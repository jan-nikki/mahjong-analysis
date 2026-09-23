"""Dependency-free pilot modeling for combo-theory candidate rows."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import exp, isfinite, log, log1p, sqrt
from random import Random
from typing import Literal

from mahjong_analysis.combo_theory import (
    UnseenCountsInput,
    calculate_simple_combo,
    normalize_unseen_counts,
    sequence_wait_patterns,
)
from mahjong_analysis.post_riichi_decisions import (
    CandidateDanger,
    PostRiichiDrawDecision,
)
from mahjong_analysis.tiles import TILE_KINDS, tile_to_index

FeatureSet = Literal[
    "simple",
    "tile_turn",
    "conventional",
    "conventional_simple",
    "conventional_simple_percentile",
    "visibility_local_saturated",
    "visibility_local_saturated_simple",
]
LabelName = Literal["structural_wait", "ron_eligible"]
CalibrationStatus = Literal[
    "ok",
    "single_class",
    "constant_logit",
    "complete_separation",
    "quasi_separation",
    "singular_hessian",
    "line_search_failed",
    "non_converged",
]

FEATURE_SETS: tuple[FeatureSet, ...] = (
    "simple",
    "tile_turn",
    "conventional",
    "conventional_simple",
    "conventional_simple_percentile",
    "visibility_local_saturated",
    "visibility_local_saturated_simple",
)
DECISION_TURN_BINS = ("1-6", "7-9", "10-12", "13+")

_FULL_UNSEEN = (4,) * len(TILE_KINDS)
_INITIAL_SIMPLE_COMBO = {
    tile: calculate_simple_combo(tile, _FULL_UNSEEN).total for tile in TILE_KINDS
}


@dataclass(frozen=True)
class ComboCandidateRow:
    """One candidate joined to its decision context and within-decision rank."""

    year: int
    game_id: str
    kyoku_index: int
    decision_id: str
    candidate_count: int
    decision: PostRiichiDrawDecision
    candidate: CandidateDanger
    simple_rank: float
    simple_percentile: float
    simple_visibility_residual: int


@dataclass(frozen=True)
class SparseExample:
    """One weighted binary example with a sparse named feature vector."""

    row_id: str
    group_id: str
    cluster_id: str
    label: bool
    weight: float
    features: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class CalibrationDiagnostics:
    """Weighted logistic recalibration diagnostics for held-out predictions."""

    joint_status: CalibrationStatus
    recalibration_intercept: float | None
    recalibration_slope: float | None
    calibration_in_the_large: float | None
    iterations: int
    clip_epsilon: float
    clipped_low: int
    clipped_high: int


@dataclass(frozen=True)
class BinaryMetrics:
    """Decision-weighted binary prediction metrics."""

    row_count: int
    decision_count: int
    positive_rate: float
    auc: float | None
    log_loss: float
    brier: float
    ece: float
    calibration: CalibrationDiagnostics
    macro_concordance: float | None
    macro_concordance_decisions: int


@dataclass(frozen=True)
class BootstrapInterval:
    """Percentile interval for one paired challenger-minus-base metric."""

    point: float | None
    lower: float | None
    upper: float | None
    valid_replicates: int


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Game-cluster bootstrap intervals for paired pooled metric differences."""

    cluster_count: int
    requested_replicates: int
    seed: int
    auc: BootstrapInterval
    log_loss: BootstrapInterval
    brier: BootstrapInterval
    ece: BootstrapInterval
    macro_concordance: BootstrapInterval


@dataclass(frozen=True)
class _ClusterAucSummary:
    positive_weights: tuple[float, ...]
    negative_weights: tuple[float, ...]
    concordance_matrix: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class _ClusterEceSummary:
    weights: tuple[tuple[float, ...], ...]
    prediction_sums: tuple[tuple[float, ...], ...]
    label_sums: tuple[tuple[float, ...], ...]


@dataclass
class SparseLogisticModel:
    """Small deterministic L2 logistic model for development pilots."""

    intercept: float
    weights: dict[str, float]
    epochs_run: int
    final_largest_gradient: float

    def predict_features(self, features: Iterable[tuple[str, float]]) -> float:
        linear = self.intercept + sum(
            self.weights.get(name, 0.0) * value for name, value in features
        )
        return _sigmoid(linear)

    def predict(self, examples: Sequence[SparseExample]) -> tuple[float, ...]:
        return tuple(self.predict_features(example.features) for example in examples)


def candidate_rows_for_decision(
    decision: PostRiichiDrawDecision,
    *,
    year: int,
    game_id: str,
    kyoku_index: int,
) -> tuple[ComboCandidateRow, ...]:
    """Attach reproducible IDs and average-tie combo ranks to a decision."""
    if isinstance(year, bool) or not isinstance(year, int):
        raise TypeError("year must be an integer")
    if not game_id:
        raise ValueError("game_id must be non-empty")
    scores = tuple(candidate.simple_combo.total for candidate in decision.candidates)
    candidate_count = len(scores)
    if candidate_count == 0:
        return ()
    decision_id = (
        f"{game_id}:k{kyoku_index}:r{decision.target_reach_accepted_event_index}"
        f":o{decision.observer_actor}:t{decision.tsumo_event_index}"
    )
    rows = []
    for candidate, score in zip(decision.candidates, scores, strict=True):
        higher = sum(other > score for other in scores)
        equal = sum(other == score for other in scores)
        average_rank = 1.0 + higher + (equal - 1) / 2
        percentile = (
            0.5
            if candidate_count == 1
            else (candidate_count - average_rank) / (candidate_count - 1)
        )
        rows.append(
            ComboCandidateRow(
                year=year,
                game_id=game_id,
                kyoku_index=kyoku_index,
                decision_id=decision_id,
                candidate_count=candidate_count,
                decision=decision,
                candidate=candidate,
                simple_rank=average_rank,
                simple_percentile=percentile,
                simple_visibility_residual=(
                    score - _INITIAL_SIMPLE_COMBO[candidate.tile]
                ),
            )
        )
    return tuple(rows)


def sparse_examples(
    rows: Sequence[ComboCandidateRow],
    feature_set: FeatureSet,
    label_name: LabelName,
) -> tuple[SparseExample, ...]:
    """Create identically weighted examples for one feature-set comparison."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set: {feature_set!r}")
    return tuple(
        SparseExample(
            row_id=f"{row.decision_id}:tile={row.candidate.tile}",
            group_id=row.decision_id,
            cluster_id=row.game_id,
            label=_label(row.candidate, label_name),
            weight=1.0 / row.candidate_count,
            features=feature_vector(row, feature_set),
        )
        for row in rows
    )


def feature_vector(
    row: ComboCandidateRow,
    feature_set: FeatureSet,
) -> tuple[tuple[str, float], ...]:
    """Build one sparse vector while keeping model comparisons nested."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set: {feature_set!r}")
    if feature_set == "simple":
        return _simple_raw_features(row)
    features = list(_tile_turn_features(row))
    if feature_set != "tile_turn":
        features.extend(_conventional_features(row))
    if feature_set in {
        "visibility_local_saturated",
        "visibility_local_saturated_simple",
    }:
        features.extend(
            visibility_local_saturated_features(
                row.candidate.tile,
                row.decision.unseen_counts,
            )
        )
    if feature_set in {
        "conventional_simple",
        "conventional_simple_percentile",
        "visibility_local_saturated_simple",
    }:
        features.extend(_simple_component_features(row))
    if feature_set == "conventional_simple_percentile":
        features.append(("combo:percentile", row.simple_percentile))
    return tuple(features)


def visibility_local_saturated_features(
    candidate_tile: str,
    unseen_counts: UnseenCountsInput,
) -> tuple[tuple[str, float], ...]:
    """One-hot the exact local unseen-count states that determine simple combo.

    The candidate count determines the tanki and shanpon terms.  Each sequence
    fragment's joint count state determines its product term.  Consequently,
    the simple raw score is in the linear span of these one-hot features; adding
    it can improve parameter sharing or regularization, but adds no information.
    """
    counts = normalize_unseen_counts(unseen_counts)
    candidate_index = tile_to_index(candidate_tile)
    candidate_count = counts[candidate_index]
    features = [
        (f"visibility_local:candidate_count:{candidate_count}", 1.0),
    ]
    for pattern in sequence_wait_patterns(candidate_tile):
        first_index = tile_to_index(pattern.tiles[0])
        second_index = tile_to_index(pattern.tiles[1])
        first_offset = first_index - candidate_index
        second_offset = second_index - candidate_index
        features.append(
            (
                "visibility_local:pair:"
                f"{first_offset:+d},{second_offset:+d}:"
                f"counts:{counts[first_index]},{counts[second_index]}",
                1.0,
            )
        )
    return tuple(features)


def fit_sparse_logistic(
    examples: Sequence[SparseExample],
    *,
    epochs: int = 80,
    learning_rate: float = 0.05,
    l2: float = 0.001,
    tolerance: float = 1e-7,
) -> SparseLogisticModel:
    """Fit deterministic full-batch Adam logistic regression.

    This small implementation is for reproducible development pilots. The main
    analysis can replace it with a separately validated production learner while
    keeping the feature sets, splits, weights, and metrics fixed.
    """
    if not examples:
        raise ValueError("at least one training example is required")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if learning_rate <= 0 or l2 < 0 or tolerance < 0:
        raise ValueError("optimizer parameters are outside their valid range")
    _validate_examples(examples)
    total_weight = sum(example.weight for example in examples)
    positive_weight = sum(example.weight for example in examples if example.label)
    prevalence = min(max(positive_weight / total_weight, 1e-6), 1 - 1e-6)
    model = SparseLogisticModel(
        intercept=log(prevalence / (1 - prevalence)),
        weights={},
        epochs_run=0,
        final_largest_gradient=float("inf"),
    )

    first_moments: defaultdict[str, float] = defaultdict(float)
    second_moments: defaultdict[str, float] = defaultdict(float)
    intercept_first_moment = 0.0
    intercept_second_moment = 0.0
    beta1 = 0.9
    beta2 = 0.999
    for epoch in range(epochs):
        intercept_gradient = 0.0
        gradients: defaultdict[str, float] = defaultdict(float)
        for example in examples:
            prediction = model.predict_features(example.features)
            error = (prediction - int(example.label)) * example.weight
            intercept_gradient += error
            for name, value in example.features:
                gradients[name] += error * value

        normalized_intercept_gradient = intercept_gradient / total_weight
        intercept_first_moment = (
            beta1 * intercept_first_moment + (1 - beta1) * normalized_intercept_gradient
        )
        intercept_second_moment = (
            beta2 * intercept_second_moment
            + (1 - beta2) * normalized_intercept_gradient**2
        )
        iteration = epoch + 1
        model.intercept -= (
            learning_rate
            * (intercept_first_moment / (1 - beta1**iteration))
            / (sqrt(intercept_second_moment / (1 - beta2**iteration)) + 1e-8)
        )
        largest_gradient = abs(normalized_intercept_gradient)
        for name in set(model.weights) | set(gradients):
            weight = model.weights.get(name, 0.0)
            gradient = gradients.get(name, 0.0) / total_weight + l2 * weight
            first_moments[name] = beta1 * first_moments[name] + (1 - beta1) * gradient
            second_moments[name] = (
                beta2 * second_moments[name] + (1 - beta2) * gradient**2
            )
            first_hat = first_moments[name] / (1 - beta1**iteration)
            second_hat = second_moments[name] / (1 - beta2**iteration)
            model.weights[name] = weight - learning_rate * first_hat / (
                sqrt(second_hat) + 1e-8
            )
            largest_gradient = max(largest_gradient, abs(gradient))
        model.epochs_run = epoch + 1
        model.final_largest_gradient = largest_gradient
        if largest_gradient <= tolerance:
            break
    return model


def evaluate_binary_predictions(
    examples: Sequence[SparseExample],
    predictions: Sequence[float],
    *,
    ece_bins: int = 10,
) -> BinaryMetrics:
    """Calculate decision-weighted row metrics and macro decision concordance."""
    if len(examples) != len(predictions) or not examples:
        raise ValueError("examples and predictions must have the same nonzero length")
    if isinstance(ece_bins, bool) or not isinstance(ece_bins, int) or ece_bins < 1:
        raise ValueError("ece_bins must be a positive integer")
    _validate_examples(examples)
    checked_predictions = tuple(_validate_probability(value) for value in predictions)
    total_weight = sum(example.weight for example in examples)
    positive_rate = (
        sum(example.weight * int(example.label) for example in examples) / total_weight
    )
    log_loss = (
        sum(
            example.weight
            * -(
                int(example.label) * log(max(prediction, 1e-15))
                + (1 - int(example.label)) * log(max(1 - prediction, 1e-15))
            )
            for example, prediction in zip(examples, checked_predictions, strict=True)
        )
        / total_weight
    )
    brier = (
        sum(
            example.weight * (prediction - int(example.label)) ** 2
            for example, prediction in zip(examples, checked_predictions, strict=True)
        )
        / total_weight
    )
    auc = _weighted_auc(examples, checked_predictions)
    ece = _expected_calibration_error(
        examples,
        checked_predictions,
        bins=ece_bins,
    )
    calibration = evaluate_calibration(
        examples,
        checked_predictions,
    )
    macro, coverage = _macro_concordance(examples, checked_predictions)
    return BinaryMetrics(
        row_count=len(examples),
        decision_count=len({example.group_id for example in examples}),
        positive_rate=positive_rate,
        auc=auc,
        log_loss=log_loss,
        brier=brier,
        ece=ece,
        calibration=calibration,
        macro_concordance=macro,
        macro_concordance_decisions=coverage,
    )


def evaluate_calibration(
    examples: Sequence[SparseExample],
    predictions: Sequence[float],
    *,
    clip_epsilon: float = 1e-15,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> CalibrationDiagnostics:
    """Fit weighted, unpenalized logistic recalibration on held-out predictions.

    ``recalibration_intercept`` and ``recalibration_slope`` are jointly fitted;
    their ideal values are zero and one. ``calibration_in_the_large`` instead
    fixes the slope at one and estimates only an offset. Non-finite joint MLEs
    from complete or quasi separation are explicitly reported as missing.
    """
    if len(examples) != len(predictions) or not examples:
        raise ValueError("examples and predictions must have the same nonzero length")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if not 0 < clip_epsilon < 0.5:
        raise ValueError("clip_epsilon must be between zero and one half")
    _validate_examples(examples)
    checked = tuple(_validate_probability(value) for value in predictions)
    clipped = tuple(
        min(max(value, clip_epsilon), 1 - clip_epsilon) for value in checked
    )
    clipped_low = sum(value < clip_epsilon for value in checked)
    clipped_high = sum(value > 1 - clip_epsilon for value in checked)
    total_weight = sum(example.weight for example in examples)
    positive_weight = sum(example.weight for example in examples if example.label)
    if positive_weight == 0 or positive_weight == total_weight:
        return CalibrationDiagnostics(
            joint_status="single_class",
            recalibration_intercept=None,
            recalibration_slope=None,
            calibration_in_the_large=None,
            iterations=0,
            clip_epsilon=clip_epsilon,
            clipped_low=clipped_low,
            clipped_high=clipped_high,
        )
    logits = tuple(log(value / (1 - value)) for value in clipped)
    calibration_in_the_large = _calibration_in_the_large(
        examples,
        logits,
        tolerance=tolerance,
    )
    weighted_mean = (
        sum(
            example.weight * value
            for example, value in zip(examples, logits, strict=True)
        )
        / total_weight
    )
    weighted_variance = (
        sum(
            example.weight * (value - weighted_mean) ** 2
            for example, value in zip(examples, logits, strict=True)
        )
        / total_weight
    )
    if weighted_variance <= 1e-14:
        return CalibrationDiagnostics(
            joint_status="constant_logit",
            recalibration_intercept=None,
            recalibration_slope=None,
            calibration_in_the_large=calibration_in_the_large,
            iterations=0,
            clip_epsilon=clip_epsilon,
            clipped_low=clipped_low,
            clipped_high=clipped_high,
        )
    separation_status = _one_dimensional_separation_status(examples, logits)
    if separation_status is not None:
        return CalibrationDiagnostics(
            joint_status=separation_status,
            recalibration_intercept=None,
            recalibration_slope=None,
            calibration_in_the_large=calibration_in_the_large,
            iterations=0,
            clip_epsilon=clip_epsilon,
            clipped_low=clipped_low,
            clipped_high=clipped_high,
        )

    standard_deviation = sqrt(weighted_variance)
    standardized_logits = tuple(
        (value - weighted_mean) / standard_deviation for value in logits
    )
    prevalence = positive_weight / total_weight
    intercept = log(prevalence / (1 - prevalence))
    slope = 0.0
    loss = (
        _calibration_log_loss(
            examples,
            standardized_logits,
            intercept,
            slope,
        )
        / total_weight
    )
    failure_status: CalibrationStatus = "non_converged"
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        gradient_intercept = 0.0
        gradient_slope = 0.0
        hessian_intercept = 0.0
        hessian_cross = 0.0
        hessian_slope = 0.0
        for example, value in zip(examples, standardized_logits, strict=True):
            fitted = _sigmoid(intercept + slope * value)
            error = (fitted - int(example.label)) * example.weight
            curvature = fitted * (1 - fitted) * example.weight
            gradient_intercept += error
            gradient_slope += error * value
            hessian_intercept += curvature
            hessian_cross += curvature * value
            hessian_slope += curvature * value * value
        gradient_intercept /= total_weight
        gradient_slope /= total_weight
        hessian_intercept /= total_weight
        hessian_cross /= total_weight
        hessian_slope /= total_weight
        if max(abs(gradient_intercept), abs(gradient_slope)) <= tolerance:
            original_slope = slope / standard_deviation
            original_intercept = intercept - original_slope * weighted_mean
            return CalibrationDiagnostics(
                joint_status="ok",
                recalibration_intercept=original_intercept,
                recalibration_slope=original_slope,
                calibration_in_the_large=calibration_in_the_large,
                iterations=iteration,
                clip_epsilon=clip_epsilon,
                clipped_low=clipped_low,
                clipped_high=clipped_high,
            )
        determinant = hessian_intercept * hessian_slope - hessian_cross * hessian_cross
        hessian_scale = hessian_intercept * hessian_slope
        if hessian_scale <= 0 or determinant <= 1e-12 * hessian_scale:
            failure_status = "singular_hessian"
            break
        step_intercept = (
            hessian_slope * gradient_intercept - hessian_cross * gradient_slope
        ) / determinant
        step_slope = (
            hessian_intercept * gradient_slope - hessian_cross * gradient_intercept
        ) / determinant

        accepted = False
        step_scale = 1.0
        for _line_search in range(30):
            candidate_intercept = intercept - step_scale * step_intercept
            candidate_slope = slope - step_scale * step_slope
            candidate_loss = (
                _calibration_log_loss(
                    examples,
                    standardized_logits,
                    candidate_intercept,
                    candidate_slope,
                )
                / total_weight
            )
            if isfinite(candidate_loss) and candidate_loss <= loss + 1e-15:
                intercept = candidate_intercept
                slope = candidate_slope
                loss = candidate_loss
                accepted = True
                break
            step_scale *= 0.5
        if not accepted:
            failure_status = "line_search_failed"
            break
    return CalibrationDiagnostics(
        joint_status=failure_status,
        recalibration_intercept=None,
        recalibration_slope=None,
        calibration_in_the_large=calibration_in_the_large,
        iterations=iteration,
        clip_epsilon=clip_epsilon,
        clipped_low=clipped_low,
        clipped_high=clipped_high,
    )


def calibration_intercept_slope(
    examples: Sequence[SparseExample],
    predictions: Sequence[float],
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> tuple[float | None, float | None]:
    """Return the jointly fitted recalibration intercept and slope."""
    diagnostics = evaluate_calibration(
        examples,
        predictions,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    return diagnostics.recalibration_intercept, diagnostics.recalibration_slope


def paired_game_cluster_bootstrap(
    examples: Sequence[SparseExample],
    base_predictions: Sequence[float],
    challenger_predictions: Sequence[float],
    *,
    replicates: int = 200,
    seed: int = 20260923,
    ece_bins: int = 10,
) -> PairedBootstrapResult:
    """Resample games and retain within-game candidate dependence."""
    if (
        len(examples) != len(base_predictions)
        or len(examples) != len(challenger_predictions)
        or not examples
    ):
        raise ValueError("examples and both prediction vectors must align")
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates < 1
    ):
        raise ValueError("replicates must be a positive integer")
    _validate_examples(examples)
    base_checked = tuple(_validate_probability(value) for value in base_predictions)
    challenger_checked = tuple(
        _validate_probability(value) for value in challenger_predictions
    )
    cluster_ids = tuple(sorted({example.cluster_id for example in examples}))
    if not cluster_ids:
        raise ValueError("at least one cluster is required")

    base_full = evaluate_binary_predictions(
        examples,
        base_checked,
        ece_bins=ece_bins,
    )
    challenger_full = evaluate_binary_predictions(
        examples,
        challenger_checked,
        ece_bins=ece_bins,
    )
    cluster_index = {cluster_id: index for index, cluster_id in enumerate(cluster_ids)}
    cluster_weights = [0.0] * len(cluster_ids)
    log_loss_differences = [0.0] * len(cluster_ids)
    brier_differences = [0.0] * len(cluster_ids)
    for example, base, challenger in zip(
        examples,
        base_checked,
        challenger_checked,
        strict=True,
    ):
        index = cluster_index[example.cluster_id]
        label = int(example.label)
        cluster_weights[index] += example.weight
        log_loss_differences[index] += example.weight * (
            _row_log_loss(label, challenger) - _row_log_loss(label, base)
        )
        brier_differences[index] += example.weight * (
            (challenger - label) ** 2 - (base - label) ** 2
        )
    base_auc_summary = _cluster_auc_summary(
        examples,
        base_checked,
        cluster_index,
    )
    challenger_auc_summary = _cluster_auc_summary(
        examples,
        challenger_checked,
        cluster_index,
    )
    base_ece_summary = _cluster_ece_summary(
        examples,
        base_checked,
        cluster_index,
        bins=ece_bins,
    )
    challenger_ece_summary = _cluster_ece_summary(
        examples,
        challenger_checked,
        cluster_index,
        bins=ece_bins,
    )
    macro_differences, macro_counts = _cluster_macro_differences(
        examples,
        base_checked,
        challenger_checked,
        cluster_index,
    )
    samples: dict[str, list[float]] = {
        "auc": [],
        "log_loss": [],
        "brier": [],
        "ece": [],
        "macro_concordance": [],
    }
    random = Random(seed)
    for _ in range(replicates):
        multiplicities = [0] * len(cluster_ids)
        for _draw in range(len(cluster_ids)):
            multiplicities[random.randrange(len(cluster_ids))] += 1
        total_weight = sum(
            multiplicity * weight
            for multiplicity, weight in zip(
                multiplicities,
                cluster_weights,
                strict=True,
            )
        )
        base_auc = _auc_from_cluster_summary(base_auc_summary, multiplicities)
        challenger_auc = _auc_from_cluster_summary(
            challenger_auc_summary,
            multiplicities,
        )
        if base_auc is not None and challenger_auc is not None:
            samples["auc"].append(challenger_auc - base_auc)
        samples["log_loss"].append(
            sum(
                multiplicity * difference
                for multiplicity, difference in zip(
                    multiplicities,
                    log_loss_differences,
                    strict=True,
                )
            )
            / total_weight
        )
        samples["brier"].append(
            sum(
                multiplicity * difference
                for multiplicity, difference in zip(
                    multiplicities,
                    brier_differences,
                    strict=True,
                )
            )
            / total_weight
        )
        samples["ece"].append(
            _ece_from_cluster_summary(challenger_ece_summary, multiplicities)
            - _ece_from_cluster_summary(base_ece_summary, multiplicities)
        )
        macro_count = sum(
            multiplicity * count
            for multiplicity, count in zip(
                multiplicities,
                macro_counts,
                strict=True,
            )
        )
        if macro_count:
            samples["macro_concordance"].append(
                sum(
                    multiplicity * difference
                    for multiplicity, difference in zip(
                        multiplicities,
                        macro_differences,
                        strict=True,
                    )
                )
                / macro_count
            )

    return PairedBootstrapResult(
        cluster_count=len(cluster_ids),
        requested_replicates=replicates,
        seed=seed,
        auc=_bootstrap_interval(
            _optional_metric_difference(challenger_full.auc, base_full.auc),
            samples["auc"],
        ),
        log_loss=_bootstrap_interval(
            challenger_full.log_loss - base_full.log_loss,
            samples["log_loss"],
        ),
        brier=_bootstrap_interval(
            challenger_full.brier - base_full.brier,
            samples["brier"],
        ),
        ece=_bootstrap_interval(
            challenger_full.ece - base_full.ece,
            samples["ece"],
        ),
        macro_concordance=_bootstrap_interval(
            _optional_metric_difference(
                challenger_full.macro_concordance,
                base_full.macro_concordance,
            ),
            samples["macro_concordance"],
        ),
    )


def _tile_turn_features(row: ComboCandidateRow) -> tuple[tuple[str, float], ...]:
    feature = row.candidate.conventional
    decision = row.decision
    return (
        (f"tile:{row.candidate.tile}", 1.0),
        (f"turn:{decision_turn_bin(decision.decision_discard_number)}", 1.0),
        (
            f"riichi_turn:{decision_turn_bin(decision.target_riichi_discard_number)}",
            1.0,
        ),
        (
            f"post_riichi_decision:{_post_riichi_bin(decision.post_riichi_decision_number)}",
            1.0,
        ),
        ("turn:scaled", min(decision.decision_discard_number, 18) / 18),
        ("riichi_turn:scaled", min(decision.target_riichi_discard_number, 18) / 18),
        ("tile:is_honor", float(feature.candidate_is_honor)),
        ("tile:is_terminal", float(feature.candidate_is_terminal)),
        ("tile:is_simple", float(feature.candidate_is_simple)),
    )


def _conventional_features(
    row: ComboCandidateRow,
) -> tuple[tuple[str, float], ...]:
    feature = row.candidate.conventional
    first_discard = feature.target_first_discard_number
    decision = row.decision
    target_score = decision.scores[decision.target_actor]
    observer_score = decision.scores[decision.observer_actor]
    target_place = 1 + sum(score > target_score for score in decision.scores)
    return (
        (f"safety:{feature.safety_class}", 1.0),
        (f"visible:{feature.visible_count}", 1.0),
        ("safety:is_target_own_discard", float(feature.is_target_own_discard)),
        ("safety:is_post_riichi_passed", float(feature.is_post_riichi_passed)),
        (
            "suji:safe_share",
            feature.suji_safe_count / feature.suji_reference_count
            if feature.suji_reference_count
            else 0.0,
        ),
        (
            f"wall:no_chance_directions:{feature.no_chance_direction_count}",
            1.0,
        ),
        (
            f"wall:one_chance_directions:{feature.one_chance_direction_count}",
            1.0,
        ),
        (
            "wall:all_ryanmen_blocked",
            float(feature.all_ryanmen_directions_blocked),
        ),
        (
            f"river:first_candidate:{_first_discard_bin(first_discard)}",
            1.0,
        ),
        ("river:matches_declaration", float(feature.matches_riichi_declaration)),
        (
            "river:same_suit_as_declaration",
            float(feature.same_suit_as_riichi_declaration),
        ),
        (
            f"river:declaration_rank_distance:{feature.rank_distance_from_riichi_declaration}",
            1.0,
        ),
        ("river:safe_kind_share", len(row.decision.target_safe_tile_kinds) / 34),
        ("round:target_is_dealer", float(decision.target_actor == decision.oya)),
        ("round:observer_is_dealer", float(decision.observer_actor == decision.oya)),
        (f"round:kyoku:{decision.kyoku_number}", 1.0),
        (f"round:honba:{min(decision.honba, 4)}", 1.0),
        (f"round:kyotaku:{min(decision.kyotaku, 4)}", 1.0),
        (f"round:target_place:{target_place}", 1.0),
        ("round:target_score_scaled", target_score / 50000),
        (
            "round:observer_minus_target_score_scaled",
            (observer_score - target_score) / 50000,
        ),
    )


def _simple_raw_features(
    row: ComboCandidateRow,
) -> tuple[tuple[str, float], ...]:
    return (("combo:total", row.candidate.simple_combo.total / 58),)


def _simple_component_features(
    row: ComboCandidateRow,
) -> tuple[tuple[str, float], ...]:
    simple = row.candidate.simple_combo
    return (
        *_simple_raw_features(row),
        (
            "combo:sequence_total",
            sum(component.count for component in simple.sequence_combos) / 48,
        ),
        ("combo:shanpon", simple.shanpon_combos / 6),
        ("combo:tanki", simple.tanki_combos / 4),
        ("combo:visibility_residual", row.simple_visibility_residual / 58),
    )


def _label(candidate: CandidateDanger, label_name: LabelName) -> bool:
    if label_name == "structural_wait":
        return candidate.is_structural_wait
    if label_name == "ron_eligible":
        return candidate.is_ron_eligible
    raise ValueError(f"unknown label: {label_name!r}")


def decision_turn_bin(turn: int) -> str:
    """Return the pre-specified discard-number bin used for subgroup analysis."""
    if isinstance(turn, bool) or not isinstance(turn, int):
        raise TypeError("turn must be an integer")
    if turn < 1:
        raise ValueError("turn must be positive")
    if turn <= 6:
        return "1-6"
    if turn <= 9:
        return "7-9"
    if turn <= 12:
        return "10-12"
    return "13+"


def _post_riichi_bin(number: int) -> str:
    if number <= 2:
        return str(number)
    if number <= 4:
        return "3-4"
    return "5+"


def _first_discard_bin(number: int | None) -> str:
    if number is None:
        return "none"
    if number <= 3:
        return "1-3"
    if number <= 6:
        return "4-6"
    if number <= 9:
        return "7-9"
    return "10+"


def _validate_examples(examples: Sequence[SparseExample]) -> None:
    for example in examples:
        if not isfinite(example.weight) or example.weight <= 0:
            raise ValueError("example weights must be finite and positive")
        if not example.group_id:
            raise ValueError("example group IDs must be non-empty")
        if not example.row_id:
            raise ValueError("example row IDs must be non-empty")
        if not example.cluster_id:
            raise ValueError("example cluster IDs must be non-empty")
        for name, value in example.features:
            if not name:
                raise ValueError("feature names must be non-empty")
            if not isfinite(value):
                raise ValueError("feature values must be finite")


def _validate_probability(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("predictions must be numeric probabilities")
    result = float(value)
    if result != result or not 0 <= result <= 1:
        raise ValueError("predictions must be between zero and one")
    return result


def _sigmoid(value: float) -> float:
    if value >= 0:
        decay = exp(-min(value, 709))
        return 1 / (1 + decay)
    growth = exp(max(value, -709))
    return growth / (1 + growth)


def _row_log_loss(label: int, prediction: float) -> float:
    return -(
        label * log(max(prediction, 1e-15))
        + (1 - label) * log(max(1 - prediction, 1e-15))
    )


def _calibration_log_loss(
    examples: Sequence[SparseExample],
    logits: Sequence[float],
    intercept: float,
    slope: float,
) -> float:
    result = 0.0
    for example, value in zip(examples, logits, strict=True):
        linear = intercept + slope * value
        if linear >= 0:
            row_loss = (1 - int(example.label)) * linear + log1p(exp(-linear))
        else:
            row_loss = -int(example.label) * linear + log1p(exp(linear))
        result += example.weight * row_loss
    return result


def _calibration_in_the_large(
    examples: Sequence[SparseExample],
    logits: Sequence[float],
    *,
    tolerance: float,
) -> float:
    total_weight = sum(example.weight for example in examples)
    lower = -100.0
    upper = 100.0
    for _iteration in range(200):
        midpoint = (lower + upper) / 2
        score = (
            sum(
                example.weight * (_sigmoid(value + midpoint) - int(example.label))
                for example, value in zip(examples, logits, strict=True)
            )
            / total_weight
        )
        if abs(score) <= tolerance:
            return midpoint
        if score > 0:
            upper = midpoint
        else:
            lower = midpoint
    return (lower + upper) / 2


def _one_dimensional_separation_status(
    examples: Sequence[SparseExample],
    logits: Sequence[float],
) -> Literal["complete_separation", "quasi_separation"] | None:
    positive = tuple(
        value for example, value in zip(examples, logits, strict=True) if example.label
    )
    negative = tuple(
        value
        for example, value in zip(examples, logits, strict=True)
        if not example.label
    )
    positive_min = min(positive)
    positive_max = max(positive)
    negative_min = min(negative)
    negative_max = max(negative)
    boundary_scale = max(
        1.0,
        abs(positive_min),
        abs(positive_max),
        abs(negative_min),
        abs(negative_max),
    )
    boundary_tolerance = 1e-12 * boundary_scale
    if (
        negative_max < positive_min - boundary_tolerance
        or positive_max < negative_min - boundary_tolerance
    ):
        return "complete_separation"
    if (
        negative_max <= positive_min + boundary_tolerance
        or positive_max <= negative_min + boundary_tolerance
    ):
        return "quasi_separation"
    return None


def _cluster_auc_summary(
    examples: Sequence[SparseExample],
    predictions: Sequence[float],
    cluster_index: dict[str, int],
) -> _ClusterAucSummary:
    cluster_count = len(cluster_index)
    positive_weights = [0.0] * cluster_count
    negative_weights = [0.0] * cluster_count
    for example in examples:
        target = positive_weights if example.label else negative_weights
        target[cluster_index[example.cluster_id]] += example.weight
    matrix = [[0.0] * cluster_count for _ in range(cluster_count)]
    cumulative_negative = [0.0] * cluster_count
    ordered = sorted(
        zip(predictions, examples, strict=True),
        key=lambda item: item[0],
    )
    position = 0
    while position < len(ordered):
        score = ordered[position][0]
        group_positive: defaultdict[int, float] = defaultdict(float)
        group_negative: defaultdict[int, float] = defaultdict(float)
        while position < len(ordered) and ordered[position][0] == score:
            example = ordered[position][1]
            index = cluster_index[example.cluster_id]
            target = group_positive if example.label else group_negative
            target[index] += example.weight
            position += 1
        for positive_index, positive_weight in group_positive.items():
            row = matrix[positive_index]
            for negative_index in range(cluster_count):
                row[negative_index] += positive_weight * (
                    cumulative_negative[negative_index]
                    + 0.5 * group_negative.get(negative_index, 0.0)
                )
        for index, weight in group_negative.items():
            cumulative_negative[index] += weight
    return _ClusterAucSummary(
        tuple(positive_weights),
        tuple(negative_weights),
        tuple(tuple(row) for row in matrix),
    )


def _auc_from_cluster_summary(
    summary: _ClusterAucSummary,
    multiplicities: Sequence[int],
) -> float | None:
    positive_weight = sum(
        multiplicity * weight
        for multiplicity, weight in zip(
            multiplicities,
            summary.positive_weights,
            strict=True,
        )
    )
    negative_weight = sum(
        multiplicity * weight
        for multiplicity, weight in zip(
            multiplicities,
            summary.negative_weights,
            strict=True,
        )
    )
    if not positive_weight or not negative_weight:
        return None
    numerator = sum(
        multiplicities[positive_index] * multiplicities[negative_index] * value
        for positive_index, row in enumerate(summary.concordance_matrix)
        for negative_index, value in enumerate(row)
    )
    return numerator / (positive_weight * negative_weight)


def _cluster_ece_summary(
    examples: Sequence[SparseExample],
    predictions: Sequence[float],
    cluster_index: dict[str, int],
    *,
    bins: int,
) -> _ClusterEceSummary:
    weights = [[0.0] * bins for _ in cluster_index]
    prediction_sums = [[0.0] * bins for _ in cluster_index]
    label_sums = [[0.0] * bins for _ in cluster_index]
    for example, prediction in zip(examples, predictions, strict=True):
        cluster = cluster_index[example.cluster_id]
        bin_index = min(int(prediction * bins), bins - 1)
        weights[cluster][bin_index] += example.weight
        prediction_sums[cluster][bin_index] += example.weight * prediction
        label_sums[cluster][bin_index] += example.weight * int(example.label)
    return _ClusterEceSummary(
        tuple(tuple(row) for row in weights),
        tuple(tuple(row) for row in prediction_sums),
        tuple(tuple(row) for row in label_sums),
    )


def _ece_from_cluster_summary(
    summary: _ClusterEceSummary,
    multiplicities: Sequence[int],
) -> float:
    bin_count = len(summary.weights[0])
    total_weight = sum(
        multiplicities[cluster] * summary.weights[cluster][bin_index]
        for cluster in range(len(multiplicities))
        for bin_index in range(bin_count)
    )
    result = 0.0
    for bin_index in range(bin_count):
        weight = sum(
            multiplicities[cluster] * summary.weights[cluster][bin_index]
            for cluster in range(len(multiplicities))
        )
        if not weight:
            continue
        prediction_sum = sum(
            multiplicities[cluster] * summary.prediction_sums[cluster][bin_index]
            for cluster in range(len(multiplicities))
        )
        label_sum = sum(
            multiplicities[cluster] * summary.label_sums[cluster][bin_index]
            for cluster in range(len(multiplicities))
        )
        result += (
            weight / total_weight * abs(prediction_sum / weight - label_sum / weight)
        )
    return result


def _optional_metric_difference(
    left: float | None,
    right: float | None,
) -> float | None:
    return None if left is None or right is None else left - right


def _bootstrap_interval(
    point: float | None,
    values: Sequence[float],
) -> BootstrapInterval:
    if not values:
        return BootstrapInterval(point, None, None, 0)
    ordered = sorted(values)
    return BootstrapInterval(
        point=point,
        lower=_percentile(ordered, 0.025),
        upper=_percentile(ordered, 0.975),
        valid_replicates=len(ordered),
    )


def _percentile(ordered: Sequence[float], probability: float) -> float:
    position = probability * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] * (1 - fraction) + ordered[upper_index] * fraction


def _weighted_auc(
    examples: Sequence[SparseExample],
    predictions: Sequence[float],
) -> float | None:
    ordered = sorted(
        zip(predictions, examples, strict=True),
        key=lambda item: item[0],
    )
    positive_weight = sum(example.weight for example in examples if example.label)
    negative_weight = sum(example.weight for example in examples if not example.label)
    if not positive_weight or not negative_weight:
        return None
    concordant = 0.0
    cumulative_negative = 0.0
    index = 0
    while index < len(ordered):
        score = ordered[index][0]
        group_positive = 0.0
        group_negative = 0.0
        while index < len(ordered) and ordered[index][0] == score:
            example = ordered[index][1]
            if example.label:
                group_positive += example.weight
            else:
                group_negative += example.weight
            index += 1
        concordant += group_positive * (cumulative_negative + 0.5 * group_negative)
        cumulative_negative += group_negative
    return concordant / (positive_weight * negative_weight)


def _expected_calibration_error(
    examples: Sequence[SparseExample],
    predictions: Sequence[float],
    *,
    bins: int,
) -> float:
    totals = [0.0] * bins
    predictions_sum = [0.0] * bins
    labels_sum = [0.0] * bins
    for example, prediction in zip(examples, predictions, strict=True):
        index = min(int(prediction * bins), bins - 1)
        totals[index] += example.weight
        predictions_sum[index] += example.weight * prediction
        labels_sum[index] += example.weight * int(example.label)
    total_weight = sum(totals)
    return sum(
        weight
        / total_weight
        * abs(predictions_sum[index] / weight - labels_sum[index] / weight)
        for index, weight in enumerate(totals)
        if weight
    )


def _macro_concordance(
    examples: Sequence[SparseExample],
    predictions: Sequence[float],
) -> tuple[float | None, int]:
    groups: defaultdict[str, list[tuple[bool, float]]] = defaultdict(list)
    for example, prediction in zip(examples, predictions, strict=True):
        groups[example.group_id].append((example.label, prediction))
    concordances = []
    for values in groups.values():
        positives = [score for label, score in values if label]
        negatives = [score for label, score in values if not label]
        if not positives or not negatives:
            continue
        pair_scores = [
            1.0 if positive > negative else 0.5 if positive == negative else 0.0
            for positive in positives
            for negative in negatives
        ]
        concordances.append(sum(pair_scores) / len(pair_scores))
    if not concordances:
        return (None, 0)
    return (sum(concordances) / len(concordances), len(concordances))


def _cluster_macro_differences(
    examples: Sequence[SparseExample],
    base_predictions: Sequence[float],
    challenger_predictions: Sequence[float],
    cluster_index: dict[str, int],
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    groups: defaultdict[
        str,
        list[tuple[bool, float, float, str]],
    ] = defaultdict(list)
    for example, base, challenger in zip(
        examples,
        base_predictions,
        challenger_predictions,
        strict=True,
    ):
        groups[example.group_id].append(
            (example.label, base, challenger, example.cluster_id)
        )
    differences = [0.0] * len(cluster_index)
    counts = [0] * len(cluster_index)
    for group_id, values in groups.items():
        cluster_ids = {value[3] for value in values}
        if len(cluster_ids) != 1:
            raise ValueError(f"decision spans multiple game clusters: {group_id}")
        positive_positions = [index for index, value in enumerate(values) if value[0]]
        negative_positions = [
            index for index, value in enumerate(values) if not value[0]
        ]
        if not positive_positions or not negative_positions:
            continue
        base_scores = [
            1.0
            if values[positive][1] > values[negative][1]
            else 0.5
            if values[positive][1] == values[negative][1]
            else 0.0
            for positive in positive_positions
            for negative in negative_positions
        ]
        challenger_scores = [
            1.0
            if values[positive][2] > values[negative][2]
            else 0.5
            if values[positive][2] == values[negative][2]
            else 0.0
            for positive in positive_positions
            for negative in negative_positions
        ]
        index = cluster_index[next(iter(cluster_ids))]
        differences[index] += sum(challenger_scores) / len(challenger_scores) - sum(
            base_scores
        ) / len(base_scores)
        counts[index] += 1
    return tuple(differences), tuple(counts)
