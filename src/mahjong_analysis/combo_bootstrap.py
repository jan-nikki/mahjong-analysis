"""Joint game-cluster bootstrap for combo-model comparisons and subgroups."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, log
from random import Random

import numpy as np
from numpy.typing import NDArray

from mahjong_analysis.combo_prediction import BootstrapInterval, SparseExample

_METRIC_NAMES = ("auc", "log_loss", "brier", "ece", "macro_concordance")


@dataclass(frozen=True)
class GroupContrast:
    """A pre-specified difference between two subgroup model effects."""

    name: str
    left_group: str
    right_group: str


@dataclass(frozen=True)
class MetricBootstrapIntervals:
    """Paired challenger-minus-base intervals for all reported metrics."""

    auc: BootstrapInterval
    log_loss: BootstrapInterval
    brier: BootstrapInterval
    ece: BootstrapInterval
    macro_concordance: BootstrapInterval


@dataclass(frozen=True)
class GroupBootstrapResult:
    """One named subgroup's paired bootstrap result."""

    group: str
    row_count: int
    decision_count: int
    cluster_count: int
    metrics: MetricBootstrapIntervals


@dataclass(frozen=True)
class ContrastBootstrapResult:
    """A replicate-wise subgroup-effect contrast."""

    name: str
    left_group: str
    right_group: str
    metrics: MetricBootstrapIntervals


@dataclass(frozen=True)
class ComparisonBootstrapResult:
    """Joint results for one named model comparison."""

    name: str
    base_model: str
    challenger_model: str
    overall: MetricBootstrapIntervals
    by_group: tuple[GroupBootstrapResult, ...]
    contrasts: tuple[ContrastBootstrapResult, ...]


@dataclass(frozen=True)
class JointBootstrapResult:
    """Metadata and results from one shared set of cluster-bootstrap draws."""

    cluster_count: int
    requested_replicates: int
    seed: int
    ece_bins: int
    stratified: bool
    stratum_cluster_counts: tuple[tuple[str, int], ...]
    group_order: tuple[str, ...]
    comparisons: tuple[ComparisonBootstrapResult, ...]


@dataclass(frozen=True)
class _AucSummary:
    cluster_indices: NDArray[np.intp]
    positive_weights: NDArray[np.float64]
    negative_weights: NDArray[np.float64]
    concordance_matrix: NDArray[np.float64]


@dataclass(frozen=True)
class _EceSummary:
    weights: NDArray[np.float64]
    prediction_sums: NDArray[np.float64]
    label_sums: NDArray[np.float64]


@dataclass(frozen=True)
class _ScopeSummary:
    row_count: int
    decision_count: int
    cluster_count: int
    cluster_weights: NDArray[np.float64]
    cluster_log_losses: NDArray[np.float64]
    cluster_briers: NDArray[np.float64]
    auc: _AucSummary | None
    ece: _EceSummary | None
    macro_sums: NDArray[np.float64]
    macro_counts: NDArray[np.int64]


def joint_paired_game_cluster_bootstrap(
    examples: Sequence[SparseExample],
    predictions_by_model: Mapping[str, Sequence[float]],
    comparisons: Mapping[str, tuple[str, str]],
    *,
    group_ids: Sequence[str] | None = None,
    group_order: Sequence[str] = (),
    stratum_by_cluster: Mapping[str, str] | None = None,
    contrasts: Sequence[GroupContrast] = (),
    replicates: int = 2_000,
    seed: int = 20260923,
    ece_bins: int = 10,
) -> JointBootstrapResult:
    """Bootstrap several paired comparisons with shared game-level draws.

    Each comparison value is ``challenger - base``. A group contrast is the
    comparison value in ``left_group`` minus that in ``right_group``. When
    strata are supplied, games are resampled independently within each stratum
    while retaining that stratum's original number of games. Metrics, including
    pooled AUC, are then evaluated over the combined stratified draw.
    """
    _validate_integer(replicates, "replicates")
    _validate_integer(ece_bins, "ece_bins")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not examples:
        raise ValueError("examples must be non-empty")
    _validate_examples(examples)

    cluster_ids = tuple(sorted({example.cluster_id for example in examples}))
    cluster_index = {cluster_id: index for index, cluster_id in enumerate(cluster_ids)}
    checked_predictions = _validate_predictions(
        examples,
        predictions_by_model,
    )
    checked_comparisons = _validate_comparisons(
        checked_predictions,
        comparisons,
    )
    checked_group_ids, checked_group_order = _validate_groups(
        examples,
        group_ids,
        group_order,
        contrasts,
    )
    checked_strata = _validate_strata(cluster_ids, stratum_by_cluster)

    scope_indices: dict[str | None, tuple[int, ...]] = {
        None: tuple(range(len(examples)))
    }
    if checked_group_ids is not None:
        for group in checked_group_order:
            scope_indices[group] = tuple(
                index for index, value in enumerate(checked_group_ids) if value == group
            )

    required_models = tuple(
        dict.fromkeys(
            model
            for _name, base, challenger in checked_comparisons
            for model in (base, challenger)
        )
    )
    bootstrap_multiplicities = _draw_multiplicity_matrix(
        len(cluster_ids),
        checked_strata,
        replicates,
        Random(seed),
    )
    multiplicities = np.empty(
        (replicates + 1, len(cluster_ids)),
        dtype=np.int32,
    )
    multiplicities[0, :] = 1
    multiplicities[1:, :] = bootstrap_multiplicities
    del bootstrap_multiplicities

    scope_metadata: dict[str | None, tuple[int, int, int]] = {}
    model_metrics: dict[
        tuple[str | None, str],
        dict[str, NDArray[np.float64]],
    ] = {}
    # Materialize one dense AUC matrix at a time. Keeping only the resulting
    # metric vectors prevents peak memory from scaling with scopes or models.
    for scope, indices in scope_indices.items():
        subset = tuple(examples[index] for index in indices)
        scope_metadata[scope] = (
            len(subset),
            len({example.group_id for example in subset}),
            len({example.cluster_id for example in subset}),
        )
        for model in required_models:
            predictions = tuple(checked_predictions[model][index] for index in indices)
            summary = _scope_summary(
                subset,
                predictions,
                cluster_index,
                ece_bins,
            )
            model_metrics[(scope, model)] = _metrics_from_summary(
                summary,
                multiplicities,
            )
            del summary

    differences: dict[
        tuple[str, str | None],
        dict[str, NDArray[np.float64]],
    ] = {}
    for comparison_name, base, challenger in checked_comparisons:
        for scope in scope_indices:
            differences[(comparison_name, scope)] = {
                metric: (
                    model_metrics[(scope, challenger)][metric]
                    - model_metrics[(scope, base)][metric]
                )
                for metric in _METRIC_NAMES
            }

    comparison_results = []
    for comparison_name, base, challenger in checked_comparisons:
        overall = _intervals_for_differences(
            differences[(comparison_name, None)],
        )
        group_results = []
        for group in checked_group_order:
            row_count, decision_count, group_cluster_count = scope_metadata[group]
            group_results.append(
                GroupBootstrapResult(
                    group=group,
                    row_count=row_count,
                    decision_count=decision_count,
                    cluster_count=group_cluster_count,
                    metrics=_intervals_for_differences(
                        differences[(comparison_name, group)],
                    ),
                )
            )
        contrast_results = []
        for contrast in contrasts:
            left = differences[(comparison_name, contrast.left_group)]
            right = differences[(comparison_name, contrast.right_group)]
            contrast_results.append(
                ContrastBootstrapResult(
                    name=contrast.name,
                    left_group=contrast.left_group,
                    right_group=contrast.right_group,
                    metrics=_intervals_for_differences(
                        {
                            metric: left[metric] - right[metric]
                            for metric in _METRIC_NAMES
                        }
                    ),
                )
            )
        comparison_results.append(
            ComparisonBootstrapResult(
                name=comparison_name,
                base_model=base,
                challenger_model=challenger,
                overall=overall,
                by_group=tuple(group_results),
                contrasts=tuple(contrast_results),
            )
        )

    return JointBootstrapResult(
        cluster_count=len(cluster_ids),
        requested_replicates=replicates,
        seed=seed,
        ece_bins=ece_bins,
        stratified=stratum_by_cluster is not None,
        stratum_cluster_counts=tuple(
            (stratum, len(indices)) for stratum, indices in checked_strata
        ),
        group_order=checked_group_order,
        comparisons=tuple(comparison_results),
    )


def _validate_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _validate_examples(examples: Sequence[SparseExample]) -> None:
    row_ids: set[str] = set()
    cluster_by_decision: dict[str, str] = {}
    for example in examples:
        if not isinstance(example.row_id, str) or not example.row_id:
            raise ValueError("example row IDs must be non-empty")
        if example.row_id in row_ids:
            raise ValueError(f"duplicate example row ID: {example.row_id}")
        row_ids.add(example.row_id)
        if not isinstance(example.group_id, str) or not example.group_id:
            raise ValueError("example decision IDs must be non-empty")
        if not isinstance(example.cluster_id, str) or not example.cluster_id:
            raise ValueError("example cluster IDs must be non-empty")
        if not isinstance(example.label, bool):
            raise TypeError("example labels must be booleans")
        if not isfinite(example.weight) or example.weight <= 0:
            raise ValueError("example weights must be finite and positive")
        previous = cluster_by_decision.setdefault(
            example.group_id,
            example.cluster_id,
        )
        if previous != example.cluster_id:
            raise ValueError(
                f"decision spans multiple game clusters: {example.group_id}"
            )


def _validate_probability(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("predictions must be numeric probabilities")
    result = float(value)
    if not isfinite(result) or not 0 <= result <= 1:
        raise ValueError("predictions must be finite and between zero and one")
    return result


def _validate_predictions(
    examples: Sequence[SparseExample],
    predictions_by_model: Mapping[str, Sequence[float]],
) -> dict[str, tuple[float, ...]]:
    if not predictions_by_model:
        raise ValueError("predictions_by_model must be non-empty")
    checked = {}
    for name, predictions in predictions_by_model.items():
        if not isinstance(name, str) or not name:
            raise ValueError("model names must be non-empty strings")
        if len(predictions) != len(examples):
            raise ValueError(f"predictions for model {name!r} do not align")
        checked[name] = tuple(_validate_probability(value) for value in predictions)
    return checked


def _validate_comparisons(
    predictions_by_model: Mapping[str, Sequence[float]],
    comparisons: Mapping[str, tuple[str, str]],
) -> tuple[tuple[str, str, str], ...]:
    if not comparisons:
        raise ValueError("comparisons must be non-empty")
    checked = []
    for name, models in comparisons.items():
        if not isinstance(name, str) or not name:
            raise ValueError("comparison names must be non-empty strings")
        if not isinstance(models, (tuple, list)) or len(models) != 2:
            raise ValueError("each comparison must contain base and challenger")
        base, challenger = models
        if (
            not isinstance(base, str)
            or not base
            or not isinstance(challenger, str)
            or not challenger
        ):
            raise ValueError("comparison model names must be non-empty strings")
        if base not in predictions_by_model or challenger not in predictions_by_model:
            raise ValueError(f"comparison {name!r} references an unknown model")
        checked.append((name, base, challenger))
    return tuple(checked)


def _validate_groups(
    examples: Sequence[SparseExample],
    group_ids: Sequence[str] | None,
    group_order: Sequence[str],
    contrasts: Sequence[GroupContrast],
) -> tuple[tuple[str, ...] | None, tuple[str, ...]]:
    if group_ids is None:
        if group_order or contrasts:
            raise ValueError("group_order and contrasts require group_ids")
        return None, ()
    if len(group_ids) != len(examples):
        raise ValueError("group_ids must align with examples")
    checked_ids = tuple(group_ids)
    if any(not isinstance(group, str) or not group for group in checked_ids):
        raise ValueError("group IDs must be non-empty strings")
    if group_order:
        checked_order = tuple(group_order)
    else:
        checked_order = tuple(dict.fromkeys(checked_ids))
    if any(not isinstance(group, str) or not group for group in checked_order):
        raise ValueError("group_order values must be non-empty strings")
    if len(set(checked_order)) != len(checked_order):
        raise ValueError("group_order values must be unique")
    unknown = set(checked_ids) - set(checked_order)
    if unknown:
        raise ValueError(f"group_order omits observed groups: {sorted(unknown)!r}")

    group_by_decision: dict[str, str] = {}
    for example, group in zip(examples, checked_ids, strict=True):
        previous = group_by_decision.setdefault(example.group_id, group)
        if previous != group:
            raise ValueError(f"decision spans multiple subgroups: {example.group_id}")

    contrast_names: set[str] = set()
    known_groups = set(checked_order)
    for contrast in contrasts:
        if not isinstance(contrast, GroupContrast):
            raise TypeError("contrasts must contain GroupContrast values")
        if not contrast.name:
            raise ValueError("contrast names must be non-empty")
        if contrast.name in contrast_names:
            raise ValueError(f"duplicate contrast name: {contrast.name}")
        if contrast.name in known_groups:
            raise ValueError("contrast names must not duplicate subgroup names")
        contrast_names.add(contrast.name)
        if contrast.left_group not in known_groups:
            raise ValueError(f"unknown contrast group: {contrast.left_group}")
        if contrast.right_group not in known_groups:
            raise ValueError(f"unknown contrast group: {contrast.right_group}")
        if contrast.left_group == contrast.right_group:
            raise ValueError("contrast groups must be different")
    return checked_ids, checked_order


def _validate_strata(
    cluster_ids: Sequence[str],
    stratum_by_cluster: Mapping[str, str] | None,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if stratum_by_cluster is None:
        return (("all", tuple(range(len(cluster_ids)))),)
    expected = set(cluster_ids)
    actual = set(stratum_by_cluster)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"stratum cluster keys mismatch; missing={missing}, extra={extra}"
        )
    cluster_index = {cluster_id: index for index, cluster_id in enumerate(cluster_ids)}
    indices_by_stratum: defaultdict[str, list[int]] = defaultdict(list)
    for cluster_id in cluster_ids:
        stratum = stratum_by_cluster[cluster_id]
        if not isinstance(stratum, str) or not stratum:
            raise ValueError("stratum IDs must be non-empty strings")
        indices_by_stratum[stratum].append(cluster_index[cluster_id])
    return tuple(
        (stratum, tuple(indices_by_stratum[stratum]))
        for stratum in sorted(indices_by_stratum)
    )


def _scope_summary(
    examples: Sequence[SparseExample],
    predictions: Sequence[float],
    cluster_index: Mapping[str, int],
    ece_bins: int,
) -> _ScopeSummary:
    cluster_count = len(cluster_index)
    weights = np.zeros(cluster_count, dtype=np.float64)
    log_losses = np.zeros(cluster_count, dtype=np.float64)
    briers = np.zeros(cluster_count, dtype=np.float64)
    for example, prediction in zip(examples, predictions, strict=True):
        index = cluster_index[example.cluster_id]
        label = int(example.label)
        weights[index] += example.weight
        log_losses[index] += example.weight * _row_log_loss(label, prediction)
        briers[index] += example.weight * (prediction - label) ** 2
    macro_sums, macro_counts = _cluster_macro_summary(
        examples,
        predictions,
        cluster_index,
    )
    return _ScopeSummary(
        row_count=len(examples),
        decision_count=len({example.group_id for example in examples}),
        cluster_count=int(np.count_nonzero(weights)),
        cluster_weights=weights,
        cluster_log_losses=log_losses,
        cluster_briers=briers,
        auc=_cluster_auc_summary(examples, predictions, cluster_index)
        if examples
        else None,
        ece=_cluster_ece_summary(examples, predictions, cluster_index, ece_bins)
        if examples
        else None,
        macro_sums=macro_sums,
        macro_counts=macro_counts,
    )


def _row_log_loss(label: int, prediction: float) -> float:
    return -(
        label * log(max(prediction, 1e-15))
        + (1 - label) * log(max(1 - prediction, 1e-15))
    )


def _cluster_auc_summary(
    examples: Sequence[SparseExample],
    predictions: Sequence[float],
    cluster_index: Mapping[str, int],
) -> _AucSummary:
    global_clusters = np.fromiter(
        (cluster_index[example.cluster_id] for example in examples),
        dtype=np.intp,
        count=len(examples),
    )
    cluster_indices = np.unique(global_clusters)
    global_to_local = np.full(len(cluster_index), -1, dtype=np.intp)
    global_to_local[cluster_indices] = np.arange(len(cluster_indices), dtype=np.intp)
    local_clusters = global_to_local[global_clusters]
    prediction_values = np.asarray(predictions, dtype=np.float64)
    labels = np.fromiter(
        (example.label for example in examples),
        dtype=np.bool_,
        count=len(examples),
    )
    weights = np.fromiter(
        (example.weight for example in examples),
        dtype=np.float64,
        count=len(examples),
    )
    local_cluster_count = len(cluster_indices)
    positive_weights = np.bincount(
        local_clusters[labels],
        weights=weights[labels],
        minlength=local_cluster_count,
    ).astype(np.float64, copy=False)
    negative_weights = np.bincount(
        local_clusters[~labels],
        weights=weights[~labels],
        minlength=local_cluster_count,
    ).astype(np.float64, copy=False)
    matrix = np.zeros(
        (local_cluster_count, local_cluster_count),
        dtype=np.float64,
        order="C",
    )
    cumulative_negative = np.zeros(local_cluster_count, dtype=np.float64)
    order = np.argsort(prediction_values, kind="stable")
    ordered_predictions = prediction_values[order]
    ordered_clusters = local_clusters[order]
    ordered_labels = labels[order]
    ordered_weights = weights[order]
    position = 0
    while position < len(order):
        end = position + 1
        while (
            end < len(order)
            and ordered_predictions[end] == ordered_predictions[position]
        ):
            end += 1
        group_slice = slice(position, end)
        group_labels = ordered_labels[group_slice]
        positive_indices, positive_group_weights = _aggregate_cluster_weights(
            ordered_clusters[group_slice][group_labels],
            ordered_weights[group_slice][group_labels],
        )
        negative_indices, negative_group_weights = _aggregate_cluster_weights(
            ordered_clusters[group_slice][~group_labels],
            ordered_weights[group_slice][~group_labels],
        )
        for positive_index, positive_weight in zip(
            positive_indices,
            positive_group_weights,
            strict=True,
        ):
            matrix[positive_index, :] += positive_weight * cumulative_negative
            matrix[positive_index, negative_indices] += (
                0.5 * positive_weight * negative_group_weights
            )
        cumulative_negative[negative_indices] += negative_group_weights
        position = end
    return _AucSummary(
        cluster_indices=np.ascontiguousarray(cluster_indices),
        positive_weights=np.ascontiguousarray(positive_weights),
        negative_weights=np.ascontiguousarray(negative_weights),
        concordance_matrix=matrix,
    )


def _aggregate_cluster_weights(
    cluster_indices: NDArray[np.intp],
    weights: NDArray[np.float64],
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    if not len(cluster_indices):
        return (
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=np.float64),
        )
    unique, inverse = np.unique(cluster_indices, return_inverse=True)
    totals = np.bincount(inverse, weights=weights).astype(np.float64, copy=False)
    return unique, totals


def _auc_from_summary(
    summary: _AucSummary | None,
    multiplicities: NDArray[np.int32],
) -> NDArray[np.float64]:
    result = np.full(len(multiplicities), np.nan, dtype=np.float64)
    if summary is None:
        return result
    if len(summary.cluster_indices) == multiplicities.shape[1] and np.array_equal(
        summary.cluster_indices,
        np.arange(multiplicities.shape[1]),
    ):
        selected = multiplicities
    else:
        selected = multiplicities[:, summary.cluster_indices]
    positive_weight = selected @ summary.positive_weights
    negative_weight = selected @ summary.negative_weights
    denominator = positive_weight * negative_weight
    valid = denominator > 0
    if not np.any(valid):
        return result

    valid_positions = np.flatnonzero(valid)
    cluster_count = len(summary.cluster_indices)
    # Bound each temporary (draw batch @ AUC matrix) product to about 32 MiB.
    target_product_bytes = 32 * 1024 * 1024
    batch_size = max(
        1,
        min(
            len(valid_positions),
            target_product_bytes // max(cluster_count * 8, 1),
        ),
    )
    for start in range(0, len(valid_positions), batch_size):
        positions = valid_positions[start : start + batch_size]
        if len(valid_positions) == len(selected):
            raw_block = selected[start : start + batch_size, :]
        else:
            raw_block = selected[positions, :]
        block = np.asarray(raw_block, dtype=np.float64)
        product = block @ summary.concordance_matrix
        numerator = np.einsum("ij,ij->i", product, block, optimize=True)
        result[positions] = numerator / denominator[positions]
    return result


def _cluster_ece_summary(
    examples: Sequence[SparseExample],
    predictions: Sequence[float],
    cluster_index: Mapping[str, int],
    bins: int,
) -> _EceSummary:
    weights = np.zeros((len(cluster_index), bins), dtype=np.float64)
    prediction_sums = np.zeros_like(weights)
    label_sums = np.zeros_like(weights)
    for example, prediction in zip(examples, predictions, strict=True):
        cluster = cluster_index[example.cluster_id]
        bin_index = min(int(prediction * bins), bins - 1)
        weights[cluster][bin_index] += example.weight
        prediction_sums[cluster][bin_index] += example.weight * prediction
        label_sums[cluster][bin_index] += example.weight * int(example.label)
    return _EceSummary(
        weights=weights,
        prediction_sums=prediction_sums,
        label_sums=label_sums,
    )


def _ece_from_summary(
    summary: _EceSummary | None,
    multiplicities: NDArray[np.int32],
) -> NDArray[np.float64]:
    result = np.full(len(multiplicities), np.nan, dtype=np.float64)
    if summary is None:
        return result
    bin_weights = multiplicities @ summary.weights
    prediction_sums = multiplicities @ summary.prediction_sums
    label_sums = multiplicities @ summary.label_sums
    total_weights = np.sum(bin_weights, axis=1)
    valid_rows = total_weights > 0
    contributions = np.zeros_like(bin_weights)
    populated = bin_weights > 0
    contributions[populated] = np.abs(
        prediction_sums[populated] / bin_weights[populated]
        - label_sums[populated] / bin_weights[populated]
    )
    result[valid_rows] = (
        np.sum(
            bin_weights[valid_rows] * contributions[valid_rows],
            axis=1,
        )
        / total_weights[valid_rows]
    )
    return result


def _cluster_macro_summary(
    examples: Sequence[SparseExample],
    predictions: Sequence[float],
    cluster_index: Mapping[str, int],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    groups: defaultdict[str, list[tuple[bool, float, str]]] = defaultdict(list)
    for example, prediction in zip(examples, predictions, strict=True):
        groups[example.group_id].append((example.label, prediction, example.cluster_id))
    sums = np.zeros(len(cluster_index), dtype=np.float64)
    counts = np.zeros(len(cluster_index), dtype=np.int64)
    for group_id, values in groups.items():
        cluster_ids = {value[2] for value in values}
        if len(cluster_ids) != 1:
            raise ValueError(f"decision spans multiple game clusters: {group_id}")
        positives = [value[1] for value in values if value[0]]
        negatives = [value[1] for value in values if not value[0]]
        if not positives or not negatives:
            continue
        pair_scores = [
            1.0 if positive > negative else 0.5 if positive == negative else 0.0
            for positive in positives
            for negative in negatives
        ]
        index = cluster_index[next(iter(cluster_ids))]
        sums[index] += sum(pair_scores) / len(pair_scores)
        counts[index] += 1
    return sums, counts


def _metrics_from_summary(
    summary: _ScopeSummary,
    multiplicities: NDArray[np.int32],
) -> dict[str, NDArray[np.float64]]:
    total_weights = multiplicities @ summary.cluster_weights
    valid = total_weights > 0
    log_losses = np.full(len(multiplicities), np.nan, dtype=np.float64)
    briers = np.full(len(multiplicities), np.nan, dtype=np.float64)
    log_losses[valid] = (
        multiplicities[valid] @ summary.cluster_log_losses
    ) / total_weights[valid]
    briers[valid] = (multiplicities[valid] @ summary.cluster_briers) / total_weights[
        valid
    ]
    macro_counts = multiplicities @ summary.macro_counts
    macro = np.full(len(multiplicities), np.nan, dtype=np.float64)
    valid_macro = macro_counts > 0
    macro[valid_macro] = (
        multiplicities[valid_macro] @ summary.macro_sums
    ) / macro_counts[valid_macro]
    return {
        "auc": _auc_from_summary(summary.auc, multiplicities),
        "log_loss": log_losses,
        "brier": briers,
        "ece": _ece_from_summary(summary.ece, multiplicities),
        "macro_concordance": macro,
    }


def _draw_multiplicity_matrix(
    cluster_count: int,
    strata: Sequence[tuple[str, tuple[int, ...]]],
    replicates: int,
    random: Random,
) -> NDArray[np.int32]:
    multiplicities = np.zeros((replicates, cluster_count), dtype=np.int32)
    for replicate in range(replicates):
        for _stratum, indices in strata:
            draws = np.fromiter(
                (
                    indices[random.randrange(len(indices))]
                    for _draw in range(len(indices))
                ),
                dtype=np.intp,
                count=len(indices),
            )
            np.add.at(multiplicities[replicate], draws, 1)
    return multiplicities


def _intervals_for_differences(
    differences: Mapping[str, NDArray[np.float64]],
) -> MetricBootstrapIntervals:
    return MetricBootstrapIntervals(
        **{
            metric: _bootstrap_interval(
                _finite_float_or_none(differences[metric][0]),
                differences[metric][1:][np.isfinite(differences[metric][1:])],
            )
            for metric in _METRIC_NAMES
        }
    )


def _finite_float_or_none(value: np.float64) -> float | None:
    return float(value) if np.isfinite(value) else None


def _bootstrap_interval(
    point: float | None,
    values: Sequence[float],
) -> BootstrapInterval:
    if len(values) == 0:
        return BootstrapInterval(
            point=point, lower=None, upper=None, valid_replicates=0
        )
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    return BootstrapInterval(
        point=point,
        lower=float(_percentile(ordered, 0.025)),
        upper=float(_percentile(ordered, 0.975)),
        valid_replicates=len(ordered),
    )


def _percentile(ordered: Sequence[float], probability: float) -> float:
    position = probability * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] * (1 - fraction) + ordered[upper_index] * fraction
