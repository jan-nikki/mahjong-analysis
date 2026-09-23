from __future__ import annotations

import math
from itertools import product

import pytest

from mahjong_analysis.combo_prediction import (
    DECISION_TURN_BINS,
    SparseExample,
    calibration_intercept_slope,
    decision_turn_bin,
    evaluate_binary_predictions,
    evaluate_calibration,
    fit_sparse_logistic,
    paired_game_cluster_bootstrap,
    visibility_local_saturated_features,
)
from mahjong_analysis.combo_theory import calculate_simple_combo, sequence_wait_patterns
from mahjong_analysis.tiles import TILE_KINDS, tile_to_index


def example(
    group: str,
    label: bool,
    x: float,
    *,
    weight: float = 1.0,
) -> SparseExample:
    return SparseExample(
        row_id=f"{group}:x={x}:label={int(label)}",
        group_id=group,
        cluster_id=group.split(":", maxsplit=1)[0],
        label=label,
        weight=weight,
        features=(("x", x),),
    )


def test_weighted_metrics_are_exact_for_perfect_predictions() -> None:
    examples = (
        example("a", False, -1, weight=0.5),
        example("a", True, 1, weight=0.5),
        example("b", False, -1, weight=0.5),
        example("b", True, 1, weight=0.5),
    )

    result = evaluate_binary_predictions(examples, (0.1, 0.9, 0.1, 0.9))

    assert result.row_count == 4
    assert result.decision_count == 2
    assert result.positive_rate == 0.5
    assert result.auc == 1.0
    assert result.macro_concordance == 1.0
    assert result.macro_concordance_decisions == 2
    assert result.log_loss == pytest.approx(-math.log(0.9))
    assert result.brier == pytest.approx(0.01)
    assert result.ece == pytest.approx(0.1)


def test_macro_concordance_excludes_single_label_decisions() -> None:
    examples = (
        example("mixed", False, -1),
        example("mixed", True, 1),
        example("negative-only", False, -1),
        example("negative-only", False, -2),
    )

    result = evaluate_binary_predictions(examples, (0.2, 0.8, 0.1, 0.3))

    assert result.macro_concordance == 1.0
    assert result.macro_concordance_decisions == 1


def test_tied_predictions_receive_half_auc_and_concordance_credit() -> None:
    examples = (example("a", False, 0), example("a", True, 0))

    result = evaluate_binary_predictions(examples, (0.5, 0.5))

    assert result.auc == 0.5
    assert result.macro_concordance == 0.5


def test_calibration_intercept_and_slope_recover_known_recalibration() -> None:
    labels = (True, False, False, False, False, True, True, True, True, False)
    examples = tuple(
        example(f"g{index}", label, 0.0) for index, label in enumerate(labels)
    )
    calibrated = (0.2,) * 5 + (0.8,) * 5
    overconfident = (1 / 17,) * 5 + (16 / 17,) * 5

    intercept, slope = calibration_intercept_slope(examples, calibrated)
    compressed_intercept, compressed_slope = calibration_intercept_slope(
        examples,
        overconfident,
    )

    assert intercept == pytest.approx(0.0, abs=1e-8)
    assert slope == pytest.approx(1.0, abs=1e-8)
    assert compressed_intercept == pytest.approx(0.0, abs=1e-8)
    assert compressed_slope == pytest.approx(0.5, abs=1e-8)


def test_calibration_reports_unidentifiable_cases_as_missing() -> None:
    mixed = (example("a", False, 0), example("b", True, 0))
    one_class = (example("a", True, 0), example("b", True, 0))

    assert calibration_intercept_slope(mixed, (0.5, 0.5)) == (None, None)
    assert calibration_intercept_slope(one_class, (0.2, 0.8)) == (None, None)


def test_calibration_recovers_fractional_weight_parameters_and_weight_scale() -> None:
    low_prediction = 1 / (1 + math.exp(1))
    high_prediction = 1 / (1 + math.exp(-1))
    low_observed = 1 / (1 + math.exp(1.0))
    high_observed = 1 / (1 + math.exp(-0.2))

    def weighted_examples(scale: float) -> tuple[SparseExample, ...]:
        return (
            example("a", True, 0, weight=scale * low_observed),
            example("a", False, 0, weight=scale * (1 - low_observed)),
            example("b", True, 0, weight=scale * high_observed),
            example("b", False, 0, weight=scale * (1 - high_observed)),
        )

    predictions = (low_prediction, low_prediction, high_prediction, high_prediction)
    first = evaluate_calibration(weighted_examples(1.0), predictions)
    scaled = evaluate_calibration(weighted_examples(7.0), predictions)

    assert first.joint_status == "ok"
    assert first.recalibration_intercept == pytest.approx(-0.4, abs=1e-8)
    assert first.recalibration_slope == pytest.approx(0.6, abs=1e-8)
    assert scaled.recalibration_intercept == pytest.approx(
        first.recalibration_intercept,
        abs=1e-8,
    )
    assert scaled.recalibration_slope == pytest.approx(
        first.recalibration_slope,
        abs=1e-8,
    )


@pytest.mark.parametrize(
    ("labels", "predictions", "expected_status"),
    [
        ((False, False, True, True), (0.2, 0.3, 0.7, 0.8), "complete_separation"),
        ((True, True, False, False), (0.2, 0.3, 0.7, 0.8), "complete_separation"),
        ((False, False, True, True), (0.2, 0.5, 0.5, 0.8), "quasi_separation"),
    ],
)
def test_calibration_detects_nonfinite_separation(
    labels: tuple[bool, ...],
    predictions: tuple[float, ...],
    expected_status: str,
) -> None:
    examples = tuple(
        example(f"g{index}", label, 0) for index, label in enumerate(labels)
    )

    result = evaluate_calibration(examples, predictions)

    assert result.joint_status == expected_status
    assert result.recalibration_intercept is None
    assert result.recalibration_slope is None
    assert result.calibration_in_the_large is not None


def test_calibration_clips_probability_endpoints_and_reports_counts() -> None:
    labels = (False, True, False, True)
    examples = tuple(
        example(f"g{index}", label, 0) for index, label in enumerate(labels)
    )

    result = evaluate_calibration(examples, (0.0, 0.0, 1.0, 1.0))

    assert result.joint_status == "ok"
    assert result.clipped_low == 2
    assert result.clipped_high == 2
    assert result.recalibration_intercept == pytest.approx(0.0, abs=1e-8)
    assert result.recalibration_slope == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("weight", [0.0, -1.0, float("nan"), float("inf")])
def test_calibration_rejects_nonpositive_or_nonfinite_weights(weight: float) -> None:
    examples = (
        example("a", False, 0, weight=weight),
        example("b", True, 0),
    )

    with pytest.raises(ValueError, match="finite and positive"):
        evaluate_calibration(examples, (0.2, 0.8))


def test_sparse_logistic_learns_direction_and_reduces_log_loss() -> None:
    examples = tuple(
        example(f"g{index}", x > 0, x)
        for index, x in enumerate((-3.0, -2.0, -1.0, 1.0, 2.0, 3.0))
    )
    baseline = evaluate_binary_predictions(examples, (0.5,) * len(examples))

    model = fit_sparse_logistic(
        examples,
        epochs=200,
        learning_rate=1.0,
        l2=0.0,
    )
    predictions = model.predict(examples)
    fitted = evaluate_binary_predictions(examples, predictions)

    assert model.weights["x"] > 0
    assert predictions[0] < predictions[-1]
    assert fitted.log_loss < baseline.log_loss


def test_visibility_local_saturated_features_span_seven_man_combo() -> None:
    unseen = {
        "5m": 1,
        "6m": 2,
        "7m": 3,
        "8m": 4,
        "9m": 1,
    }

    features = dict(visibility_local_saturated_features("7m", unseen))

    assert features == {
        "visibility_local:candidate_count:3": 1.0,
        "visibility_local:pair:-2,-1:counts:1,2": 1.0,
        "visibility_local:pair:-1,+1:counts:2,4": 1.0,
        "visibility_local:pair:+1,+2:counts:4,1": 1.0,
    }
    reconstructed_combo = 1 * 2 + 2 * 4 + 4 * 1 + 3 + 3
    assert reconstructed_combo == 20


def test_visibility_local_saturated_features_span_every_simple_combo_state() -> None:
    """The saturated state contains the simple score exactly, not approximately."""

    for candidate in TILE_KINDS:
        candidate_index = tile_to_index(candidate)
        local_indices = {candidate_index}
        for pattern in sequence_wait_patterns(candidate):
            local_indices.update(tile_to_index(tile) for tile in pattern.tiles)
        ordered_indices = tuple(sorted(local_indices))

        for local_counts in product(range(5), repeat=len(ordered_indices)):
            unseen = [0] * len(TILE_KINDS)
            for index, count in zip(ordered_indices, local_counts, strict=True):
                unseen[index] = count

            reconstructed_sequence = 0
            reconstructed_shanpon = 0
            reconstructed_tanki = 0
            for name, value in visibility_local_saturated_features(
                candidate, tuple(unseen)
            ):
                if name.startswith("visibility_local:candidate_count:"):
                    count = int(name.rsplit(":", maxsplit=1)[1])
                    reconstructed_shanpon += int(value * math.comb(count, 2))
                    reconstructed_tanki += int(value * count)
                else:
                    first, second = (
                        int(count)
                        for count in name.rsplit(":", maxsplit=1)[1].split(",")
                    )
                    reconstructed_sequence += int(value * first * second)

            simple = calculate_simple_combo(candidate, tuple(unseen))
            assert reconstructed_sequence == sum(
                combo.count for combo in simple.sequence_combos
            )
            assert reconstructed_shanpon == simple.shanpon_combos
            assert reconstructed_tanki == simple.tanki_combos
            assert (
                reconstructed_sequence
                + reconstructed_shanpon
                + reconstructed_tanki
                == simple.total
            )


@pytest.mark.parametrize(
    ("turn", "expected"),
    [
        (1, "1-6"),
        (6, "1-6"),
        (7, "7-9"),
        (9, "7-9"),
        (10, "10-12"),
        (12, "10-12"),
        (13, "13+"),
        (30, "13+"),
    ],
)
def test_decision_turn_bins_have_fixed_boundaries(turn: int, expected: str) -> None:
    assert DECISION_TURN_BINS == ("1-6", "7-9", "10-12", "13+")
    assert decision_turn_bin(turn) == expected


@pytest.mark.parametrize("turn", [0, -1])
def test_decision_turn_bin_rejects_nonpositive_values(turn: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        decision_turn_bin(turn)


@pytest.mark.parametrize("turn", [True, 1.5, "6"])
def test_decision_turn_bin_rejects_nonintegers(turn: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        decision_turn_bin(turn)  # type: ignore[arg-type]


def test_game_cluster_bootstrap_is_reproducible_and_paired() -> None:
    examples = (
        example("game-a:1", False, -1),
        example("game-a:1", True, 1),
        example("game-b:1", False, -1),
        example("game-b:1", True, 1),
    )
    base = (0.4, 0.6, 0.4, 0.6)
    challenger = (0.2, 0.8, 0.2, 0.8)

    first = paired_game_cluster_bootstrap(
        examples,
        base,
        challenger,
        replicates=20,
        seed=7,
    )
    second = paired_game_cluster_bootstrap(
        examples,
        base,
        challenger,
        replicates=20,
        seed=7,
    )

    assert first == second
    assert first.cluster_count == 2
    assert first.auc.point == 0.0
    assert first.auc.lower == pytest.approx(first.auc.point)
    assert first.auc.upper == pytest.approx(first.auc.point)
    assert first.log_loss.point is not None and first.log_loss.point < 0
    assert first.log_loss.lower == pytest.approx(first.log_loss.point)
    assert first.log_loss.upper == pytest.approx(first.log_loss.point)
    assert first.brier.point is not None and first.brier.point < 0
    assert first.log_loss.valid_replicates == 20
    assert first.macro_concordance.point == 0.0
    assert first.macro_concordance.lower == 0.0
    assert first.macro_concordance.upper == 0.0


@pytest.mark.parametrize("prediction", [-0.1, 1.1, float("nan")])
def test_metrics_reject_invalid_predictions(prediction: float) -> None:
    with pytest.raises(ValueError):
        evaluate_binary_predictions((example("a", True, 1),), (prediction,))
