from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite

import pytest

from mahjong_analysis.sparse_logistic import (
    _objective_and_gradient,
    _prepare_examples,
    fit_sparse_logistic,
)


@dataclass(frozen=True)
class Example:
    features: tuple[tuple[str, float], ...]
    label: bool
    weight: float = 1.0


def _example(
    x: float,
    label: bool,
    *,
    weight: float = 1.0,
    extra_features: tuple[tuple[str, float], ...] = (),
) -> Example:
    return Example(
        features=(("x", x), *extra_features),
        label=label,
        weight=weight,
    )


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def test_analytic_gradient_matches_finite_differences() -> None:
    examples = (
        Example((("b", -0.5), ("a", 2.0)), False, 0.7),
        Example((("a", -1.0), ("b", 1.5)), True, 2.0),
        Example((("a", 0.25), ("a", 0.75)), True, 1.3),
    )
    dataset = _prepare_examples(examples)
    parameters = [0.2, -0.3, 0.4]
    l2 = 0.17

    _, analytic = _objective_and_gradient(dataset, parameters, l2)
    finite_difference = []
    step = 1e-6
    for index in range(len(parameters)):
        lower = list(parameters)
        upper = list(parameters)
        lower[index] -= step
        upper[index] += step
        lower_objective, _ = _objective_and_gradient(dataset, lower, l2)
        upper_objective, _ = _objective_and_gradient(dataset, upper, l2)
        finite_difference.append((upper_objective - lower_objective) / (2.0 * step))

    assert analytic == pytest.approx(finite_difference, rel=2e-8, abs=2e-9)


def test_recovers_known_intercept_and_coefficient() -> None:
    expected_intercept = -0.4
    expected_coefficient = 0.85
    examples = []
    for x in (-2.0, -1.0, 0.0, 1.0, 2.0):
        probability = _sigmoid(expected_intercept + expected_coefficient * x)
        # The two weighted binary rows express an exact fractional response at
        # each x without requiring non-binary labels.
        examples.append(_example(x, True, weight=probability))
        examples.append(_example(x, False, weight=1.0 - probability))

    model = fit_sparse_logistic(
        examples,
        l2=0.0,
        tolerance=1e-10,
        max_iterations=100,
    )

    assert model.converged
    assert model.intercept == pytest.approx(expected_intercept, abs=2e-7)
    assert model.weights == {"x": pytest.approx(expected_coefficient, abs=2e-7)}


def test_common_weight_scale_does_not_change_fit() -> None:
    examples = (
        _example(-2.0, False, weight=0.5),
        _example(-1.0, False, weight=1.5),
        _example(0.5, True, weight=0.75),
        _example(2.0, True, weight=2.25),
    )
    scaled = tuple(
        Example(row.features, row.label, row.weight * 32.0) for row in examples
    )

    original_model = fit_sparse_logistic(examples, l2=0.2, tolerance=1e-11)
    scaled_model = fit_sparse_logistic(scaled, l2=0.2, tolerance=1e-11)

    assert scaled_model.intercept == pytest.approx(original_model.intercept, abs=1e-12)
    assert scaled_model.weights == pytest.approx(original_model.weights, abs=1e-12)
    assert scaled_model.final_objective == pytest.approx(
        original_model.final_objective,
        abs=1e-14,
    )


def test_row_and_feature_order_do_not_change_fit() -> None:
    examples = (
        Example((("z", 1.0), ("x", -2.0)), False, 0.5),
        Example((("x", -1.0), ("z", 0.5)), False, 1.5),
        Example((("z", -0.5), ("x", 0.5)), True, 0.75),
        Example((("x", 2.0), ("z", -1.0)), True, 2.25),
    )
    reordered = tuple(
        Example(tuple(reversed(row.features)), row.label, row.weight)
        for row in reversed(examples)
    )

    first = fit_sparse_logistic(examples, l2=0.2, tolerance=1e-11)
    second = fit_sparse_logistic(reordered, l2=0.2, tolerance=1e-11)

    assert first == second


def test_duplicate_features_are_summed_within_each_row() -> None:
    duplicated = (
        Example((("x", -3.0), ("x", 1.0)), False),
        Example((("x", -0.25), ("x", -0.75)), False),
        Example((("x", 0.75), ("x", -0.25)), True),
        Example((("x", 1.5), ("x", 0.5)), True),
    )
    combined = (
        _example(-2.0, False),
        _example(-1.0, False),
        _example(0.5, True),
        _example(2.0, True),
    )

    duplicate_model = fit_sparse_logistic(duplicated, l2=0.1, tolerance=1e-11)
    combined_model = fit_sparse_logistic(combined, l2=0.1, tolerance=1e-11)

    assert duplicate_model == combined_model
    assert duplicate_model.predict_features((("x", 0.25), ("x", 0.75))) == (
        pytest.approx(duplicate_model.predict_features((("x", 1.0),)))
    )


def test_reports_successful_optimizer_diagnostics() -> None:
    examples = (
        _example(-2.0, False),
        _example(-1.0, False),
        _example(1.0, True),
        _example(2.0, True),
    )

    model = fit_sparse_logistic(examples, l2=0.3, tolerance=1e-9)

    assert model.converged
    assert model.termination_reason == "gradient_tolerance"
    assert 0 < model.iterations_run <= 250
    assert isfinite(model.final_objective)
    assert model.final_largest_gradient <= 1e-9
    assert model.line_search_evaluations >= model.iterations_run
    assert model.skipped_curvature_updates >= 0


def test_reports_iteration_limit() -> None:
    examples = (
        _example(-3.0, False),
        _example(-1.0, False),
        _example(0.5, True),
        _example(2.0, True),
    )

    model = fit_sparse_logistic(
        examples,
        l2=0.1,
        tolerance=0.0,
        max_iterations=1,
    )

    assert not model.converged
    assert model.termination_reason == "max_iterations"
    assert model.iterations_run == 1
    assert model.line_search_evaluations >= 1


def test_repeated_fits_are_bitwise_deterministic() -> None:
    examples = (
        _example(-2.0, False, extra_features=(("z", 0.5),)),
        _example(-0.5, True, extra_features=(("z", -1.0),)),
        _example(0.25, False, extra_features=(("z", 2.0),)),
        _example(1.5, True, extra_features=(("z", -0.25),)),
    )

    first = fit_sparse_logistic(examples, l2=0.05)
    second = fit_sparse_logistic(examples, l2=0.05)

    assert first == second


@pytest.mark.parametrize("weight", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_invalid_weights(weight: float) -> None:
    with pytest.raises(ValueError, match="weight"):
        fit_sparse_logistic((_example(1.0, True, weight=weight),))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_feature_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        fit_sparse_logistic((Example((("x", value),), True),))
