"""Dependency-free sparse binary logistic regression fitted with L-BFGS.

The learner deliberately has a small, duck-typed interface: every training
example only needs ``features``, ``label``, and ``weight`` attributes.  The
objective is weighted mean binary log loss plus ``l2 / 2 * ||weights||^2``;
the intercept is not penalized.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import exp, fsum, inf, isfinite, log, log1p, sqrt
from typing import Literal, Protocol

TerminationReason = Literal[
    "gradient_tolerance",
    "max_iterations",
    "line_search_failed",
]

_INITIAL_PROBABILITY_CLIP = 1e-12
_CURVATURE_EPSILON = 1e-10


class SparseLogisticExample(Protocol):
    """Structural type accepted by :func:`fit_sparse_logistic`."""

    features: Iterable[tuple[str, float]]
    label: bool | int | float
    weight: float


@dataclass(frozen=True)
class SparseLogisticModel:
    """A fitted sparse logistic model and its optimizer diagnostics."""

    intercept: float
    weights: dict[str, float]
    converged: bool
    termination_reason: TerminationReason
    iterations_run: int
    final_objective: float
    final_largest_gradient: float
    line_search_evaluations: int
    skipped_curvature_updates: int

    def predict_features(
        self,
        features: Iterable[tuple[str, float]] | Mapping[str, float],
    ) -> float:
        """Predict one probability from a sparse named feature vector."""
        items = features.items() if isinstance(features, Mapping) else features
        linear = self.intercept
        for item in items:
            try:
                name, raw_value = item
            except (TypeError, ValueError) as error:
                raise ValueError("features must contain (name, value) pairs") from error
            value = _finite_feature_value(raw_value)
            linear += self.weights.get(name, 0.0) * value
        if not isfinite(linear):
            raise ValueError("the linear prediction must be finite")
        return _sigmoid(linear)

    def predict(self, examples: Iterable[SparseLogisticExample]) -> tuple[float, ...]:
        """Predict probabilities for examples exposing a ``features`` attribute."""
        return tuple(self.predict_features(example.features) for example in examples)


@dataclass(frozen=True)
class _RawRow:
    label: float
    weight: float
    features: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class _PreparedRow:
    label: float
    weight: float
    features: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class _PreparedDataset:
    feature_names: tuple[str, ...]
    rows: tuple[_PreparedRow, ...]


@dataclass(frozen=True)
class _HistoryEntry:
    parameter_delta: tuple[float, ...]
    gradient_delta: tuple[float, ...]
    inverse_curvature: float


def fit_sparse_logistic(
    examples: Iterable[SparseLogisticExample],
    *,
    l2: float = 0.001,
    max_iterations: int = 250,
    tolerance: float = 1e-7,
    history_size: int = 10,
    armijo_constant: float = 1e-4,
    max_line_search_evaluations: int = 50,
) -> SparseLogisticModel:
    """Fit weighted binary logistic regression with deterministic L-BFGS.

    Feature names and rows are put in canonical order, and duplicate feature
    names within a row are summed.  Consequently, fitting is invariant to row
    order, feature order, and a common rescaling of all example weights (up to
    ordinary floating-point rounding).
    """
    checked_l2 = _finite_real(l2, "l2")
    checked_tolerance = _finite_real(tolerance, "tolerance")
    checked_armijo = _finite_real(armijo_constant, "armijo_constant")
    _positive_integer(max_iterations, "max_iterations")
    _positive_integer(history_size, "history_size")
    _positive_integer(
        max_line_search_evaluations,
        "max_line_search_evaluations",
    )
    if checked_l2 < 0:
        raise ValueError("l2 must be nonnegative")
    if checked_tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    if not 0 < checked_armijo < 1:
        raise ValueError("armijo_constant must be strictly between zero and one")

    dataset = _prepare_examples(examples)
    positive_rate = fsum(row.weight * row.label for row in dataset.rows)
    clipped_rate = min(
        max(positive_rate, _INITIAL_PROBABILITY_CLIP),
        1.0 - _INITIAL_PROBABILITY_CLIP,
    )
    parameters = [
        log(clipped_rate / (1.0 - clipped_rate)),
        *([0.0] * len(dataset.feature_names)),
    ]
    objective, gradient = _objective_and_gradient(
        dataset,
        parameters,
        checked_l2,
    )
    largest_gradient = _largest_absolute(gradient)
    if largest_gradient <= checked_tolerance:
        return _make_model(
            dataset,
            parameters,
            converged=True,
            termination_reason="gradient_tolerance",
            iterations_run=0,
            final_objective=objective,
            final_largest_gradient=largest_gradient,
            line_search_evaluations=0,
            skipped_curvature_updates=0,
        )

    history: list[_HistoryEntry] = []
    iterations_run = 0
    line_search_evaluations = 0
    skipped_curvature_updates = 0

    for iteration in range(1, max_iterations + 1):
        direction = _lbfgs_direction(gradient, history)
        directional_derivative = _dot(gradient, direction)
        if not isfinite(directional_derivative) or directional_derivative >= 0:
            # Roundoff can occasionally spoil a quasi-Newton direction.  A
            # steepest-descent restart preserves Armijo's descent guarantee.
            history.clear()
            direction = [-value for value in gradient]
            directional_derivative = -_dot(gradient, gradient)

        accepted: tuple[list[float], float, list[float]] | None = None
        step = 1.0
        for _ in range(max_line_search_evaluations):
            trial_parameters = [
                value + step * delta
                for value, delta in zip(parameters, direction, strict=True)
            ]
            trial_objective, trial_gradient = _objective_and_gradient(
                dataset,
                trial_parameters,
                checked_l2,
            )
            line_search_evaluations += 1
            armijo_bound = objective + checked_armijo * step * directional_derivative
            if isfinite(trial_objective) and trial_objective <= armijo_bound:
                accepted = (
                    trial_parameters,
                    trial_objective,
                    trial_gradient,
                )
                break
            step *= 0.5

        if accepted is None:
            return _make_model(
                dataset,
                parameters,
                converged=False,
                termination_reason="line_search_failed",
                iterations_run=iterations_run,
                final_objective=objective,
                final_largest_gradient=largest_gradient,
                line_search_evaluations=line_search_evaluations,
                skipped_curvature_updates=skipped_curvature_updates,
            )

        next_parameters, next_objective, next_gradient = accepted
        parameter_delta = [
            new - old for new, old in zip(next_parameters, parameters, strict=True)
        ]
        gradient_delta = [
            new - old for new, old in zip(next_gradient, gradient, strict=True)
        ]
        curvature = _dot(parameter_delta, gradient_delta)
        delta_norm_product = sqrt(
            max(0.0, _dot(parameter_delta, parameter_delta))
            * max(0.0, _dot(gradient_delta, gradient_delta))
        )
        if isfinite(curvature) and curvature > _CURVATURE_EPSILON * delta_norm_product:
            history.append(
                _HistoryEntry(
                    parameter_delta=tuple(parameter_delta),
                    gradient_delta=tuple(gradient_delta),
                    inverse_curvature=1.0 / curvature,
                )
            )
            if len(history) > history_size:
                del history[0]
        else:
            skipped_curvature_updates += 1

        parameters = next_parameters
        objective = next_objective
        gradient = next_gradient
        iterations_run = iteration
        largest_gradient = _largest_absolute(gradient)
        if largest_gradient <= checked_tolerance:
            return _make_model(
                dataset,
                parameters,
                converged=True,
                termination_reason="gradient_tolerance",
                iterations_run=iterations_run,
                final_objective=objective,
                final_largest_gradient=largest_gradient,
                line_search_evaluations=line_search_evaluations,
                skipped_curvature_updates=skipped_curvature_updates,
            )

    return _make_model(
        dataset,
        parameters,
        converged=False,
        termination_reason="max_iterations",
        iterations_run=iterations_run,
        final_objective=objective,
        final_largest_gradient=largest_gradient,
        line_search_evaluations=line_search_evaluations,
        skipped_curvature_updates=skipped_curvature_updates,
    )


def _prepare_examples(
    examples: Iterable[SparseLogisticExample],
) -> _PreparedDataset:
    raw_rows = []
    for example_index, example in enumerate(examples):
        try:
            raw_label = example.label
            raw_weight = example.weight
            raw_features = example.features
        except AttributeError as error:
            raise TypeError(
                "each example must have features, label, and weight attributes"
            ) from error
        label = _binary_label(raw_label, example_index)
        weight = _finite_real(raw_weight, f"example {example_index} weight")
        if weight <= 0:
            raise ValueError("example weights must be finite and positive")
        features = _canonical_features(raw_features, example_index)
        raw_rows.append(_RawRow(label=label, weight=weight, features=features))

    if not raw_rows:
        raise ValueError("at least one training example is required")

    try:
        total_weight = fsum(sorted(row.weight for row in raw_rows))
    except OverflowError as error:
        raise ValueError("the total example weight must be finite") from error
    if not isfinite(total_weight) or total_weight <= 0:
        raise ValueError("the total example weight must be finite and positive")

    feature_names = tuple(
        sorted({name for row in raw_rows for name, _ in row.features})
    )
    feature_indices = {name: index for index, name in enumerate(feature_names)}
    # Canonical row ordering removes accumulation-order differences when the
    # caller supplies the same examples in another order.
    raw_rows.sort(key=lambda row: (row.features, row.label, row.weight))
    rows = tuple(
        _PreparedRow(
            label=row.label,
            weight=row.weight / total_weight,
            features=tuple(
                (feature_indices[name], value) for name, value in row.features
            ),
        )
        for row in raw_rows
    )
    return _PreparedDataset(feature_names=feature_names, rows=rows)


def _canonical_features(
    features: Iterable[tuple[str, float]] | Mapping[str, float],
    example_index: int,
) -> tuple[tuple[str, float], ...]:
    items = features.items() if isinstance(features, Mapping) else features
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    try:
        iterator = iter(items)
    except TypeError as error:
        raise TypeError(f"example {example_index} features must be iterable") from error
    for feature_index, item in enumerate(iterator):
        try:
            name, raw_value = item
        except (TypeError, ValueError) as error:
            raise ValueError("features must contain (name, value) pairs") from error
        if not isinstance(name, str) or not name:
            raise ValueError("feature names must be non-empty strings")
        grouped[name].append(
            _finite_feature_value(
                raw_value,
                context=(f"example {example_index} feature {feature_index} value"),
            )
        )

    combined = []
    for name in sorted(grouped):
        try:
            value = fsum(sorted(grouped[name]))
        except OverflowError as error:
            raise ValueError("summed feature values must be finite") from error
        if not isfinite(value):
            raise ValueError("summed feature values must be finite")
        if value != 0.0:
            combined.append((name, value))
    return tuple(combined)


def _objective_and_gradient(
    dataset: _PreparedDataset,
    parameters: Sequence[float],
    l2: float,
) -> tuple[float, list[float]]:
    """Evaluate the documented objective and its analytic gradient."""
    dimension = len(dataset.feature_names) + 1
    if len(parameters) != dimension:
        raise ValueError("parameter vector has the wrong dimension")
    if any(not isfinite(value) for value in parameters):
        return inf, [inf] * dimension

    gradient = [0.0] * dimension
    for index in range(1, dimension):
        gradient[index] = l2 * parameters[index]
    loss_terms = []
    for row in dataset.rows:
        linear = parameters[0]
        for feature_index, feature_value in row.features:
            linear += parameters[feature_index + 1] * feature_value
        if not isfinite(linear):
            return inf, [inf] * dimension
        loss = _softplus(-linear) if row.label == 1.0 else _softplus(linear)
        loss_terms.append(row.weight * loss)
        error = row.weight * (_sigmoid(linear) - row.label)
        gradient[0] += error
        for feature_index, feature_value in row.features:
            gradient[feature_index + 1] += error * feature_value

    try:
        penalty = 0.5 * l2 * fsum(value * value for value in parameters[1:])
        objective = fsum(loss_terms) + penalty
    except (OverflowError, ValueError):
        return inf, [inf] * dimension
    if not isfinite(objective) or any(not isfinite(value) for value in gradient):
        return inf, [inf] * dimension
    return objective, gradient


def _lbfgs_direction(
    gradient: Sequence[float],
    history: Sequence[_HistoryEntry],
) -> list[float]:
    if not history:
        return [-value for value in gradient]

    transformed = list(gradient)
    alphas = []
    for entry in reversed(history):
        alpha = entry.inverse_curvature * _dot(
            entry.parameter_delta,
            transformed,
        )
        alphas.append(alpha)
        for index, value in enumerate(entry.gradient_delta):
            transformed[index] -= alpha * value

    latest = history[-1]
    curvature = 1.0 / latest.inverse_curvature
    gradient_delta_squared = _dot(
        latest.gradient_delta,
        latest.gradient_delta,
    )
    scaling = curvature / gradient_delta_squared
    result = [scaling * value for value in transformed]

    for entry, alpha in zip(history, reversed(alphas), strict=True):
        beta = entry.inverse_curvature * _dot(entry.gradient_delta, result)
        for index, value in enumerate(entry.parameter_delta):
            result[index] += value * (alpha - beta)
    return [-value for value in result]


def _make_model(
    dataset: _PreparedDataset,
    parameters: Sequence[float],
    *,
    converged: bool,
    termination_reason: TerminationReason,
    iterations_run: int,
    final_objective: float,
    final_largest_gradient: float,
    line_search_evaluations: int,
    skipped_curvature_updates: int,
) -> SparseLogisticModel:
    return SparseLogisticModel(
        intercept=parameters[0],
        weights=dict(zip(dataset.feature_names, parameters[1:], strict=True)),
        converged=converged,
        termination_reason=termination_reason,
        iterations_run=iterations_run,
        final_objective=final_objective,
        final_largest_gradient=final_largest_gradient,
        line_search_evaluations=line_search_evaluations,
        skipped_curvature_updates=skipped_curvature_updates,
    )


def _binary_label(value: object, example_index: int) -> float:
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)):
        raise TypeError(f"example {example_index} label must be binary numeric")
    result = float(value)
    if not isfinite(result) or result not in (0.0, 1.0):
        raise ValueError("example labels must be zero or one")
    return result


def _finite_feature_value(value: object, context: str = "feature value") -> float:
    return _finite_real(value, context)


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return fsum(a * b for a, b in zip(left, right, strict=True))


def _largest_absolute(values: Sequence[float]) -> float:
    return max(abs(value) for value in values)


def _softplus(value: float) -> float:
    if value > 0:
        return value + log1p(exp(-value))
    return log1p(exp(value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        decay = exp(-value)
        return 1.0 / (1.0 + decay)
    growth = exp(value)
    return growth / (1.0 + growth)
