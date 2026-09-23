"""Shared-evaluator comparisons for combo-theory discard policies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Real


@dataclass(frozen=True)
class PairedPolicyEvaluation:
    """One base-versus-combo comparison scored by a common evaluator."""

    base_action: str
    combo_action: str
    action_changed: bool
    base_q_eval: float
    combo_q_eval: float
    combo_minus_base_q_eval: float
    base_target_ron: bool
    combo_target_ron: bool
    combo_minus_base_target_ron: int


def evaluate_policy_pair(
    base_action: str,
    combo_action: str,
    *,
    common_action_values: Mapping[str, float],
    target_ron_labels: Mapping[str, bool],
) -> PairedPolicyEvaluation:
    """Compare two selected actions with one shared value and label mapping.

    ``common_action_values`` is an evaluation model, not either policy's own
    prediction. Higher values are better. A positive target-ron difference is
    worse for the combo policy; a negative difference is better.
    """
    _validate_action(base_action, "base_action")
    _validate_action(combo_action, "combo_action")
    base_value = _common_value(common_action_values, base_action)
    combo_value = _common_value(common_action_values, combo_action)
    base_ron = _target_ron_label(target_ron_labels, base_action)
    combo_ron = _target_ron_label(target_ron_labels, combo_action)
    return PairedPolicyEvaluation(
        base_action=base_action,
        combo_action=combo_action,
        action_changed=base_action != combo_action,
        base_q_eval=base_value,
        combo_q_eval=combo_value,
        combo_minus_base_q_eval=combo_value - base_value,
        base_target_ron=base_ron,
        combo_target_ron=combo_ron,
        combo_minus_base_target_ron=int(combo_ron) - int(base_ron),
    )


def _validate_action(action: object, name: str) -> None:
    if not isinstance(action, str) or not action:
        raise ValueError(f"{name} must be a non-empty string")


def _common_value(values: Mapping[str, float], action: str) -> float:
    if action not in values:
        raise KeyError(f"common evaluator has no value for action {action!r}")
    value = values[action]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"common evaluator value for {action!r} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"common evaluator value for {action!r} must be finite")
    return result


def _target_ron_label(labels: Mapping[str, bool], action: str) -> bool:
    if action not in labels:
        raise KeyError(f"target-ron labels have no value for action {action!r}")
    value = labels[action]
    if not isinstance(value, bool):
        raise TypeError(f"target-ron label for {action!r} must be boolean")
    return value
