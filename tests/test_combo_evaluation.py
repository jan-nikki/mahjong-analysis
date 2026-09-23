from __future__ import annotations

import pytest

from mahjong_analysis.combo_evaluation import evaluate_policy_pair


def test_same_selected_action_has_zero_common_evaluation_difference() -> None:
    result = evaluate_policy_pair(
        "5m",
        "5m",
        common_action_values={"5m": -1200.0},
        target_ron_labels={"5m": True},
    )

    assert result.action_changed is False
    assert result.combo_minus_base_q_eval == 0.0
    assert result.combo_minus_base_target_ron == 0


def test_changed_actions_use_one_shared_evaluator_and_target_label_map() -> None:
    result = evaluate_policy_pair(
        "5m",
        "4s",
        common_action_values={"5m": -1200.0, "4s": 300.0},
        target_ron_labels={"5m": True, "4s": False},
    )

    assert result.action_changed is True
    assert result.base_q_eval == -1200.0
    assert result.combo_q_eval == 300.0
    assert result.combo_minus_base_q_eval == 1500.0
    assert result.combo_minus_base_target_ron == -1


@pytest.mark.parametrize("bad_value", [True, float("nan"), float("inf")])
def test_common_evaluator_rejects_invalid_values(bad_value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        evaluate_policy_pair(
            "5m",
            "4s",
            common_action_values={"5m": 0.0, "4s": bad_value},  # type: ignore[dict-item]
            target_ron_labels={"5m": False, "4s": False},
        )


def test_common_evaluator_requires_both_selected_actions() -> None:
    with pytest.raises(KeyError, match="4s"):
        evaluate_policy_pair(
            "5m",
            "4s",
            common_action_values={"5m": 0.0},
            target_ron_labels={"5m": False, "4s": False},
        )
