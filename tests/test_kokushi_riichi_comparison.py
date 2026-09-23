import pytest

from mahjong_analysis.kokushi_riichi_comparison import analyze_decision_document


def _record(strategy: str, outcome: str, *, turn: int = 5, wait: str = "honor_single"):
    return {
        "year": 2020,
        "strategy": strategy,
        "riichi_eligible": True,
        "wait_group": wait,
        "turn": turn,
        "unseen_wait_counts": [2],
        "dealer": False,
        "prior_riichis": 0,
        "furiten": False,
        "opponent_called_hands": 0,
        "outcome": outcome,
        "win_method": "ron" if outcome == "win" else None,
        "point_delta": 32000 if outcome == "win" else -1000,
    }


def test_primary_groups_delayed_with_initial_dama_and_standardizes() -> None:
    records = [
        _record("immediate_riichi", "win"),
        _record("immediate_riichi", "draw"),
        _record("dama", "win"),
        _record("delayed_riichi", "deal_in"),
    ]
    result = analyze_decision_document({"scope": {}, "records": records})
    primary = result["primary_immediate_vs_initial_dama"]
    assert primary["groups"]["immediate_riichi"]["win_rate"] == 0.5
    assert primary["groups"]["initial_dama"]["win_rate"] == 0.5
    assert result["eligible_strategies"]["delayed_riichi"]["count"] == 1
    standardized = primary["standardized"]["wait_turn_unseen"]
    assert standardized["common_strata"] == 1
    assert standardized["outcomes"]["win"]["difference_percentage_points"] == 0
    assert standardized["point_delta"]["difference"] == 0
    assert (
        result["by_wait_group"]["honor_single"]["groups"]["immediate_riichi"]["count"]
        == 2
    )


def test_ineligible_records_are_excluded_from_comparison() -> None:
    ineligible = _record("dama", "win")
    ineligible["riichi_eligible"] = False
    result = analyze_decision_document(
        {"records": [_record("immediate_riichi", "draw"), ineligible]}
    )
    assert result["source"]["eligible_decision_count"] == 1
    assert (
        result["primary_immediate_vs_initial_dama"]["groups"]["initial_dama"]["count"]
        == 0
    )


def test_rejects_missing_required_fields() -> None:
    with pytest.raises(ValueError, match="missing fields"):
        analyze_decision_document({"records": [{"strategy": "dama"}]})
