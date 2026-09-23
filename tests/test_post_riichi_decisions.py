from __future__ import annotations

from typing import Any

from mahjong_analysis.combo_prediction import candidate_rows_for_decision
from mahjong_analysis.post_riichi_decisions import (
    CandidateDanger,
    PostRiichiDrawDecision,
    extract_post_riichi_draw_decisions,
    visible_tile_counts,
)
from mahjong_analysis.tiles import tile_to_index


def expand_hand(hand: str) -> list[str]:
    tiles: list[str] = []
    for group in hand.split():
        if group in {"5mr", "5pr", "5sr"}:
            tiles.append(group)
        elif group[-1] in "mps":
            tiles.extend(f"{rank}{group[-1]}" for rank in group[:-1])
        else:
            tiles.extend(group)
    return tiles


TARGET_HAND = expand_hand("123m 123p 789p EE 45s")
OBSERVER_HAND = expand_hand("123m 456m 789m 3s 6s F C")
OTHER_HAND = expand_hand("147m 258p 369s ESWN")


def make_kyoku(
    *events: dict[str, Any],
    target_hand: list[str] | None = None,
    observer_hand: list[str] | None = None,
    actor2_hand: list[str] | None = None,
) -> list[dict[str, Any]]:
    hands = [
        list(target_hand or TARGET_HAND),
        list(observer_hand or OBSERVER_HAND),
        list(actor2_hand or OTHER_HAND),
        list(OTHER_HAND),
    ]
    return [
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "N",
            "tehais": hands,
        },
        *events,
        {"type": "end_kyoku"},
    ]


def target_riichi_events() -> tuple[dict[str, Any], ...]:
    return (
        {"type": "tsumo", "actor": 0, "pai": "9s"},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "9s", "tsumogiri": True},
        {"type": "reach_accepted", "actor": 0},
    )


def candidate(decision: PostRiichiDrawDecision, tile: str) -> CandidateDanger:
    return next(value for value in decision.candidates if value.tile == tile)


def test_extracts_all_normalized_candidates_with_wait_labels() -> None:
    kyoku = make_kyoku(
        *target_riichi_events(),
        {"type": "tsumo", "actor": 1, "pai": "2s"},
        {"type": "dahai", "actor": 1, "pai": "1m", "tsumogiri": False},
    )

    decisions = extract_post_riichi_draw_decisions(kyoku)

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.observer_actor == 1
    assert decision.target_actor == 0
    assert decision.bakaze == "E"
    assert decision.kyoku_number == 1
    assert decision.honba == 0
    assert decision.kyotaku == 1
    assert decision.oya == 0
    assert decision.scores == (24000, 25000, 25000, 25000)
    assert decision.target_concealed_tile_count == 13
    assert decision.target_riichi_discard_number == 1
    assert decision.target_riichi_declaration_tile_kind == "9s"
    assert decision.target_own_discard_kinds == ("9s",)
    assert decision.post_riichi_passed_tile_kinds == ()
    assert decision.target_safe_tile_kinds == ("9s",)
    assert decision.post_riichi_decision_number == 1
    assert decision.target_wait_tiles == ("3s", "6s")
    assert decision.decision_discard_number == 1
    assert decision.actual_declared_reach is False
    assert candidate(decision, "3s").is_structural_wait is True
    assert candidate(decision, "6s").is_structural_wait is True
    assert candidate(decision, "2s").is_structural_wait is False
    assert candidate(decision, "3s").is_ron_eligible is True


def test_includes_chasing_riichi_declaration_discard() -> None:
    observer_hand = list(TARGET_HAND)
    kyoku = make_kyoku(
        *target_riichi_events(),
        {"type": "tsumo", "actor": 1, "pai": "8s"},
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "8s", "tsumogiri": True},
        {"type": "reach_accepted", "actor": 1},
        observer_hand=observer_hand,
    )

    decisions = extract_post_riichi_draw_decisions(kyoku)

    assert len(decisions) == 1
    assert decisions[0].actual_declared_reach is True
    assert decisions[0].tsumo_event_index == 5
    assert decisions[0].dahai_event_index == 7


def test_non_observer_hidden_hand_does_not_change_features() -> None:
    events = (
        *target_riichi_events(),
        {"type": "tsumo", "actor": 1, "pai": "2s"},
        {"type": "dahai", "actor": 1, "pai": "1m", "tsumogiri": False},
    )
    changed_hidden_hand = expand_hand("111m 222p 333s EESS")

    first = extract_post_riichi_draw_decisions(make_kyoku(*events))[0]
    second = extract_post_riichi_draw_decisions(
        make_kyoku(*events, actor2_hand=changed_hidden_hand)
    )[0]

    assert first.visible_counts == second.visible_counts
    assert first.unseen_counts == second.unseen_counts
    assert first.candidates == second.candidates


def test_target_discard_furiten_disables_every_wait_for_ron() -> None:
    kyoku = make_kyoku(
        {"type": "tsumo", "actor": 0, "pai": "3s"},
        {"type": "dahai", "actor": 0, "pai": "3s", "tsumogiri": True},
        *target_riichi_events(),
        {"type": "tsumo", "actor": 1, "pai": "2s"},
        {"type": "dahai", "actor": 1, "pai": "1m", "tsumogiri": False},
    )

    decision = extract_post_riichi_draw_decisions(kyoku)[0]

    assert decision.target_is_ron_furiten is True
    assert candidate(decision, "3s").is_structural_wait is True
    assert candidate(decision, "6s").is_structural_wait is True
    assert candidate(decision, "3s").is_ron_eligible is False
    assert candidate(decision, "6s").is_ron_eligible is False


def test_passed_wait_after_riichi_disables_every_wait_for_ron() -> None:
    kyoku = make_kyoku(
        *target_riichi_events(),
        {"type": "tsumo", "actor": 2, "pai": "3s"},
        {"type": "dahai", "actor": 2, "pai": "3s", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "2s"},
        {"type": "dahai", "actor": 1, "pai": "1m", "tsumogiri": False},
    )

    decisions = extract_post_riichi_draw_decisions(kyoku)
    observer_decision = next(value for value in decisions if value.observer_actor == 1)

    assert observer_decision.target_is_ron_furiten is True
    assert observer_decision.post_riichi_passed_tile_kinds == ("3s",)
    assert candidate(observer_decision, "3s").conventional.is_genbutsu is True
    assert candidate(observer_decision, "3s").conventional.is_post_riichi_passed is True
    assert candidate(observer_decision, "6s").is_ron_eligible is False


def test_called_tile_is_not_counted_twice_as_public() -> None:
    observer_hand = expand_hand("123m 456m 789m 3s 6s F C")
    actor2_hand = expand_hand("333m 147p 258s ESWN")
    kyoku = make_kyoku(
        *target_riichi_events(),
        {"type": "tsumo", "actor": 1, "pai": "2s"},
        {"type": "dahai", "actor": 1, "pai": "3m", "tsumogiri": False},
        {
            "type": "pon",
            "actor": 2,
            "target": 1,
            "pai": "3m",
            "consumed": ["3m", "3m"],
        },
        {"type": "dahai", "actor": 2, "pai": "E", "tsumogiri": False},
        {"type": "tsumo", "actor": 1, "pai": "4s"},
        {"type": "dahai", "actor": 1, "pai": "1m", "tsumogiri": False},
        observer_hand=observer_hand,
        actor2_hand=actor2_hand,
    )

    decisions = extract_post_riichi_draw_decisions(kyoku)
    second = [value for value in decisions if value.observer_actor == 1][1]

    assert [
        value.post_riichi_decision_number
        for value in decisions
        if value.observer_actor == 1
    ] == [1, 2]
    assert second.public_visible_counts[tile_to_index("3m")] == 3
    assert second.visible_counts[tile_to_index("3m")] == 3


def test_rinshan_draw_is_not_extracted_as_an_ordinary_draw_decision() -> None:
    observer_hand = expand_hand("111m 234p 567s ESWN")
    kyoku = make_kyoku(
        *target_riichi_events(),
        {"type": "tsumo", "actor": 1, "pai": "1m"},
        {
            "type": "ankan",
            "actor": 1,
            "consumed": ["1m", "1m", "1m", "1m"],
        },
        {"type": "dora", "dora_marker": "2p"},
        {"type": "tsumo", "actor": 1, "pai": "8p"},
        {"type": "dahai", "actor": 1, "pai": "8p", "tsumogiri": True},
        observer_hand=observer_hand,
    )

    decisions = extract_post_riichi_draw_decisions(kyoku)

    assert decisions == ()


def test_target_post_riichi_ankan_updates_concealed_count_and_keeps_waits() -> None:
    target_hand = expand_hand("123m 666p 789p EE 45s")
    kyoku = make_kyoku(
        *target_riichi_events(),
        {"type": "tsumo", "actor": 0, "pai": "6p"},
        {
            "type": "ankan",
            "actor": 0,
            "consumed": ["6p", "6p", "6p", "6p"],
        },
        {"type": "dora", "dora_marker": "2p"},
        {"type": "tsumo", "actor": 0, "pai": "P"},
        {"type": "dahai", "actor": 0, "pai": "P", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "2s"},
        {"type": "dahai", "actor": 1, "pai": "1m", "tsumogiri": False},
        target_hand=target_hand,
    )

    decision = extract_post_riichi_draw_decisions(kyoku)[0]

    assert decision.target_wait_tiles == ("3s", "6s")
    assert decision.target_concealed_tile_count == 10
    assert decision.public_visible_counts[tile_to_index("6p")] == 4


def test_visible_count_function_has_no_other_hand_input() -> None:
    counts = visible_tile_counts(["5m", "5mr"], ["5m", "E"])

    assert counts[tile_to_index("5m")] == 3
    assert counts[tile_to_index("E")] == 1


def test_candidate_rows_have_stable_id_weights_and_monotone_combo_percentiles() -> None:
    kyoku = make_kyoku(
        *target_riichi_events(),
        {"type": "tsumo", "actor": 1, "pai": "2s"},
        {"type": "dahai", "actor": 1, "pai": "1m", "tsumogiri": False},
    )
    decision = extract_post_riichi_draw_decisions(kyoku)[0]

    rows = candidate_rows_for_decision(
        decision,
        year=2020,
        game_id="2020/artificial.mjson",
        kyoku_index=0,
    )

    assert len(rows) == len(decision.candidates)
    assert {row.candidate_count for row in rows} == {len(rows)}
    assert len({row.decision_id for row in rows}) == 1
    ordered = sorted(rows, key=lambda row: row.candidate.simple_combo.total)
    assert [row.simple_percentile for row in ordered] == sorted(
        row.simple_percentile for row in ordered
    )
