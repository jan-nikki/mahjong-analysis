"""Artificial MJAI tests for the independent reference replay path."""

import gzip
import json
from dataclasses import replace
from pathlib import Path

import pytest

from mahjong_analysis.riichi_wait_reference import (
    ReferenceActorDiscard,
    ReferenceHandWaits,
    ReferenceKyoku,
    ReferenceMeld,
    ReferenceMjaiEvent,
    ReferenceWaitDetail,
    extract_reference_riichi_candidates,
    filter_reference_east_kyokus,
    is_reference_east_kyoku,
    is_reference_target_game,
    read_reference_mjai,
    reference_log_from_events,
    reference_tile_to_index,
    split_reference_kyokus,
)

TARGET_SOURCE = "2025/2025010100gm-00a9-0000-1234abcd.mjson"


def tiles(*groups: str) -> list[str]:
    result: list[str] = []
    for group in groups:
        if group in {"5mr", "5pr", "5sr"}:
            result.append(group)
        elif group[-1:] in {"m", "p", "s"}:
            result.extend(f"{rank}{group[-1]}" for rank in group[:-1])
        else:
            result.extend(group)
    return result


DECLARATION_POST_HAND = tiles("123m", "123p", "789p", "EE", "45s")


def make_log(
    *kyoku_events: dict[str, object],
    hands: list[list[str]] | None = None,
    bakaze: str = "E",
    source_path: str = TARGET_SOURCE,
    first_line: int = 1,
):
    if hands is None:
        hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    return reference_log_from_events(
        (
            {"type": "start_game", "aka_flag": True},
            {
                "type": "start_kyoku",
                "bakaze": bakaze,
                "kyoku": 1,
                "honba": 0,
                "oya": 0,
                "tehais": hands,
            },
            *kyoku_events,
            {"type": "end_kyoku"},
            {"type": "end_game"},
        ),
        source_path=source_path,
        first_line=first_line,
    )


def make_kyoku(*events: dict[str, object], **kwargs: object):
    return split_reference_kyokus(make_log(*events, **kwargs))[0]


def established_events(actor: int, tile: str = "9s") -> tuple[dict[str, object], ...]:
    return (
        {"type": "tsumo", "actor": actor, "pai": tile},
        {"type": "reach", "actor": actor},
        {"type": "dahai", "actor": actor, "pai": tile, "tsumogiri": True},
        {"type": "reach_accepted", "actor": actor},
    )


def detail_tuples(candidate) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (detail.wait_tile, detail.hand_type, detail.wait_shape)
        for detail in candidate.wait_details
    )


def canonical_tiles(raw_tiles: list[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            raw_tiles,
            key=lambda tile: (reference_tile_to_index(tile), tile.endswith("r")),
        )
    )


def test_extracts_established_riichi_with_source_and_wait_metadata() -> None:
    candidate = extract_reference_riichi_candidates(
        make_kyoku(*established_events(1), first_line=20)
    )[0]

    assert candidate.candidate_key == (TARGET_SOURCE, 21, 23)
    assert candidate.actor == 1
    assert (
        candidate.reach_event_index,
        candidate.declaration_dahai_event_index,
        candidate.reach_accepted_event_index,
    ) == (2, 3, 4)
    assert (
        candidate.reach_line,
        candidate.declaration_dahai_line,
        candidate.reach_accepted_line,
    ) == (23, 24, 25)
    assert candidate.riichi_discard_number == 1
    assert candidate.riichi_declaration_tile == "9s"
    assert candidate.riichi_declaration_tile_kind == "9s"
    assert candidate.concealed_tiles_after_discard == canonical_tiles(
        DECLARATION_POST_HAND
    )
    assert candidate.fixed_melds == ()
    assert candidate.wait_tiles == ("3s", "6s")
    assert candidate.wait_tile_count == 2
    assert detail_tuples(candidate) == (
        ("3s", "standard", "ryanmen"),
        ("6s", "standard", "ryanmen"),
    )
    assert candidate.wait_shapes == ("ryanmen",)
    assert candidate.contains_ryanmen is True
    assert candidate.is_pure_ryanmen is True
    assert candidate.is_multiwait is False

    discard = candidate.actor_discards_before_riichi[0]
    assert (
        discard.discard_number,
        discard.tile,
        discard.tile_kind,
        discard.tsumogiri,
        discard.event_index,
        discard.is_riichi_declaration,
        discard.was_called,
    ) == (1, "9s", "9s", True, 3, True, False)


def test_riichi_discard_number_counts_only_the_actor() -> None:
    events: list[dict[str, object]] = []
    for _ in range(6):
        events.extend(
            (
                {"type": "tsumo", "actor": 1, "pai": "9m"},
                {"type": "dahai", "actor": 1, "pai": "9m", "tsumogiri": True},
            )
        )
    events.extend(established_events(1))

    candidate = extract_reference_riichi_candidates(make_kyoku(*events))[0]

    assert candidate.riichi_discard_number == 7
    assert tuple(
        discard.discard_number for discard in candidate.actor_discards_before_riichi
    ) == tuple(range(1, 8))


def test_returns_no_candidate_for_reach_ending_in_ryukyoku() -> None:
    events = (
        {"type": "tsumo", "actor": 1, "pai": "9s"},
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "9s", "tsumogiri": True},
        {"type": "ryukyoku"},
    )

    assert extract_reference_riichi_candidates(make_kyoku(*events)) == ()


@pytest.mark.parametrize(
    ("events", "message"),
    (
        (
            (
                {"type": "tsumo", "actor": 1, "pai": "9s"},
                {"type": "reach", "actor": 1},
                {"type": "dahai", "actor": 2, "pai": "9s", "tsumogiri": True},
            ),
            "actors do not match",
        ),
        (
            (
                {"type": "tsumo", "actor": 1, "pai": "9s"},
                {"type": "reach", "actor": 1},
                {"type": "dahai", "actor": 1, "pai": "9s", "tsumogiri": True},
                {"type": "reach_accepted", "actor": 2},
            ),
            "actors do not match",
        ),
        (
            (
                {"type": "tsumo", "actor": 1, "pai": "9s"},
                {"type": "reach", "actor": 1},
                {"type": "dahai", "actor": 1, "pai": "9s", "tsumogiri": True},
                {"type": "dora", "dora_marker": "1m"},
                {"type": "reach_accepted", "actor": 1},
            ),
            "followed immediately",
        ),
    ),
    ids=("dahai-actor", "accepted-actor", "intervening-event"),
)
def test_reach_state_machine_rejects_invalid_sequence(
    events: tuple[dict[str, object], ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        extract_reference_riichi_candidates(make_kyoku(*events))


def test_accepts_declaration_tile_ron_as_unestablished() -> None:
    events = (
        {"type": "tsumo", "actor": 1, "pai": "5mr"},
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "5mr", "tsumogiri": True},
        {"type": "hora", "actor": 0, "target": 1, "pai": "5m"},
    )

    assert extract_reference_riichi_candidates(make_kyoku(*events)) == ()


def test_accepts_declaration_tile_ron_without_pai_as_unestablished() -> None:
    events = (
        {"type": "tsumo", "actor": 1, "pai": "9s"},
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "9s", "tsumogiri": True},
        {"type": "hora", "actor": 0, "target": 1},
    )

    assert extract_reference_riichi_candidates(make_kyoku(*events)) == ()


@pytest.mark.parametrize(
    "hora",
    (
        {"type": "hora", "actor": 0, "target": 2, "pai": "9s"},
        {"type": "hora", "actor": 0, "target": 1, "pai": "8s"},
    ),
    ids=("wrong-target", "wrong-tile"),
)
def test_rejects_unrelated_hora_after_declaration(hora: dict[str, object]) -> None:
    events = (
        {"type": "tsumo", "actor": 1, "pai": "9s"},
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "9s", "tsumogiri": True},
        hora,
    )

    with pytest.raises(ValueError, match="declaration hora"):
        extract_reference_riichi_candidates(make_kyoku(*events))


def test_rejects_self_hora_as_declaration_tile_ron() -> None:
    events = (
        {"type": "tsumo", "actor": 1, "pai": "9s"},
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "9s", "tsumogiri": True},
        {"type": "hora", "actor": 1, "target": 1, "pai": "9s"},
    )

    with pytest.raises(ValueError, match="actor must differ from target"):
        extract_reference_riichi_candidates(make_kyoku(*events))


@pytest.mark.parametrize("invalid_pai", [None, 5, "not-a-tile"])
def test_rejects_declaration_hora_with_present_invalid_pai(
    invalid_pai: object,
) -> None:
    events = (
        {"type": "tsumo", "actor": 1, "pai": "9s"},
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "9s", "tsumogiri": True},
        {"type": "hora", "actor": 0, "target": 1, "pai": invalid_pai},
    )

    with pytest.raises(ValueError, match="pai"):
        extract_reference_riichi_candidates(make_kyoku(*events))


def test_extracts_multiple_actor_riichis_in_acceptance_order() -> None:
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    hands[3] = tiles("123m", "456m", "789m", "123p", "5s")
    events = (*established_events(1, "9s"), *established_events(3, "C"))

    candidates = extract_reference_riichi_candidates(make_kyoku(*events, hands=hands))

    assert tuple(candidate.actor for candidate in candidates) == (1, 3)
    assert candidates[0].concealed_tiles_after_discard == canonical_tiles(
        DECLARATION_POST_HAND
    )
    assert candidates[0].riichi_declaration_tile == "9s"
    assert candidates[1].concealed_tiles_after_discard == canonical_tiles(
        tiles("123m", "456m", "789m", "123p", "5s")
    )
    assert candidates[1].riichi_declaration_tile == "C"
    assert candidates[1].wait_tiles == ("5s",)


def test_multiple_actor_riichis_keep_asymmetric_discard_histories() -> None:
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    hands[3] = tiles("123m", "456m", "789m", "123p", "5s")
    events = (
        {"type": "tsumo", "actor": 1, "pai": "9m"},
        {"type": "dahai", "actor": 1, "pai": "9m", "tsumogiri": True},
        *established_events(1, "9s"),
        *established_events(3, "C"),
    )

    first, second = extract_reference_riichi_candidates(
        make_kyoku(*events, hands=hands)
    )

    assert (first.actor, first.riichi_discard_number) == (1, 2)
    assert tuple(
        (
            discard.discard_number,
            discard.event_index,
            discard.is_riichi_declaration,
        )
        for discard in first.actor_discards_before_riichi
    ) == ((1, 2, False), (2, 5, True))
    assert (second.actor, second.riichi_discard_number) == (3, 1)
    assert tuple(
        (
            discard.discard_number,
            discard.event_index,
            discard.is_riichi_declaration,
        )
        for discard in second.actor_discards_before_riichi
    ) == ((1, 9, True),)


@pytest.mark.parametrize("raw_tile", ("5m", "5mr"))
def test_tsumo_and_dahai_remove_the_same_raw_five(raw_tile: str) -> None:
    candidate = extract_reference_riichi_candidates(
        make_kyoku(*established_events(1, raw_tile))
    )[0]

    assert candidate.riichi_declaration_tile == raw_tile
    assert candidate.riichi_declaration_tile_kind == "5m"
    assert raw_tile not in candidate.concealed_tiles_after_discard


def test_dahai_does_not_substitute_normal_for_red_five() -> None:
    events = (
        {"type": "tsumo", "actor": 1, "pai": "5mr"},
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "5m", "tsumogiri": True},
        {"type": "reach_accepted", "actor": 1},
    )

    with pytest.raises(ValueError, match=r"lacks raw tiles.*5m"):
        extract_reference_riichi_candidates(make_kyoku(*events))


def call_hands() -> list[list[str]]:
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    hands[0] = tiles("9m", "123p", "456p", "789s", "ESW")
    return hands


@pytest.mark.parametrize(
    ("call_event", "caller_hand", "caller_discard"),
    (
        (
            {
                "type": "chi",
                "actor": 1,
                "target": 0,
                "pai": "3m",
                "consumed": ["1m", "2m"],
            },
            tiles("12m", "123p", "456p", "789s", "ES"),
            "E",
        ),
        (
            {
                "type": "pon",
                "actor": 2,
                "target": 0,
                "pai": "9m",
                "consumed": ["9m", "9m"],
            },
            tiles("99m", "123p", "456p", "789s", "ES"),
            "E",
        ),
    ),
    ids=("chi", "pon"),
)
def test_replays_chi_and_pon_before_another_actor_riichi(
    call_event: dict[str, object],
    caller_hand: list[str],
    caller_discard: str,
) -> None:
    hands = call_hands()
    target_tile = str(call_event["pai"])
    if target_tile == "3m":
        hands[0] = tiles("3m", "123p", "456p", "789s", "ESW")
    caller = int(call_event["actor"])
    hands[caller] = caller_hand
    events = (
        {"type": "tsumo", "actor": 0, "pai": "C"},
        {"type": "dahai", "actor": 0, "pai": target_tile, "tsumogiri": False},
        call_event,
        {"type": "dahai", "actor": caller, "pai": caller_discard, "tsumogiri": False},
        *established_events(3),
    )

    candidates = extract_reference_riichi_candidates(make_kyoku(*events, hands=hands))

    assert tuple(candidate.actor for candidate in candidates) == (3,)


def test_replays_daiminkan_dora_rinshan_tsumo_and_dahai() -> None:
    hands = call_hands()
    hands[2] = tiles("999m", "123p", "456p", "ESWN")
    events = (
        {"type": "tsumo", "actor": 0, "pai": "C"},
        {"type": "dahai", "actor": 0, "pai": "9m", "tsumogiri": False},
        {
            "type": "daiminkan",
            "actor": 2,
            "target": 0,
            "pai": "9m",
            "consumed": ["9m", "9m", "9m"],
        },
        {"type": "dora", "dora_marker": "1s"},
        {"type": "tsumo", "actor": 2, "pai": "4s"},
        {"type": "dahai", "actor": 2, "pai": "4s", "tsumogiri": True},
        *established_events(1),
    )

    candidates = extract_reference_riichi_candidates(make_kyoku(*events, hands=hands))

    assert tuple(candidate.actor for candidate in candidates) == (1,)


def ankan_riichi_kyoku():
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    hands[0] = tiles("999m", "123p", "789p", "EE", "45s")
    events = (
        {"type": "tsumo", "actor": 0, "pai": "9m"},
        {"type": "ankan", "actor": 0, "consumed": ["9m"] * 4},
        {"type": "dora", "dora_marker": "1s"},
        {"type": "tsumo", "actor": 0, "pai": "C"},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "C", "tsumogiri": True},
        {"type": "reach_accepted", "actor": 0},
    )
    return make_kyoku(*events, hands=hands)


def test_replays_ankan_dora_rinshan_and_allows_riichi() -> None:
    candidate = extract_reference_riichi_candidates(ankan_riichi_kyoku())[0]

    assert candidate.fixed_melds == (ReferenceMeld(("9m", "9m", "9m", "9m")),)
    assert candidate.concealed_tiles_after_discard == canonical_tiles(
        tiles("123p", "789p", "EE", "45s")
    )
    assert candidate.wait_tiles == ("3s", "6s")


def test_reference_establishes_fifth_copy_wait_after_ankan() -> None:
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    hands[0] = tiles("44p", "67p", "2s", "456s", "6m", "3333s")
    kyoku = make_kyoku(
        {"type": "tsumo", "actor": 0, "pai": "5p"},
        {"type": "ankan", "actor": 0, "consumed": ["3s"] * 4},
        {"type": "dora", "dora_marker": "5p"},
        {"type": "tsumo", "actor": 0, "pai": "1s"},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "6m", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 0},
        hands=hands,
    )

    candidates = extract_reference_riichi_candidates(kyoku)

    assert len(candidates) == 1
    c = candidates[0]
    assert c.actor == 0
    assert c.riichi_discard_number == 1
    assert c.riichi_declaration_tile == "6m"
    assert c.candidate_key == (TARGET_SOURCE, 2, 7)
    assert (
        c.reach_event_index,
        c.declaration_dahai_event_index,
        c.reach_accepted_event_index,
    ) == (5, 6, 7)
    assert c.concealed_tiles_after_discard == canonical_tiles(
        tiles("44p", "567p", "12s", "456s")
    )
    assert c.fixed_melds == (ReferenceMeld(("3s",) * 4),)
    assert c.wait_tiles == ("3s",)
    assert detail_tuples(c) == (("3s", "standard", "penchan"),)
    assert c.wait_tile_count == 1
    assert c.wait_shapes == ("penchan",)
    assert c.contains_ryanmen is False
    assert c.is_pure_ryanmen is False
    assert c.is_multiwait is False


def test_replays_multiple_ankan_before_established_riichi() -> None:
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    hands[0] = tiles("111m", "999m", "123p", "EE", "45s")
    events = (
        {"type": "tsumo", "actor": 0, "pai": "1m"},
        {"type": "ankan", "actor": 0, "consumed": ["1m"] * 4},
        {"type": "dora", "dora_marker": "2m"},
        {"type": "tsumo", "actor": 0, "pai": "9m"},
        {"type": "ankan", "actor": 0, "consumed": ["9m"] * 4},
        {"type": "dora", "dora_marker": "8m"},
        {"type": "tsumo", "actor": 0, "pai": "C"},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "C", "tsumogiri": True},
        {"type": "reach_accepted", "actor": 0},
    )

    candidate = extract_reference_riichi_candidates(make_kyoku(*events, hands=hands))[0]

    assert candidate.concealed_tiles_after_discard == canonical_tiles(
        tiles("123p", "EE", "45s")
    )
    assert candidate.fixed_melds == (
        ReferenceMeld(("1m",) * 4),
        ReferenceMeld(("9m",) * 4),
    )
    assert candidate.wait_tiles == ("3s", "6s")
    assert detail_tuples(candidate) == (
        ("3s", "standard", "ryanmen"),
        ("6s", "standard", "ryanmen"),
    )
    assert candidate.wait_tile_count == 2
    assert candidate.wait_shapes == ("ryanmen",)
    assert candidate.contains_ryanmen is True
    assert candidate.is_pure_ryanmen is True
    assert candidate.is_multiwait is False


def test_replays_pon_then_kakan_before_another_riichi() -> None:
    hands = call_hands()
    hands[2] = tiles("99m", "123p", "456p", "789s", "ES")
    events = (
        {"type": "tsumo", "actor": 0, "pai": "C"},
        {"type": "dahai", "actor": 0, "pai": "9m", "tsumogiri": False},
        {
            "type": "pon",
            "actor": 2,
            "target": 0,
            "pai": "9m",
            "consumed": ["9m", "9m"],
        },
        {"type": "dahai", "actor": 2, "pai": "E", "tsumogiri": False},
        {"type": "tsumo", "actor": 2, "pai": "9m"},
        {
            "type": "kakan",
            "actor": 2,
            "pai": "9m",
            "consumed": ["9m", "9m", "9m"],
        },
        {"type": "dora", "dora_marker": "1s"},
        {"type": "tsumo", "actor": 2, "pai": "4s"},
        {"type": "dahai", "actor": 2, "pai": "4s", "tsumogiri": True},
        *established_events(1),
    )

    candidates = extract_reference_riichi_candidates(make_kyoku(*events, hands=hands))

    assert tuple(candidate.actor for candidate in candidates) == (1,)


def test_second_kakan_confirms_pon_was_replaced() -> None:
    hands = call_hands()
    hands[2] = tiles("99m", "123p", "456p", "789s", "ES")
    kakan = {
        "type": "kakan",
        "actor": 2,
        "pai": "9m",
        "consumed": ["9m", "9m", "9m"],
    }
    events = (
        {"type": "tsumo", "actor": 0, "pai": "C"},
        {"type": "dahai", "actor": 0, "pai": "9m", "tsumogiri": False},
        {
            "type": "pon",
            "actor": 2,
            "target": 0,
            "pai": "9m",
            "consumed": ["9m", "9m"],
        },
        {"type": "dahai", "actor": 2, "pai": "E", "tsumogiri": False},
        {"type": "tsumo", "actor": 2, "pai": "9m"},
        kakan,
        kakan,
    )

    with pytest.raises(ValueError, match="no unique matching pon"):
        extract_reference_riichi_candidates(make_kyoku(*events, hands=hands))


def test_rejects_riichi_with_open_meld() -> None:
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    hands[0] = tiles("3m", "123p", "456p", "789s", "ESW")
    hands[1] = tiles("12m", "123p", "456p", "789s", "ES")
    events = (
        {"type": "tsumo", "actor": 0, "pai": "C"},
        {"type": "dahai", "actor": 0, "pai": "3m", "tsumogiri": False},
        {
            "type": "chi",
            "actor": 1,
            "target": 0,
            "pai": "3m",
            "consumed": ["1m", "2m"],
        },
        {"type": "dahai", "actor": 1, "pai": "E", "tsumogiri": False},
        {"type": "tsumo", "actor": 1, "pai": "9s"},
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "9s", "tsumogiri": True},
        {"type": "reach_accepted", "actor": 1},
    )

    with pytest.raises(ValueError, match="open melds"):
        extract_reference_riichi_candidates(make_kyoku(*events, hands=hands))


def test_rejects_dahai_of_absent_raw_tile() -> None:
    event = {"type": "dahai", "actor": 0, "pai": "9m", "tsumogiri": False}

    with pytest.raises(ValueError, match=r"lacks raw tiles.*9m"):
        extract_reference_riichi_candidates(make_kyoku(event))


def test_rejects_absent_consumed_tile() -> None:
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    hands[0] = tiles("3m", "123p", "456p", "789s", "ESW")
    events = (
        {"type": "tsumo", "actor": 0, "pai": "C"},
        {"type": "dahai", "actor": 0, "pai": "3m", "tsumogiri": False},
        {
            "type": "chi",
            "actor": 1,
            "target": 0,
            "pai": "3m",
            "consumed": ["4m", "5m"],
        },
    )

    with pytest.raises(ValueError, match="lacks raw tiles"):
        extract_reference_riichi_candidates(make_kyoku(*events, hands=hands))


def test_rejects_kakan_without_corresponding_pon() -> None:
    event = {
        "type": "kakan",
        "actor": 0,
        "pai": "9m",
        "consumed": ["9m", "9m", "9m"],
    }

    with pytest.raises(ValueError, match="no unique matching pon"):
        extract_reference_riichi_candidates(make_kyoku(event))


def test_rejects_chi_from_wrong_direction() -> None:
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    hands[0] = tiles("3m", "123p", "456p", "789s", "ESW")
    events = (
        {"type": "tsumo", "actor": 0, "pai": "C"},
        {"type": "dahai", "actor": 0, "pai": "3m", "tsumogiri": False},
        {
            "type": "chi",
            "actor": 2,
            "target": 0,
            "pai": "3m",
            "consumed": ["1m", "2m"],
        },
    )

    with pytest.raises(ValueError, match="next actor"):
        extract_reference_riichi_candidates(make_kyoku(*events, hands=hands))


def test_each_dahai_checks_every_actor_post_discard_count() -> None:
    events = (
        {"type": "tsumo", "actor": 1, "pai": "9m"},
        {"type": "tsumo", "actor": 0, "pai": "C"},
        {"type": "dahai", "actor": 0, "pai": "C", "tsumogiri": True},
    )

    with pytest.raises(ValueError, match=r"actor 1: post-discard.*13, got 14"):
        extract_reference_riichi_candidates(make_kyoku(*events))


def test_rejects_duplicate_raw_red_five_in_actor_state() -> None:
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    hands[0] = tiles("5mr", "5mr", "123p", "789s", "ESWNP")

    with pytest.raises(ValueError, match=r"multiple physical copies.*5mr"):
        extract_reference_riichi_candidates(make_kyoku(hands=hands))


def test_rejects_five_owned_copies_after_normalizing_red_five() -> None:
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    hands[0] = tiles("555m", "5mr", "123p", "789s", "ESW")
    event = {"type": "tsumo", "actor": 0, "pai": "5m"}

    with pytest.raises(ValueError, match=r"more than four.*5m"):
        extract_reference_riichi_candidates(make_kyoku(event, hands=hands))


def test_same_raw_tile_discards_attach_call_to_later_event_index() -> None:
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    hands[2] = tiles("78m", "123p", "456p", "789s", "ES")
    events = (
        {"type": "tsumo", "actor": 1, "pai": "9m"},
        {"type": "dahai", "actor": 1, "pai": "9m", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "9m"},
        {"type": "dahai", "actor": 1, "pai": "9m", "tsumogiri": True},
        {
            "type": "chi",
            "actor": 2,
            "target": 1,
            "pai": "9m",
            "consumed": ["7m", "8m"],
        },
        {"type": "dahai", "actor": 2, "pai": "E", "tsumogiri": False},
        *established_events(1, "9s"),
    )

    candidate = extract_reference_riichi_candidates(make_kyoku(*events, hands=hands))[0]
    first, second, declaration = candidate.actor_discards_before_riichi

    assert (first.tile, first.event_index, first.was_called) == ("9m", 2, False)
    assert (
        second.tile,
        second.event_index,
        second.was_called,
        second.call_type,
        second.called_by_actor,
        second.call_event_index,
    ) == ("9m", 4, True, "chi", 2, 5)
    assert declaration.event_index == 9
    assert declaration.is_riichi_declaration is True


def test_call_after_reach_accepted_is_finalized_on_declaration_discard() -> None:
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    hands[2] = tiles("78m", "123p", "456p", "789s", "ES")
    events = (
        *established_events(1, "9m"),
        {
            "type": "chi",
            "actor": 2,
            "target": 1,
            "pai": "9m",
            "consumed": ["7m", "8m"],
        },
        {"type": "dahai", "actor": 2, "pai": "E", "tsumogiri": False},
    )

    candidate = extract_reference_riichi_candidates(make_kyoku(*events, hands=hands))[0]
    declaration = candidate.actor_discards_before_riichi[-1]

    assert declaration.is_riichi_declaration is True
    assert declaration.was_called is True
    assert declaration.call_type == "chi"
    assert declaration.called_by_actor == 2
    assert declaration.call_event_index == 5


@pytest.mark.parametrize("call_type", ("pon", "daiminkan"))
def test_pon_and_daiminkan_after_acceptance_update_declaration_discard(
    call_type: str,
) -> None:
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    consumed_count = 2 if call_type == "pon" else 3
    hands[3] = (
        tiles("99m", "123p", "456p", "789s", "ES")
        if call_type == "pon"
        else tiles("999m", "123p", "456p", "ESWN")
    )
    events: list[dict[str, object]] = [
        *established_events(1, "9m"),
        {
            "type": call_type,
            "actor": 3,
            "target": 1,
            "pai": "9m",
            "consumed": ["9m"] * consumed_count,
        },
    ]
    if call_type == "daiminkan":
        events.extend(
            (
                {"type": "dora", "dora_marker": "1s"},
                {"type": "tsumo", "actor": 3, "pai": "4s"},
                {"type": "dahai", "actor": 3, "pai": "4s", "tsumogiri": True},
            )
        )
    else:
        events.append({"type": "dahai", "actor": 3, "pai": "E", "tsumogiri": False})

    candidate = extract_reference_riichi_candidates(make_kyoku(*events, hands=hands))[0]
    declaration = candidate.actor_discards_before_riichi[-1]

    assert declaration.was_called is True
    assert declaration.call_type == call_type
    assert declaration.called_by_actor == 3
    assert declaration.call_event_index == 5


def test_snapshot_survives_post_acceptance_call_tsumo_and_hora() -> None:
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    hands[2] = tiles("78m", "123p", "456p", "789s", "ES")
    events = (
        *established_events(1, "9m"),
        {
            "type": "chi",
            "actor": 2,
            "target": 1,
            "pai": "9m",
            "consumed": ["7m", "8m"],
        },
        {"type": "dahai", "actor": 2, "pai": "E", "tsumogiri": False},
        {"type": "tsumo", "actor": 1, "pai": "3s"},
        {"type": "hora", "actor": 1, "target": 1, "pai": "3s"},
    )

    candidate = extract_reference_riichi_candidates(make_kyoku(*events, hands=hands))[0]
    declaration = candidate.actor_discards_before_riichi[-1]

    assert candidate.concealed_tiles_after_discard == canonical_tiles(
        DECLARATION_POST_HAND
    )
    assert candidate.fixed_melds == ()
    assert candidate.riichi_discard_number == 1
    assert declaration.was_called is True
    assert declaration.call_type == "chi"
    assert declaration.called_by_actor == 2
    assert declaration.call_event_index == 5


@pytest.mark.parametrize("call_type", ("pon", "daiminkan"))
def test_pon_and_daiminkan_finalize_target_river_metadata(call_type: str) -> None:
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    consumed_count = 2 if call_type == "pon" else 3
    hands[3] = (
        tiles("99m", "123p", "456p", "789s", "ES")
        if call_type == "pon"
        else tiles("999m", "123p", "456p", "ESWN")
    )
    events: list[dict[str, object]] = [
        {"type": "tsumo", "actor": 1, "pai": "9m"},
        {"type": "dahai", "actor": 1, "pai": "9m", "tsumogiri": True},
        {
            "type": call_type,
            "actor": 3,
            "target": 1,
            "pai": "9m",
            "consumed": ["9m"] * consumed_count,
        },
    ]
    if call_type == "daiminkan":
        events.extend(
            (
                {"type": "dora", "dora_marker": "1s"},
                {"type": "tsumo", "actor": 3, "pai": "4s"},
                {"type": "dahai", "actor": 3, "pai": "4s", "tsumogiri": True},
            )
        )
    else:
        events.append({"type": "dahai", "actor": 3, "pai": "E", "tsumogiri": False})
    events.extend(established_events(1, "9s"))

    candidate = extract_reference_riichi_candidates(make_kyoku(*events, hands=hands))[0]
    called = candidate.actor_discards_before_riichi[0]

    assert called.was_called is True
    assert called.call_type == call_type
    assert called.called_by_actor == 3
    assert called.call_event_index == 3


def write_json_lines(
    path: Path, events: list[dict[str, object]], *, compressed: bool
) -> Path:
    content = "".join(json.dumps(event) + "\n" for event in events)
    path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        with gzip.open(path, mode="wt", encoding="utf-8") as file:
            file.write(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.parametrize("compressed", (False, True), ids=("plain", "gzip"))
def test_reader_tracks_physical_lines_and_detects_gzip_magic(
    tmp_path: Path, compressed: bool
) -> None:
    events = [
        {"type": "start_game", "aka_flag": True},
        {"type": "start_kyoku", "bakaze": "E", "tehais": [[]] * 4},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]
    path = write_json_lines(
        tmp_path / "raw" / "2025" / Path(TARGET_SOURCE).name,
        events,
        compressed=compressed,
    )

    log = read_reference_mjai(path, source_root=tmp_path / "raw")
    kyoku = split_reference_kyokus(log)[0]

    assert log.source_path == f"2025/{Path(TARGET_SOURCE).name}"
    assert tuple(event.line_number for event in log.events) == (1, 2, 3, 4)
    assert tuple(event.event_index for event in kyoku.events) == (0, 1)
    assert kyoku.start_kyoku_line == 2
    if compressed:
        assert path.read_bytes()[:2] == b"\x1f\x8b"


def test_target_game_and_east_scope_are_independently_determined() -> None:
    east = make_log(source_path=TARGET_SOURCE)
    south = make_log(source_path=TARGET_SOURCE, bakaze="S")
    wrong_rule = make_log(source_path="2025/2025010100gm-00e1-0000-1234abcd.mjson")
    no_red = reference_log_from_events(
        ({"type": "start_game", "aka_flag": False},),
        source_path=TARGET_SOURCE,
    )

    assert is_reference_target_game(east) is True
    assert is_reference_target_game(wrong_rule) is False
    assert is_reference_target_game(no_red) is False
    assert is_reference_east_kyoku(split_reference_kyokus(east)[0]) is True
    assert is_reference_east_kyoku(split_reference_kyokus(south)[0]) is False
    assert filter_reference_east_kyokus(
        (
            split_reference_kyokus(east)[0],
            split_reference_kyokus(south)[0],
        )
    ) == (split_reference_kyokus(east)[0],)


def test_reader_rejects_invalid_json_with_source_line(tmp_path: Path) -> None:
    path = tmp_path / Path(TARGET_SOURCE).name
    path.write_text('{"type":"start_game"}\n{invalid}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"line 2: invalid MJAI JSON"):
        read_reference_mjai(path)


def test_rejects_non_tenpai_established_reach() -> None:
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    hands[1] = tiles("123m", "456p", "789s", "ESW", "N")

    with pytest.raises(ValueError, match="has no waits"):
        extract_reference_riichi_candidates(
            make_kyoku(*established_events(1), hands=hands)
        )


def test_reader_resets_event_index_for_each_kyoku() -> None:
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    log = reference_log_from_events(
        (
            {"type": "start_game", "aka_flag": True},
            {"type": "start_kyoku", "bakaze": "E", "tehais": hands},
            {"type": "end_kyoku"},
            {"type": "start_kyoku", "bakaze": "E", "tehais": hands},
            {"type": "end_kyoku"},
            {"type": "end_game"},
        ),
        source_path=TARGET_SOURCE,
    )

    first, second = split_reference_kyokus(log)

    assert tuple(event.event_index for event in first.events) == (0, 1)
    assert tuple(event.event_index for event in second.events) == (0, 1)
    assert (first.start_kyoku_line, second.start_kyoku_line) == (2, 4)


@pytest.mark.parametrize("aka_flag", ("true", 1), ids=("string", "integer"))
def test_target_game_rejects_non_boolean_true(aka_flag: object) -> None:
    log = reference_log_from_events(
        ({"type": "start_game", "aka_flag": aka_flag},),
        source_path=TARGET_SOURCE,
    )

    with pytest.raises(ValueError, match="aka_flag must be a bool"):
        is_reference_target_game(log)


def test_reference_event_and_kyoku_defensively_freeze_nested_input() -> None:
    nested_tiles = ["1m"]
    payload: dict[str, object] = {
        "type": "start_kyoku",
        "nested": {"tiles": nested_tiles},
    }
    events = [
        ReferenceMjaiEvent(1, payload, 0),
        ReferenceMjaiEvent(2, {"type": "end_kyoku"}, 1),
    ]

    kyoku = ReferenceKyoku("x.mjson", events)
    nested_tiles.append("2m")
    payload["new"] = "value"
    events.clear()

    assert isinstance(kyoku.events, tuple)
    assert len(kyoku.events) == 2
    assert kyoku.events[0].data["nested"]["tiles"] == ("1m",)  # type: ignore[index]
    assert "new" not in kyoku.events[0].data


@pytest.mark.parametrize("invalid", ({"1m"}, object()), ids=("set", "object"))
def test_reference_event_rejects_non_json_payload_values(invalid: object) -> None:
    with pytest.raises(TypeError, match="unsupported JSON value type"):
        ReferenceMjaiEvent(1, {"type": "start_game", "invalid": invalid})


def test_reference_kyoku_rejects_non_increasing_source_lines() -> None:
    events = (
        ReferenceMjaiEvent(2, {"type": "start_kyoku"}, 0),
        ReferenceMjaiEvent(1, {"type": "end_kyoku"}, 1),
    )

    with pytest.raises(ValueError, match="source lines must be strictly increasing"):
        ReferenceKyoku("x.mjson", events)

    with pytest.raises(ValueError, match="source_path"):
        ReferenceKyoku("", tuple(reversed(events)))
    with pytest.raises(TypeError, match="ReferenceMjaiEvent"):
        ReferenceKyoku("x.mjson", [events[0], object()])  # type: ignore[list-item]


@pytest.mark.parametrize(
    ("events", "message"),
    (
        (
            (
                {"type": "tsumo", "actor": 1, "pai": "9s"},
                {"type": "reach", "actor": 1},
                {"type": "reach", "actor": 1},
            ),
            "followed immediately by dahai",
        ),
        (({"type": "reach_accepted", "actor": 1},), "no matching pending reach"),
        (
            (
                {"type": "tsumo", "actor": 1, "pai": "9s"},
                {"type": "reach", "actor": 1},
                {"type": "dahai", "actor": 1, "pai": "9s", "tsumogiri": True},
            ),
            "pending reach did not end",
        ),
    ),
    ids=("duplicate-reach", "orphan-accepted", "pending-at-end"),
)
def test_reach_state_machine_rejects_additional_invalid_boundaries(
    events: tuple[dict[str, object], ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        extract_reference_riichi_candidates(make_kyoku(*events))


def test_split_rejects_next_kyoku_while_pending_reach_kyoku_is_open() -> None:
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    log = reference_log_from_events(
        (
            {"type": "start_game", "aka_flag": True},
            {"type": "start_kyoku", "bakaze": "E", "tehais": hands},
            {"type": "tsumo", "actor": 1, "pai": "9s"},
            {"type": "reach", "actor": 1},
            {"type": "dahai", "actor": 1, "pai": "9s", "tsumogiri": True},
            {"type": "start_kyoku", "bakaze": "E", "tehais": hands},
        ),
        source_path=TARGET_SOURCE,
    )

    with pytest.raises(ValueError, match="start_kyoku encountered before end_kyoku"):
        split_reference_kyokus(log)


def _valid_reference_candidate():
    return extract_reference_riichi_candidates(make_kyoku(*established_events(1)))[0]


def _two_discard_reference_candidate():
    events = (
        {"type": "tsumo", "actor": 1, "pai": "9m"},
        {"type": "dahai", "actor": 1, "pai": "9m", "tsumogiri": True},
        *established_events(1),
    )
    return extract_reference_riichi_candidates(make_kyoku(*events))[0]


def test_reference_models_reject_basic_invalid_identity_values() -> None:
    candidate = _valid_reference_candidate()

    with pytest.raises(ValueError, match="actor"):
        replace(candidate, actor=4)
    with pytest.raises(ValueError, match="event indexes"):
        replace(candidate, reach_event_index=-1)
    with pytest.raises(ValueError, match="source lines"):
        replace(candidate, start_kyoku_line=0)
    with pytest.raises(ValueError, match="line_number"):
        ReferenceMjaiEvent(0, {"type": "start_game"})
    with pytest.raises(ValueError, match="discard_number"):
        ReferenceActorDiscard(0, "1m", "1m", False, 1, False)
    with pytest.raises(ValueError, match="event_index"):
        ReferenceActorDiscard(1, "1m", "1m", False, -1, False)
    with pytest.raises(ValueError, match="normalized raw tile"):
        ReferenceActorDiscard(1, "5mr", "5p", False, 1, False)


def test_reference_candidate_rejects_invalid_event_and_discard_relationships() -> None:
    candidate = _valid_reference_candidate()
    declaration = candidate.actor_discards_before_riichi[-1]
    two_discards = _two_discard_reference_candidate()

    with pytest.raises(ValueError, match="reach < dahai < accepted"):
        replace(
            candidate,
            reach_event_index=candidate.declaration_dahai_event_index,
        )
    with pytest.raises(ValueError, match="history length"):
        replace(candidate, riichi_discard_number=2)
    with pytest.raises(ValueError, match="consecutive"):
        replace(
            two_discards,
            actor_discards_before_riichi=(
                replace(two_discards.actor_discards_before_riichi[0], discard_number=2),
                two_discards.actor_discards_before_riichi[1],
            ),
        )
    with pytest.raises(ValueError, match="last retained discard must declare"):
        replace(
            candidate,
            actor_discards_before_riichi=(
                replace(declaration, is_riichi_declaration=False),
            ),
        )
    with pytest.raises(ValueError, match="event index does not match"):
        replace(
            candidate,
            actor_discards_before_riichi=(
                replace(declaration, event_index=declaration.event_index - 1),
            ),
        )
    with pytest.raises(ValueError, match="only the last"):
        replace(
            two_discards,
            actor_discards_before_riichi=(
                replace(
                    two_discards.actor_discards_before_riichi[0],
                    is_riichi_declaration=True,
                ),
                two_discards.actor_discards_before_riichi[1],
            ),
        )


def test_reference_candidate_recalculates_waits_and_owned_tile_invariants() -> None:
    candidate = _valid_reference_candidate()
    ankan_candidate = extract_reference_riichi_candidates(ankan_riichi_kyoku())[0]
    inconsistent_waits = ReferenceHandWaits(
        ("E",),
        (ReferenceWaitDetail("E", "standard", "tanki"),),
    )

    with pytest.raises(ValueError, match="five of one normalized tile kind"):
        replace(candidate, concealed_tiles_after_discard=("1m",) * 13)
    with pytest.raises(ValueError, match="duplicate physical red fives"):
        replace(
            candidate,
            concealed_tiles_after_discard=(
                "5mr",
                "5mr",
                *candidate.concealed_tiles_after_discard[2:],
            ),
        )
    with pytest.raises(ValueError, match="five of one normalized tile kind"):
        replace(
            ankan_candidate,
            concealed_tiles_after_discard=(
                "9m",
                *ankan_candidate.concealed_tiles_after_discard[1:],
            ),
        )
    with pytest.raises(ValueError, match="duplicate physical red fives"):
        replace(
            ankan_candidate,
            fixed_melds=(ReferenceMeld(("5mr",) * 4),),
        )
    with pytest.raises(ValueError, match="waits must match"):
        replace(candidate, waits=inconsistent_waits)
