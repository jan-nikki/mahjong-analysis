import pytest

from mahjong_analysis.hand_waits import FixedMeld, calculate_hand_waits
from mahjong_analysis.riichi import ActorDiscard, extract_established_riichis


def expand_hand(hand: str) -> list[str]:
    """Expand compact notation used by the artificial MJAI fixtures."""
    tiles: list[str] = []
    for group in hand.split():
        if group in {"5mr", "5pr", "5sr"}:
            tiles.append(group)
        elif group[-1] in "mps":
            tiles.extend(f"{rank}{group[-1]}" for rank in group[:-1])
        else:
            tiles.extend(group)
    return tiles


DECLARATION_POST_HAND = expand_hand("123m 123p 789p EE 45s")


def make_kyoku(
    *events: dict[str, object],
    hands: list[list[str]] | None = None,
    oya: int = 0,
) -> list[dict[str, object]]:
    if hands is None:
        hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    return [
        {"type": "start_kyoku", "oya": oya, "tehais": hands},
        *events,
        {"type": "end_kyoku"},
    ]


def established_events(
    actor: int,
    tile: str = "9s",
) -> tuple[dict[str, object], ...]:
    return (
        {"type": "tsumo", "actor": actor, "pai": tile},
        {"type": "reach", "actor": actor},
        {"type": "dahai", "actor": actor, "pai": tile, "tsumogiri": True},
        {"type": "reach_accepted", "actor": actor},
    )


@pytest.mark.parametrize(
    ("oya", "actor"),
    [(0, 1), (2, 2)],
    ids=("child", "dealer"),
)
def test_extracts_normal_established_riichi_for_child_and_dealer(
    oya: int,
    actor: int,
) -> None:
    result = extract_established_riichis(
        make_kyoku(*established_events(actor), oya=oya)
    )

    assert len(result) == 1
    riichi = result[0]
    assert riichi.actor == actor
    assert riichi.reach_event_index == 2
    assert riichi.declaration_dahai_event_index == 3
    assert riichi.reach_accepted_event_index == 4
    assert riichi.riichi_discard_number == 1
    assert riichi.riichi_declaration_tile == "9s"
    assert riichi.riichi_declaration_tile_kind == "9s"
    assert riichi.concealed_tiles_after_discard == tuple(DECLARATION_POST_HAND)
    assert riichi.fixed_melds == ()

    waits = calculate_hand_waits(
        riichi.concealed_tiles_after_discard,
        riichi.fixed_melds,
    )
    assert waits.wait_tiles == ("3s", "6s")


def test_first_discard_riichi_is_number_one() -> None:
    riichi = extract_established_riichis(make_kyoku(*established_events(0)))[0]

    assert riichi.riichi_discard_number == 1
    assert tuple(
        discard.discard_number for discard in riichi.actor_discards_before_riichi
    ) == (1,)


def test_seventh_discard_riichi_counts_only_actor_discards() -> None:
    events: list[dict[str, object]] = []
    for _ in range(6):
        events.extend(
            (
                {"type": "tsumo", "actor": 1, "pai": "9m"},
                {
                    "type": "dahai",
                    "actor": 1,
                    "pai": "9m",
                    "tsumogiri": True,
                },
            )
        )
    events.extend(established_events(1))

    riichi = extract_established_riichis(make_kyoku(*events))[0]

    assert riichi.riichi_discard_number == 7
    assert tuple(
        discard.discard_number for discard in riichi.actor_discards_before_riichi
    ) == tuple(range(1, 8))


def test_extracts_multiple_established_riichis_in_event_order() -> None:
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    actor_three_hand = expand_hand("123m 456m 789m 123p 5s")
    hands[3] = actor_three_hand
    events = (
        *established_events(1),
        {"type": "tsumo", "actor": 3, "pai": "N"},
        {"type": "dahai", "actor": 3, "pai": "N", "tsumogiri": True},
        *established_events(3, "C"),
    )

    result = extract_established_riichis(make_kyoku(*events, hands=hands))

    assert tuple(riichi.actor for riichi in result) == (1, 3)
    first, second = result
    assert first.concealed_tiles_after_discard == tuple(DECLARATION_POST_HAND)
    assert first.riichi_declaration_tile == "9s"
    assert first.riichi_discard_number == 1
    assert tuple(discard.tile for discard in first.actor_discards_before_riichi) == (
        "9s",
    )
    assert second.concealed_tiles_after_discard == tuple(actor_three_hand)
    assert second.riichi_declaration_tile == "C"
    assert second.riichi_discard_number == 2
    assert tuple(discard.tile for discard in second.actor_discards_before_riichi) == (
        "N",
        "C",
    )


def test_unaccepted_declaration_ending_in_ryukyoku_is_not_extracted() -> None:
    events = (
        {"type": "tsumo", "actor": 1, "pai": "9s"},
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "9s", "tsumogiri": True},
        {"type": "ryukyoku"},
    )

    assert extract_established_riichis(make_kyoku(*events)) == ()


@pytest.mark.parametrize(
    ("declaration_tile", "hora_tile"),
    [("9s", "9s"), ("5mr", "5m")],
)
def test_declaration_tile_hora_is_a_valid_unaccepted_reach(
    declaration_tile: str,
    hora_tile: str,
) -> None:
    events = (
        {"type": "tsumo", "actor": 1, "pai": declaration_tile},
        {"type": "reach", "actor": 1},
        {
            "type": "dahai",
            "actor": 1,
            "pai": declaration_tile,
            "tsumogiri": True,
        },
        {"type": "hora", "actor": 2, "target": 1, "pai": hora_tile},
    )

    assert extract_established_riichis(make_kyoku(*events)) == ()


@pytest.mark.parametrize(
    "hora",
    [
        {"type": "hora", "actor": 2, "target": 0, "pai": "9s"},
        {"type": "hora", "actor": 2, "target": 1, "pai": "8s"},
    ],
    ids=("wrong-target", "wrong-tile-kind"),
)
def test_unrelated_hora_after_reach_declaration_is_rejected(
    hora: dict[str, object],
) -> None:
    events = (
        {"type": "tsumo", "actor": 1, "pai": "9s"},
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "9s", "tsumogiri": True},
        hora,
    )

    with pytest.raises(ValueError, match=r"declaration hora (target|pai)"):
        extract_established_riichis(make_kyoku(*events))


def test_self_hora_after_reach_declaration_is_rejected() -> None:
    events = (
        {"type": "tsumo", "actor": 1, "pai": "9s"},
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "9s", "tsumogiri": True},
        {"type": "hora", "actor": 1, "target": 1, "pai": "9s"},
    )

    with pytest.raises(ValueError, match="actor must differ from target"):
        extract_established_riichis(make_kyoku(*events))


def test_rejects_actor_outside_four_players() -> None:
    with pytest.raises(ValueError, match=r"event 2: actor"):
        extract_established_riichis(make_kyoku(*established_events(4)))


def test_rejects_reach_and_declaration_dahai_actor_mismatch() -> None:
    events = (
        {"type": "tsumo", "actor": 0, "pai": "9s"},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 1, "pai": "9s", "tsumogiri": True},
        {"type": "reach_accepted", "actor": 0},
    )

    with pytest.raises(ValueError, match="reach and dahai actors do not match"):
        extract_established_riichis(make_kyoku(*events))


def test_rejects_reach_and_reach_accepted_actor_mismatch() -> None:
    events = (
        {"type": "tsumo", "actor": 0, "pai": "9s"},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "9s", "tsumogiri": True},
        {"type": "reach_accepted", "actor": 1},
    )

    with pytest.raises(
        ValueError, match="reach and reach_accepted actors do not match"
    ):
        extract_established_riichis(make_kyoku(*events))


def test_rejects_reach_not_immediately_followed_by_dahai() -> None:
    events = (
        {"type": "reach", "actor": 0},
        {"type": "tsumo", "actor": 0, "pai": "9s"},
    )

    with pytest.raises(ValueError, match="reach must be followed by dahai"):
        extract_established_riichis(make_kyoku(*events))


def test_rejects_event_between_declaration_dahai_and_reach_accepted() -> None:
    events = (
        {"type": "tsumo", "actor": 0, "pai": "9s"},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "9s", "tsumogiri": True},
        {"type": "dora", "dora_marker": "1m"},
        {"type": "reach_accepted", "actor": 0},
    )

    with pytest.raises(ValueError, match=r"reach_accepted, hora, or ryukyoku"):
        extract_established_riichis(make_kyoku(*events))


def test_rejects_unmatched_reach_accepted() -> None:
    events = ({"type": "reach_accepted", "actor": 0},)

    with pytest.raises(ValueError, match="has no matching reach"):
        extract_established_riichis(make_kyoku(*events))


def test_rejects_dahai_of_raw_tile_not_in_concealed_hand() -> None:
    events = (
        {"type": "tsumo", "actor": 0, "pai": "9s"},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "1s", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 0},
    )

    with pytest.raises(ValueError, match=r"event 3, actor 0:.*does not contain"):
        extract_established_riichis(make_kyoku(*events))


def test_rejects_invalid_post_discard_count_for_any_actor() -> None:
    event = {
        "type": "dahai",
        "actor": 0,
        "pai": "1m",
        "tsumogiri": False,
    }

    with pytest.raises(
        ValueError,
        match=r"event 1, actor 0: post-discard.*expected 13, got 12",
    ):
        extract_established_riichis(make_kyoku(event))


def test_rejects_established_riichi_with_no_waits() -> None:
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    hands[1] = expand_hand("1111m 2222p 333s EE")

    with pytest.raises(ValueError, match=r"actor 1: established riichi.*no waits"):
        extract_established_riichis(
            make_kyoku(*established_events(1, "9m"), hands=hands)
        )


def test_rejects_consumed_raw_tile_not_in_concealed_hand() -> None:
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    hands[1] = expand_hand("223m 123p 789p EE 45s")
    events = (
        {"type": "tsumo", "actor": 0, "pai": "9m"},
        {"type": "dahai", "actor": 0, "pai": "3m", "tsumogiri": False},
        {
            "type": "chi",
            "actor": 1,
            "target": 0,
            "pai": "3m",
            "consumed": ["1m", "2m"],
        },
    )

    with pytest.raises(ValueError, match=r"event 3, actor 1:.*does not contain"):
        extract_established_riichis(make_kyoku(*events, hands=hands))


def test_chi_consumed_tiles_are_removed_from_concealed_hand() -> None:
    events = (
        {"type": "tsumo", "actor": 0, "pai": "9m"},
        {"type": "dahai", "actor": 0, "pai": "3m", "tsumogiri": False},
        {
            "type": "chi",
            "actor": 1,
            "target": 0,
            "pai": "3m",
            "consumed": ["1m", "2m"],
        },
        {"type": "dahai", "actor": 1, "pai": "1m", "tsumogiri": False},
    )

    with pytest.raises(ValueError, match=r"actor 1:.*does not contain.*1m"):
        extract_established_riichis(make_kyoku(*events))


def test_pon_consumed_tiles_are_removed_from_concealed_hand() -> None:
    events = (
        {"type": "tsumo", "actor": 0, "pai": "9m"},
        {"type": "dahai", "actor": 0, "pai": "E", "tsumogiri": False},
        {
            "type": "pon",
            "actor": 1,
            "target": 0,
            "pai": "E",
            "consumed": ["E", "E"],
        },
        {"type": "dahai", "actor": 1, "pai": "E", "tsumogiri": False},
    )

    with pytest.raises(ValueError, match=r"actor 1:.*does not contain.*E"):
        extract_established_riichis(make_kyoku(*events))


def test_daiminkan_consumed_tiles_are_removed_from_concealed_hand() -> None:
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    hands[1] = expand_hand("EEE 123m 123p 789s N")
    events = (
        {"type": "tsumo", "actor": 0, "pai": "9m"},
        {"type": "dahai", "actor": 0, "pai": "E", "tsumogiri": False},
        {
            "type": "daiminkan",
            "actor": 1,
            "target": 0,
            "pai": "E",
            "consumed": ["E", "E", "E"],
        },
        {"type": "dora", "dora_marker": "1s"},
        {"type": "tsumo", "actor": 1, "pai": "4m"},
        {"type": "dahai", "actor": 1, "pai": "E", "tsumogiri": False},
    )

    with pytest.raises(ValueError, match=r"actor 1:.*does not contain.*E"):
        extract_established_riichis(make_kyoku(*events, hands=hands))


def test_rejects_call_not_matching_latest_raw_discard() -> None:
    events = (
        {"type": "tsumo", "actor": 0, "pai": "9m"},
        {"type": "dahai", "actor": 0, "pai": "3m", "tsumogiri": False},
        {
            "type": "chi",
            "actor": 1,
            "target": 0,
            "pai": "6m",
            "consumed": ["4m", "5m"],
        },
    )

    with pytest.raises(ValueError, match="latest callable raw discard"):
        extract_established_riichis(make_kyoku(*events))


def test_rejects_chi_from_actor_other_than_target_next_actor() -> None:
    events = (
        {"type": "tsumo", "actor": 0, "pai": "9m"},
        {"type": "dahai", "actor": 0, "pai": "3m", "tsumogiri": False},
        {
            "type": "chi",
            "actor": 2,
            "target": 0,
            "pai": "3m",
            "consumed": ["1m", "2m"],
        },
    )

    with pytest.raises(ValueError, match=r"actor 2: chi actor must be the next"):
        extract_established_riichis(make_kyoku(*events))


def test_other_player_chi_is_replayed_before_riichi() -> None:
    events = (
        {"type": "tsumo", "actor": 2, "pai": "9m"},
        {"type": "dahai", "actor": 2, "pai": "3m", "tsumogiri": False},
        {
            "type": "chi",
            "actor": 3,
            "target": 2,
            "pai": "3m",
            "consumed": ["1m", "2m"],
        },
        {"type": "dahai", "actor": 3, "pai": "1p", "tsumogiri": False},
        *established_events(1),
    )

    assert extract_established_riichis(make_kyoku(*events))[0].actor == 1


def test_other_player_pon_is_replayed_before_riichi() -> None:
    events = (
        {"type": "tsumo", "actor": 0, "pai": "9m"},
        {"type": "dahai", "actor": 0, "pai": "E", "tsumogiri": False},
        {
            "type": "pon",
            "actor": 3,
            "target": 0,
            "pai": "E",
            "consumed": ["E", "E"],
        },
        {"type": "dahai", "actor": 3, "pai": "1p", "tsumogiri": False},
        *established_events(1),
    )

    assert extract_established_riichis(make_kyoku(*events))[0].actor == 1


def test_other_player_daiminkan_is_replayed_before_riichi() -> None:
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    hands[3] = expand_hand("EEE 123m 123p 789s N")
    events = (
        {"type": "tsumo", "actor": 0, "pai": "9m"},
        {"type": "dahai", "actor": 0, "pai": "E", "tsumogiri": False},
        {
            "type": "daiminkan",
            "actor": 3,
            "target": 0,
            "pai": "E",
            "consumed": ["E", "E", "E"],
        },
        {"type": "dora", "dora_marker": "1s"},
        {"type": "tsumo", "actor": 3, "pai": "4m"},
        {"type": "dahai", "actor": 3, "pai": "4m", "tsumogiri": True},
        *established_events(1),
    )

    assert extract_established_riichis(make_kyoku(*events, hands=hands))[0].actor == 1


def test_other_player_ankan_is_replayed_before_riichi() -> None:
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    hands[3] = expand_hand("EEEE 123m 123p 789s")
    events = (
        {"type": "tsumo", "actor": 3, "pai": "9m"},
        {
            "type": "ankan",
            "actor": 3,
            "consumed": ["E", "E", "E", "E"],
        },
        {"type": "dora", "dora_marker": "1s"},
        {"type": "tsumo", "actor": 3, "pai": "4m"},
        {"type": "dahai", "actor": 3, "pai": "4m", "tsumogiri": True},
        *established_events(1),
    )

    assert extract_established_riichis(make_kyoku(*events, hands=hands))[0].actor == 1


def test_rejects_established_riichi_by_actor_with_open_meld() -> None:
    events = (
        {"type": "tsumo", "actor": 0, "pai": "9m"},
        {"type": "dahai", "actor": 0, "pai": "3m", "tsumogiri": False},
        {
            "type": "chi",
            "actor": 1,
            "target": 0,
            "pai": "3m",
            "consumed": ["1m", "2m"],
        },
        {"type": "dahai", "actor": 1, "pai": "1p", "tsumogiri": False},
        *established_events(1),
    )

    with pytest.raises(ValueError, match=r"actor 1: established riichi.*open meld"):
        extract_established_riichis(make_kyoku(*events))


def test_riichi_actor_ankan_is_retained_as_fixed_meld() -> None:
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    hands[1] = expand_hand("999m 123p 789p EE 45s")
    events = (
        {"type": "tsumo", "actor": 1, "pai": "9m"},
        {
            "type": "ankan",
            "actor": 1,
            "consumed": ["9m", "9m", "9m", "9m"],
        },
        *established_events(1, "1s"),
    )

    riichi = extract_established_riichis(make_kyoku(*events, hands=hands))[0]

    assert riichi.concealed_tiles_after_discard == tuple(
        expand_hand("123p 789p EE 45s")
    )
    assert riichi.fixed_melds == (FixedMeld(("9m", "9m", "9m", "9m")),)
    waits = calculate_hand_waits(
        riichi.concealed_tiles_after_discard,
        riichi.fixed_melds,
    )
    assert waits.wait_tiles == ("3s", "6s")


def test_kakan_updates_matching_pon_before_another_actor_riichi() -> None:
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    hands[0] = expand_hand("9m 123p 456p 789s E S W")
    hands[2] = expand_hand("99m 123p 456p 789s E S")
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
        {"type": "tsumo", "actor": 2, "pai": "4m"},
        {"type": "dahai", "actor": 2, "pai": "4m", "tsumogiri": True},
        *established_events(1),
    )

    assert extract_established_riichis(make_kyoku(*events, hands=hands))[0].actor == 1


def test_kakan_replaces_existing_pon_in_fixed_melds() -> None:
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    hands[0] = expand_hand("9m 123p 456p 789s E S W")
    hands[2] = expand_hand("99m 123p 456p 789s E S")
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
        {
            "type": "kakan",
            "actor": 2,
            "pai": "9m",
            "consumed": ["9m", "9m", "9m"],
        },
    )

    with pytest.raises(ValueError, match="no unique matching pon"):
        extract_established_riichis(make_kyoku(*events, hands=hands))


def test_rejects_kakan_without_matching_pon() -> None:
    event = {
        "type": "kakan",
        "actor": 0,
        "pai": "9m",
        "consumed": ["9m", "9m", "9m"],
    }

    with pytest.raises(ValueError, match="no unique matching pon"):
        extract_established_riichis(make_kyoku(event))


def test_red_five_tsumo_and_dahai_preserve_raw_and_normalized_tiles() -> None:
    riichi = extract_established_riichis(make_kyoku(*established_events(1, "5mr")))[0]

    assert riichi.riichi_declaration_tile == "5mr"
    assert riichi.riichi_declaration_tile_kind == "5m"
    assert "5mr" not in riichi.concealed_tiles_after_discard


def test_dahai_does_not_substitute_normal_five_for_red_five() -> None:
    events = (
        {"type": "tsumo", "actor": 1, "pai": "5mr"},
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "5m", "tsumogiri": True},
        {"type": "reach_accepted", "actor": 1},
    )

    with pytest.raises(ValueError, match=r"raw tiles.*5m"):
        extract_established_riichis(make_kyoku(*events))


def test_rejects_multiple_physical_copies_of_same_red_five() -> None:
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    hands[0] = expand_hand("5mr 5mr 123p 789p E S W N P")

    with pytest.raises(ValueError, match=r"actor 0: multiple physical.*5mr"):
        extract_established_riichis(make_kyoku(hands=hands))


def test_rejects_fifth_owned_copy_of_normalized_tile_kind() -> None:
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    hands[0] = expand_hand("555m 5mr 123p 789s E S W")
    events = ({"type": "tsumo", "actor": 0, "pai": "5m"},)

    with pytest.raises(ValueError, match=r"event 1, actor 0: more than four.*5m"):
        extract_established_riichis(make_kyoku(*events, hands=hands))


def test_other_players_call_does_not_change_riichi_discard_number() -> None:
    events: list[dict[str, object]] = [
        {"type": "tsumo", "actor": 2, "pai": "9m"},
        {"type": "dahai", "actor": 2, "pai": "3m", "tsumogiri": False},
        {
            "type": "chi",
            "actor": 3,
            "target": 2,
            "pai": "3m",
            "consumed": ["1m", "2m"],
        },
        {"type": "dahai", "actor": 3, "pai": "1p", "tsumogiri": False},
    ]
    for _ in range(6):
        events.extend(
            (
                {"type": "tsumo", "actor": 1, "pai": "9m"},
                {
                    "type": "dahai",
                    "actor": 1,
                    "pai": "9m",
                    "tsumogiri": True,
                },
            )
        )
    events.extend(established_events(1))

    riichi = extract_established_riichis(make_kyoku(*events))[0]

    assert riichi.riichi_discard_number == 7
    assert len(riichi.actor_discards_before_riichi) == 7


def test_actor_discard_history_preserves_raw_normalized_and_call_metadata() -> None:
    events = (
        {"type": "tsumo", "actor": 1, "pai": "5mr"},
        {"type": "dahai", "actor": 1, "pai": "5mr", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "3m"},
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "3m", "tsumogiri": True},
        {"type": "reach_accepted", "actor": 1},
        {
            "type": "chi",
            "actor": 2,
            "target": 1,
            "pai": "3m",
            "consumed": ["1m", "2m"],
        },
    )

    riichi = extract_established_riichis(make_kyoku(*events))[0]
    first, declaration = riichi.actor_discards_before_riichi

    assert (
        first.discard_number,
        first.tile,
        first.tile_kind,
        first.tsumogiri,
        first.event_index,
        first.is_riichi_declaration,
        first.was_called,
    ) == (1, "5mr", "5m", True, 2, False, False)
    assert (
        declaration.discard_number,
        declaration.tile,
        declaration.tile_kind,
        declaration.tsumogiri,
        declaration.event_index,
        declaration.is_riichi_declaration,
        declaration.was_called,
        declaration.call_type,
        declaration.called_by_actor,
        declaration.call_event_index,
    ) == (2, "3m", "3m", True, 5, True, True, "chi", 2, 7)


def test_call_marks_only_later_of_two_same_raw_discards() -> None:
    hands = [list(DECLARATION_POST_HAND) for _ in range(4)]
    hands[2] = expand_hand("78m 123p 789p EE 45s N")
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
        {"type": "dahai", "actor": 2, "pai": "N", "tsumogiri": False},
        *established_events(1),
    )

    riichi = extract_established_riichis(make_kyoku(*events, hands=hands))[0]
    first, second, declaration = riichi.actor_discards_before_riichi

    assert (first.tile, first.event_index, first.was_called) == ("9m", 2, False)
    assert (
        second.tile,
        second.event_index,
        second.was_called,
        second.call_type,
        second.called_by_actor,
        second.call_event_index,
    ) == ("9m", 4, True, "chi", 2, 5)
    assert declaration.is_riichi_declaration is True


def make_actor_discard(**changes: object) -> ActorDiscard:
    values: dict[str, object] = {
        "discard_number": 1,
        "tile": "9s",
        "tile_kind": "9s",
        "tsumogiri": True,
        "event_index": 10,
        "is_riichi_declaration": False,
        "was_called": False,
        "call_type": None,
        "called_by_actor": None,
        "call_event_index": None,
    }
    values.update(changes)
    return ActorDiscard(**values)


@pytest.mark.parametrize(
    ("tile", "tile_kind"),
    [("5m", "5m"), ("5mr", "5m"), ("5pr", "5p"), ("5sr", "5s")],
)
def test_actor_discard_accepts_valid_raw_and_normalized_tiles(
    tile: str,
    tile_kind: str,
) -> None:
    discard = make_actor_discard(tile=tile, tile_kind=tile_kind)

    assert discard.tile == tile
    assert discard.tile_kind == tile_kind


@pytest.mark.parametrize("call_type", ["chi", "pon", "daiminkan"])
def test_actor_discard_accepts_consistent_call_information(call_type: str) -> None:
    discard = make_actor_discard(
        was_called=True,
        call_type=call_type,
        called_by_actor=2,
        call_event_index=11,
    )

    assert (
        discard.was_called,
        discard.call_type,
        discard.called_by_actor,
        discard.call_event_index,
    ) == (True, call_type, 2, 11)


@pytest.mark.parametrize(
    ("changes", "expected_message"),
    [
        ({"discard_number": 0}, "discard_number"),
        ({"discard_number": True}, "discard_number"),
        ({"event_index": -1}, "event_index"),
        ({"event_index": True}, "event_index"),
        ({"tile": ["9s"]}, "tile must be a string"),
        ({"tile": "0m"}, "invalid tile"),
        ({"tile_kind": ["9s"]}, "tile_kind must be a string"),
        (
            {"tile": "5mr", "tile_kind": "5mr"},
            "tile_kind must be a normalized tile kind",
        ),
        (
            {"tile": "5mr", "tile_kind": "5p"},
            "tile_kind must equal normalize_tile",
        ),
        ({"tsumogiri": 1}, "tsumogiri"),
        ({"is_riichi_declaration": 1}, "is_riichi_declaration"),
        ({"was_called": 1}, "was_called"),
        ({"called_by_actor": 2}, "uncalled discard"),
        ({"call_type": "pon"}, "uncalled discard"),
        ({"call_event_index": 11}, "uncalled discard"),
        ({"was_called": True}, "call_type"),
        (
            {
                "was_called": True,
                "call_type": "kakan",
                "called_by_actor": 2,
                "call_event_index": 11,
            },
            "call_type",
        ),
        (
            {
                "was_called": True,
                "call_type": "pon",
                "called_by_actor": True,
                "call_event_index": 11,
            },
            "called_by_actor",
        ),
        (
            {
                "was_called": True,
                "call_type": "pon",
                "called_by_actor": 4,
                "call_event_index": 11,
            },
            "called_by_actor",
        ),
        (
            {
                "was_called": True,
                "call_type": "pon",
                "called_by_actor": 2,
                "call_event_index": True,
            },
            "call_event_index",
        ),
        (
            {
                "was_called": True,
                "call_type": "pon",
                "called_by_actor": 2,
                "call_event_index": 10,
            },
            "call_event_index must be after",
        ),
    ],
)
def test_actor_discard_rejects_invalid_direct_construction(
    changes: dict[str, object],
    expected_message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=expected_message):
        make_actor_discard(**changes)
