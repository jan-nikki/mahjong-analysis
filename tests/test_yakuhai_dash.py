from __future__ import annotations

from math import comb

import pytest

from mahjong_analysis.tiles import TILE_KINDS, normalize_tile, tiles_to_counts
from mahjong_analysis.yakuhai_dash import (
    YakuhaiDashAccumulator,
    analyze_yakuhai_dash_kyoku,
    classify_yakuhai,
    seat_wind,
    theoretical_any_pair_probability,
    theoretical_singleton_pair_by_draw,
    theoretical_specific_pair_probability,
    singleton_to_dict,
    yakuhai_kinds,
)


def make_hands(
    allocations: dict[int, list[str]] | None = None,
) -> list[list[str]]:
    allocations = allocations or {}
    hands = [list(allocations.get(actor, [])) for actor in range(4)]
    allocated = [tile for hand in hands for tile in hand]
    tiles_to_counts(allocated)
    deck = [
        tile
        for tile in TILE_KINDS
        if tile[-1:] in "mps"
        for _ in range(4 - sum(normalize_tile(value) == tile for value in allocated))
    ]
    deck_index = 0
    while any(len(hand) < 13 for hand in hands):
        for hand in hands:
            if len(hand) < 13:
                hand.append(deck[deck_index])
                deck_index += 1
    tiles_to_counts(tile for hand in hands for tile in hand)
    return hands


def make_kyoku(
    hands: list[list[str]],
    events: list[dict[str, object]] | None = None,
    *,
    dealer: int = 0,
) -> list[dict[str, object]]:
    return [
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": dealer,
            "scores": [25000] * 4,
            "dora_marker": "9s",
            "tehais": hands,
        },
        *(events or []),
        {"type": "end_kyoku"},
    ]


def observation(result, actor: int, tile: str):
    return next(
        row for row in result.singletons if row.actor == actor and row.tile == tile
    )


def non_tile(hand: list[str], excluded: str) -> str:
    return next(tile for tile in hand if normalize_tile(tile) != excluded)


def test_seat_winds_and_unique_yakuhai_kinds() -> None:
    assert [seat_wind(actor, 2) for actor in range(4)] == ["W", "N", "E", "S"]
    assert yakuhai_kinds(0, 0) == ("E", "P", "F", "C")
    assert yakuhai_kinds(1, 0) == ("E", "S", "P", "F", "C")
    assert classify_yakuhai("E", 0, 0) == "dealer_double_wind"
    assert classify_yakuhai("E", 1, 0) == "shared_round_wind"
    assert classify_yakuhai("P", 1, 0) == "shared_dragon"
    assert classify_yakuhai("S", 1, 0) == "own_seat_wind"


def test_initial_pair_and_singleton_counts_use_thirteen_tile_deal() -> None:
    hands = make_hands({0: ["E", "E", "P"], 1: ["S"]})

    result = analyze_yakuhai_dash_kyoku(make_kyoku(hands))

    dealer = result.deals[0]
    child = result.deals[1]
    assert dealer.yakuhai_counts == (("E", 2), ("P", 1), ("F", 0), ("C", 0))
    assert dealer.yakuhai_pair_kind_count == 1
    assert dealer.singleton_kind_count == 1
    assert child.yakuhai_counts == (
        ("E", 0),
        ("S", 1),
        ("P", 0),
        ("F", 0),
        ("C", 0),
    )
    assert len(result.openings) == 2
    assert observation(result, 0, "P").outcome == "round_end"
    assert observation(result, 1, "S").yakuhai_class == "own_seat_wind"


def test_self_draw_forms_pair_at_exact_draw_and_pair_break_is_not_dash() -> None:
    hands = make_hands({0: ["P"]})
    events = [
        {"type": "tsumo", "actor": 0, "pai": "P"},
        {"type": "dahai", "actor": 0, "pai": "P", "tsumogiri": True},
    ]

    result = analyze_yakuhai_dash_kyoku(make_kyoku(hands, events))
    tracked = observation(result, 0, "P")
    opening = next(row for row in result.openings if row.actor == 0)

    assert tracked.outcome == "self_pair_draw"
    assert tracked.self_pair_draw_number == 1
    assert tracked.self_pair_event_index == 1
    assert tracked.self_pair_followup_discard_event_index == 2
    assert tracked.self_pair_retained_after_discard is False
    assert tracked.self_pair_pon_shape_after_discard is False
    assert tracked.singleton_discard_number is None
    assert opening.strict_dash is False
    assert opening.dash_by_3 is False


def test_self_pair_followup_is_fixed_at_first_own_discard() -> None:
    hands = make_hands({0: ["P"]})
    events = [
        {"type": "tsumo", "actor": 0, "pai": "P"},
        {"type": "dahai", "actor": 0, "pai": "P", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "P"},
        {
            "type": "dahai",
            "actor": 0,
            "pai": non_tile(hands[0], "P"),
            "tsumogiri": False,
        },
    ]
    result = analyze_yakuhai_dash_kyoku(make_kyoku(hands, events))
    tracked = observation(result, 0, "P")

    assert tracked.outcome == "self_pair_draw"
    assert tracked.self_pair_followup_discard_event_index == 2
    assert tracked.self_pair_retained_after_discard is False
    assert tracked.self_pair_pon_shape_after_discard is False
    assert singleton_to_dict(tracked)["self_pair_retained_after_discard"] is False

    aggregate = YakuhaiDashAccumulator()
    aggregate.add(result)
    followup = aggregate.to_dict()["initial_singletons"]["all"]["self_pair_followup"]
    assert followup["pair_broken"]["count"] == 1
    assert followup["pon_shape"]["count"] == 0
    assert followup["pair_broken"]["denominator"] == 1
    assert followup["by_self_pair_draw_number"][0]["pair_broken"]["count"] == 1
    assert followup["by_self_pair_draw_number"][0]["pair_broken"]["denominator"] == 1


def test_self_pair_retained_after_first_discard_is_pon_shape() -> None:
    hands = make_hands({0: ["P"]})
    events = [
        {"type": "tsumo", "actor": 0, "pai": "P"},
        {
            "type": "dahai",
            "actor": 0,
            "pai": non_tile(hands[0], "P"),
            "tsumogiri": False,
        },
    ]
    result = analyze_yakuhai_dash_kyoku(make_kyoku(hands, events))
    tracked = observation(result, 0, "P")
    aggregate = YakuhaiDashAccumulator()
    aggregate.add(result)
    followup = aggregate.to_dict()["initial_singletons"]["all"]["self_pair_followup"]

    assert tracked.self_pair_retained_after_discard is True
    assert tracked.self_pair_pon_shape_after_discard is True
    assert followup["retained_pair"]["count"] == 1
    assert followup["pon_shape"]["count"] == 1
    assert followup["pon_shape_cumulative_by_pair_formation_draw"][0][
        "cumulative"
    ]["count"] == 1


@pytest.mark.parametrize("reach_before_pair", [False, True])
def test_riichi_pair_is_retained_but_cannot_pon(reach_before_pair: bool) -> None:
    hands = make_hands({0: ["P"]})
    events: list[dict[str, object]] = []
    if reach_before_pair:
        events.extend(
            [
                {"type": "tsumo", "actor": 0, "pai": "9s"},
                {"type": "reach", "actor": 0},
                {"type": "dahai", "actor": 0, "pai": "9s", "tsumogiri": True},
                {"type": "reach_accepted", "actor": 0},
            ]
        )
    events.append({"type": "tsumo", "actor": 0, "pai": "P"})
    if not reach_before_pair:
        events.append({"type": "reach", "actor": 0})
    events.append(
        {
            "type": "dahai",
            "actor": 0,
            "pai": non_tile(hands[0], "P"),
            "tsumogiri": False,
        }
    )
    result = analyze_yakuhai_dash_kyoku(make_kyoku(hands, events))
    tracked = observation(result, 0, "P")
    aggregate = YakuhaiDashAccumulator()
    aggregate.add(result)
    followup = aggregate.to_dict()["initial_singletons"]["all"]["self_pair_followup"]

    assert tracked.self_pair_retained_after_discard is True
    assert tracked.self_pair_pon_shape_after_discard is False
    assert followup["riichi_blocked"]["count"] == 1
    assert followup["retained_pair"]["count"] == 1
    assert followup["pon_shape"]["count"] == 0


def test_self_pair_without_followup_discard_remains_unknown() -> None:
    hands = make_hands({0: ["P"]})
    result = analyze_yakuhai_dash_kyoku(
        make_kyoku(hands, [{"type": "tsumo", "actor": 0, "pai": "P"}])
    )
    tracked = observation(result, 0, "P")
    aggregate = YakuhaiDashAccumulator()
    aggregate.add(result)
    followup = aggregate.to_dict()["initial_singletons"]["all"]["self_pair_followup"]

    assert tracked.self_pair_followup_discard_event_index is None
    assert tracked.self_pair_retained_after_discard is None
    assert tracked.self_pair_pon_shape_after_discard is None
    assert followup["no_followup_discard"]["count"] == 1


def test_self_pair_followup_after_kan_uses_original_pair_draw_cohort() -> None:
    hands = make_hands({0: ["P", "7s", "7s", "7s", "7s"]})
    events = [
        {"type": "tsumo", "actor": 0, "pai": "P"},
        {
            "type": "ankan",
            "actor": 0,
            "consumed": ["7s", "7s", "7s", "7s"],
        },
        {"type": "tsumo", "actor": 0, "pai": "8s"},
        {"type": "dahai", "actor": 0, "pai": "8s", "tsumogiri": True},
    ]
    result = analyze_yakuhai_dash_kyoku(make_kyoku(hands, events))
    tracked = observation(result, 0, "P")
    aggregate = YakuhaiDashAccumulator()
    aggregate.add(result)
    followup = aggregate.to_dict()["initial_singletons"]["all"]["self_pair_followup"]

    assert tracked.self_pair_draw_number == 1
    assert tracked.self_pair_followup_discard_event_index == 4
    assert tracked.self_pair_pon_shape_after_discard is True
    assert followup["by_self_pair_draw_number"][0][
        "pair_formation_self_draw_number"
    ] == 1
    assert followup["by_self_pair_draw_number"][0]["pon_shape"]["count"] == 1


def test_strict_dash_records_opponent_capability_and_actual_pon() -> None:
    hands = make_hands({0: ["P"], 1: ["P", "P"]})
    actor_one_discard = non_tile(hands[1], "P")
    events = [
        {"type": "tsumo", "actor": 0, "pai": "9s"},
        {"type": "dahai", "actor": 0, "pai": "P", "tsumogiri": False},
        {
            "type": "pon",
            "actor": 1,
            "target": 0,
            "pai": "P",
            "consumed": ["P", "P"],
        },
        {
            "type": "dahai",
            "actor": 1,
            "pai": actor_one_discard,
            "tsumogiri": False,
        },
    ]

    result = analyze_yakuhai_dash_kyoku(make_kyoku(hands, events))
    tracked = observation(result, 0, "P")
    opening = next(row for row in result.openings if row.actor == 0)

    assert tracked.opponent_ready_at_deal is True
    assert tracked.opponent_ready_actor_count_at_deal == 1
    assert tracked.outcome == "singleton_discard"
    assert tracked.singleton_discard_number == 1
    assert tracked.opponent_pon_capable_at_discard is True
    assert tracked.discard_was_ponned is True
    assert tracked.discard_pon_actor == 1
    assert tracked.discard_pon_event_index == 3
    assert opening.strict_dash is True
    assert opening.dash_by_2 is True
    assert opening.dash_by_3 is True


def test_pon_capable_opponent_can_decline_call() -> None:
    hands = make_hands({0: ["P"], 1: ["P", "P"]})
    events = [
        {"type": "tsumo", "actor": 0, "pai": "9s"},
        {"type": "dahai", "actor": 0, "pai": "P", "tsumogiri": False},
        {"type": "tsumo", "actor": 1, "pai": "8s"},
    ]

    tracked = observation(analyze_yakuhai_dash_kyoku(make_kyoku(hands, events)), 0, "P")

    assert tracked.opponent_pon_capable_at_discard is True
    assert tracked.discard_was_ponned is False


def test_opponent_ready_at_deal_can_stop_being_pon_capable() -> None:
    hands = make_hands({0: ["P"], 1: ["P", "P"]})
    events = [
        {"type": "tsumo", "actor": 1, "pai": "9s"},
        {"type": "dahai", "actor": 1, "pai": "P", "tsumogiri": False},
        {"type": "tsumo", "actor": 2, "pai": "8s"},
        {
            "type": "dahai",
            "actor": 2,
            "pai": non_tile(hands[2], "P"),
            "tsumogiri": False,
        },
        {"type": "tsumo", "actor": 0, "pai": "7s"},
        {"type": "dahai", "actor": 0, "pai": "P", "tsumogiri": False},
    ]

    tracked = observation(analyze_yakuhai_dash_kyoku(make_kyoku(hands, events)), 0, "P")

    assert tracked.opponent_ready_at_deal is True
    assert tracked.opponent_pon_capable_at_discard is False
    assert tracked.discard_was_ponned is False


def test_opponent_draw_can_become_pair_before_focal_resolution() -> None:
    hands = make_hands({0: ["P"], 1: ["P"]})
    actor_one_discard = non_tile(hands[1], "P")
    events = [
        {"type": "tsumo", "actor": 1, "pai": "P"},
        {
            "type": "dahai",
            "actor": 1,
            "pai": actor_one_discard,
            "tsumogiri": False,
        },
    ]

    result = analyze_yakuhai_dash_kyoku(make_kyoku(hands, events))
    focal = observation(result, 0, "P")
    opponent = observation(result, 1, "P")

    assert focal.opponent_ready_at_deal is False
    assert focal.first_opponent_pair_event_index == 1
    assert focal.first_opponent_pair_actor == 1
    assert focal.focal_draws_completed_at_opponent_pair == 0
    assert focal.outcome == "round_end"
    assert opponent.outcome == "self_pair_draw"
    assert opponent.self_pair_draw_number == 1


def test_pair_holder_can_later_pon_matching_discard() -> None:
    hands = make_hands({0: ["P"]})
    actor_zero_discard = non_tile(hands[0], "P")
    actor_zero_post_pon_discard = next(
        tile for tile in hands[0] if tile not in {actor_zero_discard, "P"}
    )
    events = [
        {"type": "tsumo", "actor": 0, "pai": "P"},
        {
            "type": "dahai",
            "actor": 0,
            "pai": actor_zero_discard,
            "tsumogiri": False,
        },
        {"type": "tsumo", "actor": 1, "pai": "P"},
        {"type": "dahai", "actor": 1, "pai": "P", "tsumogiri": True},
        {
            "type": "pon",
            "actor": 0,
            "target": 1,
            "pai": "P",
            "consumed": ["P", "P"],
        },
        {
            "type": "dahai",
            "actor": 0,
            "pai": actor_zero_post_pon_discard,
            "tsumogiri": False,
        },
    ]

    tracked = observation(analyze_yakuhai_dash_kyoku(make_kyoku(hands, events)), 0, "P")

    assert tracked.outcome == "self_pair_draw"
    assert tracked.focal_pon_event_index == 5


def test_accumulator_invariants_and_breakdowns() -> None:
    hands = make_hands({0: ["P"], 1: ["P", "P"], 2: ["W"]})
    actor_one_discard = non_tile(hands[1], "P")
    events = [
        {"type": "tsumo", "actor": 0, "pai": "9s"},
        {"type": "dahai", "actor": 0, "pai": "P", "tsumogiri": False},
        {
            "type": "pon",
            "actor": 1,
            "target": 0,
            "pai": "P",
            "consumed": ["P", "P"],
        },
        {
            "type": "dahai",
            "actor": 1,
            "pai": actor_one_discard,
            "tsumogiri": False,
        },
    ]
    result = analyze_yakuhai_dash_kyoku(make_kyoku(hands, events))
    accumulator = YakuhaiDashAccumulator()
    accumulator.add(result)

    document = accumulator.to_dict()

    assert document["initial_deals"]["all"]["player_deals"] == 4
    assert document["initial_deals"]["by_role"][0]["player_deals"] == 1
    assert document["initial_deals"]["by_role"][1]["player_deals"] == 3
    assert document["opening_behavior"]["all"]["strict_dash"]["count"] == 1
    all_singletons = document["initial_singletons"]["all"]
    assert all_singletons["actual_pon_at_singleton_discard"]["count"] == 1
    own_only = document["initial_singletons"]["by_exposure"][1]
    assert own_only["initial_singleton_observations"] >= 1


def test_after_singleton_discard_tracks_actual_log_without_changing_outcome() -> None:
    hands = make_hands({0: ["P"], 1: ["P"]})
    events = [
        {"type": "tsumo", "actor": 0, "pai": "9s"},
        {"type": "dahai", "actor": 0, "pai": "P", "tsumogiri": False},
        {"type": "tsumo", "actor": 1, "pai": "P"},
        {"type": "tsumo", "actor": 0, "pai": "P"},
    ]
    result = analyze_yakuhai_dash_kyoku(make_kyoku(hands, events))
    tracked = observation(result, 0, "P")

    assert tracked.outcome == "singleton_discard"
    assert tracked.first_opponent_pair_event_index is None
    assert tracked.post_discard_opponent_pair_event_index == 3
    assert tracked.post_discard_opponent_pair_actor == 1
    assert tracked.post_discard_self_draw_event_index == 4
    assert tracked.post_discard_self_draw_number == 2
    assert tracked.pair_race == "discard_before_either_pair"
    aggregate = YakuhaiDashAccumulator()
    aggregate.add(result)
    singleton = aggregate.to_dict()["initial_singletons"]["all"]
    assert singleton["post_discard_self_draw_in_actual_log"]["count"] == 1
    assert singleton["post_discard_opponent_pair_in_actual_log"]["count"] == 1


def test_riichi_pair_holder_is_not_pon_capable() -> None:
    hands = make_hands({0: ["P"], 1: ["P", "P"]})
    events = [
        {"type": "tsumo", "actor": 1, "pai": "9s"},
        {"type": "reach", "actor": 1},
        {
            "type": "dahai",
            "actor": 1,
            "pai": non_tile(hands[1], "P"),
            "tsumogiri": False,
        },
        {"type": "reach_accepted", "actor": 1},
        {"type": "tsumo", "actor": 0, "pai": "8s"},
        {"type": "dahai", "actor": 0, "pai": "P", "tsumogiri": False},
    ]
    result = analyze_yakuhai_dash_kyoku(make_kyoku(hands, events))
    tracked = observation(result, 0, "P")

    assert tracked.opponent_ready_at_deal is True
    assert tracked.opponent_pair_holder_actor_count == 1
    assert tracked.opponent_pon_capable_actor_count == 0
    assert tracked.opponent_pon_capable_at_discard is False
    aggregate = YakuhaiDashAccumulator()
    aggregate.add(result)
    singleton = aggregate.to_dict()["initial_singletons"]["all"]
    assert singleton["only_riichi_pair_holders_at_discard"]["count"] == 1


@pytest.mark.parametrize(
    ("allocations", "events", "expected"),
    [
        ({0: ["P"], 1: ["P", "P"]}, [], "opponent_at_deal"),
        ({0: ["P"]}, [{"type": "tsumo", "actor": 0, "pai": "P"}], "self_first"),
        (
            {0: ["P"], 1: ["P"]},
            [
                {"type": "tsumo", "actor": 1, "pai": "P"},
                {"type": "tsumo", "actor": 0, "pai": "P"},
            ],
            "opponent_first_after_deal",
        ),
        (
            {0: ["P"]},
            [
                {"type": "tsumo", "actor": 0, "pai": "9s"},
                {"type": "dahai", "actor": 0, "pai": "P", "tsumogiri": False},
            ],
            "discard_before_either_pair",
        ),
        ({0: ["P"]}, [], "round_end_before_either_pair"),
    ],
)
def test_first_pair_race_is_exclusive(allocations, events, expected) -> None:
    tracked = observation(
        analyze_yakuhai_dash_kyoku(make_kyoku(make_hands(allocations), events)),
        0,
        "P",
    )
    assert tracked.pair_race == expected


@pytest.mark.parametrize("discard_number", [1, 2, 3])
def test_dash_boundaries_and_first_discard_denominator(discard_number: int) -> None:
    hands = make_hands({0: ["P"]})
    filler = [tile for tile in hands[0] if tile != "P"]
    events = []
    for number in range(1, discard_number + 1):
        events.append({"type": "tsumo", "actor": 0, "pai": f"{10-number}s"})
        events.append(
            {
                "type": "dahai",
                "actor": 0,
                "pai": "P" if number == discard_number else filler[number - 1],
                "tsumogiri": False,
            }
        )
    result = analyze_yakuhai_dash_kyoku(make_kyoku(hands, events))
    opening = next(row for row in result.openings if row.actor == 0)
    assert opening.strict_dash is (discard_number == 1)
    assert opening.dash_by_2 is (discard_number <= 2)
    assert opening.dash_by_3 is True
    assert opening.opponent_discards_before_first_discard == 0

    no_discard = analyze_yakuhai_dash_kyoku(make_kyoku(hands))
    aggregate = YakuhaiDashAccumulator()
    aggregate.add(result)
    aggregate.add(no_discard)
    opening_counts = aggregate.to_dict()["opening_behavior"]["all"]
    assert opening_counts["first_discard_observed"]["count"] == 1
    assert opening_counts["strict_dash_given_first_discard"]["denominator"] == 1


def test_public_tile_visibility_includes_dora_marker_and_prior_discard() -> None:
    hands = make_hands({0: ["P"], 1: ["P"]})
    events = [
        {"type": "tsumo", "actor": 1, "pai": "9s"},
        {"type": "dahai", "actor": 1, "pai": "P", "tsumogiri": False},
        {"type": "tsumo", "actor": 0, "pai": "8s"},
        {
            "type": "dahai",
            "actor": 0,
            "pai": non_tile(hands[0], "P"),
            "tsumogiri": False,
        },
    ]
    kyoku = make_kyoku(hands, events)
    kyoku[0]["dora_marker"] = "P"
    result = analyze_yakuhai_dash_kyoku(kyoku)
    tracked = observation(result, 0, "P")
    opening = next(row for row in result.openings if row.actor == 0)
    assert tracked.public_same_tile_before_first_discard == 2
    assert opening.opponent_discards_before_first_discard == 1


def test_public_tile_visibility_includes_called_tiles() -> None:
    hands = make_hands({0: ["P"], 1: ["P"], 2: ["P", "P"]})
    events = [
        {"type": "tsumo", "actor": 1, "pai": "9s"},
        {"type": "dahai", "actor": 1, "pai": "P", "tsumogiri": False},
        {"type": "pon", "actor": 2, "target": 1, "pai": "P", "consumed": ["P", "P"]},
        {"type": "tsumo", "actor": 0, "pai": "8s"},
        {
            "type": "dahai",
            "actor": 0,
            "pai": non_tile(hands[0], "P"),
            "tsumogiri": False,
        },
    ]
    tracked = observation(analyze_yakuhai_dash_kyoku(make_kyoku(hands, events)), 0, "P")
    assert tracked.public_same_tile_before_first_discard == 3


def test_initial_dora_han_counts_red_five_as_dora_and_aka() -> None:
    hands = make_hands({0: ["5mr", "P"]})
    kyoku = make_kyoku(hands)
    kyoku[0]["dora_marker"] = "4m"
    deal = analyze_yakuhai_dash_kyoku(kyoku).deals[0]
    assert deal.dora_han == sum(normalize_tile(tile) == "5m" for tile in hands[0]) + 1
    assert deal.dora_han >= 2
    assert deal.all_pair_kind_count == sum(
        count >= 2 for count in tiles_to_counts(hands[0])
    )


def test_theoretical_probabilities_match_independent_combinations() -> None:
    expected_specific = 1 - (comb(132, 13) + 4 * comb(132, 12)) / comb(136, 13)
    assert theoretical_specific_pair_probability() == pytest.approx(expected_specific)
    assert theoretical_any_pair_probability(4) == pytest.approx(0.17298590000343073)
    assert theoretical_any_pair_probability(5) == pytest.approx(0.21250425345963392)
    assert theoretical_singleton_pair_by_draw(1) == pytest.approx(3 / 123)
    assert theoretical_singleton_pair_by_draw(6) == pytest.approx(0.1404099517217906)


def test_rejects_non_east_round_and_fifth_initial_copy() -> None:
    hands = make_hands({0: ["P"], 1: ["P"], 2: ["P"], 3: ["P"]})
    south = make_kyoku(hands)
    south[0]["bakaze"] = "S"
    with pytest.raises(ValueError, match="east-round"):
        analyze_yakuhai_dash_kyoku(south)

    invalid = make_kyoku(hands)
    invalid[0]["tehais"][0][1] = "P"
    with pytest.raises(ValueError, match="more than four"):
        analyze_yakuhai_dash_kyoku(invalid)
