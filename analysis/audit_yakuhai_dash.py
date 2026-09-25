"""Independently replay source lines referenced by yakuhai-dash audit samples.

This intentionally does not import the yakuhai-dash classifier or MJAI loader.
It checks the recorded sample facts against the original one-event-per-line logs.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WINDS = ("E", "S", "W", "N")
DRAGONS = ("P", "F", "C")
REDS = {"5mr": "5m", "5pr": "5p", "5sr": "5s"}
HONORS = set((*WINDS, *DRAGONS))
TILE_ORDER = (
    tuple(f"{rank}{suit}" for suit in "mps" for rank in range(1, 10))
    + WINDS + DRAGONS
)
ORPHAN_INDICES = (0, 8, 9, 17, 18, 26, *range(27, 34))
DEAL_CATEGORIES = {"initial_yakuhai_pair", "initial_no_yakuhai_pair"}
OPENING_CATEGORIES = {"first_discard_without_dash", "no_first_discard"}
SINGLETON_CATEGORIES = {
    "strict_dash",
    "self_pair_draw",
    "no_self_pair_draw",
    "opponent_ready_at_deal",
    "opponent_not_ready_at_deal",
    "opponent_became_ready_after_deal",
    "opponent_pon_capable_at_discard",
    "opponent_not_pon_capable_at_discard",
    "only_riichi_pair_holders_at_discard",
    "actual_pon",
    "post_discard_self_draw",
    "post_discard_opponent_pair",
    "own_only_seat_wind",
}
FOLLOWUP_CATEGORIES = {"self_pair_pon_shape", "self_pair_without_pon_shape"}
CHECKED_FOLLOWUP = (
    "self_pair_followup_discard_event_index",
    "self_pair_followup_discard_line",
    "self_pair_retained_after_discard",
    "self_pair_pon_shape_after_discard",
)
CHECKED_CONTEXT = (
    "bakaze",
    "kyoku",
    "honba",
    "oya",
    "east_kyoku_ordinal_in_game",
    "initial_hand",
    "role",
)
CHECKED_DEAL = (
    "seat_wind",
    "yakuhai_counts",
    "yakuhai_pair_kind_count",
    "singleton_kind_count",
    "max_suit_count",
    "honor_count",
    "terminal_honor_count",
    "all_pair_kind_count",
    "dora_han",
    "shanten",
)
CHECKED_SINGLETON = (
    "seat_wind",
    "yakuhai_class",
    "exposure",
    "opponent_ready_at_deal",
    "opponent_ready_actor_count_at_deal",
    "first_opponent_pair_event_index",
    "first_opponent_pair_actor",
    "focal_draws_completed_at_opponent_pair",
    "first_opponent_pair_line",
    "outcome",
    "self_pair_draw_number",
    "self_pair_event_index",
    "self_pair_line",
    "singleton_discard_number",
    "singleton_discard_event_index",
    "singleton_discard_line",
    "opponent_pon_capable_at_discard",
    "opponent_pon_capable_actor_count",
    "opponent_pair_holder_actor_count",
    "discard_was_ponned",
    "discard_pon_actor",
    "discard_pon_event_index",
    "discard_pon_line",
    "focal_pon_event_index",
    "focal_pon_line",
    "pair_race",
    "public_same_tile_before_first_discard",
    "post_discard_self_draw_number",
    "post_discard_self_draw_event_index",
    "post_discard_self_draw_line",
    "post_discard_opponent_pair_event_index",
    "post_discard_opponent_pair_actor",
    "post_discard_opponent_pair_line",
)


def normalize(tile: str) -> str:
    return REDS.get(tile, tile)


def _dora(marker: str) -> str:
    marker = normalize(marker)
    if marker in WINDS:
        return WINDS[(WINDS.index(marker) + 1) % 4]
    if marker in DRAGONS:
        return DRAGONS[(DRAGONS.index(marker) + 1) % 3]
    rank = int(marker[0])
    return f"{1 if rank == 9 else rank + 1}{marker[1]}"


def _seat(actor: int, dealer: int) -> str:
    return WINDS[(actor - dealer) % 4]


def reference_shanten(hand: list[str]) -> int:
    """Independent whole-hand DFS for standard, seven-pair, and orphan shapes."""
    if len(hand) != 13:
        raise ValueError("reference shanten requires a 13-tile hand")
    counts_map = Counter(normalize(tile) for tile in hand)
    counts = tuple(counts_map[tile] for tile in TILE_ORDER)
    if any(count > 4 for count in counts):
        raise ValueError("reference shanten hand has a fifth copy")

    @lru_cache(maxsize=None)
    def standard(tiles: tuple[int, ...], melds: int, blocks: int, head: int) -> int:
        first = next((index for index, count in enumerate(tiles) if count), None)
        if first is None:
            return 8 - 2 * melds - min(blocks, 4 - melds) - head

        def use(indices: tuple[int, ...], next_melds: int,
                next_blocks: int, next_head: int) -> int:
            remaining = list(tiles)
            for index in indices:
                remaining[index] -= 1
            return standard(tuple(remaining), next_melds, next_blocks, next_head)

        best = use((first,), melds, blocks, head)
        if tiles[first] >= 2:
            if head == 0:
                best = min(best, use((first, first), melds, blocks, 1))
            if blocks < 4:
                best = min(best, use((first, first), melds, blocks + 1, head))
        if tiles[first] >= 3 and melds < 4:
            best = min(
                best, use((first, first, first), melds + 1, blocks, head)
            )
        if first < 27:
            rank = first % 9
            if rank <= 7 and tiles[first + 1] and blocks < 4:
                best = min(
                    best, use((first, first + 1), melds, blocks + 1, head)
                )
            if rank <= 6 and tiles[first + 2] and blocks < 4:
                best = min(
                    best, use((first, first + 2), melds, blocks + 1, head)
                )
            if (
                rank <= 6 and tiles[first + 1] and tiles[first + 2]
                and melds < 4
            ):
                best = min(
                    best,
                    use((first, first + 1, first + 2), melds + 1, blocks, head),
                )
        return best

    pair_kinds = sum(count >= 2 for count in counts)
    unique_kinds = sum(count > 0 for count in counts)
    seven_pairs = 6 - pair_kinds + max(0, 7 - unique_kinds)
    unique_orphans = sum(counts[index] > 0 for index in ORPHAN_INDICES)
    orphan_pair = any(counts[index] >= 2 for index in ORPHAN_INDICES)
    orphans = 13 - unique_orphans - orphan_pair
    return min(standard(counts, 0, 0, 0), seven_pairs, orphans)


def _yakuhai(actor: int, dealer: int) -> tuple[str, ...]:
    kinds = {"E", _seat(actor, dealer), *DRAGONS}
    return tuple(tile for tile in (*WINDS, *DRAGONS) if tile in kinds)


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("rb") as header:
        compressed = header.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else Path.open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank MJAI line")
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError(f"{path}:{line_number}: MJAI event is not an object")
            events.append(event)
    return events


def _source_kyoku(
    events: list[dict[str, Any]], start_line: int
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    start = events[start_line - 1]
    if start.get("type") != "start_kyoku":
        raise ValueError(f"line {start_line} is not start_kyoku")
    ordinal = sum(
        event.get("type") == "start_kyoku" and event.get("bakaze") == "E"
        for event in events[:start_line]
    )
    ending = next(
        (
            index
            for index in range(start_line, len(events))
            if events[index].get("type") == "end_kyoku"
        ),
        None,
    )
    if ending is None:
        raise ValueError(f"line {start_line} has no matching end_kyoku")
    return start, events[start_line:ending], ordinal


def _deal_facts(start: dict[str, Any], actor: int) -> dict[str, Any]:
    hand = start["tehais"][actor]
    dealer = start["oya"]
    counts = Counter(normalize(tile) for tile in hand)
    yakuhai = _yakuhai(actor, dealer)
    dora = _dora(start["dora_marker"])
    suits = Counter(tile[-1] for tile in counts.elements() if tile[-1:] in "mps")
    return {
        "initial_hand": hand,
        "role": "dealer" if actor == dealer else "nondealer",
        "seat_wind": _seat(actor, dealer),
        "yakuhai_counts": [
            {"tile": tile, "copies": counts[tile]} for tile in yakuhai
        ],
        "yakuhai_pair_kind_count": sum(counts[tile] >= 2 for tile in yakuhai),
        "singleton_kind_count": sum(counts[tile] == 1 for tile in yakuhai),
        "max_suit_count": max(suits.values(), default=0),
        "honor_count": sum(counts[tile] for tile in HONORS),
        "terminal_honor_count": sum(
            count
            for tile, count in counts.items()
            if tile in HONORS or (tile[-1:] in "mps" and tile[0] in "19")
        ),
        "all_pair_kind_count": sum(count >= 2 for count in counts.values()),
        "dora_han": counts[dora] + sum(tile in REDS for tile in hand),
        "shanten": reference_shanten(hand),
    }


def _line(start_line: int, event_index: int | None) -> int | None:
    return None if event_index is None else start_line + event_index


def _replay(
    start: dict[str, Any], events: list[dict[str, Any]], actor: int,
    tile: str | None, start_line: int,
) -> dict[str, Any]:
    """Reconstruct selected facts directly from the raw chronological events."""
    dealer = start["oya"]
    hands = [[normalize(raw) for raw in hand] for hand in start["tehais"]]
    initial = Counter(hands[actor])
    initial_yakuhai = _yakuhai(actor, dealer)
    riichi: set[int] = set()
    declared_riichi: set[int] = set()
    draws = [0, 0, 0, 0]
    discards = [0, 0, 0, 0]
    first_discard_index: int | None = None
    opponent_discards_before_first: int | None = None
    first_discard_tile: str | None = None
    self_drawn_before_first: set[str] = set()
    public = Counter((normalize(start["dora_marker"]),))
    visible_before_first: int | None = None
    last_discard: tuple[int, str, int] | None = None

    opponent_pair_holders = (
        [other for other in range(4) if other != actor and hands[other].count(tile) >= 2]
        if tile is not None else []
    )
    facts: dict[str, Any] = {
        "opponent_ready_at_deal": bool(opponent_pair_holders),
        "opponent_ready_actor_count_at_deal": len(opponent_pair_holders),
        "first_opponent_pair_event_index": 0 if opponent_pair_holders else None,
        "first_opponent_pair_actor": opponent_pair_holders[0] if opponent_pair_holders else None,
        "focal_draws_completed_at_opponent_pair": 0 if opponent_pair_holders else None,
        "outcome": None,
        "self_pair_draw_number": None,
        "self_pair_event_index": None,
        "self_pair_followup_discard_event_index": None,
        "self_pair_retained_after_discard": None,
        "self_pair_pon_shape_after_discard": None,
        "singleton_discard_number": None,
        "singleton_discard_event_index": None,
        "opponent_pon_capable_at_discard": None,
        "opponent_pon_capable_actor_count": None,
        "opponent_pair_holder_actor_count": None,
        "discard_was_ponned": False,
        "discard_pon_actor": None,
        "discard_pon_event_index": None,
        "focal_pon_event_index": None,
        "post_discard_self_draw_number": None,
        "post_discard_self_draw_event_index": None,
        "post_discard_opponent_pair_event_index": None,
        "post_discard_opponent_pair_actor": None,
    }

    for event_index, event in enumerate(events, start=1):
        kind = event["type"]
        owner = event.get("actor")
        if kind == "tsumo":
            drawn = normalize(event["pai"])
            draws[owner] += 1
            if owner == actor and first_discard_index is None:
                self_drawn_before_first.add(drawn)
            if tile == drawn and owner == actor:
                if facts["outcome"] is None:
                    facts["outcome"] = "self_pair_draw"
                    facts["self_pair_draw_number"] = draws[actor]
                    facts["self_pair_event_index"] = event_index
                elif (
                    facts["outcome"] == "singleton_discard"
                    and facts["post_discard_self_draw_event_index"] is None
                ):
                    facts["post_discard_self_draw_number"] = draws[actor]
                    facts["post_discard_self_draw_event_index"] = event_index
            hands[owner].append(drawn)
            if tile == drawn and owner != actor:
                if (
                    facts["outcome"] is None
                    and facts["first_opponent_pair_event_index"] is None
                    and hands[owner].count(tile) >= 2
                ):
                    facts["first_opponent_pair_event_index"] = event_index
                    facts["first_opponent_pair_actor"] = owner
                    facts["focal_draws_completed_at_opponent_pair"] = draws[actor]
                if (
                    facts["outcome"] == "singleton_discard"
                    and facts["post_discard_opponent_pair_event_index"] is None
                    and hands[owner].count(tile) == 2
                ):
                    facts["post_discard_opponent_pair_event_index"] = event_index
                    facts["post_discard_opponent_pair_actor"] = owner
            last_discard = None
        elif kind == "dahai":
            discarded = normalize(event["pai"])
            if owner == actor and first_discard_index is None:
                first_discard_index = event_index
                first_discard_tile = discarded
                opponent_discards_before_first = sum(discards) - discards[actor]
                if tile is not None:
                    visible_before_first = public[tile]
            discards[owner] += 1
            if owner == actor and tile == discarded and facts["outcome"] is None:
                holders = [
                    other for other in range(4)
                    if other != actor and hands[other].count(tile) >= 2
                ]
                callable_holders = [other for other in holders if other not in riichi]
                facts["outcome"] = "singleton_discard"
                facts["singleton_discard_number"] = discards[actor]
                facts["singleton_discard_event_index"] = event_index
                facts["opponent_pon_capable_at_discard"] = bool(callable_holders)
                facts["opponent_pon_capable_actor_count"] = len(callable_holders)
                facts["opponent_pair_holder_actor_count"] = len(holders)
            hands[owner].remove(discarded)
            if (
                owner == actor and facts["outcome"] == "self_pair_draw"
                and facts["self_pair_followup_discard_event_index"] is None
            ):
                retained = hands[owner].count(tile) >= 2
                facts["self_pair_followup_discard_event_index"] = event_index
                facts["self_pair_retained_after_discard"] = retained
                facts["self_pair_pon_shape_after_discard"] = (
                    retained and owner not in riichi and owner not in declared_riichi
                )
            public[discarded] += 1
            last_discard = (owner, discarded, event_index)
        elif kind in {"chi", "pon", "daiminkan"}:
            called = normalize(event["pai"])
            if kind == "pon" and last_discard is not None:
                if (
                    tile == called
                    and last_discard == (
                        actor, tile, facts["singleton_discard_event_index"]
                    )
                ):
                    facts["discard_was_ponned"] = True
                    facts["discard_pon_actor"] = owner
                    facts["discard_pon_event_index"] = event_index
                if (
                    tile == called and owner == actor
                    and facts["outcome"] == "self_pair_draw"
                    and facts["focal_pon_event_index"] is None
                ):
                    facts["focal_pon_event_index"] = event_index
            for raw in event["consumed"]:
                used = normalize(raw)
                hands[owner].remove(used)
                public[used] += 1
            last_discard = None
        elif kind == "ankan":
            for raw in event["consumed"]:
                used = normalize(raw)
                hands[owner].remove(used)
                public[used] += 1
            last_discard = None
        elif kind == "kakan":
            added = normalize(event["pai"])
            hands[owner].remove(added)
            public[added] += 1
            last_discard = None
        elif kind == "reach_accepted":
            riichi.add(owner)
        elif kind == "reach":
            declared_riichi.add(owner)
            last_discard = None
        elif kind == "dora":
            public[normalize(event["dora_marker"])] += 1
            last_discard = None
        elif kind in {"hora", "ryukyoku"}:
            last_discard = None
        else:
            raise ValueError(f"unsupported source event {kind!r} at line {start_line + event_index}")

    facts["first_discard_line"] = _line(start_line, first_discard_index)
    facts["first_discard_event_index"] = first_discard_index
    facts["opponent_discards_before_first_discard"] = opponent_discards_before_first
    facts["actor_discard_count"] = discards[actor]
    facts["strict_dash"] = (
        first_discard_tile in initial_yakuhai
        and initial[first_discard_tile] == 1
        and first_discard_tile not in self_drawn_before_first
        if first_discard_tile is not None else False
    )
    if tile is None:
        return facts
    facts["outcome"] = facts["outcome"] or "round_end"
    facts["public_same_tile_before_first_discard"] = visible_before_first
    if tile in DRAGONS:
        yakuhai_class = "shared_dragon"
    elif tile == "E" and actor == dealer:
        yakuhai_class = "dealer_double_wind"
    elif tile == "E":
        yakuhai_class = "shared_round_wind"
    else:
        yakuhai_class = "own_seat_wind"
    facts["yakuhai_class"] = yakuhai_class
    facts["exposure"] = "own_only" if yakuhai_class == "own_seat_wind" else "shared"
    facts["seat_wind"] = _seat(actor, dealer)
    first_opponent = facts["first_opponent_pair_event_index"]
    first_self = facts["self_pair_event_index"]
    if facts["opponent_ready_at_deal"]:
        race = "opponent_at_deal"
    elif first_opponent is not None and (
        first_self is None or first_opponent < first_self
    ):
        race = "opponent_first_after_deal"
    elif first_self is not None:
        race = "self_first"
    elif facts["outcome"] == "singleton_discard":
        race = "discard_before_either_pair"
    else:
        race = "round_end_before_either_pair"
    facts["pair_race"] = race
    for prefix, index_field in (
        ("first_opponent_pair", "first_opponent_pair_event_index"),
        ("self_pair", "self_pair_event_index"),
        ("self_pair_followup_discard", "self_pair_followup_discard_event_index"),
        ("singleton_discard", "singleton_discard_event_index"),
        ("discard_pon", "discard_pon_event_index"),
        ("focal_pon", "focal_pon_event_index"),
        ("post_discard_self_draw", "post_discard_self_draw_event_index"),
        ("post_discard_opponent_pair", "post_discard_opponent_pair_event_index"),
    ):
        facts[f"{prefix}_line"] = _line(start_line, facts[index_field])
    return facts


def _belongs(category: str, facts: dict[str, Any]) -> bool:
    tests = {
        "initial_yakuhai_pair": facts.get("yakuhai_pair_kind_count", 0) >= 1,
        "initial_no_yakuhai_pair": facts.get("yakuhai_pair_kind_count", 0) == 0,
        "strict_dash": facts.get("singleton_discard_number") == 1,
        "first_discard_without_dash": (
            facts.get("singleton_kind_count", 0) > 0
            and facts.get("first_discard_line") is not None
            and not facts.get("strict_dash", False)
        ),
        "no_first_discard": (
            facts.get("singleton_kind_count", 0) > 0
            and facts.get("first_discard_line") is None
        ),
        "self_pair_draw": facts.get("outcome") == "self_pair_draw",
        "self_pair_pon_shape": facts.get("self_pair_pon_shape_after_discard") is True,
        "self_pair_without_pon_shape": (
            facts.get("outcome") == "self_pair_draw"
            and facts.get("self_pair_pon_shape_after_discard") is not True
        ),
        "no_self_pair_draw": facts.get("outcome") != "self_pair_draw",
        "opponent_ready_at_deal": facts.get("opponent_ready_at_deal") is True,
        "opponent_not_ready_at_deal": facts.get("opponent_ready_at_deal") is False,
        "opponent_became_ready_after_deal": (
            facts.get("first_opponent_pair_event_index") is not None
            and facts["first_opponent_pair_event_index"] > 0
        ),
        "opponent_pon_capable_at_discard": (
            facts.get("opponent_pon_capable_at_discard") is True
        ),
        "opponent_not_pon_capable_at_discard": (
            facts.get("opponent_pon_capable_at_discard") is False
        ),
        "only_riichi_pair_holders_at_discard": (
            (facts.get("opponent_pair_holder_actor_count") or 0) > 0
            and facts.get("opponent_pon_capable_at_discard") is False
        ),
        "actual_pon": facts.get("discard_was_ponned") is True,
        "post_discard_self_draw": (
            facts.get("post_discard_self_draw_event_index") is not None
        ),
        "post_discard_opponent_pair": (
            facts.get("post_discard_opponent_pair_event_index") is not None
        ),
        "own_only_seat_wind": facts.get("yakuhai_class") == "own_seat_wind",
    }
    return tests[category]


def _category_populations(annual: dict[str, Any], version: int) -> dict[str, int]:
    """Recover audit category sizes from the independently stored aggregates."""
    deals = annual["initial_deals"]["all"]
    openings = annual["opening_behavior"]["all"]
    singletons = annual["initial_singletons"]["all"]
    outcomes = {row["outcome"]: row["count"] for row in singletons["outcomes"]}
    n = singletons["initial_singleton_observations"]
    capable = singletons["opponent_pon_capable_at_singleton_discard"]["count"]
    populations = {
        "initial_yakuhai_pair": deals["any_yakuhai_pair_or_more"]["count"],
        "initial_no_yakuhai_pair": (
            deals["player_deals"] - deals["any_yakuhai_pair_or_more"]["count"]
        ),
        "strict_dash": openings["strict_dash"]["count"],
        "first_discard_without_dash": (
            openings["first_discard_observed"]["count"] - openings["strict_dash"]["count"]
        ),
        "no_first_discard": (
            openings["eligible_player_deals"] - openings["first_discard_observed"]["count"]
        ),
        "self_pair_draw": outcomes["self_pair_draw"],
        "no_self_pair_draw": n - outcomes["self_pair_draw"],
        "opponent_ready_at_deal": singletons["opponent_ready_at_deal"]["count"],
        "opponent_not_ready_at_deal": n - singletons["opponent_ready_at_deal"]["count"],
        # focal_draws_completed == 0 also includes opponents drawing before
        # the focal player's first draw, so subtract only true deal holders.
        "opponent_became_ready_after_deal": (
            sum(row["exact_count"] for row in singletons[
                "opponent_first_pair_by_focal_draws_completed"
            ]) - singletons["opponent_ready_at_deal"]["count"]
        ),
        "opponent_pon_capable_at_discard": capable,
        "opponent_not_pon_capable_at_discard": outcomes["singleton_discard"] - capable,
        "only_riichi_pair_holders_at_discard": singletons[
            "only_riichi_pair_holders_at_discard"
        ]["count"],
        "actual_pon": singletons["actual_pon_at_singleton_discard"]["count"],
        "post_discard_self_draw": singletons["post_discard_self_draw_in_actual_log"]["count"],
        "post_discard_opponent_pair": singletons[
            "post_discard_opponent_pair_in_actual_log"
        ]["count"],
        "own_only_seat_wind": next(
            row["initial_singleton_observations"]
            for row in annual["initial_singletons"]["by_yakuhai_class"]
            if row["yakuhai_class"] == "own_seat_wind"
        ),
    }
    if version >= 3:
        pon_shape = singletons["self_pair_followup"]["pon_shape"]["count"]
        populations["self_pair_pon_shape"] = pon_shape
        populations["self_pair_without_pon_shape"] = outcomes["self_pair_draw"] - pon_shape
    if any(type(count) is not int or count < 0 for count in populations.values()):
        raise ValueError("invalid audit category population in annual aggregates")
    return populations


def _validate_audit_coverage(document: dict[str, Any], version: int) -> None:
    years = document["metadata"]["years"]
    if (
        not isinstance(years, list) or not years
        or any(type(year) is not int or not 2020 <= year <= 2025 for year in years)
        or len(set(years)) != len(years)
    ):
        raise ValueError("metadata.years must contain unique analysis years")
    blocks = document["audit_samples_by_year"]
    annuals = document["years"]
    for label, rows in (("audit_samples_by_year", blocks), ("years", annuals)):
        if not isinstance(rows, list) or Counter(row["year"] for row in rows) != Counter(years):
            raise ValueError(f"{label} does not cover metadata.years exactly once")
    annual_by_year = {row["year"]: row for row in annuals}
    limit = None
    if version >= 3:
        limit = document["metadata"]["execution"]["audit_samples_per_category"]
        if type(limit) is not int or limit < 1:
            raise ValueError("saved-sample audit requires audit_samples_per_category >= 1")
    expected = DEAL_CATEGORIES | OPENING_CATEGORIES | SINGLETON_CATEGORIES
    if version >= 3:
        expected |= FOLLOWUP_CATEGORIES
    total = 0
    for block in blocks:
        year = block["year"]
        groups = block["categories"]
        if Counter(group["category"] for group in groups) != Counter({key: 1 for key in expected}):
            raise ValueError(f"{year}: missing, duplicate, or unexpected audit categories")
        populations = _category_populations(annual_by_year[year], version)
        year_total = 0
        for group in groups:
            category = group["category"]
            population = populations[category]
            samples = group["samples"]
            if not isinstance(samples, list):
                raise ValueError(f"{year}/{category}: samples must be a list")
            if version >= 3:
                if type(group["population_count"]) is not int or group["population_count"] != population:
                    raise ValueError(f"{year}/{category}: population_count disagrees with aggregates")
                expected_count = min(limit, population)
                if len(samples) != expected_count:
                    raise ValueError(f"{year}/{category}: expected {expected_count} samples, got {len(samples)}")
            elif len(samples) > population or (population > 0 and not samples):
                # v2 did not record the configured cap, so exact cap coverage
                # cannot be reconstructed. Still reject empty populated groups.
                raise ValueError(f"{year}/{category}: absent or excessive legacy audit samples")
            required = {"relative_source_path", "start_kyoku_line", "actor", *CHECKED_CONTEXT}
            if category in DEAL_CATEGORIES:
                required.update(CHECKED_DEAL)
            elif category in OPENING_CATEGORIES:
                required.update(("dora_han", "shanten"))
                if category == "first_discard_without_dash":
                    required.add("first_discard_line")
            else:
                required.update(("tile", *CHECKED_SINGLETON))
                if version >= 3:
                    required.update(CHECKED_FOLLOWUP)
            keys = set()
            for sample in samples:
                missing = required - sample.keys()
                if missing:
                    raise ValueError(f"{year}/{category}: missing required fields {sorted(missing)}")
                key = (sample["relative_source_path"], sample["start_kyoku_line"], sample["actor"], sample.get("tile"))
                if key in keys:
                    raise ValueError(f"{year}/{category}: duplicate audit observation")
                keys.add(key)
            year_total += len(samples)
        if not year_total:
            raise ValueError(f"{year}: no saved audit observations")
        total += year_total
    if not total:
        raise ValueError("no saved audit observations")


def audit_document(document: dict[str, Any], data_root: Path) -> dict[str, Any]:
    """Check every annual audit entry and return a compact discrepancy report."""
    version = document["metadata"]["schema_version"]
    if type(version) is not int or version not in (2, 3):
        raise ValueError("expected yakuhai-dash schema version 2 or 3")
    try:
        _validate_audit_coverage(document, version)
    except (KeyError, IndexError, TypeError, ValueError, StopIteration, AttributeError) as error:
        return {
            "status": "fail", "source_files": 0, "category_entries": 0,
            "unique_source_observations": 0, "yearly_counts": [],
            "failure_count": 1,
            "failures": [{"field": "audit_coverage", "message": str(error)}],
        }
    root = data_root.resolve()
    source_cache: dict[Path, list[dict[str, Any]]] = {}
    failures: list[dict[str, Any]] = []
    category_entries = 0
    unique_observations: set[tuple[str, int, int, str | None]] = set()
    yearly_counts: list[dict[str, Any]] = []

    for year_block in document["audit_samples_by_year"]:
        year = year_block["year"]
        yearly_entries = 0
        for group in year_block["categories"]:
            category = group["category"]
            if category not in DEAL_CATEGORIES | OPENING_CATEGORIES | SINGLETON_CATEGORIES | FOLLOWUP_CATEGORIES:
                raise ValueError(f"unknown audit category {category!r}")
            for sample in group["samples"]:
                category_entries += 1
                yearly_entries += 1
                relative = Path(sample["relative_source_path"])
                source = (root / relative).resolve()
                if not source.is_relative_to(root) or relative.parts[0] != str(year):
                    raise ValueError(f"invalid source path for {year}: {relative}")
                key = (
                    sample["relative_source_path"], sample["start_kyoku_line"],
                    sample["actor"], sample.get("tile"),
                )
                unique_observations.add(key)
                if source not in source_cache:
                    source_cache[source] = _read_events(source)
                events = source_cache[source]
                try:
                    start, round_events, ordinal = _source_kyoku(
                        events, sample["start_kyoku_line"]
                    )
                    actor = sample["actor"]
                    deal = _deal_facts(start, actor)
                    tile = sample.get("tile")
                    replay = _replay(
                        start, round_events, actor, tile,
                        sample["start_kyoku_line"],
                    )
                    facts = {
                        "bakaze": start["bakaze"],
                        "kyoku": start["kyoku"],
                        "honba": start["honba"],
                        "oya": start["oya"],
                        "east_kyoku_ordinal_in_game": ordinal,
                        **deal,
                        **replay,
                    }
                    if tile is not None:
                        if tile not in _yakuhai(actor, start["oya"]):
                            raise ValueError("sample tile is not focal yakuhai")
                        if Counter(normalize(raw) for raw in start["tehais"][actor])[tile] != 1:
                            raise ValueError("sample tile was not an initial singleton")
                    fields = CHECKED_CONTEXT
                    if category in DEAL_CATEGORIES:
                        fields += CHECKED_DEAL
                    elif category in OPENING_CATEGORIES:
                        fields += ("dora_han", "shanten")
                        if category == "first_discard_without_dash":
                            fields += ("first_discard_line",)
                    else:
                        fields += CHECKED_SINGLETON
                        if version >= 3:
                            fields += CHECKED_FOLLOWUP
                    for field in fields:
                        if sample[field] != facts.get(field):
                            failures.append({
                                "year": year,
                                "category": category,
                                "source": sample["relative_source_path"],
                                "start_line": sample["start_kyoku_line"],
                                "actor": actor,
                                "tile": tile,
                                "field": field,
                                "recorded": sample[field],
                                "source_replay": facts.get(field),
                            })
                    if not _belongs(category, facts):
                        failures.append({
                            "year": year,
                            "category": category,
                            "source": sample["relative_source_path"],
                            "start_line": sample["start_kyoku_line"],
                            "actor": actor,
                            "tile": tile,
                            "field": "category_membership",
                            "recorded": True,
                            "source_replay": False,
                        })
                except (KeyError, IndexError, TypeError, ValueError) as error:
                    failures.append({
                        "year": year,
                        "category": category,
                        "source": sample["relative_source_path"],
                        "start_line": sample["start_kyoku_line"],
                        "actor": sample["actor"],
                        "tile": sample.get("tile"),
                        "field": "source_replay_error",
                        "message": str(error),
                    })
        yearly_counts.append({"year": year, "category_entries": yearly_entries})

    return {
        "status": "pass" if not failures else "fail",
        "source_files": len(source_cache),
        "category_entries": category_entries,
        "unique_source_observations": len(unique_observations),
        "yearly_counts": yearly_counts,
        "coverage_count_policy": (
            "exact_minimum_of_configured_cap_and_population" if version >= 3
            else "legacy_v2_cap_unavailable_nonempty_populated_categories"
        ),
        "shanten_check": "independent whole-hand DFS for all saved deal/opening samples",
        "failure_count": len(failures),
        "failures": failures[:50],
    }


def audit_blind_logs(
    data_root: Path, years: list[int], logs_per_year: int = 24
) -> dict[str, Any]:
    """Compare production observations to the independent replay across each year.

    Evenly spaced filenames cover the full year's inventory instead of the
    first-few-log samples embedded in the result document. All expected initial
    singleton tiles are enumerated from raw starting hands, detecting omissions.
    """
    if logs_per_year < 1:
        raise ValueError("logs_per_year must be positive")
    if not years or len(set(years)) != len(years):
        raise ValueError("blind audit requires nonempty, unique years")
    from mahjong_analysis.yakuhai_dash import (
        analyze_yakuhai_dash_kyoku,
        deal_to_dict,
        singleton_to_dict,
    )

    failures: list[dict[str, Any]] = []
    yearly_counts: list[dict[str, int]] = []
    total_logs = total_kyokus = total_deals = total_singletons = 0

    def compare(
        path: Path, start_line: int, actor: int, tile: str | None,
        field: str, recorded: Any, replayed: Any,
    ) -> None:
        if recorded != replayed:
            failures.append({
                "source": str(path), "start_line": start_line,
                "actor": actor, "tile": tile, "field": field,
                "production": recorded, "source_replay": replayed,
            })

    for year in years:
        paths = sorted((data_root / str(year)).glob("*-00a9-*.mjson"))
        if not paths:
            raise FileNotFoundError(f"no logs for {year}")
        count = min(logs_per_year, len(paths))
        indices = sorted({
            round(index * (len(paths) - 1) / max(1, count - 1))
            for index in range(count)
        })
        year_logs = year_kyokus = year_singletons = 0
        for position in indices:
            path = paths[position]
            source_events = _read_events(path)
            if (
                "-00a9-" not in path.name
                or source_events[0].get("type") != "start_game"
                or source_events[0].get("aka_flag") is not True
            ):
                continue
            year_logs += 1
            line_index = 0
            while line_index < len(source_events):
                start = source_events[line_index]
                if start.get("type") != "start_kyoku":
                    line_index += 1
                    continue
                start_line = line_index + 1
                end_index = next(
                    (
                        index for index in range(line_index + 1, len(source_events))
                        if source_events[index].get("type") == "end_kyoku"
                    ),
                    None,
                )
                if end_index is None:
                    raise ValueError(f"{path}:{start_line}: missing end_kyoku")
                kyoku = source_events[line_index:end_index + 1]
                line_index = end_index + 1
                if start["bakaze"] != "E":
                    continue
                try:
                    production = analyze_yakuhai_dash_kyoku(kyoku)
                    expected_singletons: set[tuple[int, str]] = set()
                    for actor in range(4):
                        deal = _deal_facts(start, actor)
                        actual_deal = deal_to_dict(production.deals[actor])
                        for field in (*CHECKED_CONTEXT[-2:], *CHECKED_DEAL):
                            compare(
                                path, start_line, actor, None, field,
                                actual_deal.get(field), deal.get(field),
                            )
                        counts = Counter(
                            normalize(raw) for raw in start["tehais"][actor]
                        )
                        expected_singletons.update(
                            (actor, tile) for tile in _yakuhai(actor, start["oya"])
                            if counts[tile] == 1
                        )
                    recorded_singletons = {
                        (row.actor, row.tile) for row in production.singletons
                    }
                    compare(
                        path, start_line, -1, None, "singleton_keys",
                        sorted(recorded_singletons), sorted(expected_singletons),
                    )
                    compare(
                        path, start_line, -1, None, "singleton_record_count",
                        len(production.singletons), len(expected_singletons),
                    )
                    singleton_facts: dict[tuple[int, str], dict[str, Any]] = {}
                    for row in production.singletons:
                        facts = _replay(
                            start, kyoku[1:-1], row.actor, row.tile, start_line
                        )
                        singleton_facts[(row.actor, row.tile)] = facts
                        actual = singleton_to_dict(row)
                        for field in (*CHECKED_SINGLETON, *CHECKED_FOLLOWUP):
                            # Source line fields are added by the CLI, not by
                            # singleton_to_dict; every other field is required.
                            if not field.endswith("_line"):
                                compare(
                                    path, start_line, row.actor, row.tile, field,
                                    actual[field], facts.get(field),
                                )
                    opening_by_actor = {row.actor: row for row in production.openings}
                    expected_opening_actors = {
                        actor for actor, _ in expected_singletons
                    }
                    compare(
                        path, start_line, -1, None, "opening_actors",
                        sorted(opening_by_actor), sorted(expected_opening_actors),
                    )
                    for actor, opening in opening_by_actor.items():
                        timeline = _replay(start, kyoku[1:-1], actor, None, start_line)
                        discard_numbers = [
                            facts["singleton_discard_number"]
                            for (owner, _), facts in singleton_facts.items()
                            if owner == actor
                            and facts["singleton_discard_number"] is not None
                        ]
                        expected_opening = {
                            "initial_singleton_count": sum(
                                owner == actor for owner, _ in expected_singletons
                            ),
                            "strict_dash": 1 in discard_numbers,
                            "dash_by_2": any(n <= 2 for n in discard_numbers),
                            "dash_by_3": any(n <= 3 for n in discard_numbers),
                            "first_discard_event_index": timeline[
                                "first_discard_event_index"
                            ],
                            "opponent_discards_before_first_discard": timeline[
                                "opponent_discards_before_first_discard"
                            ],
                            "actor_discard_count": timeline["actor_discard_count"],
                        }
                        for field, expected in expected_opening.items():
                            compare(
                                path, start_line, actor, None, field,
                                getattr(opening, field), expected,
                            )
                except (IndexError, KeyError, TypeError, ValueError) as error:
                    failures.append({
                        "source": str(path), "start_line": start_line,
                        "field": "blind_replay_error", "message": str(error),
                    })
                    continue
                year_kyokus += 1
                year_singletons += len(expected_singletons)
        if not year_logs or not year_kyokus:
            failures.append({
                "year": year, "field": "blind_audit_coverage",
                "message": "no target logs or east-round observations audited",
            })
        yearly_counts.append({
            "year": year, "source_logs": year_logs,
            "east_kyokus": year_kyokus,
            "initial_singletons": year_singletons,
        })
        total_logs += year_logs
        total_kyokus += year_kyokus
        total_deals += year_kyokus * 4
        total_singletons += year_singletons

    return {
        "status": "pass" if not failures else "fail",
        "selection": "evenly_spaced_sorted_00a9_filenames_including_first_and_last",
        "source_logs": total_logs,
        "east_kyokus": total_kyokus,
        "player_deals": total_deals,
        "initial_singletons": total_singletons,
        "yearly_counts": yearly_counts,
        "failure_count": len(failures),
        "failures": failures[:50],
    }


def compare_v2_v3_aggregates(
    legacy: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    """Ensure adding the follow-up metric did not alter earlier full counts."""
    if legacy["metadata"]["schema_version"] != 2:
        raise ValueError("legacy comparison requires schema version 2")
    if current["metadata"]["schema_version"] != 3:
        raise ValueError("current comparison requires schema version 3")

    def without_followup(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: without_followup(item)
                for key, item in value.items()
                if key != "self_pair_followup"
            }
        if isinstance(value, list):
            return [without_followup(item) for item in value]
        return value

    checks = {
        "analysis_years": (
            legacy["metadata"]["years"] == current["metadata"]["years"]
        ),
        "input_counts": legacy["input_counts"] == current["input_counts"],
        "overall_preexisting_aggregates": (
            legacy["overall"] == without_followup(current["overall"])
        ),
        "annual_preexisting_aggregates": (
            legacy["years"] == without_followup(current["years"])
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=PROJECT_ROOT / "outputs" / "yakuhai-dash" / "summary-v3.json",
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--blind-logs-per-year", type=int, default=24)
    parser.add_argument(
        "--compare-v2", type=Path,
        help="compare all pre-existing annual and overall aggregates to a v2 result",
    )
    args = parser.parse_args(argv)
    document = json.loads(args.input.read_text(encoding="utf-8"))
    saved = audit_document(document, args.data_root)
    blind = audit_blind_logs(
        args.data_root, document["metadata"]["years"],
        args.blind_logs_per_year,
    )
    legacy = (
        compare_v2_v3_aggregates(
            json.loads(args.compare_v2.read_text(encoding="utf-8")), document
        )
        if args.compare_v2 is not None else None
    )
    result = {
        "status": (
            "pass" if saved["status"] == blind["status"] == "pass"
            and (legacy is None or legacy["status"] == "pass") else "fail"
        ),
        "saved_audit_samples": saved,
        "blind_full_year_spread": blind,
    }
    if legacy is not None:
        result["legacy_aggregate_consistency"] = legacy
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
