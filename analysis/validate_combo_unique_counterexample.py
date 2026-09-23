"""Independently validate exact candidate-wait counts by hand-set generation."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Any

from mahjong_analysis.mjai import load_mjai, split_kyoku
from mahjong_analysis.post_riichi_decisions import (
    PostRiichiDrawDecision,
    extract_post_riichi_draw_decisions,
)
from mahjong_analysis.tiles import tile_to_index
from mahjong_analysis.unique_tenpai_states import count_candidate_wait_states

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT / "data" / "raw" / "2020" / "2020010100gm-00a9-0000-03aa14b0.mjson"
)

_MELDS = tuple(
    tuple(3 if index == tile else 0 for index in range(34)) for tile in range(34)
) + tuple(
    tuple(1 if suit + start <= index < suit + start + 3 else 0 for index in range(34))
    for suit in (0, 9, 18)
    for start in range(7)
)
_ORPHANS = (0, 8, 9, 17, 18, 26, *range(27, 34))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one real-position exact combo counterexample."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--kyoku-index", type=int, default=0)
    parser.add_argument("--tsumo-event-index", type=int, default=63)
    parser.add_argument("--tiles", nargs="+", default=("5m", "4s"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    kyoku = split_kyoku(load_mjai(args.source))[args.kyoku_index]
    decision = _find_decision(kyoku, args.tsumo_event_index)
    if decision.target_concealed_tile_count != 13:
        raise ValueError("this independent validator currently requires no fixed meld")

    meld_vectors = _meld_vectors_by_count()
    rows = []
    for tile in args.tiles:
        generated = generate_waiting_hands(
            decision.unseen_counts,
            tile,
            meld_vectors,
        )
        generated_weight = sum(
            _physical_weight(decision.unseen_counts, hand) for hand in generated
        )
        factored = count_candidate_wait_states(decision.unseen_counts, tile)
        row = {
            "tile": tile,
            "generated_state_count": len(generated),
            "generated_physical_weight": generated_weight,
            "factored_state_count": factored.state_count,
            "factored_physical_weight": factored.physical_weight,
            "matches": (
                len(generated) == factored.state_count
                and generated_weight == factored.physical_weight
            ),
        }
        rows.append(row)
        if not row["matches"]:
            raise AssertionError(f"independent count mismatch for {tile}: {row}")

    print(
        json.dumps(
            {
                "source": args.source.resolve().as_posix(),
                "kyoku_index": args.kyoku_index,
                "tsumo_event_index": args.tsumo_event_index,
                "observer_actor": decision.observer_actor,
                "target_actor": decision.target_actor,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def generate_waiting_hands(
    unseen_counts: tuple[int, ...],
    candidate_tile: str,
    meld_vectors: tuple[frozenset[tuple[int, ...]], ...],
) -> set[tuple[int, ...]]:
    """Generate and deduplicate 13-tile hands through explicit wait fragments."""
    candidate_index = tile_to_index(candidate_tile)
    hands: set[tuple[int, ...]] = set()

    tanki = [0] * 34
    tanki[candidate_index] = 1
    for melds in meld_vectors[4]:
        _add_if_feasible(hands, tanki, melds, None, candidate_index, unseen_counts)

    shanpon = [0] * 34
    shanpon[candidate_index] = 2
    fragments = [tuple(shanpon)]
    fragments.extend(_sequence_fragments(candidate_index))
    for fragment in fragments:
        for melds in meld_vectors[3]:
            for pair_index in range(34):
                _add_if_feasible(
                    hands,
                    fragment,
                    melds,
                    pair_index,
                    candidate_index,
                    unseen_counts,
                )

    pair_kinds = [
        index
        for index, available in enumerate(unseen_counts)
        if index != candidate_index and available >= 2
    ]
    if unseen_counts[candidate_index] >= 1:
        for other_pairs in combinations(pair_kinds, 6):
            hand = [0] * 34
            hand[candidate_index] = 1
            for index in other_pairs:
                hand[index] = 2
            hands.add(tuple(hand))

    if candidate_index in _ORPHANS:
        for pair_index in _ORPHANS:
            completed = [0] * 34
            for index in _ORPHANS:
                completed[index] = 1
            completed[pair_index] += 1
            completed[candidate_index] -= 1
            hand = tuple(completed)
            if _is_feasible(hand, candidate_index, unseen_counts):
                hands.add(hand)
    return hands


def _meld_vectors_by_count() -> tuple[frozenset[tuple[int, ...]], ...]:
    levels: list[set[tuple[int, ...]]] = [{(0,) * 34}]
    for _ in range(4):
        next_level: set[tuple[int, ...]] = set()
        for current in levels[-1]:
            for meld in _MELDS:
                combined = tuple(current[index] + meld[index] for index in range(34))
                if max(combined) <= 4:
                    next_level.add(combined)
        levels.append(next_level)
    return tuple(frozenset(level) for level in levels)


def _sequence_fragments(candidate_index: int) -> tuple[tuple[int, ...], ...]:
    if candidate_index >= 27:
        return ()
    suit = (candidate_index // 9) * 9
    rank = candidate_index % 9
    fragments = []
    for start in range(max(0, rank - 2), min(rank, 6) + 1):
        fragment = [0] * 34
        for local_index in range(start, start + 3):
            if local_index != rank:
                fragment[suit + local_index] += 1
        fragments.append(tuple(fragment))
    return tuple(fragments)


def _add_if_feasible(
    hands: set[tuple[int, ...]],
    fragment: tuple[int, ...] | list[int],
    melds: tuple[int, ...],
    pair_index: int | None,
    candidate_index: int,
    unseen_counts: tuple[int, ...],
) -> None:
    hand = tuple(
        fragment[index] + melds[index] + (2 if pair_index == index else 0)
        for index in range(34)
    )
    if _is_feasible(hand, candidate_index, unseen_counts):
        hands.add(hand)


def _is_feasible(
    hand: tuple[int, ...],
    candidate_index: int,
    unseen_counts: tuple[int, ...],
) -> bool:
    return (
        sum(hand) == 13
        and hand[candidate_index] < 4
        and all(
            held <= available
            for held, available in zip(hand, unseen_counts, strict=True)
        )
    )


def _physical_weight(
    unseen_counts: tuple[int, ...],
    hand: tuple[int, ...],
) -> int:
    weight = 1
    for available, held in zip(unseen_counts, hand, strict=True):
        weight *= comb(available, held)
    return weight


def _find_decision(
    kyoku: list[dict[str, Any]],
    tsumo_event_index: int,
) -> PostRiichiDrawDecision:
    return next(
        decision
        for decision in extract_post_riichi_draw_decisions(kyoku)
        if decision.tsumo_event_index == tsumo_event_index
    )


if __name__ == "__main__":
    raise SystemExit(main())
