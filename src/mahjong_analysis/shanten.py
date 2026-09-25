"""Shanten of a closed 13-tile hand, independent of MJAI replay."""

from __future__ import annotations

from functools import lru_cache

from mahjong_analysis.tiles import tiles_to_counts

_ORPHANS = (0, 8, 9, 17, 18, 26, *range(27, 34))


def initial_shanten(tiles: tuple[str, ...] | list[str]) -> int:
    """Return minimum standard, seven-pairs, or thirteen-orphans shanten."""
    if len(tiles) != 13:
        raise ValueError("initial shanten requires exactly 13 tiles")
    counts = tiles_to_counts(tiles)
    unique = sum(bool(count) for count in counts)
    pair_kinds = sum(count >= 2 for count in counts)
    chiitoitsu = 6 - pair_kinds + max(0, 7 - unique)
    kokushi = 13 - sum(bool(counts[index]) for index in _ORPHANS) - any(
        counts[index] >= 2 for index in _ORPHANS
    )
    groups = (
        _group_shapes(counts[0:9], True),
        _group_shapes(counts[9:18], True),
        _group_shapes(counts[18:27], True),
        _group_shapes(counts[27:34], False),
    )
    combined = {(0, 0, 0)}
    for group in groups:
        next_combined: set[tuple[int, int, int]] = set()
        for melds, taatsu, head in combined:
            for gm, gt, gh in group:
                if melds + gm <= 4 and head + gh <= 1:
                    next_combined.add(
                        (melds + gm, min(4, taatsu + gt), head + gh)
                    )
        combined = next_combined
    standard = min(
        8 - 2 * melds - min(taatsu, 4 - melds) - head
        for melds, taatsu, head in combined
    )
    return min(standard, chiitoitsu, kokushi)


@lru_cache(maxsize=65_536)
def _group_shapes(
    counts: tuple[int, ...], suited: bool
) -> frozenset[tuple[int, int, int]]:
    """Possible (meld, taatsu, head) combinations within a suit or honors."""
    first = next((index for index, count in enumerate(counts) if count), None)
    if first is None:
        return frozenset({(0, 0, 0)})

    options: list[tuple[tuple[int, ...], int, int, int]] = []

    def subtract(indices: tuple[int, ...], meld: int, taatsu: int, head: int) -> None:
        remainder = list(counts)
        for index in indices:
            remainder[index] -= 1
        options.append((tuple(remainder), meld, taatsu, head))

    subtract((first,), 0, 0, 0)
    if counts[first] >= 2:
        subtract((first, first), 0, 1, 0)
        subtract((first, first), 0, 0, 1)
    if counts[first] >= 3:
        subtract((first, first, first), 1, 0, 0)
    if suited:
        if first <= 7 and counts[first + 1]:
            subtract((first, first + 1), 0, 1, 0)
        if first <= 6 and counts[first + 2]:
            subtract((first, first + 2), 0, 1, 0)
        if first <= 6 and counts[first + 1] and counts[first + 2]:
            subtract((first, first + 1, first + 2), 1, 0, 0)

    results: set[tuple[int, int, int]] = set()
    for remainder, meld, taatsu, head in options:
        for rm, rt, rh in _group_shapes(remainder, suited):
            if meld + rm <= 4 and head + rh <= 1:
                results.add((meld + rm, min(4, taatsu + rt), head + rh))
    return frozenset(results)
