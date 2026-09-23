"""Local wait-fragment counts used by the combo-theory analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import comb
from typing import Literal

from mahjong_analysis.tiles import (
    TILE_KINDS,
    index_to_tile,
    normalize_tile,
    tile_to_index,
)

SequenceWaitShape = Literal["ryanmen", "kanchan", "penchan"]
type UnseenCountsInput = Sequence[int] | Mapping[str, int]


@dataclass(frozen=True)
class SequenceWaitPattern:
    """One two-tile fragment completed by a suited candidate tile."""

    tiles: tuple[str, str]
    wait_shape: SequenceWaitShape


@dataclass(frozen=True)
class SequenceCombo:
    """The physical selections available for one sequence wait pattern."""

    pattern: SequenceWaitPattern
    count: int


@dataclass(frozen=True)
class SimpleComboBreakdown:
    """The specified simple combo sum and its non-overlap-corrected terms."""

    candidate_tile: str
    sequence_combos: tuple[SequenceCombo, ...]
    shanpon_combos: int
    tanki_combos: int

    @property
    def total(self) -> int:
        return (
            sum(component.count for component in self.sequence_combos)
            + self.shanpon_combos
            + self.tanki_combos
        )


def normalize_unseen_counts(unseen_counts: UnseenCountsInput) -> tuple[int, ...]:
    """Return validated unseen counts in canonical 34-tile order.

    A mapping may omit tile kinds, which then have count zero. Mapping keys must
    already be canonical tile kinds; red-five spellings are deliberately not a
    second count bucket.
    """
    if isinstance(unseen_counts, Mapping):
        counts = [0] * len(TILE_KINDS)
        for tile, count in unseen_counts.items():
            if not isinstance(tile, str):
                raise TypeError("unseen-count mapping keys must be tile strings")
            normalized = normalize_tile(tile)
            if normalized != tile:
                raise ValueError(f"unseen-count key must be normalized: {tile!r}")
            counts[tile_to_index(tile)] = _validate_unseen_count(count, tile)
        return tuple(counts)

    if isinstance(unseen_counts, (str, bytes)) or not isinstance(
        unseen_counts, Sequence
    ):
        raise TypeError("unseen_counts must be a sequence or mapping")
    if len(unseen_counts) != len(TILE_KINDS):
        raise ValueError("unseen-count sequence must contain exactly 34 values")
    return tuple(
        _validate_unseen_count(count, tile)
        for tile, count in zip(TILE_KINDS, unseen_counts, strict=True)
    )


def sequence_wait_patterns(candidate_tile: str) -> tuple[SequenceWaitPattern, ...]:
    """Return every suited two-tile sequence fragment completed by a candidate."""
    candidate = normalize_tile(candidate_tile)
    candidate_index = tile_to_index(candidate)
    if candidate_index >= 27:
        return ()

    suit_offset = (candidate_index // 9) * 9
    rank_index = candidate_index % 9
    patterns: list[SequenceWaitPattern] = []
    first_start = max(0, rank_index - 2)
    last_start = min(rank_index, 6)
    for start in range(first_start, last_start + 1):
        fragment_indices = tuple(
            suit_offset + position
            for position in range(start, start + 3)
            if position != rank_index
        )
        offset = rank_index - start
        wait_shape: SequenceWaitShape
        if offset == 1:
            wait_shape = "kanchan"
        elif (start == 0 and offset == 2) or (start == 6 and offset == 0):
            wait_shape = "penchan"
        else:
            wait_shape = "ryanmen"
        patterns.append(
            SequenceWaitPattern(
                tiles=(
                    index_to_tile(fragment_indices[0]),
                    index_to_tile(fragment_indices[1]),
                ),
                wait_shape=wait_shape,
            )
        )
    return tuple(patterns)


def calculate_simple_combo(
    candidate_tile: str,
    unseen_counts: UnseenCountsInput,
) -> SimpleComboBreakdown:
    """Calculate the specified uncorrected local combo sum for one tile kind."""
    candidate = normalize_tile(candidate_tile)
    counts = normalize_unseen_counts(unseen_counts)
    sequence_combos = tuple(
        SequenceCombo(
            pattern=pattern,
            count=(
                counts[tile_to_index(pattern.tiles[0])]
                * counts[tile_to_index(pattern.tiles[1])]
            ),
        )
        for pattern in sequence_wait_patterns(candidate)
    )
    candidate_unseen = counts[tile_to_index(candidate)]
    return SimpleComboBreakdown(
        candidate_tile=candidate,
        sequence_combos=sequence_combos,
        shanpon_combos=comb(candidate_unseen, 2),
        tanki_combos=candidate_unseen,
    )


def _validate_unseen_count(count: object, tile: str) -> int:
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError(f"unseen count for {tile} must be an integer")
    if not 0 <= count <= 4:
        raise ValueError(f"unseen count for {tile} must be between 0 and 4")
    return count
