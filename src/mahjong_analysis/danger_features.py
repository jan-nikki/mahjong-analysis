"""Observer-visible conventional danger features for one discard candidate."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from mahjong_analysis.tiles import (
    TILE_KINDS,
    normalize_tile,
    tile_to_index,
)

SafetyClass = Literal[
    "genbutsu",
    "full_suji",
    "partial_suji",
    "unsuji_numbered",
    "honor_non_genbutsu",
]


@dataclass(frozen=True)
class ConventionalDangerFeatures:
    """Conventional target-riichi features available before the discard."""

    candidate_tile: str
    safety_class: SafetyClass
    is_genbutsu: bool
    is_target_own_discard: bool
    is_post_riichi_passed: bool
    candidate_is_honor: bool
    candidate_is_terminal: bool
    candidate_is_simple: bool
    candidate_rank: int | None
    visible_count: int
    unseen_count: int
    suji_reference_count: int
    suji_safe_count: int
    ryanmen_direction_count: int
    no_chance_direction_count: int
    one_chance_direction_count: int
    all_ryanmen_directions_blocked: bool
    target_first_discard_number: int | None
    matches_riichi_declaration: bool
    same_suit_as_riichi_declaration: bool
    rank_distance_from_riichi_declaration: int | None


def calculate_conventional_danger_features(
    candidate_tile: str,
    visible_counts: Sequence[int],
    *,
    target_own_discards: Sequence[str],
    post_riichi_passed_tiles: Iterable[str],
    riichi_declaration_tile: str,
) -> ConventionalDangerFeatures:
    """Calculate target-specific safety, suji, and wall features.

    Wall features use the immediate inner tile for each possible ryanmen
    direction. Zero unseen copies is no-chance and one unseen copy is
    one-chance for that direction. Penchan directions are intentionally not
    included.
    """
    candidate = normalize_tile(candidate_tile)
    visible = _validate_visible_counts(visible_counts)
    own_discards = tuple(normalize_tile(tile) for tile in target_own_discards)
    passed_tiles = frozenset(normalize_tile(tile) for tile in post_riichi_passed_tiles)
    declaration = normalize_tile(riichi_declaration_tile)
    own_discard_set = frozenset(own_discards)
    safe_tiles = own_discard_set | passed_tiles
    suji_references = _suji_reference_tiles(candidate)
    suji_safe_count = sum(tile in safe_tiles for tile in suji_references)
    safety_class = _safety_class(
        candidate,
        safe_tiles,
        len(suji_references),
        suji_safe_count,
    )

    candidate_index = tile_to_index(candidate)
    is_honor = candidate_index >= 27
    rank = None if is_honor else candidate_index % 9 + 1
    wall_anchors = _ryanmen_wall_anchor_tiles(candidate)
    wall_unseen = tuple(4 - visible[tile_to_index(tile)] for tile in wall_anchors)
    no_chance_count = sum(count == 0 for count in wall_unseen)
    one_chance_count = sum(count == 1 for count in wall_unseen)
    first_discard = next(
        (
            index
            for index, tile in enumerate(own_discards, start=1)
            if tile == candidate
        ),
        None,
    )
    same_suit, rank_distance = _declaration_relation(candidate, declaration)
    return ConventionalDangerFeatures(
        candidate_tile=candidate,
        safety_class=safety_class,
        is_genbutsu=candidate in safe_tiles,
        is_target_own_discard=candidate in own_discard_set,
        is_post_riichi_passed=candidate in passed_tiles,
        candidate_is_honor=is_honor,
        candidate_is_terminal=rank in {1, 9},
        candidate_is_simple=rank is not None and 2 <= rank <= 8,
        candidate_rank=rank,
        visible_count=visible[candidate_index],
        unseen_count=4 - visible[candidate_index],
        suji_reference_count=len(suji_references),
        suji_safe_count=suji_safe_count,
        ryanmen_direction_count=len(wall_anchors),
        no_chance_direction_count=no_chance_count,
        one_chance_direction_count=one_chance_count,
        all_ryanmen_directions_blocked=(
            bool(wall_anchors) and no_chance_count == len(wall_anchors)
        ),
        target_first_discard_number=first_discard,
        matches_riichi_declaration=candidate == declaration,
        same_suit_as_riichi_declaration=same_suit,
        rank_distance_from_riichi_declaration=rank_distance,
    )


def suji_reference_tiles(candidate_tile: str) -> tuple[str, ...]:
    """Return already-safe tile kinds that establish candidate suji."""
    return _suji_reference_tiles(normalize_tile(candidate_tile))


def ryanmen_wall_anchor_tiles(candidate_tile: str) -> tuple[str, ...]:
    """Return immediate inner tiles for the candidate's ryanmen directions."""
    return _ryanmen_wall_anchor_tiles(normalize_tile(candidate_tile))


def _validate_visible_counts(visible_counts: Sequence[int]) -> tuple[int, ...]:
    if isinstance(visible_counts, (str, bytes)) or len(visible_counts) != len(
        TILE_KINDS
    ):
        raise ValueError("visible_counts must contain exactly 34 values")
    result: list[int] = []
    for tile, count in zip(TILE_KINDS, visible_counts, strict=True):
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"visible count for {tile} must be an integer")
        if not 0 <= count <= 4:
            raise ValueError(f"visible count for {tile} must be between 0 and 4")
        result.append(count)
    return tuple(result)


def _suji_reference_tiles(candidate: str) -> tuple[str, ...]:
    index = tile_to_index(candidate)
    if index >= 27:
        return ()
    suit = candidate[-1]
    rank = index % 9 + 1
    if rank <= 3:
        reference_ranks = (rank + 3,)
    elif rank >= 7:
        reference_ranks = (rank - 3,)
    else:
        reference_ranks = (rank - 3, rank + 3)
    return tuple(f"{reference_rank}{suit}" for reference_rank in reference_ranks)


def _ryanmen_wall_anchor_tiles(candidate: str) -> tuple[str, ...]:
    index = tile_to_index(candidate)
    if index >= 27:
        return ()
    suit = candidate[-1]
    rank = index % 9 + 1
    anchors: list[str] = []
    if rank <= 6:
        anchors.append(f"{rank + 1}{suit}")
    if rank >= 4:
        anchors.append(f"{rank - 1}{suit}")
    return tuple(anchors)


def _safety_class(
    candidate: str,
    safe_tiles: frozenset[str],
    suji_reference_count: int,
    suji_safe_count: int,
) -> SafetyClass:
    if candidate in safe_tiles:
        return "genbutsu"
    if tile_to_index(candidate) >= 27:
        return "honor_non_genbutsu"
    if suji_safe_count == suji_reference_count:
        return "full_suji"
    if suji_safe_count:
        return "partial_suji"
    return "unsuji_numbered"


def _declaration_relation(
    candidate: str,
    declaration: str,
) -> tuple[bool, int | None]:
    candidate_index = tile_to_index(candidate)
    declaration_index = tile_to_index(declaration)
    if candidate_index >= 27 or declaration_index >= 27:
        return (False, None)
    candidate_suit = candidate_index // 9
    declaration_suit = declaration_index // 9
    if candidate_suit != declaration_suit:
        return (False, None)
    return (
        True,
        abs(candidate_index % 9 - declaration_index % 9),
    )
