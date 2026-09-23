from __future__ import annotations

import pytest

from mahjong_analysis.danger_features import (
    calculate_conventional_danger_features,
    ryanmen_wall_anchor_tiles,
    suji_reference_tiles,
)
from mahjong_analysis.tiles import TILE_KINDS, tile_to_index


def visible_counts(**counts: int) -> tuple[int, ...]:
    result = [0] * len(TILE_KINDS)
    for tile, count in counts.items():
        result[tile_to_index(tile)] = count
    return tuple(result)


@pytest.mark.parametrize(
    ("tile", "expected"),
    [
        ("1m", ("4m",)),
        ("3m", ("6m",)),
        ("4m", ("1m", "7m")),
        ("6m", ("3m", "9m")),
        ("9m", ("6m",)),
        ("E", ()),
    ],
)
def test_suji_reference_tiles(tile: str, expected: tuple[str, ...]) -> None:
    assert suji_reference_tiles(tile) == expected


@pytest.mark.parametrize(
    ("tile", "expected"),
    [
        ("1p", ("2p",)),
        ("3p", ("4p",)),
        ("4p", ("5p", "3p")),
        ("6p", ("7p", "5p")),
        ("7p", ("6p",)),
        ("9p", ("8p",)),
        ("C", ()),
    ],
)
def test_ryanmen_wall_anchor_tiles(tile: str, expected: tuple[str, ...]) -> None:
    assert ryanmen_wall_anchor_tiles(tile) == expected


@pytest.mark.parametrize(
    ("safe", "expected"),
    [
        ({"4m"}, "genbutsu"),
        ({"1m", "7m"}, "full_suji"),
        ({"1m"}, "partial_suji"),
        (set(), "unsuji_numbered"),
    ],
)
def test_safety_class_uses_predecision_safe_tiles(
    safe: set[str],
    expected: str,
) -> None:
    result = calculate_conventional_danger_features(
        "4m",
        visible_counts(),
        target_own_discards=tuple(safe),
        post_riichi_passed_tiles=(),
        riichi_declaration_tile="9p",
    )

    assert result.safety_class == expected


def test_post_riichi_passed_tile_is_genbutsu_but_not_target_own_discard() -> None:
    result = calculate_conventional_danger_features(
        "5mr",
        visible_counts(**{"5m": 2}),
        target_own_discards=("2p",),
        post_riichi_passed_tiles=("5m",),
        riichi_declaration_tile="2p",
    )

    assert result.candidate_tile == "5m"
    assert result.is_genbutsu is True
    assert result.is_target_own_discard is False
    assert result.is_post_riichi_passed is True
    assert result.visible_count == 2
    assert result.unseen_count == 2


def test_wall_features_count_directions_without_treating_penchan_as_ryanmen() -> None:
    result = calculate_conventional_danger_features(
        "4s",
        visible_counts(**{"3s": 4, "5s": 3}),
        target_own_discards=(),
        post_riichi_passed_tiles=(),
        riichi_declaration_tile="6s",
    )

    assert result.ryanmen_direction_count == 2
    assert result.no_chance_direction_count == 1
    assert result.one_chance_direction_count == 1
    assert result.all_ryanmen_directions_blocked is False

    edge = calculate_conventional_danger_features(
        "3s",
        visible_counts(**{"2s": 4, "4s": 4}),
        target_own_discards=(),
        post_riichi_passed_tiles=(),
        riichi_declaration_tile="6s",
    )
    assert edge.ryanmen_direction_count == 1
    assert edge.no_chance_direction_count == 1
    assert edge.all_ryanmen_directions_blocked is True


def test_target_river_and_declaration_relationships_are_retained() -> None:
    result = calculate_conventional_danger_features(
        "5p",
        visible_counts(),
        target_own_discards=("1m", "5p", "5p"),
        post_riichi_passed_tiles=(),
        riichi_declaration_tile="5p",
    )

    assert result.target_first_discard_number == 2
    assert result.matches_riichi_declaration is True
    assert result.same_suit_as_riichi_declaration is True
    assert result.rank_distance_from_riichi_declaration == 0


def test_honor_has_no_suji_or_wall_direction() -> None:
    result = calculate_conventional_danger_features(
        "E",
        visible_counts(E=3),
        target_own_discards=(),
        post_riichi_passed_tiles=(),
        riichi_declaration_tile="9m",
    )

    assert result.safety_class == "honor_non_genbutsu"
    assert result.candidate_rank is None
    assert result.suji_reference_count == 0
    assert result.ryanmen_direction_count == 0
    assert result.all_ryanmen_directions_blocked is False
