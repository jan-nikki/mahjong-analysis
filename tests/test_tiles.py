import pytest

from mahjong_analysis.tiles import (
    TILE_KINDS,
    index_to_tile,
    normalize_tile,
    tile_to_index,
    tiles_to_counts,
)


def test_tile_kinds_have_canonical_34_kind_order() -> None:
    assert len(TILE_KINDS) == 34
    assert TILE_KINDS == (
        "1m",
        "2m",
        "3m",
        "4m",
        "5m",
        "6m",
        "7m",
        "8m",
        "9m",
        "1p",
        "2p",
        "3p",
        "4p",
        "5p",
        "6p",
        "7p",
        "8p",
        "9p",
        "1s",
        "2s",
        "3s",
        "4s",
        "5s",
        "6s",
        "7s",
        "8s",
        "9s",
        "E",
        "S",
        "W",
        "N",
        "P",
        "F",
        "C",
    )


@pytest.mark.parametrize(
    ("raw_tile", "normalized"),
    [("5mr", "5m"), ("5pr", "5p"), ("5sr", "5s")],
)
def test_normalize_tile_normalizes_red_fives(
    raw_tile: str,
    normalized: str,
) -> None:
    assert normalize_tile(raw_tile) == normalized


@pytest.mark.parametrize("tile", ["1m", "9s", "E", "C"])
def test_normalize_tile_preserves_canonical_tiles(tile: str) -> None:
    assert normalize_tile(tile) == tile


@pytest.mark.parametrize("tile", ["?", "0m", "10m", "5mrr", "e", ""])
def test_normalize_tile_rejects_invalid_tiles(tile: str) -> None:
    with pytest.raises(ValueError, match="invalid tile"):
        normalize_tile(tile)


def test_tile_indices_round_trip_in_canonical_order() -> None:
    assert tuple(index_to_tile(index) for index in range(34)) == TILE_KINDS
    assert tuple(tile_to_index(tile) for tile in TILE_KINDS) == tuple(range(34))


def test_red_five_has_same_index_as_normal_five() -> None:
    assert tile_to_index("5mr") == tile_to_index("5m") == 4


@pytest.mark.parametrize("index", [-1, 34])
def test_index_to_tile_rejects_out_of_range_indices(index: int) -> None:
    with pytest.raises(ValueError, match="tile index"):
        index_to_tile(index)


@pytest.mark.parametrize("index", [True, 1.0])
def test_index_to_tile_rejects_non_integer_indices(index: object) -> None:
    with pytest.raises(TypeError, match="tile index"):
        index_to_tile(index)  # type: ignore[arg-type]


def test_tiles_to_counts_combines_red_and_normal_fives() -> None:
    counts = tiles_to_counts(["5s", "5sr", "E"])

    assert counts[tile_to_index("5s")] == 2
    assert counts[tile_to_index("E")] == 1
    assert sum(counts) == 3


def test_tiles_to_counts_rejects_fifth_normalized_copy() -> None:
    with pytest.raises(ValueError, match="more than four"):
        tiles_to_counts(["5m", "5m", "5m", "5m", "5mr"])
