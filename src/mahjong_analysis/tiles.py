"""Tile-kind normalization and the canonical 34-tile representation."""

from collections.abc import Iterable

TILE_KINDS: tuple[str, ...] = (
    *(f"{rank}m" for rank in range(1, 10)),
    *(f"{rank}p" for rank in range(1, 10)),
    *(f"{rank}s" for rank in range(1, 10)),
    "E",
    "S",
    "W",
    "N",
    "P",
    "F",
    "C",
)

RED_FIVE_NORMALIZATION: dict[str, str] = {
    "5mr": "5m",
    "5pr": "5p",
    "5sr": "5s",
}

_TILE_TO_INDEX = {tile: index for index, tile in enumerate(TILE_KINDS)}


def normalize_tile(tile: str) -> str:
    """Return the canonical tile kind, normalizing a red five."""
    normalized = RED_FIVE_NORMALIZATION.get(tile, tile)
    if normalized not in _TILE_TO_INDEX:
        raise ValueError(f"invalid tile: {tile!r}")
    return normalized


def tile_to_index(tile: str) -> int:
    """Return the stable zero-based index of a raw or canonical tile."""
    return _TILE_TO_INDEX[normalize_tile(tile)]


def index_to_tile(index: int) -> str:
    """Return the canonical tile kind for a stable zero-based index."""
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError(f"tile index must be an integer: {index!r}")
    if index < 0 or index >= len(TILE_KINDS):
        raise ValueError(f"tile index out of range: {index}")
    return TILE_KINDS[index]


def tiles_to_counts(tiles: Iterable[str]) -> tuple[int, ...]:
    """Convert tiles to canonical 34-kind counts, rejecting a fifth copy."""
    counts = [0] * len(TILE_KINDS)
    for tile in tiles:
        index = tile_to_index(tile)
        counts[index] += 1
        if counts[index] > 4:
            raise ValueError(f"more than four copies of tile kind: {TILE_KINDS[index]}")
    return tuple(counts)
