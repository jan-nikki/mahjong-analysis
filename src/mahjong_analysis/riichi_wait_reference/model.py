"""Reference-only tile constants and immutable wait models."""

from dataclasses import dataclass
from typing import Literal

ReferenceHandType = Literal["standard", "chiitoitsu", "kokushi"]
ReferenceWaitShape = Literal[
    "ryanmen",
    "kanchan",
    "penchan",
    "shanpon",
    "tanki",
    "kokushi_single",
    "kokushi_13men",
]

REFERENCE_TILE_KINDS: tuple[str, ...] = (
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
REFERENCE_RED_FIVE_NORMALIZATION = {
    "5mr": "5m",
    "5pr": "5p",
    "5sr": "5s",
}
REFERENCE_HAND_TYPE_ORDER: tuple[ReferenceHandType, ...] = (
    "standard",
    "chiitoitsu",
    "kokushi",
)
REFERENCE_WAIT_SHAPE_ORDER: tuple[ReferenceWaitShape, ...] = (
    "ryanmen",
    "kanchan",
    "penchan",
    "shanpon",
    "tanki",
    "kokushi_single",
    "kokushi_13men",
)

_REFERENCE_TILE_INDEX = {tile: index for index, tile in enumerate(REFERENCE_TILE_KINDS)}
_REFERENCE_HAND_TYPE_INDEX = {
    hand_type: index for index, hand_type in enumerate(REFERENCE_HAND_TYPE_ORDER)
}
_REFERENCE_WAIT_SHAPE_INDEX = {
    wait_shape: index for index, wait_shape in enumerate(REFERENCE_WAIT_SHAPE_ORDER)
}
_REFERENCE_VALID_WAIT_SHAPES: dict[ReferenceHandType, frozenset[ReferenceWaitShape]] = {
    "standard": frozenset(("ryanmen", "kanchan", "penchan", "shanpon", "tanki")),
    "chiitoitsu": frozenset(("tanki",)),
    "kokushi": frozenset(("kokushi_single", "kokushi_13men")),
}


def normalize_reference_tile(tile: str) -> str:
    """Return one reference 34-kind tile without using production constants."""
    if not isinstance(tile, str):
        raise TypeError("tile must be a string")
    if tile in REFERENCE_RED_FIVE_NORMALIZATION:
        return REFERENCE_RED_FIVE_NORMALIZATION[tile]
    if tile in _REFERENCE_TILE_INDEX:
        return tile
    raise ValueError(f"invalid tile: {tile!r}")


def reference_tile_to_index(tile: str) -> int:
    """Return the reference 34-kind index for a raw or normalized tile."""
    return _REFERENCE_TILE_INDEX[normalize_reference_tile(tile)]


@dataclass(frozen=True)
class ReferenceMeld:
    """A closed quad represented independently from production melds."""

    tiles: tuple[str, ...]
    meld_type: Literal["ankan"] = "ankan"

    def __post_init__(self) -> None:
        if not isinstance(self.tiles, tuple):
            raise TypeError("tiles must be a tuple")
        if not isinstance(self.meld_type, str):
            raise TypeError("meld_type must be a string")
        if self.meld_type != "ankan":
            raise ValueError(f"unsupported reference meld type: {self.meld_type!r}")
        if len(self.tiles) != 4:
            raise ValueError("reference ankan must contain exactly four tiles")

        tile_indices = tuple(reference_tile_to_index(tile) for tile in self.tiles)
        if len(set(tile_indices)) != 1:
            raise ValueError("reference ankan tiles must have one normalized tile kind")


@dataclass(frozen=True)
class ReferenceWaitDetail:
    """One reference wait-tile, hand-type, and wait-shape relation."""

    wait_tile: str
    hand_type: ReferenceHandType
    wait_shape: ReferenceWaitShape

    def __post_init__(self) -> None:
        if not isinstance(self.wait_tile, str):
            raise TypeError("wait_tile must be a string")
        normalized = normalize_reference_tile(self.wait_tile)
        if normalized != self.wait_tile:
            raise ValueError("reference wait_tile must use a normalized tile kind")
        if not isinstance(self.hand_type, str):
            raise TypeError("hand_type must be a string")
        if self.hand_type not in _REFERENCE_HAND_TYPE_INDEX:
            raise ValueError(f"invalid reference hand_type: {self.hand_type!r}")
        if not isinstance(self.wait_shape, str):
            raise TypeError("wait_shape must be a string")
        if self.wait_shape not in _REFERENCE_WAIT_SHAPE_INDEX:
            raise ValueError(f"invalid reference wait_shape: {self.wait_shape!r}")
        if self.wait_shape not in _REFERENCE_VALID_WAIT_SHAPES[self.hand_type]:
            raise ValueError(
                "invalid reference hand_type/wait_shape combination: "
                f"{self.hand_type}/{self.wait_shape}"
            )


@dataclass(frozen=True)
class ReferenceHandWaits:
    """Canonical reference waits with independently derived values."""

    wait_tiles: tuple[str, ...]
    wait_details: tuple[ReferenceWaitDetail, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.wait_tiles, tuple):
            raise TypeError("wait_tiles must be a tuple")
        if not isinstance(self.wait_details, tuple):
            raise TypeError("wait_details must be a tuple")

        for wait_tile in self.wait_tiles:
            if not isinstance(wait_tile, str):
                raise TypeError("wait_tiles must contain strings")
            if normalize_reference_tile(wait_tile) != wait_tile:
                raise ValueError("reference wait_tiles must be normalized")
        if len(set(self.wait_tiles)) != len(self.wait_tiles):
            raise ValueError("reference wait_tiles must not contain duplicates")
        expected_tiles = tuple(sorted(self.wait_tiles, key=reference_tile_to_index))
        if self.wait_tiles != expected_tiles:
            raise ValueError("reference wait_tiles must use 34-kind order")

        if not all(
            isinstance(detail, ReferenceWaitDetail) for detail in self.wait_details
        ):
            raise TypeError("wait_details must contain ReferenceWaitDetail instances")
        if len(set(self.wait_details)) != len(self.wait_details):
            raise ValueError("reference wait_details must not contain duplicates")
        expected_details = tuple(
            sorted(self.wait_details, key=reference_wait_detail_sort_key)
        )
        if self.wait_details != expected_details:
            raise ValueError("reference wait_details must use specification order")

        detail_tiles = {detail.wait_tile for detail in self.wait_details}
        if set(self.wait_tiles) != detail_tiles:
            raise ValueError("reference wait_tiles must match tiles in wait_details")

    @property
    def wait_tile_count(self) -> int:
        """Return the number of distinct normalized wait tile kinds."""
        return len(self.wait_tiles)

    @property
    def wait_shapes(self) -> tuple[ReferenceWaitShape, ...]:
        """Return distinct reference shapes in specification order."""
        present = {detail.wait_shape for detail in self.wait_details}
        return tuple(shape for shape in REFERENCE_WAIT_SHAPE_ORDER if shape in present)

    @property
    def contains_ryanmen(self) -> bool:
        """Return whether any reference interpretation is ryanmen."""
        return any(detail.wait_shape == "ryanmen" for detail in self.wait_details)

    @property
    def is_pure_ryanmen(self) -> bool:
        """Return whether this is exactly two all-standard ryanmen waits."""
        return (
            bool(self.wait_details)
            and self.wait_tile_count == 2
            and all(
                detail.hand_type == "standard" and detail.wait_shape == "ryanmen"
                for detail in self.wait_details
            )
        )

    @property
    def is_multiwait(self) -> bool:
        """Return whether three or more normalized tile kinds complete the hand."""
        return self.wait_tile_count >= 3


def reference_wait_detail_sort_key(
    detail: ReferenceWaitDetail,
) -> tuple[int, int, int]:
    """Return the independent specification order for one detail."""
    return (
        reference_tile_to_index(detail.wait_tile),
        _REFERENCE_HAND_TYPE_INDEX[detail.hand_type],
        _REFERENCE_WAIT_SHAPE_INDEX[detail.wait_shape],
    )
