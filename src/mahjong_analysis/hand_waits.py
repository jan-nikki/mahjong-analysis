"""MJAI-independent mahjong hand wait detection."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Literal

from mahjong_analysis.tiles import (
    TILE_KINDS,
    index_to_tile,
    normalize_tile,
    tile_to_index,
    tiles_to_counts,
)

HandType = Literal["standard", "chiitoitsu", "kokushi"]
WaitShape = Literal[
    "ryanmen",
    "kanchan",
    "penchan",
    "shanpon",
    "tanki",
    "kokushi_single",
    "kokushi_13men",
]

HAND_TYPE_ORDER: tuple[HandType, ...] = (
    "standard",
    "chiitoitsu",
    "kokushi",
)
WAIT_SHAPE_ORDER: tuple[WaitShape, ...] = (
    "ryanmen",
    "kanchan",
    "penchan",
    "shanpon",
    "tanki",
    "kokushi_single",
    "kokushi_13men",
)

_HAND_TYPE_INDEX = {hand_type: index for index, hand_type in enumerate(HAND_TYPE_ORDER)}
_WAIT_SHAPE_INDEX = {
    wait_shape: index for index, wait_shape in enumerate(WAIT_SHAPE_ORDER)
}
_VALID_WAIT_SHAPES: dict[HandType, frozenset[WaitShape]] = {
    "standard": frozenset(("ryanmen", "kanchan", "penchan", "shanpon", "tanki")),
    "chiitoitsu": frozenset(("tanki",)),
    "kokushi": frozenset(("kokushi_single", "kokushi_13men")),
}
_KOKUSHI_INDICES = frozenset((0, 8, 9, 17, 18, 26, *range(27, 34)))


@dataclass(frozen=True)
class FixedMeld:
    """A fixed meld owned by the player.

    Established riichi can only have closed quads, so this rule-layer API
    intentionally accepts only ``ankan`` melds for now.
    """

    tiles: tuple[str, ...]
    meld_type: Literal["ankan"] = "ankan"

    def __post_init__(self) -> None:
        object.__setattr__(self, "tiles", tuple(self.tiles))
        if self.meld_type != "ankan":
            raise ValueError(f"unsupported fixed meld type: {self.meld_type!r}")
        if len(self.tiles) != 4:
            raise ValueError("ankan must contain exactly four physical tiles")
        normalized = {tile_to_index(tile) for tile in self.tiles}
        if len(normalized) != 1:
            raise ValueError("ankan tiles must all have the same tile kind")


@dataclass(frozen=True)
class WaitDetail:
    """One distinct wait-tile, hand-type, and wait-shape relation."""

    wait_tile: str
    hand_type: HandType
    wait_shape: WaitShape

    def __post_init__(self) -> None:
        if not isinstance(self.wait_tile, str):
            raise TypeError("wait_tile must be a string")
        normalized = normalize_tile(self.wait_tile)
        if normalized != self.wait_tile:
            raise ValueError(
                f"wait_tile must be normalized: {self.wait_tile!r} -> {normalized!r}"
            )
        if not isinstance(self.hand_type, str):
            raise TypeError("hand_type must be a string")
        if self.hand_type not in _HAND_TYPE_INDEX:
            raise ValueError(f"invalid hand_type: {self.hand_type!r}")
        if not isinstance(self.wait_shape, str):
            raise TypeError("wait_shape must be a string")
        if self.wait_shape not in _WAIT_SHAPE_INDEX:
            raise ValueError(f"invalid wait_shape: {self.wait_shape!r}")
        if self.wait_shape not in _VALID_WAIT_SHAPES[self.hand_type]:
            raise ValueError(
                "invalid hand_type/wait_shape combination: "
                f"{self.hand_type}/{self.wait_shape}"
            )


@dataclass(frozen=True)
class HandWaits:
    """Canonical wait facts and values derived exclusively from them."""

    wait_tiles: tuple[str, ...]
    wait_details: tuple[WaitDetail, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.wait_tiles, tuple):
            raise TypeError("wait_tiles must be a tuple")
        if not isinstance(self.wait_details, tuple):
            raise TypeError("wait_details must be a tuple")

        for wait_tile in self.wait_tiles:
            if not isinstance(wait_tile, str):
                raise TypeError("wait_tiles must contain strings")
            normalized = normalize_tile(wait_tile)
            if normalized != wait_tile:
                raise ValueError(
                    f"wait_tiles must be normalized: {wait_tile!r} -> {normalized!r}"
                )
        if len(set(self.wait_tiles)) != len(self.wait_tiles):
            raise ValueError("wait_tiles must not contain duplicates")
        if self.wait_tiles != tuple(sorted(self.wait_tiles, key=tile_to_index)):
            raise ValueError("wait_tiles must be in canonical 34-tile order")

        if not all(isinstance(detail, WaitDetail) for detail in self.wait_details):
            raise TypeError("wait_details must contain WaitDetail instances")
        if len(set(self.wait_details)) != len(self.wait_details):
            raise ValueError("wait_details must not contain duplicates")
        if self.wait_details != tuple(sorted(self.wait_details, key=_detail_sort_key)):
            raise ValueError("wait_details must be in canonical order")

        detail_tiles = {detail.wait_tile for detail in self.wait_details}
        if set(self.wait_tiles) != detail_tiles:
            raise ValueError(
                "wait_tiles must match the tile kinds present in wait_details"
            )

    @property
    def wait_tile_count(self) -> int:
        """Return the number of distinct normalized wait tiles."""
        return len(self.wait_tiles)

    @property
    def wait_shapes(self) -> tuple[WaitShape, ...]:
        """Return distinct wait shapes in their specification order."""
        present = {detail.wait_shape for detail in self.wait_details}
        return tuple(shape for shape in WAIT_SHAPE_ORDER if shape in present)

    @property
    def contains_ryanmen(self) -> bool:
        """Return whether at least one wait interpretation is ryanmen."""
        return any(detail.wait_shape == "ryanmen" for detail in self.wait_details)

    @property
    def is_pure_ryanmen(self) -> bool:
        """Return whether this is exactly a two-tile, all-standard ryanmen."""
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
        """Return whether at least three distinct normalized tiles win."""
        return self.wait_tile_count >= 3


@dataclass(frozen=True)
class _Component:
    kind: Literal["pair", "triplet", "sequence"]
    tile_index: int


def calculate_hand_waits(
    concealed_tiles: Iterable[str],
    fixed_melds: Iterable[FixedMeld] = (),
) -> HandWaits:
    """Calculate all structural waits for a declaration-post hand.

    The concealed physical tile count must be ``13 - 3 * fixed_meld_count``.
    Rivers, furiten, visible tiles, and live-wall availability are deliberately
    outside this function's scope.
    """
    concealed = tuple(concealed_tiles)
    fixed = tuple(fixed_melds)
    if not all(isinstance(meld, FixedMeld) for meld in fixed):
        raise TypeError("fixed_melds must contain FixedMeld instances")
    if len(fixed) > 4:
        raise ValueError("a hand cannot contain more than four fixed melds")

    expected_concealed_count = 13 - 3 * len(fixed)
    if len(concealed) != expected_concealed_count:
        raise ValueError(
            "concealed hand has invalid physical tile count: "
            f"expected {expected_concealed_count}, got {len(concealed)}"
        )

    concealed_counts = tiles_to_counts(concealed)
    # Validate actual ownership, separately from Tenhou's structural fifth wait.
    tiles_to_counts(
        (
            *concealed,
            *(tile for meld in fixed for tile in meld.tiles),
        )
    )

    details: set[WaitDetail] = set()
    melds_needed = 4 - len(fixed)
    for wait_index in range(len(TILE_KINDS)):
        # Only the pure concealed hand excludes a fifth-copy wait. Fixed melds
        # (including ankan) do not contribute to this Tenhou tenpai restriction.
        if concealed_counts[wait_index] >= 4:
            continue

        completed_counts = list(concealed_counts)
        completed_counts[wait_index] += 1
        completed = tuple(completed_counts)
        wait_tile = index_to_tile(wait_index)

        for decomposition in _standard_decompositions(completed, melds_needed):
            for component in decomposition:
                wait_shape = _shape_for_assignment(component, wait_index)
                if wait_shape is not None:
                    details.add(WaitDetail(wait_tile, "standard", wait_shape))

        if not fixed and _is_chiitoitsu(completed):
            details.add(WaitDetail(wait_tile, "chiitoitsu", "tanki"))

        if not fixed and _is_kokushi(completed):
            shape: WaitShape
            if all(concealed_counts[index] == 1 for index in _KOKUSHI_INDICES):
                shape = "kokushi_13men"
            else:
                shape = "kokushi_single"
            details.add(WaitDetail(wait_tile, "kokushi", shape))

    ordered_details = tuple(sorted(details, key=_detail_sort_key))
    wait_indices = {tile_to_index(detail.wait_tile) for detail in details}
    wait_tiles = tuple(index_to_tile(index) for index in sorted(wait_indices))
    return HandWaits(wait_tiles=wait_tiles, wait_details=ordered_details)


def _standard_decompositions(
    counts: tuple[int, ...],
    melds_needed: int,
) -> Iterator[tuple[_Component, ...]]:
    for pair_index, count in enumerate(counts):
        if count < 2:
            continue
        remainder = list(counts)
        remainder[pair_index] -= 2
        pair = _Component("pair", pair_index)
        for melds in _meld_decompositions(tuple(remainder), melds_needed):
            yield (pair, *melds)


def _meld_decompositions(
    counts: tuple[int, ...],
    melds_needed: int,
) -> Iterator[tuple[_Component, ...]]:
    try:
        first = next(index for index, count in enumerate(counts) if count)
    except StopIteration:
        if melds_needed == 0:
            yield ()
        return

    if melds_needed == 0:
        return

    if counts[first] >= 3:
        remainder = list(counts)
        remainder[first] -= 3
        component = _Component("triplet", first)
        for rest in _meld_decompositions(tuple(remainder), melds_needed - 1):
            yield (component, *rest)

    rank = first % 9
    if first < 27 and rank <= 6 and counts[first + 1] and counts[first + 2]:
        remainder = list(counts)
        remainder[first] -= 1
        remainder[first + 1] -= 1
        remainder[first + 2] -= 1
        component = _Component("sequence", first)
        for rest in _meld_decompositions(tuple(remainder), melds_needed - 1):
            yield (component, *rest)


def _shape_for_assignment(
    component: _Component,
    wait_index: int,
) -> WaitShape | None:
    if component.kind == "pair":
        return "tanki" if component.tile_index == wait_index else None
    if component.kind == "triplet":
        return "shanpon" if component.tile_index == wait_index else None

    start = component.tile_index
    if wait_index < start or wait_index > start + 2:
        return None
    offset = wait_index - start
    if offset == 1:
        return "kanchan"
    rank = start % 9
    if (rank == 0 and offset == 2) or (rank == 6 and offset == 0):
        return "penchan"
    return "ryanmen"


def _is_chiitoitsu(counts: tuple[int, ...]) -> bool:
    return sum(count == 2 for count in counts) == 7 and all(
        count in (0, 2) for count in counts
    )


def _is_kokushi(counts: tuple[int, ...]) -> bool:
    if any(
        count and index not in _KOKUSHI_INDICES for index, count in enumerate(counts)
    ):
        return False
    return (
        all(counts[index] >= 1 for index in _KOKUSHI_INDICES)
        and sum(counts[index] == 2 for index in _KOKUSHI_INDICES) == 1
    )


def _detail_sort_key(detail: WaitDetail) -> tuple[int, int, int]:
    return (
        tile_to_index(detail.wait_tile),
        _HAND_TYPE_INDEX[detail.hand_type],
        _WAIT_SHAPE_INDEX[detail.wait_shape],
    )
