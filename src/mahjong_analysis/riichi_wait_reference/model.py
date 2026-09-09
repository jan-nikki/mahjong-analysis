"""Reference-only tile constants and immutable wait models."""

from collections import Counter
from dataclasses import dataclass
from typing import Literal

ReferenceHandType = Literal["standard", "chiitoitsu", "kokushi"]
ReferenceCallType = Literal["chi", "pon", "daiminkan"]
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


@dataclass(frozen=True)
class ReferenceActorDiscard:
    """One actor discard with call metadata finalized at kyoku end."""

    discard_number: int
    tile: str
    tile_kind: str
    tsumogiri: bool
    event_index: int
    is_riichi_declaration: bool
    was_called: bool = False
    call_type: ReferenceCallType | None = None
    called_by_actor: int | None = None
    call_event_index: int | None = None

    def __post_init__(self) -> None:
        if type(self.discard_number) is not int:
            raise TypeError("discard_number must be an integer")
        if self.discard_number < 1:
            raise ValueError("discard_number must be at least 1")
        if type(self.event_index) is not int:
            raise TypeError("event_index must be an integer")
        if self.event_index < 0:
            raise ValueError("event_index must be non-negative")

        if not isinstance(self.tile, str):
            raise TypeError("tile must be a string")
        normalized = normalize_reference_tile(self.tile)
        if not isinstance(self.tile_kind, str):
            raise TypeError("tile_kind must be a string")
        if normalize_reference_tile(self.tile_kind) != self.tile_kind:
            raise ValueError("tile_kind must be a normalized tile kind")
        if normalized != self.tile_kind:
            raise ValueError("tile_kind must equal the normalized raw tile")

        for field_name, value in (
            ("tsumogiri", self.tsumogiri),
            ("is_riichi_declaration", self.is_riichi_declaration),
            ("was_called", self.was_called),
        ):
            if type(value) is not bool:
                raise TypeError(f"{field_name} must be a bool")

        if not self.was_called:
            if any(
                value is not None
                for value in (
                    self.call_type,
                    self.called_by_actor,
                    self.call_event_index,
                )
            ):
                raise ValueError("uncalled discard must not contain call metadata")
            return

        if self.call_type not in {"chi", "pon", "daiminkan"}:
            raise ValueError("called discard has an invalid call_type")
        if type(self.called_by_actor) is not int:
            raise TypeError("called_by_actor must be an integer")
        if self.called_by_actor not in range(4):
            raise ValueError("called_by_actor must be between 0 and 3")
        if type(self.call_event_index) is not int:
            raise TypeError("call_event_index must be an integer")
        if self.call_event_index <= self.event_index:
            raise ValueError("call_event_index must follow the discard event")


@dataclass(frozen=True)
class ReferenceRiichiCandidate:
    """One independently replayed established-riichi candidate."""

    source_path: str
    start_kyoku_line: int
    actor: int
    reach_event_index: int
    declaration_dahai_event_index: int
    reach_accepted_event_index: int
    reach_line: int
    declaration_dahai_line: int
    reach_accepted_line: int
    riichi_discard_number: int
    riichi_declaration_tile: str
    riichi_declaration_tile_kind: str
    concealed_tiles_after_discard: tuple[str, ...]
    fixed_melds: tuple[ReferenceMeld, ...]
    actor_discards_before_riichi: tuple[ReferenceActorDiscard, ...]
    waits: ReferenceHandWaits

    def __post_init__(self) -> None:
        if not isinstance(self.source_path, str) or not self.source_path:
            raise ValueError("source_path must be a non-empty string")
        if "\\" in self.source_path:
            raise ValueError("source_path must use '/' separators")
        if type(self.actor) is not int or self.actor not in range(4):
            raise ValueError("actor must be an integer from 0 to 3")

        indexes = (
            self.reach_event_index,
            self.declaration_dahai_event_index,
            self.reach_accepted_event_index,
        )
        if any(type(index) is not int for index in indexes):
            raise TypeError("event indexes must be integers")
        if not 0 <= indexes[0] < indexes[1] < indexes[2]:
            raise ValueError("event indexes must satisfy reach < dahai < accepted")

        lines = (
            self.start_kyoku_line,
            self.reach_line,
            self.declaration_dahai_line,
            self.reach_accepted_line,
        )
        if any(type(line) is not int for line in lines):
            raise TypeError("source line numbers must be integers")
        if not 1 <= lines[0] < lines[1] < lines[2] < lines[3]:
            raise ValueError("source lines must follow the established reach sequence")

        if type(self.riichi_discard_number) is not int:
            raise TypeError("riichi_discard_number must be an integer")
        if self.riichi_discard_number < 1:
            raise ValueError("riichi_discard_number must be at least 1")
        if not isinstance(self.riichi_declaration_tile, str):
            raise TypeError("riichi_declaration_tile must be a string")
        declaration_kind = normalize_reference_tile(self.riichi_declaration_tile)
        if declaration_kind != self.riichi_declaration_tile_kind:
            raise ValueError("riichi declaration tile and tile kind do not match")

        if not isinstance(self.concealed_tiles_after_discard, tuple):
            raise TypeError("concealed_tiles_after_discard must be a tuple")
        for tile in self.concealed_tiles_after_discard:
            normalize_reference_tile(tile)
        if not isinstance(self.fixed_melds, tuple) or not all(
            isinstance(meld, ReferenceMeld) for meld in self.fixed_melds
        ):
            raise TypeError("fixed_melds must be a tuple of ReferenceMeld")
        expected_count = 13 - 3 * len(self.fixed_melds)
        if len(self.concealed_tiles_after_discard) != expected_count:
            raise ValueError("declaration-post concealed tile count is invalid")
        _validate_candidate_owned_tiles(
            self.concealed_tiles_after_discard,
            self.fixed_melds,
        )

        if not isinstance(self.actor_discards_before_riichi, tuple) or not all(
            isinstance(discard, ReferenceActorDiscard)
            for discard in self.actor_discards_before_riichi
        ):
            raise TypeError(
                "actor_discards_before_riichi must be a tuple of ReferenceActorDiscard"
            )
        if len(self.actor_discards_before_riichi) != self.riichi_discard_number:
            raise ValueError("discard history length must equal riichi_discard_number")
        expected_numbers = tuple(range(1, self.riichi_discard_number + 1))
        actual_numbers = tuple(
            discard.discard_number for discard in self.actor_discards_before_riichi
        )
        if actual_numbers != expected_numbers:
            raise ValueError("discard numbers must be consecutive from 1")
        if any(
            earlier.event_index >= later.event_index
            for earlier, later in zip(
                self.actor_discards_before_riichi,
                self.actor_discards_before_riichi[1:],
                strict=False,
            )
        ):
            raise ValueError("discard event indexes must be strictly increasing")
        if any(
            discard.is_riichi_declaration
            for discard in self.actor_discards_before_riichi[:-1]
        ):
            raise ValueError("only the last retained discard may declare riichi")
        declaration = self.actor_discards_before_riichi[-1]
        if not declaration.is_riichi_declaration:
            raise ValueError("last retained discard must declare riichi")
        if declaration.event_index != self.declaration_dahai_event_index:
            raise ValueError("declaration discard event index does not match")
        if declaration.tile != self.riichi_declaration_tile:
            raise ValueError("declaration discard raw tile does not match")
        if declaration.tile_kind != self.riichi_declaration_tile_kind:
            raise ValueError("declaration discard tile kind does not match")
        if not isinstance(self.waits, ReferenceHandWaits):
            raise TypeError("waits must be ReferenceHandWaits")
        if not self.waits.wait_tiles:
            raise ValueError("established riichi candidate must have a non-empty wait")

        # Keep the public constructor honest without depending on production code.
        # The local import avoids the model -> waits -> model import cycle.
        from .waits import calculate_reference_hand_waits

        calculated_waits = calculate_reference_hand_waits(
            self.concealed_tiles_after_discard,
            self.fixed_melds,
        )
        if _wait_signature(self.waits) != _wait_signature(calculated_waits):
            raise ValueError(
                "waits must match the independently calculated declaration-post hand"
            )

    @property
    def candidate_key(self) -> tuple[str, int, int]:
        """Return the future production/reference comparison key."""
        return self.source_path, self.start_kyoku_line, self.reach_line

    @property
    def wait_tiles(self) -> tuple[str, ...]:
        return self.waits.wait_tiles

    @property
    def wait_details(self) -> tuple[ReferenceWaitDetail, ...]:
        return self.waits.wait_details

    @property
    def wait_tile_count(self) -> int:
        return self.waits.wait_tile_count

    @property
    def wait_shapes(self) -> tuple[ReferenceWaitShape, ...]:
        return self.waits.wait_shapes

    @property
    def contains_ryanmen(self) -> bool:
        return self.waits.contains_ryanmen

    @property
    def is_pure_ryanmen(self) -> bool:
        return self.waits.is_pure_ryanmen

    @property
    def is_multiwait(self) -> bool:
        return self.waits.is_multiwait


def reference_wait_detail_sort_key(
    detail: ReferenceWaitDetail,
) -> tuple[int, int, int]:
    """Return the independent specification order for one detail."""
    return (
        reference_tile_to_index(detail.wait_tile),
        _REFERENCE_HAND_TYPE_INDEX[detail.hand_type],
        _REFERENCE_WAIT_SHAPE_INDEX[detail.wait_shape],
    )


def _validate_candidate_owned_tiles(
    concealed_tiles: tuple[str, ...],
    fixed_melds: tuple[ReferenceMeld, ...],
) -> None:
    normalized_counts: Counter[str] = Counter()
    raw_red_counts: Counter[str] = Counter()
    owned_tiles = (
        *concealed_tiles,
        *(tile for meld in fixed_melds for tile in meld.tiles),
    )
    for tile in owned_tiles:
        normalized_counts[normalize_reference_tile(tile)] += 1
        if tile in REFERENCE_RED_FIVE_NORMALIZATION:
            raw_red_counts[tile] += 1

    if any(count > 4 for count in normalized_counts.values()):
        raise ValueError("candidate cannot own five of one normalized tile kind")
    if any(count > 1 for count in raw_red_counts.values()):
        raise ValueError("candidate cannot own duplicate physical red fives")


def _wait_signature(waits: ReferenceHandWaits) -> tuple[object, ...]:
    return (
        waits.wait_tiles,
        waits.wait_tile_count,
        waits.wait_details,
        waits.wait_shapes,
        waits.contains_ryanmen,
        waits.is_pure_ryanmen,
        waits.is_multiwait,
    )
