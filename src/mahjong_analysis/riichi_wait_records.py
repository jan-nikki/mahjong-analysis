"""Build analysis records from established-riichi hand snapshots."""

from collections.abc import Iterable
from dataclasses import InitVar, dataclass
from itertools import pairwise

from mahjong_analysis.hand_waits import (
    FixedMeld,
    HandWaits,
    WaitDetail,
    WaitShape,
    calculate_hand_waits,
)
from mahjong_analysis.riichi import ActorDiscard, EstablishedRiichi
from mahjong_analysis.tiles import normalize_tile

_FACTORY_VALIDATION_TOKEN = object()


@dataclass(frozen=True)
class _FactoryValidatedWaits:
    token: object
    waits: HandWaits


@dataclass(frozen=True)
class RiichiWaitRecord:
    """One established riichi enriched with its canonical structural waits."""

    actor: int
    riichi_discard_number: int
    riichi_declaration_tile: str
    riichi_declaration_tile_kind: str
    reach_event_index: int
    declaration_dahai_event_index: int
    reach_accepted_event_index: int
    concealed_tiles_after_discard: tuple[str, ...]
    fixed_melds: tuple[FixedMeld, ...]
    actor_discards_before_riichi: tuple[ActorDiscard, ...]
    wait_tiles: tuple[str, ...]
    wait_tile_count: int
    wait_details: tuple[WaitDetail, ...]
    wait_shapes: tuple[WaitShape, ...]
    contains_ryanmen: bool
    is_pure_ryanmen: bool
    is_multiwait: bool
    _factory_validated_waits: InitVar[_FactoryValidatedWaits | None] = None

    def __post_init__(
        self,
        _factory_validated_waits: _FactoryValidatedWaits | None,
    ) -> None:
        _validate_record_identity(self)
        _validate_immutable_collections(self)
        _validate_discard_history(self)

        waits = HandWaits(
            wait_tiles=self.wait_tiles,
            wait_details=self.wait_details,
        )
        if not waits.wait_tiles:
            raise ValueError("a riichi wait record must have at least one wait tile")

        calculated = _resolve_calculated_waits(
            self,
            _factory_validated_waits,
        )
        if waits != calculated:
            raise ValueError(
                "wait_tiles and wait_details must match the declaration-post hand"
            )

        _validate_derived_wait_values(self, waits)


def build_riichi_wait_record(
    established_riichi: EstablishedRiichi,
) -> RiichiWaitRecord:
    """Add canonical wait information to one established-riichi snapshot."""
    if not isinstance(established_riichi, EstablishedRiichi):
        raise TypeError("established_riichi must be an EstablishedRiichi")

    riichi = established_riichi
    waits = calculate_hand_waits(
        riichi.concealed_tiles_after_discard,
        riichi.fixed_melds,
    )
    if not waits.wait_tiles:
        raise ValueError("cannot build a riichi wait record with no wait tiles")

    return RiichiWaitRecord(
        actor=riichi.actor,
        riichi_discard_number=riichi.riichi_discard_number,
        riichi_declaration_tile=riichi.riichi_declaration_tile,
        riichi_declaration_tile_kind=riichi.riichi_declaration_tile_kind,
        reach_event_index=riichi.reach_event_index,
        declaration_dahai_event_index=riichi.declaration_dahai_event_index,
        reach_accepted_event_index=riichi.reach_accepted_event_index,
        concealed_tiles_after_discard=riichi.concealed_tiles_after_discard,
        fixed_melds=riichi.fixed_melds,
        actor_discards_before_riichi=riichi.actor_discards_before_riichi,
        wait_tiles=waits.wait_tiles,
        wait_tile_count=waits.wait_tile_count,
        wait_details=waits.wait_details,
        wait_shapes=waits.wait_shapes,
        contains_ryanmen=waits.contains_ryanmen,
        is_pure_ryanmen=waits.is_pure_ryanmen,
        is_multiwait=waits.is_multiwait,
        _factory_validated_waits=_FactoryValidatedWaits(
            _FACTORY_VALIDATION_TOKEN,
            waits,
        ),
    )


def build_riichi_wait_records(
    established_riichis: Iterable[EstablishedRiichi],
) -> tuple[RiichiWaitRecord, ...]:
    """Build one wait record for each established riichi, preserving order."""
    return tuple(
        build_riichi_wait_record(established_riichi)
        for established_riichi in established_riichis
    )


def _resolve_calculated_waits(
    record: RiichiWaitRecord,
    factory_validated_waits: _FactoryValidatedWaits | None,
) -> HandWaits:
    if factory_validated_waits is None:
        return calculate_hand_waits(
            record.concealed_tiles_after_discard,
            record.fixed_melds,
        )
    if (
        type(factory_validated_waits) is not _FactoryValidatedWaits
        or factory_validated_waits.token is not _FACTORY_VALIDATION_TOKEN
    ):
        raise TypeError("_factory_validated_waits is reserved for the internal factory")
    return factory_validated_waits.waits


def _validate_record_identity(record: RiichiWaitRecord) -> None:
    if type(record.actor) is not int:
        raise TypeError("actor must be an integer")
    if record.actor not in range(4):
        raise ValueError("actor must be between 0 and 3")

    event_indexes = (
        record.reach_event_index,
        record.declaration_dahai_event_index,
        record.reach_accepted_event_index,
    )
    if not all(type(event_index) is int for event_index in event_indexes):
        raise TypeError("event indexes must be integers")
    if not all(event_index >= 0 for event_index in event_indexes):
        raise ValueError("event indexes must be non-negative")
    if not (
        record.reach_event_index
        < record.declaration_dahai_event_index
        < record.reach_accepted_event_index
    ):
        raise ValueError(
            "event indexes must satisfy reach < declaration_dahai < reach_accepted"
        )

    if type(record.riichi_discard_number) is not int:
        raise TypeError("riichi_discard_number must be an integer")
    if record.riichi_discard_number < 1:
        raise ValueError("riichi_discard_number must be at least 1")

    if not isinstance(record.riichi_declaration_tile, str):
        raise TypeError("riichi_declaration_tile must be a string")
    if not isinstance(record.riichi_declaration_tile_kind, str):
        raise TypeError("riichi_declaration_tile_kind must be a string")
    normalized = normalize_tile(record.riichi_declaration_tile)
    if record.riichi_declaration_tile_kind != normalized:
        raise ValueError(
            "riichi_declaration_tile_kind must be the normalized declaration tile"
        )


def _validate_immutable_collections(record: RiichiWaitRecord) -> None:
    collections = (
        (record.concealed_tiles_after_discard, "concealed_tiles_after_discard"),
        (record.fixed_melds, "fixed_melds"),
        (record.actor_discards_before_riichi, "actor_discards_before_riichi"),
        (record.wait_tiles, "wait_tiles"),
        (record.wait_details, "wait_details"),
        (record.wait_shapes, "wait_shapes"),
    )
    for value, field_name in collections:
        if not isinstance(value, tuple):
            raise TypeError(f"{field_name} must be a tuple")

    if not all(
        isinstance(discard, ActorDiscard)
        for discard in record.actor_discards_before_riichi
    ):
        raise TypeError(
            "actor_discards_before_riichi must contain ActorDiscard instances"
        )


def _validate_derived_wait_values(
    record: RiichiWaitRecord,
    waits: HandWaits,
) -> None:
    if type(record.wait_tile_count) is not int:
        raise TypeError("wait_tile_count must be an integer")
    if record.wait_tile_count != waits.wait_tile_count:
        raise ValueError("wait_tile_count must equal len(wait_tiles)")
    if record.wait_shapes != waits.wait_shapes:
        raise ValueError("wait_shapes must be derived from wait_details")

    derived_booleans = (
        ("contains_ryanmen", record.contains_ryanmen, waits.contains_ryanmen),
        ("is_pure_ryanmen", record.is_pure_ryanmen, waits.is_pure_ryanmen),
        ("is_multiwait", record.is_multiwait, waits.is_multiwait),
    )
    for field_name, value, expected in derived_booleans:
        if type(value) is not bool:
            raise TypeError(f"{field_name} must be a bool")
        if value is not expected:
            raise ValueError(f"{field_name} must be derived from canonical waits")


def _validate_discard_history(record: RiichiWaitRecord) -> None:
    discards = record.actor_discards_before_riichi
    if len(discards) != record.riichi_discard_number:
        raise ValueError(
            "actor discard history length must equal riichi_discard_number"
        )

    discard_numbers = tuple(discard.discard_number for discard in discards)
    expected_numbers = tuple(range(1, record.riichi_discard_number + 1))
    if discard_numbers != expected_numbers:
        raise ValueError("actor discard numbers must be consecutive from 1")

    event_indexes = tuple(discard.event_index for discard in discards)
    if any(left >= right for left, right in pairwise(event_indexes)):
        raise ValueError("actor discard event indexes must be strictly increasing")

    if any(discard.is_riichi_declaration for discard in discards[:-1]):
        raise ValueError("only the final actor discard may declare riichi")

    declaration = discards[-1]
    if not declaration.is_riichi_declaration:
        raise ValueError("the final actor discard must declare riichi")
    if declaration.event_index != record.declaration_dahai_event_index:
        raise ValueError("final actor discard event_index must match declaration dahai")
    if declaration.tile != record.riichi_declaration_tile:
        raise ValueError("final actor discard tile must match the declaration tile")
    if declaration.tile_kind != record.riichi_declaration_tile_kind:
        raise ValueError(
            "final actor discard tile_kind must match the declaration tile kind"
        )
