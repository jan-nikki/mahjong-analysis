import json
from dataclasses import asdict, replace

import pytest

import mahjong_analysis.riichi_wait_records as record_module
from mahjong_analysis.hand_waits import (
    WAIT_SHAPE_ORDER,
    FixedMeld,
    WaitDetail,
)
from mahjong_analysis.riichi import (
    ActorDiscard,
    EstablishedRiichi,
    extract_established_riichis,
)
from mahjong_analysis.riichi_wait_records import (
    RiichiWaitRecord,
    build_riichi_wait_record,
    build_riichi_wait_records,
)


def expand_hand(hand: str) -> list[str]:
    """Expand compact notation used by the specification fixtures."""
    tiles: list[str] = []
    for group in hand.split():
        if group in {"5mr", "5pr", "5sr"}:
            tiles.append(group)
        elif group[-1] in "mps":
            tiles.extend(f"{rank}{group[-1]}" for rank in group[:-1])
        else:
            tiles.extend(group)
    return tiles


def details(*values: str) -> tuple[WaitDetail, ...]:
    result = []
    for value in values:
        wait_tile, hand_type, wait_shape = value.split("/")
        result.append(WaitDetail(wait_tile, hand_type, wait_shape))
    return tuple(result)


KOKUSHI_TILES = (
    "1m",
    "9m",
    "1p",
    "9p",
    "1s",
    "9s",
    "E",
    "S",
    "W",
    "N",
    "P",
    "F",
    "C",
)

SPECIFICATION_CASES = (
    (
        "R1",
        "123m 123p 789p EE 45s",
        (),
        ("3s", "6s"),
        details("3s/standard/ryanmen", "6s/standard/ryanmen"),
        True,
        True,
        False,
    ),
    (
        "R2",
        "123m 123p 789p EE 46s",
        (),
        ("5s",),
        details("5s/standard/kanchan"),
        False,
        False,
        False,
    ),
    (
        "R3",
        "123m 456m 789p EE 12s",
        (),
        ("3s",),
        details("3s/standard/penchan"),
        False,
        False,
        False,
    ),
    (
        "R4",
        "123m 456m 789m 55p 77s",
        (),
        ("5p", "7s"),
        details("5p/standard/shanpon", "7s/standard/shanpon"),
        False,
        False,
        False,
    ),
    (
        "R5",
        "123m 456m 789m 123p 5s",
        (),
        ("5s",),
        details("5s/standard/tanki"),
        False,
        False,
        False,
    ),
    (
        "R6",
        "34567m 123p 789p EE",
        (),
        ("2m", "5m", "8m"),
        details(
            "2m/standard/ryanmen",
            "5m/standard/ryanmen",
            "8m/standard/ryanmen",
        ),
        True,
        False,
        True,
    ),
    (
        "R7",
        "2345678m 123p 789p",
        (),
        ("2m", "5m", "8m"),
        details(
            "2m/standard/tanki",
            "5m/standard/tanki",
            "8m/standard/tanki",
        ),
        False,
        False,
        True,
    ),
    (
        "R8",
        "2333456m 123p 789p",
        (),
        ("1m", "2m", "4m", "7m"),
        details(
            "1m/standard/ryanmen",
            "2m/standard/tanki",
            "4m/standard/ryanmen",
            "7m/standard/ryanmen",
        ),
        True,
        False,
        True,
    ),
    (
        "R9",
        "123m 456m 789m 4556p",
        (),
        ("5p",),
        details(
            "5p/standard/kanchan",
            "5p/standard/tanki",
        ),
        False,
        False,
        False,
    ),
    (
        "R10",
        "11m 22m 33p 44p 55s 66s E",
        (),
        ("E",),
        details("E/chiitoitsu/tanki"),
        False,
        False,
        False,
    ),
    (
        "R11",
        "1m 9m 1p 9p 1s 9s EE S W N P F",
        (),
        ("C",),
        details("C/kokushi/kokushi_single"),
        False,
        False,
        False,
    ),
    (
        "R12",
        "1m 9m 1p 9p 1s 9s E S W N P F C",
        (),
        KOKUSHI_TILES,
        tuple(WaitDetail(tile, "kokushi", "kokushi_13men") for tile in KOKUSHI_TILES),
        False,
        False,
        True,
    ),
    (
        "R13",
        "123m 123p 789p EE 4s 5sr",
        (),
        ("3s", "6s"),
        details("3s/standard/ryanmen", "6s/standard/ryanmen"),
        True,
        True,
        False,
    ),
    (
        "R14",
        "123p 789p EE 45s",
        (FixedMeld(("9m", "9m", "9m", "9m")),),
        ("3s", "6s"),
        details("3s/standard/ryanmen", "6s/standard/ryanmen"),
        True,
        True,
        False,
    ),
    (
        "R15",
        "22234567m 22p 789s",
        (),
        ("2m", "5m", "8m", "2p"),
        details(
            "2m/standard/ryanmen",
            "2m/standard/shanpon",
            "5m/standard/ryanmen",
            "8m/standard/ryanmen",
            "2p/standard/shanpon",
        ),
        True,
        False,
        True,
    ),
    (
        "R16",
        "11122233m 456p EE",
        (),
        ("3m", "E"),
        details(
            "3m/standard/penchan",
            "3m/standard/shanpon",
            "E/standard/shanpon",
        ),
        False,
        False,
        False,
    ),
)


def make_established_riichi(
    concealed_hand: str,
    *,
    fixed_melds: tuple[FixedMeld, ...] = (),
    actor: int = 1,
    reach_event_index: int = 10,
    declaration_dahai_event_index: int = 11,
    reach_accepted_event_index: int = 12,
    riichi_discard_number: int = 1,
    declaration_tile: str = "9s",
    declaration_tile_kind: str = "9s",
    discards: tuple[ActorDiscard, ...] | None = None,
) -> EstablishedRiichi:
    if discards is None:
        discards = (
            ActorDiscard(
                discard_number=riichi_discard_number,
                tile=declaration_tile,
                tile_kind=declaration_tile_kind,
                tsumogiri=True,
                event_index=declaration_dahai_event_index,
                is_riichi_declaration=True,
            ),
        )
    return EstablishedRiichi(
        actor=actor,
        reach_event_index=reach_event_index,
        declaration_dahai_event_index=declaration_dahai_event_index,
        reach_accepted_event_index=reach_accepted_event_index,
        riichi_discard_number=riichi_discard_number,
        riichi_declaration_tile=declaration_tile,
        riichi_declaration_tile_kind=declaration_tile_kind,
        concealed_tiles_after_discard=tuple(expand_hand(concealed_hand)),
        fixed_melds=fixed_melds,
        actor_discards_before_riichi=discards,
    )


def build_two_discard_record() -> RiichiWaitRecord:
    discards = (
        ActorDiscard(1, "1m", "1m", False, 7, False),
        ActorDiscard(2, "9s", "9s", True, 11, True),
    )
    return build_riichi_wait_record(
        make_established_riichi(
            "123m 123p 789p EE 45s",
            riichi_discard_number=2,
            discards=discards,
        )
    )


@pytest.mark.parametrize(
    (
        "case_id",
        "concealed_hand",
        "fixed_melds",
        "expected_wait_tiles",
        "expected_wait_details",
        "contains_ryanmen",
        "is_pure_ryanmen",
        "is_multiwait",
    ),
    SPECIFICATION_CASES,
    ids=[case[0] for case in SPECIFICATION_CASES],
)
def test_builds_wait_record_for_specification_cases_r1_to_r16(
    case_id: str,
    concealed_hand: str,
    fixed_melds: tuple[FixedMeld, ...],
    expected_wait_tiles: tuple[str, ...],
    expected_wait_details: tuple[WaitDetail, ...],
    contains_ryanmen: bool,
    is_pure_ryanmen: bool,
    is_multiwait: bool,
) -> None:
    established = make_established_riichi(
        concealed_hand,
        fixed_melds=fixed_melds,
    )

    record = build_riichi_wait_record(established)

    expected_wait_shapes = tuple(
        shape
        for shape in WAIT_SHAPE_ORDER
        if any(detail.wait_shape == shape for detail in expected_wait_details)
    )
    assert record.wait_tiles == expected_wait_tiles, case_id
    assert record.wait_tile_count == len(expected_wait_tiles), case_id
    assert record.wait_details == expected_wait_details, case_id
    assert record.wait_shapes == expected_wait_shapes, case_id
    assert record.contains_ryanmen is contains_ryanmen, case_id
    assert record.is_pure_ryanmen is is_pure_ryanmen, case_id
    assert record.is_multiwait is is_multiwait, case_id
    assert record.concealed_tiles_after_discard == (
        established.concealed_tiles_after_discard
    )
    assert record.fixed_melds == established.fixed_melds


def test_preserves_riichi_metadata_and_discard_history_and_is_serializable() -> None:
    discards = (
        ActorDiscard(
            discard_number=1,
            tile="5mr",
            tile_kind="5m",
            tsumogiri=False,
            event_index=7,
            is_riichi_declaration=False,
            was_called=True,
            call_type="pon",
            called_by_actor=3,
            call_event_index=8,
        ),
        ActorDiscard(
            discard_number=2,
            tile="5pr",
            tile_kind="5p",
            tsumogiri=True,
            event_index=11,
            is_riichi_declaration=True,
        ),
    )
    established = make_established_riichi(
        "123m 123p 789p EE 45s",
        actor=2,
        riichi_discard_number=2,
        declaration_tile="5pr",
        declaration_tile_kind="5p",
        discards=discards,
    )

    record = build_riichi_wait_record(established)

    assert (
        record.actor,
        record.riichi_discard_number,
        record.riichi_declaration_tile,
        record.riichi_declaration_tile_kind,
    ) == (2, 2, "5pr", "5p")
    assert (
        record.reach_event_index,
        record.declaration_dahai_event_index,
        record.reach_accepted_event_index,
    ) == (10, 11, 12)
    assert record.actor_discards_before_riichi == discards

    payload = json.loads(json.dumps(asdict(record)))
    assert payload["riichi_declaration_tile"] == "5pr"
    assert payload["riichi_declaration_tile_kind"] == "5p"
    assert payload["actor_discards_before_riichi"] == [
        {
            "discard_number": 1,
            "tile": "5mr",
            "tile_kind": "5m",
            "tsumogiri": False,
            "event_index": 7,
            "is_riichi_declaration": False,
            "was_called": True,
            "call_type": "pon",
            "called_by_actor": 3,
            "call_event_index": 8,
        },
        {
            "discard_number": 2,
            "tile": "5pr",
            "tile_kind": "5p",
            "tsumogiri": True,
            "event_index": 11,
            "is_riichi_declaration": True,
            "was_called": False,
            "call_type": None,
            "called_by_actor": None,
            "call_event_index": None,
        },
    ]
    assert payload["wait_tiles"] == ["3s", "6s"]
    assert payload["wait_details"] == [
        {
            "wait_tile": "3s",
            "hand_type": "standard",
            "wait_shape": "ryanmen",
        },
        {
            "wait_tile": "6s",
            "hand_type": "standard",
            "wait_shape": "ryanmen",
        },
    ]


def test_builds_one_record_per_established_riichi_in_one_kyoku() -> None:
    first_hand = expand_hand("123m 123p 789p EE 45s")
    second_hand = expand_hand("123m 456m 789m 123p 5s")
    hands = [list(first_hand), list(first_hand), list(first_hand), list(second_hand)]
    kyoku = [
        {"type": "start_kyoku", "oya": 0, "tehais": hands},
        {"type": "tsumo", "actor": 1, "pai": "9s"},
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "9s", "tsumogiri": True},
        {"type": "reach_accepted", "actor": 1},
        {"type": "tsumo", "actor": 3, "pai": "N"},
        {"type": "dahai", "actor": 3, "pai": "N", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "C"},
        {"type": "reach", "actor": 3},
        {"type": "dahai", "actor": 3, "pai": "C", "tsumogiri": True},
        {"type": "reach_accepted", "actor": 3},
        {"type": "end_kyoku"},
    ]

    records = build_riichi_wait_records(extract_established_riichis(kyoku))

    assert len(records) == 2
    first, second = records
    assert (
        first.actor,
        first.concealed_tiles_after_discard,
        first.riichi_declaration_tile,
        first.riichi_discard_number,
        tuple(discard.tile for discard in first.actor_discards_before_riichi),
        first.wait_tiles,
    ) == (1, tuple(first_hand), "9s", 1, ("9s",), ("3s", "6s"))
    assert (
        second.actor,
        second.concealed_tiles_after_discard,
        second.riichi_declaration_tile,
        second.riichi_discard_number,
        tuple(discard.tile for discard in second.actor_discards_before_riichi),
        second.wait_tiles,
    ) == (3, tuple(second_hand), "C", 2, ("N", "C"), ("5s",))


def test_factory_calculates_hand_waits_once_per_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = record_module.calculate_hand_waits
    call_count = 0

    def counted_calculate_hand_waits(
        concealed_tiles: tuple[str, ...],
        fixed_melds: tuple[FixedMeld, ...],
    ) -> object:
        nonlocal call_count
        call_count += 1
        return original(concealed_tiles, fixed_melds)

    monkeypatch.setattr(
        record_module,
        "calculate_hand_waits",
        counted_calculate_hand_waits,
    )

    record = build_riichi_wait_record(make_established_riichi("123m 123p 789p EE 45s"))

    assert call_count == 1
    assert record.wait_tiles == ("3s", "6s")


def test_multiple_record_factory_calculates_once_per_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = record_module.calculate_hand_waits
    call_count = 0

    def counted_calculate_hand_waits(
        concealed_tiles: tuple[str, ...],
        fixed_melds: tuple[FixedMeld, ...],
    ) -> object:
        nonlocal call_count
        call_count += 1
        return original(concealed_tiles, fixed_melds)

    monkeypatch.setattr(
        record_module,
        "calculate_hand_waits",
        counted_calculate_hand_waits,
    )
    established = (
        make_established_riichi("123m 123p 789p EE 45s"),
        make_established_riichi("123m 456m 789m 123p 5s"),
        make_established_riichi("123m 456m 789m 55p 77s"),
    )

    records = build_riichi_wait_records(established)

    assert call_count == len(records) == 3
    assert tuple(record.wait_tiles for record in records) == (
        ("3s", "6s"),
        ("5s",),
        ("5p", "7s"),
    )


def test_rejects_established_riichi_with_empty_waits() -> None:
    established = make_established_riichi("1m 9m 1p 9p 1s EE SS W N P F")

    with pytest.raises(ValueError, match="no wait tiles"):
        build_riichi_wait_record(established)


@pytest.mark.parametrize(
    ("changes", "expected_message"),
    [
        ({"wait_tile_count": 1}, "wait_tile_count"),
        ({"wait_shapes": ("kanchan",)}, "wait_shapes"),
        ({"contains_ryanmen": False}, "contains_ryanmen"),
        ({"is_pure_ryanmen": False}, "is_pure_ryanmen"),
        ({"is_multiwait": True}, "is_multiwait"),
        ({"reach_event_index": 11}, "event indexes"),
        ({"declaration_dahai_event_index": 12}, "event indexes"),
        ({"riichi_discard_number": 0}, "riichi_discard_number"),
        ({"actor": True}, "actor must be an integer"),
        ({"actor": 4}, "actor must be between"),
        ({"reach_event_index": "10"}, "event indexes must be integers"),
        ({"reach_event_index": -1}, "event indexes must be non-negative"),
        ({"riichi_discard_number": True}, "riichi_discard_number"),
        (
            {
                "riichi_declaration_tile": "5mr",
                "riichi_declaration_tile_kind": "5p",
            },
            "normalized declaration tile",
        ),
    ],
    ids=(
        "wait-tile-count",
        "wait-shapes",
        "contains-ryanmen",
        "pure-ryanmen",
        "multiwait",
        "reach-index-order",
        "accepted-index-order",
        "discard-number",
        "actor-type",
        "actor-range",
        "event-index-type",
        "event-index-negative",
        "discard-number-type",
        "declaration-normalization",
    ),
)
def test_direct_construction_rejects_invalid_derived_or_identity_values(
    changes: dict[str, object],
    expected_message: str,
) -> None:
    record = build_riichi_wait_record(make_established_riichi("123m 123p 789p EE 45s"))

    with pytest.raises((TypeError, ValueError), match=expected_message):
        replace(record, **changes)


def test_direct_construction_rejects_mismatched_wait_tile_and_detail_sets() -> None:
    record = build_riichi_wait_record(make_established_riichi("123m 123p 789p EE 45s"))

    with pytest.raises(ValueError, match="wait_tiles must match"):
        replace(record, wait_tiles=("3s",))


def test_direct_construction_rejects_waits_from_a_different_hand() -> None:
    record = build_riichi_wait_record(make_established_riichi("123m 123p 789p EE 45s"))

    with pytest.raises(ValueError, match="declaration-post hand"):
        replace(
            record,
            wait_tiles=("5s",),
            wait_tile_count=1,
            wait_details=details("5s/standard/kanchan"),
            wait_shapes=("kanchan",),
            contains_ryanmen=False,
            is_pure_ryanmen=False,
            is_multiwait=False,
        )


def test_public_direct_construction_still_recalculates_and_rejects_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = build_riichi_wait_record(make_established_riichi("123m 123p 789p EE 45s"))
    original = record_module.calculate_hand_waits
    call_count = 0

    def counted_calculate_hand_waits(
        concealed_tiles: tuple[str, ...],
        fixed_melds: tuple[FixedMeld, ...],
    ) -> object:
        nonlocal call_count
        call_count += 1
        return original(concealed_tiles, fixed_melds)

    monkeypatch.setattr(
        record_module,
        "calculate_hand_waits",
        counted_calculate_hand_waits,
    )

    with pytest.raises(ValueError, match="declaration-post hand"):
        RiichiWaitRecord(
            actor=source.actor,
            riichi_discard_number=source.riichi_discard_number,
            riichi_declaration_tile=source.riichi_declaration_tile,
            riichi_declaration_tile_kind=source.riichi_declaration_tile_kind,
            reach_event_index=source.reach_event_index,
            declaration_dahai_event_index=source.declaration_dahai_event_index,
            reach_accepted_event_index=source.reach_accepted_event_index,
            concealed_tiles_after_discard=tuple(expand_hand("123m 456m 789m 123p 5s")),
            fixed_melds=source.fixed_melds,
            actor_discards_before_riichi=source.actor_discards_before_riichi,
            wait_tiles=source.wait_tiles,
            wait_tile_count=source.wait_tile_count,
            wait_details=source.wait_details,
            wait_shapes=source.wait_shapes,
            contains_ryanmen=source.contains_ryanmen,
            is_pure_ryanmen=source.is_pure_ryanmen,
            is_multiwait=source.is_multiwait,
        )

    assert call_count == 1


def test_direct_construction_cannot_forge_factory_validation() -> None:
    record = build_riichi_wait_record(make_established_riichi("123m 123p 789p EE 45s"))

    with pytest.raises(TypeError, match="reserved for the internal factory"):
        replace(record, _factory_validated_waits=object())


def test_direct_construction_rejects_empty_waits() -> None:
    record = build_riichi_wait_record(make_established_riichi("123m 123p 789p EE 45s"))

    with pytest.raises(ValueError, match="at least one wait tile"):
        replace(
            record,
            wait_tiles=(),
            wait_tile_count=0,
            wait_details=(),
            wait_shapes=(),
            contains_ryanmen=False,
            is_pure_ryanmen=False,
            is_multiwait=False,
        )


@pytest.mark.parametrize(
    ("case_id", "expected_message"),
    [
        ("history-length", "history length"),
        ("discard-numbers", "discard numbers"),
        ("event-index-order", "strictly increasing"),
        ("early-declaration", "only the final"),
        ("final-not-declaration", "final actor discard must declare"),
        ("declaration-event", "event_index must match"),
        ("declaration-tile", "tile must match"),
    ],
)
def test_direct_construction_rejects_invalid_actor_discard_history(
    case_id: str,
    expected_message: str,
) -> None:
    record = build_two_discard_record()
    first, declaration = record.actor_discards_before_riichi

    if case_id == "history-length":
        invalid_history = (declaration,)
    elif case_id == "discard-numbers":
        invalid_history = (first, replace(declaration, discard_number=3))
    elif case_id == "event-index-order":
        invalid_history = (replace(first, event_index=11), declaration)
    elif case_id == "early-declaration":
        invalid_history = (replace(first, is_riichi_declaration=True), declaration)
    elif case_id == "final-not-declaration":
        invalid_history = (first, replace(declaration, is_riichi_declaration=False))
    elif case_id == "declaration-event":
        invalid_history = (first, replace(declaration, event_index=10))
    else:
        invalid_history = (
            first,
            replace(declaration, tile="1s", tile_kind="1s"),
        )

    with pytest.raises(ValueError, match=expected_message):
        replace(record, actor_discards_before_riichi=invalid_history)


def test_direct_construction_rejects_non_tuple_actor_discard_history() -> None:
    record = build_two_discard_record()

    with pytest.raises(TypeError, match="actor_discards_before_riichi must be a tuple"):
        replace(
            record,
            actor_discards_before_riichi=list(record.actor_discards_before_riichi),
        )
