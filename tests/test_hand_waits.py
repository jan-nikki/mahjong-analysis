from collections.abc import Iterable

import pytest

from mahjong_analysis.hand_waits import (
    FixedMeld,
    HandWaits,
    WaitDetail,
    calculate_hand_waits,
)


def expand_hand(hand: str) -> list[str]:
    """Expand compact notation used only by the specification fixtures."""
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
    [
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
            tuple(
                WaitDetail(tile, "kokushi", "kokushi_13men") for tile in KOKUSHI_TILES
            ),
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
    ],
    ids=lambda value: (
        value if isinstance(value, str) and value.startswith("R") else None
    ),
)
def test_specification_hands_r1_to_r16(
    case_id: str,
    concealed_hand: str,
    fixed_melds: Iterable[FixedMeld],
    expected_wait_tiles: tuple[str, ...],
    expected_wait_details: tuple[WaitDetail, ...],
    contains_ryanmen: bool,
    is_pure_ryanmen: bool,
    is_multiwait: bool,
) -> None:
    waits = calculate_hand_waits(expand_hand(concealed_hand), fixed_melds)

    assert waits.wait_tiles == expected_wait_tiles, case_id
    assert waits.wait_details == expected_wait_details, case_id
    assert waits.contains_ryanmen is contains_ryanmen, case_id
    assert waits.is_pure_ryanmen is is_pure_ryanmen, case_id
    assert waits.is_multiwait is is_multiwait, case_id


@pytest.mark.parametrize(
    ("hand", "wait_tile", "detail"),
    [
        (
            "34567m 123p 789p EE",
            "5m",
            WaitDetail("5m", "standard", "ryanmen"),
        ),
        (
            "2333456m 123p 789p",
            "4m",
            WaitDetail("4m", "standard", "ryanmen"),
        ),
    ],
)
def test_duplicate_decompositions_produce_one_wait_detail(
    hand: str,
    wait_tile: str,
    detail: WaitDetail,
) -> None:
    waits = calculate_hand_waits(expand_hand(hand))

    assert waits.wait_details.count(detail) == 1
    assert sum(item.wait_tile == wait_tile for item in waits.wait_details) == 1


def test_wait_tile_owned_four_times_is_not_a_candidate() -> None:
    waits = calculate_hand_waits(expand_hand("11222233m 456p EE"))

    assert "2m" not in waits.wait_tiles


def test_wait_detection_can_keep_standard_and_chiitoitsu_details() -> None:
    waits = calculate_hand_waits(expand_hand("1122334455667m"))

    seven_wait_details = tuple(
        detail for detail in waits.wait_details if detail.wait_tile == "7m"
    )
    assert WaitDetail("7m", "standard", "ryanmen") in seven_wait_details
    assert WaitDetail("7m", "standard", "tanki") in seven_wait_details
    assert WaitDetail("7m", "chiitoitsu", "tanki") in seven_wait_details


def test_chiitoitsu_does_not_count_four_copies_as_two_pairs() -> None:
    waits = calculate_hand_waits(expand_hand("1111m 22p 33p 44s 55s E"))

    assert waits.wait_tiles == ()
    assert waits.wait_details == ()


def test_kokushi_requires_all_thirteen_distinct_terminal_and_honor_kinds() -> None:
    waits = calculate_hand_waits(expand_hand("1m 9m 1p 9p 1s EE SS W N P F"))

    assert waits.wait_tiles == ()
    assert waits.wait_details == ()


def test_upper_edge_penchan_is_classified_correctly() -> None:
    waits = calculate_hand_waits(expand_hand("123m 123p 789p EE 89s"))

    assert waits.wait_tiles == ("7s",)
    assert waits.wait_details == details("7s/standard/penchan")


def test_upper_edge_ryanmen_is_classified_correctly() -> None:
    waits = calculate_hand_waits(expand_hand("123m 123p 789p EE 78s"))

    assert waits.wait_tiles == ("6s", "9s")
    assert waits.wait_details == details(
        "6s/standard/ryanmen",
        "9s/standard/ryanmen",
    )


def test_red_five_pair_produces_normalized_five_wait() -> None:
    waits = calculate_hand_waits(expand_hand("123p 456p 789p EE 5m 5mr"))

    assert waits.wait_tiles == ("5m", "E")
    assert waits.wait_details == details(
        "5m/standard/shanpon",
        "E/standard/shanpon",
    )


def test_fixed_meld_tiles_count_toward_four_copy_limit() -> None:
    fixed_meld = FixedMeld(("5m", "5m", "5m", "5mr"))
    hand = expand_hand("34m 123p 789p EE")

    waits = calculate_hand_waits(hand, [fixed_meld])

    assert "2m" in waits.wait_tiles
    assert "5m" not in waits.wait_tiles


def test_multiple_ankan_reduce_required_concealed_meld_count() -> None:
    fixed_melds = (
        FixedMeld(("1m", "1m", "1m", "1m")),
        FixedMeld(("9m", "9m", "9m", "9m")),
    )

    waits = calculate_hand_waits(expand_hand("123p EE 45s"), fixed_melds)

    assert waits.wait_tiles == ("3s", "6s")
    assert waits.wait_details == details(
        "3s/standard/ryanmen",
        "6s/standard/ryanmen",
    )


def test_four_ankan_leave_only_the_pair_to_complete() -> None:
    fixed_melds = tuple(
        FixedMeld((tile, tile, tile, tile)) for tile in ("1m", "2m", "3p", "4p")
    )

    waits = calculate_hand_waits(["E"], fixed_melds)

    assert waits.wait_tiles == ("E",)
    assert waits.wait_details == details("E/standard/tanki")


def test_calculate_hand_waits_rejects_wrong_concealed_tile_count() -> None:
    with pytest.raises(ValueError, match="physical tile count"):
        calculate_hand_waits(expand_hand("123m"))


def test_fixed_meld_rejects_non_matching_ankan_tiles() -> None:
    with pytest.raises(ValueError, match="same tile kind"):
        FixedMeld(("9m", "9m", "9m", "8m"))


def test_fixed_meld_rejects_non_ankan_type() -> None:
    with pytest.raises(ValueError, match="unsupported fixed meld"):
        FixedMeld(("1m", "2m", "3m"), meld_type="chi")  # type: ignore[arg-type]


def test_wait_shapes_and_count_are_derived_in_definition_order() -> None:
    waits = calculate_hand_waits(expand_hand("22234567m 22p 789s"))

    assert waits.wait_tile_count == 4
    assert waits.wait_shapes == ("ryanmen", "shanpon")


@pytest.mark.parametrize("wait_tile", ["5mr", "invalid"])
def test_wait_detail_rejects_noncanonical_wait_tile(wait_tile: str) -> None:
    with pytest.raises(ValueError, match="wait_tile|invalid tile"):
        WaitDetail(wait_tile, "standard", "tanki")


def test_wait_detail_rejects_unknown_hand_type() -> None:
    with pytest.raises(ValueError, match="hand_type"):
        WaitDetail("E", "unknown", "tanki")  # type: ignore[arg-type]


def test_wait_detail_rejects_unknown_wait_shape() -> None:
    with pytest.raises(ValueError, match="wait_shape"):
        WaitDetail("E", "standard", "unknown")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("hand_type", "wait_shape"),
    [
        ("standard", "kokushi_single"),
        ("chiitoitsu", "ryanmen"),
        ("kokushi", "tanki"),
    ],
)
def test_wait_detail_rejects_invalid_hand_type_shape_combination(
    hand_type: str,
    wait_shape: str,
) -> None:
    with pytest.raises(ValueError, match="combination"):
        WaitDetail("E", hand_type, wait_shape)  # type: ignore[arg-type]


def test_hand_waits_rejects_noncanonical_wait_tile() -> None:
    with pytest.raises(ValueError, match="normalized"):
        HandWaits(
            ("5mr",),
            (WaitDetail("5m", "standard", "tanki"),),
        )


def test_hand_waits_rejects_duplicate_wait_tiles() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        HandWaits(
            ("E", "E"),
            (WaitDetail("E", "standard", "tanki"),),
        )


def test_hand_waits_rejects_noncanonical_wait_tile_order() -> None:
    with pytest.raises(ValueError, match="34-tile order"):
        HandWaits(
            ("1p", "2m"),
            (
                WaitDetail("2m", "standard", "tanki"),
                WaitDetail("1p", "standard", "tanki"),
            ),
        )


def test_hand_waits_rejects_duplicate_wait_details() -> None:
    detail = WaitDetail("E", "standard", "tanki")

    with pytest.raises(ValueError, match="duplicates"):
        HandWaits(("E",), (detail, detail))


def test_hand_waits_rejects_mismatched_wait_tile_sets() -> None:
    with pytest.raises(ValueError, match="must match"):
        HandWaits(("3s", "6s"), ())


def test_hand_waits_rejects_noncanonical_wait_detail_order() -> None:
    with pytest.raises(ValueError, match="canonical order"):
        HandWaits(
            ("2m", "1p"),
            (
                WaitDetail("1p", "standard", "tanki"),
                WaitDetail("2m", "standard", "tanki"),
            ),
        )


def test_empty_hand_waits_is_not_pure_ryanmen() -> None:
    waits = HandWaits((), ())

    assert waits.is_pure_ryanmen is False
