from __future__ import annotations

import pytest

from mahjong_analysis.combo_theory import (
    calculate_simple_combo,
    normalize_unseen_counts,
    sequence_wait_patterns,
)
from mahjong_analysis.tiles import TILE_KINDS, tile_to_index
from mahjong_analysis.unique_tenpai_states import (
    bounded_count_vector_count,
    count_candidate_wait_states,
    count_unique_tenpai_states,
)


def test_seven_simple_combo_matches_specified_formula() -> None:
    unseen = {
        "5m": 3,
        "6m": 2,
        "7m": 3,
        "8m": 1,
        "9m": 4,
    }

    result = calculate_simple_combo("7m", unseen)

    assert [component.pattern.tiles for component in result.sequence_combos] == [
        ("5m", "6m"),
        ("6m", "8m"),
        ("8m", "9m"),
    ]
    assert [component.count for component in result.sequence_combos] == [6, 2, 4]
    assert result.shanpon_combos == 3
    assert result.tanki_combos == 3
    assert result.total == 18


@pytest.mark.parametrize(
    ("tile", "expected"),
    [
        ("1m", [("2m", "3m", "ryanmen")]),
        (
            "2m",
            [("1m", "3m", "kanchan"), ("3m", "4m", "ryanmen")],
        ),
        (
            "3m",
            [
                ("1m", "2m", "penchan"),
                ("2m", "4m", "kanchan"),
                ("4m", "5m", "ryanmen"),
            ],
        ),
        (
            "8m",
            [("6m", "7m", "ryanmen"), ("7m", "9m", "kanchan")],
        ),
        ("9m", [("7m", "8m", "ryanmen")]),
        ("E", []),
    ],
)
def test_sequence_wait_patterns_cover_edges_and_honors(
    tile: str,
    expected: list[tuple[str, str, str]],
) -> None:
    assert [
        (*pattern.tiles, pattern.wait_shape) for pattern in sequence_wait_patterns(tile)
    ] == expected


def test_red_candidate_is_normalized() -> None:
    unseen = {"5m": 3, "4m": 2, "6m": 1}

    result = calculate_simple_combo("5mr", unseen)

    assert result.candidate_tile == "5m"
    assert result.shanpon_combos == 3
    assert result.tanki_combos == 3


@pytest.mark.parametrize(
    "counts",
    [
        [0] * 33,
        [0] * 35,
        [0] * 33 + [5],
        [0] * 33 + [True],
    ],
)
def test_unseen_count_sequence_validation(counts: list[object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        normalize_unseen_counts(counts)  # type: ignore[arg-type]


def test_unseen_count_mapping_requires_normalized_keys() -> None:
    with pytest.raises(ValueError, match="must be normalized"):
        normalize_unseen_counts({"5mr": 1})


def test_bounded_count_vector_count() -> None:
    assert bounded_count_vector_count((2, 2), 2) == 3
    assert bounded_count_vector_count((1, 1, 1), 2) == 3
    assert bounded_count_vector_count((1, 1), 3) == 0


def test_reduced_universe_reverses_simple_and_unique_rankings() -> None:
    unseen = dict(
        zip(
            TILE_KINDS[:9],
            (3, 1, 1, 1, 1, 1, 4, 4, 1),
            strict=True,
        )
    )

    simple_2m = calculate_simple_combo("2m", unseen)
    simple_4m = calculate_simple_combo("4m", unseen)
    exact = count_unique_tenpai_states(unseen)
    fast_2m = count_candidate_wait_states(unseen, "2m")
    fast_4m = count_candidate_wait_states(unseen, "4m")

    assert simple_2m.total == 5
    assert simple_4m.total == 4
    assert exact.tenpai_state_count == 176
    assert exact.for_tile("2m").state_count == 39
    assert exact.for_tile("4m").state_count == 46
    assert exact.for_tile("2m").physical_weight == 265
    assert exact.for_tile("4m").physical_weight == 396
    assert (fast_2m.state_count, fast_2m.physical_weight) == (39, 265)
    assert (fast_4m.state_count, fast_4m.physical_weight) == (46, 396)


@pytest.mark.parametrize("tile", [f"{rank}m" for rank in range(1, 10)])
def test_block_factorization_matches_brute_force_for_every_reduced_wait(
    tile: str,
) -> None:
    unseen = dict(zip(TILE_KINDS[:9], (3, 1, 1, 1, 1, 1, 4, 4, 1), strict=True))

    brute_force = count_unique_tenpai_states(unseen)
    factored = count_candidate_wait_states(unseen, tile)

    assert factored.state_count == brute_force.for_tile(tile).state_count
    assert factored.physical_weight == brute_force.for_tile(tile).physical_weight


def test_factored_count_unions_standard_and_chiitoitsu_once() -> None:
    unseen = {
        "1m": 2,
        "2m": 2,
        "3m": 2,
        "4m": 2,
        "5m": 2,
        "6m": 2,
        "7m": 1,
    }

    result = count_candidate_wait_states(unseen, "7m")

    assert result.state_count == 1
    assert result.standard_state_count == 1
    assert result.chiitoitsu_state_count == 1
    assert result.multi_hand_type_state_count == 1


def test_factored_count_includes_chiitoitsu_only_wait() -> None:
    unseen = {
        "1m": 2,
        "2m": 2,
        "3p": 2,
        "4p": 2,
        "5s": 2,
        "6s": 2,
        "E": 1,
    }

    result = count_candidate_wait_states(unseen, "E")

    assert result.state_count == 1
    assert result.standard_state_count == 0
    assert result.chiitoitsu_state_count == 1
    assert result.kokushi_state_count == 0


def test_factored_count_includes_kokushi_single_wait() -> None:
    unseen = {
        "1m": 1,
        "9m": 1,
        "1p": 1,
        "9p": 1,
        "1s": 1,
        "9s": 1,
        "E": 2,
        "S": 1,
        "W": 1,
        "N": 1,
        "P": 1,
        "F": 1,
    }

    result = count_candidate_wait_states(unseen, "C")

    assert result.state_count == 1
    assert result.standard_state_count == 0
    assert result.chiitoitsu_state_count == 0
    assert result.kokushi_state_count == 1


def test_exact_enumerator_refuses_unbounded_full_state_space() -> None:
    unseen = [4] * len(TILE_KINDS)

    with pytest.raises(ValueError, match="exceeding limit"):
        count_unique_tenpai_states(unseen, max_candidate_vectors=10)


def test_exact_summary_keeps_all_canonical_wait_tiles() -> None:
    unseen = {
        "1m": 3,
        "2m": 1,
        "3m": 1,
        "4m": 1,
        "5m": 1,
        "6m": 1,
        "7m": 4,
        "8m": 4,
        "9m": 1,
    }

    result = count_unique_tenpai_states(unseen)

    assert len(result.wait_counts) == len(TILE_KINDS)
    assert result.wait_counts[tile_to_index("2m")].wait_tile == "2m"
