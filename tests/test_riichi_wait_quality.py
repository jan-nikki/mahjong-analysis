import gzip
from pathlib import Path

import pytest

from test_riichi_wait_summary import _record

from mahjong_analysis.hand_waits import FixedMeld, calculate_hand_waits
from mahjong_analysis.riichi_wait_dataset import (
    DatasetFixedMeld,
    DatasetWaitDetail,
    serialize_dataset_record,
)
from mahjong_analysis.riichi_wait_quality import (
    AnnualQualityTask,
    QualityAccumulator,
    WaitQualityFacts,
    aggregate_annual_quality,
    evaluate_wait_quality,
    quality_results_document,
    run_quality_tasks,
)


def expand_hand(hand: str) -> tuple[str, ...]:
    tiles: list[str] = []
    for group in hand.split():
        if group in {"5mr", "5pr", "5sr"}:
            tiles.append(group)
        elif group[-1] in "mps":
            tiles.extend(f"{rank}{group[-1]}" for rank in group[:-1])
        else:
            tiles.extend(group)
    return tuple(tiles)


RED_FIVE_ANKAN = (FixedMeld(("5m", "5m", "5m", "5mr")),)
THREE_SOU_ANKAN = (FixedMeld(("3s",) * 4),)
KOKUSHI_13MEN = "1m 9m 1p 9p 1s 9s E S W N P F C"


def evaluate_hand(
    hand: str, fixed_melds: tuple[FixedMeld, ...] = ()
) -> WaitQualityFacts:
    concealed = expand_hand(hand)
    waits = calculate_hand_waits(concealed, fixed_melds)
    return evaluate_wait_quality(
        concealed,
        tuple(DatasetFixedMeld(meld.meld_type, meld.tiles) for meld in fixed_melds),
        tuple(
            DatasetWaitDetail(detail.wait_tile, detail.hand_type, detail.wait_shape)
            for detail in waits.wait_details
        ),
    )


@pytest.mark.parametrize(
    ("hand", "fixed_melds", "expected_wait_tiles", "expected"),
    [
        pytest.param(
            "2345m 123p 789p 456s",
            (),
            ("2m", "5m"),
            (False, True, True, 6, True),
            id="nobetan-six-copies",
        ),
        pytest.param(
            "3455m 123p 789p 456s",
            (),
            ("2m", "5m"),
            (True, False, True, 6, True),
            id="one-tanki-interpretation-is-not-nobetan",
        ),
        pytest.param(
            "2255m 123p 789p 456s",
            (),
            ("2m", "5m"),
            (False, False, False, 4, False),
            id="suji-spaced-shanpon-is-not-nobetan",
        ),
        pytest.param(
            "22234m EE 123p 789s",
            (),
            ("2m", "5m", "E"),
            (True, False, True, 7, True),
            id="ryanmen-shanpon-overlap-counted-once",
        ),
        pytest.param(
            "1234444567m EEE",
            (),
            ("1m", "7m"),
            (False, True, True, 6, True),
            id="nobetan-six-ranks-apart",
        ),
        pytest.param(
            "2345678m 123p 789s",
            (),
            ("2m", "5m", "8m"),
            (False, True, True, 9, True),
            id="three-sided-nobetan",
        ),
        pytest.param(
            "44p 567p 12s 456s",
            THREE_SOU_ANKAN,
            ("3s",),
            (False, False, False, 0, False),
            id="formal-fifth-copy-wait-contributes-zero",
        ),
        pytest.param(
            "34m 123p 789p EE",
            RED_FIVE_ANKAN,
            ("2m", "5m"),
            (True, False, True, 4, False),
            id="red-ankan-removes-four-physical-copies",
        ),
        pytest.param(
            "11m 22m 33p 44p 55s 66s E",
            (),
            ("E",),
            (False, False, False, 3, False),
            id="chiitoitsu-is-not-nobetan",
        ),
        pytest.param(
            KOKUSHI_13MEN,
            (),
            ("1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C"),
            (False, False, False, 39, True),
            id="kokushi-can-be-good-without-suji",
        ),
        pytest.param(
            "1m 9m 1p 9p 1s 9s EE S W N P F",
            (),
            ("C",),
            (False, False, False, 4, False),
            id="kokushi-single-wait-below-threshold",
        ),
        pytest.param(
            "1122334455667m",
            (),
            ("1m", "4m", "7m"),
            (True, True, True, 7, True),
            id="standard-and-chiitoitsu-interpretations-counted-once",
        ),
        pytest.param(
            "2345m 123m 789p 456s",
            (),
            ("2m", "5m"),
            (False, True, True, 5, True),
            id="good-wait-threshold-is-inclusive",
        ),
        pytest.param(
            "234m 5mr 123p 789p 456s",
            (),
            ("2m", "5m"),
            (False, True, True, 6, True),
            id="concealed-red-five-is-normalized",
        ),
    ],
)
def test_wait_quality_from_complete_hands(
    hand: str,
    fixed_melds: tuple[FixedMeld, ...],
    expected_wait_tiles: tuple[str, ...],
    expected: tuple[bool, bool, bool, int, bool],
) -> None:
    waits = calculate_hand_waits(expand_hand(hand), fixed_melds)
    assert waits.wait_tiles == expected_wait_tiles

    facts = evaluate_hand(hand, fixed_melds)

    assert (
        facts.contains_ryanmen,
        facts.contains_nobetan,
        facts.contains_suji,
        facts.self_excluded_wait_copies,
        facts.is_good_wait,
    ) == expected


def test_quality_rejects_five_actual_owned_copies() -> None:
    with pytest.raises(ValueError):
        evaluate_wait_quality(
            expand_hand("34m 123p 789p EE"),
            (DatasetFixedMeld("ankan", ("3m",) * 4),),
            (
                DatasetWaitDetail("2m", "standard", "ryanmen"),
                DatasetWaitDetail("5m", "standard", "ryanmen"),
            ),
        )


def test_quality_rejects_incomplete_hand() -> None:
    with pytest.raises(ValueError, match="concealed/fixed meld counts"):
        evaluate_wait_quality((), (), (DatasetWaitDetail("2m", "standard", "ryanmen"),))


def sample_facts() -> tuple[WaitQualityFacts, ...]:
    return (
        evaluate_hand("2345m 123p 789p 456s"),
        evaluate_hand("3455m 123p 789p 456s"),
        evaluate_hand("2255m 123p 789p 456s"),
        evaluate_hand("34m 123p 789p EE", RED_FIVE_ANKAN),
        evaluate_hand(KOKUSHI_13MEN),
        evaluate_hand("1122334455667m"),
    )


def test_accumulator_uses_all_records_as_each_denominator() -> None:
    accumulator = QualityAccumulator()
    for facts in sample_facts():
        accumulator.add(facts)

    assert accumulator.to_dict() == {
        "record_count": 6,
        "contains_ryanmen": {"count": 3, "denominator": 6, "rate": 0.5},
        "contains_nobetan": {"count": 2, "denominator": 6, "rate": 2 / 6},
        "contains_suji": {"count": 4, "denominator": 6, "rate": 4 / 6},
        "good_wait": {"count": 4, "denominator": 6, "rate": 4 / 6},
        "wait_copy_distribution": [
            {"wait_copies": 4, "count": 2},
            {"wait_copies": 6, "count": 2},
            {"wait_copies": 7, "count": 1},
            {"wait_copies": 39, "count": 1},
        ],
        "suji_origin_counts": {
            "ryanmen_only": 2,
            "nobetan_only": 1,
            "both": 1,
            "neither": 2,
        },
        "suji_good_cross_counts": {
            "suji_and_good": 3,
            "suji_only": 1,
            "good_only": 1,
            "neither": 1,
        },
    }


def test_empty_accumulator_has_null_rates() -> None:
    result = QualityAccumulator().to_dict()

    assert result["record_count"] == 0
    for name in ("contains_ryanmen", "contains_nobetan", "contains_suji", "good_wait"):
        assert result[name] == {"count": 0, "denominator": 0, "rate": None}
    assert result["wait_copy_distribution"] == []
    assert result["suji_origin_counts"] == {
        "ryanmen_only": 0,
        "nobetan_only": 0,
        "both": 0,
        "neither": 0,
    }
    assert result["suji_good_cross_counts"] == {
        "suji_and_good": 0,
        "suji_only": 0,
        "good_only": 0,
        "neither": 0,
    }


def test_merging_accumulators_matches_adding_all_records() -> None:
    complete = QualityAccumulator()
    first = QualityAccumulator()
    second = QualityAccumulator()
    facts = sample_facts()
    for index, item in enumerate(facts):
        complete.add(item)
        (first if index < 3 else second).add(item)

    first.merge(second)

    assert first.to_dict() == complete.to_dict()


def annual_task(tmp_path: Path, year: int) -> AnnualQualityTask:
    path = tmp_path / f"{year}.jsonl.gz"
    with gzip.open(path, "wb") as stream:
        for turn, hand in ((2, "2345m 123p 789p 456s"), (5, "2255m 123p 789p 456s")):
            concealed = expand_hand(hand)
            waits = calculate_hand_waits(concealed)
            record = _record(
                year=year,
                turn=turn,
                concealed=concealed,
                waits=tuple(
                    (d.wait_tile, d.hand_type, d.wait_shape) for d in waits.wait_details
                ),
            )
            stream.write(serialize_dataset_record(record) + b"\n")
    return AnnualQualityTask(year, str(path), 2)


def test_annual_streaming_counts_and_exact_turns(tmp_path: Path) -> None:
    result = aggregate_annual_quality(annual_task(tmp_path, 2025))
    document = quality_results_document((result,))
    assert document["overall"]["record_count"] == 2
    assert document["overall"]["contains_suji"] == {
        "count": 1,
        "denominator": 2,
        "rate": 0.5,
    }
    assert document["overall"]["good_wait"] == {
        "count": 1,
        "denominator": 2,
        "rate": 0.5,
    }
    assert [r["riichi_discard_number"] for r in document["by_turn"]] == [2, 5]
    assert [r["contains_suji"]["count"] for r in document["by_turn"]] == [1, 0]
    assert document["years"][0]["by_turn"] == document["by_turn"]


def test_serial_parallel_and_result_order_match(tmp_path: Path) -> None:
    tasks = (annual_task(tmp_path, 2025), annual_task(tmp_path, 2009))
    serial = run_quality_tasks(tasks, workers=1)
    parallel = run_quality_tasks(tasks, workers=2)
    assert serial == parallel
    assert [r.year for r in parallel] == [2025, 2009]
    assert quality_results_document(serial) == quality_results_document(
        tuple(reversed(parallel))
    )
    assert [r["year"] for r in quality_results_document(serial)["years"]] == [
        2009,
        2025,
    ]


@pytest.mark.parametrize("workers", [1, 2])
def test_bad_annual_count_propagates(tmp_path: Path, workers: int) -> None:
    task = annual_task(tmp_path, 2025)
    wrong = AnnualQualityTask(task.year, task.path, 3)
    with pytest.raises(ValueError, match="annual record count"):
        run_quality_tasks((wrong,), workers=workers)


def test_malformed_input_is_not_skipped(tmp_path: Path) -> None:
    task = annual_task(tmp_path, 2025)
    with gzip.open(task.path, "ab") as stream:
        stream.write(b"{broken\n")
    with pytest.raises(ValueError, match="invalid dataset JSON"):
        aggregate_annual_quality(task)


def test_wrong_annual_year_is_rejected(tmp_path: Path) -> None:
    task = annual_task(tmp_path, 2025)
    with pytest.raises(ValueError, match="unexpected record year"):
        aggregate_annual_quality(AnnualQualityTask(2024, task.path, 2))
