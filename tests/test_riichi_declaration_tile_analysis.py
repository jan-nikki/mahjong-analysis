import copy
import gzip
from dataclasses import replace
from pathlib import Path

import pytest
from test_riichi_wait_summary import _record

from analysis.analyze_riichi_declaration_tiles import check_strong_results_unchanged
from mahjong_analysis.hand_waits import calculate_hand_waits
from mahjong_analysis.riichi_declaration_tile_analysis import (
    RANK_WAIT_PAIRS,
    AnnualDeclarationTileTask,
    DeclarationTileAccumulator,
    StrengthAccumulator,
    WaitStrengthFacts,
    aggregate_annual_declaration_tiles,
    classify_declaration_tile,
    declaration_tile_results_document,
    dora_from_marker,
    evaluate_wait_strength,
    run_declaration_tile_tasks,
    same_suji_line_wait_rank_shapes,
    same_suji_line_wait_shapes,
)
from mahjong_analysis.riichi_wait_dataset import (
    DatasetWaitDetail,
    serialize_dataset_record,
)
from mahjong_analysis.riichi_wait_quality import (
    WaitQualityFacts,
    evaluate_wait_quality,
)


@pytest.mark.parametrize(
    ("marker", "expected"),
    [
        ("1m", "2m"),
        ("9p", "1p"),
        ("5sr", "6s"),
        ("E", "S"),
        ("N", "E"),
        ("P", "F"),
        ("C", "P"),
    ],
)
def test_dora_from_marker_cycles(marker: str, expected: str) -> None:
    assert dora_from_marker(marker) == expected


@pytest.mark.parametrize(
    ("raw", "kind", "marker", "expected"),
    [
        ("1m", "1m", "1p", "19"),
        ("8p", "8p", "1m", "28"),
        ("3s", "3s", "1m", "37"),
        ("6m", "6m", "1p", "46"),
        ("5p", "5p", "1m", "5"),
        ("E", "E", "1m", "honor"),
        ("2m", "2m", "1m", "dora"),
        ("S", "S", "E", "dora"),
        ("5sr", "5s", "9m", "dora"),
    ],
)
def test_declaration_classification(
    raw: str, kind: str, marker: str, expected: str
) -> None:
    assert classify_declaration_tile(raw, kind, marker) == expected


def _strength(
    *, suji: bool, ryanmen: bool, copies: int, shape: str
) -> WaitStrengthFacts:
    return evaluate_wait_strength(
        WaitQualityFacts(ryanmen, suji and not ryanmen, copies),
        (DatasetWaitDetail("5m", "standard", shape),),
    )


def test_wait_strength_definitions_are_exclusive() -> None:
    suji = _strength(suji=True, ryanmen=True, copies=4, shape="ryanmen")
    five_plus = _strength(suji=False, ryanmen=False, copies=5, shape="kanchan")
    weak = _strength(suji=False, ryanmen=False, copies=4, shape="tanki")
    penchan = _strength(suji=False, ryanmen=False, copies=4, shape="penchan")

    assert (suji.is_strong_wait, suji.is_weak_wait) == (True, False)
    assert (five_plus.is_strong_wait, five_plus.is_weak_wait) == (True, False)
    assert (weak.is_strong_wait, weak.is_weak_wait) == (False, True)
    assert (penchan.is_strong_wait, penchan.is_weak_wait) == (False, True)


def _expand_hand(hand: str) -> tuple[str, ...]:
    tiles: list[str] = []
    for group in hand.split():
        if group[-1] in "mps":
            tiles.extend(f"{rank}{group[-1]}" for rank in group[:-1])
        else:
            tiles.extend(group)
    return tuple(tiles)


def _strength_from_hand(hand: str) -> tuple[WaitStrengthFacts, tuple[str, ...]]:
    concealed = _expand_hand(hand)
    waits = calculate_hand_waits(concealed)
    details = tuple(
        DatasetWaitDetail(
            detail.wait_tile,
            detail.hand_type,
            detail.wait_shape,
        )
        for detail in waits.wait_details
    )
    quality = evaluate_wait_quality(concealed, (), details)
    return evaluate_wait_strength(quality, details), waits.wait_shapes


@pytest.mark.parametrize(
    ("hand", "expected_shape"),
    [
        ("123m 456m 789p EE 12s", "penchan"),
        ("123m 456m 789p EE 46s", "kanchan"),
    ],
)
def test_actual_four_copy_penchan_and_kanchan_are_weak(
    hand: str, expected_shape: str
) -> None:
    strength, shapes = _strength_from_hand(hand)

    assert shapes == (expected_shape,)
    assert strength.self_excluded_wait_copies == 4
    assert (strength.is_strong_wait, strength.is_weak_wait) == (False, True)


def test_actual_penchan_multiwait_with_five_plus_copies_is_strong() -> None:
    strength, shapes = _strength_from_hand("1112m 456p 789p 123s")

    assert "penchan" in shapes
    assert strength.self_excluded_wait_copies == 7
    assert (strength.is_strong_wait, strength.is_weak_wait) == (True, False)


def test_strong_shape_condition_takes_priority_over_weak_shapes() -> None:
    strength = evaluate_wait_strength(
        WaitQualityFacts(True, False, 4),
        (
            DatasetWaitDetail("2m", "standard", "ryanmen"),
            DatasetWaitDetail("5m", "standard", "penchan"),
        ),
    )

    assert (strength.is_strong_wait, strength.is_weak_wait) == (True, False)


@pytest.mark.parametrize(
    ("hand", "copies", "expected"),
    [
        (
            "1m 9m 1p 9p 1s 9s EE S W N P F",
            4,
            (False, False),
        ),
        (
            "1m 9m 1p 9p 1s 9s E S W N P F C",
            39,
            (True, False),
        ),
    ],
)
def test_actual_kokushi_single_is_other_and_thirteen_way_is_strong(
    hand: str, copies: int, expected: tuple[bool, bool]
) -> None:
    strength, _ = _strength_from_hand(hand)

    assert strength.self_excluded_wait_copies == copies
    assert (strength.is_strong_wait, strength.is_weak_wait) == expected


def test_same_suji_line_shapes_are_nonexclusive_record_flags() -> None:
    shapes = same_suji_line_wait_shapes(
        "5m",
        (
            DatasetWaitDetail("2m", "standard", "kanchan"),
            DatasetWaitDetail("2m", "standard", "tanki"),
            DatasetWaitDetail("8m", "standard", "shanpon"),
            DatasetWaitDetail("2p", "standard", "tanki"),
        ),
    )

    assert shapes == frozenset(("kanchan", "tanki", "shanpon"))
    assert same_suji_line_wait_shapes("E", ()) == frozenset()


def test_same_suji_line_wait_rank_shapes_preserve_target_rank() -> None:
    details = (
        DatasetWaitDetail("2m", "standard", "kanchan"),
        DatasetWaitDetail("2m", "standard", "tanki"),
        DatasetWaitDetail("8m", "standard", "shanpon"),
        DatasetWaitDetail("2p", "standard", "tanki"),
    )

    assert same_suji_line_wait_rank_shapes("5m", details) == {
        2: frozenset(("kanchan", "tanki")),
        8: frozenset(("shanpon",)),
    }
    assert same_suji_line_wait_rank_shapes("E", details) == {}


def test_non_dora_rank_main_table_is_exclusive_and_pairs_are_not() -> None:
    accumulator = DeclarationTileAccumulator()
    quality = WaitQualityFacts(False, False, 4)
    details = (
        DatasetWaitDetail("4m", "standard", "kanchan"),
        DatasetWaitDetail("7m", "standard", "shanpon"),
    )
    accumulator.add("19", "1m", quality, details)
    result = accumulator.to_dict()["non_dora_suited_rank_breakdown"]
    ranks = {row["declaration_rank"]: row for row in result["ranks"]}
    pairs = {
        (row["declaration_rank"], row["wait_rank"]): row
        for row in result["rank_wait_pairs"]["pairs"]
    }

    assert result["record_count"] == 1
    assert result["ranks_exclusive"] is True
    assert ranks[1]["record_count"] == 1
    assert ranks[1]["same_suji_line_wait"]["count"] == 1
    assert pairs[(1, 4)]["record_count"] == 1
    assert pairs[(1, 7)]["record_count"] == 1
    assert result["rank_wait_pairs"]["exclusive"] is False
    assert pairs[(1, 4)]["weak_wait_shape_record_counts"]["counts"]["kanchan"] == 1
    assert pairs[(1, 7)]["weak_wait_shape_record_counts"]["counts"]["shanpon"] == 1
    assert pairs[(1, 4)]["wait_copy_distribution"] == [{"wait_copies": 4, "count": 1}]


def test_non_dora_rank_rows_reconstruct_five_suited_categories() -> None:
    accumulator = DeclarationTileAccumulator()
    category_by_rank = {
        1: "19",
        2: "28",
        3: "37",
        4: "46",
        5: "5",
        6: "46",
        7: "37",
        8: "28",
        9: "19",
    }
    quality = WaitQualityFacts(False, False, 4)
    for rank, category in category_by_rank.items():
        wait_rank = abs(rank - 3) if rank >= 4 else rank + 3
        details = (DatasetWaitDetail(f"{wait_rank}m", "standard", "kanchan"),)
        accumulator.add(category, f"{rank}m", quality, details)
    accumulator.add(
        "dora",
        "5p",
        quality,
        (DatasetWaitDetail("2p", "standard", "kanchan"),),
    )

    result = accumulator.to_dict()
    ranks = {
        row["declaration_rank"]: row
        for row in result["non_dora_suited_rank_breakdown"]["ranks"]
    }
    categories = {row["declaration_category"]: row for row in result["categories"]}
    assert result["non_dora_suited_rank_breakdown"]["record_count"] == 9
    assert all(ranks[rank]["record_count"] == 1 for rank in range(1, 10))
    assert categories["19"]["record_count"] == 2
    assert categories["28"]["record_count"] == 2
    assert categories["37"]["record_count"] == 2
    assert categories["46"]["record_count"] == 2
    assert categories["5"]["record_count"] == 1
    assert categories["dora"]["record_count"] == 1
    assert (
        tuple(
            (row["declaration_rank"], row["wait_rank"])
            for row in result["non_dora_suited_rank_breakdown"]["rank_wait_pairs"][
                "pairs"
            ]
        )
        == RANK_WAIT_PAIRS
    )


def test_weak_same_suji_shape_breakdown_uses_all_record_interpretations() -> None:
    quality = WaitQualityFacts(False, False, 4)
    details = (
        DatasetWaitDetail("2m", "standard", "kanchan"),
        DatasetWaitDetail("3p", "standard", "tanki"),
    )
    accumulator = StrengthAccumulator()
    accumulator.add(quality, details, same_suji_line_wait_shapes("5m", details))

    shape_counts = accumulator.to_dict()[
        "weak_same_suji_line_wait_shape_record_counts"
    ]["counts"]
    assert shape_counts["kanchan"] == 1
    assert shape_counts["tanki"] == 1


def _weak_record(year: int, raw_tile: str, marker: str):
    record = _record(
        year=year,
        waits=(
            ("2m", "standard", "shanpon"),
            ("5m", "standard", "shanpon"),
        ),
        concealed=(
            "2m",
            "2m",
            "5m",
            "5m",
            "1p",
            "2p",
            "3p",
            "7p",
            "8p",
            "9p",
            "4s",
            "5s",
            "6s",
        ),
    )
    kind = "5s" if raw_tile == "5sr" else raw_tile
    river = (
        *record.actor_discards_before_riichi[:-1],
        replace(
            record.actor_discards_before_riichi[-1],
            tile=raw_tile,
            normalized_tile=kind,
        ),
    )
    return replace(
        record,
        dora_marker=marker,
        riichi_declaration_tile=raw_tile,
        riichi_declaration_tile_kind=kind,
        actor_discards_before_riichi=river,
    )


def test_annual_aggregation_partitions_declarations_and_strength(
    tmp_path: Path,
) -> None:
    path = tmp_path / "2025.jsonl.gz"
    records = (
        _weak_record(2025, "1m", "1p"),
        _weak_record(2025, "2m", "1m"),
        _weak_record(2025, "5sr", "9m"),
        _weak_record(2025, "E", "1m"),
    )
    with gzip.open(path, "wb") as stream:
        for record in records:
            stream.write(serialize_dataset_record(record) + b"\n")

    result = aggregate_annual_declaration_tiles(
        AnnualDeclarationTileTask(2025, str(path), len(records))
    )
    document = declaration_tile_results_document((result,))
    categories = {
        row["declaration_category"]: row for row in document["overall"]["categories"]
    }

    assert document["overall"]["record_count"] == 4
    assert categories["19"]["record_count"] == 1
    assert categories["honor"]["record_count"] == 1
    assert categories["dora"]["record_count"] == 2
    rank_breakdown = document["overall"]["non_dora_suited_rank_breakdown"]
    rank_rows = {row["declaration_rank"]: row for row in rank_breakdown["ranks"]}
    assert rank_breakdown["record_count"] == 1
    assert rank_rows[1]["record_count"] == 1
    assert sum(row["record_count"] for row in rank_breakdown["ranks"]) == 1
    assert "non_dora_suited_rank_breakdown" in document["years"][0]["overall"]
    assert "non_dora_suited_rank_breakdown" in document["by_turn"][0]
    assert document["overall"]["all"]["weak_wait"]["count"] == 4
    assert document["overall"]["all"]["strong_wait"]["count"] == 0
    assert document["overall"]["all"]["other_wait"]["count"] == 0
    assert (
        sum(
            document["overall"]["all"][name]["count"]
            for name in ("strong_wait", "weak_wait", "other_wait")
        )
        == document["overall"]["record_count"]
    )
    assert document["overall"]["all"]["wait_shape_record_counts"]["shanpon"] == 4
    assert document["overall"]["all"]["same_suji_line_wait"] == {
        "count": 1,
        "denominator": 4,
        "rate": 0.25,
    }
    assert document["overall"]["all"]["weak_and_same_suji_line"]["count"] == 1
    assert document["overall"]["all"]["weak_and_not_same_suji_line"]["count"] == 3
    assert (
        document["overall"]["all"]["weak_wait_rate_within_same_suji_line"]["rate"]
        == 1.0
    )
    assert (
        document["overall"]["all"]["weak_wait_rate_excluding_same_suji_line"]["rate"]
        == 1.0
    )
    assert document["overall"]["all"][
        "weak_same_suji_line_wait_shape_record_counts"
    ] == {
        "exclusive": False,
        "counts": {
            "ryanmen": 0,
            "kanchan": 0,
            "penchan": 0,
            "shanpon": 1,
            "tanki": 0,
            "kokushi_single": 0,
            "kokushi_13men": 0,
        },
    }
    audit = {
        row["declaration_category"]: row["samples"]
        for row in document["audit"]["categories"]
    }
    assert len(audit["dora"]) == 2
    assert audit["dora"][0]["relative_source_path"] == "2025/game.mjson"
    assert audit["dora"][0]["declaration_category"] == "dora"


def test_annual_count_mismatch_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "2025.jsonl.gz"
    with gzip.open(path, "wb") as stream:
        stream.write(serialize_dataset_record(_weak_record(2025, "1m", "1p")) + b"\n")
    with pytest.raises(ValueError, match="annual record count"):
        aggregate_annual_declaration_tiles(
            AnnualDeclarationTileTask(2025, str(path), 2)
        )


def test_serial_and_parallel_results_match(tmp_path: Path) -> None:
    tasks = []
    for year in (2024, 2025):
        path = tmp_path / f"{year}.jsonl.gz"
        with gzip.open(path, "wb") as stream:
            stream.write(
                serialize_dataset_record(_weak_record(year, "5m", "1p")) + b"\n"
            )
        tasks.append(AnnualDeclarationTileTask(year, str(path), 1))

    serial = run_declaration_tile_tasks(tuple(tasks), workers=1)
    parallel = run_declaration_tile_tasks(tuple(tasks), workers=2)

    assert serial == parallel
    assert declaration_tile_results_document(
        serial
    ) == declaration_tile_results_document(tuple(reversed(parallel)))


def test_strong_wait_comparison_covers_saved_slices(tmp_path: Path) -> None:
    path = tmp_path / "2025.jsonl.gz"
    with gzip.open(path, "wb") as stream:
        stream.write(serialize_dataset_record(_weak_record(2025, "1m", "1p")) + b"\n")
    result = aggregate_annual_declaration_tiles(
        AnnualDeclarationTileTask(2025, str(path), 1)
    )
    document = declaration_tile_results_document((result,))
    previous = copy.deepcopy(document)

    check_strong_results_unchanged(document, previous)
    previous["years"][0]["by_turn"][0]["categories"][0]["strong_wait"]["count"] += 1
    with pytest.raises(ValueError, match="strong wait changed"):
        check_strong_results_unchanged(document, previous)
