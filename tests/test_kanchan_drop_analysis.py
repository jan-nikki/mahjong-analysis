import gzip
from dataclasses import replace
from itertools import product
from pathlib import Path

import pytest
from test_riichi_wait_summary import _record

from mahjong_analysis.hand_waits import calculate_hand_waits
from mahjong_analysis.kanchan_drop_analysis import (
    AnnualKanchanDropTask,
    KanchanDropAccumulator,
    aggregate_annual_kanchan_drop,
    detect_kanchan_drop,
    kanchan_drop_results_document,
    run_kanchan_drop_tasks,
    select_audit_samples,
)
from mahjong_analysis.riichi_wait_dataset import (
    DatasetActorDiscard,
    RiichiWaitDatasetRecord,
    serialize_dataset_record,
)
from mahjong_analysis.riichi_wait_quality import evaluate_wait_quality
from mahjong_analysis.tiles import TILE_KINDS


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


def make_discard(
    number: int,
    tile: str,
    *,
    tsumogiri: bool = False,
    declaration: bool = False,
    event_index: int | None = None,
    call_event_index: int | None = None,
) -> DatasetActorDiscard:
    return DatasetActorDiscard(
        discard_number=number,
        tile=tile,
        normalized_tile=tile.removesuffix("r"),
        tsumogiri=tsumogiri,
        event_index=event_index if event_index is not None else number * 2,
        is_riichi_declaration=declaration,
        was_called=call_event_index is not None,
        call_type="chi" if call_event_index is not None else None,
        called_by_actor=1 if call_event_index is not None else None,
        call_event_index=call_event_index,
    )


def make_record(
    river_spec: tuple[tuple[str, bool], ...],
    *,
    year: int = 2025,
    hand: str = "2345m 123p 789p 456s",
    source_suffix: str = "game.mjson",
    call_first_before_acceptance: bool = False,
) -> RiichiWaitDatasetRecord:
    concealed = expand_hand(hand)
    waits = calculate_hand_waits(concealed)
    turn = len(river_spec)
    base = _record(
        year=year,
        turn=turn,
        concealed=concealed,
        waits=tuple(
            (detail.wait_tile, detail.hand_type, detail.wait_shape)
            for detail in waits.wait_details
        ),
        source_suffix=source_suffix,
    )
    river = []
    for index, (tile, tsumogiri) in enumerate(river_spec, 1):
        declaration = index == turn
        event_index = base.declaration_dahai_event_index if declaration else index * 2
        river.append(
            make_discard(
                index,
                tile,
                tsumogiri=tsumogiri,
                declaration=declaration,
                event_index=event_index,
                call_event_index=(
                    3 if call_first_before_acceptance and index == 1 else None
                ),
            )
        )
    declaration = river[-1]
    return replace(
        base,
        riichi_declaration_tile=declaration.tile,
        riichi_declaration_tile_kind=declaration.normalized_tile,
        actor_discards_before_riichi=tuple(river),
    )


@pytest.mark.parametrize(
    ("first", "second", "shape", "direction", "order"),
    [
        ("3m", "1m", "13", "high_to_low", "inner_first"),
        ("4m", "2m", "24", "high_to_low", "inner_first"),
        ("5m", "3m", "35", "high_to_low", "inner_first"),
        ("4m", "6m", "46", "low_to_high", "symmetric"),
        ("6m", "4m", "46", "high_to_low", "symmetric"),
        ("5m", "7m", "57", "low_to_high", "inner_first"),
        ("6m", "8m", "68", "low_to_high", "inner_first"),
        ("7m", "9m", "79", "low_to_high", "inner_first"),
        ("1m", "3m", "13", "low_to_high", "outer_first"),
        ("2m", "4m", "24", "low_to_high", "outer_first"),
        ("3m", "5m", "35", "low_to_high", "outer_first"),
        ("7m", "5m", "57", "high_to_low", "outer_first"),
        ("8m", "6m", "68", "high_to_low", "outer_first"),
        ("9m", "7m", "79", "high_to_low", "outer_first"),
    ],
)
def test_shape_direction_and_centrality(
    first: str,
    second: str,
    shape: str,
    direction: str,
    order: str,
) -> None:
    record = make_record(((first, False), (second, False)))

    facts = detect_kanchan_drop(record.actor_discards_before_riichi)

    assert facts.classification == "a_only"
    assert facts.distance == 0
    assert facts.candidates[0].shape == shape
    assert facts.candidates[0].direction == direction
    assert facts.candidates[0].order_class == order


def test_tsumogiri_first_is_equal_b_pattern_and_red_is_normalized() -> None:
    record = make_record((("5mr", True), ("7m", False)))

    facts = detect_kanchan_drop(record.actor_discards_before_riichi)

    assert facts.classification == "b_only"
    assert facts.has_pattern
    assert facts.candidates[0].candidate_type == "B"
    assert facts.candidates[0].first_normalized_tile == "5m"
    assert facts.candidates[0].order_class == "inner_first"


@pytest.mark.parametrize(
    "river",
    [
        (("2m", False), ("4m", True)),
        (("2m", False), ("3m", False)),
        (("2m", False), ("5m", False)),
        (("2m", False), ("4p", False)),
        (("E", False), ("W", False)),
        (("2m", False), ("E", False)),
    ],
)
def test_non_matching_pairs_are_rejected(river: tuple[tuple[str, bool], ...]) -> None:
    assert detect_kanchan_drop(
        make_record(river).actor_discards_before_riichi
    ).classification == ("none")


def test_same_shape_separated_by_own_discard_is_not_adjacent() -> None:
    record = make_record((("2m", False), ("E", False), ("4m", False)))
    assert detect_kanchan_drop(record.actor_discards_before_riichi).classification == (
        "none"
    )


def test_overlapping_candidates_use_latest_representative() -> None:
    record = make_record((("2m", True), ("4m", False), ("2m", False)))

    facts = detect_kanchan_drop(record.actor_discards_before_riichi)

    assert facts.classification == "both"
    assert [value.candidate_type for value in facts.candidates] == ["B", "A"]
    assert [value.order_class for value in facts.candidates] == [
        "outer_first",
        "inner_first",
    ]
    assert facts.representative_candidate == facts.candidates[-1]
    assert facts.distance == 0


def test_call_metadata_and_intervening_events_do_not_change_classification() -> None:
    plain = make_record((("4m", False), ("2m", False)))
    called = make_record(
        (("4m", False), ("2m", False)), call_first_before_acceptance=True
    )
    first, second = plain.actor_discards_before_riichi
    with_event_gap = (first, replace(second, event_index=second.event_index + 8))

    assert detect_kanchan_drop(
        plain.actor_discards_before_riichi
    ) == detect_kanchan_drop(called.actor_discards_before_riichi)
    assert detect_kanchan_drop(with_event_gap).classification == "a_only"


def test_invalid_river_sequence_is_rejected() -> None:
    record = make_record((("4m", False), ("2m", False)))
    first, second = record.actor_discards_before_riichi
    with pytest.raises(ValueError, match="consecutive"):
        detect_kanchan_drop((first, replace(second, discard_number=3)))


def _oracle_matches(first: str, second: str, second_tsumogiri: bool) -> bool:
    if second_tsumogiri or first[-1] not in "mps" or second[-1] not in "mps":
        return False
    return first[-1] == second[-1] and abs(int(first[0]) - int(second[0])) == 2


def test_exhaustive_tile_pair_oracle() -> None:
    for first, second, first_tsumogiri, second_tsumogiri in product(
        TILE_KINDS, TILE_KINDS, (False, True), (False, True)
    ):
        record = make_record(((first, first_tsumogiri), (second, second_tsumogiri)))
        facts = detect_kanchan_drop(record.actor_discards_before_riichi)
        assert facts.has_pattern == _oracle_matches(first, second, second_tsumogiri), (
            first,
            second,
            first_tsumogiri,
            second_tsumogiri,
        )


def test_accumulator_partitions_orders_and_shapes_once_per_record() -> None:
    records = (
        make_record((("1m", False),)),
        make_record((("4m", False), ("2m", False))),
        make_record((("2m", True), ("4m", False))),
        make_record((("4m", False), ("6m", False))),
    )
    accumulator = KanchanDropAccumulator()
    for record in records:
        accumulator.add(
            detect_kanchan_drop(record.actor_discards_before_riichi),
            evaluate_wait_quality(
                record.concealed_tiles_after_discard,
                record.fixed_melds,
                record.wait_details,
            ),
        )

    result = accumulator.to_dict()
    assert result["record_count"] == 4
    assert result["classification_counts"] == {
        "none": 1,
        "a_only": 2,
        "b_only": 1,
        "both": 0,
    }
    assert result["representative_order_groups"]["inner_first"]["record_count"] == 1
    assert result["representative_order_groups"]["outer_first"]["record_count"] == 1
    assert result["representative_order_groups"]["symmetric"]["record_count"] == 1
    shape24 = next(
        row for row in result["representative_shape_direction"] if row["shape"] == "24"
    )
    assert shape24["high_to_low"]["record_count"] == 1
    assert shape24["low_to_high"]["record_count"] == 1


def annual_task(tmp_path: Path, year: int) -> AnnualKanchanDropTask:
    records = (
        make_record(
            (("4m", False), ("2m", False)),
            year=year,
            source_suffix="a.mjson",
        ),
        make_record(
            (("1p", False), ("9p", False), ("E", False)),
            year=year,
            hand="2255m 123p 789p 456s",
            source_suffix="b.mjson",
        ),
        make_record(
            (("5m", False), ("5m", False)),
            year=year,
            source_suffix="c.mjson",
        ),
    )
    path = tmp_path / f"{year}.jsonl.gz"
    with gzip.open(path, "wb") as stream:
        for record in records:
            stream.write(serialize_dataset_record(record) + b"\n")
    return AnnualKanchanDropTask(year, str(path), len(records))


def test_annual_streaming_counts_turns_distances_quality_and_toitsu(
    tmp_path: Path,
) -> None:
    result = aggregate_annual_kanchan_drop(annual_task(tmp_path, 2025))
    document = kanchan_drop_results_document((result,))

    assert document["overall"]["record_count"] == 3
    assert document["overall"]["pattern_present"]["count"] == 1
    assert (
        document["overall"]["representative_order_groups"]["inner_first"][
            "record_count"
        ]
        == 1
    )
    assert [row["riichi_discard_number"] for row in document["by_turn"]] == [2, 3]
    assert document["by_distance"][0]["record_count"] == 1
    assert document["by_toitsu_pattern"]["present"]["record_count"] == 1
    assert document["by_toitsu_pattern"]["absent"]["record_count"] == 2


def test_serial_parallel_and_output_order_match(tmp_path: Path) -> None:
    tasks = (annual_task(tmp_path, 2025), annual_task(tmp_path, 2009))
    serial = run_kanchan_drop_tasks(tasks, workers=1)
    parallel = run_kanchan_drop_tasks(tasks, workers=2)

    assert serial == parallel
    assert [result.year for result in parallel] == [2025, 2009]
    assert kanchan_drop_results_document(serial) == kanchan_drop_results_document(
        tuple(reversed(parallel))
    )


def test_audit_selection_covers_order_and_called_discard(tmp_path: Path) -> None:
    base = aggregate_annual_kanchan_drop(annual_task(tmp_path, 2025))
    called = make_record(
        (("5mr", True), ("7m", False)),
        source_suffix="0-called.mjson",
        call_first_before_acceptance=True,
    )
    path = tmp_path / "called.jsonl.gz"
    with gzip.open(path, "wb") as stream:
        stream.write(serialize_dataset_record(called) + b"\n")
    called_result = aggregate_annual_kanchan_drop(
        AnnualKanchanDropTask(2025, str(path), 1)
    )

    audit = select_audit_samples((base, called_result))
    selected = next(
        row
        for row in audit["samples"]
        if "called_before_reach_accepted" in row["selection_reasons"]
    )
    assert selected["called_before_acceptance_discard_numbers"] == [1]
    assert "red_five_candidate" in selected["selection_reasons"]
    assert "inner_first" in selected["selection_reasons"]


@pytest.mark.parametrize("workers", [1, 2])
def test_bad_annual_count_propagates(tmp_path: Path, workers: int) -> None:
    task = annual_task(tmp_path, 2025)
    wrong = AnnualKanchanDropTask(task.year, task.path, 4)
    with pytest.raises(ValueError, match="annual record count"):
        run_kanchan_drop_tasks((wrong,), workers=workers)


def test_malformed_input_is_not_skipped(tmp_path: Path) -> None:
    task = annual_task(tmp_path, 2025)
    with gzip.open(task.path, "ab") as stream:
        stream.write(b"{broken\n")
    with pytest.raises(ValueError, match="invalid dataset JSON"):
        aggregate_annual_kanchan_drop(task)


def test_wrong_annual_year_is_rejected(tmp_path: Path) -> None:
    task = annual_task(tmp_path, 2025)
    with pytest.raises(ValueError, match="unexpected record year"):
        aggregate_annual_kanchan_drop(AnnualKanchanDropTask(2024, task.path, 3))
