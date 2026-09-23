import gzip
from dataclasses import replace
from pathlib import Path

import pytest
from test_riichi_wait_summary import _record

from mahjong_analysis.hand_waits import calculate_hand_waits
from mahjong_analysis.riichi_wait_dataset import (
    DatasetActorDiscard,
    RiichiWaitDatasetRecord,
    serialize_dataset_record,
)
from mahjong_analysis.toitsu_drop_analysis import (
    AnnualToitsuDropTask,
    ToitsuDropAccumulator,
    aggregate_annual_toitsu_drop,
    detect_toitsu_drop,
    run_toitsu_drop_tasks,
    select_audit_samples,
    toitsu_drop_results_document,
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


def make_discard(
    number: int,
    tile: str,
    *,
    normalized: str | None = None,
    tsumogiri: bool = False,
    declaration: bool = False,
    event_index: int | None = None,
    call_event_index: int | None = None,
) -> DatasetActorDiscard:
    return DatasetActorDiscard(
        discard_number=number,
        tile=tile,
        normalized_tile=normalized or tile.removesuffix("r"),
        tsumogiri=tsumogiri,
        event_index=event_index if event_index is not None else number * 2,
        is_riichi_declaration=declaration,
        was_called=call_event_index is not None,
        call_type="pon" if call_event_index is not None else None,
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


def test_tedashi_then_tedashi_is_a_and_declaration_distance_zero() -> None:
    record = make_record((("5m", False), ("5m", False)))

    facts = detect_toitsu_drop(record.actor_discards_before_riichi)

    assert facts.classification == "a_only"
    assert [pair.pair_type for pair in facts.pairs] == ["A"]
    assert facts.representative_pair == facts.pairs[0]
    assert facts.distance == 0


def test_tsumogiri_then_tedashi_is_b_and_red_is_normalized() -> None:
    record = make_record((("5mr", True), ("5m", False)))

    facts = detect_toitsu_drop(record.actor_discards_before_riichi)

    assert facts.classification == "b_only"
    assert facts.pairs[0].normalized_tile == "5m"
    assert facts.pairs[0].first_raw_tile == "5mr"
    assert facts.pairs[0].second_raw_tile == "5m"


def test_second_tsumogiri_is_not_a_pattern() -> None:
    record = make_record((("5m", False), ("5m", True)))

    facts = detect_toitsu_drop(record.actor_discards_before_riichi)

    assert facts.classification == "none"
    assert facts.pairs == ()
    assert facts.distance is None


def test_same_tile_separated_by_own_discard_is_not_adjacent() -> None:
    record = make_record((("5m", False), ("E", False), ("5m", False)))

    facts = detect_toitsu_drop(record.actor_discards_before_riichi)

    assert facts.classification == "none"


def test_three_identical_discards_produce_both_and_latest_representative() -> None:
    record = make_record((("E", True), ("E", False), ("E", False)))

    facts = detect_toitsu_drop(record.actor_discards_before_riichi)

    assert facts.classification == "both"
    assert [pair.pair_type for pair in facts.pairs] == ["B", "A"]
    assert [pair.second_discard_number for pair in facts.pairs] == [2, 3]
    assert facts.representative_pair == facts.pairs[-1]
    assert facts.distance == 0


def test_last_completed_pair_controls_distance() -> None:
    record = make_record(
        (("2p", False), ("2p", False), ("7s", False), ("7s", False), ("E", False))
    )

    facts = detect_toitsu_drop(record.actor_discards_before_riichi)

    assert [pair.second_discard_number for pair in facts.pairs] == [2, 4]
    assert facts.distance == 1


def test_call_metadata_does_not_change_pattern_classification() -> None:
    plain = make_record((("5m", False), ("5m", False)))
    called = make_record(
        (("5m", False), ("5m", False)), call_first_before_acceptance=True
    )

    assert detect_toitsu_drop(plain.actor_discards_before_riichi) == detect_toitsu_drop(
        called.actor_discards_before_riichi
    )


def test_invalid_river_sequence_is_rejected() -> None:
    record = make_record((("5m", False), ("5m", False)))
    first, second = record.actor_discards_before_riichi
    bad = (first, replace(second, discard_number=3))

    with pytest.raises(ValueError, match="consecutive"):
        detect_toitsu_drop(bad)


def test_accumulator_partitions_each_record_once() -> None:
    records = (
        make_record((("1m", False),)),
        make_record((("5m", False), ("5m", False))),
        make_record((("5m", True), ("5m", False))),
        make_record((("E", True), ("E", False), ("E", False))),
    )
    accumulator = ToitsuDropAccumulator()
    from mahjong_analysis.riichi_wait_quality import evaluate_wait_quality

    for record in records:
        accumulator.add(
            detect_toitsu_drop(record.actor_discards_before_riichi),
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
        "a_only": 1,
        "b_only": 1,
        "both": 1,
    }
    assert result["pattern_present"] == {
        "count": 3,
        "denominator": 4,
        "rate": 0.75,
    }
    assert sum(result["classification_counts"].values()) == 4
    assert (
        sum(row["record_count"] for row in result["matching_pair_count_distribution"])
        == 4
    )


def annual_task(tmp_path: Path, year: int) -> AnnualToitsuDropTask:
    records = (
        make_record(
            (("5m", False), ("5m", False)),
            year=year,
            source_suffix="a.mjson",
        ),
        make_record(
            (("1p", False), ("9p", False), ("E", False)),
            year=year,
            hand="2255m 123p 789p 456s",
            source_suffix="b.mjson",
        ),
    )
    path = tmp_path / f"{year}.jsonl.gz"
    with gzip.open(path, "wb") as stream:
        for record in records:
            stream.write(serialize_dataset_record(record) + b"\n")
    return AnnualToitsuDropTask(year, str(path), len(records))


def test_annual_streaming_counts_turns_distances_and_quality(tmp_path: Path) -> None:
    result = aggregate_annual_toitsu_drop(annual_task(tmp_path, 2025))
    document = toitsu_drop_results_document((result,))

    assert document["overall"]["record_count"] == 2
    assert document["overall"]["pattern_present"]["count"] == 1
    assert document["overall"]["groups"]["pattern_present"]["contains_suji"] == {
        "count": 1,
        "denominator": 1,
        "rate": 1.0,
    }
    assert document["overall"]["groups"]["no_pattern"]["good_wait"] == {
        "count": 0,
        "denominator": 1,
        "rate": 0.0,
    }
    assert [row["riichi_discard_number"] for row in document["by_turn"]] == [2, 3]
    assert document["by_distance"][0]["distance"] == 0
    assert document["by_distance"][0]["record_count"] == 1


def test_serial_parallel_and_output_order_match(tmp_path: Path) -> None:
    tasks = (annual_task(tmp_path, 2025), annual_task(tmp_path, 2009))

    serial = run_toitsu_drop_tasks(tasks, workers=1)
    parallel = run_toitsu_drop_tasks(tasks, workers=2)

    assert serial == parallel
    assert [result.year for result in parallel] == [2025, 2009]
    assert toitsu_drop_results_document(serial) == toitsu_drop_results_document(
        tuple(reversed(parallel))
    )
    assert [row["year"] for row in toitsu_drop_results_document(serial)["years"]] == [
        2009,
        2025,
    ]


def test_audit_selection_is_deduplicated_sorted_and_tracks_preacceptance_call(
    tmp_path: Path,
) -> None:
    task = annual_task(tmp_path, 2025)
    result = aggregate_annual_toitsu_drop(task)
    # Add one B record with a pre-acceptance called discard to the source.
    called = make_record(
        (("5mr", True), ("5m", False)),
        source_suffix="0-called.mjson",
        call_first_before_acceptance=True,
    )
    called_path = tmp_path / "called.jsonl.gz"
    with gzip.open(called_path, "wb") as stream:
        stream.write(serialize_dataset_record(called) + b"\n")
    called_result = aggregate_annual_toitsu_drop(
        AnnualToitsuDropTask(2025, str(called_path), 1)
    )

    audit = select_audit_samples((result, called_result))
    keys = [
        (row["relative_source_path"], row["start_kyoku_line"], row["reach_line"])
        for row in audit["samples"]
    ]
    assert keys == sorted(set(keys))
    selected = next(
        row
        for row in audit["samples"]
        if "called_before_reach_accepted" in row["selection_reasons"]
    )
    assert selected["called_before_acceptance_discard_numbers"] == [1]
    assert "red_five_pair" in selected["selection_reasons"]
    assert audit["selected_count"] <= 24


@pytest.mark.parametrize("workers", [1, 2])
def test_bad_annual_count_propagates(tmp_path: Path, workers: int) -> None:
    task = annual_task(tmp_path, 2025)
    wrong = AnnualToitsuDropTask(task.year, task.path, 3)
    with pytest.raises(ValueError, match="annual record count"):
        run_toitsu_drop_tasks((wrong,), workers=workers)


def test_malformed_input_is_not_skipped(tmp_path: Path) -> None:
    task = annual_task(tmp_path, 2025)
    with gzip.open(task.path, "ab") as stream:
        stream.write(b"{broken\n")
    with pytest.raises(ValueError, match="invalid dataset JSON"):
        aggregate_annual_toitsu_drop(task)


def test_wrong_annual_year_is_rejected(tmp_path: Path) -> None:
    task = annual_task(tmp_path, 2025)
    with pytest.raises(ValueError, match="unexpected record year"):
        aggregate_annual_toitsu_drop(AnnualToitsuDropTask(2024, task.path, 2))
