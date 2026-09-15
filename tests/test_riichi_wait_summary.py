import gzip
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

import mahjong_analysis.riichi_wait_summary as summary_module
from mahjong_analysis.hand_waits import HAND_TYPE_ORDER, WAIT_SHAPE_ORDER
from mahjong_analysis.riichi_wait_dataset import (
    DATASET_NAME,
    DatasetActorDiscard,
    DatasetFixedMeld,
    DatasetWaitDetail,
    RiichiWaitDatasetRecord,
    dataset_record_to_dict,
    serialize_dataset_record,
)
from mahjong_analysis.riichi_wait_summary import (
    ANALYSIS_NAME,
    CANONICAL_RECORD_COUNT,
    CANONICAL_SOURCE_REPOSITORY,
    COMPLETION_MARKER_FILENAME,
    OUTPUT_FILENAMES,
    PUBLICATION_GENERATIONS_DIRECTORY,
    PUBLICATION_POINTER_FILENAME,
    AnalysisGitMetadata,
    SummaryAnalysis,
    SummaryProgress,
    aggregate_riichi_wait_records,
    build_summary_documents,
    collect_analysis_git_metadata,
    load_published_summary_paths,
    summarize_riichi_wait_dataset,
    write_summary_outputs,
)
from mahjong_analysis.tiles import TILE_KINDS

ORPHANS = (
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


def _record(
    *,
    year: int = 2025,
    turn: int = 1,
    waits: tuple[tuple[str, str, str], ...],
    concealed: tuple[str, ...] = (),
    fixed_melds: tuple[DatasetFixedMeld, ...] = (),
    source_suffix: str = "game.mjson",
) -> RiichiWaitDatasetRecord:
    tile_index = {tile: index for index, tile in enumerate(TILE_KINDS)}
    hand_index = {value: index for index, value in enumerate(HAND_TYPE_ORDER)}
    shape_index = {value: index for index, value in enumerate(WAIT_SHAPE_ORDER)}
    details = tuple(
        DatasetWaitDetail(*value)
        for value in sorted(
            waits,
            key=lambda value: (
                tile_index[value[0]],
                hand_index[value[1]],
                shape_index[value[2]],
            ),
        )
    )
    wait_tiles = tuple(
        tile for tile in TILE_KINDS if any(detail.tile == tile for detail in details)
    )
    wait_shapes = tuple(
        shape
        for shape in WAIT_SHAPE_ORDER
        if any(detail.wait_shape == shape for detail in details)
    )
    reach_index = turn + 10
    declaration_index = reach_index + 1
    accepted_index = declaration_index + 1
    river = tuple(
        DatasetActorDiscard(
            discard_number=number,
            tile="9s",
            normalized_tile="9s",
            tsumogiri=False,
            event_index=number if number < turn else declaration_index,
            is_riichi_declaration=number == turn,
            was_called=False,
            call_type=None,
            called_by_actor=None,
            call_event_index=None,
        )
        for number in range(1, turn + 1)
    )
    pure = (
        bool(details)
        and len(wait_tiles) == 2
        and all(
            detail.hand_type == "standard" and detail.wait_shape == "ryanmen"
            for detail in details
        )
    )
    return RiichiWaitDatasetRecord(
        year=year,
        relative_source_path=f"{year}/{source_suffix}",
        start_kyoku_line=100,
        reach_line=100 + reach_index,
        declaration_dahai_line=100 + declaration_index,
        reach_accepted_line=100 + accepted_index,
        bakaze="E",
        kyoku=1,
        honba=0,
        oya=0,
        scores_at_start=(25000, 25000, 25000, 25000),
        dora_marker="1m",
        actor=0,
        riichi_discard_number=turn,
        riichi_declaration_tile="9s",
        riichi_declaration_tile_kind="9s",
        reach_event_index=reach_index,
        declaration_dahai_event_index=declaration_index,
        reach_accepted_event_index=accepted_index,
        concealed_tiles_after_discard=concealed,
        fixed_melds=fixed_melds,
        actor_discards_before_riichi=river,
        wait_tiles=wait_tiles,
        wait_tile_count=len(wait_tiles),
        wait_details=details,
        wait_shapes=wait_shapes,
        contains_ryanmen=any(detail.wait_shape == "ryanmen" for detail in details),
        is_pure_ryanmen=pure,
        is_multiwait=len(wait_tiles) >= 3,
    )


def _summary_r17(*, year: int = 2025, turn: int = 1) -> RiichiWaitDatasetRecord:
    return _record(
        year=year,
        turn=turn,
        waits=(("3s", "standard", "penchan"),),
        concealed=("4p", "4p", "5p", "6p", "7p", "1s", "2s", "4s", "5s", "6s"),
        fixed_melds=(DatasetFixedMeld("ankan", ("3s", "3s", "3s", "3s")),),
        source_suffix="summary-r17.mjson",
    )


def _summary_r18(*, year: int = 2025, turn: int = 1) -> RiichiWaitDatasetRecord:
    return _record(
        year=year,
        turn=turn,
        waits=(
            ("2m", "standard", "ryanmen"),
            ("5m", "standard", "ryanmen"),
            ("8m", "standard", "ryanmen"),
        ),
        concealed=("3m", "4m", "5m", "6m", "7m", "7p", "8p", "9p", "E", "E"),
        fixed_melds=(DatasetFixedMeld("ankan", ("2m", "2m", "2m", "2m")),),
        source_suffix="summary-r18.mjson",
    )


def _summary_r19(*, year: int = 2025, turn: int = 1) -> RiichiWaitDatasetRecord:
    return _record(
        year=year,
        turn=turn,
        waits=(
            ("2m", "standard", "ryanmen"),
            ("5m", "standard", "ryanmen"),
        ),
        concealed=("3m", "4m", "1p", "2p", "3p", "7p", "8p", "9p", "E", "E"),
        fixed_melds=(DatasetFixedMeld("ankan", ("5m", "5m", "5m", "5mr")),),
        source_suffix="summary-r19.mjson",
    )


def _metadata(
    *,
    years: tuple[int, ...] = (2025,),
    record_count: int = 1,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "analysis_name": ANALYSIS_NAME,
        "observation_unit": "established_riichi_record",
        "years": list(years),
        "scope": {"rule_code": "00a9", "aka_flag": True, "bakaze": "E"},
        "input_dataset": {
            "dataset_name": DATASET_NAME,
            "schema_version": 1,
            "manifest_path": "data/processed/riichi-waits-v1/manifest.json",
            "manifest_sha256": "a" * 64,
            "generator_git_commit": "dataset-commit",
            "source_repository": "NikkeTryHard/tenhou-to-mjai",
            "source_release_tag": "v2.0.0",
            "totals": {"output_records": record_count},
        },
        "analysis_generator": {"git_commit": "analysis-commit", "worktree_clean": True},
        "formal_wait_semantics": {
            "name": "tenhou_formal_wait",
            "fixed_meld_self_owned_four_waits_retained": True,
            "concealed_four_candidate_excluded": True,
        },
        "ryanmen_metrics": {
            "headline": "is_pure_ryanmen",
            "headline_definition": {
                "wait_tile_count": 2,
                "all_hand_types": "standard",
                "all_wait_shapes": "ryanmen",
                "empty_details_is_pure_ryanmen": False,
            },
            "supplementary": "contains_ryanmen",
        },
    }


def _overall(records: tuple[RiichiWaitDatasetRecord, ...]) -> dict[str, object]:
    aggregation = aggregate_riichi_wait_records(records)
    analysis = SummaryAnalysis(
        aggregation,
        _metadata(
            years=tuple(sorted({r.year for r in records})),
            record_count=len(records),
        ),
    )
    return json.loads(build_summary_documents(analysis)["overall"])["result"]


def _metric_by_value(
    values: list[dict[str, object]], value: object
) -> dict[str, object]:
    return next(item for item in values if item["value"] == value)


def _set_category(
    values: list[dict[str, object]], members: list[str]
) -> dict[str, object]:
    return next(item for item in values if item["members"] == members)


def _formal_shapes(value: dict[str, object]) -> dict[str, object]:
    return value["wait_shape_record_level"]  # type: ignore[return-value]


def _formal_hand_types(value: dict[str, object]) -> dict[str, object]:
    return value["hand_type_record_level"]  # type: ignore[return-value]


def _adjusted_shapes(value: dict[str, object]) -> dict[str, object]:
    return value["adjusted_wait_shape_record_level"]  # type: ignore[return-value]


def _adjusted_hand_types(value: dict[str, object]) -> dict[str, object]:
    return value["adjusted_hand_type_record_level"]  # type: ignore[return-value]


def _expected_metric(count: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "count": count,
        "denominator": denominator,
        "rate": count / denominator if denominator else None,
    }


def _expected_membership(
    order: tuple[str, ...],
    counts: dict[str, int],
    denominator: int,
) -> list[dict[str, object]]:
    return [
        {"value": value, **_expected_metric(counts.get(value, 0), denominator)}
        for value in order
    ]


def _expected_set_distribution(
    counts: tuple[tuple[tuple[str, ...], int], ...],
    denominator: int,
) -> list[dict[str, object]]:
    return [
        {"members": list(members), **_expected_metric(count, denominator)}
        for members, count in counts
    ]


def _expected_record_level(
    *,
    denominator: int,
    wait_counts: tuple[tuple[int, int], ...],
    pure: int,
    contains: int,
    multi: int,
    shape_sets: tuple[tuple[tuple[str, ...], int], ...],
    hand_type_sets: tuple[tuple[tuple[str, ...], int], ...],
) -> dict[str, object]:
    shape_membership: dict[str, int] = {}
    hand_type_membership: dict[str, int] = {}
    for members, count in shape_sets:
        for member in members:
            shape_membership[member] = shape_membership.get(member, 0) + count
    for members, count in hand_type_sets:
        for member in members:
            hand_type_membership[member] = hand_type_membership.get(member, 0) + count
    multiple_shapes = sum(count for members, count in shape_sets if len(members) > 1)
    multiple_hand_types = sum(
        count for members, count in hand_type_sets if len(members) > 1
    )
    return {
        "record_count": denominator,
        "pure_ryanmen": _expected_metric(pure, denominator),
        "contains_ryanmen": _expected_metric(contains, denominator),
        "multiwait": _expected_metric(multi, denominator),
        "wait_tile_count_distribution": [
            {"value": value, **_expected_metric(count, denominator)}
            for value, count in wait_counts
        ],
        "wait_shape_record_level": {
            "wait_shape_membership": _expected_membership(
                WAIT_SHAPE_ORDER, shape_membership, denominator
            ),
            "wait_shape_set_distribution": _expected_set_distribution(
                shape_sets, denominator
            ),
            "records_with_multiple_wait_shapes": _expected_metric(
                multiple_shapes, denominator
            ),
        },
        "hand_type_record_level": {
            "hand_type_membership": _expected_membership(
                HAND_TYPE_ORDER, hand_type_membership, denominator
            ),
            "hand_type_set_distribution": _expected_set_distribution(
                hand_type_sets, denominator
            ),
            "records_with_multiple_hand_types": _expected_metric(
                multiple_hand_types, denominator
            ),
        },
    }


def _expected_adjusted(
    *,
    zero_wait: int,
    **record_level: object,
) -> dict[str, object]:
    raw = _expected_record_level(**record_level)  # type: ignore[arg-type]
    shape_values = raw["wait_shape_record_level"]
    hand_type_values = raw["hand_type_record_level"]
    return {
        "record_count": raw["record_count"],
        "adjusted_zero_wait_count": zero_wait,
        "adjusted_is_pure_ryanmen": raw["pure_ryanmen"],
        "adjusted_contains_ryanmen": raw["contains_ryanmen"],
        "adjusted_is_multiwait": raw["multiwait"],
        "adjusted_wait_tile_count_distribution": raw["wait_tile_count_distribution"],
        "adjusted_wait_shape_record_level": {
            "adjusted_wait_shape_membership": shape_values["wait_shape_membership"],
            "adjusted_wait_shape_set_distribution": shape_values[
                "wait_shape_set_distribution"
            ],
            "adjusted_records_with_multiple_wait_shapes": shape_values[
                "records_with_multiple_wait_shapes"
            ],
        },
        "adjusted_hand_type_record_level": {
            "adjusted_hand_type_membership": hand_type_values["hand_type_membership"],
            "adjusted_hand_type_set_distribution": hand_type_values[
                "hand_type_set_distribution"
            ],
            "adjusted_records_with_multiple_hand_types": hand_type_values[
                "records_with_multiple_hand_types"
            ],
        },
    }


def test_summary_r17_adjusts_to_zero_without_dropping_record() -> None:
    record = _summary_r17()
    contribution = summary_module._build_contribution(record)
    result = _overall((record,))
    sensitivity = result["fifth_tile_sensitivity"]
    adjusted = sensitivity["adjusted_record_level"]

    assert result["established_riichi_count"] == 1
    assert contribution.fifth_tile_waits == ("3s",)
    assert contribution.adjusted.wait_tiles == ()
    assert contribution.adjusted.details == ()
    assert contribution.adjusted.shapes == ()
    assert contribution.adjusted.hand_types == ()
    assert sensitivity["records_with_only_fifth_tile_waits"]["count"] == 1
    assert adjusted["record_count"] == 1
    assert adjusted["adjusted_zero_wait_count"] == 1
    assert (
        _metric_by_value(adjusted["adjusted_wait_tile_count_distribution"], 0)["count"]
        == 1
    )
    assert adjusted["adjusted_is_multiwait"]["count"] == 0
    assert adjusted["adjusted_contains_ryanmen"]["count"] == 0
    assert adjusted["adjusted_is_pure_ryanmen"]["count"] == 0
    adjusted_shapes = _adjusted_shapes(adjusted)
    adjusted_hand_types = _adjusted_hand_types(adjusted)
    assert all(
        item["count"] == 0 for item in adjusted_shapes["adjusted_wait_shape_membership"]
    )
    assert all(
        item["count"] == 0
        for item in adjusted_hand_types["adjusted_hand_type_membership"]
    )
    assert (
        _set_category(adjusted_shapes["adjusted_wait_shape_set_distribution"], [])[
            "count"
        ]
        == 1
    )
    assert (
        _set_category(adjusted_hand_types["adjusted_hand_type_set_distribution"], [])[
            "count"
        ]
        == 1
    )


def test_summary_r18_recomputes_pure_and_multiwait_after_removal() -> None:
    record = _summary_r18()
    contribution = summary_module._build_contribution(record)
    result = _overall((record,))
    formal = result["formal_baseline"]
    adjusted = result["fifth_tile_sensitivity"]["adjusted_record_level"]
    adjusted_shapes = _adjusted_shapes(adjusted)
    adjusted_hand_types = _adjusted_hand_types(adjusted)

    assert contribution.fifth_tile_waits == ("2m",)
    assert contribution.adjusted.wait_tiles == ("5m", "8m")
    assert tuple(
        (detail.tile, detail.hand_type, detail.wait_shape)
        for detail in contribution.adjusted.details
    ) == (
        ("5m", "standard", "ryanmen"),
        ("8m", "standard", "ryanmen"),
    )
    assert formal["pure_ryanmen"]["count"] == 0
    assert formal["multiwait"]["count"] == 1
    assert adjusted["adjusted_is_pure_ryanmen"]["count"] == 1
    assert adjusted["adjusted_is_multiwait"]["count"] == 0
    assert adjusted["adjusted_contains_ryanmen"]["count"] == 1
    assert (
        _metric_by_value(adjusted["adjusted_wait_tile_count_distribution"], 2)["count"]
        == 1
    )
    assert (
        _set_category(
            adjusted_shapes["adjusted_wait_shape_set_distribution"], ["ryanmen"]
        )["count"]
        == 1
    )
    assert (
        _set_category(
            adjusted_hand_types["adjusted_hand_type_set_distribution"], ["standard"]
        )["count"]
        == 1
    )


def test_summary_r19_counts_red_five_and_turns_pure_false() -> None:
    record = _summary_r19()
    contribution = summary_module._build_contribution(record)
    result = _overall((record,))
    formal = result["formal_baseline"]
    sensitivity = result["fifth_tile_sensitivity"]
    adjusted = sensitivity["adjusted_record_level"]
    adjusted_shapes = _adjusted_shapes(adjusted)
    adjusted_hand_types = _adjusted_hand_types(adjusted)

    assert contribution.fifth_tile_waits == ("5m",)
    assert contribution.adjusted.wait_tiles == ("2m",)
    assert sensitivity["self_owned_four_fifth_tile_wait_observations"]["count"] == 1
    assert formal["pure_ryanmen"]["count"] == 1
    assert adjusted["adjusted_is_pure_ryanmen"]["count"] == 0
    assert adjusted["adjusted_contains_ryanmen"]["count"] == 1
    assert adjusted["adjusted_is_multiwait"]["count"] == 0
    assert (
        _metric_by_value(adjusted["adjusted_wait_tile_count_distribution"], 1)["count"]
        == 1
    )
    assert _set_category(
        adjusted_shapes["adjusted_wait_shape_set_distribution"], ["ryanmen"]
    ) == {"members": ["ryanmen"], "count": 1, "denominator": 1, "rate": 1.0}
    assert _set_category(
        adjusted_hand_types["adjusted_hand_type_set_distribution"], ["standard"]
    ) == {"members": ["standard"], "count": 1, "denominator": 1, "rate": 1.0}


def test_formal_record_memberships_hand_types_and_details_are_not_exclusive() -> None:
    pure = _record(
        waits=(
            ("2m", "standard", "ryanmen"),
            ("5m", "standard", "ryanmen"),
        ),
        source_suffix="pure.mjson",
    )
    composite = _record(
        waits=(
            ("2m", "standard", "ryanmen"),
            ("2m", "standard", "shanpon"),
            ("3s", "standard", "kanchan"),
        ),
        source_suffix="composite.mjson",
    )
    chiitoitsu = _record(
        waits=(("E", "chiitoitsu", "tanki"),),
        source_suffix="chiitoitsu.mjson",
    )
    kokushi = _record(
        waits=(("C", "kokushi", "kokushi_single"),),
        source_suffix="kokushi.mjson",
    )
    dual_hand_type = _record(
        waits=(
            ("E", "standard", "tanki"),
            ("E", "chiitoitsu", "tanki"),
        ),
        source_suffix="dual.mjson",
    )
    result = _overall((pure, composite, chiitoitsu, kokushi, dual_hand_type))
    formal = result["formal_baseline"]
    formal_shapes = _formal_shapes(formal)
    formal_hand_types = _formal_hand_types(formal)

    assert formal["record_count"] == 5
    assert formal["pure_ryanmen"] == {"count": 1, "denominator": 5, "rate": 0.2}
    assert formal["contains_ryanmen"]["count"] == 2
    assert (
        _metric_by_value(formal_shapes["wait_shape_membership"], "ryanmen")["count"]
        == 2
    )
    assert (
        _metric_by_value(formal_shapes["wait_shape_membership"], "shanpon")["count"]
        == 1
    )
    assert formal_shapes["records_with_multiple_wait_shapes"]["count"] == 1
    assert (
        _metric_by_value(formal_hand_types["hand_type_membership"], "standard")["count"]
        == 3
    )
    assert (
        _metric_by_value(formal_hand_types["hand_type_membership"], "chiitoitsu")[
            "count"
        ]
        == 2
    )
    assert (
        _metric_by_value(formal_hand_types["hand_type_membership"], "kokushi")["count"]
        == 1
    )
    assert formal_hand_types["records_with_multiple_hand_types"]["count"] == 1
    assert result["wait_detail_level"]["wait_detail_count"] == 9


def test_kokushi_13men_is_kept_as_thirteen_wait_tile_observations() -> None:
    record = _record(
        waits=tuple((tile, "kokushi", "kokushi_13men") for tile in ORPHANS),
        source_suffix="kokushi-13men.mjson",
    )
    result = _overall((record,))

    assert result["formal_baseline"]["multiwait"]["count"] == 1
    assert result["wait_tile_level"]["wait_tile_observation_count"] == 13
    assert result["wait_detail_level"]["wait_detail_count"] == 13
    assert (
        _metric_by_value(
            _formal_hand_types(result["formal_baseline"])["hand_type_membership"],
            "kokushi",
        )["count"]
        == 1
    )


def test_wait_tile_level_uses_each_tiles_own_observations_as_denominator() -> None:
    records = (
        _record(
            waits=(
                ("2m", "standard", "ryanmen"),
                ("5m", "standard", "ryanmen"),
            ),
            source_suffix="a.mjson",
        ),
        _record(
            waits=(("3s", "standard", "ryanmen"),),
            source_suffix="b.mjson",
        ),
        _record(
            waits=(("E", "standard", "tanki"),),
            source_suffix="c.mjson",
        ),
    )
    result = _overall(records)
    wait_tile_level = result["wait_tile_level"]
    three_s = next(item for item in wait_tile_level["by_tile"] if item["tile"] == "3s")
    ryanmen = _metric_by_value(three_s["shape_membership"], "ryanmen")

    assert wait_tile_level["wait_tile_observation_count"] == 4
    assert three_s["tile_frequency"] == {
        "count": 1,
        "denominator": 4,
        "share": 0.25,
    }
    assert ryanmen == {"value": "ryanmen", "count": 1, "denominator": 1, "rate": 1.0}


def test_wait_tile_multiple_interpretations_are_counted_once_per_tile() -> None:
    record = _record(
        waits=(
            ("2m", "standard", "ryanmen"),
            ("2m", "standard", "shanpon"),
        )
    )
    wait_tile_level = _overall((record,))["wait_tile_level"]

    assert wait_tile_level["wait_tile_observation_count"] == 1
    assert wait_tile_level["wait_tiles_with_multiple_shapes"]["count"] == 1
    assert (
        _set_category(
            wait_tile_level["shape_set_distribution"], ["ryanmen", "shanpon"]
        )["count"]
        == 1
    )


def test_year_and_exact_turn_slices_sum_to_overall() -> None:
    records = (
        _record(
            year=2024,
            turn=1,
            waits=(("3s", "standard", "penchan"),),
            source_suffix="a.mjson",
        ),
        _record(
            year=2025,
            turn=2,
            waits=(
                ("2m", "standard", "ryanmen"),
                ("5m", "standard", "ryanmen"),
            ),
            source_suffix="b.mjson",
        ),
        _record(
            year=2025,
            turn=2,
            waits=(("E", "chiitoitsu", "tanki"),),
            source_suffix="c.mjson",
        ),
    )
    aggregation = aggregate_riichi_wait_records(records)
    analysis = SummaryAnalysis(
        aggregation,
        _metadata(years=(2024, 2025), record_count=3),
    )
    documents = build_summary_documents(analysis)
    yearly = json.loads(documents["yearly"])
    by_turn = json.loads(documents["by_turn"])

    assert [value["year"] for value in yearly["years"]] == [2024, 2025]
    assert [value["established_riichi_count"] for value in yearly["years"]] == [1, 2]
    assert [value["riichi_discard_number"] for value in by_turn["all_years"]] == [1, 2]
    assert [value["established_riichi_count"] for value in by_turn["all_years"]] == [
        1,
        2,
    ]
    assert by_turn["by_year"][1]["turns"][0]["riichi_discard_number"] == 2
    assert by_turn["by_year"][1]["turns"][0]["established_riichi_count"] == 2


def test_overall_json_uses_the_complete_formal_and_adjusted_schema() -> None:
    analysis = SummaryAnalysis(
        aggregate_riichi_wait_records((_summary_r17(),)), _metadata()
    )
    actual = json.loads(build_summary_documents(analysis)["overall"])
    formal = _expected_record_level(
        denominator=1,
        wait_counts=((1, 1),),
        pure=0,
        contains=0,
        multi=0,
        shape_sets=((("penchan",), 1),),
        hand_type_sets=((("standard",), 1),),
    )
    adjusted = _expected_adjusted(
        denominator=1,
        wait_counts=((0, 1),),
        pure=0,
        contains=0,
        multi=0,
        shape_sets=(((), 1),),
        hand_type_sets=(((), 1),),
        zero_wait=1,
    )
    empty_tile_shapes = _expected_membership(WAIT_SHAPE_ORDER, {}, 0)
    expected_wait_tiles = {
        "wait_tile_observation_count": 1,
        "shape_membership": _expected_membership(WAIT_SHAPE_ORDER, {"penchan": 1}, 1),
        "shape_set_distribution": _expected_set_distribution(((("penchan",), 1),), 1),
        "wait_tiles_with_multiple_shapes": _expected_metric(0, 1),
        "by_tile": [
            {
                "tile": tile,
                "tile_frequency": {
                    "count": int(tile == "3s"),
                    "denominator": 1,
                    "share": float(tile == "3s"),
                },
                "shape_membership": (
                    _expected_membership(WAIT_SHAPE_ORDER, {"penchan": 1}, 1)
                    if tile == "3s"
                    else empty_tile_shapes
                ),
                "shape_set_distribution": (
                    _expected_set_distribution(((("penchan",), 1),), 1)
                    if tile == "3s"
                    else []
                ),
                "wait_tiles_with_multiple_shapes": _expected_metric(
                    0, int(tile == "3s")
                ),
            }
            for tile in TILE_KINDS
        ],
    }
    expected_details = {
        "wait_detail_count": 1,
        "hand_type_by_wait_shape": [
            {
                "hand_type": hand_type,
                "wait_shape": shape,
                "count": int(hand_type == "standard" and shape == "penchan"),
                "detail_denominator": 1,
                "share_of_details": (
                    1.0 if hand_type == "standard" and shape == "penchan" else 0.0
                ),
            }
            for hand_type in HAND_TYPE_ORDER
            for shape in WAIT_SHAPE_ORDER
        ],
    }
    expected = {
        "metadata": _metadata(),
        "result": {
            "established_riichi_count": 1,
            "formal_baseline": formal,
            "riichi_discard_number_distribution": [
                {"value": 1, **_expected_metric(1, 1)}
            ],
            "wait_tile_level": expected_wait_tiles,
            "wait_detail_level": expected_details,
            "fifth_tile_sensitivity": {
                "records_with_self_owned_four_fifth_tile_wait": _expected_metric(1, 1),
                "self_owned_four_fifth_tile_wait_observations": _expected_metric(1, 1),
                "records_with_only_fifth_tile_waits": _expected_metric(1, 1),
                "records_with_fifth_tile_and_other_waits": _expected_metric(0, 1),
                "adjusted_record_level": adjusted,
                "formal_adjusted_deltas": {
                    "is_pure_ryanmen": {
                        "count_delta": 0,
                        "percentage_point_delta": 0.0,
                    },
                    "contains_ryanmen": {
                        "count_delta": 0,
                        "percentage_point_delta": 0.0,
                    },
                    "is_multiwait": {
                        "count_delta": 0,
                        "percentage_point_delta": 0.0,
                    },
                },
            },
        },
    }

    assert actual == expected


def test_formal_and_adjusted_denominators_and_rates_match_in_every_slice() -> None:
    records = (
        _summary_r17(year=2024, turn=1),
        _summary_r18(year=2025, turn=2),
        _summary_r19(year=2025, turn=3),
    )
    analysis = SummaryAnalysis(
        aggregate_riichi_wait_records(records),
        _metadata(years=(2024, 2025), record_count=3),
    )
    documents = build_summary_documents(analysis)
    overall = json.loads(documents["overall"])["result"]
    yearly_values = {
        value["year"]: value for value in json.loads(documents["yearly"])["years"]
    }
    by_turn = json.loads(documents["by_turn"])
    turn_values = {
        value["riichi_discard_number"]: value for value in by_turn["all_years"]
    }
    year_turn_values = {
        (year_value["year"], turn["riichi_discard_number"]): turn
        for year_value in by_turn["by_year"]
        for turn in year_value["turns"]
    }
    formal_r17 = _expected_record_level(
        denominator=1,
        wait_counts=((1, 1),),
        pure=0,
        contains=0,
        multi=0,
        shape_sets=((("penchan",), 1),),
        hand_type_sets=((("standard",), 1),),
    )
    adjusted_r17 = _expected_adjusted(
        denominator=1,
        wait_counts=((0, 1),),
        pure=0,
        contains=0,
        multi=0,
        shape_sets=(((), 1),),
        hand_type_sets=(((), 1),),
        zero_wait=1,
    )
    formal_r18 = _expected_record_level(
        denominator=1,
        wait_counts=((3, 1),),
        pure=0,
        contains=1,
        multi=1,
        shape_sets=((("ryanmen",), 1),),
        hand_type_sets=((("standard",), 1),),
    )
    adjusted_r18 = _expected_adjusted(
        denominator=1,
        wait_counts=((2, 1),),
        pure=1,
        contains=1,
        multi=0,
        shape_sets=((("ryanmen",), 1),),
        hand_type_sets=((("standard",), 1),),
        zero_wait=0,
    )
    formal_r19 = _expected_record_level(
        denominator=1,
        wait_counts=((2, 1),),
        pure=1,
        contains=1,
        multi=0,
        shape_sets=((("ryanmen",), 1),),
        hand_type_sets=((("standard",), 1),),
    )
    adjusted_r19 = _expected_adjusted(
        denominator=1,
        wait_counts=((1, 1),),
        pure=0,
        contains=1,
        multi=0,
        shape_sets=((("ryanmen",), 1),),
        hand_type_sets=((("standard",), 1),),
        zero_wait=0,
    )
    formal_2025 = _expected_record_level(
        denominator=2,
        wait_counts=((2, 1), (3, 1)),
        pure=1,
        contains=2,
        multi=1,
        shape_sets=((("ryanmen",), 2),),
        hand_type_sets=((("standard",), 2),),
    )
    adjusted_2025 = _expected_adjusted(
        denominator=2,
        wait_counts=((1, 1), (2, 1)),
        pure=1,
        contains=2,
        multi=0,
        shape_sets=((("ryanmen",), 2),),
        hand_type_sets=((("standard",), 2),),
        zero_wait=0,
    )
    formal_all = _expected_record_level(
        denominator=3,
        wait_counts=((1, 1), (2, 1), (3, 1)),
        pure=1,
        contains=2,
        multi=1,
        shape_sets=((("ryanmen",), 2), (("penchan",), 1)),
        hand_type_sets=((("standard",), 3),),
    )
    adjusted_all = _expected_adjusted(
        denominator=3,
        wait_counts=((0, 1), (1, 1), (2, 1)),
        pure=1,
        contains=2,
        multi=0,
        shape_sets=(((), 1), (("ryanmen",), 2)),
        hand_type_sets=(((), 1), (("standard",), 2)),
        zero_wait=1,
    )

    full_slices = (
        (overall, formal_all, adjusted_all),
        (yearly_values[2024], formal_r17, adjusted_r17),
        (yearly_values[2025], formal_2025, adjusted_2025),
    )
    turn_slices = (
        (turn_values[1], formal_r17, adjusted_r17),
        (turn_values[2], formal_r18, adjusted_r18),
        (turn_values[3], formal_r19, adjusted_r19),
        (year_turn_values[(2024, 1)], formal_r17, adjusted_r17),
        (year_turn_values[(2025, 2)], formal_r18, adjusted_r18),
        (year_turn_values[(2025, 3)], formal_r19, adjusted_r19),
    )
    for value, expected_formal, expected_adjusted in full_slices:
        assert value["formal_baseline"] == expected_formal
        assert value["fifth_tile_sensitivity"]["adjusted_record_level"] == (
            expected_adjusted
        )
    for value, expected_formal, expected_adjusted in turn_slices:
        assert value["formal_baseline"] == expected_formal
        assert value["adjusted_record_level"] == expected_adjusted


def test_surviving_multiple_interpretations_are_preserved_after_adjustment() -> None:
    record = _record(
        waits=(
            ("2m", "standard", "ryanmen"),
            ("2m", "standard", "shanpon"),
            ("2m", "chiitoitsu", "tanki"),
            ("5m", "standard", "ryanmen"),
        ),
        concealed=("3m", "4m"),
        fixed_melds=(DatasetFixedMeld("ankan", ("5m", "5m", "5m", "5mr")),),
    )
    adjusted = _overall((record,))["fifth_tile_sensitivity"]["adjusted_record_level"]
    adjusted_shapes = _adjusted_shapes(adjusted)
    adjusted_hand_types = _adjusted_hand_types(adjusted)

    assert (
        _metric_by_value(adjusted["adjusted_wait_tile_count_distribution"], 1)["count"]
        == 1
    )
    assert (
        _metric_by_value(adjusted_shapes["adjusted_wait_shape_membership"], "ryanmen")[
            "count"
        ]
        == 1
    )
    assert (
        _metric_by_value(adjusted_shapes["adjusted_wait_shape_membership"], "shanpon")[
            "count"
        ]
        == 1
    )
    assert (
        _metric_by_value(adjusted_shapes["adjusted_wait_shape_membership"], "tanki")[
            "count"
        ]
        == 1
    )
    assert (
        _set_category(
            adjusted_shapes["adjusted_wait_shape_set_distribution"],
            ["ryanmen", "shanpon", "tanki"],
        )["count"]
        == 1
    )
    assert (
        _metric_by_value(
            adjusted_hand_types["adjusted_hand_type_membership"], "standard"
        )["count"]
        == 1
    )
    assert (
        _metric_by_value(
            adjusted_hand_types["adjusted_hand_type_membership"], "chiitoitsu"
        )["count"]
        == 1
    )
    assert (
        _set_category(
            adjusted_hand_types["adjusted_hand_type_set_distribution"],
            ["standard", "chiitoitsu"],
        )["count"]
        == 1
    )


def test_concealed_four_without_a_formal_candidate_is_not_adjusted() -> None:
    record = _record(
        waits=(("3s", "standard", "penchan"),),
        concealed=("5m", "5m", "5m", "5mr"),
    )
    result = _overall((record,))

    assert (
        result["fifth_tile_sensitivity"][
            "records_with_self_owned_four_fifth_tile_wait"
        ]["count"]
        == 0
    )
    assert (
        result["fifth_tile_sensitivity"]["adjusted_record_level"][
            "adjusted_wait_tile_count_distribution"
        ][0]["value"]
        == 1
    )


def test_documents_are_deterministic_finite_and_share_metadata() -> None:
    aggregation = aggregate_riichi_wait_records((_summary_r17(), _summary_r19()))
    analysis = SummaryAnalysis(aggregation, _metadata(record_count=2))

    first = build_summary_documents(analysis)
    second = build_summary_documents(analysis)

    assert first == second
    assert all(value.endswith("\n") for value in first.values())
    json_documents = [
        json.loads(first[key]) for key in ("overall", "yearly", "by_turn")
    ]
    assert all(
        value["metadata"] == json_documents[0]["metadata"] for value in json_documents
    )
    assert "NaN" not in "".join(first.values())
    assert "Infinity" not in "".join(first.values())


def test_summary_analysis_rejects_metadata_record_total_mismatch() -> None:
    aggregation = aggregate_riichi_wait_records((_summary_r17(), _summary_r19()))

    with pytest.raises(ValueError, match="record total disagrees"):
        SummaryAnalysis(aggregation, _metadata(record_count=1))


def test_summary_analysis_rejects_empty_metadata() -> None:
    with pytest.raises(ValueError, match="required fields"):
        SummaryAnalysis(
            aggregate_riichi_wait_records((_summary_r17(),)),
            {},
        )


@pytest.mark.parametrize(
    ("path", "invalid_value", "error"),
    (
        (("schema_version",), 999, "schema_version"),
        (("analysis_name",), "another-analysis", "analysis_name"),
        (("years",), [2026], "years"),
        (("scope", "rule_code"), "another-rule", "rule_code"),
        (("input_dataset", "schema_version"), 999, "input schema_version"),
        (("input_dataset", "manifest_sha256"), None, "manifest SHA256"),
        (("input_dataset", "generator_git_commit"), None, "dataset generator commit"),
        (("input_dataset", "source_release_tag"), "v0", "source release"),
        (("analysis_generator", "git_commit"), None, "analysis generator commit"),
        (("input_dataset", "source_repository"), "another/repository", "repository"),
    ),
)
def test_summary_analysis_rejects_invalid_required_metadata(
    path: tuple[str, ...],
    invalid_value: object,
    error: str,
) -> None:
    metadata = _metadata()
    target = metadata
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment,index]
    target[path[-1]] = invalid_value  # type: ignore[index]

    with pytest.raises((TypeError, ValueError), match=error):
        SummaryAnalysis(
            aggregate_riichi_wait_records((_summary_r17(),)),
            metadata,
        )


def test_document_generation_revalidates_mutated_metadata() -> None:
    metadata = _metadata()
    analysis = SummaryAnalysis(
        aggregate_riichi_wait_records((_summary_r17(),)),
        metadata,
    )
    metadata["schema_version"] = 999

    with pytest.raises(ValueError, match="schema_version"):
        build_summary_documents(analysis)


def test_input_record_order_does_not_change_canonical_documents() -> None:
    records = (_summary_r17(), _summary_r18(), _summary_r19())
    forward = SummaryAnalysis(
        aggregate_riichi_wait_records(records),
        _metadata(record_count=3),
    )
    reverse = SummaryAnalysis(
        aggregate_riichi_wait_records(tuple(reversed(records))),
        _metadata(record_count=3),
    )

    assert build_summary_documents(forward) == build_summary_documents(reverse)


def test_writer_publishes_all_four_documents(tmp_path: Path) -> None:
    aggregation = aggregate_riichi_wait_records((_summary_r17(),))
    analysis = SummaryAnalysis(aggregation, _metadata())

    paths = write_summary_outputs(analysis, tmp_path)

    assert len(paths) == 4
    assert all(path.is_file() for path in paths)
    assert paths == load_published_summary_paths(tmp_path)
    assert (tmp_path / PUBLICATION_POINTER_FILENAME).is_file()
    generation_root = paths[0].parent
    assert generation_root.parent.name == PUBLICATION_GENERATIONS_DIRECTORY
    assert (generation_root / COMPLETION_MARKER_FILENAME).is_file()
    assert not tuple(tmp_path.rglob("*.tmp"))
    assert not any(
        (tmp_path / filename).exists() for filename in OUTPUT_FILENAMES.values()
    )
    assert (
        json.loads(paths[0].read_text(encoding="utf-8"))["result"][
            "established_riichi_count"
        ]
        == 1
    )


def _read_json_file(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_json_file(path: Path, value: dict[str, object]) -> None:
    path.write_text(summary_module._json_document(value), encoding="utf-8", newline="")


def _install_self_consistent_generation(
    root: Path,
    documents: dict[str, str],
) -> tuple[Path, ...]:
    document_bytes = {key: value.encode("utf-8") for key, value in documents.items()}
    descriptor = summary_module._generation_descriptor(document_bytes)
    generation_id = summary_module._generation_id(descriptor)
    generation_root = root / PUBLICATION_GENERATIONS_DIRECTORY / generation_id
    generation_root.mkdir(parents=True)
    for key, content in document_bytes.items():
        (generation_root / OUTPUT_FILENAMES[key]).write_bytes(content)
    _write_json_file(
        generation_root / COMPLETION_MARKER_FILENAME,
        summary_module._completion_marker(generation_id, descriptor),
    )
    _write_json_file(
        root / PUBLICATION_POINTER_FILENAME,
        {
            "schema_version": 1,
            "analysis_name": ANALYSIS_NAME,
            "generation_id": generation_id,
            "completion_marker": (
                f"{PUBLICATION_GENERATIONS_DIRECTORY}/{generation_id}/"
                f"{COMPLETION_MARKER_FILENAME}"
            ),
        },
    )
    return tuple(
        generation_root / OUTPUT_FILENAMES[key]
        for key in ("overall", "yearly", "by_turn", "markdown")
    )


def _valid_documents() -> dict[str, str]:
    return build_summary_documents(
        SummaryAnalysis(
            aggregate_riichi_wait_records((_summary_r17(),)),
            _metadata(),
        )
    )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("schema_version", 2),
        ("schema_version", True),
        ("analysis_name", "another-analysis"),
    ),
)
def test_publication_pointer_rejects_invalid_schema_or_name(
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    paths = write_summary_outputs(
        SummaryAnalysis(aggregate_riichi_wait_records((_summary_r17(),)), _metadata()),
        tmp_path,
    )
    pointer_path = tmp_path / PUBLICATION_POINTER_FILENAME
    pointer = _read_json_file(pointer_path)
    pointer[field] = invalid_value
    _write_json_file(pointer_path, pointer)

    with pytest.raises(ValueError, match="publication pointer"):
        load_published_summary_paths(tmp_path)
    assert all(path.is_file() for path in paths)


def test_publication_pointer_rejects_malformed_json(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / PUBLICATION_POINTER_FILENAME).write_text("{broken\n", encoding="utf-8")

    with pytest.raises(ValueError, match="publication pointer"):
        load_published_summary_paths(tmp_path)


def test_publication_pointer_rejects_missing_generation(tmp_path: Path) -> None:
    generation_id = "f" * 64
    tmp_path.mkdir(exist_ok=True)
    _write_json_file(
        tmp_path / PUBLICATION_POINTER_FILENAME,
        {
            "schema_version": 1,
            "analysis_name": ANALYSIS_NAME,
            "generation_id": generation_id,
            "completion_marker": (
                f"{PUBLICATION_GENERATIONS_DIRECTORY}/{generation_id}/"
                f"{COMPLETION_MARKER_FILENAME}"
            ),
        },
    )

    with pytest.raises(ValueError, match="completion marker"):
        load_published_summary_paths(tmp_path)


def test_publication_pointer_rejects_incomplete_generation(tmp_path: Path) -> None:
    paths = _install_self_consistent_generation(tmp_path, _valid_documents())
    (paths[0].parent / COMPLETION_MARKER_FILENAME).unlink()

    with pytest.raises(ValueError, match="completion marker"):
        load_published_summary_paths(tmp_path)


def test_publication_pointer_rejects_generation_marker_mismatch(tmp_path: Path) -> None:
    paths = _install_self_consistent_generation(tmp_path, _valid_documents())
    marker_path = paths[0].parent / COMPLETION_MARKER_FILENAME
    marker = _read_json_file(marker_path)
    marker["generation_id"] = "f" * 64
    _write_json_file(marker_path, marker)

    with pytest.raises(ValueError, match="generation_id"):
        load_published_summary_paths(tmp_path)


def test_completion_marker_rejects_malformed_json(tmp_path: Path) -> None:
    paths = _install_self_consistent_generation(tmp_path, _valid_documents())
    marker_path = paths[0].parent / COMPLETION_MARKER_FILENAME
    marker_path.write_text("{broken\n", encoding="utf-8")

    with pytest.raises(ValueError, match="completion marker"):
        load_published_summary_paths(tmp_path)


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("required_field", "unexpected fields"),
        ("descriptor_entry", "four-item array"),
        ("filename", "filename"),
        ("size", "content does not match"),
        ("sha256", "content does not match"),
    ),
)
def test_completion_marker_rejects_invalid_descriptor(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    paths = _install_self_consistent_generation(tmp_path, _valid_documents())
    marker_path = paths[0].parent / COMPLETION_MARKER_FILENAME
    marker = _read_json_file(marker_path)
    if mutation == "required_field":
        marker.pop("files")
    else:
        files = marker["files"]
        assert isinstance(files, list)
        if mutation == "descriptor_entry":
            files.pop()
        else:
            first = files[0]
            assert isinstance(first, dict)
            if mutation == "filename":
                first["filename"] = "wrong.json"
            elif mutation == "size":
                first["size_bytes"] += 1
            else:
                first["sha256"] = "0" * 64
    _write_json_file(marker_path, marker)

    with pytest.raises((TypeError, ValueError), match=error):
        load_published_summary_paths(tmp_path)


@pytest.mark.parametrize("corruption", ("missing", "size", "sha256"))
def test_published_generation_rejects_corrupt_artifact(
    tmp_path: Path,
    corruption: str,
) -> None:
    paths = _install_self_consistent_generation(tmp_path, _valid_documents())
    target = paths[1]
    content = target.read_bytes()
    if corruption == "missing":
        target.unlink()
    elif corruption == "size":
        target.write_bytes(content + b" ")
    else:
        target.write_bytes(bytes((content[0] ^ 1,)) + content[1:])

    errors = {
        "missing": "file is unavailable",
        "size": "file size mismatch",
        "sha256": "file SHA256 mismatch",
    }
    with pytest.raises(ValueError, match=errors[corruption]):
        load_published_summary_paths(tmp_path)


@pytest.mark.parametrize(
    ("path", "invalid_value", "error"),
    (
        ((), {}, "required fields"),
        (("schema_version",), 999, "schema_version"),
        (("input_dataset", "manifest_sha256"), None, "manifest SHA256"),
        (("analysis_generator", "git_commit"), None, "analysis generator commit"),
        (("input_dataset", "source_repository"), "another/repository", "repository"),
    ),
)
def test_published_generation_rejects_invalid_metadata(
    tmp_path: Path,
    path: tuple[str, ...],
    invalid_value: object,
    error: str,
) -> None:
    documents = _valid_documents()
    for key in ("overall", "yearly", "by_turn"):
        document = json.loads(documents[key])
        if not path:
            document["metadata"] = invalid_value
        else:
            target = document["metadata"]
            for part in path[:-1]:
                target = target[part]
            target[path[-1]] = invalid_value
        documents[key] = summary_module._json_document(document)
    _install_self_consistent_generation(tmp_path, documents)

    with pytest.raises((TypeError, ValueError), match=error):
        load_published_summary_paths(tmp_path)


@pytest.mark.parametrize("document_key", ("overall", "yearly", "by_turn"))
@pytest.mark.parametrize(
    ("path", "invalid_value", "error"),
    (
        (("schema_version",), True, "schema_version"),
        (("scope", "aka_flag"), 1, "aka_flag"),
        (("years",), [2025.0], "years"),
        (("input_dataset", "totals", "output_records"), True, "output_records"),
    ),
)
def test_published_generation_rejects_invalid_metadata_in_each_document(
    tmp_path: Path,
    document_key: str,
    path: tuple[str, ...],
    invalid_value: object,
    error: str,
) -> None:
    documents = _valid_documents()
    document = json.loads(documents[document_key])
    target = document["metadata"]
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = invalid_value
    documents[document_key] = summary_module._json_document(document)
    _install_self_consistent_generation(tmp_path, documents)

    with pytest.raises((TypeError, ValueError), match=error):
        load_published_summary_paths(tmp_path)


def test_published_generation_rejects_valid_but_different_metadata(
    tmp_path: Path,
) -> None:
    documents = _valid_documents()
    document = json.loads(documents["yearly"])
    document["metadata"]["analysis_generator"]["git_commit"] = "another-valid-commit"
    documents["yearly"] = summary_module._json_document(document)
    _install_self_consistent_generation(tmp_path, documents)

    with pytest.raises(ValueError, match="different metadata"):
        load_published_summary_paths(tmp_path)


def test_same_incomplete_generation_is_safely_regenerated(tmp_path: Path) -> None:
    documents = _valid_documents()
    descriptor = summary_module._generation_descriptor(
        {key: value.encode("utf-8") for key, value in documents.items()}
    )
    generation_id = summary_module._generation_id(descriptor)
    generation_root = tmp_path / PUBLICATION_GENERATIONS_DIRECTORY / generation_id
    generation_root.mkdir(parents=True)
    (generation_root / OUTPUT_FILENAMES["overall"]).write_bytes(b"partial")
    analysis = SummaryAnalysis(
        aggregate_riichi_wait_records((_summary_r17(),)),
        _metadata(),
    )

    paths = write_summary_outputs(analysis, tmp_path)

    assert paths[0].parent == generation_root
    assert tuple(path.read_text(encoding="utf-8") for path in paths) == tuple(
        documents[key] for key in ("overall", "yearly", "by_turn", "markdown")
    )


def test_same_complete_generation_is_revalidated_without_artifact_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = SummaryAnalysis(
        aggregate_riichi_wait_records((_summary_r17(),)),
        _metadata(),
    )
    paths = write_summary_outputs(analysis, tmp_path)
    written_paths: list[Path] = []
    real_write = summary_module._write_bytes_atomically

    def record_write(path: Path, content: bytes) -> None:
        written_paths.append(path)
        real_write(path, content)

    monkeypatch.setattr(summary_module, "_write_bytes_atomically", record_write)

    assert write_summary_outputs(analysis, tmp_path) == paths
    assert written_paths == [tmp_path / PUBLICATION_POINTER_FILENAME]


def test_same_complete_generation_corruption_is_not_reused(tmp_path: Path) -> None:
    analysis = SummaryAnalysis(
        aggregate_riichi_wait_records((_summary_r17(),)),
        _metadata(),
    )
    paths = write_summary_outputs(analysis, tmp_path)
    content = paths[0].read_bytes()
    paths[0].write_bytes(bytes((content[0] ^ 1,)) + content[1:])

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        write_summary_outputs(analysis, tmp_path)


def test_generation_file_failure_never_publishes_partial_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregation = aggregate_riichi_wait_records((_summary_r17(),))
    analysis = SummaryAnalysis(aggregation, _metadata())
    real_replace = os.replace

    def fail_yearly_replace(source: str | Path, target: str | Path) -> None:
        if Path(target).name == OUTPUT_FILENAMES["yearly"]:
            raise OSError("injected publish failure")
        real_replace(source, target)

    monkeypatch.setattr(
        "mahjong_analysis.riichi_wait_summary.os.replace",
        fail_yearly_replace,
    )

    with pytest.raises(OSError, match="injected publish failure"):
        write_summary_outputs(analysis, tmp_path)

    assert not (tmp_path / PUBLICATION_POINTER_FILENAME).exists()
    assert not tuple(tmp_path.rglob(COMPLETION_MARKER_FILENAME))
    with pytest.raises(ValueError, match="publication pointer"):
        load_published_summary_paths(tmp_path)


def test_completion_marker_failure_leaves_no_published_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = SummaryAnalysis(
        aggregate_riichi_wait_records((_summary_r17(),)), _metadata()
    )
    real_write = summary_module._write_bytes_atomically

    def fail_marker(path: Path, content: bytes) -> None:
        if path.name == COMPLETION_MARKER_FILENAME:
            raise OSError("injected marker failure")
        real_write(path, content)

    monkeypatch.setattr(summary_module, "_write_bytes_atomically", fail_marker)

    with pytest.raises(OSError, match="injected marker failure"):
        write_summary_outputs(analysis, tmp_path)

    assert not (tmp_path / PUBLICATION_POINTER_FILENAME).exists()
    assert not tuple(tmp_path.rglob(COMPLETION_MARKER_FILENAME))


def test_pointer_failure_keeps_the_previous_complete_generation_official(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_analysis = SummaryAnalysis(
        aggregate_riichi_wait_records((_summary_r17(),)), _metadata()
    )
    new_analysis = SummaryAnalysis(
        aggregate_riichi_wait_records((_summary_r19(),)), _metadata()
    )
    old_paths = write_summary_outputs(old_analysis, tmp_path)
    old_pointer = (tmp_path / PUBLICATION_POINTER_FILENAME).read_bytes()
    old_contents = tuple(path.read_bytes() for path in old_paths)
    real_replace = os.replace

    def fail_pointer(source: str | Path, target: str | Path) -> None:
        if Path(target).name == PUBLICATION_POINTER_FILENAME:
            raise PermissionError("injected pointer failure")
        real_replace(source, target)

    monkeypatch.setattr(summary_module.os, "replace", fail_pointer)

    with pytest.raises(PermissionError, match="injected pointer failure"):
        write_summary_outputs(new_analysis, tmp_path)

    assert (tmp_path / PUBLICATION_POINTER_FILENAME).read_bytes() == old_pointer
    assert load_published_summary_paths(tmp_path) == old_paths
    assert tuple(path.read_bytes() for path in old_paths) == old_contents
    assert len(tuple(tmp_path.rglob(COMPLETION_MARKER_FILENAME))) == 2


def test_replace_and_cleanup_failures_do_not_mix_published_generations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_analysis = SummaryAnalysis(
        aggregate_riichi_wait_records((_summary_r17(),)), _metadata()
    )
    new_analysis = SummaryAnalysis(
        aggregate_riichi_wait_records((_summary_r19(),)), _metadata()
    )
    old_paths = write_summary_outputs(old_analysis, tmp_path)
    old_contents = tuple(path.read_bytes() for path in old_paths)
    real_replace = os.replace
    real_unlink = Path.unlink

    def fail_new_yearly(source: str | Path, target: str | Path) -> None:
        if Path(target).name == OUTPUT_FILENAMES["yearly"]:
            raise PermissionError("injected generation replace failure")
        real_replace(source, target)

    def fail_temp_cleanup(path: Path, *args: object, **kwargs: object) -> None:
        if path.name.endswith(".tmp"):
            raise PermissionError("injected cleanup failure")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(summary_module.os, "replace", fail_new_yearly)
    monkeypatch.setattr(Path, "unlink", fail_temp_cleanup)

    with pytest.raises(PermissionError, match="generation replace failure"):
        write_summary_outputs(new_analysis, tmp_path)

    assert load_published_summary_paths(tmp_path) == old_paths
    assert tuple(path.read_bytes() for path in old_paths) == old_contents
    assert tuple(tmp_path.rglob("*.tmp"))


def test_unreferenced_generation_residue_is_not_a_published_result(
    tmp_path: Path,
) -> None:
    residue = tmp_path / PUBLICATION_GENERATIONS_DIRECTORY / ("0" * 64)
    residue.mkdir(parents=True)
    (residue / OUTPUT_FILENAMES["overall"]).write_text("partial", encoding="utf-8")

    with pytest.raises(ValueError, match="publication pointer"):
        load_published_summary_paths(tmp_path)

    analysis = SummaryAnalysis(
        aggregate_riichi_wait_records((_summary_r17(),)), _metadata()
    )
    paths = write_summary_outputs(analysis, tmp_path)

    assert paths == load_published_summary_paths(tmp_path)
    assert residue not in {path.parent for path in paths}


def _write_fixture_dataset(
    root: Path,
    records: tuple[RiichiWaitDatasetRecord, ...],
    *,
    output_records: int | None = None,
    malformed: bool = False,
) -> dict[str, object]:
    root.mkdir(parents=True)
    annual_path = root / "2025.jsonl.gz"
    with gzip.open(annual_path, "wb") as file:
        if malformed:
            file.write(b"{broken}\n")
        else:
            for record in records:
                file.write(serialize_dataset_record(record) + b"\n")
    count = len(records) if output_records is None else output_records
    annual_bytes = annual_path.read_bytes()
    year_entry = {
        "year": 2025,
        "scanned_files": count,
        "target_games": count,
        "east_kyokus": count,
        "established_riichis": count,
        "output_records": count,
        "output_filename": "2025.jsonl.gz",
        "compressed_size_bytes": len(annual_bytes),
        "sha256": hashlib.sha256(annual_bytes).hexdigest(),
        "elapsed_seconds": 1.0,
    }
    manifest = {
        "schema_version": 1,
        "dataset_name": DATASET_NAME,
        "created_at_utc": "2026-09-14T00:00:00Z",
        "source": {
            "repository": "NikkeTryHard/tenhou-to-mjai",
            "release_tag": "v2.0.0",
            "validation_summary_path": "data/validation/summary.json",
            "validation_summary_sha256": "1" * 64,
            "archive_hashes_verified": True,
        },
        "scope": {
            "years": [2025],
            "rule_code": "00a9",
            "aka_flag": True,
            "bakaze": "E",
            "input_selection": {
                "ordering": "raw-root-relative POSIX path lexicographic",
                "max_files_before_target_filtering": None,
            },
            "extraction_mode": "full",
        },
        "serialization": {
            "format": "JSON Lines",
            "encoding": "UTF-8",
            "json_options": {
                "ensure_ascii": False,
                "sort_keys": True,
                "separators": [",", ":"],
                "allow_nan": False,
                "line_terminator": "LF",
            },
            "compression": "gzip",
            "compression_level": 6,
            "gzip_mtime": 0,
            "gzip_header_filename": "",
        },
        "generator": {
            "git_commit": "dataset-commit",
            "worktree_clean": True,
            "python_version": "3.12.10",
            "zlib_version": "1.3.1",
            "invocation": {"years": [2025], "mode": "full", "max_files": None},
        },
        "years": [year_entry],
        "totals": {
            "scanned_files": count,
            "target_games": count,
            "east_kyokus": count,
            "established_riichis": count,
            "output_records": count,
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_streaming_dataset_path_validates_and_aggregates_once(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data" / "processed" / "riichi-waits-v1"
    manifest = _write_fixture_dataset(dataset_root, (_summary_r17(), _summary_r19()))
    (dataset_root / "unlisted.jsonl.gz").write_bytes(b"not a gzip stream")
    progress: list[SummaryProgress] = []

    analysis = summarize_riichi_wait_dataset(
        dataset_root,
        analysis_git=AnalysisGitMetadata("analysis-commit", True),
        project_root=tmp_path,
        progress_interval=1,
        progress_callback=progress.append,
        require_canonical=False,
    )

    assert analysis.aggregation.record_count == 2
    assert [
        (item.year, item.annual_records, item.total_records) for item in progress
    ] == [
        (2025, 1, 1),
        (2025, 2, 2),
    ]
    assert all(item.elapsed_seconds >= 0 for item in progress)
    assert (
        analysis.metadata["input_dataset"]["generator_git_commit"]
        == (manifest["generator"]["git_commit"])
    )
    assert analysis.metadata["input_dataset"]["manifest_path"] == (
        "data/processed/riichi-waits-v1/manifest.json"
    )


def test_streaming_dataset_rejects_dto_validation_failure(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    manifest = _write_fixture_dataset(dataset_root, (_summary_r17(),))
    payload = dataset_record_to_dict(_summary_r17())
    payload["waits"]["wait_tile_count"] = 2
    annual_path = dataset_root / "2025.jsonl.gz"
    with gzip.open(annual_path, "wt", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(payload, separators=(",", ":")) + "\n")
    annual_bytes = annual_path.read_bytes()
    manifest["years"][0]["compressed_size_bytes"] = len(annual_bytes)
    manifest["years"][0]["sha256"] = hashlib.sha256(annual_bytes).hexdigest()
    (dataset_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="derived from canonical waits"):
        summarize_riichi_wait_dataset(
            dataset_root,
            analysis_git=AnalysisGitMetadata("analysis-commit", True),
            project_root=tmp_path,
            require_canonical=False,
        )


def test_streaming_dataset_rejects_annual_row_count_mismatch(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_fixture_dataset(dataset_root, (_summary_r17(),), output_records=2)

    with pytest.raises(ValueError, match="record count mismatch"):
        summarize_riichi_wait_dataset(
            dataset_root,
            analysis_git=AnalysisGitMetadata("analysis-commit", True),
            project_root=tmp_path,
            require_canonical=False,
        )


def test_streaming_dataset_rejects_malformed_json(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_fixture_dataset(dataset_root, (_summary_r17(),), malformed=True)

    with pytest.raises(ValueError, match="invalid dataset JSON"):
        summarize_riichi_wait_dataset(
            dataset_root,
            analysis_git=AnalysisGitMetadata("analysis-commit", True),
            project_root=tmp_path,
            require_canonical=False,
        )


def test_streaming_dataset_rejects_size_mismatch_before_aggregation(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    _write_fixture_dataset(dataset_root, (_summary_r17(),))
    path = dataset_root / "2025.jsonl.gz"
    path.write_bytes(path.read_bytes() + b"corruption")

    with pytest.raises(ValueError, match="size mismatch"):
        summarize_riichi_wait_dataset(
            dataset_root,
            analysis_git=AnalysisGitMetadata("analysis-commit", True),
            project_root=tmp_path,
            require_canonical=False,
        )


def test_streaming_dataset_rejects_sha_mismatch_before_aggregation(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    manifest = _write_fixture_dataset(dataset_root, (_summary_r17(),))
    manifest["years"][0]["sha256"] = "0" * 64
    (dataset_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        summarize_riichi_wait_dataset(
            dataset_root,
            analysis_git=AnalysisGitMetadata("analysis-commit", True),
            project_root=tmp_path,
            require_canonical=False,
        )


def test_formal_run_rejects_dirty_analysis_code_before_reading_input(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="clean worktree"):
        summarize_riichi_wait_dataset(
            tmp_path / "missing",
            analysis_git=AnalysisGitMetadata("analysis-commit", False),
            project_root=tmp_path,
            require_canonical=False,
        )


def test_git_clean_guard_overrides_hidden_untracked_configuration(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(("git", "init", str(repository)), check=True, capture_output=True)
    (repository / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    tracked = repository / "tracked.txt"
    tracked.write_text("committed\n", encoding="utf-8")
    subprocess.run(
        ("git", "-C", str(repository), "add", ".gitignore", "tracked.txt"),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        (
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Summary Test",
            "-c",
            "user.email=summary@example.invalid",
            "commit",
            "-m",
            "initial",
        ),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        (
            "git",
            "-C",
            str(repository),
            "config",
            "status.showUntrackedFiles",
            "no",
        ),
        check=True,
        capture_output=True,
    )

    (repository / "ignored.txt").write_text("ignored\n", encoding="utf-8")
    assert collect_analysis_git_metadata(repository).worktree_clean is True

    untracked = repository / "analysis-not-in-head.py"
    untracked.write_text("print('untracked')\n", encoding="utf-8")
    metadata = collect_analysis_git_metadata(repository)
    assert metadata.worktree_clean is False
    with pytest.raises(ValueError, match="clean worktree"):
        summarize_riichi_wait_dataset(
            repository / "missing-dataset",
            analysis_git=metadata,
            project_root=repository,
            require_canonical=False,
        )

    untracked.unlink()
    tracked.write_text("modified\n", encoding="utf-8")
    assert collect_analysis_git_metadata(repository).worktree_clean is False


def test_canonical_manifest_requires_the_fixed_source_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    years = tuple(range(2009, 2026))
    manifest = {
        "source": {"repository": CANONICAL_SOURCE_REPOSITORY},
        "scope": {"years": list(years)},
        "years": [{"year": year} for year in years],
        "totals": {
            "output_records": CANONICAL_RECORD_COUNT,
            "established_riichis": CANONICAL_RECORD_COUNT,
        },
    }

    summary_module._validate_canonical_manifest(manifest, years)
    manifest["source"]["repository"] = "another/repository"
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        summary_module,
        "validate_dataset_integrity",
        lambda *args, **kwargs: manifest,
    )

    with pytest.raises(ValueError, match="source repository"):
        summarize_riichi_wait_dataset(
            dataset_root,
            analysis_git=AnalysisGitMetadata("analysis-commit", True),
            project_root=tmp_path,
        )
