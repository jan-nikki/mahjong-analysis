from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import subprocess
import sys
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal, Inexact, ROUND_DOWN, localcontext
from fractions import Fraction
from math import isqrt
from pathlib import Path
from typing import Any

import pytest

import mahjong_analysis.dealer_riichi_defense as defense
from mahjong_analysis.dealer_child_riichi_points import extract_established_riichis
from mahjong_analysis.dealer_riichi_defense import (
    DECIMAL_PRECISION_DIGITS,
    WEIGHTED_SUM_REPRESENTATION,
    BinaryCount,
    DefenseAnalysisBuilder,
    analyze_focal_kyoku,
    canonical_json,
    classify_discard,
    publish_output_bundle,
    standardize_binary_counts,
)
from mahjong_analysis.hand_waits import HandWaits, WaitDetail
from mahjong_analysis.riichi_wait_dataset import (
    DatasetActorDiscard,
    DatasetWaitDetail,
    RiichiWaitDatasetRecord,
    dataset_record_to_dict,
)
from mahjong_analysis.tiles import normalize_tile

_CLI_PATH = Path(__file__).parents[1] / "analysis/analyze_dealer_riichi_defense.py"
_CLI_SPEC = importlib.util.spec_from_file_location(
    "dealer_riichi_defense_cli", _CLI_PATH
)
assert _CLI_SPEC is not None and _CLI_SPEC.loader is not None
defense_cli = importlib.util.module_from_spec(_CLI_SPEC)
sys.modules[_CLI_SPEC.name] = defense_cli
_CLI_SPEC.loader.exec_module(defense_cli)

_VALID_SOURCE = "2025/2025010100gm-00a9-0000-00000001.mjson"


def _start(*, oya: int = 0, kyoku: int = 1) -> dict[str, Any]:
    return {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": kyoku,
        "honba": 0,
        "oya": oya,
    }


def _record_for_first_reach(
    events: list[dict[str, Any]],
    *,
    wait_specs: tuple[tuple[str, str], ...] = (("2p", "tanki"),),
    source_path: str = "2025/artificial.mjson",
    start_kyoku_line: int = 1,
) -> RiichiWaitDatasetRecord:
    """Build a strict wait-record DTO for a compact artificial MJAI kyoku."""
    focal = extract_established_riichis(events)[0]
    declaration_index = focal.reach_event_index + 1
    declaration = events[declaration_index]
    actor_discards = [
        (index, event)
        for index, event in enumerate(events[: declaration_index + 1])
        if event.get("type") == "dahai" and event.get("actor") == focal.actor
    ]
    river = tuple(
        DatasetActorDiscard(
            discard_number=number,
            tile=event["pai"],
            normalized_tile=normalize_tile(event["pai"]),
            tsumogiri=event["tsumogiri"],
            event_index=index,
            is_riichi_declaration=index == declaration_index,
            was_called=False,
            call_type=None,
            called_by_actor=None,
            call_event_index=None,
        )
        for number, (index, event) in enumerate(actor_discards, 1)
    )
    wait_details = tuple(
        WaitDetail(tile, "standard", shape)  # type: ignore[arg-type]
        for tile, shape in wait_specs
    )
    waits = HandWaits(
        wait_tiles=tuple(tile for tile, _ in wait_specs),
        wait_details=wait_details,
    )
    start = events[0]
    start_line = start_kyoku_line
    return RiichiWaitDatasetRecord(
        year=2025,
        relative_source_path=source_path,
        start_kyoku_line=start_line,
        reach_line=start_line + focal.reach_event_index,
        declaration_dahai_line=start_line + declaration_index,
        reach_accepted_line=start_line + focal.reach_accepted_event_index,
        bakaze="E",
        kyoku=start["kyoku"],
        honba=start["honba"],
        oya=start["oya"],
        scores_at_start=(25_000, 25_000, 25_000, 25_000),
        dora_marker="1p",
        actor=focal.actor,
        riichi_discard_number=len(river),
        riichi_declaration_tile=declaration["pai"],
        riichi_declaration_tile_kind=normalize_tile(declaration["pai"]),
        reach_event_index=focal.reach_event_index,
        declaration_dahai_event_index=declaration_index,
        reach_accepted_event_index=focal.reach_accepted_event_index,
        concealed_tiles_after_discard=(),
        fixed_melds=(),
        actor_discards_before_riichi=river,
        wait_tiles=waits.wait_tiles,
        wait_tile_count=waits.wait_tile_count,
        wait_details=tuple(
            DatasetWaitDetail(detail.wait_tile, detail.hand_type, detail.wait_shape)
            for detail in waits.wait_details
        ),
        wait_shapes=waits.wait_shapes,
        contains_ryanmen=waits.contains_ryanmen,
        is_pure_ryanmen=waits.is_pure_ryanmen,
        is_multiwait=waits.is_multiwait,
    )


@pytest.mark.parametrize(
    ("tile", "safe", "expected"),
    (
        ("5mr", {"5m"}, "genbutsu"),
        ("4m", {"1m", "7m"}, "full_suji"),
        ("4m", {"1m"}, "partial_suji"),
        ("1p", set(), "unsuji_numbered"),
        ("E", set(), "honor_non_genbutsu"),
        ("5pr", {"2p", "8p"}, "full_suji"),
    ),
)
def test_classify_discard_covers_red_genbutsu_suji_and_honor(
    tile: str,
    safe: set[str],
    expected: str,
) -> None:
    assert classify_discard(tile, safe) == expected


def test_dealer_focal_uses_all_children_and_updates_safe_set_after_decision() -> None:
    events = [
        _start(oya=0),
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "5mr", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 0},
        {"type": "dahai", "actor": 1, "pai": "1p", "tsumogiri": False},
        {"type": "dahai", "actor": 2, "pai": "1p", "tsumogiri": True},
        {"type": "dahai", "actor": 3, "pai": "E", "tsumogiri": False},
        {"type": "ryukyoku"},
        {"type": "end_kyoku"},
    ]

    result = analyze_focal_kyoku(_record_for_first_reach(events), events)

    assert result.focal_role == "dealer"
    assert [decision.primary_child_responder for decision in result.decisions] == [
        True,
        True,
        True,
    ]
    assert [decision.safety_class for decision in result.decisions] == [
        "unsuji_numbered",
        "genbutsu",
        "honor_non_genbutsu",
    ]
    assert [decision.relative_seat for decision in result.decisions] == [1, 2, 3]
    assert result.primary_child_discard_exposures == 3
    assert result.all_responder_discard_exposures == 3
    assert result.window_end_reason == "hand_end"
    assert result.terminal_outcome == "draw"
    assert result.outcome == "draw"


def test_child_focal_tracks_secondary_dealer_open_state_calls_and_censor() -> None:
    events = [
        _start(oya=0),
        {"type": "dahai", "actor": 1, "pai": "2m", "tsumogiri": False},
        {"type": "pon", "actor": 2},
        {"type": "ankan", "actor": 3},
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "3m", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 1},
        {"type": "tsumo", "actor": 1, "pai": "4p"},
        {"type": "ankan", "actor": 2},
        {"type": "ankan", "actor": 3},
        {"type": "pon", "actor": 3},
        {"type": "dahai", "actor": 3, "pai": "7s", "tsumogiri": False},
        {"type": "dahai", "actor": 0, "pai": "9p", "tsumogiri": True},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 0},
        {"type": "dahai", "actor": 2, "pai": "E", "tsumogiri": False},
        {"type": "ryukyoku"},
        {"type": "end_kyoku"},
    ]
    record = _record_for_first_reach(events, wait_specs=(("2m", "tanki"),))

    result = analyze_focal_kyoku(record, events)

    assert result.focal_role == "nondealer"
    assert result.reach_type == "riichi"
    assert result.focal_furiten is True
    assert result.focal_draw_opportunities == 1
    assert result.window_end_reason == "second_reach_accepted"
    assert result.terminal_outcome == "draw"
    assert result.outcome == "censored_second_reach"
    assert result.censor_actor == 0
    assert result.second_reach_declaration_established is True
    assert result.second_reach_declaration_unestablished is False
    # The later player's declaration discard is included; play after its
    # reach_accepted is not.
    assert [(item.responder, item.event_index) for item in result.decisions] == [
        (3, 11),
        (0, 12),
        (0, 14),
    ]
    assert [item.primary_child_responder for item in result.decisions] == [
        True,
        False,
        False,
    ]
    assert result.primary_child_discard_exposures == 1
    assert result.all_responder_discard_exposures == 3

    by_actor = {item.responder: item for item in result.responders}
    assert by_actor[0].later_established_riichi is True
    assert by_actor[0].primary_child_responder is False
    assert by_actor[2].responder_open_at_focal is True
    assert by_actor[2].any_call is True
    assert by_actor[3].responder_open_at_focal is False
    assert by_actor[3].any_call is True
    actor_three_decision = result.decisions[0]
    assert actor_three_decision.responder_open_at_focal is False
    assert actor_three_decision.responder_open_at_decision is True
    assert actor_three_decision.relative_seat == 2


def test_unestablished_second_reach_discard_and_immediate_multi_ron_are_counted() -> (
    None
):
    events = [
        _start(oya=0),
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 0},
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "6s", "tsumogiri": False},
        {"type": "hora", "actor": 3, "target": 1},
        {"type": "hora", "actor": 0, "target": 1},
        {"type": "end_kyoku"},
    ]

    result = analyze_focal_kyoku(
        _record_for_first_reach(events, wait_specs=(("6s", "tanki"),)),
        events,
    )

    assert result.censor_actor is None
    assert result.window_end_reason == "hand_end"
    assert result.terminal_outcome == "focal_ron"
    assert result.outcome == "focal_ron"
    assert result.terminal_hora_events == 2
    assert result.terminal_multi_ron is True
    assert result.second_reach_declaration_established is False
    assert result.second_reach_declaration_unestablished is True
    assert result.ron_events_on_second_reach_declaration == 2
    assert result.focal_ron_on_second_reach_declaration is True
    assert result.focal_ron_without_discard is False
    assert len(result.decisions) == 1
    decision = result.decisions[0]
    assert decision.event_index == 5
    assert decision.focal_ron_on_discard is True
    assert result.ron_details == (
        defense.RonDetail(
            responder=1,
            relative_seat=1,
            responder_open=False,
            safety_class="unsuji_numbered",
            tsumogiri=False,
            event_index=5,
        ),
    )


@pytest.mark.parametrize(
    ("terminal_events", "expected_outcome", "hora_events", "multi_ron"),
    (
        (
            ({"type": "hora", "actor": 0, "target": 0},),
            "focal_tsumo",
            1,
            False,
        ),
        (
            (
                {
                    "type": "dahai",
                    "actor": 2,
                    "pai": "7p",
                    "tsumogiri": False,
                },
                {"type": "hora", "actor": 0, "target": 2, "pai": "7p"},
            ),
            "focal_ron",
            1,
            False,
        ),
        (
            ({"type": "hora", "actor": 2, "target": 3},),
            "other_win",
            1,
            False,
        ),
        (({"type": "ryukyoku"},), "draw", 0, False),
        (
            (
                {
                    "type": "dahai",
                    "actor": 2,
                    "pai": "7p",
                    "tsumogiri": True,
                },
                {"type": "hora", "actor": 3, "target": 2, "pai": "7p"},
                {"type": "hora", "actor": 0, "target": 2, "pai": "7p"},
            ),
            "focal_ron",
            2,
            True,
        ),
    ),
)
def test_terminal_outcome_is_retained_after_response_window_is_censored(
    terminal_events: tuple[dict[str, Any], ...],
    expected_outcome: str,
    hora_events: int,
    multi_ron: bool,
) -> None:
    events = [
        _start(oya=0),
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 0},
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "6m", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 1},
        *terminal_events,
        {"type": "end_kyoku"},
    ]

    result = analyze_focal_kyoku(
        _record_for_first_reach(events, wait_specs=(("7p", "tanki"),)),
        events,
    )

    assert result.window_end_reason == "second_reach_accepted"
    assert result.outcome == "censored_second_reach"
    assert result.terminal_outcome == expected_outcome
    assert result.terminal_hora_events == hora_events
    assert result.terminal_multi_ron is multi_ron
    assert result.censor_actor == 1
    assert result.second_reach_declaration_established is True
    assert [
        (decision.responder, decision.event_index) for decision in result.decisions
    ] == [(1, 5)]
    assert result.ron_details == ()


def test_chankan_is_terminal_focal_ron_but_not_a_discard_opportunity() -> None:
    events = [
        _start(oya=0),
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 0},
        {
            "type": "kakan",
            "actor": 1,
            "pai": "2p",
            "consumed": ["2p", "2p", "2p"],
        },
        {"type": "hora", "actor": 0, "target": 1, "pai": "2p"},
        {"type": "end_kyoku"},
    ]

    result = analyze_focal_kyoku(
        _record_for_first_reach(events, wait_specs=(("2p", "tanki"),)),
        events,
    )

    assert result.window_end_reason == "hand_end"
    assert result.terminal_outcome == "focal_ron"
    assert result.terminal_hora_events == 1
    assert result.terminal_multi_ron is False
    assert result.focal_ron_without_discard is True
    assert result.decisions == ()
    assert result.ron_details == ()
    assert result.primary_child_discard_exposures == 0
    assert {item.responder: item.any_call for item in result.responders}[1] is True


def test_focal_analysis_rejects_nonfirst_record_and_malformed_dahai() -> None:
    events = [
        _start(oya=0),
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 0},
        {"type": "dahai", "actor": 1, "pai": "1p"},
        {"type": "ryukyoku"},
        {"type": "end_kyoku"},
    ]
    record = _record_for_first_reach(events)

    with pytest.raises(ValueError, match="tsumogiri must be boolean"):
        analyze_focal_kyoku(record, events)
    with pytest.raises(ValueError, match="not the first established riichi"):
        analyze_focal_kyoku(replace(record, actor=1), events)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    (
        ("riichi_discard_number", 3),
        ("riichi_declaration_tile", "6m"),
        ("riichi_declaration_tile_kind", "6m"),
        ("declaration_dahai_event_index", 1),
        ("reach_line", 99),
        ("declaration_dahai_line", 99),
        ("reach_accepted_line", 99),
    ),
)
def test_focal_analysis_rejects_wait_record_scalar_source_mismatches(
    field_name: str,
    bad_value: object,
) -> None:
    events = [
        _start(oya=0),
        {"type": "dahai", "actor": 0, "pai": "2m", "tsumogiri": False},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "5mr", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 0},
        {"type": "ryukyoku"},
        {"type": "end_kyoku"},
    ]
    record = _record_for_first_reach(events)

    with pytest.raises(ValueError):
        analyze_focal_kyoku(replace(record, **{field_name: bad_value}), events)


@pytest.mark.parametrize("malformation", ("missing", "tile", "index"))
def test_focal_analysis_rejects_recorded_river_mismatch(
    malformation: str,
) -> None:
    events = [
        _start(oya=0),
        {"type": "dahai", "actor": 0, "pai": "2m", "tsumogiri": False},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 0},
        {"type": "ryukyoku"},
        {"type": "end_kyoku"},
    ]
    record = _record_for_first_reach(events)
    river = list(record.actor_discards_before_riichi)
    if malformation == "missing":
        river.pop(0)
    elif malformation == "tile":
        river[0] = replace(river[0], tile="3m", normalized_tile="3m")
    else:
        river[0] = replace(river[0], event_index=99)

    with pytest.raises(ValueError):
        analyze_focal_kyoku(
            replace(record, actor_discards_before_riichi=tuple(river)),
            events,
        )


def test_focal_river_matches_call_of_declaration_after_reach_accepted() -> None:
    events = [
        _start(oya=0),
        {"type": "reach", "actor": 2},
        {"type": "dahai", "actor": 2, "pai": "4m", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 2},
        {
            "type": "chi",
            "actor": 3,
            "target": 2,
            "pai": "4m",
            "consumed": ["2m", "3m"],
        },
        {"type": "dahai", "actor": 3, "pai": "8p", "tsumogiri": False},
        {"type": "ryukyoku"},
        {"type": "end_kyoku"},
    ]
    record = _record_for_first_reach(events)
    declaration = replace(
        record.actor_discards_before_riichi[-1],
        was_called=True,
        call_type="chi",
        called_by_actor=3,
        call_event_index=4,
    )
    record = replace(
        record,
        actor_discards_before_riichi=(
            *record.actor_discards_before_riichi[:-1],
            declaration,
        ),
    )

    result = analyze_focal_kyoku(record, events)

    assert result.focal_actor == 2
    assert result.decisions[0].responder == 3


def test_focal_ron_discard_must_be_one_of_the_normalized_wait_tiles() -> None:
    events = [
        _start(oya=0),
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 0},
        {"type": "dahai", "actor": 1, "pai": "6s", "tsumogiri": False},
        {"type": "hora", "actor": 0, "target": 1, "pai": "6s"},
        {"type": "end_kyoku"},
    ]
    record = _record_for_first_reach(events, wait_specs=(("2p", "tanki"),))

    with pytest.raises(ValueError, match="wait|winning tile|ron tile"):
        analyze_focal_kyoku(record, events)


def test_direct_standardization_uses_pooled_common_support_exactly() -> None:
    counts = {
        ("dealer", ("A",)): BinaryCount(8, 10),
        ("nondealer", ("A",)): BinaryCount(1, 2),
        ("dealer", ("B",)): BinaryCount(1, 2),
        ("nondealer", ("B",)): BinaryCount(3, 6),
        ("dealer", ("dealer-only",)): BinaryCount(1, 1),
        ("nondealer", ("dealer-only",)): BinaryCount(0, 0),
    }

    result = standardize_binary_counts(counts)  # type: ignore[arg-type]

    assert type(result.dealer_rate) is Decimal
    assert type(result.nondealer_rate) is Decimal
    assert type(result.difference) is Decimal
    assert result.dealer_rate == Decimal("0.68")
    assert result.nondealer_rate == Decimal("0.5")
    assert result.difference == Decimal("0.18")
    assert result.common_strata == 2
    assert result.union_strata == 3
    assert result.dealer_coverage == Fraction(12, 13)
    assert result.nondealer_coverage == Fraction(1)
    assert result.status == "established"
    assert result.dealer_total_eligible_denominator == 13
    assert result.nondealer_total_eligible_denominator == 8
    assert result.dealer_common_denominator == 12
    assert result.nondealer_common_denominator == 8
    assert result.common_pooled_denominator == 20


def test_decimal_weighted_term_multiplies_integers_before_decimal_rounding() -> None:
    value = defense._decimal_weighted_term(  # noqa: SLF001
        72_531_090_881_370_229_615_073_900_526_333_547_575_203_444_285_636_616_999_101_068_196_291_206_952_028_889,
        3_397_999_960_958_613_908_122_789_715_419_071_464_495_285_924_068_035_357_932_477_608_767_820_975_360_254,
        78_670_559_949_049_913_131_394_476_658_471_849_501_136_901_237_255_612_477_204_366_952_700_074_153_377_047,
        82_936_765_407_523_603_348_328_108_357_960_216_098_317_851_609_362_444_789_888_818_722_713_531_665_445_621,
    )

    assert str(value) == (
        "0.0377735884267254302827494329499289895678764928374475162293272"
    )


def test_decimal_standardization_is_independent_of_ambient_context() -> None:
    counts = {
        ("dealer", ("A",)): BinaryCount(1, 3),
        ("nondealer", ("A",)): BinaryCount(2, 7),
        ("dealer", ("B",)): BinaryCount(4, 9),
        ("nondealer", ("B",)): BinaryCount(5, 11),
    }
    baseline = standardize_binary_counts(counts)  # type: ignore[arg-type]

    with localcontext() as ambient:
        ambient.prec = 7
        ambient.rounding = ROUND_DOWN
        ambient.traps[Inexact] = True
        changed = standardize_binary_counts(counts)  # type: ignore[arg-type]

    assert changed.dealer_rate == baseline.dealer_rate
    assert changed.nondealer_rate == baseline.nondealer_rate
    assert changed.difference == baseline.difference


def test_direct_standardization_returns_none_without_common_support() -> None:
    result = standardize_binary_counts(
        {
            ("dealer", ("A",)): BinaryCount(1, 1),
            ("nondealer", ("B",)): BinaryCount(0, 2),
        }  # type: ignore[arg-type]
    )

    assert result.dealer_rate is None
    assert result.nondealer_rate is None
    assert result.difference is None
    assert result.common_strata == 0
    assert result.union_strata == 2
    assert result.dealer_coverage == 0
    assert result.nondealer_coverage == 0


def test_direct_standardization_empty_input_has_null_coverage_metadata() -> None:
    result = standardize_binary_counts({})

    assert result.dealer_rate is None
    assert result.nondealer_rate is None
    assert result.difference is None
    assert result.common_strata == 0
    assert result.union_strata == 0
    assert result.dealer_coverage is None
    assert result.nondealer_coverage is None
    assert result.status == "not_estimable"
    assert result.dealer_total_eligible_denominator == 0
    assert result.nondealer_total_eligible_denominator == 0
    assert result.dealer_common_denominator == 0
    assert result.nondealer_common_denominator == 0
    assert result.common_pooled_denominator == 0


def test_decimal_weighted_sum_handles_over_5000_digit_equivalent_lcm() -> None:
    primes: list[int] = []
    candidate = 10_007
    while len(primes) < 1_251:
        if all(candidate % divisor for divisor in range(3, isqrt(candidate) + 1, 2)):
            primes.append(candidate)
        candidate += 2
    # Every denominator exceeds 10,000 and is prime, so their product (and the
    # naive aggregate Fraction LCM) exceeds 5,000 decimal digits.
    assert len(primes) * 4 > 5_000
    counts = {
        (role, (index,)): BinaryCount(
            1 if role == "dealer" else 0,
            denominator if role == "dealer" else 1,
        )
        for index, denominator in enumerate(primes)
        for role in ("dealer", "nondealer")
    }

    result = standardize_binary_counts(counts)  # type: ignore[arg-type]
    summary = defense._metric_document(  # noqa: SLF001 - serialization boundary test
        {
            "dealer": BinaryCount(len(primes), sum(primes)),
            "nondealer": BinaryCount(0, len(primes)),
        },
        result,
    )
    rows = [
        [
            index,
            "dealer",
            1,
            denominator,
        ]
        for index, denominator in enumerate(primes)
    ] + [[index, "nondealer", 0, 1] for index in range(len(primes))]
    loaded = json.loads(canonical_json({"summary": summary, "rows": rows}))
    reconstructed = {
        (row[1], (row[0],)): BinaryCount(row[2], row[3]) for row in loaded["rows"]
    }
    recomputed = standardize_binary_counts(reconstructed)  # type: ignore[arg-type]
    recomputed_summary = defense._metric_document(  # noqa: SLF001
        {
            "dealer": BinaryCount(len(primes), sum(primes)),
            "nondealer": BinaryCount(0, len(primes)),
        },
        recomputed,
    )

    adjustment = loaded["summary"]["common_support_standardized"]
    raw_dealer = loaded["summary"]["raw"]["dealer"]
    assert raw_dealer["rate"]["numerator"] == len(primes)
    assert raw_dealer["rate"]["denominator"] == sum(primes)
    assert isinstance(raw_dealer["rate"]["decimal"], str)
    assert raw_dealer["rate"]["rounding"] == "ROUND_HALF_EVEN"
    assert (
        adjustment["dealer_rate"]
        == recomputed_summary["common_support_standardized"]["dealer_rate"]
    )
    assert (
        adjustment["dealer_minus_nondealer"]
        == recomputed_summary["common_support_standardized"]["dealer_minus_nondealer"]
    )
    for field_name in (
        "dealer_rate",
        "nondealer_rate",
        "dealer_minus_nondealer",
    ):
        weighted = adjustment[field_name]
        assert weighted["representation"] == WEIGHTED_SUM_REPRESENTATION
        assert isinstance(weighted["decimal"], str)
        assert weighted["precision_digits"] == DECIMAL_PRECISION_DIGITS
        assert "numerator" not in weighted
        assert "denominator" not in weighted


def test_ron_standardization_support_uses_only_nonfuriten_decisions() -> None:
    dealer_furiten_events = [
        _start(oya=0),
        {"type": "dahai", "actor": 0, "pai": "2p", "tsumogiri": False},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 0},
        {"type": "dahai", "actor": 1, "pai": "6s", "tsumogiri": False},
        {"type": "ryukyoku"},
        {"type": "end_kyoku"},
    ]
    child_nonfuriten_events = [
        _start(oya=0, kyoku=2),
        {"type": "dahai", "actor": 1, "pai": "3p", "tsumogiri": False},
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "5m", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 1},
        {"type": "dahai", "actor": 2, "pai": "6s", "tsumogiri": False},
        {"type": "ryukyoku"},
        {"type": "end_kyoku"},
    ]
    builder = DefenseAnalysisBuilder((2025,))
    for events, source in (
        (dealer_furiten_events, "2025/dealer-furiten.mjson"),
        (child_nonfuriten_events, "2025/child-nonfuriten.mjson"),
    ):
        builder.add(
            analyze_focal_kyoku(
                _record_for_first_reach(
                    events,
                    wait_specs=(("2p", "tanki"),),
                    source_path=source,
                ),
                events,
            )
        )

    metrics = builder.to_dict()["selected"]["all_riichi"]["primary_child_responders"]
    ordinary = metrics["genbutsu_share"]
    ron = metrics["focal_ron_per_discard"]
    assert ordinary["raw"]["dealer"]["denominator"] == 1
    assert ordinary["raw"]["nondealer"]["denominator"] == 1
    assert ordinary["common_support_standardized"]["common_strata"] == 1
    assert ron["raw"]["dealer"]["denominator"] == 0
    assert ron["raw"]["nondealer"]["denominator"] == 1
    adjustment = ron["common_support_standardized"]
    assert adjustment["dealer_rate"] is None
    assert adjustment["nondealer_rate"] is None
    assert adjustment["dealer_minus_nondealer"] is None
    assert adjustment["common_strata"] == 0
    assert adjustment["union_strata"] == 1
    assert adjustment["status"] == "not_estimable"
    assert adjustment["common_pooled_denominator"] == 0
    assert adjustment["eligible_denominators"] == {
        "dealer": {"total": 0, "common": 0, "outside_common_support": 0},
        "nondealer": {"total": 1, "common": 0, "outside_common_support": 1},
    }
    assert adjustment["coverage"] == {
        "dealer": None,
        "nondealer": {
            "numerator": 0,
            "denominator": 1,
            "decimal": "0",
            "precision_digits": DECIMAL_PRECISION_DIGITS,
            "rounding": "ROUND_HALF_EVEN",
        },
    }
    assert "strata" not in ron
    statistics = builder.standardization_sufficient_statistics()
    layout = statistics["units"]["decision"]["row_layout"]
    table = next(
        table
        for table in statistics["tables"]
        if table["year"] == 2025 and table["sensitivity"] == "all_riichi"
    )
    by_role = {
        row[layout.index("focal_role")]: dict(zip(layout, row, strict=True))
        for row in table["decision_rows"]
    }
    assert by_role["dealer"]["focal_ron_per_discard.numerator"] == 0
    assert by_role["dealer"]["focal_ron_per_discard.denominator"] == 0
    assert by_role["nondealer"]["focal_ron_per_discard.numerator"] == 0
    assert by_role["nondealer"]["focal_ron_per_discard.denominator"] == 1


def _without_audits(value: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(value)
    for period in result.values():
        for sensitivity in period.values():
            sensitivity["audit_samples"] = []
    return result


def test_streaming_builder_is_order_invariant_and_normal_sensitivity_excludes_double() -> (
    None
):
    double_events = [
        _start(oya=0),
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 0},
        {"type": "ryukyoku"},
        {"type": "end_kyoku"},
    ]
    normal_events = [
        _start(oya=0, kyoku=2),
        {"type": "dahai", "actor": 1, "pai": "9m", "tsumogiri": False},
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "5p", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 1},
        {"type": "ryukyoku"},
        {"type": "end_kyoku"},
    ]
    focals = (
        analyze_focal_kyoku(
            _record_for_first_reach(double_events, source_path="2025/double.mjson"),
            double_events,
        ),
        analyze_focal_kyoku(
            _record_for_first_reach(normal_events, source_path="2025/normal.mjson"),
            normal_events,
        ),
    )
    forward = DefenseAnalysisBuilder((2025,))
    backward = DefenseAnalysisBuilder((2025,))
    for focal in focals:
        forward.add(focal)
    for focal in reversed(focals):
        backward.add(focal)

    forward_document = forward.to_dict()
    assert _without_audits(forward_document) == _without_audits(backward.to_dict())
    assert forward_document["selected"]["all_riichi"]["focals"] == {
        "dealer": 1,
        "nondealer": 1,
    }
    assert forward_document["selected"]["normal_riichi"]["focals"] == {
        "dealer": 0,
        "nondealer": 1,
    }
    assert forward_document["year_2025"]["all_riichi"]["focals"] == {
        "dealer": 1,
        "nondealer": 1,
    }


def test_annual_sufficient_statistics_reconstruct_a_multi_year_period() -> None:
    dealer_events = [
        _start(oya=0),
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 0},
        {"type": "dahai", "actor": 1, "pai": "6s", "tsumogiri": False},
        {"type": "ryukyoku"},
        {"type": "end_kyoku"},
    ]
    nondealer_events = [
        _start(oya=0, kyoku=2),
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "5p", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 1},
        {"type": "dahai", "actor": 2, "pai": "6s", "tsumogiri": False},
        {"type": "ryukyoku"},
        {"type": "end_kyoku"},
    ]
    base_focals = (
        analyze_focal_kyoku(
            _record_for_first_reach(dealer_events, source_path="2025/dealer.mjson"),
            dealer_events,
        ),
        analyze_focal_kyoku(
            _record_for_first_reach(
                nondealer_events, source_path="2025/nondealer.mjson"
            ),
            nondealer_events,
        ),
    )
    builder = DefenseAnalysisBuilder((2024, 2025))
    for year in (2024, 2025):
        for focal in base_focals:
            source_path = f"{year}/{focal.focal_role}.mjson"
            builder.add(
                replace(
                    focal,
                    year=year,
                    source_path=source_path,
                    decisions=tuple(
                        replace(decision, year=year, source_path=source_path)
                        for decision in focal.decisions
                    ),
                    responders=tuple(
                        replace(responder, year=year) for responder in focal.responders
                    ),
                )
            )

    periods = builder.to_dict()
    statistics = builder.standardization_sufficient_statistics()

    assert builder.period_definitions() == {
        "selected": [2024, 2025],
        "year_2024": [2024],
        "year_2025": [2025],
    }
    assert len(statistics["tables"]) == 4
    assert all(
        "dimensions" not in table and "metric_columns" not in table
        for table in statistics["tables"]
    )
    period_metric = periods["selected"]["all_riichi"]["primary_child_responders"][
        "genbutsu_share"
    ]
    assert "strata" not in period_metric

    unit = statistics["units"]["decision"]
    layout = unit["row_layout"]
    role_index = layout.index("focal_role")
    numerator_index = layout.index("genbutsu_share.numerator")
    denominator_index = layout.index("genbutsu_share.denominator")
    dimension_indexes = [layout.index(name) for name in unit["dimensions"]]
    reconstructed: dict[tuple[str, tuple[Any, ...]], BinaryCount] = {}
    for table in statistics["tables"]:
        if table["sensitivity"] != "all_riichi":
            continue
        for row in table["decision_rows"]:
            stratum = (
                table["year"],
                *(row[index] for index in dimension_indexes),
            )
            reconstructed[(row[role_index], stratum)] = BinaryCount(
                row[numerator_index], row[denominator_index]
            )
    recalculated = standardize_binary_counts(reconstructed)  # type: ignore[arg-type]

    for role in ("dealer", "nondealer"):
        role_counts = [
            value for (key_role, _), value in reconstructed.items() if key_role == role
        ]
        assert period_metric["raw"][role]["numerator"] == sum(
            value.numerator for value in role_counts
        )
        assert period_metric["raw"][role]["denominator"] == sum(
            value.denominator for value in role_counts
        )
    adjustment = period_metric["common_support_standardized"]
    assert adjustment["dealer_rate"] == defense._weighted_decimal_document(  # noqa: SLF001
        recalculated.dealer_rate
    )
    assert adjustment["nondealer_rate"] == defense._weighted_decimal_document(  # noqa: SLF001
        recalculated.nondealer_rate
    )
    assert adjustment["dealer_minus_nondealer"] == (
        defense._weighted_decimal_document(recalculated.difference)  # noqa: SLF001
    )


def test_output_bundle_rolls_back_originals_on_second_publish_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "summary.json"
    second = tmp_path / "summary.md"
    first.write_text("old-json", encoding="utf-8")
    second.write_text("old-markdown", encoding="utf-8")
    real_replace = defense.os.replace
    replace_calls = 0

    def fail_second_publish(source: Path, target: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 4:
            raise OSError("injected second publish failure")
        real_replace(source, target)

    monkeypatch.setattr(defense.os, "replace", fail_second_publish)

    with pytest.raises(OSError, match="injected second publish failure"):
        publish_output_bundle({first: "new-json", second: "new-markdown"})

    assert first.read_text(encoding="utf-8") == "old-json"
    assert second.read_text(encoding="utf-8") == "old-markdown"
    assert set(tmp_path.iterdir()) == {first, second}


@pytest.mark.parametrize("interrupt_after_replace", (1, 2, 3, 4))
def test_output_bundle_rolls_back_keyboard_interrupt_after_every_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_after_replace: int,
) -> None:
    first = tmp_path / "summary.json"
    second = tmp_path / "summary.md"
    first.write_text("old-json", encoding="utf-8")
    second.write_text("old-markdown", encoding="utf-8")
    real_replace = defense.os.replace
    replace_calls = 0

    def interrupt_immediately_after_replace(source: Path, target: Path) -> None:
        nonlocal replace_calls
        real_replace(source, target)
        replace_calls += 1
        if replace_calls == interrupt_after_replace:
            raise KeyboardInterrupt

    monkeypatch.setattr(defense.os, "replace", interrupt_immediately_after_replace)

    with pytest.raises(KeyboardInterrupt):
        publish_output_bundle({first: "new-json", second: "new-markdown"})

    assert first.read_text(encoding="utf-8") == "old-json"
    assert second.read_text(encoding="utf-8") == "old-markdown"
    assert set(tmp_path.iterdir()) == {first, second}


@pytest.mark.parametrize("interrupt_after_replace", (1, 2, 3))
def test_output_bundle_rolls_back_mixed_existing_and_new_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_after_replace: int,
) -> None:
    existing = tmp_path / "summary.json"
    new_target = tmp_path / "summary.md"
    existing.write_text("old-json", encoding="utf-8")
    real_replace = defense.os.replace
    replace_calls = 0

    def interrupt_immediately_after_replace(source: Path, target: Path) -> None:
        nonlocal replace_calls
        real_replace(source, target)
        replace_calls += 1
        if replace_calls == interrupt_after_replace:
            raise KeyboardInterrupt

    monkeypatch.setattr(defense.os, "replace", interrupt_immediately_after_replace)

    with pytest.raises(KeyboardInterrupt):
        publish_output_bundle({existing: "new-json", new_target: "new-markdown"})

    assert existing.read_text(encoding="utf-8") == "old-json"
    assert not new_target.exists()
    assert set(tmp_path.iterdir()) == {existing}


def test_output_bundle_retains_backup_when_rollback_restoration_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "summary.json"
    target.write_text("old-json", encoding="utf-8")
    real_replace = defense.os.replace
    replace_calls = 0

    def fail_publish_then_rollback(source: Path, destination: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("injected publish failure")
        if replace_calls == 3:
            raise OSError("injected rollback failure")
        real_replace(source, destination)

    monkeypatch.setattr(defense.os, "replace", fail_publish_then_rollback)

    with pytest.raises(RuntimeError, match="retained backups") as raised:
        publish_output_bundle({target: "new-json"})

    backups = tuple(tmp_path.glob("*.backup"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "old-json"
    assert str(backups[0]) in str(raised.value)


def test_manifest_and_article1_identity_are_strict() -> None:
    root = Path(__file__).parents[1]
    manifest = json.loads(
        (root / "data/processed/riichi-waits-v1/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    summary = json.loads(
        (root / "outputs/dealer-child-riichi-points/summary-v1.json").read_text(
            encoding="utf-8"
        )
    )

    defense_cli.validate_manifest_identity(manifest)
    defense_cli.validate_summary_identity(summary)

    invalid_manifest = deepcopy(manifest)
    invalid_manifest["source"]["repository"] = "somewhere/else"
    with pytest.raises(ValueError, match="repository"):
        defense_cli.validate_manifest_identity(invalid_manifest)

    invalid_manifest = deepcopy(manifest)
    invalid_manifest["scope"]["years"][1] = 2009
    with pytest.raises(ValueError, match="unique"):
        defense_cli.validate_manifest_identity(invalid_manifest)

    invalid_summary = deepcopy(summary)
    invalid_summary["scope"]["bakaze"] = "S"
    with pytest.raises(ValueError, match="bakaze"):
        defense_cli.validate_summary_identity(invalid_summary)

    invalid_summary = deepcopy(summary)
    invalid_summary["years"] = list(reversed(invalid_summary["years"]))
    with pytest.raises(ValueError, match="ordered"):
        defense_cli.validate_summary_identity(invalid_summary)


def _oracle_counts_from_year_entry(
    entry: dict[str, Any],
) -> defense_cli.ReconciliationCounts:
    counts = defense_cli.ReconciliationCounts()
    for sensitivity, container in (
        ("all_riichi", entry),
        ("normal_riichi", entry["sensitivity_normal_riichi_only"]),
    ):
        for role in ("dealer", "nondealer"):
            for field_name in (
                "riichis",
                "wins",
                "other_wins",
                "draws",
                "tsumo_wins",
                "ron_wins",
            ):
                counts.values[(sensitivity, role, field_name)] = container[role][
                    field_name
                ]
    return counts


def test_article1_selected_period_is_reconciled_only_for_its_member_years() -> None:
    root = Path(__file__).parents[1]
    summary = json.loads(
        (root / "outputs/dealer-child-riichi-points/summary-v1.json").read_text(
            encoding="utf-8"
        )
    )
    invalid = deepcopy(summary)
    invalid["periods"]["selected"]["dealer"]["riichis"] += 1
    annual = {
        entry["year"]: _oracle_counts_from_year_entry(entry)
        for entry in summary["years"]
    }

    with pytest.raises(ValueError, match="period selected"):
        defense_cli.reconcile_periods(
            annual,
            defense_cli.SUPPORTED_YEARS,
            invalid,
        )

    subset = defense_cli.reconcile_periods(
        {2025: annual[2025]},
        (2025,),
        invalid,
    )
    assert subset["selected"] == {
        "status": "skipped",
        "reason": "cli_selected_years_do_not_equal_article1_selected_years",
        "required_years": list(defense_cli.SUPPORTED_YEARS),
    }


def test_source_processing_reconciles_every_riichi_but_uses_first_as_focal() -> None:
    kyoku = [
        _start(oya=0),
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 0},
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "6m", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 1},
        {"type": "ryukyoku"},
        {"type": "end_kyoku"},
    ]
    events = [{"type": "start_game", "aka_flag": True}, *kyoku, {"type": "end_game"}]
    first = _record_for_first_reach(
        kyoku,
        source_path=_VALID_SOURCE,
        start_kyoku_line=2,
    )
    second_reach = extract_established_riichis(kyoku)[1]
    second = replace(
        first,
        actor=1,
        reach_event_index=second_reach.reach_event_index,
        declaration_dahai_event_index=second_reach.reach_event_index + 1,
        reach_accepted_event_index=second_reach.reach_accepted_event_index,
        reach_line=2 + second_reach.reach_event_index,
        declaration_dahai_line=3 + second_reach.reach_event_index,
        reach_accepted_line=2 + second_reach.reach_accepted_event_index,
        riichi_declaration_tile="6m",
        riichi_declaration_tile_kind="6m",
        actor_discards_before_riichi=(
            DatasetActorDiscard(
                discard_number=1,
                tile="6m",
                normalized_tile="6m",
                tsumogiri=False,
                event_index=second_reach.reach_event_index + 1,
                is_riichi_declaration=True,
                was_called=False,
                call_type=None,
                called_by_actor=None,
                call_event_index=None,
            ),
        ),
    )
    builder = DefenseAnalysisBuilder((2025,))
    reconciliation = defense_cli.ReconciliationCounts()

    focal_count = defense_cli.process_source_file(
        year=2025,
        source_path=first.relative_source_path,
        events=events,
        records=(first, second),
        builder=builder,
        reconciliation=reconciliation,
    )

    assert focal_count == 1
    assert builder.to_dict()["selected"]["all_riichi"]["focals"] == {
        "dealer": 1,
        "nondealer": 0,
    }
    assert reconciliation.group("all_riichi", "dealer")["draws"] == 1
    assert reconciliation.group("all_riichi", "nondealer")["draws"] == 1


def _draw_riichi_kyoku(*, kyoku: int, bakaze: str = "E") -> list[dict[str, Any]]:
    return [
        {
            **_start(oya=0, kyoku=kyoku),
            "bakaze": bakaze,
        },
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 0},
        {"type": "ryukyoku"},
        {"type": "end_kyoku"},
    ]


@pytest.mark.parametrize("record_mode", ("missing", "extra"))
def test_source_processing_requires_bidirectional_identity_for_every_east_hand(
    record_mode: str,
) -> None:
    first_kyoku = _draw_riichi_kyoku(kyoku=1)
    second_kyoku = _draw_riichi_kyoku(kyoku=2)
    second_start_line = 2 + len(first_kyoku)
    first_record = _record_for_first_reach(
        first_kyoku,
        source_path=_VALID_SOURCE,
        start_kyoku_line=2,
    )
    second_record = _record_for_first_reach(
        second_kyoku,
        source_path=_VALID_SOURCE,
        start_kyoku_line=second_start_line,
    )
    if record_mode == "missing":
        records = (first_record,)
    else:
        records = (first_record, first_record, second_record)
    events = [
        {"type": "start_game", "aka_flag": True},
        *first_kyoku,
        *second_kyoku,
        {"type": "end_game"},
    ]

    with pytest.raises(ValueError, match="established-riichi|wait record|identity"):
        defense_cli.process_source_file(
            year=2025,
            source_path=_VALID_SOURCE,
            events=events,
            records=records,
            builder=DefenseAnalysisBuilder((2025,)),
            reconciliation=defense_cli.ReconciliationCounts(),
        )


def test_source_processing_does_not_require_wait_records_for_south_hands() -> None:
    east = _draw_riichi_kyoku(kyoku=1)
    south = _draw_riichi_kyoku(kyoku=1, bakaze="S")
    record = _record_for_first_reach(
        east,
        source_path=_VALID_SOURCE,
        start_kyoku_line=2,
    )
    events = [
        {"type": "start_game", "aka_flag": True},
        *east,
        *south,
        {"type": "end_game"},
    ]

    assert (
        defense_cli.process_source_file(
            year=2025,
            source_path=_VALID_SOURCE,
            events=events,
            records=(record,),
            builder=DefenseAnalysisBuilder((2025,)),
            reconciliation=defense_cli.ReconciliationCounts(),
        )
        == 1
    )


@pytest.mark.parametrize(
    ("source_path", "aka_flag"),
    (
        ("2025/2025010100gm-00e1-0000-00000001.mjson", True),
        (_VALID_SOURCE, False),
    ),
)
def test_source_processing_rejects_wrong_raw_game_rule_or_aka(
    source_path: str,
    aka_flag: bool,
) -> None:
    kyoku = _draw_riichi_kyoku(kyoku=1)
    record = _record_for_first_reach(
        kyoku,
        source_path=source_path,
        start_kyoku_line=2,
    )
    events = [
        {"type": "start_game", "aka_flag": aka_flag},
        *kyoku,
        {"type": "end_game"},
    ]

    with pytest.raises(ValueError, match="target|rule|aka"):
        defense_cli.process_source_file(
            year=2025,
            source_path=source_path,
            events=events,
            records=(record,),
            builder=DefenseAnalysisBuilder((2025,)),
            reconciliation=defense_cli.ReconciliationCounts(),
        )


def test_source_group_streaming_and_partial_cap(tmp_path: Path) -> None:
    dataset_root = tmp_path / "waits"
    dataset_root.mkdir()
    base_events = [
        _start(oya=0),
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 0},
        {"type": "ryukyoku"},
        {"type": "end_kyoku"},
    ]
    records = (
        _record_for_first_reach(base_events, source_path="2025/a.mjson"),
        _record_for_first_reach(base_events, source_path="2025/b.mjson"),
    )
    with gzip.open(dataset_root / "2025.jsonl.gz", "wt", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(dataset_record_to_dict(record), sort_keys=True) + "\n"
            )

    groups = list(
        defense_cli.iter_source_groups(dataset_root, 2025, partial=True, max_files=1)
    )

    assert [(path, len(group)) for path, group in groups] == [("2025/a.mjson", 1)]


def test_reconciliation_mismatch_and_partial_default_output_guard() -> None:
    counts = defense_cli.ReconciliationCounts()
    counts.add("dealer", "riichi", "draws", None)
    summary = {
        "years": [
            {
                "year": 2025,
                "dealer": {
                    "riichis": 1,
                    "wins": 0,
                    "other_wins": 0,
                    "draws": 0,
                    "tsumo_wins": 0,
                    "ron_wins": 0,
                },
                "nondealer": {
                    "riichis": 0,
                    "wins": 0,
                    "other_wins": 0,
                    "draws": 0,
                    "tsumo_wins": 0,
                    "ron_wins": 0,
                },
                "sensitivity_normal_riichi_only": {
                    "dealer": {
                        "riichis": 1,
                        "wins": 0,
                        "other_wins": 0,
                        "draws": 0,
                        "tsumo_wins": 0,
                        "ron_wins": 0,
                    },
                    "nondealer": {
                        "riichis": 0,
                        "wins": 0,
                        "other_wins": 0,
                        "draws": 0,
                        "tsumo_wins": 0,
                        "ron_wins": 0,
                    },
                },
            }
        ]
    }
    with pytest.raises(ValueError, match="reconciliation mismatch"):
        defense_cli.reconcile_year(2025, counts, summary)

    with pytest.raises(SystemExit):
        defense_cli.parse_args(["--years", "2025", "--max-files", "1"])


def test_partial_default_output_guard_is_independent_of_current_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_default = (
        Path(__file__).parents[1] / defense_cli.DEFAULT_OUTPUT_DIRECTORY
    ).resolve()
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit):
        defense_cli.parse_args(
            [
                "--years",
                "2025",
                "--max-files",
                "1",
                "--output-dir",
                str(repository_default),
            ]
        )


def _dummy_source_task(index: int) -> Any:
    source_path = f"2025/2025010100gm-00a9-0000-{index:08d}.mjson"
    return defense_cli.SourceTask(
        year=2025,
        source_path=source_path,
        raw_path=Path(source_path),
        records=(),
    )


def _dummy_source_result(task: Any) -> Any:
    return defense_cli.SourceResult(
        year=task.year,
        source_path=task.source_path,
        raw_sha256="0" * 64,
        raw_bytes=0,
        raw_events=0,
        wait_records=0,
        observations=(),
        reconciliation_items=(),
    )


def test_parallel_results_preserve_submission_order_when_futures_finish_reverse() -> (
    None
):
    tasks = tuple(_dummy_source_task(index) for index in range(1, 5))
    completion_order: list[str] = []

    class ReverseCompletionExecutor:
        def __init__(self) -> None:
            self.submissions: list[tuple[tuple[Any, ...], Future[Any]]] = []

        def submit(self, function: Any, batch: tuple[Any, ...]) -> Future[Any]:
            assert function is defense_cli.process_source_batch
            future: Future[Any] = Future()
            self.submissions.append((batch, future))
            if len(self.submissions) == 4:
                for submitted, submitted_future in reversed(self.submissions):
                    completion_order.append(submitted[0].source_path)
                    submitted_future.set_result(
                        tuple(_dummy_source_result(task) for task in submitted)
                    )
            return future

    executor = ReverseCompletionExecutor()
    results = list(
        defense_cli._iter_parallel_source_results(
            iter(tasks), executor=executor, workers=2, batch_size=1
        )
    )

    assert completion_order == [task.source_path for task in reversed(tasks)]
    assert [result.source_path for result in results] == [
        task.source_path for task in tasks
    ]


def test_parallel_submission_is_bounded_by_workers_times_two() -> None:
    tasks = tuple(_dummy_source_task(index) for index in range(1, 26))

    class TrackingFuture:
        def __init__(self, owner: Any, batch: tuple[Any, ...]) -> None:
            self.owner = owner
            self.batch = batch
            self.finished = False

        def result(self) -> tuple[Any, ...]:
            if not self.finished:
                self.finished = True
                self.owner.outstanding -= 1
            return tuple(_dummy_source_result(task) for task in self.batch)

        def cancel(self) -> bool:
            if not self.finished:
                self.finished = True
                self.owner.outstanding -= 1
            return True

    class TrackingExecutor:
        def __init__(self) -> None:
            self.outstanding = 0
            self.maximum = 0

        def submit(self, function: Any, batch: tuple[Any, ...]) -> TrackingFuture:
            assert function is defense_cli.process_source_batch
            self.outstanding += 1
            self.maximum = max(self.maximum, self.outstanding)
            return TrackingFuture(self, batch)

    executor = TrackingExecutor()
    results = list(
        defense_cli._iter_parallel_source_results(
            iter(tasks), executor=executor, workers=3, batch_size=2
        )
    )

    assert len(results) == len(tasks)
    assert executor.maximum == 6
    assert executor.outstanding == 0


def test_parallel_failure_has_source_context_and_cancels_pending() -> None:
    tasks = tuple(_dummy_source_task(index) for index in range(1, 5))

    class FailingExecutor:
        def __init__(self) -> None:
            self.futures: list[Future[Any]] = []

        def submit(self, function: Any, batch: tuple[Any, ...]) -> Future[Any]:
            assert function is defense_cli.process_source_batch
            future: Future[Any] = Future()
            if not self.futures:
                future.set_exception(RuntimeError(f"{batch[0].source_path}: boom"))
            self.futures.append(future)
            return future

    executor = FailingExecutor()
    with pytest.raises(RuntimeError, match=tasks[0].source_path):
        list(
            defense_cli._iter_parallel_source_results(
                iter(tasks), executor=executor, workers=2, batch_size=1
            )
        )

    assert all(future.cancelled() for future in executor.futures[1:])

    missing = replace(tasks[0], raw_path=Path("definitely-missing.mjson"))
    with pytest.raises(RuntimeError, match=missing.source_path):
        defense_cli.process_source_task(missing)


def test_parallel_cli_arguments_are_positive_and_default_to_serial() -> None:
    args = defense_cli.parse_args(["--years", "2025"])
    assert args.workers == 1
    assert args.batch_size == 32

    for option in ("--workers", "--batch-size"):
        with pytest.raises(SystemExit):
            defense_cli.parse_args(["--years", "2025", option, "0"])

    with pytest.raises(ValueError, match="workers"):
        defense_cli.run_analysis(
            years=(2025,),
            dataset_root=Path("unused"),
            raw_root=Path("unused"),
            point_summary_path=Path("unused"),
            max_files=1,
            workers=True,
        )


def _write_parallel_cli_fixture(tmp_path: Path) -> tuple[Path, Path]:
    dataset_root = tmp_path / "waits"
    raw_root = tmp_path / "raw"
    (raw_root / "2025").mkdir(parents=True)
    dataset_root.mkdir()
    records: list[RiichiWaitDatasetRecord] = []
    for source_number in (1, 2):
        source_path = f"2025/2025010100gm-00a9-0000-{source_number:08d}.mjson"
        kyoku = [
            _start(oya=0),
            {"type": "reach", "actor": 0},
            {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
            {"type": "reach_accepted", "actor": 0},
            {"type": "ryukyoku"},
            {"type": "end_kyoku"},
        ]
        events = [
            {"type": "start_game", "aka_flag": True},
            *kyoku,
            {"type": "end_game"},
        ]
        raw_payload = b"".join(
            json.dumps(event, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            for event in events
        )
        (raw_root / source_path).write_bytes(raw_payload)
        records.append(
            _record_for_first_reach(
                kyoku,
                source_path=source_path,
                start_kyoku_line=2,
            )
        )

    annual_path = dataset_root / "2025.jsonl.gz"
    with annual_path.open("wb") as raw_file:
        with gzip.GzipFile(
            filename="", mode="wb", compresslevel=6, mtime=0, fileobj=raw_file
        ) as compressed:
            for record in records:
                compressed.write(
                    json.dumps(
                        dataset_record_to_dict(record),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                    + b"\n"
                )
    annual_bytes = annual_path.read_bytes()
    year_entries: list[dict[str, Any]] = []
    count_fields = (
        "scanned_files",
        "target_games",
        "east_kyokus",
        "established_riichis",
        "output_records",
    )
    for year in defense_cli.SUPPORTED_YEARS:
        count = len(records) if year == 2025 else 0
        year_entries.append(
            {
                "year": year,
                **{field: count for field in count_fields},
                "output_filename": f"{year}.jsonl.gz",
                "compressed_size_bytes": len(annual_bytes) if year == 2025 else 0,
                "sha256": (
                    hashlib.sha256(annual_bytes).hexdigest()
                    if year == 2025
                    else "0" * 64
                ),
                "elapsed_seconds": 0.0,
            }
        )
    totals = {
        field: sum(entry[field] for entry in year_entries) for field in count_fields
    }
    manifest = {
        "schema_version": 1,
        "dataset_name": "riichi-waits-v1",
        "created_at_utc": "2026-09-23T00:00:00Z",
        "source": {
            "repository": "NikkeTryHard/tenhou-to-mjai",
            "release_tag": "v2.0.0",
            "validation_summary_path": "data/validation/summary.json",
            "validation_summary_sha256": "1" * 64,
            "archive_hashes_verified": True,
        },
        "scope": {
            "years": list(defense_cli.SUPPORTED_YEARS),
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
            "git_commit": "parallel-test",
            "worktree_clean": True,
            "python_version": "3.12.10",
            "zlib_version": "1.3.1",
            "invocation": {
                "years": list(defense_cli.SUPPORTED_YEARS),
                "mode": "full",
                "max_files": None,
            },
        },
        "years": year_entries,
        "totals": totals,
    }
    (dataset_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return dataset_root, raw_root


def test_workers_one_and_eight_publish_identical_bytes(tmp_path: Path) -> None:
    dataset_root, raw_root = _write_parallel_cli_fixture(tmp_path)
    repository_root = Path(__file__).parents[1]
    outputs = (tmp_path / "serial", tmp_path / "parallel")
    configurations = ((1, 32), (8, 1))
    for output, (workers, batch_size) in zip(outputs, configurations, strict=True):
        completed = subprocess.run(
            [
                sys.executable,
                str(_CLI_PATH),
                "--years",
                "2025",
                "--max-files",
                "2",
                "--workers",
                str(workers),
                "--batch-size",
                str(batch_size),
                "--dataset-root",
                str(dataset_root),
                "--raw-root",
                str(raw_root),
                "--output-dir",
                str(output),
            ],
            cwd=repository_root,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr

    assert (outputs[0] / "summary-v3.json").read_bytes() == (
        outputs[1] / "summary-v3.json"
    ).read_bytes()
    assert (outputs[0] / "summary-v3.md").read_bytes() == (
        outputs[1] / "summary-v3.md"
    ).read_bytes()
    document = json.loads((outputs[1] / "summary-v3.json").read_text())
    assert set(document["run_mode"]) == {
        "partial",
        "max_files",
        "article1_reconciliation",
    }
