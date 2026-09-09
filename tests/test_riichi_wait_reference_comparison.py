"""Tests for the formal production/reference comparison bridge."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

import mahjong_analysis.riichi_wait_reference_comparison as comparison_module
from analysis.validate_riichi_wait_reference import (
    render_comparison_report,
    select_input_files,
)
from mahjong_analysis.riichi_wait_reference_comparison import (
    ComparableDiscard,
    ComparableMeld,
    ComparableRiichiCandidate,
    ComparableWaitDetail,
    RiichiWaitComparisonReport,
    compare_candidates,
    compare_riichi_wait_files,
)


def candidate(
    *,
    source_path: str = "2025/2025010100gm-00a9-0000-1234abcd.mjson",
    start_line: int = 2,
    reach_line: int = 8,
) -> ComparableRiichiCandidate:
    return ComparableRiichiCandidate(
        relative_source_path=source_path,
        start_kyoku_line=start_line,
        reach_line=reach_line,
        actor=1,
        bakaze="E",
        kyoku=1,
        honba=0,
        oya=0,
        declaration_dahai_line=9,
        reach_accepted_line=10,
        riichi_discard_number=2,
        riichi_declaration_tile="5mr",
        riichi_declaration_tile_kind="5m",
        concealed_tiles_after_discard=(
            "5mr",
            "1m",
            "2m",
            "3m",
            "1p",
            "2p",
            "3p",
            "7p",
            "8p",
            "9p",
            "E",
            "E",
            "4s",
        ),
        fixed_melds=(
            ComparableMeld("ankan", ("9m",) * 4),
            ComparableMeld("ankan", ("1s",) * 4),
        ),
        actor_discards_before_riichi=(
            ComparableDiscard(1, "9m", "9m", False, 2, False, True, "chi", 2, 3),
            ComparableDiscard(2, "5mr", "5m", True, 7, True, False, None, None, None),
        ),
        wait_tiles=("6s", "3s"),
        wait_tile_count=2,
        wait_details=(
            ComparableWaitDetail("6s", "standard", "ryanmen"),
            ComparableWaitDetail("3s", "standard", "ryanmen"),
        ),
        wait_shapes=("ryanmen",),
        contains_ryanmen=True,
        is_pure_ryanmen=True,
        is_multiwait=False,
    )


def side_result(
    source_path: str,
    *,
    target_game: bool = True,
    east_lines: tuple[int, ...] = (2,),
    candidates: tuple[ComparableRiichiCandidate, ...] | None = None,
):
    if candidates is None:
        candidates = (candidate(source_path=source_path),)
    return comparison_module._SideFileResult(
        source_path,
        target_game,
        east_lines,
        candidates,
    )


def test_complete_candidate_match_passes() -> None:
    production = candidate()
    reference = candidate()

    result = compare_candidates((production,), (reference,))

    assert result.is_pass is True
    assert result.production_candidates == result.reference_candidates == 1
    assert result.production_only == ()
    assert result.reference_only == ()
    assert result.field_mismatches == ()


def test_reports_production_only_candidate() -> None:
    production = candidate()

    result = compare_candidates((production,), ())

    assert result.is_pass is False
    assert result.production_only[0].key == production.candidate_key
    assert result.reference_only == ()


def test_reports_reference_only_candidate() -> None:
    reference = candidate()

    result = compare_candidates((), (reference,))

    assert result.is_pass is False
    assert result.reference_only[0].key == reference.candidate_key
    assert result.production_only == ()


def test_reports_actor_only_field_mismatch_with_trace_context() -> None:
    production = candidate()
    reference = replace(production, actor=2)

    result = compare_candidates((production,), (reference,))

    assert tuple(value.field_name for value in result.field_mismatches) == ("actor",)
    mismatch = result.field_mismatches[0]
    assert (mismatch.production_value, mismatch.reference_value) == (1, 2)
    assert (mismatch.production_actor, mismatch.reference_actor) == (1, 2)
    assert mismatch.source_path == production.relative_source_path
    assert mismatch.production_lines == production.related_lines
    assert mismatch.reference_lines == reference.related_lines


def test_raw_red_representation_difference_is_not_normalized_away() -> None:
    production = candidate()
    reference = replace(
        production,
        concealed_tiles_after_discard=tuple(
            "5m" if tile == "5mr" else tile
            for tile in production.concealed_tiles_after_discard
        ),
    )

    result = compare_candidates((production,), (reference,))

    assert tuple(value.field_name for value in result.field_mismatches) == (
        "concealed_tiles_after_discard",
    )
    assert "5mr" in result.field_mismatches[0].production_value
    assert "5m" in result.field_mismatches[0].reference_value


def test_concealed_order_difference_is_ignored_without_losing_duplicates() -> None:
    production = candidate()
    reference = replace(
        production,
        concealed_tiles_after_discard=tuple(
            reversed(production.concealed_tiles_after_discard)
        ),
    )

    result = compare_candidates((production,), (reference,))

    assert result.is_pass is True
    assert production.concealed_tiles_after_discard.count("E") == 2


def test_concealed_multiset_difference_is_reported() -> None:
    production = candidate()
    reference = replace(
        production,
        concealed_tiles_after_discard=(
            "2s",
            *production.concealed_tiles_after_discard[1:],
        ),
    )

    result = compare_candidates((production,), (reference,))

    assert tuple(value.field_name for value in result.field_mismatches) == (
        "concealed_tiles_after_discard",
    )


def test_fixed_meld_order_difference_is_ignored() -> None:
    production = candidate()
    reference = replace(production, fixed_melds=tuple(reversed(production.fixed_melds)))

    assert compare_candidates((production,), (reference,)).is_pass is True


def test_wait_tile_and_detail_order_differences_are_ignored() -> None:
    production = candidate()
    reference = replace(
        production,
        wait_tiles=tuple(reversed(production.wait_tiles)),
        wait_details=tuple(reversed(production.wait_details)),
    )

    assert compare_candidates((production,), (reference,)).is_pass is True


def test_wait_tile_duplicates_are_not_hidden_by_normalization() -> None:
    production = candidate()
    reference = replace(production, wait_tiles=(*production.wait_tiles, "3s"))

    result = compare_candidates((production,), (reference,))

    assert tuple(value.field_name for value in result.field_mismatches) == (
        "wait_tiles",
    )


def test_missing_wait_detail_is_reported() -> None:
    production = candidate()
    reference = replace(production, wait_details=production.wait_details[:-1])

    result = compare_candidates((production,), (reference,))

    assert tuple(value.field_name for value in result.field_mismatches) == (
        "wait_details",
    )


def test_added_wait_shape_is_reported() -> None:
    production = candidate()
    reference = replace(production, wait_shapes=("tanki", *production.wait_shapes))

    result = compare_candidates((production,), (reference,))

    assert tuple(value.field_name for value in result.field_mismatches) == (
        "wait_shapes",
    )


def test_boolean_difference_is_reported() -> None:
    production = candidate()
    reference = replace(production, contains_ryanmen=False)

    result = compare_candidates((production,), (reference,))

    assert tuple(value.field_name for value in result.field_mismatches) == (
        "contains_ryanmen",
    )


def test_river_order_difference_is_not_normalized_away() -> None:
    production = candidate()
    reference = replace(
        production,
        actor_discards_before_riichi=tuple(
            reversed(production.actor_discards_before_riichi)
        ),
    )

    result = compare_candidates((production,), (reference,))

    assert tuple(value.field_name for value in result.field_mismatches) == (
        "actor_discards_before_riichi",
    )


def test_river_call_metadata_difference_is_reported() -> None:
    production = candidate()
    first, declaration = production.actor_discards_before_riichi
    reference = replace(
        production,
        actor_discards_before_riichi=(
            replace(
                first,
                was_called=False,
                call_type=None,
                called_by_actor=None,
                call_event_index=None,
            ),
            declaration,
        ),
    )

    result = compare_candidates((production,), (reference,))

    assert tuple(value.field_name for value in result.field_mismatches) == (
        "actor_discards_before_riichi",
    )


def test_one_field_mismatch_is_retained_for_each_differing_field() -> None:
    production = candidate()
    reference = replace(
        production,
        actor=3,
        honba=2,
        is_multiwait=True,
    )

    result = compare_candidates((production,), (reference,))

    assert tuple(value.field_name for value in result.field_mismatches) == (
        "actor",
        "honba",
        "is_multiwait",
    )


@pytest.mark.parametrize("side", ("production", "reference"))
def test_duplicate_candidate_key_is_rejected(side: str) -> None:
    duplicate = candidate()

    with pytest.raises(ValueError, match=rf"duplicate {side} candidate key"):
        compare_candidates(
            (duplicate, duplicate) if side == "production" else (),
            (duplicate, duplicate) if side == "reference" else (),
        )


@pytest.mark.parametrize(
    ("production_scope", "reference_scope", "field_name"),
    (
        ((True, (2,)), (False, ()), "target_game"),
        ((True, (2,)), (True, (3,)), "east_kyoku_start_lines"),
    ),
    ids=("target-game", "east-kyokus"),
)
def test_scope_mismatch_is_classified_without_candidate_cascade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    production_scope: tuple[bool, tuple[int, ...]],
    reference_scope: tuple[bool, tuple[int, ...]],
    field_name: str,
) -> None:
    path = tmp_path / "game.mjson"
    source = path.name
    production = side_result(
        source,
        target_game=production_scope[0],
        east_lines=production_scope[1],
        candidates=(candidate(source_path=source),) if production_scope[0] else (),
    )
    reference = side_result(
        source,
        target_game=reference_scope[0],
        east_lines=reference_scope[1],
        candidates=(candidate(source_path=source),) if reference_scope[0] else (),
    )
    monkeypatch.setattr(
        comparison_module, "_process_production_file", lambda *_: production
    )
    monkeypatch.setattr(
        comparison_module, "_process_reference_file", lambda *_: reference
    )

    report = compare_riichi_wait_files((path,), raw_root=tmp_path)

    assert report.is_pass is False
    assert tuple(value.field_name for value in report.scope_mismatches) == (field_name,)
    assert report.production_only == ()
    assert report.reference_only == ()


@pytest.mark.parametrize(
    ("failing_sides", "expected_production", "expected_reference"),
    (
        (("production",), True, False),
        (("reference",), False, True),
        (("production", "reference"), True, True),
    ),
    ids=("production", "reference", "both"),
)
def test_processing_errors_are_classified_without_candidate_cascade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_sides: tuple[str, ...],
    expected_production: bool,
    expected_reference: bool,
) -> None:
    path = tmp_path / "game.mjson"
    good = side_result(path.name)

    def processor(side: str):
        def run(*_args: object):
            if side in failing_sides:
                raise RuntimeError(f"{side} failed")
            return good

        return run

    monkeypatch.setattr(
        comparison_module, "_process_production_file", processor("production")
    )
    monkeypatch.setattr(
        comparison_module, "_process_reference_file", processor("reference")
    )

    report = compare_riichi_wait_files((path,), raw_root=tmp_path)

    assert report.is_pass is False
    assert len(report.processing_errors) == 1
    error = report.processing_errors[0]
    assert (error.production_error is not None) is expected_production
    assert (error.reference_error is not None) is expected_reference
    assert report.production_only == ()
    assert report.reference_only == ()


def test_artificial_mjai_matches_through_both_complete_pipelines(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    path = raw_root / "2025" / "2025010100gm-00a9-0000-1234abcd.mjson"
    hand = ["1m", "2m", "3m", "1p", "2p", "3p", "7p", "8p", "9p", "E", "E", "4s", "5s"]
    events = (
        {"type": "start_game", "aka_flag": True},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "tehais": [list(hand) for _ in range(4)],
        },
        {"type": "tsumo", "actor": 1, "pai": "9s"},
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "9s", "tsumogiri": True},
        {"type": "reach_accepted", "actor": 1},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        "".join(f"{json.dumps(event, separators=(',', ':'))}\n" for event in events),
        encoding="utf-8",
    )

    production = comparison_module._process_production_file(path, raw_root)
    reference = comparison_module._process_reference_file(path, raw_root)
    assert production.target_game is reference.target_game is True
    assert production.east_kyoku_start_lines == reference.east_kyoku_start_lines == (2,)
    assert production.candidates == reference.candidates
    normalized = production.candidates[0]
    assert normalized.candidate_key == (
        "2025/2025010100gm-00a9-0000-1234abcd.mjson",
        2,
        4,
    )
    assert (
        normalized.actor,
        normalized.bakaze,
        normalized.kyoku,
        normalized.honba,
        normalized.oya,
        normalized.declaration_dahai_line,
        normalized.reach_accepted_line,
        normalized.riichi_discard_number,
        normalized.riichi_declaration_tile,
        normalized.riichi_declaration_tile_kind,
    ) == (1, "E", 1, 0, 0, 5, 6, 1, "9s", "9s")
    assert normalized.fixed_melds == ()
    assert normalized.actor_discards_before_riichi == (
        ComparableDiscard(1, "9s", "9s", True, 3, True, False, None, None, None),
    )
    assert normalized.wait_tiles == ("3s", "6s")
    assert normalized.wait_details == (
        ComparableWaitDetail("3s", "standard", "ryanmen"),
        ComparableWaitDetail("6s", "standard", "ryanmen"),
    )
    assert (
        normalized.wait_tile_count,
        normalized.wait_shapes,
        normalized.contains_ryanmen,
        normalized.is_pure_ryanmen,
        normalized.is_multiwait,
    ) == (2, ("ryanmen",), True, True, False)

    report = compare_riichi_wait_files((path,), raw_root=raw_root)

    assert report.is_pass is True
    assert (
        report.scanned_files,
        report.production_target_games,
        report.reference_target_games,
        report.production_east_kyokus,
        report.reference_east_kyokus,
        report.production_candidates,
        report.reference_candidates,
    ) == (1, 1, 1, 1, 1, 1, 1)


def test_max_files_selection_sorts_relative_paths_before_scope_filtering(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    paths = (
        raw_root / "2025" / "z" / "c.mjson",
        raw_root / "2025" / "a.mjson",
        raw_root / "2025" / "z" / "b.mjson",
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not inspected during selection\n", encoding="utf-8")

    selected = select_input_files(raw_root, 2025, max_files=2)

    assert tuple(path.relative_to(raw_root).as_posix() for path in selected) == (
        "2025/a.mjson",
        "2025/z/b.mjson",
    )


def test_explicit_files_are_resolved_and_sorted(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    first = raw_root / "2025" / "a.mjson"
    second = raw_root / "2025" / "b.mjson"
    first.parent.mkdir(parents=True)
    first.write_text("{}\n", encoding="utf-8")
    second.write_text("{}\n", encoding="utf-8")

    selected = select_input_files(
        raw_root,
        2025,
        max_files=None,
        files=("b.mjson", "2025/a.mjson"),
    )

    assert selected == (first.resolve(), second.resolve())


def test_report_renders_all_counts_and_strict_result() -> None:
    report = RiichiWaitComparisonReport(
        scanned_files=1,
        production_target_games=1,
        reference_target_games=1,
        production_east_kyokus=1,
        reference_east_kyokus=1,
        production_candidates=1,
        reference_candidates=0,
        production_only=compare_candidates((candidate(),), ()).production_only,
        reference_only=(),
        field_mismatches=(),
        scope_mismatches=(),
        processing_errors=(),
    )

    rendered = render_comparison_report(report)

    assert "scanned_files: 1" in rendered
    assert "production_candidates: 1" in rendered
    assert "reference_candidates: 0" in rendered
    assert "production_only: 1" in rendered
    assert "reference_only: 0" in rendered
    assert "field_mismatch: 0" in rendered
    assert "scope_mismatch: 0" in rendered
    assert "processing_error: 0" in rendered
    assert "result: FAIL" in rendered
    assert "[production_only]" in rendered
