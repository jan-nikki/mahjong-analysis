import json
from dataclasses import replace
from pathlib import Path

import pytest

import mahjong_analysis.yearly_aggregation as yearly_aggregation
from analysis.dealer_double_riichi_win_rate_yearly import (
    main,
    parse_args,
    selected_years_from_args,
)
from mahjong_analysis.yearly_aggregation import (
    SUPPORTED_YEARS,
    DatasetValidationMetadata,
    OutputConsistencyError,
    YearlyDealerDoubleRiichiResult,
    aggregate_dealer_double_riichi_years,
    load_dataset_validation_metadata,
    load_known_results,
    normalize_years,
    render_yearly_markdown,
    write_yearly_outputs,
    yearly_summary_to_dict,
)


KNOWN_RESULTS_PATH = (
    Path(__file__).parents[1]
    / "data"
    / "validation"
    / "dealer-double-riichi-v2.0.0-known-results.json"
)
DATASET_SUMMARY_PATH = (
    Path(__file__).parents[1]
    / "data"
    / "validation"
    / "tenhou-to-mjai-v2.0.0-summary.json"
)


def _filename(year: int, index: int, *, rule_code: str = "00a9") -> str:
    return f"{year}010100gm-{rule_code}-0000-{index:08x}.mjson"


def _double_riichi_kyoku(
    result: str,
    *,
    bakaze: str = "E",
    kyoku: int = 1,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = [
        {
            "type": "start_kyoku",
            "bakaze": bakaze,
            "kyoku": kyoku,
            "honba": 0,
            "oya": 0,
        },
        {"type": "tsumo", "actor": 0},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0},
        {"type": "reach_accepted", "actor": 0},
    ]
    if result == "dealer_win":
        events.append({"type": "hora", "actor": 0, "target": 0})
    elif result == "other_win":
        events.append({"type": "hora", "actor": 1, "target": 1})
    elif result == "draw":
        events.append({"type": "ryukyoku"})
    else:
        raise ValueError(f"unsupported result: {result}")
    events.append({"type": "end_kyoku"})
    return events


def _kyoku_without_double_riichi() -> list[dict[str, object]]:
    return [
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "oya": 0,
        },
        {"type": "tsumo", "actor": 0},
        {"type": "dahai", "actor": 0},
        {"type": "ryukyoku"},
        {"type": "end_kyoku"},
    ]


def _write_game(
    raw_root: Path,
    year: int,
    index: int,
    kyokus: list[list[dict[str, object]]],
    *,
    rule_code: str = "00a9",
    aka_flag: object = True,
) -> Path:
    path = raw_root / str(year) / _filename(year, index, rule_code=rule_code)
    path.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {"type": "start_game", "aka_flag": aka_flag},
        *(event for kyoku in kyokus for event in kyoku),
        {"type": "end_game"},
    ]
    path.write_text(
        "".join(f"{json.dumps(event)}\n" for event in events),
        encoding="utf-8",
    )
    return path


def _dataset(counts: dict[int, int]) -> DatasetValidationMetadata:
    return DatasetValidationMetadata(
        repository="example/repository",
        release_tag="v2.0.0",
        raw_file_counts=counts,
    )


def _write_dataset_summary(
    path: Path,
    counts: dict[int, int],
    *,
    release_tag: str = "v2.0.0",
    validation_failed: object = False,
    archive_hashes_verified: object = True,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "repository": "example/repository",
                "release_tag": release_tag,
                "validation_failed": validation_failed,
                "archive_hashes_verified": archive_hashes_verified,
                "years": [
                    {"year": year, "raw": {"total_mjson_files": count}}
                    for year, count in sorted(counts.items())
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_known_results(
    path: Path,
    results: list[YearlyDealerDoubleRiichiResult],
) -> Path:
    path.write_text(
        json.dumps(
            {
                "release_tag": "v2.0.0",
                "years": [
                    {
                        "year": result.year,
                        "scanned_files": result.scanned_files,
                        "target_games": result.target_games,
                        "east_kyokus": result.east_kyokus,
                        "dealer_double_riichi": result.dealer_double_riichi,
                        "dealer_win": result.dealer_win,
                        "other_win": result.other_win,
                        "draw": result.draw,
                    }
                    for result in results
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_cli_requires_exactly_one_year_selection() -> None:
    assert selected_years_from_args(parse_args(["--years", "2025"])) == (2025,)
    assert selected_years_from_args(parse_args(["--all"])) == SUPPORTED_YEARS

    with pytest.raises(SystemExit):
        parse_args([])
    with pytest.raises(SystemExit):
        parse_args(["--all", "--years", "2025"])
    with pytest.raises(SystemExit):
        parse_args(["--years", "2008"])


def test_normalize_years_sorts_and_rejects_duplicates() -> None:
    assert normalize_years([2025, 2009]) == (2009, 2025)

    with pytest.raises(ValueError, match="duplicate"):
        normalize_years([2025, 2025])


def test_aggregate_one_year_with_no_double_riichi(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _write_game(raw_root, 2020, 1, [_kyoku_without_double_riichi()])

    summary = aggregate_dealer_double_riichi_years(
        [2020],
        raw_root,
        _dataset({2020: 1}),
    )

    result = summary.years[0]
    assert result.year == 2020
    assert result.scanned_files == 1
    assert result.target_games == 1
    assert result.east_kyokus == 1
    assert result.dealer_double_riichi == 0
    assert result.win_rate is None
    assert summary.totals.win_rate is None


def test_aggregate_multiple_years_keeps_results_separate_and_builds_totals(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    _write_game(raw_root, 2020, 1, [_double_riichi_kyoku("dealer_win")])
    _write_game(
        raw_root,
        2020,
        2,
        [_double_riichi_kyoku("dealer_win")],
        rule_code="00e1",
    )
    _write_game(
        raw_root,
        2021,
        1,
        [
            _double_riichi_kyoku("other_win", kyoku=1),
            _double_riichi_kyoku("draw", kyoku=2),
        ],
    )
    progress: list[tuple[int, int]] = []

    summary = aggregate_dealer_double_riichi_years(
        [2021, 2020],
        raw_root,
        _dataset({2020: 2, 2021: 1}),
        progress_interval=1,
        progress_callback=lambda year, count: progress.append((year, count)),
    )

    assert summary.selected_years == (2020, 2021)
    assert [(result.year, result.scanned_files) for result in summary.years] == [
        (2020, 2),
        (2021, 1),
    ]
    assert summary.years[0].target_games == 1
    assert summary.years[0].dealer_win == 1
    assert summary.years[1].target_games == 1
    assert summary.years[1].other_win == 1
    assert summary.years[1].draw == 1
    assert summary.totals.scanned_files == 3
    assert summary.totals.target_games == 2
    assert summary.totals.dealer_double_riichi == 3
    assert summary.totals.dealer_win == 1
    assert summary.totals.other_win == 1
    assert summary.totals.draw == 1
    assert summary.totals.win_rate == pytest.approx(1 / 3)
    assert progress == [(2020, 1), (2020, 2), (2021, 1)]

    json_path = tmp_path / "yearly.json"
    markdown_path = tmp_path / "yearly.md"
    write_yearly_outputs(summary, json_path, markdown_path)
    output = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert output["years"][0] == {
        "year": 2020,
        "scanned_files": 2,
        "target_games": 1,
        "east_kyokus": 1,
        "dealer_double_riichi": 1,
        "dealer_win": 1,
        "other_win": 0,
        "draw": 0,
        "win_rate": 1.0,
    }
    assert output["years"][1] == {
        "year": 2021,
        "scanned_files": 1,
        "target_games": 1,
        "east_kyokus": 2,
        "dealer_double_riichi": 2,
        "dealer_win": 0,
        "other_win": 1,
        "draw": 1,
        "win_rate": 0.0,
    }
    assert output["totals"] == {
        "scanned_files": 3,
        "target_games": 2,
        "east_kyokus": 3,
        "dealer_double_riichi": 3,
        "dealer_win": 1,
        "other_win": 1,
        "draw": 1,
        "win_rate": pytest.approx(1 / 3),
    }
    assert "| 2020 | 2 | 1 | 1 | 1 | 1 | 0 | 0 | 100.00% |" in markdown
    assert "| 2021 | 1 | 1 | 2 | 2 | 0 | 1 | 1 | 0.00% |" in markdown
    assert (
        "| **Total** | **3** | **2** | **3** | **3** | **1** | **1** | "
        "**1** | **33.33%** |"
    ) in markdown


def test_yearly_result_rejects_inconsistent_result_counts() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        YearlyDealerDoubleRiichiResult(
            year=2020,
            scanned_files=1,
            target_games=1,
            east_kyokus=1,
            dealer_double_riichi=1,
            dealer_win=1,
            other_win=1,
            draw=0,
        )


def test_aggregate_rejects_missing_year_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="2020"):
        aggregate_dealer_double_riichi_years(
            [2020],
            tmp_path / "raw",
            _dataset({2020: 0}),
        )


def test_aggregate_rejects_raw_file_count_mismatch_before_reading(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    path = raw_root / "2020" / _filename(2020, 1)
    path.parent.mkdir(parents=True)
    path.write_text("not JSON\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="expected 2, found 1"):
        aggregate_dealer_double_riichi_years(
            [2020],
            raw_root,
            _dataset({2020: 2}),
        )


def test_aggregate_error_identifies_year_and_file(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    path = raw_root / "2020" / _filename(2020, 1)
    path.parent.mkdir(parents=True)
    path.write_text("not JSON\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="failed to aggregate year 2020") as error:
        aggregate_dealer_double_riichi_years(
            [2020],
            raw_root,
            _dataset({2020: 1}),
        )

    assert error.value.__cause__ is not None
    assert path.name in str(error.value.__cause__)


@pytest.mark.parametrize(
    ("release_tag", "validation_failed", "hashes_verified", "message"),
    [
        ("v1.0.0", False, True, "release_tag"),
        ("v2.0.0", True, True, "has not passed"),
        ("v2.0.0", False, False, "hashes have not been verified"),
    ],
)
def test_dataset_summary_preconditions(
    tmp_path: Path,
    release_tag: str,
    validation_failed: bool,
    hashes_verified: bool,
    message: str,
) -> None:
    path = _write_dataset_summary(
        tmp_path / "summary.json",
        {2020: 1},
        release_tag=release_tag,
        validation_failed=validation_failed,
        archive_hashes_verified=hashes_verified,
    )

    with pytest.raises(ValueError, match=message):
        load_dataset_validation_metadata(path, [2020])


def test_dataset_summary_requires_requested_year(tmp_path: Path) -> None:
    path = _write_dataset_summary(tmp_path / "summary.json", {2020: 1})

    with pytest.raises(ValueError, match=r"missing years: \[2021\]"):
        load_dataset_validation_metadata(path, [2021])


def test_tracked_dataset_summary_schema_and_file_counts() -> None:
    metadata = load_dataset_validation_metadata(
        DATASET_SUMMARY_PATH,
        [2009, 2025],
    )

    assert metadata.release_tag == "v2.0.0"
    assert metadata.raw_file_counts == {2009: 6_897, 2025: 178_888}


def test_known_results_manifest_contains_verified_2025_counts() -> None:
    known = load_known_results(KNOWN_RESULTS_PATH)

    assert known[2025] == YearlyDealerDoubleRiichiResult(
        year=2025,
        scanned_files=178_888,
        target_games=178_887,
        east_kyokus=1_028_072,
        dealer_double_riichi=702,
        dealer_win=507,
        other_win=87,
        draw=108,
    )
    assert known[2025].win_rate == 507 / 702


def test_aggregate_checks_known_results_when_present(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _write_game(raw_root, 2020, 1, [_double_riichi_kyoku("dealer_win")])
    expected = YearlyDealerDoubleRiichiResult(
        year=2020,
        scanned_files=1,
        target_games=1,
        east_kyokus=1,
        dealer_double_riichi=1,
        dealer_win=1,
        other_win=0,
        draw=0,
    )

    summary = aggregate_dealer_double_riichi_years(
        [2020],
        raw_root,
        _dataset({2020: 1}),
        known_results={2020: expected},
    )

    assert summary.years == (expected,)

    with pytest.raises(RuntimeError, match="dealer_win=expected 0, actual 1"):
        aggregate_dealer_double_riichi_years(
            [2020],
            raw_root,
            _dataset({2020: 1}),
            known_results={2020: replace(expected, dealer_win=0, other_win=1)},
        )


def test_json_and_markdown_outputs_are_deterministic(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _write_game(raw_root, 2020, 1, [_kyoku_without_double_riichi()])
    summary = aggregate_dealer_double_riichi_years(
        [2020],
        raw_root,
        _dataset({2020: 1}),
    )
    json_path = tmp_path / "outputs" / "yearly.json"
    markdown_path = tmp_path / "outputs" / "yearly.md"

    write_yearly_outputs(summary, json_path, markdown_path)
    first_json = json_path.read_text(encoding="utf-8")
    first_markdown = markdown_path.read_text(encoding="utf-8")
    write_yearly_outputs(summary, json_path, markdown_path)

    assert json_path.read_text(encoding="utf-8") == first_json
    assert markdown_path.read_text(encoding="utf-8") == first_markdown
    assert first_json.endswith("\n")
    assert first_markdown == render_yearly_markdown(summary)
    assert str(tmp_path) not in first_json
    assert "generated_at" not in first_json
    output = yearly_summary_to_dict(summary)
    assert output["schema_version"] == 1
    assert output["dataset"] == {
        "repository": "example/repository",
        "release_tag": "v2.0.0",
    }
    assert output["analysis"] == {
        "rule_code": "00a9",
        "aka_flag": True,
        "bakaze": "E",
        "dealer_double_riichi": "established",
    }
    assert output["years"][0]["win_rate"] is None
    assert output["totals"]["win_rate"] is None
    assert not list(json_path.parent.glob(".*.tmp"))


def test_cli_rejects_same_resolved_output_path_before_writing(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    output_path = tmp_path / "yearly.out"
    equivalent_path = nested / ".." / "yearly.out"

    with pytest.raises(ValueError, match="output paths must be different"):
        main(
            [
                "--years",
                "2020",
                "--output-json",
                str(output_path),
                "--output-markdown",
                str(equivalent_path),
            ]
        )

    assert not output_path.exists()


def test_second_output_replace_failure_restores_existing_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "raw"
    _write_game(raw_root, 2020, 1, [_kyoku_without_double_riichi()])
    summary = aggregate_dealer_double_riichi_years(
        [2020],
        raw_root,
        _dataset({2020: 1}),
    )
    json_path = tmp_path / "yearly.json"
    markdown_path = tmp_path / "yearly.md"
    json_path.write_text("old JSON", encoding="utf-8")
    markdown_path.write_text("old Markdown", encoding="utf-8")
    real_replace = yearly_aggregation.os.replace
    markdown_failed = False

    def fail_first_markdown_replace(source: Path, destination: Path) -> None:
        nonlocal markdown_failed
        if Path(destination) == markdown_path and not markdown_failed:
            markdown_failed = True
            raise PermissionError("Markdown replace failed")
        real_replace(source, destination)

    monkeypatch.setattr(
        yearly_aggregation.os,
        "replace",
        fail_first_markdown_replace,
    )

    with pytest.raises(PermissionError, match="Markdown replace failed"):
        write_yearly_outputs(summary, json_path, markdown_path)

    assert json_path.read_text(encoding="utf-8") == "old JSON"
    assert markdown_path.read_text(encoding="utf-8") == "old Markdown"
    assert not list(tmp_path.glob(".*.tmp"))


def test_rollback_failure_reports_possible_output_inconsistency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "raw"
    _write_game(raw_root, 2020, 1, [_kyoku_without_double_riichi()])
    summary = aggregate_dealer_double_riichi_years(
        [2020],
        raw_root,
        _dataset({2020: 1}),
    )
    json_path = tmp_path / "yearly.json"
    markdown_path = tmp_path / "yearly.md"
    json_path.write_text("old JSON", encoding="utf-8")
    markdown_path.write_text("old Markdown", encoding="utf-8")
    real_replace = yearly_aggregation.os.replace
    markdown_failed = False

    def fail_markdown_and_json_rollback(source: Path, destination: Path) -> None:
        nonlocal markdown_failed
        target = Path(destination)
        if target == markdown_path and not markdown_failed:
            markdown_failed = True
            raise PermissionError("Markdown replace failed")
        if target == json_path and markdown_failed:
            raise PermissionError("JSON rollback failed")
        real_replace(source, destination)

    monkeypatch.setattr(
        yearly_aggregation.os,
        "replace",
        fail_markdown_and_json_rollback,
    )

    with pytest.raises(OutputConsistencyError, match="not guaranteed") as error:
        write_yearly_outputs(summary, json_path, markdown_path)

    assert isinstance(error.value.__cause__, PermissionError)
    assert str(error.value.__cause__) == "Markdown replace failed"
    assert error.value.write_error is error.value.__cause__
    assert [path for path, _ in error.value.rollback_errors] == [json_path]
    assert markdown_path.read_text(encoding="utf-8") == "old Markdown"
    assert not list(tmp_path.glob(".*.tmp"))


def test_cli_does_not_update_outputs_when_known_result_fails(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    _write_game(raw_root, 2020, 1, [_double_riichi_kyoku("dealer_win")])
    dataset_path = _write_dataset_summary(
        tmp_path / "dataset-summary.json",
        {2020: 1},
    )
    incorrect = YearlyDealerDoubleRiichiResult(
        year=2020,
        scanned_files=1,
        target_games=1,
        east_kyokus=1,
        dealer_double_riichi=1,
        dealer_win=0,
        other_win=1,
        draw=0,
    )
    known_path = _write_known_results(tmp_path / "known.json", [incorrect])
    json_path = tmp_path / "yearly.json"
    markdown_path = tmp_path / "yearly.md"
    json_path.write_text("old JSON", encoding="utf-8")
    markdown_path.write_text("old Markdown", encoding="utf-8")

    with pytest.raises(RuntimeError, match="known result mismatch"):
        main(
            [
                "--years",
                "2020",
                "--raw-root",
                str(raw_root),
                "--dataset-summary",
                str(dataset_path),
                "--known-results",
                str(known_path),
                "--output-json",
                str(json_path),
                "--output-markdown",
                str(markdown_path),
            ]
        )

    assert json_path.read_text(encoding="utf-8") == "old JSON"
    assert markdown_path.read_text(encoding="utf-8") == "old Markdown"
    assert not list(tmp_path.glob("*partial*"))


def test_cli_writes_outputs_after_all_selected_years_succeed(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    _write_game(raw_root, 2020, 1, [_double_riichi_kyoku("draw")])
    dataset_path = _write_dataset_summary(
        tmp_path / "dataset-summary.json",
        {2020: 1},
    )
    known_path = _write_known_results(tmp_path / "known.json", [])
    json_path = tmp_path / "yearly.json"
    markdown_path = tmp_path / "yearly.md"

    exit_code = main(
        [
            "--years",
            "2020",
            "--raw-root",
            str(raw_root),
            "--dataset-summary",
            str(dataset_path),
            "--known-results",
            str(known_path),
            "--output-json",
            str(json_path),
            "--output-markdown",
            str(markdown_path),
        ]
    )

    assert exit_code == 0
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["selected_years"] == [2020]
    assert data["years"][0]["draw"] == 1
    assert markdown_path.exists()
