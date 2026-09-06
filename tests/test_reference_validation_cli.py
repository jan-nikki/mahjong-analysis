import gzip
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

import analysis.validate_dealer_double_riichi as validator
from mahjong_analysis.reference_validation import (
    CandidateRound,
    ReferenceGameResult,
)
from mahjong_analysis.yearly_aggregation import YearlyDealerDoubleRiichiResult


def _write_game(
    raw_root: Path,
    year: int,
    *,
    compressed: bool,
) -> Path:
    path = raw_root / str(year) / f"{year}010100gm-00a9-0000-00000001.mjson"
    path.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {"type": "start_game", "aka_flag": True},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "oya": 0,
        },
        {"type": "tsumo", "actor": 0},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0},
        {"type": "reach_accepted", "actor": 0},
        {"type": "hora", "actor": 0, "target": 0},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]
    content = "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events)
    if compressed:
        with gzip.open(path, mode="wt", encoding="utf-8") as file:
            file.write(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def _candidate(
    *,
    filename: str = "2009010100gm-00a9-0000-00000001.mjson",
    start_kyoku_line: int = 2,
    reach_line: int = 4,
    result: str = "dealer_win",
) -> CandidateRound:
    return CandidateRound(
        filename=filename,
        start_kyoku_line=start_kyoku_line,
        reach_line=reach_line,
        bakaze="E",
        kyoku=1,
        honba=0,
        oya=0,
        result=result,
    )


def _reference_result(
    *,
    target_game: bool = True,
    east_kyokus: int = 1,
    candidates: tuple[CandidateRound, ...] = (),
) -> ReferenceGameResult:
    return ReferenceGameResult(
        target_game=target_game,
        east_kyokus=east_kyokus,
        candidates=candidates,
        reach_audits=(),
    )


def _existing_result(
    *,
    target_game: bool = True,
    east_kyokus: int = 1,
    candidates: tuple[CandidateRound, ...] = (),
) -> validator._ExistingGameResult:
    return validator._ExistingGameResult(
        target_game=target_game,
        east_kyokus=east_kyokus,
        candidates=candidates,
    )


def _mock_results(
    monkeypatch: pytest.MonkeyPatch,
    reference: ReferenceGameResult,
    existing: validator._ExistingGameResult,
) -> None:
    monkeypatch.setattr(
        validator,
        "analyze_reference_mjai",
        lambda path: reference,
    )
    monkeypatch.setattr(
        validator,
        "_analyze_with_existing_implementation",
        lambda path: existing,
    )


def _known_result(year: int = 2025) -> YearlyDealerDoubleRiichiResult:
    return YearlyDealerDoubleRiichiResult(
        year=year,
        scanned_files=1,
        target_games=1,
        east_kyokus=1,
        dealer_double_riichi=1,
        dealer_win=1,
        other_win=0,
        draw=0,
    )


def _write_known_results(
    path: Path,
    results: list[YearlyDealerDoubleRiichiResult],
) -> Path:
    path.write_text(
        json.dumps(
            {
                "release_tag": "v2.0.0",
                "years": [asdict(result) for result in results],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_cli_accepts_2009_and_2025_and_rejects_unsupported_years() -> None:
    assert validator.parse_args(["--year", "2009"]).year == 2009
    assert validator.parse_args(["--year", "2025"]).year == 2025

    with pytest.raises(SystemExit):
        validator.parse_args(["--year", "2008"])
    with pytest.raises(SystemExit):
        validator.parse_args(["--year", "2026"])


@pytest.mark.parametrize(
    ("year", "compressed"),
    [(2009, True), (2025, False)],
)
def test_validation_accepts_supported_plain_and_gzip_years(
    tmp_path: Path,
    year: int,
    compressed: bool,
) -> None:
    path = _write_game(tmp_path / "raw", year, compressed=compressed)

    report = validator.validate_paths(year, [path])

    assert report.passed is True
    assert report.scanned_files == 1
    assert report.reference_stats == validator.ValidationStats(
        target_games=1,
        east_kyokus=1,
        dealer_double_riichi=1,
        dealer_win=1,
        other_win=0,
        draw=0,
    )
    assert report.reference_stats == report.existing_stats
    assert report.reference_candidates == report.existing_candidates
    candidate = report.reference_candidates[0]
    assert candidate.filename == path.name
    assert candidate.start_kyoku_line == 2
    assert candidate.reach_line == 4
    assert candidate.result == "dealer_win"


def test_validation_fails_for_candidate_set_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _candidate()
    existing = replace(reference, start_kyoku_line=20)
    _mock_results(
        monkeypatch,
        _reference_result(candidates=(reference,)),
        _existing_result(candidates=(existing,)),
    )

    report = validator.validate_paths(2009, [tmp_path / reference.filename])

    assert report.passed is False
    assert report.candidate_comparison.reference_only == (reference,)
    assert report.candidate_comparison.existing_only == (existing,)


def test_validation_fails_for_candidate_metadata_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _candidate()
    existing = replace(reference, reach_line=40)
    _mock_results(
        monkeypatch,
        _reference_result(candidates=(reference,)),
        _existing_result(candidates=(existing,)),
    )

    report = validator.validate_paths(2009, [tmp_path / reference.filename])

    assert report.passed is False
    assert report.candidate_comparison.metadata_mismatches == ((reference, existing),)


def test_validation_fails_for_candidate_result_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _candidate(result="dealer_win")
    existing = replace(reference, result="other_win")
    _mock_results(
        monkeypatch,
        _reference_result(candidates=(reference,)),
        _existing_result(candidates=(existing,)),
    )

    report = validator.validate_paths(2009, [tmp_path / reference.filename])

    assert report.passed is False
    assert report.candidate_comparison.result_mismatches == ((reference, existing),)


def test_validation_fails_for_target_and_east_count_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filename = "2009010100gm-00a9-0000-00000001.mjson"
    _mock_results(
        monkeypatch,
        _reference_result(target_game=True, east_kyokus=2),
        _existing_result(target_game=False, east_kyokus=0),
    )

    report = validator.validate_paths(2009, [tmp_path / filename])

    assert report.passed is False
    assert report.reference_stats.target_games == 1
    assert report.existing_stats.target_games == 0
    assert report.reference_stats.east_kyokus == 2
    assert report.existing_stats.east_kyokus == 0
    assert report.file_count_mismatches == (
        {
            "filename": filename,
            "reference_target_game": True,
            "existing_target_game": False,
            "reference_east_kyokus": 2,
            "existing_east_kyokus": 0,
        },
    )


def test_cli_applies_known_2025_only_and_separates_year_outputs(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    _write_game(raw_root, 2009, compressed=True)
    _write_game(raw_root, 2025, compressed=False)
    known_path = _write_known_results(
        tmp_path / "known.json",
        [_known_result()],
    )
    output_root = tmp_path / "outputs"

    for year in (2009, 2025):
        exit_code = validator.main(
            [
                "--year",
                str(year),
                "--raw-root",
                str(raw_root),
                "--known-results",
                str(known_path),
                "--output-root",
                str(output_root),
            ]
        )
        assert exit_code == 0

    output_2009 = json.loads(
        (output_root / "2009" / "comparison.json").read_text(encoding="utf-8")
    )
    output_2025 = json.loads(
        (output_root / "2025" / "comparison.json").read_text(encoding="utf-8")
    )
    assert output_2009["year"] == 2009
    assert output_2009["known_result_checked"] is False
    assert output_2009["known_value_mismatches"] == {}
    assert output_2025["year"] == 2025
    assert output_2025["known_result_checked"] is True
    assert output_2025["known_value_mismatches"] == {}
    assert output_2009["passed"] is True
    assert output_2025["passed"] is True
    assert output_2009["file_count_mismatches"] == []
    assert output_2025["file_count_mismatches"] == []
    for year in (2009, 2025):
        output_directory = output_root / str(year)
        assert (output_directory / "reference_candidates.jsonl").exists()
        assert (output_directory / "existing_candidates.jsonl").exists()
        assert (output_directory / "reach_audit_samples.jsonl").exists()
        comparison_text = (output_directory / "comparison.json").read_text(
            encoding="utf-8"
        )
        assert str(tmp_path) not in comparison_text
        assert "generated_at" not in comparison_text


def test_cli_fails_when_known_2025_result_differs(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _write_game(raw_root, 2025, compressed=False)
    expected = replace(_known_result(), dealer_win=0, other_win=1)
    known_path = _write_known_results(tmp_path / "known.json", [expected])
    output_root = tmp_path / "outputs"

    exit_code = validator.main(
        [
            "--year",
            "2025",
            "--raw-root",
            str(raw_root),
            "--known-results",
            str(known_path),
            "--output-root",
            str(output_root),
        ]
    )

    assert exit_code == 1
    comparison = json.loads(
        (output_root / "2025" / "comparison.json").read_text(encoding="utf-8")
    )
    assert comparison["known_value_mismatches"] == {
        "dealer_win": {"expected": 0, "reference": 1, "existing": 1},
        "other_win": {"expected": 1, "reference": 0, "existing": 0},
    }


def test_cli_automatically_checks_tracked_2009_known_result(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    (raw_root / "2009").mkdir(parents=True)
    output_root = tmp_path / "outputs"

    exit_code = validator.main(
        [
            "--year",
            "2009",
            "--raw-root",
            str(raw_root),
            "--output-root",
            str(output_root),
        ]
    )

    assert exit_code == 1
    comparison = json.loads(
        (output_root / "2009" / "comparison.json").read_text(encoding="utf-8")
    )
    assert comparison["known_result_checked"] is True
    assert comparison["known_value_mismatches"]["scanned_files"] == {
        "expected": 6_897,
        "reference": 0,
        "existing": 0,
    }


def test_cli_progress_includes_year_and_processed_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_root = tmp_path / "raw"
    _write_game(raw_root, 2009, compressed=True)
    known_path = _write_known_results(tmp_path / "known.json", [])
    monkeypatch.setattr(validator, "PROGRESS_INTERVAL", 1)

    exit_code = validator.main(
        [
            "--year",
            "2009",
            "--raw-root",
            str(raw_root),
            "--known-results",
            str(known_path),
            "--output-root",
            str(tmp_path / "outputs"),
        ]
    )

    assert exit_code == 0
    assert "2009: processed 1 files" in capsys.readouterr().out


@pytest.mark.parametrize("side", ["reference", "production"])
def test_validation_error_identifies_side_year_and_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    side: str,
) -> None:
    filename = "2009010100gm-00a9-0000-00000001.mjson"
    path = tmp_path / filename

    if side == "reference":

        def fail_reference(path: Path) -> ReferenceGameResult:
            raise ValueError("bad reference input")

        monkeypatch.setattr(
            validator,
            "analyze_reference_mjai",
            fail_reference,
        )
    else:
        monkeypatch.setattr(
            validator,
            "analyze_reference_mjai",
            lambda path: _reference_result(),
        )

        def fail_production(path: Path) -> validator._ExistingGameResult:
            raise ValueError("bad production input")

        monkeypatch.setattr(
            validator,
            "_analyze_with_existing_implementation",
            fail_production,
        )

    with pytest.raises(RuntimeError, match=side) as error:
        validator.validate_paths(2009, [path])

    assert "year 2009" in str(error.value)
    assert filename in str(error.value)
