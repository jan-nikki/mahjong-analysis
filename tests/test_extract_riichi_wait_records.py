import json
from pathlib import Path

import pytest

import analysis.extract_riichi_wait_records as cli
from mahjong_analysis.riichi_wait_dataset import iter_dataset_records
from mahjong_analysis.riichi_wait_export import GitMetadata


def write_summary(path: Path, year: int, count: int) -> None:
    write_multi_year_summary(path, {year: count})


def write_multi_year_summary(path: Path, counts: dict[int, int]) -> None:
    path.write_text(
        json.dumps(
            {
                "archive_hashes_verified": True,
                "release_tag": "v2.0.0",
                "repository": "example/source",
                "target_rule_code": "00a9",
                "validation_failed": False,
                "years": [
                    {"year": year, "raw": {"total_mjson_files": count}}
                    for year, count in sorted(counts.items())
                ],
            }
        ),
        encoding="utf-8",
    )


def write_game(path: Path) -> None:
    hand = [
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
        "5s",
    ]
    events = (
        {"type": "start_game", "aka_flag": True},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "dora_marker": "5pr",
            "honba": 0,
            "kyoku": 1,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "tehais": [list(hand) for _ in range(4)],
        },
        {"type": "tsumo", "actor": 1, "pai": "5mr"},
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "5mr", "tsumogiri": True},
        {"type": "reach_accepted", "actor": 1},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        "".join(f"{json.dumps(event, separators=(',', ':'))}\n" for event in events),
        encoding="utf-8",
    )


def test_cli_exports_exact_production_record_with_source_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "sample-output"
    summary = tmp_path / "summary.json"
    game = raw / "2025" / "2025010100gm-00a9-0000-1234abcd.mjson"
    write_game(game)
    write_summary(summary, 2025, 1)
    monkeypatch.setattr(
        cli,
        "collect_git_metadata",
        lambda _root: GitMetadata("abc123", False),
    )

    result = cli.main(
        (
            "--raw-root",
            str(raw),
            "--output-root",
            str(output),
            "--dataset-summary",
            str(summary),
            "--dataset-summary-logical-path",
            "fixtures/summary.json",
            "--year",
            "2025",
            "--max-files",
            "1",
            "--allow-dirty",
        )
    )

    assert result == 0
    records = tuple(iter_dataset_records(output / "2025.jsonl.gz"))
    assert len(records) == 1
    record = records[0]
    assert (
        record.relative_source_path,
        record.start_kyoku_line,
        record.reach_line,
        record.declaration_dahai_line,
        record.reach_accepted_line,
    ) == (
        "2025/2025010100gm-00a9-0000-1234abcd.mjson",
        2,
        4,
        5,
        6,
    )
    assert record.scores_at_start == (25000, 25000, 25000, 25000)
    assert record.dora_marker == "5pr"
    assert record.riichi_declaration_tile == "5mr"
    assert record.riichi_declaration_tile_kind == "5m"
    assert record.wait_tiles == ("3s", "6s")
    assert tuple(
        (detail.tile, detail.hand_type, detail.wait_shape)
        for detail in record.wait_details
    ) == (
        ("3s", "standard", "ryanmen"),
        ("6s", "standard", "ryanmen"),
    )


def test_cli_year_selection_is_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(("--year", "2025", "--all"))


def test_cli_all_and_multiple_year_selection() -> None:
    all_args = cli.parse_args(("--all",))
    years_args = cli.parse_args(("--years", "2025", "2009"))

    assert cli._selected_years(all_args) == tuple(range(2009, 2026))
    assert cli._selected_years(years_args) == (2009, 2025)


def test_cli_exports_multiple_years_in_ascending_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    summary = tmp_path / "summary.json"
    write_game(raw / "2009" / "2009010100gm-00a9-0000-1234abcd.mjson")
    write_game(raw / "2010" / "2010010100gm-00a9-0000-1234abcd.mjson")
    write_multi_year_summary(summary, {2009: 1, 2010: 1})
    monkeypatch.setattr(
        cli,
        "collect_git_metadata",
        lambda _root: GitMetadata("abc123", True),
    )

    result = cli.main(
        (
            "--raw-root",
            str(raw),
            "--output-root",
            str(output),
            "--dataset-summary",
            str(summary),
            "--dataset-summary-logical-path",
            "fixtures/summary.json",
            "--years",
            "2010",
            "2009",
        )
    )

    assert result == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert [entry["year"] for entry in manifest["years"]] == [2009, 2010]
    assert manifest["totals"]["output_records"] == 2


def test_cli_rejects_max_files_with_multiple_years() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        cli.main(("--years", "2009", "2010", "--max-files", "1"))


def test_malformed_input_aborts_without_final_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    summary = tmp_path / "summary.json"
    path = raw / "2025" / "2025010100gm-00a9-0000-deadbeef.mjson"
    path.parent.mkdir(parents=True)
    path.write_text("not json\n", encoding="utf-8")
    write_summary(summary, 2025, 1)
    monkeypatch.setattr(
        cli,
        "collect_git_metadata",
        lambda _root: GitMetadata("abc123", False),
    )

    with pytest.raises(json.JSONDecodeError):
        cli.main(
            (
                "--raw-root",
                str(raw),
                "--output-root",
                str(output),
                "--dataset-summary",
                str(summary),
                "--dataset-summary-logical-path",
                "fixtures/summary.json",
                "--year",
                "2025",
                "--max-files",
                "1",
                "--allow-dirty",
            )
        )

    assert (output / "2025.jsonl.gz.part").exists()
    assert not (output / "2025.jsonl.gz").exists()
    assert not (output / "manifest.json").exists()


def test_external_summary_requires_and_preserves_logical_identifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    summary = tmp_path / "outside-summary.json"
    write_game(raw / "2025" / "2025010100gm-00a9-0000-1234abcd.mjson")
    write_summary(summary, 2025, 1)
    monkeypatch.setattr(
        cli,
        "collect_git_metadata",
        lambda _root: GitMetadata("abc123", False),
    )

    base_args = (
        "--raw-root",
        str(raw),
        "--output-root",
        str(output),
        "--dataset-summary",
        str(summary),
        "--year",
        "2025",
        "--max-files",
        "1",
        "--allow-dirty",
    )
    with pytest.raises(ValueError, match="logical-path"):
        cli.main(base_args)

    assert (
        cli.main(
            (
                *base_args,
                "--dataset-summary-logical-path",
                "fixtures/custom-summary.json",
            )
        )
        == 0
    )
    manifest_text = (output / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["source"]["validation_summary_path"] == (
        "fixtures/custom-summary.json"
    )
    assert str(tmp_path) not in manifest_text


def test_project_summary_path_is_derived_as_relative_posix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    summary = tmp_path / "data" / "validation" / "summary.json"
    assert cli._summary_logical_path(summary, None) == ("data/validation/summary.json")

    with pytest.raises(ValueError, match="relative POSIX"):
        cli._summary_logical_path(summary, "C:/private/summary.json")
