from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis import analyze_yakuhai_dash as analysis
from analysis.analyze_yakuhai_dash import (
    YearTask,
    build_document,
    main,
    run_tasks,
    scan_year,
)
from test_yakuhai_dash import make_hands, make_kyoku, non_tile


def artificial_game() -> list[dict[str, object]]:
    hands = make_hands({0: ["P"], 1: ["P", "P"], 2: ["W"]})
    events = [
        {"type": "tsumo", "actor": 0, "pai": "9s"},
        {"type": "dahai", "actor": 0, "pai": "P", "tsumogiri": False},
        {
            "type": "pon",
            "actor": 1,
            "target": 0,
            "pai": "P",
            "consumed": ["P", "P"],
        },
        {
            "type": "dahai",
            "actor": 1,
            "pai": non_tile(hands[1], "P"),
            "tsumogiri": False,
        },
    ]
    return make_kyoku(hands, events)


def write_game(path: Path, kyoku: list[dict[str, object]]) -> None:
    events = [
        {"type": "start_game", "aka_flag": True, "names": ["a", "b", "c", "d"]},
        *kyoku,
        {"type": "end_game"},
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(event, separators=(",", ":")) + "\n" for event in events),
        encoding="utf-8",
    )


def test_scan_year_builds_summary_and_source_audits(tmp_path: Path) -> None:
    path = tmp_path / "2025" / "2025010100gm-00a9-0000-1234abcd.mjson"
    write_game(path, artificial_game())

    result = scan_year(YearTask(2025, str(tmp_path), None, 2))
    document = build_document((result,))

    assert document["input_counts"] == {
        "available_logs": 1,
        "scanned_logs": 1,
        "target_logs": 1,
        "east_kyokus": 1,
    }
    assert document["overall"]["initial_deals"]["all"]["player_deals"] == 4
    assert document["overall"]["opening_behavior"]["all"]["strict_dash"] == {
        "count": 1,
        "denominator": document["overall"]["opening_behavior"]["all"][
            "eligible_player_deals"
        ],
        "rate": 1
        / document["overall"]["opening_behavior"]["all"]["eligible_player_deals"],
    }
    audits = {row["category"]: row["samples"] for row in document["audit_samples"]}
    strict = audits["strict_dash"][0]
    assert strict["relative_source_path"] == path.relative_to(tmp_path).as_posix()
    assert strict["start_kyoku_line"] == 2
    assert strict["singleton_discard_line"] == 4
    assert strict["discard_pon_line"] == 5
    assert strict["first_opponent_pair_line"] == 2
    assert audits["actual_pon"][0]["discard_was_ponned"] is True
    assert audits["initial_yakuhai_pair"][0]["yakuhai_pair_kind_count"] == 1
    assert audits["initial_no_yakuhai_pair"]
    assert audits["no_first_discard"]
    annual = document["audit_samples_by_year"][0]
    assert annual["year"] == 2025
    assert [category["category"] for category in annual["categories"]] == [
        category["category"] for category in document["audit_samples"]
    ]
    assert all(
        category["population_count"] >= len(category["samples"])
        for category in annual["categories"]
    )


def test_serial_and_parallel_results_match(tmp_path: Path) -> None:
    tasks = []
    for year in (2024, 2025):
        path = tmp_path / str(year) / f"{year}010100gm-00a9-0000-1234abcd.mjson"
        write_game(path, artificial_game())
        tasks.append(YearTask(year, str(tmp_path), None, 1))

    serial = build_document(run_tasks(tuple(tasks), workers=1))
    parallel = build_document(run_tasks(tuple(tasks), workers=2))

    assert serial == parallel
    assert [row["year"] for row in serial["audit_samples_by_year"]] == [2024, 2025]
    assert all(
        any(category["samples"] for category in row["categories"])
        for row in serial["audit_samples_by_year"]
    )


def test_post_discard_audits_keep_source_lines(tmp_path: Path) -> None:
    hands = make_hands({0: ["P"], 1: ["P"]})
    events = [
        {"type": "tsumo", "actor": 0, "pai": "9s"},
        {"type": "dahai", "actor": 0, "pai": "P", "tsumogiri": False},
        {"type": "tsumo", "actor": 1, "pai": "P"},
        {"type": "tsumo", "actor": 0, "pai": "P"},
    ]
    path = tmp_path / "2025" / "2025010100gm-00a9-0000-1234abcd.mjson"
    write_game(path, make_kyoku(hands, events))

    document = build_document((scan_year(YearTask(2025, str(tmp_path), None, 2)),))
    audits = {row["category"]: row["samples"] for row in document["audit_samples"]}
    assert audits["post_discard_self_draw"][0]["post_discard_self_draw_line"] == 6
    assert audits["post_discard_opponent_pair"][0][
        "post_discard_opponent_pair_line"
    ] == 5
    assert audits["opponent_not_pon_capable_at_discard"]
    assert audits["no_self_pair_draw"][0]["outcome"] == "singleton_discard"


def test_cli_writes_non_overwriting_sample_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "raw"
    path = data_root / "2025" / "2025010100gm-00a9-0000-1234abcd.mjson"
    write_game(path, artificial_game())
    full_output = tmp_path / "outputs" / "summary-v3.json"
    monkeypatch.setattr(analysis, "DEFAULT_OUTPUT", full_output)

    args = [
        "--data-root",
        str(data_root),
        "--years",
        "2025",
        "--max-logs-per-year",
        "1",
    ]
    assert main(args) == 0
    sample_output = full_output.with_name("summary-v3-sample-1.json")
    document = json.loads(sample_output.read_text(encoding="utf-8"))
    assert document["metadata"]["analysis_name"] == "yakuhai-dash-v3"
    assert document["metadata"]["schema_version"] == 3
    assert document["metadata"]["execution"]["run_mode"] == "sample"
    assert document["metadata"]["execution"]["audit_samples_per_category"] == 3
    assert document["metadata"]["theoretical_baselines"][
        "specific_kind_pair_or_more"
    ] == pytest.approx(0.04555777796847987)
    assert not full_output.exists()

    previous = sample_output.read_bytes()
    with pytest.raises(SystemExit, match="output already exists"):
        main(args)
    assert sample_output.read_bytes() == previous


def test_sample_cannot_target_full_default_before_scanning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    full_output = tmp_path / "summary-v3.json"
    monkeypatch.setattr(analysis, "DEFAULT_OUTPUT", full_output)

    with pytest.raises(SystemExit, match="sample output cannot use"):
        main(
            [
                "--data-root",
                str(tmp_path / "missing"),
                "--years",
                "2025",
                "--max-logs-per-year",
                "1",
                "--output",
                str(full_output),
            ]
        )


def test_zero_sample_limit_still_records_population(tmp_path: Path) -> None:
    path = tmp_path / "2025" / "2025010100gm-00a9-0000-1234abcd.mjson"
    write_game(path, artificial_game())
    document = build_document((scan_year(YearTask(2025, str(tmp_path), None, 0)),))
    categories = document["audit_samples_by_year"][0]["categories"]
    strict = next(row for row in categories if row["category"] == "strict_dash")
    assert strict["population_count"] == 1
    assert strict["samples"] == []
    assert document["metadata"]["execution"]["audit_samples_per_category"] == 0


def test_progress_reports_during_scan_and_on_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for index in range(2):
        path = tmp_path / "2025" / f"202501010{index}gm-00a9-0000-1234abcd.mjson"
        write_game(path, artificial_game())
    monkeypatch.setattr(analysis, "PROGRESS_LOG_INTERVAL", 1)
    scan_year(YearTask(2025, str(tmp_path), None, 0, progress=True))
    progress = capsys.readouterr().err.splitlines()
    assert len(progress) == 3
    assert "start: scanned 0/2 logs, 0 target logs, 0 east kyokus" in progress[0]
    assert "progress: scanned 1/2 logs, 1 target logs, 1 east kyokus" in progress[1]
    assert "complete: scanned 2/2 logs, 2 target logs, 2 east kyokus" in progress[2]


def test_completed_year_checkpoint_survives_later_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "raw"
    good = data_root / "2024" / "2024010100gm-00a9-0000-1234abcd.mjson"
    write_game(good, artificial_game())
    bad = data_root / "2025" / "2025010100gm-00a9-0000-1234abcd.mjson"
    bad.parent.mkdir(parents=True)
    bad.write_text("not json\n", encoding="utf-8")
    output = tmp_path / "summary-v3.json"
    checkpoint_dir = tmp_path / "checkpoints"
    monkeypatch.setattr(analysis, "DEFAULT_OUTPUT", output)
    with pytest.raises(SystemExit):
        main(
            [
                "--data-root",
                str(data_root),
                "--years",
                "2024",
                "2025",
                "--workers",
                "1",
                "--output",
                str(output),
                "--checkpoint-dir",
                str(checkpoint_dir),
            ]
        )
    checkpoint = checkpoint_dir / "year-2024-v3.json"
    assert checkpoint.is_file()
    annual = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert annual["metadata"]["years"] == [2024]
    assert annual["input_counts"]["east_kyokus"] == 1
    assert not output.exists()
    with pytest.raises(SystemExit, match="output already exists"):
        main(
            [
                "--data-root",
                str(data_root),
                "--years",
                "2024",
                "2025",
                "--checkpoint-dir",
                str(checkpoint_dir),
            ]
        )


def test_parallel_checkpoint_preserves_success_when_other_year_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "raw"
    good = data_root / "2024" / "2024010100gm-00a9-0000-1234abcd.mjson"
    write_game(good, artificial_game())
    bad = data_root / "2025" / "2025010100gm-00a9-0000-1234abcd.mjson"
    bad.parent.mkdir(parents=True)
    bad.write_text("not json\n", encoding="utf-8")
    output = tmp_path / "summary-v3.json"
    monkeypatch.setattr(analysis, "DEFAULT_OUTPUT", output)
    with pytest.raises(SystemExit):
        main(
            [
                "--data-root",
                str(data_root),
                "--years",
                "2024",
                "2025",
                "--workers",
                "2",
                "--checkpoint-dir",
                str(tmp_path / "checkpoints"),
            ]
        )
    assert (tmp_path / "checkpoints" / "year-2024-v3.json").is_file()
    assert not output.exists()


def test_sample_cannot_target_legacy_full_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        analysis, "LEGACY_DEFAULT_OUTPUT", tmp_path / "summary-v2.json"
    )
    with pytest.raises(SystemExit, match="sample output cannot use"):
        main(
            [
                "--data-root",
                str(tmp_path / "missing"),
                "--years",
                "2025",
                "--max-logs-per-year",
                "1",
                "--output",
                str(analysis.LEGACY_DEFAULT_OUTPUT),
            ]
        )
