import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import analysis.find_kokushi_riichi as cli


def test_cli_defaults_cover_full_archive_range() -> None:
    args = cli.parse_args([])

    assert args.db_root == cli.DEFAULT_DB_ROOT
    assert args.output == cli.DEFAULT_OUTPUT
    assert args.years == list(range(2009, 2026))
    assert args.workers == 1
    assert args.max_logs_per_year is None


def test_cli_rejects_bad_worker_sample_and_year() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--workers", "0"])
    with pytest.raises(SystemExit):
        cli.parse_args(["--max-logs-per-year", "0"])
    with pytest.raises(SystemExit):
        cli.parse_args(["--years", "2026"])


def test_cli_runs_scan_writes_result_and_prints_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = SimpleNamespace(name="result")
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        cli,
        "scan_kokushi_riichi",
        lambda root, **kwargs: calls.append((root, kwargs)) or result,
    )
    monkeypatch.setattr(
        cli,
        "write_result_json",
        lambda value, output: calls.append(("write", value, output)),
    )
    monkeypatch.setattr(
        cli,
        "result_to_dict",
        lambda value: {"summary": {"total_riichis": 12, "kokushi_riichis": 2}},
    )
    output = tmp_path / "result.json"

    exit_code = cli.main(
        [
            "--db-root",
            str(tmp_path),
            "--output",
            str(output),
            "--workers",
            "3",
            "--years",
            "2009",
            "2025",
            "--max-logs-per-year",
            "10",
        ]
    )

    assert exit_code == 0
    assert calls == [
        (
            tmp_path,
            {
                "years": [2009, 2025],
                "workers": 3,
                "max_logs_per_year": 10,
                "on_year_complete": None,
            },
        ),
        ("write", result, output),
    ]
    printed = json.loads(capsys.readouterr().out)
    assert printed["output"] == str(output.resolve())
    assert printed["summary"] == {"total_riichis": 12, "kokushi_riichis": 2}
