import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import analysis.find_rare_yakuman as cli


def test_cli_defaults_and_required_yaku() -> None:
    args = cli.parse_args(["--yaku", "緑一色"])

    assert args.raw_root == cli.DEFAULT_RAW_ROOT
    assert args.yaku == "緑一色"
    assert args.workers == 1
    assert args.continue_on_replay_error is False


def test_cli_rejects_non_positive_workers() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--yaku", "緑一色", "--workers", "0"])


def test_cli_passes_arguments_and_prints_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = SimpleNamespace(name="result")
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        cli,
        "scan_rare_yakuman",
        lambda root, **kwargs: calls.append((root, kwargs)) or result,
    )
    monkeypatch.setattr(
        cli,
        "result_to_dict",
        lambda value: {"result": value.name, "yaku": "緑一色"},
    )

    exit_code = cli.main(
        [
            "--raw-root",
            str(tmp_path),
            "--yaku",
            "緑一色",
            "--workers",
            "3",
            "--continue-on-replay-error",
        ]
    )

    assert exit_code == 0
    assert calls == [
        (
            tmp_path,
            {
                "yaku": "緑一色",
                "workers": 3,
                "continue_on_replay_error": True,
            },
        )
    ]
    assert (
        capsys.readouterr().out == '{\n  "result": "result",\n  "yaku": "緑一色"\n}\n'
    )


def test_cli_reports_unsupported_yaku_without_scanning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "scan_rare_yakuman",
        lambda root, **kwargs: (_ for _ in ()).throw(
            ValueError("unsupported yaku; available yaku: 緑一色")
        ),
    )

    with pytest.raises(SystemExit, match="available yaku: 緑一色"):
        cli.main(["--yaku", "九蓮宝燈"])


def test_cli_subprocess_stdout_is_utf8_independent_of_windows_locale(
    tmp_path: Path,
) -> None:
    for year in range(2009, 2026):
        (tmp_path / str(year)).mkdir()
    environment = os.environ.copy()
    source_root = Path(cli.__file__).resolve().parents[1] / "src"
    previous_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(source_root), previous_pythonpath) if part
    )
    environment["PYTHONIOENCODING"] = "cp932"

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(cli.__file__).resolve()),
            "--raw-root",
            str(tmp_path),
            "--yaku",
            "緑一色",
            "--workers",
            "1",
        ],
        check=True,
        capture_output=True,
        env=environment,
    )

    decoded = completed.stdout.decode("utf-8")
    document = json.loads(decoded)
    assert document["yaku"] == "緑一色"
    assert document["total_games"] == 0
    assert document["total_kyokus"] == 0
    assert document["analyzed_kyokus"] == 0
    assert document["anomaly_kyokus"] == 0
    assert completed.stderr == b""
