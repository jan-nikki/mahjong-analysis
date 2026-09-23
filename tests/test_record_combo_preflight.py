from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

import analysis.record_combo_preflight as preflight
from analysis.record_combo_preflight import (
    full_pytest_command,
    parse_pytest_transcript,
    record_preflight_report,
    run_and_record_preflight,
    verify_preflight_report,
)

_BUNDLE_HASHES = {
    "bundle_file_sha256": "a" * 64,
    "payload_sha256": "b" * 64,
}
_PASSING_TRANSCRIPT = (
    b"============================= test session starts =============================\n"
    b"platform win32 -- Python 3.12.10, pytest-9.0.0\n"
    b"collected 3 items\n"
    b"\n"
    b"tests/test_sample.py ..s                                              [100%]\n"
    b"\n"
    b"======================== 2 passed, 1 skipped in 0.12s ========================\n"
)


def _project(tmp_path: Path) -> Path:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_sample.py").write_text(
        "def test_sample():\n    assert True\n", encoding="utf-8"
    )
    if os.name == "nt":
        executable = tmp_path / ".venv" / "Scripts" / "python.exe"
    else:
        executable = tmp_path / ".venv" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"test launcher")
    return tmp_path


def _record(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    root = _project(tmp_path)
    report_path = root / "outputs" / "preflight.json"
    report = record_preflight_report(
        report_path,
        root,
        _BUNDLE_HASHES,
        command=full_pytest_command(root),
        exit_code=0,
        transcript=_PASSING_TRANSCRIPT,
        created_at_utc="2026-09-23T00:00:00Z",
    )
    return root, report_path, report


def test_record_and_verify_binds_transcript_tests_and_bundle(tmp_path: Path) -> None:
    root, report_path, report = _record(tmp_path)

    verified = verify_preflight_report(report_path, root, _BUNDLE_HASHES)

    assert verified == report
    assert verified["status"] == "PASS"
    assert verified["pytest"] == {
        "collected": 3,
        "passed": 2,
        "skipped": 1,
        "failed": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
        "deselected": 0,
        "warnings": 0,
        "duration_seconds": 0.12,
        "summary_line": (
            "======================== 2 passed, 1 skipped in 0.12s "
            "========================"
        ),
    }
    transcript = root.joinpath(*Path(verified["transcript"]["path"]).parts)
    assert transcript.read_bytes() == _PASSING_TRANSCRIPT
    assert verified["transcript"]["sha256"] in transcript.name
    assert verified["tests"]["files"] == [
        {
            "path": "tests/test_sample.py",
            "sha256": verified["tests"]["files"][0]["sha256"],
            "size_bytes": (root / "tests" / "test_sample.py").stat().st_size,
        }
    ]


def test_parser_accepts_skips_xfails_and_warnings() -> None:
    transcript = (
        b"================ test session starts ================\n"
        b"collected 5 items\n"
        b"tests/test_sample.py ...sx [100%]\n"
        b"=========== 3 passed, 1 skipped, 1 xfailed, 2 warnings in 1.25s ===========\n"
    )

    result = parse_pytest_transcript(transcript, exit_code=0)

    assert result["collected"] == 5
    assert result["passed"] == 3
    assert result["skipped"] == 1
    assert result["xfailed"] == 1
    assert result["warnings"] == 2


def test_parser_accepts_pytest_long_duration_suffix() -> None:
    transcript = (
        b"================ test session starts ================\n"
        b"collected 1380 items\n"
        b"tests/test_sample.py . [100%]\n"
        b"====== 1380 passed, 1 warning in 82.14s (0:01:22) ======\n"
    )

    result = parse_pytest_transcript(transcript, exit_code=0)

    assert result["collected"] == 1380
    assert result["passed"] == 1380
    assert result["warnings"] == 1
    assert result["duration_seconds"] == 82.14


@pytest.mark.parametrize(
    ("transcript", "exit_code", "message"),
    [
        (
            b"collected 1 item\n================ 1 failed in 0.01s ================\n",
            1,
            "exit code",
        ),
        (b"collected 1 item\ntests/test_sample.py . [100%]\n", 0, "terminal"),
        (
            b"collected 2 items\n================ 1 passed in 0.01s ================\n",
            0,
            "collected count",
        ),
        (
            b"collected 1 item\n"
            b"================ 1 deselected in 0.01s ================\n",
            0,
            "collected count",
        ),
    ],
)
def test_record_rejects_nonzero_or_incomplete_results_without_replacing_report(
    tmp_path: Path,
    transcript: bytes,
    exit_code: int,
    message: str,
) -> None:
    root = _project(tmp_path)
    report_path = root / "outputs" / "preflight.json"
    report_path.parent.mkdir()
    report_path.write_text("existing report\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        record_preflight_report(
            report_path,
            root,
            _BUNDLE_HASHES,
            command=full_pytest_command(root),
            exit_code=exit_code,
            transcript=transcript,
        )

    assert report_path.read_text(encoding="utf-8") == "existing report\n"


def test_verify_rejects_tampered_transcript(tmp_path: Path) -> None:
    root, report_path, report = _record(tmp_path)
    transcript = root.joinpath(*Path(report["transcript"]["path"]).parts)
    transcript.write_bytes(_PASSING_TRANSCRIPT + b"tampered\n")

    with pytest.raises(ValueError, match="size mismatch"):
        verify_preflight_report(report_path, root, _BUNDLE_HASHES)


def test_verify_rejects_tampered_summary(tmp_path: Path) -> None:
    root, report_path, report = _record(tmp_path)
    report["pytest"]["passed"] = 1
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="summary does not match"):
        verify_preflight_report(report_path, root, _BUNDLE_HASHES)


def test_verify_rejects_current_test_inventory_change(tmp_path: Path) -> None:
    root, report_path, _report = _record(tmp_path)
    (root / "tests" / "test_added.py").write_text(
        "def test_added():\n    assert True\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="inventory no longer matches"):
        verify_preflight_report(report_path, root, _BUNDLE_HASHES)


def test_verify_rejects_bundle_mismatch(tmp_path: Path) -> None:
    root, report_path, _report = _record(tmp_path)
    mismatched = {**_BUNDLE_HASHES, "payload_sha256": "c" * 64}

    with pytest.raises(ValueError, match="bundle hashes"):
        verify_preflight_report(report_path, root, mismatched)


def test_verify_rejects_report_shape_and_transcript_path_tampering(
    tmp_path: Path,
) -> None:
    root, report_path, report = _record(tmp_path)
    report["unexpected"] = True
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="top-level shape"):
        verify_preflight_report(report_path, root, _BUNDLE_HASHES)

    report.pop("unexpected")
    report["transcript"]["path"] = "tests/test_sample.py"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="content-addressed path"):
        verify_preflight_report(report_path, root, _BUNDLE_HASHES)


def test_run_helper_executes_only_full_suite_and_records_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _project(tmp_path)
    report_path = root / "outputs" / "preflight.json"
    calls: list[list[str]] = []

    def fake_run(
        args: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append(args)
        assert kwargs["cwd"] == root
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert "PYTEST_ADDOPTS" not in environment
        assert kwargs["stderr"] == subprocess.STDOUT
        return subprocess.CompletedProcess(args, 0, stdout=_PASSING_TRANSCRIPT)

    monkeypatch.setattr(preflight.subprocess, "run", fake_run)

    report = run_and_record_preflight(report_path, root, _BUNDLE_HASHES)

    assert calls == [[str(preflight._venv_python_path(root)), "-m", "pytest"]]
    assert report["command"] == {
        "cwd": ".",
        "argv": list(full_pytest_command(root)),
    }
    assert verify_preflight_report(report_path, root, _BUNDLE_HASHES) == report


def test_record_rejects_filtered_pytest_command(tmp_path: Path) -> None:
    root = _project(tmp_path)
    filtered = (*full_pytest_command(root), "tests/test_sample.py")

    with pytest.raises(ValueError, match="complete pytest command"):
        record_preflight_report(
            root / "outputs" / "preflight.json",
            root,
            _BUNDLE_HASHES,
            command=filtered,
            exit_code=0,
            transcript=_PASSING_TRANSCRIPT,
        )


def test_record_is_exclusive_and_preserves_existing_report_bytes(
    tmp_path: Path,
) -> None:
    root, report_path, _report = _record(tmp_path)
    original = report_path.read_bytes()

    with pytest.raises(FileExistsError, match="already exists and is immutable"):
        record_preflight_report(
            report_path,
            root,
            _BUNDLE_HASHES,
            command=full_pytest_command(root),
            exit_code=0,
            transcript=_PASSING_TRANSCRIPT,
            created_at_utc="2026-09-23T00:01:00Z",
        )

    assert report_path.read_bytes() == original
