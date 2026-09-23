"""Record and verify the full pre-2024 pytest evidence.

This module deliberately has no data-extraction imports.  Its only job is to
bind a complete ``.venv`` pytest run to the frozen development bundle and to
the exact ``tests/**/*.py`` inventory that produced the run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.freeze_combo_bundle import DEFAULT_OUTPUT as DEFAULT_FREEZE_BUNDLE
from analysis.freeze_combo_bundle import file_sha256, verify_freeze_bundle

SCHEMA_VERSION = 1
ANALYSIS_ID = "combo-pre2024-test-preflight-v1"
PASS_STATUS = "PASS"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / f"{ANALYSIS_ID}.json"

_BUNDLE_HASH_KEYS = {"bundle_file_sha256", "payload_sha256"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_COLLECTED_RE = re.compile(r"\bcollected\s+(\d+)\s+items?\b")
_SUMMARY_RE = re.compile(
    r"(?P<body>.+?)\s+in\s+(?P<duration>\d+(?:\.\d+)?)s"
    r"(?:\s+\(\d+:\d{2}:\d{2}\))?\Z"
)
_SUMMARY_TOKEN_RE = re.compile(r"(?P<count>\d+)\s+(?P<name>[A-Za-z]+)\Z")
_SUMMARY_NAMES = {
    "passed": "passed",
    "failed": "failed",
    "skipped": "skipped",
    "error": "errors",
    "errors": "errors",
    "xfailed": "xfailed",
    "xpassed": "xpassed",
    "deselected": "deselected",
    "warning": "warnings",
    "warnings": "warnings",
}
_PYTEST_SUMMARY_KEYS = {
    "collected",
    "passed",
    "skipped",
    "failed",
    "errors",
    "xfailed",
    "xpassed",
    "deselected",
    "warnings",
    "duration_seconds",
    "summary_line",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete project pytest suite and atomically record "
            "pre-2024 evidence."
        )
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--freeze-bundle", type=Path, default=DEFAULT_FREEZE_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    bundle_path = args.freeze_bundle.resolve()
    bundle = verify_freeze_bundle(bundle_path, project_root=project_root)
    expected_bundle_hashes = {
        "bundle_file_sha256": file_sha256(bundle_path),
        "payload_sha256": _sha256_text(
            bundle.get("payload_sha256"), "freeze bundle payload SHA-256"
        ),
    }
    report = run_and_record_preflight(
        report_path=args.output.resolve(),
        project_root=project_root,
        expected_bundle_hashes=expected_bundle_hashes,
    )
    print(
        "preflight PASS "
        f"passed={report['pytest']['passed']} "
        f"skipped={report['pytest']['skipped']} "
        f"transcript_sha256={report['transcript']['sha256']}",
        file=sys.stderr,
        flush=True,
    )
    return 0


def run_and_record_preflight(
    report_path: Path,
    project_root: Path,
    expected_bundle_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Run the exact full-suite command and write its verified evidence."""

    root = project_root.resolve()
    command = full_pytest_command(root)
    before = _test_inventory(root)
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)
    environment.pop("PYTEST_PLUGINS", None)
    environment["NO_COLOR"] = "1"
    environment["PY_COLORS"] = "0"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    completed = subprocess.run(
        [str(_venv_python_path(root)), "-m", "pytest"],
        cwd=root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    after = _test_inventory(root)
    if after != before:
        raise ValueError("test inventory changed while pytest was running")
    return record_preflight_report(
        report_path=report_path,
        project_root=root,
        expected_bundle_hashes=expected_bundle_hashes,
        command=command,
        exit_code=completed.returncode,
        transcript=completed.stdout,
    )


def record_preflight_report(
    report_path: Path,
    project_root: Path,
    expected_bundle_hashes: Mapping[str, str],
    *,
    command: Sequence[str],
    exit_code: int,
    transcript: bytes,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Validate captured pytest output and atomically record a report.

    ``command`` is intentionally restricted to the complete ``.venv`` suite;
    selectors and pytest options are rejected.  This helper is useful when the
    caller owns process capture, while :func:`run_and_record_preflight` is the
    normal CLI path.
    """

    root = project_root.resolve()
    output = _safe_path(report_path, root, must_exist=False)
    if output.suffix.lower() != ".json":
        raise ValueError("preflight report path must end in .json")
    normalized_command = _validate_full_pytest_command(command, root)
    bundle_hashes = _normalize_bundle_hashes(expected_bundle_hashes)
    summary = parse_pytest_transcript(transcript, exit_code=exit_code)
    inventory = _test_inventory(root)
    transcript_digest = sha256(transcript).hexdigest()
    transcript_path = _transcript_path(output, transcript_digest)
    _write_content_addressed(transcript_path, transcript)
    timestamp = created_at_utc or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    _validate_utc_timestamp(timestamp)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": ANALYSIS_ID,
        "status": PASS_STATUS,
        "created_at_utc": timestamp,
        "command": {"cwd": ".", "argv": list(normalized_command)},
        "exit_code": 0,
        "pytest": summary,
        "transcript": {
            "path": _relative_path(transcript_path, root),
            "sha256": transcript_digest,
            "size_bytes": len(transcript),
            "encoding": "utf-8",
        },
        "tests": inventory,
        "freeze_bundle": bundle_hashes,
    }
    _atomic_write_json(output, report)
    return report


def verify_preflight_report(
    report_path: Path,
    project_root: Path,
    expected_bundle_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Fail closed unless a preflight report still matches every bound input."""

    root = project_root.resolve()
    path = _safe_path(report_path, root, must_exist=True)
    report = _read_json_object(path)
    expected_top_level = {
        "schema_version",
        "analysis_id",
        "status",
        "created_at_utc",
        "command",
        "exit_code",
        "pytest",
        "transcript",
        "tests",
        "freeze_bundle",
    }
    if set(report) != expected_top_level:
        raise ValueError("preflight report has an unexpected top-level shape")
    schema_version = report.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != SCHEMA_VERSION
    ):
        raise ValueError("preflight schema_version mismatch")
    if report.get("analysis_id") != ANALYSIS_ID:
        raise ValueError("preflight analysis_id mismatch")
    if report.get("status") != PASS_STATUS:
        raise ValueError("preflight status is not PASS")
    _validate_utc_timestamp(report.get("created_at_utc"))
    exit_code = report.get("exit_code")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int) or exit_code != 0:
        raise ValueError("preflight pytest exit code is not zero")

    expected_hashes = _normalize_bundle_hashes(expected_bundle_hashes)
    recorded_hashes = _require_mapping(report.get("freeze_bundle"), "freeze_bundle")
    if dict(recorded_hashes) != expected_hashes:
        raise ValueError("preflight freeze bundle hashes do not match")

    command = _require_mapping(report.get("command"), "command")
    if set(command) != {"cwd", "argv"} or command.get("cwd") != ".":
        raise ValueError("preflight command shape is invalid")
    argv = command.get("argv")
    if not isinstance(argv, list):
        raise TypeError("preflight command argv must be a list")
    _validate_full_pytest_command(argv, root)

    transcript_record = _require_mapping(report.get("transcript"), "transcript")
    if set(transcript_record) != {"path", "sha256", "size_bytes", "encoding"}:
        raise ValueError("preflight transcript record shape is invalid")
    if transcript_record.get("encoding") != "utf-8":
        raise ValueError("preflight transcript encoding must be utf-8")
    transcript_sha = _sha256_text(transcript_record.get("sha256"), "transcript SHA-256")
    transcript_path = _path_from_record(transcript_record.get("path"), root)
    expected_transcript_path = _transcript_path(path, transcript_sha)
    if transcript_path != expected_transcript_path:
        raise ValueError("preflight transcript is not at its content-addressed path")
    transcript = transcript_path.read_bytes()
    size = transcript_record.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("preflight transcript size is invalid")
    if len(transcript) != size:
        raise ValueError("preflight transcript size mismatch")
    if sha256(transcript).hexdigest() != transcript_sha:
        raise ValueError("preflight transcript SHA-256 mismatch")
    parsed_summary = parse_pytest_transcript(transcript, exit_code=0)
    recorded_summary = _require_mapping(report.get("pytest"), "pytest")
    if set(recorded_summary) != _PYTEST_SUMMARY_KEYS:
        raise ValueError("preflight pytest summary shape is invalid")
    _validate_summary_types(recorded_summary)
    if _canonical_json_sha256(recorded_summary) != _canonical_json_sha256(
        parsed_summary
    ):
        raise ValueError("preflight pytest summary does not match transcript")

    recorded_inventory = _require_mapping(report.get("tests"), "tests")
    current_inventory = _test_inventory(root)
    if _canonical_json_sha256(recorded_inventory) != _canonical_json_sha256(
        current_inventory
    ):
        raise ValueError("preflight test inventory no longer matches current tests")
    return report


def parse_pytest_transcript(
    transcript: bytes,
    *,
    exit_code: int,
) -> dict[str, Any]:
    """Parse one complete plain-pytest transcript and enforce a clean result."""

    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        raise TypeError("pytest exit code must be an integer")
    if exit_code != 0:
        raise ValueError(f"pytest exit code must be zero, got {exit_code}")
    if not isinstance(transcript, bytes) or not transcript:
        raise ValueError("pytest transcript must be nonempty bytes")
    try:
        text = transcript.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("pytest transcript is not valid UTF-8") from error
    if "\x00" in text:
        raise ValueError("pytest transcript contains a NUL byte")
    clean_lines = [
        _ANSI_ESCAPE_RE.sub("", line).strip()
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ]
    nonempty = [line for line in clean_lines if line]
    if not nonempty:
        raise ValueError("pytest transcript is empty")
    collected_matches = [
        int(match.group(1))
        for line in nonempty
        for match in _COLLECTED_RE.finditer(line)
    ]
    if len(collected_matches) != 1:
        raise ValueError("pytest transcript must contain exactly one collection count")
    collected = collected_matches[0]
    if collected < 1:
        raise ValueError("pytest collected no tests")

    terminal = nonempty[-1]
    unframed = terminal.strip("= ")
    match = _SUMMARY_RE.fullmatch(unframed)
    if match is None:
        raise ValueError("pytest transcript lacks a complete terminal summary")
    counts = {
        "passed": 0,
        "skipped": 0,
        "failed": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
        "deselected": 0,
        "warnings": 0,
    }
    seen: set[str] = set()
    for raw_token in match.group("body").split(","):
        token = raw_token.strip()
        token_match = _SUMMARY_TOKEN_RE.fullmatch(token)
        if token_match is None:
            raise ValueError(f"unknown pytest summary token: {token!r}")
        raw_name = token_match.group("name").lower()
        normalized_name = _SUMMARY_NAMES.get(raw_name)
        if normalized_name is None:
            raise ValueError(f"unknown pytest summary outcome: {raw_name!r}")
        if normalized_name in seen:
            raise ValueError(f"duplicate pytest summary outcome: {normalized_name}")
        seen.add(normalized_name)
        counts[normalized_name] = int(token_match.group("count"))

    outcome_total = sum(
        counts[name]
        for name in ("passed", "skipped", "failed", "errors", "xfailed", "xpassed")
    )
    if outcome_total != collected:
        raise ValueError(
            "pytest terminal outcomes do not equal the collected count: "
            f"{outcome_total} != {collected}"
        )
    if counts["passed"] < 1:
        raise ValueError("pytest transcript contains no passing tests")
    if counts["failed"] or counts["errors"] or counts["deselected"]:
        raise ValueError("pytest transcript is not a clean full-suite pass")
    return {
        "collected": collected,
        **counts,
        "duration_seconds": float(match.group("duration")),
        "summary_line": terminal,
    }


def full_pytest_command(project_root: Path) -> tuple[str, str, str]:
    """Return the sole accepted logical command for a complete suite run."""

    root = project_root.resolve()
    executable = _venv_python_path(root)
    if not executable.is_file():
        raise ValueError(f".venv Python does not exist: {executable}")
    return (_relative_path(executable, root), "-m", "pytest")


def _validate_full_pytest_command(
    command: Sequence[str], project_root: Path
) -> tuple[str, str, str]:
    if isinstance(command, (str, bytes)):
        raise TypeError("pytest command must be a sequence of arguments")
    normalized = tuple(command)
    if any(not isinstance(value, str) or not value for value in normalized):
        raise ValueError("pytest command arguments must be nonempty strings")
    expected = full_pytest_command(project_root)
    if normalized != expected:
        raise ValueError(f"only the complete pytest command is accepted: {expected!r}")
    return expected


def _venv_python_path(project_root: Path) -> Path:
    windows = project_root / ".venv" / "Scripts" / "python.exe"
    posix = project_root / ".venv" / "bin" / "python"
    if windows.is_file():
        return windows.resolve()
    if posix.is_file():
        return posix.resolve()
    return (windows if os.name == "nt" else posix).resolve()


def _test_inventory(project_root: Path) -> dict[str, Any]:
    tests_root = (project_root / "tests").resolve()
    _relative_path(tests_root, project_root)
    if not tests_root.is_dir():
        raise ValueError(f"tests directory does not exist: {tests_root}")
    paths = sorted(
        (path.resolve() for path in tests_root.rglob("*.py") if path.is_file()),
        key=lambda path: _relative_path(path, project_root),
    )
    if not paths:
        raise ValueError("tests/**/*.py inventory is empty")
    files = [
        {
            "path": _relative_path(path, project_root),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]
    return {
        "root": "tests",
        "file_count": len(files),
        "inventory_sha256": _canonical_json_sha256(files),
        "files": files,
    }


def _normalize_bundle_hashes(values: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(values, Mapping) or set(values) != _BUNDLE_HASH_KEYS:
        raise ValueError(
            "expected bundle hashes must contain exactly bundle_file_sha256 "
            "and payload_sha256"
        )
    return {
        key: _sha256_text(values[key], f"freeze bundle {key}")
        for key in sorted(_BUNDLE_HASH_KEYS)
    }


def _validate_summary_types(summary: Mapping[str, Any]) -> None:
    count_keys = _PYTEST_SUMMARY_KEYS - {"duration_seconds", "summary_line"}
    for key in count_keys:
        value = summary.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"preflight pytest {key} count is invalid")
    duration = summary.get("duration_seconds")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise TypeError("preflight pytest duration is invalid")
    if duration < 0:
        raise ValueError("preflight pytest duration is invalid")
    if not isinstance(summary.get("summary_line"), str):
        raise TypeError("preflight pytest summary_line must be a string")


def _transcript_path(report_path: Path, transcript_sha256: str) -> Path:
    return report_path.with_name(
        f"{report_path.stem}.pytest-{transcript_sha256}.txt"
    ).resolve()


def _write_content_addressed(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise ValueError("content-addressed transcript path contains other bytes")
        return
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = (
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"preflight report already exists and is immutable: {path}"
            ) from error
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json_object(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read preflight report: {path}") from error
    if not isinstance(value, dict):
        raise TypeError("preflight report root must be an object")
    return value


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _sha256_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _canonical_json_sha256(value: Any) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(rendered).hexdigest()


def _safe_path(path: Path, project_root: Path, *, must_exist: bool) -> Path:
    resolved = path.resolve()
    _relative_path(resolved, project_root)
    if must_exist and not resolved.is_file():
        raise ValueError(f"preflight path does not exist: {resolved}")
    return resolved


def _path_from_record(value: Any, project_root: Path) -> Path:
    if not isinstance(value, str):
        raise TypeError("preflight transcript path must be a string")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"unsafe preflight transcript path: {value!r}")
    path = project_root.joinpath(*pure.parts).resolve()
    _relative_path(path, project_root)
    if not path.is_file():
        raise ValueError(f"preflight transcript does not exist: {path}")
    return path


def _relative_path(path: Path, project_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(project_root.resolve())
    except ValueError as error:
        raise ValueError(f"path is outside project root: {path}") from error
    if not relative.parts:
        return "."
    return relative.as_posix()


def _validate_utc_timestamp(value: Any) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("created_at_utc must be an ISO-8601 UTC string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("created_at_utc is not valid ISO-8601") from error
    if parsed.tzinfo != UTC:
        raise ValueError("created_at_utc must use UTC")


if __name__ == "__main__":
    raise SystemExit(main())
