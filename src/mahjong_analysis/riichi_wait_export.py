"""Streaming, failure-safe export of production riichi-wait records."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import zlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from mahjong_analysis.mjai import (
    filter_east_kyokus,
    is_target_game,
    load_mjai,
    split_kyoku,
)
from mahjong_analysis.riichi import extract_established_riichis
from mahjong_analysis.riichi_wait_dataset import (
    COMPRESSION_LEVEL,
    DATASET_NAME,
    INPUT_ORDERING,
    RULE_CODE,
    SCHEMA_VERSION,
    RiichiWaitDatasetRecord,
    build_dataset_record,
    serialize_dataset_record,
    validate_checkpoint_manifest,
    validate_dataset_integrity,
    validate_manifest,
)
from mahjong_analysis.riichi_wait_records import build_riichi_wait_record

FIRST_YEAR = 2009
LAST_YEAR = 2025
SUPPORTED_YEARS = tuple(range(FIRST_YEAR, LAST_YEAR + 1))
EXPECTED_RELEASE_TAG = "v2.0.0"
EXPECTED_RULE_CODE = RULE_CODE

ExtractionMode = Literal["full", "sample"]


@dataclass(frozen=True)
class GitMetadata:
    """Reproducibility metadata separated for deterministic unit tests."""

    commit: str
    worktree_clean: bool

    def __post_init__(self) -> None:
        if not isinstance(self.commit, str) or not self.commit:
            raise ValueError("git commit must be a non-empty string")
        if type(self.worktree_clean) is not bool:
            raise TypeError("worktree_clean must be a bool")


@dataclass(frozen=True)
class SourceValidation:
    """Validated source release identity and expected annual file counts."""

    repository: str
    release_tag: str
    validation_summary_path: str
    validation_summary_sha256: str
    archive_hashes_verified: bool
    raw_file_counts: Mapping[int, int]


@dataclass(frozen=True)
class ProgressUpdate:
    """A bounded progress event independent of wall-clock formatting."""

    year: int
    processed_files: int
    total_files: int
    target_games: int
    east_kyokus: int
    records: int
    elapsed_seconds: float


@dataclass(frozen=True)
class FileExtractionCounts:
    """Counts returned after one source file was processed successfully."""

    target_games: int
    east_kyokus: int
    established_riichis: int

    def __post_init__(self) -> None:
        values = (self.target_games, self.east_kyokus, self.established_riichis)
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("file extraction counts must be non-negative integers")
        if self.target_games not in {0, 1}:
            raise ValueError("one input file can contain at most one target game")
        if self.target_games == 0 and any(values[1:]):
            raise ValueError("a non-target game cannot contain exported results")


RecordWriter = Callable[[RiichiWaitDatasetRecord], None]
FileExtractor = Callable[[Path, Path, int, RecordWriter], FileExtractionCounts]
ProgressCallback = Callable[[ProgressUpdate], None]


def normalize_years(years: Iterable[int]) -> tuple[int, ...]:
    """Return unique supported years in ascending order."""
    values = tuple(years)
    if not values:
        raise ValueError("at least one year must be selected")
    if any(type(year) is not int for year in values):
        raise TypeError("years must be integers")
    if len(set(values)) != len(values):
        raise ValueError("duplicate years are not allowed")
    invalid = tuple(sorted(year for year in values if year not in SUPPORTED_YEARS))
    if invalid:
        raise ValueError(f"unsupported years: {invalid}")
    return tuple(sorted(values))


def load_source_validation(
    path: str | Path,
    years: Iterable[int],
    *,
    logical_path: str,
) -> SourceValidation:
    """Validate the source summary and return only export-relevant metadata."""
    selected_years = normalize_years(years)
    if not logical_path or "\\" in logical_path or Path(logical_path).is_absolute():
        raise ValueError("validation summary logical_path must be relative POSIX text")
    raw = Path(path).read_bytes()
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid dataset validation summary JSON") from error
    if not isinstance(data, dict):
        raise TypeError("dataset validation summary must be an object")
    if data.get("release_tag") != EXPECTED_RELEASE_TAG:
        raise ValueError(f"source release must be {EXPECTED_RELEASE_TAG}")
    repository = data.get("repository")
    if not isinstance(repository, str) or not repository:
        raise ValueError("source repository must be a non-empty string")
    if data.get("validation_failed") is not False:
        raise ValueError("source validation summary did not pass")
    if data.get("archive_hashes_verified") is not True:
        raise ValueError("source archives were not SHA256-verified")
    if data.get("target_rule_code") != EXPECTED_RULE_CODE:
        raise ValueError("source validation summary has an unexpected target rule")
    year_values = data.get("years")
    if not isinstance(year_values, list):
        raise TypeError("source validation years must be an array")
    counts: dict[int, int] = {}
    for value in year_values:
        if not isinstance(value, dict):
            raise TypeError("source validation year entries must be objects")
        year = value.get("year")
        raw_summary = value.get("raw")
        if type(year) is not int or not isinstance(raw_summary, dict):
            raise ValueError("source validation year entry is malformed")
        count = raw_summary.get("total_mjson_files")
        if type(count) is not int or count < 0:
            raise ValueError(f"invalid raw file count for year {year!r}")
        if year in counts:
            raise ValueError(f"duplicate source validation year: {year}")
        counts[year] = count
    missing = tuple(year for year in selected_years if year not in counts)
    if missing:
        raise ValueError(f"source validation summary is missing years: {missing}")
    return SourceValidation(
        repository=repository,
        release_tag=EXPECTED_RELEASE_TAG,
        validation_summary_path=logical_path,
        validation_summary_sha256=_sha256_bytes(raw),
        archive_hashes_verified=True,
        raw_file_counts={year: counts[year] for year in selected_years},
    )


def collect_git_metadata(project_root: str | Path) -> GitMetadata:
    """Read only the commit identity and clean/dirty status from Git."""
    root = Path(project_root)
    commit = subprocess.run(
        ("git", "-C", str(root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "-C", str(root), "status", "--porcelain"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return GitMetadata(commit=commit, worktree_clean=not bool(status))


def select_year_files(
    raw_root: str | Path,
    year: int,
    *,
    max_files: int | None,
) -> tuple[Path, ...]:
    """Select raw-root-relative POSIX-sorted files before target filtering."""
    if year not in SUPPORTED_YEARS:
        raise ValueError(f"unsupported year: {year}")
    if max_files is not None and (type(max_files) is not int or max_files < 1):
        raise ValueError("max_files must be a positive integer")
    root = Path(raw_root).resolve()
    year_root = root / str(year)
    if not year_root.is_dir():
        return ()
    paths = sorted(
        year_root.rglob("*.mjson"),
        key=lambda path: path.resolve().relative_to(root).as_posix(),
    )
    if max_files is not None:
        paths = paths[:max_files]
    return tuple(paths)


def extract_production_file(
    path: Path,
    raw_root: Path,
    year: int,
    write_record: RecordWriter,
) -> FileExtractionCounts:
    """Stream production records from one game into the supplied writer."""
    events = load_mjai(path)
    if not is_target_game(path, events):
        return FileExtractionCounts(0, 0, 0)
    relative_source_path = path.resolve().relative_to(raw_root.resolve()).as_posix()
    line_by_event_id = {
        id(event): line_number for line_number, event in enumerate(events, start=1)
    }
    east_kyokus = filter_east_kyokus(split_kyoku(events))
    record_count = 0
    for kyoku_events in east_kyokus:
        start = kyoku_events[0]
        start_line = line_by_event_id[id(start)]
        established = extract_established_riichis(kyoku_events)
        previous_reach_index = -1
        for riichi in established:
            if riichi.reach_event_index <= previous_reach_index:
                raise ValueError("established riichis are not in source event order")
            previous_reach_index = riichi.reach_event_index
            record = build_riichi_wait_record(riichi)
            write_record(
                build_dataset_record(
                    record,
                    year=year,
                    relative_source_path=relative_source_path,
                    start_kyoku_line=start_line,
                    reach_line=_event_line(
                        kyoku_events, record.reach_event_index, line_by_event_id
                    ),
                    declaration_dahai_line=_event_line(
                        kyoku_events,
                        record.declaration_dahai_event_index,
                        line_by_event_id,
                    ),
                    reach_accepted_line=_event_line(
                        kyoku_events,
                        record.reach_accepted_event_index,
                        line_by_event_id,
                    ),
                    bakaze=_required_string(start, "bakaze"),
                    kyoku=_required_int(start, "kyoku"),
                    honba=_required_int(start, "honba"),
                    oya=_required_int(start, "oya"),
                    scores_at_start=_required_scores(start),
                    dora_marker=_required_string(start, "dora_marker"),
                )
            )
            record_count += 1
    return FileExtractionCounts(1, len(east_kyokus), record_count)


def export_riichi_wait_dataset(
    *,
    raw_root: str | Path,
    output_root: str | Path,
    years: Iterable[int],
    validation_summary_path: str | Path,
    validation_summary_logical_path: str,
    git_metadata: GitMetadata,
    max_files: int | None = None,
    force: bool = False,
    resume: bool = False,
    allow_dirty: bool = False,
    progress_interval: int = 10_000,
    progress_callback: ProgressCallback | None = None,
    file_extractor: FileExtractor = extract_production_file,
    utc_now: Callable[[], datetime] = lambda: datetime.now(UTC),
    monotonic: Callable[[], float] = time.monotonic,
) -> Mapping[str, Any]:
    """Export selected years and atomically publish a completed manifest."""
    selected_years = normalize_years(years)
    if max_files is not None and len(selected_years) != 1:
        raise ValueError("max_files may only be used with exactly one year")
    if force and resume:
        raise ValueError("force and resume are mutually exclusive")
    if type(progress_interval) is not int or progress_interval < 1:
        raise ValueError("progress_interval must be a positive integer")
    if not isinstance(git_metadata, GitMetadata):
        raise TypeError("git_metadata must be GitMetadata")
    mode: ExtractionMode = "sample" if max_files is not None else "full"
    if allow_dirty and mode != "sample":
        raise ValueError("allow_dirty is only valid for a sample export")
    if not git_metadata.worktree_clean and not (mode == "sample" and allow_dirty):
        raise ValueError("a full export requires a clean Git worktree")

    source = load_source_validation(
        validation_summary_path,
        selected_years,
        logical_path=validation_summary_logical_path,
    )
    root = Path(raw_root).resolve()
    destination = Path(output_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    invocation = {
        "years": list(selected_years),
        "mode": mode,
        "max_files": max_files,
    }
    base = _base_manifest(
        selected_years,
        source,
        git_metadata,
        mode,
        max_files,
        invocation,
        utc_now,
    )
    checkpoint_path = destination / "manifest.json.part"
    manifest_path = destination / "manifest.json"
    previous_manifest_path = destination / "manifest.json.previous"

    if resume:
        checkpoint = _load_checkpoint(checkpoint_path, base)
    else:
        replacement = _prepare_new_run(
            destination,
            manifest_path,
            checkpoint_path,
            previous_manifest_path,
            base,
            force,
        )
        checkpoint = {
            **base,
            "checkpoint": True,
            "checkpoint_state": "processing",
            "replacement": replacement,
            "years": [],
        }
        _write_checkpoint(checkpoint_path, checkpoint)

    completed = _reconcile_checkpoint(
        destination,
        checkpoint,
        checkpoint_path,
        selected_years,
    )
    for year in selected_years:
        if year in completed:
            continue
        year_paths = select_year_files(root, year, max_files=max_files)
        if not year_paths:
            raise FileNotFoundError(f"year {year} has no input .mjson files")
        if mode == "full" and len(year_paths) != source.raw_file_counts[year]:
            raise ValueError(
                f"year {year} raw file count mismatch: "
                f"expected {source.raw_file_counts[year]}, found {len(year_paths)}"
            )
        output_name = f"{year}.jsonl.gz"
        part_path = destination / f"{output_name}.part"
        started = monotonic()
        counters = _write_year(
            year=year,
            paths=year_paths,
            raw_root=root,
            part_path=part_path,
            progress_interval=progress_interval,
            progress_callback=progress_callback,
            file_extractor=file_extractor,
            monotonic=monotonic,
            started=started,
        )
        elapsed = max(0.0, monotonic() - started)
        if counters["established_riichis"] != counters["output_records"]:
            raise RuntimeError("established riichi and output record counts differ")
        entry = {
            "year": year,
            **counters,
            "output_filename": output_name,
            "compressed_size_bytes": part_path.stat().st_size,
            "sha256": _sha256_file(part_path),
            "elapsed_seconds": elapsed,
            "status": "ready",
        }
        _replace_year_entry(checkpoint, entry)
        _write_checkpoint(checkpoint_path, checkpoint)
        _reconcile_ready_output(
            destination,
            checkpoint,
            checkpoint_path,
            entry,
        )
        entry["status"] = "complete"
        _write_checkpoint(checkpoint_path, checkpoint)
        completed[year] = entry

    checkpoint["checkpoint_state"] = "final_ready"
    _write_checkpoint(checkpoint_path, checkpoint)
    return _publish_final_ready_checkpoint(
        destination,
        checkpoint,
        checkpoint_path,
        manifest_path,
        previous_manifest_path,
        selected_years,
    )


def _write_year(
    *,
    year: int,
    paths: Sequence[Path],
    raw_root: Path,
    part_path: Path,
    progress_interval: int,
    progress_callback: ProgressCallback | None,
    file_extractor: FileExtractor,
    monotonic: Callable[[], float],
    started: float,
) -> dict[str, int]:
    scanned_files = 0
    target_games = 0
    east_kyokus = 0
    established_riichis = 0
    output_records = 0
    with open(part_path, "wb") as raw_file:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=COMPRESSION_LEVEL,
            fileobj=raw_file,
            mtime=0,
        ) as gzip_file:

            def write_record(record: RiichiWaitDatasetRecord) -> None:
                nonlocal output_records
                gzip_file.write(serialize_dataset_record(record))
                gzip_file.write(b"\n")
                output_records += 1

            for path in paths:
                counts = file_extractor(path, raw_root, year, write_record)
                if not isinstance(counts, FileExtractionCounts):
                    raise TypeError("file_extractor must return FileExtractionCounts")
                scanned_files += 1
                target_games += counts.target_games
                east_kyokus += counts.east_kyokus
                established_riichis += counts.established_riichis
                if progress_callback is not None and (
                    scanned_files % progress_interval == 0
                    or scanned_files == len(paths)
                ):
                    progress_callback(
                        ProgressUpdate(
                            year=year,
                            processed_files=scanned_files,
                            total_files=len(paths),
                            target_games=target_games,
                            east_kyokus=east_kyokus,
                            records=output_records,
                            elapsed_seconds=max(0.0, monotonic() - started),
                        )
                    )
            if progress_callback is not None and not paths:
                progress_callback(
                    ProgressUpdate(
                        year=year,
                        processed_files=0,
                        total_files=0,
                        target_games=0,
                        east_kyokus=0,
                        records=0,
                        elapsed_seconds=max(0.0, monotonic() - started),
                    )
                )
        raw_file.flush()
        os.fsync(raw_file.fileno())
    return {
        "scanned_files": scanned_files,
        "target_games": target_games,
        "east_kyokus": east_kyokus,
        "established_riichis": established_riichis,
        "output_records": output_records,
    }


def _base_manifest(
    years: tuple[int, ...],
    source: SourceValidation,
    git_metadata: GitMetadata,
    mode: ExtractionMode,
    max_files: int | None,
    invocation: Mapping[str, Any],
    utc_now: Callable[[], datetime],
) -> dict[str, Any]:
    created = utc_now()
    if created.tzinfo is None:
        raise ValueError("utc_now must return a timezone-aware datetime")
    created_text = created.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": DATASET_NAME,
        "created_at_utc": created_text,
        "source": {
            "repository": source.repository,
            "release_tag": source.release_tag,
            "validation_summary_path": source.validation_summary_path,
            "validation_summary_sha256": source.validation_summary_sha256,
            "archive_hashes_verified": source.archive_hashes_verified,
        },
        "scope": {
            "years": list(years),
            "rule_code": EXPECTED_RULE_CODE,
            "aka_flag": True,
            "bakaze": "E",
            "input_selection": {
                "ordering": INPUT_ORDERING,
                "max_files_before_target_filtering": max_files,
            },
            "extraction_mode": mode,
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
            "compression_level": COMPRESSION_LEVEL,
            "gzip_mtime": 0,
            "gzip_header_filename": "",
        },
        "generator": {
            "git_commit": git_metadata.commit,
            "worktree_clean": git_metadata.worktree_clean,
            "python_version": platform.python_version(),
            "zlib_version": zlib.ZLIB_VERSION,
            "invocation": dict(invocation),
        },
    }


def _prepare_new_run(
    destination: Path,
    manifest_path: Path,
    checkpoint_path: Path,
    previous_manifest_path: Path,
    expected_base: Mapping[str, Any],
    force: bool,
) -> dict[str, bool]:
    """Validate existing artifacts and prepare a same-identity replacement."""
    years = tuple(expected_base["scope"]["years"])
    existing_manifest: Mapping[str, Any] | None = None
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as file:
            existing_manifest = validate_manifest(
                json.load(file, parse_constant=_reject_json_constant)
            )
        _validate_replacement_identity(existing_manifest, expected_base)
        if not force:
            raise FileExistsError(f"completed dataset already exists: {manifest_path}")
        validate_dataset_integrity(
            destination,
            expected_mode=existing_manifest["scope"]["extraction_mode"],
        )

    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as file:
            validate_checkpoint_manifest(
                json.load(file, parse_constant=_reject_json_constant)
            )
        raise FileExistsError(
            "incomplete export checkpoint exists; use --resume "
            "(--force cannot reset an incomplete export)"
        )

    previous_manifest: Mapping[str, Any] | None = None
    if previous_manifest_path.exists():
        with open(previous_manifest_path, encoding="utf-8") as file:
            previous_manifest = validate_manifest(
                json.load(file, parse_constant=_reject_json_constant)
            )
        _validate_replacement_identity(previous_manifest, expected_base)
        if existing_manifest is None:
            raise ValueError("backup manifest exists without a resumable checkpoint")

    had_completed_manifest = existing_manifest is not None
    parts = _annual_part_files(destination)
    if parts:
        raise ValueError("unexpected annual part exists without a checkpoint")

    final_files = _annual_final_files(destination)
    unselected = tuple(
        path for path in final_files if _artifact_year(path) not in years
    )
    if unselected:
        raise ValueError("output root contains annual files outside selected years")
    if final_files and existing_manifest is None:
        raise ValueError("annual output exists without a manifest or checkpoint")

    if previous_manifest is not None and existing_manifest is not None:
        previous_manifest_path.unlink()
        _fsync_directory(destination)
    return {
        "force": force,
        "had_completed_manifest": had_completed_manifest,
    }


def _load_checkpoint(
    checkpoint_path: Path,
    expected_base: Mapping[str, Any],
) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError("resume requires manifest.json.part")
    with open(checkpoint_path, encoding="utf-8") as file:
        value = json.load(file, parse_constant=_reject_json_constant)
    data = dict(validate_checkpoint_manifest(value))
    for field in (
        "schema_version",
        "dataset_name",
        "source",
        "scope",
        "serialization",
        "generator",
    ):
        if data.get(field) != expected_base.get(field):
            raise ValueError(f"resume checkpoint configuration mismatch: {field}")
    return data


def _reconcile_checkpoint(
    destination: Path,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    selected_years: tuple[int, ...],
) -> dict[int, dict[str, Any]]:
    processing_part = _validate_checkpoint_artifacts(destination, checkpoint)
    if processing_part is not None:
        processing_part.unlink()
        _fsync_directory(destination)
    completed: dict[int, dict[str, Any]] = {}
    entries = checkpoint.get("years", [])
    if not isinstance(entries, list):
        raise TypeError("checkpoint years must be an array")
    for value in entries:
        if not isinstance(value, dict):
            raise TypeError("checkpoint year entry must be an object")
        year = value.get("year")
        status = value.get("status")
        if type(year) is not int or status not in {"ready", "complete"}:
            raise ValueError("invalid checkpoint year entry")
        if year not in selected_years:
            raise ValueError(f"checkpoint contains an unselected year: {year}")
        if year in completed:
            raise ValueError(f"duplicate checkpoint year: {year}")
        output_filename = value.get("output_filename")
        if output_filename != f"{year}.jsonl.gz":
            raise ValueError(f"invalid checkpoint output filename for year {year}")
        final_path = destination / output_filename
        part_path = destination / f"{output_filename}.part"
        if status == "ready":
            _reconcile_ready_output(
                destination,
                checkpoint,
                checkpoint_path,
                value,
            )
            value["status"] = "complete"
            _write_checkpoint(checkpoint_path, checkpoint)
        _verify_output_file(final_path, value)
        if part_path.exists():
            raise ValueError(f"complete year has unexpected part: {part_path.name}")
        completed[year] = value
    return completed


def _reconcile_ready_output(
    destination: Path,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    entry: Mapping[str, Any],
) -> None:
    """Resolve a ready year from final/part hashes without guessing."""
    output_filename = entry["output_filename"]
    final_path = destination / output_filename
    part_path = destination / f"{output_filename}.part"
    final_matches = _output_matches(final_path, entry)
    part_matches = _output_matches(part_path, entry)

    if final_path.exists() and part_path.exists():
        if final_matches and part_matches:
            part_path.unlink()
            _fsync_directory(destination)
            return
        if final_matches:
            raise ValueError(f"ready year has a corrupt extra part: {part_path.name}")
        if not part_matches:
            raise ValueError(
                f"ready year final and part both fail integrity: {output_filename}"
            )
    elif final_path.exists():
        if final_matches:
            return
        raise ValueError(
            f"ready year final fails integrity and part is missing: {output_filename}"
        )
    elif not part_matches:
        raise ValueError(f"ready year part is missing or corrupt: {part_path.name}")

    _invalidate_completion_marker_before_year_replace(
        destination,
        checkpoint,
        checkpoint_path,
    )
    os.replace(part_path, final_path)
    _fsync_directory(destination)
    _verify_output_file(final_path, entry)


def _invalidate_completion_marker_before_year_replace(
    destination: Path,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    replacement = checkpoint["replacement"]
    if replacement["had_completed_manifest"] is not True:
        if (destination / "manifest.json").exists():
            raise ValueError("unexpected completed manifest during new export")
        return
    manifest_path = destination / "manifest.json"
    previous_path = destination / "manifest.json.previous"
    if manifest_path.exists():
        current = validate_dataset_integrity(destination)
        _validate_replacement_identity(current, checkpoint)
        os.replace(manifest_path, previous_path)
        _fsync_directory(destination)
        return
    if not previous_path.is_file():
        raise ValueError("force replacement lost both current and backup manifests")
    with open(previous_path, encoding="utf-8") as file:
        previous = validate_manifest(
            json.load(file, parse_constant=_reject_json_constant)
        )
    _validate_replacement_identity(previous, checkpoint)


def _validate_checkpoint_artifacts(
    destination: Path,
    checkpoint: Mapping[str, Any],
) -> Path | None:
    """Validate the annual artifact set for one checkpoint state."""
    years = tuple(checkpoint["scope"]["years"])
    entries = {entry["year"]: entry for entry in checkpoint["years"]}
    manifest_path = destination / "manifest.json"
    previous_path = destination / "manifest.json.previous"
    replacement = checkpoint["replacement"]
    if replacement["had_completed_manifest"] is True:
        if not manifest_path.exists() and not previous_path.exists():
            raise ValueError("force checkpoint has neither current nor backup manifest")
        if manifest_path.exists():
            current = validate_dataset_integrity(destination)
            _validate_replacement_identity(current, checkpoint)
        if previous_path.exists():
            with open(previous_path, encoding="utf-8") as file:
                previous = validate_manifest(
                    json.load(file, parse_constant=_reject_json_constant)
                )
            _validate_replacement_identity(previous, checkpoint)
        if (
            manifest_path.exists()
            and previous_path.exists()
            and checkpoint["checkpoint_state"] != "final_ready"
        ):
            raise ValueError(
                "processing checkpoint has both current and backup manifests"
            )
    elif manifest_path.exists() or previous_path.exists():
        if (
            checkpoint["checkpoint_state"] != "final_ready"
            or not manifest_path.exists()
            or previous_path.exists()
        ):
            raise ValueError(
                "new export checkpoint has an unexpected completion marker"
            )
        current = validate_dataset_integrity(destination)
        _validate_replacement_identity(current, checkpoint)

    final_paths = _annual_final_files(destination)
    part_paths = _annual_part_files(destination)
    if any(_artifact_year(path) not in years for path in (*final_paths, *part_paths)):
        raise ValueError(
            "output root contains annual artifacts outside checkpoint scope"
        )

    for year, entry in entries.items():
        final_path = destination / entry["output_filename"]
        part_path = destination / f"{entry['output_filename']}.part"
        if entry["status"] == "complete":
            _verify_output_file(final_path, entry)
            if part_path.exists():
                raise ValueError(f"complete year has unexpected part: {part_path.name}")
        else:
            if not final_path.exists() and not part_path.exists():
                raise ValueError(f"ready year has neither final nor part: {year}")
            if part_path.exists() and not _output_matches(part_path, entry):
                raise ValueError(f"ready year part fails integrity: {part_path.name}")
            if (
                final_path.exists()
                and not _output_matches(final_path, entry)
                and not part_path.exists()
            ):
                raise ValueError(f"ready year final fails integrity: {final_path.name}")

    unrecorded_parts = tuple(
        path for path in part_paths if _artifact_year(path) not in entries
    )
    missing_years = tuple(year for year in years if year not in entries)
    processing_part: Path | None = None
    if unrecorded_parts:
        if (
            len(unrecorded_parts) != 1
            or not missing_years
            or _artifact_year(unrecorded_parts[0]) != missing_years[0]
            or checkpoint["checkpoint_state"] != "processing"
        ):
            raise ValueError("unexpected annual part is not the active processing year")
        processing_part = unrecorded_parts[0]

    if replacement["had_completed_manifest"] is not True:
        unrecorded_finals = tuple(
            path for path in final_paths if _artifact_year(path) not in entries
        )
        if unrecorded_finals:
            raise ValueError("unexpected annual final is not recorded by checkpoint")
    return processing_part


def _validate_replacement_identity(
    existing: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    if existing.get("schema_version") != expected.get("schema_version"):
        raise ValueError("existing dataset schema identity differs")
    if existing.get("dataset_name") != expected.get("dataset_name"):
        raise ValueError("existing dataset name differs")
    existing_scope = existing["scope"]
    expected_scope = expected["scope"]
    if existing_scope["extraction_mode"] != expected_scope["extraction_mode"]:
        raise ValueError("sample and full outputs must use separate roots")
    if tuple(existing_scope["years"]) != tuple(expected_scope["years"]):
        raise ValueError("force requires the same year selection as existing state")
    for field in ("source", "scope", "serialization"):
        if existing.get(field) != expected.get(field):
            raise ValueError(f"existing dataset {field} identity differs")


def _annual_final_files(destination: Path) -> tuple[Path, ...]:
    return tuple(sorted(destination.glob("[0-9][0-9][0-9][0-9].jsonl.gz")))


def _annual_part_files(destination: Path) -> tuple[Path, ...]:
    return tuple(sorted(destination.glob("[0-9][0-9][0-9][0-9].jsonl.gz.part")))


def _artifact_year(path: Path) -> int:
    return int(path.name[:4])


def _verify_output_file(path: Path, entry: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise ValueError(f"checkpoint output is missing: {path.name}")
    if path.stat().st_size != entry.get("compressed_size_bytes"):
        raise ValueError(f"checkpoint output size mismatch: {path.name}")
    if _sha256_file(path) != entry.get("sha256"):
        raise ValueError(f"checkpoint output SHA256 mismatch: {path.name}")


def _output_matches(path: Path, entry: Mapping[str, Any]) -> bool:
    return bool(
        path.is_file()
        and path.stat().st_size == entry.get("compressed_size_bytes")
        and _sha256_file(path) == entry.get("sha256")
    )


def _replace_year_entry(checkpoint: dict[str, Any], entry: dict[str, Any]) -> None:
    entries = checkpoint["years"]
    entries[:] = [value for value in entries if value.get("year") != entry["year"]]
    entries.append(entry)
    entries.sort(key=lambda value: value["year"])


def _completed_manifest(
    checkpoint: Mapping[str, Any],
    selected_years: tuple[int, ...],
) -> dict[str, Any]:
    if checkpoint.get("checkpoint") is not True:
        raise ValueError("completed manifest must be built from a checkpoint")
    if checkpoint.get("checkpoint_state") != "final_ready":
        raise ValueError("completed manifest requires a final_ready checkpoint")
    entries = checkpoint.get("years")
    if not isinstance(entries, list):
        raise TypeError("checkpoint years must be an array")
    by_year = {entry["year"]: entry for entry in entries}
    if tuple(sorted(by_year)) != selected_years:
        raise ValueError("cannot complete manifest before every year is complete")
    years = []
    for year in selected_years:
        entry = dict(by_year[year])
        if entry.pop("status", None) != "complete":
            raise ValueError(f"year {year} is not complete")
        years.append(entry)
    count_fields = (
        "scanned_files",
        "target_games",
        "east_kyokus",
        "established_riichis",
        "output_records",
    )
    totals = {field: sum(entry[field] for entry in years) for field in count_fields}
    result = {
        key: value
        for key, value in checkpoint.items()
        if key not in {"checkpoint", "checkpoint_state", "replacement", "years"}
    }
    result["years"] = years
    result["totals"] = totals
    return result


def _publish_final_ready_checkpoint(
    destination: Path,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    manifest_path: Path,
    previous_manifest_path: Path,
    selected_years: tuple[int, ...],
) -> Mapping[str, Any]:
    validate_checkpoint_manifest(checkpoint)
    _validate_checkpoint_artifacts(destination, checkpoint)
    final_manifest = _completed_manifest(checkpoint, selected_years)
    validate_manifest(final_manifest)
    if manifest_path.exists():
        current = validate_dataset_integrity(destination)
        _validate_replacement_identity(current, checkpoint)
    _write_checkpoint(manifest_path, final_manifest)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    if previous_manifest_path.exists():
        previous_manifest_path.unlink()
    _fsync_directory(destination)
    return final_manifest


def _write_checkpoint(path: Path, value: Mapping[str, Any]) -> None:
    temp_path = path.with_name(f"{path.name}.tmp")
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    with open(temp_path, "wb") as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp_path, path)
    _fsync_directory(path.parent)


def _event_line(
    kyoku: Sequence[dict[str, Any]],
    event_index: int,
    line_by_event_id: Mapping[int, int],
) -> int:
    try:
        return line_by_event_id[id(kyoku[event_index])]
    except (IndexError, KeyError) as error:
        raise ValueError(f"kyoku event {event_index} has no physical line") from error


def _required_string(event: Mapping[str, Any], name: str) -> str:
    value = event.get(name)
    if not isinstance(value, str):
        raise TypeError(f"start_kyoku {name} must be a string")
    return value


def _required_int(event: Mapping[str, Any], name: str) -> int:
    value = event.get(name)
    if type(value) is not int:
        raise TypeError(f"start_kyoku {name} must be an integer")
    return value


def _required_scores(event: Mapping[str, Any]) -> tuple[int, int, int, int]:
    value = event.get("scores")
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(type(score) is not int for score in value)
    ):
        raise TypeError("start_kyoku scores must contain four integers")
    return value[0], value[1], value[2], value[3]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _fsync_directory(path: Path) -> None:
    if sys.platform == "win32":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
