"""Fast validation utilities for the extracted MJAI dataset."""

import gzip
import hashlib
import json
import os
import zlib
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from zipfile import BadZipFile, ZipFile

from mahjong_analysis.mjai import extract_rule_code


Compression = Literal["gzip", "plain", "unknown"]
FirstEventStatus = Literal[
    "readable_start_game",
    "empty",
    "json_decode_error",
    "not_object",
    "not_start_game",
    "unreadable",
]
AkaFlagStatus = Literal[
    "true",
    "false",
    "missing",
    "invalid",
    "unavailable",
]

GZIP_MAGIC = b"\x1f\x8b"
COMPRESSION_VALUES = ("gzip", "plain", "unknown")
FIRST_EVENT_STATUS_VALUES = (
    "readable_start_game",
    "empty",
    "json_decode_error",
    "not_object",
    "not_start_game",
    "unreadable",
)
AKA_FLAG_STATUS_VALUES = (
    "true",
    "false",
    "missing",
    "invalid",
    "unavailable",
)
FILENAME_YEAR_STATUS_VALUES = ("match", "mismatch", "unavailable")
DEFAULT_ERROR_SAMPLE_LIMIT = 10
HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ValidationIssue:
    """One validation problem tied to a repository-relative path."""

    category: str
    relative_path: str
    exception_type: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class FileValidationResult:
    """The information obtained from one MJAI filename and its first line."""

    relative_path: str
    filename_valid: bool
    filename_year: int | None
    filename_year_status: Literal["match", "mismatch", "unavailable"]
    rule_code: str | None
    compression: Compression
    first_event_status: FirstEventStatus
    aka_flag_status: AkaFlagStatus
    target_game: bool
    issues: tuple[ValidationIssue, ...]


@dataclass(frozen=True)
class ReleaseAsset:
    """Expected GitHub release metadata for one yearly ZIP asset."""

    year: int
    filename: str
    expected_size_bytes: int
    expected_sha256: str


@dataclass(frozen=True)
class ReleaseAssetManifest:
    """Expected release metadata that is independent of local observations."""

    repository: str
    release_tag: str
    assets: tuple[ReleaseAsset, ...]


@dataclass(frozen=True)
class ArchiveValidationResult:
    """Observed and expected properties for one yearly ZIP archive."""

    filename: str
    expected_size_bytes: int
    observed_size_bytes: int | None
    size_matches: bool | None
    expected_sha256: str
    observed_sha256: str | None
    sha256_matches: bool | None
    sha256_skipped: bool
    observed_mjson_entries: int | None
    unexpected_archive_entries: int | None
    mjson_filenames: frozenset[str] | None
    issues: tuple[ValidationIssue, ...]


@dataclass(frozen=True)
class RawYearValidationResult:
    """Aggregated first-line observations for one raw yearly directory."""

    total_mjson_files: int
    valid_filenames: int
    invalid_filenames: int
    filename_year_counts: dict[str, int]
    rule_code_counts: dict[str, int]
    compression_counts: dict[str, int]
    first_event_status_counts: dict[str, int]
    aka_flag_counts: dict[str, int]
    aka_flag_by_rule_code: dict[str, dict[str, int]]
    target_games: int
    unexpected_raw_entries: dict[str, int]
    unexpected_raw_entry_samples: dict[str, tuple[str, ...]]
    files_with_errors: int
    error_counts: dict[str, int]
    error_samples: dict[str, tuple[ValidationIssue, ...]]
    mjson_filenames: frozenset[str] | None


@dataclass(frozen=True)
class MjsonFilenameSetComparison:
    """Bounded summary of the ZIP/raw MJAI filename-set comparison."""

    matches: bool | None
    archive_only_count: int | None
    raw_only_count: int | None
    archive_only_samples: tuple[str, ...]
    raw_only_samples: tuple[str, ...]


@dataclass(frozen=True)
class ValidationSummary:
    """Combined file and dataset-level validation issues for one year."""

    files_with_errors: int
    error_counts: dict[str, int]
    error_samples: dict[str, tuple[ValidationIssue, ...]]
    failed: bool


@dataclass(frozen=True)
class YearValidationSummary:
    """Archive and extracted-data validation results for one year."""

    year: int
    archive: ArchiveValidationResult
    raw: RawYearValidationResult
    mjson_filename_set: MjsonFilenameSetComparison
    validation: ValidationSummary


@dataclass(frozen=True)
class DatasetValidationSummary:
    """Validation summary for selected years of one dataset release."""

    repository: str
    release_tag: str
    target_rule_code: str
    selected_years: tuple[int, ...]
    totals: dict[str, Any]
    years: tuple[YearValidationSummary, ...]
    archive_hashes_verified: bool
    validation_failed: bool


class _IssueCollector:
    def __init__(self, sample_limit: int) -> None:
        if sample_limit < 0:
            raise ValueError("sample_limit must not be negative")
        self.sample_limit = min(sample_limit, DEFAULT_ERROR_SAMPLE_LIMIT)
        self.counts: Counter[str] = Counter()
        self.samples: defaultdict[str, list[ValidationIssue]] = defaultdict(list)

    def add(self, issue: ValidationIssue) -> None:
        self.counts[issue.category] += 1
        samples = self.samples[issue.category]
        if len(samples) < self.sample_limit:
            samples.append(issue)

    def add_count_and_samples(
        self,
        category: str,
        count: int,
        samples: Iterable[ValidationIssue],
    ) -> None:
        self.counts[category] += count
        destination = self.samples[category]
        for issue in samples:
            if len(destination) >= self.sample_limit:
                break
            destination.append(issue)

    def frozen_counts(self) -> dict[str, int]:
        return dict(sorted(self.counts.items()))

    def frozen_samples(self) -> dict[str, tuple[ValidationIssue, ...]]:
        return {
            category: tuple(self.samples[category])
            for category in sorted(self.samples)
            if self.samples[category]
        }


def _relative_mjson_path(expected_year: int, path: Path) -> str:
    return f"{expected_year}/{path.name}"


def _safe_error_message(
    error: BaseException,
    path: Path,
    relative_path: str,
) -> str:
    message = str(error)
    candidates = {str(path), str(path.absolute())}
    for candidate in sorted(candidates, key=len, reverse=True):
        if candidate:
            message = message.replace(candidate, relative_path)
    return message


def _issue(
    category: str,
    relative_path: str,
    *,
    error: BaseException | None = None,
    path: Path | None = None,
) -> ValidationIssue:
    if error is None:
        return ValidationIssue(category, relative_path)
    message = str(error)
    if path is not None:
        message = _safe_error_message(error, path, relative_path)
    return ValidationIssue(
        category=category,
        relative_path=relative_path,
        exception_type=type(error).__name__,
        message=message,
    )


def inspect_mjai_file(
    path: str | PathLike[str],
    expected_year: int,
    *,
    target_rule_code: str = "00a9",
) -> FileValidationResult:
    """Inspect one filename, compression marker, and first JSON line only."""
    file_path = Path(path)
    relative_path = _relative_mjson_path(expected_year, file_path)
    issues: list[ValidationIssue] = []
    filename_valid = False
    filename_year: int | None = None
    filename_year_status: Literal["match", "mismatch", "unavailable"]
    filename_year_status = "unavailable"
    rule_code: str | None = None
    compression: Compression = "unknown"
    first_event_status: FirstEventStatus = "unreadable"
    aka_flag_status: AkaFlagStatus = "unavailable"

    try:
        rule_code = extract_rule_code(file_path)
    except ValueError as error:
        issues.append(
            _issue(
                "invalid_filename",
                relative_path,
                error=error,
                path=file_path,
            )
        )
    else:
        filename_valid = True
        filename_year = int(file_path.name[:4])
        if filename_year == expected_year:
            filename_year_status = "match"
        else:
            filename_year_status = "mismatch"
            issues.append(_issue("year_mismatch", relative_path))

    def result() -> FileValidationResult:
        return FileValidationResult(
            relative_path=relative_path,
            filename_valid=filename_valid,
            filename_year=filename_year,
            filename_year_status=filename_year_status,
            rule_code=rule_code,
            compression=compression,
            first_event_status=first_event_status,
            aka_flag_status=aka_flag_status,
            target_game=(rule_code == target_rule_code and aka_flag_status == "true"),
            issues=tuple(issues),
        )

    try:
        binary_file = open(file_path, "rb")
    except OSError as error:
        issues.append(
            _issue(
                "binary_open_error",
                relative_path,
                error=error,
                path=file_path,
            )
        )
        return result()

    with binary_file:
        try:
            magic = binary_file.read(2)
            binary_file.seek(0)
        except OSError as error:
            issues.append(
                _issue(
                    "binary_open_error",
                    relative_path,
                    error=error,
                    path=file_path,
                )
            )
            return result()

        compression = "gzip" if magic == GZIP_MAGIC else "plain"
        try:
            if compression == "gzip":
                with gzip.GzipFile(fileobj=binary_file, mode="rb") as gzip_file:
                    first_line_bytes = gzip_file.readline()
            else:
                first_line_bytes = binary_file.readline()
        except (gzip.BadGzipFile, EOFError, zlib.error) as error:
            issues.append(
                _issue(
                    "gzip_first_line_read_error",
                    relative_path,
                    error=error,
                    path=file_path,
                )
            )
            return result()
        except OSError as error:
            category = (
                "gzip_first_line_read_error"
                if compression == "gzip"
                else "plain_first_line_read_error"
            )
            issues.append(
                _issue(
                    category,
                    relative_path,
                    error=error,
                    path=file_path,
                )
            )
            return result()

    if not first_line_bytes:
        first_event_status = "empty"
        issues.append(_issue("empty_file", relative_path))
        return result()

    try:
        first_line = first_line_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        issues.append(
            _issue(
                "utf8_decode_error",
                relative_path,
                error=error,
                path=file_path,
            )
        )
        return result()

    try:
        first_event = json.loads(first_line)
    except json.JSONDecodeError as error:
        first_event_status = "json_decode_error"
        issues.append(
            _issue(
                "json_decode_error",
                relative_path,
                error=error,
                path=file_path,
            )
        )
        return result()

    if not isinstance(first_event, dict):
        first_event_status = "not_object"
        issues.append(_issue("first_event_not_object", relative_path))
        return result()

    if first_event.get("type") != "start_game":
        first_event_status = "not_start_game"
        issues.append(_issue("first_event_not_start_game", relative_path))
        return result()

    first_event_status = "readable_start_game"
    if "aka_flag" not in first_event:
        aka_flag_status = "missing"
        issues.append(_issue("aka_flag_missing", relative_path))
    else:
        aka_flag = first_event["aka_flag"]
        if aka_flag is True:
            aka_flag_status = "true"
        elif aka_flag is False:
            aka_flag_status = "false"
        else:
            aka_flag_status = "invalid"
            issues.append(_issue("aka_flag_invalid", relative_path))

    return result()


def _complete_counts(
    counter: Mapping[str, int],
    values: Iterable[str],
) -> dict[str, int]:
    return {value: counter.get(value, 0) for value in values}


def _validate_raw_summary_invariants(summary: RawYearValidationResult) -> None:
    total = summary.total_mjson_files
    if summary.valid_filenames + summary.invalid_filenames != total:
        raise RuntimeError("filename counts do not equal total_mjson_files")
    for name, counts in (
        ("filename year", summary.filename_year_counts),
        ("compression", summary.compression_counts),
        ("first event", summary.first_event_status_counts),
        ("aka_flag", summary.aka_flag_counts),
    ):
        if sum(counts.values()) != total:
            raise RuntimeError(f"{name} counts do not equal total_mjson_files")
    if sum(summary.rule_code_counts.values()) != summary.valid_filenames:
        raise RuntimeError("rule_code counts do not equal valid filenames")
    for rule_code, count in summary.rule_code_counts.items():
        if sum(summary.aka_flag_by_rule_code[rule_code].values()) != count:
            raise RuntimeError(
                f"aka_flag counts for {rule_code} do not equal rule count"
            )


def validate_raw_year_directory(
    year: int,
    raw_year_directory: str | PathLike[str],
    *,
    target_rule_code: str = "00a9",
    error_sample_limit: int = DEFAULT_ERROR_SAMPLE_LIMIT,
    progress_callback: Callable[[int], None] | None = None,
) -> RawYearValidationResult:
    """Validate direct child MJAI files in one yearly raw directory."""
    raw_directory = Path(raw_year_directory)
    issue_collector = _IssueCollector(error_sample_limit)
    unexpected_counts: Counter[str] = Counter()
    unexpected_samples: defaultdict[str, list[str]] = defaultdict(list)
    filename_year_counts: Counter[str] = Counter()
    rule_code_counts: Counter[str] = Counter()
    compression_counts: Counter[str] = Counter()
    first_event_counts: Counter[str] = Counter()
    aka_flag_counts: Counter[str] = Counter()
    aka_by_rule_code: defaultdict[str, Counter[str]] = defaultdict(Counter)
    total_mjson_files = 0
    valid_filenames = 0
    target_games = 0
    files_with_errors = 0
    mjson_filenames: set[str] = set()

    try:
        entries = sorted(raw_directory.iterdir(), key=lambda entry: entry.name)
    except OSError as error:
        relative_path = f"data/raw/{year}"
        issue_collector.add(
            _issue(
                "raw_directory_error",
                relative_path,
                error=error,
                path=raw_directory,
            )
        )
        return RawYearValidationResult(
            total_mjson_files=0,
            valid_filenames=0,
            invalid_filenames=0,
            filename_year_counts=_complete_counts({}, FILENAME_YEAR_STATUS_VALUES),
            rule_code_counts={},
            compression_counts=_complete_counts({}, COMPRESSION_VALUES),
            first_event_status_counts=_complete_counts({}, FIRST_EVENT_STATUS_VALUES),
            aka_flag_counts=_complete_counts({}, AKA_FLAG_STATUS_VALUES),
            aka_flag_by_rule_code={},
            target_games=0,
            unexpected_raw_entries={"files": 0, "subdirectories": 0},
            unexpected_raw_entry_samples={},
            files_with_errors=0,
            error_counts=issue_collector.frozen_counts(),
            error_samples=issue_collector.frozen_samples(),
            mjson_filenames=None,
        )

    for entry in entries:
        if entry.suffix == ".mjson" and entry.is_file():
            mjson_filenames.add(entry.name)
            file_result = inspect_mjai_file(
                entry,
                year,
                target_rule_code=target_rule_code,
            )
            total_mjson_files += 1
            valid_filenames += int(file_result.filename_valid)
            filename_year_counts[file_result.filename_year_status] += 1
            compression_counts[file_result.compression] += 1
            first_event_counts[file_result.first_event_status] += 1
            aka_flag_counts[file_result.aka_flag_status] += 1
            target_games += int(file_result.target_game)
            if file_result.rule_code is not None:
                rule_code_counts[file_result.rule_code] += 1
                aka_by_rule_code[file_result.rule_code][
                    file_result.aka_flag_status
                ] += 1
            if file_result.issues:
                files_with_errors += 1
                for issue in file_result.issues:
                    issue_collector.add(issue)
            if progress_callback is not None:
                progress_callback(total_mjson_files)
            continue

        if entry.is_dir():
            category = "unexpected_raw_subdirectory"
            unexpected_key = "subdirectories"
        else:
            category = "unexpected_raw_file"
            unexpected_key = "files"
        unexpected_counts[unexpected_key] += 1
        relative_path = f"{year}/{entry.name}"
        if len(unexpected_samples[unexpected_key]) < issue_collector.sample_limit:
            unexpected_samples[unexpected_key].append(relative_path)
        issue_collector.add(_issue(category, relative_path))

    aka_by_rule = {
        rule_code: _complete_counts(aka_by_rule_code[rule_code], AKA_FLAG_STATUS_VALUES)
        for rule_code in sorted(aka_by_rule_code)
    }
    result = RawYearValidationResult(
        total_mjson_files=total_mjson_files,
        valid_filenames=valid_filenames,
        invalid_filenames=total_mjson_files - valid_filenames,
        filename_year_counts=_complete_counts(
            filename_year_counts, FILENAME_YEAR_STATUS_VALUES
        ),
        rule_code_counts=dict(sorted(rule_code_counts.items())),
        compression_counts=_complete_counts(compression_counts, COMPRESSION_VALUES),
        first_event_status_counts=_complete_counts(
            first_event_counts, FIRST_EVENT_STATUS_VALUES
        ),
        aka_flag_counts=_complete_counts(aka_flag_counts, AKA_FLAG_STATUS_VALUES),
        aka_flag_by_rule_code=aka_by_rule,
        target_games=target_games,
        unexpected_raw_entries={
            "files": unexpected_counts["files"],
            "subdirectories": unexpected_counts["subdirectories"],
        },
        unexpected_raw_entry_samples={
            category: tuple(unexpected_samples[category])
            for category in ("files", "subdirectories")
            if unexpected_samples[category]
        },
        files_with_errors=files_with_errors,
        error_counts=issue_collector.frozen_counts(),
        error_samples=issue_collector.frozen_samples(),
        mjson_filenames=frozenset(mjson_filenames),
    )
    _validate_raw_summary_invariants(result)
    return result


def load_release_asset_manifest(
    path: str | PathLike[str],
) -> ReleaseAssetManifest:
    """Load and validate the expected GitHub release asset manifest."""
    with open(path, encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("release asset manifest must be a JSON object")
    repository = data.get("repository")
    release_tag = data.get("release_tag")
    asset_values = data.get("assets")
    if not isinstance(repository, str) or not repository:
        raise ValueError("manifest repository must be a non-empty string")
    if not isinstance(release_tag, str) or not release_tag:
        raise ValueError("manifest release_tag must be a non-empty string")
    if not isinstance(asset_values, list):
        raise ValueError("manifest assets must be a list")

    assets: list[ReleaseAsset] = []
    seen_years: set[int] = set()
    for value in asset_values:
        if not isinstance(value, dict):
            raise ValueError("each manifest asset must be an object")
        year = value.get("year")
        filename = value.get("filename")
        expected_size = value.get("expected_size_bytes")
        expected_sha256 = value.get("expected_sha256")
        if type(year) is not int:
            raise ValueError("asset year must be an integer")
        if year in seen_years:
            raise ValueError(f"duplicate asset year: {year}")
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"asset filename must be a string for {year}")
        if type(expected_size) is not int or expected_size < 0:
            raise ValueError(f"asset size must be a non-negative int for {year}")
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise ValueError(f"asset SHA-256 must be lowercase hex for {year}")
        seen_years.add(year)
        assets.append(
            ReleaseAsset(
                year=year,
                filename=filename,
                expected_size_bytes=expected_size,
                expected_sha256=expected_sha256,
            )
        )

    return ReleaseAssetManifest(
        repository=repository,
        release_tag=release_tag,
        assets=tuple(sorted(assets, key=lambda asset: asset.year)),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_release_archive(
    asset: ReleaseAsset,
    archive_path: str | PathLike[str],
    *,
    skip_sha256: bool = False,
    relative_path: str | None = None,
) -> ArchiveValidationResult:
    """Observe one local ZIP and compare release-provided size and SHA-256."""
    path = Path(archive_path)
    issue_path = relative_path or path.name
    issues: list[ValidationIssue] = []
    observed_size: int | None = None
    size_matches: bool | None = None
    observed_sha256: str | None = None
    sha256_matches: bool | None = None
    observed_mjson_entries: int | None = None
    unexpected_archive_entries: int | None = None
    mjson_filenames: frozenset[str] | None = None

    try:
        observed_size = path.stat().st_size
    except OSError as error:
        issues.append(
            _issue(
                "archive_open_error",
                issue_path,
                error=error,
                path=path,
            )
        )
    else:
        size_matches = observed_size == asset.expected_size_bytes
        if not size_matches:
            issues.append(_issue("archive_size_mismatch", issue_path))

        if not skip_sha256:
            try:
                observed_sha256 = _sha256_file(path)
            except OSError as error:
                issues.append(
                    _issue(
                        "archive_hash_error",
                        issue_path,
                        error=error,
                        path=path,
                    )
                )
            else:
                sha256_matches = observed_sha256 == asset.expected_sha256
                if not sha256_matches:
                    issues.append(_issue("archive_sha256_mismatch", issue_path))

        try:
            with ZipFile(path) as archive:
                entries = archive.infolist()
        except (BadZipFile, OSError) as error:
            issues.append(
                _issue(
                    "archive_read_error",
                    issue_path,
                    error=error,
                    path=path,
                )
            )
        else:
            mjson_entries = [
                entry
                for entry in entries
                if not entry.is_dir() and entry.filename.endswith(".mjson")
            ]
            observed_mjson_entries = len(mjson_entries)
            unexpected_archive_entries = len(entries) - observed_mjson_entries
            mjson_filenames = frozenset(
                PurePosixPath(entry.filename).name for entry in mjson_entries
            )

    return ArchiveValidationResult(
        filename=asset.filename,
        expected_size_bytes=asset.expected_size_bytes,
        observed_size_bytes=observed_size,
        size_matches=size_matches,
        expected_sha256=asset.expected_sha256,
        observed_sha256=observed_sha256,
        sha256_matches=sha256_matches,
        sha256_skipped=skip_sha256,
        observed_mjson_entries=observed_mjson_entries,
        unexpected_archive_entries=unexpected_archive_entries,
        mjson_filenames=mjson_filenames,
        issues=tuple(issues),
    )


def build_year_validation_summary(
    year: int,
    archive: ArchiveValidationResult,
    raw: RawYearValidationResult,
    *,
    error_sample_limit: int = DEFAULT_ERROR_SAMPLE_LIMIT,
) -> YearValidationSummary:
    """Combine archive and raw observations and compare counts and names."""
    collector = _IssueCollector(error_sample_limit)
    for category, count in raw.error_counts.items():
        collector.add_count_and_samples(
            category,
            count,
            raw.error_samples.get(category, ()),
        )
    for issue in archive.issues:
        collector.add(issue)

    if (
        archive.observed_mjson_entries is not None
        and archive.observed_mjson_entries != raw.total_mjson_files
    ):
        collector.add(_issue("raw_archive_count_mismatch", f"data/raw/{year}"))

    if archive.mjson_filenames is None or raw.mjson_filenames is None:
        filename_set = MjsonFilenameSetComparison(
            matches=None,
            archive_only_count=None,
            raw_only_count=None,
            archive_only_samples=(),
            raw_only_samples=(),
        )
    else:
        archive_only = archive.mjson_filenames - raw.mjson_filenames
        raw_only = raw.mjson_filenames - archive.mjson_filenames
        matches = not archive_only and not raw_only
        filename_set = MjsonFilenameSetComparison(
            matches=matches,
            archive_only_count=len(archive_only),
            raw_only_count=len(raw_only),
            archive_only_samples=tuple(sorted(archive_only)[: collector.sample_limit]),
            raw_only_samples=tuple(sorted(raw_only)[: collector.sample_limit]),
        )
        if not matches:
            collector.add(_issue("raw_archive_filename_mismatch", f"data/raw/{year}"))

    error_counts = collector.frozen_counts()
    validation = ValidationSummary(
        files_with_errors=raw.files_with_errors,
        error_counts=error_counts,
        error_samples=collector.frozen_samples(),
        failed=bool(error_counts),
    )
    return YearValidationSummary(
        year=year,
        archive=replace(archive, mjson_filenames=None),
        raw=replace(raw, mjson_filenames=None),
        mjson_filename_set=filename_set,
        validation=validation,
    )


def _sum_nested_counts(
    mappings: Iterable[Mapping[str, int]],
) -> dict[str, int]:
    result: Counter[str] = Counter()
    for mapping in mappings:
        result.update(mapping)
    return dict(sorted(result.items()))


def _sum_aka_counts_by_rule_code(
    year_summaries: Iterable[YearValidationSummary],
) -> dict[str, dict[str, int]]:
    totals: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for year_summary in year_summaries:
        for rule_code, counts in year_summary.raw.aka_flag_by_rule_code.items():
            totals[rule_code].update(counts)
    return {
        rule_code: _complete_counts(totals[rule_code], AKA_FLAG_STATUS_VALUES)
        for rule_code in sorted(totals)
    }


def build_dataset_validation_summary(
    manifest: ReleaseAssetManifest,
    year_summaries: Iterable[YearValidationSummary],
    *,
    target_rule_code: str = "00a9",
) -> DatasetValidationSummary:
    """Build release-level totals solely from observed yearly summaries."""
    years = tuple(sorted(year_summaries, key=lambda summary: summary.year))
    selected_years = tuple(summary.year for summary in years)
    archive_entry_values = [
        summary.archive.observed_mjson_entries
        for summary in years
        if summary.archive.observed_mjson_entries is not None
    ]
    totals: dict[str, Any] = {
        "total_mjson_files": sum(summary.raw.total_mjson_files for summary in years),
        "archive_mjson_entries": sum(archive_entry_values),
        "archive_entry_count_unavailable_years": sum(
            summary.archive.observed_mjson_entries is None for summary in years
        ),
        "target_games": sum(summary.raw.target_games for summary in years),
        "valid_filenames": sum(summary.raw.valid_filenames for summary in years),
        "invalid_filenames": sum(summary.raw.invalid_filenames for summary in years),
        "filename_year_counts": _sum_nested_counts(
            summary.raw.filename_year_counts for summary in years
        ),
        "rule_code_counts": _sum_nested_counts(
            summary.raw.rule_code_counts for summary in years
        ),
        "compression_counts": _sum_nested_counts(
            summary.raw.compression_counts for summary in years
        ),
        "first_event_status_counts": _sum_nested_counts(
            summary.raw.first_event_status_counts for summary in years
        ),
        "aka_flag_counts": _sum_nested_counts(
            summary.raw.aka_flag_counts for summary in years
        ),
        "aka_flag_by_rule_code": _sum_aka_counts_by_rule_code(years),
        "files_with_errors": sum(
            summary.validation.files_with_errors for summary in years
        ),
        "error_counts": _sum_nested_counts(
            summary.validation.error_counts for summary in years
        ),
        "unexpected_raw_entries": _sum_nested_counts(
            summary.raw.unexpected_raw_entries for summary in years
        ),
        "unexpected_archive_entries": sum(
            summary.archive.unexpected_archive_entries or 0 for summary in years
        ),
    }
    return DatasetValidationSummary(
        repository=manifest.repository,
        release_tag=manifest.release_tag,
        target_rule_code=target_rule_code,
        selected_years=selected_years,
        totals=totals,
        years=years,
        archive_hashes_verified=all(
            summary.archive.sha256_matches is True for summary in years
        ),
        validation_failed=any(summary.validation.failed for summary in years),
    )


def _issue_to_dict(issue: ValidationIssue) -> dict[str, Any]:
    value: dict[str, Any] = {
        "category": issue.category,
        "relative_path": issue.relative_path,
    }
    if issue.exception_type is not None:
        value["exception_type"] = issue.exception_type
    if issue.message is not None:
        value["message"] = issue.message
    return value


def dataset_validation_summary_to_dict(
    summary: DatasetValidationSummary,
) -> dict[str, Any]:
    """Convert a validation summary to its stable JSON-compatible form."""
    year_values: list[dict[str, Any]] = []
    for year_summary in summary.years:
        archive = year_summary.archive
        raw = year_summary.raw
        validation = year_summary.validation
        year_values.append(
            {
                "year": year_summary.year,
                "archive": {
                    "filename": archive.filename,
                    "expected_size_bytes": archive.expected_size_bytes,
                    "observed_size_bytes": archive.observed_size_bytes,
                    "size_matches": archive.size_matches,
                    "expected_sha256": archive.expected_sha256,
                    "observed_sha256": archive.observed_sha256,
                    "sha256_matches": archive.sha256_matches,
                    "sha256_skipped": archive.sha256_skipped,
                    "observed_mjson_entries": archive.observed_mjson_entries,
                    "unexpected_archive_entries": (archive.unexpected_archive_entries),
                },
                "raw": {
                    "total_mjson_files": raw.total_mjson_files,
                    "filenames": {
                        "valid": raw.valid_filenames,
                        "invalid": raw.invalid_filenames,
                    },
                    "filename_year": raw.filename_year_counts,
                    "rule_code_counts": raw.rule_code_counts,
                    "compression_counts": raw.compression_counts,
                    "first_event_status_counts": (raw.first_event_status_counts),
                    "aka_flag_counts": raw.aka_flag_counts,
                    "aka_flag_by_rule_code": raw.aka_flag_by_rule_code,
                    "target_games": raw.target_games,
                    "unexpected_raw_entries": raw.unexpected_raw_entries,
                    "unexpected_raw_entry_samples": (raw.unexpected_raw_entry_samples),
                },
                "mjson_filename_set": {
                    "matches": year_summary.mjson_filename_set.matches,
                    "archive_only_count": (
                        year_summary.mjson_filename_set.archive_only_count
                    ),
                    "raw_only_count": year_summary.mjson_filename_set.raw_only_count,
                    "archive_only_samples": list(
                        year_summary.mjson_filename_set.archive_only_samples
                    ),
                    "raw_only_samples": list(
                        year_summary.mjson_filename_set.raw_only_samples
                    ),
                },
                "validation": {
                    "files_with_errors": validation.files_with_errors,
                    "error_counts": validation.error_counts,
                    "error_samples": {
                        category: [_issue_to_dict(issue) for issue in issues]
                        for category, issues in validation.error_samples.items()
                    },
                    "failed": validation.failed,
                },
            }
        )

    return {
        "repository": summary.repository,
        "release_tag": summary.release_tag,
        "target_rule_code": summary.target_rule_code,
        "selected_years": list(summary.selected_years),
        "totals": summary.totals,
        "archive_hashes_verified": summary.archive_hashes_verified,
        "validation_failed": summary.validation_failed,
        "years": year_values,
    }


def render_dataset_validation_markdown(
    summary: DatasetValidationSummary,
) -> str:
    """Render a deterministic human-readable report from summary data."""
    lines = [
        f"# Dataset validation: {summary.release_tag}",
        "",
        f"- Repository: `{summary.repository}`",
        f"- Target rule code: `{summary.target_rule_code}`",
        "- Selected years: " + ", ".join(str(year) for year in summary.selected_years),
        f"- Total MJAI files: {summary.totals['total_mjson_files']}",
        f"- Target games: {summary.totals['target_games']}",
        "- Archive hashes verified: "
        + ("yes" if summary.archive_hashes_verified else "no"),
        "- Validation: " + ("FAILED" if summary.validation_failed else "passed"),
        "",
        "## Yearly summary",
        "",
        "| Year | ZIP MJAI | Raw MJAI | Count match | Name match | Gzip | "
        "Plain | Target | Errors | Status |",
        "|---:|---:|---:|:---:|:---:|---:|---:|---:|---:|:---:|",
    ]
    for year_summary in summary.years:
        archive_count = year_summary.archive.observed_mjson_entries
        archive_count_text = "N/A" if archive_count is None else str(archive_count)
        count_matches = (
            archive_count is not None
            and archive_count == year_summary.raw.total_mjson_files
        )
        filename_matches = year_summary.mjson_filename_set.matches
        filename_match_text = (
            "N/A" if filename_matches is None else ("yes" if filename_matches else "no")
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    str(year_summary.year),
                    archive_count_text,
                    str(year_summary.raw.total_mjson_files),
                    "yes" if count_matches else "no",
                    filename_match_text,
                    str(year_summary.raw.compression_counts["gzip"]),
                    str(year_summary.raw.compression_counts["plain"]),
                    str(year_summary.raw.target_games),
                    str(sum(year_summary.validation.error_counts.values())),
                    "FAILED" if year_summary.validation.failed else "passed",
                )
            )
            + " |"
        )

    lines.extend(["", "## Validation issues", ""])
    error_counts = summary.totals["error_counts"]
    if not error_counts:
        lines.append("No validation issues were found.")
    else:
        lines.extend(("| Category | Count |", "|---|---:|"))
        for category, count in error_counts.items():
            lines.append(f"| `{category}` | {count} |")
    lines.append("")
    return "\n".join(lines)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_dataset_validation_outputs(
    summary: DatasetValidationSummary,
    json_path: str | PathLike[str],
    markdown_path: str | PathLike[str],
) -> None:
    """Atomically write canonical JSON and derived Markdown summaries."""
    json_content = (
        json.dumps(
            dataset_validation_summary_to_dict(summary),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    markdown_content = render_dataset_validation_markdown(summary)
    _atomic_write_text(Path(json_path), json_content)
    _atomic_write_text(Path(markdown_path), markdown_content)
