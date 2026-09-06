"""Yearly orchestration for dealer double-riichi aggregation."""

import json
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

from mahjong_analysis.aggregation import (
    DealerDoubleRiichiStats,
    aggregate_dealer_double_riichi,
)


FIRST_YEAR = 2009
LAST_YEAR = 2025
SUPPORTED_YEARS = tuple(range(FIRST_YEAR, LAST_YEAR + 1))
EXPECTED_RELEASE_TAG = "v2.0.0"
SCHEMA_VERSION = 1
TARGET_RULE_CODE = "00a9"


@dataclass(frozen=True)
class DatasetValidationMetadata:
    """Validated dataset identity and observed raw-file counts."""

    repository: str
    release_tag: str
    raw_file_counts: dict[int, int]


@dataclass(frozen=True)
class YearlyDealerDoubleRiichiResult:
    """Dealer double-riichi aggregation result for one year."""

    year: int
    scanned_files: int
    target_games: int
    east_kyokus: int
    dealer_double_riichi: int
    dealer_win: int
    other_win: int
    draw: int

    def __post_init__(self) -> None:
        values = (
            self.year,
            self.scanned_files,
            self.target_games,
            self.east_kyokus,
            self.dealer_double_riichi,
            self.dealer_win,
            self.other_win,
            self.draw,
        )
        if any(type(value) is not int for value in values):
            raise TypeError("yearly result values must be integers")
        if self.year not in SUPPORTED_YEARS:
            raise ValueError(f"unsupported year: {self.year}")
        if any(value < 0 for value in values[1:]):
            raise ValueError("yearly result counts must not be negative")
        if self.dealer_win + self.other_win + self.draw != (self.dealer_double_riichi):
            raise ValueError("dealer double-riichi result counts are inconsistent")

    @property
    def win_rate(self) -> float | None:
        """Return the dealer win rate, or None with no observations."""
        if self.dealer_double_riichi == 0:
            return None
        return self.dealer_win / self.dealer_double_riichi


@dataclass(frozen=True)
class DealerDoubleRiichiTotals:
    """Counts combined across all selected years."""

    scanned_files: int
    target_games: int
    east_kyokus: int
    dealer_double_riichi: int
    dealer_win: int
    other_win: int
    draw: int

    def __post_init__(self) -> None:
        values = (
            self.scanned_files,
            self.target_games,
            self.east_kyokus,
            self.dealer_double_riichi,
            self.dealer_win,
            self.other_win,
            self.draw,
        )
        if any(type(value) is not int for value in values):
            raise TypeError("total result values must be integers")
        if any(value < 0 for value in values):
            raise ValueError("total result counts must not be negative")
        if self.dealer_win + self.other_win + self.draw != (self.dealer_double_riichi):
            raise ValueError("dealer double-riichi total counts are inconsistent")

    @property
    def win_rate(self) -> float | None:
        """Return the combined dealer win rate, or None with no observations."""
        if self.dealer_double_riichi == 0:
            return None
        return self.dealer_win / self.dealer_double_riichi


@dataclass(frozen=True)
class YearlyDealerDoubleRiichiSummary:
    """Complete deterministic output for selected years."""

    repository: str
    release_tag: str
    selected_years: tuple[int, ...]
    years: tuple[YearlyDealerDoubleRiichiResult, ...]
    totals: DealerDoubleRiichiTotals


class OutputConsistencyError(RuntimeError):
    """Report that an output update and its best-effort rollback both failed."""

    def __init__(
        self,
        write_error: OSError,
        rollback_errors: tuple[tuple[Path, OSError], ...],
    ) -> None:
        super().__init__(
            "output update failed and best-effort rollback did not complete; "
            "output consistency is not guaranteed"
        )
        self.write_error = write_error
        self.rollback_errors = rollback_errors


@dataclass(frozen=True)
class _OutputSnapshot:
    existed: bool
    content: bytes | None


ProgressCallback = Callable[[int, int], None]


def normalize_years(years: Iterable[int]) -> tuple[int, ...]:
    """Validate requested years and return them in ascending order."""
    values = tuple(years)
    if not values:
        raise ValueError("at least one year must be selected")
    if any(type(year) is not int for year in values):
        raise TypeError("years must be integers")

    invalid = sorted(year for year in values if year not in SUPPORTED_YEARS)
    if invalid:
        raise ValueError(f"unsupported years: {invalid}")
    if len(set(values)) != len(values):
        raise ValueError("duplicate years are not allowed")
    return tuple(sorted(values))


def load_dataset_validation_metadata(
    path: str | PathLike[str],
    years: Iterable[int],
    *,
    expected_release_tag: str = EXPECTED_RELEASE_TAG,
) -> DatasetValidationMetadata:
    """Load the validated dataset identity and requested yearly file counts."""
    selected_years = normalize_years(years)
    with open(path, encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("dataset validation summary must be a JSON object")
    repository = data.get("repository")
    release_tag = data.get("release_tag")
    if not isinstance(repository, str) or not repository:
        raise ValueError("dataset repository must be a non-empty string")
    if release_tag != expected_release_tag:
        raise ValueError(
            f"dataset release_tag must be {expected_release_tag!r}: {release_tag!r}"
        )
    if data.get("validation_failed") is not False:
        raise ValueError("dataset validation summary has not passed")
    if data.get("archive_hashes_verified") is not True:
        raise ValueError("dataset archive hashes have not been verified")

    year_values = data.get("years")
    if not isinstance(year_values, list):
        raise ValueError("dataset validation years must be a list")

    raw_file_counts: dict[int, int] = {}
    for value in year_values:
        if not isinstance(value, dict):
            raise ValueError("dataset validation year must be an object")
        year = value.get("year")
        raw = value.get("raw")
        if type(year) is not int or not isinstance(raw, dict):
            raise ValueError("dataset validation year has invalid fields")
        count = raw.get("total_mjson_files")
        if type(count) is not int or count < 0:
            raise ValueError(f"invalid raw file count for year {year}")
        if year in raw_file_counts:
            raise ValueError(f"duplicate year in dataset validation summary: {year}")
        raw_file_counts[year] = count

    missing = [year for year in selected_years if year not in raw_file_counts]
    if missing:
        raise ValueError(f"dataset validation summary is missing years: {missing}")

    return DatasetValidationMetadata(
        repository=repository,
        release_tag=release_tag,
        raw_file_counts={year: raw_file_counts[year] for year in selected_years},
    )


def _result_from_dict(value: object) -> YearlyDealerDoubleRiichiResult:
    if not isinstance(value, dict):
        raise ValueError("known yearly result must be an object")
    field_names = (
        "year",
        "scanned_files",
        "target_games",
        "east_kyokus",
        "dealer_double_riichi",
        "dealer_win",
        "other_win",
        "draw",
    )
    if any(field not in value for field in field_names):
        raise ValueError("known yearly result is missing required fields")
    return YearlyDealerDoubleRiichiResult(
        year=value["year"],
        scanned_files=value["scanned_files"],
        target_games=value["target_games"],
        east_kyokus=value["east_kyokus"],
        dealer_double_riichi=value["dealer_double_riichi"],
        dealer_win=value["dealer_win"],
        other_win=value["other_win"],
        draw=value["draw"],
    )


def load_known_results(
    path: str | PathLike[str],
    *,
    expected_release_tag: str = EXPECTED_RELEASE_TAG,
) -> dict[int, YearlyDealerDoubleRiichiResult]:
    """Load independently verified integer results by year."""
    with open(path, encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("known results manifest must be a JSON object")
    if data.get("release_tag") != expected_release_tag:
        raise ValueError("known results release_tag does not match the dataset")
    values = data.get("years")
    if not isinstance(values, list):
        raise ValueError("known results years must be a list")

    results: dict[int, YearlyDealerDoubleRiichiResult] = {}
    for value in values:
        result = _result_from_dict(value)
        if result.year in results:
            raise ValueError(f"duplicate year in known results: {result.year}")
        results[result.year] = result
    return results


def _from_aggregate(
    scanned_files: int,
    stats: DealerDoubleRiichiStats,
) -> YearlyDealerDoubleRiichiResult:
    return YearlyDealerDoubleRiichiResult(
        year=stats.year,
        scanned_files=scanned_files,
        target_games=stats.target_games,
        east_kyokus=stats.east_kyokus,
        dealer_double_riichi=stats.dealer_double_riichi,
        dealer_win=stats.dealer_win,
        other_win=stats.other_win,
        draw=stats.draw,
    )


def _paths_with_progress(
    year: int,
    paths: Iterable[Path],
    progress_interval: int,
    progress_callback: ProgressCallback | None,
) -> Iterable[Path]:
    for processed, path in enumerate(paths, 1):
        yield path
        if progress_callback is not None and processed % progress_interval == 0:
            progress_callback(year, processed)


def _compare_known_result(
    actual: YearlyDealerDoubleRiichiResult,
    expected: YearlyDealerDoubleRiichiResult,
) -> None:
    if actual == expected:
        return

    field_names = (
        "scanned_files",
        "target_games",
        "east_kyokus",
        "dealer_double_riichi",
        "dealer_win",
        "other_win",
        "draw",
    )
    differences = ", ".join(
        f"{field}=expected {getattr(expected, field)}, actual {getattr(actual, field)}"
        for field in field_names
        if getattr(actual, field) != getattr(expected, field)
    )
    raise RuntimeError(f"known result mismatch for year {actual.year}: {differences}")


def _build_totals(
    years: Iterable[YearlyDealerDoubleRiichiResult],
) -> DealerDoubleRiichiTotals:
    values = tuple(years)
    return DealerDoubleRiichiTotals(
        scanned_files=sum(value.scanned_files for value in values),
        target_games=sum(value.target_games for value in values),
        east_kyokus=sum(value.east_kyokus for value in values),
        dealer_double_riichi=sum(value.dealer_double_riichi for value in values),
        dealer_win=sum(value.dealer_win for value in values),
        other_win=sum(value.other_win for value in values),
        draw=sum(value.draw for value in values),
    )


def aggregate_dealer_double_riichi_years(
    years: Iterable[int],
    raw_root: str | PathLike[str],
    dataset: DatasetValidationMetadata,
    *,
    known_results: Mapping[int, YearlyDealerDoubleRiichiResult] | None = None,
    progress_interval: int = 10_000,
    progress_callback: ProgressCallback | None = None,
) -> YearlyDealerDoubleRiichiSummary:
    """Aggregate selected years sequentially using the production aggregator."""
    selected_years = normalize_years(years)
    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive")
    missing_expectations = [
        year for year in selected_years if year not in dataset.raw_file_counts
    ]
    if missing_expectations:
        raise ValueError(f"dataset metadata is missing years: {missing_expectations}")

    root = Path(raw_root)
    missing_directories = [
        year for year in selected_years if not (root / str(year)).is_dir()
    ]
    if missing_directories:
        raise FileNotFoundError(
            f"raw year directories do not exist: {missing_directories}"
        )

    results: list[YearlyDealerDoubleRiichiResult] = []
    expected_results = known_results or {}
    for year in selected_years:
        year_directory = root / str(year)
        try:
            paths = sorted(
                (
                    path
                    for path in year_directory.iterdir()
                    if path.is_file() and path.suffix == ".mjson"
                ),
                key=lambda path: path.name,
            )
        except OSError as error:
            raise RuntimeError(f"failed to enumerate raw year {year}") from error

        scanned_files = len(paths)
        expected_files = dataset.raw_file_counts[year]
        if scanned_files != expected_files:
            raise RuntimeError(
                f"raw file count mismatch for year {year}: "
                f"expected {expected_files}, found {scanned_files}"
            )

        try:
            stats = aggregate_dealer_double_riichi(
                year,
                _paths_with_progress(
                    year,
                    paths,
                    progress_interval,
                    progress_callback,
                ),
            )
        except Exception as error:
            raise RuntimeError(f"failed to aggregate year {year}") from error

        result = _from_aggregate(scanned_files, stats)
        expected = expected_results.get(year)
        if expected is not None:
            _compare_known_result(result, expected)
        results.append(result)
        del paths

    result_values = tuple(results)
    return YearlyDealerDoubleRiichiSummary(
        repository=dataset.repository,
        release_tag=dataset.release_tag,
        selected_years=selected_years,
        years=result_values,
        totals=_build_totals(result_values),
    )


def _result_to_dict(
    result: YearlyDealerDoubleRiichiResult | DealerDoubleRiichiTotals,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "scanned_files": result.scanned_files,
        "target_games": result.target_games,
        "east_kyokus": result.east_kyokus,
        "dealer_double_riichi": result.dealer_double_riichi,
        "dealer_win": result.dealer_win,
        "other_win": result.other_win,
        "draw": result.draw,
        "win_rate": result.win_rate,
    }
    if isinstance(result, YearlyDealerDoubleRiichiResult):
        value = {"year": result.year, **value}
    return value


def yearly_summary_to_dict(
    summary: YearlyDealerDoubleRiichiSummary,
) -> dict[str, Any]:
    """Convert yearly results to the canonical JSON-compatible structure."""
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "repository": summary.repository,
            "release_tag": summary.release_tag,
        },
        "analysis": {
            "rule_code": TARGET_RULE_CODE,
            "aka_flag": True,
            "bakaze": "E",
            "dealer_double_riichi": "established",
        },
        "selected_years": list(summary.selected_years),
        "years": [_result_to_dict(result) for result in summary.years],
        "totals": _result_to_dict(summary.totals),
    }


def render_yearly_markdown(summary: YearlyDealerDoubleRiichiSummary) -> str:
    """Render a deterministic human-readable yearly report."""

    def format_rate(rate: float | None) -> str:
        return "N/A" if rate is None else f"{rate:.2%}"

    lines = [
        "# Dealer double-riichi yearly results",
        "",
        f"- Dataset: `{summary.repository}` `{summary.release_tag}`",
        f"- Selected years: {', '.join(map(str, summary.selected_years))}",
        f"- Target: rule code `{TARGET_RULE_CODE}`, `aka_flag is True`",
        '- Rounds: East only (`bakaze == "E"`)',
        "- Event: established dealer double-riichi",
        "",
        "| Year | Scanned | Target games | East kyokus | Dealer double riichi | "
        "Dealer win | Other win | Draw | Win rate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in summary.years:
        lines.append(
            f"| {result.year} | {result.scanned_files} | {result.target_games} | "
            f"{result.east_kyokus} | {result.dealer_double_riichi} | "
            f"{result.dealer_win} | {result.other_win} | {result.draw} | "
            f"{format_rate(result.win_rate)} |"
        )

    totals = summary.totals
    lines.append(
        f"| **Total** | **{totals.scanned_files}** | "
        f"**{totals.target_games}** | **{totals.east_kyokus}** | "
        f"**{totals.dealer_double_riichi}** | **{totals.dealer_win}** | "
        f"**{totals.other_win}** | **{totals.draw}** | "
        f"**{format_rate(totals.win_rate)}** |"
    )
    lines.append("")
    return "\n".join(lines)


def validate_output_paths(
    json_path: str | PathLike[str],
    markdown_path: str | PathLike[str],
) -> tuple[Path, Path]:
    """Reject JSON and Markdown targets that resolve to the same path."""
    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    if json_target.resolve() == markdown_target.resolve():
        raise ValueError("JSON and Markdown output paths must be different")
    return json_target, markdown_target


def _write_temporary_bytes(target: Path, content: bytes) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            file.write(content)
    except BaseException:
        if temporary_path is not None:
            _remove_temporary_file(temporary_path)
        raise
    return temporary_path


def _remove_temporary_file(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _snapshot_output(path: Path) -> _OutputSnapshot:
    try:
        return _OutputSnapshot(True, path.read_bytes())
    except FileNotFoundError:
        return _OutputSnapshot(False, None)


def _restore_output(path: Path, snapshot: _OutputSnapshot) -> None:
    if not snapshot.existed:
        path.unlink(missing_ok=True)
        return

    if snapshot.content is None:
        raise RuntimeError("existing output snapshot has no content")
    temporary_path = _write_temporary_bytes(path, snapshot.content)
    try:
        os.replace(temporary_path, path)
    finally:
        _remove_temporary_file(temporary_path)


def write_yearly_outputs(
    summary: YearlyDealerDoubleRiichiSummary,
    json_path: str | PathLike[str],
    markdown_path: str | PathLike[str],
) -> None:
    """Atomically replace each output, with best-effort pair rollback."""
    json_target, markdown_target = validate_output_paths(json_path, markdown_path)
    json_content = (
        json.dumps(
            yearly_summary_to_dict(summary),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    markdown_content = render_yearly_markdown(summary)
    json_temporary: Path | None = None
    markdown_temporary: Path | None = None

    try:
        json_temporary = _write_temporary_bytes(
            json_target,
            json_content.encode("utf-8"),
        )
        markdown_temporary = _write_temporary_bytes(
            markdown_target,
            markdown_content.encode("utf-8"),
        )
        json_snapshot = _snapshot_output(json_target)
        markdown_snapshot = _snapshot_output(markdown_target)

        os.replace(json_temporary, json_target)
        try:
            os.replace(markdown_temporary, markdown_target)
        except OSError as write_error:
            rollback_errors: list[tuple[Path, OSError]] = []
            for target, snapshot in (
                (json_target, json_snapshot),
                (markdown_target, markdown_snapshot),
            ):
                try:
                    _restore_output(target, snapshot)
                except OSError as rollback_error:
                    rollback_errors.append((target, rollback_error))
            if rollback_errors:
                raise OutputConsistencyError(
                    write_error,
                    tuple(rollback_errors),
                ) from write_error
            raise
    finally:
        _remove_temporary_file(json_temporary)
        _remove_temporary_file(markdown_temporary)
