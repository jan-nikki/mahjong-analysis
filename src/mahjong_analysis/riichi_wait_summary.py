"""Streaming descriptive summaries for the canonical riichi-wait dataset."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mahjong_analysis.hand_waits import HAND_TYPE_ORDER, WAIT_SHAPE_ORDER
from mahjong_analysis.riichi_wait_dataset import (
    DATASET_NAME,
    RULE_CODE,
    SCHEMA_VERSION,
    SOURCE_RELEASE_TAG,
    SUPPORTED_YEARS,
    DatasetWaitDetail,
    RiichiWaitDatasetRecord,
    iter_dataset_records,
    validate_dataset_integrity,
)
from mahjong_analysis.tiles import TILE_KINDS, normalize_tile

ANALYSIS_NAME = "riichi-wait-summary-v1"
ANALYSIS_SCHEMA_VERSION = 1
CANONICAL_RECORD_COUNT = 10_706_714
CANONICAL_SOURCE_REPOSITORY = "NikkeTryHard/tenhou-to-mjai"
DEFAULT_DATASET_ROOT = Path("data/processed/riichi-waits-v1")
DEFAULT_OUTPUT_ROOT = Path("research/results/riichi-wait-summary-v1")
PUBLICATION_POINTER_FILENAME = "current.json"
PUBLICATION_GENERATIONS_DIRECTORY = "generations"
COMPLETION_MARKER_FILENAME = "complete.json"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
OUTPUT_FILENAMES = {
    "overall": "riichi-wait-summary-v1-overall.json",
    "yearly": "riichi-wait-summary-v1-yearly.json",
    "by_turn": "riichi-wait-summary-v1-by-turn.json",
    "markdown": "riichi-wait-summary-v1.md",
}


@dataclass(frozen=True)
class AnalysisGitMetadata:
    """Commit identity used by deterministic result provenance."""

    commit: str
    worktree_clean: bool

    def __post_init__(self) -> None:
        if not isinstance(self.commit, str) or not self.commit:
            raise ValueError("analysis git commit must be non-empty text")
        if type(self.worktree_clean) is not bool:
            raise TypeError("analysis worktree_clean must be a bool")


@dataclass(frozen=True)
class SummaryProgress:
    """Bounded progress information for a long canonical aggregation."""

    year: int
    annual_records: int
    total_records: int
    elapsed_seconds: float


ProgressCallback = Callable[[SummaryProgress], None]


@dataclass(frozen=True)
class SummaryBenchmarkYearResult:
    """One annual worker measurement from a non-publishing benchmark."""

    year: int
    records_processed: int
    elapsed_seconds: float


@dataclass(frozen=True)
class SummaryBenchmarkResult:
    """One worker-count measurement from a non-publishing benchmark."""

    workers_requested: int
    workers_used: int
    years: tuple[int, ...]
    record_limit_per_year: int
    records_processed: int
    elapsed_seconds: float
    records_per_second: float
    year_results: tuple[SummaryBenchmarkYearResult, ...]


@dataclass(frozen=True)
class _AnnualSummaryTask:
    year: int
    annual_path: str
    expected_output_records: int
    record_limit: int | None = None


@dataclass(frozen=True)
class _AnnualSummaryResult:
    year: int
    aggregation: SummaryAggregation
    records_processed: int
    elapsed_seconds: float


@dataclass(frozen=True)
class _WaitFacts:
    wait_tiles: tuple[str, ...]
    details: tuple[DatasetWaitDetail, ...]
    shapes: tuple[str, ...]
    hand_types: tuple[str, ...]
    is_multiwait: bool
    contains_ryanmen: bool
    is_pure_ryanmen: bool


@dataclass(frozen=True)
class _RecordContribution:
    formal: _WaitFacts
    adjusted: _WaitFacts
    fifth_tile_waits: tuple[str, ...]
    riichi_discard_number: int


@dataclass
class _RecordLevelAccumulator:
    record_count: int = 0
    wait_tile_counts: Counter[int] = field(default_factory=Counter)
    pure_ryanmen_count: int = 0
    contains_ryanmen_count: int = 0
    multiwait_count: int = 0
    shape_membership: Counter[str] = field(default_factory=Counter)
    shape_sets: Counter[tuple[str, ...]] = field(default_factory=Counter)
    hand_type_membership: Counter[str] = field(default_factory=Counter)
    hand_type_sets: Counter[tuple[str, ...]] = field(default_factory=Counter)
    multiple_shape_count: int = 0
    multiple_hand_type_count: int = 0

    def add(self, facts: _WaitFacts) -> None:
        self.record_count += 1
        self.wait_tile_counts[len(facts.wait_tiles)] += 1
        self.pure_ryanmen_count += facts.is_pure_ryanmen
        self.contains_ryanmen_count += facts.contains_ryanmen
        self.multiwait_count += facts.is_multiwait
        self.shape_sets[facts.shapes] += 1
        self.hand_type_sets[facts.hand_types] += 1
        self.multiple_shape_count += len(facts.shapes) > 1
        self.multiple_hand_type_count += len(facts.hand_types) > 1
        self.shape_membership.update(facts.shapes)
        self.hand_type_membership.update(facts.hand_types)

    def validate(self) -> None:
        denominator = self.record_count
        if sum(self.wait_tile_counts.values()) != denominator:
            raise ValueError("wait-tile-count distribution does not sum to records")
        if sum(self.shape_sets.values()) != denominator:
            raise ValueError("shape-set distribution does not sum to records")
        if sum(self.hand_type_sets.values()) != denominator:
            raise ValueError("hand-type-set distribution does not sum to records")
        if not (
            0 <= self.pure_ryanmen_count <= self.contains_ryanmen_count <= denominator
        ):
            raise ValueError("ryanmen counts violate record-level invariants")
        if not 0 <= self.multiwait_count <= denominator:
            raise ValueError("multiwait count exceeds record count")
        if not 0 <= self.multiple_shape_count <= denominator:
            raise ValueError("multiple-shape count exceeds record count")
        if not 0 <= self.multiple_hand_type_count <= denominator:
            raise ValueError("multiple-hand-type count exceeds record count")
        if any(shape not in WAIT_SHAPE_ORDER for shape in self.shape_membership):
            raise ValueError("unknown wait shape in membership")
        if any(
            hand_type not in HAND_TYPE_ORDER for hand_type in self.hand_type_membership
        ):
            raise ValueError("unknown hand type in membership")
        expected_shapes: Counter[str] = Counter()
        expected_hand_types: Counter[str] = Counter()
        for members, count in self.shape_sets.items():
            expected_shapes.update({member: count for member in members})
        for members, count in self.hand_type_sets.items():
            expected_hand_types.update({member: count for member in members})
        if expected_shapes != self.shape_membership:
            raise ValueError("shape membership disagrees with shape sets")
        if expected_hand_types != self.hand_type_membership:
            raise ValueError("hand-type membership disagrees with hand-type sets")
        if (
            sum(count for members, count in self.shape_sets.items() if len(members) > 1)
            != self.multiple_shape_count
        ):
            raise ValueError("multiple-shape count disagrees with shape sets")
        if (
            sum(
                count
                for members, count in self.hand_type_sets.items()
                if len(members) > 1
            )
            != self.multiple_hand_type_count
        ):
            raise ValueError("multiple-hand-type count disagrees with hand-type sets")


@dataclass
class _WaitTileAccumulator:
    observation_count: int = 0
    shape_membership: Counter[str] = field(default_factory=Counter)
    shape_sets: Counter[tuple[str, ...]] = field(default_factory=Counter)
    multiple_shape_count: int = 0

    def add(self, shapes: tuple[str, ...]) -> None:
        self.observation_count += 1
        self.shape_membership.update(shapes)
        self.shape_sets[shapes] += 1
        self.multiple_shape_count += len(shapes) > 1

    def validate(self) -> None:
        if sum(self.shape_sets.values()) != self.observation_count:
            raise ValueError("wait-tile shape sets do not sum to observations")
        if not 0 <= self.multiple_shape_count <= self.observation_count:
            raise ValueError("multiple-shape wait tiles exceed observations")
        expected_membership: Counter[str] = Counter()
        for members, count in self.shape_sets.items():
            expected_membership.update({member: count for member in members})
        if expected_membership != self.shape_membership:
            raise ValueError("wait-tile shape membership disagrees with shape sets")
        if (
            sum(count for members, count in self.shape_sets.items() if len(members) > 1)
            != self.multiple_shape_count
        ):
            raise ValueError("multiple-shape wait tiles disagree with shape sets")


@dataclass
class _DiagnosticsAccumulator:
    wait_tiles: _WaitTileAccumulator = field(default_factory=_WaitTileAccumulator)
    by_tile: dict[str, _WaitTileAccumulator] = field(
        default_factory=lambda: {tile: _WaitTileAccumulator() for tile in TILE_KINDS}
    )
    detail_count: int = 0
    detail_cross_table: Counter[tuple[str, str]] = field(default_factory=Counter)

    def add(self, facts: _WaitFacts) -> None:
        details_by_tile: dict[str, list[DatasetWaitDetail]] = defaultdict(list)
        for detail in facts.details:
            details_by_tile[detail.tile].append(detail)
            self.detail_cross_table[(detail.hand_type, detail.wait_shape)] += 1
            self.detail_count += 1
        for tile in facts.wait_tiles:
            shapes = _ordered_members(
                (detail.wait_shape for detail in details_by_tile[tile]),
                WAIT_SHAPE_ORDER,
            )
            self.wait_tiles.add(shapes)
            self.by_tile[tile].add(shapes)

    def validate(self, expected_observations: int) -> None:
        self.wait_tiles.validate()
        if self.wait_tiles.observation_count != expected_observations:
            raise ValueError("wait-tile observations disagree with record waits")
        for value in self.by_tile.values():
            value.validate()
        if sum(value.observation_count for value in self.by_tile.values()) != (
            expected_observations
        ):
            raise ValueError("per-tile observations do not sum to all observations")
        if sum(self.detail_cross_table.values()) != self.detail_count:
            raise ValueError("detail cross table does not sum to detail count")


@dataclass
class _SliceAccumulator:
    formal: _RecordLevelAccumulator = field(default_factory=_RecordLevelAccumulator)
    adjusted: _RecordLevelAccumulator = field(default_factory=_RecordLevelAccumulator)
    adjusted_zero_wait_count: int = 0

    def add(self, contribution: _RecordContribution) -> None:
        self.formal.add(contribution.formal)
        self.adjusted.add(contribution.adjusted)
        self.adjusted_zero_wait_count += not contribution.adjusted.wait_tiles

    def validate(self) -> None:
        self.formal.validate()
        self.adjusted.validate()
        if self.formal.record_count != self.adjusted.record_count:
            raise ValueError("formal and adjusted denominators differ")
        zero_from_distribution = self.adjusted.wait_tile_counts.get(0, 0)
        if self.adjusted_zero_wait_count != zero_from_distribution:
            raise ValueError("adjusted zero-wait count is inconsistent")


@dataclass
class _FullAccumulator:
    record_level: _SliceAccumulator = field(default_factory=_SliceAccumulator)
    riichi_discard_numbers: Counter[int] = field(default_factory=Counter)
    diagnostics: _DiagnosticsAccumulator = field(
        default_factory=_DiagnosticsAccumulator
    )
    fifth_tile_record_count: int = 0
    fifth_tile_observation_count: int = 0
    fifth_tile_only_record_count: int = 0
    fifth_tile_with_other_record_count: int = 0

    def add(self, contribution: _RecordContribution) -> None:
        self.record_level.add(contribution)
        self.riichi_discard_numbers[contribution.riichi_discard_number] += 1
        self.diagnostics.add(contribution.formal)
        if contribution.fifth_tile_waits:
            self.fifth_tile_record_count += 1
            self.fifth_tile_observation_count += len(contribution.fifth_tile_waits)
            if contribution.adjusted.wait_tiles:
                self.fifth_tile_with_other_record_count += 1
            else:
                self.fifth_tile_only_record_count += 1

    def validate(self) -> None:
        self.record_level.validate()
        denominator = self.record_level.formal.record_count
        if sum(self.riichi_discard_numbers.values()) != denominator:
            raise ValueError("discard-number distribution does not sum to records")
        expected_observations = sum(
            count * wait_count
            for wait_count, count in self.record_level.formal.wait_tile_counts.items()
        )
        self.diagnostics.validate(expected_observations)
        if self.fifth_tile_record_count != (
            self.fifth_tile_only_record_count + self.fifth_tile_with_other_record_count
        ):
            raise ValueError("fifth-tile affected records are not partitioned")
        if not 0 <= self.fifth_tile_record_count <= denominator:
            raise ValueError("fifth-tile record count exceeds records")
        if not (
            self.fifth_tile_record_count
            <= self.fifth_tile_observation_count
            <= expected_observations
        ):
            raise ValueError("fifth-tile observations violate wait-tile bounds")
        if self.fifth_tile_only_record_count != (
            self.record_level.adjusted_zero_wait_count
        ):
            raise ValueError("fifth-only and adjusted zero-wait counts differ")


@dataclass
class SummaryAggregation:
    """Bounded in-memory aggregate; never retains input records."""

    overall: _FullAccumulator = field(default_factory=_FullAccumulator)
    years: dict[int, _FullAccumulator] = field(default_factory=dict)
    turns: dict[int, _SliceAccumulator] = field(default_factory=dict)
    year_turns: dict[int, dict[int, _SliceAccumulator]] = field(default_factory=dict)

    def add(self, record: RiichiWaitDatasetRecord) -> None:
        if not isinstance(record, RiichiWaitDatasetRecord):
            raise TypeError("record must be a RiichiWaitDatasetRecord")
        contribution = _build_contribution(record)
        self.overall.add(contribution)
        year_result = self.years.get(record.year)
        if year_result is None:
            year_result = _FullAccumulator()
            self.years[record.year] = year_result
        year_result.add(contribution)
        turn_result = self.turns.get(record.riichi_discard_number)
        if turn_result is None:
            turn_result = _SliceAccumulator()
            self.turns[record.riichi_discard_number] = turn_result
        turn_result.add(contribution)
        year_turns = self.year_turns.get(record.year)
        if year_turns is None:
            year_turns = {}
            self.year_turns[record.year] = year_turns
        year_turn_result = year_turns.get(record.riichi_discard_number)
        if year_turn_result is None:
            year_turn_result = _SliceAccumulator()
            year_turns[record.riichi_discard_number] = year_turn_result
        year_turn_result.add(contribution)

    @property
    def record_count(self) -> int:
        return self.overall.record_level.formal.record_count

    def merge(self, other: SummaryAggregation) -> None:
        """Merge a bounded aggregate without retaining any source records."""
        if not isinstance(other, SummaryAggregation):
            raise TypeError("other must be a SummaryAggregation")
        other.validate()
        overlapping_years = set(self.years).intersection(other.years)
        if overlapping_years:
            raise ValueError(
                f"cannot merge duplicate years: {sorted(overlapping_years)}"
            )
        _merge_full_accumulator(self.overall, other.overall)
        for year in sorted(other.years):
            self.years[year] = other.years[year]
            self.year_turns[year] = other.year_turns[year]
        for turn in sorted(other.turns):
            target = self.turns.get(turn)
            if target is None:
                target = _SliceAccumulator()
                self.turns[turn] = target
            _merge_slice_accumulator(target, other.turns[turn])

    def validate(self, expected_years: tuple[int, ...] | None = None) -> None:
        self.overall.validate()
        for value in self.years.values():
            value.validate()
        for value in self.turns.values():
            value.validate()
        for turn_values in self.year_turns.values():
            for value in turn_values.values():
                value.validate()
        _validate_full_partition(self.overall, tuple(self.years.values()), "year")
        _validate_slice_partition(
            self.overall.record_level,
            tuple(self.turns.values()),
            "turn",
        )
        if set(self.years) != set(self.year_turns):
            raise ValueError("year and year-by-turn slices differ")
        for year, result in self.years.items():
            _validate_slice_partition(
                result.record_level,
                tuple(self.year_turns[year].values()),
                f"year {year} turn",
            )
        if expected_years is not None and tuple(sorted(self.years)) != expected_years:
            raise ValueError("aggregated years do not match the manifest")


@dataclass(frozen=True)
class SummaryAnalysis:
    """Validated aggregation and deterministic provenance envelope."""

    aggregation: SummaryAggregation
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.aggregation, SummaryAggregation):
            raise TypeError("aggregation must be SummaryAggregation")
        _validate_summary_metadata(
            self.metadata,
            expected_record_count=self.aggregation.record_count,
        )
        years = self.metadata["years"]
        self.aggregation.validate(expected_years=tuple(years))


def aggregate_riichi_wait_records(
    records: Iterable[RiichiWaitDatasetRecord],
) -> SummaryAggregation:
    """Aggregate validated DTOs without retaining them."""
    aggregation = SummaryAggregation()
    for record in records:
        aggregation.add(record)
    aggregation.validate()
    return aggregation


def summarize_riichi_wait_dataset(
    dataset_root: str | Path,
    *,
    analysis_git: AnalysisGitMetadata,
    project_root: str | Path,
    workers: int = 1,
    progress_interval: int = 100_000,
    progress_callback: ProgressCallback | None = None,
    require_canonical: bool = True,
) -> SummaryAnalysis:
    """Validate and aggregate manifest-listed annual streams deterministically."""
    if not isinstance(analysis_git, AnalysisGitMetadata):
        raise TypeError("analysis_git must be AnalysisGitMetadata")
    if not analysis_git.worktree_clean:
        raise ValueError("formal summary generation requires a clean worktree")
    _validate_workers(workers)
    if type(progress_interval) is not int or progress_interval < 1:
        raise ValueError("progress_interval must be a positive integer")
    root = Path(dataset_root)
    project = Path(project_root).resolve()
    manifest_path = root / "manifest.json"
    try:
        manifest_logical_path = manifest_path.resolve().relative_to(project).as_posix()
    except ValueError as error:
        raise ValueError(
            "canonical manifest must be inside the project root"
        ) from error
    manifest_bytes = manifest_path.read_bytes()
    manifest = validate_dataset_integrity(root, expected_mode="full", deep=False)
    if manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("input manifest changed during validation")
    years = tuple(entry["year"] for entry in manifest["years"])
    if require_canonical:
        _validate_canonical_manifest(manifest, years)

    started = time.monotonic()
    if workers == 1:
        aggregation = _aggregate_manifest_serially(
            root,
            manifest["years"],
            progress_interval=progress_interval,
            progress_callback=progress_callback,
            started=started,
        )
    else:
        tasks = _annual_summary_tasks(root, manifest["years"])
        processed_records = 0

        def report_result(result: _AnnualSummaryResult) -> None:
            nonlocal processed_records
            processed_records += result.records_processed
            if progress_callback is not None:
                progress_callback(
                    SummaryProgress(
                        result.year,
                        result.records_processed,
                        processed_records,
                        time.monotonic() - started,
                    )
                )

        annual_results = _run_annual_summary_tasks(
            tasks,
            workers=workers,
            result_callback=report_result,
        )
        aggregation = _merge_annual_summary_results(annual_results, years)
    aggregation.validate(expected_years=years)
    if aggregation.record_count != manifest["totals"]["output_records"]:
        raise ValueError("total record count does not match manifest")
    if aggregation.record_count != manifest["totals"]["established_riichis"]:
        raise ValueError("total record count does not match established riichis")
    if require_canonical and aggregation.record_count != CANONICAL_RECORD_COUNT:
        raise ValueError("canonical record count does not equal 10,706,714")

    metadata = _build_metadata(
        manifest,
        manifest_logical_path=manifest_logical_path,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        analysis_git=analysis_git,
    )
    return SummaryAnalysis(aggregation=aggregation, metadata=metadata)


def benchmark_riichi_wait_dataset(
    dataset_root: str | Path,
    *,
    years: tuple[int, ...] = tuple(range(2018, 2026)),
    record_limit_per_year: int = 100_000,
    worker_counts: tuple[int, ...] = (1, 2, 4, 6),
) -> tuple[SummaryBenchmarkResult, ...]:
    """Benchmark bounded annual aggregation without publishing any result."""
    if (
        not isinstance(years, tuple)
        or not years
        or any(type(year) is not int for year in years)
        or years != tuple(sorted(set(years)))
    ):
        raise ValueError("benchmark years must be a non-empty ascending tuple")
    if type(record_limit_per_year) is not int or record_limit_per_year < 1:
        raise ValueError("benchmark record limit must be a positive integer")
    if not isinstance(worker_counts, tuple) or not worker_counts:
        raise ValueError("benchmark worker counts must be a non-empty tuple")
    for workers in worker_counts:
        _validate_workers(workers)

    root = Path(dataset_root)
    manifest = validate_dataset_integrity(root, expected_mode="full", deep=False)
    entries_by_year = {entry["year"]: entry for entry in manifest["years"]}
    if any(year not in entries_by_year for year in years):
        raise ValueError("benchmark year is absent from the dataset manifest")
    selected_entries = tuple(entries_by_year[year] for year in years)
    tasks = _annual_summary_tasks(
        root,
        selected_entries,
        record_limit=record_limit_per_year,
    )

    benchmark_results: list[SummaryBenchmarkResult] = []
    expected_aggregation: SummaryAggregation | None = None
    for workers_requested in worker_counts:
        started = time.monotonic()
        annual_results = _run_annual_summary_tasks(
            tasks,
            workers=workers_requested,
        )
        aggregation = _merge_annual_summary_results(annual_results, years)
        elapsed = time.monotonic() - started
        if expected_aggregation is None:
            expected_aggregation = aggregation
        elif aggregation != expected_aggregation:
            raise ValueError("benchmark aggregation changed with worker count")
        records_processed = aggregation.record_count
        benchmark_results.append(
            SummaryBenchmarkResult(
                workers_requested=workers_requested,
                workers_used=min(workers_requested, len(tasks)),
                years=years,
                record_limit_per_year=record_limit_per_year,
                records_processed=records_processed,
                elapsed_seconds=elapsed,
                records_per_second=(
                    records_processed / elapsed if elapsed > 0 else float("inf")
                ),
                year_results=tuple(
                    SummaryBenchmarkYearResult(
                        result.year,
                        result.records_processed,
                        result.elapsed_seconds,
                    )
                    for result in sorted(annual_results, key=lambda item: item.year)
                ),
            )
        )
    return tuple(benchmark_results)


def _validate_workers(workers: object) -> None:
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")


def _aggregate_manifest_serially(
    root: Path,
    entries: Iterable[Mapping[str, Any]],
    *,
    progress_interval: int,
    progress_callback: ProgressCallback | None,
    started: float,
) -> SummaryAggregation:
    aggregation = SummaryAggregation()
    for entry in entries:
        year = entry["year"]
        annual_count = 0
        annual_path = root / entry["output_filename"]
        for record in iter_dataset_records(annual_path):
            if record.year != year:
                raise ValueError(f"record year mismatch in {annual_path.name}")
            aggregation.add(record)
            annual_count += 1
            if progress_callback is not None and annual_count % progress_interval == 0:
                progress_callback(
                    SummaryProgress(
                        year,
                        annual_count,
                        aggregation.record_count,
                        time.monotonic() - started,
                    )
                )
        if annual_count != entry["output_records"]:
            raise ValueError(f"record count mismatch for year {year}")
        if progress_callback is not None and annual_count % progress_interval != 0:
            progress_callback(
                SummaryProgress(
                    year,
                    annual_count,
                    aggregation.record_count,
                    time.monotonic() - started,
                )
            )
    return aggregation


def _annual_summary_tasks(
    root: Path,
    entries: Iterable[Mapping[str, Any]],
    *,
    record_limit: int | None = None,
) -> tuple[_AnnualSummaryTask, ...]:
    return tuple(
        _AnnualSummaryTask(
            year=entry["year"],
            annual_path=str((root / entry["output_filename"]).resolve()),
            expected_output_records=entry["output_records"],
            record_limit=record_limit,
        )
        for entry in entries
    )


def _aggregate_annual_summary_task(task: _AnnualSummaryTask) -> _AnnualSummaryResult:
    """Process one annual gzip stream; kept top-level for Windows spawn."""
    started = time.monotonic()
    aggregation = SummaryAggregation()
    count = 0
    annual_path = Path(task.annual_path)
    for record in iter_dataset_records(annual_path):
        if record.year != task.year:
            raise ValueError(f"record year mismatch in {annual_path.name}")
        aggregation.add(record)
        count += 1
        if task.record_limit is not None and count >= task.record_limit:
            break
    if task.record_limit is None and count != task.expected_output_records:
        raise ValueError(f"record count mismatch for year {task.year}")
    aggregation.validate(expected_years=(task.year,))
    return _AnnualSummaryResult(
        year=task.year,
        aggregation=aggregation,
        records_processed=count,
        elapsed_seconds=time.monotonic() - started,
    )


def _run_annual_summary_tasks(
    tasks: tuple[_AnnualSummaryTask, ...],
    *,
    workers: int,
    result_callback: Callable[[_AnnualSummaryResult], None] | None = None,
) -> tuple[_AnnualSummaryResult, ...]:
    _validate_workers(workers)
    if not tasks:
        raise ValueError("at least one annual summary task is required")
    if workers == 1:
        results = []
        for task in tasks:
            result = _aggregate_annual_summary_task(task)
            results.append(result)
            if result_callback is not None:
                result_callback(result)
        return tuple(results)

    workers_used = min(workers, len(tasks))
    executor = ProcessPoolExecutor(max_workers=workers_used)
    futures = {}
    try:
        for task in tasks:
            futures[executor.submit(_aggregate_annual_summary_task, task)] = task.year
        results_by_year: dict[int, _AnnualSummaryResult] = {}
        for future in as_completed(futures):
            expected_year = futures[future]
            result = future.result()
            if result.year != expected_year:
                raise ValueError("annual summary worker returned the wrong year")
            if result.year in results_by_year:
                raise ValueError(
                    f"duplicate annual summary result for year {result.year}"
                )
            results_by_year[result.year] = result
            if result_callback is not None:
                result_callback(result)
    except BaseException:
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    if set(results_by_year) != {task.year for task in tasks}:
        raise ValueError("annual summary results are incomplete")
    return tuple(results_by_year.values())


def _merge_annual_summary_results(
    results: tuple[_AnnualSummaryResult, ...],
    expected_years: tuple[int, ...],
) -> SummaryAggregation:
    results_by_year: dict[int, _AnnualSummaryResult] = {}
    for result in results:
        if result.year in results_by_year:
            raise ValueError(f"duplicate annual summary result for year {result.year}")
        results_by_year[result.year] = result
    if set(results_by_year) != set(expected_years):
        raise ValueError("annual summary result years do not match the manifest")
    aggregation = SummaryAggregation()
    for year in expected_years:
        result = results_by_year[year]
        if result.records_processed != result.aggregation.record_count:
            raise ValueError(f"annual summary count disagrees for year {year}")
        aggregation.merge(result.aggregation)
    aggregation.validate(expected_years=expected_years)
    return aggregation


def collect_analysis_git_metadata(project_root: str | Path) -> AnalysisGitMetadata:
    """Collect commit identity and fail closed if Git metadata is unavailable."""
    root = Path(project_root)
    commit = subprocess.run(
        ("git", "-C", str(root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        (
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return AnalysisGitMetadata(commit=commit, worktree_clean=not bool(status))


def build_summary_documents(analysis: SummaryAnalysis) -> dict[str, str]:
    """Build all canonical result documents from the same in-memory aggregate."""
    _validate_summary_metadata(
        analysis.metadata,
        expected_record_count=analysis.aggregation.record_count,
    )
    analysis.aggregation.validate(expected_years=tuple(analysis.metadata["years"]))
    overall = {
        "metadata": dict(analysis.metadata),
        "result": _full_result_to_dict(analysis.aggregation.overall),
    }
    yearly = {
        "metadata": dict(analysis.metadata),
        "years": [
            {
                "year": year,
                **_full_result_to_dict(analysis.aggregation.years[year]),
            }
            for year in sorted(analysis.aggregation.years)
        ],
    }
    by_turn = {
        "metadata": dict(analysis.metadata),
        "all_years": [
            {
                "riichi_discard_number": turn,
                **_slice_result_to_dict(analysis.aggregation.turns[turn]),
            }
            for turn in sorted(analysis.aggregation.turns)
        ],
        "by_year": [
            {
                "year": year,
                "turns": [
                    {
                        "riichi_discard_number": turn,
                        **_slice_result_to_dict(
                            analysis.aggregation.year_turns[year][turn]
                        ),
                    }
                    for turn in sorted(analysis.aggregation.year_turns[year])
                ],
            }
            for year in sorted(analysis.aggregation.year_turns)
        ],
    }
    return {
        "overall": _json_document(overall),
        "yearly": _json_document(yearly),
        "by_turn": _json_document(by_turn),
        "markdown": _render_markdown(analysis, overall, yearly, by_turn),
    }


def write_summary_outputs(
    analysis: SummaryAnalysis,
    output_root: str | Path,
) -> tuple[Path, ...]:
    """Publish one complete content-addressed generation with an atomic pointer."""
    documents = build_summary_documents(analysis)
    document_bytes = _validated_document_bytes(documents)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    descriptor = _generation_descriptor(document_bytes)
    generation_id = _generation_id(descriptor)
    generations_root = root / PUBLICATION_GENERATIONS_DIRECTORY
    generations_root.mkdir(parents=True, exist_ok=True)
    generation_root = generations_root / generation_id
    generation_root.mkdir(parents=True, exist_ok=True)
    marker_path = generation_root / COMPLETION_MARKER_FILENAME
    marker = _completion_marker(generation_id, descriptor)
    marker_bytes = _json_document(marker).encode("utf-8")

    if marker_path.exists():
        _validate_completed_generation(generation_root, generation_id, descriptor)
    else:
        pointer_path = root / PUBLICATION_POINTER_FILENAME
        if (
            pointer_path.exists()
            and _read_publication_pointer(pointer_path) == generation_id
        ):
            raise ValueError("published generation is missing its completion marker")
        for key in ("overall", "yearly", "by_turn", "markdown"):
            _write_bytes_atomically(
                generation_root / OUTPUT_FILENAMES[key],
                document_bytes[key],
            )
        _validate_generation_files(generation_root, descriptor)
        _write_bytes_atomically(marker_path, marker_bytes)
        _validate_completed_generation(generation_root, generation_id, descriptor)

    pointer = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_name": ANALYSIS_NAME,
        "generation_id": generation_id,
        "completion_marker": (
            f"{PUBLICATION_GENERATIONS_DIRECTORY}/{generation_id}/"
            f"{COMPLETION_MARKER_FILENAME}"
        ),
    }
    _write_bytes_atomically(
        root / PUBLICATION_POINTER_FILENAME,
        _json_document(pointer).encode("utf-8"),
    )
    return load_published_summary_paths(root)


def load_published_summary_paths(output_root: str | Path) -> tuple[Path, ...]:
    """Resolve and validate the single generation named by the publication pointer."""
    root = Path(output_root)
    pointer_path = root / PUBLICATION_POINTER_FILENAME
    generation_id = _read_publication_pointer(pointer_path)
    generation_root = root / PUBLICATION_GENERATIONS_DIRECTORY / generation_id
    descriptor = _read_completion_marker(generation_root, generation_id)
    _validate_generation_files(generation_root, descriptor)
    return tuple(
        generation_root / OUTPUT_FILENAMES[key]
        for key in ("overall", "yearly", "by_turn", "markdown")
    )


def _read_completion_marker(
    generation_root: Path,
    generation_id: str,
) -> tuple[dict[str, Any], ...]:
    marker = _read_json_object(
        generation_root / COMPLETION_MARKER_FILENAME, "summary completion marker"
    )
    expected_marker_fields = {
        "schema_version",
        "analysis_name",
        "generation_id",
        "files",
    }
    if set(marker) != expected_marker_fields:
        raise ValueError("summary completion marker has unexpected fields")
    if (
        type(marker["schema_version"]) is not int
        or marker["schema_version"] != ANALYSIS_SCHEMA_VERSION
    ):
        raise ValueError("summary completion marker schema_version is invalid")
    if marker["analysis_name"] != ANALYSIS_NAME:
        raise ValueError("summary completion marker analysis_name is invalid")
    if marker["generation_id"] != generation_id:
        raise ValueError("summary completion marker generation_id is invalid")
    descriptor = _validate_generation_descriptor(marker["files"])
    if _generation_id(descriptor) != generation_id:
        raise ValueError("summary generation content does not match its identity")
    return descriptor


def _validated_document_bytes(documents: Mapping[str, str]) -> dict[str, bytes]:
    expected_keys = ("overall", "yearly", "by_turn", "markdown")
    if tuple(documents) != expected_keys:
        raise ValueError("summary document bundle has unexpected members or order")
    if any(not isinstance(documents[key], str) for key in expected_keys):
        raise TypeError("summary documents must be text")
    if any(not documents[key].endswith("\n") for key in expected_keys):
        raise ValueError("summary documents must end with LF")
    json_values = [
        json.loads(documents[key], parse_constant=_reject_json_constant)
        for key in ("overall", "yearly", "by_turn")
    ]
    if any(not isinstance(value, dict) for value in json_values):
        raise TypeError("summary JSON documents must contain objects")
    _validate_document_metadata(json_values)
    return {key: documents[key].encode("utf-8") for key in expected_keys}


def _generation_descriptor(
    document_bytes: Mapping[str, bytes],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "key": key,
            "filename": OUTPUT_FILENAMES[key],
            "size_bytes": len(document_bytes[key]),
            "sha256": hashlib.sha256(document_bytes[key]).hexdigest(),
        }
        for key in ("overall", "yearly", "by_turn", "markdown")
    )


def _generation_id(descriptor: tuple[dict[str, Any], ...]) -> str:
    payload = json.dumps(
        descriptor,
        ensure_ascii=True,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _completion_marker(
    generation_id: str,
    descriptor: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_name": ANALYSIS_NAME,
        "generation_id": generation_id,
        "files": list(descriptor),
    }


def _write_bytes_atomically(path: Path, content: bytes) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                # A leftover temp is never referenced by current.json or complete.json.
                pass


def _read_json_object(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid {context}: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{context} must contain a JSON object")
    return value


def _read_publication_pointer(path: Path) -> str:
    pointer = _read_json_object(path, "summary publication pointer")
    expected_fields = {
        "schema_version",
        "analysis_name",
        "generation_id",
        "completion_marker",
    }
    if set(pointer) != expected_fields:
        raise ValueError("summary publication pointer has unexpected fields")
    generation_id = pointer["generation_id"]
    if (
        type(pointer["schema_version"]) is not int
        or pointer["schema_version"] != ANALYSIS_SCHEMA_VERSION
        or pointer["analysis_name"] != ANALYSIS_NAME
        or not isinstance(generation_id, str)
        or _SHA256_PATTERN.fullmatch(generation_id) is None
        or pointer["completion_marker"]
        != (
            f"{PUBLICATION_GENERATIONS_DIRECTORY}/{generation_id}/"
            f"{COMPLETION_MARKER_FILENAME}"
        )
    ):
        raise ValueError("summary publication pointer is invalid")
    return generation_id


def _validate_generation_descriptor(value: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or len(value) != len(OUTPUT_FILENAMES):
        raise TypeError("summary generation files must be a four-item array")
    expected_keys = ("overall", "yearly", "by_turn", "markdown")
    descriptor: list[dict[str, Any]] = []
    for expected_key, item in zip(expected_keys, value, strict=True):
        if not isinstance(item, dict):
            raise TypeError("summary generation file entry must be an object")
        if set(item) != {"key", "filename", "size_bytes", "sha256"}:
            raise ValueError("summary generation file entry has unexpected fields")
        if (
            item["key"] != expected_key
            or item["filename"] != OUTPUT_FILENAMES[expected_key]
        ):
            raise ValueError(
                "summary generation file entry order or filename is invalid"
            )
        if type(item["size_bytes"]) is not int or item["size_bytes"] < 0:
            raise ValueError("summary generation file size must be non-negative")
        if (
            not isinstance(item["sha256"], str)
            or _SHA256_PATTERN.fullmatch(item["sha256"]) is None
        ):
            raise ValueError("summary generation file SHA256 is invalid")
        descriptor.append(dict(item))
    return tuple(descriptor)


def _validate_generation_files(
    generation_root: Path,
    descriptor: tuple[dict[str, Any], ...],
) -> None:
    for item in descriptor:
        path = generation_root / item["filename"]
        try:
            content = path.read_bytes()
        except OSError as error:
            raise ValueError(
                f"summary generation file is unavailable: {path}"
            ) from error
        if len(content) != item["size_bytes"]:
            raise ValueError(f"summary generation file size mismatch: {path.name}")
        if hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError(f"summary generation file SHA256 mismatch: {path.name}")
    json_values = []
    for key in ("overall", "yearly", "by_turn"):
        value = _read_json_object(
            generation_root / OUTPUT_FILENAMES[key], "summary JSON"
        )
        json_values.append(value)
    _validate_document_metadata(json_values)


def _validate_completed_generation(
    generation_root: Path,
    generation_id: str,
    expected_descriptor: tuple[dict[str, Any], ...],
) -> None:
    actual_descriptor = _read_completion_marker(generation_root, generation_id)
    if actual_descriptor != expected_descriptor:
        raise ValueError(
            "existing summary generation does not match the requested result"
        )
    _validate_generation_files(generation_root, expected_descriptor)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _validate_document_metadata(json_values: list[dict[str, Any]]) -> None:
    metadata_values = [value.get("metadata") for value in json_values]
    for metadata in metadata_values:
        _validate_summary_metadata(metadata)
    if any(metadata != metadata_values[0] for metadata in metadata_values[1:]):
        raise ValueError("summary JSON documents have different metadata")


def _validate_summary_metadata(
    value: object,
    *,
    expected_record_count: int | None = None,
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("summary metadata must be a mapping")
    _require_metadata_fields(
        value,
        (
            "schema_version",
            "analysis_name",
            "observation_unit",
            "years",
            "scope",
            "input_dataset",
            "analysis_generator",
            "formal_wait_semantics",
            "ryanmen_metrics",
        ),
        "summary metadata",
    )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != ANALYSIS_SCHEMA_VERSION
    ):
        raise ValueError("summary metadata schema_version is invalid")
    if value["analysis_name"] != ANALYSIS_NAME:
        raise ValueError("summary metadata analysis_name is invalid")
    if value["observation_unit"] != "established_riichi_record":
        raise ValueError("summary metadata observation_unit is invalid")

    years = value["years"]
    if not isinstance(years, list) or not years:
        raise TypeError("summary metadata years must be a non-empty array")
    if any(type(year) is not int for year in years):
        raise TypeError("summary metadata years must contain integers")
    if tuple(years) != tuple(sorted(set(years))) or any(
        year not in SUPPORTED_YEARS for year in years
    ):
        raise ValueError("summary metadata years are invalid")

    scope = _metadata_mapping(value["scope"], "summary metadata scope")
    _require_metadata_fields(
        scope,
        ("rule_code", "aka_flag", "bakaze"),
        "summary metadata scope",
    )
    if scope["rule_code"] != RULE_CODE:
        raise ValueError("summary metadata rule_code is invalid")
    if scope["aka_flag"] is not True:
        raise ValueError("summary metadata aka_flag is invalid")
    if scope["bakaze"] != "E":
        raise ValueError("summary metadata bakaze is invalid")

    dataset = _metadata_mapping(value["input_dataset"], "summary input dataset")
    _require_metadata_fields(
        dataset,
        (
            "dataset_name",
            "schema_version",
            "manifest_path",
            "manifest_sha256",
            "generator_git_commit",
            "source_repository",
            "source_release_tag",
            "totals",
        ),
        "summary input dataset",
    )
    if dataset["dataset_name"] != DATASET_NAME:
        raise ValueError("summary input dataset_name is invalid")
    if (
        type(dataset["schema_version"]) is not int
        or dataset["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("summary input schema_version is invalid")
    manifest_path = dataset["manifest_path"]
    if (
        not isinstance(manifest_path, str)
        or not manifest_path
        or "\\" in manifest_path
        or Path(manifest_path).is_absolute()
        or ".." in manifest_path.split("/")
    ):
        raise ValueError("summary input manifest_path is invalid")
    _validate_sha256_text(dataset["manifest_sha256"], "summary manifest SHA256")
    _validate_commit_text(dataset["generator_git_commit"], "dataset generator commit")
    if dataset["source_repository"] != CANONICAL_SOURCE_REPOSITORY:
        raise ValueError("summary input source repository is invalid")
    if dataset["source_release_tag"] != SOURCE_RELEASE_TAG:
        raise ValueError("summary input source release is invalid")
    totals = _metadata_mapping(dataset["totals"], "summary input totals")
    _require_metadata_fields(totals, ("output_records",), "summary input totals")
    record_count = totals["output_records"]
    if type(record_count) is not int or record_count < 0:
        raise ValueError("summary input output_records is invalid")
    if expected_record_count is not None and record_count != expected_record_count:
        raise ValueError("metadata input record total disagrees with aggregation")
    if (
        "established_riichis" in totals
        and totals["established_riichis"] != record_count
    ):
        raise ValueError("summary input riichi total disagrees with output records")

    generator = _metadata_mapping(
        value["analysis_generator"], "summary analysis generator"
    )
    _require_metadata_fields(
        generator,
        ("git_commit", "worktree_clean"),
        "summary analysis generator",
    )
    _validate_commit_text(generator["git_commit"], "analysis generator commit")
    if generator["worktree_clean"] is not True:
        raise ValueError("summary metadata requires a clean analysis generator")

    semantics = _metadata_mapping(
        value["formal_wait_semantics"], "summary formal wait semantics"
    )
    _require_metadata_fields(
        semantics,
        (
            "name",
            "fixed_meld_self_owned_four_waits_retained",
            "concealed_four_candidate_excluded",
        ),
        "summary formal wait semantics",
    )
    if (
        semantics["name"] != "tenhou_formal_wait"
        or semantics["fixed_meld_self_owned_four_waits_retained"] is not True
        or semantics["concealed_four_candidate_excluded"] is not True
    ):
        raise ValueError("summary formal wait semantics are invalid")

    ryanmen = _metadata_mapping(value["ryanmen_metrics"], "summary ryanmen metrics")
    _require_metadata_fields(
        ryanmen,
        ("headline", "headline_definition", "supplementary"),
        "summary ryanmen metrics",
    )
    definition = _metadata_mapping(
        ryanmen["headline_definition"], "summary ryanmen headline definition"
    )
    _require_metadata_fields(
        definition,
        (
            "wait_tile_count",
            "all_hand_types",
            "all_wait_shapes",
            "empty_details_is_pure_ryanmen",
        ),
        "summary ryanmen headline definition",
    )
    if (
        ryanmen["headline"] != "is_pure_ryanmen"
        or ryanmen["supplementary"] != "contains_ryanmen"
        or type(definition["wait_tile_count"]) is not int
        or definition["wait_tile_count"] != 2
        or definition["all_hand_types"] != "standard"
        or definition["all_wait_shapes"] != "ryanmen"
        or definition["empty_details_is_pure_ryanmen"] is not False
    ):
        raise ValueError("summary ryanmen metric definitions are invalid")


def _metadata_mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def _require_metadata_fields(
    value: Mapping[str, Any],
    fields: tuple[str, ...],
    context: str,
) -> None:
    missing = [field for field in fields if field not in value]
    if missing:
        raise ValueError(f"{context} is missing required fields: {', '.join(missing)}")


def _validate_sha256_text(value: object, context: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{context} must be 64 lowercase hexadecimal characters")


def _validate_commit_text(value: object, context: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            f"{context} must be non-empty text without surrounding whitespace"
        )


def _merge_record_level_accumulator(
    target: _RecordLevelAccumulator,
    source: _RecordLevelAccumulator,
) -> None:
    target.record_count += source.record_count
    target.wait_tile_counts.update(source.wait_tile_counts)
    target.pure_ryanmen_count += source.pure_ryanmen_count
    target.contains_ryanmen_count += source.contains_ryanmen_count
    target.multiwait_count += source.multiwait_count
    target.shape_membership.update(source.shape_membership)
    target.shape_sets.update(source.shape_sets)
    target.hand_type_membership.update(source.hand_type_membership)
    target.hand_type_sets.update(source.hand_type_sets)
    target.multiple_shape_count += source.multiple_shape_count
    target.multiple_hand_type_count += source.multiple_hand_type_count


def _merge_slice_accumulator(
    target: _SliceAccumulator,
    source: _SliceAccumulator,
) -> None:
    _merge_record_level_accumulator(target.formal, source.formal)
    _merge_record_level_accumulator(target.adjusted, source.adjusted)
    target.adjusted_zero_wait_count += source.adjusted_zero_wait_count


def _merge_wait_tile_accumulator(
    target: _WaitTileAccumulator,
    source: _WaitTileAccumulator,
) -> None:
    target.observation_count += source.observation_count
    target.shape_membership.update(source.shape_membership)
    target.shape_sets.update(source.shape_sets)
    target.multiple_shape_count += source.multiple_shape_count


def _merge_diagnostics_accumulator(
    target: _DiagnosticsAccumulator,
    source: _DiagnosticsAccumulator,
) -> None:
    _merge_wait_tile_accumulator(target.wait_tiles, source.wait_tiles)
    for tile in TILE_KINDS:
        _merge_wait_tile_accumulator(target.by_tile[tile], source.by_tile[tile])
    target.detail_count += source.detail_count
    target.detail_cross_table.update(source.detail_cross_table)


def _merge_full_accumulator(
    target: _FullAccumulator,
    source: _FullAccumulator,
) -> None:
    _merge_slice_accumulator(target.record_level, source.record_level)
    target.riichi_discard_numbers.update(source.riichi_discard_numbers)
    _merge_diagnostics_accumulator(target.diagnostics, source.diagnostics)
    target.fifth_tile_record_count += source.fifth_tile_record_count
    target.fifth_tile_observation_count += source.fifth_tile_observation_count
    target.fifth_tile_only_record_count += source.fifth_tile_only_record_count
    target.fifth_tile_with_other_record_count += (
        source.fifth_tile_with_other_record_count
    )


def _validate_record_level_partition(
    total: _RecordLevelAccumulator,
    parts: tuple[_RecordLevelAccumulator, ...],
    context: str,
) -> None:
    combined = _RecordLevelAccumulator()
    for part in parts:
        combined.record_count += part.record_count
        combined.wait_tile_counts.update(part.wait_tile_counts)
        combined.pure_ryanmen_count += part.pure_ryanmen_count
        combined.contains_ryanmen_count += part.contains_ryanmen_count
        combined.multiwait_count += part.multiwait_count
        combined.shape_membership.update(part.shape_membership)
        combined.shape_sets.update(part.shape_sets)
        combined.hand_type_membership.update(part.hand_type_membership)
        combined.hand_type_sets.update(part.hand_type_sets)
        combined.multiple_shape_count += part.multiple_shape_count
        combined.multiple_hand_type_count += part.multiple_hand_type_count
    if combined != total:
        raise ValueError(f"{context} record-level aggregates do not sum to total")


def _validate_slice_partition(
    total: _SliceAccumulator,
    parts: tuple[_SliceAccumulator, ...],
    context: str,
) -> None:
    _validate_record_level_partition(
        total.formal,
        tuple(part.formal for part in parts),
        f"{context} formal",
    )
    _validate_record_level_partition(
        total.adjusted,
        tuple(part.adjusted for part in parts),
        f"{context} adjusted",
    )
    if sum(part.adjusted_zero_wait_count for part in parts) != (
        total.adjusted_zero_wait_count
    ):
        raise ValueError(f"{context} adjusted zero-wait counts do not sum to total")


def _validate_wait_tile_partition(
    total: _WaitTileAccumulator,
    parts: tuple[_WaitTileAccumulator, ...],
    context: str,
) -> None:
    combined = _WaitTileAccumulator()
    for part in parts:
        combined.observation_count += part.observation_count
        combined.shape_membership.update(part.shape_membership)
        combined.shape_sets.update(part.shape_sets)
        combined.multiple_shape_count += part.multiple_shape_count
    if combined != total:
        raise ValueError(f"{context} wait-tile aggregates do not sum to total")


def _validate_diagnostics_partition(
    total: _DiagnosticsAccumulator,
    parts: tuple[_DiagnosticsAccumulator, ...],
    context: str,
) -> None:
    _validate_wait_tile_partition(
        total.wait_tiles,
        tuple(part.wait_tiles for part in parts),
        context,
    )
    for tile in TILE_KINDS:
        _validate_wait_tile_partition(
            total.by_tile[tile],
            tuple(part.by_tile[tile] for part in parts),
            f"{context} {tile}",
        )
    if sum(part.detail_count for part in parts) != total.detail_count:
        raise ValueError(f"{context} detail counts do not sum to total")
    combined_cross_table: Counter[tuple[str, str]] = Counter()
    for part in parts:
        combined_cross_table.update(part.detail_cross_table)
    if combined_cross_table != total.detail_cross_table:
        raise ValueError(f"{context} detail aggregates do not sum to total")


def _validate_full_partition(
    total: _FullAccumulator,
    parts: tuple[_FullAccumulator, ...],
    context: str,
) -> None:
    _validate_slice_partition(
        total.record_level,
        tuple(part.record_level for part in parts),
        context,
    )
    combined_discards: Counter[int] = Counter()
    for part in parts:
        combined_discards.update(part.riichi_discard_numbers)
    if combined_discards != total.riichi_discard_numbers:
        raise ValueError(f"{context} discard distributions do not sum to total")
    _validate_diagnostics_partition(
        total.diagnostics,
        tuple(part.diagnostics for part in parts),
        context,
    )
    for name in (
        "fifth_tile_record_count",
        "fifth_tile_observation_count",
        "fifth_tile_only_record_count",
        "fifth_tile_with_other_record_count",
    ):
        if sum(getattr(part, name) for part in parts) != getattr(total, name):
            raise ValueError(f"{context} {name} does not sum to total")


def _build_contribution(record: RiichiWaitDatasetRecord) -> _RecordContribution:
    formal = _facts_from_details(record.wait_details)
    if not formal.wait_tiles:
        raise ValueError(
            "an established riichi record must have at least one formal wait"
        )
    if formal.wait_tiles != record.wait_tiles:
        raise ValueError("formal wait tiles are not derived from wait details")
    if formal.shapes != record.wait_shapes:
        raise ValueError("formal wait shapes are not derived from wait details")
    if (
        formal.is_multiwait != record.is_multiwait
        or formal.contains_ryanmen != record.contains_ryanmen
        or formal.is_pure_ryanmen != record.is_pure_ryanmen
    ):
        raise ValueError("formal wait booleans are not derived from wait details")

    concealed_counts = Counter(
        normalize_tile(tile) for tile in record.concealed_tiles_after_discard
    )
    owned_counts = concealed_counts.copy()
    for meld in record.fixed_melds:
        owned_counts.update(normalize_tile(tile) for tile in meld.tiles)
    fifth_tile_waits = tuple(
        tile
        for tile in record.wait_tiles
        if owned_counts[tile] == 4 and concealed_counts[tile] < 4
    )
    removed = frozenset(fifth_tile_waits)
    adjusted_details = tuple(
        detail for detail in record.wait_details if detail.tile not in removed
    )
    adjusted = _facts_from_details(adjusted_details)
    return _RecordContribution(
        formal=formal,
        adjusted=adjusted,
        fifth_tile_waits=fifth_tile_waits,
        riichi_discard_number=record.riichi_discard_number,
    )


def _facts_from_details(details: tuple[DatasetWaitDetail, ...]) -> _WaitFacts:
    wait_tiles = _ordered_members((detail.tile for detail in details), TILE_KINDS)
    shapes = _ordered_members(
        (detail.wait_shape for detail in details),
        WAIT_SHAPE_ORDER,
    )
    hand_types = _ordered_members(
        (detail.hand_type for detail in details),
        HAND_TYPE_ORDER,
    )
    return _WaitFacts(
        wait_tiles=wait_tiles,
        details=details,
        shapes=shapes,
        hand_types=hand_types,
        is_multiwait=len(wait_tiles) >= 3,
        contains_ryanmen="ryanmen" in shapes,
        is_pure_ryanmen=(
            bool(details)
            and len(wait_tiles) == 2
            and all(
                detail.hand_type == "standard" and detail.wait_shape == "ryanmen"
                for detail in details
            )
        ),
    )


def _ordered_members(values: Iterable[str], order: tuple[str, ...]) -> tuple[str, ...]:
    members = frozenset(values)
    unknown = members.difference(order)
    if unknown:
        raise ValueError(f"values outside canonical order: {sorted(unknown)}")
    return tuple(value for value in order if value in members)


def _metric(count: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "count": count,
        "denominator": denominator,
        "rate": count / denominator if denominator else None,
    }


def _share(count: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "count": count,
        "denominator": denominator,
        "share": count / denominator if denominator else None,
    }


def _integer_distribution(
    values: Mapping[int, int],
    denominator: int,
) -> list[dict[str, int | float | None]]:
    return [
        {"value": value, **_metric(values[value], denominator)}
        for value in sorted(values)
    ]


def _membership(
    values: Mapping[str, int],
    order: tuple[str, ...],
    denominator: int,
) -> list[dict[str, int | float | None | str]]:
    return [
        {"value": value, **_metric(values.get(value, 0), denominator)}
        for value in order
    ]


def _set_distribution(
    values: Mapping[tuple[str, ...], int],
    order: tuple[str, ...],
    denominator: int,
) -> list[dict[str, Any]]:
    indexes = {value: index for index, value in enumerate(order)}
    categories = sorted(
        values,
        key=lambda members: tuple(indexes[value] for value in members),
    )
    return [
        {"members": list(members), **_metric(values[members], denominator)}
        for members in categories
    ]


def _record_level_to_dict(value: _RecordLevelAccumulator) -> dict[str, Any]:
    denominator = value.record_count
    return {
        "record_count": denominator,
        "pure_ryanmen": _metric(value.pure_ryanmen_count, denominator),
        "contains_ryanmen": _metric(value.contains_ryanmen_count, denominator),
        "multiwait": _metric(value.multiwait_count, denominator),
        "wait_tile_count_distribution": _integer_distribution(
            value.wait_tile_counts,
            denominator,
        ),
        "wait_shape_record_level": {
            "wait_shape_membership": _membership(
                value.shape_membership,
                WAIT_SHAPE_ORDER,
                denominator,
            ),
            "wait_shape_set_distribution": _set_distribution(
                value.shape_sets,
                WAIT_SHAPE_ORDER,
                denominator,
            ),
            "records_with_multiple_wait_shapes": _metric(
                value.multiple_shape_count,
                denominator,
            ),
        },
        "hand_type_record_level": {
            "hand_type_membership": _membership(
                value.hand_type_membership,
                HAND_TYPE_ORDER,
                denominator,
            ),
            "hand_type_set_distribution": _set_distribution(
                value.hand_type_sets,
                HAND_TYPE_ORDER,
                denominator,
            ),
            "records_with_multiple_hand_types": _metric(
                value.multiple_hand_type_count,
                denominator,
            ),
        },
    }


def _adjusted_record_level_to_dict(value: _SliceAccumulator) -> dict[str, Any]:
    raw = _record_level_to_dict(value.adjusted)
    raw_shapes = raw["wait_shape_record_level"]
    raw_hand_types = raw["hand_type_record_level"]
    return {
        "record_count": raw["record_count"],
        "adjusted_zero_wait_count": value.adjusted_zero_wait_count,
        "adjusted_is_pure_ryanmen": raw["pure_ryanmen"],
        "adjusted_contains_ryanmen": raw["contains_ryanmen"],
        "adjusted_is_multiwait": raw["multiwait"],
        "adjusted_wait_tile_count_distribution": raw["wait_tile_count_distribution"],
        "adjusted_wait_shape_record_level": {
            "adjusted_wait_shape_membership": raw_shapes["wait_shape_membership"],
            "adjusted_wait_shape_set_distribution": raw_shapes[
                "wait_shape_set_distribution"
            ],
            "adjusted_records_with_multiple_wait_shapes": raw_shapes[
                "records_with_multiple_wait_shapes"
            ],
        },
        "adjusted_hand_type_record_level": {
            "adjusted_hand_type_membership": raw_hand_types["hand_type_membership"],
            "adjusted_hand_type_set_distribution": raw_hand_types[
                "hand_type_set_distribution"
            ],
            "adjusted_records_with_multiple_hand_types": raw_hand_types[
                "records_with_multiple_hand_types"
            ],
        },
    }


def _wait_tile_level_to_dict(value: _DiagnosticsAccumulator) -> dict[str, Any]:
    total = value.wait_tiles.observation_count
    per_tile = []
    for tile in TILE_KINDS:
        tile_value = value.by_tile[tile]
        per_tile.append(
            {
                "tile": tile,
                "tile_frequency": _share(tile_value.observation_count, total),
                "shape_membership": _membership(
                    tile_value.shape_membership,
                    WAIT_SHAPE_ORDER,
                    tile_value.observation_count,
                ),
                "shape_set_distribution": _set_distribution(
                    tile_value.shape_sets,
                    WAIT_SHAPE_ORDER,
                    tile_value.observation_count,
                ),
                "wait_tiles_with_multiple_shapes": _metric(
                    tile_value.multiple_shape_count,
                    tile_value.observation_count,
                ),
            }
        )
    return {
        "wait_tile_observation_count": total,
        "shape_membership": _membership(
            value.wait_tiles.shape_membership,
            WAIT_SHAPE_ORDER,
            total,
        ),
        "shape_set_distribution": _set_distribution(
            value.wait_tiles.shape_sets,
            WAIT_SHAPE_ORDER,
            total,
        ),
        "wait_tiles_with_multiple_shapes": _metric(
            value.wait_tiles.multiple_shape_count,
            total,
        ),
        "by_tile": per_tile,
    }


def _detail_level_to_dict(value: _DiagnosticsAccumulator) -> dict[str, Any]:
    denominator = value.detail_count
    cross_table = []
    for hand_type in HAND_TYPE_ORDER:
        for shape in WAIT_SHAPE_ORDER:
            count = value.detail_cross_table.get((hand_type, shape), 0)
            cross_table.append(
                {
                    "hand_type": hand_type,
                    "wait_shape": shape,
                    "count": count,
                    "detail_denominator": denominator,
                    "share_of_details": count / denominator if denominator else None,
                }
            )
    return {"wait_detail_count": denominator, "hand_type_by_wait_shape": cross_table}


def _slice_result_to_dict(value: _SliceAccumulator) -> dict[str, Any]:
    return {
        "established_riichi_count": value.formal.record_count,
        "formal_baseline": _record_level_to_dict(value.formal),
        "adjusted_record_level": _adjusted_record_level_to_dict(value),
    }


def _full_result_to_dict(value: _FullAccumulator) -> dict[str, Any]:
    denominator = value.record_level.formal.record_count
    formal = _record_level_to_dict(value.record_level.formal)
    adjusted = _adjusted_record_level_to_dict(value.record_level)
    formal_adjusted_deltas = {}
    for name, formal_key, adjusted_key in (
        ("is_pure_ryanmen", "pure_ryanmen", "adjusted_is_pure_ryanmen"),
        ("contains_ryanmen", "contains_ryanmen", "adjusted_contains_ryanmen"),
        ("is_multiwait", "multiwait", "adjusted_is_multiwait"),
    ):
        formal_metric = formal[formal_key]
        adjusted_metric = adjusted[adjusted_key]
        formal_rate = formal_metric["rate"]
        adjusted_rate = adjusted_metric["rate"]
        formal_adjusted_deltas[name] = {
            "count_delta": adjusted_metric["count"] - formal_metric["count"],
            "percentage_point_delta": (
                (adjusted_rate - formal_rate) * 100
                if adjusted_rate is not None and formal_rate is not None
                else None
            ),
        }
    return {
        "established_riichi_count": denominator,
        "formal_baseline": formal,
        "riichi_discard_number_distribution": _integer_distribution(
            value.riichi_discard_numbers,
            denominator,
        ),
        "wait_tile_level": _wait_tile_level_to_dict(value.diagnostics),
        "wait_detail_level": _detail_level_to_dict(value.diagnostics),
        "fifth_tile_sensitivity": {
            "records_with_self_owned_four_fifth_tile_wait": _metric(
                value.fifth_tile_record_count,
                denominator,
            ),
            "self_owned_four_fifth_tile_wait_observations": _metric(
                value.fifth_tile_observation_count,
                value.diagnostics.wait_tiles.observation_count,
            ),
            "records_with_only_fifth_tile_waits": _metric(
                value.fifth_tile_only_record_count,
                denominator,
            ),
            "records_with_fifth_tile_and_other_waits": _metric(
                value.fifth_tile_with_other_record_count,
                denominator,
            ),
            "adjusted_record_level": adjusted,
            "formal_adjusted_deltas": formal_adjusted_deltas,
        },
    }


def _validate_canonical_manifest(
    manifest: Mapping[str, Any],
    years: tuple[int, ...],
) -> None:
    if years != SUPPORTED_YEARS:
        raise ValueError("canonical analysis requires all years 2009 through 2025")
    if manifest["source"]["repository"] != CANONICAL_SOURCE_REPOSITORY:
        raise ValueError("canonical manifest has an unexpected source repository")
    totals = manifest["totals"]
    if totals["output_records"] != CANONICAL_RECORD_COUNT:
        raise ValueError("canonical manifest has an unexpected record total")
    if totals["established_riichis"] != CANONICAL_RECORD_COUNT:
        raise ValueError("canonical manifest has an unexpected riichi total")


def _build_metadata(
    manifest: Mapping[str, Any],
    *,
    manifest_logical_path: str,
    manifest_sha256: str,
    analysis_git: AnalysisGitMetadata,
) -> dict[str, Any]:
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_name": ANALYSIS_NAME,
        "observation_unit": "established_riichi_record",
        "years": list(manifest["scope"]["years"]),
        "scope": {
            "rule_code": manifest["scope"]["rule_code"],
            "aka_flag": manifest["scope"]["aka_flag"],
            "bakaze": manifest["scope"]["bakaze"],
        },
        "input_dataset": {
            "dataset_name": manifest["dataset_name"],
            "schema_version": manifest["schema_version"],
            "manifest_path": manifest_logical_path,
            "manifest_sha256": manifest_sha256,
            "generator_git_commit": manifest["generator"]["git_commit"],
            "source_repository": manifest["source"]["repository"],
            "source_release_tag": manifest["source"]["release_tag"],
            "totals": dict(manifest["totals"]),
        },
        "analysis_generator": {
            "git_commit": analysis_git.commit,
            "worktree_clean": analysis_git.worktree_clean,
        },
        "formal_wait_semantics": {
            "name": "tenhou_formal_wait",
            "fixed_meld_self_owned_four_waits_retained": True,
            "concealed_four_candidate_excluded": True,
        },
        "ryanmen_metrics": {
            "headline": "is_pure_ryanmen",
            "headline_definition": {
                "wait_tile_count": 2,
                "all_hand_types": "standard",
                "all_wait_shapes": "ryanmen",
                "empty_details_is_pure_ryanmen": False,
            },
            "supplementary": "contains_ryanmen",
        },
    }


def _json_document(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            indent=2,
        )
        + "\n"
    )


def _format_rate(metric: Mapping[str, Any]) -> str:
    rate = metric["rate"]
    return "N/A" if rate is None else f"{rate * 100:.2f}%"


def _render_markdown(
    analysis: SummaryAnalysis,
    overall_document: Mapping[str, Any],
    yearly_document: Mapping[str, Any],
    by_turn_document: Mapping[str, Any],
) -> str:
    result = overall_document["result"]
    formal = result["formal_baseline"]
    sensitivity = result["fifth_tile_sensitivity"]
    adjusted = sensitivity["adjusted_record_level"]
    metadata = analysis.metadata
    lines = [
        "# Established-riichi wait baseline",
        "",
        f"- Input: `{metadata['input_dataset']['manifest_path']}`",
        f"- Records: {result['established_riichi_count']:,}",
        "- Observation unit: one established-riichi record",
        "- Headline ryanmen metric: `is_pure_ryanmen`",
        "",
        "## Overall",
        "",
        "| Metric | Formal | Self-ownership adjusted |",
        "|---|---:|---:|",
        (
            f"| Pure ryanmen | {_format_rate(formal['pure_ryanmen'])} | "
            f"{_format_rate(adjusted['adjusted_is_pure_ryanmen'])} |"
        ),
        (
            f"| Contains ryanmen | {_format_rate(formal['contains_ryanmen'])} | "
            f"{_format_rate(adjusted['adjusted_contains_ryanmen'])} |"
        ),
        (
            f"| Multiwait | {_format_rate(formal['multiwait'])} | "
            f"{_format_rate(adjusted['adjusted_is_multiwait'])} |"
        ),
        "",
        (
            "Wait-shape and hand-type memberships are non-exclusive; their rates need "
            "not sum to 100%."
        ),
        "",
        "## Yearly headline",
        "",
        "| Year | Records | Pure ryanmen | Contains ryanmen | Multiwait |",
        "|---:|---:|---:|---:|---:|",
    ]
    for year_result in yearly_document["years"]:
        year_formal = year_result["formal_baseline"]
        lines.append(
            f"| {year_result['year']} | "
            f"{year_result['established_riichi_count']:,} | "
            f"{_format_rate(year_formal['pure_ryanmen'])} | "
            f"{_format_rate(year_formal['contains_ryanmen'])} | "
            f"{_format_rate(year_formal['multiwait'])} |"
        )
    lines.extend(
        [
            "",
            "## Exact riichi discard number",
            "",
            "| Discard number | Records | Pure ryanmen | Contains ryanmen | Multiwait |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for turn in by_turn_document["all_years"]:
        turn_formal = turn["formal_baseline"]
        lines.append(
            f"| {turn['riichi_discard_number']} | "
            f"{turn['established_riichi_count']:,} | "
            f"{_format_rate(turn_formal['pure_ryanmen'])} | "
            f"{_format_rate(turn_formal['contains_ryanmen'])} | "
            f"{_format_rate(turn_formal['multiwait'])} |"
        )
    fifth_metric = sensitivity["records_with_self_owned_four_fifth_tile_wait"]
    lines.extend(
        [
            "",
            "## Fifth-tile sensitivity",
            "",
            (
                f"- Affected records: {fifth_metric['count']:,} "
                f"({_format_rate(fifth_metric)})"
            ),
            f"- Adjusted zero-wait records: {adjusted['adjusted_zero_wait_count']:,}",
            (
                "- The adjusted series removes only self-owned-four formal wait tiles; "
                "it is not a live-wait estimate."
            ),
            "",
            "## Validation",
            "",
            f"- Manifest SHA256: `{metadata['input_dataset']['manifest_sha256']}`",
            f"- Analysis commit: `{metadata['analysis_generator']['git_commit']}`",
            (
                "- Annual file integrity, DTO validation, annual counts, total counts, "
                "and aggregation invariants passed before these documents were built."
            ),
            "",
        ]
    )
    return "\n".join(lines)
