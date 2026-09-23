"""Run the first-riichi child-responder defense analysis."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter, deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from itertools import groupby, islice
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from mahjong_analysis.dealer_child_riichi_points import extract_established_riichis
from mahjong_analysis.dealer_riichi_defense import (
    DECIMAL_MAX_EXPONENT,
    DECIMAL_MIN_EXPONENT,
    DECIMAL_PRECISION_DIGITS,
    DECISION_METRICS,
    RESPONDER_METRICS,
    DefenseAnalysisBuilder,
    FocalObservation,
    FocalRole,
    analyze_focal_kyoku,
    canonical_json,
    publish_output_bundle,
)
from mahjong_analysis.mjai import is_target_game, split_kyoku
from mahjong_analysis.riichi_wait_dataset import (
    RiichiWaitDatasetRecord,
    iter_dataset_records,
    iter_validated_year_records,
    load_manifest,
)

SUPPORTED_YEARS = tuple(range(2009, 2026))
PRIMARY_YEARS = tuple(range(2020, 2026))
EXPECTED_REPOSITORY = "NikkeTryHard/tenhou-to-mjai"
EXPECTED_RELEASE = "v2.0.0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs/dealer-riichi-defense"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_DIRECTORY / "summary-v3.json"
DEFAULT_OUTPUT_MARKDOWN = DEFAULT_OUTPUT_DIRECTORY / "summary-v3.md"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data/processed/riichi-waits-v1"
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data/raw"
DEFAULT_POINT_SUMMARY = (
    PROJECT_ROOT / "outputs/dealer-child-riichi-points/summary-v1.json"
)
DEFAULT_WORKERS = 1
DEFAULT_BATCH_SIZE = 32
IN_FLIGHT_BATCHES_PER_WORKER = 2


@dataclass
class ReconciliationCounts:
    """Article-1-compatible integer outcome counts for all established riichis."""

    values: Counter[tuple[str, FocalRole, str]] = field(default_factory=Counter)

    def add(
        self,
        role: FocalRole,
        reach_type: str,
        outcome: str,
        method: str | None,
    ) -> None:
        sensitivities = ["all_riichi"]
        if reach_type == "riichi":
            sensitivities.append("normal_riichi")
        for sensitivity in sensitivities:
            self.values[(sensitivity, role, "riichis")] += 1
            if outcome == "win":
                self.values[(sensitivity, role, "wins")] += 1
                self.values[(sensitivity, role, f"{method}_wins")] += 1
            else:
                self.values[(sensitivity, role, outcome)] += 1

    def group(self, sensitivity: str, role: FocalRole) -> dict[str, int]:
        return {
            key: self.values[(sensitivity, role, key)]
            for key in (
                "riichis",
                "wins",
                "other_wins",
                "draws",
                "tsumo_wins",
                "ron_wins",
            )
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            sensitivity: {
                role: self.group(sensitivity, role) for role in ("dealer", "nondealer")
            }
            for sensitivity in ("all_riichi", "normal_riichi")
        }

    def merge(self, other: ReconciliationCounts) -> None:
        self.values.update(other.values)

    def frozen_items(self) -> tuple[tuple[tuple[str, FocalRole, str], int], ...]:
        """Return a deterministic process-safe representation of nonzero counts."""
        return tuple(sorted(self.values.items()))

    def merge_items(
        self,
        items: Sequence[tuple[tuple[str, FocalRole, str], int]],
    ) -> None:
        """Merge a worker's integer reconciliation delta."""
        self.values.update(dict(items))


@dataclass
class RunCounts:
    source_files: int = 0
    wait_records: int = 0
    riichi_kyokus: int = 0
    focal_riichis: int = 0
    raw_bytes: int = 0
    raw_events: int = 0


@dataclass(frozen=True)
class SourceTask:
    """One canonical raw source and its already-validated wait records."""

    year: int
    source_path: str
    raw_path: Path
    records: tuple[RiichiWaitDatasetRecord, ...]


@dataclass(frozen=True)
class SourceResult:
    """Pure worker output applied by the parent in canonical source order."""

    year: int
    source_path: str
    raw_sha256: str
    raw_bytes: int
    raw_events: int
    wait_records: int
    observations: tuple[FocalObservation, ...]
    reconciliation_items: tuple[tuple[tuple[str, FocalRole, str], int], ...]

    @property
    def focal_riichis(self) -> int:
        return len(self.observations)


def validate_manifest_identity(value: Mapping[str, Any]) -> None:
    """Fail closed unless the wait dataset is the exact research input."""
    if value.get("dataset_name") != "riichi-waits-v1":
        raise ValueError("unexpected wait dataset name")
    if value.get("schema_version") != 1:
        raise ValueError("unexpected wait dataset schema version")
    source = _mapping(value.get("source"), "wait manifest source")
    if source.get("repository") != EXPECTED_REPOSITORY:
        raise ValueError("unexpected wait dataset repository")
    if source.get("release_tag") != EXPECTED_RELEASE:
        raise ValueError("unexpected wait dataset release")
    scope = _mapping(value.get("scope"), "wait manifest scope")
    expected_scope = {
        "rule_code": "00a9",
        "aka_flag": True,
        "bakaze": "E",
        "extraction_mode": "full",
    }
    for key, expected in expected_scope.items():
        if scope.get(key) != expected or type(scope.get(key)) is not type(expected):
            raise ValueError(f"unexpected wait dataset scope {key}")
    if _strict_years(scope.get("years"), "wait manifest scope years") != (
        SUPPORTED_YEARS
    ):
        raise ValueError("wait manifest scope must contain 2009-2025")
    serialization = _mapping(value.get("serialization"), "wait serialization")
    if serialization.get("format") != "JSON Lines":
        raise ValueError("wait dataset must use JSON Lines")
    entries = _sequence(value.get("years"), "wait manifest years")
    entry_years = tuple(
        _integer(_mapping(entry, "wait year entry").get("year"), "wait year")
        for entry in entries
    )
    if entry_years != SUPPORTED_YEARS:
        raise ValueError("wait manifest year entries must be ordered 2009-2025")


def validate_summary_identity(value: Mapping[str, Any]) -> None:
    """Validate the independent Article-1 reconciliation oracle identity."""
    if value.get("analysis_name") != "dealer-child-riichi-points-v1":
        raise ValueError("unexpected Article-1 analysis name")
    if value.get("schema_version") != 2:
        raise ValueError("unexpected Article-1 schema version")
    source = _mapping(value.get("source"), "Article-1 source")
    expected_source = {
        "repository": EXPECTED_REPOSITORY,
        "release_tag": EXPECTED_RELEASE,
        "format": "MJAI JSON Lines",
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise ValueError(f"unexpected Article-1 source {key}")
    scope = _mapping(value.get("scope"), "Article-1 scope")
    expected_scope = {
        "rule_code": "00a9",
        "aka_flag": True,
        "bakaze": "E",
        "riichi_population": "established",
    }
    for key, expected in expected_scope.items():
        if scope.get(key) != expected or type(scope.get(key)) is not type(expected):
            raise ValueError(f"unexpected Article-1 scope {key}")
    if _strict_years(scope.get("primary_years"), "Article-1 primary years") != (
        PRIMARY_YEARS
    ):
        raise ValueError("Article-1 primary years must be 2020-2025")
    if _strict_years(scope.get("selected_years"), "Article-1 selected years") != (
        SUPPORTED_YEARS
    ):
        raise ValueError("Article-1 selected years must be 2009-2025")
    entries = _sequence(value.get("years"), "Article-1 years")
    if (
        tuple(
            _integer(_mapping(entry, "Article-1 year entry").get("year"), "year")
            for entry in entries
        )
        != SUPPORTED_YEARS
    ):
        raise ValueError("Article-1 year entries must be ordered 2009-2025")
    annual_counts = {
        _integer(_mapping(entry, "Article-1 year entry").get("year"), "year"): (
            _article1_counts(_mapping(entry, "Article-1 year entry"))
        )
        for entry in entries
    }
    reconcile_periods(annual_counts, SUPPORTED_YEARS, value)


def _article1_counts(container: Mapping[str, Any]) -> ReconciliationCounts:
    counts = ReconciliationCounts()
    for sensitivity, source in (
        ("all_riichi", container),
        (
            "normal_riichi",
            _mapping(
                container.get("sensitivity_normal_riichi_only"),
                "Article-1 normal sensitivity",
            ),
        ),
    ):
        for role in ("dealer", "nondealer"):
            group = _mapping(source.get(role), f"Article-1 {role}")
            for field_name in (
                "riichis",
                "wins",
                "other_wins",
                "draws",
                "tsumo_wins",
                "ron_wins",
            ):
                counts.values[(sensitivity, role, field_name)] = _integer(
                    group.get(field_name), f"Article-1 {role} {field_name}"
                )
    return counts


def reconcile_year(
    year: int,
    actual: ReconciliationCounts,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact all/normal role/outcome equality with Article 1."""
    entries = _sequence(summary.get("years"), "Article-1 years")
    matches = [
        _mapping(entry, "Article-1 year")
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("year") == year
    ]
    if len(matches) != 1:
        raise ValueError(f"Article-1 summary has no unique year {year}")
    return _reconcile_container(actual, matches[0], f"year {year}")


def _reconcile_container(
    actual: ReconciliationCounts,
    container_value: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    expected_document: dict[str, Any] = {}
    for sensitivity, container in (
        ("all_riichi", container_value),
        (
            "normal_riichi",
            _mapping(
                container_value.get("sensitivity_normal_riichi_only"),
                f"Article-1 {label} normal sensitivity",
            ),
        ),
    ):
        expected_document[sensitivity] = {}
        for role in ("dealer", "nondealer"):
            expected = _mapping(container.get(role), f"Article-1 {label} {role}")
            observed = actual.group(sensitivity, role)
            expected_group: dict[str, int] = {}
            for field_name, observed_value in observed.items():
                expected_value = _integer(
                    expected.get(field_name),
                    f"Article-1 {label} {sensitivity} {role} {field_name}",
                )
                expected_group[field_name] = expected_value
                if observed_value != expected_value:
                    raise ValueError(
                        f"Article-1 reconciliation mismatch for {label} "
                        f"{sensitivity} {role} {field_name}: "
                        f"{observed_value} != {expected_value}"
                    )
            expected_document[sensitivity][role] = expected_group
    return {
        "status": "matched",
        "expected": expected_document,
        "observed": actual.to_dict(),
    }


def reconcile_periods(
    year_counts: Mapping[int, ReconciliationCounts],
    selected_years: tuple[int, ...],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild every fully covered Article-1 period from annual integers."""
    period_values = _mapping(summary.get("periods"), "Article-1 periods")
    summary_selected = _strict_years(
        _mapping(summary.get("scope"), "Article-1 scope").get("selected_years"),
        "Article-1 selected years",
    )
    definitions = (
        ("primary_2020_2025", PRIMARY_YEARS),
        ("long_term_2009_2025", SUPPORTED_YEARS),
        ("selected", summary_selected),
    )
    documents: dict[str, Any] = {}
    available = set(year_counts)
    for name, required_years in definitions:
        if name == "selected" and selected_years != summary_selected:
            documents[name] = {
                "status": "skipped",
                "reason": "cli_selected_years_do_not_equal_article1_selected_years",
                "required_years": list(required_years),
            }
            continue
        missing = tuple(year for year in required_years if year not in available)
        if missing:
            documents[name] = {
                "status": "skipped",
                "reason": "not_all_period_years_processed",
                "required_years": list(required_years),
                "missing_years": list(missing),
            }
            continue
        period = _mapping(period_values.get(name), f"Article-1 period {name}")
        if _strict_years(period.get("years"), f"Article-1 period {name} years") != (
            required_years
        ):
            raise ValueError(f"Article-1 period {name} years are inconsistent")
        combined = ReconciliationCounts()
        for year in required_years:
            combined.merge(year_counts[year])
        document = _reconcile_container(combined, period, f"period {name}")
        document["years"] = list(required_years)
        documents[name] = document
    return documents


def run_analysis(
    *,
    years: tuple[int, ...],
    dataset_root: Path,
    raw_root: Path,
    point_summary_path: Path,
    max_files: int | None,
    workers: int = DEFAULT_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Stream the selected years and return a deterministic result document."""
    _validate_parallelism(workers, batch_size)
    manifest_path = dataset_root / "manifest.json"
    manifest = load_manifest(manifest_path, expected_mode="full")
    validate_manifest_identity(manifest)
    partial = max_files is not None
    article1: Mapping[str, Any] | None = None
    if not partial:
        article1 = _load_json_object(point_summary_path, "Article-1 summary")
        validate_summary_identity(article1)
    builder = DefenseAnalysisBuilder(years)
    counts = RunCounts()
    raw_digest = hashlib.sha256()
    year_reconciliation: dict[int, ReconciliationCounts] = {
        year: ReconciliationCounts() for year in years
    }
    year_reconciliation_documents: dict[str, Any] = {}
    annual_entries = {
        _integer(_mapping(value, "wait year").get("year"), "wait year"): _mapping(
            value, "wait year"
        )
        for value in _sequence(manifest.get("years"), "wait years")
    }

    remaining = max_files
    progress_interval = 100 if partial else 10_000
    executor = (
        ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
        if workers > 1
        else None
    )
    failed = False
    try:
        for year in years:
            if remaining == 0:
                break
            limit = remaining
            tasks = iter_source_tasks(
                dataset_root,
                raw_root,
                year,
                partial=partial,
                max_files=limit,
            )
            results = iter_source_results(
                tasks,
                executor=executor,
                workers=workers,
                batch_size=batch_size,
            )
            for result in results:
                _apply_source_result(
                    result,
                    builder=builder,
                    reconciliation=year_reconciliation[year],
                    counts=counts,
                    raw_digest=raw_digest,
                )
                if counts.source_files % progress_interval == 0:
                    print(
                        f"processed {counts.source_files} source files, "
                        f"{counts.focal_riichis} focal riichis",
                        file=sys.stderr,
                        flush=True,
                    )
            if remaining is not None:
                remaining = max(0, max_files - counts.source_files)
            if not partial and article1 is not None:
                year_reconciliation_documents[str(year)] = reconcile_year(
                    year, year_reconciliation[year], article1
                )
            print(
                f"completed {year}: {counts.source_files} cumulative source files, "
                f"{counts.focal_riichis} cumulative focal riichis",
                file=sys.stderr,
                flush=True,
            )
    except BaseException:  # noqa: BLE001 - executor cleanup must survive interrupts
        failed = True
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=failed)

    input_entries = [annual_entries[year] for year in years]
    periods = builder.to_dict()
    period_definitions = builder.period_definitions()
    standardization_statistics = builder.standardization_sufficient_statistics()
    selected_all = periods["selected"]["all_riichi"]
    focal_total = sum(selected_all["focals"].values())
    primary_responder_total = sum(
        selected_all["focals"][role] * (3 if role == "dealer" else 2)
        for role in ("dealer", "nondealer")
    )
    primary_decision_total = sum(
        selected_all["exposures"][role]["primary_child_discard_opportunities"]
        for role in ("dealer", "nondealer")
    )
    all_decision_total = sum(
        selected_all["exposures"][role]["all_responder_discard_opportunities"]
        for role in ("dealer", "nondealer")
    )
    if partial:
        reconciliation_document: dict[str, Any] = {
            "status": "skipped_partial_run",
            "skipped_items": [
                "full_annual_wait_file_hash_and_record_count",
                "article1_annual_outcome_counts",
                "article1_period_outcome_counts",
            ],
            "observed_partial_counts": {
                str(year): year_reconciliation[year].to_dict()
                for year in years
                if year_reconciliation[year].values
            },
        }
    else:
        if article1 is None:
            raise RuntimeError("full analysis requires the Article-1 summary")
        reconciliation_document = {
            "status": "matched",
            "years": year_reconciliation_documents,
            "periods": reconcile_periods(year_reconciliation, years, article1),
            "skipped_items": [],
        }
    result = {
        "analysis_name": "dealer-riichi-defense-v1",
        "schema_version": 3,
        "status": "partial_smoke" if partial else "complete",
        "source": {
            "repository": EXPECTED_REPOSITORY,
            "release_tag": EXPECTED_RELEASE,
            "format": "MJAI JSON Lines",
        },
        "scope": {
            "rule_code": "00a9",
            "aka_flag": True,
            "bakaze": "E",
            "selected_years": list(years),
            "primary_years": list(PRIMARY_YEARS),
            "focal_population": "first established riichi in each east-round hand",
            "primary_responders": "child seats other than the focal actor",
        },
        "run_mode": {
            "partial": partial,
            "max_files": max_files,
            "article1_reconciliation": "disabled" if partial else "exact",
        },
        "inputs": {
            "wait_manifest": {
                "logical_path": "data/processed/riichi-waits-v1/manifest.json",
                "sha256": sha256_path(manifest_path),
            },
            "wait_year_files": [
                {
                    "year": entry["year"],
                    "logical_path": (
                        f"data/processed/riichi-waits-v1/{entry['output_filename']}"
                    ),
                    "manifest_sha256": entry["sha256"],
                    "sha256_verification": (
                        "not_recomputed_for_partial_smoke"
                        if partial
                        else "computed_and_matched_manifest"
                    ),
                    "manifest_records": entry["output_records"],
                }
                for entry in input_entries
            ],
            "raw_files": {
                "logical_pattern": "data/raw/YYYY/*.mjson",
                "processed_files": counts.source_files,
                "processed_bytes": counts.raw_bytes,
                "ordered_path_and_content_digest_sha256": raw_digest.hexdigest(),
            },
            "article1_summary": (
                {
                    "logical_path": (
                        "outputs/dealer-child-riichi-points/summary-v1.json"
                    ),
                    "sha256": sha256_path(point_summary_path),
                }
                if not partial
                else None
            ),
        },
        "definitions": {
            "headline_genbutsu": (
                "observed genbutsu discards divided by observed primary child "
                "discard decisions; tiles available in hand are not reconstructed"
            ),
            "safe_set_timing": (
                "focal river plus tiles already passed after acceptance, evaluated "
                "before the current discard"
            ),
            "response_window": (
                "from immediately after focal reach_accepted through the next "
                "reach_accepted or the hand result; the second declaration discard "
                "is included and the accepting event ends exposure"
            ),
            "window_end_reason": (
                "hand_end or second_reach_accepted; this controls only the exposure "
                "window and is distinct from the actual hand result"
            ),
            "terminal_outcome": (
                "focal_tsumo, focal_ron, other_win, or draw, classified from the "
                "complete hand even when the response window was censored"
            ),
            "terminal_auxiliary": (
                "audits multi-ron, established and unestablished second declaration "
                "discards, ron events on those discards, and focal ron not attributable "
                "to a discard opportunity"
            ),
            "wait_adjustment": (
                "actual wait shape is unobserved by responders and is used only "
                "as a compositional adjustment covariate, not a causal explanation"
            ),
            "comparison": "descriptive association, not a causal effect",
            "standardization": (
                "exact common-support direct standardization with pooled-denominator "
                "stratum weights; annual integer stratum rows are the exact source of "
                "truth and fixed-precision ROUND_HALF_EVEN Decimal weighted sums are "
                "display values, never aggregate Fractions or float inputs"
            ),
            "input_cross_check": (
                "raw focal river, declaration identity and lines, every east-round "
                "established riichi, normalized focal-ron tile, rule 00a9, and red-five "
                "flag are matched bidirectionally to the wait dataset"
            ),
        },
        "input_counts": {
            "source_files": counts.source_files,
            "wait_records": counts.wait_records,
            "riichi_kyokus": counts.riichi_kyokus,
            "focal_riichis": counts.focal_riichis,
            "raw_events": counts.raw_events,
            "all_responder_pairs": focal_total * 3,
            "primary_child_responder_pairs": primary_responder_total,
            "all_responder_discard_decisions": all_decision_total,
            "primary_child_discard_decisions": primary_decision_total,
        },
        "article1_reconciliation": reconciliation_document,
        "period_definitions": period_definitions,
        "standardization_sufficient_statistics": standardization_statistics,
        "periods": periods,
    }
    return result


def _validate_parallelism(workers: int, batch_size: int) -> None:
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch size must be a positive integer")


def iter_source_tasks(
    dataset_root: Path,
    raw_root: Path,
    year: int,
    *,
    partial: bool,
    max_files: int | None,
) -> Iterator[SourceTask]:
    """Yield canonical source tasks without reading raw MJAI in the parent."""
    resolved_root = raw_root.resolve()
    for source_path, records in iter_source_groups(
        dataset_root,
        year,
        partial=partial,
        max_files=max_files,
    ):
        raw_path = (resolved_root / Path(source_path)).resolve()
        try:
            raw_path.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(f"source path escapes raw root: {source_path}") from error
        yield SourceTask(
            year=year,
            source_path=source_path,
            raw_path=raw_path,
            records=records,
        )


def iter_source_results(
    tasks: Iterator[SourceTask],
    *,
    executor: ProcessPoolExecutor | None,
    workers: int,
    batch_size: int,
) -> Iterator[SourceResult]:
    """Process source tasks serially or in bounded deterministic batches."""
    _validate_parallelism(workers, batch_size)
    if workers == 1:
        if executor is not None:
            raise ValueError("serial source processing must not receive an executor")
        for task in tasks:
            yield process_source_task(task)
        return
    if executor is None:
        raise ValueError("parallel source processing requires an executor")
    yield from _iter_parallel_source_results(
        tasks,
        executor=executor,
        workers=workers,
        batch_size=batch_size,
    )


def _iter_source_batches(
    tasks: Iterator[SourceTask], batch_size: int
) -> Iterator[tuple[SourceTask, ...]]:
    iterator = iter(tasks)
    try:
        while True:
            batch = tuple(islice(iterator, batch_size))
            if not batch:
                return
            yield batch
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


def _iter_parallel_source_results(
    tasks: Iterator[SourceTask],
    *,
    executor: Any,
    workers: int,
    batch_size: int,
) -> Iterator[SourceResult]:
    """Yield worker results in submission order with bounded futures."""
    batches = _iter_source_batches(tasks, batch_size)
    pending: deque[tuple[tuple[SourceTask, ...], Future[tuple[SourceResult, ...]]]] = (
        deque()
    )
    in_flight_limit = workers * IN_FLIGHT_BATCHES_PER_WORKER
    try:
        for batch in islice(batches, in_flight_limit):
            pending.append((batch, executor.submit(process_source_batch, batch)))
        while pending:
            batch, future = pending.popleft()
            try:
                results = future.result()
            except Exception as error:
                _cancel_source_futures(pending)
                first = batch[0].source_path
                last = batch[-1].source_path
                raise RuntimeError(
                    f"source batch {first} through {last} failed: {error}"
                ) from error
            if len(results) != len(batch):
                raise RuntimeError("worker result count does not match source batch")
            for task, result in zip(batch, results, strict=True):
                if (result.year, result.source_path) != (task.year, task.source_path):
                    raise RuntimeError(
                        "worker result identity does not match source task"
                    )
                yield result
            next_batch = next(batches, None)
            if next_batch is not None:
                pending.append(
                    (next_batch, executor.submit(process_source_batch, next_batch))
                )
    finally:
        _cancel_source_futures(pending)
        close = getattr(batches, "close", None)
        if close is not None:
            close()


def _cancel_source_futures(
    pending: Sequence[tuple[tuple[SourceTask, ...], Future[tuple[SourceResult, ...]]]],
) -> None:
    for _, future in pending:
        future.cancel()


def process_source_batch(
    tasks: tuple[SourceTask, ...],
) -> tuple[SourceResult, ...]:
    """Spawn-safe worker entry point for one bounded source batch."""
    return tuple(process_source_task(task) for task in tasks)


def process_source_task(task: SourceTask) -> SourceResult:
    """Read and validate one raw source without mutating parent state."""
    try:
        events, raw_sha, raw_size = load_mjai_once(task.raw_path)
        observations, reconciliation = analyze_source_events(
            year=task.year,
            source_path=task.source_path,
            events=events,
            records=task.records,
        )
    except Exception as error:  # noqa: BLE001 - process boundary adds source context
        raise RuntimeError(
            f"{task.source_path}: source worker failed: {error}"
        ) from error
    return SourceResult(
        year=task.year,
        source_path=task.source_path,
        raw_sha256=raw_sha,
        raw_bytes=raw_size,
        raw_events=len(events),
        wait_records=len(task.records),
        observations=observations,
        reconciliation_items=reconciliation.frozen_items(),
    )


def _apply_source_result(
    result: SourceResult,
    *,
    builder: DefenseAnalysisBuilder,
    reconciliation: ReconciliationCounts,
    counts: RunCounts,
    raw_digest: Any,
) -> None:
    """Apply one pure result to all parent-owned state in canonical order."""
    raw_digest.update(result.source_path.encode("utf-8"))
    raw_digest.update(b"\0")
    raw_digest.update(bytes.fromhex(result.raw_sha256))
    counts.raw_bytes += result.raw_bytes
    counts.raw_events += result.raw_events
    counts.source_files += 1
    counts.wait_records += result.wait_records
    counts.riichi_kyokus += result.focal_riichis
    counts.focal_riichis += result.focal_riichis
    reconciliation.merge_items(result.reconciliation_items)
    for observation in result.observations:
        if (
            observation.year != result.year
            or observation.source_path != result.source_path
        ):
            raise RuntimeError(
                "worker observation identity does not match source result"
            )
        builder.add(observation)


def iter_source_groups(
    dataset_root: Path,
    year: int,
    *,
    partial: bool,
    max_files: int | None,
) -> Iterator[tuple[str, tuple[RiichiWaitDatasetRecord, ...]]]:
    """Yield adjacent source groups while retaining only one source at a time."""
    iterator: Iterator[RiichiWaitDatasetRecord]
    if partial:
        iterator = iter_dataset_records(dataset_root / f"{year}.jsonl.gz")
    else:
        iterator = iter_validated_year_records(dataset_root, year)
    previous_source: str | None = None
    emitted = 0
    try:
        for source_path, group in groupby(
            iterator, key=lambda record: record.relative_source_path
        ):
            if previous_source is not None and source_path <= previous_source:
                raise ValueError("wait records are duplicated or not ordered by source")
            if max_files is not None and emitted >= max_files:
                break
            records = tuple(group)
            if not records:
                raise ValueError("empty source group")
            previous_key: tuple[int, int, int] | None = None
            for record in records:
                if record.year != year:
                    raise ValueError("wait record year does not match annual file")
                key = (
                    record.start_kyoku_line,
                    record.reach_accepted_event_index,
                    record.actor,
                )
                if previous_key is not None and key <= previous_key:
                    raise ValueError(
                        "wait records are duplicated or not canonically ordered"
                    )
                previous_key = key
            yield source_path, records
            previous_source = source_path
            emitted += 1
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


def load_mjai_once(path: Path) -> tuple[list[dict[str, Any]], str, int]:
    """Read, hash, and parse one raw source file with one filesystem read."""
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    payload = gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid MJAI JSON at {path.name}:{line_number}"
            ) from error
        if not isinstance(event, dict):
            raise ValueError(  # noqa: TRY004 - malformed MJAI is a data-value error
                f"MJAI event must be an object at {path.name}:{line_number}"
            )
        events.append(event)
    return events, digest, len(raw)


def analyze_source_events(
    *,
    year: int,
    source_path: str,
    events: list[dict[str, Any]],
    records: Sequence[RiichiWaitDatasetRecord],
) -> tuple[tuple[FocalObservation, ...], ReconciliationCounts]:
    """Validate one source and return observations plus reconciliation delta."""
    if not is_target_game(source_path, events):
        raise ValueError(
            f"{source_path}: source must have rule code 00a9 and aka_flag=true"
        )
    reconciliation = ReconciliationCounts()
    observations: list[FocalObservation] = []
    line_by_id = {id(event): index for index, event in enumerate(events, start=1)}
    kyokus = split_kyoku(events)
    records_by_line: dict[int, tuple[RiichiWaitDatasetRecord, ...]] = {}
    for start_line, group in groupby(
        records, key=lambda record: record.start_kyoku_line
    ):
        if start_line in records_by_line:
            raise ValueError(f"{source_path}:{start_line}: duplicate wait-record group")
        records_by_line[start_line] = tuple(group)
    for kyoku in kyokus:
        start_line = line_by_id[id(kyoku[0])]
        start = _mapping(kyoku[0], "start_kyoku")
        if start.get("bakaze") != "E":
            continue
        established = extract_established_riichis(kyoku)
        group_records = records_by_line.pop(start_line, ())
        if len(established) != len(group_records):
            raise ValueError(
                f"{source_path}:{start_line}: east-round established-riichi "
                "identity mismatch"
            )
        if not established:
            continue
        ordered_records = tuple(
            sorted(group_records, key=lambda value: value.reach_accepted_event_index)
        )
        for record, riichi in zip(ordered_records, established, strict=True):
            if (
                record.year != year
                or record.relative_source_path != source_path
                or record.bakaze != start.get("bakaze")
                or record.kyoku != start.get("kyoku")
                or record.honba != start.get("honba")
                or record.oya != start.get("oya")
                or record.actor != riichi.actor
                or record.reach_event_index != riichi.reach_event_index
                or record.declaration_dahai_event_index != riichi.reach_event_index + 1
                or record.reach_accepted_event_index
                != riichi.reach_accepted_event_index
                or record.reach_line != start_line + riichi.reach_event_index
                or record.declaration_dahai_line
                != start_line + riichi.reach_event_index + 1
                or record.reach_accepted_line
                != start_line + riichi.reach_accepted_event_index
            ):
                raise ValueError(
                    f"{source_path}:{start_line}: wait/source riichi mismatch"
                )
            outcome, method = _established_outcome(kyoku, riichi.actor)
            role: FocalRole = "dealer" if riichi.actor == record.oya else "nondealer"
            reconciliation.add(role, riichi.reach_type, outcome, method)
        focal_record = ordered_records[0]
        observations.append(analyze_focal_kyoku(focal_record, kyoku))
    if records_by_line:
        extra_lines = tuple(sorted(records_by_line))
        raise ValueError(
            f"{source_path}: wait records have no matching east-round kyoku: "
            f"{extra_lines}"
        )
    return tuple(observations), reconciliation


def process_source_file(
    *,
    year: int,
    source_path: str,
    events: list[dict[str, Any]],
    records: Sequence[RiichiWaitDatasetRecord],
    builder: DefenseAnalysisBuilder,
    reconciliation: ReconciliationCounts,
) -> int:
    """Compatibility wrapper applying one pure source result immediately."""
    observations, delta = analyze_source_events(
        year=year,
        source_path=source_path,
        events=events,
        records=records,
    )
    reconciliation.merge(delta)
    for observation in observations:
        builder.add(observation)
    return len(observations)


def _established_outcome(
    kyoku: Sequence[dict[str, Any]], actor: int
) -> tuple[str, str | None]:
    result_events: list[Mapping[str, Any]] = []
    started = False
    for index, event in enumerate(kyoku[:-1]):
        event_type = event.get("type") if isinstance(event, dict) else None
        if event_type in {"hora", "ryukyoku"}:
            result_events.append(event)
            started = True
        elif started:
            raise ValueError(f"event {index}: result events must be contiguous")
    if not result_events:
        raise ValueError("kyoku has no result event")
    horas = [event for event in result_events if event.get("type") == "hora"]
    draws = [event for event in result_events if event.get("type") == "ryukyoku"]
    if horas and draws or len(draws) > 1:
        raise ValueError("kyoku has inconsistent result events")
    winners: set[int] = set()
    for hora in horas:
        winner = _actor(hora.get("actor"), "hora actor")
        target = _actor(hora.get("target"), "hora target")
        if winner in winners:
            raise ValueError("one actor cannot have multiple hora events")
        winners.add(winner)
        if winner == actor:
            return "win", "tsumo" if target == winner else "ron"
    return ("draws", None) if draws else ("other_wins", None)


def render_markdown(document: Mapping[str, Any]) -> str:
    """Render all periods, both sensitivities, and raw/adjusted rates."""
    mode = document["status"]
    lines = [
        "# 親リーチ後の子respondersの観測行動",
        "",
        f"- 実行状態: `{mode}`",
        f"- 対象年: {', '.join(map(str, document['scope']['selected_years']))}",
        f"- raw source files: {document['input_counts']['source_files']:,}",
        f"- focal riichis: {document['input_counts']['focal_riichis']:,}",
        "",
        "> 本表は観察的な記述比較であり、因果効果ではない。現物率の分母は",
        "> 観測された子responderの捨て牌であり、手牌中の選択肢は復元していない。",
        "> 実待ちは見えない構成調整変数で、他家の意思決定原因とは解釈しない。",
        "",
    ]
    periods = _mapping(document["periods"], "periods")
    for period_name, period_value in periods.items():
        period = _mapping(period_value, period_name)
        lines.extend((f"## {period_name}", ""))
        for sensitivity in ("all_riichi", "normal_riichi"):
            result = _mapping(period[sensitivity], sensitivity)
            lines.extend((f"### {sensitivity}", ""))
            lines.extend(
                (
                    "| 指標 | 親 raw | 子 raw | 親 adjusted | 子 adjusted | 差 | status | common/excluded/union strata | 親 common/total(outside) | 子 common/total(outside) | coverage 親/子 | common pooled denom |",
                    "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
                )
            )
            metrics = _mapping(result["primary_child_responders"], "metrics")
            for metric in (*DECISION_METRICS, *RESPONDER_METRICS):
                value = _mapping(metrics[metric], metric)
                raw = _mapping(value["raw"], "raw")
                adjusted = _mapping(value["common_support_standardized"], "adjusted")
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            metric,
                            _format_rate(raw["dealer"]["rate"]),
                            _format_rate(raw["nondealer"]["rate"]),
                            _format_rate(adjusted["dealer_rate"]),
                            _format_rate(adjusted["nondealer_rate"]),
                            _format_rate(adjusted["dealer_minus_nondealer"]),
                            adjusted["status"],
                            (
                                f"{adjusted['common_strata']}/"
                                f"{adjusted['union_strata'] - adjusted['common_strata']}/"
                                f"{adjusted['union_strata']}"
                            ),
                            _format_support_denominator(adjusted, "dealer"),
                            _format_support_denominator(adjusted, "nondealer"),
                            (
                                f"{_format_rate(adjusted['coverage']['dealer'])}/"
                                f"{_format_rate(adjusted['coverage']['nondealer'])}"
                            ),
                            str(adjusted["common_pooled_denominator"]),
                        )
                    )
                    + " |"
                )
            lines.extend(("", "#### Exposure / outcome", ""))
            lines.append(
                "| role | focals | child discard exposure | focal draw exposure | "
                "terminal focal ron | terminal focal tsumo | second-reach censored | "
                "multi-ron hands | unestablished declaration rons | non-discard focal rons |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for role in ("dealer", "nondealer"):
                lines.append(
                    f"| {role} | {result['focals'][role]} | "
                    f"{result['exposures'][role]['primary_child_discard_opportunities']} | "
                    f"{result['exposures'][role]['focal_draw_opportunities']} | "
                    f"{result['terminal_outcomes'][role]['focal_ron']} | "
                    f"{result['terminal_outcomes'][role]['focal_tsumo']} | "
                    f"{result['window_end_reasons'][role]['second_reach_accepted']} | "
                    f"{result['terminal_auxiliary'][role]['terminal_multi_ron_hands']} | "
                    f"{result['terminal_auxiliary'][role]['ron_events_on_second_reach_declaration']} | "
                    f"{result['terminal_auxiliary'][role]['focal_rons_without_discard']} |"
                )
            lines.extend(
                (
                    "",
                    "#### 安全度分類（primary child decisions）",
                    "",
                    "| focal role | safety class | decisions |",
                    "|---|---|---:|",
                )
            )
            for row in result["safety_class_distribution"]:
                lines.append(
                    f"| {row['focal_role']} | {row['safety_class']} | "
                    f"{row['decisions']} |"
                )
            lines.extend(
                (
                    "",
                    "#### Secondary: all responders raw rates",
                    "",
                    "| metric | 親 focal | 子 focal |",
                    "|---|---:|---:|",
                )
            )
            secondary = _mapping(result["secondary_all_responders"], "secondary")
            for metric in (*DECISION_METRICS, *RESPONDER_METRICS):
                raw = _mapping(secondary[metric], metric)
                lines.append(
                    f"| {metric} | {_format_rate(raw['dealer']['rate'])} | "
                    f"{_format_rate(raw['nondealer']['rate'])} |"
                )
            lines.extend(
                (
                    "",
                    "#### Primary raw breakdowns",
                    "",
                    "| unit | metric | dimension | value | 親 focal | 子 focal |",
                    "|---|---|---|---|---:|---:|",
                )
            )
            for row in result["breakdowns"]:
                raw = row["raw"]
                lines.append(
                    f"| {row['unit']} | {row['metric']} | {row['dimension']} | "
                    f"{row['value']} | {_format_rate(raw['dealer']['rate'])} | "
                    f"{_format_rate(raw['nondealer']['rate'])} |"
                )
            lines.extend(
                (
                    "",
                    f"監査例: {len(result['audit_samples'])} focal（先頭固定上限）",
                    "",
                )
            )
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--years", type=int, nargs="+")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--max-files", type=_positive_int)
    parser.add_argument("--workers", type=_positive_int, default=DEFAULT_WORKERS)
    parser.add_argument("--batch-size", type=_positive_int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--point-summary", type=Path, default=DEFAULT_POINT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    args = parser.parse_args(argv)
    years = SUPPORTED_YEARS if args.all else tuple(sorted(set(args.years)))
    if not years or any(year not in SUPPORTED_YEARS for year in years):
        parser.error("years must be selected from 2009 through 2025")
    if args.years is not None and len(years) != len(args.years):
        parser.error("--years must not contain duplicates")
    args.selected_years = years
    output_json = args.output_dir / "summary-v3.json"
    output_markdown = args.output_dir / "summary-v3.md"
    if args.max_files is not None and (
        _same_path(output_json, DEFAULT_OUTPUT_JSON)
        or _same_path(output_markdown, DEFAULT_OUTPUT_MARKDOWN)
    ):
        parser.error("--max-files requires a non-default --output-dir")
    if args.max_files is not None and len(years) != 1:
        parser.error("--max-files requires exactly one selected year")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_analysis(
        years=args.selected_years,
        dataset_root=args.dataset_root,
        raw_root=args.raw_root,
        point_summary_path=args.point_summary,
        max_files=args.max_files,
        workers=args.workers,
        batch_size=args.batch_size,
    )
    json_target = args.output_dir / "summary-v3.json"
    markdown_target = args.output_dir / "summary-v3.md"
    json_text = canonical_json(result)
    json.loads(json_text)
    publish_output_bundle(
        {
            json_target: json_text,
            markdown_target: render_markdown(result),
        }
    )
    print(
        f"published {json_target} and {markdown_target} "
        f"({result['input_counts']['source_files']} files)",
        file=sys.stderr,
        flush=True,
    )
    return 0


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, label: str) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    return _mapping(value, label)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    return value


def _actor(value: object, label: str) -> int:
    actor = _integer(value, label)
    if actor not in range(4):
        raise ValueError(f"{label} must be from 0 through 3")
    return actor


def _strict_years(value: object, label: str) -> tuple[int, ...]:
    years = tuple(_integer(year, label) for year in _sequence(value, label))
    if not years or tuple(sorted(set(years))) != years:
        raise ValueError(f"{label} must be non-empty, unique, and ordered")
    return years


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def _format_rate(value: object) -> str:
    if value is None:
        return "N/A"
    mapping = _mapping(value, "fraction")
    decimal = mapping.get("decimal")
    if not isinstance(decimal, str):
        raise TypeError("rate decimal must be a string")
    context = Context(
        prec=DECIMAL_PRECISION_DIGITS,
        rounding=ROUND_HALF_EVEN,
        Emin=DECIMAL_MIN_EXPONENT,
        Emax=DECIMAL_MAX_EXPONENT,
    )
    with localcontext(context):
        return f"{Decimal(decimal) * Decimal(100):.4f}%"


def _format_support_denominator(value: Mapping[str, Any], role: str) -> str:
    denominators = _mapping(value["eligible_denominators"], "denominators")
    role_value = _mapping(denominators[role], f"{role} denominators")
    return (
        f"{role_value['common']}/{role_value['total']}"
        f"({role_value['outside_common_support']})"
    )


if __name__ == "__main__":
    raise SystemExit(main())
