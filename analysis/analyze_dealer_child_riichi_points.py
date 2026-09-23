"""Aggregate dealer/nondealer riichi points directly from raw MJAI."""

import argparse
import gzip
import io
import json
import multiprocessing
import os
import shutil
import tempfile
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mahjong_analysis.dealer_child_riichi_points import (
    SUPPORTED_YEARS,
    CohortReference,
    RiichiPointAggregationCancelled,
    RiichiPointAnalysis,
    RiichiPointRecord,
    YearlyRiichiPointResult,
    aggregate_riichi_point_years,
    load_cohort_reference,
    normalize_years,
    riichi_point_record_document,
    write_outputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_COHORT_MANIFEST = (
    PROJECT_ROOT / "data" / "processed" / "riichi-waits-v1" / "manifest.json"
)
DEFAULT_OUTPUT_JSON = (
    PROJECT_ROOT / "outputs" / "dealer-child-riichi-points" / "summary-v1.json"
)
DEFAULT_OUTPUT_MARKDOWN = (
    PROJECT_ROOT / "outputs" / "dealer-child-riichi-points" / "summary-v1.md"
)
DEFAULT_OUTPUT_RECORDS = (
    PROJECT_ROOT / "outputs" / "dealer-child-riichi-points" / "records-v1.jsonl.gz"
)
DEFAULT_PROGRESS_INTERVAL = 10_000
COHORT_MANIFEST_LOGICAL_PATH = "data/processed/riichi-waits-v1/manifest.json"
_WORKER_CANCELLATION_EVENT: Any | None = None


@dataclass(frozen=True)
class _YearTask:
    """One independently runnable and picklable annual scan."""

    year: int
    raw_root: Path
    cohort: CohortReference
    records_path: Path
    progress_interval: int


def _initialize_worker(cancellation_event: Any) -> None:
    """Install a shared cancellation event during spawned-worker startup."""
    global _WORKER_CANCELLATION_EVENT
    _WORKER_CANCELLATION_EVENT = cancellation_event


def _positive_int(value: str) -> int:
    """Parse a strictly positive command-line integer."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate dealer/nondealer established-riichi points from raw MJAI."
        )
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--years",
        nargs="+",
        type=int,
        choices=SUPPORTED_YEARS,
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="process every year from 2009 through 2025",
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--cohort-manifest",
        type=Path,
        default=DEFAULT_COHORT_MANIFEST,
        help="canonical riichi-waits manifest used only for count validation",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=DEFAULT_OUTPUT_MARKDOWN,
    )
    parser.add_argument(
        "--output-records",
        type=Path,
        default=DEFAULT_OUTPUT_RECORDS,
        help="gzip JSON Lines audit dataset, one row per established riichi",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=DEFAULT_PROGRESS_INTERVAL,
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help="number of years to process concurrently (default: 1)",
    )
    return parser.parse_args(argv)


def selected_years_from_args(args: argparse.Namespace) -> tuple[int, ...]:
    """Resolve the mutually exclusive year selection."""
    return normalize_years(SUPPORTED_YEARS if args.all else args.years)


def _write_record(
    text: io.TextIOWrapper,
    record: RiichiPointRecord,
) -> None:
    """Write one canonical audit record to an annual gzip stream."""
    document = riichi_point_record_document(record)
    text.write(
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )


def _run_year(task: _YearTask) -> YearlyRiichiPointResult:
    """Scan, aggregate, and validate one year; safe for Windows spawn."""
    record_count = 0

    def report_progress(year: int, processed: int) -> None:
        print(f"{year}: processed {processed:,} files", flush=True)

    def write_record(record: RiichiPointRecord) -> None:
        nonlocal record_count
        _write_record(text, record)
        record_count += 1

    try:
        if (
            _WORKER_CANCELLATION_EVENT is not None
            and _WORKER_CANCELLATION_EVENT.is_set()
        ):
            raise RiichiPointAggregationCancelled(
                f"aggregation cancelled for {task.year} before start"
            )
        with (
            task.records_path.open("xb") as raw_file,
            gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_file,
                compresslevel=9,
                mtime=0,
            ) as gzip_file,
            io.TextIOWrapper(gzip_file, encoding="utf-8", newline="\n") as text,
        ):
            analysis = aggregate_riichi_point_years(
                (task.year,),
                task.raw_root,
                task.cohort,
                progress_interval=task.progress_interval,
                progress_callback=report_progress,
                record_callback=write_record,
                cancellation_callback=(
                    None
                    if _WORKER_CANCELLATION_EVENT is None
                    else _WORKER_CANCELLATION_EVENT.is_set
                ),
            )
        result = analysis.years[0]
        if record_count != result.established_riichis:
            raise RuntimeError(
                f"audit record count mismatch for {task.year}: "
                f"expected {result.established_riichis}, wrote {record_count}"
            )
        return result
    except RiichiPointAggregationCancelled:
        print(f"{task.year}: cancelled after peer failure", flush=True)
        task.records_path.unlink(missing_ok=True)
        raise
    except BaseException:
        task.records_path.unlink(missing_ok=True)
        raise


def _run_years(
    years: tuple[int, ...],
    raw_root: Path,
    cohort: CohortReference,
    member_root: Path,
    *,
    progress_interval: int,
    workers: int,
) -> RiichiPointAnalysis:
    """Run annual scans and return results in deterministic year order."""
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if type(progress_interval) is not int or progress_interval < 1:
        raise ValueError("progress_interval must be a positive integer")
    tasks = tuple(
        _YearTask(
            year=year,
            raw_root=raw_root,
            cohort=cohort,
            records_path=member_root / f"{year}.jsonl.gz",
            progress_interval=progress_interval,
        )
        for year in years
    )
    workers_used = min(workers, len(tasks))
    print(
        f"processing {len(tasks)} year(s) with {workers_used} worker(s)",
        flush=True,
    )
    results: dict[int, YearlyRiichiPointResult] = {}
    if workers_used == 1:
        for task in tasks:
            result = _run_year(task)
            results[result.year] = result
            print(
                f"{result.year}: complete "
                f"({result.scanned_files:,} files, "
                f"{result.established_riichis:,} records)",
                flush=True,
            )
    else:
        process_context = multiprocessing.get_context("spawn")
        cancellation_event = process_context.Event()
        executor = ProcessPoolExecutor(
            max_workers=workers_used,
            mp_context=process_context,
            initializer=_initialize_worker,
            initargs=(cancellation_event,),
        )
        futures = {}
        try:
            futures = {executor.submit(_run_year, task): task.year for task in tasks}
            for future in as_completed(futures):
                year = futures[future]
                try:
                    result = future.result()
                except BaseException as error:
                    raise RuntimeError(f"annual scan failed for {year}") from error
                results[result.year] = result
                print(
                    f"{result.year}: complete "
                    f"({result.scanned_files:,} files, "
                    f"{result.established_riichis:,} records)",
                    flush=True,
                )
        except BaseException:
            cancellation_event.set()
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
    ordered_results = tuple(results[year] for year in years)
    return RiichiPointAnalysis(
        repository=cohort.repository,
        release_tag=cohort.release_tag,
        cohort_manifest_path=cohort.manifest_path,
        selected_years=years,
        years=ordered_results,
    )


def _reserve_temporary_path(target: Path) -> Path:
    """Reserve a same-directory path suitable for an atomic replacement."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        return Path(file.name)


def _combine_gzip_members(member_paths: Sequence[Path], output: Path) -> None:
    """Concatenate complete annual gzip members in the supplied order."""
    with output.open("wb") as destination:
        for member_path in member_paths:
            with member_path.open("rb") as source:
                shutil.copyfileobj(source, destination)


def _validate_combined_records(path: Path, expected_records: int) -> None:
    """Read every concatenated member and validate the final line count."""
    record_count = 0
    with gzip.open(path, "rb") as records:
        for line in records:
            if not line.endswith(b"\n"):
                raise RuntimeError("audit record is not newline terminated")
            record_count += 1
    if record_count != expected_records:
        raise RuntimeError(
            "combined audit record count mismatch: "
            f"expected {expected_records}, found {record_count}"
        )


def _stage_outputs(
    analysis: RiichiPointAnalysis,
    member_paths: Sequence[Path],
    json_target: Path,
    markdown_target: Path,
    records_target: Path,
) -> tuple[Path, Path, Path]:
    """Stage and validate every artifact without touching final targets."""
    temporaries: list[Path] = []
    try:
        json_temporary = _reserve_temporary_path(json_target)
        temporaries.append(json_temporary)
        markdown_temporary = _reserve_temporary_path(markdown_target)
        temporaries.append(markdown_temporary)
        records_temporary = _reserve_temporary_path(records_target)
        temporaries.append(records_temporary)
        write_outputs(analysis, json_temporary, markdown_temporary)
        _combine_gzip_members(member_paths, records_temporary)
        expected_records = sum(year.established_riichis for year in analysis.years)
        _validate_combined_records(records_temporary, expected_records)
    except BaseException:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)
        raise
    return json_temporary, markdown_temporary, records_temporary


def _publish_staged_outputs(
    staged: tuple[Path, Path, Path],
    targets: tuple[Path, Path, Path],
) -> None:
    """Replace the output set, restoring every prior target on failure."""
    backups: list[Path | None] = []
    moved_originals: set[int] = set()
    published: set[int] = set()
    retained_backups: set[Path] = set()
    try:
        for target in targets:
            backups.append(_reserve_temporary_path(target) if target.exists() else None)
        for index, (target, backup) in enumerate(zip(targets, backups, strict=True)):
            if backup is not None:
                os.replace(target, backup)
                moved_originals.add(index)
        for index, (temporary, target) in enumerate(zip(staged, targets, strict=True)):
            os.replace(temporary, target)
            published.add(index)
    except BaseException as publish_error:
        rollback_errors: list[BaseException] = []
        for index in reversed(range(len(targets))):
            target = targets[index]
            backup = backups[index] if index < len(backups) else None
            try:
                if index in moved_originals:
                    assert backup is not None
                    os.replace(backup, target)
                elif index in published:
                    target.unlink(missing_ok=True)
            except OSError as rollback_error:
                rollback_errors.append(rollback_error)
                if index in moved_originals and backup is not None:
                    retained_backups.add(backup)
        if rollback_errors:
            retained_text = ", ".join(str(path) for path in sorted(retained_backups))
            raise RuntimeError(
                "output publication failed and rollback could not restore "
                f"{len(rollback_errors)} target(s); retained backups: "
                f"{retained_text or 'none'}"
            ) from publish_error
        raise
    finally:
        disposable_backups = (
            path
            for path in backups
            if path is not None and path not in retained_backups
        )
        for temporary in (*staged, *disposable_backups):
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def main(argv: Sequence[str] | None = None) -> int:
    """Run a complete validated aggregation and write all outputs."""
    args = parse_args(argv)
    years = selected_years_from_args(args)
    cohort = load_cohort_reference(
        args.cohort_manifest,
        years,
        logical_path=COHORT_MANIFEST_LOGICAL_PATH,
    )

    outputs = (
        args.output_json.resolve(),
        args.output_markdown.resolve(),
        args.output_records.resolve(),
    )
    if len(set(outputs)) != len(outputs):
        raise ValueError("JSON, Markdown, and records output paths must differ")

    json_target = args.output_json.resolve()
    markdown_target = args.output_markdown.resolve()
    records_target = args.output_records.resolve()
    records_target.parent.mkdir(parents=True, exist_ok=True)
    staged: tuple[Path, Path, Path] | None = None
    try:
        with tempfile.TemporaryDirectory(
            dir=records_target.parent,
            prefix=f".{records_target.name}.years.",
        ) as member_directory:
            member_root = Path(member_directory)
            analysis = _run_years(
                years,
                args.raw_root,
                cohort,
                member_root,
                progress_interval=args.progress_interval,
                workers=args.workers,
            )
            member_paths = tuple(member_root / f"{year}.jsonl.gz" for year in years)
            staged = _stage_outputs(
                analysis,
                member_paths,
                json_target,
                markdown_target,
                records_target,
            )
        _publish_staged_outputs(
            staged,
            (json_target, markdown_target, records_target),
        )
        staged = None
    finally:
        if staged is not None:
            for temporary in staged:
                temporary.unlink(missing_ok=True)
    record_count = sum(year.established_riichis for year in analysis.years)
    print(f"wrote: {args.output_json}", flush=True)
    print(f"wrote: {args.output_markdown}", flush=True)
    print(f"wrote: {args.output_records} ({record_count:,} records)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
