"""Export the versioned established-riichi wait dataset."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from mahjong_analysis.riichi_wait_export import (
    SUPPORTED_YEARS,
    ProgressUpdate,
    collect_git_metadata,
    export_riichi_wait_dataset,
    normalize_years,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "riichi-waits-v1"
DEFAULT_VALIDATION_SUMMARY = (
    PROJECT_ROOT / "data" / "validation" / "tenhou-to-mjai-v2.0.0-summary.json"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse an explicit year selection and failure-safety options."""
    parser = argparse.ArgumentParser(
        description="Export one gzip JSONL file per year of established riichis."
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--dataset-summary",
        "--validation-summary",
        dest="dataset_summary",
        type=Path,
        default=DEFAULT_VALIDATION_SUMMARY,
    )
    parser.add_argument(
        "--dataset-summary-logical-path",
        help=(
            "relative POSIX identifier required when --dataset-summary is outside "
            "the project root"
        ),
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--year", type=int)
    selection.add_argument("--years", nargs="+", type=int)
    selection.add_argument("--all", action="store_true")
    parser.add_argument(
        "--max-files",
        type=_positive_int,
        help="sample only: sorted input count before target filtering",
    )
    replacement = parser.add_mutually_exclusive_group()
    replacement.add_argument("--force", action="store_true")
    replacement.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--progress-interval", type=_positive_int, default=10_000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the streaming export, propagating every processing failure."""
    args = parse_args(argv)
    years = _selected_years(args)
    if args.max_files is not None and len(years) != 1:
        raise ValueError("--max-files requires exactly one selected year")
    output_root = args.output_root
    if output_root is None:
        output_root = (
            DEFAULT_OUTPUT_ROOT
            if args.max_files is None
            else DEFAULT_OUTPUT_ROOT.with_name(
                f"{DEFAULT_OUTPUT_ROOT.name}-sample-{args.max_files}"
            )
        )
    git_metadata = collect_git_metadata(PROJECT_ROOT)
    summary_logical_path = _summary_logical_path(
        args.dataset_summary,
        args.dataset_summary_logical_path,
    )
    manifest = export_riichi_wait_dataset(
        raw_root=args.raw_root,
        output_root=output_root,
        years=years,
        validation_summary_path=args.dataset_summary,
        validation_summary_logical_path=summary_logical_path,
        git_metadata=git_metadata,
        max_files=args.max_files,
        force=args.force,
        resume=args.resume,
        allow_dirty=args.allow_dirty,
        progress_interval=args.progress_interval,
        progress_callback=_print_progress,
    )
    for year in manifest["years"]:
        print(
            "year complete: "
            f"year={year['year']} files={year['scanned_files']} "
            f"target_games={year['target_games']} "
            f"east_kyokus={year['east_kyokus']} "
            f"records={year['output_records']} "
            f"elapsed={year['elapsed_seconds']:.1f}s"
        )
    totals = manifest["totals"]
    print(
        "complete: "
        f"years={','.join(str(year) for year in years)} "
        f"files={totals['scanned_files']} "
        f"target_games={totals['target_games']} "
        f"east_kyokus={totals['east_kyokus']} "
        f"records={totals['output_records']}"
    )
    return 0


def _selected_years(args: argparse.Namespace) -> tuple[int, ...]:
    if args.all:
        return SUPPORTED_YEARS
    if args.year is not None:
        return normalize_years((args.year,))
    return normalize_years(args.years)


def _print_progress(update: ProgressUpdate) -> None:
    elapsed = update.elapsed_seconds
    rate = update.processed_files / elapsed if elapsed > 0 else 0.0
    print(
        f"year={update.year} "
        f"files={update.processed_files}/{update.total_files} "
        f"target_games={update.target_games} "
        f"east_kyokus={update.east_kyokus} "
        f"records={update.records} "
        f"elapsed={elapsed:.1f}s files_per_second={rate:.1f}",
        file=sys.stderr,
    )


def _summary_logical_path(path: Path, explicit: str | None) -> str:
    if explicit is not None:
        if (
            not explicit
            or "\\" in explicit
            or Path(explicit).is_absolute()
            or ".." in explicit.split("/")
        ):
            raise ValueError(
                "--dataset-summary-logical-path must be relative POSIX text"
            )
        return explicit
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(
            "a dataset summary outside the project root requires "
            "--dataset-summary-logical-path"
        ) from error


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
