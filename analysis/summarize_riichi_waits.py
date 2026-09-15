"""Generate deterministic baseline summaries for established-riichi waits."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from mahjong_analysis.riichi_wait_summary import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUTPUT_ROOT,
    SummaryProgress,
    collect_analysis_git_metadata,
    summarize_riichi_wait_dataset,
    write_summary_outputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse canonical summary input and output paths."""
    parser = argparse.ArgumentParser(
        description="Summarize the canonical established-riichi wait dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_DATASET_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument("--progress-interval", type=_positive_int, default=100_000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the canonical one-pass aggregation and publish all result files."""
    args = parse_args(argv)
    git_metadata = collect_analysis_git_metadata(PROJECT_ROOT)
    analysis = summarize_riichi_wait_dataset(
        args.dataset_root,
        analysis_git=git_metadata,
        project_root=PROJECT_ROOT,
        workers=args.workers,
        progress_interval=args.progress_interval,
        progress_callback=_print_progress,
    )
    paths = write_summary_outputs(analysis, args.output_root)
    for path in paths:
        print(path.resolve())
    return 0


def _print_progress(update: SummaryProgress) -> None:
    print(
        f"year={update.year} annual_records={update.annual_records} "
        f"total_records={update.total_records} elapsed={update.elapsed_seconds:.1f}s",
        file=sys.stderr,
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
