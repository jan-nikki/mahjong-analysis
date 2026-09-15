"""Benchmark annual-process riichi-wait aggregation without publishing results."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from mahjong_analysis.riichi_wait_summary import (
    DEFAULT_DATASET_ROOT,
    SummaryBenchmarkResult,
    benchmark_riichi_wait_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YEARS = tuple(range(2018, 2026))
DEFAULT_WORKERS = (1, 2, 4, 6)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse one fixed-sample benchmark configuration."""
    parser = argparse.ArgumentParser(
        description="Benchmark summary aggregation without publishing artifacts."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_DATASET_ROOT,
    )
    parser.add_argument("--years", nargs="+", type=int, default=list(DEFAULT_YEARS))
    parser.add_argument(
        "--record-limit-per-year",
        type=_positive_int,
        default=100_000,
    )
    parser.add_argument(
        "--workers",
        nargs="+",
        type=_positive_int,
        default=list(DEFAULT_WORKERS),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark suite and print timing diagnostics."""
    args = parse_args(argv)
    results = benchmark_riichi_wait_dataset(
        args.dataset_root,
        years=tuple(args.years),
        record_limit_per_year=args.record_limit_per_year,
        worker_counts=tuple(args.workers),
    )
    for result in results:
        _print_result(result)
    print(
        "Caution: filesystem cache and worker startup can affect sequential timings; "
        "compare results as local diagnostics."
    )
    return 0


def _print_result(result: SummaryBenchmarkResult) -> None:
    years = ",".join(str(year) for year in result.years)
    print(
        f"workers_requested={result.workers_requested} "
        f"workers_used={result.workers_used} years={years} "
        f"record_limit_per_year={result.record_limit_per_year} "
        f"records_processed={result.records_processed} "
        f"elapsed_seconds={result.elapsed_seconds:.6f} "
        f"records_per_second={result.records_per_second:.2f}"
    )
    for annual in result.year_results:
        print(
            f"  year={annual.year} records_processed={annual.records_processed} "
            f"elapsed_seconds={annual.elapsed_seconds:.6f}"
        )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
