"""Aggregate dealer double-riichi results for selected years."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from mahjong_analysis.yearly_aggregation import (
    SUPPORTED_YEARS,
    aggregate_dealer_double_riichi_years,
    load_dataset_validation_metadata,
    load_known_results,
    normalize_years,
    validate_output_paths,
    write_yearly_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_DATASET_SUMMARY = (
    PROJECT_ROOT / "data" / "validation" / "tenhou-to-mjai-v2.0.0-summary.json"
)
DEFAULT_KNOWN_RESULTS = (
    PROJECT_ROOT
    / "data"
    / "validation"
    / "dealer-double-riichi-v2.0.0-known-results.json"
)
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "outputs" / "dealer-double-riichi" / "yearly.json"
DEFAULT_OUTPUT_MARKDOWN = (
    PROJECT_ROOT / "outputs" / "dealer-double-riichi" / "yearly.md"
)
PROGRESS_INTERVAL = 10_000


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate yearly dealer double-riichi win rates."
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
        "--dataset-summary",
        type=Path,
        default=DEFAULT_DATASET_SUMMARY,
    )
    parser.add_argument(
        "--known-results",
        type=Path,
        default=DEFAULT_KNOWN_RESULTS,
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
    return parser.parse_args(argv)


def selected_years_from_args(args: argparse.Namespace) -> tuple[int, ...]:
    """Resolve and validate the mutually exclusive year selection."""
    years = SUPPORTED_YEARS if args.all else args.years
    return normalize_years(years)


def main(argv: Sequence[str] | None = None) -> int:
    """Aggregate all selected years and write outputs only after success."""
    args = parse_args(argv)
    validate_output_paths(args.output_json, args.output_markdown)
    years = selected_years_from_args(args)
    dataset = load_dataset_validation_metadata(args.dataset_summary, years)
    known_results = load_known_results(args.known_results)

    def report_progress(year: int, processed: int) -> None:
        print(f"{year}: processed {processed:,} files", flush=True)

    summary = aggregate_dealer_double_riichi_years(
        years,
        args.raw_root,
        dataset,
        known_results=known_results,
        progress_interval=PROGRESS_INTERVAL,
        progress_callback=report_progress,
    )
    write_yearly_outputs(summary, args.output_json, args.output_markdown)
    print(f"wrote: {args.output_json}", flush=True)
    print(f"wrote: {args.output_markdown}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
