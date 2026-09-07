"""Analyze uncertainty and heterogeneity in reviewed yearly results."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from mahjong_analysis.yearly_inference import (
    analyze_yearly_file,
    write_inference_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT / "research" / "results" / "dealer-double-riichi-v2.0.0-yearly.json"
)
DEFAULT_OUTPUT_JSON = (
    PROJECT_ROOT / "outputs" / "dealer-double-riichi" / "yearly-inference.json"
)
DEFAULT_OUTPUT_MARKDOWN = (
    PROJECT_ROOT / "outputs" / "dealer-double-riichi" / "yearly-inference.md"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse source and output path overrides."""
    parser = argparse.ArgumentParser(
        description=(
            "Calculate Wilson intervals and yearly Pearson heterogeneity "
            "from reviewed dealer double-riichi results."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run inference using only the reviewed aggregate JSON."""
    args = parse_args(argv)
    summary = analyze_yearly_file(args.input)
    write_inference_outputs(
        summary,
        args.output_json,
        args.output_markdown,
    )
    print(f"wrote: {args.output_json}", flush=True)
    print(f"wrote: {args.output_markdown}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
