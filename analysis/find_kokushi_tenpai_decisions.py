"""Extract first kokushi-tenpai decisions for riichi-versus-dama analysis."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from mahjong_analysis.kokushi_riichi import SUPPORTED_YEARS
from mahjong_analysis.kokushi_tenpai_decisions import (
    KokushiTenpaiYearSummary,
    result_to_dict,
    scan_kokushi_tenpai_decisions,
    write_result_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_ROOT = PROJECT_ROOT / "data" / "archives" / "v1.2.0-db"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "kokushi-tenpai-decisions-2009-2025.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the first post-discard kokushi tenpai for every actor/kyoku, "
            "classify immediate riichi, delayed riichi, or dama, and follow outcomes."
        )
    )
    parser.add_argument("--db-root", type=Path, default=DEFAULT_DB_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--years", type=_supported_year, nargs="+", default=list(SUPPORTED_YEARS)
    )
    parser.add_argument("--max-logs-per-year", type=_positive_int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = scan_kokushi_tenpai_decisions(
            args.db_root,
            years=args.years,
            workers=args.workers,
            max_logs_per_year=args.max_logs_per_year,
            on_year_complete=_print_progress if args.progress else None,
        )
        write_result_json(result, args.output)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    summary = {
        "output": str(args.output.resolve()),
        "summary": result_to_dict(result)["summary"],
    }
    encoded = (
        json.dumps(
            summary, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
        )
        + "\n"
    ).encode("utf-8")
    stdout = getattr(sys.stdout, "buffer", None)
    if stdout is None:
        sys.stdout.write(encoded.decode("utf-8"))
    else:
        stdout.write(encoded)
        stdout.flush()
    return 0


def _print_progress(summary: KokushiTenpaiYearSummary) -> None:
    print(
        f"{summary.year}: {summary.scanned_logs:,} logs, "
        f"{summary.decisions:,} decisions "
        f"(immediate={summary.immediate_riichi}, delayed={summary.delayed_riichi}, "
        f"dama={summary.dama}, established={summary.established_riichis})",
        file=sys.stderr,
        flush=True,
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _supported_year(value: str) -> int:
    year = int(value)
    if year not in SUPPORTED_YEARS:
        raise argparse.ArgumentTypeError("must be a year from 2009 through 2025")
    return year


if __name__ == "__main__":
    raise SystemExit(main())
