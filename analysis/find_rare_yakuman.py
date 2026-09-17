"""Search all local MJAI games for a supported rare yakuman."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from mahjong_analysis.rare_yakuman import (
    DETECTORS,
    result_to_dict,
    scan_rare_yakuman,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the deliberately small rare-yakuman search CLI."""
    parser = argparse.ArgumentParser(
        description="Search complete local MJAI games for a supported rare yakuman."
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--yaku",
        required=True,
        help=f"implemented yaku name ({', '.join(DETECTORS)})",
    )
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument(
        "--continue-on-replay-error",
        action="store_true",
        help="record kyoku replay validation errors as anomalies and continue",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the search and write one compact JSON document to stdout."""
    args = parse_args(argv)
    try:
        result = scan_rare_yakuman(
            args.raw_root,
            yaku=args.yaku,
            workers=args.workers,
            continue_on_replay_error=args.continue_on_replay_error,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    document = json.dumps(
        result_to_dict(result),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    encoded = f"{document}\n".encode("utf-8")
    stdout_buffer = getattr(sys.stdout, "buffer", None)
    if stdout_buffer is None:
        sys.stdout.write(encoded.decode("utf-8"))
    else:
        stdout_buffer.write(encoded)
        stdout_buffer.flush()
    return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
