"""Add kyoku outcomes to the established kokushi-riichi extraction."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from mahjong_analysis.kokushi_riichi_outcomes import (
    enrich_result_document,
    load_result_json,
    write_outcome_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_ROOT = PROJECT_ROOT / "data" / "archives" / "v1.2.0-db"
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "kokushi-riichi-2009-2025.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "kokushi-riichi-outcomes-2009-2025.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enrich extracted kokushi riichis with win/draw outcomes."
    )
    parser.add_argument("--db-root", type=Path, default=DEFAULT_DB_ROOT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = load_result_json(args.input)
    result = enrich_result_document(source, args.db_root)
    write_outcome_json(result, args.output)
    print(
        json.dumps(
            {"output": str(args.output.resolve()), "summary": result["summary"]},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
