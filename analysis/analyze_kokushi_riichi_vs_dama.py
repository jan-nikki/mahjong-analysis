"""Compare immediate kokushi riichi with initially staying dama."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mahjong_analysis.kokushi_riichi_comparison import analyze_decision_document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "kokushi-tenpai-decisions-2009-2025.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "kokushi-riichi-vs-dama-analysis.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    document = json.loads(args.input.read_text(encoding="utf-8"))
    result = analyze_decision_document(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.name}.part")
    temporary.write_text(
        json.dumps(
            result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "primary": result["primary_immediate_vs_initial_dama"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
