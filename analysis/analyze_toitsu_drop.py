"""Analyze wait quality after a pair-drop-looking river pattern."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from mahjong_analysis.riichi_wait_dataset import validate_dataset_integrity
from mahjong_analysis.toitsu_drop_analysis import (
    AnnualToitsuDropTask,
    run_toitsu_drop_tasks,
    toitsu_drop_results_document,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_PATHS = (
    "analysis/analyze_toitsu_drop.py",
    "src/mahjong_analysis/toitsu_drop_analysis.py",
    "src/mahjong_analysis/riichi_wait_quality.py",
    "src/mahjong_analysis/riichi_wait_dataset.py",
    "src/mahjong_analysis/tiles.py",
    "docs/specs/toitsu-drop-analysis.md",
    "docs/specs/riichi-wait-quality.md",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_hashes() -> dict[str, str]:
    return {name: _sha256(PROJECT_ROOT / name) for name in CODE_PATHS}


def _quality_slice(value: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "record_count",
        "contains_ryanmen",
        "contains_nobetan",
        "contains_suji",
        "good_wait",
        "suji_origin_counts",
        "suji_good_cross_counts",
        "wait_copy_distribution",
    )
    missing = tuple(field for field in fields if field not in value)
    if missing:
        raise ValueError(f"quality baseline slice is missing fields: {missing}")
    return {field: value[field] for field in fields}


def check_quality_baseline(document: dict, baseline: dict) -> None:
    """Require identical all-record quality counts at every saved slice."""
    if document["overall"]["groups"]["all"] != _quality_slice(baseline["overall"]):
        raise ValueError("overall quality totals disagree with baseline")

    old_turns = {row["riichi_discard_number"]: row for row in baseline["by_turn"]}
    new_turns = {row["riichi_discard_number"]: row for row in document["by_turn"]}
    if set(old_turns) != set(new_turns):
        raise ValueError("turn coverage differs from quality baseline")
    for turn, new in new_turns.items():
        if new["groups"]["all"] != _quality_slice(old_turns[turn]):
            raise ValueError(f"turn {turn} quality totals disagree with baseline")

    old_years = {row["year"]: row for row in baseline["years"]}
    new_years = {row["year"]: row for row in document["years"]}
    if set(old_years) != set(new_years):
        raise ValueError("year coverage differs from quality baseline")
    for year, new in new_years.items():
        old = old_years[year]
        if new["overall"]["groups"]["all"] != _quality_slice(old):
            raise ValueError(f"year {year} quality totals disagree with baseline")
        old_year_turns = {row["riichi_discard_number"]: row for row in old["by_turn"]}
        new_year_turns = {row["riichi_discard_number"]: row for row in new["by_turn"]}
        if set(old_year_turns) != set(new_year_turns):
            raise ValueError(f"year {year} turn coverage differs from baseline")
        for turn, new_turn in new_year_turns.items():
            if new_turn["groups"]["all"] != _quality_slice(old_year_turns[turn]):
                raise ValueError(
                    f"year {year} turn {turn} quality totals disagree with baseline"
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data/processed/riichi-waits-v1",
    )
    parser.add_argument(
        "--quality-baseline",
        type=Path,
        default=PROJECT_ROOT / "research/results/riichi-wait-quality-v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "research/results/toitsu-drop-analysis-v1.json",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.output.exists():
        raise FileExistsError(f"will not overwrite {args.output}")

    started = time.monotonic()
    code_hashes = _code_hashes()
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    clean = not subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    manifest_path = args.dataset_root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    print("Checking canonical manifest, annual sizes and SHA256...", file=sys.stderr)
    manifest = validate_dataset_integrity(args.dataset_root, expected_mode="full")
    if manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("manifest changed during validation")
    years = tuple(entry["year"] for entry in manifest["years"])
    if years != tuple(range(2009, 2026)) or (
        manifest["totals"]["output_records"] != 10_706_714
        or manifest["source"]["repository"] != "NikkeTryHard/tenhou-to-mjai"
    ):
        raise ValueError("unexpected canonical dataset")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()

    baseline_bytes = args.quality_baseline.read_bytes()
    baseline = json.loads(baseline_bytes)
    if baseline.get("metadata", {}).get("input_manifest_sha256") != manifest_hash:
        raise ValueError("quality baseline and analysis use different manifests")
    if baseline.get("overall", {}).get("record_count") != 10_706_714:
        raise ValueError("unexpected quality baseline total")

    def report(result) -> None:
        pattern_count = result.overall.quality_for(
            ("a_only", "b_only", "both")
        ).record_count
        print(
            f"year={result.year} records={result.overall.record_count:,} "
            f"pattern={pattern_count:,} elapsed={time.monotonic() - started:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    results = run_toitsu_drop_tasks(
        tuple(
            AnnualToitsuDropTask(
                entry["year"],
                str((args.dataset_root / entry["output_filename"]).resolve()),
                entry["output_records"],
            )
            for entry in manifest["years"]
        ),
        workers=args.workers,
        on_result=report,
    )
    document = toitsu_drop_results_document(results)
    if document["overall"]["record_count"] != 10_706_714:
        raise ValueError("canonical total differs from 10,706,714")
    check_quality_baseline(document, baseline)
    if (
        _code_hashes() != code_hashes
        or manifest_path.read_bytes() != manifest_bytes
        or args.quality_baseline.read_bytes() != baseline_bytes
    ):
        raise ValueError("code or validated input changed during aggregation")

    document["metadata"] = {
        "analysis_name": "toitsu-drop-analysis-v1",
        "schema_version": 1,
        "observation_unit": "established_riichi_record",
        "input_manifest_sha256": manifest_hash,
        "quality_baseline_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
        "dataset_generator_git_commit": manifest["generator"]["git_commit"],
        "source": dict(manifest["source"]),
        "scope": dict(manifest["scope"]),
        "analysis_generator": {
            "base_git_commit": git_commit,
            "worktree_clean": clean,
            "source_sha256": code_hashes,
        },
        "definitions": {
            "pattern": "adjacent own discards with equal normalized tile and second tsumogiri=false",
            "A": "tedashi then tedashi",
            "B": "tsumogiri then tedashi",
            "classification": "exclusive none/a_only/b_only/both using all matching pairs",
            "representative_pair": "last completed matching pair",
            "distance": "riichi_discard_number minus representative second discard_number",
            "contains_suji_and_good_wait": "delegated to riichi_wait_quality.evaluate_wait_quality",
            "call_metadata_used_for_main_pattern": False,
        },
        "validation": {
            "annual_size_sha256": "PASS",
            "streaming_dto_validation": "PASS",
            "annual_and_total_counts": "PASS",
            "classification_partitions": "PASS",
            "year_turn_distance_partitions": "PASS",
            "quality_baseline_all_slices": "PASS",
        },
    }
    content = (
        json.dumps(
            document, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as file:
        file.write(content)
    print(f"Saved {args.output}; elapsed={time.monotonic() - started:.1f}s")
    print(json.dumps(document["overall"], ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
