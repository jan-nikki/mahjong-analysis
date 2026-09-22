"""Compute the two new descriptive metrics without replacing published baseline v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from mahjong_analysis.riichi_wait_dataset import validate_dataset_integrity
from mahjong_analysis.riichi_wait_quality import (
    AnnualQualityTask,
    quality_results_document,
    run_quality_tasks,
)
from mahjong_analysis.riichi_wait_summary import load_published_summary_paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_PATHS = (
    "analysis/summarize_riichi_wait_quality.py",
    "src/mahjong_analysis/riichi_wait_quality.py",
    "src/mahjong_analysis/riichi_wait_dataset.py",
    "src/mahjong_analysis/riichi_wait_summary.py",
    "src/mahjong_analysis/tiles.py",
    "src/mahjong_analysis/hand_waits.py",
    "docs/specs/riichi-wait-quality.md",
)


def _code_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
        for name in CODE_PATHS
    }


def _check_slice(new: dict, old: dict) -> None:
    if new["record_count"] != old["record_count"] or (
        new["contains_ryanmen"] != old["contains_ryanmen"]
    ):
        raise ValueError("new aggregation disagrees with existing baseline")


def check_baseline(document: dict, baseline: dict[str, dict]) -> None:
    """Compare old integer counts and denominators at every saved granularity."""
    _check_slice(document["overall"], baseline["overall"]["result"]["formal_baseline"])
    old_years = {r["year"]: r for r in baseline["yearly"]["years"]}
    old_turns = {
        r["riichi_discard_number"]: r for r in baseline["by-turn"]["all_years"]
    }
    old_year_turns = {r["year"]: r["turns"] for r in baseline["by-turn"]["by_year"]}
    if set(old_years) != {r["year"] for r in document["years"]}:
        raise ValueError("baseline year coverage differs")
    if set(old_year_turns) != set(old_years):
        raise ValueError("baseline year/turn coverage differs")
    if set(old_turns) != {r["riichi_discard_number"] for r in document["by_turn"]}:
        raise ValueError("baseline turn coverage differs")
    for row in document["by_turn"]:
        _check_slice(row, old_turns[row["riichi_discard_number"]]["formal_baseline"])
    for row in document["years"]:
        _check_slice(row, old_years[row["year"]]["formal_baseline"])
        previous = {t["riichi_discard_number"]: t for t in old_year_turns[row["year"]]}
        if set(previous) != {t["riichi_discard_number"] for t in row["by_turn"]}:
            raise ValueError("baseline annual turn coverage differs")
        for turn in row["by_turn"]:
            _check_slice(
                turn, previous[turn["riichi_discard_number"]]["formal_baseline"]
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data/processed/riichi-waits-v1",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=PROJECT_ROOT / "research/results/riichi-wait-summary-v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "research/results/riichi-wait-quality-v1.json",
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
    baseline_paths = load_published_summary_paths(args.baseline_root)
    baseline = {
        label: json.loads(
            next(
                p for p in baseline_paths if p.name.endswith(f"-{label}.json")
            ).read_text(encoding="utf-8")
        )
        for label in ("overall", "yearly", "by-turn")
    }
    for previous in baseline.values():
        if previous["metadata"]["input_dataset"]["manifest_sha256"] != manifest_hash:
            raise ValueError("baseline and new metrics use different input manifests")

    def report(result) -> None:
        print(
            f"year={result.year} records={result.overall.record_count:,} "
            f"elapsed={time.monotonic() - started:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    results = run_quality_tasks(
        tuple(
            AnnualQualityTask(
                e["year"],
                str((args.dataset_root / e["output_filename"]).resolve()),
                e["output_records"],
            )
            for e in manifest["years"]
        ),
        workers=args.workers,
        on_result=report,
    )
    document = quality_results_document(results)
    if document["overall"]["record_count"] != 10_706_714:
        raise ValueError("canonical total differs from 10,706,714")
    check_baseline(document, baseline)
    if _code_hashes() != code_hashes or manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("code or input manifest changed during aggregation")
    document["metadata"] = {
        "analysis_name": "riichi-wait-quality-v1",
        "schema_version": 1,
        "status": "additional_metrics_not_a_replacement_for_published_baseline",
        "observation_unit": "established_riichi_record",
        "input_manifest_sha256": manifest_hash,
        "dataset_generator_git_commit": manifest["generator"]["git_commit"],
        "source": dict(manifest["source"]),
        "scope": dict(manifest["scope"]),
        "analysis_generator": {
            "base_git_commit": git_commit,
            "worktree_clean": clean,
            "source_sha256": code_hashes,
        },
        "definitions": {
            "contains_suji": "any standard/ryanmen OR two distinct standard/tanki waits of same suit with rank difference 3 or 6",
            "good_wait": "sum(4 - normalized self-owned copies) over unique formal wait tiles >= 5",
            "ownership": "concealed after declaration discard plus all fixed meld tiles",
            "not_subtracted": "own river, other players, dora markers, wall",
            "zero_available_records_retained": True,
        },
        "validation": {
            "annual_size_sha256": "PASS",
            "streaming_dto_validation": "PASS",
            "annual_and_total_counts": "PASS",
            "year_and_turn_partitions": "PASS",
            "baseline_record_and_ryanmen_counts_all_slices": "PASS",
        },
    }
    content = (
        json.dumps(
            document, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # A distinct output, never an overwrite of the established baseline pointer.
    with args.output.open("x", encoding="utf-8", newline="\n") as file:
        file.write(content)
    print(f"Saved {args.output}; elapsed={time.monotonic() - started:.1f}s")
    print(json.dumps(document["overall"], ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
