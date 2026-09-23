"""Analyze established-riichi wait strength by declaration discard tile."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from mahjong_analysis.riichi_declaration_tile_analysis import (
    AnnualDeclarationTileTask,
    declaration_tile_results_document,
    run_declaration_tile_tasks,
)
from mahjong_analysis.riichi_wait_dataset import validate_dataset_integrity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_YEARS = tuple(range(2020, 2026))
CODE_PATHS = (
    "analysis/analyze_riichi_declaration_tiles.py",
    "src/mahjong_analysis/riichi_declaration_tile_analysis.py",
    "src/mahjong_analysis/riichi_wait_quality.py",
    "src/mahjong_analysis/riichi_wait_dataset.py",
    "src/mahjong_analysis/tiles.py",
    "docs/specs/riichi-declaration-tile-analysis.md",
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
        raise ValueError(f"quality slice is missing fields: {missing}")
    return {field: value[field] for field in fields}


def check_quality_baseline(document: dict, baseline: dict) -> None:
    """Require identical annual and annual-by-turn all-record quality slices."""
    old_years = {row["year"]: row for row in baseline["years"]}
    new_years = {row["year"]: row for row in document["years"]}
    if set(new_years) != set(ANALYSIS_YEARS):
        raise ValueError("analysis year coverage differs from specification")
    if not set(new_years) <= set(old_years):
        raise ValueError("analysis years are absent from quality baseline")
    for year, new in new_years.items():
        old = old_years[year]
        if _quality_slice(new["overall"]["all"]) != _quality_slice(old):
            raise ValueError(f"year {year} quality totals disagree with baseline")
        old_turns = {row["riichi_discard_number"]: row for row in old["by_turn"]}
        new_turns = {row["riichi_discard_number"]: row for row in new["by_turn"]}
        if set(old_turns) != set(new_turns):
            raise ValueError(f"year {year} turn coverage differs from baseline")
        for turn, new_turn in new_turns.items():
            if _quality_slice(new_turn["all"]) != _quality_slice(old_turns[turn]):
                raise ValueError(
                    f"year {year} turn {turn} quality totals disagree with baseline"
                )


def check_strong_results_unchanged(document: dict, previous: dict) -> None:
    """Compare every saved strong-wait slice with the pre-definition result."""

    def check_accumulator(new: dict, old: dict, context: str) -> None:
        if new["record_count"] != old["record_count"]:
            raise ValueError(f"{context}: record count changed")
        for name in ("all", "non_dora"):
            if new[name]["strong_wait"] != old[name]["strong_wait"]:
                raise ValueError(f"{context}/{name}: strong wait changed")
        new_categories = {row["declaration_category"]: row for row in new["categories"]}
        old_categories = {row["declaration_category"]: row for row in old["categories"]}
        if set(new_categories) != set(old_categories):
            raise ValueError(f"{context}: declaration categories changed")
        for category in new_categories:
            if (
                new_categories[category]["record_count"]
                != old_categories[category]["record_count"]
                or new_categories[category]["strong_wait"]
                != old_categories[category]["strong_wait"]
            ):
                raise ValueError(f"{context}/{category}: strong wait changed")

    check_accumulator(document["overall"], previous["overall"], "overall")
    new_turns = {row["riichi_discard_number"]: row for row in document["by_turn"]}
    old_turns = {row["riichi_discard_number"]: row for row in previous["by_turn"]}
    if set(new_turns) != set(old_turns):
        raise ValueError("overall exact-turn coverage changed")
    for turn, new_turn in new_turns.items():
        check_accumulator(new_turn, old_turns[turn], f"turn {turn}")

    new_years = {row["year"]: row for row in document["years"]}
    old_years = {row["year"]: row for row in previous["years"]}
    if set(new_years) != set(old_years):
        raise ValueError("year coverage changed")
    for year in new_years:
        check_accumulator(
            new_years[year]["overall"],
            old_years[year]["overall"],
            f"year {year}",
        )
        new_year_turns = {
            row["riichi_discard_number"]: row for row in new_years[year]["by_turn"]
        }
        old_year_turns = {
            row["riichi_discard_number"]: row for row in old_years[year]["by_turn"]
        }
        if set(new_year_turns) != set(old_year_turns):
            raise ValueError(f"year {year}: exact-turn coverage changed")
        for turn, new_year_turn in new_year_turns.items():
            check_accumulator(
                new_year_turn,
                old_year_turns[turn],
                f"year {year}/turn {turn}",
            )


def _combined_baseline_quality(baseline: dict) -> dict[str, Any]:
    """Merge integer baseline slices for the selected years independently."""
    rows = [row for row in baseline["years"] if row["year"] in ANALYSIS_YEARS]
    record_count = sum(row["record_count"] for row in rows)

    def metric(name: str) -> dict[str, Any]:
        count = sum(row[name]["count"] for row in rows)
        return {
            "count": count,
            "denominator": record_count,
            "rate": count / record_count if record_count else None,
        }

    origins = Counter()
    cross = Counter()
    copies = Counter()
    for row in rows:
        origins.update(row["suji_origin_counts"])
        cross.update(row["suji_good_cross_counts"])
        copies.update(
            {
                item["wait_copies"]: item["count"]
                for item in row["wait_copy_distribution"]
            }
        )
    return {
        "record_count": record_count,
        "contains_ryanmen": metric("contains_ryanmen"),
        "contains_nobetan": metric("contains_nobetan"),
        "contains_suji": metric("contains_suji"),
        "good_wait": metric("good_wait"),
        "suji_origin_counts": dict(origins),
        "suji_good_cross_counts": dict(cross),
        "wait_copy_distribution": [
            {"wait_copies": value, "count": copies[value]} for value in sorted(copies)
        ],
    }


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
        default=PROJECT_ROOT
        / "research/results/riichi-declaration-tile-analysis-v1.json",
    )
    parser.add_argument(
        "--previous-result",
        type=Path,
        help="pre-definition result used to prove strong-wait slices are unchanged",
    )
    parser.add_argument("--workers", type=int, default=6)
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
    if (
        tuple(entry["year"] for entry in manifest["years"]) != tuple(range(2009, 2026))
        or manifest["totals"]["output_records"] != 10_706_714
        or manifest["source"]["repository"] != "NikkeTryHard/tenhou-to-mjai"
    ):
        raise ValueError("unexpected canonical dataset")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()

    baseline_bytes = args.quality_baseline.read_bytes()
    baseline = json.loads(baseline_bytes)
    if baseline.get("metadata", {}).get("input_manifest_sha256") != manifest_hash:
        raise ValueError("quality baseline and analysis use different manifests")
    previous_bytes = args.previous_result.read_bytes() if args.previous_result else None
    previous = json.loads(previous_bytes) if previous_bytes is not None else None
    if previous is not None and (
        previous.get("metadata", {}).get("analysis_name")
        != "riichi-declaration-tile-analysis-v1"
        or previous.get("metadata", {}).get("input_manifest_sha256") != manifest_hash
    ):
        raise ValueError("previous result is not the matching declaration analysis")

    selected_entries = tuple(
        entry for entry in manifest["years"] if entry["year"] in ANALYSIS_YEARS
    )

    def report(result) -> None:
        categories = {
            category: accumulator.record_count
            for category, accumulator in result.overall.by_category.items()
        }
        print(
            f"year={result.year} records={result.overall.record_count:,} "
            f"dora={categories['dora']:,} elapsed={time.monotonic() - started:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    results = run_declaration_tile_tasks(
        tuple(
            AnnualDeclarationTileTask(
                entry["year"],
                str((args.dataset_root / entry["output_filename"]).resolve()),
                entry["output_records"],
            )
            for entry in selected_entries
        ),
        workers=args.workers,
        on_result=report,
    )
    document = declaration_tile_results_document(results)
    expected_records = sum(entry["output_records"] for entry in selected_entries)
    if document["overall"]["record_count"] != expected_records:
        raise ValueError("selected-period record total differs from manifest")
    check_quality_baseline(document, baseline)
    if _quality_slice(document["overall"]["all"]) != _combined_baseline_quality(
        baseline
    ):
        raise ValueError("selected-period overall quality disagrees with baseline")
    if previous is not None:
        check_strong_results_unchanged(document, previous)
    if (
        _code_hashes() != code_hashes
        or manifest_path.read_bytes() != manifest_bytes
        or args.quality_baseline.read_bytes() != baseline_bytes
        or (
            args.previous_result is not None
            and args.previous_result.read_bytes() != previous_bytes
        )
    ):
        raise ValueError("code or validated input changed during aggregation")

    document["metadata"] = {
        "analysis_name": "riichi-declaration-tile-analysis-v1",
        "schema_version": 3,
        "observation_unit": "established_riichi_record",
        "analysis_years": list(ANALYSIS_YEARS),
        "input_manifest_sha256": manifest_hash,
        "quality_baseline_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
        "previous_result_sha256": (
            hashlib.sha256(previous_bytes).hexdigest()
            if previous_bytes is not None
            else None
        ),
        "dataset_generator_git_commit": manifest["generator"]["git_commit"],
        "source": dict(manifest["source"]),
        "scope": {**manifest["scope"], "years": list(ANALYSIS_YEARS)},
        "analysis_generator": {
            "base_git_commit": git_commit,
            "worktree_clean": clean,
            "source_sha256": code_hashes,
        },
        "definitions": {
            "category_order": ["19", "28", "37", "46", "5", "honor", "dora"],
            "dora_override": "red five or initial visible dora from start_kyoku.dora_marker",
            "kan_dora_included": False,
            "strong_wait": "contains_suji OR contains_ryanmen OR self-excluded formal wait copies >= 5; evaluated first",
            "weak_wait": "not strong, copies <= 4, and every formal interpretation is tanki/kanchan/penchan/shanpon",
            "other_wait": "neither strong nor weak (including kokushi single)",
            "wait_shape_record_counts": "record-level flags; non-exclusive when multiple wait shapes apply",
            "same_suji_line_wait": "at least one formal wait tile in the declaration tile's suit with rank difference 3 or 6",
            "weak_same_suji_line_wait_shape_record_counts": "record-level flags among weak_and_same_suji_line; non-exclusive",
            "non_dora_suited_rank_breakdown": "exclusive declaration-rank 1-9 rows for non-dora suited declarations only",
            "rank_wait_pairs": "18 declaration-rank/wait-rank pairs in the same suit at rank difference 3 or 6; pair rows are non-exclusive",
            "rank_wait_pair_weak_shapes": "record-level shape flags for matching wait-rank interpretations within weak records; non-exclusive",
        },
        "validation": {
            "annual_size_sha256": "PASS",
            "streaming_dto_validation": "PASS",
            "annual_and_total_counts": "PASS",
            "declaration_category_partition": "PASS",
            "strong_weak_other_partition": "PASS",
            "same_suji_line_weak_partition": "PASS",
            "non_dora_suited_rank_category_reconciliation": "PASS",
            "rank_wait_pair_nonexclusive_semantics": "PASS",
            "quality_baseline_annual_and_turn_slices": "PASS",
            "quality_baseline_selected_period": "PASS",
            "strong_wait_unchanged_from_previous_result": (
                "PASS" if previous is not None else "NOT_RUN"
            ),
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
