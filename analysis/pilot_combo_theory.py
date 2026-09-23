"""Run a bounded 2020-2023 pilot of post-riichi combo extraction."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

from mahjong_analysis.mjai import is_target_game, load_mjai, split_kyoku
from mahjong_analysis.post_riichi_decisions import (
    PostRiichiDrawDecision,
    extract_post_riichi_draw_decisions,
)
from mahjong_analysis.unique_tenpai_states import (
    CandidateWaitStateCount,
    bounded_count_vector_count,
    count_candidate_wait_states,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw"
DEVELOPMENT_YEARS = tuple(range(2020, 2024))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pilot combo extraction without touching the 2025 holdout."
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--year", type=int, choices=DEVELOPMENT_YEARS, default=2020)
    parser.add_argument("--max-files", type=_positive_int, default=20)
    parser.add_argument("--audit-samples", type=_nonnegative_int, default=10)
    parser.add_argument("--exact-decisions", type=_nonnegative_int, default=10)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def collect_pilot(
    paths: Sequence[Path],
    *,
    raw_root: Path,
    max_files: int,
    audit_samples: int,
    exact_decisions: int,
) -> dict[str, Any]:
    """Extract deterministic bounded counts and source-locatable audit rows."""
    selected = tuple(sorted(paths, key=lambda path: path.as_posix())[:max_files])
    counts: Counter[str] = Counter()
    turn_bins: Counter[str] = Counter()
    exact_space_sizes: list[int] = []
    audits: list[dict[str, Any]] = []
    exact_counts: Counter[str] = Counter()
    inversion_samples: list[dict[str, Any]] = []

    for path in selected:
        counts["scanned_files"] += 1
        events = load_mjai(path)
        if not is_target_game(path, events):
            continue
        counts["target_games"] += 1
        for kyoku_index, kyoku in enumerate(split_kyoku(events)):
            if kyoku[0].get("bakaze") != "E":
                continue
            counts["east_kyokus"] += 1
            decisions = extract_post_riichi_draw_decisions(kyoku)
            for decision in decisions:
                _accumulate_decision(counts, turn_bins, decision)
                state_space_size = bounded_count_vector_count(
                    decision.unseen_counts,
                    decision.target_concealed_tile_count,
                )
                exact_space_sizes.append(state_space_size)
                if len(audits) < audit_samples:
                    audits.append(
                        _audit_row(
                            path,
                            raw_root,
                            kyoku_index,
                            decision,
                            state_space_size,
                        )
                    )
                if exact_counts["decisions"] < exact_decisions:
                    exact_values = _exact_candidate_counts(decision)
                    _accumulate_exact_ranking(
                        exact_counts,
                        inversion_samples,
                        path,
                        raw_root,
                        kyoku_index,
                        decision,
                        exact_values,
                    )

    return {
        "scope": {
            "development_years_only": list(DEVELOPMENT_YEARS),
            "selected_file_count": len(selected),
            "max_files": max_files,
            "audit_sample_limit": audit_samples,
            "exact_decision_limit": exact_decisions,
            "holdout_2025_touched": False,
        },
        "counts": dict(sorted(counts.items())),
        "decision_turn_bins": dict(sorted(turn_bins.items())),
        "exact_enumeration_space": _space_summary(exact_space_sizes),
        "exact_ranking": {
            "counts": dict(sorted(exact_counts.items())),
            "inversion_samples": inversion_samples,
        },
        "audit_samples": audits,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    year_root = args.raw_root / str(args.year)
    paths = tuple(year_root.rglob("*.mjson"))
    report = collect_pilot(
        paths,
        raw_root=args.raw_root,
        max_files=args.max_files,
        audit_samples=args.audit_samples,
        exact_decisions=args.exact_decisions,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


def _accumulate_decision(
    counts: Counter[str],
    turn_bins: Counter[str],
    decision: PostRiichiDrawDecision,
) -> None:
    counts["decisions"] += 1
    counts["candidate_rows"] += len(decision.candidates)
    counts["structural_wait_rows"] += sum(
        candidate.is_structural_wait for candidate in decision.candidates
    )
    counts["ron_eligible_rows"] += sum(
        candidate.is_ron_eligible for candidate in decision.candidates
    )
    counts["decisions_with_structural_wait"] += any(
        candidate.is_structural_wait for candidate in decision.candidates
    )
    counts["chasing_riichi_decisions"] += decision.actual_declared_reach
    counts["target_furiten_decisions"] += decision.target_is_ron_furiten
    turn_bins[_turn_bin(decision.decision_discard_number)] += 1


def _audit_row(
    path: Path,
    raw_root: Path,
    kyoku_index: int,
    decision: PostRiichiDrawDecision,
    state_space_size: int,
) -> dict[str, Any]:
    try:
        source_path = path.resolve().relative_to(raw_root.resolve()).as_posix()
    except ValueError:
        source_path = path.resolve().as_posix()
    return {
        "source_path": source_path,
        "kyoku_index": kyoku_index,
        "tsumo_event_index": decision.tsumo_event_index,
        "dahai_event_index": decision.dahai_event_index,
        "observer_actor": decision.observer_actor,
        "target_actor": decision.target_actor,
        "turn": decision.decision_discard_number,
        "actual_discard": decision.actual_discard_tile,
        "actual_declared_reach": decision.actual_declared_reach,
        "target_wait_tiles": list(decision.target_wait_tiles),
        "target_is_ron_furiten": decision.target_is_ron_furiten,
        "bounded_count_vectors": str(state_space_size),
        "candidates": [
            {
                "tile": candidate.tile,
                "simple_combo": candidate.simple_combo.total,
                "is_structural_wait": candidate.is_structural_wait,
                "is_ron_eligible": candidate.is_ron_eligible,
            }
            for candidate in decision.candidates
        ],
    }


def _exact_candidate_counts(
    decision: PostRiichiDrawDecision,
) -> dict[str, CandidateWaitStateCount]:
    fixed_meld_count = (13 - decision.target_concealed_tile_count) // 3
    return {
        candidate.tile: count_candidate_wait_states(
            decision.unseen_counts,
            candidate.tile,
            fixed_meld_count=fixed_meld_count,
        )
        for candidate in decision.candidates
    }


def _accumulate_exact_ranking(
    counts: Counter[str],
    inversion_samples: list[dict[str, Any]],
    path: Path,
    raw_root: Path,
    kyoku_index: int,
    decision: PostRiichiDrawDecision,
    exact_values: dict[str, CandidateWaitStateCount],
) -> None:
    counts["decisions"] += 1
    counts["candidate_rows"] += len(decision.candidates)
    decision_has_state_inversion = False
    decision_has_weight_inversion = False
    for left, right in combinations(decision.candidates, 2):
        simple_difference = left.simple_combo.total - right.simple_combo.total
        state_difference = (
            exact_values[left.tile].state_count - exact_values[right.tile].state_count
        )
        weight_difference = (
            exact_values[left.tile].physical_weight
            - exact_values[right.tile].physical_weight
        )
        counts["candidate_pairs"] += 1
        if simple_difference == 0:
            counts["pairs_with_simple_tie"] += 1
        state_inversion = _accumulate_order_comparison(
            counts,
            "state",
            simple_difference,
            state_difference,
        )
        weight_inversion = _accumulate_order_comparison(
            counts,
            "physical_weight",
            simple_difference,
            weight_difference,
        )
        decision_has_state_inversion |= state_inversion
        decision_has_weight_inversion |= weight_inversion
        if not state_inversion and not weight_inversion:
            continue
        if len(inversion_samples) >= 20:
            continue
        try:
            source_path = path.resolve().relative_to(raw_root.resolve()).as_posix()
        except ValueError:
            source_path = path.resolve().as_posix()
        inversion_samples.append(
            {
                "source_path": source_path,
                "kyoku_index": kyoku_index,
                "tsumo_event_index": decision.tsumo_event_index,
                "observer_actor": decision.observer_actor,
                "target_actor": decision.target_actor,
                "inversion_types": [
                    name
                    for name, present in (
                        ("unique_state_count", state_inversion),
                        ("unique_physical_weight", weight_inversion),
                    )
                    if present
                ],
                "left": {
                    "tile": left.tile,
                    "simple_combo": left.simple_combo.total,
                    "unique_state_count": exact_values[left.tile].state_count,
                    "unique_physical_weight": str(
                        exact_values[left.tile].physical_weight
                    ),
                },
                "right": {
                    "tile": right.tile,
                    "simple_combo": right.simple_combo.total,
                    "unique_state_count": exact_values[right.tile].state_count,
                    "unique_physical_weight": str(
                        exact_values[right.tile].physical_weight
                    ),
                },
            }
        )
    counts["decisions_with_state_inversion"] += decision_has_state_inversion
    counts["decisions_with_physical_weight_inversion"] += decision_has_weight_inversion


def _accumulate_order_comparison(
    counts: Counter[str],
    name: str,
    simple_difference: int,
    comparison_difference: int,
) -> bool:
    if simple_difference == 0 or comparison_difference == 0:
        counts[f"{name}_pairs_with_tie"] += 1
        return False
    if (simple_difference > 0) == (comparison_difference > 0):
        counts[f"{name}_strict_order_agreements"] += 1
        return False
    counts[f"{name}_strict_order_inversions"] += 1
    return True


def _space_summary(sizes: list[int]) -> dict[str, Any]:
    if not sizes:
        return {"observations": 0}
    ordered = sorted(sizes)
    return {
        "observations": len(ordered),
        "minimum": str(ordered[0]),
        "median": str(ordered[len(ordered) // 2]),
        "maximum": str(ordered[-1]),
        "maximum_decimal_digits": len(str(ordered[-1])),
    }


def _turn_bin(turn: int) -> str:
    if turn <= 6:
        return "1-6"
    if turn <= 9:
        return "7-9"
    if turn <= 12:
        return "10-12"
    return "13+"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
