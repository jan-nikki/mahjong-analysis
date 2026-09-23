"""Run a bounded out-of-time pilot of combo danger prediction."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mahjong_analysis.combo_prediction import (
    BinaryMetrics,
    ComboCandidateRow,
    FeatureSet,
    LabelName,
    candidate_rows_for_decision,
    evaluate_binary_predictions,
    fit_sparse_logistic,
    paired_game_cluster_bootstrap,
    sparse_examples,
)
from mahjong_analysis.mjai import is_target_game, load_mjai, split_kyoku
from mahjong_analysis.post_riichi_decisions import (
    extract_post_riichi_draw_decisions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw"
DEVELOPMENT_YEARS = tuple(range(2020, 2025))
LABELS: tuple[LabelName, ...] = ("structural_wait", "ron_eligible")
PILOT_FEATURE_SETS: tuple[FeatureSet, ...] = (
    "simple",
    "tile_turn",
    "conventional",
    "conventional_simple",
    "conventional_simple_percentile",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bounded out-of-time combo prediction without the 2025 holdout."
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--train-year", type=int, choices=DEVELOPMENT_YEARS, default=2020
    )
    parser.add_argument(
        "--test-year", type=int, choices=DEVELOPMENT_YEARS, default=2021
    )
    parser.add_argument("--max-files", type=_positive_int, default=100)
    parser.add_argument("--epochs", type=_positive_int, default=100)
    parser.add_argument("--learning-rate", type=_positive_float, default=0.05)
    parser.add_argument("--l2", type=_nonnegative_float, default=0.001)
    parser.add_argument("--bootstrap-replicates", type=_nonnegative_int, default=200)
    parser.add_argument("--bootstrap-seed", type=int, default=20260923)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.test_year <= args.train_year:
        parser.error("--test-year must be later than --train-year")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    train_rows, train_counts, train_sources = _collect_year(
        args.raw_root,
        args.train_year,
        args.max_files,
    )
    test_rows, test_counts, test_sources = _collect_year(
        args.raw_root,
        args.test_year,
        args.max_files,
    )
    results: dict[str, Any] = {}
    for label in LABELS:
        metrics_by_model: dict[FeatureSet, BinaryMetrics] = {}
        examples_by_model = {}
        predictions_by_model = {}
        model_documents: dict[str, Any] = {}
        for feature_set in PILOT_FEATURE_SETS:
            train_examples = sparse_examples(train_rows, feature_set, label)
            test_examples = sparse_examples(test_rows, feature_set, label)
            model = fit_sparse_logistic(
                train_examples,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                l2=args.l2,
            )
            predictions = model.predict(test_examples)
            metrics = evaluate_binary_predictions(test_examples, predictions)
            metrics_by_model[feature_set] = metrics
            examples_by_model[feature_set] = test_examples
            predictions_by_model[feature_set] = predictions
            model_documents[feature_set] = {
                "metrics": asdict(metrics),
                "feature_count": len(model.weights),
                "intercept": model.intercept,
                "epochs_run": model.epochs_run,
                "final_largest_gradient": model.final_largest_gradient,
                "largest_absolute_coefficients": _largest_coefficients(model.weights),
            }
        results[label] = {
            "models": model_documents,
            "paired_deltas_from_conventional": {
                feature_set: _metric_delta(
                    metrics_by_model["conventional"],
                    metrics_by_model[feature_set],
                )
                for feature_set in (
                    "conventional_simple",
                    "conventional_simple_percentile",
                )
            },
            "paired_game_cluster_bootstrap": {
                feature_set: asdict(
                    paired_game_cluster_bootstrap(
                        examples_by_model["conventional"],
                        predictions_by_model["conventional"],
                        predictions_by_model[feature_set],
                        replicates=args.bootstrap_replicates,
                        seed=args.bootstrap_seed,
                    )
                )
                for feature_set in (
                    "conventional_simple",
                    "conventional_simple_percentile",
                )
            }
            if args.bootstrap_replicates
            else None,
        }

    report = {
        "scope": {
            "development_years_only": list(DEVELOPMENT_YEARS),
            "train_year": args.train_year,
            "test_year": args.test_year,
            "max_files_per_year": args.max_files,
            "holdout_2025_touched": False,
            "selection": "lexicographically first source paths within each year",
            "status": "development pilot; not a representative population estimate",
        },
        "optimizer": {
            "type": "dependency-free deterministic full-batch Adam logistic regression",
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
            "candidate_weight": "1 / candidate_count_in_decision",
        },
        "train": {
            "counts": dict(sorted(train_counts.items())),
            "first_source": train_sources[0] if train_sources else None,
            "last_source": train_sources[-1] if train_sources else None,
        },
        "test": {
            "counts": dict(sorted(test_counts.items())),
            "first_source": test_sources[0] if test_sources else None,
            "last_source": test_sources[-1] if test_sources else None,
        },
        "results": results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


def _collect_year(
    raw_root: Path,
    year: int,
    max_files: int,
) -> tuple[list[ComboCandidateRow], Counter[str], list[str]]:
    year_root = raw_root / str(year)
    selected = tuple(
        sorted(year_root.rglob("*.mjson"), key=lambda path: path.as_posix())[:max_files]
    )
    rows: list[ComboCandidateRow] = []
    counts: Counter[str] = Counter()
    sources: list[str] = []
    for path in selected:
        counts["scanned_files"] += 1
        events = load_mjai(path)
        if not is_target_game(path, events):
            continue
        counts["target_games"] += 1
        relative_source = path.resolve().relative_to(raw_root.resolve()).as_posix()
        sources.append(relative_source)
        for kyoku_index, kyoku in enumerate(split_kyoku(events)):
            if kyoku[0].get("bakaze") != "E":
                continue
            counts["east_kyokus"] += 1
            decisions = extract_post_riichi_draw_decisions(kyoku)
            counts["decisions"] += len(decisions)
            for decision in decisions:
                candidate_rows = candidate_rows_for_decision(
                    decision,
                    year=year,
                    game_id=relative_source,
                    kyoku_index=kyoku_index,
                )
                rows.extend(candidate_rows)
                counts["candidate_rows"] += len(candidate_rows)
                counts["structural_wait_rows"] += sum(
                    row.candidate.is_structural_wait for row in candidate_rows
                )
                counts["ron_eligible_rows"] += sum(
                    row.candidate.is_ron_eligible for row in candidate_rows
                )
    if not rows:
        raise ValueError(f"no candidate rows extracted for {year}")
    return rows, counts, sources


def _metric_delta(
    base: BinaryMetrics, challenger: BinaryMetrics
) -> dict[str, float | None]:
    return {
        "auc": _optional_difference(challenger.auc, base.auc),
        "log_loss": challenger.log_loss - base.log_loss,
        "brier": challenger.brier - base.brier,
        "ece": challenger.ece - base.ece,
        "macro_concordance": _optional_difference(
            challenger.macro_concordance,
            base.macro_concordance,
        ),
    }


def _optional_difference(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else left - right


def _largest_coefficients(
    weights: dict[str, float],
    *,
    limit: int = 12,
) -> list[dict[str, float | str]]:
    return [
        {"feature": name, "coefficient": coefficient}
        for name, coefficient in sorted(
            weights.items(),
            key=lambda item: (-abs(item[1]), item[0]),
        )[:limit]
    ]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
