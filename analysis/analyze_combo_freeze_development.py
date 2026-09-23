"""Run the pre-2024 primary combo-development freeze analysis."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import scipy

from analysis.analyze_combo_development import (
    _canonical_json_sha256,
    _read_json_object,
    _samples_from_manifest,
)
from mahjong_analysis.combo_bootstrap import (
    GroupContrast,
    joint_paired_game_cluster_bootstrap,
)
from mahjong_analysis.combo_dataset import (
    DEVELOPMENT_YEARS,
    StableYearSample,
    collect_candidate_dataset,
    expanding_year_splits,
    file_sha256,
)
from mahjong_analysis.combo_prediction import (
    DECISION_TURN_BINS,
    SparseExample,
    evaluate_binary_predictions,
)
from mahjong_analysis.combo_sparse_pipeline import (
    SparseLogisticModel,
    YearSparseArtifact,
    build_year_sparse_artifact,
    conventional_feature_mask,
    fit_sparse_problem,
    global_feature_vocabulary,
    load_year_sparse_artifact,
    prepare_logistic_problem,
    save_year_sparse_artifact,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "combo-development-sample-2020-2023-random1000.manifest.json"
)
DEFAULT_CACHE_DIR = (
    PROJECT_ROOT / "data" / "processed" / "combo-development-primary-r1000-v1"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs" / "combo-development-primary-r1000-v2.json"
)

ANALYSIS_ID = "combo-development-primary-r1000-v2"
FILES_PER_YEAR = 1_000
SAMPLE_SEED = 20260923
LABEL = "ron_eligible"
BASE_MODEL = "conventional"
CHALLENGER_MODEL = "conventional_simple"
COMPARISON_NAME = "conventional_simple_minus_conventional"
L2_GRID = (
    0.0,
    1e-7,
    3e-7,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
)
L2_TIE_TOLERANCE = 1e-6
FIXED_BOOTSTRAP_REPLICATES = 2_000
FIXED_BOOTSTRAP_SEED = SAMPLE_SEED
FIXED_MAX_ITERATIONS = 300
FIXED_STABILITY_MAX_ITERATIONS = 600
FIXED_GRADIENT_TOLERANCE = 1e-7


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the primary combo model using development years only."
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--build-cache-only", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--source-prefix", type=_source_prefix, default=FILES_PER_YEAR)
    parser.add_argument(
        "--bootstrap-replicates",
        type=_nonnegative_int,
        default=FIXED_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=FIXED_BOOTSTRAP_SEED)
    parser.add_argument(
        "--max-iterations", type=_positive_int, default=FIXED_MAX_ITERATIONS
    )
    parser.add_argument(
        "--stability-max-iterations",
        type=_positive_int,
        default=FIXED_STABILITY_MAX_ITERATIONS,
    )
    parser.add_argument(
        "--gradient-tolerance",
        type=_positive_float,
        default=FIXED_GRADIENT_TOLERANCE,
    )
    parser.add_argument("--skip-stability-check", action="store_true")
    args = parser.parse_args(argv)
    if args.stability_max_iterations < args.max_iterations:
        parser.error("--stability-max-iterations must be at least --max-iterations")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = perf_counter()
    manifest = _read_json_object(args.manifest)
    if manifest.get("schema_version") != 2:
        raise ValueError("the freeze analysis requires a schema-v2 manifest")
    samples = _samples_from_manifest(
        manifest,
        argparse.Namespace(
            raw_root=args.raw_root,
            sample_seed=SAMPLE_SEED,
            files_per_year=FILES_PER_YEAR,
        ),
    )
    manifest_canonical_sha256 = _canonical_json_sha256(manifest)
    artifacts, cache_documents = _load_or_build_artifacts(
        samples,
        raw_root=args.raw_root,
        cache_dir=args.cache_dir,
        manifest_canonical_sha256=manifest_canonical_sha256,
        rebuild=args.rebuild_cache,
    )
    if args.build_cache_only:
        document = {
            "analysis_id": ANALYSIS_ID,
            "status": "cache_built; modeling_not_run",
            "manifest_canonical_sha256": manifest_canonical_sha256,
            "cache_artifacts": cache_documents,
            "validation_2024_touched": False,
            "holdout_2025_touched": False,
        }
        _write_json(args.output, document)
        return 0

    fold_state, lambda_scores, lambda_fold_diagnostics = _tune_base_models(
        artifacts,
        source_prefix=args.source_prefix,
        max_iterations=args.max_iterations,
        gradient_tolerance=args.gradient_tolerance,
    )
    selected_l2 = _choose_lambda(lambda_scores)
    fold_documents: dict[str, Any] = {}
    pooled_examples: list[SparseExample] = []
    pooled_turn_bins: list[str] = []
    pooled_base: list[float] = []
    pooled_challenger: list[float] = []
    stratum_by_cluster: dict[str, str] = {}
    all_fit_accepted = True
    all_stability_accepted = True

    for train_years, test_year in expanding_year_splits():
        state = fold_state[test_year]
        base_model = state["models"][selected_l2]
        base_predictions = state["predictions"][selected_l2]
        vocabulary = base_model.feature_names
        problem = prepare_logistic_problem(
            (artifacts[year] for year in train_years),
            label=LABEL,
            vocabulary=vocabulary,
            source_prefix=args.source_prefix,
        )
        challenger = fit_sparse_problem(
            problem,
            l2=selected_l2,
            max_iterations=args.max_iterations,
            gradient_tolerance=args.gradient_tolerance,
            function_tolerance=0.0,
            max_line_search_steps=50,
            history_size=20,
        )
        _require_accepted_fit(
            challenger,
            args.gradient_tolerance,
            max_iterations=args.max_iterations,
        )
        test_artifact = artifacts[test_year]
        test_mask = test_artifact.source_prefix_mask(args.source_prefix)
        challenger_predictions = challenger.predict_artifact(test_artifact)[test_mask]
        examples, turn_bins = _examples_from_artifact(
            test_artifact,
            source_prefix=args.source_prefix,
            label=LABEL,
        )
        if len(examples) != len(base_predictions):
            raise RuntimeError("test examples and predictions are misaligned")

        stability = None
        if not args.skip_stability_check:
            base_stable = fit_sparse_problem(
                problem,
                active_features=conventional_feature_mask(vocabulary),
                l2=selected_l2,
                max_iterations=args.stability_max_iterations,
                gradient_tolerance=args.gradient_tolerance,
                function_tolerance=0.0,
                max_line_search_steps=50,
                history_size=20,
            )
            challenger_stable = fit_sparse_problem(
                problem,
                l2=selected_l2,
                max_iterations=args.stability_max_iterations,
                gradient_tolerance=args.gradient_tolerance,
                function_tolerance=0.0,
                max_line_search_steps=50,
                history_size=20,
            )
            base_stable_predictions = base_stable.predict_artifact(test_artifact)[
                test_mask
            ]
            challenger_stable_predictions = challenger_stable.predict_artifact(
                test_artifact
            )[test_mask]
            stability = _stability_document(
                base_model,
                base_stable,
                base_predictions,
                base_stable_predictions,
                challenger,
                challenger_stable,
                challenger_predictions,
                challenger_stable_predictions,
                gradient_tolerance=args.gradient_tolerance,
                primary_max_iterations=args.max_iterations,
                stability_max_iterations=args.stability_max_iterations,
                labels=test_artifact.labels(LABEL)[test_mask],
                weights=test_artifact.weights[test_mask],
            )
            all_stability_accepted &= bool(stability["accepted"])

        base_metrics = evaluate_binary_predictions(examples, base_predictions)
        challenger_metrics = evaluate_binary_predictions(
            examples, challenger_predictions
        )
        all_fit_accepted &= _fit_is_accepted(
            base_model,
            args.gradient_tolerance,
            max_iterations=args.max_iterations,
        )
        all_fit_accepted &= _fit_is_accepted(
            challenger,
            args.gradient_tolerance,
            max_iterations=args.max_iterations,
        )
        fold_documents[str(test_year)] = {
            "train_years": list(train_years),
            "test_year": test_year,
            "row_count": len(examples),
            "decision_count": len({example.group_id for example in examples}),
            "game_count": len({example.cluster_id for example in examples}),
            "models": {
                BASE_MODEL: _model_document(base_model, base_metrics),
                CHALLENGER_MODEL: _model_document(challenger, challenger_metrics),
            },
            "delta": _metric_delta(base_metrics, challenger_metrics),
            "stability_check": stability,
        }
        pooled_examples.extend(examples)
        pooled_turn_bins.extend(turn_bins)
        pooled_base.extend(float(value) for value in base_predictions)
        pooled_challenger.extend(float(value) for value in challenger_predictions)
        for example in examples:
            stratum_by_cluster[example.cluster_id] = str(test_year)

    pooled_examples_tuple = tuple(pooled_examples)
    pooled_base_tuple = tuple(pooled_base)
    pooled_challenger_tuple = tuple(pooled_challenger)
    pooled_base_metrics = evaluate_binary_predictions(
        pooled_examples_tuple, pooled_base_tuple
    )
    pooled_challenger_metrics = evaluate_binary_predictions(
        pooled_examples_tuple, pooled_challenger_tuple
    )
    joint_bootstrap = (
        joint_paired_game_cluster_bootstrap(
            pooled_examples_tuple,
            {
                BASE_MODEL: pooled_base_tuple,
                CHALLENGER_MODEL: pooled_challenger_tuple,
            },
            {COMPARISON_NAME: (BASE_MODEL, CHALLENGER_MODEL)},
            group_ids=tuple(pooled_turn_bins),
            group_order=DECISION_TURN_BINS,
            stratum_by_cluster=stratum_by_cluster,
            contrasts=(GroupContrast("late_minus_early", "13+", "1-6"),),
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed,
        )
        if args.bootstrap_replicates
        else None
    )
    fixed_configuration = (
        args.source_prefix == FILES_PER_YEAR
        and args.bootstrap_replicates == FIXED_BOOTSTRAP_REPLICATES
        and args.bootstrap_seed == FIXED_BOOTSTRAP_SEED
        and args.max_iterations == FIXED_MAX_ITERATIONS
        and args.stability_max_iterations == FIXED_STABILITY_MAX_ITERATIONS
        and args.gradient_tolerance == FIXED_GRADIENT_TOLERANCE
        and not args.skip_stability_check
    )
    freeze_ready = (
        fixed_configuration
        and all_fit_accepted
        and all_stability_accepted
    )
    report = {
        "analysis_id": ANALYSIS_ID,
        "status": "GO_to_freeze" if freeze_ready else "NO_GO_development_incomplete",
        "scope": {
            "development_years": list(DEVELOPMENT_YEARS),
            "prediction_years": list(DEVELOPMENT_YEARS[1:]),
            "files_per_year": args.source_prefix,
            "fixed_primary_files_per_year": FILES_PER_YEAR,
            "sample_seed": SAMPLE_SEED,
            "label": LABEL,
            "base_model": BASE_MODEL,
            "challenger_model": CHALLENGER_MODEL,
            "primary_metric": "weighted_log_loss_delta",
            "validation_2024_touched": False,
            "holdout_2025_touched": False,
        },
        "manifest": {
            "path": str(args.manifest.resolve()),
            "canonical_sha256": manifest_canonical_sha256,
            "file_sha256": file_sha256(args.manifest),
        },
        "cache_artifacts": cache_documents,
        "optimizer": {
            "type": "SciPy L-BFGS-B with analytic gradient",
            "objective": "weighted mean log loss + l2/2 * ||beta||^2",
            "intercept_penalized": False,
            "lambda_grid": list(L2_GRID),
            "lambda_selection": (
                "base-only pooled 2021-2023 OOT log loss; within 1e-6 choose "
                "larger lambda"
            ),
            "selected_l2": selected_l2,
            "max_iterations": args.max_iterations,
            "stability_max_iterations": args.stability_max_iterations,
            "gradient_tolerance": args.gradient_tolerance,
            "function_tolerance": 0.0,
            "lambda_scores": {
                _float_key(value): score for value, score in lambda_scores.items()
            },
            "lambda_fold_diagnostics": lambda_fold_diagnostics,
        },
        "folds": fold_documents,
        "pooled": {
            "models": {
                BASE_MODEL: {"metrics": asdict(pooled_base_metrics)},
                CHALLENGER_MODEL: {"metrics": asdict(pooled_challenger_metrics)},
            },
            "delta": _metric_delta(
                pooled_base_metrics,
                pooled_challenger_metrics,
            ),
            "joint_game_cluster_bootstrap": (
                asdict(joint_bootstrap) if joint_bootstrap is not None else None
            ),
        },
        "freeze_checks": {
            "all_primary_fits_accepted": all_fit_accepted,
            "all_doubled_iteration_stability_checks_accepted": (
                all_stability_accepted and not args.skip_stability_check
            ),
            "fixed_configuration": fixed_configuration,
            "bootstrap_replicates_exactly_2000": (
                args.bootstrap_replicates == FIXED_BOOTSTRAP_REPLICATES
            ),
            "bootstrap_seed_fixed": args.bootstrap_seed == FIXED_BOOTSTRAP_SEED,
            "optimizer_limits_fixed": (
                args.max_iterations == FIXED_MAX_ITERATIONS
                and args.stability_max_iterations
                == FIXED_STABILITY_MAX_ITERATIONS
                and args.gradient_tolerance == FIXED_GRADIENT_TOLERANCE
            ),
            "full_1000_file_prefix": args.source_prefix == FILES_PER_YEAR,
            "freeze_ready": freeze_ready,
        },
        "environment": _environment_document(),
        "elapsed_seconds": perf_counter() - started,
    }
    _write_json(args.output, report)
    return 0


def _load_or_build_artifacts(
    samples: dict[int, StableYearSample],
    *,
    raw_root: Path,
    cache_dir: Path,
    manifest_canonical_sha256: str,
    rebuild: bool,
) -> tuple[dict[int, YearSparseArtifact], dict[str, Any]]:
    artifacts: dict[int, YearSparseArtifact] = {}
    documents: dict[str, Any] = {}
    for year in DEVELOPMENT_YEARS:
        sample = samples[year]
        cache_path = cache_dir / f"{year}-conventional-simple.npz"
        if cache_path.is_file() and not rebuild:
            artifact = load_year_sparse_artifact(cache_path)
        else:
            print(f"[cache] extract/build {year}", file=sys.stderr, flush=True)
            dataset = collect_candidate_dataset(raw_root, year, sample.paths)
            counts_json = json.dumps(
                dataset.counts,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            artifact = build_year_sparse_artifact(
                dataset.rows,
                year=year,
                source_rank_by_game={
                    source: rank
                    for rank, source in enumerate(sample.relative_sources)
                },
                provenance={
                    "analysis_id": ANALYSIS_ID,
                    "manifest_canonical_sha256": manifest_canonical_sha256,
                    "counts_json": counts_json,
                },
            )
            save_year_sparse_artifact(cache_path, artifact)
            artifact = load_year_sparse_artifact(cache_path)
        provenance = dict(artifact.provenance)
        if artifact.year != year:
            raise ValueError(f"cache year mismatch for {cache_path}")
        if artifact.source_ids_by_rank != sample.relative_sources:
            raise ValueError(f"cache selected-source order mismatch for {year}")
        if provenance.get("manifest_canonical_sha256") != manifest_canonical_sha256:
            raise ValueError(f"cache manifest digest mismatch for {year}")
        artifacts[year] = artifact
        documents[str(year)] = {
            "path": str(cache_path.resolve()),
            "sha256": file_sha256(cache_path),
            "row_count": artifact.row_count,
            "feature_count": artifact.feature_count,
            "selected_source_count": artifact.selected_source_count,
            "counts": json.loads(provenance["counts_json"]),
        }
    return artifacts, documents


def _tune_base_models(
    artifacts: dict[int, YearSparseArtifact],
    *,
    source_prefix: int,
    max_iterations: int,
    gradient_tolerance: float,
) -> tuple[
    dict[int, dict[str, Any]],
    dict[float, float | None],
    dict[str, dict[str, Any]],
]:
    fold_state: dict[int, dict[str, Any]] = {}
    loss_sums = {value: 0.0 for value in L2_GRID}
    eligible = {value: True for value in L2_GRID}
    diagnostics_by_year: dict[str, dict[str, Any]] = {}
    total_weight = 0.0
    for train_years, test_year in expanding_year_splits():
        print(
            f"[tune] train={','.join(map(str, train_years))} test={test_year}",
            file=sys.stderr,
            flush=True,
        )
        train_artifacts = tuple(artifacts[year] for year in train_years)
        vocabulary = global_feature_vocabulary(train_artifacts)
        problem = prepare_logistic_problem(
            train_artifacts,
            label=LABEL,
            vocabulary=vocabulary,
            source_prefix=source_prefix,
        )
        active = conventional_feature_mask(vocabulary)
        test_artifact = artifacts[test_year]
        test_mask = test_artifact.source_prefix_mask(source_prefix)
        test_labels = test_artifact.labels(LABEL)[test_mask]
        test_weights = test_artifact.weights[test_mask]
        fold_weight = float(np.sum(test_weights, dtype=np.float64))
        total_weight += fold_weight
        models: dict[float, SparseLogisticModel] = {}
        predictions: dict[float, np.ndarray] = {}
        fold_diagnostics: dict[str, Any] = {}
        for l2 in L2_GRID:
            model = fit_sparse_problem(
                problem,
                active_features=active,
                l2=l2,
                max_iterations=max_iterations,
                gradient_tolerance=gradient_tolerance,
                function_tolerance=0.0,
                max_line_search_steps=50,
                history_size=20,
            )
            models[l2] = model
            accepted = _fit_is_accepted(
                model,
                gradient_tolerance,
                max_iterations=max_iterations,
            )
            fold_diagnostics[_float_key(l2)] = {
                "accepted": accepted,
                **asdict(model.diagnostics),
            }
            if not accepted:
                eligible[l2] = False
                continue
            values = model.predict_artifact(test_artifact)[test_mask]
            loss_sums[l2] += _weighted_log_loss_sum(
                test_labels,
                values,
                test_weights,
            )
            predictions[l2] = values
        fold_state[test_year] = {
            "models": models,
            "predictions": predictions,
        }
        diagnostics_by_year[str(test_year)] = fold_diagnostics
    return (
        fold_state,
        {
            value: loss_sums[value] / total_weight if eligible[value] else None
            for value in L2_GRID
        },
        diagnostics_by_year,
    )


def _choose_lambda(scores: dict[float, float | None]) -> float:
    if set(scores) != set(L2_GRID):
        raise ValueError("lambda scores must exactly cover the fixed grid")
    finite_scores = {
        l2: score
        for l2, score in scores.items()
        if score is not None and np.isfinite(score)
    }
    if any(
        score is not None and not np.isfinite(score) for score in scores.values()
    ):
        raise ValueError("lambda scores must be finite")
    if not finite_scores:
        raise ValueError("no lambda candidate passed the convergence gate")
    minimum = min(finite_scores.values())
    eligible = [
        l2
        for l2, score in finite_scores.items()
        if score <= minimum + L2_TIE_TOLERANCE
    ]
    return max(eligible)


def _examples_from_artifact(
    artifact: YearSparseArtifact,
    *,
    source_prefix: int,
    label: str,
) -> tuple[tuple[SparseExample, ...], tuple[str, ...]]:
    mask = artifact.source_prefix_mask(source_prefix)
    positions = np.flatnonzero(mask)
    labels = artifact.labels(label)  # type: ignore[arg-type]
    examples = tuple(
        SparseExample(
            row_id=f"{artifact.year}:{int(position)}",
            group_id=artifact.group_ids[int(artifact.group_indices[position])],
            cluster_id=artifact.cluster_ids[int(artifact.cluster_indices[position])],
            label=bool(labels[position]),
            weight=float(artifact.weights[position]),
            features=(),
        )
        for position in positions
    )
    turn_bins = tuple(
        DECISION_TURN_BINS[int(artifact.turn_bins[position])] for position in positions
    )
    return examples, turn_bins


def _weighted_log_loss_sum(
    labels: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray,
) -> float:
    clipped = np.clip(predictions, 1e-15, 1 - 1e-15)
    losses = -(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
    return float(np.dot(weights, losses))


def _fit_is_accepted(
    model: SparseLogisticModel,
    tolerance: float,
    *,
    max_iterations: int,
) -> bool:
    diagnostics = model.diagnostics
    return bool(
        diagnostics.converged
        and diagnostics.iterations < max_iterations
        and np.isfinite(diagnostics.final_objective)
        and diagnostics.final_gradient_inf_norm <= tolerance
        and np.isfinite(model.intercept)
        and np.all(np.isfinite(model.coefficients))
    )


def _require_accepted_fit(
    model: SparseLogisticModel,
    tolerance: float,
    *,
    max_iterations: int,
) -> None:
    if not _fit_is_accepted(
        model,
        tolerance,
        max_iterations=max_iterations,
    ):
        raise RuntimeError(
            "optimizer did not meet the fixed convergence gate: "
            f"{asdict(model.diagnostics)}"
        )


def _stability_document(
    base: SparseLogisticModel,
    base_stable: SparseLogisticModel,
    base_predictions: Sequence[float],
    base_stable_predictions: Sequence[float],
    challenger: SparseLogisticModel,
    challenger_stable: SparseLogisticModel,
    challenger_predictions: Sequence[float],
    challenger_stable_predictions: Sequence[float],
    *,
    gradient_tolerance: float,
    primary_max_iterations: int,
    stability_max_iterations: int,
    labels: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    base_difference = _prediction_difference(base_predictions, base_stable_predictions)
    challenger_difference = _prediction_difference(
        challenger_predictions, challenger_stable_predictions
    )
    base_loss_difference = abs(
        _weighted_log_loss_sum(
            labels,
            np.asarray(base_predictions),
            weights,
        )
        - _weighted_log_loss_sum(
            labels,
            np.asarray(base_stable_predictions),
            weights,
        )
    )
    challenger_loss_difference = abs(
        _weighted_log_loss_sum(
            labels,
            np.asarray(challenger_predictions),
            weights,
        )
        - _weighted_log_loss_sum(
            labels,
            np.asarray(challenger_stable_predictions),
            weights,
        )
    )
    accepted = (
        _fit_is_accepted(
            base,
            gradient_tolerance,
            max_iterations=primary_max_iterations,
        )
        and _fit_is_accepted(
            base_stable,
            gradient_tolerance,
            max_iterations=stability_max_iterations,
        )
        and _fit_is_accepted(
            challenger,
            gradient_tolerance,
            max_iterations=primary_max_iterations,
        )
        and _fit_is_accepted(
            challenger_stable,
            gradient_tolerance,
            max_iterations=stability_max_iterations,
        )
        and base_difference <= 1e-8
        and challenger_difference <= 1e-8
        and base_loss_difference <= 1e-10
        and challenger_loss_difference <= 1e-10
    )
    return {
        "accepted": accepted,
        "thresholds": {
            "prediction_max_absolute_difference": 1e-8,
            "log_loss_absolute_difference": 1e-10,
            "gradient_inf_norm": gradient_tolerance,
        },
        "base": {
            "prediction_max_absolute_difference": base_difference,
            "log_loss_absolute_difference": base_loss_difference,
            "primary_fit": asdict(base.diagnostics),
            "doubled_iteration_fit": asdict(base_stable.diagnostics),
        },
        "challenger": {
            "prediction_max_absolute_difference": challenger_difference,
            "log_loss_absolute_difference": challenger_loss_difference,
            "primary_fit": asdict(challenger.diagnostics),
            "doubled_iteration_fit": asdict(challenger_stable.diagnostics),
        },
    }


def _prediction_difference(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        raise ValueError("prediction vectors must align")
    return float(np.max(np.abs(left_array - right_array), initial=0.0))


def _model_document(model: SparseLogisticModel, metrics: Any) -> dict[str, Any]:
    coefficients = [
        (name, float(value))
        for name, value in zip(
            model.feature_names,
            model.coefficients,
            strict=True,
        )
        if value != 0
    ]
    coefficients.sort(key=lambda item: (-abs(item[1]), item[0]))
    return {
        "metrics": asdict(metrics),
        "intercept": model.intercept,
        "active_feature_count": int(np.sum(model.active_mask)),
        "diagnostics": asdict(model.diagnostics),
        "largest_absolute_coefficients": [
            {"feature": name, "coefficient": value}
            for name, value in coefficients[:12]
        ],
    }


def _metric_delta(base: Any, challenger: Any) -> dict[str, float | None]:
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


def _environment_document() -> dict[str, Any]:
    tracked_files = (
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "uv.lock",
        PROJECT_ROOT / "src" / "mahjong_analysis" / "combo_prediction.py",
        PROJECT_ROOT / "src" / "mahjong_analysis" / "combo_bootstrap.py",
        PROJECT_ROOT / "src" / "mahjong_analysis" / "combo_sparse_pipeline.py",
        Path(__file__),
    )
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "argv": [str(value) for value in sys.argv],
        "file_sha256": {
            str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
            for path in tracked_files
            if path.is_file()
        },
    }


def _float_key(value: float) -> str:
    return format(value, ".12g")


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        document,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
        default=_json_default,
    ) + "\n"
    path.write_text(rendered, encoding="utf-8")
    digest = sha256(rendered.encode()).hexdigest()
    print(f"[output] {path} sha256={digest}", file=sys.stderr, flush=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _source_prefix(value: str) -> int:
    parsed = int(value)
    if parsed not in {50, 100, 250, 500, 1_000}:
        raise argparse.ArgumentTypeError("must be one of 50, 100, 250, 500, 1000")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
