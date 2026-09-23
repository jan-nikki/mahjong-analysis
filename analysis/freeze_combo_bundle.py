"""Validate and seal the pre-2024 combo-development result.

The bundle deliberately contains hashes of its payload rather than a hash of
the bundle file itself.  The latter is printed after the atomic write, avoiding
a self-hash fixed-point problem.  Verification checks both the semantic freeze
gates and every file recorded by the bundle.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import zipfile
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

import numpy as np
import scipy

if __package__:
    from analysis.audit_combo_cache_reproducibility import verify_audit_report
else:
    from audit_combo_cache_reproducibility import verify_audit_report

ANALYSIS_ID = "combo-development-primary-r1000-v2"
DEVELOPMENT_YEARS = (2020, 2021, 2022, 2023)
PREDICTION_YEARS = (2021, 2022, 2023)
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
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = SAMPLE_SEED
MAX_ITERATIONS = 300
STABILITY_MAX_ITERATIONS = 600
GRADIENT_TOLERANCE = 1e-7
FUNCTION_TOLERANCE = 0.0
MAX_LINE_SEARCH_STEPS = 50
HISTORY_SIZE = 20
SELECTION_ALGORITHM = "lowest sha256(namespace, seed, relative_source), v1"
LAMBDA_SELECTION_RECORD = (
    "base-only pooled 2021-2023 OOT log loss; within 1e-6 choose larger lambda"
)
TURN_BINS = ("1-6", "7-9", "10-12", "13+")
METRIC_NAMES = ("auc", "log_loss", "brier", "ece", "macro_concordance")
RESULT_CODE_PATHS = (
    "pyproject.toml",
    "uv.lock",
    "src/mahjong_analysis/combo_prediction.py",
    "src/mahjong_analysis/combo_bootstrap.py",
    "src/mahjong_analysis/combo_sparse_pipeline.py",
    "analysis/analyze_combo_freeze_development.py",
)
BUNDLE_SCHEMA_VERSION = 1
BUNDLE_STATUS = "GO_FOR_2024_VALIDATION"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = PROJECT_ROOT / "outputs" / f"{ANALYSIS_ID}.json"
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "combo-development-sample-2020-2023-random1000.manifest.json"
)
DEFAULT_CACHE_DIR = (
    PROJECT_ROOT / "data" / "processed" / "combo-development-primary-r1000-v1"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / f"{ANALYSIS_ID}.freeze.json"
DEFAULT_CACHE_AUDIT = (
    PROJECT_ROOT / "outputs" / "combo-development-cache-reproducibility-v1.json"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and atomically seal the pre-2024 combo result."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cache-audit", type=Path)
    parser.add_argument(
        "--cache",
        action="append",
        default=[],
        metavar="YEAR=PATH",
        help="repeat once for each of 2020, 2021, 2022, and 2023",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--verify",
        type=Path,
        metavar="BUNDLE",
        help="verify an existing bundle instead of creating one",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    if args.verify is not None:
        if (
            args.result is not None
            or args.manifest is not None
            or args.cache
            or args.cache_audit is not None
        ):
            raise ValueError(
                "--verify cannot be combined with result, manifest, cache, "
                "or audit inputs"
            )
        bundle_path = args.verify.resolve()
        verified = verify_freeze_bundle(bundle_path, project_root=project_root)
        print(
            f"verified bundle_sha256={file_sha256(bundle_path)} "
            f"payload_sha256={verified['payload_sha256']}",
            file=sys.stderr,
            flush=True,
        )
        return 0

    result_path = (args.result or DEFAULT_RESULT).resolve()
    manifest_path = (args.manifest or DEFAULT_MANIFEST).resolve()
    cache_paths = (
        _parse_cache_arguments(args.cache)
        if args.cache
        else {
            year: (DEFAULT_CACHE_DIR / f"{year}-conventional-simple.npz").resolve()
            for year in DEVELOPMENT_YEARS
        }
    )
    output_path = (args.output or DEFAULT_OUTPUT).resolve()
    bundle = create_freeze_bundle(
        result_path=result_path,
        manifest_path=manifest_path,
        cache_paths=cache_paths,
        project_root=project_root,
        output_path=output_path,
        cache_audit_path=(args.cache_audit or DEFAULT_CACHE_AUDIT).resolve(),
    )
    print(
        f"wrote bundle_sha256={file_sha256(output_path)} "
        f"payload_sha256={bundle['payload_sha256']}",
        file=sys.stderr,
        flush=True,
    )
    return 0


def create_freeze_bundle(
    *,
    result_path: Path,
    manifest_path: Path,
    cache_paths: Mapping[int, Path],
    project_root: Path,
    output_path: Path,
    cache_audit_path: Path | None = None,
) -> dict[str, Any]:
    """Validate all fixed gates and atomically write a freeze bundle."""

    root = project_root.resolve()
    result = _read_json_object(result_path)
    manifest = _read_json_object(manifest_path)
    _ensure_finite_json_numbers(result, "result")
    _ensure_finite_json_numbers(manifest, "manifest")
    _validate_manifest(manifest)
    selected_l2 = _validate_result(
        result,
        manifest=manifest,
        manifest_file_sha256=file_sha256(manifest_path),
        project_root=root,
    )
    normalized_caches = _validate_caches(
        cache_paths,
        result=result,
        project_root=root,
    )
    audit_path = (
        cache_audit_path or root / "outputs" / DEFAULT_CACHE_AUDIT.name
    ).resolve()
    _validate_cache_audit_binding(
        audit_path,
        project_root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        cache_paths=normalized_caches,
        result=result,
    )

    input_paths = {
        "result": _target_record(result_path, root),
        "manifest": _target_record(manifest_path, root),
        "cache_audit": _target_record(audit_path, root),
        "caches": {
            str(year): _target_record(normalized_caches[year], root)
            for year in DEVELOPMENT_YEARS
        },
    }
    code_and_specs = _code_and_spec_records(root)
    payload: dict[str, Any] = {
        "analysis_id": ANALYSIS_ID,
        "status": BUNDLE_STATUS,
        "selected_l2": selected_l2,
        "development_inputs": input_paths,
        "code_and_specs": code_and_specs,
        "validation_2024_protocol": _validation_2024_protocol(selected_l2),
    }
    wrapper = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "payload_sha256": canonical_json_sha256(payload),
        "payload": payload,
    }

    output = output_path.resolve()
    _relative_target(output, root)
    input_targets = _flatten_target_paths(input_paths)
    if _relative_target(output, root) in input_targets:
        raise ValueError("bundle output must not overwrite a hashed input")
    _atomic_write_json(output, wrapper)
    return wrapper


def verify_freeze_bundle(
    bundle_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Fail closed unless an existing bundle and all its targets still match."""

    root = project_root.resolve()
    bundle = _read_json_object(bundle_path)
    if set(bundle) != {"schema_version", "payload_sha256", "payload"}:
        raise ValueError("bundle has an unexpected top-level shape")
    _require_equal(
        bundle["schema_version"], BUNDLE_SCHEMA_VERSION, "bundle.schema_version"
    )
    payload = _require_mapping(bundle["payload"], "bundle.payload")
    digest = canonical_json_sha256(payload)
    _require_equal(bundle["payload_sha256"], digest, "bundle.payload_sha256")
    _require_equal(payload.get("analysis_id"), ANALYSIS_ID, "payload.analysis_id")
    _require_equal(payload.get("status"), BUNDLE_STATUS, "payload.status")

    protocol = _require_mapping(
        payload.get("validation_2024_protocol"),
        "payload.validation_2024_protocol",
    )
    selected_l2 = _finite_number(payload.get("selected_l2"), "payload.selected_l2")
    _require_equal(
        protocol,
        _validation_2024_protocol(selected_l2),
        "payload.validation_2024_protocol",
    )

    development_inputs = _require_mapping(
        payload.get("development_inputs"), "payload.development_inputs"
    )
    _require_equal(
        set(development_inputs),
        {"result", "manifest", "caches", "cache_audit"},
        "payload.development_inputs",
    )
    _verify_target_tree(development_inputs, root)
    stored_code = _require_mapping(
        payload.get("code_and_specs"), "payload.code_and_specs"
    )
    expected_code = _code_and_spec_records(root)
    _require_equal(stored_code, expected_code, "payload.code_and_specs")

    result_record = _require_mapping(
        development_inputs.get("result"), "development_inputs.result"
    )
    manifest_record = _require_mapping(
        development_inputs.get("manifest"), "development_inputs.manifest"
    )
    cache_records = _require_mapping(
        development_inputs.get("caches"), "development_inputs.caches"
    )
    result_path = _path_from_record(result_record, root)
    manifest_path = _path_from_record(manifest_record, root)
    result = _read_json_object(result_path)
    manifest = _read_json_object(manifest_path)
    _ensure_finite_json_numbers(result, "result")
    _ensure_finite_json_numbers(manifest, "manifest")
    _validate_manifest(manifest)
    verified_l2 = _validate_result(
        result,
        manifest=manifest,
        manifest_file_sha256=file_sha256(manifest_path),
        project_root=root,
    )
    _require_equal(verified_l2, selected_l2, "payload.selected_l2")
    normalized_caches = _validate_caches(
        {
            int(year): _path_from_record(
                _require_mapping(record, f"development_inputs.caches.{year}"),
                root,
            )
            for year, record in cache_records.items()
        },
        result=result,
        project_root=root,
    )
    audit_record = _require_mapping(
        development_inputs.get("cache_audit"), "development_inputs.cache_audit"
    )
    _validate_cache_audit_binding(
        _path_from_record(audit_record, root),
        project_root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        cache_paths=normalized_caches,
        result=result,
    )
    return bundle


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    _require_equal(manifest.get("schema_version"), 2, "manifest.schema_version")
    _require_equal(
        manifest.get("selection_algorithm"),
        SELECTION_ALGORITHM,
        "manifest.selection_algorithm",
    )
    _require_equal(manifest.get("sample_seed"), SAMPLE_SEED, "manifest.sample_seed")
    _require_equal(
        manifest.get("files_per_year"), FILES_PER_YEAR, "manifest.files_per_year"
    )
    _require_equal(
        manifest.get("development_years"),
        list(DEVELOPMENT_YEARS),
        "manifest.development_years",
    )
    _require_equal(
        manifest.get("validation_2024_touched"),
        False,
        "manifest.validation_2024_touched",
    )
    _require_equal(
        manifest.get("holdout_2025_touched"),
        False,
        "manifest.holdout_2025_touched",
    )
    years = _require_mapping(manifest.get("years"), "manifest.years")
    _require_equal(
        set(years), {str(year) for year in DEVELOPMENT_YEARS}, "manifest.years"
    )
    for year in DEVELOPMENT_YEARS:
        entry = _require_mapping(years[str(year)], f"manifest.years.{year}")
        candidate_count = _integer(
            entry.get("candidate_count"), f"manifest.{year}.candidate_count"
        )
        if candidate_count < FILES_PER_YEAR:
            raise ValueError(f"manifest.{year}.candidate_count is too small")
        _sha256_text(
            entry.get("candidate_population_sha256"),
            f"manifest.{year}.candidate_population_sha256",
        )
        selected = entry.get("selected")
        if not isinstance(selected, list) or len(selected) != FILES_PER_YEAR:
            raise ValueError(f"manifest.{year}.selected must contain exactly 1000 rows")
        selection_hashes: list[str] = []
        relative_sources: set[str] = set()
        for index, raw_item in enumerate(selected):
            item = _require_mapping(raw_item, f"manifest.{year}.selected[{index}]")
            relative_source = item.get("relative_source")
            if not isinstance(relative_source, str) or not relative_source.startswith(
                f"{year}/"
            ):
                raise ValueError(
                    f"manifest.{year}.selected[{index}].relative_source is invalid"
                )
            if relative_source in relative_sources:
                raise ValueError(f"manifest.{year}.selected has a duplicate source")
            relative_sources.add(relative_source)
            selection_hashes.append(
                _sha256_text(
                    item.get("selection_sha256"),
                    f"manifest.{year}.selected[{index}].selection_sha256",
                )
            )
            _sha256_text(
                item.get("content_sha256"),
                f"manifest.{year}.selected[{index}].content_sha256",
            )
        if (
            selection_hashes != sorted(selection_hashes)
            or len(set(selection_hashes)) != FILES_PER_YEAR
        ):
            raise ValueError(
                f"manifest.{year}.selected must have unique sorted selection hashes"
            )


def _validate_result(
    result: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    manifest_file_sha256: str,
    project_root: Path,
) -> float:
    _require_equal(result.get("analysis_id"), ANALYSIS_ID, "result.analysis_id")
    _require_equal(result.get("status"), "GO_to_freeze", "result.status")
    scope = _require_mapping(result.get("scope"), "result.scope")
    expected_scope = {
        "development_years": list(DEVELOPMENT_YEARS),
        "prediction_years": list(PREDICTION_YEARS),
        "files_per_year": FILES_PER_YEAR,
        "fixed_primary_files_per_year": FILES_PER_YEAR,
        "sample_seed": SAMPLE_SEED,
        "label": LABEL,
        "base_model": BASE_MODEL,
        "challenger_model": CHALLENGER_MODEL,
        "primary_metric": "weighted_log_loss_delta",
        "validation_2024_touched": False,
        "holdout_2025_touched": False,
    }
    for key, expected in expected_scope.items():
        _require_equal(scope.get(key), expected, f"result.scope.{key}")

    result_manifest = _require_mapping(result.get("manifest"), "result.manifest")
    _require_equal(
        result_manifest.get("canonical_sha256"),
        canonical_json_sha256(manifest),
        "result.manifest.canonical_sha256",
    )
    _require_equal(
        result_manifest.get("file_sha256"),
        manifest_file_sha256,
        "result.manifest.file_sha256",
    )

    optimizer = _require_mapping(result.get("optimizer"), "result.optimizer")
    _require_equal(
        optimizer.get("type"),
        "SciPy L-BFGS-B with analytic gradient",
        "result.optimizer.type",
    )
    _require_equal(
        optimizer.get("objective"),
        "weighted mean log loss + l2/2 * ||beta||^2",
        "result.optimizer.objective",
    )
    _require_equal(
        optimizer.get("intercept_penalized"),
        False,
        "result.optimizer.intercept_penalized",
    )
    _require_equal(
        optimizer.get("lambda_grid"), list(L2_GRID), "result.optimizer.lambda_grid"
    )
    _require_equal(
        optimizer.get("lambda_selection"),
        LAMBDA_SELECTION_RECORD,
        "result.optimizer.lambda_selection",
    )
    _require_equal(
        optimizer.get("max_iterations"),
        MAX_ITERATIONS,
        "result.optimizer.max_iterations",
    )
    _require_equal(
        optimizer.get("stability_max_iterations"),
        STABILITY_MAX_ITERATIONS,
        "result.optimizer.stability_max_iterations",
    )
    _require_equal(
        optimizer.get("gradient_tolerance"),
        GRADIENT_TOLERANCE,
        "result.optimizer.gradient_tolerance",
    )
    _require_equal(
        optimizer.get("function_tolerance"),
        FUNCTION_TOLERANCE,
        "result.optimizer.function_tolerance",
    )
    selected_l2 = _finite_number(
        optimizer.get("selected_l2"), "result.optimizer.selected_l2"
    )
    if selected_l2 not in L2_GRID:
        raise ValueError("result.optimizer.selected_l2 is outside the closed grid")
    lambda_scores = _validate_lambda_scores(
        optimizer.get("lambda_scores"), optimizer.get("lambda_fold_diagnostics")
    )
    minimum = min(lambda_scores.values())
    independently_selected = max(
        l2 for l2, score in lambda_scores.items() if score <= minimum + L2_TIE_TOLERANCE
    )
    _require_equal(
        selected_l2,
        independently_selected,
        "result.optimizer.selected_l2",
    )

    folds = _require_mapping(result.get("folds"), "result.folds")
    _require_equal(set(folds), {str(year) for year in PREDICTION_YEARS}, "result.folds")
    for test_year in PREDICTION_YEARS:
        fold = _require_mapping(folds[str(test_year)], f"result.folds.{test_year}")
        _require_equal(
            fold.get("train_years"),
            list(range(2020, test_year)),
            f"result.folds.{test_year}.train_years",
        )
        _require_equal(
            fold.get("test_year"), test_year, f"result.folds.{test_year}.test_year"
        )
        models = _require_mapping(
            fold.get("models"), f"result.folds.{test_year}.models"
        )
        _require_equal(
            set(models),
            {BASE_MODEL, CHALLENGER_MODEL},
            f"result.folds.{test_year}.models",
        )
        for model_name in (BASE_MODEL, CHALLENGER_MODEL):
            model = _require_mapping(
                models[model_name], f"result.folds.{test_year}.models.{model_name}"
            )
            _finite_number(
                model.get("intercept"),
                f"result.folds.{test_year}.models.{model_name}.intercept",
            )
            _validate_fit_diagnostics(
                model.get("diagnostics"),
                limit=MAX_ITERATIONS,
                field=f"result.folds.{test_year}.models.{model_name}.diagnostics",
                selected_l2=selected_l2,
            )
        _validate_stability(
            fold.get("stability_check"),
            field=f"result.folds.{test_year}.stability_check",
            selected_l2=selected_l2,
        )
        metrics = _validate_metric_pair(fold, f"result.folds.{test_year}")
        for count in ("row_count", "decision_count"):
            _require_equal(
                fold.get(count), metrics[count], f"result.folds.{test_year}.{count}"
            )
        games = _positive_count(fold.get("game_count"), f"fold {test_year}.game_count")
        if games > metrics["decision_count"]:
            raise ValueError(f"fold {test_year} game_count exceeds decision_count")

    pooled = _require_mapping(result.get("pooled"), "result.pooled")
    pooled_metrics = _validate_metric_pair(pooled, "result.pooled")
    _close_number(
        pooled_metrics["log_loss"],
        lambda_scores[selected_l2],
        "result.pooled.selected_lambda_base_log_loss",
    )
    for count in ("row_count", "decision_count"):
        _require_equal(
            pooled_metrics[count],
            sum(fold[count] for fold in folds.values()),
            f"result.pooled.{count}",
        )
    _validate_bootstrap(pooled, folds=folds, pooled_metrics=pooled_metrics)
    _validate_result_environment(result.get("environment"), project_root)
    freeze_checks = _require_mapping(
        result.get("freeze_checks"), "result.freeze_checks"
    )
    for key in (
        "all_primary_fits_accepted",
        "all_doubled_iteration_stability_checks_accepted",
        "fixed_configuration",
        "bootstrap_replicates_exactly_2000",
        "bootstrap_seed_fixed",
        "optimizer_limits_fixed",
        "full_1000_file_prefix",
        "freeze_ready",
    ):
        _require_equal(freeze_checks.get(key), True, f"result.freeze_checks.{key}")
    return selected_l2


def _validate_result_environment(raw: Any, project_root: Path) -> None:
    environment = _require_mapping(raw, "result.environment")
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    for name, expected in versions.items():
        _require_equal(environment.get(name), expected, f"result.environment.{name}")
    expected_hashes = {
        str(Path(relative)): file_sha256(project_root / relative)
        for relative in RESULT_CODE_PATHS
    }
    hashes = _require_mapping(
        environment.get("file_sha256"), "result.environment.file_sha256"
    )
    _require_equal(dict(hashes), expected_hashes, "result.environment.file_sha256")
    argv = environment.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(value, str) or not value for value in argv)
    ):
        raise ValueError("result.environment.argv must contain nonempty strings")


def _positive_count(raw: Any, field: str) -> int:
    value = _integer(raw, field)
    if value < 1:
        raise ValueError(f"{field} must be positive")
    return value


def _close_number(actual: Any, expected: float, field: str) -> None:
    value = _finite_number(actual, field)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{field} mismatch: expected {expected!r}, got {value!r}")


def _unit_interval(raw: Any, field: str) -> float:
    value = _finite_number(raw, field)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be between zero and one")
    return value


def _validate_calibration(raw: Any, *, row_count: int, field: str) -> None:
    calibration = _require_mapping(raw, field)
    expected_keys = {
        "joint_status",
        "recalibration_intercept",
        "recalibration_slope",
        "calibration_in_the_large",
        "iterations",
        "clip_epsilon",
        "clipped_low",
        "clipped_high",
    }
    _require_equal(set(calibration), expected_keys, field)
    statuses = {
        "ok",
        "single_class",
        "constant_logit",
        "complete_separation",
        "quasi_separation",
        "singular_hessian",
        "line_search_failed",
        "non_converged",
    }
    status = calibration["joint_status"]
    if not isinstance(status, str) or status not in statuses:
        raise ValueError(f"{field}.joint_status is unknown")
    for name in ("recalibration_intercept", "recalibration_slope"):
        if status == "ok":
            _finite_number(calibration[name], f"{field}.{name}")
        elif calibration[name] is not None:
            raise ValueError(f"{field}.{name} must be missing for {status}")
    if status == "single_class":
        if calibration["calibration_in_the_large"] is not None:
            raise ValueError(f"{field}.calibration_in_the_large must be missing")
    else:
        _finite_number(
            calibration["calibration_in_the_large"], f"{field}.calibration_in_the_large"
        )
    iterations = _integer(calibration["iterations"], f"{field}.iterations")
    if not 0 <= iterations <= 100 or (status == "ok" and iterations == 0):
        raise ValueError(f"{field}.iterations is outside the fixed bounds")
    _require_equal(calibration["clip_epsilon"], 1e-15, f"{field}.clip_epsilon")
    clipped = []
    for name in ("clipped_low", "clipped_high"):
        count = _integer(calibration[name], f"{field}.{name}")
        if count < 0:
            raise ValueError(f"{field}.{name} must be nonnegative")
        clipped.append(count)
    if sum(clipped) > row_count:
        raise ValueError(f"{field} clipped counts exceed row_count")


def _validate_binary_metrics(raw: Any, field: str) -> Mapping[str, Any]:
    metrics = _require_mapping(raw, field)
    _require_equal(
        set(metrics),
        {
            *METRIC_NAMES,
            "row_count",
            "decision_count",
            "positive_rate",
            "calibration",
            "macro_concordance_decisions",
        },
        field,
    )
    rows = _positive_count(metrics["row_count"], f"{field}.row_count")
    decisions = _positive_count(metrics["decision_count"], f"{field}.decision_count")
    if decisions > rows:
        raise ValueError(f"{field}.decision_count exceeds row_count")
    rate = _unit_interval(metrics["positive_rate"], f"{field}.positive_rate")
    if rate in (0.0, 1.0):
        if metrics["auc"] is not None:
            raise ValueError(f"{field}.auc must be missing for a single class")
    else:
        _unit_interval(metrics["auc"], f"{field}.auc")
    if _finite_number(metrics["log_loss"], f"{field}.log_loss") < 0:
        raise ValueError(f"{field}.log_loss must be nonnegative")
    for name in ("brier", "ece"):
        _unit_interval(metrics[name], f"{field}.{name}")
    coverage = _integer(
        metrics["macro_concordance_decisions"], f"{field}.macro_concordance_decisions"
    )
    if not 0 <= coverage <= decisions:
        raise ValueError(f"{field}.macro_concordance_decisions is invalid")
    if coverage == 0:
        if metrics["macro_concordance"] is not None:
            raise ValueError(
                f"{field}.macro_concordance must be missing at zero coverage"
            )
    else:
        _unit_interval(metrics["macro_concordance"], f"{field}.macro_concordance")
    _validate_calibration(
        metrics["calibration"], row_count=rows, field=f"{field}.calibration"
    )
    if (metrics["calibration"]["joint_status"] == "single_class") != (
        rate in (0.0, 1.0)
    ):
        raise ValueError(
            f"{field}.calibration single_class status conflicts with positive_rate"
        )
    return metrics


def _validate_metric_pair(scope: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    models = _require_mapping(scope.get("models"), f"{field}.models")
    _require_equal(set(models), {BASE_MODEL, CHALLENGER_MODEL}, f"{field}.models")
    metrics = {}
    for name in (BASE_MODEL, CHALLENGER_MODEL):
        model = _require_mapping(models[name], f"{field}.models.{name}")
        metrics[name] = _validate_binary_metrics(
            model.get("metrics"), f"{field}.models.{name}.metrics"
        )
    base, challenger = metrics[BASE_MODEL], metrics[CHALLENGER_MODEL]
    for name in (
        "row_count",
        "decision_count",
        "positive_rate",
        "macro_concordance_decisions",
    ):
        _require_equal(challenger[name], base[name], f"{field}.paired.{name}")
    delta = _require_mapping(scope.get("delta"), f"{field}.delta")
    _require_equal(set(delta), set(METRIC_NAMES), f"{field}.delta")
    for name in METRIC_NAMES:
        if base[name] is None or challenger[name] is None:
            if delta[name] is not None:
                raise ValueError(f"{field}.delta.{name} must be missing")
        else:
            _close_number(
                delta[name], challenger[name] - base[name], f"{field}.delta.{name}"
            )
    return base


def _validate_lambda_scores(
    raw_scores: Any, raw_diagnostics: Any
) -> dict[float, float]:
    scores = _require_mapping(raw_scores, "result.optimizer.lambda_scores")
    diagnostics = _require_mapping(
        raw_diagnostics, "result.optimizer.lambda_fold_diagnostics"
    )
    _require_equal(
        set(diagnostics),
        {str(year) for year in PREDICTION_YEARS},
        "result.optimizer.lambda_fold_diagnostics.years",
    )
    keys = {format(value, ".12g") for value in L2_GRID}
    _require_equal(set(scores), keys, "result.optimizer.lambda_scores")
    all_accepted = {key: True for key in keys}
    for year, raw_fold in diagnostics.items():
        fold = _require_mapping(raw_fold, f"lambda diagnostics {year}")
        _require_equal(set(fold), keys, f"lambda diagnostics {year}.grid")
        for key, raw_fit in fold.items():
            field = f"lambda diagnostics {year}.{key}"
            fit = _require_mapping(raw_fit, field)
            if not isinstance(fit.get("accepted"), bool) or not isinstance(
                fit.get("converged"), bool
            ):
                raise ValueError(
                    f"{field} acceptance and convergence must be booleans"
                )
            iterations = _integer(fit.get("iterations"), f"{field}.iterations")
            if iterations < 0:
                raise ValueError(f"{field}.iterations must be nonnegative")
            gradient = _finite_number(
                fit.get("final_gradient_inf_norm"), f"{field}.gradient"
            )
            if gradient < 0:
                raise ValueError(f"{field}.gradient must be nonnegative")
            for name in (
                "initial_objective",
                "final_objective",
                "initial_gradient_inf_norm",
                "total_weight",
                "l2",
            ):
                value = _finite_number(fit.get(name), f"{field}.{name}")
                if value < 0 or (name == "total_weight" and value == 0):
                    raise ValueError(f"{field}.{name} is outside its valid range")
            _require_equal(fit["l2"], float(key), f"{field}.l2")
            _integer(fit.get("status"), f"{field}.status")
            if not isinstance(fit.get("message"), str) or not fit["message"]:
                raise ValueError(f"{field}.message must be a nonempty diagnostic")
            for name in (
                "function_evaluations",
                "gradient_evaluations",
                "row_count",
                "active_feature_count",
            ):
                _positive_count(fit.get(name), f"{field}.{name}")
            expected_acceptance = (
                fit["converged"]
                and iterations < MAX_ITERATIONS
                and gradient <= GRADIENT_TOLERANCE
            )
            _require_equal(fit["accepted"], expected_acceptance, f"{field}.accepted")
            all_accepted[key] &= fit["accepted"]
    parsed: dict[float, float] = {}
    for raw_key, raw_score in scores.items():
        try:
            key = float(raw_key)
        except (TypeError, ValueError) as error:
            raise ValueError("lambda score keys must be numeric") from error
        if format(key, ".12g") != raw_key:
            raise ValueError("lambda score keys must use the canonical float format")
        if not all_accepted[raw_key]:
            if raw_score is not None:
                raise ValueError(f"ineligible lambda {raw_key} must have a null score")
            continue
        parsed[key] = _finite_number(
            raw_score, f"result.optimizer.lambda_scores.{raw_key}"
        )
        if parsed[key] < 0:
            raise ValueError(f"lambda {raw_key} log loss must be nonnegative")
    if not parsed:
        raise ValueError("no lambda passed all three convergence gates")
    return parsed


def _validate_fit_diagnostics(
    raw: Any, *, limit: int, field: str, selected_l2: float
) -> None:
    diagnostics = _require_mapping(raw, field)
    _require_equal(diagnostics.get("converged"), True, f"{field}.converged")
    iterations = _integer(diagnostics.get("iterations"), f"{field}.iterations")
    if not 0 <= iterations < limit:
        raise ValueError(f"{field}.iterations hit its fixed limit")
    gradient = _finite_number(
        diagnostics.get("final_gradient_inf_norm"),
        f"{field}.final_gradient_inf_norm",
    )
    if not 0 <= gradient <= GRADIENT_TOLERANCE:
        raise ValueError(f"{field}.final_gradient_inf_norm failed the gate")
    if (
        _finite_number(diagnostics.get("final_objective"), f"{field}.final_objective")
        < 0
    ):
        raise ValueError(f"{field}.final_objective must be nonnegative")
    _require_equal(diagnostics.get("l2"), selected_l2, f"{field}.l2")


def _validate_stability(raw: Any, *, field: str, selected_l2: float) -> None:
    stability = _require_mapping(raw, field)
    _require_equal(stability.get("accepted"), True, f"{field}.accepted")
    thresholds = _require_mapping(stability.get("thresholds"), f"{field}.thresholds")
    _require_equal(
        thresholds.get("prediction_max_absolute_difference"),
        1e-8,
        f"{field}.thresholds.prediction_max_absolute_difference",
    )
    _require_equal(
        thresholds.get("log_loss_absolute_difference"),
        1e-10,
        f"{field}.thresholds.log_loss_absolute_difference",
    )
    _require_equal(
        thresholds.get("gradient_inf_norm"),
        GRADIENT_TOLERANCE,
        f"{field}.thresholds.gradient_inf_norm",
    )
    for model_name in ("base", "challenger"):
        model = _require_mapping(stability.get(model_name), f"{field}.{model_name}")
        prediction_difference = _finite_number(
            model.get("prediction_max_absolute_difference"),
            f"{field}.{model_name}.prediction_max_absolute_difference",
        )
        loss_difference = _finite_number(
            model.get("log_loss_absolute_difference"),
            f"{field}.{model_name}.log_loss_absolute_difference",
        )
        if not 0 <= prediction_difference <= 1e-8 or not 0 <= loss_difference <= 1e-10:
            raise ValueError(f"{field}.{model_name} failed the stability gate")
        _validate_fit_diagnostics(
            model.get("primary_fit"),
            limit=MAX_ITERATIONS,
            field=f"{field}.{model_name}.primary_fit",
            selected_l2=selected_l2,
        )
        _validate_fit_diagnostics(
            model.get("doubled_iteration_fit"),
            limit=STABILITY_MAX_ITERATIONS,
            field=f"{field}.{model_name}.doubled_iteration_fit",
            selected_l2=selected_l2,
        )


def _validate_bootstrap(
    raw_pooled: Any,
    *,
    folds: Mapping[str, Any],
    pooled_metrics: Mapping[str, Any],
) -> None:
    pooled = _require_mapping(raw_pooled, "result.pooled")
    bootstrap = _require_mapping(
        pooled.get("joint_game_cluster_bootstrap"),
        "result.pooled.joint_game_cluster_bootstrap",
    )
    _require_equal(
        bootstrap.get("requested_replicates"),
        BOOTSTRAP_REPLICATES,
        "result.pooled.bootstrap.requested_replicates",
    )
    _require_equal(
        bootstrap.get("seed"), BOOTSTRAP_SEED, "result.pooled.bootstrap.seed"
    )
    _require_equal(
        bootstrap.get("stratified"), True, "result.pooled.bootstrap.stratified"
    )
    _require_equal(bootstrap.get("ece_bins"), 10, "result.pooled.bootstrap.ece_bins")
    cluster_count = _positive_count(
        bootstrap.get("cluster_count"), "result.pooled.bootstrap.cluster_count"
    )
    _require_equal(
        bootstrap.get("group_order"),
        list(TURN_BINS),
        "result.pooled.bootstrap.group_order",
    )
    raw_counts = bootstrap.get("stratum_cluster_counts")
    if not isinstance(raw_counts, list) or len(raw_counts) != len(PREDICTION_YEARS):
        raise ValueError(
            "result.pooled.bootstrap.stratum_cluster_counts must be a list"
        )
    total_clusters = 0
    for year, item in zip(PREDICTION_YEARS, raw_counts, strict=True):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("bootstrap stratum counts must be year/count pairs")
        _require_equal(item[0], str(year), "bootstrap stratum year/order")
        count = _positive_count(item[1], f"bootstrap stratum {year} count")
        _require_equal(
            count, folds[str(year)]["game_count"], f"bootstrap stratum {year} count"
        )
        total_clusters += count
    _require_equal(total_clusters, cluster_count, "bootstrap stratum count sum")
    comparisons = bootstrap.get("comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != 1:
        raise ValueError("result.pooled.bootstrap must contain exactly one comparison")
    comparison = _require_mapping(
        comparisons[0], "result.pooled.bootstrap.comparisons[0]"
    )
    _require_equal(comparison.get("name"), COMPARISON_NAME, "bootstrap.comparison.name")
    _require_equal(
        comparison.get("base_model"), BASE_MODEL, "bootstrap.comparison.base_model"
    )
    _require_equal(
        comparison.get("challenger_model"),
        CHALLENGER_MODEL,
        "bootstrap.comparison.challenger_model",
    )
    overall = _validate_metric_intervals(comparison.get("overall"), "bootstrap.overall")
    for metric in METRIC_NAMES:
        _close_number(
            overall[metric]["point"],
            pooled["delta"][metric],
            f"bootstrap.overall.{metric}.point",
        )
    groups = comparison.get("by_group")
    if not isinstance(groups, list) or len(groups) != len(TURN_BINS):
        raise ValueError("bootstrap by_group must contain exactly four bins")
    group_metrics = {}
    row_sum = decision_sum = 0
    for name, raw_group in zip(TURN_BINS, groups, strict=True):
        field = f"bootstrap.by_group.{name}"
        group = _require_mapping(raw_group, field)
        _require_equal(
            set(group),
            {"group", "row_count", "decision_count", "cluster_count", "metrics"},
            field,
        )
        _require_equal(group["group"], name, f"{field}.group/order")
        rows = _positive_count(group["row_count"], f"{field}.row_count")
        decisions = _positive_count(group["decision_count"], f"{field}.decision_count")
        games = _positive_count(group["cluster_count"], f"{field}.cluster_count")
        if not games <= decisions <= rows or games > cluster_count:
            raise ValueError(f"{field} has inconsistent row/decision/cluster counts")
        row_sum += rows
        decision_sum += decisions
        group_metrics[name] = _validate_metric_intervals(
            group["metrics"], f"{field}.metrics"
        )
    _require_equal(row_sum, pooled_metrics["row_count"], "bootstrap bin row count sum")
    _require_equal(
        decision_sum,
        pooled_metrics["decision_count"],
        "bootstrap bin decision count sum",
    )
    contrasts = comparison.get("contrasts")
    if not isinstance(contrasts, list) or len(contrasts) != 1:
        raise ValueError("bootstrap comparison must contain the fixed contrast")
    contrast = _require_mapping(contrasts[0], "bootstrap.comparison.contrasts[0]")
    _require_equal(contrast.get("name"), "late_minus_early", "bootstrap.contrast.name")
    _require_equal(contrast.get("left_group"), "13+", "bootstrap.contrast.left_group")
    _require_equal(contrast.get("right_group"), "1-6", "bootstrap.contrast.right_group")
    contrast_metrics = _validate_metric_intervals(
        contrast.get("metrics"), "bootstrap.contrast.metrics", difference_bound=2.0
    )
    for metric in METRIC_NAMES:
        expected = (
            group_metrics["13+"][metric]["point"]
            - group_metrics["1-6"][metric]["point"]
        )
        _close_number(
            contrast_metrics[metric]["point"],
            expected,
            f"bootstrap.contrast.{metric}.point",
        )


def _validate_metric_intervals(
    raw: Any, field: str, *, difference_bound: float = 1.0
) -> Mapping[str, Any]:
    metrics = _require_mapping(raw, field)
    _require_equal(set(metrics), set(METRIC_NAMES), field)
    for metric, raw_interval in metrics.items():
        label = f"{field}.{metric}"
        interval = _require_mapping(raw_interval, label)
        _require_equal(
            set(interval), {"point", "lower", "upper", "valid_replicates"}, label
        )
        _require_equal(
            interval["valid_replicates"],
            BOOTSTRAP_REPLICATES,
            f"{label}.valid_replicates",
        )
        values = {
            key: _finite_number(interval[key], f"{label}.{key}")
            for key in ("point", "lower", "upper")
        }
        if values["lower"] > values["upper"]:
            raise ValueError(f"{label} lower bound exceeds upper bound")
        if metric != "log_loss" and any(
            abs(value) > difference_bound for value in values.values()
        ):
            raise ValueError(f"{label} is outside the metric difference range")
    return metrics


def _validate_caches(
    cache_paths: Mapping[int, Path],
    *,
    result: Mapping[str, Any],
    project_root: Path,
) -> dict[int, Path]:
    _require_equal(set(cache_paths), set(DEVELOPMENT_YEARS), "cache years")
    result_caches = _require_mapping(
        result.get("cache_artifacts"), "result.cache_artifacts"
    )
    _require_equal(
        set(result_caches),
        {str(year) for year in DEVELOPMENT_YEARS},
        "result.cache_artifacts",
    )
    normalized: dict[int, Path] = {}
    for year in DEVELOPMENT_YEARS:
        path = cache_paths[year].resolve()
        _relative_target(path, project_root)
        if path.suffix.lower() != ".npz" or not path.is_file():
            raise ValueError(f"cache {year} must be an existing NPZ file")
        if not zipfile.is_zipfile(path):
            raise ValueError(f"cache {year} is not a valid NPZ/ZIP artifact")
        record = _require_mapping(
            result_caches[str(year)], f"result.cache_artifacts.{year}"
        )
        _require_equal(
            record.get("selected_source_count"),
            FILES_PER_YEAR,
            f"result.cache_artifacts.{year}.selected_source_count",
        )
        _require_equal(
            record.get("sha256"),
            file_sha256(path),
            f"result.cache_artifacts.{year}.sha256",
        )
        normalized[year] = path
    return normalized


def _validate_cache_audit_binding(
    audit_path: Path,
    *,
    project_root: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    cache_paths: Mapping[int, Path],
    result: Mapping[str, Any],
) -> None:
    """Verify reproduction evidence and bind it to these exact freeze inputs."""

    _relative_target(audit_path, project_root)
    wrapper = verify_audit_report(audit_path, project_root=project_root)
    _require_equal(wrapper.get("schema_version"), 1, "cache audit schema_version")
    report = _require_mapping(wrapper.get("report"), "cache audit report")
    _require_equal(
        wrapper.get("report_sha256"),
        canonical_json_sha256(report),
        "cache audit report_sha256",
    )
    _require_equal(
        report.get("audit_id"),
        "combo-development-cache-reproducibility-v1",
        "cache audit audit_id",
    )
    _require_equal(report.get("analysis_id"), ANALYSIS_ID, "cache audit analysis_id")
    _require_equal(report.get("status"), "PASS", "cache audit status")
    scope = _require_mapping(report.get("scope"), "cache audit scope")
    for key, expected in {
        "development_years": list(DEVELOPMENT_YEARS),
        "files_per_year": FILES_PER_YEAR,
        "sample_seed": SAMPLE_SEED,
        "cache_producer_analysis_id": "combo-development-primary-r1000-v1",
        "validation_2024_touched": False,
        "holdout_2025_touched": False,
    }.items():
        _require_equal(scope.get(key), expected, f"cache audit scope.{key}")

    manifest_record = _require_mapping(report.get("manifest"), "cache audit manifest")
    audit_manifest_path = _path_from_record(
        {key: manifest_record.get(key) for key in ("path", "sha256")}, project_root
    )
    _require_equal(
        audit_manifest_path, manifest_path.resolve(), "cache audit manifest path"
    )
    _require_equal(
        manifest_record.get("sha256"),
        file_sha256(manifest_path),
        "cache audit manifest SHA-256",
    )
    _require_equal(
        manifest_record.get("canonical_sha256"),
        canonical_json_sha256(manifest),
        "cache audit manifest canonical SHA-256",
    )
    _require_equal(
        manifest_record.get("schema_version"), 2, "cache audit manifest schema"
    )
    _require_equal(
        manifest_record.get("validation_status"), "PASS", "cache audit manifest status"
    )
    result_manifest = _require_mapping(result.get("manifest"), "result.manifest")
    _require_equal(
        manifest_record["sha256"],
        result_manifest.get("file_sha256"),
        "cache audit/result manifest SHA-256",
    )
    _require_equal(
        manifest_record["canonical_sha256"],
        result_manifest.get("canonical_sha256"),
        "cache audit/result canonical manifest SHA-256",
    )

    years = _require_mapping(report.get("years"), "cache audit years")
    _require_equal(
        set(years), {str(year) for year in DEVELOPMENT_YEARS}, "cache audit years"
    )
    result_caches = _require_mapping(
        result.get("cache_artifacts"), "result.cache_artifacts"
    )
    for year in DEVELOPMENT_YEARS:
        field = f"cache audit year {year}"
        year_report = _require_mapping(years[str(year)], field)
        cache_record = _require_mapping(year_report.get("cache"), f"{field}.cache")
        _require_equal(
            _path_from_record(cache_record, project_root),
            cache_paths[year].resolve(),
            f"{field}.cache path",
        )
        result_cache = _require_mapping(
            result_caches[str(year)], f"result cache {year}"
        )
        _require_equal(
            cache_record.get("sha256"),
            result_cache.get("sha256"),
            f"{field}.result cache SHA-256",
        )
        _require_equal(
            cache_record.get("sha256"),
            file_sha256(cache_paths[year]),
            f"{field}.cache SHA-256",
        )
        comparison = _require_mapping(
            year_report.get("comparison"), f"{field}.comparison"
        )
        _require_equal(comparison.get("status"), "PASS", f"{field}.comparison status")
        _require_equal(
            comparison.get("comparison"), "exact", f"{field}.comparison mode"
        )
        semantic_digest = _sha256_text(
            year_report.get("semantic_sha256"), f"{field}.semantic_sha256"
        )
        for key in ("cached_semantic_sha256", "rebuilt_semantic_sha256"):
            _require_equal(
                _sha256_text(comparison.get(key), f"{field}.{key}"),
                semantic_digest,
                f"{field}.{key}",
            )
        summary = _require_mapping(year_report.get("summary"), f"{field}.summary")
        for key in ("row_count", "feature_count", "selected_source_count"):
            _require_equal(
                summary.get(key), result_cache.get(key), f"{field}.summary.{key}"
            )
        _require_equal(
            summary.get("counts"), result_cache.get("counts"), f"{field}.summary.counts"
        )
        _require_equal(
            year_report.get("extraction_counts"),
            result_cache.get("counts"),
            f"{field}.extraction_counts",
        )


def _validation_2024_protocol(selected_l2: float) -> dict[str, Any]:
    return {
        "validation_year": 2024,
        "holdout_year": 2025,
        "training_years": list(DEVELOPMENT_YEARS),
        "files_per_year": FILES_PER_YEAR,
        "sample_seed": SAMPLE_SEED,
        "selection_algorithm": SELECTION_ALGORITHM,
        "manifest_schema_version": 2,
        "label": LABEL,
        "base_model": BASE_MODEL,
        "challenger_model": CHALLENGER_MODEL,
        "comparison": COMPARISON_NAME,
        "selected_l2": selected_l2,
        "lambda_selection_on_2024": False,
        "recalibration_on_2024": False,
        "optimizer": {
            "type": "SciPy L-BFGS-B with analytic gradient",
            "max_iterations": MAX_ITERATIONS,
            "gradient_tolerance": GRADIENT_TOLERANCE,
            "function_tolerance": FUNCTION_TOLERANCE,
            "max_line_search_steps": MAX_LINE_SEARCH_STEPS,
            "history_size": HISTORY_SIZE,
            "intercept_penalized": False,
        },
        "bootstrap": {
            "cluster": "game",
            "paired": True,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "refit": False,
        },
        "primary_metric": "weighted_log_loss_delta",
        "success_rule": "point_estimate < 0 and percentile_95_upper < 0",
        "validation_2024_touched_at_freeze": False,
        "holdout_2025_touched_at_freeze": False,
    }


def _code_and_spec_records(project_root: Path) -> dict[str, dict[str, str]]:
    required = (
        project_root / "pyproject.toml",
        project_root / "uv.lock",
        project_root / "docs" / "specs" / "combo-prediction-freeze.md",
        project_root / "docs" / "specs" / "combo-2024-validation.md",
        project_root / "analysis" / "analyze_combo_freeze_development.py",
        project_root / "analysis" / "freeze_combo_bundle.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"required freeze inputs are missing: {missing}")
    python_paths = sorted(
        {
            *project_root.joinpath("src", "mahjong_analysis").rglob("*.py"),
            *project_root.joinpath("analysis").rglob("*.py"),
        },
        key=lambda path: path.as_posix(),
    )
    spec_paths = sorted(
        project_root.joinpath("docs", "specs").rglob("*.md"),
        key=lambda path: path.as_posix(),
    )
    if not python_paths or not spec_paths:
        raise ValueError("Python source and specification inventories must be nonempty")
    return {
        "project_metadata": {
            _relative_target(path, project_root): file_sha256(path)
            for path in (project_root / "pyproject.toml", project_root / "uv.lock")
        },
        "python": {
            _relative_target(path, project_root): file_sha256(path)
            for path in python_paths
        },
        "specs": {
            _relative_target(path, project_root): file_sha256(path)
            for path in spec_paths
        },
    }


def _target_record(path: Path, project_root: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"hashed target does not exist: {resolved}")
    return {
        "path": _relative_target(resolved, project_root),
        "sha256": file_sha256(resolved),
    }


def _verify_target_tree(raw: Mapping[str, Any], project_root: Path) -> None:
    for name, value in raw.items():
        if name == "caches":
            caches = _require_mapping(value, "development_inputs.caches")
            _require_equal(
                set(caches), {str(year) for year in DEVELOPMENT_YEARS}, "cache records"
            )
            for year, cache_record in caches.items():
                _verify_target_record(
                    _require_mapping(cache_record, f"development_inputs.caches.{year}"),
                    project_root,
                )
        else:
            _verify_target_record(
                _require_mapping(value, f"development_inputs.{name}"), project_root
            )


def _verify_target_record(record: Mapping[str, Any], project_root: Path) -> None:
    path = _path_from_record(record, project_root)
    expected = _sha256_text(record.get("sha256"), f"hash for {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )


def _path_from_record(record: Mapping[str, Any], project_root: Path) -> Path:
    if set(record) != {"path", "sha256"}:
        raise ValueError("target records must contain exactly path and sha256")
    raw_path = record.get("path")
    if not isinstance(raw_path, str):
        raise ValueError("target record path must be a string")
    pure = PurePosixPath(raw_path)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"unsafe target path in bundle: {raw_path!r}")
    resolved = project_root.joinpath(*pure.parts).resolve()
    _relative_target(resolved, project_root)
    if not resolved.is_file():
        raise ValueError(f"bundle target is missing: {resolved}")
    return resolved


def _flatten_target_paths(inputs: Mapping[str, Any]) -> set[str]:
    paths = {
        str(_require_mapping(inputs["result"], "result record")["path"]),
        str(_require_mapping(inputs["manifest"], "manifest record")["path"]),
        str(_require_mapping(inputs["cache_audit"], "cache audit record")["path"]),
    }
    caches = _require_mapping(inputs["caches"], "cache records")
    paths.update(
        str(_require_mapping(record, f"cache {year}")["path"])
        for year, record in caches.items()
    )
    return paths


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = (
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        # Publish the complete bundle atomically and never replace an existing
        # seal, even if two creators race.
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"JSON input does not exist: {path}")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    loaded = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )
    if not isinstance(loaded, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return loaded


def canonical_json_sha256(value: Any) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256(rendered.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_cache_arguments(values: Sequence[str]) -> dict[int, Path]:
    parsed: dict[int, Path] = {}
    for value in values:
        raw_year, separator, raw_path = value.partition("=")
        if not separator:
            raise ValueError("--cache values must have the form YEAR=PATH")
        try:
            year = int(raw_year)
        except ValueError as error:
            raise ValueError(f"invalid cache year: {raw_year}") from error
        if year in parsed:
            raise ValueError(f"duplicate cache year: {year}")
        parsed[year] = Path(raw_path).resolve()
    _require_equal(set(parsed), set(DEVELOPMENT_YEARS), "--cache years")
    return parsed


def _relative_target(path: Path, project_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(project_root.resolve())
    except ValueError as error:
        raise ValueError(
            f"freeze target is outside the project root: {path}"
        ) from error
    if not relative.parts:
        raise ValueError("the project root itself cannot be a freeze target")
    return relative.as_posix()


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    # Keep malformed freeze evidence on the existing public ValueError contract.
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected or isinstance(actual, bool) != isinstance(expected, bool):
        raise ValueError(f"{field} mismatch: expected {expected!r}, got {actual!r}")


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _sha256_text(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _ensure_finite_json_numbers(value: Any, field: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _ensure_finite_json_numbers(child, f"{field}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _ensure_finite_json_numbers(child, f"{field}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{field} must be finite")


if __name__ == "__main__":
    raise SystemExit(main())
