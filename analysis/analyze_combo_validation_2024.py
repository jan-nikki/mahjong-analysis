"""Run the sealed, one-shot 2024 combo-model validation.

The normal-run ordering in this module is intentional.  The development freeze
bundle is verified, its four caches are loaded, and both fixed models pass their
solver gates before any function capable of enumerating 2024 is called.  A
separate receipt-recovery path verifies immutable artifacts without fitting or
accessing 2024/2025 raw data.  The 2025 holdout is neither parameterized nor
inspected by this runner.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np
import scipy

if __package__:
    from analysis.freeze_combo_bundle import (
        DEVELOPMENT_YEARS,
        canonical_json_sha256,
        file_sha256,
        verify_freeze_bundle,
    )
    from analysis.record_combo_preflight import verify_preflight_report
else:
    from freeze_combo_bundle import (
        DEVELOPMENT_YEARS,
        canonical_json_sha256,
        file_sha256,
        verify_freeze_bundle,
    )
    from record_combo_preflight import verify_preflight_report
from mahjong_analysis.combo_bootstrap import (
    GroupContrast,
    JointBootstrapResult,
    joint_paired_game_cluster_bootstrap,
)
from mahjong_analysis.combo_prediction import (
    DECISION_TURN_BINS,
    BinaryMetrics,
    SparseExample,
    evaluate_binary_predictions,
)
from mahjong_analysis.combo_sparse_pipeline import (
    SparseLogisticModel,
    SparseLogisticProblem,
    YearSparseArtifact,
    build_year_sparse_artifact,
    conventional_feature_mask,
    fit_sparse_problem,
    global_feature_vocabulary,
    load_year_sparse_artifact,
    prepare_logistic_problem,
    save_year_sparse_artifact,
)
from mahjong_analysis.combo_validation_2024 import (
    VALIDATION_SAMPLE_SEED,
    VALIDATION_SAMPLE_SIZE,
    VALIDATION_YEAR,
    build_validation_2024_manifest,
    collect_validation_2024_dataset,
    load_validation_2024_manifest,
    select_validation_2024_sample,
    write_validation_2024_manifest,
)

ANALYSIS_ID = "combo-validation-2024-r1000-v1"
EXPECTED_FREEZE_ANALYSIS_ID = "combo-development-primary-r1000-v2"
EXPECTED_FREEZE_STATUS = "GO_FOR_2024_VALIDATION"
LABEL = "ron_eligible"
BASE_MODEL = "conventional"
CHALLENGER_MODEL = "conventional_simple"
COMPARISON_NAME = "conventional_simple_minus_conventional"

BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20260923
MAX_ITERATIONS = 300
GRADIENT_TOLERANCE = 1e-7
FUNCTION_TOLERANCE = 0.0
MAX_LINE_SEARCH_STEPS = 50
HISTORY_SIZE = 20

METRIC_NAMES = ("auc", "log_loss", "brier", "ece", "macro_concordance")
CALIBRATION_STATUSES = {
    "ok",
    "single_class",
    "constant_logit",
    "complete_separation",
    "quasi_separation",
    "singular_hessian",
    "line_search_failed",
    "non_converged",
}

TOUCH_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
RECEIPT_SCHEMA_VERSION = 1

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ValidationPaths:
    """All paths accepted by the fixed runner; no statistical knobs live here."""

    project_root: Path
    freeze_bundle: Path
    preflight_report: Path
    raw_root: Path
    manifest: Path
    cache: Path
    touch_marker: Path
    output: Path
    receipt: Path


@dataclass(frozen=True)
class FrozenDevelopmentInputs:
    artifacts: tuple[YearSparseArtifact, ...]
    manifest_path: Path
    manifest_file_sha256: str
    manifest_canonical_sha256: str
    cache_records: tuple[tuple[int, str, str], ...]


@dataclass(frozen=True)
class FrozenModels:
    vocabulary: tuple[str, ...]
    problem: SparseLogisticProblem
    base: SparseLogisticModel
    challenger: SparseLogisticModel


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the sealed 2024 combo-model validation exactly once."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--freeze-bundle", type=Path)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--touch-marker", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument(
        "--recover-receipt",
        action="store_true",
        help=(
            "recover or verify only the receipt for an existing immutable "
            "result; never fit, evaluate, or inspect raw validation data"
        ),
    )
    return parser.parse_args(argv)


def paths_from_args(args: argparse.Namespace) -> ValidationPaths:
    root = args.project_root.resolve()

    def resolved(value: Path | None, default: Path) -> Path:
        return (value if value is not None else default).resolve()

    return ValidationPaths(
        project_root=root,
        freeze_bundle=resolved(
            args.freeze_bundle,
            root / "outputs" / "combo-development-primary-r1000-v2.freeze.json",
        ),
        preflight_report=resolved(
            args.preflight_report,
            root / "outputs" / "combo-pre2024-test-preflight-v1.json",
        ),
        raw_root=resolved(args.raw_root, root / "data" / "raw"),
        manifest=resolved(
            args.manifest,
            root / "outputs" / "combo-validation-2024-random1000.manifest.json",
        ),
        cache=resolved(
            args.cache,
            root
            / "data"
            / "processed"
            / "combo-validation-2024-r1000-v1"
            / "2024-conventional-simple.npz",
        ),
        touch_marker=resolved(
            args.touch_marker,
            root / "outputs" / "combo-validation-2024.touched.json",
        ),
        output=resolved(
            args.output,
            root / "outputs" / "combo-validation-2024-r1000-v1.json",
        ),
        receipt=resolved(
            args.receipt,
            root / "outputs" / "combo-validation-2024-r1000-v1.receipt.json",
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = paths_from_args(args)
    if args.recover_receipt:
        result, receipt = recover_validation_receipt(paths)
    else:
        result, receipt = run_validation(paths)
    print(
        f"status={result['status']} result_sha256={receipt['result']['sha256']}",
        file=sys.stderr,
        flush=True,
    )
    return 0


def run_validation(
    paths: ValidationPaths,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute the fixed protocol, failing closed at every integrity boundary."""

    started = perf_counter()

    # This must remain the first operation capable of authorizing validation.
    # In particular, no 2024 sampler/manifest loader/extractor appears above it.
    bundle = verify_freeze_bundle(
        paths.freeze_bundle,
        project_root=paths.project_root,
    )
    bundle_file_digest = file_sha256(paths.freeze_bundle)
    preflight_report = verify_preflight_report(
        paths.preflight_report,
        paths.project_root,
        {
            "bundle_file_sha256": bundle_file_digest,
            "payload_sha256": str(bundle["payload_sha256"]),
        },
    )
    preflight_report_digest = file_sha256(paths.preflight_report)
    payload, selected_l2 = _validated_bundle_protocol(bundle)

    # Reject a completed/partial output before loading caches or fitting.  A
    # result without a receipt can only be handled by --recover-receipt.
    _validate_destination_state(paths)
    development = _load_frozen_development_inputs(
        payload,
        project_root=paths.project_root,
    )
    models = _fit_frozen_models(development.artifacts, selected_l2=selected_l2)

    touch_document = _touch_document(
        paths,
        bundle_file_sha256=bundle_file_digest,
        payload_sha256=str(bundle["payload_sha256"]),
    )
    _ensure_touch_marker(paths.touch_marker, touch_document)

    sample, manifest_document = _load_or_create_validation_manifest(
        paths,
        bundle=bundle,
    )
    manifest_file_digest = file_sha256(paths.manifest)
    manifest_canonical_digest = canonical_json_sha256(manifest_document)
    validation_artifact, validation_counts = _load_or_create_validation_cache(
        paths,
        sample=sample,
        manifest_file_sha256=manifest_file_digest,
        manifest_canonical_sha256=manifest_canonical_digest,
        bundle_file_sha256=bundle_file_digest,
        payload_sha256=str(bundle["payload_sha256"]),
    )
    cache_file_digest = file_sha256(paths.cache)

    evaluation = _evaluate_fixed_predictions(
        models,
        validation_artifact,
    )
    primary = _primary_validation_decision(evaluation["bootstrap"])
    status = "VALIDATION_SUCCESS" if primary["success"] else "VALIDATION_FAILURE"

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "analysis_id": ANALYSIS_ID,
        "status": status,
        "primary_validation_success": primary["success"],
        "scope": {
            "training_years": list(DEVELOPMENT_YEARS),
            "validation_year": VALIDATION_YEAR,
            "selected_validation_sources": VALIDATION_SAMPLE_SIZE,
            "sample_seed": VALIDATION_SAMPLE_SEED,
            "label": LABEL,
            "base_model": BASE_MODEL,
            "challenger_model": CHALLENGER_MODEL,
            "comparison": COMPARISON_NAME,
            "validation_2024_touched": True,
            "holdout_2025_touched": False,
        },
        "freeze_bundle": {
            "path": _relative_path(paths.freeze_bundle, paths.project_root),
            "file_sha256": bundle_file_digest,
            "payload_sha256": bundle["payload_sha256"],
            "analysis_id": payload["analysis_id"],
            "status": payload["status"],
            "validation_2024_protocol": payload["validation_2024_protocol"],
            "code_and_specs": payload["code_and_specs"],
            "code_and_specs_sha256": canonical_json_sha256(payload["code_and_specs"]),
        },
        "preflight_tests": {
            "path": _relative_path(paths.preflight_report, paths.project_root),
            "file_sha256": preflight_report_digest,
            "status": preflight_report["status"],
            "pytest": preflight_report["pytest"],
            "test_inventory_sha256": preflight_report["tests"]["inventory_sha256"],
            "transcript_sha256": preflight_report["transcript"]["sha256"],
            "freeze_bundle": preflight_report["freeze_bundle"],
        },
        "development_inputs": {
            "manifest": {
                "path": _relative_path(
                    development.manifest_path,
                    paths.project_root,
                ),
                "file_sha256": development.manifest_file_sha256,
                "canonical_sha256": development.manifest_canonical_sha256,
            },
            "caches": [
                {"year": year, "path": path, "sha256": digest}
                for year, path, digest in development.cache_records
            ],
        },
        "validation_inputs": {
            "touch_marker": {
                "path": _relative_path(paths.touch_marker, paths.project_root),
                "sha256": file_sha256(paths.touch_marker),
            },
            "manifest": {
                "path": _relative_path(paths.manifest, paths.project_root),
                "file_sha256": manifest_file_digest,
                "canonical_sha256": manifest_canonical_digest,
                "candidate_count": sample.candidate_count,
                "selected_source_count": len(sample.relative_sources),
                "candidate_population_sha256": (sample.candidate_population_sha256),
            },
            "cache": {
                "path": _relative_path(paths.cache, paths.project_root),
                "sha256": cache_file_digest,
                "row_count": validation_artifact.row_count,
                "feature_count": validation_artifact.feature_count,
                "selected_source_count": validation_artifact.selected_source_count,
                "counts": validation_counts,
            },
        },
        "optimizer": {
            "type": "SciPy L-BFGS-B with analytic gradient",
            "objective": "weighted mean log loss + l2/2 * ||beta||^2",
            "selected_l2": selected_l2,
            "selected_l2_source": "verified freeze bundle only",
            "max_iterations": MAX_ITERATIONS,
            "gradient_tolerance": GRADIENT_TOLERANCE,
            "function_tolerance": FUNCTION_TOLERANCE,
            "max_line_search_steps": MAX_LINE_SEARCH_STEPS,
            "history_size": HISTORY_SIZE,
            "intercept_penalized": False,
            "models": {
                BASE_MODEL: _model_document(models.base),
                CHALLENGER_MODEL: _model_document(models.challenger),
            },
        },
        "feature_vocabulary": {
            "source": "frozen 2020-2023 development caches only",
            "feature_count": len(models.vocabulary),
            "sha256": canonical_json_sha256(list(models.vocabulary)),
            "validation_only_feature_count": len(
                evaluation["validation_only_features"]
            ),
            "validation_only_features_sha256": canonical_json_sha256(
                list(evaluation["validation_only_features"])
            ),
            "validation_only_features_coefficient": 0.0,
        },
        "predictions": {
            BASE_MODEL: {
                "sha256": evaluation["prediction_sha256"][BASE_MODEL],
                "metrics": asdict(evaluation["metrics"][BASE_MODEL]),
            },
            CHALLENGER_MODEL: {
                "sha256": evaluation["prediction_sha256"][CHALLENGER_MODEL],
                "metrics": asdict(evaluation["metrics"][CHALLENGER_MODEL]),
            },
            "recalibration_applied": False,
        },
        "delta": _metric_delta(
            evaluation["metrics"][BASE_MODEL],
            evaluation["metrics"][CHALLENGER_MODEL],
        ),
        "turn_bins": evaluation["turn_bins"],
        "bootstrap_protocol": {
            "cluster": "game",
            "paired": True,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "refit": False,
            "interval": "pointwise percentile 95%",
            "shared_multiplicity_across_models_and_turn_bins": True,
        },
        "joint_game_cluster_bootstrap": asdict(evaluation["bootstrap"]),
        "primary_decision": primary,
        "counts": {
            "games": len({example.cluster_id for example in evaluation["examples"]}),
            "decisions": len({example.group_id for example in evaluation["examples"]}),
            "candidate_rows": len(evaluation["examples"]),
        },
        "environment": {
            "argv": [str(value) for value in sys.argv],
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "elapsed_seconds": perf_counter() - started,
    }

    result = _json_normalized_object(result)
    _validate_completed_result_for_receipt(
        paths,
        result,
        bundle=bundle,
        payload=payload,
        bundle_file_sha256=bundle_file_digest,
        preflight_report=preflight_report,
        development=development,
        validation_artifact=validation_artifact,
    )
    _atomic_write_new_json(paths.output, result)
    result_digest = file_sha256(paths.output)
    receipt = _receipt_document(paths, result, result_sha256=result_digest)
    _atomic_write_new_json(paths.receipt, receipt)
    return result, receipt


def recover_validation_receipt(
    paths: ValidationPaths,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recover a receipt without fitting, evaluating, or reading validation raw."""

    bundle = verify_freeze_bundle(
        paths.freeze_bundle,
        project_root=paths.project_root,
    )
    bundle_file_digest = file_sha256(paths.freeze_bundle)
    payload, _selected_l2 = _validated_bundle_protocol(bundle)
    preflight_report = verify_preflight_report(
        paths.preflight_report,
        paths.project_root,
        {
            "bundle_file_sha256": bundle_file_digest,
            "payload_sha256": str(bundle["payload_sha256"]),
        },
    )
    if not paths.output.is_file():
        raise FileNotFoundError(
            f"immutable validation result does not exist: {paths.output}"
        )
    result = _read_json_object(paths.output)
    _validate_completed_result_for_receipt(
        paths,
        result,
        bundle=bundle,
        payload=payload,
        bundle_file_sha256=bundle_file_digest,
        preflight_report=preflight_report,
    )
    result_digest = file_sha256(paths.output)
    expected_receipt = _receipt_document(
        paths,
        result,
        result_sha256=result_digest,
    )
    if paths.receipt.exists():
        actual_receipt = _read_json_object(paths.receipt)
        _require_equal(
            actual_receipt,
            expected_receipt,
            "existing validation receipt",
        )
        verify_validation_receipt(
            paths.receipt,
            project_root=paths.project_root,
        )
        return result, actual_receipt

    _atomic_write_new_json(paths.receipt, expected_receipt)
    verify_validation_receipt(
        paths.receipt,
        project_root=paths.project_root,
    )
    return result, expected_receipt


def _receipt_document(
    paths: ValidationPaths,
    result: Mapping[str, Any],
    *,
    result_sha256: str,
) -> dict[str, Any]:
    freeze = _require_mapping(result.get("freeze_bundle"), "result.freeze_bundle")
    _require_equal(
        set(freeze),
        {
            "path",
            "file_sha256",
            "payload_sha256",
            "analysis_id",
            "status",
            "validation_2024_protocol",
            "code_and_specs",
            "code_and_specs_sha256",
        },
        "result.freeze_bundle fields",
    )
    preflight = _require_mapping(
        result.get("preflight_tests"),
        "result.preflight_tests",
    )
    _require_equal(
        set(preflight),
        {
            "path",
            "file_sha256",
            "status",
            "pytest",
            "test_inventory_sha256",
            "transcript_sha256",
            "freeze_bundle",
        },
        "result.preflight_tests fields",
    )
    validation_inputs = _require_mapping(
        result.get("validation_inputs"),
        "result.validation_inputs",
    )
    touch = _require_mapping(
        validation_inputs.get("touch_marker"),
        "result.validation_inputs.touch_marker",
    )
    manifest = _require_mapping(
        validation_inputs.get("manifest"),
        "result.validation_inputs.manifest",
    )
    cache = _require_mapping(
        validation_inputs.get("cache"),
        "result.validation_inputs.cache",
    )
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "analysis_id": ANALYSIS_ID,
        "status": "SEALED_VALIDATION_RESULT",
        "result": {
            "path": _relative_path(paths.output, paths.project_root),
            "sha256": _sha256_text(result_sha256, "result SHA-256"),
        },
        "freeze_bundle": {
            "path": _relative_path(paths.freeze_bundle, paths.project_root),
            "sha256": _sha256_text(
                freeze.get("file_sha256"),
                "result.freeze_bundle.file_sha256",
            ),
            "payload_sha256": _sha256_text(
                freeze.get("payload_sha256"),
                "result.freeze_bundle.payload_sha256",
            ),
        },
        "preflight_report": {
            "path": _relative_path(paths.preflight_report, paths.project_root),
            "sha256": _sha256_text(
                preflight.get("file_sha256"),
                "result.preflight_tests.file_sha256",
            ),
        },
        "touch_marker": {
            "path": _relative_path(paths.touch_marker, paths.project_root),
            "sha256": _sha256_text(
                touch.get("sha256"),
                "result.validation_inputs.touch_marker.sha256",
            ),
        },
        "validation_manifest": {
            "path": _relative_path(paths.manifest, paths.project_root),
            "sha256": _sha256_text(
                manifest.get("file_sha256"),
                "result.validation_inputs.manifest.file_sha256",
            ),
        },
        "validation_cache": {
            "path": _relative_path(paths.cache, paths.project_root),
            "sha256": _sha256_text(
                cache.get("sha256"),
                "result.validation_inputs.cache.sha256",
            ),
        },
        "validation_2024_touched": True,
        "holdout_2025_touched": False,
    }


def _validate_completed_result_for_receipt(
    paths: ValidationPaths,
    result: Mapping[str, Any],
    *,
    bundle: Mapping[str, Any],
    payload: Mapping[str, Any],
    bundle_file_sha256: str,
    preflight_report: Mapping[str, Any],
    development: FrozenDevelopmentInputs | None = None,
    validation_artifact: YearSparseArtifact | None = None,
) -> None:
    expected_top_level = {
        "schema_version",
        "analysis_id",
        "status",
        "primary_validation_success",
        "scope",
        "freeze_bundle",
        "preflight_tests",
        "development_inputs",
        "validation_inputs",
        "optimizer",
        "feature_vocabulary",
        "predictions",
        "delta",
        "turn_bins",
        "bootstrap_protocol",
        "joint_game_cluster_bootstrap",
        "primary_decision",
        "counts",
        "environment",
        "elapsed_seconds",
    }
    _require_equal(set(result), expected_top_level, "result top-level fields")
    _require_equal(
        result.get("schema_version"),
        RESULT_SCHEMA_VERSION,
        "result.schema_version",
    )
    _require_equal(result.get("analysis_id"), ANALYSIS_ID, "result.analysis_id")
    status = result.get("status")
    if status not in {"VALIDATION_SUCCESS", "VALIDATION_FAILURE"}:
        raise ValueError("result.status is not a completed validation status")
    primary_success = result.get("primary_validation_success")
    if not isinstance(primary_success, bool):
        raise TypeError("result.primary_validation_success must be boolean")
    _require_equal(
        status,
        "VALIDATION_SUCCESS" if primary_success else "VALIDATION_FAILURE",
        "result status/success",
    )

    scope = _require_mapping(result.get("scope"), "result.scope")
    scope_expected = {
        "training_years": list(DEVELOPMENT_YEARS),
        "validation_year": VALIDATION_YEAR,
        "selected_validation_sources": VALIDATION_SAMPLE_SIZE,
        "sample_seed": VALIDATION_SAMPLE_SEED,
        "label": LABEL,
        "base_model": BASE_MODEL,
        "challenger_model": CHALLENGER_MODEL,
        "comparison": COMPARISON_NAME,
        "validation_2024_touched": True,
        "holdout_2025_touched": False,
    }
    _require_equal(scope, scope_expected, "result.scope")

    bundle_digest = _sha256_text(bundle_file_sha256, "freeze bundle SHA-256")
    payload_digest = _sha256_text(
        bundle.get("payload_sha256"),
        "freeze bundle payload SHA-256",
    )
    freeze = _require_mapping(result.get("freeze_bundle"), "result.freeze_bundle")
    _require_equal(
        freeze.get("path"),
        _relative_path(paths.freeze_bundle, paths.project_root),
        "result.freeze_bundle.path",
    )
    _require_equal(
        freeze.get("file_sha256"),
        bundle_digest,
        "result.freeze_bundle.file_sha256",
    )
    _require_equal(
        freeze.get("payload_sha256"),
        payload_digest,
        "result.freeze_bundle.payload_sha256",
    )
    _require_equal(
        freeze.get("analysis_id"),
        payload.get("analysis_id"),
        "result.freeze_bundle.analysis_id",
    )
    _require_equal(
        freeze.get("status"),
        payload.get("status"),
        "result.freeze_bundle.status",
    )
    _require_equal(
        freeze.get("validation_2024_protocol"),
        payload.get("validation_2024_protocol"),
        "result.freeze_bundle.validation_2024_protocol",
    )
    _require_equal(
        freeze.get("code_and_specs"),
        payload.get("code_and_specs"),
        "result.freeze_bundle.code_and_specs",
    )
    _require_equal(
        freeze.get("code_and_specs_sha256"),
        canonical_json_sha256(payload.get("code_and_specs")),
        "result.freeze_bundle.code_and_specs_sha256",
    )

    preflight = _require_mapping(
        result.get("preflight_tests"),
        "result.preflight_tests",
    )
    preflight_digest = _verify_embedded_path_hash(
        paths.preflight_report,
        preflight,
        hash_key="file_sha256",
        project_root=paths.project_root,
        field="result.preflight_tests",
    )
    _require_equal(
        preflight_digest,
        file_sha256(paths.preflight_report),
        "result.preflight_tests.file_sha256",
    )
    _require_equal(
        preflight.get("status"),
        preflight_report.get("status"),
        "result.preflight_tests.status",
    )
    _require_equal(
        preflight.get("pytest"),
        preflight_report.get("pytest"),
        "result.preflight_tests.pytest",
    )
    recorded_tests = _require_mapping(
        preflight_report.get("tests"),
        "preflight tests",
    )
    recorded_transcript = _require_mapping(
        preflight_report.get("transcript"),
        "preflight transcript",
    )
    _require_equal(
        preflight.get("test_inventory_sha256"),
        recorded_tests.get("inventory_sha256"),
        "result.preflight_tests.test_inventory_sha256",
    )
    _require_equal(
        preflight.get("transcript_sha256"),
        recorded_transcript.get("sha256"),
        "result.preflight_tests.transcript_sha256",
    )
    _require_equal(
        preflight.get("freeze_bundle"),
        preflight_report.get("freeze_bundle"),
        "result.preflight_tests.freeze_bundle",
    )

    checked_development = _validate_result_development_inputs(
        result,
        payload,
        paths.project_root,
        development=development,
    )
    checked_validation_artifact = _validate_result_validation_inputs(
        paths,
        result,
        bundle_file_sha256=bundle_digest,
        payload_sha256=payload_digest,
        validation_artifact=validation_artifact,
    )
    _validate_result_fixed_protocol(
        result,
        payload,
        development=checked_development,
        validation_artifact=checked_validation_artifact,
        primary_success=primary_success,
    )


def _validate_result_development_inputs(
    result: Mapping[str, Any],
    payload: Mapping[str, Any],
    project_root: Path,
    *,
    development: FrozenDevelopmentInputs | None,
) -> FrozenDevelopmentInputs:
    result_inputs = _require_mapping(
        result.get("development_inputs"),
        "result.development_inputs",
    )
    _require_equal(
        set(result_inputs),
        {"manifest", "caches"},
        "result.development_inputs fields",
    )
    checked = development or _load_frozen_development_inputs(
        payload,
        project_root=project_root,
    )
    expected_manifest = {
        "path": _relative_path(checked.manifest_path, project_root),
        "file_sha256": checked.manifest_file_sha256,
        "canonical_sha256": checked.manifest_canonical_sha256,
    }
    _require_equal(
        result_inputs.get("manifest"),
        expected_manifest,
        "result development manifest",
    )
    expected_caches = [
        {"year": year, "path": path, "sha256": digest}
        for year, path, digest in checked.cache_records
    ]
    _require_equal(
        result_inputs.get("caches"),
        expected_caches,
        "result development caches",
    )
    return checked


def _validate_result_validation_inputs(
    paths: ValidationPaths,
    result: Mapping[str, Any],
    *,
    bundle_file_sha256: str,
    payload_sha256: str,
    validation_artifact: YearSparseArtifact | None,
) -> YearSparseArtifact:
    inputs = _require_mapping(
        result.get("validation_inputs"),
        "result.validation_inputs",
    )
    _require_equal(
        set(inputs),
        {"touch_marker", "manifest", "cache"},
        "result.validation_inputs fields",
    )
    touch = _require_mapping(
        inputs.get("touch_marker"),
        "result validation touch marker",
    )
    _require_equal(
        set(touch),
        {"path", "sha256"},
        "result validation touch marker fields",
    )
    _verify_embedded_path_hash(
        paths.touch_marker,
        touch,
        hash_key="sha256",
        project_root=paths.project_root,
        field="result validation touch marker",
    )
    expected_touch = _touch_document(
        paths,
        bundle_file_sha256=bundle_file_sha256,
        payload_sha256=payload_sha256,
    )
    _require_equal(
        _read_json_object(paths.touch_marker),
        expected_touch,
        "validation touch marker",
    )

    manifest_record = _require_mapping(
        inputs.get("manifest"),
        "result validation manifest",
    )
    _require_equal(
        set(manifest_record),
        {
            "path",
            "file_sha256",
            "canonical_sha256",
            "candidate_count",
            "selected_source_count",
            "candidate_population_sha256",
        },
        "result validation manifest fields",
    )
    manifest_file_digest = _verify_embedded_path_hash(
        paths.manifest,
        manifest_record,
        hash_key="file_sha256",
        project_root=paths.project_root,
        field="result validation manifest",
    )
    manifest_document = _read_json_object(paths.manifest)
    manifest_canonical_digest = canonical_json_sha256(manifest_document)
    _require_equal(
        manifest_record.get("canonical_sha256"),
        manifest_canonical_digest,
        "result validation manifest canonical SHA-256",
    )
    _require_equal(
        manifest_document.get("schema_version"),
        2,
        "validation manifest schema_version",
    )
    _require_equal(
        manifest_document.get("validation_year"),
        VALIDATION_YEAR,
        "validation manifest year",
    )
    _require_equal(
        manifest_document.get("validation_2024_touched"),
        True,
        "validation manifest touched flag",
    )
    _require_equal(
        manifest_document.get("holdout_2025_touched"),
        False,
        "validation manifest holdout flag",
    )
    _require_equal(
        manifest_document.get("development_freeze_bundle"),
        {
            "file_sha256": bundle_file_sha256,
            "payload_sha256": payload_sha256,
        },
        "validation manifest freeze identity",
    )
    manifest_years = _require_mapping(
        manifest_document.get("years"),
        "validation manifest years",
    )
    _require_equal(
        set(manifest_years),
        {str(VALIDATION_YEAR)},
        "validation manifest year keys",
    )
    year_document = _require_mapping(
        manifest_years[str(VALIDATION_YEAR)],
        "validation manifest 2024",
    )
    candidate_count = _nonnegative_integer(
        year_document.get("candidate_count"),
        "validation manifest candidate_count",
    )
    if candidate_count < VALIDATION_SAMPLE_SIZE:
        raise ValueError("validation manifest candidate_count is too small")
    _sha256_text(
        year_document.get("candidate_population_sha256"),
        "validation manifest candidate population SHA-256",
    )
    selected = year_document.get("selected")
    if not isinstance(selected, list) or len(selected) != VALIDATION_SAMPLE_SIZE:
        raise ValueError("validation manifest selected list is invalid")
    selected_sources_list: list[str] = []
    for item in selected:
        relative_source = _require_mapping(
            item,
            "validation selected item",
        ).get("relative_source")
        if (
            not isinstance(relative_source, str)
            or not relative_source.startswith(f"{VALIDATION_YEAR}/")
            or not relative_source.endswith(".mjson")
            or "\\" in relative_source
            or ".." in PurePosixPath(relative_source).parts
        ):
            raise ValueError("validation manifest relative_source is invalid")
        selected_sources_list.append(relative_source)
    selected_sources = tuple(selected_sources_list)
    if len(set(selected_sources)) != VALIDATION_SAMPLE_SIZE:
        raise ValueError("validation manifest selected sources are not unique")
    _require_equal(
        manifest_record.get("candidate_count"),
        year_document.get("candidate_count"),
        "result validation manifest candidate_count",
    )
    _require_equal(
        manifest_record.get("selected_source_count"),
        len(selected),
        "result validation manifest selected_source_count",
    )
    _require_equal(
        manifest_record.get("candidate_population_sha256"),
        year_document.get("candidate_population_sha256"),
        "result validation manifest population SHA-256",
    )

    cache_record = _require_mapping(
        inputs.get("cache"),
        "result validation cache",
    )
    _require_equal(
        set(cache_record),
        {
            "path",
            "sha256",
            "row_count",
            "feature_count",
            "selected_source_count",
            "counts",
        },
        "result validation cache fields",
    )
    _verify_embedded_path_hash(
        paths.cache,
        cache_record,
        hash_key="sha256",
        project_root=paths.project_root,
        field="result validation cache",
    )
    artifact = validation_artifact or load_year_sparse_artifact(paths.cache)
    _require_equal(artifact.year, VALIDATION_YEAR, "validation cache year")
    _require_equal(
        artifact.source_ids_by_rank,
        selected_sources,
        "validation cache selected sources",
    )
    _validate_artifact_rows(artifact, "validation cache")
    provenance = dict(artifact.provenance)
    provenance_expected = {
        "analysis_id": ANALYSIS_ID,
        "freeze_bundle_file_sha256": bundle_file_sha256,
        "freeze_bundle_payload_sha256": payload_sha256,
        "manifest_file_sha256": manifest_file_digest,
        "manifest_canonical_sha256": manifest_canonical_digest,
    }
    for key, expected in provenance_expected.items():
        _require_equal(
            provenance.get(key),
            expected,
            f"validation cache provenance {key}",
        )
    _require_equal(cache_record.get("row_count"), artifact.row_count, "cache rows")
    _require_equal(
        cache_record.get("feature_count"),
        artifact.feature_count,
        "cache features",
    )
    _require_equal(
        cache_record.get("selected_source_count"),
        artifact.selected_source_count,
        "cache selected sources",
    )
    raw_counts = provenance.get("counts_json")
    if not isinstance(raw_counts, str):
        raise TypeError("validation cache counts provenance is missing")
    counts = json.loads(raw_counts)
    if not isinstance(counts, dict):
        raise TypeError("validation cache counts must be an object")
    allowed_count_fields = {
        "scanned_files",
        "excluded_non_target_games",
        "target_games",
        "east_kyokus",
        "decisions",
        "candidate_rows",
        "structural_wait_rows",
        "ron_eligible_rows",
    }
    if set(counts) - allowed_count_fields:
        raise ValueError("validation cache counts have unexpected fields")
    for key, value in counts.items():
        _nonnegative_integer(value, f"validation cache counts.{key}")
    required_counts = {
        "scanned_files": VALIDATION_SAMPLE_SIZE,
        "candidate_rows": artifact.row_count,
        "decisions": len(artifact.group_ids),
        "structural_wait_rows": int(np.sum(artifact.y_structural)),
        "ron_eligible_rows": int(np.sum(artifact.y_ron)),
    }
    for key, expected in required_counts.items():
        _require_equal(counts.get(key), expected, f"validation cache counts.{key}")
    _require_equal(
        cache_record.get("counts"),
        dict(sorted(counts.items())),
        "validation cache counts",
    )
    return artifact


def _validate_result_fixed_protocol(
    result: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    development: FrozenDevelopmentInputs,
    validation_artifact: YearSparseArtifact,
    primary_success: bool,
) -> None:
    """Validate the complete normal-run result schema before sealing a receipt."""

    _validate_result_fixed_protocol_core(
        result,
        payload,
        primary_success=primary_success,
    )
    _validate_result_optimizer_models(result, payload, development)
    _validate_result_feature_vocabulary(
        result,
        development,
        validation_artifact,
    )
    counts = _validate_result_counts(result, validation_artifact)
    overall_delta, turn_bin_deltas = _validate_result_metrics(
        result,
        validation_artifact,
        counts=counts,
    )
    _validate_result_bootstrap_schema(
        result,
        overall_delta=overall_delta,
        turn_bin_deltas=turn_bin_deltas,
        counts=counts,
    )
    _validate_result_environment(result)


def _validate_result_fixed_protocol_core(
    result: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    primary_success: bool,
) -> None:
    optimizer = _require_mapping(result.get("optimizer"), "result.optimizer")
    optimizer_expected = {
        "type": "SciPy L-BFGS-B with analytic gradient",
        "objective": "weighted mean log loss + l2/2 * ||beta||^2",
        "selected_l2": payload.get("selected_l2"),
        "selected_l2_source": "verified freeze bundle only",
        "max_iterations": MAX_ITERATIONS,
        "gradient_tolerance": GRADIENT_TOLERANCE,
        "function_tolerance": FUNCTION_TOLERANCE,
        "max_line_search_steps": MAX_LINE_SEARCH_STEPS,
        "history_size": HISTORY_SIZE,
        "intercept_penalized": False,
    }
    for key, expected in optimizer_expected.items():
        _require_equal(optimizer.get(key), expected, f"result.optimizer.{key}")

    predictions = _require_mapping(result.get("predictions"), "result.predictions")
    _require_equal(
        set(predictions),
        {BASE_MODEL, CHALLENGER_MODEL, "recalibration_applied"},
        "result.predictions fields",
    )
    _require_equal(
        predictions.get("recalibration_applied"),
        False,
        "result.predictions.recalibration_applied",
    )
    for model_name in (BASE_MODEL, CHALLENGER_MODEL):
        model = _require_mapping(
            predictions.get(model_name),
            f"result.predictions.{model_name}",
        )
        _sha256_text(model.get("sha256"), f"result prediction {model_name}")

    turn_bins = _require_mapping(result.get("turn_bins"), "result.turn_bins")
    _require_equal(
        set(turn_bins),
        set(DECISION_TURN_BINS),
        "result.turn_bins fields",
    )

    bootstrap_protocol = _require_mapping(
        result.get("bootstrap_protocol"),
        "result.bootstrap_protocol",
    )
    bootstrap_expected = {
        "cluster": "game",
        "paired": True,
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "refit": False,
        "interval": "pointwise percentile 95%",
        "shared_multiplicity_across_models_and_turn_bins": True,
    }
    _require_equal(
        bootstrap_protocol,
        bootstrap_expected,
        "result.bootstrap_protocol",
    )
    bootstrap = _require_mapping(
        result.get("joint_game_cluster_bootstrap"),
        "result.joint_game_cluster_bootstrap",
    )
    _require_equal(
        bootstrap.get("requested_replicates"),
        BOOTSTRAP_REPLICATES,
        "result bootstrap replicates",
    )
    _require_equal(bootstrap.get("seed"), BOOTSTRAP_SEED, "result bootstrap seed")
    _require_equal(
        bootstrap.get("stratified"),
        False,
        "result bootstrap stratified",
    )
    _require_equal(
        bootstrap.get("group_order"),
        list(DECISION_TURN_BINS),
        "result bootstrap turn bins",
    )
    comparisons = bootstrap.get("comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != 1:
        raise ValueError("result bootstrap must have exactly one comparison")
    comparison = _require_mapping(comparisons[0], "result bootstrap comparison")
    for key, expected in {
        "name": COMPARISON_NAME,
        "base_model": BASE_MODEL,
        "challenger_model": CHALLENGER_MODEL,
    }.items():
        _require_equal(
            comparison.get(key),
            expected,
            f"result bootstrap comparison {key}",
        )
    by_group = comparison.get("by_group")
    if not isinstance(by_group, list):
        raise TypeError("result bootstrap by_group must be a list")
    _require_equal(
        [
            _require_mapping(item, "result bootstrap group").get("group")
            for item in by_group
        ],
        list(DECISION_TURN_BINS),
        "result bootstrap group order",
    )
    contrasts = comparison.get("contrasts")
    if not isinstance(contrasts, list) or len(contrasts) != 1:
        raise ValueError("result bootstrap must have the fixed turn-bin contrast")
    contrast = _require_mapping(contrasts[0], "result bootstrap contrast")
    for key, expected in {
        "name": "late_minus_early",
        "left_group": "13+",
        "right_group": "1-6",
    }.items():
        _require_equal(
            contrast.get(key),
            expected,
            f"result bootstrap contrast {key}",
        )

    overall = _require_mapping(
        comparison.get("overall"),
        "result bootstrap comparison overall",
    )
    log_loss = _require_mapping(
        overall.get("log_loss"),
        "result bootstrap comparison log_loss",
    )
    bootstrap_point = _finite_number(log_loss.get("point"), "bootstrap point")
    bootstrap_lower = _finite_number(log_loss.get("lower"), "bootstrap lower")
    bootstrap_upper = _finite_number(log_loss.get("upper"), "bootstrap upper")
    _require_equal(
        log_loss.get("valid_replicates"),
        BOOTSTRAP_REPLICATES,
        "bootstrap log-loss valid replicates",
    )

    base_prediction = _require_mapping(
        predictions.get(BASE_MODEL),
        f"result.predictions.{BASE_MODEL}",
    )
    challenger_prediction = _require_mapping(
        predictions.get(CHALLENGER_MODEL),
        f"result.predictions.{CHALLENGER_MODEL}",
    )
    base_metrics = _require_mapping(
        base_prediction.get("metrics"),
        f"result.predictions.{BASE_MODEL}.metrics",
    )
    challenger_metrics = _require_mapping(
        challenger_prediction.get("metrics"),
        f"result.predictions.{CHALLENGER_MODEL}.metrics",
    )
    metric_point = _finite_number(
        challenger_metrics.get("log_loss"),
        "challenger log loss",
    ) - _finite_number(base_metrics.get("log_loss"), "base log loss")
    if not math.isclose(bootstrap_point, metric_point, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("bootstrap point does not match result prediction metrics")
    delta = _require_mapping(result.get("delta"), "result.delta")
    if not math.isclose(
        _finite_number(delta.get("log_loss"), "result.delta.log_loss"),
        metric_point,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("result log-loss delta does not match prediction metrics")

    primary = _require_mapping(
        result.get("primary_decision"),
        "result.primary_decision",
    )
    expected_primary = {
        "metric": "weighted_log_loss_delta",
        "estimand": "conventional_simple - conventional",
        "point": bootstrap_point,
        "percentile_95_lower": bootstrap_lower,
        "percentile_95_upper": bootstrap_upper,
        "valid_replicates": BOOTSTRAP_REPLICATES,
        "required_replicates": BOOTSTRAP_REPLICATES,
        "point_below_zero": bootstrap_point < 0.0,
        "upper_below_zero": bootstrap_upper < 0.0,
        "success": bootstrap_point < 0.0 and bootstrap_upper < 0.0,
    }
    _require_equal(primary, expected_primary, "result.primary_decision")
    _require_equal(
        primary_success,
        expected_primary["success"],
        "primary success rule",
    )


def _validate_result_optimizer_models(
    result: Mapping[str, Any],
    payload: Mapping[str, Any],
    development: FrozenDevelopmentInputs,
) -> None:
    optimizer = _require_mapping(result.get("optimizer"), "result.optimizer")
    _require_equal(
        set(optimizer),
        {
            "type",
            "objective",
            "selected_l2",
            "selected_l2_source",
            "max_iterations",
            "gradient_tolerance",
            "function_tolerance",
            "max_line_search_steps",
            "history_size",
            "intercept_penalized",
            "models",
        },
        "result.optimizer fields",
    )
    vocabulary = global_feature_vocabulary(development.artifacts)
    expected_active_counts = {
        BASE_MODEL: int(np.sum(conventional_feature_mask(vocabulary))),
        CHALLENGER_MODEL: len(vocabulary),
    }
    training_rows = sum(artifact.row_count for artifact in development.artifacts)
    training_weight = sum(
        float(np.sum(artifact.weights)) for artifact in development.artifacts
    )
    models = _require_mapping(optimizer.get("models"), "result.optimizer.models")
    _require_equal(
        set(models),
        {BASE_MODEL, CHALLENGER_MODEL},
        "result.optimizer.models fields",
    )
    selected_l2 = _finite_number(payload.get("selected_l2"), "selected_l2")
    for model_name, expected_active in expected_active_counts.items():
        field = f"result.optimizer.models.{model_name}"
        model = _require_mapping(models.get(model_name), field)
        _require_equal(
            set(model),
            {"intercept", "active_feature_count", "coefficient_sha256", "diagnostics"},
            f"{field} fields",
        )
        _finite_number(model.get("intercept"), f"{field}.intercept")
        _require_equal(
            model.get("active_feature_count"),
            expected_active,
            f"{field}.active_feature_count",
        )
        _sha256_text(model.get("coefficient_sha256"), f"{field}.coefficient_sha256")
        _validate_result_optimizer_diagnostics(
            _require_mapping(model.get("diagnostics"), f"{field}.diagnostics"),
            field=f"{field}.diagnostics",
            selected_l2=selected_l2,
            expected_rows=training_rows,
            expected_weight=training_weight,
            expected_active=expected_active,
        )


def _validate_result_optimizer_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    field: str,
    selected_l2: float,
    expected_rows: int,
    expected_weight: float,
    expected_active: int,
) -> None:
    _require_equal(
        set(diagnostics),
        {
            "converged",
            "status",
            "message",
            "iterations",
            "function_evaluations",
            "gradient_evaluations",
            "initial_objective",
            "final_objective",
            "initial_gradient_inf_norm",
            "final_gradient_inf_norm",
            "row_count",
            "total_weight",
            "active_feature_count",
            "l2",
        },
        f"{field} fields",
    )
    _require_equal(diagnostics.get("converged"), True, f"{field}.converged")
    _require_equal(diagnostics.get("status"), 0, f"{field}.status")
    message = diagnostics.get("message")
    if not isinstance(message, str) or not message:
        raise ValueError(f"{field}.message must be a non-empty string")
    iterations = _nonnegative_integer(
        diagnostics.get("iterations"), f"{field}.iterations"
    )
    if iterations >= MAX_ITERATIONS:
        raise ValueError(f"{field}.iterations reached the fixed limit")
    for key in ("function_evaluations", "gradient_evaluations"):
        if _nonnegative_integer(diagnostics.get(key), f"{field}.{key}") < 1:
            raise ValueError(f"{field}.{key} must be positive")
    initial_objective = _finite_number(
        diagnostics.get("initial_objective"),
        f"{field}.initial_objective",
    )
    final_objective = _finite_number(
        diagnostics.get("final_objective"),
        f"{field}.final_objective",
    )
    if initial_objective < 0 or final_objective < 0:
        raise ValueError(f"{field} objectives must be nonnegative")
    if final_objective > initial_objective + 1e-12:
        raise ValueError(f"{field}.final_objective exceeds initial objective")
    initial_gradient = _finite_number(
        diagnostics.get("initial_gradient_inf_norm"),
        f"{field}.initial_gradient_inf_norm",
    )
    final_gradient = _finite_number(
        diagnostics.get("final_gradient_inf_norm"),
        f"{field}.final_gradient_inf_norm",
    )
    if initial_gradient < 0 or not 0 <= final_gradient <= GRADIENT_TOLERANCE:
        raise ValueError(f"{field} gradient diagnostics fail the acceptance gate")
    _require_equal(diagnostics.get("row_count"), expected_rows, f"{field}.row_count")
    _require_close(
        diagnostics.get("total_weight"),
        expected_weight,
        f"{field}.total_weight",
    )
    _require_equal(
        diagnostics.get("active_feature_count"),
        expected_active,
        f"{field}.active_feature_count",
    )
    _require_close(diagnostics.get("l2"), selected_l2, f"{field}.l2")


def _validate_result_feature_vocabulary(
    result: Mapping[str, Any],
    development: FrozenDevelopmentInputs,
    validation_artifact: YearSparseArtifact,
) -> None:
    document = _require_mapping(
        result.get("feature_vocabulary"),
        "result.feature_vocabulary",
    )
    vocabulary = global_feature_vocabulary(development.artifacts)
    validation_only = tuple(
        sorted(set(validation_artifact.feature_names) - set(vocabulary))
    )
    expected = {
        "source": "frozen 2020-2023 development caches only",
        "feature_count": len(vocabulary),
        "sha256": canonical_json_sha256(list(vocabulary)),
        "validation_only_feature_count": len(validation_only),
        "validation_only_features_sha256": canonical_json_sha256(list(validation_only)),
        "validation_only_features_coefficient": 0.0,
    }
    _require_equal(document, expected, "result.feature_vocabulary")


def _validate_result_counts(
    result: Mapping[str, Any],
    artifact: YearSparseArtifact,
) -> dict[str, int]:
    counts = _require_mapping(result.get("counts"), "result.counts")
    expected = {
        "games": len(artifact.cluster_ids),
        "decisions": len(artifact.group_ids),
        "candidate_rows": artifact.row_count,
    }
    _require_equal(counts, expected, "result.counts")
    if any(value < 1 for value in expected.values()):
        raise ValueError("result.counts must all be positive")
    return expected


def _validate_result_metrics(
    result: Mapping[str, Any],
    artifact: YearSparseArtifact,
    *,
    counts: Mapping[str, int],
) -> tuple[dict[str, float | None], dict[str, dict[str, float | None]]]:
    predictions = _require_mapping(result.get("predictions"), "result.predictions")
    _require_equal(
        set(predictions),
        {BASE_MODEL, CHALLENGER_MODEL, "recalibration_applied"},
        "result.predictions fields",
    )
    _require_equal(
        predictions.get("recalibration_applied"),
        False,
        "result.predictions.recalibration_applied",
    )
    overall_metrics: dict[str, dict[str, float | None]] = {}
    positive_rate = _artifact_positive_rate(artifact)
    macro_decisions = _artifact_macro_decision_count(artifact)
    for model_name in (BASE_MODEL, CHALLENGER_MODEL):
        field = f"result.predictions.{model_name}"
        prediction = _require_mapping(predictions.get(model_name), field)
        _require_equal(set(prediction), {"sha256", "metrics"}, f"{field} fields")
        _sha256_text(prediction.get("sha256"), f"{field}.sha256")
        overall_metrics[model_name] = _validate_binary_metrics(
            prediction.get("metrics"),
            field=f"{field}.metrics",
            expected_rows=counts["candidate_rows"],
            expected_decisions=counts["decisions"],
            expected_positive_rate=positive_rate,
            expected_macro_decisions=macro_decisions,
        )
    overall_delta = _validate_metric_delta(
        result.get("delta"),
        overall_metrics[BASE_MODEL],
        overall_metrics[CHALLENGER_MODEL],
        field="result.delta",
    )

    turn_bins = _require_mapping(result.get("turn_bins"), "result.turn_bins")
    _require_equal(set(turn_bins), set(DECISION_TURN_BINS), "result.turn_bins fields")
    turn_bin_deltas: dict[str, dict[str, float | None]] = {}
    total_rows = 0
    total_decisions = 0
    for code, turn_bin in enumerate(DECISION_TURN_BINS):
        mask = artifact.turn_bins == code
        row_count = int(np.sum(mask))
        decision_count = len(np.unique(artifact.group_indices[mask]))
        game_count = len(np.unique(artifact.cluster_indices[mask]))
        if min(row_count, decision_count, game_count) < 1:
            raise ValueError(f"validation artifact has an empty turn bin: {turn_bin}")
        field = f"result.turn_bins.{turn_bin}"
        document = _require_mapping(turn_bins.get(turn_bin), field)
        _require_equal(
            set(document),
            {"row_count", "decision_count", "game_count", "models", "delta"},
            f"{field} fields",
        )
        for key, expected in {
            "row_count": row_count,
            "decision_count": decision_count,
            "game_count": game_count,
        }.items():
            _require_equal(document.get(key), expected, f"{field}.{key}")
        models = _require_mapping(document.get("models"), f"{field}.models")
        _require_equal(
            set(models),
            {BASE_MODEL, CHALLENGER_MODEL},
            f"{field}.models fields",
        )
        bin_metrics = {
            model_name: _validate_binary_metrics(
                models.get(model_name),
                field=f"{field}.models.{model_name}",
                expected_rows=row_count,
                expected_decisions=decision_count,
                expected_positive_rate=_artifact_positive_rate(artifact, mask=mask),
                expected_macro_decisions=_artifact_macro_decision_count(
                    artifact,
                    mask=mask,
                ),
            )
            for model_name in (BASE_MODEL, CHALLENGER_MODEL)
        }
        turn_bin_deltas[turn_bin] = _validate_metric_delta(
            document.get("delta"),
            bin_metrics[BASE_MODEL],
            bin_metrics[CHALLENGER_MODEL],
            field=f"{field}.delta",
        )
        total_rows += row_count
        total_decisions += decision_count
    _require_equal(total_rows, counts["candidate_rows"], "turn-bin row total")
    _require_equal(total_decisions, counts["decisions"], "turn-bin decision total")
    return overall_delta, turn_bin_deltas


def _validate_binary_metrics(
    value: Any,
    *,
    field: str,
    expected_rows: int,
    expected_decisions: int,
    expected_positive_rate: float,
    expected_macro_decisions: int,
) -> dict[str, float | None]:
    metrics = _require_mapping(value, field)
    _require_equal(
        set(metrics),
        {
            "row_count",
            "decision_count",
            "positive_rate",
            "auc",
            "log_loss",
            "brier",
            "ece",
            "calibration",
            "macro_concordance",
            "macro_concordance_decisions",
        },
        f"{field} fields",
    )
    _require_equal(metrics.get("row_count"), expected_rows, f"{field}.row_count")
    _require_equal(
        metrics.get("decision_count"),
        expected_decisions,
        f"{field}.decision_count",
    )
    _require_close(
        metrics.get("positive_rate"),
        expected_positive_rate,
        f"{field}.positive_rate",
    )
    auc = _optional_finite_number(metrics.get("auc"), f"{field}.auc")
    if auc is not None and not 0 <= auc <= 1:
        raise ValueError(f"{field}.auc must be in [0, 1]")
    has_both_classes = 0 < expected_positive_rate < 1
    if (auc is not None) != has_both_classes:
        raise ValueError(f"{field}.auc availability is inconsistent with labels")
    log_loss = _finite_number(metrics.get("log_loss"), f"{field}.log_loss")
    brier = _finite_number(metrics.get("brier"), f"{field}.brier")
    ece = _finite_number(metrics.get("ece"), f"{field}.ece")
    if log_loss < 0 or not 0 <= brier <= 1 or not 0 <= ece <= 1:
        raise ValueError(f"{field} metric range is invalid")
    macro = _optional_finite_number(
        metrics.get("macro_concordance"),
        f"{field}.macro_concordance",
    )
    if macro is not None and not 0 <= macro <= 1:
        raise ValueError(f"{field}.macro_concordance must be in [0, 1]")
    _require_equal(
        metrics.get("macro_concordance_decisions"),
        expected_macro_decisions,
        f"{field}.macro_concordance_decisions",
    )
    if (macro is not None) != (expected_macro_decisions > 0):
        raise ValueError(f"{field}.macro_concordance availability is inconsistent")
    _validate_calibration(
        metrics.get("calibration"),
        field=f"{field}.calibration",
        row_count=expected_rows,
        single_class=not has_both_classes,
    )
    return {
        "auc": auc,
        "log_loss": log_loss,
        "brier": brier,
        "ece": ece,
        "macro_concordance": macro,
    }


def _validate_calibration(
    value: Any,
    *,
    field: str,
    row_count: int,
    single_class: bool,
) -> None:
    calibration = _require_mapping(value, field)
    _require_equal(
        set(calibration),
        {
            "joint_status",
            "recalibration_intercept",
            "recalibration_slope",
            "calibration_in_the_large",
            "iterations",
            "clip_epsilon",
            "clipped_low",
            "clipped_high",
        },
        f"{field} fields",
    )
    status = calibration.get("joint_status")
    if status not in CALIBRATION_STATUSES:
        raise ValueError(f"{field}.joint_status is invalid")
    if single_class != (status == "single_class"):
        raise ValueError(f"{field}.joint_status is inconsistent with labels")
    intercept = _optional_finite_number(
        calibration.get("recalibration_intercept"),
        f"{field}.recalibration_intercept",
    )
    slope = _optional_finite_number(
        calibration.get("recalibration_slope"),
        f"{field}.recalibration_slope",
    )
    calibration_large = _optional_finite_number(
        calibration.get("calibration_in_the_large"),
        f"{field}.calibration_in_the_large",
    )
    iterations = _nonnegative_integer(
        calibration.get("iterations"),
        f"{field}.iterations",
    )
    if iterations > 100:
        raise ValueError(f"{field}.iterations exceeds the fixed diagnostic limit")
    _require_close(
        calibration.get("clip_epsilon"),
        1e-15,
        f"{field}.clip_epsilon",
        absolute_tolerance=0.0,
    )
    clipped_low = _nonnegative_integer(
        calibration.get("clipped_low"),
        f"{field}.clipped_low",
    )
    clipped_high = _nonnegative_integer(
        calibration.get("clipped_high"),
        f"{field}.clipped_high",
    )
    if clipped_low + clipped_high > row_count:
        raise ValueError(f"{field} clipped counts exceed row_count")
    if status == "ok":
        if intercept is None or slope is None or calibration_large is None:
            raise ValueError(f"{field} successful diagnostics must be finite")
        if iterations < 1:
            raise ValueError(f"{field}.iterations must be positive for status ok")
    else:
        if intercept is not None or slope is not None:
            raise ValueError(f"{field} failed diagnostics must not report joint fit")
        if status == "single_class":
            if calibration_large is not None or iterations != 0:
                raise ValueError(f"{field} single-class diagnostics are inconsistent")
        elif calibration_large is None:
            raise ValueError(f"{field}.calibration_in_the_large must be finite")
        elif status in {"constant_logit", "complete_separation", "quasi_separation"}:
            if iterations != 0:
                raise ValueError(f"{field}.iterations must be zero for {status}")
        elif iterations < 1:
            raise ValueError(f"{field}.iterations must be positive for {status}")


def _validate_metric_delta(
    value: Any,
    base: Mapping[str, float | None],
    challenger: Mapping[str, float | None],
    *,
    field: str,
) -> dict[str, float | None]:
    delta = _require_mapping(value, field)
    _require_equal(set(delta), set(METRIC_NAMES), f"{field} fields")
    checked: dict[str, float | None] = {}
    for metric in METRIC_NAMES:
        base_value = base[metric]
        challenger_value = challenger[metric]
        expected = (
            None
            if base_value is None or challenger_value is None
            else challenger_value - base_value
        )
        actual = _optional_finite_number(delta.get(metric), f"{field}.{metric}")
        _require_optional_close(actual, expected, f"{field}.{metric}")
        checked[metric] = actual
    return checked


def _artifact_positive_rate(
    artifact: YearSparseArtifact,
    *,
    mask: np.ndarray | None = None,
) -> float:
    selected = np.ones(artifact.row_count, dtype=np.bool_) if mask is None else mask
    weights = artifact.weights[selected]
    labels = artifact.labels(LABEL)[selected]
    return float(np.sum(weights * labels) / np.sum(weights))


def _artifact_macro_decision_count(
    artifact: YearSparseArtifact,
    *,
    mask: np.ndarray | None = None,
) -> int:
    selected = np.ones(artifact.row_count, dtype=np.bool_) if mask is None else mask
    labels = artifact.labels(LABEL)
    return sum(
        bool(np.any(labels[selected & (artifact.group_indices == group)]))
        and bool(np.any(~labels[selected & (artifact.group_indices == group)]))
        for group in np.unique(artifact.group_indices[selected])
    )


def _validate_result_bootstrap_schema(
    result: Mapping[str, Any],
    *,
    overall_delta: Mapping[str, float | None],
    turn_bin_deltas: Mapping[str, Mapping[str, float | None]],
    counts: Mapping[str, int],
) -> None:
    protocol = _require_mapping(
        result.get("bootstrap_protocol"),
        "result.bootstrap_protocol",
    )
    _require_equal(
        protocol,
        {
            "cluster": "game",
            "paired": True,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "refit": False,
            "interval": "pointwise percentile 95%",
            "shared_multiplicity_across_models_and_turn_bins": True,
        },
        "result.bootstrap_protocol",
    )
    bootstrap = _require_mapping(
        result.get("joint_game_cluster_bootstrap"),
        "result.joint_game_cluster_bootstrap",
    )
    _require_equal(
        set(bootstrap),
        {
            "cluster_count",
            "requested_replicates",
            "seed",
            "ece_bins",
            "stratified",
            "stratum_cluster_counts",
            "group_order",
            "comparisons",
        },
        "result bootstrap fields",
    )
    for key, expected in {
        "cluster_count": counts["games"],
        "requested_replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "ece_bins": 10,
        "stratified": False,
        "stratum_cluster_counts": [["all", counts["games"]]],
        "group_order": list(DECISION_TURN_BINS),
    }.items():
        _require_equal(bootstrap.get(key), expected, f"result bootstrap {key}")
    comparisons = bootstrap.get("comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != 1:
        raise ValueError("result bootstrap must have exactly one comparison")
    comparison = _require_mapping(comparisons[0], "result bootstrap comparison")
    _require_equal(
        set(comparison),
        {"name", "base_model", "challenger_model", "overall", "by_group", "contrasts"},
        "result bootstrap comparison fields",
    )
    for key, expected in {
        "name": COMPARISON_NAME,
        "base_model": BASE_MODEL,
        "challenger_model": CHALLENGER_MODEL,
    }.items():
        _require_equal(comparison.get(key), expected, f"comparison.{key}")
    _validate_bootstrap_metric_set(
        comparison.get("overall"),
        expected_points=overall_delta,
        field="result bootstrap comparison overall",
        non_log_loss_absolute_limit=1.0,
    )

    by_group = comparison.get("by_group")
    if not isinstance(by_group, list) or len(by_group) != len(DECISION_TURN_BINS):
        raise ValueError("result bootstrap by_group must contain all fixed bins")
    turn_bins = _require_mapping(result.get("turn_bins"), "result.turn_bins")
    for expected_group, raw_group in zip(DECISION_TURN_BINS, by_group, strict=True):
        field = f"result bootstrap group {expected_group}"
        group = _require_mapping(raw_group, field)
        _require_equal(
            set(group),
            {"group", "row_count", "decision_count", "cluster_count", "metrics"},
            f"{field} fields",
        )
        _require_equal(group.get("group"), expected_group, f"{field}.group")
        turn_document = _require_mapping(
            turn_bins.get(expected_group),
            f"result.turn_bins.{expected_group}",
        )
        for bootstrap_key, result_key in (
            ("row_count", "row_count"),
            ("decision_count", "decision_count"),
            ("cluster_count", "game_count"),
        ):
            _require_equal(
                group.get(bootstrap_key),
                turn_document.get(result_key),
                f"{field}.{bootstrap_key}",
            )
        _validate_bootstrap_metric_set(
            group.get("metrics"),
            expected_points=turn_bin_deltas[expected_group],
            field=f"{field}.metrics",
            non_log_loss_absolute_limit=1.0,
        )

    contrasts = comparison.get("contrasts")
    if not isinstance(contrasts, list) or len(contrasts) != 1:
        raise ValueError("result bootstrap must have the fixed turn-bin contrast")
    contrast = _require_mapping(contrasts[0], "result bootstrap contrast")
    _require_equal(
        set(contrast),
        {"name", "left_group", "right_group", "metrics"},
        "result bootstrap contrast fields",
    )
    for key, expected in {
        "name": "late_minus_early",
        "left_group": "13+",
        "right_group": "1-6",
    }.items():
        _require_equal(contrast.get(key), expected, f"result contrast.{key}")
    contrast_points = {
        metric: _optional_difference_values(
            turn_bin_deltas["13+"][metric],
            turn_bin_deltas["1-6"][metric],
        )
        for metric in METRIC_NAMES
    }
    _validate_bootstrap_metric_set(
        contrast.get("metrics"),
        expected_points=contrast_points,
        field="result bootstrap contrast metrics",
        non_log_loss_absolute_limit=2.0,
    )


def _validate_bootstrap_metric_set(
    value: Any,
    *,
    expected_points: Mapping[str, float | None],
    field: str,
    non_log_loss_absolute_limit: float,
) -> None:
    absolute_limit = _finite_number(
        non_log_loss_absolute_limit,
        f"{field} non-log-loss absolute limit",
    )
    if absolute_limit <= 0:
        raise ValueError(f"{field} non-log-loss absolute limit must be positive")
    metrics = _require_mapping(value, field)
    _require_equal(set(metrics), set(METRIC_NAMES), f"{field} fields")
    for metric in METRIC_NAMES:
        interval = _require_mapping(metrics.get(metric), f"{field}.{metric}")
        _require_equal(
            set(interval),
            {"point", "lower", "upper", "valid_replicates"},
            f"{field}.{metric} fields",
        )
        point = _optional_finite_number(
            interval.get("point"), f"{field}.{metric}.point"
        )
        _require_optional_close(
            point, expected_points[metric], f"{field}.{metric}.point"
        )
        valid = _nonnegative_integer(
            interval.get("valid_replicates"),
            f"{field}.{metric}.valid_replicates",
        )
        if valid > BOOTSTRAP_REPLICATES:
            raise ValueError(f"{field}.{metric}.valid_replicates is too large")
        lower = _optional_finite_number(
            interval.get("lower"), f"{field}.{metric}.lower"
        )
        upper = _optional_finite_number(
            interval.get("upper"), f"{field}.{metric}.upper"
        )
        if valid == 0:
            if lower is not None or upper is not None:
                raise ValueError(
                    f"{field}.{metric} empty interval must have null bounds"
                )
        elif lower is None or upper is None or lower > upper:
            raise ValueError(f"{field}.{metric} interval bounds are invalid")
        if metric != "log_loss" and any(
            bound is not None and not -absolute_limit <= bound <= absolute_limit
            for bound in (point, lower, upper)
        ):
            raise ValueError(
                f"{field}.{metric} interval exceeds absolute limit {absolute_limit}"
            )
        if metric in {"log_loss", "brier", "ece"} and valid != BOOTSTRAP_REPLICATES:
            raise ValueError(f"{field}.{metric} must use all bootstrap replicates")


def _validate_result_environment(result: Mapping[str, Any]) -> None:
    environment = _require_mapping(result.get("environment"), "result.environment")
    _require_equal(
        set(environment),
        {"argv", "python", "numpy", "scipy"},
        "result.environment fields",
    )
    argv = environment.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(value, str) for value in argv)
    ):
        raise ValueError("result.environment.argv must be a non-empty string list")
    if "--recover-receipt" in argv:
        raise ValueError("result.environment.argv cannot be a recovery invocation")
    for key in ("python", "numpy", "scipy"):
        value = environment.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"result.environment.{key} must be a non-empty string")
    elapsed = _finite_number(result.get("elapsed_seconds"), "result.elapsed_seconds")
    if elapsed < 0:
        raise ValueError("result.elapsed_seconds must be nonnegative")


def _optional_difference_values(
    left: float | None,
    right: float | None,
) -> float | None:
    return None if left is None or right is None else left - right


def _verify_embedded_path_hash(
    path: Path,
    record: Mapping[str, Any],
    *,
    hash_key: str,
    project_root: Path,
    field: str,
) -> str:
    _require_equal(
        record.get("path"),
        _relative_path(path, project_root),
        f"{field}.path",
    )
    if not path.is_file():
        raise ValueError(f"{field} target does not exist: {path}")
    expected = _sha256_text(record.get(hash_key), f"{field}.{hash_key}")
    actual = file_sha256(path)
    _require_equal(actual, expected, f"{field}.{hash_key}")
    return expected


def verify_validation_receipt(
    receipt_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Verify an immutable result receipt and every artifact that it seals."""

    root = project_root.resolve()
    receipt = _read_json_object(receipt_path)
    _require_equal(
        receipt.get("schema_version"),
        RECEIPT_SCHEMA_VERSION,
        "receipt.schema_version",
    )
    _require_equal(receipt.get("analysis_id"), ANALYSIS_ID, "receipt.analysis_id")
    _require_equal(
        receipt.get("status"),
        "SEALED_VALIDATION_RESULT",
        "receipt.status",
    )
    _require_equal(
        receipt.get("validation_2024_touched"),
        True,
        "receipt.validation_2024_touched",
    )
    _require_equal(
        receipt.get("holdout_2025_touched"),
        False,
        "receipt.holdout_2025_touched",
    )
    for field in (
        "result",
        "preflight_report",
        "touch_marker",
        "validation_manifest",
        "validation_cache",
    ):
        record = _require_mapping(receipt.get(field), f"receipt.{field}")
        _verify_hash_record(record, root, f"receipt.{field}")
    bundle_record = _require_mapping(
        receipt.get("freeze_bundle"),
        "receipt.freeze_bundle",
    )
    _verify_hash_record(bundle_record, root, "receipt.freeze_bundle")
    payload_digest = _sha256_text(
        bundle_record.get("payload_sha256"),
        "receipt.freeze_bundle.payload_sha256",
    )
    bundle = verify_freeze_bundle(
        _resolve_record_path(bundle_record, root, "receipt.freeze_bundle"),
        project_root=root,
    )
    _require_equal(
        bundle.get("payload_sha256"),
        payload_digest,
        "receipt.freeze_bundle.payload_sha256",
    )
    preflight_record = _require_mapping(
        receipt["preflight_report"],
        "receipt.preflight_report",
    )
    verify_preflight_report(
        _resolve_record_path(preflight_record, root, "receipt.preflight_report"),
        root,
        {
            "bundle_file_sha256": _sha256_text(
                bundle_record.get("sha256"),
                "receipt.freeze_bundle.sha256",
            ),
            "payload_sha256": payload_digest,
        },
    )
    result_record = _require_mapping(receipt["result"], "receipt.result")
    result = _read_json_object(
        _resolve_record_path(result_record, root, "receipt.result")
    )
    _require_equal(result.get("analysis_id"), ANALYSIS_ID, "result.analysis_id")
    if result.get("status") not in {"VALIDATION_SUCCESS", "VALIDATION_FAILURE"}:
        raise ValueError("result.status is not a completed validation status")
    return receipt


def _validated_bundle_protocol(
    bundle: Mapping[str, Any],
) -> tuple[Mapping[str, Any], float]:
    payload = _require_mapping(bundle.get("payload"), "bundle.payload")
    _require_equal(
        payload.get("analysis_id"),
        EXPECTED_FREEZE_ANALYSIS_ID,
        "bundle.payload.analysis_id",
    )
    _require_equal(
        payload.get("status"),
        EXPECTED_FREEZE_STATUS,
        "bundle.payload.status",
    )
    selected_l2 = _finite_number(
        payload.get("selected_l2"),
        "bundle.payload.selected_l2",
    )
    if selected_l2 < 0:
        raise ValueError("bundle selected_l2 must be nonnegative")
    protocol = _require_mapping(
        payload.get("validation_2024_protocol"),
        "bundle.payload.validation_2024_protocol",
    )
    expected = {
        "validation_year": VALIDATION_YEAR,
        "holdout_year": 2025,
        "training_years": list(DEVELOPMENT_YEARS),
        "files_per_year": VALIDATION_SAMPLE_SIZE,
        "sample_seed": VALIDATION_SAMPLE_SEED,
        "label": LABEL,
        "base_model": BASE_MODEL,
        "challenger_model": CHALLENGER_MODEL,
        "comparison": COMPARISON_NAME,
        "selected_l2": selected_l2,
        "lambda_selection_on_2024": False,
        "recalibration_on_2024": False,
        "primary_metric": "weighted_log_loss_delta",
        "success_rule": "point_estimate < 0 and percentile_95_upper < 0",
        "validation_2024_touched_at_freeze": False,
        "holdout_2025_touched_at_freeze": False,
    }
    for key, value in expected.items():
        _require_equal(protocol.get(key), value, f"bundle protocol {key}")
    optimizer = _require_mapping(protocol.get("optimizer"), "bundle protocol optimizer")
    optimizer_expected = {
        "max_iterations": MAX_ITERATIONS,
        "gradient_tolerance": GRADIENT_TOLERANCE,
        "function_tolerance": FUNCTION_TOLERANCE,
        "max_line_search_steps": MAX_LINE_SEARCH_STEPS,
        "history_size": HISTORY_SIZE,
        "intercept_penalized": False,
    }
    for key, value in optimizer_expected.items():
        _require_equal(optimizer.get(key), value, f"bundle protocol optimizer {key}")
    bootstrap = _require_mapping(protocol.get("bootstrap"), "bundle protocol bootstrap")
    bootstrap_expected = {
        "cluster": "game",
        "paired": True,
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "refit": False,
    }
    for key, value in bootstrap_expected.items():
        _require_equal(bootstrap.get(key), value, f"bundle protocol bootstrap {key}")
    return payload, selected_l2


def _load_frozen_development_inputs(
    payload: Mapping[str, Any],
    *,
    project_root: Path,
) -> FrozenDevelopmentInputs:
    inputs = _require_mapping(
        payload.get("development_inputs"),
        "bundle.payload.development_inputs",
    )
    manifest_record = _require_mapping(
        inputs.get("manifest"),
        "bundle development manifest",
    )
    manifest_path = _verify_hash_record(
        manifest_record,
        project_root,
        "bundle development manifest",
    )
    manifest = _read_json_object(manifest_path)
    _require_equal(manifest.get("schema_version"), 2, "development manifest schema")
    _require_equal(
        manifest.get("development_years"),
        list(DEVELOPMENT_YEARS),
        "development manifest years",
    )
    _require_equal(
        manifest.get("validation_2024_touched"),
        False,
        "development manifest validation flag",
    )
    _require_equal(
        manifest.get("holdout_2025_touched"),
        False,
        "development manifest holdout flag",
    )
    years_document = _require_mapping(
        manifest.get("years"),
        "development manifest years",
    )
    _require_equal(
        set(years_document),
        {str(year) for year in DEVELOPMENT_YEARS},
        "development manifest year keys",
    )
    manifest_canonical_digest = canonical_json_sha256(manifest)

    cache_records_raw = _require_mapping(
        inputs.get("caches"),
        "bundle development caches",
    )
    _require_equal(
        set(cache_records_raw),
        {str(year) for year in DEVELOPMENT_YEARS},
        "bundle development cache years",
    )
    artifacts: list[YearSparseArtifact] = []
    cache_records: list[tuple[int, str, str]] = []
    for year in DEVELOPMENT_YEARS:
        year_document = _require_mapping(
            years_document[str(year)],
            f"development manifest {year}",
        )
        selected = year_document.get("selected")
        if not isinstance(selected, list) or len(selected) != VALIDATION_SAMPLE_SIZE:
            raise ValueError(f"development manifest {year} must select 1000 sources")
        sources = tuple(
            str(
                _require_mapping(item, f"manifest {year} selected item")[
                    "relative_source"
                ]
            )
            for item in selected
        )
        if len(set(sources)) != VALIDATION_SAMPLE_SIZE or any(
            not source.startswith(f"{year}/") for source in sources
        ):
            raise ValueError(f"development manifest {year} source list is invalid")

        record = _require_mapping(
            cache_records_raw[str(year)],
            f"bundle development cache {year}",
        )
        cache_path = _verify_hash_record(
            record,
            project_root,
            f"bundle development cache {year}",
        )
        artifact = load_year_sparse_artifact(cache_path)
        if artifact.year != year:
            raise ValueError(f"development cache {year} has the wrong year")
        if artifact.selected_source_count != VALIDATION_SAMPLE_SIZE:
            raise ValueError(f"development cache {year} source count is not 1000")
        if artifact.source_ids_by_rank != sources:
            raise ValueError(f"development cache {year} source order mismatch")
        provenance = dict(artifact.provenance)
        _require_equal(
            provenance.get("manifest_canonical_sha256"),
            manifest_canonical_digest,
            f"development cache {year} manifest hash",
        )
        _validate_artifact_rows(artifact, f"development cache {year}")
        digest = _sha256_text(record.get("sha256"), f"cache {year} sha256")
        artifacts.append(artifact)
        cache_records.append(
            (
                year,
                _relative_path(cache_path, project_root),
                digest,
            )
        )
    return FrozenDevelopmentInputs(
        artifacts=tuple(artifacts),
        manifest_path=manifest_path,
        manifest_file_sha256=file_sha256(manifest_path),
        manifest_canonical_sha256=manifest_canonical_digest,
        cache_records=tuple(cache_records),
    )


def _fit_frozen_models(
    artifacts: Sequence[YearSparseArtifact],
    *,
    selected_l2: float,
) -> FrozenModels:
    if tuple(artifact.year for artifact in artifacts) != DEVELOPMENT_YEARS:
        raise ValueError("development artifacts must be exactly 2020-2023 in order")
    vocabulary = global_feature_vocabulary(artifacts)
    problem = prepare_logistic_problem(
        artifacts,
        label=LABEL,
        vocabulary=vocabulary,
        source_prefix=VALIDATION_SAMPLE_SIZE,
    )
    fixed_options = {
        "l2": selected_l2,
        "max_iterations": MAX_ITERATIONS,
        "gradient_tolerance": GRADIENT_TOLERANCE,
        "function_tolerance": FUNCTION_TOLERANCE,
        "max_line_search_steps": MAX_LINE_SEARCH_STEPS,
        "history_size": HISTORY_SIZE,
    }
    base_mask = conventional_feature_mask(vocabulary)
    base = fit_sparse_problem(
        problem,
        active_features=base_mask,
        **fixed_options,
    )
    _require_accepted_fit(
        base,
        vocabulary=vocabulary,
        expected_active=base_mask,
        selected_l2=selected_l2,
        model_name=BASE_MODEL,
    )
    challenger_mask = np.ones(len(vocabulary), dtype=np.bool_)
    challenger = fit_sparse_problem(
        problem,
        **fixed_options,
    )
    _require_accepted_fit(
        challenger,
        vocabulary=vocabulary,
        expected_active=challenger_mask,
        selected_l2=selected_l2,
        model_name=CHALLENGER_MODEL,
    )
    return FrozenModels(
        vocabulary=vocabulary,
        problem=problem,
        base=base,
        challenger=challenger,
    )


def _require_accepted_fit(
    model: SparseLogisticModel,
    *,
    vocabulary: tuple[str, ...],
    expected_active: np.ndarray,
    selected_l2: float,
    model_name: str,
) -> None:
    diagnostics = model.diagnostics
    numeric_diagnostics = (
        diagnostics.initial_objective,
        diagnostics.final_objective,
        diagnostics.initial_gradient_inf_norm,
        diagnostics.final_gradient_inf_norm,
        diagnostics.total_weight,
        diagnostics.l2,
    )
    accepted = (
        diagnostics.converged
        and 0 <= diagnostics.iterations < MAX_ITERATIONS
        and all(math.isfinite(value) for value in numeric_diagnostics)
        and diagnostics.final_gradient_inf_norm <= GRADIENT_TOLERANCE
        and diagnostics.l2 == selected_l2
        and model.label == LABEL
        and model.feature_names == vocabulary
        and np.array_equal(model.active_mask, expected_active)
        and math.isfinite(model.intercept)
        and np.all(np.isfinite(model.coefficients))
        and np.all(model.coefficients[~expected_active] == 0.0)
    )
    if not accepted:
        raise RuntimeError(
            f"{model_name} failed the sealed optimizer gate: {asdict(diagnostics)}"
        )


def _validate_destination_state(paths: ValidationPaths) -> None:
    for path, name in (
        (paths.output, "result"),
        (paths.receipt, "receipt"),
    ):
        if path.exists():
            raise FileExistsError(
                f"{name} already exists and will not be overwritten: {path}"
            )
    prior_validation_artifact = paths.manifest.exists() or paths.cache.exists()
    if prior_validation_artifact and not paths.touch_marker.is_file():
        raise RuntimeError(
            "a validation manifest/cache exists without the mandatory touch marker"
        )
    if paths.cache.exists() and not paths.manifest.is_file():
        raise RuntimeError("validation cache exists without its manifest")


def _touch_document(
    paths: ValidationPaths,
    *,
    bundle_file_sha256: str,
    payload_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": TOUCH_SCHEMA_VERSION,
        "analysis_id": ANALYSIS_ID,
        "validation_year": VALIDATION_YEAR,
        "freeze_bundle": {
            "path": _relative_path(paths.freeze_bundle, paths.project_root),
            "file_sha256": bundle_file_sha256,
            "payload_sha256": payload_sha256,
        },
        "validation_2024_touched": True,
        "holdout_2025_touched": False,
    }


def _ensure_touch_marker(path: Path, expected: Mapping[str, Any]) -> None:
    if path.exists():
        actual = _read_json_object(path)
        _require_equal(actual, dict(expected), "touch marker")
        return
    _atomic_write_new_json(path, expected)


def _load_or_create_validation_manifest(
    paths: ValidationPaths,
    *,
    bundle: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    if not paths.manifest.exists():
        sample = select_validation_2024_sample(
            paths.raw_root,
            sample_size=VALIDATION_SAMPLE_SIZE,
            seed=VALIDATION_SAMPLE_SEED,
            year=VALIDATION_YEAR,
        )
        document = build_validation_2024_manifest(
            sample,
            freeze_bundle_path=paths.freeze_bundle,
            freeze_bundle=bundle,
        )
        write_validation_2024_manifest(
            paths.manifest,
            document,
            overwrite=False,
        )
    sample = load_validation_2024_manifest(
        paths.manifest,
        raw_root=paths.raw_root,
        freeze_bundle_path=paths.freeze_bundle,
        freeze_bundle=bundle,
    )
    if sample.year != VALIDATION_YEAR:
        raise ValueError("validation manifest sample has the wrong year")
    if len(sample.relative_sources) != VALIDATION_SAMPLE_SIZE:
        raise ValueError("validation manifest must select exactly 1000 sources")
    if len(set(sample.relative_sources)) != VALIDATION_SAMPLE_SIZE or any(
        not source.startswith(f"{VALIDATION_YEAR}/")
        for source in sample.relative_sources
    ):
        raise ValueError("validation manifest source list is invalid")
    return sample, _read_json_object(paths.manifest)


def _load_or_create_validation_cache(
    paths: ValidationPaths,
    *,
    sample: Any,
    manifest_file_sha256: str,
    manifest_canonical_sha256: str,
    bundle_file_sha256: str,
    payload_sha256: str,
) -> tuple[YearSparseArtifact, dict[str, int]]:
    required_provenance = {
        "analysis_id": ANALYSIS_ID,
        "freeze_bundle_file_sha256": bundle_file_sha256,
        "freeze_bundle_payload_sha256": payload_sha256,
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_canonical_sha256": manifest_canonical_sha256,
    }
    if not paths.cache.exists():
        dataset = collect_validation_2024_dataset(
            paths.raw_root,
            sample.paths,
            year=VALIDATION_YEAR,
            expected_content_sha256=sample.source_content_sha256,
        )
        if dataset.selected_sources != sample.relative_sources:
            raise ValueError("validation extraction source order differs from manifest")
        counts_json = json.dumps(
            dataset.counts,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        artifact = build_year_sparse_artifact(
            dataset.rows,
            year=VALIDATION_YEAR,
            source_rank_by_game={
                source: rank for rank, source in enumerate(sample.relative_sources)
            },
            provenance={**required_provenance, "counts_json": counts_json},
        )
        _save_new_sparse_artifact(paths.cache, artifact)
    artifact = load_year_sparse_artifact(paths.cache)
    if artifact.year != VALIDATION_YEAR:
        raise ValueError("validation cache has the wrong year")
    if artifact.source_ids_by_rank != sample.relative_sources:
        raise ValueError("validation cache source order differs from manifest")
    if artifact.selected_source_count != VALIDATION_SAMPLE_SIZE:
        raise ValueError("validation cache source count is not exactly 1000")
    provenance = dict(artifact.provenance)
    for key, expected in required_provenance.items():
        _require_equal(provenance.get(key), expected, f"validation cache {key}")
    raw_counts = provenance.get("counts_json")
    if not isinstance(raw_counts, str):
        raise TypeError("validation cache has no extraction counts")
    counts = json.loads(raw_counts)
    if not isinstance(counts, dict) or any(
        not isinstance(key, str)
        or isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for key, value in counts.items()
    ):
        raise ValueError("validation cache extraction counts are invalid")
    _validate_artifact_rows(artifact, "validation cache")
    expected_counts = {
        "scanned_files": VALIDATION_SAMPLE_SIZE,
        "candidate_rows": artifact.row_count,
        "decisions": len(artifact.group_ids),
        "structural_wait_rows": int(np.sum(artifact.y_structural)),
        "ron_eligible_rows": int(np.sum(artifact.y_ron)),
    }
    for name, expected in expected_counts.items():
        _require_equal(counts.get(name), expected, f"validation cache counts {name}")
    return artifact, dict(sorted(counts.items()))


def _save_new_sparse_artifact(path: Path, artifact: YearSparseArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.tmp-{uuid4().hex}.npz")
    try:
        save_year_sparse_artifact(staged, artifact)
        # A hard link publishes an already-complete file and fails if the
        # immutable destination appeared concurrently.
        os.link(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def _evaluate_fixed_predictions(
    models: FrozenModels,
    artifact: YearSparseArtifact,
) -> dict[str, Any]:
    examples, turn_bins = _examples_from_artifact(artifact)
    if set(turn_bins) != set(DECISION_TURN_BINS):
        raise RuntimeError("all four fixed turn bins must be represented")

    base_predictions = tuple(
        float(value) for value in models.base.predict_artifact(artifact)
    )
    challenger_predictions = tuple(
        float(value) for value in models.challenger.predict_artifact(artifact)
    )
    predictions = {
        BASE_MODEL: base_predictions,
        CHALLENGER_MODEL: challenger_predictions,
    }
    for model_name, values in predictions.items():
        if len(values) != len(examples) or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values
        ):
            raise RuntimeError(f"{model_name} produced invalid fixed predictions")
    prediction_sha256 = {
        name: canonical_json_sha256(list(values))
        for name, values in predictions.items()
    }

    metrics = {
        name: evaluate_binary_predictions(examples, values)
        for name, values in predictions.items()
    }
    turn_documents: dict[str, Any] = {}
    for turn_bin in DECISION_TURN_BINS:
        indices = tuple(
            index for index, value in enumerate(turn_bins) if value == turn_bin
        )
        bin_examples = tuple(examples[index] for index in indices)
        bin_metrics = {
            name: evaluate_binary_predictions(
                bin_examples,
                tuple(values[index] for index in indices),
            )
            for name, values in predictions.items()
        }
        turn_documents[turn_bin] = {
            "row_count": len(indices),
            "decision_count": len({example.group_id for example in bin_examples}),
            "game_count": len({example.cluster_id for example in bin_examples}),
            "models": {name: asdict(value) for name, value in bin_metrics.items()},
            "delta": _metric_delta(
                bin_metrics[BASE_MODEL],
                bin_metrics[CHALLENGER_MODEL],
            ),
        }

    bootstrap = joint_paired_game_cluster_bootstrap(
        examples,
        predictions,
        {COMPARISON_NAME: (BASE_MODEL, CHALLENGER_MODEL)},
        group_ids=turn_bins,
        group_order=DECISION_TURN_BINS,
        contrasts=(GroupContrast("late_minus_early", "13+", "1-6"),),
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    bootstrap_point = bootstrap.comparisons[0].overall.log_loss.point
    metric_point = metrics[CHALLENGER_MODEL].log_loss - metrics[BASE_MODEL].log_loss
    if bootstrap_point is None or not math.isclose(
        bootstrap_point,
        metric_point,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("bootstrap point does not match the fixed prediction delta")
    # Diagnostic calibration is allowed to inspect predictions but never to
    # replace them.  Immutable tuples and this hash recheck enforce that.
    for name, values in predictions.items():
        if canonical_json_sha256(list(values)) != prediction_sha256[name]:
            raise RuntimeError("fixed predictions changed during evaluation")

    validation_only = tuple(
        sorted(set(artifact.feature_names) - set(models.vocabulary))
    )
    return {
        "examples": examples,
        "turn_bins": turn_documents,
        "metrics": metrics,
        "bootstrap": bootstrap,
        "prediction_sha256": prediction_sha256,
        "validation_only_features": validation_only,
    }


def _primary_validation_decision(
    bootstrap: JointBootstrapResult,
) -> dict[str, Any]:
    if bootstrap.requested_replicates != BOOTSTRAP_REPLICATES:
        raise RuntimeError("bootstrap replicate count differs from the fixed protocol")
    if bootstrap.seed != BOOTSTRAP_SEED:
        raise RuntimeError("bootstrap seed differs from the fixed protocol")
    if bootstrap.stratified:
        raise RuntimeError("single-year validation bootstrap must not be stratified")
    if len(bootstrap.comparisons) != 1:
        raise RuntimeError("bootstrap must contain exactly one primary comparison")
    comparison = bootstrap.comparisons[0]
    if (
        comparison.name != COMPARISON_NAME
        or comparison.base_model != BASE_MODEL
        or comparison.challenger_model != CHALLENGER_MODEL
    ):
        raise RuntimeError("bootstrap primary comparison is not the sealed comparison")
    interval = comparison.overall.log_loss
    values = (interval.point, interval.lower, interval.upper)
    if interval.valid_replicates != BOOTSTRAP_REPLICATES or any(
        value is None or not math.isfinite(value) for value in values
    ):
        raise RuntimeError("primary bootstrap interval is incomplete")
    point = float(interval.point)
    lower = float(interval.lower)
    upper = float(interval.upper)
    return {
        "metric": "weighted_log_loss_delta",
        "estimand": "conventional_simple - conventional",
        "point": point,
        "percentile_95_lower": lower,
        "percentile_95_upper": upper,
        "valid_replicates": interval.valid_replicates,
        "required_replicates": BOOTSTRAP_REPLICATES,
        "point_below_zero": point < 0.0,
        "upper_below_zero": upper < 0.0,
        "success": point < 0.0 and upper < 0.0,
    }


def _examples_from_artifact(
    artifact: YearSparseArtifact,
) -> tuple[tuple[SparseExample, ...], tuple[str, ...]]:
    if artifact.year != VALIDATION_YEAR:
        raise ValueError("examples must come from the 2024 validation artifact")
    labels = artifact.labels(LABEL)
    examples = tuple(
        SparseExample(
            row_id=f"{VALIDATION_YEAR}:{position}",
            group_id=artifact.group_ids[int(artifact.group_indices[position])],
            cluster_id=artifact.cluster_ids[int(artifact.cluster_indices[position])],
            label=bool(labels[position]),
            weight=float(artifact.weights[position]),
            features=(),
        )
        for position in range(artifact.row_count)
    )
    turn_bins = tuple(
        DECISION_TURN_BINS[int(artifact.turn_bins[position])]
        for position in range(artifact.row_count)
    )
    return examples, turn_bins


def _validate_artifact_rows(artifact: YearSparseArtifact, field: str) -> None:
    if (
        artifact.row_count < 1
        or artifact.selected_source_count != VALIDATION_SAMPLE_SIZE
    ):
        raise ValueError(f"{field} has invalid row/source counts")
    if not np.all(np.isfinite(artifact.matrix.data)):
        raise ValueError(f"{field} contains non-finite feature values")
    if not np.all(np.isfinite(artifact.weights)) or np.any(artifact.weights <= 0):
        raise ValueError(f"{field} contains invalid weights")
    group_count = len(artifact.group_ids)
    observed = np.bincount(artifact.group_indices, minlength=group_count)
    weight_sums = np.bincount(
        artifact.group_indices,
        weights=artifact.weights,
        minlength=group_count,
    )
    if np.any(observed < 1) or not np.allclose(weight_sums, 1.0, rtol=0, atol=1e-12):
        raise ValueError(f"{field} decision weights do not sum to one")
    expected_weights = 1.0 / observed[artifact.group_indices]
    if not np.allclose(artifact.weights, expected_weights, rtol=0, atol=1e-15):
        raise ValueError(f"{field} weights are not 1/candidate_count")
    source_rank_by_id = {
        source: rank for rank, source in enumerate(artifact.source_ids_by_rank)
    }
    try:
        cluster_source_ranks = np.fromiter(
            (source_rank_by_id[source] for source in artifact.cluster_ids),
            dtype=np.int32,
            count=len(artifact.cluster_ids),
        )
    except KeyError as error:
        raise ValueError(f"{field} cluster is absent from selected sources") from error
    if not np.array_equal(
        cluster_source_ranks[artifact.cluster_indices],
        artifact.source_ranks,
    ):
        raise ValueError(f"{field} row source ranks do not match game clusters")


def _model_document(model: SparseLogisticModel) -> dict[str, Any]:
    return {
        "intercept": model.intercept,
        "active_feature_count": int(np.sum(model.active_mask)),
        "coefficient_sha256": canonical_json_sha256(
            {
                "feature_names": list(model.feature_names),
                "coefficients": [float(value) for value in model.coefficients],
            }
        ),
        "diagnostics": asdict(model.diagnostics),
    }


def _metric_delta(
    base: BinaryMetrics,
    challenger: BinaryMetrics,
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


def _save_json_staged(path: Path, document: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    rendered = (
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=_json_default,
        )
        + "\n"
    )
    try:
        with staged.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return staged


def _atomic_write_new_json(path: Path, document: Mapping[str, Any]) -> None:
    staged = _save_json_staged(path, document)
    try:
        os.link(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def _read_json_object(path: Path) -> dict[str, Any]:
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
        raise TypeError(f"JSON root must be an object: {path}")
    return loaded


def _json_normalized_object(document: Mapping[str, Any]) -> dict[str, Any]:
    rendered = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    )
    loaded = json.loads(rendered)
    if not isinstance(loaded, dict):
        raise TypeError("normalized JSON root must be an object")
    return loaded


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _verify_hash_record(
    record: Mapping[str, Any],
    project_root: Path,
    field: str,
) -> Path:
    path = _resolve_record_path(record, project_root, field)
    expected = _sha256_text(record.get("sha256"), f"{field}.sha256")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{field} SHA-256 mismatch: expected {expected}, got {actual}")
    return path


def _resolve_record_path(
    record: Mapping[str, Any],
    project_root: Path,
    field: str,
) -> Path:
    allowed_keys = {"path", "sha256", "payload_sha256"}
    if not {"path", "sha256"}.issubset(record) or set(record) - allowed_keys:
        raise ValueError(f"{field} has an unexpected record shape")
    raw_path = record.get("path")
    if not isinstance(raw_path, str):
        raise TypeError(f"{field}.path must be a string")
    pure = PurePosixPath(raw_path)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"{field}.path is unsafe")
    path = project_root.resolve().joinpath(*pure.parts).resolve()
    _relative_path(path, project_root)
    if not path.is_file():
        raise ValueError(f"{field} target does not exist: {path}")
    return path


def _relative_path(path: Path, project_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(project_root.resolve())
    except ValueError as error:
        raise ValueError(f"path is outside project root: {path}") from error
    if not relative.parts:
        raise ValueError("project root itself cannot be an artifact path")
    return relative.as_posix()


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object")
    return value


def _require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected or isinstance(actual, bool) != isinstance(expected, bool):
        raise ValueError(f"{field} mismatch: expected {expected!r}, got {actual!r}")


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _optional_finite_number(value: Any, field: str) -> float | None:
    return None if value is None else _finite_number(value, field)


def _nonnegative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < 0:
        raise ValueError(f"{field} must be nonnegative")
    return value


def _require_close(
    actual: Any,
    expected: float,
    field: str,
    *,
    absolute_tolerance: float = 1e-12,
) -> None:
    checked = _finite_number(actual, field)
    if not math.isclose(
        checked,
        expected,
        rel_tol=0.0,
        abs_tol=absolute_tolerance,
    ):
        raise ValueError(f"{field} mismatch: expected {expected!r}, got {checked!r}")


def _require_optional_close(
    actual: float | None,
    expected: float | None,
    field: str,
) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(f"{field} mismatch: expected {expected!r}, got {actual!r}")
        return
    _require_close(actual, expected, field)


def _sha256_text(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
