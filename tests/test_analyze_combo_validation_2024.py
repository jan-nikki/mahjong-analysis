from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.special import expit

import analysis.analyze_combo_validation_2024 as validation
from mahjong_analysis.combo_bootstrap import (
    ComparisonBootstrapResult,
    JointBootstrapResult,
    MetricBootstrapIntervals,
)
from mahjong_analysis.combo_prediction import BootstrapInterval
from mahjong_analysis.combo_sparse_pipeline import (
    LbfgsDiagnostics,
    SparseLogisticModel,
    YearSparseArtifact,
    save_year_sparse_artifact,
)


def test_cli_has_paths_but_no_validation_or_tuning_knobs(tmp_path: Path) -> None:
    args = validation.parse_args(["--project-root", str(tmp_path)])
    paths = validation.paths_from_args(args)

    assert args.recover_receipt is False
    assert validation.parse_args(["--recover-receipt"]).recover_receipt is True
    assert paths.project_root == tmp_path.resolve()
    assert paths.raw_root == (tmp_path / "data" / "raw").resolve()
    assert paths.preflight_report.name == "combo-pre2024-test-preflight-v1.json"
    for forbidden in (
        "year",
        "sample_size",
        "seed",
        "l2",
        "bootstrap_replicates",
        "max_iterations",
        "gradient_tolerance",
    ):
        assert not hasattr(args, forbidden)

    with pytest.raises(SystemExit):
        validation.parse_args(["--year", "2024"])
    with pytest.raises(SystemExit):
        validation.parse_args(["--l2", "0.001"])


def test_bundle_failure_occurs_before_any_2024_touch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    sampled = False

    def fail_bundle(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise ValueError("freeze bundle rejected")

    def forbidden_sample(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal sampled
        sampled = True
        raise AssertionError("2024 sampler must not run")

    monkeypatch.setattr(validation, "verify_freeze_bundle", fail_bundle)
    monkeypatch.setattr(
        validation,
        "select_validation_2024_sample",
        forbidden_sample,
    )

    with pytest.raises(ValueError, match="freeze bundle rejected"):
        validation.run_validation(paths)

    assert not sampled
    assert not paths.touch_marker.exists()


def test_preflight_failure_occurs_before_dev_fit_or_2024_touch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    paths.freeze_bundle.parent.mkdir(parents=True)
    paths.freeze_bundle.write_text("bundle", encoding="utf-8")
    events: list[str] = []

    def verify_bundle(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        events.append("bundle")
        return {"payload_sha256": "a" * 64}

    def fail_preflight(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        events.append("preflight")
        raise ValueError("preflight rejected")

    monkeypatch.setattr(validation, "verify_freeze_bundle", verify_bundle)
    monkeypatch.setattr(validation, "verify_preflight_report", fail_preflight)
    monkeypatch.setattr(
        validation,
        "_load_frozen_development_inputs",
        lambda *_args, **_kwargs: events.append("development"),
    )

    with pytest.raises(ValueError, match="preflight rejected"):
        validation.run_validation(paths)

    assert events == ["bundle", "preflight"]
    assert not paths.touch_marker.exists()


def test_development_fit_failure_occurs_before_touch_or_sampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    paths.freeze_bundle.parent.mkdir(parents=True)
    paths.freeze_bundle.write_text("bundle", encoding="utf-8")
    paths.preflight_report.write_text("preflight", encoding="utf-8")
    sampled = False

    monkeypatch.setattr(
        validation,
        "verify_freeze_bundle",
        lambda *_args, **_kwargs: {"payload_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        validation,
        "verify_preflight_report",
        lambda *_args, **_kwargs: {"status": "PASS"},
    )
    monkeypatch.setattr(
        validation,
        "_validated_bundle_protocol",
        lambda _bundle: ({}, 3e-4),
    )
    monkeypatch.setattr(
        validation,
        "_load_frozen_development_inputs",
        lambda *_args, **_kwargs: SimpleNamespace(artifacts=()),
    )

    def fail_fit(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("sealed fit failed")

    def forbidden_sample(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal sampled
        sampled = True
        raise AssertionError("2024 sampler must not run")

    monkeypatch.setattr(validation, "_fit_frozen_models", fail_fit)
    monkeypatch.setattr(
        validation,
        "select_validation_2024_sample",
        forbidden_sample,
    )

    with pytest.raises(RuntimeError, match="sealed fit failed"):
        validation.run_validation(paths)

    assert not sampled
    assert not paths.touch_marker.exists()


def test_touch_marker_precedes_validation_manifest_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    paths.freeze_bundle.parent.mkdir(parents=True)
    paths.freeze_bundle.write_text("bundle", encoding="utf-8")
    paths.preflight_report.write_text("preflight", encoding="utf-8")
    events: list[str] = []

    monkeypatch.setattr(
        validation,
        "verify_freeze_bundle",
        lambda *_args, **_kwargs: {"payload_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        validation,
        "verify_preflight_report",
        lambda *_args, **_kwargs: {"status": "PASS"},
    )
    monkeypatch.setattr(
        validation,
        "_validated_bundle_protocol",
        lambda _bundle: ({}, 3e-4),
    )
    monkeypatch.setattr(
        validation,
        "_load_frozen_development_inputs",
        lambda *_args, **_kwargs: SimpleNamespace(artifacts=()),
    )
    monkeypatch.setattr(
        validation,
        "_fit_frozen_models",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        validation,
        "_ensure_touch_marker",
        lambda *_args, **_kwargs: events.append("touch"),
    )

    def stop_at_manifest(*_args: Any, **_kwargs: Any) -> Any:
        events.append("manifest")
        raise RuntimeError("stop after order check")

    monkeypatch.setattr(
        validation,
        "_load_or_create_validation_manifest",
        stop_at_manifest,
    )

    with pytest.raises(RuntimeError, match="order check"):
        validation.run_validation(paths)

    assert events == ["touch", "manifest"]


def test_fixed_fit_uses_dev_vocabulary_and_bundle_lambda_for_both_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = tuple(
        SimpleNamespace(
            year=year,
            feature_names=("base:known", "combo:known")
            if year != 2023
            else ("base:known", "combo:known", "base:late"),
        )
        for year in validation.DEVELOPMENT_YEARS
    )
    prepared = object()
    calls: list[dict[str, Any]] = []

    def fake_prepare(
        received: Any,
        *,
        label: str,
        vocabulary: tuple[str, ...],
        source_prefix: int,
    ) -> object:
        assert tuple(received) == artifacts
        assert label == "ron_eligible"
        assert vocabulary == ("base:known", "base:late", "combo:known")
        assert source_prefix == 1_000
        return prepared

    def fake_fit(
        problem: object,
        *,
        active_features: np.ndarray | None = None,
        **options: Any,
    ) -> SparseLogisticModel:
        assert problem is prepared
        active = (
            np.ones(3, dtype=np.bool_)
            if active_features is None
            else np.asarray(active_features, dtype=np.bool_)
        )
        calls.append({"active": active.copy(), **options})
        return _model(
            ("base:known", "base:late", "combo:known"),
            active,
            l2=float(options["l2"]),
        )

    monkeypatch.setattr(validation, "prepare_logistic_problem", fake_prepare)
    monkeypatch.setattr(validation, "fit_sparse_problem", fake_fit)

    fitted = validation._fit_frozen_models(artifacts, selected_l2=3e-4)  # type: ignore[arg-type]

    assert fitted.vocabulary == ("base:known", "base:late", "combo:known")
    assert len(calls) == 2
    assert all(call["l2"] == 3e-4 for call in calls)
    assert all(call["max_iterations"] == 300 for call in calls)
    assert all(call["gradient_tolerance"] == 1e-7 for call in calls)
    assert all(call["function_tolerance"] == 0.0 for call in calls)
    assert all(call["max_line_search_steps"] == 50 for call in calls)
    assert all(call["history_size"] == 20 for call in calls)
    np.testing.assert_array_equal(calls[0]["active"], [True, True, False])
    np.testing.assert_array_equal(calls[1]["active"], [True, True, True])


def test_solver_gate_rejects_bad_gradient_and_nonfinite_coefficients() -> None:
    vocabulary = ("base:known",)
    active = np.ones(1, dtype=np.bool_)
    accepted = _model(vocabulary, active, l2=0.001)

    bad_gradient = replace(
        accepted,
        diagnostics=replace(
            accepted.diagnostics,
            final_gradient_inf_norm=1.01e-7,
        ),
    )
    with pytest.raises(RuntimeError, match="optimizer gate"):
        validation._require_accepted_fit(
            bad_gradient,
            vocabulary=vocabulary,
            expected_active=active,
            selected_l2=0.001,
            model_name="base",
        )

    bad_coefficients = replace(
        accepted,
        coefficients=np.asarray([np.nan], dtype=np.float64),
    )
    with pytest.raises(RuntimeError, match="optimizer gate"):
        validation._require_accepted_fit(
            bad_coefficients,
            vocabulary=vocabulary,
            expected_active=active,
            selected_l2=0.001,
            model_name="base",
        )


def test_validation_only_feature_is_unknown_and_predictions_stay_untouched() -> None:
    artifact = _validation_artifact()
    vocabulary = ("base:known",)
    active = np.ones(1, dtype=np.bool_)
    base = _model(vocabulary, active, l2=0.001, intercept=-0.4, coefficient=0.2)
    challenger = _model(
        vocabulary,
        active,
        l2=0.001,
        intercept=-0.4,
        coefficient=0.4,
    )
    models = validation.FrozenModels(
        vocabulary=vocabulary,
        problem=SimpleNamespace(),  # type: ignore[arg-type]
        base=base,
        challenger=challenger,
    )

    evaluation = validation._evaluate_fixed_predictions(models, artifact)

    assert evaluation["validation_only_features"] == ("validation:only",)
    known = artifact.matrix[:, 0].toarray().ravel()
    expected = tuple(float(value) for value in expit(-0.4 + 0.2 * known))
    assert evaluation["prediction_sha256"]["conventional"] == (
        validation.canonical_json_sha256(list(expected))
    )
    assert evaluation["bootstrap"].requested_replicates == 2_000
    assert evaluation["bootstrap"].seed == 20260923


@pytest.mark.parametrize(
    ("point", "upper", "expected"),
    [
        (-0.01, -0.001, True),
        (0.0, -0.001, False),
        (-0.01, 0.0, False),
        (0.01, 0.02, False),
    ],
)
def test_primary_success_rule_is_strict(
    point: float,
    upper: float,
    expected: bool,
) -> None:
    decision = validation._primary_validation_decision(
        _bootstrap(point=point, upper=upper)
    )
    assert decision["success"] is expected


def test_primary_requires_all_fixed_bootstrap_replicates() -> None:
    with pytest.raises(RuntimeError, match="incomplete"):
        validation._primary_validation_decision(
            _bootstrap(point=-0.01, upper=-0.001, valid_replicates=1_999)
        )


def test_contrast_bootstrap_accepts_bounded_metric_interval_at_1_5() -> None:
    points = {
        metric: 0.0 if metric == "log_loss" else 1.5
        for metric in validation.METRIC_NAMES
    }
    intervals = _bootstrap_metric_document(points)
    for metric in validation.METRIC_NAMES:
        if metric != "log_loss":
            intervals[metric].update({"point": 1.5, "lower": 1.5, "upper": 1.5})

    validation._validate_bootstrap_metric_set(
        intervals,
        expected_points=points,
        field="artificial contrast",
        non_log_loss_absolute_limit=2.0,
    )


def test_contrast_bootstrap_rejects_bounded_metric_beyond_2() -> None:
    points = {metric: 0.0 for metric in validation.METRIC_NAMES}
    points["auc"] = 2.000_001
    intervals = _bootstrap_metric_document(points)

    with pytest.raises(ValueError, match=r"absolute limit 2\.0"):
        validation._validate_bootstrap_metric_set(
            intervals,
            expected_points=points,
            field="artificial contrast",
            non_log_loss_absolute_limit=2.0,
        )


def test_atomic_json_and_touch_marker_are_immutable(tmp_path: Path) -> None:
    output = tmp_path / "outputs" / "artifact.json"
    validation._atomic_write_new_json(output, {"value": 1})
    first = output.read_bytes()

    with pytest.raises(FileExistsError):
        validation._atomic_write_new_json(output, {"value": 2})
    assert output.read_bytes() == first

    marker = tmp_path / "outputs" / "touch.json"
    expected = {"validation_2024_touched": True, "holdout_2025_touched": False}
    validation._ensure_touch_marker(marker, expected)
    marker.write_text(
        json.dumps({"validation_2024_touched": True, "holdout_2025_touched": True}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="touch marker"):
        validation._ensure_touch_marker(marker, expected)


def test_hash_record_detects_content_tampering(tmp_path: Path) -> None:
    target = tmp_path / "outputs" / "target.json"
    target.parent.mkdir(parents=True)
    target.write_text("original\n", encoding="utf-8")
    record = {
        "path": "outputs/target.json",
        "sha256": validation.file_sha256(target),
    }
    assert validation._verify_hash_record(record, tmp_path, "target") == target

    target.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validation._verify_hash_record(record, tmp_path, "target")


def test_loads_all_four_frozen_caches_and_checks_manifest_order(
    tmp_path: Path,
) -> None:
    payload = _development_payload(tmp_path)
    loaded = validation._load_frozen_development_inputs(
        payload,
        project_root=tmp_path,
    )

    assert tuple(artifact.year for artifact in loaded.artifacts) == (
        2020,
        2021,
        2022,
        2023,
    )
    assert len(loaded.cache_records) == 4


def test_development_cache_source_order_tampering_is_rejected(
    tmp_path: Path,
) -> None:
    payload = _development_payload(tmp_path, wrong_order_year=2022)

    with pytest.raises(ValueError, match="source order mismatch"):
        validation._load_frozen_development_inputs(
            payload,
            project_root=tmp_path,
        )


def test_existing_validation_cache_with_wrong_binding_is_rejected(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    sources = tuple(f"2024/{index:04d}.mjson" for index in range(1_000))
    artifact = _one_row_artifact(
        2024,
        sources,
        provenance={
            "analysis_id": validation.ANALYSIS_ID,
            "freeze_bundle_file_sha256": "wrong",
            "freeze_bundle_payload_sha256": "b" * 64,
            "manifest_file_sha256": "c" * 64,
            "manifest_canonical_sha256": "d" * 64,
            "counts_json": "{}",
        },
    )
    paths.cache.parent.mkdir(parents=True)
    save_year_sparse_artifact(paths.cache, artifact)
    sample = SimpleNamespace(
        paths=(),
        relative_sources=sources,
    )

    with pytest.raises(ValueError, match="freeze_bundle_file_sha256"):
        validation._load_or_create_validation_cache(
            paths,
            sample=sample,
            manifest_file_sha256="c" * 64,
            manifest_canonical_sha256="d" * 64,
            bundle_file_sha256="a" * 64,
            payload_sha256="b" * 64,
        )


def test_receipt_recovery_never_runs_raw_fit_evaluation_or_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, bundle, preflight = _recovery_fixture(tmp_path)
    _stub_recovery_authorities(monkeypatch, bundle=bundle, preflight=preflight)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("receipt recovery entered the validation pipeline")

    for name in (
        "select_validation_2024_sample",
        "load_validation_2024_manifest",
        "collect_validation_2024_dataset",
        "_fit_frozen_models",
        "_evaluate_fixed_predictions",
        "joint_paired_game_cluster_bootstrap",
    ):
        monkeypatch.setattr(validation, name, forbidden)

    recovered_result, receipt = validation.recover_validation_receipt(paths)

    assert recovered_result["status"] == "VALIDATION_SUCCESS"
    assert receipt["result"]["sha256"] == validation.file_sha256(paths.output)
    assert receipt["validation_manifest"]["sha256"] == validation.file_sha256(
        paths.manifest
    )
    assert paths.receipt.is_file()
    assert not paths.raw_root.exists()


def test_normal_run_rejects_existing_result_before_dev_load_or_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, bundle, preflight = _recovery_fixture(tmp_path)
    _stub_recovery_authorities(monkeypatch, bundle=bundle, preflight=preflight)
    original_result = paths.output.read_bytes()

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("existing result must fail before development work")

    monkeypatch.setattr(validation, "_load_frozen_development_inputs", forbidden)
    monkeypatch.setattr(validation, "_fit_frozen_models", forbidden)
    monkeypatch.setattr(validation, "select_validation_2024_sample", forbidden)

    with pytest.raises(FileExistsError, match="result already exists"):
        validation.run_validation(paths)

    assert paths.output.read_bytes() == original_result
    assert not paths.receipt.exists()


def test_receipt_recovery_rejects_tampered_result_hash_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, bundle, preflight = _recovery_fixture(tmp_path)
    _stub_recovery_authorities(monkeypatch, bundle=bundle, preflight=preflight)
    result = json.loads(paths.output.read_text(encoding="utf-8"))
    result["validation_inputs"]["manifest"]["file_sha256"] = "0" * 64
    _write_json(paths.output, result)

    with pytest.raises(ValueError, match=r"manifest\.file_sha256"):
        validation.recover_validation_receipt(paths)

    assert not paths.receipt.exists()


@pytest.mark.parametrize(
    ("operation", "field_path", "replacement"),
    [
        ("delete", ("optimizer", "models"), None),
        (
            "set",
            (
                "optimizer",
                "models",
                validation.BASE_MODEL,
                "diagnostics",
                "converged",
            ),
            False,
        ),
        ("set", ("feature_vocabulary", "feature_count"), 99),
        ("set", ("counts", "games"), 99),
        (
            "set",
            ("predictions", validation.BASE_MODEL, "metrics", "brier"),
            1.5,
        ),
        (
            "set",
            (
                "predictions",
                validation.BASE_MODEL,
                "metrics",
                "calibration",
                "clip_epsilon",
            ),
            0.01,
        ),
        ("delete", ("predictions", validation.BASE_MODEL, "metrics", "ece"), None),
        ("set", ("turn_bins", "1-6", "delta", "log_loss"), 0.0),
        (
            "set",
            (
                "joint_game_cluster_bootstrap",
                "comparisons",
                0,
                "by_group",
                0,
                "metrics",
                "log_loss",
                "valid_replicates",
            ),
            1_999,
        ),
        ("set", ("environment", "numpy"), ""),
    ],
    ids=(
        "missing-optimizer-models",
        "unaccepted-fit",
        "vocabulary-count",
        "result-counts",
        "metric-range",
        "calibration-config",
        "missing-metric",
        "turn-bin-delta",
        "bootstrap-interval",
        "environment",
    ),
)
def test_receipt_recovery_rejects_nested_result_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    field_path: tuple[str | int, ...],
    replacement: Any,
) -> None:
    paths, bundle, preflight = _recovery_fixture(tmp_path)
    _stub_recovery_authorities(monkeypatch, bundle=bundle, preflight=preflight)
    result = json.loads(paths.output.read_text(encoding="utf-8"))
    target: Any = result
    for key in field_path[:-1]:
        target = target[key]
    final_key = field_path[-1]
    if operation == "delete":
        del target[final_key]
    else:
        target[final_key] = replacement
    _write_json(paths.output, result)

    with pytest.raises((TypeError, ValueError)):
        validation.recover_validation_receipt(paths)

    assert not paths.receipt.exists()


@pytest.mark.parametrize("artifact_name", ["manifest", "cache", "touch_marker"])
def test_receipt_recovery_rejects_tampered_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
) -> None:
    paths, bundle, preflight = _recovery_fixture(tmp_path)
    _stub_recovery_authorities(monkeypatch, bundle=bundle, preflight=preflight)
    artifact_path = {
        "manifest": paths.manifest,
        "cache": paths.cache,
        "touch_marker": paths.touch_marker,
    }[artifact_name]
    artifact_path.write_bytes(artifact_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="sha256 mismatch"):
        validation.recover_validation_receipt(paths)

    assert not paths.receipt.exists()


def test_existing_valid_receipt_is_verified_without_republication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, bundle, preflight = _recovery_fixture(tmp_path)
    _stub_recovery_authorities(monkeypatch, bundle=bundle, preflight=preflight)
    result = json.loads(paths.output.read_text(encoding="utf-8"))
    expected = validation._receipt_document(
        paths,
        result,
        result_sha256=validation.file_sha256(paths.output),
    )
    _write_json(paths.receipt, expected)
    original_receipt = paths.receipt.read_bytes()

    def forbidden_write(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("an existing valid receipt must not be republished")

    monkeypatch.setattr(validation, "_atomic_write_new_json", forbidden_write)

    _result, recovered = validation.recover_validation_receipt(paths)

    assert recovered == expected
    assert paths.receipt.read_bytes() == original_receipt


def test_existing_mismatched_receipt_is_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, bundle, preflight = _recovery_fixture(tmp_path)
    _stub_recovery_authorities(monkeypatch, bundle=bundle, preflight=preflight)
    _write_json(paths.receipt, {"status": "tampered"})
    original_receipt = paths.receipt.read_bytes()

    with pytest.raises(ValueError, match="existing validation receipt"):
        validation.recover_validation_receipt(paths)

    assert paths.receipt.read_bytes() == original_receipt


def _paths(root: Path) -> validation.ValidationPaths:
    return validation.ValidationPaths(
        project_root=root.resolve(),
        freeze_bundle=(root / "outputs" / "freeze.json").resolve(),
        preflight_report=(root / "outputs" / "preflight.json").resolve(),
        raw_root=(root / "data" / "raw").resolve(),
        manifest=(root / "outputs" / "validation.manifest.json").resolve(),
        cache=(root / "data" / "processed" / "validation.npz").resolve(),
        touch_marker=(root / "outputs" / "touch.json").resolve(),
        output=(root / "outputs" / "result.json").resolve(),
        receipt=(root / "outputs" / "receipt.json").resolve(),
    )


def _diagnostics(*, l2: float, active_count: int) -> LbfgsDiagnostics:
    return LbfgsDiagnostics(
        converged=True,
        status=0,
        message="ok",
        iterations=12,
        function_evaluations=15,
        gradient_evaluations=15,
        initial_objective=0.5,
        final_objective=0.2,
        initial_gradient_inf_norm=0.1,
        final_gradient_inf_norm=1e-8,
        row_count=8,
        total_weight=4.0,
        active_feature_count=active_count,
        l2=l2,
    )


def _model(
    vocabulary: tuple[str, ...],
    active: np.ndarray,
    *,
    l2: float,
    intercept: float = 0.0,
    coefficient: float = 0.0,
) -> SparseLogisticModel:
    coefficients = np.zeros(len(vocabulary), dtype=np.float64)
    coefficients[active] = coefficient
    return SparseLogisticModel(
        label="ron_eligible",
        feature_names=vocabulary,
        active_mask=np.asarray(active, dtype=np.bool_),
        intercept=intercept,
        coefficients=coefficients,
        diagnostics=_diagnostics(l2=l2, active_count=int(np.sum(active))),
    )


def _validation_artifact() -> YearSparseArtifact:
    known = np.asarray([0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 0.5, 0.5])
    validation_only = np.asarray([100.0, -100.0] * 4)
    matrix = csr_matrix(np.column_stack((known, validation_only)))
    groups = np.repeat(np.arange(4, dtype=np.int32), 2)
    return YearSparseArtifact(
        year=2024,
        feature_names=("base:known", "validation:only"),
        matrix=matrix,
        y_structural=np.asarray([False, True] * 4, dtype=np.bool_),
        y_ron=np.asarray([False, True] * 4, dtype=np.bool_),
        weights=np.full(8, 0.5, dtype=np.float64),
        group_indices=groups,
        group_ids=tuple(f"decision-{index}" for index in range(4)),
        cluster_indices=groups.copy(),
        cluster_ids=tuple(f"game-{index}" for index in range(4)),
        turn_bins=np.repeat(np.arange(4, dtype=np.uint8), 2),
        source_ranks=np.zeros(8, dtype=np.int32),
        source_ids_by_rank=("2024/game.mjson",),
    )


def _bootstrap(
    *,
    point: float,
    upper: float,
    valid_replicates: int = 2_000,
) -> JointBootstrapResult:
    interval = BootstrapInterval(
        point=point,
        lower=point - 0.01,
        upper=upper,
        valid_replicates=valid_replicates,
    )
    intervals = MetricBootstrapIntervals(
        auc=interval,
        log_loss=interval,
        brier=interval,
        ece=interval,
        macro_concordance=interval,
    )
    comparison = ComparisonBootstrapResult(
        name=validation.COMPARISON_NAME,
        base_model=validation.BASE_MODEL,
        challenger_model=validation.CHALLENGER_MODEL,
        overall=intervals,
        by_group=(),
        contrasts=(),
    )
    return JointBootstrapResult(
        cluster_count=4,
        requested_replicates=2_000,
        seed=20260923,
        ece_bins=10,
        stratified=False,
        stratum_cluster_counts=(("all", 4),),
        group_order=("1-6", "7-9", "10-12", "13+"),
        comparisons=(comparison,),
    )


def _development_payload(
    root: Path,
    *,
    wrong_order_year: int | None = None,
) -> dict[str, Any]:
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    years: dict[str, Any] = {}
    sources_by_year: dict[int, tuple[str, ...]] = {}
    for year in validation.DEVELOPMENT_YEARS:
        sources = tuple(f"{year}/{index:04d}.mjson" for index in range(1_000))
        sources_by_year[year] = sources
        years[str(year)] = {
            "selected": [{"relative_source": source} for source in sources]
        }
    manifest = {
        "schema_version": 2,
        "development_years": list(validation.DEVELOPMENT_YEARS),
        "validation_2024_touched": False,
        "holdout_2025_touched": False,
        "years": years,
    }
    manifest_path = outputs / "development.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_digest = validation.canonical_json_sha256(manifest)

    cache_records: dict[str, dict[str, str]] = {}
    for year in validation.DEVELOPMENT_YEARS:
        sources = sources_by_year[year]
        if year == wrong_order_year:
            sources = tuple(reversed(sources))
        artifact = _one_row_artifact(
            year,
            sources,
            provenance={"manifest_canonical_sha256": manifest_digest},
        )
        cache_path = root / "data" / "processed" / f"{year}.npz"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_year_sparse_artifact(cache_path, artifact)
        cache_records[str(year)] = {
            "path": cache_path.relative_to(root).as_posix(),
            "sha256": validation.file_sha256(cache_path),
        }
    return {
        "development_inputs": {
            "manifest": {
                "path": manifest_path.relative_to(root).as_posix(),
                "sha256": validation.file_sha256(manifest_path),
            },
            "caches": cache_records,
        }
    }


def _one_row_artifact(
    year: int,
    sources: tuple[str, ...],
    *,
    provenance: dict[str, str],
) -> YearSparseArtifact:
    return YearSparseArtifact(
        year=year,
        feature_names=("base:known",),
        matrix=csr_matrix(np.asarray([[1.0]], dtype=np.float64)),
        y_structural=np.asarray([False], dtype=np.bool_),
        y_ron=np.asarray([False], dtype=np.bool_),
        weights=np.asarray([1.0], dtype=np.float64),
        group_indices=np.asarray([0], dtype=np.int32),
        group_ids=(f"{year}:decision",),
        cluster_indices=np.asarray([0], dtype=np.int32),
        cluster_ids=(sources[0],),
        turn_bins=np.asarray([0], dtype=np.uint8),
        source_ranks=np.asarray([0], dtype=np.int32),
        source_ids_by_rank=sources,
        provenance=tuple(sorted(provenance.items())),
    )


def _recovery_fixture_legacy(
    root: Path,
) -> tuple[
    validation.ValidationPaths,
    dict[str, Any],
    dict[str, Any],
]:
    paths = _paths(root)
    paths.freeze_bundle.parent.mkdir(parents=True, exist_ok=True)
    paths.freeze_bundle.write_text("sealed freeze fixture\n", encoding="utf-8")
    bundle_file_digest = validation.file_sha256(paths.freeze_bundle)
    payload_digest = "b" * 64

    development_manifest_path = root / "outputs" / "development.manifest.json"
    development_manifest = {
        "schema_version": 2,
        "development_years": list(validation.DEVELOPMENT_YEARS),
    }
    _write_json(development_manifest_path, development_manifest)
    development_caches: dict[str, dict[str, str]] = {}
    for year in validation.DEVELOPMENT_YEARS:
        cache_path = root / "data" / "processed" / f"development-{year}.npz"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(f"sealed development cache {year}\n".encode())
        development_caches[str(year)] = {
            "path": cache_path.relative_to(root).as_posix(),
            "sha256": validation.file_sha256(cache_path),
        }
    payload: dict[str, Any] = {
        "analysis_id": validation.EXPECTED_FREEZE_ANALYSIS_ID,
        "status": validation.EXPECTED_FREEZE_STATUS,
        "selected_l2": 0.001,
        "validation_2024_protocol": {"fixture": "sealed"},
        "code_and_specs": {},
        "development_inputs": {
            "manifest": {
                "path": development_manifest_path.relative_to(root).as_posix(),
                "sha256": validation.file_sha256(development_manifest_path),
            },
            "caches": development_caches,
        },
    }
    bundle = {"payload_sha256": payload_digest, "payload": payload}

    paths.preflight_report.write_text("sealed preflight fixture\n", encoding="utf-8")
    preflight = {
        "status": "PASS",
        "pytest": {
            "passed": 1,
            "failed": 0,
            "errors": 0,
            "deselected": 0,
        },
        "tests": {"inventory_sha256": "c" * 64},
        "transcript": {"sha256": "d" * 64},
        "freeze_bundle": {
            "bundle_file_sha256": bundle_file_digest,
            "payload_sha256": payload_digest,
        },
    }

    touch = validation._touch_document(
        paths,
        bundle_file_sha256=bundle_file_digest,
        payload_sha256=payload_digest,
    )
    _write_json(paths.touch_marker, touch)

    sources = tuple(f"2024/{index:04d}.mjson" for index in range(1_000))
    population_digest = "e" * 64
    validation_manifest = {
        "schema_version": 2,
        "validation_year": validation.VALIDATION_YEAR,
        "validation_2024_touched": True,
        "holdout_2025_touched": False,
        "development_freeze_bundle": {
            "file_sha256": bundle_file_digest,
            "payload_sha256": payload_digest,
        },
        "years": {
            "2024": {
                "candidate_count": 1_000,
                "candidate_population_sha256": population_digest,
                "selected": [{"relative_source": source} for source in sources],
            }
        },
    }
    _write_json(paths.manifest, validation_manifest)
    manifest_file_digest = validation.file_sha256(paths.manifest)
    manifest_canonical_digest = validation.canonical_json_sha256(validation_manifest)

    counts = {
        "candidate_rows": 1,
        "decisions": 1,
        "ron_eligible_rows": 0,
        "scanned_files": 1_000,
        "structural_wait_rows": 0,
    }
    validation_artifact = _one_row_artifact(
        2024,
        sources,
        provenance={
            "analysis_id": validation.ANALYSIS_ID,
            "freeze_bundle_file_sha256": bundle_file_digest,
            "freeze_bundle_payload_sha256": payload_digest,
            "manifest_file_sha256": manifest_file_digest,
            "manifest_canonical_sha256": manifest_canonical_digest,
            "counts_json": json.dumps(counts, sort_keys=True),
        },
    )
    paths.cache.parent.mkdir(parents=True, exist_ok=True)
    save_year_sparse_artifact(paths.cache, validation_artifact)

    base_log_loss = 0.5
    challenger_log_loss = 0.49
    point = challenger_log_loss - base_log_loss
    lower = -0.02
    upper = -0.001
    primary = {
        "metric": "weighted_log_loss_delta",
        "estimand": "conventional_simple - conventional",
        "point": point,
        "percentile_95_lower": lower,
        "percentile_95_upper": upper,
        "valid_replicates": validation.BOOTSTRAP_REPLICATES,
        "required_replicates": validation.BOOTSTRAP_REPLICATES,
        "point_below_zero": True,
        "upper_below_zero": True,
        "success": True,
    }
    result: dict[str, Any] = {
        "schema_version": validation.RESULT_SCHEMA_VERSION,
        "analysis_id": validation.ANALYSIS_ID,
        "status": "VALIDATION_SUCCESS",
        "primary_validation_success": True,
        "scope": {
            "training_years": list(validation.DEVELOPMENT_YEARS),
            "validation_year": validation.VALIDATION_YEAR,
            "selected_validation_sources": validation.VALIDATION_SAMPLE_SIZE,
            "sample_seed": validation.VALIDATION_SAMPLE_SEED,
            "label": validation.LABEL,
            "base_model": validation.BASE_MODEL,
            "challenger_model": validation.CHALLENGER_MODEL,
            "comparison": validation.COMPARISON_NAME,
            "validation_2024_touched": True,
            "holdout_2025_touched": False,
        },
        "freeze_bundle": {
            "path": paths.freeze_bundle.relative_to(root).as_posix(),
            "file_sha256": bundle_file_digest,
            "payload_sha256": payload_digest,
            "analysis_id": payload["analysis_id"],
            "status": payload["status"],
            "validation_2024_protocol": payload["validation_2024_protocol"],
            "code_and_specs": payload["code_and_specs"],
            "code_and_specs_sha256": validation.canonical_json_sha256(
                payload["code_and_specs"]
            ),
        },
        "preflight_tests": {
            "path": paths.preflight_report.relative_to(root).as_posix(),
            "file_sha256": validation.file_sha256(paths.preflight_report),
            "status": preflight["status"],
            "pytest": preflight["pytest"],
            "test_inventory_sha256": preflight["tests"]["inventory_sha256"],
            "transcript_sha256": preflight["transcript"]["sha256"],
            "freeze_bundle": preflight["freeze_bundle"],
        },
        "development_inputs": {
            "manifest": {
                "path": payload["development_inputs"]["manifest"]["path"],
                "file_sha256": payload["development_inputs"]["manifest"]["sha256"],
                "canonical_sha256": validation.canonical_json_sha256(
                    development_manifest
                ),
            },
            "caches": [
                {
                    "year": year,
                    "path": development_caches[str(year)]["path"],
                    "sha256": development_caches[str(year)]["sha256"],
                }
                for year in validation.DEVELOPMENT_YEARS
            ],
        },
        "validation_inputs": {
            "touch_marker": {
                "path": paths.touch_marker.relative_to(root).as_posix(),
                "sha256": validation.file_sha256(paths.touch_marker),
            },
            "manifest": {
                "path": paths.manifest.relative_to(root).as_posix(),
                "file_sha256": manifest_file_digest,
                "canonical_sha256": manifest_canonical_digest,
                "candidate_count": 1_000,
                "selected_source_count": 1_000,
                "candidate_population_sha256": population_digest,
            },
            "cache": {
                "path": paths.cache.relative_to(root).as_posix(),
                "sha256": validation.file_sha256(paths.cache),
                "row_count": validation_artifact.row_count,
                "feature_count": validation_artifact.feature_count,
                "selected_source_count": validation_artifact.selected_source_count,
                "counts": counts,
            },
        },
        "optimizer": {
            "type": "SciPy L-BFGS-B with analytic gradient",
            "objective": "weighted mean log loss + l2/2 * ||beta||^2",
            "selected_l2": payload["selected_l2"],
            "selected_l2_source": "verified freeze bundle only",
            "max_iterations": validation.MAX_ITERATIONS,
            "gradient_tolerance": validation.GRADIENT_TOLERANCE,
            "function_tolerance": validation.FUNCTION_TOLERANCE,
            "max_line_search_steps": validation.MAX_LINE_SEARCH_STEPS,
            "history_size": validation.HISTORY_SIZE,
            "intercept_penalized": False,
            "models": {},
        },
        "feature_vocabulary": {},
        "predictions": {
            validation.BASE_MODEL: {
                "sha256": "1" * 64,
                "metrics": {"log_loss": base_log_loss},
            },
            validation.CHALLENGER_MODEL: {
                "sha256": "2" * 64,
                "metrics": {"log_loss": challenger_log_loss},
            },
            "recalibration_applied": False,
        },
        "delta": {"log_loss": point},
        "turn_bins": {turn_bin: {} for turn_bin in validation.DECISION_TURN_BINS},
        "bootstrap_protocol": {
            "cluster": "game",
            "paired": True,
            "replicates": validation.BOOTSTRAP_REPLICATES,
            "seed": validation.BOOTSTRAP_SEED,
            "refit": False,
            "interval": "pointwise percentile 95%",
            "shared_multiplicity_across_models_and_turn_bins": True,
        },
        "joint_game_cluster_bootstrap": {
            "cluster_count": 1,
            "requested_replicates": validation.BOOTSTRAP_REPLICATES,
            "seed": validation.BOOTSTRAP_SEED,
            "ece_bins": 10,
            "stratified": False,
            "stratum_cluster_counts": [["all", 1]],
            "group_order": list(validation.DECISION_TURN_BINS),
            "comparisons": [
                {
                    "name": validation.COMPARISON_NAME,
                    "base_model": validation.BASE_MODEL,
                    "challenger_model": validation.CHALLENGER_MODEL,
                    "overall": {
                        "log_loss": {
                            "point": point,
                            "lower": lower,
                            "upper": upper,
                            "valid_replicates": validation.BOOTSTRAP_REPLICATES,
                        }
                    },
                    "by_group": [
                        {"group": turn_bin}
                        for turn_bin in validation.DECISION_TURN_BINS
                    ],
                    "contrasts": [
                        {
                            "name": "late_minus_early",
                            "left_group": "13+",
                            "right_group": "1-6",
                        }
                    ],
                }
            ],
        },
        "primary_decision": primary,
        "counts": {"games": 1, "decisions": 1, "candidate_rows": 1},
        "environment": {},
        "elapsed_seconds": 1.0,
    }
    _write_json(paths.output, result)
    return paths, bundle, preflight


def _recovery_fixture(
    root: Path,
) -> tuple[
    validation.ValidationPaths,
    dict[str, Any],
    dict[str, Any],
]:
    paths = _paths(root)
    paths.freeze_bundle.parent.mkdir(parents=True, exist_ok=True)
    paths.freeze_bundle.write_text("sealed freeze fixture\n", encoding="utf-8")
    bundle_file_digest = validation.file_sha256(paths.freeze_bundle)
    payload_digest = "b" * 64

    frozen_development = _development_payload(root)["development_inputs"]
    development_manifest_path = root.joinpath(
        *Path(frozen_development["manifest"]["path"]).parts
    )
    development_manifest = json.loads(
        development_manifest_path.read_text(encoding="utf-8")
    )
    development_caches = frozen_development["caches"]
    payload: dict[str, Any] = {
        "analysis_id": validation.EXPECTED_FREEZE_ANALYSIS_ID,
        "status": validation.EXPECTED_FREEZE_STATUS,
        "selected_l2": 0.001,
        "validation_2024_protocol": {"fixture": "sealed"},
        "code_and_specs": {},
        "development_inputs": frozen_development,
    }
    bundle = {"payload_sha256": payload_digest, "payload": payload}

    paths.preflight_report.write_text("sealed preflight fixture\n", encoding="utf-8")
    preflight = {
        "status": "PASS",
        "pytest": {"passed": 1, "failed": 0, "errors": 0, "deselected": 0},
        "tests": {"inventory_sha256": "c" * 64},
        "transcript": {"sha256": "d" * 64},
        "freeze_bundle": {
            "bundle_file_sha256": bundle_file_digest,
            "payload_sha256": payload_digest,
        },
    }

    _write_json(
        paths.touch_marker,
        validation._touch_document(
            paths,
            bundle_file_sha256=bundle_file_digest,
            payload_sha256=payload_digest,
        ),
    )
    sources = tuple(f"2024/{index:04d}.mjson" for index in range(1_000))
    population_digest = "e" * 64
    validation_manifest = {
        "schema_version": 2,
        "validation_year": validation.VALIDATION_YEAR,
        "validation_2024_touched": True,
        "holdout_2025_touched": False,
        "development_freeze_bundle": {
            "file_sha256": bundle_file_digest,
            "payload_sha256": payload_digest,
        },
        "years": {
            "2024": {
                "candidate_count": 1_000,
                "candidate_population_sha256": population_digest,
                "selected": [{"relative_source": source} for source in sources],
            }
        },
    }
    _write_json(paths.manifest, validation_manifest)
    manifest_file_digest = validation.file_sha256(paths.manifest)
    manifest_canonical_digest = validation.canonical_json_sha256(validation_manifest)
    extraction_counts = {
        "candidate_rows": 4,
        "decisions": 4,
        "ron_eligible_rows": 2,
        "scanned_files": 1_000,
        "structural_wait_rows": 0,
    }
    artifact = _four_bin_artifact(
        sources,
        provenance={
            "analysis_id": validation.ANALYSIS_ID,
            "freeze_bundle_file_sha256": bundle_file_digest,
            "freeze_bundle_payload_sha256": payload_digest,
            "manifest_file_sha256": manifest_file_digest,
            "manifest_canonical_sha256": manifest_canonical_digest,
            "counts_json": json.dumps(extraction_counts, sort_keys=True),
        },
    )
    paths.cache.parent.mkdir(parents=True, exist_ok=True)
    save_year_sparse_artifact(paths.cache, artifact)

    overall_base = _metrics_document(
        rows=4,
        decisions=4,
        positive_rate=0.5,
        auc=0.60,
        log_loss=0.50,
        brier=0.25,
        ece=0.10,
    )
    overall_challenger = _metrics_document(
        rows=4,
        decisions=4,
        positive_rate=0.5,
        auc=0.61,
        log_loss=0.49,
        brier=0.24,
        ece=0.09,
    )
    overall_delta = _delta_document(overall_base, overall_challenger)
    turn_documents: dict[str, Any] = {}
    turn_deltas: dict[str, dict[str, float | None]] = {}
    for index, turn_bin in enumerate(validation.DECISION_TURN_BINS):
        positive_rate = float(index % 2)
        base_metrics = _metrics_document(
            rows=1,
            decisions=1,
            positive_rate=positive_rate,
            auc=None,
            log_loss=0.40 + 0.01 * index,
            brier=0.20 + 0.01 * index,
            ece=0.10 + 0.01 * index,
        )
        challenger_metrics = _metrics_document(
            rows=1,
            decisions=1,
            positive_rate=positive_rate,
            auc=None,
            log_loss=0.39 + 0.01 * index,
            brier=0.19 + 0.01 * index,
            ece=0.09 + 0.01 * index,
        )
        delta = _delta_document(base_metrics, challenger_metrics)
        turn_deltas[turn_bin] = delta
        turn_documents[turn_bin] = {
            "row_count": 1,
            "decision_count": 1,
            "game_count": 1,
            "models": {
                validation.BASE_MODEL: base_metrics,
                validation.CHALLENGER_MODEL: challenger_metrics,
            },
            "delta": delta,
        }

    overall_intervals = _bootstrap_metric_document(overall_delta)
    contrast_points = {
        metric: (
            None
            if turn_deltas["13+"][metric] is None or turn_deltas["1-6"][metric] is None
            else turn_deltas["13+"][metric] - turn_deltas["1-6"][metric]
        )
        for metric in validation.METRIC_NAMES
    }
    log_interval = overall_intervals["log_loss"]
    primary = {
        "metric": "weighted_log_loss_delta",
        "estimand": "conventional_simple - conventional",
        "point": log_interval["point"],
        "percentile_95_lower": log_interval["lower"],
        "percentile_95_upper": log_interval["upper"],
        "valid_replicates": validation.BOOTSTRAP_REPLICATES,
        "required_replicates": validation.BOOTSTRAP_REPLICATES,
        "point_below_zero": True,
        "upper_below_zero": True,
        "success": True,
    }
    optimizer_models = {
        model_name: _optimizer_model_document(active_features=1)
        for model_name in (validation.BASE_MODEL, validation.CHALLENGER_MODEL)
    }
    result: dict[str, Any] = {
        "schema_version": validation.RESULT_SCHEMA_VERSION,
        "analysis_id": validation.ANALYSIS_ID,
        "status": "VALIDATION_SUCCESS",
        "primary_validation_success": True,
        "scope": {
            "training_years": list(validation.DEVELOPMENT_YEARS),
            "validation_year": validation.VALIDATION_YEAR,
            "selected_validation_sources": validation.VALIDATION_SAMPLE_SIZE,
            "sample_seed": validation.VALIDATION_SAMPLE_SEED,
            "label": validation.LABEL,
            "base_model": validation.BASE_MODEL,
            "challenger_model": validation.CHALLENGER_MODEL,
            "comparison": validation.COMPARISON_NAME,
            "validation_2024_touched": True,
            "holdout_2025_touched": False,
        },
        "freeze_bundle": {
            "path": paths.freeze_bundle.relative_to(root).as_posix(),
            "file_sha256": bundle_file_digest,
            "payload_sha256": payload_digest,
            "analysis_id": payload["analysis_id"],
            "status": payload["status"],
            "validation_2024_protocol": payload["validation_2024_protocol"],
            "code_and_specs": payload["code_and_specs"],
            "code_and_specs_sha256": validation.canonical_json_sha256(
                payload["code_and_specs"]
            ),
        },
        "preflight_tests": {
            "path": paths.preflight_report.relative_to(root).as_posix(),
            "file_sha256": validation.file_sha256(paths.preflight_report),
            "status": preflight["status"],
            "pytest": preflight["pytest"],
            "test_inventory_sha256": preflight["tests"]["inventory_sha256"],
            "transcript_sha256": preflight["transcript"]["sha256"],
            "freeze_bundle": preflight["freeze_bundle"],
        },
        "development_inputs": {
            "manifest": {
                "path": frozen_development["manifest"]["path"],
                "file_sha256": frozen_development["manifest"]["sha256"],
                "canonical_sha256": validation.canonical_json_sha256(
                    development_manifest
                ),
            },
            "caches": [
                {
                    "year": year,
                    "path": development_caches[str(year)]["path"],
                    "sha256": development_caches[str(year)]["sha256"],
                }
                for year in validation.DEVELOPMENT_YEARS
            ],
        },
        "validation_inputs": {
            "touch_marker": {
                "path": paths.touch_marker.relative_to(root).as_posix(),
                "sha256": validation.file_sha256(paths.touch_marker),
            },
            "manifest": {
                "path": paths.manifest.relative_to(root).as_posix(),
                "file_sha256": manifest_file_digest,
                "canonical_sha256": manifest_canonical_digest,
                "candidate_count": 1_000,
                "selected_source_count": 1_000,
                "candidate_population_sha256": population_digest,
            },
            "cache": {
                "path": paths.cache.relative_to(root).as_posix(),
                "sha256": validation.file_sha256(paths.cache),
                "row_count": artifact.row_count,
                "feature_count": artifact.feature_count,
                "selected_source_count": artifact.selected_source_count,
                "counts": extraction_counts,
            },
        },
        "optimizer": {
            "type": "SciPy L-BFGS-B with analytic gradient",
            "objective": "weighted mean log loss + l2/2 * ||beta||^2",
            "selected_l2": payload["selected_l2"],
            "selected_l2_source": "verified freeze bundle only",
            "max_iterations": validation.MAX_ITERATIONS,
            "gradient_tolerance": validation.GRADIENT_TOLERANCE,
            "function_tolerance": validation.FUNCTION_TOLERANCE,
            "max_line_search_steps": validation.MAX_LINE_SEARCH_STEPS,
            "history_size": validation.HISTORY_SIZE,
            "intercept_penalized": False,
            "models": optimizer_models,
        },
        "feature_vocabulary": {
            "source": "frozen 2020-2023 development caches only",
            "feature_count": 1,
            "sha256": validation.canonical_json_sha256(["base:known"]),
            "validation_only_feature_count": 1,
            "validation_only_features_sha256": validation.canonical_json_sha256(
                ["validation:only"]
            ),
            "validation_only_features_coefficient": 0.0,
        },
        "predictions": {
            validation.BASE_MODEL: {"sha256": "1" * 64, "metrics": overall_base},
            validation.CHALLENGER_MODEL: {
                "sha256": "2" * 64,
                "metrics": overall_challenger,
            },
            "recalibration_applied": False,
        },
        "delta": overall_delta,
        "turn_bins": turn_documents,
        "bootstrap_protocol": {
            "cluster": "game",
            "paired": True,
            "replicates": validation.BOOTSTRAP_REPLICATES,
            "seed": validation.BOOTSTRAP_SEED,
            "refit": False,
            "interval": "pointwise percentile 95%",
            "shared_multiplicity_across_models_and_turn_bins": True,
        },
        "joint_game_cluster_bootstrap": {
            "cluster_count": 4,
            "requested_replicates": validation.BOOTSTRAP_REPLICATES,
            "seed": validation.BOOTSTRAP_SEED,
            "ece_bins": 10,
            "stratified": False,
            "stratum_cluster_counts": [["all", 4]],
            "group_order": list(validation.DECISION_TURN_BINS),
            "comparisons": [
                {
                    "name": validation.COMPARISON_NAME,
                    "base_model": validation.BASE_MODEL,
                    "challenger_model": validation.CHALLENGER_MODEL,
                    "overall": overall_intervals,
                    "by_group": [
                        {
                            "group": turn_bin,
                            "row_count": 1,
                            "decision_count": 1,
                            "cluster_count": 1,
                            "metrics": _bootstrap_metric_document(
                                turn_deltas[turn_bin]
                            ),
                        }
                        for turn_bin in validation.DECISION_TURN_BINS
                    ],
                    "contrasts": [
                        {
                            "name": "late_minus_early",
                            "left_group": "13+",
                            "right_group": "1-6",
                            "metrics": _bootstrap_metric_document(contrast_points),
                        }
                    ],
                }
            ],
        },
        "primary_decision": primary,
        "counts": {"games": 4, "decisions": 4, "candidate_rows": 4},
        "environment": {
            "argv": ["analysis/analyze_combo_validation_2024.py"],
            "python": "3.12.0",
            "numpy": "fixture",
            "scipy": "fixture",
        },
        "elapsed_seconds": 1.0,
    }
    _write_json(paths.output, result)
    return paths, bundle, preflight


def _four_bin_artifact(
    sources: tuple[str, ...],
    *,
    provenance: dict[str, str],
) -> YearSparseArtifact:
    return YearSparseArtifact(
        year=2024,
        feature_names=("base:known", "validation:only"),
        matrix=csr_matrix(np.ones((4, 2), dtype=np.float64)),
        y_structural=np.zeros(4, dtype=np.bool_),
        y_ron=np.asarray([False, True, False, True], dtype=np.bool_),
        weights=np.ones(4, dtype=np.float64),
        group_indices=np.arange(4, dtype=np.int32),
        group_ids=tuple(f"2024:decision:{index}" for index in range(4)),
        cluster_indices=np.arange(4, dtype=np.int32),
        cluster_ids=sources[:4],
        turn_bins=np.arange(4, dtype=np.uint8),
        source_ranks=np.arange(4, dtype=np.int32),
        source_ids_by_rank=sources,
        provenance=tuple(sorted(provenance.items())),
    )


def _optimizer_model_document(*, active_features: int) -> dict[str, Any]:
    return {
        "intercept": 0.0,
        "active_feature_count": active_features,
        "coefficient_sha256": "f" * 64,
        "diagnostics": {
            "converged": True,
            "status": 0,
            "message": "CONVERGENCE",
            "iterations": 12,
            "function_evaluations": 15,
            "gradient_evaluations": 15,
            "initial_objective": 0.5,
            "final_objective": 0.2,
            "initial_gradient_inf_norm": 0.1,
            "final_gradient_inf_norm": 1e-8,
            "row_count": 4,
            "total_weight": 4.0,
            "active_feature_count": active_features,
            "l2": 0.001,
        },
    }


def _metrics_document(
    *,
    rows: int,
    decisions: int,
    positive_rate: float,
    auc: float | None,
    log_loss: float,
    brier: float,
    ece: float,
) -> dict[str, Any]:
    single_class = positive_rate in {0.0, 1.0}
    return {
        "row_count": rows,
        "decision_count": decisions,
        "positive_rate": positive_rate,
        "auc": auc,
        "log_loss": log_loss,
        "brier": brier,
        "ece": ece,
        "calibration": {
            "joint_status": "single_class" if single_class else "ok",
            "recalibration_intercept": None if single_class else 0.0,
            "recalibration_slope": None if single_class else 1.0,
            "calibration_in_the_large": None if single_class else 0.0,
            "iterations": 0 if single_class else 1,
            "clip_epsilon": 1e-15,
            "clipped_low": 0,
            "clipped_high": 0,
        },
        "macro_concordance": None,
        "macro_concordance_decisions": 0,
    }


def _delta_document(
    base: dict[str, Any],
    challenger: dict[str, Any],
) -> dict[str, float | None]:
    return {
        metric: (
            None
            if base[metric] is None or challenger[metric] is None
            else challenger[metric] - base[metric]
        )
        for metric in validation.METRIC_NAMES
    }


def _bootstrap_metric_document(
    points: dict[str, float | None],
) -> dict[str, dict[str, float | int | None]]:
    return {
        metric: {
            "point": point,
            "lower": None if point is None else point - 0.005,
            "upper": None if point is None else point + 0.005,
            "valid_replicates": (
                0 if point is None else validation.BOOTSTRAP_REPLICATES
            ),
        }
        for metric, point in points.items()
    }


def _stub_recovery_authorities(
    monkeypatch: pytest.MonkeyPatch,
    *,
    bundle: dict[str, Any],
    preflight: dict[str, Any],
) -> None:
    monkeypatch.setattr(
        validation,
        "verify_freeze_bundle",
        lambda *_args, **_kwargs: bundle,
    )
    monkeypatch.setattr(
        validation,
        "verify_preflight_report",
        lambda *_args, **_kwargs: preflight,
    )
    monkeypatch.setattr(
        validation,
        "_validated_bundle_protocol",
        lambda _bundle: (_bundle["payload"], _bundle["payload"]["selected_l2"]),
    )


def _write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
