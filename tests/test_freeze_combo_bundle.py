from __future__ import annotations

import copy
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy
from scipy.sparse import csr_matrix

import analysis.audit_combo_cache_reproducibility as audit
import analysis.freeze_combo_bundle as freeze
from analysis.freeze_combo_bundle import (
    ANALYSIS_ID,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    DEVELOPMENT_YEARS,
    L2_GRID,
    LAMBDA_SELECTION_RECORD,
    METRIC_NAMES,
    RESULT_CODE_PATHS,
    create_freeze_bundle,
    file_sha256,
    verify_freeze_bundle,
)
from mahjong_analysis.combo_sparse_pipeline import (
    YearSparseArtifact,
    save_year_sparse_artifact,
)


@pytest.fixture(autouse=True)
def _synthetic_source_inventory_only(monkeypatch: pytest.MonkeyPatch) -> None:
    # Audit verification is real; only raw-directory enumeration is substituted
    # by this module's artificial development population, never the real MJAI.
    monkeypatch.setattr(
        audit,
        "source_inventory_snapshot",
        lambda _root: {
            str(year): {
                "candidate_count": 1001,
                "candidate_population_sha256": f"{year:064x}",
                "path_size_mtime_inventory_sha256": "a" * 64,
            }
            for year in DEVELOPMENT_YEARS
        },
    )


def test_create_and_verify_bundle_then_detect_target_tampering(tmp_path: Path) -> None:
    fixture = _freeze_fixture(tmp_path)
    bundle_path = tmp_path / "outputs" / "development.freeze.json"

    bundle = create_freeze_bundle(
        result_path=fixture["result_path"],
        manifest_path=fixture["manifest_path"],
        cache_paths=fixture["cache_paths"],
        project_root=tmp_path,
        output_path=bundle_path,
    )

    assert bundle["payload"]["status"] == "GO_FOR_2024_VALIDATION"
    assert bundle["payload"]["selected_l2"] == 0.001
    assert len(bundle["payload_sha256"]) == 64
    assert verify_freeze_bundle(bundle_path, project_root=tmp_path) == bundle

    source = tmp_path / "src" / "mahjong_analysis" / "rules.py"
    source.write_text("TAMPERED = True\n", encoding="utf-8")
    with pytest.raises(ValueError, match="code_and_specs"):
        verify_freeze_bundle(bundle_path, project_root=tmp_path)


def test_create_rejects_fixed_configuration_mismatch_atomically(
    tmp_path: Path,
) -> None:
    fixture = _freeze_fixture(tmp_path)
    result = copy.deepcopy(fixture["result"])
    result["optimizer"]["max_iterations"] = 301
    _write_json(fixture["result_path"], result)
    output = tmp_path / "outputs" / "development.freeze.json"
    output.write_text("existing bundle remains\n", encoding="utf-8")

    with pytest.raises(ValueError, match="max_iterations"):
        create_freeze_bundle(
            result_path=fixture["result_path"],
            manifest_path=fixture["manifest_path"],
            cache_paths=fixture["cache_paths"],
            project_root=tmp_path,
            output_path=output,
        )

    assert output.read_text(encoding="utf-8") == "existing bundle remains\n"


def test_create_rejects_schema_v1_manifest(tmp_path: Path) -> None:
    fixture = _freeze_fixture(tmp_path)
    manifest = copy.deepcopy(fixture["manifest"])
    manifest["schema_version"] = 1
    _write_json(fixture["manifest_path"], manifest)

    with pytest.raises(ValueError, match="schema_version"):
        create_freeze_bundle(
            result_path=fixture["result_path"],
            manifest_path=fixture["manifest_path"],
            cache_paths=fixture["cache_paths"],
            project_root=tmp_path,
            output_path=tmp_path / "outputs" / "development.freeze.json",
        )


def test_create_never_overwrites_an_existing_bundle(tmp_path: Path) -> None:
    fixture = _freeze_fixture(tmp_path)
    output = tmp_path / "outputs" / "development.freeze.json"
    output.write_text("existing bundle remains\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        create_freeze_bundle(
            result_path=fixture["result_path"],
            manifest_path=fixture["manifest_path"],
            cache_paths=fixture["cache_paths"],
            project_root=tmp_path,
            output_path=output,
        )

    assert output.read_text(encoding="utf-8") == "existing bundle remains\n"
    assert not tuple(output.parent.glob(f".{output.name}.tmp-*"))


def test_verify_rejects_bundle_payload_tampering(tmp_path: Path) -> None:
    fixture = _freeze_fixture(tmp_path)
    bundle_path = tmp_path / "outputs" / "development.freeze.json"
    create_freeze_bundle(
        result_path=fixture["result_path"],
        manifest_path=fixture["manifest_path"],
        cache_paths=fixture["cache_paths"],
        project_root=tmp_path,
        output_path=bundle_path,
    )
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["payload"]["validation_2024_protocol"]["bootstrap"]["replicates"] = 1
    _write_json(bundle_path, bundle)

    with pytest.raises(ValueError, match="payload_sha256"):
        verify_freeze_bundle(bundle_path, project_root=tmp_path)


def test_cache_audit_is_mandatory_and_public_verifier_runs_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _freeze_fixture(tmp_path)
    real_verifier = freeze.verify_audit_report
    calls: list[Path] = []

    def recorded_verifier(path: Path, *, project_root: Path) -> dict[str, Any]:
        calls.append(path)
        return real_verifier(path, project_root=project_root)

    monkeypatch.setattr(freeze, "verify_audit_report", recorded_verifier)
    bundle = _seal_fixture(tmp_path, fixture)
    record = bundle["payload"]["development_inputs"]["cache_audit"]
    assert record == {
        "path": fixture["audit_path"].relative_to(tmp_path).as_posix(),
        "sha256": file_sha256(fixture["audit_path"]),
    }
    verify_freeze_bundle(
        tmp_path / "outputs/development.freeze.json", project_root=tmp_path
    )
    assert calls == [fixture["audit_path"], fixture["audit_path"]]


def test_missing_cache_audit_fails_without_creating_seal(tmp_path: Path) -> None:
    fixture = _freeze_fixture(tmp_path)
    with pytest.raises(ValueError, match=r"does not exist|missing|input"):
        create_freeze_bundle(
            result_path=fixture["result_path"],
            manifest_path=fixture["manifest_path"],
            cache_paths=fixture["cache_paths"],
            project_root=tmp_path,
            output_path=tmp_path / "outputs/development.freeze.json",
            cache_audit_path=tmp_path / "outputs/missing-audit.json",
        )
    assert not (tmp_path / "outputs/development.freeze.json").exists()


def test_cache_audit_tampering_is_detected_after_sealing(tmp_path: Path) -> None:
    fixture = _freeze_fixture(tmp_path)
    _seal_fixture(tmp_path, fixture)
    with fixture["audit_path"].open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_freeze_bundle(
            tmp_path / "outputs/development.freeze.json", project_root=tmp_path
        )


@pytest.mark.parametrize("which", ["manifest", "cache"])
def test_valid_audit_for_different_input_path_is_not_reused(
    tmp_path: Path, which: str
) -> None:
    fixture = _freeze_fixture(tmp_path)
    wrapper = fixture["audit_document"]
    report = wrapper["report"]
    original = (
        fixture["manifest_path"]
        if which == "manifest"
        else fixture["cache_paths"][2020]
    )
    copy_path = original.with_name(f"other-{original.name}")
    copy_path.write_bytes(original.read_bytes())
    record = (
        report["manifest"] if which == "manifest" else report["years"]["2020"]["cache"]
    )
    record["path"] = copy_path.relative_to(tmp_path).as_posix()
    _write_audit_fixture(fixture)
    # This report is internally valid, but it belongs to a different input path.
    audit.verify_audit_report(fixture["audit_path"], project_root=tmp_path)
    with pytest.raises(ValueError, match=r"manifest path|cache path"):
        _seal_fixture(tmp_path, fixture)


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("status",), "FAIL", "status"),
        (("scope", "validation_2024_touched"), True, "validation_2024_touched"),
        (
            ("scope", "cache_producer_analysis_id"),
            ANALYSIS_ID,
            "cache_producer_analysis_id",
        ),
        (("years", "2020", "comparison", "status"), "FAIL", "status"),
        (("years", "2020", "comparison", "comparison"), "approximate", "comparison"),
        (
            ("years", "2020", "comparison", "rebuilt_semantic_sha256"),
            "f" * 64,
            "semantic",
        ),
        (("years", "2020", "cache", "sha256"), "f" * 64, "sha256|SHA-256"),
        (
            ("manifest", "canonical_sha256"),
            "f" * 64,
            "canonical_sha256|canonical SHA-256",
        ),
        (
            ("input_stability", "execution_context_sha256_final"),
            "f" * 64,
            "execution_context_sha256_final",
        ),
    ],
)
def test_cache_audit_rejects_inexact_or_mismatched_evidence(
    tmp_path: Path, path: tuple[str, ...], replacement: Any, message: str
) -> None:
    fixture = _freeze_fixture(tmp_path)
    target = fixture["audit_document"]["report"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    _write_audit_fixture(fixture)
    with pytest.raises(ValueError, match=message):
        _seal_fixture(tmp_path, fixture)


def test_bundle_output_cannot_replace_its_cache_audit(tmp_path: Path) -> None:
    fixture = _freeze_fixture(tmp_path)
    original = fixture["audit_path"].read_bytes()
    with pytest.raises(ValueError, match="overwrite a hashed input"):
        create_freeze_bundle(
            result_path=fixture["result_path"],
            manifest_path=fixture["manifest_path"],
            cache_paths=fixture["cache_paths"],
            project_root=tmp_path,
            output_path=fixture["audit_path"],
        )
    assert fixture["audit_path"].read_bytes() == original


def _write_audit_fixture(fixture: dict[str, Any]) -> None:
    wrapper = fixture["audit_document"]
    wrapper["report_sha256"] = freeze.canonical_json_sha256(wrapper["report"])
    _write_json(fixture["audit_path"], wrapper)


@pytest.mark.parametrize("version", ["python", "numpy", "scipy"])
def test_seal_rejects_different_result_runtime(tmp_path: Path, version: str) -> None:
    fixture = _freeze_fixture(tmp_path)
    fixture["result"]["environment"][version] = "0.0.0"
    _write_json(fixture["result_path"], fixture["result"])
    with pytest.raises(ValueError, match=f"environment.{version}"):
        _seal_fixture(tmp_path, fixture)


def test_seal_rejects_code_changed_after_development_run(tmp_path: Path) -> None:
    fixture = _freeze_fixture(tmp_path)
    source = tmp_path / "src/mahjong_analysis/combo_prediction.py"
    source.write_text("CHANGED_AFTER_MODELING = True\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"environment\.file_sha256"):
        _seal_fixture(tmp_path, fixture)


def test_verify_rechecks_runtime_after_sealing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _freeze_fixture(tmp_path)
    _seal_fixture(tmp_path, fixture)
    monkeypatch.setattr(freeze.platform, "python_version", lambda: "0.0.0")
    with pytest.raises(ValueError, match=r"environment\.python"):
        verify_freeze_bundle(
            tmp_path / "outputs/development.freeze.json", project_root=tmp_path
        )


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("folds", "2021", "models", "conventional", "metrics", "auc"), 1.1, "auc"),
        (
            ("folds", "2021", "models", "conventional", "metrics", "log_loss"),
            -0.1,
            "log_loss",
        ),
        (
            (
                "folds",
                "2021",
                "models",
                "conventional",
                "metrics",
                "macro_concordance_decisions",
            ),
            99,
            "macro_concordance_decisions",
        ),
        (
            (
                "folds",
                "2021",
                "models",
                "conventional",
                "metrics",
                "calibration",
                "joint_status",
            ),
            "unknown",
            "joint_status",
        ),
        (
            (
                "folds",
                "2021",
                "models",
                "conventional",
                "metrics",
                "calibration",
                "recalibration_slope",
            ),
            None,
            "recalibration_slope",
        ),
        (
            (
                "folds",
                "2021",
                "models",
                "conventional",
                "metrics",
                "calibration",
                "iterations",
            ),
            101,
            "iterations",
        ),
        (
            (
                "folds",
                "2021",
                "models",
                "conventional",
                "metrics",
                "calibration",
                "clipped_low",
            ),
            25,
            "clipped counts",
        ),
        (("folds", "2021", "delta", "auc"), 0.8, "delta.auc"),
        (("folds", "2021", "row_count"), 25, "row_count"),
        (("pooled", "delta", "brier"), 0.2, "delta.brier"),
        (
            ("pooled", "models", "conventional_simple", "metrics", "positive_rate"),
            0.26,
            "paired.positive_rate",
        ),
        (
            ("pooled", "models", "conventional", "metrics", "ece"),
            float("inf"),
            "non-finite",
        ),
    ],
)
def test_seal_rejects_invalid_metrics_or_deltas(
    tmp_path: Path, path: tuple[str, ...], replacement: Any, message: str
) -> None:
    fixture = _freeze_fixture(tmp_path)
    target = fixture["result"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    _write_json(fixture["result_path"], fixture["result"])
    with pytest.raises(ValueError, match=message):
        _seal_fixture(tmp_path, fixture)
    assert not (tmp_path / "outputs/development.freeze.json").exists()


@pytest.mark.parametrize(
    "scope", ["fold", "pooled", "calibration", "bootstrap", "environment"]
)
def test_seal_rejects_missing_evidence(tmp_path: Path, scope: str) -> None:
    fixture = _freeze_fixture(tmp_path)
    result = fixture["result"]
    if scope == "fold":
        del result["folds"]["2021"]["models"]["conventional"]["metrics"]
    elif scope == "pooled":
        del result["pooled"]["models"]
    elif scope == "calibration":
        del result["pooled"]["models"]["conventional"]["metrics"]["calibration"][
            "clipped_high"
        ]
    elif scope == "bootstrap":
        del result["pooled"]["joint_game_cluster_bootstrap"]["comparisons"][0][
            "overall"
        ]
    else:
        result["environment"]["file_sha256"].pop(str(Path(RESULT_CODE_PATHS[0])))
    _write_json(fixture["result_path"], result)
    with pytest.raises(ValueError):
        _seal_fixture(tmp_path, fixture)


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("overall", "auc", "valid_replicates"), 1999, "valid_replicates"),
        (("overall", "log_loss", "point"), 0.1, "point"),
        (("overall", "brier", "lower"), 0.9, "lower bound"),
        (("by_group", 0, "group"), "13+", "group/order"),
        (("by_group", 0, "row_count"), 19, "row count sum"),
        (("by_group", 0, "cluster_count"), 0, "positive"),
        (("by_group", 0, "metrics", "ece", "upper"), None, "upper"),
        (("contrasts", 0, "metrics", "auc", "point"), 0.8, "point"),
        (
            ("contrasts", 0, "metrics", "macro_concordance", "valid_replicates"),
            1,
            "valid_replicates",
        ),
    ],
)
def test_seal_rejects_incomplete_or_inconsistent_bootstrap(
    tmp_path: Path, path: tuple[str | int, ...], replacement: Any, message: str
) -> None:
    fixture = _freeze_fixture(tmp_path)
    target = fixture["result"]["pooled"]["joint_game_cluster_bootstrap"]["comparisons"][
        0
    ]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    _write_json(fixture["result_path"], fixture["result"])
    with pytest.raises(ValueError, match=message):
        _seal_fixture(tmp_path, fixture)


@pytest.mark.parametrize(
    "counts",
    [
        [["2021", 0], ["2022", 3], ["2023", 3]],
        [["2021", 3], ["2021", 3], ["2023", 3]],
        [["2021", 3], ["2022", 3]],
        [["2021", 3], ["2022", 4], ["2023", 3]],
    ],
)
def test_seal_rejects_wrong_prediction_year_strata(tmp_path: Path, counts: Any) -> None:
    fixture = _freeze_fixture(tmp_path)
    fixture["result"]["pooled"]["joint_game_cluster_bootstrap"][
        "stratum_cluster_counts"
    ] = counts
    _write_json(fixture["result_path"], fixture["result"])
    with pytest.raises(ValueError, match="stratum"):
        _seal_fixture(tmp_path, fixture)


def test_lambda_null_requires_a_failed_fold_and_preserves_selection(
    tmp_path: Path,
) -> None:
    fixture = _freeze_fixture(tmp_path)
    optimizer = fixture["result"]["optimizer"]
    optimizer["lambda_scores"]["0"] = None
    fit = optimizer["lambda_fold_diagnostics"]["2021"]["0"]
    fit.update(accepted=False, converged=False, iterations=300, status=1)
    _write_json(fixture["result_path"], fixture["result"])
    bundle = _seal_fixture(tmp_path, fixture)
    assert bundle["payload"]["selected_l2"] == 0.001


@pytest.mark.parametrize(
    "damage",
    [
        "null_eligible",
        "finite_ineligible",
        "bad_acceptance",
        "missing_year",
        "missing_lambda",
        "score_not_pooled",
    ],
)
def test_lambda_eligibility_evidence_is_required(tmp_path: Path, damage: str) -> None:
    fixture = _freeze_fixture(tmp_path)
    optimizer = fixture["result"]["optimizer"]
    diagnostics = optimizer["lambda_fold_diagnostics"]
    if damage == "null_eligible":
        optimizer["lambda_scores"]["0"] = None
    elif damage == "finite_ineligible":
        diagnostics["2021"]["0"].update(accepted=False, converged=False)
    elif damage == "bad_acceptance":
        diagnostics["2021"]["0"]["accepted"] = False
    elif damage == "missing_year":
        del diagnostics["2023"]
    elif damage == "score_not_pooled":
        optimizer["lambda_scores"]["0.001"] = 0.49
    else:
        del diagnostics["2021"]["0"]
    _write_json(fixture["result_path"], fixture["result"])
    with pytest.raises(ValueError):
        _seal_fixture(tmp_path, fixture)


def test_nonconverged_calibration_is_explicitly_missing_not_rejected(
    tmp_path: Path,
) -> None:
    fixture = _freeze_fixture(tmp_path)
    calibration = fixture["result"]["folds"]["2021"]["models"]["conventional"][
        "metrics"
    ]["calibration"]
    calibration.update(
        joint_status="non_converged",
        recalibration_intercept=None,
        recalibration_slope=None,
        iterations=100,
    )
    _write_json(fixture["result_path"], fixture["result"])
    _seal_fixture(tmp_path, fixture)


def _seal_fixture(root: Path, fixture: dict[str, Any]) -> dict[str, Any]:
    return create_freeze_bundle(
        result_path=fixture["result_path"],
        manifest_path=fixture["manifest_path"],
        cache_paths=fixture["cache_paths"],
        project_root=root,
        output_path=root / "outputs/development.freeze.json",
    )


def _freeze_fixture(root: Path) -> dict[str, Any]:
    _write_project_sources(root)
    manifest = _valid_manifest()
    manifest_path = root / "outputs" / "manifest.json"
    _write_json(manifest_path, manifest)

    cache_paths: dict[int, Path] = {}
    artifacts: dict[int, YearSparseArtifact] = {}
    for year in DEVELOPMENT_YEARS:
        path = root / "data" / "processed" / f"{year}.npz"
        artifact = _synthetic_artifact(year, freeze.canonical_json_sha256(manifest))
        save_year_sparse_artifact(path, artifact)
        cache_paths[year] = path
        artifacts[year] = artifact

    result = _valid_result(manifest_path, manifest, cache_paths)
    result_path = root / "outputs" / f"{ANALYSIS_ID}.json"
    _write_json(result_path, result)
    audit_path = root / "outputs" / freeze.DEFAULT_CACHE_AUDIT.name
    audit_document = _synthetic_audit_report(
        root, manifest_path, manifest, cache_paths, artifacts
    )
    _write_json(audit_path, audit_document)
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "cache_paths": cache_paths,
        "result": result,
        "result_path": result_path,
        "audit_path": audit_path,
        "audit_document": audit_document,
    }


def _synthetic_counts() -> dict[str, int]:
    return {
        "scanned_files": 1000,
        "candidate_rows": 24,
        "decisions": 12,
        "structural_wait_rows": 6,
        "ron_eligible_rows": 6,
    }


def _synthetic_artifact(year: int, manifest_digest: str) -> YearSparseArtifact:
    sources = tuple(f"{year}/{index:04d}.mjson" for index in range(1000))
    groups = np.repeat(np.arange(12, dtype=np.int32), 2)
    clusters = groups // 4
    return YearSparseArtifact(
        year=year,
        feature_names=("x",),
        matrix=csr_matrix(np.arange(24, dtype=np.float64).reshape(-1, 1)),
        y_structural=np.asarray([index % 4 == 0 for index in range(24)]),
        y_ron=np.asarray([index % 4 == 0 for index in range(24)]),
        weights=np.full(24, 0.5, dtype=np.float64),
        group_indices=groups,
        group_ids=tuple(f"{year}:decision-{index}" for index in range(12)),
        cluster_indices=clusters,
        cluster_ids=sources[:3],
        turn_bins=(groups % 4).astype(np.uint8),
        source_ranks=clusters.copy(),
        source_ids_by_rank=sources,
        provenance=tuple(
            sorted(
                {
                    "analysis_id": "combo-development-primary-r1000-v1",
                    "manifest_canonical_sha256": manifest_digest,
                    "counts_json": json.dumps(
                        _synthetic_counts(), sort_keys=True, separators=(",", ":")
                    ),
                }.items()
            )
        ),
    )


def _synthetic_audit_report(
    root: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    cache_paths: dict[int, Path],
    artifacts: dict[int, YearSparseArtifact],
) -> dict[str, Any]:
    raw_root = root / "data/raw"
    raw_root.mkdir(parents=True)
    snapshot = audit._input_snapshot(
        manifest_path, cache_paths, raw_root=raw_root, project_root=root
    )
    environment = audit.environment_document()
    project_inventory = audit.current_project_inventory(root)
    execution_context_sha256 = freeze.canonical_json_sha256(
        {
            "environment": environment,
            "project_inventory": project_inventory,
        }
    )
    summaries = {}
    for year, value in manifest["years"].items():
        selected = value["selected"]
        summaries[year] = {
            "candidate_count": value["candidate_count"],
            "selected_source_count": len(selected),
            "candidate_population_sha256": value["candidate_population_sha256"],
            "selected_sources_sha256": freeze.canonical_json_sha256(
                [entry["relative_source"] for entry in selected]
            ),
            "selection_digests_sha256": freeze.canonical_json_sha256(
                [entry["selection_sha256"] for entry in selected]
            ),
            "selected_content_sha256": freeze.canonical_json_sha256(
                [entry["content_sha256"] for entry in selected]
            ),
        }
    report = {
        "audit_id": audit.AUDIT_ID,
        "analysis_id": ANALYSIS_ID,
        "status": "PASS",
        "scope": {
            "development_years": list(DEVELOPMENT_YEARS),
            "files_per_year": 1000,
            "sample_seed": freeze.SAMPLE_SEED,
            "cache_producer_analysis_id": "combo-development-primary-r1000-v1",
            "validation_2024_touched": False,
            "holdout_2025_touched": False,
        },
        "manifest": {
            "path": manifest_path.relative_to(root).as_posix(),
            "sha256": file_sha256(manifest_path),
            "canonical_sha256": freeze.canonical_json_sha256(manifest),
            "schema_version": 2,
            "validation_status": "PASS",
            "year_summaries": summaries,
        },
        "input_stability": {
            "before": snapshot,
            "after": snapshot,
            "final": snapshot,
            "identical": True,
            "manifest_samples_identical": True,
            "execution_context_identical": True,
            "execution_context_sha256_before": execution_context_sha256,
            "execution_context_sha256_after": execution_context_sha256,
            "execution_context_sha256_final": execution_context_sha256,
        },
        "years": {
            str(year): {
                "cache": {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                },
                "semantic_sha256": audit.artifact_semantic_sha256(artifacts[year]),
                "summary": audit.artifact_summary(artifacts[year]),
                "extraction_counts": _synthetic_counts(),
                "comparison": audit.compare_year_artifacts(
                    artifacts[year], artifacts[year]
                ),
            }
            for year, path in cache_paths.items()
        },
        "environment": environment,
        "project_inventory": project_inventory,
    }
    return {
        "schema_version": 1,
        "report_sha256": freeze.canonical_json_sha256(report),
        "report": report,
    }


def _write_project_sources(root: Path) -> None:
    paths = {
        "pyproject.toml": "[project]\nname='fixture'\n",
        "uv.lock": "version = 1\n",
        "docs/specs/combo-prediction-freeze.md": "# freeze\n",
        "docs/specs/combo-2024-validation.md": "# validation\n",
        "docs/specs/another.md": "# another spec\n",
        "analysis/analyze_combo_freeze_development.py": "ANALYSIS = 'fixture'\n",
        "analysis/freeze_combo_bundle.py": "BUNDLE = 'fixture'\n",
        "analysis/audit_combo_cache_reproducibility.py": "AUDIT = 'fixture'\n",
        "analysis/helper.py": "HELPER = True\n",
        "src/mahjong_analysis/rules.py": "RULE = True\n",
        "src/mahjong_analysis/combo_prediction.py": "PREDICTION = True\n",
        "src/mahjong_analysis/combo_bootstrap.py": "BOOTSTRAP = True\n",
        "src/mahjong_analysis/combo_sparse_pipeline.py": "SPARSE = True\n",
    }
    for relative, content in paths.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _valid_manifest() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "selection_algorithm": ("lowest sha256(namespace, seed, relative_source), v1"),
        "sample_seed": 20260923,
        "files_per_year": 1000,
        "development_years": list(DEVELOPMENT_YEARS),
        "validation_2024_touched": False,
        "holdout_2025_touched": False,
        "years": {
            str(year): {
                "candidate_count": 1001,
                "candidate_population_sha256": f"{year:064x}",
                "selected": [
                    {
                        "relative_source": f"{year}/{index:04d}.mjson",
                        "selection_sha256": f"{index:064x}",
                        "content_sha256": f"{index + 1000:064x}",
                    }
                    for index in range(1000)
                ],
            }
            for year in DEVELOPMENT_YEARS
        },
    }


def _valid_result(
    manifest_path: Path,
    manifest: dict[str, Any],
    caches: dict[int, Path],
) -> dict[str, Any]:
    from analysis.freeze_combo_bundle import canonical_json_sha256

    selected_l2 = 0.001
    root = manifest_path.parent.parent
    scores = {format(value, ".12g"): 0.6 for value in L2_GRID}
    scores[format(selected_l2, ".12g")] = 0.5
    return {
        "analysis_id": ANALYSIS_ID,
        "status": "GO_to_freeze",
        "scope": {
            "development_years": list(DEVELOPMENT_YEARS),
            "prediction_years": [2021, 2022, 2023],
            "files_per_year": 1000,
            "fixed_primary_files_per_year": 1000,
            "sample_seed": 20260923,
            "label": "ron_eligible",
            "base_model": "conventional",
            "challenger_model": "conventional_simple",
            "primary_metric": "weighted_log_loss_delta",
            "validation_2024_touched": False,
            "holdout_2025_touched": False,
        },
        "manifest": {
            "path": str(manifest_path),
            "canonical_sha256": canonical_json_sha256(manifest),
            "file_sha256": file_sha256(manifest_path),
        },
        "cache_artifacts": {
            str(year): {
                "path": str(path),
                "sha256": file_sha256(path),
                "row_count": 24,
                "feature_count": 1,
                "selected_source_count": 1000,
                "counts": _synthetic_counts(),
            }
            for year, path in caches.items()
        },
        "optimizer": {
            "type": "SciPy L-BFGS-B with analytic gradient",
            "objective": "weighted mean log loss + l2/2 * ||beta||^2",
            "intercept_penalized": False,
            "lambda_grid": list(L2_GRID),
            "lambda_selection": LAMBDA_SELECTION_RECORD,
            "selected_l2": selected_l2,
            "max_iterations": 300,
            "stability_max_iterations": 600,
            "gradient_tolerance": 1e-7,
            "function_tolerance": 0.0,
            "lambda_scores": scores,
            "lambda_fold_diagnostics": {
                str(year): {
                    format(l2, ".12g"): {
                        "accepted": True,
                        **_valid_diagnostics(),
                        "l2": l2,
                    }
                    for l2 in L2_GRID
                }
                for year in (2021, 2022, 2023)
            },
        },
        "folds": {
            str(test_year): {
                "train_years": list(range(2020, test_year)),
                "test_year": test_year,
                "row_count": 24,
                "decision_count": 12,
                "game_count": 3,
                "models": {
                    "conventional": _valid_model(),
                    "conventional_simple": _valid_model(challenger=True),
                },
                "delta": _valid_delta(),
                "stability_check": _valid_stability(),
            }
            for test_year in (2021, 2022, 2023)
        },
        "pooled": {
            "models": {
                "conventional": {"metrics": _valid_metrics(rows=72, decisions=36)},
                "conventional_simple": {
                    "metrics": _valid_metrics(challenger=True, rows=72, decisions=36)
                },
            },
            "delta": _valid_delta(),
            "joint_game_cluster_bootstrap": {
                "cluster_count": 9,
                "requested_replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "ece_bins": 10,
                "stratified": True,
                "group_order": ["1-6", "7-9", "10-12", "13+"],
                "stratum_cluster_counts": [
                    ["2021", 3],
                    ["2022", 3],
                    ["2023", 3],
                ],
                "comparisons": [
                    {
                        "name": "conventional_simple_minus_conventional",
                        "base_model": "conventional",
                        "challenger_model": "conventional_simple",
                        "overall": _valid_intervals(),
                        "by_group": [
                            {
                                "group": name,
                                "row_count": 18,
                                "decision_count": 9,
                                "cluster_count": 3,
                                "metrics": _valid_intervals(multiplier),
                            }
                            for name, multiplier in zip(
                                freeze.TURN_BINS, (0.7, 0.9, 1.1, 1.3), strict=True
                            )
                        ],
                        "contrasts": [
                            {
                                "name": "late_minus_early",
                                "left_group": "13+",
                                "right_group": "1-6",
                                "metrics": _valid_intervals(0.6),
                            }
                        ],
                    }
                ],
            },
        },
        "freeze_checks": {
            "all_primary_fits_accepted": True,
            "all_doubled_iteration_stability_checks_accepted": True,
            "fixed_configuration": True,
            "bootstrap_replicates_exactly_2000": True,
            "bootstrap_seed_fixed": True,
            "optimizer_limits_fixed": True,
            "full_1000_file_prefix": True,
            "freeze_ready": True,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "argv": [str(root / "analysis/analyze_combo_freeze_development.py")],
            "file_sha256": {
                str(Path(relative)): file_sha256(root / relative)
                for relative in RESULT_CODE_PATHS
            },
        },
    }


def _valid_model(*, challenger: bool = False) -> dict[str, Any]:
    return {
        "intercept": 0.0,
        "diagnostics": _valid_diagnostics(),
        "metrics": _valid_metrics(challenger=challenger),
    }


def _valid_metrics(
    *, challenger: bool = False, rows: int = 24, decisions: int = 12
) -> dict[str, Any]:
    return {
        "row_count": rows,
        "decision_count": decisions,
        "positive_rate": 0.25,
        "auc": 0.72 if challenger else 0.7,
        "log_loss": 0.49 if challenger else 0.5,
        "brier": 0.15 if challenger else 0.16,
        "ece": 0.02 if challenger else 0.03,
        "macro_concordance": 0.66 if challenger else 0.65,
        "macro_concordance_decisions": decisions * 2 // 3,
        "calibration": {
            "joint_status": "ok",
            "recalibration_intercept": 0.0,
            "recalibration_slope": 1.0,
            "calibration_in_the_large": 0.0,
            "iterations": 5,
            "clip_epsilon": 1e-15,
            "clipped_low": 0,
            "clipped_high": 0,
        },
    }


def _valid_delta() -> dict[str, float]:
    base, challenger = _valid_metrics(), _valid_metrics(challenger=True)
    return {name: challenger[name] - base[name] for name in METRIC_NAMES}


def _valid_intervals(multiplier: float = 1.0) -> dict[str, Any]:
    return {
        name: {
            "point": value * multiplier,
            "lower": value * multiplier - 0.001,
            "upper": value * multiplier + 0.001,
            "valid_replicates": BOOTSTRAP_REPLICATES,
        }
        for name, value in _valid_delta().items()
    }


def _valid_diagnostics() -> dict[str, Any]:
    return {
        "converged": True,
        "status": 0,
        "message": "CONVERGENCE",
        "iterations": 12,
        "function_evaluations": 15,
        "gradient_evaluations": 15,
        "initial_objective": 0.6,
        "initial_gradient_inf_norm": 0.01,
        "final_gradient_inf_norm": 1e-8,
        "final_objective": 0.5,
        "row_count": 24,
        "total_weight": 12.0,
        "active_feature_count": 2,
        "l2": 0.001,
    }


def _valid_stability() -> dict[str, Any]:
    return {
        "accepted": True,
        "thresholds": {
            "prediction_max_absolute_difference": 1e-8,
            "log_loss_absolute_difference": 1e-10,
            "gradient_inf_norm": 1e-7,
        },
        "base": _valid_stability_model(),
        "challenger": _valid_stability_model(),
    }


def _valid_stability_model() -> dict[str, Any]:
    return {
        "prediction_max_absolute_difference": 1e-9,
        "log_loss_absolute_difference": 1e-11,
        "primary_fit": _valid_diagnostics(),
        "doubled_iteration_fit": _valid_diagnostics(),
    }


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
