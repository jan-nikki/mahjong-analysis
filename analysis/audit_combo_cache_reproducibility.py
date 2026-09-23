"""Re-extract and semantically audit the 2020--2023 combo caches.

Creation is intentionally expensive: the schema-v2 manifest is validated by
the existing production loader before and after extraction, and each cache is
rebuilt in memory with the current extraction/feature code.  Verification is
cheaper: it rechecks all bindings and cache semantics without re-extracting
MJAI rows.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

import numpy as np
import scipy

from mahjong_analysis.combo_dataset import (
    DEVELOPMENT_YEARS,
    StableYearSample,
    collect_candidate_dataset,
    file_sha256,
)
from mahjong_analysis.combo_sparse_pipeline import (
    YearSparseArtifact,
    build_year_sparse_artifact,
    load_year_sparse_artifact,
)

if __package__:
    from analysis.analyze_combo_development import _samples_from_manifest
else:
    from analyze_combo_development import _samples_from_manifest

ANALYSIS_ID = "combo-development-primary-r1000-v2"
CACHE_PRODUCER_ANALYSIS_ID = "combo-development-primary-r1000-v1"
AUDIT_ID = "combo-development-cache-reproducibility-v1"
REPORT_SCHEMA_VERSION = 1
FILES_PER_YEAR = 1_000
SAMPLE_SEED = 20260923
PASS_STATUS = "PASS"

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
    PROJECT_ROOT
    / "outputs"
    / "combo-development-cache-reproducibility-v1.json"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantically reproduce and audit the development caches."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--cache",
        action="append",
        default=[],
        metavar="YEAR=PATH",
        help="repeat for 2020, 2021, 2022, and 2023",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--verify",
        type=Path,
        metavar="REPORT",
        help="verify bindings without re-extracting MJAI rows",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    if args.verify is not None:
        if args.manifest is not None or args.cache or args.raw_root is not None:
            raise ValueError(
                "--verify cannot be combined with --manifest, --cache, or --raw-root"
            )
        report_path = args.verify.resolve()
        report = verify_audit_report(report_path, project_root=project_root)
        print(
            f"verified report_file_sha256={file_sha256(report_path)} "
            f"report_sha256={report['report_sha256']}",
            file=sys.stderr,
            flush=True,
        )
        return 0

    raw_root = (args.raw_root or DEFAULT_RAW_ROOT).resolve()
    manifest_path = (args.manifest or DEFAULT_MANIFEST).resolve()
    cache_paths = (
        _parse_cache_arguments(args.cache)
        if args.cache
        else {
            year: (
                DEFAULT_CACHE_DIR / f"{year}-conventional-simple.npz"
            ).resolve()
            for year in DEVELOPMENT_YEARS
        }
    )
    output_path = (args.output or DEFAULT_OUTPUT).resolve()
    wrapper = create_audit_report(
        manifest_path=manifest_path,
        cache_paths=cache_paths,
        raw_root=raw_root,
        project_root=project_root,
        output_path=output_path,
    )
    print(
        f"wrote report_file_sha256={file_sha256(output_path)} "
        f"report_sha256={wrapper['report_sha256']}",
        file=sys.stderr,
        flush=True,
    )
    return 0


def create_audit_report(
    *,
    manifest_path: Path,
    cache_paths: Mapping[int, Path],
    raw_root: Path,
    project_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Rebuild all four development artifacts and write an exclusive report."""

    root = project_root.resolve()
    raw = raw_root.resolve()
    manifest_file = manifest_path.resolve()
    normalized_caches = _normalize_cache_paths(cache_paths, root)
    _relative_path(raw, root)
    _relative_path(manifest_file, root)
    output = output_path.resolve()
    _relative_path(output, root)
    if output.exists():
        raise FileExistsError(f"audit report already exists: {output}")

    manifest = _read_json_object(manifest_file)
    if manifest.get("schema_version") != 2:
        raise ValueError("cache reproducibility audit requires manifest schema v2")
    manifest_canonical_sha256 = canonical_json_sha256(manifest)
    execution_context_before = _execution_context_snapshot(root)
    before = _input_snapshot(
        manifest_file,
        normalized_caches,
        raw_root=raw,
        project_root=root,
    )
    samples_before = _validated_samples(manifest, raw)
    sample_summaries = _sample_summaries(samples_before)
    _validate_manifest_summary_binding(manifest, sample_summaries)

    year_reports: dict[str, Any] = {}
    for year in DEVELOPMENT_YEARS:
        print(f"[audit-cache] re-extract {year}", file=sys.stderr, flush=True)
        sample = samples_before[year]
        dataset = collect_candidate_dataset(raw, year, sample.paths)
        if dataset.selected_sources != sample.relative_sources:
            raise ValueError(f"extracted selected-source order mismatch for {year}")
        counts_json = json.dumps(
            dataset.counts,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        expected_provenance = {
            "analysis_id": CACHE_PRODUCER_ANALYSIS_ID,
            "manifest_canonical_sha256": manifest_canonical_sha256,
            "counts_json": counts_json,
        }
        rebuilt = build_year_sparse_artifact(
            dataset.rows,
            year=year,
            source_rank_by_game={
                source: rank for rank, source in enumerate(sample.relative_sources)
            },
            provenance=expected_provenance,
        )
        cached = load_year_sparse_artifact(normalized_caches[year])
        comparison = compare_year_artifacts(cached, rebuilt)
        _validate_cache_provenance(
            cached,
            expected_provenance=expected_provenance,
            counts=dataset.counts,
            year=year,
        )
        year_reports[str(year)] = {
            "cache": _file_record(normalized_caches[year], root),
            "semantic_sha256": artifact_semantic_sha256(cached),
            "summary": artifact_summary(cached),
            "extraction_counts": dataset.counts,
            "comparison": comparison,
        }

    samples_after = _validated_samples(manifest, raw)
    after = _input_snapshot(
        manifest_file,
        normalized_caches,
        raw_root=raw,
        project_root=root,
    )
    if _sample_bindings(samples_before) != _sample_bindings(samples_after):
        raise RuntimeError("manifest/source bindings changed during re-extraction")
    if before != after:
        raise RuntimeError("manifest, cache, or source inventory changed during audit")

    execution_context_after = _execution_context_snapshot(root)
    if execution_context_after != execution_context_before:
        raise RuntimeError(
            "project code, specifications, metadata, or environment changed "
            "during re-extraction"
        )
    execution_context_digest = canonical_json_sha256(execution_context_before)
    report: dict[str, Any] = {
        "audit_id": AUDIT_ID,
        "analysis_id": ANALYSIS_ID,
        "status": PASS_STATUS,
        "scope": {
            "development_years": list(DEVELOPMENT_YEARS),
            "files_per_year": FILES_PER_YEAR,
            "sample_seed": SAMPLE_SEED,
            "cache_producer_analysis_id": CACHE_PRODUCER_ANALYSIS_ID,
            "validation_2024_touched": False,
            "holdout_2025_touched": False,
        },
        "manifest": {
            **_file_record(manifest_file, root),
            "canonical_sha256": manifest_canonical_sha256,
            "schema_version": 2,
            "validation_status": PASS_STATUS,
            "year_summaries": sample_summaries,
        },
        "input_stability": {
            "before": before,
            "after": after,
            "final": after,
            "identical": True,
            "manifest_samples_identical": True,
            "execution_context_identical": True,
            "execution_context_sha256_before": execution_context_digest,
            "execution_context_sha256_after": execution_context_digest,
            "execution_context_sha256_final": execution_context_digest,
        },
        "years": year_reports,
        "environment": execution_context_before["environment"],
        "project_inventory": execution_context_before["project_inventory"],
    }
    wrapper = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_sha256": canonical_json_sha256(report),
        "report": report,
    }
    final_snapshot = _input_snapshot(
        manifest_file,
        normalized_caches,
        raw_root=raw,
        project_root=root,
    )
    if final_snapshot != after:
        raise RuntimeError("audit inputs changed while finalizing the report")
    execution_context_final = _execution_context_snapshot(root)
    if execution_context_final != execution_context_before:
        raise RuntimeError(
            "project code, specifications, metadata, or environment changed "
            "while finalizing the report"
        )
    _atomic_exclusive_write_json(output, wrapper)
    return wrapper


def verify_audit_report(
    report_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Recheck an audit's hashes and cached semantics without MJAI extraction."""

    root = project_root.resolve()
    wrapper = _read_json_object(report_path)
    if set(wrapper) != {"schema_version", "report_sha256", "report"}:
        raise ValueError("audit wrapper has an unexpected shape")
    _require_equal(
        wrapper["schema_version"], REPORT_SCHEMA_VERSION, "schema_version"
    )
    report = _mapping(wrapper["report"], "report")
    _require_equal(
        wrapper["report_sha256"], canonical_json_sha256(report), "report_sha256"
    )
    _require_equal(report.get("audit_id"), AUDIT_ID, "report.audit_id")
    _require_equal(report.get("analysis_id"), ANALYSIS_ID, "report.analysis_id")
    _require_equal(report.get("status"), PASS_STATUS, "report.status")
    _verify_scope(report.get("scope"))
    _require_equal(
        report.get("environment"), environment_document(), "report.environment"
    )
    _require_equal(
        report.get("project_inventory"),
        current_project_inventory(root),
        "report.project_inventory",
    )

    manifest_record = _mapping(report.get("manifest"), "report.manifest")
    manifest_path = _bound_path(
        manifest_record,
        root,
        extra_keys={
            "canonical_sha256",
            "schema_version",
            "validation_status",
            "year_summaries",
        },
    )
    _verify_file_hash(manifest_path, manifest_record, "report.manifest")
    manifest = _read_json_object(manifest_path)
    _require_equal(manifest.get("schema_version"), 2, "manifest.schema_version")
    _require_equal(
        manifest_record.get("validation_status"),
        PASS_STATUS,
        "report.manifest.validation_status",
    )
    _require_equal(
        manifest_record.get("canonical_sha256"),
        canonical_json_sha256(manifest),
        "report.manifest.canonical_sha256",
    )
    _validate_manifest_summary_binding(
        manifest,
        _mapping(
            manifest_record.get("year_summaries"),
            "report.manifest.year_summaries",
        ),
    )

    stability = _mapping(report.get("input_stability"), "report.input_stability")
    _require_equal(stability.get("identical"), True, "input_stability.identical")
    _require_equal(
        stability.get("manifest_samples_identical"),
        True,
        "input_stability.manifest_samples_identical",
    )
    _require_equal(
        stability.get("before"), stability.get("after"), "input_stability snapshots"
    )
    _require_equal(
        stability.get("after"), stability.get("final"), "input_stability final snapshot"
    )
    _require_equal(
        stability.get("execution_context_identical"),
        True,
        "input_stability.execution_context_identical",
    )
    execution_context = {
        "environment": report.get("environment"),
        "project_inventory": report.get("project_inventory"),
    }
    execution_context_digest = canonical_json_sha256(execution_context)
    for phase in ("before", "after", "final"):
        _require_equal(
            stability.get(f"execution_context_sha256_{phase}"),
            execution_context_digest,
            f"input_stability.execution_context_sha256_{phase}",
        )
    after = _mapping(stability.get("after"), "input_stability.after")
    raw_root = _path_from_relative(
        after.get("raw_root"), root, "input_stability.raw_root"
    )

    years = _mapping(report.get("years"), "report.years")
    _require_equal(
        set(years),
        {str(year) for year in DEVELOPMENT_YEARS},
        "report.years",
    )
    cache_paths: dict[int, Path] = {}
    for year in DEVELOPMENT_YEARS:
        year_report = _mapping(years[str(year)], f"report.years.{year}")
        cache_record = _mapping(year_report.get("cache"), f"report.years.{year}.cache")
        cache_path = _bound_path(cache_record, root)
        _verify_file_hash(cache_path, cache_record, f"report.years.{year}.cache")
        artifact = load_year_sparse_artifact(cache_path)
        _require_equal(artifact.year, year, f"cache {year} year")
        _require_equal(
            year_report.get("semantic_sha256"),
            artifact_semantic_sha256(artifact),
            f"report.years.{year}.semantic_sha256",
        )
        _require_equal(
            year_report.get("summary"),
            artifact_summary(artifact),
            f"report.years.{year}.summary",
        )
        cache_paths[year] = cache_path

    current_snapshot = _input_snapshot(
        manifest_path,
        cache_paths,
        raw_root=raw_root,
        project_root=root,
    )
    _require_equal(current_snapshot, after, "current input snapshot")
    return wrapper


def compare_year_artifacts(
    cached: YearSparseArtifact,
    rebuilt: YearSparseArtifact,
) -> dict[str, Any]:
    """Require deterministic semantic equality of two year artifacts."""

    scalar_fields = {
        "year": (cached.year, rebuilt.year),
        "feature_names": (cached.feature_names, rebuilt.feature_names),
        "matrix_shape": (cached.matrix.shape, rebuilt.matrix.shape),
        "group_ids": (cached.group_ids, rebuilt.group_ids),
        "cluster_ids": (cached.cluster_ids, rebuilt.cluster_ids),
        "source_ids_by_rank": (
            cached.source_ids_by_rank,
            rebuilt.source_ids_by_rank,
        ),
        "provenance": (cached.provenance, rebuilt.provenance),
    }
    for field, (left, right) in scalar_fields.items():
        if left != right:
            raise ValueError(f"cache semantic mismatch: {field}")
    array_fields = {
        "matrix.indptr": (cached.matrix.indptr, rebuilt.matrix.indptr),
        "matrix.indices": (cached.matrix.indices, rebuilt.matrix.indices),
        "matrix.data": (cached.matrix.data, rebuilt.matrix.data),
        "y_structural": (cached.y_structural, rebuilt.y_structural),
        "y_ron": (cached.y_ron, rebuilt.y_ron),
        "weights": (cached.weights, rebuilt.weights),
        "group_indices": (cached.group_indices, rebuilt.group_indices),
        "cluster_indices": (cached.cluster_indices, rebuilt.cluster_indices),
        "turn_bins": (cached.turn_bins, rebuilt.turn_bins),
        "source_ranks": (cached.source_ranks, rebuilt.source_ranks),
    }
    for field, (left, right) in array_fields.items():
        if left.dtype != right.dtype or left.shape != right.shape:
            raise ValueError(f"cache semantic mismatch: {field} dtype/shape")
        if not np.array_equal(left, right, equal_nan=False):
            raise ValueError(f"cache semantic mismatch: {field} values")
    cached_digest = artifact_semantic_sha256(cached)
    rebuilt_digest = artifact_semantic_sha256(rebuilt)
    _require_equal(cached_digest, rebuilt_digest, "artifact semantic SHA-256")
    return {
        "status": PASS_STATUS,
        "comparison": "exact",
        "fields_checked": [*scalar_fields, *array_fields],
        "cached_semantic_sha256": cached_digest,
        "rebuilt_semantic_sha256": rebuilt_digest,
    }


def artifact_summary(artifact: YearSparseArtifact) -> dict[str, Any]:
    provenance = dict(artifact.provenance)
    counts: dict[str, Any] = {}
    if "counts_json" in provenance:
        loaded_counts = json.loads(provenance["counts_json"])
        if not isinstance(loaded_counts, dict):
            raise ValueError("artifact counts_json must decode to an object")
        counts = loaded_counts
    return {
        "year": artifact.year,
        "row_count": artifact.row_count,
        "feature_count": artifact.feature_count,
        "nnz": int(artifact.matrix.nnz),
        "selected_source_count": artifact.selected_source_count,
        "group_count": len(artifact.group_ids),
        "cluster_count": len(artifact.cluster_ids),
        "structural_positive_rows": int(np.count_nonzero(artifact.y_structural)),
        "ron_positive_rows": int(np.count_nonzero(artifact.y_ron)),
        "total_weight": float(np.sum(artifact.weights, dtype=np.float64)),
        "counts": counts,
    }


def artifact_semantic_sha256(artifact: YearSparseArtifact) -> str:
    digest = sha256()
    _digest_text(digest, str(artifact.year))
    for values in (
        artifact.feature_names,
        artifact.group_ids,
        artifact.cluster_ids,
        artifact.source_ids_by_rank,
    ):
        _digest_strings(digest, values)
    _digest_strings(
        digest,
        tuple(f"{key}\0{value}" for key, value in artifact.provenance),
    )
    _digest_text(digest, json.dumps(artifact.matrix.shape, separators=(",", ":")))
    for array in (
        artifact.matrix.indptr,
        artifact.matrix.indices,
        artifact.matrix.data,
        artifact.y_structural,
        artifact.y_ron,
        artifact.weights,
        artifact.group_indices,
        artifact.cluster_indices,
        artifact.turn_bins,
        artifact.source_ranks,
    ):
        contiguous = np.ascontiguousarray(array)
        _digest_text(digest, contiguous.dtype.str)
        _digest_text(digest, json.dumps(contiguous.shape, separators=(",", ":")))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def current_project_inventory(project_root: Path) -> dict[str, Any]:
    required = (
        project_root / "pyproject.toml",
        project_root / "uv.lock",
        project_root / "analysis" / "audit_combo_cache_reproducibility.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"required project files are missing: {missing}")
    groups = {
        "project_metadata": [project_root / "pyproject.toml", project_root / "uv.lock"],
        "python": sorted(
            {
                *project_root.joinpath("src", "mahjong_analysis").rglob("*.py"),
                *project_root.joinpath("analysis").rglob("*.py"),
            },
            key=lambda path: path.as_posix(),
        ),
        "specs": sorted(
            project_root.joinpath("docs", "specs").rglob("*.md"),
            key=lambda path: path.as_posix(),
        ),
    }
    if not groups["python"] or not groups["specs"]:
        raise ValueError("project Python/spec inventories must be nonempty")
    files = {
        name: {
            _relative_path(path, project_root): file_sha256(path) for path in paths
        }
        for name, paths in groups.items()
    }
    return {"inventory_sha256": canonical_json_sha256(files), "files": files}


def environment_document() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def _execution_context_snapshot(project_root: Path) -> dict[str, Any]:
    return {
        "environment": environment_document(),
        "project_inventory": current_project_inventory(project_root),
    }


def _validated_samples(
    manifest: dict[str, Any], raw_root: Path
) -> dict[int, StableYearSample]:
    args = argparse.Namespace(
        raw_root=raw_root,
        sample_seed=SAMPLE_SEED,
        files_per_year=FILES_PER_YEAR,
    )
    samples = _samples_from_manifest(manifest, args)
    _require_equal(set(samples), set(DEVELOPMENT_YEARS), "manifest samples")
    return samples


def _input_snapshot(
    manifest_path: Path,
    cache_paths: Mapping[int, Path],
    *,
    raw_root: Path,
    project_root: Path,
) -> dict[str, Any]:
    return {
        "raw_root": _relative_path(raw_root, project_root),
        "manifest_file_sha256": file_sha256(manifest_path),
        "cache_file_sha256": {
            str(year): file_sha256(cache_paths[year]) for year in DEVELOPMENT_YEARS
        },
        "source_inventory": source_inventory_snapshot(raw_root),
    }


def source_inventory_snapshot(raw_root: Path) -> dict[str, Any]:
    """Bind development source paths and stat metadata without opening MJAI."""

    result: dict[str, Any] = {}
    canonical_root = raw_root.resolve()
    for year in DEVELOPMENT_YEARS:
        year_root = canonical_root / str(year)
        if not year_root.is_dir():
            raise FileNotFoundError(f"raw development year is missing: {year_root}")
        paths = sorted(
            (path for path in year_root.rglob("*.mjson") if path.is_file()),
            key=lambda path: path.as_posix(),
        )
        population = sha256()
        stat_inventory = sha256()
        for path in paths:
            relative = path.relative_to(canonical_root).as_posix()
            stat = path.stat()
            population.update(relative.encode("utf-8"))
            population.update(b"\n")
            stat_inventory.update(relative.encode("utf-8"))
            stat_inventory.update(b"\0")
            stat_inventory.update(str(stat.st_size).encode("ascii"))
            stat_inventory.update(b"\0")
            stat_inventory.update(str(stat.st_mtime_ns).encode("ascii"))
            stat_inventory.update(b"\n")
        result[str(year)] = {
            "candidate_count": len(paths),
            "candidate_population_sha256": population.hexdigest(),
            "path_size_mtime_inventory_sha256": stat_inventory.hexdigest(),
        }
    return result


def _sample_bindings(samples: Mapping[int, StableYearSample]) -> dict[str, Any]:
    return {
        str(year): {
            "candidate_count": sample.candidate_count,
            "relative_sources": list(sample.relative_sources),
            "selection_digests": list(sample.selection_digests),
            "candidate_population_sha256": sample.candidate_population_sha256,
            "source_content_sha256": list(sample.source_content_sha256),
        }
        for year, sample in sorted(samples.items())
    }


def _sample_summaries(samples: Mapping[int, StableYearSample]) -> dict[str, Any]:
    return {
        str(year): {
            "candidate_count": sample.candidate_count,
            "selected_source_count": len(sample.relative_sources),
            "candidate_population_sha256": sample.candidate_population_sha256,
            "selected_sources_sha256": canonical_json_sha256(
                list(sample.relative_sources)
            ),
            "selection_digests_sha256": canonical_json_sha256(
                list(sample.selection_digests)
            ),
            "selected_content_sha256": canonical_json_sha256(
                list(sample.source_content_sha256)
            ),
        }
        for year, sample in sorted(samples.items())
    }


def _validate_manifest_summary_binding(
    manifest: Mapping[str, Any], summaries: Mapping[str, Any]
) -> None:
    years = _mapping(manifest.get("years"), "manifest.years")
    expected_years = {str(year) for year in DEVELOPMENT_YEARS}
    _require_equal(set(years), expected_years, "manifest.years")
    _require_equal(set(summaries), expected_years, "manifest year summaries")
    for year in DEVELOPMENT_YEARS:
        entry = _mapping(years[str(year)], f"manifest.years.{year}")
        summary = _mapping(summaries[str(year)], f"manifest summary {year}")
        selected = entry.get("selected")
        if not isinstance(selected, list) or len(selected) != FILES_PER_YEAR:
            raise ValueError(f"manifest year {year} must select exactly 1000 files")
        selected_items = [
            _mapping(item, f"manifest.years.{year}.selected[{index}]")
            for index, item in enumerate(selected)
        ]
        expected = {
            "candidate_count": entry.get("candidate_count"),
            "selected_source_count": len(selected),
            "candidate_population_sha256": entry.get(
                "candidate_population_sha256"
            ),
            "selected_sources_sha256": canonical_json_sha256(
                [item.get("relative_source") for item in selected_items]
            ),
            "selection_digests_sha256": canonical_json_sha256(
                [item.get("selection_sha256") for item in selected_items]
            ),
            "selected_content_sha256": canonical_json_sha256(
                [item.get("content_sha256") for item in selected_items]
            ),
        }
        _require_equal(summary, expected, f"manifest year {year} summary")


def _validate_cache_provenance(
    artifact: YearSparseArtifact,
    *,
    expected_provenance: Mapping[str, str],
    counts: Mapping[str, int],
    year: int,
) -> None:
    _require_equal(
        dict(artifact.provenance), dict(expected_provenance), f"cache {year} provenance"
    )
    decoded = json.loads(dict(artifact.provenance)["counts_json"])
    _require_equal(decoded, dict(counts), f"cache {year} extraction counts")


def _normalize_cache_paths(
    cache_paths: Mapping[int, Path], project_root: Path
) -> dict[int, Path]:
    _require_equal(set(cache_paths), set(DEVELOPMENT_YEARS), "cache years")
    normalized: dict[int, Path] = {}
    for year in DEVELOPMENT_YEARS:
        path = cache_paths[year].resolve()
        _relative_path(path, project_root)
        if path.suffix.lower() != ".npz" or not path.is_file():
            raise ValueError(f"cache {year} must be an existing NPZ file")
        normalized[year] = path
    return normalized


def _verify_scope(raw: Any) -> None:
    scope = _mapping(raw, "report.scope")
    expected = {
        "development_years": list(DEVELOPMENT_YEARS),
        "files_per_year": FILES_PER_YEAR,
        "sample_seed": SAMPLE_SEED,
        "cache_producer_analysis_id": CACHE_PRODUCER_ANALYSIS_ID,
        "validation_2024_touched": False,
        "holdout_2025_touched": False,
    }
    _require_equal(scope, expected, "report.scope")


def _file_record(path: Path, project_root: Path) -> dict[str, str]:
    return {
        "path": _relative_path(path, project_root),
        "sha256": file_sha256(path),
    }


def _bound_path(
    record: Mapping[str, Any],
    project_root: Path,
    *,
    extra_keys: set[str] | None = None,
) -> Path:
    expected_keys = {"path", "sha256"} | (extra_keys or set())
    _require_equal(set(record), expected_keys, "file record keys")
    return _path_from_relative(record.get("path"), project_root, "file record path")


def _verify_file_hash(path: Path, record: Mapping[str, Any], field: str) -> None:
    expected = record.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{field}.sha256 is invalid")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"{field}.sha256 mismatch: expected {expected}, got {actual}"
        )


def _path_from_relative(value: Any, project_root: Path, field: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a relative path string")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"{field} is unsafe")
    resolved = project_root.joinpath(*pure.parts).resolve()
    _relative_path(resolved, project_root)
    if not resolved.exists():
        raise ValueError(f"{field} does not exist: {resolved}")
    return resolved


def _relative_path(path: Path, project_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(project_root.resolve())
    except ValueError as error:
        raise ValueError(f"audit target is outside project root: {path}") from error
    if not relative.parts:
        raise ValueError("project root itself cannot be an audit target")
    return relative.as_posix()


def _parse_cache_arguments(values: Sequence[str]) -> dict[int, Path]:
    parsed: dict[int, Path] = {}
    for value in values:
        raw_year, separator, raw_path = value.partition("=")
        if not separator:
            raise ValueError("--cache must have the form YEAR=PATH")
        try:
            year = int(raw_year)
        except ValueError as error:
            raise ValueError(f"invalid cache year: {raw_year}") from error
        if year in parsed:
            raise ValueError(f"duplicate cache year: {year}")
        parsed[year] = Path(raw_path).resolve()
    _require_equal(set(parsed), set(DEVELOPMENT_YEARS), "--cache years")
    return parsed


def _atomic_exclusive_write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        document, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"audit report already exists: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"JSON input does not exist: {path}")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value: {value}")

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
        raise ValueError("JSON root must be an object")
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


def _digest_text(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _digest_strings(digest: Any, values: Sequence[str]) -> None:
    digest.update(len(values).to_bytes(8, "big"))
    for value in values:
        _digest_text(digest, value)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise ValueError(f"{field} mismatch: expected {expected!r}, got {actual!r}")


if __name__ == "__main__":
    raise SystemExit(main())
