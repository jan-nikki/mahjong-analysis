from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import analysis.audit_combo_cache_reproducibility as audit
from mahjong_analysis.combo_dataset import CandidateDataset, StableYearSample
from mahjong_analysis.combo_sparse_pipeline import (
    YearSparseArtifact,
    save_year_sparse_artifact,
)


def test_create_verify_and_detect_cache_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "outputs" / "cache-audit.json"

    wrapper = audit.create_audit_report(
        manifest_path=fixture.manifest_path,
        cache_paths=fixture.cache_paths,
        raw_root=fixture.raw_root,
        project_root=tmp_path,
        output_path=output,
    )

    assert wrapper["report"]["status"] == "PASS"
    assert (
        wrapper["report"]["scope"]["cache_producer_analysis_id"]
        == "combo-development-primary-r1000-v1"
    )
    assert wrapper["report"]["scope"]["validation_2024_touched"] is False
    assert wrapper["report"]["input_stability"]["identical"] is True
    assert (
        wrapper["report"]["input_stability"]["execution_context_identical"]
        is True
    )
    assert audit.verify_audit_report(output, project_root=tmp_path) == wrapper

    with fixture.cache_paths[2022].open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        audit.verify_audit_report(output, project_root=tmp_path)


def test_create_rejects_semantically_different_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    changed = _artifact(
        2021,
        fixture.manifest_canonical_sha256,
        weights=np.asarray([0.4, 0.6]),
    )
    original_builder = audit.build_year_sparse_artifact

    def changed_builder(*args: Any, **kwargs: Any) -> YearSparseArtifact:
        if kwargs["year"] == 2021:
            return changed
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(audit, "build_year_sparse_artifact", changed_builder)

    with pytest.raises(ValueError, match="weights values"):
        audit.create_audit_report(
            manifest_path=fixture.manifest_path,
            cache_paths=fixture.cache_paths,
            raw_root=fixture.raw_root,
            project_root=tmp_path,
            output_path=tmp_path / "outputs" / "cache-audit.json",
        )


def test_create_rejects_concurrent_source_inventory_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    calls = 0

    def changing_inventory(_raw_root: Path) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        suffix = "0" * 64 if calls == 1 else "1" * 64
        return {
            str(year): {
                "candidate_count": 1000,
                "candidate_population_sha256": "a" * 64,
                "path_size_mtime_inventory_sha256": suffix,
            }
            for year in audit.DEVELOPMENT_YEARS
        }

    monkeypatch.setattr(audit, "source_inventory_snapshot", changing_inventory)

    with pytest.raises(RuntimeError, match="changed during audit"):
        audit.create_audit_report(
            manifest_path=fixture.manifest_path,
            cache_paths=fixture.cache_paths,
            raw_root=fixture.raw_root,
            project_root=tmp_path,
            output_path=tmp_path / "outputs" / "cache-audit.json",
        )


def test_create_rejects_concurrent_project_inventory_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    original_inventory = audit.current_project_inventory
    calls = 0

    def changing_inventory(project_root: Path) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        inventory = original_inventory(project_root)
        if calls >= 2:
            inventory["inventory_sha256"] = "f" * 64
        return inventory

    monkeypatch.setattr(audit, "current_project_inventory", changing_inventory)

    with pytest.raises(RuntimeError, match=r"project code.*changed"):
        audit.create_audit_report(
            manifest_path=fixture.manifest_path,
            cache_paths=fixture.cache_paths,
            raw_root=fixture.raw_root,
            project_root=tmp_path,
            output_path=tmp_path / "outputs" / "cache-audit.json",
        )


def test_create_rejects_finalization_environment_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    calls = 0

    def changing_environment() -> dict[str, str]:
        nonlocal calls
        calls += 1
        return {
            "python": "3.12.10",
            "numpy": "test",
            "scipy": "before" if calls < 3 else "after",
        }

    monkeypatch.setattr(audit, "environment_document", changing_environment)

    with pytest.raises(RuntimeError, match="while finalizing"):
        audit.create_audit_report(
            manifest_path=fixture.manifest_path,
            cache_paths=fixture.cache_paths,
            raw_root=fixture.raw_root,
            project_root=tmp_path,
            output_path=tmp_path / "outputs" / "cache-audit.json",
        )


def test_create_rejects_schema_v1_before_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    manifest = json.loads(fixture.manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 1
    fixture.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="schema v2"):
        audit.create_audit_report(
            manifest_path=fixture.manifest_path,
            cache_paths=fixture.cache_paths,
            raw_root=fixture.raw_root,
            project_root=tmp_path,
            output_path=tmp_path / "outputs" / "cache-audit.json",
        )


def test_report_writer_is_exclusive(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    output.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        audit._atomic_exclusive_write_json(output, {"replacement": True})

    assert output.read_text(encoding="utf-8") == "existing\n"


def _fixture(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> SimpleNamespace:
    _write_project_files(root)
    raw_root = root / "data" / "raw"
    raw_root.mkdir(parents=True)
    manifest = {
        "schema_version": 2,
        "development_years": list(audit.DEVELOPMENT_YEARS),
        "files_per_year": 1000,
        "sample_seed": audit.SAMPLE_SEED,
        "validation_2024_touched": False,
        "holdout_2025_touched": False,
        "years": {
            str(year): {
                "candidate_count": 1000,
                "candidate_population_sha256": "a" * 64,
                "selected": [
                    {
                        "relative_source": f"{year}/{index:04d}.mjson",
                        "selection_sha256": f"{index:064x}",
                        "content_sha256": "b" * 64,
                    }
                    for index in range(1000)
                ],
            }
            for year in audit.DEVELOPMENT_YEARS
        },
    }
    manifest_path = root / "outputs" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_digest = audit.canonical_json_sha256(manifest)

    samples = {
        year: StableYearSample(
            year=year,
            candidate_count=1000,
            paths=tuple(
                raw_root / f"{year}/{index:04d}.mjson" for index in range(1000)
            ),
            relative_sources=tuple(
                f"{year}/{index:04d}.mjson" for index in range(1000)
            ),
            selection_digests=tuple(f"{index:064x}" for index in range(1000)),
            candidate_population_sha256="a" * 64,
            source_content_sha256=tuple("b" * 64 for _ in range(1000)),
        )
        for year in audit.DEVELOPMENT_YEARS
    }
    cache_paths: dict[int, Path] = {}
    artifacts: dict[int, YearSparseArtifact] = {}
    for year in audit.DEVELOPMENT_YEARS:
        artifact = _artifact(year, manifest_digest)
        cache_path = root / "data" / "processed" / f"{year}.npz"
        save_year_sparse_artifact(cache_path, artifact)
        cache_paths[year] = cache_path
        artifacts[year] = artifact

    counts = {"candidate_rows": 2, "scanned_files": 1000}

    def fake_loader(
        _manifest: dict[str, Any], _args: Any
    ) -> dict[int, StableYearSample]:
        return samples

    def fake_collect(
        _raw_root: Path, year: int, _paths: Any
    ) -> CandidateDataset:
        return CandidateDataset(
            year=year,
            rows=(object(), object()),  # type: ignore[arg-type]
            counts=counts,
            selected_sources=samples[year].relative_sources,
            target_sources=(samples[year].relative_sources[0],),
        )

    def fake_build(
        _rows: Any,
        *,
        year: int,
        source_rank_by_game: dict[str, int],
        provenance: dict[str, str],
    ) -> YearSparseArtifact:
        assert len(source_rank_by_game) == 1000
        assert provenance == dict(artifacts[year].provenance)
        return artifacts[year]

    inventory = {
        str(year): {
            "candidate_count": 1000,
            "candidate_population_sha256": "a" * 64,
            "path_size_mtime_inventory_sha256": "0" * 64,
        }
        for year in audit.DEVELOPMENT_YEARS
    }
    monkeypatch.setattr(audit, "_samples_from_manifest", fake_loader)
    monkeypatch.setattr(audit, "collect_candidate_dataset", fake_collect)
    monkeypatch.setattr(audit, "build_year_sparse_artifact", fake_build)
    monkeypatch.setattr(
        audit, "source_inventory_snapshot", lambda _raw_root: inventory
    )
    return SimpleNamespace(
        raw_root=raw_root,
        manifest_path=manifest_path,
        manifest_canonical_sha256=manifest_digest,
        cache_paths=cache_paths,
    )


def _artifact(
    year: int,
    manifest_digest: str,
    *,
    weights: np.ndarray | None = None,
) -> YearSparseArtifact:
    sources = tuple(f"{year}/{index:04d}.mjson" for index in range(1000))
    counts_json = json.dumps(
        {"candidate_rows": 2, "scanned_files": 1000},
        sort_keys=True,
        separators=(",", ":"),
    )
    return YearSparseArtifact(
        year=year,
        feature_names=("x",),
        matrix=csr_matrix(np.asarray([[1.0], [2.0]])),
        y_structural=np.asarray([False, True]),
        y_ron=np.asarray([False, False]),
        weights=(np.asarray([0.5, 0.5]) if weights is None else weights),
        group_indices=np.asarray([0, 0], dtype=np.int32),
        group_ids=(f"{year}:decision",),
        cluster_indices=np.asarray([0, 0], dtype=np.int32),
        cluster_ids=(sources[0],),
        turn_bins=np.asarray([0, 0], dtype=np.uint8),
        source_ranks=np.asarray([0, 0], dtype=np.int32),
        source_ids_by_rank=sources,
        provenance=tuple(
            sorted(
                {
                    "analysis_id": audit.CACHE_PRODUCER_ANALYSIS_ID,
                    "manifest_canonical_sha256": manifest_digest,
                    "counts_json": counts_json,
                }.items()
            )
        ),
    )


def _write_project_files(root: Path) -> None:
    files = {
        "pyproject.toml": "[project]\nname='fixture'\n",
        "uv.lock": "version = 1\n",
        "analysis/audit_combo_cache_reproducibility.py": "AUDIT = True\n",
        "analysis/helper.py": "HELPER = True\n",
        "src/mahjong_analysis/rules.py": "RULE = True\n",
        "docs/specs/combo-prediction-freeze.md": "# freeze\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
