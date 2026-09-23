from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import mahjong_analysis.combo_validation_2024 as validation
from mahjong_analysis.combo_dataset import (
    SELECTION_ALGORITHM,
    file_sha256,
    selection_digest,
)


def _create_sources(raw_root: Path, count: int = 1_002) -> tuple[str, ...]:
    year_root = raw_root / "2024"
    year_root.mkdir(parents=True)
    sources: list[str] = []
    for index in range(count):
        relative_source = f"2024/game-{index:04d}.mjson"
        path = raw_root.joinpath(*relative_source.split("/"))
        path.write_text(f"synthetic-{index}\n", encoding="utf-8")
        sources.append(relative_source)
    (year_root / "ignored.txt").write_text("ignored", encoding="utf-8")
    return tuple(sources)


def _canonical_sha256(document: dict[str, Any]) -> str:
    encoded = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256(encoded).hexdigest()


def _write_freeze_bundle(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    payload = {
        "analysis_id": "combo-development-primary-r1000-v2",
        "status": "GO_FOR_2024_VALIDATION",
        "validation_2024_protocol": {
            "validation_year": 2024,
            "holdout_year": 2025,
            "files_per_year": 1_000,
            "sample_seed": 20260923,
            "selection_algorithm": SELECTION_ALGORITHM,
            "manifest_schema_version": 2,
        },
    }
    bundle = {
        "schema_version": 1,
        "payload_sha256": _canonical_sha256(payload),
        "payload": payload,
    }
    path = tmp_path / "development.freeze.json"
    path.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path, bundle


def _build_case(
    tmp_path: Path,
    *,
    candidate_count: int = 1_000,
) -> tuple[
    Path,
    tuple[str, ...],
    validation.StableYearSample,
    Path,
    dict[str, Any],
    dict[str, Any],
]:
    raw_root = tmp_path / "raw"
    sources = _create_sources(raw_root, candidate_count)
    sample = validation.select_validation_2024_sample(raw_root)
    freeze_path, freeze_bundle = _write_freeze_bundle(tmp_path)
    manifest = validation.build_validation_2024_manifest(
        sample,
        freeze_bundle_path=freeze_path,
        freeze_bundle=freeze_bundle,
    )
    return raw_root, sources, sample, freeze_path, freeze_bundle, manifest


@pytest.fixture(scope="module")
def validation_case(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[
    Path,
    tuple[str, ...],
    validation.StableYearSample,
    Path,
    dict[str, Any],
    dict[str, Any],
]:
    return _build_case(tmp_path_factory.mktemp("combo-validation-2024"))


def _write_document(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_exact_2024_sample_is_deterministic_and_population_complete(
    validation_case: tuple[
        Path,
        tuple[str, ...],
        validation.StableYearSample,
        Path,
        dict[str, Any],
        dict[str, Any],
    ],
) -> None:
    raw_root, all_sources, first, _freeze_path, _freeze_bundle, _manifest = (
        validation_case
    )
    second = validation.select_validation_2024_sample(raw_root)

    expected_ranked = sorted(
        (selection_digest(source, 20260923), source) for source in all_sources
    )[:1000]
    population_payload = "".join(f"{source}\n" for source in sorted(all_sources))
    assert validation.VALIDATION_YEAR == 2024
    assert first == second
    assert first.year == 2024
    assert first.candidate_count == 1_000
    assert first.selection_digests == tuple(item[0] for item in expected_ranked)
    assert first.relative_sources == tuple(item[1] for item in expected_ranked)
    assert first.candidate_population_sha256 == sha256(
        population_payload.encode()
    ).hexdigest()
    assert first.source_content_sha256 == tuple(
        file_sha256(path) for path in first.paths
    )


def test_sampler_rejects_nonfixed_protocol_before_enumeration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def forbidden_rglob(_path: Path, _pattern: str) -> tuple[Path, ...]:
        nonlocal calls
        calls += 1
        raise AssertionError("no year directory may be enumerated")

    monkeypatch.setattr(Path, "rglob", forbidden_rglob)

    with pytest.raises(ValueError, match="exactly 2024"):
        validation.select_validation_2024_sample(tmp_path, year=2025)
    with pytest.raises(ValueError, match="sample_size must be exactly 1000"):
        validation.select_validation_2024_sample(tmp_path, sample_size=999)
    with pytest.raises(ValueError, match="seed must be exactly 20260923"):
        validation.select_validation_2024_sample(tmp_path, seed=1)
    assert calls == 0


def test_manifest_round_trip_binds_freeze_and_refuses_overwrite(
    tmp_path: Path,
    validation_case: tuple[
        Path,
        tuple[str, ...],
        validation.StableYearSample,
        Path,
        dict[str, Any],
        dict[str, Any],
    ],
) -> None:
    raw_root, _sources, sample, freeze_path, freeze_bundle, manifest = (
        validation_case
    )
    manifest_path = tmp_path / "validation-2024.manifest.json"
    staged_pattern = f".{manifest_path.name}.tmp-*"

    validation.write_validation_2024_manifest(manifest_path, manifest)
    original_bytes = manifest_path.read_bytes()
    assert list(tmp_path.glob(staged_pattern)) == []
    with pytest.raises(FileExistsError):
        validation.write_validation_2024_manifest(manifest_path, manifest)
    assert manifest_path.read_bytes() == original_bytes
    assert list(tmp_path.glob(staged_pattern)) == []
    with pytest.raises(ValueError, match=r"overwriting.*forbidden"):
        validation.write_validation_2024_manifest(
            manifest_path,
            manifest,
            overwrite=True,
        )
    assert manifest_path.read_bytes() == original_bytes
    assert list(tmp_path.glob(staged_pattern)) == []

    loaded = validation.load_validation_2024_manifest(
        manifest_path,
        raw_root=raw_root,
        freeze_bundle_path=freeze_path,
        freeze_bundle=freeze_bundle,
    )
    assert loaded == sample
    assert manifest_path.read_bytes() == original_bytes
    assert manifest["schema_version"] == 2
    assert manifest["sample_seed"] == 20260923
    assert manifest["files_per_year"] == 1_000
    assert manifest["validation_2024_touched"] is True
    assert manifest["holdout_2025_touched"] is False
    assert manifest["development_freeze_bundle"] == {
        "file_sha256": file_sha256(freeze_path),
        "payload_sha256": freeze_bundle["payload_sha256"],
    }


def test_manifest_detects_selected_content_and_population_tampering(
    tmp_path: Path,
    validation_case: tuple[
        Path,
        tuple[str, ...],
        validation.StableYearSample,
        Path,
        dict[str, Any],
        dict[str, Any],
    ],
) -> None:
    raw_root, _sources, _sample, freeze_path, freeze_bundle, manifest = (
        validation_case
    )

    content_tamper = deepcopy(manifest)
    content_tamper["years"]["2024"]["selected"][-1]["content_sha256"] = "0" * 64
    content_path = tmp_path / "content-tampered.json"
    _write_document(content_path, content_tamper)
    with pytest.raises(ValueError, match="selected content SHA-256 mismatch"):
        validation.load_validation_2024_manifest(
            content_path,
            raw_root=raw_root,
            freeze_bundle_path=freeze_path,
            freeze_bundle=freeze_bundle,
        )

    population_tamper = deepcopy(manifest)
    population_tamper["years"]["2024"][
        "candidate_population_sha256"
    ] = "0" * 64
    population_path = tmp_path / "population-tampered.json"
    _write_document(population_path, population_tamper)
    with pytest.raises(ValueError, match="candidate population SHA-256 mismatch"):
        validation.load_validation_2024_manifest(
            population_path,
            raw_root=raw_root,
            freeze_bundle_path=freeze_path,
            freeze_bundle=freeze_bundle,
        )


def test_manifest_recomputes_full_population_and_freeze_file_hash(
    tmp_path: Path,
    validation_case: tuple[
        Path,
        tuple[str, ...],
        validation.StableYearSample,
        Path,
        dict[str, Any],
        dict[str, Any],
    ],
) -> None:
    raw_root, _sources, _sample, freeze_path, freeze_bundle, manifest = (
        validation_case
    )
    manifest_path = tmp_path / "validation-2024.manifest.json"
    validation.write_validation_2024_manifest(manifest_path, manifest)

    added = raw_root / "2024" / "added-after-manifest.mjson"
    added.write_text("new population member\n", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="candidate_count mismatch"):
            validation.load_validation_2024_manifest(
                manifest_path,
                raw_root=raw_root,
                freeze_bundle_path=freeze_path,
                freeze_bundle=freeze_bundle,
            )
    finally:
        added.unlink(missing_ok=True)

    tampered_freeze_path = tmp_path / "whitespace-tampered.freeze.json"
    tampered_freeze_path.write_text(
        freeze_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="freeze bundle identity mismatch"):
        validation.load_validation_2024_manifest(
            manifest_path,
            raw_root=raw_root,
            freeze_bundle_path=tampered_freeze_path,
            freeze_bundle=freeze_bundle,
        )


def test_manifest_rejects_selection_tamper_and_non_2024_sample(
    tmp_path: Path,
    validation_case: tuple[
        Path,
        tuple[str, ...],
        validation.StableYearSample,
        Path,
        dict[str, Any],
        dict[str, Any],
    ],
) -> None:
    raw_root, _sources, sample, freeze_path, freeze_bundle, manifest = (
        validation_case
    )
    tampered_manifest = deepcopy(manifest)
    tampered_manifest["years"]["2024"]["selected"][0][
        "selection_sha256"
    ] = "0" * 64
    manifest_path = tmp_path / "selection-tampered.json"
    _write_document(manifest_path, tampered_manifest)
    with pytest.raises(ValueError, match="selection hash mismatch"):
        validation.load_validation_2024_manifest(
            manifest_path,
            raw_root=raw_root,
            freeze_bundle_path=freeze_path,
            freeze_bundle=freeze_bundle,
        )

    with pytest.raises(ValueError, match="exactly 2024"):
        validation.build_validation_2024_manifest(
            replace(sample, year=2025),
            freeze_bundle_path=freeze_path,
            freeze_bundle=freeze_bundle,
        )


def test_2024_extraction_uses_existing_lower_level_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "raw"
    source = raw_root / "2024" / "target.mjson"
    source.parent.mkdir(parents=True)
    source.write_text("synthetic", encoding="utf-8")
    captured: list[dict[str, Any]] = []
    rows = (
        SimpleNamespace(
            candidate=SimpleNamespace(
                is_structural_wait=True,
                is_ron_eligible=False,
            )
        ),
        SimpleNamespace(
            candidate=SimpleNamespace(
                is_structural_wait=False,
                is_ron_eligible=True,
            )
        ),
    )

    monkeypatch.setattr(validation, "load_mjai", lambda _path: ["events"])
    monkeypatch.setattr(validation, "is_target_game", lambda _path, _events: True)
    monkeypatch.setattr(
        validation,
        "split_kyoku",
        lambda _events: ([{"bakaze": "E"}], [{"bakaze": "S"}]),
    )
    monkeypatch.setattr(
        validation,
        "extract_post_riichi_draw_decisions",
        lambda _kyoku: ("decision",),
    )

    def candidate_rows(_decision: str, **kwargs: Any) -> tuple[Any, ...]:
        captured.append(kwargs)
        return rows

    monkeypatch.setattr(validation, "candidate_rows_for_decision", candidate_rows)

    dataset = validation.collect_validation_2024_dataset(raw_root, (source,))

    assert dataset.year == 2024
    assert dataset.rows == rows
    assert dataset.selected_sources == ("2024/target.mjson",)
    assert dataset.target_sources == ("2024/target.mjson",)
    assert dataset.counts == {
        "candidate_rows": 2,
        "decisions": 1,
        "east_kyokus": 1,
        "ron_eligible_rows": 1,
        "scanned_files": 1,
        "structural_wait_rows": 1,
        "target_games": 1,
    }
    assert captured == [
        {
            "year": 2024,
            "game_id": "2024/target.mjson",
            "kyoku_index": 0,
        }
    ]

    outside = raw_root / "2025" / "forbidden.mjson"
    outside.parent.mkdir()
    outside.write_text("do not load", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 2024"):
        validation.collect_validation_2024_dataset(
            raw_root,
            (outside,),
            year=2025,
        )
    with pytest.raises(ValueError, match="not in 2024"):
        validation.collect_validation_2024_dataset(raw_root, (outside,))


def test_2024_extraction_rejects_source_change_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "raw"
    source = raw_root / "2024" / "target.mjson"
    source.parent.mkdir(parents=True)
    source.write_text("original", encoding="utf-8")
    expected_hash = file_sha256(source)

    def mutating_load(path: Path) -> list[dict[str, Any]]:
        path.write_text("changed", encoding="utf-8")
        return []

    monkeypatch.setattr(validation, "load_mjai", mutating_load)

    with pytest.raises(ValueError, match="changed during extraction"):
        validation.collect_validation_2024_dataset(
            raw_root,
            (source,),
            expected_content_sha256=(expected_hash,),
        )


def test_2024_extraction_rechecks_all_sources_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "raw"
    first = raw_root / "2024" / "first.mjson"
    second = raw_root / "2024" / "second.mjson"
    first.parent.mkdir(parents=True)
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    expected_hashes = (file_sha256(first), file_sha256(second))
    calls = 0

    def mutating_later_load(_path: Path) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        if calls == 2:
            first.write_text("changed after first read", encoding="utf-8")
        return []

    monkeypatch.setattr(validation, "load_mjai", mutating_later_load)
    monkeypatch.setattr(validation, "is_target_game", lambda *_args: False)

    with pytest.raises(ValueError, match="before cache publication"):
        validation.collect_validation_2024_dataset(
            raw_root,
            (first, second),
            expected_content_sha256=expected_hashes,
        )
