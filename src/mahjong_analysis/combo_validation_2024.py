"""Fail-closed sampling and extraction for the fixed 2024 validation set.

This module is intentionally separate from :mod:`combo_dataset`.  The latter
only permits the development years, while this module permits exactly 2024
and never scans a sibling year directory.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
from heapq import nsmallest
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from mahjong_analysis.combo_dataset import (
    SELECTION_ALGORITHM,
    CandidateDataset,
    StableYearSample,
    file_sha256,
    selection_digest,
)
from mahjong_analysis.combo_prediction import (
    ComboCandidateRow,
    candidate_rows_for_decision,
)
from mahjong_analysis.mjai import is_target_game, load_mjai, split_kyoku
from mahjong_analysis.post_riichi_decisions import (
    extract_post_riichi_draw_decisions,
)

VALIDATION_YEAR = 2024
FILES_PER_YEAR = 1_000
SAMPLE_SEED = 20260923
VALIDATION_SAMPLE_SIZE = FILES_PER_YEAR
VALIDATION_SAMPLE_SEED = SAMPLE_SEED
MANIFEST_SCHEMA_VERSION = 2

_FREEZE_BUNDLE_SCHEMA_VERSION = 1
_FREEZE_ANALYSIS_ID = "combo-development-primary-r1000-v2"
_FREEZE_STATUS = "GO_FOR_2024_VALIDATION"


def select_validation_2024_sample(
    raw_root: Path,
    sample_size: int = VALIDATION_SAMPLE_SIZE,
    *,
    seed: int = VALIDATION_SAMPLE_SEED,
    year: int = VALIDATION_YEAR,
) -> StableYearSample:
    """Return the fixed lowest-hash 1,000-file sample from only ``2024/``."""

    _validate_fixed_protocol(year=year, sample_size=sample_size, seed=seed)
    canonical_root = Path(raw_root).resolve()
    year_root = (canonical_root / str(VALIDATION_YEAR)).resolve()
    _require_exact_year_root(canonical_root, year_root)
    if not year_root.is_dir():
        raise FileNotFoundError(
            f"raw validation year directory does not exist: {year_root}"
        )

    candidate_count = 0
    candidate_sources: list[str] = []
    seen_sources: set[str] = set()

    def scored_paths() -> Iterable[tuple[str, str, Path]]:
        nonlocal candidate_count
        for discovered_path in year_root.rglob("*.mjson"):
            if not discovered_path.is_file():
                continue
            path = discovered_path.resolve()
            try:
                path.relative_to(year_root)
                relative_source = path.relative_to(canonical_root).as_posix()
            except ValueError as error:
                raise ValueError(
                    f"validation candidate escapes raw_root/2024: {discovered_path}"
                ) from error
            _validate_relative_source(relative_source)
            if relative_source in seen_sources:
                raise ValueError(
                    f"duplicate validation candidate source: {relative_source}"
                )
            seen_sources.add(relative_source)
            candidate_count += 1
            candidate_sources.append(relative_source)
            yield selection_digest(relative_source, seed), relative_source, path

    selected = nsmallest(sample_size, scored_paths())
    if len(selected) != sample_size:
        raise ValueError(
            f"requested exactly {sample_size} files for {VALIDATION_YEAR}, "
            f"found {candidate_count}"
        )
    selection_hashes = tuple(item[0] for item in selected)
    if len(set(selection_hashes)) != sample_size:
        raise ValueError("selected validation sources have colliding selection hashes")
    return StableYearSample(
        year=VALIDATION_YEAR,
        candidate_count=candidate_count,
        paths=tuple(item[2] for item in selected),
        relative_sources=tuple(item[1] for item in selected),
        selection_digests=selection_hashes,
        candidate_population_sha256=_population_sha256(candidate_sources),
        source_content_sha256=tuple(file_sha256(item[2]) for item in selected),
    )


def collect_validation_2024_dataset(
    raw_root: Path,
    paths: Sequence[Path],
    *,
    year: int = VALIDATION_YEAR,
    expected_content_sha256: Sequence[str] | None = None,
) -> CandidateDataset:
    """Extract the same candidate rows as development, but only from 2024."""

    _validate_validation_year(year)
    if not paths:
        raise ValueError("at least one 2024 source path is required")
    expected_hashes: tuple[str, ...] | None = None
    if expected_content_sha256 is not None:
        expected_hashes = tuple(expected_content_sha256)
        if len(expected_hashes) != len(paths):
            raise ValueError("2024 expected content hash count differs from paths")
        for index, expected_hash in enumerate(expected_hashes):
            _sha256_text(expected_hash, f"expected content hash at index {index}")

    canonical_root = Path(raw_root).resolve()
    year_root = (canonical_root / str(VALIDATION_YEAR)).resolve()
    _require_exact_year_root(canonical_root, year_root)
    rows: list[ComboCandidateRow] = []
    counts: Counter[str] = Counter()
    selected_sources: list[str] = []
    target_sources: list[str] = []
    seen_sources: set[str] = set()

    for index, raw_path in enumerate(paths):
        path = Path(raw_path).resolve()
        try:
            path.relative_to(year_root)
            relative_source = path.relative_to(canonical_root).as_posix()
        except ValueError as error:
            raise ValueError(f"source path is not in 2024: {raw_path}") from error
        _validate_relative_source(relative_source)
        if not path.is_file():
            raise FileNotFoundError(f"2024 source does not exist: {path}")
        if relative_source in seen_sources:
            raise ValueError(f"duplicate source path: {relative_source}")
        seen_sources.add(relative_source)
        selected_sources.append(relative_source)
        counts["scanned_files"] += 1

        expected_hash = None if expected_hashes is None else expected_hashes[index]
        if expected_hash is not None and file_sha256(path) != expected_hash:
            raise ValueError(
                f"2024 source content changed before extraction at index {index}"
            )
        events = load_mjai(path)
        if expected_hash is not None and file_sha256(path) != expected_hash:
            raise ValueError(
                f"2024 source content changed during extraction at index {index}"
            )
        if not is_target_game(path, events):
            counts["excluded_non_target_games"] += 1
            continue
        counts["target_games"] += 1
        target_sources.append(relative_source)
        for kyoku_index, kyoku in enumerate(split_kyoku(events)):
            if kyoku[0].get("bakaze") != "E":
                continue
            counts["east_kyokus"] += 1
            decisions = extract_post_riichi_draw_decisions(kyoku)
            counts["decisions"] += len(decisions)
            for decision in decisions:
                candidate_rows = candidate_rows_for_decision(
                    decision,
                    year=VALIDATION_YEAR,
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

    if expected_hashes is not None:
        for index, (raw_path, expected_hash) in enumerate(
            zip(paths, expected_hashes, strict=True)
        ):
            if file_sha256(Path(raw_path).resolve()) != expected_hash:
                raise ValueError(
                    "2024 source content changed before cache publication "
                    f"at index {index}"
                )
    if not rows:
        raise ValueError("no candidate rows extracted for 2024")
    return CandidateDataset(
        year=VALIDATION_YEAR,
        rows=tuple(rows),
        counts=dict(sorted(counts.items())),
        selected_sources=tuple(selected_sources),
        target_sources=tuple(target_sources),
    )


def build_validation_2024_manifest(
    sample: StableYearSample,
    *,
    freeze_bundle_path: Path,
    freeze_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a schema-v2 manifest bound to the verified development freeze."""

    _validate_sample(sample)
    freeze_identity = _freeze_bundle_identity(freeze_bundle_path, freeze_bundle)
    document: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "selection_algorithm": SELECTION_ALGORITHM,
        "sample_seed": VALIDATION_SAMPLE_SEED,
        "files_per_year": VALIDATION_SAMPLE_SIZE,
        "validation_year": VALIDATION_YEAR,
        "validation_2024_touched": True,
        "holdout_2025_touched": False,
        "development_freeze_bundle": freeze_identity,
        "years": {str(VALIDATION_YEAR): _sample_document(sample)},
    }
    _validate_manifest_shape(document)
    return document


def write_validation_2024_manifest(
    manifest_path: Path,
    document: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Write a new manifest and refuse to replace any existing filesystem item."""

    if overwrite is not False:
        raise ValueError("overwriting a 2024 validation manifest is forbidden")
    _validate_manifest_shape(document)
    output = Path(manifest_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = (
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    staged = output.with_name(f".{output.name}.tmp-{uuid4().hex}")
    staged_created = False
    try:
        with staged.open("x", encoding="utf-8", newline="\n") as destination:
            staged_created = True
            destination.write(rendered)
            destination.flush()
            os.fsync(destination.fileno())
        # Publishing by hard link is atomic and fails if an immutable manifest
        # already exists, including one created by a concurrent process.
        os.link(staged, output)
    finally:
        if staged_created:
            staged.unlink(missing_ok=True)


def load_validation_2024_manifest(
    manifest_path: Path,
    *,
    raw_root: Path,
    freeze_bundle_path: Path,
    freeze_bundle: Mapping[str, Any],
) -> StableYearSample:
    """Load a manifest and recompute every population and selected-file gate."""

    document = _read_json_object(manifest_path, label="2024 manifest")
    year_document = _validate_manifest_shape(document)
    actual_freeze = _freeze_bundle_identity(freeze_bundle_path, freeze_bundle)
    if document["development_freeze_bundle"] != actual_freeze:
        raise ValueError("2024 manifest freeze bundle identity mismatch")

    actual = select_validation_2024_sample(raw_root)
    if year_document["candidate_count"] != actual.candidate_count:
        raise ValueError("2024 manifest candidate_count mismatch")
    if (
        year_document["candidate_population_sha256"]
        != actual.candidate_population_sha256
    ):
        raise ValueError("2024 manifest candidate population SHA-256 mismatch")

    selected = year_document["selected"]
    stored_sources = tuple(item["relative_source"] for item in selected)
    stored_selection_hashes = tuple(item["selection_sha256"] for item in selected)
    stored_content_hashes = tuple(item["content_sha256"] for item in selected)
    if stored_sources != actual.relative_sources:
        raise ValueError("2024 manifest does not contain the lowest-hash sources")
    if stored_selection_hashes != actual.selection_digests:
        raise ValueError("2024 manifest selection hash order mismatch")
    if stored_content_hashes != actual.source_content_sha256:
        raise ValueError("2024 manifest selected content SHA-256 mismatch")
    return actual


def _validate_fixed_protocol(*, year: int, sample_size: int, seed: int) -> None:
    _validate_validation_year(year)
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise TypeError("sample_size must be an integer")
    if sample_size != VALIDATION_SAMPLE_SIZE:
        raise ValueError(
            f"2024 validation sample_size must be exactly {VALIDATION_SAMPLE_SIZE}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed != VALIDATION_SAMPLE_SEED:
        raise ValueError(
            f"2024 validation seed must be exactly {VALIDATION_SAMPLE_SEED}"
        )


def _validate_validation_year(year: int) -> None:
    if isinstance(year, bool) or not isinstance(year, int):
        raise TypeError("year must be an integer")
    if year != VALIDATION_YEAR:
        raise ValueError(
            f"validation year must be exactly {VALIDATION_YEAR}, got {year}"
        )


def _require_exact_year_root(canonical_root: Path, year_root: Path) -> None:
    try:
        relative = year_root.relative_to(canonical_root)
    except ValueError as error:
        raise ValueError("resolved raw_root/2024 escapes raw_root") from error
    if relative.parts != (str(VALIDATION_YEAR),):
        raise ValueError("resolved validation directory must be exactly raw_root/2024")


def _population_sha256(relative_sources: Sequence[str]) -> str:
    digest = sha256()
    for source in sorted(relative_sources):
        digest.update(source.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _sample_document(sample: StableYearSample) -> dict[str, Any]:
    return {
        "candidate_count": sample.candidate_count,
        "candidate_population_sha256": sample.candidate_population_sha256,
        "selected": [
            {
                "relative_source": source,
                "selection_sha256": selection_hash,
                "content_sha256": content_hash,
            }
            for source, selection_hash, content_hash in zip(
                sample.relative_sources,
                sample.selection_digests,
                sample.source_content_sha256,
                strict=True,
            )
        ],
    }


def _validate_sample(sample: StableYearSample) -> None:
    if not isinstance(sample, StableYearSample):
        raise TypeError("sample must be a StableYearSample")
    _validate_validation_year(sample.year)
    if (
        isinstance(sample.candidate_count, bool)
        or not isinstance(sample.candidate_count, int)
        or sample.candidate_count < VALIDATION_SAMPLE_SIZE
    ):
        raise ValueError("2024 sample candidate_count is invalid")
    lengths = {
        len(sample.paths),
        len(sample.relative_sources),
        len(sample.selection_digests),
        len(sample.source_content_sha256),
    }
    if lengths != {VALIDATION_SAMPLE_SIZE}:
        raise ValueError("2024 sample must contain exactly 1000 selected files")
    _sha256_text(
        sample.candidate_population_sha256,
        "sample.candidate_population_sha256",
    )
    if len(set(sample.relative_sources)) != VALIDATION_SAMPLE_SIZE:
        raise ValueError("2024 sample contains duplicate sources")
    if len(set(sample.selection_digests)) != VALIDATION_SAMPLE_SIZE:
        raise ValueError("2024 sample contains duplicate selection hashes")
    if sample.selection_digests != tuple(sorted(sample.selection_digests)):
        raise ValueError("2024 sample selection hashes are not sorted")

    roots: set[Path] = set()
    for index, (path, source, selection_hash, content_hash) in enumerate(
        zip(
            sample.paths,
            sample.relative_sources,
            sample.selection_digests,
            sample.source_content_sha256,
            strict=True,
        )
    ):
        parts = _validate_relative_source(source)
        expected_selection_hash = selection_digest(source, VALIDATION_SAMPLE_SEED)
        if selection_hash != expected_selection_hash:
            raise ValueError(f"sample selection hash mismatch at index {index}")
        _sha256_text(content_hash, f"sample content hash at index {index}")
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"selected 2024 source does not exist: {resolved}")
        if tuple(resolved.parts[-len(parts) :]) != parts:
            raise ValueError(f"sample path/source mismatch at index {index}")
        root = resolved
        for _part in parts:
            root = root.parent
        roots.add(root)
        if file_sha256(resolved) != content_hash:
            raise ValueError(f"sample content hash mismatch at index {index}")
    if len(roots) != 1:
        raise ValueError("sample paths do not share one raw_root")


def _validate_relative_source(relative_source: str) -> tuple[str, ...]:
    if not isinstance(relative_source, str) or not relative_source:
        raise ValueError("relative_source must be a non-empty string")
    if "\\" in relative_source:
        raise ValueError(f"unsafe 2024 relative source: {relative_source!r}")
    pure = PurePosixPath(relative_source)
    if (
        pure.is_absolute()
        or pure.as_posix() != relative_source
        or not pure.parts
        or pure.parts[0] != str(VALIDATION_YEAR)
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.suffix != ".mjson"
    ):
        raise ValueError(f"source is not an exact 2024 MJAI path: {relative_source!r}")
    return pure.parts


def _freeze_bundle_identity(
    freeze_bundle_path: Path,
    freeze_bundle: Mapping[str, Any],
) -> dict[str, str]:
    if not isinstance(freeze_bundle, Mapping):
        raise TypeError("freeze_bundle must be a mapping")
    path = Path(freeze_bundle_path).resolve()
    file_document = _read_json_object(path, label="development freeze bundle")
    if file_document != dict(freeze_bundle):
        raise ValueError("development freeze bundle argument does not match its file")
    if set(file_document) != {"schema_version", "payload_sha256", "payload"}:
        raise ValueError("development freeze bundle has an unexpected top-level shape")
    _exact_integer(
        file_document["schema_version"],
        _FREEZE_BUNDLE_SCHEMA_VERSION,
        "freeze bundle schema_version",
    )
    payload = file_document["payload"]
    if not isinstance(payload, dict):
        raise TypeError("development freeze bundle payload must be an object")
    payload_sha256 = _sha256_text(
        file_document["payload_sha256"],
        "freeze bundle payload_sha256",
    )
    if _canonical_json_sha256(payload) != payload_sha256:
        raise ValueError("development freeze bundle payload SHA-256 mismatch")
    if payload.get("analysis_id") != _FREEZE_ANALYSIS_ID:
        raise ValueError("development freeze bundle analysis_id mismatch")
    if payload.get("status") != _FREEZE_STATUS:
        raise ValueError("development freeze bundle is not GO for 2024 validation")
    protocol = payload.get("validation_2024_protocol")
    if not isinstance(protocol, dict):
        raise TypeError("development freeze bundle lacks the 2024 protocol")
    expected_protocol = {
        "validation_year": VALIDATION_YEAR,
        "holdout_year": 2025,
        "files_per_year": VALIDATION_SAMPLE_SIZE,
        "sample_seed": VALIDATION_SAMPLE_SEED,
        "selection_algorithm": SELECTION_ALGORITHM,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"development freeze bundle protocol {key!r} mismatch")
    return {
        "file_sha256": file_sha256(path),
        "payload_sha256": payload_sha256,
    }


def _validate_manifest_shape(
    document: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(document, Mapping):
        raise TypeError("2024 manifest must be a mapping")
    expected_keys = {
        "schema_version",
        "selection_algorithm",
        "sample_seed",
        "files_per_year",
        "validation_year",
        "validation_2024_touched",
        "holdout_2025_touched",
        "development_freeze_bundle",
        "years",
    }
    if set(document) != expected_keys:
        raise ValueError("2024 manifest has an unexpected top-level shape")
    _exact_integer(
        document["schema_version"],
        MANIFEST_SCHEMA_VERSION,
        "manifest schema_version",
    )
    if document["selection_algorithm"] != SELECTION_ALGORITHM:
        raise ValueError("2024 manifest selection_algorithm mismatch")
    _exact_integer(
        document["sample_seed"],
        VALIDATION_SAMPLE_SEED,
        "manifest sample_seed",
    )
    _exact_integer(
        document["files_per_year"],
        VALIDATION_SAMPLE_SIZE,
        "manifest files_per_year",
    )
    _exact_integer(
        document["validation_year"],
        VALIDATION_YEAR,
        "manifest validation_year",
    )
    if document["validation_2024_touched"] is not True:
        raise ValueError("2024 manifest must mark validation_2024_touched=true")
    if document["holdout_2025_touched"] is not False:
        raise ValueError("2024 manifest must mark holdout_2025_touched=false")

    freeze_identity = document["development_freeze_bundle"]
    if not isinstance(freeze_identity, Mapping) or set(freeze_identity) != {
        "file_sha256",
        "payload_sha256",
    }:
        raise ValueError("2024 manifest freeze bundle identity is invalid")
    _sha256_text(freeze_identity["file_sha256"], "freeze bundle file SHA-256")
    _sha256_text(
        freeze_identity["payload_sha256"],
        "freeze bundle payload SHA-256",
    )

    years = document["years"]
    if not isinstance(years, Mapping) or set(years) != {str(VALIDATION_YEAR)}:
        raise ValueError("2024 manifest years must contain exactly 2024")
    year_document = years[str(VALIDATION_YEAR)]
    if not isinstance(year_document, Mapping) or set(year_document) != {
        "candidate_count",
        "candidate_population_sha256",
        "selected",
    }:
        raise ValueError("2024 manifest year document is invalid")
    candidate_count = year_document["candidate_count"]
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count < VALIDATION_SAMPLE_SIZE
    ):
        raise ValueError("2024 manifest candidate_count is invalid")
    _sha256_text(
        year_document["candidate_population_sha256"],
        "candidate population SHA-256",
    )
    selected = year_document["selected"]
    if not isinstance(selected, list) or len(selected) != VALIDATION_SAMPLE_SIZE:
        raise ValueError("2024 manifest must select exactly 1000 files")

    sources: list[str] = []
    selection_hashes: list[str] = []
    for index, item in enumerate(selected):
        if not isinstance(item, Mapping) or set(item) != {
            "relative_source",
            "selection_sha256",
            "content_sha256",
        }:
            raise ValueError(f"2024 manifest selected[{index}] is invalid")
        source = item["relative_source"]
        _validate_relative_source(source)
        selection_hash = _sha256_text(
            item["selection_sha256"],
            f"selected[{index}] selection SHA-256",
        )
        if selection_hash != selection_digest(source, VALIDATION_SAMPLE_SEED):
            raise ValueError(f"2024 manifest selection hash mismatch: {source}")
        _sha256_text(
            item["content_sha256"],
            f"selected[{index}] content SHA-256",
        )
        sources.append(source)
        selection_hashes.append(selection_hash)
    if len(set(sources)) != VALIDATION_SAMPLE_SIZE:
        raise ValueError("2024 manifest selected sources are not unique")
    if len(set(selection_hashes)) != VALIDATION_SAMPLE_SIZE:
        raise ValueError("2024 manifest selection hashes are not unique")
    if selection_hashes != sorted(selection_hashes):
        raise ValueError("2024 manifest selection hashes are not in lowest-hash order")
    return year_document


def _exact_integer(value: Any, expected: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"{label} must be exactly {expected}")


def _sha256_text(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _canonical_json_sha256(document: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return sha256(encoded).hexdigest()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    document = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"{label} root must be an object")
    return document
