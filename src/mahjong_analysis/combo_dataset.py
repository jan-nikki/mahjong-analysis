"""Reproducible sampling and extraction for combo-theory prediction data."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from heapq import nsmallest
from pathlib import Path

from mahjong_analysis.combo_prediction import (
    ComboCandidateRow,
    candidate_rows_for_decision,
)
from mahjong_analysis.mjai import is_target_game, load_mjai, split_kyoku
from mahjong_analysis.post_riichi_decisions import (
    extract_post_riichi_draw_decisions,
)

DEVELOPMENT_YEARS = tuple(range(2020, 2024))
SELECTION_ALGORITHM = "lowest sha256(namespace, seed, relative_source), v1"
_SELECTION_NAMESPACE = "mahjong-analysis/combo-development-sample/v1"


@dataclass(frozen=True)
class StableYearSample:
    """A deterministic, filesystem-order-independent sample of source files."""

    year: int
    candidate_count: int
    paths: tuple[Path, ...]
    relative_sources: tuple[str, ...]
    selection_digests: tuple[str, ...]
    candidate_population_sha256: str
    source_content_sha256: tuple[str, ...]


@dataclass(frozen=True)
class CandidateDataset:
    """Candidate rows and extraction counts for one selected year sample."""

    year: int
    rows: tuple[ComboCandidateRow, ...]
    counts: dict[str, int]
    selected_sources: tuple[str, ...]
    target_sources: tuple[str, ...]


def select_stable_year_sample(
    raw_root: Path,
    year: int,
    sample_size: int,
    *,
    seed: int,
) -> StableYearSample:
    """Select the lowest stable path hashes without relying on directory order."""
    _validate_development_year(year)
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise TypeError("sample_size must be an integer")
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    canonical_root = raw_root.resolve()
    year_root = canonical_root / str(year)
    if not year_root.is_dir():
        raise FileNotFoundError(f"raw year directory does not exist: {year_root}")

    candidate_count = 0
    candidate_sources: list[str] = []

    def scored_paths() -> Iterable[tuple[str, str, Path]]:
        nonlocal candidate_count
        for path in year_root.rglob("*.mjson"):
            if not path.is_file():
                continue
            candidate_count += 1
            relative_source = path.relative_to(canonical_root).as_posix()
            candidate_sources.append(relative_source)
            digest = selection_digest(relative_source, seed)
            yield digest, relative_source, path

    selected = nsmallest(sample_size, scored_paths())
    if len(selected) != sample_size:
        raise ValueError(
            f"requested {sample_size} files for {year}, found {candidate_count}"
        )
    return StableYearSample(
        year=year,
        candidate_count=candidate_count,
        paths=tuple(path for _digest, _source, path in selected),
        relative_sources=tuple(source for _digest, source, _path in selected),
        selection_digests=tuple(digest for digest, _source, _path in selected),
        candidate_population_sha256=_population_sha256(candidate_sources),
        source_content_sha256=tuple(
            file_sha256(path) for _digest, _source, path in selected
        ),
    )


def collect_candidate_dataset(
    raw_root: Path,
    year: int,
    paths: Sequence[Path],
) -> CandidateDataset:
    """Extract East-round post-riichi candidate rows from selected source files."""
    _validate_development_year(year)
    if not paths:
        raise ValueError("at least one source path is required")

    canonical_root = raw_root.resolve()
    rows: list[ComboCandidateRow] = []
    counts: Counter[str] = Counter()
    selected_sources: list[str] = []
    target_sources: list[str] = []
    seen_sources: set[str] = set()
    for path in paths:
        canonical_path = path.resolve()
        try:
            relative_source = canonical_path.relative_to(canonical_root).as_posix()
        except ValueError as error:
            raise ValueError(f"source path is outside raw_root: {path}") from error
        try:
            canonical_path.relative_to(canonical_root / str(year))
        except ValueError as error:
            raise ValueError(
                f"source path is not in year {year}: {relative_source}"
            ) from error
        if relative_source in seen_sources:
            raise ValueError(f"duplicate source path: {relative_source}")
        seen_sources.add(relative_source)
        selected_sources.append(relative_source)
        counts["scanned_files"] += 1
        events = load_mjai(canonical_path)
        if not is_target_game(canonical_path, events):
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
    return CandidateDataset(
        year=year,
        rows=tuple(rows),
        counts=dict(sorted(counts.items())),
        selected_sources=tuple(selected_sources),
        target_sources=tuple(target_sources),
    )


def expanding_year_splits(
    years: Sequence[int] = DEVELOPMENT_YEARS,
) -> tuple[tuple[tuple[int, ...], int], ...]:
    """Return strictly out-of-time expanding-window train/test year splits."""
    normalized = tuple(years)
    if len(normalized) < 2:
        raise ValueError("at least two years are required")
    if any(isinstance(year, bool) or not isinstance(year, int) for year in normalized):
        raise TypeError("years must contain integers")
    if tuple(sorted(set(normalized))) != normalized:
        raise ValueError("years must be strictly increasing and unique")
    return tuple(
        (normalized[:index], year) for index, year in enumerate(normalized[1:], 1)
    )


def selection_digest(relative_source: str, seed: int) -> str:
    """Return the cross-platform selection digest stored in sample manifests."""
    if not isinstance(relative_source, str) or not relative_source:
        raise ValueError("relative_source must be a non-empty string")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    payload = f"{_SELECTION_NAMESPACE}\0{seed}\0{relative_source}".encode()
    return sha256(payload).hexdigest()


def _population_sha256(relative_sources: Sequence[str]) -> str:
    digest = sha256()
    for source in sorted(relative_sources):
        digest.update(source.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """Hash source content in bounded chunks for manifest verification."""
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_development_year(year: int) -> None:
    if isinstance(year, bool) or not isinstance(year, int):
        raise TypeError("year must be an integer")
    if year not in DEVELOPMENT_YEARS:
        raise ValueError(
            f"year {year} is outside development years {DEVELOPMENT_YEARS}"
        )
