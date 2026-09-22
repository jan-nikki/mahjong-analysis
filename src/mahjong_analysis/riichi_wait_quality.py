"""Additional wait-quality metrics; canonical waits and old summaries are untouched."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

from mahjong_analysis.riichi_wait_dataset import (
    DatasetFixedMeld,
    DatasetWaitDetail,
    iter_dataset_records,
)
from mahjong_analysis.tiles import TILE_KINDS, tile_to_index, tiles_to_counts


@dataclass(frozen=True)
class WaitQualityFacts:
    contains_ryanmen: bool
    contains_nobetan: bool
    self_excluded_wait_copies: int

    @property
    def contains_suji(self) -> bool:
        return self.contains_ryanmen or self.contains_nobetan

    @property
    def is_good_wait(self) -> bool:
        return self.self_excluded_wait_copies >= 5


def evaluate_wait_quality(
    concealed_tiles: tuple[str, ...],
    fixed_melds: tuple[DatasetFixedMeld, ...],
    wait_details: tuple[DatasetWaitDetail, ...],
) -> WaitQualityFacts:
    """Evaluate validated formal details, counting each wait tile only once.

    Nobetan-family membership requires two standard/tanki interpretations on
    one suited suji line (rank difference 3 or 6). Shanpon and chiitoitsu tanki
    do not qualify. A zero-copy formal wait remains part of shape membership.
    """
    if len(fixed_melds) > 4 or len(concealed_tiles) != 13 - 3 * len(fixed_melds):
        raise ValueError("declaration-post concealed/fixed meld counts disagree")
    owned = tiles_to_counts(
        (*concealed_tiles, *(tile for meld in fixed_melds for tile in meld.tiles))
    )
    wait_indexes = {tile_to_index(detail.tile) for detail in wait_details}
    tanki_indexes = {
        tile_to_index(detail.tile)
        for detail in wait_details
        if detail.hand_type == "standard" and detail.wait_shape == "tanki"
    }
    nobetan = any(
        a < 27 and b < 27 and a // 9 == b // 9 and abs(a - b) in (3, 6)
        for a, b in combinations(tanki_indexes, 2)
    )
    return WaitQualityFacts(
        contains_ryanmen=any(
            detail.hand_type == "standard" and detail.wait_shape == "ryanmen"
            for detail in wait_details
        ),
        contains_nobetan=nobetan,
        self_excluded_wait_copies=sum(4 - owned[index] for index in wait_indexes),
    )


_ORIGINS = ("ryanmen_only", "nobetan_only", "both", "neither")
_CROSS = ("suji_and_good", "suji_only", "good_only", "neither")


@dataclass
class QualityAccumulator:
    record_count: int = 0
    origins: Counter[str] = field(default_factory=Counter)
    cross: Counter[str] = field(default_factory=Counter)
    copies: Counter[int] = field(default_factory=Counter)

    def add(self, facts: WaitQualityFacts) -> None:
        self.record_count += 1
        origin = (
            "both"
            if facts.contains_ryanmen and facts.contains_nobetan
            else "ryanmen_only"
            if facts.contains_ryanmen
            else "nobetan_only"
            if facts.contains_nobetan
            else "neither"
        )
        cross = (
            "suji_and_good"
            if facts.contains_suji and facts.is_good_wait
            else "suji_only"
            if facts.contains_suji
            else "good_only"
            if facts.is_good_wait
            else "neither"
        )
        self.origins[origin] += 1
        self.cross[cross] += 1
        self.copies[facts.self_excluded_wait_copies] += 1

    def merge(self, other: QualityAccumulator) -> None:
        self.record_count += other.record_count
        self.origins.update(other.origins)
        self.cross.update(other.cross)
        self.copies.update(other.copies)

    def validate(self) -> None:
        if any(
            sum(counter.values()) != self.record_count
            for counter in (self.origins, self.cross, self.copies)
        ):
            raise ValueError("quality distributions do not sum to record count")
        if set(self.origins) - set(_ORIGINS) or set(self.cross) - set(_CROSS):
            raise ValueError("unknown quality category")
        if any(k < 0 or k > 4 * len(TILE_KINDS) for k in self.copies):
            raise ValueError("invalid wait-copy count")
        suji = self.origins["ryanmen_only"] + self.origins["nobetan_only"]
        suji += self.origins["both"]
        if suji != self.cross["suji_and_good"] + self.cross["suji_only"]:
            raise ValueError("suji counts disagree")
        if sum(v for k, v in self.copies.items() if k >= 5) != (
            self.cross["suji_and_good"] + self.cross["good_only"]
        ):
            raise ValueError("good-wait counts disagree")

    def to_dict(self) -> dict[str, Any]:
        self.validate()

        def metric(count: int) -> dict[str, Any]:
            return {
                "count": count,
                "denominator": self.record_count,
                "rate": count / self.record_count if self.record_count else None,
            }

        return {
            "record_count": self.record_count,
            "contains_ryanmen": metric(
                self.origins["ryanmen_only"] + self.origins["both"]
            ),
            "contains_nobetan": metric(
                self.origins["nobetan_only"] + self.origins["both"]
            ),
            "contains_suji": metric(self.record_count - self.origins["neither"]),
            "good_wait": metric(self.cross["suji_and_good"] + self.cross["good_only"]),
            "suji_origin_counts": {k: self.origins[k] for k in _ORIGINS},
            "suji_good_cross_counts": {k: self.cross[k] for k in _CROSS},
            "wait_copy_distribution": [
                {"wait_copies": k, "count": v} for k, v in sorted(self.copies.items())
            ],
        }


@dataclass(frozen=True)
class AnnualQualityTask:
    year: int
    path: str
    expected_records: int


@dataclass
class AnnualQualityResult:
    year: int
    overall: QualityAccumulator
    by_turn: dict[int, QualityAccumulator]

    def validate(self) -> None:
        self.overall.validate()
        merged = QualityAccumulator()
        for accumulator in self.by_turn.values():
            accumulator.validate()
            merged.merge(accumulator)
        if merged != self.overall:
            raise ValueError("turn partitions disagree with annual total")


def aggregate_annual_quality(task: AnnualQualityTask) -> AnnualQualityResult:
    """One streaming DTO-validation pass; top-level for Windows spawn."""
    overall = QualityAccumulator()
    by_turn: dict[int, QualityAccumulator] = {}
    for line, record in enumerate(iter_dataset_records(Path(task.path)), 1):
        if record.year != task.year:
            raise ValueError(f"{task.path}:{line}: unexpected record year")
        facts = evaluate_wait_quality(
            record.concealed_tiles_after_discard,
            record.fixed_melds,
            record.wait_details,
        )
        if facts.contains_ryanmen != record.contains_ryanmen:
            raise ValueError(f"{task.path}:{line}: ryanmen interpretation mismatch")
        overall.add(facts)
        by_turn.setdefault(record.riichi_discard_number, QualityAccumulator()).add(
            facts
        )
    if overall.record_count != task.expected_records:
        raise ValueError(f"{task.path}: annual record count differs from manifest")
    result = AnnualQualityResult(task.year, overall, by_turn)
    result.validate()
    return result


def run_quality_tasks(
    tasks: tuple[AnnualQualityTask, ...],
    *,
    workers: int = 1,
    on_result: Callable[[AnnualQualityResult], None] | None = None,
) -> tuple[AnnualQualityResult, ...]:
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if not tasks or len({t.year for t in tasks}) != len(tasks):
        raise ValueError("tasks must have nonempty distinct years")
    results = {}
    if workers == 1:
        for task in tasks:
            result = aggregate_annual_quality(task)
            results[task.year] = result
            if on_result:
                on_result(result)
    else:
        executor = ProcessPoolExecutor(max_workers=min(workers, len(tasks)))
        futures = {}
        try:
            futures = {
                executor.submit(aggregate_annual_quality, task): task.year
                for task in tasks
            }
            for future in as_completed(futures):
                result = future.result()
                if result.year != futures[future]:
                    raise ValueError("worker returned the wrong year")
                results[result.year] = result
                if on_result:
                    on_result(result)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    return tuple(results[t.year] for t in tasks)


def quality_results_document(
    results: tuple[AnnualQualityResult, ...],
) -> dict[str, Any]:
    overall = QualityAccumulator()
    by_turn: dict[int, QualityAccumulator] = {}
    if len({r.year for r in results}) != len(results):
        raise ValueError("duplicate annual result")
    for result in results:
        result.validate()
        overall.merge(result.overall)
        for turn, accumulator in result.by_turn.items():
            by_turn.setdefault(turn, QualityAccumulator()).merge(accumulator)
    AnnualQualityResult(0, overall, by_turn).validate()

    def turns(values: dict[int, QualityAccumulator]) -> list[dict[str, Any]]:
        return [
            {"riichi_discard_number": turn, **acc.to_dict()}
            for turn, acc in sorted(values.items())
        ]

    return {
        "overall": overall.to_dict(),
        "by_turn": turns(by_turn),
        "years": [
            {"year": r.year, **r.overall.to_dict(), "by_turn": turns(r.by_turn)}
            for r in sorted(results, key=lambda r: r.year)
        ],
    }
