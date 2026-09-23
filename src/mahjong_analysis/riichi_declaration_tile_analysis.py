"""Wait strength by the tile discarded to declare an established riichi."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mahjong_analysis.hand_waits import WAIT_SHAPE_ORDER
from mahjong_analysis.riichi_wait_dataset import (
    DatasetWaitDetail,
    iter_dataset_records,
)
from mahjong_analysis.riichi_wait_quality import (
    QualityAccumulator,
    WaitQualityFacts,
    evaluate_wait_quality,
)
from mahjong_analysis.tiles import RED_FIVE_NORMALIZATION, normalize_tile

DECLARATION_CATEGORIES = ("19", "28", "37", "46", "5", "honor", "dora")
NON_DORA_SUITED_CATEGORIES = ("19", "28", "37", "46", "5")
DECLARATION_RANKS = tuple(range(1, 10))
SUJI_WAIT_RANKS = {
    1: (4, 7),
    2: (5, 8),
    3: (6, 9),
    4: (1, 7),
    5: (2, 8),
    6: (3, 9),
    7: (1, 4),
    8: (2, 5),
    9: (3, 6),
}
RANK_WAIT_PAIRS = tuple(
    (declaration_rank, wait_rank)
    for declaration_rank in DECLARATION_RANKS
    for wait_rank in SUJI_WAIT_RANKS[declaration_rank]
)
CATEGORY_RANKS = {
    "19": (1, 9),
    "28": (2, 8),
    "37": (3, 7),
    "46": (4, 6),
    "5": (5,),
}
WEAK_WAIT_SHAPES = frozenset(("tanki", "kanchan", "penchan", "shanpon"))
PAIR_WEAK_WAIT_SHAPES = ("kanchan", "penchan", "shanpon", "tanki")

_DORA_SUCCESSOR = {
    "E": "S",
    "S": "W",
    "W": "N",
    "N": "E",
    "P": "F",
    "F": "C",
    "C": "P",
}


def dora_from_marker(marker: str) -> str:
    """Return the normalized visible dora kind for one marker."""
    normalized = normalize_tile(marker)
    if normalized[-1:] in {"m", "p", "s"}:
        rank = int(normalized[0])
        return f"{1 if rank == 9 else rank + 1}{normalized[1]}"
    return _DORA_SUCCESSOR[normalized]


def classify_declaration_tile(
    raw_tile: str,
    tile_kind: str,
    dora_marker: str,
) -> str:
    """Classify one declaration discard, with red/visible dora overriding rank."""
    normalized = normalize_tile(raw_tile)
    if normalized != tile_kind:
        raise ValueError("raw and normalized declaration tiles disagree")
    if raw_tile in RED_FIVE_NORMALIZATION or normalized == dora_from_marker(
        dora_marker
    ):
        return "dora"
    if normalized[-1:] not in {"m", "p", "s"}:
        return "honor"
    rank = int(normalized[0])
    if rank in (1, 9):
        return "19"
    if rank in (2, 8):
        return "28"
    if rank in (3, 7):
        return "37"
    if rank in (4, 6):
        return "46"
    return "5"


@dataclass(frozen=True)
class WaitStrengthFacts:
    contains_suji: bool
    contains_ryanmen: bool
    five_plus_wait: bool
    is_strong_wait: bool
    is_weak_wait: bool
    self_excluded_wait_copies: int


def evaluate_wait_strength(
    quality: WaitQualityFacts,
    wait_details: tuple[DatasetWaitDetail, ...],
) -> WaitStrengthFacts:
    """Apply the analysis-specific strong/weak definitions to canonical waits."""
    shapes = frozenset(detail.wait_shape for detail in wait_details)
    strong = quality.contains_suji or quality.is_good_wait
    weak = (
        not strong
        and quality.self_excluded_wait_copies <= 4
        and bool(shapes)
        and shapes <= WEAK_WAIT_SHAPES
    )
    return WaitStrengthFacts(
        contains_suji=quality.contains_suji,
        contains_ryanmen=quality.contains_ryanmen,
        five_plus_wait=quality.is_good_wait,
        is_strong_wait=strong,
        is_weak_wait=weak,
        self_excluded_wait_copies=quality.self_excluded_wait_copies,
    )


def same_suji_line_wait_rank_shapes(
    declaration_tile_kind: str,
    wait_details: tuple[DatasetWaitDetail, ...],
) -> dict[int, frozenset[str]]:
    """Return matching wait-rank shape flags for the declaration tile's suji line."""
    declaration = normalize_tile(declaration_tile_kind)
    if declaration[-1:] not in {"m", "p", "s"}:
        return {}
    rank = int(declaration[0])
    suit = declaration[1]
    shapes: dict[int, set[str]] = {
        wait_rank: set() for wait_rank in SUJI_WAIT_RANKS[rank]
    }
    for detail in wait_details:
        if detail.tile[-1:] != suit:
            continue
        wait_rank = int(detail.tile[0])
        if wait_rank in shapes:
            shapes[wait_rank].add(detail.wait_shape)
    return {
        wait_rank: frozenset(shapes[wait_rank])
        for wait_rank in SUJI_WAIT_RANKS[rank]
        if shapes[wait_rank]
    }


def same_suji_line_wait_shapes(
    declaration_tile_kind: str,
    wait_details: tuple[DatasetWaitDetail, ...],
) -> frozenset[str]:
    """Return non-exclusive shapes on waits in the declaration tile's suji line."""
    return frozenset(
        shape
        for shapes in same_suji_line_wait_rank_shapes(
            declaration_tile_kind, wait_details
        ).values()
        for shape in shapes
    )


@dataclass
class StrengthAccumulator:
    quality: QualityAccumulator = field(default_factory=QualityAccumulator)
    strength_counts: Counter[str] = field(default_factory=Counter)
    wait_shape_counts: Counter[str] = field(default_factory=Counter)
    same_suji_line_wait_count: int = 0
    weak_and_same_suji_line_count: int = 0
    weak_and_not_same_suji_line_count: int = 0
    weak_same_suji_line_shape_counts: Counter[str] = field(default_factory=Counter)

    @property
    def record_count(self) -> int:
        return self.quality.record_count

    def add(
        self,
        quality: WaitQualityFacts,
        wait_details: tuple[DatasetWaitDetail, ...],
        same_suji_shapes: frozenset[str],
    ) -> None:
        facts = evaluate_wait_strength(quality, wait_details)
        self.quality.add(quality)
        record_shapes = {detail.wait_shape for detail in wait_details}
        for shape in record_shapes:
            self.wait_shape_counts[shape] += 1
        category = (
            "strong"
            if facts.is_strong_wait
            else "weak"
            if facts.is_weak_wait
            else "other"
        )
        self.strength_counts[category] += 1
        if same_suji_shapes:
            self.same_suji_line_wait_count += 1
        if facts.is_weak_wait:
            if same_suji_shapes:
                self.weak_and_same_suji_line_count += 1
                self.weak_same_suji_line_shape_counts.update(record_shapes)
            else:
                self.weak_and_not_same_suji_line_count += 1

    def merge(self, other: StrengthAccumulator) -> None:
        self.quality.merge(other.quality)
        self.strength_counts.update(other.strength_counts)
        self.wait_shape_counts.update(other.wait_shape_counts)
        self.same_suji_line_wait_count += other.same_suji_line_wait_count
        self.weak_and_same_suji_line_count += other.weak_and_same_suji_line_count
        self.weak_and_not_same_suji_line_count += (
            other.weak_and_not_same_suji_line_count
        )
        self.weak_same_suji_line_shape_counts.update(
            other.weak_same_suji_line_shape_counts
        )

    def validate(self) -> None:
        self.quality.validate()
        if set(self.strength_counts) - {"strong", "weak", "other"}:
            raise ValueError("unknown wait-strength category")
        if sum(self.strength_counts.values()) != self.record_count:
            raise ValueError("wait-strength counts do not sum to record count")
        expected_strong = self.quality.cross["suji_and_good"]
        expected_strong += self.quality.cross["suji_only"]
        expected_strong += self.quality.cross["good_only"]
        if self.strength_counts["strong"] != expected_strong:
            raise ValueError("strong-wait count disagrees with canonical quality")
        if set(self.wait_shape_counts) - set(WAIT_SHAPE_ORDER):
            raise ValueError("unknown wait shape")
        if any(
            count < 0 or count > self.record_count
            for count in self.wait_shape_counts.values()
        ):
            raise ValueError("invalid wait-shape record count")
        if not (
            0
            <= self.weak_and_same_suji_line_count
            <= self.same_suji_line_wait_count
            <= self.record_count
        ):
            raise ValueError("invalid same-suji-line counts")
        if (
            self.weak_and_same_suji_line_count + self.weak_and_not_same_suji_line_count
            != self.strength_counts["weak"]
        ):
            raise ValueError("same-suji-line weak counts do not partition weak waits")
        if set(self.weak_same_suji_line_shape_counts) - set(WAIT_SHAPE_ORDER):
            raise ValueError("unknown same-suji-line wait shape")
        if any(
            count < 0 or count > self.weak_and_same_suji_line_count
            for count in self.weak_same_suji_line_shape_counts.values()
        ):
            raise ValueError("invalid same-suji-line shape count")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = self.quality.to_dict()

        def metric(count: int, denominator: int | None = None) -> dict[str, Any]:
            if denominator is None:
                denominator = self.record_count
            return {
                "count": count,
                "denominator": denominator,
                "rate": count / denominator if denominator else None,
            }

        weak_count = self.strength_counts["weak"]
        not_same_count = self.record_count - self.same_suji_line_wait_count
        result.update(
            {
                "strong_wait": metric(self.strength_counts["strong"]),
                "weak_wait": metric(weak_count),
                "other_wait": metric(self.strength_counts["other"]),
                "wait_shape_record_counts": {
                    shape: self.wait_shape_counts[shape] for shape in WAIT_SHAPE_ORDER
                },
                "same_suji_line_wait": metric(self.same_suji_line_wait_count),
                "weak_and_same_suji_line": metric(self.weak_and_same_suji_line_count),
                "weak_and_not_same_suji_line": metric(
                    self.weak_and_not_same_suji_line_count
                ),
                "same_suji_line_share_within_weak_wait": metric(
                    self.weak_and_same_suji_line_count,
                    weak_count,
                ),
                "weak_wait_rate_within_same_suji_line": metric(
                    self.weak_and_same_suji_line_count,
                    self.same_suji_line_wait_count,
                ),
                "weak_wait_rate_excluding_same_suji_line": metric(
                    self.weak_and_not_same_suji_line_count,
                    not_same_count,
                ),
                "weak_same_suji_line_wait_shape_record_counts": {
                    "exclusive": False,
                    "counts": {
                        shape: self.weak_same_suji_line_shape_counts[shape]
                        for shape in WAIT_SHAPE_ORDER
                    },
                },
            }
        )
        return result


@dataclass
class RankWaitPairAccumulator:
    record_count: int = 0
    strength_counts: Counter[str] = field(default_factory=Counter)
    weak_shape_counts: Counter[str] = field(default_factory=Counter)
    wait_copy_counts: Counter[int] = field(default_factory=Counter)

    def add(
        self,
        quality: WaitQualityFacts,
        wait_details: tuple[DatasetWaitDetail, ...],
        matching_wait_shapes: frozenset[str],
    ) -> None:
        if not matching_wait_shapes:
            raise ValueError("rank-wait pair requires at least one matching wait")
        facts = evaluate_wait_strength(quality, wait_details)
        self.record_count += 1
        category = (
            "strong"
            if facts.is_strong_wait
            else "weak"
            if facts.is_weak_wait
            else "other"
        )
        self.strength_counts[category] += 1
        self.wait_copy_counts[facts.self_excluded_wait_copies] += 1
        if facts.is_weak_wait:
            self.weak_shape_counts.update(matching_wait_shapes)

    def merge(self, other: RankWaitPairAccumulator) -> None:
        self.record_count += other.record_count
        self.strength_counts.update(other.strength_counts)
        self.weak_shape_counts.update(other.weak_shape_counts)
        self.wait_copy_counts.update(other.wait_copy_counts)

    def validate(self, declaration_rank_count: int) -> None:
        if self.record_count < 0 or self.record_count > declaration_rank_count:
            raise ValueError("invalid rank-wait pair record count")
        if set(self.strength_counts) - {"strong", "weak", "other"}:
            raise ValueError("unknown rank-wait pair strength category")
        if sum(self.strength_counts.values()) != self.record_count:
            raise ValueError("rank-wait pair strengths do not sum to record count")
        if set(self.weak_shape_counts) - set(PAIR_WEAK_WAIT_SHAPES):
            raise ValueError("unknown weak rank-wait pair shape")
        if any(
            count < 0 or count > self.strength_counts["weak"]
            for count in self.weak_shape_counts.values()
        ):
            raise ValueError("invalid weak rank-wait pair shape count")
        if any(
            copies < 0 or count < 0 for copies, count in self.wait_copy_counts.items()
        ):
            raise ValueError("invalid rank-wait pair copy distribution")
        if sum(self.wait_copy_counts.values()) != self.record_count:
            raise ValueError("rank-wait pair copies do not sum to record count")

    def to_dict(
        self,
        declaration_rank: int,
        wait_rank: int,
        declaration_rank_count: int,
    ) -> dict[str, Any]:
        self.validate(declaration_rank_count)

        def metric(count: int, denominator: int) -> dict[str, Any]:
            return {
                "count": count,
                "denominator": denominator,
                "rate": count / denominator if denominator else None,
            }

        return {
            "declaration_rank": declaration_rank,
            "wait_rank": wait_rank,
            "record_count": self.record_count,
            "share_within_declaration_rank": metric(
                self.record_count, declaration_rank_count
            ),
            "strong_wait": metric(self.strength_counts["strong"], self.record_count),
            "weak_wait": metric(self.strength_counts["weak"], self.record_count),
            "other_wait": metric(self.strength_counts["other"], self.record_count),
            "weak_wait_shape_record_counts": {
                "exclusive": False,
                "scope": "matching wait-rank interpretations within weak records",
                "counts": {
                    shape: self.weak_shape_counts[shape]
                    for shape in PAIR_WEAK_WAIT_SHAPES
                },
            },
            "wait_copy_distribution": [
                {"wait_copies": copies, "count": self.wait_copy_counts[copies]}
                for copies in sorted(self.wait_copy_counts)
            ],
        }


@dataclass
class DeclarationTileAccumulator:
    all_records: StrengthAccumulator = field(default_factory=StrengthAccumulator)
    by_category: dict[str, StrengthAccumulator] = field(
        default_factory=lambda: {
            category: StrengthAccumulator() for category in DECLARATION_CATEGORIES
        }
    )
    by_non_dora_suited_rank: dict[int, StrengthAccumulator] = field(
        default_factory=lambda: {
            rank: StrengthAccumulator() for rank in DECLARATION_RANKS
        }
    )
    by_rank_wait_pair: dict[tuple[int, int], RankWaitPairAccumulator] = field(
        default_factory=lambda: {
            pair: RankWaitPairAccumulator() for pair in RANK_WAIT_PAIRS
        }
    )

    @property
    def record_count(self) -> int:
        return self.all_records.record_count

    def add(
        self,
        declaration_category: str,
        declaration_tile_kind: str,
        quality: WaitQualityFacts,
        wait_details: tuple[DatasetWaitDetail, ...],
    ) -> None:
        if declaration_category not in self.by_category:
            raise ValueError("unknown declaration category")
        wait_rank_shapes = same_suji_line_wait_rank_shapes(
            declaration_tile_kind, wait_details
        )
        suji_shapes = frozenset(
            shape for shapes in wait_rank_shapes.values() for shape in shapes
        )
        self.all_records.add(quality, wait_details, suji_shapes)
        self.by_category[declaration_category].add(quality, wait_details, suji_shapes)
        if declaration_category in NON_DORA_SUITED_CATEGORIES:
            declaration_rank = int(declaration_tile_kind[0])
            self.by_non_dora_suited_rank[declaration_rank].add(
                quality, wait_details, suji_shapes
            )
            for wait_rank, matching_shapes in wait_rank_shapes.items():
                self.by_rank_wait_pair[(declaration_rank, wait_rank)].add(
                    quality, wait_details, matching_shapes
                )

    def merge(self, other: DeclarationTileAccumulator) -> None:
        self.all_records.merge(other.all_records)
        for category in DECLARATION_CATEGORIES:
            self.by_category[category].merge(other.by_category[category])
        for rank in DECLARATION_RANKS:
            self.by_non_dora_suited_rank[rank].merge(
                other.by_non_dora_suited_rank[rank]
            )
        for pair in RANK_WAIT_PAIRS:
            self.by_rank_wait_pair[pair].merge(other.by_rank_wait_pair[pair])

    def validate(self) -> None:
        if tuple(self.by_category) != DECLARATION_CATEGORIES:
            raise ValueError("declaration category keys or order differ")
        if tuple(self.by_non_dora_suited_rank) != DECLARATION_RANKS:
            raise ValueError("declaration rank keys or order differ")
        if tuple(self.by_rank_wait_pair) != RANK_WAIT_PAIRS:
            raise ValueError("rank-wait pair keys or order differ")
        self.all_records.validate()
        merged = StrengthAccumulator()
        for accumulator in self.by_category.values():
            accumulator.validate()
            merged.merge(accumulator)
        if merged != self.all_records:
            raise ValueError("declaration categories do not partition all records")
        for rank, accumulator in self.by_non_dora_suited_rank.items():
            accumulator.validate()
            for wait_rank in SUJI_WAIT_RANKS[rank]:
                self.by_rank_wait_pair[(rank, wait_rank)].validate(
                    accumulator.record_count
                )
        for category, ranks in CATEGORY_RANKS.items():
            rank_merged = StrengthAccumulator()
            for rank in ranks:
                rank_merged.merge(self.by_non_dora_suited_rank[rank])
            if rank_merged != self.by_category[category]:
                raise ValueError(
                    f"declaration ranks do not reconstruct category {category}"
                )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        non_dora = StrengthAccumulator()
        for category in DECLARATION_CATEGORIES:
            if category != "dora":
                non_dora.merge(self.by_category[category])
        suited_record_count = sum(
            accumulator.record_count
            for accumulator in self.by_non_dora_suited_rank.values()
        )
        return {
            "record_count": self.record_count,
            "all": self.all_records.to_dict(),
            "non_dora": non_dora.to_dict(),
            "categories": [
                {
                    "declaration_category": category,
                    **self.by_category[category].to_dict(),
                }
                for category in DECLARATION_CATEGORIES
            ],
            "non_dora_suited_rank_breakdown": {
                "record_count": suited_record_count,
                "ranks_exclusive": True,
                "ranks": [
                    {
                        "declaration_rank": rank,
                        **self.by_non_dora_suited_rank[rank].to_dict(),
                    }
                    for rank in DECLARATION_RANKS
                ],
                "rank_wait_pairs": {
                    "exclusive": False,
                    "definition": (
                        "same-suit formal wait rank with declaration/wait rank "
                        "difference 3 or 6; one record may appear in both pairs"
                    ),
                    "pairs": [
                        self.by_rank_wait_pair[(declaration_rank, wait_rank)].to_dict(
                            declaration_rank,
                            wait_rank,
                            self.by_non_dora_suited_rank[declaration_rank].record_count,
                        )
                        for declaration_rank, wait_rank in RANK_WAIT_PAIRS
                    ],
                },
            },
        }


@dataclass(frozen=True)
class AnnualDeclarationTileTask:
    year: int
    path: str
    expected_records: int


@dataclass(frozen=True)
class DeclarationAuditSample:
    relative_source_path: str
    start_kyoku_line: int
    reach_line: int
    declaration_dahai_line: int
    actor: int
    kyoku: int
    riichi_discard_number: int
    raw_tile: str
    tile_kind: str
    dora_marker: str
    declaration_category: str
    wait_tiles: tuple[str, ...]
    self_excluded_wait_copies: int
    is_strong_wait: bool
    is_weak_wait: bool
    same_suji_line_wait: bool

    @property
    def stable_key(self) -> tuple[str, int, int]:
        return self.relative_source_path, self.start_kyoku_line, self.reach_line

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_source_path": self.relative_source_path,
            "start_kyoku_line": self.start_kyoku_line,
            "reach_line": self.reach_line,
            "declaration_dahai_line": self.declaration_dahai_line,
            "actor": self.actor,
            "kyoku": self.kyoku,
            "riichi_discard_number": self.riichi_discard_number,
            "raw_tile": self.raw_tile,
            "tile_kind": self.tile_kind,
            "dora_marker": self.dora_marker,
            "declaration_category": self.declaration_category,
            "wait_tiles": list(self.wait_tiles),
            "self_excluded_wait_copies": self.self_excluded_wait_copies,
            "is_strong_wait": self.is_strong_wait,
            "is_weak_wait": self.is_weak_wait,
            "same_suji_line_wait": self.same_suji_line_wait,
        }


@dataclass
class AnnualDeclarationTileResult:
    year: int
    overall: DeclarationTileAccumulator
    by_turn: dict[int, DeclarationTileAccumulator]
    audit_by_category: dict[str, tuple[DeclarationAuditSample, ...]]

    def validate(self) -> None:
        self.overall.validate()
        merged = DeclarationTileAccumulator()
        for turn, accumulator in self.by_turn.items():
            if type(turn) is not int or turn < 1:
                raise ValueError("turn keys must be positive integers")
            accumulator.validate()
            merged.merge(accumulator)
        if merged != self.overall:
            raise ValueError("turn partitions disagree with annual total")
        if tuple(self.audit_by_category) != DECLARATION_CATEGORIES:
            raise ValueError("audit category keys or order differ")
        for category, samples in self.audit_by_category.items():
            if (
                len(samples) > 3
                or tuple(sorted(samples, key=lambda sample: sample.stable_key))
                != samples
            ):
                raise ValueError("audit samples must be sorted and capped")
            if any(sample.declaration_category != category for sample in samples):
                raise ValueError("audit sample category disagrees")


def aggregate_annual_declaration_tiles(
    task: AnnualDeclarationTileTask,
) -> AnnualDeclarationTileResult:
    """Stream and aggregate one annual canonical dataset file."""
    overall = DeclarationTileAccumulator()
    by_turn: dict[int, DeclarationTileAccumulator] = {}
    audit: dict[str, list[DeclarationAuditSample]] = {
        category: [] for category in DECLARATION_CATEGORIES
    }
    for line, record in enumerate(iter_dataset_records(Path(task.path)), 1):
        if record.year != task.year:
            raise ValueError(f"{task.path}:{line}: unexpected record year")
        quality = evaluate_wait_quality(
            record.concealed_tiles_after_discard,
            record.fixed_melds,
            record.wait_details,
        )
        if quality.contains_ryanmen != record.contains_ryanmen:
            raise ValueError(f"{task.path}:{line}: ryanmen interpretation mismatch")
        category = classify_declaration_tile(
            record.riichi_declaration_tile,
            record.riichi_declaration_tile_kind,
            record.dora_marker,
        )
        overall.add(
            category,
            record.riichi_declaration_tile_kind,
            quality,
            record.wait_details,
        )
        by_turn.setdefault(
            record.riichi_discard_number, DeclarationTileAccumulator()
        ).add(
            category,
            record.riichi_declaration_tile_kind,
            quality,
            record.wait_details,
        )
        if len(audit[category]) < 3:
            strength = evaluate_wait_strength(quality, record.wait_details)
            audit[category].append(
                DeclarationAuditSample(
                    relative_source_path=record.relative_source_path,
                    start_kyoku_line=record.start_kyoku_line,
                    reach_line=record.reach_line,
                    declaration_dahai_line=record.declaration_dahai_line,
                    actor=record.actor,
                    kyoku=record.kyoku,
                    riichi_discard_number=record.riichi_discard_number,
                    raw_tile=record.riichi_declaration_tile,
                    tile_kind=record.riichi_declaration_tile_kind,
                    dora_marker=record.dora_marker,
                    declaration_category=category,
                    wait_tiles=record.wait_tiles,
                    self_excluded_wait_copies=strength.self_excluded_wait_copies,
                    is_strong_wait=strength.is_strong_wait,
                    is_weak_wait=strength.is_weak_wait,
                    same_suji_line_wait=bool(
                        same_suji_line_wait_shapes(
                            record.riichi_declaration_tile_kind,
                            record.wait_details,
                        )
                    ),
                )
            )
    if overall.record_count != task.expected_records:
        raise ValueError(f"{task.path}: annual record count differs from manifest")
    result = AnnualDeclarationTileResult(
        task.year,
        overall,
        by_turn,
        {category: tuple(samples) for category, samples in audit.items()},
    )
    result.validate()
    return result


def run_declaration_tile_tasks(
    tasks: tuple[AnnualDeclarationTileTask, ...],
    *,
    workers: int = 1,
    on_result: Callable[[AnnualDeclarationTileResult], None] | None = None,
) -> tuple[AnnualDeclarationTileResult, ...]:
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if not tasks or len({task.year for task in tasks}) != len(tasks):
        raise ValueError("tasks must have nonempty distinct years")
    results: dict[int, AnnualDeclarationTileResult] = {}
    if workers == 1:
        for task in tasks:
            result = aggregate_annual_declaration_tiles(task)
            results[result.year] = result
            if on_result:
                on_result(result)
    else:
        executor = ProcessPoolExecutor(max_workers=min(workers, len(tasks)))
        futures = {}
        try:
            futures = {
                executor.submit(aggregate_annual_declaration_tiles, task): task.year
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
    return tuple(results[task.year] for task in tasks)


def _merge_results(
    results: Iterable[AnnualDeclarationTileResult],
) -> tuple[DeclarationTileAccumulator, dict[int, DeclarationTileAccumulator]]:
    overall = DeclarationTileAccumulator()
    by_turn: dict[int, DeclarationTileAccumulator] = {}
    for result in results:
        result.validate()
        overall.merge(result.overall)
        for turn, accumulator in result.by_turn.items():
            by_turn.setdefault(turn, DeclarationTileAccumulator()).merge(accumulator)
    AnnualDeclarationTileResult(
        0,
        overall,
        by_turn,
        {category: () for category in DECLARATION_CATEGORIES},
    ).validate()
    return overall, by_turn


def _block(results: tuple[AnnualDeclarationTileResult, ...]) -> dict[str, Any]:
    overall, by_turn = _merge_results(results)
    return {
        "overall": overall.to_dict(),
        "by_turn": [
            {"riichi_discard_number": turn, **accumulator.to_dict()}
            for turn, accumulator in sorted(by_turn.items())
        ],
    }


def declaration_tile_results_document(
    results: tuple[AnnualDeclarationTileResult, ...],
) -> dict[str, Any]:
    if not results or len({result.year for result in results}) != len(results):
        raise ValueError("results must have nonempty distinct years")
    ordered = tuple(sorted(results, key=lambda result: result.year))
    document = _block(ordered)
    document["years"] = [
        {"year": result.year, **_block((result,))} for result in ordered
    ]
    document["audit"] = {
        "samples_per_category": 3,
        "categories": [
            {
                "declaration_category": category,
                "samples": [
                    sample.to_dict()
                    for sample in sorted(
                        (
                            sample
                            for result in ordered
                            for sample in result.audit_by_category[category]
                        ),
                        key=lambda sample: sample.stable_key,
                    )[:3]
                ],
            }
            for category in DECLARATION_CATEGORIES
        ],
    }
    return document
