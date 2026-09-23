"""Record-level analysis of riichi after a pair-drop-looking river pattern."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

from mahjong_analysis.riichi_wait_dataset import (
    DatasetActorDiscard,
    RiichiWaitDatasetRecord,
    iter_dataset_records,
)
from mahjong_analysis.riichi_wait_quality import (
    QualityAccumulator,
    WaitQualityFacts,
    evaluate_wait_quality,
)

CLASSIFICATIONS = ("none", "a_only", "b_only", "both")
PATTERN_CLASSIFICATIONS = ("a_only", "b_only", "both")
AUDIT_REASONS = (
    "called_before_reach_accepted",
    "none",
    "a_only",
    "b_only",
    "both",
    "representative_pair_ends_on_declaration",
    "red_five_pair",
    "multiple_matching_pairs",
)


@dataclass(frozen=True)
class ToitsuDropPair:
    """One adjacent same-kind discard pair whose second tile was tedashi."""

    pair_type: str
    normalized_tile: str
    first_discard_number: int
    second_discard_number: int
    first_raw_tile: str
    second_raw_tile: str
    first_tsumogiri: bool
    first_event_index: int
    second_event_index: int

    def __post_init__(self) -> None:
        if self.pair_type not in {"A", "B"}:
            raise ValueError("pair_type must be A or B")
        if self.second_discard_number != self.first_discard_number + 1:
            raise ValueError("matching pair discard numbers must be adjacent")
        if self.first_event_index >= self.second_event_index:
            raise ValueError("matching pair event indexes must be increasing")
        if (self.pair_type == "B") != self.first_tsumogiri:
            raise ValueError("pair_type must agree with first tsumogiri")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_type": self.pair_type,
            "normalized_tile": self.normalized_tile,
            "first_discard_number": self.first_discard_number,
            "second_discard_number": self.second_discard_number,
            "first_raw_tile": self.first_raw_tile,
            "second_raw_tile": self.second_raw_tile,
            "first_tsumogiri": self.first_tsumogiri,
            "first_event_index": self.first_event_index,
            "second_event_index": self.second_event_index,
        }


@dataclass(frozen=True)
class ToitsuDropFacts:
    """All matching adjacent pairs and the exclusive record classification."""

    classification: str
    pairs: tuple[ToitsuDropPair, ...]
    riichi_discard_number: int

    def __post_init__(self) -> None:
        if self.classification not in CLASSIFICATIONS:
            raise ValueError("unknown toitsu-drop classification")
        pair_types = {pair.pair_type for pair in self.pairs}
        expected = (
            "both"
            if pair_types == {"A", "B"}
            else "a_only"
            if pair_types == {"A"}
            else "b_only"
            if pair_types == {"B"}
            else "none"
        )
        if self.classification != expected:
            raise ValueError("classification does not agree with matching pairs")
        if self.riichi_discard_number < 1:
            raise ValueError("riichi_discard_number must be positive")
        if any(
            left.second_discard_number >= right.second_discard_number
            for left, right in pairwise(self.pairs)
        ):
            raise ValueError("matching pairs must be in completion order")
        if self.pairs and self.pairs[-1].second_discard_number > (
            self.riichi_discard_number
        ):
            raise ValueError("matching pair cannot occur after declaration")

    @property
    def has_pattern(self) -> bool:
        return bool(self.pairs)

    @property
    def representative_pair(self) -> ToitsuDropPair | None:
        return self.pairs[-1] if self.pairs else None

    @property
    def distance(self) -> int | None:
        representative = self.representative_pair
        if representative is None:
            return None
        return self.riichi_discard_number - representative.second_discard_number


def detect_toitsu_drop(
    discards: tuple[DatasetActorDiscard, ...],
) -> ToitsuDropFacts:
    """Classify one validated actor river without using call or wait metadata."""
    if not isinstance(discards, tuple) or not discards:
        raise ValueError("actor river must be a nonempty tuple")
    if tuple(discard.discard_number for discard in discards) != tuple(
        range(1, len(discards) + 1)
    ):
        raise ValueError("actor river discard numbers must be consecutive")
    if any(left.event_index >= right.event_index for left, right in pairwise(discards)):
        raise ValueError("actor river event indexes must be strictly increasing")
    if any(discard.is_riichi_declaration for discard in discards[:-1]) or not (
        discards[-1].is_riichi_declaration
    ):
        raise ValueError("only the final discard may declare riichi")

    matches: list[ToitsuDropPair] = []
    for first, second in pairwise(discards):
        if first.normalized_tile != second.normalized_tile or second.tsumogiri:
            continue
        matches.append(
            ToitsuDropPair(
                pair_type="B" if first.tsumogiri else "A",
                normalized_tile=first.normalized_tile,
                first_discard_number=first.discard_number,
                second_discard_number=second.discard_number,
                first_raw_tile=first.tile,
                second_raw_tile=second.tile,
                first_tsumogiri=first.tsumogiri,
                first_event_index=first.event_index,
                second_event_index=second.event_index,
            )
        )
    pair_types = {match.pair_type for match in matches}
    classification = (
        "both"
        if pair_types == {"A", "B"}
        else "a_only"
        if pair_types == {"A"}
        else "b_only"
        if pair_types == {"B"}
        else "none"
    )
    return ToitsuDropFacts(classification, tuple(matches), len(discards))


@dataclass
class ToitsuDropAccumulator:
    """Exclusive record partitions with shared wait-quality aggregation."""

    by_classification: dict[str, QualityAccumulator] = field(
        default_factory=lambda: {name: QualityAccumulator() for name in CLASSIFICATIONS}
    )
    pair_counts: Counter[int] = field(default_factory=Counter)

    def add(self, pattern: ToitsuDropFacts, quality: WaitQualityFacts) -> None:
        self.by_classification[pattern.classification].add(quality)
        self.pair_counts[len(pattern.pairs)] += 1

    def merge(self, other: ToitsuDropAccumulator) -> None:
        for name in CLASSIFICATIONS:
            self.by_classification[name].merge(other.by_classification[name])
        self.pair_counts.update(other.pair_counts)

    @property
    def record_count(self) -> int:
        return sum(
            accumulator.record_count for accumulator in self.by_classification.values()
        )

    def quality_for(self, names: Iterable[str]) -> QualityAccumulator:
        result = QualityAccumulator()
        for name in names:
            if name not in CLASSIFICATIONS:
                raise ValueError(f"unknown classification: {name}")
            result.merge(self.by_classification[name])
        return result

    def validate(self) -> None:
        if set(self.by_classification) != set(CLASSIFICATIONS):
            raise ValueError("classification accumulator keys differ")
        for accumulator in self.by_classification.values():
            accumulator.validate()
        if any(type(count) is not int or count < 0 for count in self.pair_counts):
            raise ValueError("matching-pair count must be a non-negative integer")
        if sum(self.pair_counts.values()) != self.record_count:
            raise ValueError("pair-count distribution does not sum to records")
        if self.pair_counts[0] != self.by_classification["none"].record_count:
            raise ValueError("zero-pair count must equal none classification")
        if sum(count for pairs, count in self.pair_counts.items() if pairs > 0) != (
            self.quality_for(PATTERN_CLASSIFICATIONS).record_count
        ):
            raise ValueError("positive-pair count must equal pattern-present records")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        all_quality = self.quality_for(CLASSIFICATIONS)
        pattern_quality = self.quality_for(PATTERN_CLASSIFICATIONS)
        no_pattern_quality = self.by_classification["none"]

        def record_metric(count: int) -> dict[str, Any]:
            return {
                "count": count,
                "denominator": self.record_count,
                "rate": count / self.record_count if self.record_count else None,
            }

        comparisons: dict[str, float | None] = {}
        for metric in ("contains_ryanmen", "contains_suji", "good_wait"):
            pattern_rate = pattern_quality.to_dict()[metric]["rate"]
            no_pattern_rate = no_pattern_quality.to_dict()[metric]["rate"]
            comparisons[metric] = (
                None
                if pattern_rate is None or no_pattern_rate is None
                else (pattern_rate - no_pattern_rate) * 100
            )

        return {
            "record_count": self.record_count,
            "pattern_present": record_metric(pattern_quality.record_count),
            "classification_counts": {
                name: self.by_classification[name].record_count
                for name in CLASSIFICATIONS
            },
            "matching_pair_count_distribution": [
                {"matching_pair_count": pairs, "record_count": count}
                for pairs, count in sorted(self.pair_counts.items())
            ],
            "groups": {
                "all": all_quality.to_dict(),
                "pattern_present": pattern_quality.to_dict(),
                "no_pattern": no_pattern_quality.to_dict(),
                "a_only": self.by_classification["a_only"].to_dict(),
                "b_only": self.by_classification["b_only"].to_dict(),
                "both": self.by_classification["both"].to_dict(),
            },
            "pattern_minus_no_pattern_percentage_points": comparisons,
        }


@dataclass(frozen=True)
class AuditSample:
    relative_source_path: str
    start_kyoku_line: int
    reach_line: int
    actor: int
    kyoku: int
    riichi_discard_number: int
    classification: str
    matching_pairs: tuple[ToitsuDropPair, ...]
    representative_distance: int | None
    called_before_acceptance_discard_numbers: tuple[int, ...]
    reasons: tuple[str, ...]
    contains_suji: bool
    is_good_wait: bool
    self_excluded_wait_copies: int

    @property
    def stable_key(self) -> tuple[str, int, int]:
        return self.relative_source_path, self.start_kyoku_line, self.reach_line

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_source_path": self.relative_source_path,
            "start_kyoku_line": self.start_kyoku_line,
            "reach_line": self.reach_line,
            "actor": self.actor,
            "kyoku": self.kyoku,
            "riichi_discard_number": self.riichi_discard_number,
            "classification": self.classification,
            "matching_pairs": [pair.to_dict() for pair in self.matching_pairs],
            "representative_distance": self.representative_distance,
            "called_before_acceptance_discard_numbers": list(
                self.called_before_acceptance_discard_numbers
            ),
            "selection_reasons": list(self.reasons),
            "contains_suji": self.contains_suji,
            "is_good_wait": self.is_good_wait,
            "self_excluded_wait_copies": self.self_excluded_wait_copies,
        }


def _audit_reasons(
    record: RiichiWaitDatasetRecord,
    pattern: ToitsuDropFacts,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    reasons = {pattern.classification}
    representative = pattern.representative_pair
    if representative is not None and (
        representative.second_discard_number == record.riichi_discard_number
    ):
        reasons.add("representative_pair_ends_on_declaration")
    if any(
        pair.first_raw_tile.endswith("r") or pair.second_raw_tile.endswith("r")
        for pair in pattern.pairs
    ):
        reasons.add("red_five_pair")
    if len(pattern.pairs) > 1:
        reasons.add("multiple_matching_pairs")
    called = tuple(
        discard.discard_number
        for discard in record.actor_discards_before_riichi
        if discard.call_event_index is not None
        and discard.call_event_index <= record.reach_accepted_event_index
    )
    if called:
        reasons.add("called_before_reach_accepted")
    ordered = tuple(reason for reason in AUDIT_REASONS if reason in reasons)
    return ordered, called


def _make_audit_sample(
    record: RiichiWaitDatasetRecord,
    pattern: ToitsuDropFacts,
    quality: WaitQualityFacts,
    reasons: tuple[str, ...],
    called: tuple[int, ...],
) -> AuditSample:
    return AuditSample(
        relative_source_path=record.relative_source_path,
        start_kyoku_line=record.start_kyoku_line,
        reach_line=record.reach_line,
        actor=record.actor,
        kyoku=record.kyoku,
        riichi_discard_number=record.riichi_discard_number,
        classification=pattern.classification,
        matching_pairs=pattern.pairs,
        representative_distance=pattern.distance,
        called_before_acceptance_discard_numbers=called,
        reasons=reasons,
        contains_suji=quality.contains_suji,
        is_good_wait=quality.is_good_wait,
        self_excluded_wait_copies=quality.self_excluded_wait_copies,
    )


def _consider_audit_sample(
    buckets: dict[str, list[AuditSample]],
    record: RiichiWaitDatasetRecord,
    pattern: ToitsuDropFacts,
    quality: WaitQualityFacts,
) -> None:
    reasons, called = _audit_reasons(record, pattern)
    stable_key = (
        record.relative_source_path,
        record.start_kyoku_line,
        record.reach_line,
    )
    relevant = tuple(
        reason
        for reason in reasons
        if len(buckets[reason]) < 3 or stable_key < buckets[reason][-1].stable_key
    )
    if not relevant:
        return
    sample = _make_audit_sample(record, pattern, quality, reasons, called)
    for reason in relevant:
        values = {item.stable_key: item for item in buckets[reason]}
        values[sample.stable_key] = sample
        buckets[reason] = sorted(values.values(), key=lambda item: item.stable_key)[:3]


@dataclass(frozen=True)
class AnnualToitsuDropTask:
    year: int
    path: str
    expected_records: int


@dataclass
class AnnualToitsuDropResult:
    year: int
    overall: ToitsuDropAccumulator
    by_turn: dict[int, ToitsuDropAccumulator]
    by_distance: dict[int, QualityAccumulator]
    audit_by_reason: dict[str, tuple[AuditSample, ...]]

    def validate(self) -> None:
        self.overall.validate()
        merged_turns = ToitsuDropAccumulator()
        for turn, accumulator in self.by_turn.items():
            if type(turn) is not int or turn < 1:
                raise ValueError("turn keys must be positive integers")
            accumulator.validate()
            merged_turns.merge(accumulator)
        if merged_turns != self.overall:
            raise ValueError("turn partitions disagree with annual total")

        merged_distances = QualityAccumulator()
        for distance, accumulator in self.by_distance.items():
            if type(distance) is not int or distance < 0:
                raise ValueError("distance keys must be non-negative integers")
            accumulator.validate()
            merged_distances.merge(accumulator)
        if merged_distances != self.overall.quality_for(PATTERN_CLASSIFICATIONS):
            raise ValueError("distance partitions disagree with pattern records")

        if set(self.audit_by_reason) != set(AUDIT_REASONS):
            raise ValueError("audit reason keys differ")
        for reason, samples in self.audit_by_reason.items():
            if (
                len(samples) > 3
                or tuple(sorted(samples, key=lambda item: item.stable_key)) != samples
            ):
                raise ValueError("audit candidates must be sorted and capped")
            if any(reason not in sample.reasons for sample in samples):
                raise ValueError("audit candidate does not contain its reason")


def aggregate_annual_toitsu_drop(
    task: AnnualToitsuDropTask,
) -> AnnualToitsuDropResult:
    """Stream and aggregate one annual dataset file; safe for Windows spawn."""
    overall = ToitsuDropAccumulator()
    by_turn: dict[int, ToitsuDropAccumulator] = {}
    by_distance: dict[int, QualityAccumulator] = {}
    audit_buckets: dict[str, list[AuditSample]] = {
        reason: [] for reason in AUDIT_REASONS
    }
    for line, record in enumerate(iter_dataset_records(Path(task.path)), 1):
        if record.year != task.year:
            raise ValueError(f"{task.path}:{line}: unexpected record year")
        pattern = detect_toitsu_drop(record.actor_discards_before_riichi)
        quality = evaluate_wait_quality(
            record.concealed_tiles_after_discard,
            record.fixed_melds,
            record.wait_details,
        )
        if quality.contains_ryanmen != record.contains_ryanmen:
            raise ValueError(f"{task.path}:{line}: ryanmen interpretation mismatch")
        overall.add(pattern, quality)
        by_turn.setdefault(record.riichi_discard_number, ToitsuDropAccumulator()).add(
            pattern, quality
        )
        if pattern.distance is not None:
            by_distance.setdefault(pattern.distance, QualityAccumulator()).add(quality)
        _consider_audit_sample(audit_buckets, record, pattern, quality)

    if overall.record_count != task.expected_records:
        raise ValueError(f"{task.path}: annual record count differs from manifest")
    result = AnnualToitsuDropResult(
        year=task.year,
        overall=overall,
        by_turn=by_turn,
        by_distance=by_distance,
        audit_by_reason={
            reason: tuple(samples) for reason, samples in audit_buckets.items()
        },
    )
    result.validate()
    return result


def run_toitsu_drop_tasks(
    tasks: tuple[AnnualToitsuDropTask, ...],
    *,
    workers: int = 1,
    on_result: Callable[[AnnualToitsuDropResult], None] | None = None,
) -> tuple[AnnualToitsuDropResult, ...]:
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if not tasks or len({task.year for task in tasks}) != len(tasks):
        raise ValueError("tasks must have nonempty distinct years")
    results: dict[int, AnnualToitsuDropResult] = {}
    if workers == 1:
        for task in tasks:
            result = aggregate_annual_toitsu_drop(task)
            results[result.year] = result
            if on_result:
                on_result(result)
    else:
        executor = ProcessPoolExecutor(max_workers=min(workers, len(tasks)))
        futures = {}
        try:
            futures = {
                executor.submit(aggregate_annual_toitsu_drop, task): task.year
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
    results: Iterable[AnnualToitsuDropResult],
) -> tuple[
    ToitsuDropAccumulator,
    dict[int, ToitsuDropAccumulator],
    dict[int, QualityAccumulator],
]:
    overall = ToitsuDropAccumulator()
    by_turn: dict[int, ToitsuDropAccumulator] = {}
    by_distance: dict[int, QualityAccumulator] = {}
    for result in results:
        result.validate()
        overall.merge(result.overall)
        for turn, accumulator in result.by_turn.items():
            by_turn.setdefault(turn, ToitsuDropAccumulator()).merge(accumulator)
        for distance, accumulator in result.by_distance.items():
            by_distance.setdefault(distance, QualityAccumulator()).merge(accumulator)
    merged = AnnualToitsuDropResult(
        year=0,
        overall=overall,
        by_turn=by_turn,
        by_distance=by_distance,
        audit_by_reason={reason: () for reason in AUDIT_REASONS},
    )
    merged.validate()
    return overall, by_turn, by_distance


def _block(results: tuple[AnnualToitsuDropResult, ...]) -> dict[str, Any]:
    overall, by_turn, by_distance = _merge_results(results)
    return {
        "overall": overall.to_dict(),
        "by_turn": [
            {"riichi_discard_number": turn, **accumulator.to_dict()}
            for turn, accumulator in sorted(by_turn.items())
        ],
        "by_distance": [
            {"distance": distance, **accumulator.to_dict()}
            for distance, accumulator in sorted(by_distance.items())
        ],
    }


def select_audit_samples(
    results: tuple[AnnualToitsuDropResult, ...],
    *,
    limit: int = 24,
) -> dict[str, Any]:
    if type(limit) is not int or limit < 1:
        raise ValueError("audit limit must be a positive integer")
    candidates: dict[str, list[AuditSample]] = {reason: [] for reason in AUDIT_REASONS}
    for reason in AUDIT_REASONS:
        unique: dict[tuple[str, int, int], AuditSample] = {}
        for result in results:
            for sample in result.audit_by_reason[reason]:
                unique[sample.stable_key] = sample
        candidates[reason] = sorted(unique.values(), key=lambda item: item.stable_key)[
            :3
        ]

    selected: dict[tuple[str, int, int], AuditSample] = {}
    called = candidates["called_before_reach_accepted"]
    if called:
        selected[called[0].stable_key] = called[0]
    for rank in range(3):
        for reason in AUDIT_REASONS:
            values = candidates[reason]
            if rank >= len(values):
                continue
            sample = values[rank]
            selected.setdefault(sample.stable_key, sample)
            if len(selected) >= limit:
                break
        if len(selected) >= limit:
            break
    samples = sorted(selected.values(), key=lambda item: item.stable_key)
    return {
        "limit": limit,
        "selected_count": len(samples),
        "selection_reason_order": list(AUDIT_REASONS),
        "missing_reasons": [
            reason for reason in AUDIT_REASONS if not candidates[reason]
        ],
        "samples": [sample.to_dict() for sample in samples],
    }


def toitsu_drop_results_document(
    results: tuple[AnnualToitsuDropResult, ...],
) -> dict[str, Any]:
    if not results or len({result.year for result in results}) != len(results):
        raise ValueError("results must have nonempty distinct years")
    ordered = tuple(sorted(results, key=lambda result: result.year))
    overall_block = _block(ordered)
    recent = tuple(result for result in ordered if result.year >= 2020)
    return {
        **overall_block,
        "recent_2020_2025": _block(recent) if recent else None,
        "years": [
            {
                "year": result.year,
                **_block((result,)),
            }
            for result in ordered
        ],
        "audit": select_audit_samples(ordered),
    }
