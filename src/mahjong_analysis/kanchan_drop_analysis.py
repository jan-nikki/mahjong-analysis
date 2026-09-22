"""Record-level analysis of riichi after a kanchan-drop-looking river pattern."""

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
from mahjong_analysis.tiles import tile_to_index
from mahjong_analysis.toitsu_drop_analysis import detect_toitsu_drop

CLASSIFICATIONS = ("none", "a_only", "b_only", "both")
PATTERN_CLASSIFICATIONS = ("a_only", "b_only", "both")
ORDER_CLASSES = ("inner_first", "outer_first", "symmetric")
DIRECTIONS = ("low_to_high", "high_to_low")
SHAPES = ("13", "24", "35", "46", "57", "68", "79")
TOITSU_STATUSES = ("absent", "present")
AUDIT_REASONS = (
    "called_before_reach_accepted",
    "none",
    "a_only",
    "b_only",
    "both",
    "representative_candidate_ends_on_declaration",
    "red_five_candidate",
    "multiple_matching_candidates",
    "inner_first",
    "outer_first",
    "symmetric",
    "low_to_high",
    "high_to_low",
)


def _quality_metric_slice(accumulator: QualityAccumulator) -> dict[str, Any]:
    value = accumulator.to_dict()
    return {
        "record_count": value["record_count"],
        "contains_ryanmen": value["contains_ryanmen"],
        "contains_suji": value["contains_suji"],
        "good_wait": value["good_wait"],
    }


def _percentage_point_difference(
    left: QualityAccumulator,
    right: QualityAccumulator,
) -> dict[str, float | None]:
    left_value = left.to_dict()
    right_value = right.to_dict()
    result: dict[str, float | None] = {}
    for metric in ("contains_ryanmen", "contains_suji", "good_wait"):
        left_rate = left_value[metric]["rate"]
        right_rate = right_value[metric]["rate"]
        result[metric] = (
            None
            if left_rate is None or right_rate is None
            else (left_rate - right_rate) * 100
        )
    return result


@dataclass(frozen=True)
class KanchanDropCandidate:
    """One adjacent same-suit, two-rank-apart discard candidate."""

    candidate_type: str
    suit: str
    shape: str
    direction: str
    order_class: str
    middle_tile: str
    first_discard_number: int
    second_discard_number: int
    first_raw_tile: str
    second_raw_tile: str
    first_normalized_tile: str
    second_normalized_tile: str
    first_tsumogiri: bool
    first_event_index: int
    second_event_index: int

    def __post_init__(self) -> None:
        if self.candidate_type not in {"A", "B"}:
            raise ValueError("candidate_type must be A or B")
        if self.suit not in "mps" or self.shape not in SHAPES:
            raise ValueError("candidate suit or shape is invalid")
        if self.direction not in DIRECTIONS or self.order_class not in ORDER_CLASSES:
            raise ValueError("candidate direction or order class is invalid")
        if self.second_discard_number != self.first_discard_number + 1:
            raise ValueError("candidate discard numbers must be adjacent")
        if self.first_event_index >= self.second_event_index:
            raise ValueError("candidate event indexes must be increasing")
        if (self.candidate_type == "B") != self.first_tsumogiri:
            raise ValueError("candidate_type must agree with first tsumogiri")
        low, high = (int(self.shape[0]), int(self.shape[1]))
        first_rank = int(self.first_normalized_tile[0])
        second_rank = int(self.second_normalized_tile[0])
        if (
            self.first_normalized_tile[1] != self.suit
            or self.second_normalized_tile[1] != self.suit
            or {first_rank, second_rank} != {low, high}
            or high - low != 2
        ):
            raise ValueError("candidate tiles disagree with suit and shape")
        expected_direction = (
            "low_to_high" if first_rank < second_rank else "high_to_low"
        )
        if self.direction != expected_direction:
            raise ValueError("candidate direction disagrees with discard order")
        first_distance = abs(first_rank - 5)
        second_distance = abs(second_rank - 5)
        expected_order = (
            "inner_first"
            if first_distance < second_distance
            else "outer_first"
            if first_distance > second_distance
            else "symmetric"
        )
        if self.order_class != expected_order:
            raise ValueError("candidate order class disagrees with centrality")
        if self.middle_tile != f"{(low + high) // 2}{self.suit}":
            raise ValueError("candidate middle tile disagrees with shape")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_type": self.candidate_type,
            "suit": self.suit,
            "shape": self.shape,
            "direction": self.direction,
            "order_class": self.order_class,
            "middle_tile": self.middle_tile,
            "first_discard_number": self.first_discard_number,
            "second_discard_number": self.second_discard_number,
            "first_raw_tile": self.first_raw_tile,
            "second_raw_tile": self.second_raw_tile,
            "first_normalized_tile": self.first_normalized_tile,
            "second_normalized_tile": self.second_normalized_tile,
            "first_tsumogiri": self.first_tsumogiri,
            "first_event_index": self.first_event_index,
            "second_event_index": self.second_event_index,
        }


@dataclass(frozen=True)
class KanchanDropFacts:
    classification: str
    candidates: tuple[KanchanDropCandidate, ...]
    riichi_discard_number: int

    def __post_init__(self) -> None:
        if self.classification not in CLASSIFICATIONS:
            raise ValueError("unknown kanchan-drop classification")
        candidate_types = {candidate.candidate_type for candidate in self.candidates}
        expected = (
            "both"
            if candidate_types == {"A", "B"}
            else "a_only"
            if candidate_types == {"A"}
            else "b_only"
            if candidate_types == {"B"}
            else "none"
        )
        if self.classification != expected:
            raise ValueError("classification does not agree with candidates")
        if self.riichi_discard_number < 1:
            raise ValueError("riichi_discard_number must be positive")
        if any(
            left.second_discard_number >= right.second_discard_number
            for left, right in pairwise(self.candidates)
        ):
            raise ValueError("candidates must be in completion order")
        if self.candidates and self.candidates[-1].second_discard_number > (
            self.riichi_discard_number
        ):
            raise ValueError("candidate cannot occur after declaration")

    @property
    def has_pattern(self) -> bool:
        return bool(self.candidates)

    @property
    def representative_candidate(self) -> KanchanDropCandidate | None:
        return self.candidates[-1] if self.candidates else None

    @property
    def distance(self) -> int | None:
        representative = self.representative_candidate
        if representative is None:
            return None
        return self.riichi_discard_number - representative.second_discard_number


def detect_kanchan_drop(
    discards: tuple[DatasetActorDiscard, ...],
) -> KanchanDropFacts:
    """Classify a validated actor river without using call or wait metadata."""
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

    candidates: list[KanchanDropCandidate] = []
    for first, second in pairwise(discards):
        if second.tsumogiri:
            continue
        first_index = tile_to_index(first.normalized_tile)
        second_index = tile_to_index(second.normalized_tile)
        if first_index >= 27 or second_index >= 27:
            continue
        if first_index // 9 != second_index // 9:
            continue
        first_rank = first_index % 9 + 1
        second_rank = second_index % 9 + 1
        if abs(first_rank - second_rank) != 2:
            continue
        low, high = sorted((first_rank, second_rank))
        first_distance = abs(first_rank - 5)
        second_distance = abs(second_rank - 5)
        order_class = (
            "inner_first"
            if first_distance < second_distance
            else "outer_first"
            if first_distance > second_distance
            else "symmetric"
        )
        suit = first.normalized_tile[1]
        candidates.append(
            KanchanDropCandidate(
                candidate_type="B" if first.tsumogiri else "A",
                suit=suit,
                shape=f"{low}{high}",
                direction=(
                    "low_to_high" if first_rank < second_rank else "high_to_low"
                ),
                order_class=order_class,
                middle_tile=f"{(low + high) // 2}{suit}",
                first_discard_number=first.discard_number,
                second_discard_number=second.discard_number,
                first_raw_tile=first.tile,
                second_raw_tile=second.tile,
                first_normalized_tile=first.normalized_tile,
                second_normalized_tile=second.normalized_tile,
                first_tsumogiri=first.tsumogiri,
                first_event_index=first.event_index,
                second_event_index=second.event_index,
            )
        )
    candidate_types = {candidate.candidate_type for candidate in candidates}
    classification = (
        "both"
        if candidate_types == {"A", "B"}
        else "a_only"
        if candidate_types == {"A"}
        else "b_only"
        if candidate_types == {"B"}
        else "none"
    )
    return KanchanDropFacts(classification, tuple(candidates), len(discards))


@dataclass
class KanchanDropAccumulator:
    """Record partitions plus representative-candidate order comparisons."""

    by_classification: dict[str, QualityAccumulator] = field(
        default_factory=lambda: {name: QualityAccumulator() for name in CLASSIFICATIONS}
    )
    candidate_counts: Counter[int] = field(default_factory=Counter)
    by_order: dict[str, QualityAccumulator] = field(
        default_factory=lambda: {name: QualityAccumulator() for name in ORDER_CLASSES}
    )
    by_direction: dict[str, QualityAccumulator] = field(
        default_factory=lambda: {name: QualityAccumulator() for name in DIRECTIONS}
    )
    by_shape_direction: dict[tuple[str, str], QualityAccumulator] = field(
        default_factory=lambda: {
            (shape, direction): QualityAccumulator()
            for shape in SHAPES
            for direction in DIRECTIONS
        }
    )
    by_type_order: dict[tuple[str, str], QualityAccumulator] = field(
        default_factory=lambda: {
            (candidate_type, order): QualityAccumulator()
            for candidate_type in ("A", "B")
            for order in ORDER_CLASSES
        }
    )

    def add(self, pattern: KanchanDropFacts, quality: WaitQualityFacts) -> None:
        self.by_classification[pattern.classification].add(quality)
        self.candidate_counts[len(pattern.candidates)] += 1
        representative = pattern.representative_candidate
        if representative is None:
            return
        self.by_order[representative.order_class].add(quality)
        self.by_direction[representative.direction].add(quality)
        self.by_shape_direction[(representative.shape, representative.direction)].add(
            quality
        )
        self.by_type_order[
            (representative.candidate_type, representative.order_class)
        ].add(quality)

    def merge(self, other: KanchanDropAccumulator) -> None:
        for name in CLASSIFICATIONS:
            self.by_classification[name].merge(other.by_classification[name])
        self.candidate_counts.update(other.candidate_counts)
        for name in ORDER_CLASSES:
            self.by_order[name].merge(other.by_order[name])
        for name in DIRECTIONS:
            self.by_direction[name].merge(other.by_direction[name])
        for key in self.by_shape_direction:
            self.by_shape_direction[key].merge(other.by_shape_direction[key])
        for key in self.by_type_order:
            self.by_type_order[key].merge(other.by_type_order[key])

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

    def pattern_projection(self) -> KanchanDropAccumulator:
        result = KanchanDropAccumulator()
        for name in PATTERN_CLASSIFICATIONS:
            result.by_classification[name].merge(self.by_classification[name])
        result.candidate_counts.update(
            {
                count: records
                for count, records in self.candidate_counts.items()
                if count
            }
        )
        for name in ORDER_CLASSES:
            result.by_order[name].merge(self.by_order[name])
        for name in DIRECTIONS:
            result.by_direction[name].merge(self.by_direction[name])
        for key in result.by_shape_direction:
            result.by_shape_direction[key].merge(self.by_shape_direction[key])
        for key in result.by_type_order:
            result.by_type_order[key].merge(self.by_type_order[key])
        result.validate()
        return result

    def validate(self) -> None:
        if set(self.by_classification) != set(CLASSIFICATIONS):
            raise ValueError("classification accumulator keys differ")
        for accumulator in self.by_classification.values():
            accumulator.validate()
        if any(type(count) is not int or count < 0 for count in self.candidate_counts):
            raise ValueError("candidate count must be a non-negative integer")
        if sum(self.candidate_counts.values()) != self.record_count:
            raise ValueError("candidate-count distribution does not sum to records")
        if self.candidate_counts[0] != self.by_classification["none"].record_count:
            raise ValueError("zero-candidate count must equal none classification")
        pattern_quality = self.quality_for(PATTERN_CLASSIFICATIONS)
        if (
            sum(
                records for count, records in self.candidate_counts.items() if count > 0
            )
            != pattern_quality.record_count
        ):
            raise ValueError("positive-candidate count must equal pattern records")

        partitions = (
            self.by_order,
            self.by_direction,
            self.by_shape_direction,
            self.by_type_order,
        )
        for partition in partitions:
            merged = QualityAccumulator()
            for accumulator in partition.values():
                accumulator.validate()
                merged.merge(accumulator)
            if merged != pattern_quality:
                raise ValueError("representative-candidate partition disagrees")

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

        shape_comparisons = []
        for shape in SHAPES:
            low, high = int(shape[0]), int(shape[1])
            low_to_high = self.by_shape_direction[(shape, "low_to_high")]
            high_to_low = self.by_shape_direction[(shape, "high_to_low")]
            low_first_distance = abs(low - 5)
            high_first_distance = abs(high - 5)
            if low_first_distance == high_first_distance:
                inner_direction = None
                comparison = _percentage_point_difference(high_to_low, low_to_high)
                comparison_name = "high_to_low_minus_low_to_high"
            else:
                inner_direction = (
                    "low_to_high"
                    if low_first_distance < high_first_distance
                    else "high_to_low"
                )
                inner = self.by_shape_direction[(shape, inner_direction)]
                outer_direction = (
                    "high_to_low" if inner_direction == "low_to_high" else "low_to_high"
                )
                outer = self.by_shape_direction[(shape, outer_direction)]
                comparison = _percentage_point_difference(inner, outer)
                comparison_name = "inner_minus_outer_percentage_points"
            shape_comparisons.append(
                {
                    "shape": shape,
                    "inner_direction": inner_direction,
                    "low_to_high": _quality_metric_slice(low_to_high),
                    "high_to_low": _quality_metric_slice(high_to_low),
                    comparison_name: comparison,
                }
            )

        return {
            "record_count": self.record_count,
            "pattern_present": record_metric(pattern_quality.record_count),
            "classification_counts": {
                name: self.by_classification[name].record_count
                for name in CLASSIFICATIONS
            },
            "matching_candidate_count_distribution": [
                {"matching_candidate_count": count, "record_count": records}
                for count, records in sorted(self.candidate_counts.items())
            ],
            "groups": {
                "all": all_quality.to_dict(),
                "pattern_present": pattern_quality.to_dict(),
                "no_pattern": no_pattern_quality.to_dict(),
                "a_only": self.by_classification["a_only"].to_dict(),
                "b_only": self.by_classification["b_only"].to_dict(),
                "both": self.by_classification["both"].to_dict(),
            },
            "pattern_minus_no_pattern_percentage_points": (
                _percentage_point_difference(pattern_quality, no_pattern_quality)
            ),
            "representative_order_groups": {
                name: self.by_order[name].to_dict() for name in ORDER_CLASSES
            },
            "inner_minus_outer_percentage_points": _percentage_point_difference(
                self.by_order["inner_first"], self.by_order["outer_first"]
            ),
            "representative_direction_groups": {
                name: _quality_metric_slice(self.by_direction[name])
                for name in DIRECTIONS
            },
            "representative_type_order_groups": [
                {
                    "candidate_type": candidate_type,
                    "order_class": order,
                    **_quality_metric_slice(
                        self.by_type_order[(candidate_type, order)]
                    ),
                }
                for candidate_type in ("A", "B")
                for order in ORDER_CLASSES
            ],
            "representative_shape_direction": shape_comparisons,
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
    candidates: tuple[KanchanDropCandidate, ...]
    representative_distance: int | None
    called_before_acceptance_discard_numbers: tuple[int, ...]
    reasons: tuple[str, ...]
    contains_suji: bool
    is_good_wait: bool
    contains_ryanmen: bool
    self_excluded_wait_copies: int

    @property
    def stable_key(self) -> tuple[str, int, int]:
        return self.relative_source_path, self.start_kyoku_line, self.reach_line

    def to_dict(self) -> dict[str, Any]:
        representative = self.candidates[-1] if self.candidates else None
        return {
            "relative_source_path": self.relative_source_path,
            "start_kyoku_line": self.start_kyoku_line,
            "reach_line": self.reach_line,
            "actor": self.actor,
            "kyoku": self.kyoku,
            "riichi_discard_number": self.riichi_discard_number,
            "classification": self.classification,
            "matching_candidates": [value.to_dict() for value in self.candidates],
            "representative_candidate": (
                representative.to_dict() if representative is not None else None
            ),
            "representative_distance": self.representative_distance,
            "called_before_acceptance_discard_numbers": list(
                self.called_before_acceptance_discard_numbers
            ),
            "selection_reasons": list(self.reasons),
            "contains_suji": self.contains_suji,
            "is_good_wait": self.is_good_wait,
            "contains_ryanmen": self.contains_ryanmen,
            "self_excluded_wait_copies": self.self_excluded_wait_copies,
        }


def _audit_reasons(
    record: RiichiWaitDatasetRecord,
    pattern: KanchanDropFacts,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    reasons = {pattern.classification}
    representative = pattern.representative_candidate
    if representative is not None:
        reasons.add(representative.order_class)
        reasons.add(representative.direction)
        if representative.second_discard_number == record.riichi_discard_number:
            reasons.add("representative_candidate_ends_on_declaration")
    if any(
        candidate.first_raw_tile.endswith("r")
        or candidate.second_raw_tile.endswith("r")
        for candidate in pattern.candidates
    ):
        reasons.add("red_five_candidate")
    if len(pattern.candidates) > 1:
        reasons.add("multiple_matching_candidates")
    called = tuple(
        discard.discard_number
        for discard in record.actor_discards_before_riichi
        if discard.call_event_index is not None
        and discard.call_event_index <= record.reach_accepted_event_index
    )
    if called:
        reasons.add("called_before_reach_accepted")
    return tuple(reason for reason in AUDIT_REASONS if reason in reasons), called


def _make_audit_sample(
    record: RiichiWaitDatasetRecord,
    pattern: KanchanDropFacts,
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
        candidates=pattern.candidates,
        representative_distance=pattern.distance,
        called_before_acceptance_discard_numbers=called,
        reasons=reasons,
        contains_suji=quality.contains_suji,
        is_good_wait=quality.is_good_wait,
        contains_ryanmen=quality.contains_ryanmen,
        self_excluded_wait_copies=quality.self_excluded_wait_copies,
    )


def _consider_audit_sample(
    buckets: dict[str, list[AuditSample]],
    record: RiichiWaitDatasetRecord,
    pattern: KanchanDropFacts,
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
class AnnualKanchanDropTask:
    year: int
    path: str
    expected_records: int


@dataclass
class AnnualKanchanDropResult:
    year: int
    overall: KanchanDropAccumulator
    by_turn: dict[int, KanchanDropAccumulator]
    by_distance: dict[int, KanchanDropAccumulator]
    by_toitsu_status: dict[str, KanchanDropAccumulator]
    audit_by_reason: dict[str, tuple[AuditSample, ...]]

    def validate(self) -> None:
        self.overall.validate()
        merged_turns = KanchanDropAccumulator()
        for turn, accumulator in self.by_turn.items():
            if type(turn) is not int or turn < 1:
                raise ValueError("turn keys must be positive integers")
            accumulator.validate()
            merged_turns.merge(accumulator)
        if merged_turns != self.overall:
            raise ValueError("turn partitions disagree with annual total")

        merged_distances = KanchanDropAccumulator()
        for distance, accumulator in self.by_distance.items():
            if type(distance) is not int or distance < 0:
                raise ValueError("distance keys must be non-negative integers")
            accumulator.validate()
            merged_distances.merge(accumulator)
        if merged_distances != self.overall.pattern_projection():
            raise ValueError("distance partitions disagree with pattern records")

        if set(self.by_toitsu_status) != set(TOITSU_STATUSES):
            raise ValueError("toitsu status keys differ")
        merged_toitsu = KanchanDropAccumulator()
        for accumulator in self.by_toitsu_status.values():
            accumulator.validate()
            merged_toitsu.merge(accumulator)
        if merged_toitsu != self.overall:
            raise ValueError("toitsu status partitions disagree with annual total")

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


def aggregate_annual_kanchan_drop(
    task: AnnualKanchanDropTask,
) -> AnnualKanchanDropResult:
    """Stream and aggregate one annual dataset file; safe for Windows spawn."""
    overall = KanchanDropAccumulator()
    by_turn: dict[int, KanchanDropAccumulator] = {}
    by_distance: dict[int, KanchanDropAccumulator] = {}
    by_toitsu_status = {status: KanchanDropAccumulator() for status in TOITSU_STATUSES}
    audit_buckets: dict[str, list[AuditSample]] = {
        reason: [] for reason in AUDIT_REASONS
    }
    for line, record in enumerate(iter_dataset_records(Path(task.path)), 1):
        if record.year != task.year:
            raise ValueError(f"{task.path}:{line}: unexpected record year")
        pattern = detect_kanchan_drop(record.actor_discards_before_riichi)
        toitsu = detect_toitsu_drop(record.actor_discards_before_riichi)
        quality = evaluate_wait_quality(
            record.concealed_tiles_after_discard,
            record.fixed_melds,
            record.wait_details,
        )
        if quality.contains_ryanmen != record.contains_ryanmen:
            raise ValueError(f"{task.path}:{line}: ryanmen interpretation mismatch")
        overall.add(pattern, quality)
        by_turn.setdefault(record.riichi_discard_number, KanchanDropAccumulator()).add(
            pattern, quality
        )
        if pattern.distance is not None:
            by_distance.setdefault(pattern.distance, KanchanDropAccumulator()).add(
                pattern, quality
            )
        by_toitsu_status["present" if toitsu.has_pattern else "absent"].add(
            pattern, quality
        )
        _consider_audit_sample(audit_buckets, record, pattern, quality)

    if overall.record_count != task.expected_records:
        raise ValueError(f"{task.path}: annual record count differs from manifest")
    result = AnnualKanchanDropResult(
        year=task.year,
        overall=overall,
        by_turn=by_turn,
        by_distance=by_distance,
        by_toitsu_status=by_toitsu_status,
        audit_by_reason={
            reason: tuple(samples) for reason, samples in audit_buckets.items()
        },
    )
    result.validate()
    return result


def run_kanchan_drop_tasks(
    tasks: tuple[AnnualKanchanDropTask, ...],
    *,
    workers: int = 1,
    on_result: Callable[[AnnualKanchanDropResult], None] | None = None,
) -> tuple[AnnualKanchanDropResult, ...]:
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if not tasks or len({task.year for task in tasks}) != len(tasks):
        raise ValueError("tasks must have nonempty distinct years")
    results: dict[int, AnnualKanchanDropResult] = {}
    if workers == 1:
        for task in tasks:
            result = aggregate_annual_kanchan_drop(task)
            results[result.year] = result
            if on_result:
                on_result(result)
    else:
        executor = ProcessPoolExecutor(max_workers=min(workers, len(tasks)))
        futures = {}
        try:
            futures = {
                executor.submit(aggregate_annual_kanchan_drop, task): task.year
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
    results: Iterable[AnnualKanchanDropResult],
) -> tuple[
    KanchanDropAccumulator,
    dict[int, KanchanDropAccumulator],
    dict[int, KanchanDropAccumulator],
    dict[str, KanchanDropAccumulator],
]:
    overall = KanchanDropAccumulator()
    by_turn: dict[int, KanchanDropAccumulator] = {}
    by_distance: dict[int, KanchanDropAccumulator] = {}
    by_toitsu_status = {status: KanchanDropAccumulator() for status in TOITSU_STATUSES}
    for result in results:
        result.validate()
        overall.merge(result.overall)
        for turn, accumulator in result.by_turn.items():
            by_turn.setdefault(turn, KanchanDropAccumulator()).merge(accumulator)
        for distance, accumulator in result.by_distance.items():
            by_distance.setdefault(distance, KanchanDropAccumulator()).merge(
                accumulator
            )
        for status in TOITSU_STATUSES:
            by_toitsu_status[status].merge(result.by_toitsu_status[status])
    merged = AnnualKanchanDropResult(
        year=0,
        overall=overall,
        by_turn=by_turn,
        by_distance=by_distance,
        by_toitsu_status=by_toitsu_status,
        audit_by_reason={reason: () for reason in AUDIT_REASONS},
    )
    merged.validate()
    return overall, by_turn, by_distance, by_toitsu_status


def _block(results: tuple[AnnualKanchanDropResult, ...]) -> dict[str, Any]:
    overall, by_turn, by_distance, by_toitsu_status = _merge_results(results)
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
        "by_toitsu_pattern": {
            status: by_toitsu_status[status].to_dict() for status in TOITSU_STATUSES
        },
    }


def select_audit_samples(
    results: tuple[AnnualKanchanDropResult, ...],
    *,
    limit: int = 30,
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


def kanchan_drop_results_document(
    results: tuple[AnnualKanchanDropResult, ...],
) -> dict[str, Any]:
    if not results or len({result.year for result in results}) != len(results):
        raise ValueError("results must have nonempty distinct years")
    ordered = tuple(sorted(results, key=lambda result: result.year))
    overall_block = _block(ordered)
    recent = tuple(result for result in ordered if 2020 <= result.year <= 2025)
    return {
        **overall_block,
        "recent_2020_2025": _block(recent) if recent else None,
        "years": [{"year": result.year, **_block((result,))} for result in ordered],
        "audit": select_audit_samples(ordered),
    }
