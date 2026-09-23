"""Dealer/nondealer score-gap counterfactual and Shapley decomposition."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from fractions import Fraction
from os import PathLike
from pathlib import Path
from typing import Any, Literal

from mahjong_analysis.dealer_child_riichi_points import (
    PRIMARY_YEARS,
    SUPPORTED_YEARS,
    ReachType,
    RiichiPointRecord,
    normalize_years,
)
from mahjong_analysis.score_table import (
    PaymentPattern,
    WinMethod,
    WinnerRole,
    convert_payment_pattern,
    convert_total_points,
    finite_basic_point_candidates,
    score_points,
)

FactorName = Literal["payment_rule", "method_mix", "within_method_score"]
FACTORS: tuple[FactorName, ...] = (
    "payment_rule",
    "method_mix",
    "within_method_score",
)


@dataclass(frozen=True)
class DecompositionWinRecord:
    """One winning established riichi with both role-rule point values."""

    year: int
    source_path: str
    kyoku_index: int
    actor: int
    is_dealer: bool
    reach_type: ReachType
    method: WinMethod
    observed_points: int
    dealer_points: int
    nondealer_points: int
    candidate_basic_points: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.year not in SUPPORTED_YEARS:
            raise ValueError(f"unsupported year: {self.year}")
        if not self.source_path or "\\" in self.source_path:
            raise ValueError("source_path must be relative POSIX text")
        if type(self.kyoku_index) is not int or self.kyoku_index < 0:
            raise ValueError("kyoku_index must be a non-negative integer")
        if type(self.actor) is not int or self.actor not in range(4):
            raise ValueError("actor must be between 0 and 3")
        if type(self.is_dealer) is not bool:
            raise TypeError("is_dealer must be boolean")
        if self.reach_type not in ("riichi", "double_riichi"):
            raise ValueError("unsupported reach_type")
        if self.method not in ("tsumo", "ron"):
            raise ValueError("unsupported win method")
        for field_name in ("observed_points", "dealer_points", "nondealer_points"):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0 or value % 100 != 0:
                raise ValueError(f"{field_name} must be a positive multiple of 100")
        expected = self.dealer_points if self.is_dealer else self.nondealer_points
        if self.observed_points != expected:
            raise ValueError(
                "observed points do not match the observed role conversion"
            )
        candidates = self.candidate_basic_points
        if not candidates:
            converted = convert_total_points(
                self.observed_points,
                self.role,
                self.method,
            )
            if (
                converted.dealer_points != self.dealer_points
                or converted.nondealer_points != self.nondealer_points
            ):
                raise ValueError("counterfactual points do not match score candidates")
            candidates = tuple(
                sorted({tier.base_points for tier in converted.candidates})
            )
            object.__setattr__(self, "candidate_basic_points", candidates)
        if (
            any(type(value) is not int or value <= 0 for value in candidates)
            or tuple(sorted(set(candidates))) != candidates
        ):
            raise ValueError(
                "candidate_basic_points must be sorted unique positive integers"
            )
        if any(
            score_points("dealer", self.method, value) != self.dealer_points
            or score_points("nondealer", self.method, value) != self.nondealer_points
            for value in candidates
        ):
            raise ValueError(
                "candidate basic points imply different counterfactual points"
            )

    @property
    def role(self) -> WinnerRole:
        return "dealer" if self.is_dealer else "nondealer"


@dataclass(frozen=True, order=True)
class CounterfactualState:
    """Sources used for R, M, and B in one of the eight means."""

    payment_rule: WinnerRole
    method_mix: WinnerRole
    within_method_score: WinnerRole


@dataclass(frozen=True)
class CounterfactualMean:
    state: CounterfactualState
    mean_points: Fraction


@dataclass(frozen=True)
class ShapleyContribution:
    factor: FactorName
    points: Fraction


@dataclass(frozen=True)
class RoundingEffect:
    """Mean departure from an exact 1.5 dealer/nondealer multiplier."""

    score_source: Literal["dealer", "nondealer", "pooled"]
    mean_points: Fraction


@dataclass(frozen=True, order=True)
class ConversionTier:
    """One canonical conversion shared by one or more legal basic points."""

    method: WinMethod
    candidate_basic_points: tuple[int, ...]
    dealer_points: int
    nondealer_points: int


@dataclass(frozen=True)
class MethodWinCount:
    score_source: WinnerRole
    method: WinMethod
    wins: int


@dataclass(frozen=True)
class ScoreDistributionEntry:
    score_source: WinnerRole
    tier: ConversionTier
    wins: int


@dataclass(frozen=True)
class ScoreDecompositionResult:
    """Eight means and exact three-factor Shapley contributions."""

    dealer_wins: int
    nondealer_wins: int
    observed_dealer_mean: Fraction
    observed_nondealer_mean: Fraction
    counterfactual_means: tuple[CounterfactualMean, ...]
    contributions: tuple[ShapleyContribution, ...]
    rounding_effects: tuple[RoundingEffect, ...]
    method_counts: tuple[MethodWinCount, ...]
    score_distributions: tuple[ScoreDistributionEntry, ...]

    def __post_init__(self) -> None:
        if self.dealer_wins <= 0 or self.nondealer_wins <= 0:
            raise ValueError("both dealer and nondealer wins are required")
        if len(self.counterfactual_means) != 8:
            raise ValueError("exactly eight counterfactual means are required")
        if tuple(value.factor for value in self.contributions) != FACTORS:
            raise ValueError("Shapley factors are incomplete or out of order")
        if tuple(value.score_source for value in self.rounding_effects) != (
            "dealer",
            "nondealer",
            "pooled",
        ):
            raise ValueError("rounding-effect sources are incomplete or out of order")
        if sum((value.points for value in self.contributions), Fraction()) != self.gap:
            raise ValueError("Shapley contributions do not sum to the observed gap")
        expected_method_keys = tuple(
            (role, method)
            for role in ("dealer", "nondealer")
            for method in ("tsumo", "ron")
        )
        if (
            tuple((value.score_source, value.method) for value in self.method_counts)
            != expected_method_keys
        ):
            raise ValueError("method counts are incomplete or out of order")
        for value in self.method_counts:
            distributed = sum(
                entry.wins
                for entry in self.score_distributions
                if entry.score_source == value.score_source
                and entry.tier.method == value.method
            )
            if distributed != value.wins:
                raise ValueError("score distribution does not sum to method wins")

    @property
    def gap(self) -> Fraction:
        return self.observed_dealer_mean - self.observed_nondealer_mean

    def mean_for(self, state: CounterfactualState) -> Fraction:
        for value in self.counterfactual_means:
            if value.state == state:
                return value.mean_points
        raise KeyError(state)

    def contribution_for(self, factor: FactorName) -> Fraction:
        for value in self.contributions:
            if value.factor == factor:
                return value.points
        raise KeyError(factor)

    def contribution_share(self, factor: FactorName) -> Fraction | None:
        if self.gap == 0:
            return None
        return self.contribution_for(factor) / self.gap

    @property
    def efficiency_residual(self) -> Fraction:
        return (
            sum((value.points for value in self.contributions), Fraction()) - self.gap
        )

    def method_count(self, role: WinnerRole, method: WinMethod) -> int:
        for value in self.method_counts:
            if value.score_source == role and value.method == method:
                return value.wins
        raise KeyError((role, method))

    def rounding_effect_for(
        self,
        score_source: Literal["dealer", "nondealer", "pooled"],
    ) -> Fraction:
        for value in self.rounding_effects:
            if value.score_source == score_source:
                return value.mean_points
        raise KeyError(score_source)


@dataclass(frozen=True)
class DecompositionSlice:
    all_riichi: ScoreDecompositionResult
    normal_riichi: ScoreDecompositionResult


@dataclass(frozen=True)
class DecompositionInputMetadata:
    summary_logical_path: str
    records_logical_path: str
    summary_sha256: str
    records_sha256: str
    all_record_count: int
    selected_record_count: int
    selected_win_count: int
    converted_win_count: int
    unsupported_score_count: int
    ambiguous_score_count: int

    def __post_init__(self) -> None:
        for name in ("summary_logical_path", "records_logical_path"):
            value = getattr(self, name)
            if not value or "\\" in value or Path(value).is_absolute():
                raise ValueError(f"{name} must be relative POSIX text")
        for name in ("summary_sha256", "records_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        count_names = (
            "all_record_count",
            "selected_record_count",
            "selected_win_count",
            "converted_win_count",
            "unsupported_score_count",
            "ambiguous_score_count",
        )
        if any(
            type(getattr(self, name)) is not int or getattr(self, name) < 0
            for name in count_names
        ):
            raise ValueError("input metadata counts must be non-negative integers")
        if self.selected_record_count > self.all_record_count:
            raise ValueError("selected record count cannot exceed all records")
        if self.selected_win_count > self.selected_record_count:
            raise ValueError("selected win count cannot exceed selected records")
        if self.selected_win_count != self.converted_win_count:
            raise ValueError("every selected win must be converted")
        if self.unsupported_score_count or self.ambiguous_score_count:
            raise ValueError("successful metadata cannot contain conversion failures")


@dataclass(frozen=True)
class ScoreDecompositionAnalysis:
    selected_years: tuple[int, ...]
    selected: DecompositionSlice
    years: tuple[tuple[int, DecompositionSlice], ...]
    primary_2020_2025: DecompositionSlice | None
    long_term_2009_2025: DecompositionSlice | None
    input_metadata: DecompositionInputMetadata | None = None


def decomposition_record_from_article_one(
    record: RiichiPointRecord,
    *,
    payment_pattern: PaymentPattern | None = None,
) -> DecompositionWinRecord:
    """Reuse Article 1's validated hora/honba/multi-ron classification.

    Payer-level patterns are preferred.  The total-only adapter is intentionally
    fail closed when the standard table permits different conversions.
    """
    if not isinstance(record, RiichiPointRecord):
        raise TypeError("record must be a RiichiPointRecord")
    if record.outcome != "win" or record.hand_points is None:
        raise ValueError("only winning Article 1 records can be decomposed")
    if record.win_method not in ("tsumo", "ron"):
        raise ValueError("winning Article 1 record has no valid method")
    role: WinnerRole = "dealer" if record.actor == record.oya else "nondealer"
    if payment_pattern is None:
        converted = convert_total_points(record.hand_points, role, record.win_method)
    else:
        if payment_pattern.role != role or payment_pattern.method != record.win_method:
            raise ValueError("payment pattern does not match the Article 1 record")
        if payment_pattern.total != record.hand_points:
            raise ValueError("payment pattern does not reproduce Article 1 hand_points")
        converted = convert_payment_pattern(payment_pattern)
    if converted.observed_points != record.hand_points:
        raise ValueError("score conversion does not reproduce Article 1 hand_points")
    return DecompositionWinRecord(
        year=record.year,
        source_path=record.source_path,
        kyoku_index=record.kyoku_index,
        actor=record.actor,
        is_dealer=record.actor == record.oya,
        reach_type=record.reach_type,
        method=record.win_method,
        observed_points=record.hand_points,
        dealer_points=converted.dealer_points,
        nondealer_points=converted.nondealer_points,
        candidate_basic_points=tuple(
            sorted({tier.base_points for tier in converted.candidates})
        ),
    )


def observed_group_means(
    records: Iterable[DecompositionWinRecord],
) -> tuple[Fraction, Fraction]:
    """Return Article 1-compatible dealer and nondealer observed means."""
    counts: Counter[WinnerRole] = Counter()
    sums: Counter[WinnerRole] = Counter()
    for record in records:
        counts[record.role] += 1
        sums[record.role] += record.observed_points
    if not counts["dealer"] or not counts["nondealer"]:
        raise ValueError("both dealer and nondealer wins are required")
    return (
        Fraction(sums["dealer"], counts["dealer"]),
        Fraction(sums["nondealer"], counts["nondealer"]),
    )


@dataclass
class _ScoreAccumulator:
    counts: Counter[tuple[WinnerRole, WinMethod]] = field(default_factory=Counter)
    dealer_sums: Counter[tuple[WinnerRole, WinMethod]] = field(default_factory=Counter)
    nondealer_sums: Counter[tuple[WinnerRole, WinMethod]] = field(
        default_factory=Counter
    )
    rounding_twice_sums: Counter[WinnerRole] = field(default_factory=Counter)
    tier_counts: Counter[tuple[WinnerRole, ConversionTier]] = field(
        default_factory=Counter
    )

    def add(self, record: DecompositionWinRecord) -> None:
        key = (record.role, record.method)
        self.counts[key] += 1
        self.dealer_sums[key] += record.dealer_points
        self.nondealer_sums[key] += record.nondealer_points
        self.rounding_twice_sums[record.role] += (
            2 * record.dealer_points - 3 * record.nondealer_points
        )
        tier = ConversionTier(
            method=record.method,
            candidate_basic_points=record.candidate_basic_points,
            dealer_points=record.dealer_points,
            nondealer_points=record.nondealer_points,
        )
        self.tier_counts[record.role, tier] += 1

    def role_count(self, role: WinnerRole) -> int:
        return sum(self.counts[role, method] for method in ("tsumo", "ron"))


@dataclass
class _SliceAccumulator:
    all_riichi: _ScoreAccumulator = field(default_factory=_ScoreAccumulator)
    normal_riichi: _ScoreAccumulator = field(default_factory=_ScoreAccumulator)

    def add(self, record: DecompositionWinRecord) -> None:
        self.all_riichi.add(record)
        if record.reach_type == "riichi":
            self.normal_riichi.add(record)

    def freeze(self) -> DecompositionSlice:
        return DecompositionSlice(
            all_riichi=_freeze_score_accumulator(self.all_riichi),
            normal_riichi=_freeze_score_accumulator(self.normal_riichi),
        )


class ScoreDecompositionBuilder:
    """Fixed-memory accumulator for all requested periods and sensitivities."""

    def __init__(self, selected_years: Iterable[int]) -> None:
        self.selected_years = normalize_years(selected_years)
        self._selected = _SliceAccumulator()
        self._years = {year: _SliceAccumulator() for year in self.selected_years}
        self._primary = (
            _SliceAccumulator()
            if set(PRIMARY_YEARS).issubset(self.selected_years)
            else None
        )
        self._long_term = (
            _SliceAccumulator() if self.selected_years == SUPPORTED_YEARS else None
        )

    def add(self, record: DecompositionWinRecord) -> None:
        """Add one converted win without retaining the record."""
        if record.year not in self._years:
            raise ValueError("record year is outside selected years")
        self._selected.add(record)
        self._years[record.year].add(record)
        if self._primary is not None and record.year in PRIMARY_YEARS:
            self._primary.add(record)
        if self._long_term is not None:
            self._long_term.add(record)

    def freeze(
        self,
        *,
        input_metadata: DecompositionInputMetadata | None = None,
    ) -> ScoreDecompositionAnalysis:
        """Validate all strata and return immutable exact results."""
        return ScoreDecompositionAnalysis(
            selected_years=self.selected_years,
            selected=self._selected.freeze(),
            years=tuple(
                (year, self._years[year].freeze()) for year in self.selected_years
            ),
            primary_2020_2025=(
                None if self._primary is None else self._primary.freeze()
            ),
            long_term_2009_2025=(
                None if self._long_term is None else self._long_term.freeze()
            ),
            input_metadata=input_metadata,
        )


def decompose_score_gap(
    records: Iterable[DecompositionWinRecord],
) -> ScoreDecompositionResult:
    """Compute all eight means and an exact, order-invariant Shapley split."""
    accumulator = _ScoreAccumulator()
    record_count = 0
    for record in records:
        accumulator.add(record)
        record_count += 1
    if record_count == 0:
        raise ValueError("at least one win record is required")
    return _freeze_score_accumulator(accumulator)


def _freeze_score_accumulator(
    accumulator: _ScoreAccumulator,
) -> ScoreDecompositionResult:
    counts = accumulator.counts
    dealer_sums = accumulator.dealer_sums
    nondealer_sums = accumulator.nondealer_sums
    role_counts = {
        role: accumulator.role_count(role) for role in ("dealer", "nondealer")
    }
    if any(count <= 0 for count in role_counts.values()):
        raise ValueError("both dealer and nondealer wins are required")
    missing_strata = tuple(
        (role, method)
        for role in ("dealer", "nondealer")
        for method in ("tsumo", "ron")
        if counts[role, method] == 0
    )
    if missing_strata:
        raise ValueError(
            f"within-method score distribution is missing for strata: {missing_strata}"
        )

    means: dict[CounterfactualState, Fraction] = {}
    for payment_role in ("nondealer", "dealer"):
        for method_source in ("nondealer", "dealer"):
            for score_source in ("nondealer", "dealer"):
                mean = Fraction()
                for method in ("tsumo", "ron"):
                    method_count = counts[method_source, method]
                    if method_count == 0:
                        continue
                    score_count = counts[score_source, method]
                    if score_count == 0:
                        raise ValueError(
                            "within-method score distribution is missing for "
                            f"source={score_source}, method={method}"
                        )
                    method_weight = Fraction(method_count, role_counts[method_source])
                    score_sum = (
                        dealer_sums[score_source, method]
                        if payment_role == "dealer"
                        else nondealer_sums[score_source, method]
                    )
                    mean += method_weight * Fraction(score_sum, score_count)
                state = CounterfactualState(
                    payment_rule=payment_role,
                    method_mix=method_source,
                    within_method_score=score_source,
                )
                means[state] = mean

    observed_dealer = Fraction(
        sum(dealer_sums["dealer", method] for method in ("tsumo", "ron")),
        role_counts["dealer"],
    )
    observed_nondealer = Fraction(
        sum(nondealer_sums["nondealer", method] for method in ("tsumo", "ron")),
        role_counts["nondealer"],
    )
    dealer_state = CounterfactualState("dealer", "dealer", "dealer")
    nondealer_state = CounterfactualState("nondealer", "nondealer", "nondealer")
    if means[dealer_state] != observed_dealer:
        raise ValueError("dealer endpoint does not reproduce Article 1 mean")
    if means[nondealer_state] != observed_nondealer:
        raise ValueError("nondealer endpoint does not reproduce Article 1 mean")

    bit_for_factor = {factor: 1 << index for index, factor in enumerate(FACTORS)}

    def state_for(mask: int) -> CounterfactualState:
        roles = tuple(
            "dealer" if mask & (1 << index) else "nondealer" for index in range(3)
        )
        return CounterfactualState(*roles)

    contributions: list[ShapleyContribution] = []
    weights = {0: Fraction(1, 3), 1: Fraction(1, 6), 2: Fraction(1, 3)}
    for factor in FACTORS:
        bit = bit_for_factor[factor]
        contribution = Fraction()
        for mask in range(8):
            if mask & bit:
                continue
            contribution += weights[mask.bit_count()] * (
                means[state_for(mask | bit)] - means[state_for(mask)]
            )
        contributions.append(ShapleyContribution(factor, contribution))

    ordered_means = tuple(
        CounterfactualMean(state_for(mask), means[state_for(mask)]) for mask in range(8)
    )
    pooled_count = role_counts["dealer"] + role_counts["nondealer"]
    rounding_effects = (
        RoundingEffect(
            "dealer",
            Fraction(
                accumulator.rounding_twice_sums["dealer"],
                2 * role_counts["dealer"],
            ),
        ),
        RoundingEffect(
            "nondealer",
            Fraction(
                accumulator.rounding_twice_sums["nondealer"],
                2 * role_counts["nondealer"],
            ),
        ),
        RoundingEffect(
            "pooled",
            Fraction(
                accumulator.rounding_twice_sums["dealer"]
                + accumulator.rounding_twice_sums["nondealer"],
                2 * pooled_count,
            ),
        ),
    )
    method_counts = tuple(
        MethodWinCount(role, method, counts[role, method])
        for role in ("dealer", "nondealer")
        for method in ("tsumo", "ron")
    )
    score_distributions = tuple(
        ScoreDistributionEntry(role, tier, accumulator.tier_counts[role, tier])
        for role in ("dealer", "nondealer")
        for method in ("tsumo", "ron")
        for tier in sorted(
            (
                candidate
                for source, candidate in accumulator.tier_counts
                if source == role and candidate.method == method
            ),
            key=lambda candidate: (
                candidate.candidate_basic_points,
                candidate.dealer_points,
                candidate.nondealer_points,
            ),
        )
    )
    return ScoreDecompositionResult(
        dealer_wins=role_counts["dealer"],
        nondealer_wins=role_counts["nondealer"],
        observed_dealer_mean=observed_dealer,
        observed_nondealer_mean=observed_nondealer,
        counterfactual_means=ordered_means,
        contributions=tuple(contributions),
        rounding_effects=rounding_effects,
        method_counts=method_counts,
        score_distributions=score_distributions,
    )


def analyze_score_decomposition(
    records: Iterable[DecompositionWinRecord],
    selected_years: Iterable[int],
    *,
    input_metadata: DecompositionInputMetadata | None = None,
) -> ScoreDecompositionAnalysis:
    """Build all requested slices in one pass and fixed memory."""
    builder = ScoreDecompositionBuilder(selected_years)
    for record in records:
        builder.add(record)
    return builder.freeze(input_metadata=input_metadata)


def _fraction_document(value: Fraction) -> dict[str, int | float]:
    return {
        "value": float(value),
        "numerator": value.numerator,
        "denominator": value.denominator,
    }


def _result_document(result: ScoreDecompositionResult) -> dict[str, Any]:
    return {
        "dealer_wins": result.dealer_wins,
        "nondealer_wins": result.nondealer_wins,
        "observed_dealer_mean": _fraction_document(result.observed_dealer_mean),
        "observed_nondealer_mean": _fraction_document(result.observed_nondealer_mean),
        "observed_gap": _fraction_document(result.gap),
        "counterfactual_means": [
            {
                "payment_rule": value.state.payment_rule,
                "method_mix": value.state.method_mix,
                "within_method_score": value.state.within_method_score,
                "mean_points": _fraction_document(value.mean_points),
            }
            for value in result.counterfactual_means
        ],
        "shapley": {
            value.factor: _fraction_document(value.points)
            for value in result.contributions
        },
        "shapley_share_of_gap": {
            factor: (
                None
                if result.contribution_share(factor) is None
                else _fraction_document(result.contribution_share(factor))
            )
            for factor in FACTORS
        },
        "efficiency_residual": _fraction_document(result.efficiency_residual),
        "rounding_effect": {
            value.score_source: _fraction_document(value.mean_points)
            for value in result.rounding_effects
        },
        "method_counts": {
            role: {
                method: result.method_count(role, method) for method in ("tsumo", "ron")
            }
            for role in ("dealer", "nondealer")
        },
        "score_distributions": [
            {
                "score_source": value.score_source,
                "method": value.tier.method,
                "candidate_basic_points": list(value.tier.candidate_basic_points),
                "dealer_points": value.tier.dealer_points,
                "nondealer_points": value.tier.nondealer_points,
                "wins": value.wins,
            }
            for value in result.score_distributions
        ],
    }


def decomposition_document(analysis: ScoreDecompositionAnalysis) -> dict[str, Any]:
    """Return deterministic JSON-compatible output."""

    def slice_document(value: DecompositionSlice) -> dict[str, Any]:
        return {
            "all_riichi": _result_document(value.all_riichi),
            "normal_riichi": _result_document(value.normal_riichi),
        }

    periods: dict[str, Any] = {"selected": slice_document(analysis.selected)}
    if analysis.primary_2020_2025 is not None:
        periods["primary_2020_2025"] = slice_document(analysis.primary_2020_2025)
    if analysis.long_term_2009_2025 is not None:
        periods["long_term_2009_2025"] = slice_document(analysis.long_term_2009_2025)
    input_dataset: dict[str, Any] = {
        "analysis_name": "dealer-child-riichi-points-v1",
        "schema_version": 2,
    }
    if analysis.input_metadata is not None:
        metadata = analysis.input_metadata
        input_dataset.update(
            {
                "summary_logical_path": metadata.summary_logical_path,
                "records_logical_path": metadata.records_logical_path,
                "summary_sha256": metadata.summary_sha256,
                "records_sha256": metadata.records_sha256,
                "all_record_count": metadata.all_record_count,
                "selected_record_count": metadata.selected_record_count,
                "selected_win_count": metadata.selected_win_count,
                "converted_win_count": metadata.converted_win_count,
                "unsupported_score_count": metadata.unsupported_score_count,
                "ambiguous_score_count": metadata.ambiguous_score_count,
            }
        )
    return {
        "analysis_name": "dealer-child-score-decomposition-v1",
        "schema_version": 1,
        "input_dataset": input_dataset,
        "selected_years": list(analysis.selected_years),
        "population": "winning established riichis",
        "counterfactual": (
            "This is not full hand rescoring. Keep the inferred standard score "
            "tier and observed tsumo/ron method fixed, and swap only the "
            "dealer/nondealer payment rule."
        ),
        "score_conversion": {
            "counterfactual_unit": (
                "one observed win with its tsumo/ron method and canonical "
                "conversion tier fixed"
            ),
            "basic_point_candidates": {
                method: list(finite_basic_point_candidates(method))
                for method in ("tsumo", "ron")
            },
            "yakuman_basic_points": "8000 * k for positive integer k",
            "rounding": "ceil each payer payment independently to 100 points",
            "payment_formulas": {
                "dealer_ron": "ceil100(6 * b)",
                "dealer_tsumo": "3 * ceil100(2 * b)",
                "nondealer_ron": "ceil100(4 * b)",
                "nondealer_tsumo": "ceil100(2 * b) + 2 * ceil100(b)",
            },
        },
        "factors": list(FACTORS),
        "periods": periods,
        "years": [
            {"year": year, **slice_document(value)} for year, value in analysis.years
        ],
    }


def render_decomposition_markdown(analysis: ScoreDecompositionAnalysis) -> str:
    """Render every requested period, sensitivity, and counterfactual mean."""

    def share_text(value: Fraction | None) -> str:
        return "N/A" if value is None else f"{float(value) * 100:.3f}%"

    periods = [("selected", analysis.selected)]
    if analysis.primary_2020_2025 is not None:
        periods.append(("primary_2020_2025", analysis.primary_2020_2025))
    if analysis.long_term_2009_2025 is not None:
        periods.append(("long_term_2009_2025", analysis.long_term_2009_2025))
    lines = [
        "# 親子別リーチ打点差のShapley分解",
        "",
        f"- 対象年: {', '.join(map(str, analysis.selected_years))}",
        "- 母集団: 和了した成立リーチ",
        (
            "- 反実仮想: 完全な牌姿再採点ではなく、得点tierとツモ／ロンを固定し、"
            "親子の支払い規則だけ交換"
        ),
        "",
        (
            "| Period | Population | Dealer mean | Nondealer mean | Gap | "
            "D T/R | C T/R | R | M | B | R share | M share | B share | "
            "Residual | Rounding D | Rounding C | Rounding pooled |"
        ),
        (
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
            "---:|---:|---:|---:|---:|---:|"
        ),
    ]
    for period_name, period in periods:
        for population, value in (
            ("all_riichi", period.all_riichi),
            ("normal_riichi", period.normal_riichi),
        ):
            lines.append(
                f"| {period_name} | {population} | "
                f"{float(value.observed_dealer_mean):.3f} | "
                f"{float(value.observed_nondealer_mean):.3f} | "
                f"{float(value.gap):.3f} | "
                f"{value.method_count('dealer', 'tsumo')}/"
                f"{value.method_count('dealer', 'ron')} | "
                f"{value.method_count('nondealer', 'tsumo')}/"
                f"{value.method_count('nondealer', 'ron')} | "
                f"{float(value.contribution_for('payment_rule')):.3f} | "
                f"{float(value.contribution_for('method_mix')):.3f} | "
                f"{float(value.contribution_for('within_method_score')):.3f} | "
                f"{share_text(value.contribution_share('payment_rule'))} | "
                f"{share_text(value.contribution_share('method_mix'))} | "
                f"{share_text(value.contribution_share('within_method_score'))} | "
                f"{float(value.efficiency_residual):.3f} | "
                f"{float(value.rounding_effect_for('dealer')):.3f} | "
                f"{float(value.rounding_effect_for('nondealer')):.3f} | "
                f"{float(value.rounding_effect_for('pooled')):.3f} |"
            )
    lines.extend(["", "## 8 counterfactual means", ""])
    for period_name, period in periods:
        for population, value in (
            ("all_riichi", period.all_riichi),
            ("normal_riichi", period.normal_riichi),
        ):
            lines.extend(
                [
                    f"### {period_name} / {population}",
                    "",
                    "| Payment rule | Method mix | Score mix | Mean points |",
                    "|---|---|---|---:|",
                ]
            )
            lines.extend(
                f"| {cell.state.payment_rule} | {cell.state.method_mix} | "
                f"{cell.state.within_method_score} | "
                f"{float(cell.mean_points):.3f} |"
                for cell in value.counterfactual_means
            )
            lines.extend(
                [
                    "",
                    "#### Canonical conversion-tier distribution",
                    "",
                    (
                        "| Score source | Method | Candidate basic points | "
                        "Dealer points | Nondealer points | Wins |"
                    ),
                    "|---|---|---|---:|---:|---:|",
                ]
            )
            lines.extend(
                f"| {entry.score_source} | {entry.tier.method} | "
                f"{', '.join(map(str, entry.tier.candidate_basic_points))} | "
                f"{entry.tier.dealer_points} | {entry.tier.nondealer_points} | "
                f"{entry.wins} |"
                for entry in value.score_distributions
            )
            lines.append("")
    return "\n".join(lines)


def _reserve_output_temporary(target: Path, content: bytes) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        temporary = Path(file.name)
        file.write(content)
    return temporary


def _reserve_backup_path(target: Path) -> Path:
    """Reserve a same-directory name that does not yet exist."""
    backup = _reserve_output_temporary(target, b"")
    backup.unlink()
    return backup


def _publish_output_bundle(
    staged: tuple[Path, Path],
    targets: tuple[Path, Path],
) -> None:
    backups: list[Path | None] = []
    retained_backups: set[Path] = set()
    try:
        for target in targets:
            backups.append(_reserve_backup_path(target) if target.exists() else None)
        for target, backup in zip(targets, backups, strict=True):
            if backup is not None:
                os.replace(target, backup)
        for temporary, target in zip(staged, targets, strict=True):
            os.replace(temporary, target)
    except BaseException as publish_error:
        rollback_errors: list[BaseException] = []
        for index in reversed(range(len(targets))):
            target = targets[index]
            backup = backups[index] if index < len(backups) else None
            try:
                if backup is not None and backup.exists():
                    os.replace(backup, target)
                elif backup is None and not staged[index].exists():
                    target.unlink(missing_ok=True)
            # Rollback must survive a second interrupt, then re-raise below.
            except BaseException as rollback_error:  # noqa: BLE001
                rollback_errors.append(rollback_error)
                if backup is not None:
                    retained_backups.add(backup)
        if rollback_errors:
            retained = ", ".join(str(path) for path in sorted(retained_backups))
            raise RuntimeError(
                "decomposition output rollback failed; retained backups: "
                f"{retained or 'none'}"
            ) from publish_error
        raise
    finally:
        disposable = (
            path
            for path in backups
            if path is not None and path not in retained_backups
        )
        for path in (*staged, *disposable):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def write_decomposition_outputs(
    analysis: ScoreDecompositionAnalysis,
    json_path: str | PathLike[str],
    markdown_path: str | PathLike[str],
) -> None:
    """Stage, validate, and rollback-publish JSON and Markdown as one bundle."""
    targets = (Path(json_path), Path(markdown_path))
    if targets[0].resolve() == targets[1].resolve():
        raise ValueError("JSON and Markdown output paths must differ")
    contents = (
        (
            json.dumps(
                decomposition_document(analysis),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode(),
        render_decomposition_markdown(analysis).encode(),
    )
    staged: list[Path] = []
    try:
        for target, content in zip(targets, contents, strict=True):
            staged.append(_reserve_output_temporary(target, content))
        _publish_output_bundle((staged[0], staged[1]), targets)
        staged.clear()
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)
