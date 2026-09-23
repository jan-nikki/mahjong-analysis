"""Pure standard-payment conversion for dealer/nondealer counterfactuals."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

WinnerRole = Literal["dealer", "nondealer"]
WinMethod = Literal["tsumo", "ron"]


class ScorePatternError(ValueError):
    """Base class for fail-closed score-pattern errors."""


class UnmatchedScorePatternError(ScorePatternError):
    """Raised when an observation is not on the standard score table."""


class AmbiguousScorePatternError(ScorePatternError):
    """Raised when legal candidates imply different counterfactual scores."""


@dataclass(frozen=True, order=True)
class PaymentPattern:
    """Honba-excluded payer losses in a canonical order.

    Ron contains the single payment.  Dealer tsumo contains the three equal
    child payments.  Nondealer tsumo places the dealer payment first, followed
    by the two equal child payments.
    """

    role: WinnerRole
    method: WinMethod
    payments: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.role not in ("dealer", "nondealer"):
            raise ValueError("role must be dealer or nondealer")
        if self.method not in ("tsumo", "ron"):
            raise ValueError("method must be tsumo or ron")
        expected_length = 1 if self.method == "ron" else 3
        if len(self.payments) != expected_length:
            raise ValueError(f"{self.method} requires {expected_length} payment(s)")
        if any(
            type(payment) is not int or payment <= 0 or payment % 100 != 0
            for payment in self.payments
        ):
            raise ValueError("payments must be positive multiples of 100")
        if self.method == "tsumo":
            if self.role == "dealer" and len(set(self.payments)) != 1:
                raise ValueError("dealer tsumo requires three equal child payments")
            if self.role == "nondealer" and not (
                self.payments[1] == self.payments[2]
                and self.payments[0] >= self.payments[1]
            ):
                raise ValueError(
                    "nondealer tsumo requires dealer payment first and equal "
                    "child payments"
                )

    @property
    def total(self) -> int:
        """Return the total paid to the winner."""
        return sum(self.payments)


@dataclass(frozen=True, order=True)
class ScoreTier:
    """One fixed basic-point tier before dealer/nondealer payment rules."""

    base_points: int
    name: str

    def __post_init__(self) -> None:
        if type(self.base_points) is not int or self.base_points <= 0:
            raise ValueError("base_points must be a positive integer")
        if not self.name:
            raise ValueError("score tier name cannot be empty")


@dataclass(frozen=True)
class ConvertedScore:
    """A validated observation converted under both payment-rule roles."""

    observed: PaymentPattern
    candidates: tuple[ScoreTier, ...]
    dealer: PaymentPattern
    nondealer: PaymentPattern

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("at least one score-tier candidate is required")
        if self.dealer.role != "dealer" or self.nondealer.role != "nondealer":
            raise ValueError("converted patterns have incorrect roles")
        if self.dealer.method != self.observed.method:
            raise ValueError("dealer conversion changed the win method")
        if self.nondealer.method != self.observed.method:
            raise ValueError("nondealer conversion changed the win method")

    @property
    def observed_points(self) -> int:
        return self.observed.total

    @property
    def dealer_points(self) -> int:
        return self.dealer.total

    @property
    def nondealer_points(self) -> int:
        return self.nondealer.total

    @property
    def dealer_minus_one_point_five_nondealer(self) -> Fraction:
        """Return dealer points minus 1.5 times nondealer points exactly."""
        return Fraction(self.dealer_points) - Fraction(3, 2) * self.nondealer_points


_FU_VALUES = (20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110)
_LIMIT_TIERS = (
    ScoreTier(2_000, "mangan"),
    ScoreTier(3_000, "haneman"),
    ScoreTier(4_000, "baiman"),
    ScoreTier(6_000, "sanbaiman"),
)


def _round_up_100(points: int) -> int:
    return ((points + 99) // 100) * 100


def _nonlimit_tiers(method: WinMethod) -> tuple[ScoreTier, ...]:
    tiers: list[ScoreTier] = []
    for han in range(1, 5):
        for fu in _FU_VALUES:
            if fu == 20 and (method == "ron" or han < 2):
                continue
            if fu == 25 and han < 2:
                continue
            base_points = fu * 2 ** (han + 2)
            if base_points < 2_000:
                tiers.append(ScoreTier(base_points, f"{han}_han_{fu}_fu"))
    return tuple(sorted(tiers))


def _candidate_tiers(method: WinMethod, observed_total: int) -> tuple[ScoreTier, ...]:
    if observed_total <= 0:
        raise ValueError("observed_total must be positive")
    maximum_yakuman = max(1, observed_total // 8_000 + 1)
    yakuman_tiers = tuple(
        ScoreTier(8_000 * multiplier, f"yakuman_x{multiplier}")
        for multiplier in range(1, maximum_yakuman + 1)
    )
    return _nonlimit_tiers(method) + _LIMIT_TIERS + yakuman_tiers


def finite_basic_point_candidates(method: WinMethod) -> tuple[int, ...]:
    """Return finite non-limit and named-limit candidates for audit output.

    Yakuman is an unbounded ``8_000 * k`` family and is intentionally described
    separately by callers rather than truncated into this tuple.
    """
    if method not in ("tsumo", "ron"):
        raise ValueError("method must be tsumo or ron")
    return tuple(
        sorted({tier.base_points for tier in (*_nonlimit_tiers(method), *_LIMIT_TIERS)})
    )


def standard_payment_pattern(
    tier: ScoreTier,
    role: WinnerRole,
    method: WinMethod,
) -> PaymentPattern:
    """Return the standard honba-excluded payments for one fixed score tier."""
    if role not in ("dealer", "nondealer"):
        raise ValueError("role must be dealer or nondealer")
    if method not in ("tsumo", "ron"):
        raise ValueError("method must be tsumo or ron")
    base = tier.base_points
    if method == "ron":
        multiplier = 6 if role == "dealer" else 4
        payments = (_round_up_100(multiplier * base),)
    elif role == "dealer":
        child_payment = _round_up_100(2 * base)
        payments = (child_payment, child_payment, child_payment)
    else:
        dealer_payment = _round_up_100(2 * base)
        child_payment = _round_up_100(base)
        payments = (dealer_payment, child_payment, child_payment)
    return PaymentPattern(role=role, method=method, payments=payments)


def score_points(
    role: WinnerRole,
    method: WinMethod,
    basic_points: int,
) -> int:
    """Return total points for the standard role/method payment rule."""
    return standard_payment_pattern(
        ScoreTier(basic_points, "explicit"),
        role,
        method,
    ).total


def infer_basic_points(
    role: WinnerRole,
    method: WinMethod,
    hand_points: int,
) -> int:
    """Infer the unique standard basic-point tier from an observed total."""
    converted = convert_total_points(hand_points, role, method)
    bases = {candidate.base_points for candidate in converted.candidates}
    if len(bases) != 1:
        raise AmbiguousScorePatternError(
            "observed total maps to multiple basic-point tiers: "
            f"role={role}, method={method}, points={hand_points}, "
            f"candidates={sorted(bases)}"
        )
    return bases.pop()


def convert_payment_pattern(observed: PaymentPattern) -> ConvertedScore:
    """Match an observation and convert it under both role payment rules.

    Multiple legal basic-point candidates are accepted only when all of them
    produce exactly the same dealer and nondealer counterfactual patterns.
    """
    return resolve_payment_pattern(
        observed,
        _candidate_tiers(observed.method, observed.total),
    )


def resolve_payment_pattern(
    observed: PaymentPattern,
    candidates: tuple[ScoreTier, ...],
) -> ConvertedScore:
    """Resolve one pattern against an explicit candidate score table."""
    matches = tuple(
        tier
        for tier in candidates
        if standard_payment_pattern(tier, observed.role, observed.method) == observed
    )
    if not matches:
        raise UnmatchedScorePatternError(
            f"payment pattern is not on the standard table: {observed}"
        )
    conversions = {
        (
            standard_payment_pattern(tier, "dealer", observed.method),
            standard_payment_pattern(tier, "nondealer", observed.method),
        )
        for tier in matches
    }
    if len(conversions) != 1:
        raise AmbiguousScorePatternError(
            "legal score-tier candidates imply different dealer/nondealer "
            f"conversions: {observed}"
        )
    dealer, nondealer = conversions.pop()
    return ConvertedScore(
        observed=observed,
        candidates=matches,
        dealer=dealer,
        nondealer=nondealer,
    )


def convert_total_points(
    observed_points: int,
    role: WinnerRole,
    method: WinMethod,
) -> ConvertedScore:
    """Convert an Article 1 total when payer-level losses are unavailable.

    This convenience adapter remains fail closed: every standard pattern with
    the observed total must imply the same two counterfactual outputs.
    """
    if type(observed_points) is not int or observed_points <= 0:
        raise ValueError("observed_points must be a positive integer")
    matching_patterns = {
        standard_payment_pattern(tier, role, method)
        for tier in _candidate_tiers(method, observed_points)
        if standard_payment_pattern(tier, role, method).total == observed_points
    }
    if not matching_patterns:
        raise UnmatchedScorePatternError(
            f"point total is not on the standard table: {observed_points}"
        )
    converted = tuple(convert_payment_pattern(pattern) for pattern in matching_patterns)
    conversions = {(value.dealer, value.nondealer) for value in converted}
    if len(conversions) != 1:
        raise AmbiguousScorePatternError(
            "point total admits payer patterns with different counterfactual "
            f"conversions: role={role}, method={method}, points={observed_points}"
        )
    candidates = tuple(
        sorted(
            {tier for value in converted for tier in value.candidates},
            key=lambda tier: (tier.base_points, tier.name),
        )
    )
    dealer, nondealer = conversions.pop()
    observed = dealer if role == "dealer" else nondealer
    return ConvertedScore(
        observed=observed,
        candidates=candidates,
        dealer=dealer,
        nondealer=nondealer,
    )
