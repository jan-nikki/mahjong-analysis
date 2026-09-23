from fractions import Fraction

import pytest

from mahjong_analysis.score_table import (
    AmbiguousScorePatternError,
    PaymentPattern,
    ScoreTier,
    UnmatchedScorePatternError,
    convert_payment_pattern,
    convert_total_points,
    resolve_payment_pattern,
    standard_payment_pattern,
)


def test_low_ron_converts_in_both_directions() -> None:
    child = convert_payment_pattern(PaymentPattern("nondealer", "ron", (1000,)))
    dealer = convert_payment_pattern(PaymentPattern("dealer", "ron", (1500,)))

    assert child.nondealer_points == dealer.nondealer_points == 1000
    assert child.dealer_points == dealer.dealer_points == 1500
    assert child.dealer_minus_one_point_five_nondealer == 0


def test_tsumo_rounding_and_role_swap() -> None:
    child = convert_payment_pattern(
        PaymentPattern("nondealer", "tsumo", (500, 300, 300))
    )
    dealer = convert_payment_pattern(PaymentPattern("dealer", "tsumo", (500, 500, 500)))

    assert child.observed_points == 1100
    assert child.dealer.payments == (500, 500, 500)
    assert dealer.nondealer.payments == (500, 300, 300)
    assert child.dealer_minus_one_point_five_nondealer == Fraction(-150)


@pytest.mark.parametrize(
    ("base_points", "child_ron", "dealer_ron", "child_tsumo", "dealer_tsumo"),
    (
        (2_000, 8_000, 12_000, 8_000, 12_000),
        (3_000, 12_000, 18_000, 12_000, 18_000),
        (4_000, 16_000, 24_000, 16_000, 24_000),
        (6_000, 24_000, 36_000, 24_000, 36_000),
        (8_000, 32_000, 48_000, 32_000, 48_000),
        (16_000, 64_000, 96_000, 64_000, 96_000),
    ),
)
def test_limit_and_multiple_yakuman_payment_table(
    base_points: int,
    child_ron: int,
    dealer_ron: int,
    child_tsumo: int,
    dealer_tsumo: int,
) -> None:
    tier = ScoreTier(base_points, "test")

    assert standard_payment_pattern(tier, "nondealer", "ron").total == child_ron
    assert standard_payment_pattern(tier, "dealer", "ron").total == dealer_ron
    assert standard_payment_pattern(tier, "nondealer", "tsumo").total == child_tsumo
    assert standard_payment_pattern(tier, "dealer", "tsumo").total == dealer_tsumo


def test_same_conversion_allows_multiple_legal_han_fu_candidates() -> None:
    converted = convert_payment_pattern(PaymentPattern("nondealer", "ron", (7700,)))

    assert {(tier.base_points, tier.name) for tier in converted.candidates} == {
        (1920, "3_han_60_fu"),
        (1920, "4_han_30_fu"),
    }
    assert converted.dealer_points == 11_600


def test_different_counterfactual_candidates_fail_closed() -> None:
    with pytest.raises(AmbiguousScorePatternError):
        resolve_payment_pattern(
            PaymentPattern("nondealer", "ron", (1000,)),
            (ScoreTier(226, "candidate_a"), ScoreTier(249, "candidate_b")),
        )


def test_nonstandard_pattern_fails_closed() -> None:
    with pytest.raises(UnmatchedScorePatternError):
        convert_payment_pattern(PaymentPattern("nondealer", "ron", (900,)))


def test_total_adapter_preserves_article_one_points_when_unambiguous() -> None:
    converted = convert_total_points(12_000, "dealer", "tsumo")

    assert converted.observed_points == 12_000
    assert converted.dealer_points == 12_000
    assert converted.nondealer_points == 8_000
