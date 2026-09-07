import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest

from analysis.dealer_double_riichi_yearly_inference import main
from mahjong_analysis.yearly_inference import (
    BinomialObservation,
    analyze_yearly_file,
    chi_square_survival_probability,
    inference_summary_to_dict,
    load_yearly_source,
    pearson_heterogeneity_test,
    render_inference_markdown,
    wilson_score_interval,
    write_inference_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKED_YEARLY_RESULTS = (
    PROJECT_ROOT / "research" / "results" / "dealer-double-riichi-v2.0.0-yearly.json"
)


def _source_document() -> dict[str, Any]:
    years = [
        {
            "year": year,
            "scanned_files": 1,
            "target_games": 1,
            "east_kyokus": 2,
            "dealer_double_riichi": 10,
            "dealer_win": 5,
            "other_win": 3,
            "draw": 2,
            "win_rate": 0.5,
        }
        for year in range(2009, 2026)
    ]
    return {
        "schema_version": 1,
        "dataset": {
            "repository": "NikkeTryHard/tenhou-to-mjai",
            "release_tag": "v2.0.0",
        },
        "analysis": {
            "rule_code": "00a9",
            "aka_flag": True,
            "bakaze": "E",
            "dealer_double_riichi": "established",
        },
        "selected_years": list(range(2009, 2026)),
        "years": years,
        "totals": {
            "scanned_files": 17,
            "target_games": 17,
            "east_kyokus": 34,
            "dealer_double_riichi": 170,
            "dealer_win": 85,
            "other_win": 51,
            "draw": 34,
            "win_rate": 0.5,
        },
    }


def _write_source(
    path: Path,
    document: dict[str, Any] | None = None,
) -> Path:
    path.write_text(
        json.dumps(
            _source_document() if document is None else document,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def test_wilson_score_interval_for_fifty_of_one_hundred() -> None:
    interval = wilson_score_interval(50, 100)

    assert interval is not None
    assert interval.lower == pytest.approx(0.4038315303659957)
    assert interval.upper == pytest.approx(0.5961684696340044)


def test_wilson_score_interval_handles_boundaries() -> None:
    no_successes = wilson_score_interval(0, 10)
    all_successes = wilson_score_interval(10, 10)

    assert no_successes is not None
    assert no_successes.lower == 0.0
    assert no_successes.upper == pytest.approx(0.2775327998628892)
    assert all_successes is not None
    assert all_successes.lower == pytest.approx(0.7224672001371106)
    assert all_successes.upper == 1.0


def test_wilson_score_interval_returns_none_for_no_observations() -> None:
    assert wilson_score_interval(0, 0) is None


@pytest.mark.parametrize(
    ("successes", "observations"),
    [
        (-1, 10),
        (0, -1),
        (11, 10),
        (True, 10),
        (1, False),
        (1.0, 10),
        (1, 10.0),
    ],
)
def test_wilson_score_interval_rejects_invalid_counts(
    successes: object,
    observations: object,
) -> None:
    with pytest.raises(ValueError):
        wilson_score_interval(successes, observations)  # type: ignore[arg-type]


def test_pearson_test_returns_zero_and_one_for_equal_rates() -> None:
    result = pearson_heterogeneity_test(
        [
            BinomialObservation(2009, 5, 10),
            BinomialObservation(2010, 10, 20),
            BinomialObservation(2011, 15, 30),
        ]
    )

    assert result.applicable is True
    assert result.statistic == 0.0
    assert result.degrees_of_freedom == 2
    assert result.p_value == 1.0
    assert result.cramers_v == 0.0


def test_pearson_test_matches_known_three_year_example() -> None:
    result = pearson_heterogeneity_test(
        [
            BinomialObservation(2009, 9, 10),
            BinomialObservation(2010, 5, 10),
            BinomialObservation(2011, 1, 10),
        ]
    )

    assert result.applicable is True
    assert result.statistic == pytest.approx(12.8)
    assert result.degrees_of_freedom == 2
    assert result.p_value == pytest.approx(math.exp(-6.4))
    assert result.cramers_v == pytest.approx(math.sqrt(12.8 / 30))
    assert result.minimum_expected_count == 5.0
    assert result.cells_below_5 == 0
    assert result.cells_below_1 == 0
    assert result.asymptotic_conditions_met is True


def test_chi_square_survival_probability_supports_even_df() -> None:
    assert chi_square_survival_probability(
        9.487729036781154,
        4,
    ) == pytest.approx(0.05, abs=1e-12)


def test_chi_square_survival_probability_supports_odd_df() -> None:
    assert chi_square_survival_probability(
        7.814727903251179,
        3,
    ) == pytest.approx(0.05, abs=1e-12)


@pytest.mark.parametrize("degrees_of_freedom", [2, 3])
def test_chi_square_survival_probability_handles_underflowed_half_statistic(
    degrees_of_freedom: int,
) -> None:
    smallest_positive_float = float.fromhex("0x0.0000000000001p-1022")

    assert smallest_positive_float > 0.0
    assert smallest_positive_float / 2.0 == 0.0
    assert (
        chi_square_survival_probability(
            smallest_positive_float,
            degrees_of_freedom,
        )
        == 1.0
    )


@pytest.mark.parametrize(
    ("statistic", "degrees_of_freedom"),
    [(-1.0, 1), (math.nan, 1), (True, 1), (1.0, 0), (1.0, True)],
)
def test_chi_square_survival_probability_rejects_invalid_input(
    statistic: object,
    degrees_of_freedom: object,
) -> None:
    with pytest.raises(ValueError):
        chi_square_survival_probability(  # type: ignore[arg-type]
            statistic,
            degrees_of_freedom,
        )


def test_pearson_test_excludes_zero_observation_years() -> None:
    result = pearson_heterogeneity_test(
        [
            BinomialObservation(2009, 0, 0),
            BinomialObservation(2010, 5, 10),
            BinomialObservation(2011, 5, 10),
        ]
    )

    assert result.applicable is True
    assert result.included_years == (2010, 2011)
    assert result.excluded_zero_observation_years == (2009,)
    assert result.degrees_of_freedom == 1
    assert result.statistic == 0.0
    assert result.p_value == 1.0


def test_expected_count_diagnostics_detect_cells_below_one() -> None:
    result = pearson_heterogeneity_test(
        [
            BinomialObservation(2009, 0, 1),
            BinomialObservation(2010, 5, 9),
        ]
    )

    assert result.minimum_expected_count == 0.5
    assert result.cells_below_1 == 2
    assert result.cells_below_5 == 4
    assert result.asymptotic_conditions_met is False


def test_expected_count_diagnostics_detect_cells_below_five() -> None:
    result = pearson_heterogeneity_test(
        [
            BinomialObservation(2009, 2, 4),
            BinomialObservation(2010, 8, 16),
        ]
    )

    assert result.minimum_expected_count == 2.0
    assert result.cells_below_1 == 0
    assert result.cells_below_5 == 2
    assert result.asymptotic_conditions_met is False


@pytest.mark.parametrize(
    ("year_count", "expected_fraction", "conditions_met"),
    [
        (6, 2 / 12, True),
        (5, 2 / 10, True),
        (4, 2 / 8, False),
    ],
)
def test_expected_count_diagnostics_apply_twenty_percent_boundary(
    year_count: int,
    expected_fraction: float,
    conditions_met: bool,
) -> None:
    observations = [BinomialObservation(2009, 4, 8)]
    observations.extend(
        BinomialObservation(2009 + index, 5, 10) for index in range(1, year_count)
    )

    result = pearson_heterogeneity_test(observations)
    total_cells = year_count * 2

    assert result.cells_below_1 == 0
    assert result.cells_below_5 == 2
    assert result.cells_below_5 / total_cells == expected_fraction
    assert result.asymptotic_conditions_met is conditions_met


@pytest.mark.parametrize("successes", [0, 10])
def test_pearson_test_reports_single_outcome_as_not_applicable(
    successes: int,
) -> None:
    result = pearson_heterogeneity_test(
        [
            BinomialObservation(2009, successes, 10),
            BinomialObservation(2010, successes, 10),
        ]
    )

    assert result.applicable is False
    assert result.reason == "one outcome column has a zero total"
    assert result.statistic is None
    assert result.p_value is None
    assert result.cramers_v is None
    assert result.minimum_expected_count == 0.0
    assert result.asymptotic_conditions_met is False


def test_analysis_uses_integer_totals_and_calculates_yearly_values(
    tmp_path: Path,
) -> None:
    summary = analyze_yearly_file(_write_source(tmp_path / "yearly.json"))

    assert len(summary.years) == 17
    assert summary.overall.dealer_win == 85
    assert summary.overall.dealer_not_win == 85
    assert summary.overall.observations == 170
    assert summary.overall.win_rate == 0.5
    assert summary.observed_rate_range == 0.0
    assert all(row.dealer_not_win == 5 for row in summary.years)
    assert all(row.difference_from_overall == 0.0 for row in summary.years)


def test_analysis_represents_zero_observation_year_as_undefined(
    tmp_path: Path,
) -> None:
    document = _source_document()
    document["years"][0].update(
        {
            "dealer_double_riichi": 0,
            "dealer_win": 0,
            "other_win": 0,
            "draw": 0,
            "win_rate": None,
        }
    )
    document["totals"].update(
        {
            "dealer_double_riichi": 160,
            "dealer_win": 80,
            "other_win": 48,
            "draw": 32,
            "win_rate": 0.5,
        }
    )

    summary = analyze_yearly_file(_write_source(tmp_path / "zero-year.json", document))

    row = summary.years[0]
    assert row.win_rate is None
    assert row.wilson_lower is None
    assert row.wilson_upper is None
    assert row.difference_from_overall is None
    assert summary.heterogeneity.excluded_zero_observation_years == (2009,)


def _invalid_document(case: str) -> dict[str, Any]:
    document = copy.deepcopy(_source_document())
    if case == "schema":
        document["schema_version"] = True
    elif case == "release":
        document["dataset"]["release_tag"] = "v1.0.0"
    elif case == "analysis":
        document["analysis"]["aka_flag"] = 1
    elif case == "missing_year":
        document["selected_years"].pop()
    elif case == "duplicate_year":
        document["selected_years"][-1] = 2024
    elif case == "unordered_rows":
        document["years"][0]["year"] = 2010
    elif case == "bool_count":
        document["years"][0]["dealer_win"] = True
    elif case == "negative_count":
        document["years"][0]["dealer_win"] = -1
    elif case == "year_result_sum":
        document["years"][0]["draw"] = 3
    elif case == "year_rate":
        document["years"][0]["win_rate"] = 0.6
    elif case == "total_count":
        document["totals"]["dealer_win"] = 84
    elif case == "total_rate":
        document["totals"]["win_rate"] = 0.6
    else:
        raise AssertionError(f"unknown case: {case}")
    return document


@pytest.mark.parametrize(
    "case",
    [
        "schema",
        "release",
        "analysis",
        "missing_year",
        "duplicate_year",
        "unordered_rows",
        "bool_count",
        "negative_count",
        "year_result_sum",
        "year_rate",
        "total_count",
        "total_rate",
    ],
)
def test_input_validation_rejects_inconsistent_source(
    tmp_path: Path,
    case: str,
) -> None:
    source = _write_source(tmp_path / "invalid.json", _invalid_document(case))

    with pytest.raises(ValueError):
        load_yearly_source(source)


def test_json_and_markdown_are_consistent_and_deterministic(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "source.json")
    json_output = tmp_path / "out" / "inference.json"
    markdown_output = tmp_path / "out" / "inference.md"

    assert (
        main(
            [
                "--input",
                str(source),
                "--output-json",
                str(json_output),
                "--output-markdown",
                str(markdown_output),
            ]
        )
        == 0
    )
    first_json = json_output.read_text(encoding="utf-8")
    first_markdown = markdown_output.read_text(encoding="utf-8")

    assert (
        main(
            [
                "--input",
                str(source),
                "--output-json",
                str(json_output),
                "--output-markdown",
                str(markdown_output),
            ]
        )
        == 0
    )
    assert json_output.read_text(encoding="utf-8") == first_json
    assert markdown_output.read_text(encoding="utf-8") == first_markdown

    data = json.loads(first_json)
    assert data["years"][0]["dealer_win"] == 5
    assert data["years"][0]["dealer_not_win"] == 5
    assert data["years"][0]["win_rate"] == 0.5
    assert data["overall"]["dealer_win"] == 85
    assert data["overall"]["observations"] == 170
    assert data["overall"]["win_rate"] == 0.5
    assert "| 2009 | 5 | 5 | 10 | 50.00% |" in first_markdown
    assert "| **Overall** | **85** | **85** | **170** | **50.00%**" in (first_markdown)
    assert "Degrees of freedom: 16" in first_markdown
    assert str(tmp_path.resolve()) not in first_json
    assert str(tmp_path.resolve()) not in first_markdown
    for forbidden in ("generated_at", "created_at", "duration", "elapsed"):
        assert forbidden not in first_json
        assert forbidden not in first_markdown


def test_output_paths_must_be_different(tmp_path: Path) -> None:
    summary = analyze_yearly_file(_write_source(tmp_path / "source.json"))
    output = tmp_path / "same-output"

    with pytest.raises(ValueError, match="must be different"):
        write_inference_outputs(summary, output, output)


def test_summary_serializers_describe_the_same_result(tmp_path: Path) -> None:
    summary = analyze_yearly_file(_write_source(tmp_path / "source.json"))
    data = inference_summary_to_dict(summary)
    markdown = render_inference_markdown(summary)

    assert data["heterogeneity"]["statistic"] == 0.0
    assert data["heterogeneity"]["p_value"] == 1.0
    assert "Chi-square: 0.000000" in markdown
    assert "p-value: 1.000000" in markdown
    assert "Cramér's V: 0.000000" in markdown


def test_tracked_yearly_result_is_valid_and_analyzable() -> None:
    source = load_yearly_source(TRACKED_YEARLY_RESULTS)
    summary = analyze_yearly_file(TRACKED_YEARLY_RESULTS)

    assert source.release_tag == "v2.0.0"
    assert len(source.years) == 17
    assert source.years[0].year == 2009
    assert source.years[-1].year == 2025
    assert summary.overall.observations == 9398
    assert summary.overall.dealer_win == 6607
    assert summary.heterogeneity.applicable is True
    assert summary.heterogeneity.degrees_of_freedom == 16
    assert summary.heterogeneity.minimum_expected_count == pytest.approx(
        6.83049585018089
    )
    assert summary.heterogeneity.asymptotic_conditions_met is True
