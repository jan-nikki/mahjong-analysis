"""Statistical inference for reviewed yearly dealer double-riichi results."""

import json
import math
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from statistics import NormalDist
from typing import Any


FIRST_YEAR = 2009
LAST_YEAR = 2025
EXPECTED_YEARS = tuple(range(FIRST_YEAR, LAST_YEAR + 1))
EXPECTED_SCHEMA_VERSION = 1
EXPECTED_RELEASE_TAG = "v2.0.0"
CONFIDENCE_LEVEL = 0.95
_RATE_ABS_TOLERANCE = 1e-15
_COUNT_FIELDS = (
    "scanned_files",
    "target_games",
    "east_kyokus",
    "dealer_double_riichi",
    "dealer_win",
    "other_win",
    "draw",
)
_EXPECTED_ANALYSIS = {
    "rule_code": "00a9",
    "aka_flag": True,
    "bakaze": "E",
    "dealer_double_riichi": "established",
}


@dataclass(frozen=True)
class WilsonInterval:
    """Two-sided Wilson score interval."""

    lower: float
    upper: float


@dataclass(frozen=True)
class BinomialObservation:
    """One labelled binomial count used by the heterogeneity test."""

    year: int
    successes: int
    observations: int

    def __post_init__(self) -> None:
        if type(self.year) is not int:
            raise ValueError("year must be an integer")
        _validate_binomial_counts(self.successes, self.observations)

    @property
    def failures(self) -> int:
        """Return observations that are not successes."""
        return self.observations - self.successes


@dataclass(frozen=True)
class ExpectedCounts:
    """Expected counts for one row of the Pearson table."""

    year: int
    dealer_win: float
    dealer_not_win: float


@dataclass(frozen=True)
class PearsonHeterogeneityResult:
    """Pearson chi-square result and its applicability diagnostics."""

    applicable: bool
    reason: str | None
    statistic: float | None
    degrees_of_freedom: int
    p_value: float | None
    cramers_v: float | None
    minimum_expected_count: float | None
    cells_below_5: int
    cells_below_1: int
    asymptotic_conditions_met: bool
    included_years: tuple[int, ...]
    excluded_zero_observation_years: tuple[int, ...]
    expected_counts: tuple[ExpectedCounts, ...]


@dataclass(frozen=True)
class YearlySourceCount:
    """Validated integer counts from one source year."""

    year: int
    scanned_files: int
    target_games: int
    east_kyokus: int
    dealer_double_riichi: int
    dealer_win: int
    other_win: int
    draw: int


@dataclass(frozen=True)
class YearlySourceData:
    """Validated source metadata and yearly integer counts."""

    source_filename: str
    repository: str
    release_tag: str
    analysis: dict[str, Any]
    years: tuple[YearlySourceCount, ...]


@dataclass(frozen=True)
class InferenceRow:
    """Rate estimate and interval for one year."""

    year: int
    dealer_win: int
    dealer_not_win: int
    observations: int
    win_rate: float | None
    wilson_lower: float | None
    wilson_upper: float | None
    difference_from_overall: float | None


@dataclass(frozen=True)
class OverallInference:
    """Combined rate estimate and interval."""

    dealer_win: int
    dealer_not_win: int
    observations: int
    win_rate: float | None
    wilson_lower: float | None
    wilson_upper: float | None


@dataclass(frozen=True)
class YearlyInferenceSummary:
    """Complete deterministic inference result."""

    source_filename: str
    repository: str
    release_tag: str
    analysis: dict[str, Any]
    confidence_level: float
    z_value: float
    years: tuple[InferenceRow, ...]
    overall: OverallInference
    observed_rate_range: float | None
    heterogeneity: PearsonHeterogeneityResult


def _validate_binomial_counts(successes: int, observations: int) -> None:
    if type(successes) is not int or type(observations) is not int:
        raise ValueError("successes and observations must be integers")
    if successes < 0 or observations < 0:
        raise ValueError("successes and observations must not be negative")
    if successes > observations:
        raise ValueError("successes must not exceed observations")


def _z_value(confidence_level: float) -> float:
    if type(confidence_level) not in {int, float}:
        raise ValueError("confidence_level must be a number")
    value = float(confidence_level)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    return NormalDist().inv_cdf((1.0 + value) / 2.0)


def wilson_score_interval(
    successes: int,
    observations: int,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> WilsonInterval | None:
    """Return a Wilson score interval, or None for zero observations."""
    _validate_binomial_counts(successes, observations)
    z_value = _z_value(confidence_level)
    if observations == 0:
        return None

    proportion = successes / observations
    z_squared = z_value * z_value
    denominator = 1.0 + z_squared / observations
    center = (proportion + z_squared / (2.0 * observations)) / denominator
    half_width = (
        z_value
        / denominator
        * math.sqrt(
            proportion * (1.0 - proportion) / observations
            + z_squared / (4.0 * observations * observations)
        )
    )
    lower = 0.0 if successes == 0 else max(0.0, center - half_width)
    upper = 1.0 if successes == observations else min(1.0, center + half_width)
    return WilsonInterval(lower=lower, upper=upper)


def chi_square_survival_probability(
    statistic: float,
    degrees_of_freedom: int,
) -> float:
    """Return the chi-square survival probability for an integer df."""
    if type(statistic) not in {int, float}:
        raise ValueError("chi-square statistic must be a number")
    statistic_value = float(statistic)
    if math.isnan(statistic_value) or statistic_value < 0.0:
        raise ValueError("chi-square statistic must not be negative or NaN")
    if type(degrees_of_freedom) is not int or degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be a positive integer")
    if statistic_value == 0.0:
        return 1.0
    if math.isinf(statistic_value):
        return 0.0

    half_statistic = statistic_value / 2.0
    if half_statistic == 0.0:
        return 1.0
    if degrees_of_freedom % 2 == 0:
        shape = degrees_of_freedom // 2
        probability = sum(
            math.exp(
                -half_statistic
                + index * math.log(half_statistic)
                - math.lgamma(index + 1.0)
            )
            for index in range(shape)
        )
    else:
        probability = math.erfc(math.sqrt(half_statistic))
        for index in range((degrees_of_freedom - 1) // 2):
            shape = index + 0.5
            probability += math.exp(
                shape * math.log(half_statistic)
                - half_statistic
                - math.lgamma(shape + 1.0)
            )

    return min(1.0, max(0.0, probability))


def pearson_heterogeneity_test(
    observations: Iterable[BinomialObservation],
) -> PearsonHeterogeneityResult:
    """Test homogeneity of binomial rates with Pearson's chi-square."""
    values = tuple(observations)
    years = [value.year for value in values]
    if len(set(years)) != len(years):
        raise ValueError("observation years must be unique")

    included = tuple(value for value in values if value.observations > 0)
    excluded = tuple(value.year for value in values if value.observations == 0)
    degrees_of_freedom = max(0, len(included) - 1)
    if not included:
        return PearsonHeterogeneityResult(
            applicable=False,
            reason="no years have observations",
            statistic=None,
            degrees_of_freedom=degrees_of_freedom,
            p_value=None,
            cramers_v=None,
            minimum_expected_count=None,
            cells_below_5=0,
            cells_below_1=0,
            asymptotic_conditions_met=False,
            included_years=(),
            excluded_zero_observation_years=excluded,
            expected_counts=(),
        )

    total_observations = sum(value.observations for value in included)
    total_successes = sum(value.successes for value in included)
    total_failures = total_observations - total_successes
    expected_counts = tuple(
        ExpectedCounts(
            year=value.year,
            dealer_win=value.observations * total_successes / total_observations,
            dealer_not_win=value.observations * total_failures / total_observations,
        )
        for value in included
    )
    expected_values = tuple(
        count
        for value in expected_counts
        for count in (value.dealer_win, value.dealer_not_win)
    )
    minimum_expected_count = min(expected_values)
    cells_below_5 = sum(value < 5.0 for value in expected_values)
    cells_below_1 = sum(value < 1.0 for value in expected_values)
    asymptotic_conditions_met = (
        cells_below_1 == 0 and cells_below_5 / len(expected_values) <= 0.2
    )
    common_values = {
        "degrees_of_freedom": degrees_of_freedom,
        "minimum_expected_count": minimum_expected_count,
        "cells_below_5": cells_below_5,
        "cells_below_1": cells_below_1,
        "asymptotic_conditions_met": asymptotic_conditions_met,
        "included_years": tuple(value.year for value in included),
        "excluded_zero_observation_years": excluded,
        "expected_counts": expected_counts,
    }

    if len(included) < 2:
        return PearsonHeterogeneityResult(
            applicable=False,
            reason="fewer than two years have observations",
            statistic=None,
            p_value=None,
            cramers_v=None,
            **common_values,
        )
    if total_successes == 0 or total_failures == 0:
        return PearsonHeterogeneityResult(
            applicable=False,
            reason="one outcome column has a zero total",
            statistic=None,
            p_value=None,
            cramers_v=None,
            **common_values,
        )

    statistic = 0.0
    for observed, expected in zip(included, expected_counts, strict=True):
        statistic += (
            observed.successes - expected.dealer_win
        ) ** 2 / expected.dealer_win
        statistic += (
            observed.failures - expected.dealer_not_win
        ) ** 2 / expected.dealer_not_win
    return PearsonHeterogeneityResult(
        applicable=True,
        reason=None,
        statistic=statistic,
        p_value=chi_square_survival_probability(
            statistic,
            degrees_of_freedom,
        ),
        cramers_v=math.sqrt(statistic / total_observations),
        **common_values,
    )


def _nonnegative_integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _validate_stored_rate(
    value: object,
    successes: int,
    observations: int,
    label: str,
) -> None:
    if observations == 0:
        if value is not None:
            raise ValueError(f"{label} must be null with zero observations")
        return
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
    expected = successes / observations
    if not math.isclose(
        float(value),
        expected,
        rel_tol=_RATE_ABS_TOLERANCE,
        abs_tol=_RATE_ABS_TOLERANCE,
    ):
        raise ValueError(f"{label} does not match integer counts")


def _source_count(value: object, expected_year: int) -> YearlySourceCount:
    if not isinstance(value, dict):
        raise ValueError(f"year {expected_year} must be an object")
    year = _nonnegative_integer(value.get("year"), "year")
    if year != expected_year:
        raise ValueError(f"year entries must be ordered 2009 through 2025: {year}")
    counts = {
        field: _nonnegative_integer(
            value.get(field),
            f"year {year} {field}",
        )
        for field in _COUNT_FIELDS
    }
    if (
        counts["dealer_win"] + counts["other_win"] + counts["draw"]
        != (counts["dealer_double_riichi"])
    ):
        raise ValueError(f"year {year} result counts are inconsistent")
    _validate_stored_rate(
        value.get("win_rate"),
        counts["dealer_win"],
        counts["dealer_double_riichi"],
        f"year {year} win_rate",
    )
    return YearlySourceCount(year=year, **counts)


def load_yearly_source(
    path: str | PathLike[str],
) -> YearlySourceData:
    """Load and strictly validate a reviewed yearly aggregate JSON file."""
    source_path = Path(path)
    with source_path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("yearly source must be a JSON object")
    if type(data.get("schema_version")) is not int or data["schema_version"] != (
        EXPECTED_SCHEMA_VERSION
    ):
        raise ValueError("yearly source schema_version must be 1")

    dataset = data.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("yearly source dataset must be an object")
    repository = dataset.get("repository")
    release_tag = dataset.get("release_tag")
    if not isinstance(repository, str) or not repository:
        raise ValueError("dataset repository must be a non-empty string")
    if release_tag != EXPECTED_RELEASE_TAG:
        raise ValueError("dataset release_tag must be 'v2.0.0'")

    analysis = data.get("analysis")
    if not isinstance(analysis, dict):
        raise ValueError("yearly source analysis must be an object")
    for field, expected in _EXPECTED_ANALYSIS.items():
        if field == "aka_flag":
            valid = analysis.get(field) is expected
        else:
            valid = analysis.get(field) == expected
        if not valid:
            raise ValueError(f"unexpected analysis condition: {field}")

    selected_years_value = data.get("selected_years")
    if not isinstance(selected_years_value, list) or any(
        type(year) is not int for year in selected_years_value
    ):
        raise ValueError("selected_years must be a list of integers")
    selected_years = tuple(selected_years_value)
    if len(set(selected_years)) != len(selected_years):
        raise ValueError("selected_years must not contain duplicates")
    if selected_years != EXPECTED_YEARS:
        raise ValueError("selected_years must be exactly 2009 through 2025")

    year_values = data.get("years")
    if not isinstance(year_values, list) or len(year_values) != len(EXPECTED_YEARS):
        raise ValueError("years must contain exactly 17 entries")
    years = tuple(
        _source_count(value, expected_year)
        for value, expected_year in zip(
            year_values,
            EXPECTED_YEARS,
            strict=True,
        )
    )

    totals = data.get("totals")
    if not isinstance(totals, dict):
        raise ValueError("totals must be an object")
    total_counts = {
        field: _nonnegative_integer(
            totals.get(field),
            f"totals {field}",
        )
        for field in _COUNT_FIELDS
    }
    for field, total in total_counts.items():
        calculated = sum(getattr(year, field) for year in years)
        if total != calculated:
            raise ValueError(
                f"totals {field} does not equal yearly sum: {total} != {calculated}"
            )
    if (
        total_counts["dealer_win"] + total_counts["other_win"] + total_counts["draw"]
        != total_counts["dealer_double_riichi"]
    ):
        raise ValueError("total result counts are inconsistent")
    _validate_stored_rate(
        totals.get("win_rate"),
        total_counts["dealer_win"],
        total_counts["dealer_double_riichi"],
        "totals win_rate",
    )
    return YearlySourceData(
        source_filename=source_path.name,
        repository=repository,
        release_tag=release_tag,
        analysis=dict(analysis),
        years=years,
    )


def _rate_and_interval(
    successes: int,
    observations: int,
) -> tuple[float | None, WilsonInterval | None]:
    interval = wilson_score_interval(successes, observations)
    rate = None if observations == 0 else successes / observations
    return rate, interval


def analyze_yearly_source(source: YearlySourceData) -> YearlyInferenceSummary:
    """Calculate yearly intervals and the omnibus heterogeneity test."""
    overall_successes = sum(year.dealer_win for year in source.years)
    overall_observations = sum(year.dealer_double_riichi for year in source.years)
    overall_rate, overall_interval = _rate_and_interval(
        overall_successes,
        overall_observations,
    )
    rows: list[InferenceRow] = []
    rates: list[float] = []
    observations: list[BinomialObservation] = []
    for year in source.years:
        count = year.dealer_double_riichi
        rate, interval = _rate_and_interval(year.dealer_win, count)
        if rate is not None:
            rates.append(rate)
        rows.append(
            InferenceRow(
                year=year.year,
                dealer_win=year.dealer_win,
                dealer_not_win=count - year.dealer_win,
                observations=count,
                win_rate=rate,
                wilson_lower=None if interval is None else interval.lower,
                wilson_upper=None if interval is None else interval.upper,
                difference_from_overall=(
                    None
                    if rate is None or overall_rate is None
                    else rate - overall_rate
                ),
            )
        )
        observations.append(
            BinomialObservation(
                year=year.year,
                successes=year.dealer_win,
                observations=count,
            )
        )

    return YearlyInferenceSummary(
        source_filename=source.source_filename,
        repository=source.repository,
        release_tag=source.release_tag,
        analysis=dict(source.analysis),
        confidence_level=CONFIDENCE_LEVEL,
        z_value=_z_value(CONFIDENCE_LEVEL),
        years=tuple(rows),
        overall=OverallInference(
            dealer_win=overall_successes,
            dealer_not_win=overall_observations - overall_successes,
            observations=overall_observations,
            win_rate=overall_rate,
            wilson_lower=(None if overall_interval is None else overall_interval.lower),
            wilson_upper=(None if overall_interval is None else overall_interval.upper),
        ),
        observed_rate_range=(max(rates) - min(rates) if rates else None),
        heterogeneity=pearson_heterogeneity_test(observations),
    )


def analyze_yearly_file(
    path: str | PathLike[str],
) -> YearlyInferenceSummary:
    """Load a reviewed aggregate and calculate its yearly inference."""
    return analyze_yearly_source(load_yearly_source(path))


def _inference_row_to_dict(row: InferenceRow) -> dict[str, Any]:
    return {
        "year": row.year,
        "dealer_win": row.dealer_win,
        "dealer_not_win": row.dealer_not_win,
        "observations": row.observations,
        "win_rate": row.win_rate,
        "wilson_lower": row.wilson_lower,
        "wilson_upper": row.wilson_upper,
        "difference_from_overall": row.difference_from_overall,
    }


def inference_summary_to_dict(
    summary: YearlyInferenceSummary,
) -> dict[str, Any]:
    """Convert an inference summary to its canonical JSON structure."""
    heterogeneity = summary.heterogeneity
    return {
        "schema_version": 1,
        "source": {
            "result_file": summary.source_filename,
            "dataset": {
                "repository": summary.repository,
                "release_tag": summary.release_tag,
            },
            "analysis": dict(summary.analysis),
        },
        "confidence": {
            "level": summary.confidence_level,
            "method": "wilson_score",
            "z_value": summary.z_value,
        },
        "years": [_inference_row_to_dict(row) for row in summary.years],
        "overall": {
            "dealer_win": summary.overall.dealer_win,
            "dealer_not_win": summary.overall.dealer_not_win,
            "observations": summary.overall.observations,
            "win_rate": summary.overall.win_rate,
            "wilson_lower": summary.overall.wilson_lower,
            "wilson_upper": summary.overall.wilson_upper,
        },
        "observed_rate_range": summary.observed_rate_range,
        "heterogeneity": {
            "method": "pearson_chi_square",
            "yates_correction": False,
            "applicable": heterogeneity.applicable,
            "reason": heterogeneity.reason,
            "statistic": heterogeneity.statistic,
            "degrees_of_freedom": heterogeneity.degrees_of_freedom,
            "p_value": heterogeneity.p_value,
            "cramers_v": heterogeneity.cramers_v,
            "minimum_expected_count": heterogeneity.minimum_expected_count,
            "cells_below_5": heterogeneity.cells_below_5,
            "cells_below_1": heterogeneity.cells_below_1,
            "asymptotic_conditions_met": (heterogeneity.asymptotic_conditions_met),
            "included_years": list(heterogeneity.included_years),
            "excluded_zero_observation_years": list(
                heterogeneity.excluded_zero_observation_years
            ),
            "expected_counts": [
                {
                    "year": value.year,
                    "dealer_win": value.dealer_win,
                    "dealer_not_win": value.dealer_not_win,
                }
                for value in heterogeneity.expected_counts
            ],
        },
    }


def render_inference_markdown(summary: YearlyInferenceSummary) -> str:
    """Render a deterministic human-readable inference report."""

    def percentage(value: float | None, *, signed: bool = False) -> str:
        if value is None:
            return "N/A"
        return f"{value:+.2%}" if signed else f"{value:.2%}"

    def interval(row: InferenceRow | OverallInference) -> str:
        if row.wilson_lower is None or row.wilson_upper is None:
            return "N/A"
        return f"[{row.wilson_lower:.2%}, {row.wilson_upper:.2%}]"

    def p_value(value: float | None) -> str:
        if value is None:
            return "N/A"
        if 0.0 < value < 1e-6:
            return f"{value:.3e}"
        return f"{value:.6f}"

    lines = [
        "# Dealer double-riichi yearly inference",
        "",
        f"- Source: `{summary.source_filename}`",
        f"- Dataset: `{summary.repository}` `{summary.release_tag}`",
        "- Interval: two-sided 95% Wilson score interval",
        "- Heterogeneity: Pearson chi-square without Yates correction",
        "",
        "| Year | Dealer win | Dealer not win | Observations | Win rate | "
        "95% Wilson CI | Difference from overall |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.years:
        lines.append(
            f"| {row.year} | {row.dealer_win} | {row.dealer_not_win} | "
            f"{row.observations} | {percentage(row.win_rate)} | "
            f"{interval(row)} | "
            f"{percentage(row.difference_from_overall, signed=True)} |"
        )
    overall = summary.overall
    overall_difference = "N/A" if overall.win_rate is None else "0.00%"
    lines.extend(
        [
            f"| **Overall** | **{overall.dealer_win}** | "
            f"**{overall.dealer_not_win}** | **{overall.observations}** | "
            f"**{percentage(overall.win_rate)}** | **{interval(overall)}** | "
            f"**{overall_difference}** |",
            "",
            "## Heterogeneity",
            "",
        ]
    )
    test = summary.heterogeneity
    if test.applicable:
        lines.extend(
            [
                f"- Chi-square: {test.statistic:.6f}",
                f"- Degrees of freedom: {test.degrees_of_freedom}",
                f"- p-value: {p_value(test.p_value)}",
                f"- Cramér's V: {test.cramers_v:.6f}",
            ]
        )
    else:
        lines.extend(
            [
                "- Test: N/A",
                f"- Reason: {test.reason}",
                f"- Degrees of freedom: {test.degrees_of_freedom}",
            ]
        )
    minimum_expected = (
        "N/A"
        if test.minimum_expected_count is None
        else f"{test.minimum_expected_count:.6f}"
    )
    excluded = (
        "none"
        if not test.excluded_zero_observation_years
        else ", ".join(map(str, test.excluded_zero_observation_years))
    )
    lines.extend(
        [
            f"- Minimum expected count: {minimum_expected}",
            f"- Cells below 5: {test.cells_below_5}",
            f"- Cells below 1: {test.cells_below_1}",
            "- Asymptotic conditions met: "
            f"{'yes' if test.asymptotic_conditions_met else 'no'}",
            f"- Zero-observation years excluded: {excluded}",
            "",
            "## Interpretation limits",
            "",
            "- Confidence-interval overlap is not used to decide statistical "
            "significance.",
            "- The p-value does not measure the practical size of yearly "
            "differences; yearly deviations, the observed range, and "
            "Cramér's V must also be considered.",
            "- A p-value above 0.05 means there is insufficient evidence to "
            "reject homogeneity, not that yearly differences do not exist.",
            "- The aggregate input cannot evaluate dependence among rounds "
            "from the same game or player.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            file.write(content.encode("utf-8"))
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_inference_outputs(
    summary: YearlyInferenceSummary,
    json_path: str | PathLike[str],
    markdown_path: str | PathLike[str],
) -> None:
    """Write deterministic JSON and Markdown reports."""
    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    if json_target.resolve() == markdown_target.resolve():
        raise ValueError("JSON and Markdown output paths must be different")
    json_content = (
        json.dumps(
            inference_summary_to_dict(summary),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    markdown_content = render_inference_markdown(summary)
    _atomic_write_text(json_target, json_content)
    _atomic_write_text(markdown_target, markdown_content)
