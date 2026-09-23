"""Descriptive and standardized riichi-versus-dama kokushi comparisons."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any


def analyze_decision_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Analyze extracted decisions without claiming a causal riichi effect."""
    raw_records = document.get("records")
    if not isinstance(raw_records, list):
        raise TypeError("decision document records must be a list")
    records = [_validate_record(value) for value in raw_records]
    eligible = [record for record in records if record["riichi_eligible"] is True]
    immediate = [
        record for record in eligible if record["strategy"] == "immediate_riichi"
    ]
    initial_dama = [
        record for record in eligible if record["strategy"] != "immediate_riichi"
    ]
    never_riichi = [record for record in eligible if record["strategy"] == "dama"]
    delayed = [record for record in eligible if record["strategy"] == "delayed_riichi"]

    primary = _comparison(immediate, initial_dama, "immediate_riichi", "initial_dama")
    never = _comparison(immediate, never_riichi, "immediate_riichi", "never_riichi")
    return {
        "source": {
            "decision_count": len(records),
            "eligible_decision_count": len(eligible),
            "scope": document.get("scope"),
        },
        "interpretation": {
            "primary_estimand": (
                "Outcome after choosing immediate riichi versus initially staying dama "
                "at the first post-discard kokushi tenpai. The initial-dama group includes "
                "players who declared a kokushi riichi later."
            ),
            "warning": (
                "This is an observational comparison. Standardization balances only the "
                "recorded strata and does not identify a causal riichi effect."
            ),
        },
        "all_decisions": _metrics(records),
        "eligible_strategies": {
            "immediate_riichi": _metrics(immediate),
            "delayed_riichi": _metrics(delayed),
            "never_riichi": _metrics(never_riichi),
        },
        "by_wait_group": {
            wait_group: _comparison(
                [record for record in immediate if record["wait_group"] == wait_group],
                [
                    record
                    for record in initial_dama
                    if record["wait_group"] == wait_group
                ],
                "immediate_riichi",
                "initial_dama",
            )
            for wait_group in ("honor_single", "terminal_single", "thirteen_sided")
        },
        "by_era": {
            era: _comparison(
                [record for record in immediate if _era(record["year"]) == era],
                [record for record in initial_dama if _era(record["year"]) == era],
                "immediate_riichi",
                "initial_dama",
            )
            for era in ("2009-2014", "2015-2019", "2020-2025")
        },
        "primary_immediate_vs_initial_dama": primary,
        "sensitivity_immediate_vs_never_riichi": never,
    }


def _comparison(
    treated: Sequence[dict[str, Any]],
    control: Sequence[dict[str, Any]],
    treated_name: str,
    control_name: str,
) -> dict[str, Any]:
    return {
        "groups": {
            treated_name: _metrics(treated),
            control_name: _metrics(control),
        },
        "raw_differences_percentage_points": _rate_differences(treated, control),
        "standardized": {
            "wait_turn_unseen": _standardize(
                treated,
                control,
                lambda record: (
                    record["wait_group"],
                    _turn_band(record["turn"]),
                    _unseen_bucket(record["unseen_wait_counts"]),
                ),
            ),
            "extended_recorded_state": _standardize(
                treated,
                control,
                lambda record: (
                    record["wait_group"],
                    _turn_band(record["turn"]),
                    _unseen_bucket(record["unseen_wait_counts"]),
                    record["dealer"],
                    min(record["prior_riichis"], 1),
                    record["furiten"],
                    min(record["opponent_called_hands"], 1),
                ),
            ),
        },
    }


def _metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    outcomes = Counter(record["outcome"] for record in records)
    wins = outcomes["win"]
    ron = sum(record["win_method"] == "ron" for record in records)
    tsumo = sum(record["win_method"] == "tsumo" for record in records)
    deltas = [
        record["point_delta"] for record in records if record["point_delta"] is not None
    ]
    return {
        "count": total,
        "wins": wins,
        "win_rate": wins / total if total else None,
        "win_rate_wilson_95": _wilson(wins, total),
        "ron_wins": ron,
        "ron_rate": ron / total if total else None,
        "tsumo_wins": tsumo,
        "tsumo_rate": tsumo / total if total else None,
        "deal_ins": outcomes["deal_in"],
        "deal_in_rate": outcomes["deal_in"] / total if total else None,
        "opponent_tsumo": outcomes["opponent_tsumo"],
        "other_ron": outcomes["other_ron"],
        "draws": outcomes["draw"],
        "draw_rate": outcomes["draw"] / total if total else None,
        "point_delta_available": len(deltas),
        "mean_point_delta": statistics.fmean(deltas) if deltas else None,
        "median_point_delta": statistics.median(deltas) if deltas else None,
    }


def _rate_differences(
    treated: Sequence[dict[str, Any]], control: Sequence[dict[str, Any]]
) -> dict[str, float | None]:
    return {
        metric: _difference(treated, control, predicate)
        for metric, predicate in _OUTCOME_PREDICATES.items()
    }


_OUTCOME_PREDICATES: dict[str, Callable[[dict[str, Any]], bool]] = {
    "win": lambda row: row["outcome"] == "win",
    "ron": lambda row: row["win_method"] == "ron",
    "tsumo": lambda row: row["win_method"] == "tsumo",
    "deal_in": lambda row: row["outcome"] == "deal_in",
    "draw": lambda row: row["outcome"] == "draw",
}


def _difference(
    treated: Sequence[dict[str, Any]],
    control: Sequence[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
) -> float | None:
    if not treated or not control:
        return None
    return 100 * (
        sum(predicate(record) for record in treated) / len(treated)
        - sum(predicate(record) for record in control) / len(control)
    )


def _standardize(
    treated: Sequence[dict[str, Any]],
    control: Sequence[dict[str, Any]],
    stratum: Callable[[dict[str, Any]], tuple[object, ...]],
) -> dict[str, Any]:
    treated_by: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
    control_by: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in treated:
        treated_by[stratum(record)].append(record)
    for record in control:
        control_by[stratum(record)].append(record)
    common = sorted(set(treated_by) & set(control_by), key=repr)
    common_treated = sum(len(treated_by[key]) for key in common)
    common_control = sum(len(control_by[key]) for key in common)
    pooled = common_treated + common_control
    estimates: dict[str, dict[str, float | None]] = {}
    for metric, predicate in _OUTCOME_PREDICATES.items():
        if not pooled:
            estimates[metric] = {
                "treated_rate": None,
                "control_rate": None,
                "difference_percentage_points": None,
            }
            continue
        treated_rate = control_rate = 0.0
        for key in common:
            weight = (len(treated_by[key]) + len(control_by[key])) / pooled
            treated_rate += (
                weight
                * sum(predicate(row) for row in treated_by[key])
                / len(treated_by[key])
            )
            control_rate += (
                weight
                * sum(predicate(row) for row in control_by[key])
                / len(control_by[key])
            )
        estimates[metric] = {
            "treated_rate": treated_rate,
            "control_rate": control_rate,
            "difference_percentage_points": 100 * (treated_rate - control_rate),
        }
    point_delta = _standardized_continuous(treated_by, control_by, common, pooled)
    return {
        "common_strata": len(common),
        "treated_on_common_support": common_treated,
        "control_on_common_support": common_control,
        "treated_excluded": len(treated) - common_treated,
        "control_excluded": len(control) - common_control,
        "target_weights": "pooled distribution within common-support strata",
        "outcomes": estimates,
        "point_delta": point_delta,
    }


def _standardized_continuous(
    treated_by: Mapping[tuple[object, ...], Sequence[dict[str, Any]]],
    control_by: Mapping[tuple[object, ...], Sequence[dict[str, Any]]],
    common: Sequence[tuple[object, ...]],
    pooled: int,
) -> dict[str, float | int | None]:
    usable = [
        key
        for key in common
        if all(row["point_delta"] is not None for row in treated_by[key])
        and all(row["point_delta"] is not None for row in control_by[key])
    ]
    usable_pooled = sum(len(treated_by[key]) + len(control_by[key]) for key in usable)
    if not usable_pooled:
        return {
            "strata": 0,
            "treated_mean": None,
            "control_mean": None,
            "difference": None,
        }
    treated_mean = control_mean = 0.0
    for key in usable:
        weight = (len(treated_by[key]) + len(control_by[key])) / usable_pooled
        treated_mean += weight * statistics.fmean(
            row["point_delta"] for row in treated_by[key]
        )
        control_mean += weight * statistics.fmean(
            row["point_delta"] for row in control_by[key]
        )
    return {
        "strata": len(usable),
        "pooled_records": usable_pooled,
        "common_support_pooled_records": pooled,
        "treated_mean": treated_mean,
        "control_mean": control_mean,
        "difference": treated_mean - control_mean,
    }


def _turn_band(turn: int) -> str:
    if turn <= 6:
        return "1-6"
    if turn <= 12:
        return "7-12"
    return "13+"


def _era(year: int) -> str:
    if year <= 2014:
        return "2009-2014"
    if year <= 2019:
        return "2015-2019"
    return "2020-2025"


def _unseen_bucket(values: Sequence[int]) -> str:
    value = min(values)
    return str(value) if value <= 2 else "3+"


def _wilson(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [center - half, center + half]


def _validate_record(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("every decision record must be an object")
    required = {
        "strategy",
        "year",
        "riichi_eligible",
        "wait_group",
        "turn",
        "unseen_wait_counts",
        "dealer",
        "prior_riichis",
        "furiten",
        "opponent_called_hands",
        "outcome",
        "win_method",
        "point_delta",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"decision record is missing fields: {missing}")
    return value
