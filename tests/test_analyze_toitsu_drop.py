from copy import deepcopy

import pytest
from test_toitsu_drop_analysis import annual_task

from analysis.analyze_toitsu_drop import check_quality_baseline
from mahjong_analysis.toitsu_drop_analysis import (
    aggregate_annual_toitsu_drop,
    toitsu_drop_results_document,
)


def make_quality_baseline(document: dict) -> dict:
    def flatten(group: dict, **extra) -> dict:
        return {**extra, **deepcopy(group)}

    return {
        "overall": deepcopy(document["overall"]["groups"]["all"]),
        "by_turn": [
            flatten(
                row["groups"]["all"],
                riichi_discard_number=row["riichi_discard_number"],
            )
            for row in document["by_turn"]
        ],
        "years": [
            flatten(
                row["overall"]["groups"]["all"],
                year=row["year"],
                by_turn=[
                    flatten(
                        turn["groups"]["all"],
                        riichi_discard_number=turn["riichi_discard_number"],
                    )
                    for turn in row["by_turn"]
                ],
            )
            for row in document["years"]
        ],
    }


def test_quality_baseline_check_accepts_identical_all_record_slices(tmp_path) -> None:
    document = toitsu_drop_results_document(
        (aggregate_annual_toitsu_drop(annual_task(tmp_path, 2025)),)
    )

    check_quality_baseline(document, make_quality_baseline(document))


def test_quality_baseline_check_rejects_count_mismatch(tmp_path) -> None:
    document = toitsu_drop_results_document(
        (aggregate_annual_toitsu_drop(annual_task(tmp_path, 2025)),)
    )
    baseline = make_quality_baseline(document)
    baseline["overall"]["record_count"] += 1

    with pytest.raises(ValueError, match="overall quality totals"):
        check_quality_baseline(document, baseline)
