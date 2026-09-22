import json
from pathlib import Path

import pytest
from test_kanchan_drop_analysis import annual_task

from analysis.analyze_kanchan_drop import check_quality_baseline
from mahjong_analysis.kanchan_drop_analysis import (
    AnnualKanchanDropTask,
    aggregate_annual_kanchan_drop,
    kanchan_drop_results_document,
)


def make_quality_baseline(document: dict) -> dict:
    def all_quality(value: dict) -> dict:
        return json.loads(json.dumps(value["groups"]["all"]))

    return {
        "overall": all_quality(document["overall"]),
        "by_turn": [
            {
                "riichi_discard_number": row["riichi_discard_number"],
                **all_quality(row),
            }
            for row in document["by_turn"]
        ],
        "years": [
            {
                "year": row["year"],
                **all_quality(row["overall"]),
                "by_turn": [
                    {
                        "riichi_discard_number": turn["riichi_discard_number"],
                        **all_quality(turn),
                    }
                    for turn in row["by_turn"]
                ],
            }
            for row in document["years"]
        ],
    }


def result_document(tmp_path: Path) -> dict:
    result = aggregate_annual_kanchan_drop(annual_task(tmp_path, 2025))
    return kanchan_drop_results_document((result,))


def test_quality_baseline_check_accepts_identical_all_record_slices(
    tmp_path: Path,
) -> None:
    document = result_document(tmp_path)
    check_quality_baseline(document, make_quality_baseline(document))


def test_quality_baseline_check_rejects_count_mismatch(tmp_path: Path) -> None:
    document = result_document(tmp_path)
    baseline = make_quality_baseline(document)
    baseline["overall"]["record_count"] += 1
    with pytest.raises(ValueError, match="overall quality totals"):
        check_quality_baseline(document, baseline)


def test_wrong_task_count_remains_fail_closed(tmp_path: Path) -> None:
    task = annual_task(tmp_path, 2025)
    with pytest.raises(ValueError, match="annual record count"):
        aggregate_annual_kanchan_drop(
            AnnualKanchanDropTask(task.year, task.path, task.expected_records + 1)
        )
