"""Independently recheck saved kanchan-drop audit samples against raw MJAI."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _normalize(tile: str) -> str:
    return tile.removesuffix("r")


def _candidate(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any] | None:
    first_tile = _normalize(first["pai"])
    second_tile = _normalize(second["pai"])
    if second["tsumogiri"]:
        return None
    if first_tile[-1] not in "mps" or second_tile[-1] not in "mps":
        return None
    if first_tile[-1] != second_tile[-1]:
        return None
    first_rank = int(first_tile[0])
    second_rank = int(second_tile[0])
    if abs(first_rank - second_rank) != 2:
        return None
    low, high = sorted((first_rank, second_rank))
    first_distance = abs(first_rank - 5)
    second_distance = abs(second_rank - 5)
    return {
        "candidate_type": "B" if first["tsumogiri"] else "A",
        "suit": first_tile[-1],
        "shape": f"{low}{high}",
        "direction": "low_to_high" if first_rank < second_rank else "high_to_low",
        "order_class": (
            "inner_first"
            if first_distance < second_distance
            else "outer_first"
            if first_distance > second_distance
            else "symmetric"
        ),
        "middle_tile": f"{(low + high) // 2}{first_tile[-1]}",
        "first_discard_number": first["discard_number"],
        "second_discard_number": second["discard_number"],
        "first_raw_tile": first["pai"],
        "second_raw_tile": second["pai"],
        "first_normalized_tile": first_tile,
        "second_normalized_tile": second_tile,
        "first_tsumogiri": first["tsumogiri"],
        "first_event_index": first["event_index"],
        "second_event_index": second["event_index"],
    }


def _read_events(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _check_sample(sample: dict[str, Any], events: list[dict[str, Any]]) -> None:
    start_line = sample["start_kyoku_line"]
    reach_line = sample["reach_line"]
    actor = sample["actor"]
    if events[start_line - 1].get("type") != "start_kyoku":
        raise ValueError("start_kyoku_line does not point to start_kyoku")
    reach = events[reach_line - 1]
    declaration = events[reach_line]
    accepted = events[reach_line + 1]
    if reach.get("type") != "reach" or reach.get("actor") != actor:
        raise ValueError("reach_line does not point to the expected reach")
    if declaration.get("type") != "dahai" or declaration.get("actor") != actor:
        raise ValueError("reach is not followed by the expected declaration discard")
    if accepted.get("type") != "reach_accepted" or accepted.get("actor") != actor:
        raise ValueError("declaration is not followed by reach_accepted")

    river = []
    for line_number in range(start_line, reach_line + 2):
        event = events[line_number - 1]
        if event.get("type") == "dahai" and event.get("actor") == actor:
            river.append(
                {
                    "pai": event["pai"],
                    "tsumogiri": event["tsumogiri"],
                    "discard_number": len(river) + 1,
                    "event_index": line_number - start_line,
                }
            )
    if len(river) != sample["riichi_discard_number"]:
        raise ValueError("raw river length disagrees with saved riichi discard number")

    actual = []
    for first, second in pairwise(river):
        value = _candidate(first, second)
        if value is not None:
            actual.append(value)
    expected = sample["matching_candidates"]
    if actual != expected:
        raise ValueError("raw candidates disagree with saved audit candidates")
    types = {value["candidate_type"] for value in actual}
    classification = (
        "both"
        if types == {"A", "B"}
        else "a_only"
        if types == {"A"}
        else "b_only"
        if types == {"B"}
        else "none"
    )
    if classification != sample["classification"]:
        raise ValueError("raw classification disagrees with saved classification")
    representative = actual[-1] if actual else None
    if representative != sample["representative_candidate"]:
        raise ValueError("raw representative candidate disagrees")
    distance = (
        sample["riichi_discard_number"] - representative["second_discard_number"]
        if representative is not None
        else None
    )
    if distance != sample["representative_distance"]:
        raise ValueError("raw representative distance disagrees")


def validate(result_path: Path, raw_root: Path) -> dict[str, Any]:
    document = json.loads(result_path.read_text(encoding="utf-8"))
    if document.get("metadata", {}).get("analysis_name") != (
        "kanchan-drop-analysis-v1"
    ):
        raise ValueError("unexpected analysis result")
    samples = document["audit"]["samples"]
    by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_path[sample["relative_source_path"]].append(sample)
    checked = 0
    for relative_path in sorted(by_path):
        events = _read_events(raw_root / Path(relative_path))
        for sample in by_path[relative_path]:
            try:
                _check_sample(sample, events)
            except Exception as exc:
                key = (
                    sample["relative_source_path"],
                    sample["start_kyoku_line"],
                    sample["reach_line"],
                )
                raise ValueError(f"audit sample {key} failed") from exc
            checked += 1
    return {
        "selected_count": len(samples),
        "checked_count": checked,
        "matched_count": checked,
        "mismatch_count": 0,
        "missing_reasons": document["audit"]["missing_reasons"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        type=Path,
        default=PROJECT_ROOT / "research/results/kanchan-drop-analysis-v1.json",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=PROJECT_ROOT / "data/raw",
    )
    args = parser.parse_args()
    print(json.dumps(validate(args.result, args.raw_root), sort_keys=True))


if __name__ == "__main__":
    main()
