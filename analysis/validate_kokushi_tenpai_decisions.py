"""Independently audit XML decision records against the MJAI v2 archives."""

from __future__ import annotations

import argparse
import gzip
import json
import random
import zipfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mahjong_analysis.riichi_wait_reference.waits import (
    calculate_reference_hand_waits,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "kokushi-tenpai-decisions-2009-2025.json"
DEFAULT_ARCHIVES = PROJECT_ROOT / "data" / "archives" / "v2.0.0"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "kokushi-tenpai-decisions-validation.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--archives", type=Path, default=DEFAULT_ARCHIVES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dama-sample", type=int, default=500)
    args = parser.parse_args()
    document = json.loads(args.input.read_text(encoding="utf-8"))
    records = document["records"]
    rng = random.Random(20260921)
    treatment = [record for record in records if record["strategy"] != "dama"]
    dama = [record for record in records if record["strategy"] == "dama"]
    sampled = sorted(
        [*treatment, *rng.sample(dama, min(args.dama_sample, len(dama)))],
        key=lambda row: (row["year"], row["log_id"], row["kyoku_index"], row["who"]),
    )
    result = validate_records(sampled, args.archives)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.name}.part")
    temporary.write_text(
        json.dumps(
            result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(args.output)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0 if result["summary"]["status"] == "PASS" else 1


def validate_records(
    records: Sequence[Mapping[str, Any]], archive_root: Path
) -> dict[str, Any]:
    by_year: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_year[int(record["year"])].append(record)
    mismatches: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    omitted_acceptances = 0
    checked = 0
    for year, annual in sorted(by_year.items()):
        archive_path = archive_root / f"{year}.zip"
        with zipfile.ZipFile(archive_path) as archive:
            by_log: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for record in annual:
                by_log[str(record["log_id"])].append(record)
            for log_id, target_records in sorted(by_log.items()):
                try:
                    raw = archive.read(f"{log_id}.mjson")
                except KeyError:
                    missing.append({"year": year, "log_id": log_id})
                    continue
                if raw.startswith(b"\x1f\x8b"):
                    raw = gzip.decompress(raw)
                events = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
                kyokus = _split_kyokus(events)
                for target in target_records:
                    checked += 1
                    kyoku_index = int(target["kyoku_index"])
                    actor = int(target["who"])
                    reference = _reference_decisions(kyokus[kyoku_index]).get(actor)
                    if reference is None:
                        mismatches.append(_mismatch(target, None, ["missing_decision"]))
                        continue
                    fields = [
                        "turn",
                        "wait_type",
                        "waits",
                        "strategy",
                        "riichi_declared",
                        "outcome",
                        "win_method",
                    ]
                    differing = [
                        field
                        for field in fields
                        if _normalized(target.get(field))
                        != _normalized(reference.get(field))
                    ]
                    if bool(target["riichi_established"]) != bool(
                        reference["riichi_established"]
                    ):
                        if (
                            target["riichi_established"]
                            and target["riichi_declared"]
                            and not reference["riichi_established"]
                        ):
                            omitted_acceptances += 1
                        else:
                            differing.append("riichi_established")
                    if differing:
                        mismatches.append(_mismatch(target, reference, differing))
    # The independently converted MJAI archive is not a perfect superset of the
    # XML DB. Missing source files reduce coverage but are not a disagreement.
    status = "PASS" if not mismatches else "FAIL"
    return {
        "summary": {
            "status": status,
            "checked": checked,
            "mismatches": len(mismatches),
            "missing_mjai_logs": len(missing),
            "requested": len(records),
            "mjai_omitted_reach_acceptances": omitted_acceptances,
        },
        "notes": [
            "All non-dama records plus a deterministic dama sample are checked.",
            "Known upstream MJAI conversion can omit the fourth reach_accepted immediately before reach4; the XML declaration and result remain authoritative.",
        ],
        "mismatches": mismatches[:100],
        "missing": missing[:100],
    }


def _split_kyokus(events: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    result: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] | None = None
    for event in events:
        event_type = event.get("type")
        if event_type == "start_kyoku":
            current = [event]
        elif event_type == "end_kyoku":
            if current is None:
                raise ValueError("end_kyoku without start_kyoku")
            current.append(event)
            result.append(current)
            current = None
        elif current is not None:
            current.append(event)
    if current is not None:
        raise ValueError("MJAI ended before end_kyoku")
    return result


def _reference_decisions(kyoku: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    start = kyoku[0]
    hands = [Counter(hand) for hand in start["tehais"]]
    called = [False] * 4
    turns = [0] * 4
    decisions: dict[int, dict[str, Any]] = {}
    pending_actor: int | None = None
    pending_decision: dict[str, Any] | None = None
    hora_events: list[dict[str, Any]] = []
    drew: list[str | None] = [None] * 4
    for event in kyoku[1:]:
        event_type = event["type"]
        if event_type == "tsumo":
            actor = event["actor"]
            drew[actor] = event["pai"]
            if not called[actor]:
                hands[actor][event["pai"]] += 1
        elif event_type == "reach":
            pending_actor = event["actor"]
            pending_decision = None
        elif event_type == "dahai":
            actor = event["actor"]
            tile = event["pai"]
            if not called[actor]:
                _remove_counter_tile(hands[actor], tile)
            turns[actor] += 1
            drew[actor] = None
            kokushi = _reference_kokushi(hands[actor]) if not called[actor] else None
            if kokushi is not None and actor not in decisions:
                wait_type, waits = kokushi
                decisions[actor] = {
                    "turn": turns[actor],
                    "wait_type": wait_type,
                    "waits": list(waits),
                    "strategy": "dama",
                    "riichi_declared": False,
                    "riichi_established": False,
                }
            if pending_actor == actor and kokushi is not None and actor in decisions:
                decision = decisions[actor]
                decision["strategy"] = (
                    "immediate_riichi"
                    if turns[actor] == decision["turn"]
                    else "delayed_riichi"
                )
                decision["riichi_declared"] = True
                pending_decision = decision
        elif event_type == "reach_accepted":
            if pending_actor != event["actor"]:
                raise ValueError("reach_accepted actor mismatch")
            if pending_decision is not None:
                pending_decision["riichi_established"] = True
            pending_actor = None
            pending_decision = None
        elif event_type in {"chi", "pon", "daiminkan", "ankan"}:
            actor = event["actor"]
            if not called[actor]:
                for tile in event["consumed"]:
                    _remove_counter_tile(hands[actor], tile)
            called[actor] = True
        elif event_type == "kakan":
            actor = event["actor"]
            if not called[actor]:
                _remove_counter_tile(hands[actor], event["pai"])
            called[actor] = True
        elif event_type == "hora":
            hora_events.append(event)
        elif event_type in {"ryukyoku", "dora", "end_kyoku"}:
            pass
    winners = tuple(event["actor"] for event in hora_events)
    targets = tuple(event["target"] for event in hora_events)
    for actor, decision in decisions.items():
        if actor in winners:
            index = winners.index(actor)
            decision["outcome"] = "win"
            decision["win_method"] = "tsumo" if targets[index] == actor else "ron"
        elif hora_events and actor in targets:
            decision["outcome"] = "deal_in"
            decision["win_method"] = None
        elif any(
            winner == target for winner, target in zip(winners, targets, strict=True)
        ):
            decision["outcome"] = "opponent_tsumo"
            decision["win_method"] = None
        elif hora_events:
            decision["outcome"] = "other_ron"
            decision["win_method"] = None
        else:
            decision["outcome"] = "draw"
            decision["win_method"] = None
    return decisions


def _reference_kokushi(hand: Counter[str]) -> tuple[str, tuple[str, ...]] | None:
    if sum(hand.values()) != 13:
        return None
    waits = calculate_reference_hand_waits(tuple(hand.elements()), ())
    details = [detail for detail in waits.wait_details if detail.hand_type == "kokushi"]
    if not details:
        return None
    shapes = {detail.wait_shape for detail in details}
    tiles = tuple(
        tile
        for tile in waits.wait_tiles
        if any(detail.wait_tile == tile for detail in details)
    )
    if shapes == {"kokushi_13men"} and len(tiles) == 13:
        return "kokushi_13men", tiles
    if shapes == {"kokushi_single"} and len(tiles) == 1:
        return "kokushi_single", tiles
    raise ValueError("inconsistent reference kokushi waits")


def _remove_counter_tile(hand: Counter[str], tile: str) -> None:
    if hand[tile] < 1:
        raise ValueError(f"tile not in reference hand: {tile}")
    hand[tile] -= 1
    if hand[tile] == 0:
        del hand[tile]


def _normalized(value: object) -> object:
    return tuple(value) if isinstance(value, list) else value


def _mismatch(
    production: Mapping[str, Any],
    reference: Mapping[str, Any] | None,
    fields: Sequence[str],
) -> dict[str, Any]:
    return {
        "key": [
            production["year"],
            production["log_id"],
            production["kyoku_index"],
            production["who"],
        ],
        "fields": list(fields),
        "production": dict(production),
        "reference": dict(reference) if reference is not None else None,
    }


if __name__ == "__main__":
    raise SystemExit(main())
