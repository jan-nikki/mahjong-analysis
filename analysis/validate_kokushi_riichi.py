"""Independent closed-hand replay and full-result comparison for the DB search.

No production replay, meld decoder, tile constants, or wait solver are imported.
Only INIT/N/REACH and draw/discard XML tags are needed; terminal score attributes
are outside this reference's scope (including archives' duplicated owari fields).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sqlite3
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlencode

_STATE_TAG = re.compile(rb"<(?:INIT|N|REACH)\b[^<>]*/>|<[TUVWDEFGdefg]\d*\s*/>")
_INIT_TAG = re.compile(rb"<INIT\b[^<>]*/>")
_REACH_TAG = re.compile(rb"<REACH\b[^<>]*/>")
_CALL_TAG = re.compile(rb"<N\b[^<>]*/>")
_ACTOR_ACTIONS = tuple(
    re.compile(
        f"<[{discard}{discard.lower()}](\\d*)\\s*/>|<{draw}(\\d+)\\s*/>".encode()
    )
    for draw, discard in zip("TUVW", "DEFG", strict=True)
)


def targeted_reference_log(payload: bytes, year: int, date: str, log_id: str) -> dict:
    """Replay each accepted actor's prefix, independent of the forward replay.

    Select candidates from acceptance tags first. Only that actor's draw/discard
    history up to acceptance is needed; other players' actions are not replayed.
    """
    xml = gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload
    starts = list(_INIT_TAG.finditer(xml))
    required = {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33}
    names = [f"{rank}{suit}" for suit in "mps" for rank in range(1, 10)]
    names.extend(["E", "S", "W", "N", "P", "F", "C"])
    accepted = 0
    hits, audit = [], []
    for kyoku, init in enumerate(starts):
        end = starts[kyoku + 1].start() if kyoku + 1 < len(starts) else len(xml)
        round_xml = xml[init.end() : end]
        reaches = list(_REACH_TAG.finditer(round_xml))
        if not reaches:
            continue
        initial = ET.fromstring(init.group()).attrib
        calls = [
            (call.start(), int(ET.fromstring(call.group()).attrib["who"]))
            for call in _CALL_TAG.finditer(round_xml)
        ]
        declared = {}
        for reach in reaches:
            attrs = ET.fromstring(reach.group()).attrib
            who = int(attrs["who"])
            if attrs["step"] == "1":
                declared[who] = reach.start()
                continue
            assert attrs["step"] == "2"
            accepted += 1
            if any(
                actor == who and position < reach.start() for position, actor in calls
            ):
                continue
            hand = set(map(int, initial[f"hai{who}"].split(",")))
            assert len(hand) == 13
            turn, last_draw = 0, None
            for action in _ACTOR_ACTIONS[who].finditer(round_xml, 0, reach.start()):
                discard, draw = action.groups()
                if draw is not None:
                    last_draw = int(draw)
                    assert last_draw not in hand
                    hand.add(last_draw)
                else:
                    tile = int(discard) if discard else last_draw
                    hand.remove(tile)
                    turn += 1
                    last_draw = None
            assert len(hand) == 13, (log_id, kyoku, who)
            counts = Counter(tile // 4 for tile in hand)
            if set(counts) == required and all(count == 1 for count in counts.values()):
                shape, waiting = "kokushi_13men", [names[k] for k in sorted(required)]
            elif (
                set(counts) < required
                and len(counts) == 12
                and sorted(counts.values()) == [1] * 11 + [2]
            ):
                shape, waiting = (
                    "kokushi_single",
                    [names[next(iter(required - set(counts)))]],
                )
            else:
                continue
            interrupted = any(position < declared[who] for position, _ in calls)
            hits.append(
                {
                    "year": year,
                    "date": date,
                    "log_id": log_id,
                    "kyoku_index": kyoku,
                    "who": who,
                    "turn": turn,
                    "reach_type": "double_riichi"
                    if turn == 1 and not interrupted
                    else "riichi",
                    "wait_type": shape,
                    "waits": waiting,
                    "url": "https://tenhou.net/5/?"
                    + urlencode({"log": log_id, "tw": who, "ts": kyoku}),
                }
            )
            audit.append(
                {
                    "log_id": log_id,
                    "year": year,
                    "kyoku_index": kyoku,
                    "who": who,
                    "accepted_xml_byte_offset": init.end() + reach.start(),
                    "concealed_tile_ids": sorted(hand),
                    "concealed_tiles": [names[tile // 4] for tile in sorted(hand)],
                }
            )
    return {
        "total_kyokus": len(starts),
        "total_riichis": accepted,
        "records": hits,
        "audit": audit,
    }


def reference_log(payload: bytes, year: int, date: str, log_id: str) -> dict:
    """Independently inspect every accepted riichi without decoding melds."""
    xml = gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload
    required = {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33}
    names = [f"{rank}{suit}" for suit in "mps" for rank in range(1, 10)]
    names.extend(["E", "S", "W", "N", "P", "F", "C"])
    hands = []
    discard_counts = []
    declarations = {}
    last_draw = [None] * 4
    calls = 0
    kyoku = -1
    accepted = 0
    hits = []
    audit = []
    for match in _STATE_TAG.finditer(xml):
        raw = match.group()
        if raw.startswith(b"<INIT"):
            event = ET.fromstring(raw)
            hands = [
                set(map(int, event.attrib[f"hai{i}"].split(","))) for i in range(4)
            ]
            assert all(len(hand) == 13 for hand in hands)
            discard_counts = [0] * 4
            last_draw = [None] * 4
            calls = 0
            declarations = {}
            kyoku += 1
        elif raw.startswith(b"<N"):
            event = ET.fromstring(raw)
            # A fixed meld (including a closed kan) permanently excludes kokushi.
            hands[int(event.attrib["who"])] = None
            calls += 1
        elif raw.startswith(b"<REACH"):
            event = ET.fromstring(raw)
            who = int(event.attrib["who"])
            if event.attrib["step"] == "1":
                declarations[who] = calls
                continue
            assert event.attrib["step"] == "2"
            accepted += 1
            hand = hands[who]
            if hand is None:
                continue
            assert len(hand) == 13, (log_id, kyoku, who, sorted(hand))
            counts = Counter(tile // 4 for tile in hand)
            if set(counts) == required and all(count == 1 for count in counts.values()):
                shape = "kokushi_13men"
                waiting = [names[kind] for kind in sorted(required)]
            elif (
                set(counts) < required
                and len(counts) == 12
                and sorted(counts.values()) == [1] * 11 + [2]
            ):
                shape = "kokushi_single"
                waiting = [names[next(iter(required - set(counts)))]]
            else:
                continue
            reach_type = (
                "double_riichi"
                if discard_counts[who] == 1 and declarations[who] == 0
                else "riichi"
            )
            hits.append(
                {
                    "year": year,
                    "date": date,
                    "log_id": log_id,
                    "kyoku_index": kyoku,
                    "who": who,
                    "turn": discard_counts[who],
                    "reach_type": reach_type,
                    "wait_type": shape,
                    "waits": waiting,
                    "url": "https://tenhou.net/5/?"
                    + urlencode({"log": log_id, "tw": who, "ts": kyoku}),
                }
            )
            audit.append(
                {
                    "log_id": log_id,
                    "year": year,
                    "kyoku_index": kyoku,
                    "who": who,
                    "accepted_xml_byte_offset": match.start(),
                    "concealed_tile_ids": sorted(hand),
                    "concealed_tiles": [names[tile // 4] for tile in sorted(hand)],
                }
            )
        else:
            tag = raw[1:-2].strip().decode("ascii")
            initial = tag[0].upper()
            is_draw = initial in "TUVW"
            who = ("TUVW" if is_draw else "DEFG").index(initial)
            if not is_draw:
                discard_counts[who] += 1
            hand = hands[who]
            if hand is None:
                continue
            digits = tag[1:]
            tile = int(digits) if digits else last_draw[who]
            assert tile is not None
            if is_draw:
                assert tile not in hand
                hand.add(tile)
                last_draw[who] = tile
            else:
                hand.remove(tile)
                last_draw[who] = None
    return {
        "total_kyokus": kyoku + 1,
        "total_riichis": accepted,
        "records": hits,
        "audit": audit,
    }


def _year(task: tuple) -> dict:
    root, year, limit = task
    path = Path(root) / f"{year}.db"
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(logs)")}
    scope = "num_players = 4 AND is_tonpu = 0"
    if "game_type" in columns:
        scope += " AND game_type = 'houou'"
    eligible, available = connection.execute(
        f"SELECT COUNT(*), COUNT(log) FROM logs WHERE {scope}"
    ).fetchone()
    sql = (
        f"SELECT id, date, log FROM logs WHERE {scope} AND log IS NOT NULL ORDER BY id"
    )
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    result = {
        "year": year,
        "eligible_logs": eligible,
        "available_logs": available,
        "unavailable_logs": eligible - available,
        "scanned_logs": 0,
        "total_kyokus": 0,
        "total_riichis": 0,
        "records": [],
        "audit": [],
    }
    try:
        for log_id, date, blob in connection.execute(sql):
            try:
                analyzed = targeted_reference_log(blob, year, date, log_id)
            except Exception as error:
                raise ValueError(f"reference {year}/{log_id}: {error}") from error
            result["scanned_logs"] += 1
            for field in ("total_kyokus", "total_riichis"):
                result[field] += analyzed[field]
            result["records"].extend(analyzed["records"])
            result["audit"].extend(analyzed["audit"])
    finally:
        connection.close()
    result["kokushi_riichis"] = len(result["records"])
    result["kokushi_13men"] = sum(
        hit["wait_type"] == "kokushi_13men" for hit in result["records"]
    )
    return result


def _identity(record: dict) -> tuple:
    return tuple(
        record[key] for key in ("year", "date", "log_id", "kyoku_index", "who")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    expected_bytes = args.result.read_bytes()
    expected = json.loads(expected_bytes)
    tasks = [
        (str(args.db_root), year, expected["scope"]["max_logs_per_year"])
        for year in expected["scope"]["years"]
    ]
    annual = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(_year, task) for task in tasks]):
            result = future.result()
            annual.append(result)
            print(
                f"reference {result['year']}: {result['scanned_logs']:,} logs, {result['kokushi_riichis']} hits",
                flush=True,
            )
    report = compare_annual(expected_bytes, annual)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "differences": len(report["differences"]),
                "totals": report["totals"],
            },
            indent=2,
        )
    )
    return int(report["status"] != "PASS")


def compare_annual(expected_bytes: bytes, annual: list[dict]) -> dict:
    """Compare a completed independent replay to every exported field."""
    expected = json.loads(expected_bytes)
    annual = sorted(annual, key=lambda item: item["year"])
    records = sorted([hit for item in annual for hit in item["records"]], key=_identity)
    fields = (
        "eligible_logs",
        "available_logs",
        "unavailable_logs",
        "scanned_logs",
        "total_kyokus",
        "total_riichis",
        "kokushi_riichis",
        "kokushi_13men",
    )
    differences = []
    for actual, wanted in zip(annual, expected["yearly"], strict=True):
        for field in ("year", *fields):
            if actual[field] != wanted[field]:
                differences.append(
                    {
                        "year": actual["year"],
                        "field": field,
                        "reference": actual[field],
                        "production": wanted[field],
                    }
                )
    actual_by_key = {_identity(record): record for record in records}
    expected_by_key = {_identity(record): record for record in expected["records"]}
    if len(expected_by_key) != len(expected["records"]):
        differences.append({"field": "duplicate_production_record"})
    for key in sorted(actual_by_key.keys() | expected_by_key.keys()):
        if actual_by_key.get(key) != expected_by_key.get(key):
            differences.append(
                {
                    "key": key,
                    "reference": actual_by_key.get(key),
                    "production": expected_by_key.get(key),
                }
            )
    for field in fields:
        if sum(item[field] for item in annual) != expected["summary"][field]:
            differences.append({"field": f"summary.{field}"})
    report = {
        "status": "PASS" if not differences else "FAIL",
        "result_sha256": hashlib.sha256(expected_bytes).hexdigest(),
        "method": "Independent XML state-tag replay; fixed-meld actors excluded; direct orphan counts; no production imports",
        "scope": expected["scope"],
        "totals": {field: sum(item[field] for item in annual) for field in fields},
        "yearly": [
            {
                key: value
                for key, value in item.items()
                if key not in {"records", "audit"}
            }
            for item in annual
        ],
        "differences": differences,
        "hit_hands": [audit for item in annual for audit in item["audit"]],
    }
    return report


if __name__ == "__main__":
    raise SystemExit(main())
