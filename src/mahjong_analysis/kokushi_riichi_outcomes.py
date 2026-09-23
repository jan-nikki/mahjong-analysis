"""Enrich extracted kokushi-riichi records with their kyoku outcomes."""

from __future__ import annotations

import gzip
import json
import sqlite3
import xml.etree.ElementTree as ET
import zlib
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from mahjong_analysis.kokushi_riichi import (
    KokushiRiichiParseError,
    _remove_identical_terminal_attributes,
)

Outcome = Literal["kokushi_win", "other_win", "draw"]
WinMethod = Literal["ron", "tsumo"]


@dataclass(frozen=True)
class KyokuOutcome:
    """The terminal result of one target actor's kyoku."""

    outcome: Outcome
    won: bool
    win_method: WinMethod | None
    winners: tuple[int, ...]
    from_who: int | None

    def __post_init__(self) -> None:
        if self.outcome not in ("kokushi_win", "other_win", "draw"):
            raise ValueError("unsupported outcome")
        if self.won != (self.outcome == "kokushi_win"):
            raise ValueError("won must match outcome")
        if self.won != (self.win_method is not None):
            raise ValueError("win_method must be present exactly when won")
        if self.won != (self.from_who is not None):
            raise ValueError("from_who must be present exactly when won")
        if any(
            type(actor) is not int or actor not in range(4) for actor in self.winners
        ):
            raise ValueError("winner seats must be integers from 0 through 3")
        if len(self.winners) != len(set(self.winners)):
            raise ValueError("winner seats must be unique")
        if self.outcome == "draw" and self.winners:
            raise ValueError("a draw cannot have winners")


def classify_kyoku_outcome(
    payload: bytes,
    *,
    kyoku_index: int,
    actor: int,
) -> KyokuOutcome:
    """Classify whether ``actor`` won the requested kyoku."""
    if type(kyoku_index) is not int or kyoku_index < 0:
        raise ValueError("kyoku_index must be a non-negative integer")
    if type(actor) is not int or actor not in range(4):
        raise ValueError("actor must be an integer from 0 through 3")
    root = _parse_root(payload)

    current_kyoku = -1
    terminal_events: list[ET.Element] = []
    target_seen = False
    for event in root:
        if event.tag == "INIT":
            if current_kyoku == kyoku_index:
                break
            current_kyoku += 1
            target_seen = current_kyoku == kyoku_index
            continue
        if target_seen and event.tag in {"AGARI", "RYUUKYOKU"}:
            terminal_events.append(event)

    if not target_seen:
        raise ValueError(f"kyoku_index {kyoku_index} does not exist")
    if not terminal_events:
        raise ValueError(f"kyoku_index {kyoku_index} has no terminal result")

    agari_events = [event for event in terminal_events if event.tag == "AGARI"]
    draw_events = [event for event in terminal_events if event.tag == "RYUUKYOKU"]
    if agari_events and draw_events:
        raise ValueError("a kyoku cannot contain both AGARI and RYUUKYOKU")
    if len(draw_events) > 1:
        raise ValueError("a kyoku cannot contain multiple RYUUKYOKU events")

    if draw_events:
        return KyokuOutcome("draw", False, None, (), None)

    winners = tuple(_seat(event, "who") for event in agari_events)
    if len(winners) != len(set(winners)):
        raise ValueError("duplicate AGARI winner in one kyoku")
    actor_events = [
        event
        for event, winner in zip(agari_events, winners, strict=True)
        if winner == actor
    ]
    if not actor_events:
        return KyokuOutcome("other_win", False, None, winners, None)
    if len(actor_events) != 1:
        raise ValueError("target actor has multiple AGARI events")
    actor_event = actor_events[0]
    from_who = _seat(actor_event, "fromWho")
    method: WinMethod = "tsumo" if from_who == actor else "ron"
    return KyokuOutcome("kokushi_win", True, method, winners, from_who)


def find_accepted_riichi_tj(
    payload: bytes,
    *,
    kyoku_index: int,
    actor: int,
) -> int:
    """Return the Tenhou viewer event index for the actor's accepted riichi."""
    if type(kyoku_index) is not int or kyoku_index < 0:
        raise ValueError("kyoku_index must be a non-negative integer")
    if type(actor) is not int or actor not in range(4):
        raise ValueError("actor must be an integer from 0 through 3")
    root = _parse_root(payload)
    current_kyoku = -1
    viewer_tj = -1
    matches: list[int] = []
    for event in root:
        if event.tag == "INIT":
            if current_kyoku == kyoku_index:
                break
            current_kyoku += 1
            viewer_tj = -1
            continue
        if current_kyoku != kyoku_index:
            continue
        viewer_tj += 1
        if (
            event.tag == "REACH"
            and event.attrib.get("step") == "2"
            and _seat(event, "who") == actor
        ):
            matches.append(viewer_tj)
    if len(matches) != 1:
        raise ValueError(
            "target kyoku must contain exactly one accepted riichi for the actor"
        )
    return matches[0]


def enrich_result_document(
    document: Mapping[str, Any],
    db_root: str | Path,
) -> dict[str, Any]:
    """Return a copy of an extraction document with outcome fields added."""
    records_raw = document.get("records")
    if not isinstance(records_raw, list):
        raise TypeError("result document records must be a list")
    records = [_validate_record(record) for record in records_raw]
    root = Path(db_root).resolve()

    by_year: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_year[record["year"]].append(record)

    payloads: dict[tuple[int, str], bytes] = {}
    for year, annual_records in sorted(by_year.items()):
        database = root / f"{year}.db"
        if not database.is_file():
            raise FileNotFoundError(f"annual database does not exist: {database}")
        log_ids = sorted({record["log_id"] for record in annual_records})
        placeholders = ",".join("?" for _ in log_ids)
        connection = sqlite3.connect(
            f"file:{database.as_posix()}?mode=ro",
            uri=True,
        )
        try:
            rows = connection.execute(
                f"SELECT id, log FROM logs WHERE id IN ({placeholders})",
                log_ids,
            ).fetchall()
        finally:
            connection.close()
        for log_id, payload in rows:
            if payload is not None:
                payloads[(year, log_id)] = bytes(payload)
        missing = sorted(log_id for log_id in log_ids if (year, log_id) not in payloads)
        if missing:
            raise ValueError(
                f"{database.name}: missing payloads for {len(missing)} target logs: "
                + ", ".join(missing[:5])
            )

    enriched_records: list[dict[str, Any]] = []
    for record in records:
        payload = payloads[(record["year"], record["log_id"])]
        outcome = classify_kyoku_outcome(
            payload,
            kyoku_index=record["kyoku_index"],
            actor=record["who"],
        )
        riichi_tj = find_accepted_riichi_tj(
            payload,
            kyoku_index=record["kyoku_index"],
            actor=record["who"],
        )
        enriched_records.append(
            {
                **record,
                "riichi_tj": riichi_tj,
                "outcome": outcome.outcome,
                "won": outcome.won,
                "win_method": outcome.win_method,
                "winners": list(outcome.winners),
                "from_who": outcome.from_who,
            }
        )

    outcome_counts = Counter(record["outcome"] for record in enriched_records)
    method_counts = Counter(
        record["win_method"]
        for record in enriched_records
        if record["win_method"] is not None
    )
    total = len(enriched_records)
    wins = outcome_counts["kokushi_win"]
    return {
        "source": {
            "input_summary": document.get("summary"),
            "record_count": total,
        },
        "summary": {
            "total": total,
            "kokushi_wins": wins,
            "not_won": total - wins,
            "other_wins": outcome_counts["other_win"],
            "draws": outcome_counts["draw"],
            "ron_wins": method_counts["ron"],
            "tsumo_wins": method_counts["tsumo"],
            "win_rate": wins / total if total else None,
        },
        "records": enriched_records,
    }


def load_result_json(path: str | Path) -> dict[str, Any]:
    """Load one UTF-8 extraction result document."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError("result document must be a JSON object")
    return document


def write_outcome_json(document: Mapping[str, Any], path: str | Path) -> None:
    """Atomically write one outcome document."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.part")
    text = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    temporary.write_text(f"{text}\n", encoding="utf-8", newline="\n")
    temporary.replace(output)


def _parse_root(payload: bytes) -> ET.Element:
    if not isinstance(payload, bytes) or not payload:
        raise TypeError("payload must be non-empty bytes")
    try:
        xml = gzip.decompress(payload) if payload.startswith(b"\x1f\x8b") else payload
    except (gzip.BadGzipFile, EOFError, OSError, zlib.error) as error:
        raise KokushiRiichiParseError(f"invalid gzip payload: {error}") from error
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as error:
        if "duplicate attribute" not in str(error):
            raise KokushiRiichiParseError(f"invalid XML: {error}") from error
        normalized_xml, removed_count = _remove_identical_terminal_attributes(xml)
        if removed_count == 0:
            raise KokushiRiichiParseError(f"invalid XML: {error}") from error
        try:
            root = ET.fromstring(normalized_xml)
        except ET.ParseError as retry_error:
            raise KokushiRiichiParseError(
                "invalid XML after duplicate terminal attribute normalization: "
                f"{retry_error}"
            ) from retry_error
    if root.tag != "mjloggm":
        raise KokushiRiichiParseError("root element must be mjloggm")
    return root


def _seat(event: ET.Element, attribute: str) -> int:
    raw = event.attrib.get(attribute)
    try:
        seat = int(raw) if raw is not None else -1
    except ValueError as error:
        raise ValueError(f"{event.tag} {attribute} is invalid") from error
    if seat not in range(4):
        raise ValueError(f"{event.tag} {attribute} must be from 0 through 3")
    return seat


def _validate_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("each result record must be an object")
    record = dict(value)
    required = ("year", "log_id", "kyoku_index", "who")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"result record is missing fields: {', '.join(missing)}")
    if type(record["year"]) is not int:
        raise ValueError("record year must be an integer")
    if not isinstance(record["log_id"], str) or not record["log_id"]:
        raise ValueError("record log_id must be a non-empty string")
    if type(record["kyoku_index"]) is not int or record["kyoku_index"] < 0:
        raise ValueError("record kyoku_index must be a non-negative integer")
    if type(record["who"]) is not int or record["who"] not in range(4):
        raise ValueError("record who must be from 0 through 3")
    return record
