"""Extract established kokushi-musou riichis from Tenhou SQLite archives."""

from __future__ import annotations

import gzip
import json
import re
import sqlite3
import xml.etree.ElementTree as ET
import zlib
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode

from mahjong_analysis.hand_waits import calculate_hand_waits
from mahjong_analysis.tiles import index_to_tile

SUPPORTED_YEARS: tuple[int, ...] = tuple(range(2009, 2026))
ReachType = Literal["riichi", "double_riichi"]
KokushiWaitType = Literal["kokushi_single", "kokushi_13men"]

_DRAW_ACTORS = {"T": 0, "U": 1, "V": 2, "W": 3}
_DISCARD_ACTORS = {"D": 0, "E": 1, "F": 2, "G": 3}
_ORPHAN_KINDS = frozenset((0, 8, 9, 17, 18, 26, *range(27, 34)))
_REQUIRED_COLUMNS = frozenset(
    {
        "id",
        "date",
        "num_players",
        "is_tonpu",
        "is_processed",
        "was_error",
        "log",
    }
)
_TERMINAL_TAG_PATTERN = re.compile(rb"<(?:AGARI|RYUUKYOKU)\b[^<>]*>")
_XML_ATTRIBUTE_PATTERN = re.compile(
    rb"\s+([A-Za-z_:][A-Za-z0-9_.:-]*)=(\"[^\"]*\"|'[^']*')"
)


class KokushiRiichiParseError(ValueError):
    """A source log cannot be replayed without guessing."""


@dataclass(frozen=True)
class KokushiRiichiRecord:
    """One established riichi whose post-discard hand is kokushi tenpai."""

    year: int
    date: str
    log_id: str
    kyoku_index: int
    who: int
    turn: int
    reach_type: ReachType
    wait_type: KokushiWaitType
    waits: tuple[str, ...]
    url: str

    def __post_init__(self) -> None:
        if type(self.year) is not int or self.year not in SUPPORTED_YEARS:
            raise ValueError("year is outside the supported archive range")
        if not isinstance(self.date, str) or not self.date:
            raise TypeError("date must be a non-empty string")
        if not isinstance(self.log_id, str) or not self.log_id:
            raise TypeError("log_id must be a non-empty string")
        if type(self.kyoku_index) is not int or self.kyoku_index < 0:
            raise ValueError("kyoku_index must be a non-negative integer")
        if type(self.who) is not int or self.who not in range(4):
            raise ValueError("who must be an integer from 0 through 3")
        if type(self.turn) is not int or self.turn < 1:
            raise ValueError("turn must be a positive integer")
        if self.reach_type not in ("riichi", "double_riichi"):
            raise ValueError("unsupported reach_type")
        if self.wait_type not in ("kokushi_single", "kokushi_13men"):
            raise ValueError("unsupported wait_type")
        if not isinstance(self.waits, tuple):
            raise TypeError("waits must be a tuple")
        expected_count = 13 if self.wait_type == "kokushi_13men" else 1
        if len(self.waits) != expected_count or len(set(self.waits)) != expected_count:
            raise ValueError("wait count does not match wait_type")
        if self.url != tenhou_replay_url(self.log_id, self.who, self.kyoku_index):
            raise ValueError("url does not match record identity")


@dataclass(frozen=True)
class KokushiRiichiYearSummary:
    """Counters for one annual archive."""

    year: int
    eligible_logs: int
    available_logs: int
    scanned_logs: int
    unavailable_logs: int
    normalized_xml_logs: int
    post_discard_reach_sequences: int
    total_kyokus: int
    total_riichis: int
    kokushi_riichis: int
    kokushi_13men: int

    def __post_init__(self) -> None:
        if type(self.year) is not int or self.year not in SUPPORTED_YEARS:
            raise ValueError("year is outside the supported archive range")
        for name in (
            "eligible_logs",
            "available_logs",
            "scanned_logs",
            "unavailable_logs",
            "normalized_xml_logs",
            "post_discard_reach_sequences",
            "total_kyokus",
            "total_riichis",
            "kokushi_riichis",
            "kokushi_13men",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.available_logs + self.unavailable_logs != self.eligible_logs:
            raise ValueError(
                "available and unavailable logs must partition eligible logs"
            )
        if self.scanned_logs > self.available_logs:
            raise ValueError("scanned_logs cannot exceed available_logs")
        if self.normalized_xml_logs > self.scanned_logs:
            raise ValueError("normalized_xml_logs cannot exceed scanned_logs")
        if self.post_discard_reach_sequences > self.total_riichis:
            raise ValueError("post_discard_reach_sequences cannot exceed total_riichis")
        if self.kokushi_riichis > self.total_riichis:
            raise ValueError("kokushi_riichis cannot exceed total_riichis")
        if self.kokushi_13men > self.kokushi_riichis:
            raise ValueError("kokushi_13men cannot exceed kokushi_riichis")


@dataclass(frozen=True)
class KokushiRiichiResult:
    """Deterministic multi-year extraction result."""

    years: tuple[int, ...]
    max_logs_per_year: int | None
    yearly: tuple[KokushiRiichiYearSummary, ...]
    records: tuple[KokushiRiichiRecord, ...]

    def __post_init__(self) -> None:
        if not self.years or self.years != tuple(sorted(set(self.years))):
            raise ValueError("years must be non-empty, unique, and sorted")
        if tuple(summary.year for summary in self.yearly) != self.years:
            raise ValueError("yearly summaries must match years")
        if self.max_logs_per_year is not None and (
            type(self.max_logs_per_year) is not int or self.max_logs_per_year < 1
        ):
            raise ValueError("max_logs_per_year must be a positive integer or None")
        if sum(summary.kokushi_riichis for summary in self.yearly) != len(self.records):
            raise ValueError("yearly kokushi count does not match records")
        keys = [
            (record.year, record.date, record.log_id, record.kyoku_index, record.who)
            for record in self.records
        ]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("records must be unique and deterministically sorted")

    @property
    def total_riichis(self) -> int:
        return sum(summary.total_riichis for summary in self.yearly)

    @property
    def kokushi_riichis(self) -> int:
        return len(self.records)

    @property
    def kokushi_13men(self) -> int:
        return sum(record.wait_type == "kokushi_13men" for record in self.records)


@dataclass(frozen=True)
class _LogResult:
    total_kyokus: int
    total_riichis: int
    normalized_terminal_attributes: bool
    post_discard_reach_sequences: int
    records: tuple[KokushiRiichiRecord, ...]


@dataclass
class _PendingReach:
    actor: int
    reach_type: ReachType
    turn: int | None = None
    kokushi: tuple[KokushiWaitType, tuple[str, ...]] | None = None
    post_discard_sequence: bool = False


def tenhou_replay_url(log_id: str, who: int, kyoku_index: int) -> str:
    """Return an HTML5 replay URL focused on the matching player and kyoku."""
    if not isinstance(log_id, str) or not log_id:
        raise TypeError("log_id must be a non-empty string")
    if type(who) is not int or who not in range(4):
        raise ValueError("who must be an integer from 0 through 3")
    if type(kyoku_index) is not int or kyoku_index < 0:
        raise ValueError("kyoku_index must be a non-negative integer")
    query = urlencode({"log": log_id, "tw": who, "ts": kyoku_index})
    return f"https://tenhou.net/5/?{query}"


def classify_kokushi_wait(
    concealed_tile_ids: Sequence[int],
) -> tuple[KokushiWaitType, tuple[str, ...]] | None:
    """Classify kokushi tenpai from exactly 13 physical Tenhou tile IDs."""
    tile_ids = tuple(concealed_tile_ids)
    if len(tile_ids) != 13:
        raise ValueError("kokushi classification requires exactly 13 tiles")
    _validate_tile_ids(tile_ids, "concealed hand")
    # Necessary conditions only: the existing wait solver remains authoritative.
    # Avoid solving standard-hand decompositions for millions of ordinary riichis.
    kinds = {tile_id // 4 for tile_id in tile_ids}
    if not kinds <= _ORPHAN_KINDS or len(kinds) < 12:
        return None
    tiles = tuple(index_to_tile(tile_id // 4) for tile_id in tile_ids)
    waits = calculate_hand_waits(tiles)
    details = tuple(
        detail for detail in waits.wait_details if detail.hand_type == "kokushi"
    )
    if not details:
        return None
    wait_tiles = tuple(
        tile for tile in waits.wait_tiles if any(d.wait_tile == tile for d in details)
    )
    shapes = {detail.wait_shape for detail in details}
    if shapes == {"kokushi_13men"} and len(wait_tiles) == 13:
        return "kokushi_13men", wait_tiles
    if shapes == {"kokushi_single"} and len(wait_tiles) == 1:
        return "kokushi_single", wait_tiles
    raise KokushiRiichiParseError("inconsistent kokushi wait classification")


def analyze_tenhou_log(
    payload: bytes,
    *,
    year: int,
    date: str,
    log_id: str,
) -> _LogResult:
    """Replay one gzip/plain Tenhou XML log and return established riichis."""
    if type(year) is not int or year not in SUPPORTED_YEARS:
        raise ValueError("year is outside the supported archive range")
    if not isinstance(date, str) or not date:
        raise TypeError("date must be a non-empty string")
    if not isinstance(log_id, str) or not log_id:
        raise TypeError("log_id must be a non-empty string")
    if not isinstance(payload, bytes) or not payload:
        raise TypeError("payload must be non-empty bytes")

    try:
        xml = gzip.decompress(payload) if payload.startswith(b"\x1f\x8b") else payload
    except (gzip.BadGzipFile, EOFError, OSError, zlib.error) as error:
        raise KokushiRiichiParseError(f"invalid gzip payload: {error}") from error
    normalized_terminal_attributes = False
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
                f"invalid XML after duplicate terminal attribute normalization: {retry_error}"
            ) from retry_error
        normalized_terminal_attributes = True
    if root.tag != "mjloggm":
        raise KokushiRiichiParseError("root element must be mjloggm")

    hands: list[list[int]] | None = None
    last_drawn: list[int | None] = [None] * 4
    discards = [0] * 4
    fixed_melds = [0] * 4
    open_hands = [False] * 4
    accepted_actors: set[int] = set()
    round_ended = True
    any_call = False
    pending: _PendingReach | None = None
    kyoku_index = -1
    total_riichis = 0
    post_discard_reach_sequences = 0
    records: list[KokushiRiichiRecord] = []

    for event_index, event in enumerate(root):
        tag = event.tag
        if tag == "INIT":
            if pending is not None:
                raise _parse_error(event_index, "new INIT before reach sequence ended")
            if not round_ended:
                raise _parse_error(event_index, "new INIT before previous kyoku result")
            kyoku_index += 1
            hands = [
                _parse_initial_hand(event, actor, event_index) for actor in range(4)
            ]
            _validate_unique_initial_tiles(hands, event_index)
            last_drawn = [None] * 4
            discards = [0] * 4
            fixed_melds = [0] * 4
            open_hands = [False] * 4
            accepted_actors = set()
            round_ended = False
            any_call = False
            continue
        if hands is None:
            if tag in {"REACH", "N", "AGARI", "RYUUKYOKU"} or (
                _draw_actor(tag) is not None or _discard_actor(tag) is not None
            ):
                raise _parse_error(event_index, "state event before INIT")
            continue

        if tag in {"SHUFFLE", "GO", "UN", "TAIKYOKU", "BYE", "DORA"}:
            continue
        if round_ended and tag != "AGARI":
            raise _parse_error(event_index, "state event after kyoku result")
        if pending is not None:
            if pending.turn is None and _discard_actor(tag) != pending.actor:
                raise _parse_error(
                    event_index, "reach must be followed by actor discard"
                )
            if pending.turn is not None and (
                _draw_actor(tag) is not None or _discard_actor(tag) is not None
            ):
                raise _parse_error(event_index, "draw/discard before reach acceptance")

        draw_actor = _draw_actor(tag)
        if draw_actor is not None:
            tile_id = _tile_id_from_tag(tag, event_index, allow_missing=False)
            if open_hands[draw_actor]:
                # Open hands cannot become kokushi or legally declare riichi.
                # Their later physical-tile bookkeeping is outside this search.
                continue
            if tile_id in hands[draw_actor]:
                raise _parse_error(
                    event_index, "draw duplicates a concealed physical tile"
                )
            hands[draw_actor].append(tile_id)
            last_drawn[draw_actor] = tile_id
            continue

        discard_actor = _discard_actor(tag)
        if discard_actor is not None:
            if open_hands[discard_actor]:
                discards[discard_actor] += 1
                continue
            tile_id = _tile_id_from_tag(tag, event_index, allow_missing=True)
            if tile_id is None:
                tile_id = last_drawn[discard_actor]
                if tile_id is None:
                    raise _parse_error(event_index, "implicit tsumogiri has no draw")
            _remove_exact_tiles(hands[discard_actor], (tile_id,), event_index)
            discards[discard_actor] += 1
            last_drawn[discard_actor] = None
            if pending is not None and pending.turn is None:
                if discard_actor != pending.actor:
                    raise _parse_error(
                        event_index,
                        "reach declaration discard actor does not match",
                    )
                pending.turn = discards[discard_actor]
                if fixed_melds[discard_actor] == 0:
                    if len(hands[discard_actor]) != 13:
                        raise _parse_error(
                            event_index,
                            "closed declaration hand does not contain 13 tiles",
                        )
                    pending.kokushi = classify_kokushi_wait(hands[discard_actor])
            continue

        if tag == "N":
            actor = _parse_actor_attribute(event, event_index)
            try:
                meld_code = int(event.attrib["m"])
            except (KeyError, ValueError) as error:
                raise _parse_error(
                    event_index, "N has an invalid m attribute"
                ) from error
            any_call = True
            if open_hands[actor]:
                continue
            consumed = _decode_concealed_meld_tiles(meld_code)
            _validate_tile_ids(consumed, "meld consumed tiles")
            _remove_exact_tiles(hands[actor], consumed, event_index)
            is_kakan = bool(meld_code & 0x10 and not meld_code & 0xC)
            if not is_kakan:
                fixed_melds[actor] += 1
            is_ankan = not (meld_code & 0x3F)
            if not is_ankan:
                open_hands[actor] = True
            last_drawn[actor] = None
            continue

        if tag == "REACH":
            actor = _parse_actor_attribute(event, event_index)
            step = event.attrib.get("step")
            if step == "1":
                if pending is not None:
                    raise _parse_error(event_index, "overlapping reach declarations")
                if actor in accepted_actors:
                    raise _parse_error(
                        event_index, "duplicate riichi by the same actor"
                    )
                if open_hands[actor]:
                    raise _parse_error(event_index, "riichi actor has open melds")
                pre_discard_count = 14 - 3 * fixed_melds[actor]
                post_discard_count = pre_discard_count - 1
                if len(hands[actor]) == pre_discard_count:
                    reach_type: ReachType = (
                        "double_riichi"
                        if discards[actor] == 0 and not any_call
                        else "riichi"
                    )
                    pending = _PendingReach(actor=actor, reach_type=reach_type)
                elif len(hands[actor]) == post_discard_count:
                    if (
                        event_index == 0
                        or _discard_actor(root[event_index - 1].tag) != actor
                    ):
                        raise _parse_error(
                            event_index,
                            "post-discard reach is not immediately preceded by actor discard",
                        )
                    if discards[actor] < 1:
                        raise _parse_error(
                            event_index,
                            "post-discard reach has no actor discard count",
                        )
                    reach_type = (
                        "double_riichi"
                        if discards[actor] == 1 and not any_call
                        else "riichi"
                    )
                    kokushi = (
                        classify_kokushi_wait(hands[actor])
                        if fixed_melds[actor] == 0
                        else None
                    )
                    pending = _PendingReach(
                        actor=actor,
                        reach_type=reach_type,
                        turn=discards[actor],
                        kokushi=kokushi,
                        post_discard_sequence=True,
                    )
                else:
                    raise _parse_error(
                        event_index,
                        "reach declaration hand has an invalid tile count",
                    )
            elif step == "2":
                if pending is None or pending.turn is None:
                    raise _parse_error(
                        event_index, "reach acceptance has no declaration"
                    )
                if actor != pending.actor:
                    raise _parse_error(
                        event_index, "reach acceptance actor does not match"
                    )
                total_riichis += 1
                accepted_actors.add(actor)
                post_discard_reach_sequences += pending.post_discard_sequence
                if pending.kokushi is not None:
                    wait_type, wait_tiles = pending.kokushi
                    records.append(
                        KokushiRiichiRecord(
                            year=year,
                            date=date,
                            log_id=log_id,
                            kyoku_index=kyoku_index,
                            who=actor,
                            turn=pending.turn,
                            reach_type=pending.reach_type,
                            wait_type=wait_type,
                            waits=wait_tiles,
                            url=tenhou_replay_url(log_id, actor, kyoku_index),
                        )
                    )
                pending = None
            else:
                raise _parse_error(event_index, "REACH has an invalid step")
            continue

        if tag in {"AGARI", "RYUUKYOKU"}:
            pending = None
            round_ended = True
        else:
            raise _parse_error(event_index, f"unsupported event {tag!r}")

    if kyoku_index < 0:
        raise KokushiRiichiParseError("log contains no INIT events")
    if pending is not None:
        raise KokushiRiichiParseError("log ended before reach sequence ended")
    if not round_ended:
        raise KokushiRiichiParseError("log ended before kyoku result")
    return _LogResult(
        total_kyokus=kyoku_index + 1,
        total_riichis=total_riichis,
        normalized_terminal_attributes=normalized_terminal_attributes,
        post_discard_reach_sequences=post_discard_reach_sequences,
        records=tuple(records),
    )


def scan_kokushi_riichi(
    db_root: str | Path,
    *,
    years: Sequence[int] = SUPPORTED_YEARS,
    workers: int = 1,
    max_logs_per_year: int | None = None,
    on_year_complete: Callable[[KokushiRiichiYearSummary], None] | None = None,
) -> KokushiRiichiResult:
    """Scan every available payload, reporting missing ones and rejecting malformed ones."""
    selected_years = tuple(years)
    if not selected_years:
        raise ValueError("years must not be empty")
    if any(
        type(year) is not int or year not in SUPPORTED_YEARS for year in selected_years
    ):
        raise ValueError("years must be integers from 2009 through 2025")
    if len(set(selected_years)) != len(selected_years):
        raise ValueError("years must not contain duplicates")
    selected_years = tuple(sorted(selected_years))
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if max_logs_per_year is not None and (
        type(max_logs_per_year) is not int or max_logs_per_year < 1
    ):
        raise ValueError("max_logs_per_year must be a positive integer or None")

    root = Path(db_root).resolve()
    tasks: list[tuple[str, int, int | None]] = []
    for year in selected_years:
        database = root / f"{year}.db"
        if not database.is_file():
            raise FileNotFoundError(f"annual database does not exist: {database}")
        tasks.append((str(database), year, max_logs_per_year))

    outcomes = []
    if workers == 1:
        for task in tasks:
            outcome = _scan_annual_database_safe(task)
            outcomes.append(outcome)
            if on_year_complete is not None and outcome[0] is not None:
                on_year_complete(outcome[0])
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            futures = [
                executor.submit(_scan_annual_database_safe, task) for task in tasks
            ]
            for future in as_completed(futures):
                outcome = future.result()
                outcomes.append(outcome)
                if on_year_complete is not None and outcome[0] is not None:
                    on_year_complete(outcome[0])
    errors = tuple(error for _, _, error in outcomes if error is not None)
    if errors:
        raise ValueError("annual scan errors:\n" + "\n".join(errors))
    annual = sorted(
        ((summary, records) for summary, records, _ in outcomes if summary is not None),
        key=lambda item: item[0].year,
    )

    yearly = tuple(item[0] for item in annual)
    records = tuple(
        sorted(
            (record for _, annual_records in annual for record in annual_records),
            key=lambda record: (
                record.year,
                record.date,
                record.log_id,
                record.kyoku_index,
                record.who,
            ),
        )
    )
    return KokushiRiichiResult(
        years=selected_years,
        max_logs_per_year=max_logs_per_year,
        yearly=yearly,
        records=records,
    )


def result_to_dict(result: KokushiRiichiResult) -> dict[str, Any]:
    """Return the stable JSON-compatible public document."""
    if not isinstance(result, KokushiRiichiResult):
        raise TypeError("result must be KokushiRiichiResult")
    return {
        "scope": {
            "archive_format": "tenhou_xml_sqlite_v1.2.0",
            "game_type": "houou",
            "is_tonpu": False,
            "max_logs_per_year": result.max_logs_per_year,
            "num_players": 4,
            "years": list(result.years),
        },
        "summary": {
            "available_logs": sum(item.available_logs for item in result.yearly),
            "eligible_logs": sum(item.eligible_logs for item in result.yearly),
            "kokushi_13men": result.kokushi_13men,
            "kokushi_riichis": result.kokushi_riichis,
            "normalized_xml_logs": sum(
                item.normalized_xml_logs for item in result.yearly
            ),
            "post_discard_reach_sequences": sum(
                item.post_discard_reach_sequences for item in result.yearly
            ),
            "scanned_logs": sum(item.scanned_logs for item in result.yearly),
            "total_kyokus": sum(item.total_kyokus for item in result.yearly),
            "total_riichis": result.total_riichis,
            "unavailable_logs": sum(item.unavailable_logs for item in result.yearly),
        },
        "yearly": [
            {
                "available_logs": item.available_logs,
                "eligible_logs": item.eligible_logs,
                "kokushi_13men": item.kokushi_13men,
                "kokushi_riichis": item.kokushi_riichis,
                "normalized_xml_logs": item.normalized_xml_logs,
                "post_discard_reach_sequences": item.post_discard_reach_sequences,
                "scanned_logs": item.scanned_logs,
                "total_kyokus": item.total_kyokus,
                "total_riichis": item.total_riichis,
                "unavailable_logs": item.unavailable_logs,
                "year": item.year,
            }
            for item in result.yearly
        ],
        "records": [
            {
                "date": record.date,
                "kyoku_index": record.kyoku_index,
                "log_id": record.log_id,
                "reach_type": record.reach_type,
                "turn": record.turn,
                "url": record.url,
                "wait_type": record.wait_type,
                "waits": list(record.waits),
                "who": record.who,
                "year": record.year,
            }
            for record in result.records
        ],
    }


def write_result_json(result: KokushiRiichiResult, path: str | Path) -> None:
    """Atomically write one indented UTF-8 result document."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.part")
    document = json.dumps(
        result_to_dict(result),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    temporary.write_text(f"{document}\n", encoding="utf-8", newline="\n")
    temporary.replace(output)


def _scan_annual_database(
    task: tuple[str, int, int | None],
) -> tuple[KokushiRiichiYearSummary, tuple[KokushiRiichiRecord, ...]]:
    database_text, year, max_logs = task
    database = Path(database_text)
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(logs)").fetchall()
        }
        missing = sorted(_REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(
                f"{database.name}: logs table is missing columns {missing}"
            )
        scope = "num_players = 4 AND is_tonpu = 0"
        parameters: tuple[object, ...] = ()
        if "game_type" in columns:
            scope += " AND game_type = ?"
            parameters = ("houou",)
        eligible_logs, available_logs = connection.execute(
            f"SELECT COUNT(*), COUNT(log) FROM logs WHERE {scope}",
            parameters,
        ).fetchone()
        unavailable_logs = eligible_logs - available_logs
        sql = f"SELECT id, date, log FROM logs WHERE {scope} AND log IS NOT NULL ORDER BY id"
        query_parameters: tuple[object, ...] = parameters
        if max_logs is not None:
            sql += " LIMIT ?"
            query_parameters = (*parameters, max_logs)
        cursor = connection.execute(sql, query_parameters)
        scanned_logs = 0
        total_kyokus = 0
        total_riichis = 0
        normalized_xml_logs = 0
        post_discard_reach_sequences = 0
        records: list[KokushiRiichiRecord] = []
        for log_id, date, payload in cursor:
            scanned_logs += 1
            try:
                log_result = analyze_tenhou_log(
                    payload,
                    year=year,
                    date=date,
                    log_id=log_id,
                )
            except (KokushiRiichiParseError, TypeError, ValueError) as error:
                raise ValueError(
                    f"{database.name}, log_id={log_id}: {error}"
                ) from error
            total_kyokus += log_result.total_kyokus
            total_riichis += log_result.total_riichis
            normalized_xml_logs += log_result.normalized_terminal_attributes
            post_discard_reach_sequences += log_result.post_discard_reach_sequences
            records.extend(log_result.records)
    finally:
        connection.close()

    records_tuple = tuple(records)
    summary = KokushiRiichiYearSummary(
        year=year,
        eligible_logs=eligible_logs,
        available_logs=available_logs,
        scanned_logs=scanned_logs,
        unavailable_logs=unavailable_logs,
        normalized_xml_logs=normalized_xml_logs,
        post_discard_reach_sequences=post_discard_reach_sequences,
        total_kyokus=total_kyokus,
        total_riichis=total_riichis,
        kokushi_riichis=len(records_tuple),
        kokushi_13men=sum(
            record.wait_type == "kokushi_13men" for record in records_tuple
        ),
    )
    return summary, records_tuple


def _scan_annual_database_safe(
    task: tuple[str, int, int | None],
) -> tuple[
    KokushiRiichiYearSummary | None,
    tuple[KokushiRiichiRecord, ...],
    str | None,
]:
    """Return an error value so one bad year does not hide later-year anomalies."""
    try:
        summary, records = _scan_annual_database(task)
    except (OSError, ValueError, sqlite3.Error) as error:
        return None, (), str(error)
    return summary, records, None


def _parse_initial_hand(event: ET.Element, actor: int, event_index: int) -> list[int]:
    raw = event.attrib.get(f"hai{actor}")
    if raw is None:
        raise _parse_error(event_index, f"INIT is missing hai{actor}")
    try:
        tiles = [int(value) for value in raw.split(",")]
    except ValueError as error:
        raise _parse_error(event_index, f"INIT hai{actor} is invalid") from error
    if len(tiles) != 13:
        raise _parse_error(event_index, f"INIT hai{actor} must contain 13 tiles")
    _validate_tile_ids(tiles, f"INIT hai{actor}")
    return tiles


def _remove_identical_terminal_attributes(xml: bytes) -> tuple[bytes, int]:
    """Remove only byte-identical duplicate attrs on terminal result tags.

    Some archived Tenhou logs repeat the final ``owari`` attribute verbatim.
    Those attributes do not affect hand replay. Conflicting duplicates, or
    duplicates on INIT/N/REACH and other state-bearing tags, remain fatal.
    """
    removed_count = 0

    def normalize_tag(match: re.Match[bytes]) -> bytes:
        nonlocal removed_count
        tag = match.group(0)
        seen: dict[bytes, bytes] = {}
        chunks: list[bytes] = []
        previous_end = 0
        for attribute in _XML_ATTRIBUTE_PATTERN.finditer(tag):
            name, value = attribute.groups()
            if name not in seen:
                seen[name] = value
                continue
            if seen[name] != value:
                raise KokushiRiichiParseError(
                    "terminal result tag has conflicting duplicate attributes"
                )
            chunks.append(tag[previous_end : attribute.start()])
            previous_end = attribute.end()
            removed_count += 1
        if not chunks:
            return tag
        chunks.append(tag[previous_end:])
        return b"".join(chunks)

    return _TERMINAL_TAG_PATTERN.sub(normalize_tag, xml), removed_count


def _validate_unique_initial_tiles(
    hands: Sequence[Sequence[int]], event_index: int
) -> None:
    all_tiles = [tile for hand in hands for tile in hand]
    if len(all_tiles) != len(set(all_tiles)):
        raise _parse_error(event_index, "INIT contains duplicate physical tile IDs")


def _validate_tile_ids(tile_ids: Sequence[int], context: str) -> None:
    if any(type(tile_id) is not int or not 0 <= tile_id < 136 for tile_id in tile_ids):
        raise ValueError(f"{context} contains an invalid Tenhou tile ID")
    if len(tile_ids) != len(set(tile_ids)):
        raise ValueError(f"{context} contains a duplicate physical tile ID")


def _draw_actor(tag: str) -> int | None:
    if not tag or tag[0] not in _DRAW_ACTORS or not tag[1:].isdigit():
        return None
    return _DRAW_ACTORS[tag[0]]


def _discard_actor(tag: str) -> int | None:
    if not tag:
        return None
    first = tag[0].upper()
    if first not in _DISCARD_ACTORS or (tag[1:] and not tag[1:].isdigit()):
        return None
    return _DISCARD_ACTORS[first]


def _tile_id_from_tag(
    tag: str,
    event_index: int,
    *,
    allow_missing: bool,
) -> int | None:
    raw = tag[1:]
    if not raw and allow_missing:
        return None
    try:
        tile_id = int(raw)
    except ValueError as error:
        raise _parse_error(event_index, f"invalid tile tag {tag!r}") from error
    if not 0 <= tile_id < 136:
        raise _parse_error(event_index, f"tile ID is out of range in {tag!r}")
    return tile_id


def _parse_actor_attribute(event: ET.Element, event_index: int) -> int:
    try:
        actor = int(event.attrib["who"])
    except (KeyError, ValueError) as error:
        raise _parse_error(event_index, f"{event.tag} has an invalid who") from error
    if actor not in range(4):
        raise _parse_error(event_index, f"{event.tag} who is out of range")
    return actor


def _decode_concealed_meld_tiles(meld_code: int) -> tuple[int, ...]:
    """Decode only the physical tiles removed from the caller's concealed hand."""
    if type(meld_code) is not int or meld_code < 0:
        raise KokushiRiichiParseError("meld code must be a non-negative integer")
    if meld_code & 0x4:  # chi
        encoded = meld_code >> 10
        called_index = encoded % 3
        base = encoded // 3
        base = (base // 7) * 9 + (base % 7)
        base *= 4
        tiles = [
            base + ((meld_code >> 3) & 0x3),
            base + 4 + ((meld_code >> 5) & 0x3),
            base + 8 + ((meld_code >> 7) & 0x3),
        ]
        return tuple(tile for index, tile in enumerate(tiles) if index != called_index)
    if meld_code & 0x8:  # pon
        unused_index = (meld_code >> 5) & 0x3
        encoded = meld_code >> 9
        called_index = encoded % 3
        base = (encoded // 3) * 4
        tiles = [base + copy for copy in range(4) if copy != unused_index]
        return tuple(tile for index, tile in enumerate(tiles) if index != called_index)
    if meld_code & 0x10:  # added kan
        added_index = (meld_code >> 5) & 0x3
        encoded = meld_code >> 9
        base = (encoded // 3) * 4
        return (base + added_index,)
    if meld_code & 0x20:
        raise KokushiRiichiParseError(
            "north extraction is invalid in a four-player log"
        )

    relative_target = meld_code & 0x3
    called_tile = meld_code >> 8
    base = (called_tile // 4) * 4
    tiles = tuple(base + copy for copy in range(4))
    if relative_target == 0:  # concealed kan
        return tiles
    called_copy = called_tile % 4  # open kan
    return tuple(tile for copy, tile in enumerate(tiles) if copy != called_copy)


def _remove_exact_tiles(
    concealed: list[int],
    tile_ids: Sequence[int],
    event_index: int,
) -> None:
    for tile_id in tile_ids:
        try:
            concealed.remove(tile_id)
        except ValueError as error:
            raise _parse_error(
                event_index,
                f"physical tile {tile_id} is not in the actor's concealed hand",
            ) from error


def _parse_error(event_index: int, message: str) -> KokushiRiichiParseError:
    return KokushiRiichiParseError(f"event {event_index}: {message}")
