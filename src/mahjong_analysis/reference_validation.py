"""Independent reference implementation for dealer double-riichi validation."""

import gzip
import json
from collections import defaultdict
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Literal


RoundResult = Literal["dealer_win", "other_win", "draw"]
_HEX_DIGITS = frozenset("0123456789abcdef")
_REFERENCE_GZIP_MAGIC = b"\x1f\x8b"


@dataclass(frozen=True, order=True)
class CandidateRound:
    """A dealer double-riichi round identifiable in its source file."""

    filename: str
    start_kyoku_line: int
    reach_line: int
    bakaze: str
    kyoku: int
    honba: int
    oya: int
    result: RoundResult

    @property
    def key(self) -> tuple[str, int]:
        """Return the stable key used to compare candidate sets."""
        return self.filename, self.start_kyoku_line


@dataclass(frozen=True)
class ReachAuditRecord:
    """Information for manually auditing a reach near the target condition."""

    filename: str
    start_kyoku_line: int
    reach_line: int
    bakaze: str
    kyoku: int
    honba: int
    oya: int
    reach_actor: int
    actor_prior_dahai_count: int
    ankan_before_reach: bool
    event_after_dahai: str | None
    established_dealer_double_riichi: bool


@dataclass(frozen=True)
class ReferenceGameResult:
    """Reference-analysis results for one MJAI file."""

    target_game: bool
    east_kyokus: int
    candidates: tuple[CandidateRound, ...]
    reach_audits: tuple[ReachAuditRecord, ...]


@dataclass(frozen=True)
class CandidateComparison:
    """Differences between independently detected candidate collections."""

    reference_only: tuple[CandidateRound, ...]
    existing_only: tuple[CandidateRound, ...]
    metadata_mismatches: tuple[tuple[CandidateRound, CandidateRound], ...]
    result_mismatches: tuple[tuple[CandidateRound, CandidateRound], ...]

    @property
    def matches(self) -> bool:
        """Return whether candidate identities, metadata, and results match."""
        return not any(
            (
                self.reference_only,
                self.existing_only,
                self.metadata_mismatches,
                self.result_mismatches,
            )
        )


@dataclass(frozen=True)
class _LocatedEvent:
    line: int
    event: dict[str, Any]


def _reference_rule_code(path: str | PathLike[str]) -> str:
    filename = Path(path).name
    if not filename.endswith(".mjson"):
        raise ValueError(f"invalid MJAI filename: {filename}")

    parts = filename.removesuffix(".mjson").split("-")
    if len(parts) != 4:
        raise ValueError(f"invalid MJAI filename: {filename}")

    date_and_kind, rule_code, room_code, game_id = parts
    valid = (
        len(date_and_kind) == 12
        and date_and_kind[:10].isdigit()
        and date_and_kind[10:] == "gm"
        and len(rule_code) == 4
        and set(rule_code) <= _HEX_DIGITS
        and len(room_code) == 4
        and set(room_code) <= _HEX_DIGITS
        and len(game_id) == 8
        and set(game_id) <= _HEX_DIGITS
    )
    if not valid:
        raise ValueError(f"invalid MJAI filename: {filename}")
    return rule_code


def _read_event(
    path: str | PathLike[str],
    line_number: int,
    line: str,
) -> dict[str, Any]:
    try:
        event = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    if not isinstance(event, dict):
        raise ValueError(f"MJAI event must be an object at {path}:{line_number}")
    return event


def _child_event_after_dahai(
    kyoku: list[_LocatedEvent],
    reach_index: int,
    actor: int,
) -> str | None:
    if reach_index + 2 >= len(kyoku):
        return None
    dahai = kyoku[reach_index + 1].event
    if dahai.get("type") != "dahai" or dahai.get("actor") != actor:
        return None
    event_after_dahai = kyoku[reach_index + 2].event
    event_type = event_after_dahai.get("type")
    return event_type if isinstance(event_type, str) else None


def _classify_result(
    path: str | PathLike[str],
    kyoku: list[_LocatedEvent],
    dealer: int,
) -> RoundResult:
    result_events: list[dict[str, Any]] = []
    result_started = False

    for located_event in kyoku[:-1]:
        event = located_event.event
        event_type = event["type"]
        if event_type in {"hora", "ryukyoku"}:
            result_events.append(event)
            result_started = True
        elif result_started:
            raise ValueError(
                "result events must be contiguous before end_kyoku "
                f"at {path}:{located_event.line}"
            )

    horas = [event for event in result_events if event["type"] == "hora"]
    ryukyokus = [
        event for event in result_events if event["type"] == "ryukyoku"
    ]
    if not result_events:
        raise ValueError(f"kyoku has no result at {path}:{kyoku[0].line}")
    if horas and ryukyokus:
        raise ValueError(
            f"hora and ryukyoku coexist at {path}:{kyoku[0].line}"
        )
    if len(ryukyokus) > 1:
        raise ValueError(
            f"multiple ryukyoku events at {path}:{kyoku[0].line}"
        )
    if ryukyokus:
        return "draw"
    if any(hora["actor"] == dealer for hora in horas):
        return "dealer_win"
    return "other_win"


def _analyze_east_kyoku(
    path: str | PathLike[str],
    kyoku: list[_LocatedEvent],
) -> tuple[CandidateRound | None, tuple[ReachAuditRecord, ...]]:
    start = kyoku[0]
    start_event = start.event
    dealer = start_event["oya"]
    filename = Path(path).name
    discard_counts: defaultdict[int, int] = defaultdict(int)
    ankan_occurred = False
    dealer_reach_evaluated = False
    candidate_reach_line: int | None = None
    audits: list[ReachAuditRecord] = []

    for index, located_event in enumerate(kyoku):
        event = located_event.event
        event_type = event["type"]

        if event_type == "ankan":
            ankan_occurred = True
            continue
        if event_type == "dahai":
            discard_counts[event["actor"]] += 1
            continue
        if event_type != "reach":
            continue

        actor = event["actor"]
        prior_dahai_count = discard_counts[actor]
        event_after_dahai = _child_event_after_dahai(kyoku, index, actor)
        established = False

        if actor == dealer and not dealer_reach_evaluated:
            dealer_reach_evaluated = True
            if index + 1 >= len(kyoku):
                raise ValueError(
                    f"dealer reach must be followed by dahai at "
                    f"{path}:{located_event.line}"
                )

            dahai = kyoku[index + 1]
            if dahai.event["type"] != "dahai":
                raise ValueError(
                    f"dealer reach must be followed by dahai at "
                    f"{path}:{located_event.line}"
                )
            if dahai.event.get("actor") != dealer:
                raise ValueError(
                    f"dealer reach and dahai actors differ at "
                    f"{path}:{located_event.line}"
                )
            if index + 2 >= len(kyoku):
                raise ValueError(
                    f"dealer reach dahai lacks a following event at "
                    f"{path}:{located_event.line}"
                )

            event_after_dahai_event = kyoku[index + 2].event
            event_after_dahai = event_after_dahai_event["type"]
            if event_after_dahai == "reach_accepted":
                if event_after_dahai_event.get("actor") != dealer:
                    raise ValueError(
                        f"dealer reach and reach_accepted actors differ at "
                        f"{path}:{located_event.line}"
                    )
                established = prior_dahai_count == 0 and not ankan_occurred
                if established:
                    candidate_reach_line = located_event.line
            elif event_after_dahai not in {"hora", "ryukyoku"}:
                raise ValueError(
                    "dealer reach dahai must be followed by "
                    f"reach_accepted, hora, or ryukyoku at "
                    f"{path}:{located_event.line}"
                )

        audits.append(
            ReachAuditRecord(
                filename=filename,
                start_kyoku_line=start.line,
                reach_line=located_event.line,
                bakaze=start_event["bakaze"],
                kyoku=start_event["kyoku"],
                honba=start_event["honba"],
                oya=dealer,
                reach_actor=actor,
                actor_prior_dahai_count=prior_dahai_count,
                ankan_before_reach=ankan_occurred,
                event_after_dahai=event_after_dahai,
                established_dealer_double_riichi=established,
            )
        )

    if candidate_reach_line is None:
        return None, tuple(audits)

    result = _classify_result(path, kyoku, dealer)
    candidate = CandidateRound(
        filename=filename,
        start_kyoku_line=start.line,
        reach_line=candidate_reach_line,
        bakaze=start_event["bakaze"],
        kyoku=start_event["kyoku"],
        honba=start_event["honba"],
        oya=dealer,
        result=result,
    )
    return candidate, tuple(audits)


def analyze_reference_mjai(
    path: str | PathLike[str],
) -> ReferenceGameResult:
    """Independently analyze one plain or gzip-compressed MJAI file."""
    rule_code = _reference_rule_code(path)
    target_game: bool | None = None
    current_kyoku: list[_LocatedEvent] | None = None
    east_kyokus = 0
    candidates: list[CandidateRound] = []
    reach_audits: list[ReachAuditRecord] = []

    with open(path, "rb") as file:
        is_gzip = file.read(2) == _REFERENCE_GZIP_MAGIC

    if is_gzip:
        text_file = gzip.open(path, mode="rt", encoding="utf-8")
    else:
        text_file = open(path, encoding="utf-8")

    with text_file as file:
        for line_number, line in enumerate(file, 1):
            event = _read_event(path, line_number, line)

            if target_game is None:
                if event.get("type") != "start_game":
                    raise ValueError(f"first event must be start_game: {path}")
                if "aka_flag" not in event:
                    raise ValueError(f"start_game is missing aka_flag: {path}")
                aka_flag = event["aka_flag"]
                if type(aka_flag) is not bool:
                    raise ValueError(f"start_game aka_flag must be bool: {path}")
                target_game = rule_code == "00a9" and aka_flag is True
                continue

            if not target_game:
                continue

            event_type = event["type"]
            located_event = _LocatedEvent(line_number, event)

            if event_type == "start_kyoku":
                if current_kyoku is not None:
                    raise ValueError(
                        f"start_kyoku inside kyoku at {path}:{line_number}"
                    )
                current_kyoku = [located_event]
            elif event_type == "end_kyoku":
                if current_kyoku is None:
                    raise ValueError(
                        f"end_kyoku without start_kyoku at {path}:{line_number}"
                    )
                current_kyoku.append(located_event)
                if current_kyoku[0].event["bakaze"] == "E":
                    east_kyokus += 1
                    candidate, audits = _analyze_east_kyoku(path, current_kyoku)
                    if candidate is not None:
                        candidates.append(candidate)
                    reach_audits.extend(audits)
                current_kyoku = None
            elif event_type in {"start_game", "end_game"}:
                if current_kyoku is not None:
                    raise ValueError(
                        f"{event_type} inside kyoku at {path}:{line_number}"
                    )
            elif current_kyoku is None:
                raise ValueError(
                    f"{event_type} outside kyoku at {path}:{line_number}"
                )
            else:
                current_kyoku.append(located_event)

    if target_game is None:
        raise ValueError(f"event list is empty: {path}")
    if target_game and current_kyoku is not None:
        raise ValueError(
            f"input ended before end_kyoku at {path}:{current_kyoku[0].line}"
        )

    return ReferenceGameResult(
        target_game=target_game,
        east_kyokus=east_kyokus,
        candidates=tuple(candidates),
        reach_audits=tuple(reach_audits),
    )


def compare_candidate_rounds(
    reference: tuple[CandidateRound, ...],
    existing: tuple[CandidateRound, ...],
) -> CandidateComparison:
    """Compare candidate identities, metadata, and results."""

    def index_by_key(
        records: tuple[CandidateRound, ...],
    ) -> dict[tuple[str, int], CandidateRound]:
        indexed: dict[tuple[str, int], CandidateRound] = {}
        for record in records:
            if record.key in indexed:
                raise ValueError(f"duplicate candidate key: {record.key}")
            indexed[record.key] = record
        return indexed

    reference_by_key = index_by_key(reference)
    existing_by_key = index_by_key(existing)
    reference_keys = set(reference_by_key)
    existing_keys = set(existing_by_key)

    reference_only = tuple(
        reference_by_key[key] for key in sorted(reference_keys - existing_keys)
    )
    existing_only = tuple(
        existing_by_key[key] for key in sorted(existing_keys - reference_keys)
    )
    metadata_mismatches: list[tuple[CandidateRound, CandidateRound]] = []
    result_mismatches: list[tuple[CandidateRound, CandidateRound]] = []

    for key in sorted(reference_keys & existing_keys):
        reference_record = reference_by_key[key]
        existing_record = existing_by_key[key]
        if (
            reference_record.reach_line,
            reference_record.bakaze,
            reference_record.kyoku,
            reference_record.honba,
            reference_record.oya,
        ) != (
            existing_record.reach_line,
            existing_record.bakaze,
            existing_record.kyoku,
            existing_record.honba,
            existing_record.oya,
        ):
            metadata_mismatches.append((reference_record, existing_record))
        if reference_record.result != existing_record.result:
            result_mismatches.append((reference_record, existing_record))

    return CandidateComparison(
        reference_only=reference_only,
        existing_only=existing_only,
        metadata_mismatches=tuple(metadata_mismatches),
        result_mismatches=tuple(result_mismatches),
    )
