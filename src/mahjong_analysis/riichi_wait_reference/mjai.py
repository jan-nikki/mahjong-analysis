"""Independent source-aware MJAI replay for reference riichi candidates."""

from __future__ import annotations

import gzip
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from os import PathLike
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal

from .model import (
    ReferenceActorDiscard,
    ReferenceMeld,
    ReferenceRiichiCandidate,
    normalize_reference_tile,
    reference_tile_to_index,
)
from .waits import calculate_reference_hand_waits

_GZIP_MAGIC = b"\x1f\x8b"
_RED_FIVES = frozenset(("5mr", "5pr", "5sr"))
_RULE_FILENAME = re.compile(
    r"^\d{10}gm-(?P<rule>[0-9a-f]{4})-[0-9a-f]{4}-[0-9a-f]{8}\.mjson$"
)
_KNOWN_KYOKU_EVENTS = frozenset(
    (
        "start_kyoku",
        "tsumo",
        "dahai",
        "chi",
        "pon",
        "daiminkan",
        "ankan",
        "kakan",
        "dora",
        "reach",
        "reach_accepted",
        "hora",
        "ryukyoku",
        "end_kyoku",
    )
)
_OPEN_MELDS = frozenset(("chi", "pon", "daiminkan", "kakan"))

ReplayMeldType = Literal["chi", "pon", "daiminkan", "ankan", "kakan"]


@dataclass(frozen=True)
class ReferenceMjaiEvent:
    """One immutable MJAI event with source and kyoku positions."""

    line_number: int
    data: Mapping[str, Any]
    event_index: int | None = None

    def __post_init__(self) -> None:
        if type(self.line_number) is not int or self.line_number < 1:
            raise ValueError("line_number must be a positive integer")
        if self.event_index is not None and (
            type(self.event_index) is not int or self.event_index < 0
        ):
            raise ValueError("event_index must be a non-negative integer or None")
        if not isinstance(self.data, Mapping):
            raise TypeError("MJAI event data must be a mapping")
        object.__setattr__(self, "data", _freeze_json(self.data))
        if not isinstance(self.data.get("type"), str):
            raise TypeError(f"line {self.line_number}: event type must be a string")


@dataclass(frozen=True)
class ReferenceMjaiLog:
    """Source-aware immutable JSON Lines input."""

    source_path: str
    events: tuple[ReferenceMjaiEvent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_path, str) or not self.source_path:
            raise ValueError("source_path must be a non-empty string")
        if "\\" in self.source_path:
            raise ValueError("source_path must use '/' separators")
        try:
            events = tuple(self.events)
        except TypeError as error:
            raise TypeError(
                "events must be an iterable of ReferenceMjaiEvent"
            ) from error
        if not all(isinstance(event, ReferenceMjaiEvent) for event in events):
            raise TypeError("events must contain only ReferenceMjaiEvent")
        object.__setattr__(self, "events", events)
        lines = tuple(event.line_number for event in self.events)
        if any(left >= right for left, right in pairwise(lines)):
            raise ValueError("event source lines must be strictly increasing")


@dataclass(frozen=True)
class ReferenceKyoku:
    """One inclusive start_kyoku/end_kyoku event group."""

    source_path: str
    events: tuple[ReferenceMjaiEvent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_path, str) or not self.source_path:
            raise ValueError("source_path must be a non-empty string")
        if "\\" in self.source_path:
            raise ValueError("source_path must use '/' separators")
        try:
            events = tuple(self.events)
        except TypeError as error:
            raise TypeError(
                "events must be an iterable of ReferenceMjaiEvent"
            ) from error
        if not all(isinstance(event, ReferenceMjaiEvent) for event in events):
            raise TypeError("events must contain only ReferenceMjaiEvent")
        object.__setattr__(self, "events", events)
        if not self.events:
            raise ValueError("reference kyoku must contain events")
        if _event_type(self.events[0]) != "start_kyoku":
            raise ValueError("reference kyoku must start with start_kyoku")
        if _event_type(self.events[-1]) != "end_kyoku":
            raise ValueError("reference kyoku must end with end_kyoku")
        if tuple(event.event_index for event in self.events) != tuple(
            range(len(self.events))
        ):
            raise ValueError(
                "start_kyoku must be event_index 0 and indexes consecutive"
            )
        lines = tuple(event.line_number for event in self.events)
        if any(left >= right for left, right in pairwise(lines)):
            raise ValueError("kyoku event source lines must be strictly increasing")

    @property
    def start_kyoku_line(self) -> int:
        return self.events[0].line_number


@dataclass
class _ReplayMeld:
    meld_type: ReplayMeldType
    tiles: Counter[str]
    event_index: int


@dataclass
class _ActorState:
    concealed: Counter[str]
    fixed_melds: list[_ReplayMeld]
    discards: list[ReferenceActorDiscard]


@dataclass(frozen=True)
class _CallableDiscard:
    actor: int
    event_index: int
    tile: str


@dataclass
class _PendingReach:
    actor: int
    reach_event_index: int
    reach_line: int
    declaration_dahai_event_index: int | None = None
    declaration_dahai_line: int | None = None
    riichi_discard_number: int | None = None
    declaration_tile: str | None = None
    concealed_after_discard: tuple[str, ...] | None = None
    fixed_melds: tuple[tuple[ReplayMeldType, tuple[str, ...]], ...] | None = None
    discard_count: int | None = None

    @property
    def awaits_dahai(self) -> bool:
        return self.declaration_dahai_event_index is None


@dataclass(frozen=True)
class _AcceptedSnapshot:
    actor: int
    reach_event_index: int
    declaration_dahai_event_index: int
    reach_accepted_event_index: int
    reach_line: int
    declaration_dahai_line: int
    reach_accepted_line: int
    riichi_discard_number: int
    declaration_tile: str
    concealed_after_discard: tuple[str, ...]
    fixed_melds: tuple[ReferenceMeld, ...]
    discard_count: int


def read_reference_mjai(
    path: str | PathLike[str],
    *,
    source_root: str | PathLike[str] | None = None,
) -> ReferenceMjaiLog:
    """Read plain or gzip-compressed UTF-8 MJAI JSON Lines independently."""
    input_path = Path(path)
    source_path = _source_identifier(input_path, source_root)
    with input_path.open("rb") as binary_file:
        compressed = binary_file.read(2) == _GZIP_MAGIC

    events: list[ReferenceMjaiEvent] = []
    open_text = gzip.open if compressed else open
    with open_text(input_path, mode="rt", encoding="utf-8") as lines:
        for line_number, line in enumerate(lines, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{source_path}: line {line_number}: invalid MJAI JSON"
                ) from error
            if not isinstance(value, dict):
                raise TypeError(
                    f"{source_path}: line {line_number}: MJAI event must be an object"
                )
            events.append(
                ReferenceMjaiEvent(
                    line_number=line_number,
                    data=_freeze_json(value),
                )
            )
    return ReferenceMjaiLog(source_path=source_path, events=tuple(events))


def reference_log_from_events(
    events: Iterable[Mapping[str, Any]],
    *,
    source_path: str = "artificial.mjson",
    first_line: int = 1,
) -> ReferenceMjaiLog:
    """Build a source-aware log for independent artificial MJAI tests."""
    normalized_source = source_path.replace("\\", "/")
    if type(first_line) is not int or first_line < 1:
        raise ValueError("first_line must be a positive integer")
    sourced_events = tuple(
        ReferenceMjaiEvent(
            line_number=first_line + offset,
            data=_freeze_json(dict(event)),
        )
        for offset, event in enumerate(events)
    )
    return ReferenceMjaiLog(normalized_source, sourced_events)


def split_reference_kyokus(log: ReferenceMjaiLog) -> tuple[ReferenceKyoku, ...]:
    """Split a source log and assign zero-based indexes within each kyoku."""
    kyokus: list[ReferenceKyoku] = []
    current: list[ReferenceMjaiEvent] | None = None
    for source_event in log.events:
        event_type = _event_type(source_event)
        if event_type == "start_kyoku":
            if current is not None:
                raise _source_error(
                    log.source_path,
                    source_event.line_number,
                    "start_kyoku encountered before end_kyoku",
                )
            current = [replace(source_event, event_index=0)]
        elif event_type == "end_kyoku":
            if current is None:
                raise _source_error(
                    log.source_path,
                    source_event.line_number,
                    "end_kyoku encountered without start_kyoku",
                )
            current.append(replace(source_event, event_index=len(current)))
            kyokus.append(ReferenceKyoku(log.source_path, tuple(current)))
            current = None
        elif event_type in {"start_game", "end_game"}:
            if current is not None:
                raise _source_error(
                    log.source_path,
                    source_event.line_number,
                    f"{event_type} encountered inside kyoku",
                )
        elif current is None:
            raise _source_error(
                log.source_path,
                source_event.line_number,
                f"{event_type} encountered outside kyoku",
            )
        else:
            current.append(replace(source_event, event_index=len(current)))

    if current is not None:
        raise _source_error(
            log.source_path,
            current[0].line_number,
            "input ended before end_kyoku",
        )
    return tuple(kyokus)


def extract_reference_rule_code(source_path: str) -> str:
    """Extract the four-character rule code from a Tenhou MJAI filename."""
    filename = PurePosixPath(source_path).name
    match = _RULE_FILENAME.fullmatch(filename)
    if match is None:
        raise ValueError(f"invalid MJAI filename: {filename!r}")
    return match.group("rule")


def is_reference_target_game(log: ReferenceMjaiLog) -> bool:
    """Return whether the independent input scope is 00a9 with red fives."""
    if not log.events:
        raise ValueError("MJAI log is empty")
    start_game = log.events[0]
    if _event_type(start_game) != "start_game":
        raise _source_error(
            log.source_path,
            start_game.line_number,
            "first event must be start_game",
        )
    aka_flag = start_game.data.get("aka_flag")
    if type(aka_flag) is not bool:
        raise _source_error(
            log.source_path,
            start_game.line_number,
            "start_game aka_flag must be a bool",
        )
    return extract_reference_rule_code(log.source_path) == "00a9" and aka_flag


def is_reference_east_kyoku(kyoku: ReferenceKyoku) -> bool:
    """Return whether a reference kyoku belongs to the east round."""
    bakaze = kyoku.events[0].data.get("bakaze")
    if not isinstance(bakaze, str):
        raise _kyoku_error(kyoku, kyoku.events[0], None, "bakaze must be a string")
    return bakaze == "E"


def filter_reference_east_kyokus(
    kyokus: Iterable[ReferenceKyoku],
) -> tuple[ReferenceKyoku, ...]:
    """Keep east-round kyokus in source order."""
    return tuple(kyoku for kyoku in kyokus if is_reference_east_kyoku(kyoku))


def extract_reference_riichi_candidates(
    kyoku: ReferenceKyoku,
) -> tuple[ReferenceRiichiCandidate, ...]:
    """Replay one kyoku and finalize every independently established riichi."""
    states = _initial_states(kyoku)
    pending: _PendingReach | None = None
    callable_discard: _CallableDiscard | None = None
    accepted: list[_AcceptedSnapshot] = []

    for source_event in kyoku.events:
        event_index = _kyoku_event_index(source_event)
        event_type = _event_type(source_event)
        if event_type not in _KNOWN_KYOKU_EVENTS:
            raise _kyoku_error(
                kyoku, source_event, None, f"unsupported kyoku event: {event_type}"
            )
        if event_index == 0:
            continue
        if event_type == "end_kyoku":
            if event_index != len(kyoku.events) - 1:
                raise _kyoku_error(
                    kyoku, source_event, None, "end_kyoku must be the final event"
                )
            if pending is not None:
                raise _kyoku_error(
                    kyoku,
                    source_event,
                    pending.actor,
                    "pending reach did not end with reach_accepted, hora, or ryukyoku",
                )
            break

        if pending is not None and pending.awaits_dahai:
            if event_type != "dahai":
                raise _kyoku_error(
                    kyoku,
                    source_event,
                    pending.actor,
                    "reach must be followed immediately by dahai",
                )
            if _require_actor(kyoku, source_event) != pending.actor:
                raise _kyoku_error(
                    kyoku,
                    source_event,
                    pending.actor,
                    "reach and declaration dahai actors do not match",
                )
        elif pending is not None:
            pending, snapshot = _finish_pending_reach(kyoku, source_event, pending)
            if snapshot is not None:
                accepted.append(snapshot)
            if event_type == "reach_accepted":
                continue
            callable_discard = None
            continue
        elif event_type == "reach_accepted":
            actor = source_event.data.get("actor")
            raise _kyoku_error(
                kyoku,
                source_event,
                actor if type(actor) is int else None,
                "reach_accepted has no matching pending reach",
            )

        if event_type == "tsumo":
            actor = _require_actor(kyoku, source_event)
            tile = _require_tile(kyoku, source_event, "pai", actor)
            states[actor].concealed[tile] += 1
            _validate_owned_tiles(kyoku, source_event, actor, states[actor])
            callable_discard = None
        elif event_type == "dahai":
            actor = _require_actor(kyoku, source_event)
            tile = _require_tile(kyoku, source_event, "pai", actor)
            tsumogiri = source_event.data.get("tsumogiri")
            if type(tsumogiri) is not bool:
                raise _kyoku_error(
                    kyoku, source_event, actor, "dahai tsumogiri must be a bool"
                )
            _remove_raw_tiles(kyoku, source_event, actor, states[actor], (tile,))
            _validate_all_post_discard_counts(kyoku, source_event, states)
            is_declaration = pending is not None and pending.awaits_dahai
            discard = ReferenceActorDiscard(
                discard_number=len(states[actor].discards) + 1,
                tile=tile,
                tile_kind=normalize_reference_tile(tile),
                tsumogiri=tsumogiri,
                event_index=event_index,
                is_riichi_declaration=is_declaration,
            )
            states[actor].discards.append(discard)
            callable_discard = _CallableDiscard(actor, event_index, tile)
            if is_declaration:
                _capture_declaration(pending, states[actor], source_event, discard)
        elif event_type in {"chi", "pon", "daiminkan"}:
            actor = _require_actor(kyoku, source_event)
            target = _require_target(kyoku, source_event, actor)
            if event_type == "chi" and actor != (target + 1) % 4:
                raise _kyoku_error(
                    kyoku,
                    source_event,
                    actor,
                    "chi actor must be the next actor after target",
                )
            called_tile = _require_tile(kyoku, source_event, "pai", actor)
            consumed = _require_consumed(
                kyoku,
                source_event,
                actor,
                {"chi": 2, "pon": 2, "daiminkan": 3}[event_type],
            )
            meld_tiles = (*consumed, called_tile)
            _validate_call_meld(kyoku, source_event, actor, event_type, meld_tiles)
            _require_callable_discard(
                kyoku,
                source_event,
                actor,
                target,
                called_tile,
                callable_discard,
            )
            _remove_raw_tiles(kyoku, source_event, actor, states[actor], consumed)
            states[actor].fixed_melds.append(
                _ReplayMeld(event_type, Counter(meld_tiles), event_index)
            )
            _validate_owned_tiles(kyoku, source_event, actor, states[actor])
            _mark_discard_called(
                kyoku,
                source_event,
                states[target],
                callable_discard.event_index,
                event_type,
                actor,
            )
            callable_discard = None
        elif event_type == "ankan":
            actor = _require_actor(kyoku, source_event)
            consumed = _require_consumed(kyoku, source_event, actor, 4)
            _validate_same_kind_meld(kyoku, source_event, actor, "ankan", consumed)
            _remove_raw_tiles(kyoku, source_event, actor, states[actor], consumed)
            states[actor].fixed_melds.append(
                _ReplayMeld("ankan", Counter(consumed), event_index)
            )
            _validate_owned_tiles(kyoku, source_event, actor, states[actor])
            callable_discard = None
        elif event_type == "kakan":
            actor = _require_actor(kyoku, source_event)
            added_tile = _require_tile(kyoku, source_event, "pai", actor)
            consumed = _require_consumed(kyoku, source_event, actor, 3)
            _apply_kakan(
                kyoku, source_event, actor, states[actor], consumed, added_tile
            )
            callable_discard = None
        elif event_type == "reach":
            actor = _require_actor(kyoku, source_event)
            pending = _PendingReach(actor, event_index, source_event.line_number)
            callable_discard = None
        elif event_type == "dora":
            callable_discard = None
        elif event_type == "hora":
            _require_actor(kyoku, source_event)
            callable_discard = None
        elif event_type == "ryukyoku":
            callable_discard = None
        else:
            raise _kyoku_error(
                kyoku,
                source_event,
                None,
                f"unexpected event during replay: {event_type}",
            )

    return tuple(_finalize_candidate(kyoku, states, snapshot) for snapshot in accepted)


def _finish_pending_reach(
    kyoku: ReferenceKyoku,
    event: ReferenceMjaiEvent,
    pending: _PendingReach,
) -> tuple[None, _AcceptedSnapshot | None]:
    event_type = _event_type(event)
    if event_type == "reach_accepted":
        actor = _require_actor(kyoku, event)
        if actor != pending.actor:
            raise _kyoku_error(
                kyoku,
                event,
                actor,
                "reach and reach_accepted actors do not match",
            )
        return None, _accept_pending_reach(kyoku, event, pending)
    if event_type == "hora":
        _validate_declaration_hora(kyoku, event, pending)
        return None, None
    if event_type == "ryukyoku":
        return None, None
    raise _kyoku_error(
        kyoku,
        event,
        pending.actor,
        "declaration dahai must be followed immediately by "
        "reach_accepted, hora, or ryukyoku",
    )


def _capture_declaration(
    pending: _PendingReach,
    state: _ActorState,
    event: ReferenceMjaiEvent,
    discard: ReferenceActorDiscard,
) -> None:
    pending.declaration_dahai_event_index = _kyoku_event_index(event)
    pending.declaration_dahai_line = event.line_number
    pending.riichi_discard_number = discard.discard_number
    pending.declaration_tile = discard.tile
    pending.concealed_after_discard = _counter_tiles(state.concealed)
    pending.fixed_melds = tuple(
        (meld.meld_type, _counter_tiles(meld.tiles)) for meld in state.fixed_melds
    )
    pending.discard_count = len(state.discards)


def _accept_pending_reach(
    kyoku: ReferenceKyoku,
    event: ReferenceMjaiEvent,
    pending: _PendingReach,
) -> _AcceptedSnapshot:
    declaration_index = _pending_value(pending.declaration_dahai_event_index)
    declaration_line = _pending_value(pending.declaration_dahai_line)
    discard_number = _pending_value(pending.riichi_discard_number)
    declaration_tile = _pending_value(pending.declaration_tile)
    concealed = _pending_value(pending.concealed_after_discard)
    replay_melds = _pending_value(pending.fixed_melds)
    discard_count = _pending_value(pending.discard_count)
    open_types = [
        meld_type for meld_type, _ in replay_melds if meld_type in _OPEN_MELDS
    ]
    if open_types:
        raise _kyoku_error(
            kyoku,
            event,
            pending.actor,
            f"established riichi actor has open melds: {open_types}",
        )
    fixed_melds = tuple(ReferenceMeld(tiles) for _, tiles in replay_melds)
    return _AcceptedSnapshot(
        actor=pending.actor,
        reach_event_index=pending.reach_event_index,
        declaration_dahai_event_index=declaration_index,
        reach_accepted_event_index=_kyoku_event_index(event),
        reach_line=pending.reach_line,
        declaration_dahai_line=declaration_line,
        reach_accepted_line=event.line_number,
        riichi_discard_number=discard_number,
        declaration_tile=declaration_tile,
        concealed_after_discard=concealed,
        fixed_melds=fixed_melds,
        discard_count=discard_count,
    )


def _validate_declaration_hora(
    kyoku: ReferenceKyoku,
    event: ReferenceMjaiEvent,
    pending: _PendingReach,
) -> None:
    target = event.data.get("target")
    if type(target) is not int or target != pending.actor:
        raise _kyoku_error(
            kyoku, event, pending.actor, "declaration hora target must be reach actor"
        )
    winner = _require_actor(kyoku, event)
    if winner == target:
        raise _kyoku_error(
            kyoku,
            event,
            winner,
            "declaration hora actor must differ from target",
        )
    hora_tile = _require_tile(kyoku, event, "pai", winner)
    declaration_tile = _pending_value(pending.declaration_tile)
    if normalize_reference_tile(hora_tile) != normalize_reference_tile(
        declaration_tile
    ):
        raise _kyoku_error(
            kyoku,
            event,
            pending.actor,
            "declaration hora tile kind does not match declaration dahai",
        )


def _finalize_candidate(
    kyoku: ReferenceKyoku,
    states: list[_ActorState],
    snapshot: _AcceptedSnapshot,
) -> ReferenceRiichiCandidate:
    try:
        waits = calculate_reference_hand_waits(
            snapshot.concealed_after_discard, snapshot.fixed_melds
        )
    except (TypeError, ValueError) as error:
        event = kyoku.events[snapshot.declaration_dahai_event_index]
        raise _kyoku_error(
            kyoku,
            event,
            snapshot.actor,
            f"declaration-post wait calculation failed: {error}",
        ) from error
    if not waits.wait_tiles:
        event = kyoku.events[snapshot.declaration_dahai_event_index]
        raise _kyoku_error(
            kyoku,
            event,
            snapshot.actor,
            "established riichi declaration-post hand has no waits",
        )
    return ReferenceRiichiCandidate(
        source_path=kyoku.source_path,
        start_kyoku_line=kyoku.start_kyoku_line,
        actor=snapshot.actor,
        reach_event_index=snapshot.reach_event_index,
        declaration_dahai_event_index=snapshot.declaration_dahai_event_index,
        reach_accepted_event_index=snapshot.reach_accepted_event_index,
        reach_line=snapshot.reach_line,
        declaration_dahai_line=snapshot.declaration_dahai_line,
        reach_accepted_line=snapshot.reach_accepted_line,
        riichi_discard_number=snapshot.riichi_discard_number,
        riichi_declaration_tile=snapshot.declaration_tile,
        riichi_declaration_tile_kind=normalize_reference_tile(
            snapshot.declaration_tile
        ),
        concealed_tiles_after_discard=snapshot.concealed_after_discard,
        fixed_melds=snapshot.fixed_melds,
        actor_discards_before_riichi=tuple(
            states[snapshot.actor].discards[: snapshot.discard_count]
        ),
        waits=waits,
    )


def _initial_states(kyoku: ReferenceKyoku) -> list[_ActorState]:
    start = kyoku.events[0]
    tehais = start.data.get("tehais")
    if not isinstance(tehais, Sequence) or isinstance(tehais, (str, bytes)):
        raise _kyoku_error(kyoku, start, None, "start_kyoku tehais must be a sequence")
    if len(tehais) != 4:
        raise _kyoku_error(kyoku, start, None, "start_kyoku must contain four tehais")
    states: list[_ActorState] = []
    for actor, tehai in enumerate(tehais):
        if not isinstance(tehai, Sequence) or isinstance(tehai, (str, bytes)):
            raise _kyoku_error(kyoku, start, actor, "initial tehai must be a sequence")
        if len(tehai) != 13:
            raise _kyoku_error(
                kyoku, start, actor, "initial tehai must contain exactly 13 tiles"
            )
        if not all(isinstance(tile, str) for tile in tehai):
            raise _kyoku_error(kyoku, start, actor, "initial tehai contains non-tile")
        state = _ActorState(Counter(tehai), [], [])
        _validate_owned_tiles(kyoku, start, actor, state)
        states.append(state)
    return states


def _require_actor(kyoku: ReferenceKyoku, event: ReferenceMjaiEvent) -> int:
    actor = event.data.get("actor")
    if type(actor) is not int or actor not in range(4):
        raise _kyoku_error(kyoku, event, None, "actor must be an integer from 0 to 3")
    return actor


def _require_target(
    kyoku: ReferenceKyoku, event: ReferenceMjaiEvent, actor: int
) -> int:
    target = event.data.get("target")
    if type(target) is not int or target not in range(4) or target == actor:
        raise _kyoku_error(kyoku, event, actor, "call target is invalid")
    return target


def _require_tile(
    kyoku: ReferenceKyoku,
    event: ReferenceMjaiEvent,
    field: str,
    actor: int,
) -> str:
    tile = event.data.get(field)
    if not isinstance(tile, str):
        raise _kyoku_error(kyoku, event, actor, f"{field} must be a tile string")
    try:
        normalize_reference_tile(tile)
    except (TypeError, ValueError) as error:
        raise _kyoku_error(
            kyoku, event, actor, f"{field} contains invalid tile: {tile!r}"
        ) from error
    return tile


def _require_consumed(
    kyoku: ReferenceKyoku,
    event: ReferenceMjaiEvent,
    actor: int,
    expected_count: int,
) -> tuple[str, ...]:
    consumed = event.data.get("consumed")
    if not isinstance(consumed, Sequence) or isinstance(consumed, (str, bytes)):
        raise _kyoku_error(kyoku, event, actor, "consumed must be a tile sequence")
    if len(consumed) != expected_count:
        raise _kyoku_error(
            kyoku,
            event,
            actor,
            f"consumed must contain exactly {expected_count} tiles",
        )
    result = tuple(consumed)
    for tile in result:
        if not isinstance(tile, str):
            raise _kyoku_error(kyoku, event, actor, "consumed contains non-tile")
        try:
            normalize_reference_tile(tile)
        except (TypeError, ValueError) as error:
            raise _kyoku_error(
                kyoku, event, actor, f"consumed contains invalid tile: {tile!r}"
            ) from error
    return result


def _remove_raw_tiles(
    kyoku: ReferenceKyoku,
    event: ReferenceMjaiEvent,
    actor: int,
    state: _ActorState,
    tiles: tuple[str, ...],
) -> None:
    needed = Counter(tiles)
    missing = needed - state.concealed
    if missing:
        raise _kyoku_error(
            kyoku,
            event,
            actor,
            f"concealed hand lacks raw tiles: {dict(missing)}",
        )
    state.concealed.subtract(needed)
    state.concealed += Counter()


def _validate_owned_tiles(
    kyoku: ReferenceKyoku,
    event: ReferenceMjaiEvent,
    actor: int,
    state: _ActorState,
) -> None:
    owned = state.concealed.copy()
    for meld in state.fixed_melds:
        owned.update(meld.tiles)
    normalized: Counter[str] = Counter()
    for tile, count in owned.items():
        try:
            tile_kind = normalize_reference_tile(tile)
        except (TypeError, ValueError) as error:
            raise _kyoku_error(
                kyoku, event, actor, f"owned state contains invalid tile: {tile!r}"
            ) from error
        normalized[tile_kind] += count
        if tile in _RED_FIVES and count > 1:
            raise _kyoku_error(
                kyoku, event, actor, f"multiple physical copies of red five: {tile}"
            )
    overfull = [tile for tile, count in normalized.items() if count > 4]
    if overfull:
        raise _kyoku_error(
            kyoku,
            event,
            actor,
            f"more than four owned copies of tile kind: {overfull[0]}",
        )


def _validate_all_post_discard_counts(
    kyoku: ReferenceKyoku,
    event: ReferenceMjaiEvent,
    states: list[_ActorState],
) -> None:
    for actor, state in enumerate(states):
        expected = 13 - 3 * len(state.fixed_melds)
        actual = state.concealed.total()
        if actual != expected:
            raise _kyoku_error(
                kyoku,
                event,
                actor,
                f"post-discard concealed count must be {expected}, got {actual}",
            )


def _validate_call_meld(
    kyoku: ReferenceKyoku,
    event: ReferenceMjaiEvent,
    actor: int,
    meld_type: str,
    tiles: tuple[str, ...],
) -> None:
    if meld_type != "chi":
        _validate_same_kind_meld(kyoku, event, actor, meld_type, tiles)
        return
    indices = sorted(reference_tile_to_index(tile) for tile in tiles)
    if (
        indices[-1] >= 27
        or indices[0] // 9 != indices[-1] // 9
        or len(set(indices)) != 3
        or indices != list(range(indices[0], indices[0] + 3))
    ):
        raise _kyoku_error(kyoku, event, actor, "chi tiles do not form a sequence")


def _validate_same_kind_meld(
    kyoku: ReferenceKyoku,
    event: ReferenceMjaiEvent,
    actor: int,
    meld_type: str,
    tiles: tuple[str, ...],
) -> None:
    if len({normalize_reference_tile(tile) for tile in tiles}) != 1:
        raise _kyoku_error(
            kyoku, event, actor, f"{meld_type} tiles do not have one tile kind"
        )


def _require_callable_discard(
    kyoku: ReferenceKyoku,
    event: ReferenceMjaiEvent,
    actor: int,
    target: int,
    called_tile: str,
    callable_discard: _CallableDiscard | None,
) -> None:
    if callable_discard is None:
        raise _kyoku_error(kyoku, event, actor, "call has no preceding callable dahai")
    if callable_discard.actor != target:
        raise _kyoku_error(kyoku, event, actor, "call target does not match last dahai")
    if callable_discard.tile != called_tile:
        raise _kyoku_error(
            kyoku, event, actor, "call pai does not match last raw dahai tile"
        )


def _mark_discard_called(
    kyoku: ReferenceKyoku,
    event: ReferenceMjaiEvent,
    target_state: _ActorState,
    discard_event_index: int,
    call_type: str,
    called_by_actor: int,
) -> None:
    matching = [
        index
        for index, discard in enumerate(target_state.discards)
        if discard.event_index == discard_event_index
    ]
    if len(matching) != 1:
        raise _kyoku_error(
            kyoku, event, called_by_actor, "call cannot resolve one target discard"
        )
    discard_index = matching[0]
    discard = target_state.discards[discard_index]
    if discard.was_called:
        raise _kyoku_error(kyoku, event, called_by_actor, "discard was already called")
    target_state.discards[discard_index] = replace(
        discard,
        was_called=True,
        call_type=call_type,
        called_by_actor=called_by_actor,
        call_event_index=_kyoku_event_index(event),
    )


def _apply_kakan(
    kyoku: ReferenceKyoku,
    event: ReferenceMjaiEvent,
    actor: int,
    state: _ActorState,
    consumed: tuple[str, ...],
    added_tile: str,
) -> None:
    consumed_counter = Counter(consumed)
    matching = [
        index
        for index, meld in enumerate(state.fixed_melds)
        if meld.meld_type == "pon" and meld.tiles == consumed_counter
    ]
    if len(matching) != 1:
        raise _kyoku_error(
            kyoku, event, actor, "kakan has no unique matching pon fixed meld"
        )
    pon_index = matching[0]
    pon = state.fixed_melds[pon_index]
    pon_kind = normalize_reference_tile(next(iter(pon.tiles)))
    if normalize_reference_tile(added_tile) != pon_kind:
        raise _kyoku_error(kyoku, event, actor, "kakan pai does not match pon")
    _remove_raw_tiles(kyoku, event, actor, state, (added_tile,))
    upgraded = pon.tiles.copy()
    upgraded[added_tile] += 1
    state.fixed_melds[pon_index] = _ReplayMeld(
        "kakan", upgraded, _kyoku_event_index(event)
    )
    _validate_owned_tiles(kyoku, event, actor, state)


def _counter_tiles(counter: Counter[str]) -> tuple[str, ...]:
    return tuple(
        tile
        for tile, count in sorted(
            counter.items(), key=lambda item: _raw_tile_key(item[0])
        )
        for _ in range(count)
    )


def _raw_tile_key(tile: str) -> tuple[int, bool]:
    return reference_tile_to_index(tile), tile in _RED_FIVES


def _event_type(event: ReferenceMjaiEvent) -> str:
    event_type = event.data.get("type")
    if not isinstance(event_type, str):
        raise TypeError(f"line {event.line_number}: event type must be a string")
    return event_type


def _kyoku_event_index(event: ReferenceMjaiEvent) -> int:
    if event.event_index is None:
        raise ValueError(f"line {event.line_number}: kyoku event has no event_index")
    return event.event_index


def _source_identifier(
    input_path: Path, source_root: str | PathLike[str] | None
) -> str:
    if source_root is None:
        return input_path.name.replace("\\", "/")
    try:
        relative = input_path.resolve().relative_to(Path(source_root).resolve())
    except ValueError as error:
        raise ValueError("MJAI path must be inside source_root") from error
    return relative.as_posix()


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            frozen[key] = _freeze_json(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or type(value) in {str, int, bool}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def _pending_value(value: Any) -> Any:
    if value is None:
        raise RuntimeError("pending reach declaration snapshot is incomplete")
    return value


def _source_error(source_path: str, line_number: int, message: str) -> ValueError:
    return ValueError(f"{source_path}: line {line_number}: {message}")


def _kyoku_error(
    kyoku: ReferenceKyoku,
    event: ReferenceMjaiEvent,
    actor: int | None,
    message: str,
) -> ValueError:
    actor_context = "" if actor is None else f", actor {actor}"
    return ValueError(
        f"{kyoku.source_path}: line {event.line_number}, "
        f"event {_kyoku_event_index(event)}{actor_context}: {message}"
    )
