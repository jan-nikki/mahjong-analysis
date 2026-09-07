"""MJAI log file loading utilities."""

import gzip
import json
import re
from os import PathLike
from pathlib import Path
from typing import Any, Literal

from mahjong_analysis.tiles import normalize_tile

_MJAI_FILENAME_PATTERN = re.compile(
    r"^\d{10}gm-"
    r"(?P<rule_code>[0-9a-f]{4})-"
    r"(?P<room_code>[0-9a-f]{4})-"
    r"(?P<game_id>[0-9a-f]{8})\.mjson$"
)
_GZIP_MAGIC = b"\x1f\x8b"


def load_mjai(path: str | PathLike[str]) -> list[dict[str, Any]]:
    """Load events from a plain or gzip-compressed UTF-8 JSON Lines file."""
    events: list[dict[str, Any]] = []

    with open(path, "rb") as file:
        is_gzip = file.read(2) == _GZIP_MAGIC

    if is_gzip:
        text_file = gzip.open(path, mode="rt", encoding="utf-8")
    else:
        text_file = open(path, encoding="utf-8")

    with text_file as file:
        for line in file:
            events.append(json.loads(line))

    return events


def split_kyoku(
    events: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Split MJAI events into inclusive start_kyoku/end_kyoku groups."""
    kyokus: list[list[dict[str, Any]]] = []
    current_kyoku: list[dict[str, Any]] | None = None

    for event in events:
        event_type = event["type"]

        if event_type == "start_kyoku":
            if current_kyoku is not None:
                raise ValueError(
                    "start_kyoku encountered before the current kyoku ended"
                )
            current_kyoku = [event]
        elif event_type == "end_kyoku":
            if current_kyoku is None:
                raise ValueError("end_kyoku encountered without start_kyoku")
            current_kyoku.append(event)
            kyokus.append(current_kyoku)
            current_kyoku = None
        elif event_type in {"start_game", "end_game"}:
            if current_kyoku is not None:
                raise ValueError(f"{event_type} encountered inside a kyoku")
        elif current_kyoku is None:
            raise ValueError(f"{event_type} encountered outside a kyoku")
        elif current_kyoku is not None:
            current_kyoku.append(event)

    if current_kyoku is not None:
        raise ValueError("input ended before the current kyoku reached end_kyoku")

    return kyokus


def filter_east_kyokus(
    kyokus: list[list[dict[str, Any]]],
) -> list[list[dict[str, Any]]]:
    """Return east-round kyokus in their original order."""
    return [kyoku for kyoku in kyokus if kyoku[0]["bakaze"] == "E"]


def match_reach_sequence(
    kyoku: list[dict[str, Any]],
    reach_event_index: int,
    *,
    context: str = "reach",
) -> tuple[int, int, int, int] | None:
    """Validate a reach/dahai/final-event sequence.

    Return ``(actor, reach index, dahai index, accepted index)`` for an
    established reach. A declaration followed by hora or ryukyoku is a valid
    unestablished reach and returns ``None``.
    """
    if not 0 <= reach_event_index < len(kyoku):
        raise ValueError(f"{context} event index is out of range")

    reach = _require_sequence_event(kyoku, reach_event_index, context)
    if reach["type"] != "reach":
        raise ValueError(
            f"event {reach_event_index}: {context} event index does not point to reach"
        )
    actor = _require_sequence_actor(reach, reach_event_index, context)

    dahai_index = reach_event_index + 1
    if dahai_index >= len(kyoku):
        raise ValueError(
            f"event {reach_event_index}, actor {actor}: "
            f"{context} must be followed by dahai"
        )

    dahai = _require_sequence_event(kyoku, dahai_index, context, actor)
    if dahai["type"] != "dahai":
        raise ValueError(
            f"event {dahai_index}, actor {actor}: {context} must be followed by dahai"
        )
    dahai_actor = _require_sequence_actor(dahai, dahai_index, context)
    if dahai_actor != actor:
        raise ValueError(
            f"event {dahai_index}, actor {dahai_actor}: "
            f"{context} and dahai actors do not match"
        )

    final_index = reach_event_index + 2
    if final_index >= len(kyoku):
        raise ValueError(
            f"event {dahai_index}, actor {actor}: "
            f"{context} dahai must be followed by an event"
        )

    final_event = _require_sequence_event(kyoku, final_index, context, actor)
    if final_event["type"] == "reach_accepted":
        accepted_actor = _require_sequence_actor(final_event, final_index, context)
        if accepted_actor != actor:
            raise ValueError(
                f"event {final_index}, actor {accepted_actor}: "
                f"{context} and reach_accepted actors do not match"
            )
        return actor, reach_event_index, dahai_index, final_index

    if final_event["type"] == "hora":
        _validate_declaration_tile_hora(
            dahai,
            final_event,
            final_index,
            actor,
            context,
        )
        return None

    if final_event["type"] == "ryukyoku":
        return None

    raise ValueError(
        f"event {final_index}, actor {actor}: {context} dahai must be followed by "
        "reach_accepted, hora, or ryukyoku"
    )


def _require_sequence_event(
    kyoku: list[dict[str, Any]],
    event_index: int,
    context: str,
    actor: int | None = None,
) -> dict[str, Any]:
    event = kyoku[event_index]
    actor_context = "" if actor is None else f", actor {actor}"
    if not isinstance(event, dict):
        raise ValueError(  # noqa: TRY004 - malformed MJAI is a data-value error
            f"event {event_index}{actor_context}: "
            f"{context} sequence event must be an object"
        )
    if not isinstance(event.get("type"), str):
        raise ValueError(  # noqa: TRY004 - malformed MJAI is a data-value error
            f"event {event_index}{actor_context}: "
            f"{context} sequence event type must be a string"
        )
    return event


def _require_sequence_actor(
    event: dict[str, Any],
    event_index: int,
    context: str,
) -> int:
    actor = event.get("actor")
    if type(actor) is not int or not 0 <= actor <= 3:
        raise ValueError(
            f"event {event_index}: {context} sequence actor must be "
            "an integer from 0 to 3"
        )
    return actor


def _validate_declaration_tile_hora(
    dahai: dict[str, Any],
    hora: dict[str, Any],
    hora_event_index: int,
    reach_actor: int,
    context: str,
) -> None:
    target = hora.get("target")
    if type(target) is not int or target != reach_actor:
        raise ValueError(
            f"event {hora_event_index}, actor {reach_actor}: "
            f"{context} declaration hora target must be the reach actor"
        )

    dahai_tile = dahai.get("pai")
    hora_tile = hora.get("pai")
    if not isinstance(dahai_tile, str) or not isinstance(hora_tile, str):
        raise ValueError(  # noqa: TRY004 - malformed MJAI is a data-value error
            f"event {hora_event_index}, actor {reach_actor}: "
            f"{context} declaration dahai and hora pai must be tile strings"
        )
    try:
        same_tile_kind = normalize_tile(dahai_tile) == normalize_tile(hora_tile)
    except ValueError as error:
        raise ValueError(
            f"event {hora_event_index}, actor {reach_actor}: "
            f"{context} declaration dahai or hora has an invalid pai"
        ) from error
    if not same_tile_kind:
        raise ValueError(
            f"event {hora_event_index}, actor {reach_actor}: "
            f"{context} declaration hora pai does not match the declaration tile kind"
        )


def is_dealer_double_riichi(kyoku: list[dict[str, Any]]) -> bool:
    """Return whether the dealer completed riichi on their first discard."""
    dealer = kyoku[0]["oya"]
    dealer_has_discarded = False
    ankan_occurred = False

    for index, event in enumerate(kyoku):
        event_type = event["type"]

        if event_type == "ankan":
            ankan_occurred = True
        elif event_type == "dahai" and event["actor"] == dealer:
            dealer_has_discarded = True
        elif event_type == "reach" and event["actor"] == dealer:
            sequence = match_reach_sequence(
                kyoku,
                index,
                context="dealer reach",
            )
            if sequence is None:
                return False
            return not dealer_has_discarded and not ankan_occurred

    return False


def classify_dealer_double_riichi_result(
    kyoku: list[dict[str, Any]],
) -> Literal["dealer_win", "other_win", "draw"]:
    """Classify the result of an established dealer double-riichi kyoku."""
    if not kyoku or kyoku[-1]["type"] != "end_kyoku":
        raise ValueError("kyoku must end with end_kyoku")

    dealer = kyoku[0]["oya"]
    result_events: list[dict[str, Any]] = []
    result_started = False

    for event in kyoku[:-1]:
        if event["type"] in {"hora", "ryukyoku"}:
            result_events.append(event)
            result_started = True
        elif result_started:
            raise ValueError(
                "result events must be contiguous immediately before end_kyoku"
            )

    horas = [event for event in result_events if event["type"] == "hora"]
    ryukyokus = [
        event for event in result_events if event["type"] == "ryukyoku"
    ]

    if not result_events:
        raise ValueError("kyoku has no hora or ryukyoku result")
    if horas and ryukyokus:
        raise ValueError("hora and ryukyoku cannot coexist in one kyoku")
    if len(ryukyokus) > 1:
        raise ValueError("kyoku cannot contain multiple ryukyoku events")
    if ryukyokus:
        return "draw"
    if any(hora["actor"] == dealer for hora in horas):
        return "dealer_win"
    return "other_win"


def extract_rule_code(path: str | PathLike[str]) -> str:
    """Extract the rule code from a valid MJAI filename."""
    filename = Path(path).name
    match = _MJAI_FILENAME_PATTERN.fullmatch(filename)

    if match is None:
        raise ValueError(f"invalid MJAI filename: {filename}")

    return match.group("rule_code")


def is_target_game(
    path: str | PathLike[str],
    events: list[dict[str, Any]],
) -> bool:
    """Return whether an MJAI log is a target four-player red-five game."""
    rule_code = extract_rule_code(path)

    if not events:
        raise ValueError("event list is empty")

    start_game = events[0]
    if start_game.get("type") != "start_game":
        raise ValueError("first event must be start_game")
    if "aka_flag" not in start_game:
        raise ValueError("start_game is missing aka_flag")

    aka_flag = start_game["aka_flag"]
    if type(aka_flag) is not bool:
        raise ValueError("start_game aka_flag must be bool")

    return rule_code == "00a9" and aka_flag is True
