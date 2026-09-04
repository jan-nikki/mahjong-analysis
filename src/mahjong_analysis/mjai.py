"""MJAI log file loading utilities."""

import json
import re
from os import PathLike
from pathlib import Path
from typing import Any


_MJAI_FILENAME_PATTERN = re.compile(
    r"^\d{10}gm-"
    r"(?P<rule_code>[0-9a-f]{4})-"
    r"(?P<room_code>[0-9a-f]{4})-"
    r"(?P<game_id>[0-9a-f]{8})\.mjson$"
)


def load_mjai(path: str | PathLike[str]) -> list[dict[str, Any]]:
    """Load events from an uncompressed UTF-8 MJAI JSON Lines file."""
    events: list[dict[str, Any]] = []

    with open(path, encoding="utf-8") as file:
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
            if index + 1 >= len(kyoku) or kyoku[index + 1]["type"] != "dahai":
                raise ValueError("dealer reach must be followed by dahai")

            dahai = kyoku[index + 1]
            if dahai.get("actor") != dealer:
                raise ValueError("dealer reach and dahai actors do not match")

            if index + 2 >= len(kyoku):
                raise ValueError("dealer reach dahai must be followed by an event")

            event_after_dahai = kyoku[index + 2]
            if event_after_dahai["type"] == "reach_accepted":
                reach_accepted = event_after_dahai
                if reach_accepted.get("actor") != dealer:
                    raise ValueError(
                        "dealer reach and reach_accepted actors do not match"
                    )
                return not dealer_has_discarded and not ankan_occurred

            if event_after_dahai["type"] in {"hora", "ryukyoku"}:
                return False

            raise ValueError(
                "dealer reach dahai must be followed by "
                "reach_accepted, hora, or ryukyoku"
            )

    return False


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
