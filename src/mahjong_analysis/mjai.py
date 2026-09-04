"""MJAI log file loading utilities."""

import json
from os import PathLike
from typing import Any


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
