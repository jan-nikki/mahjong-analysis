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
