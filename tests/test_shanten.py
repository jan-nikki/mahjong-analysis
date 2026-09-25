from __future__ import annotations

import random

import pytest

from mahjong_analysis.hand_waits import calculate_hand_waits
from mahjong_analysis.shanten import initial_shanten
from mahjong_analysis.tiles import TILE_KINDS


@pytest.mark.parametrize(
    ("hand", "expected"),
    [
        (
            [
                "1m", "2m", "3m", "4m", "5m", "6m", "7p", "8p", "9p",
                "5s", "5s", "2s", "3s",
            ],
            0,
        ),
        (
            [
                "1m", "2m", "3m", "4m", "5m", "6m", "7p", "8p", "9p",
                "E", "E", "1s", "9s",
            ],
            1,
        ),
        (
            [
                "1m", "2m", "3m", "4m", "5m", "6m", "7p", "8p", "9p",
                "E", "S", "W", "N",
            ],
            2,
        ),
        (
            [
                "1m", "1m", "2m", "2m", "3p", "3p", "4p", "4p",
                "5s", "5s", "6s", "6s", "E",
            ],
            0,
        ),
        (
            [
                "1m", "9m", "1p", "9p", "1s", "9s", "E", "S",
                "W", "N", "P", "F", "C",
            ],
            0,
        ),
    ],
)
def test_known_thirteen_tile_shanten(hand: list[str], expected: int) -> None:
    assert initial_shanten(hand) == expected


def test_shanten_zero_agrees_with_independent_wait_detector() -> None:
    deck = [tile for tile in TILE_KINDS for _ in range(4)]
    rng = random.Random(20260924)
    for _ in range(100):
        hand = rng.sample(deck, 13)
        assert (initial_shanten(hand) == 0) == bool(
            calculate_hand_waits(hand).wait_tiles
        )
    tenpai = [
        "1m", "2m", "3m", "4m", "5m", "6m", "7p", "8p", "9p",
        "5s", "5s", "2s", "3s",
    ]
    for _ in range(100):
        hand = tenpai.copy()
        hand[rng.randrange(13)] = rng.choice(TILE_KINDS)
        assert (initial_shanten(hand) == 0) == bool(
            calculate_hand_waits(hand).wait_tiles
        )


def test_initial_shanten_rejects_invalid_hands() -> None:
    with pytest.raises(ValueError, match="exactly 13"):
        initial_shanten(["1m"] * 12)
    with pytest.raises(ValueError, match="more than four"):
        initial_shanten(["1m"] * 5 + ["2m"] * 4 + ["3m"] * 4)
