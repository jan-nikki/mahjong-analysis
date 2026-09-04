from pathlib import Path

import pytest

from mahjong_analysis.mjai import load_mjai, split_kyoku


TEST_DATA_DIR = Path(__file__).parent / "data"


def test_load_mjai_preserves_event_order() -> None:
    events = load_mjai(TEST_DATA_DIR / "sample.mjson")

    assert [event["type"] for event in events] == [
        "start_game",
        "start_kyoku",
        "tsumo",
        "dahai",
    ]
    assert events[0]["names"] == [
        "テスト東家",
        "テスト南家",
        "テスト西家",
        "テスト北家",
    ]
    assert events[2] == {"type": "tsumo", "actor": 0, "pai": "1m"}


def artificial_game_events() -> list[dict[str, object]]:
    return [
        {"type": "start_game"},
        {"type": "start_kyoku", "kyoku": 1},
        {"type": "tsumo", "actor": 0, "pai": "1m"},
        {"type": "hora", "actor": 0},
        {"type": "end_kyoku"},
        {"type": "start_kyoku", "kyoku": 2},
        {"type": "tsumo", "actor": 1, "pai": "2m"},
        {"type": "ryukyoku"},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]


def test_split_kyoku_splits_multiple_kyokus() -> None:
    events = artificial_game_events()

    kyokus = split_kyoku(events)

    assert kyokus == [events[1:5], events[5:9]]


def test_split_kyoku_starts_each_kyoku_with_start_kyoku() -> None:
    kyokus = split_kyoku(artificial_game_events())

    assert all(kyoku[0]["type"] == "start_kyoku" for kyoku in kyokus)


def test_split_kyoku_ends_each_kyoku_with_end_kyoku() -> None:
    kyokus = split_kyoku(artificial_game_events())

    assert all(kyoku[-1]["type"] == "end_kyoku" for kyoku in kyokus)


def test_split_kyoku_preserves_event_order() -> None:
    kyokus = split_kyoku(artificial_game_events())

    assert [event["type"] for event in kyokus[0]] == [
        "start_kyoku",
        "tsumo",
        "hora",
        "end_kyoku",
    ]


def test_split_kyoku_excludes_game_boundary_events() -> None:
    kyokus = split_kyoku(artificial_game_events())

    event_types = [event["type"] for kyoku in kyokus for event in kyoku]
    assert "start_game" not in event_types
    assert "end_game" not in event_types


def test_split_kyoku_keeps_ryukyoku_until_end_kyoku() -> None:
    events = [
        {"type": "start_kyoku"},
        {"type": "ryukyoku"},
        {"type": "end_kyoku"},
    ]

    kyokus = split_kyoku(events)

    assert [event["type"] for event in kyokus[0]] == [
        "start_kyoku",
        "ryukyoku",
        "end_kyoku",
    ]


def test_split_kyoku_keeps_multiple_hora_until_end_kyoku() -> None:
    events = [
        {"type": "start_kyoku"},
        {"type": "hora", "actor": 1},
        {"type": "hora", "actor": 2},
        {"type": "end_kyoku"},
    ]

    kyokus = split_kyoku(events)

    assert [event["type"] for event in kyokus[0]] == [
        "start_kyoku",
        "hora",
        "hora",
        "end_kyoku",
    ]


def test_split_kyoku_rejects_start_kyoku_inside_kyoku() -> None:
    events = [
        {"type": "start_kyoku"},
        {"type": "start_kyoku"},
        {"type": "end_kyoku"},
    ]

    with pytest.raises(ValueError, match="start_kyoku"):
        split_kyoku(events)


def test_split_kyoku_rejects_end_kyoku_without_start_kyoku() -> None:
    with pytest.raises(ValueError, match="end_kyoku"):
        split_kyoku([{"type": "end_kyoku"}])


def test_split_kyoku_rejects_unclosed_kyoku() -> None:
    events = [
        {"type": "start_kyoku"},
        {"type": "tsumo", "actor": 0, "pai": "1m"},
    ]

    with pytest.raises(ValueError, match="end_kyoku"):
        split_kyoku(events)


@pytest.mark.parametrize("event_type", ["start_game", "end_game"])
def test_split_kyoku_rejects_game_boundary_inside_kyoku(
    event_type: str,
) -> None:
    events = [
        {"type": "start_kyoku"},
        {"type": event_type},
        {"type": "end_kyoku"},
    ]

    with pytest.raises(ValueError, match=event_type):
        split_kyoku(events)


def test_split_kyoku_rejects_non_game_event_outside_kyoku() -> None:
    with pytest.raises(ValueError, match="tsumo"):
        split_kyoku([{"type": "tsumo", "actor": 0, "pai": "1m"}])
