from copy import deepcopy
from pathlib import Path

import pytest

from mahjong_analysis.mjai import (
    extract_rule_code,
    filter_east_kyokus,
    is_target_game,
    load_mjai,
    split_kyoku,
)


TEST_DATA_DIR = Path(__file__).parent / "data"
VALID_00A9_FILENAME = "2025010100gm-00a9-0000-1234abcd.mjson"
VALID_00E1_FILENAME = "2025010100gm-00e1-0000-abcdef12.mjson"


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


def artificial_kyoku(
    bakaze: str,
    kyoku: int,
    honba: int = 0,
) -> list[dict[str, object]]:
    return [
        {
            "type": "start_kyoku",
            "bakaze": bakaze,
            "kyoku": kyoku,
            "honba": honba,
        },
        {"type": "tsumo", "actor": 0, "pai": "1m"},
        {"type": "end_kyoku"},
    ]


def test_filter_east_kyokus_returns_east_kyoku() -> None:
    east = artificial_kyoku("E", 1)

    assert filter_east_kyokus([east]) == [east]


def test_filter_east_kyokus_excludes_south_kyoku() -> None:
    south = artificial_kyoku("S", 1)

    assert filter_east_kyokus([south]) == []


def test_filter_east_kyokus_excludes_west_kyoku() -> None:
    west = artificial_kyoku("W", 1)

    assert filter_east_kyokus([west]) == []


def test_filter_east_kyokus_includes_east_one_through_four() -> None:
    east_kyokus = [artificial_kyoku("E", number) for number in range(1, 5)]

    assert filter_east_kyokus(east_kyokus) == east_kyokus


def test_filter_east_kyokus_includes_east_renchan() -> None:
    renchan_kyokus = [artificial_kyoku("E", 2, honba) for honba in range(3)]

    assert filter_east_kyokus(renchan_kyokus) == renchan_kyokus


def test_filter_east_kyokus_preserves_kyoku_order() -> None:
    east_four = artificial_kyoku("E", 4)
    south_one = artificial_kyoku("S", 1)
    east_two = artificial_kyoku("E", 2)
    west_one = artificial_kyoku("W", 1)
    east_three = artificial_kyoku("E", 3)

    filtered = filter_east_kyokus(
        [east_four, south_one, east_two, west_one, east_three]
    )

    assert filtered == [east_four, east_two, east_three]


def test_filter_east_kyokus_does_not_change_kyoku_events() -> None:
    east = artificial_kyoku("E", 1)
    original_events = deepcopy(east)

    filtered = filter_east_kyokus([east])

    assert filtered[0] is east
    assert filtered[0] == original_events


def test_filter_east_kyokus_rejects_missing_bakaze() -> None:
    kyoku = [[{"type": "start_kyoku"}, {"type": "end_kyoku"}]]

    with pytest.raises(KeyError, match="bakaze"):
        filter_east_kyokus(kyoku)


def test_extract_rule_code_returns_00a9() -> None:
    assert extract_rule_code(VALID_00A9_FILENAME) == "00a9"


def test_extract_rule_code_returns_00e1() -> None:
    assert extract_rule_code(VALID_00E1_FILENAME) == "00e1"


def test_extract_rule_code_accepts_path_object() -> None:
    path = Path("logs") / VALID_00A9_FILENAME

    assert extract_rule_code(path) == "00a9"


def test_extract_rule_code_rejects_invalid_filename() -> None:
    with pytest.raises(ValueError, match="filename"):
        extract_rule_code("invalid.mjson")


def test_extract_rule_code_rejects_wrong_rule_code_length() -> None:
    filename = "2025010100gm-0a9-0000-1234abcd.mjson"

    with pytest.raises(ValueError, match="filename"):
        extract_rule_code(filename)


def test_extract_rule_code_rejects_extra_filename_component() -> None:
    filename = "2025010100gm-00a9-extra-0000-1234abcd.mjson"

    with pytest.raises(ValueError, match="filename"):
        extract_rule_code(filename)


def test_is_target_game_accepts_00a9_with_red_fives() -> None:
    events = [{"type": "start_game", "aka_flag": True}]

    assert is_target_game(VALID_00A9_FILENAME, events) is True


def test_is_target_game_rejects_00e1() -> None:
    events = [{"type": "start_game", "aka_flag": True}]

    assert is_target_game(VALID_00E1_FILENAME, events) is False


def test_is_target_game_rejects_game_without_red_fives() -> None:
    events = [{"type": "start_game", "aka_flag": False}]

    assert is_target_game(VALID_00A9_FILENAME, events) is False


def test_is_target_game_rejects_empty_events() -> None:
    with pytest.raises(ValueError, match="empty"):
        is_target_game(VALID_00A9_FILENAME, [])


def test_is_target_game_rejects_first_event_other_than_start_game() -> None:
    events = [{"type": "start_kyoku", "aka_flag": True}]

    with pytest.raises(ValueError, match="start_game"):
        is_target_game(VALID_00A9_FILENAME, events)


def test_is_target_game_rejects_missing_aka_flag() -> None:
    events = [{"type": "start_game"}]

    with pytest.raises(ValueError, match="aka_flag"):
        is_target_game(VALID_00A9_FILENAME, events)


def test_is_target_game_rejects_string_aka_flag() -> None:
    events = [{"type": "start_game", "aka_flag": "true"}]

    with pytest.raises(ValueError, match="bool"):
        is_target_game(VALID_00A9_FILENAME, events)


def test_is_target_game_rejects_integer_aka_flag() -> None:
    events = [{"type": "start_game", "aka_flag": 1}]

    with pytest.raises(ValueError, match="bool"):
        is_target_game(VALID_00A9_FILENAME, events)


def test_is_target_game_validates_aka_flag_for_non_target_rule() -> None:
    events = [{"type": "start_game", "aka_flag": "true"}]

    with pytest.raises(ValueError, match="bool"):
        is_target_game(VALID_00E1_FILENAME, events)
