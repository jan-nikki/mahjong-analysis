import gzip
import json
from copy import deepcopy
from pathlib import Path

import pytest

from mahjong_analysis.mjai import (
    classify_dealer_double_riichi_result,
    extract_rule_code,
    filter_east_kyokus,
    is_dealer_double_riichi,
    is_target_game,
    load_mjai,
    split_kyoku,
)


TEST_DATA_DIR = Path(__file__).parent / "data"
VALID_00A9_FILENAME = "2025010100gm-00a9-0000-1234abcd.mjson"
VALID_00E1_FILENAME = "2025010100gm-00e1-0000-abcdef12.mjson"


def write_json_lines(
    path: Path,
    events: list[dict[str, object]],
    *,
    compressed: bool,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(event, ensure_ascii=False) + "\n" for event in events
    )
    if compressed:
        with gzip.open(path, mode="wt", encoding="utf-8") as file:
            file.write(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


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


def test_load_mjai_reads_plain_mjson(tmp_path: Path) -> None:
    expected = [
        {"type": "start_game", "names": ["東家", "南家", "西家", "北家"]},
        {"type": "end_game"},
    ]
    path = write_json_lines(
        tmp_path / "plain" / "sample.mjson",
        expected,
        compressed=False,
    )

    assert load_mjai(path) == expected


def test_load_mjai_reads_gzip_compressed_mjson(tmp_path: Path) -> None:
    expected = [{"type": "start_game"}, {"type": "end_game"}]
    path = write_json_lines(
        tmp_path / "gzip" / "sample.mjson",
        expected,
        compressed=True,
    )

    assert path.suffix == ".mjson"
    assert path.read_bytes()[:2] == b"\x1f\x8b"
    assert load_mjai(path) == expected


def test_load_mjai_uses_magic_not_year_or_extension(tmp_path: Path) -> None:
    expected = [
        {"type": "start_game", "names": ["テスト東家"]},
        {"type": "end_game"},
    ]
    plain_path = write_json_lines(
        tmp_path / "2009" / "same.mjson",
        expected,
        compressed=False,
    )
    gzip_path = write_json_lines(
        tmp_path / "2025" / "same.mjson",
        expected,
        compressed=True,
    )

    assert plain_path.read_bytes()[:2] != b"\x1f\x8b"
    assert gzip_path.read_bytes()[:2] == b"\x1f\x8b"
    assert load_mjai(plain_path) == load_mjai(gzip_path) == expected


def test_load_mjai_rejects_invalid_plain_json(tmp_path: Path) -> None:
    path = tmp_path / "invalid.mjson"
    path.write_text('{"type":"start_game"}\n{invalid}\n', encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_mjai(path)


def test_load_mjai_rejects_invalid_json_inside_gzip(tmp_path: Path) -> None:
    path = tmp_path / "invalid.mjson"
    with gzip.open(path, mode="wt", encoding="utf-8") as file:
        file.write('{"type":"start_game"}\n{invalid}\n')

    with pytest.raises(json.JSONDecodeError):
        load_mjai(path)


def test_load_mjai_rejects_broken_gzip_with_magic(tmp_path: Path) -> None:
    path = tmp_path / "broken.mjson"
    path.write_bytes(b"\x1f\x8b\x00broken gzip")

    with pytest.raises((gzip.BadGzipFile, EOFError)):
        load_mjai(path)


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


def dealer_double_riichi_kyoku(
    *,
    tsumogiri: bool = False,
    bakaze: str = "E",
) -> list[dict[str, object]]:
    return [
        {"type": "start_kyoku", "bakaze": bakaze, "oya": 0},
        {"type": "tsumo", "actor": 0, "pai": "1m"},
        {"type": "reach", "actor": 0},
        {
            "type": "dahai",
            "actor": 0,
            "pai": "1m" if tsumogiri else "2m",
            "tsumogiri": tsumogiri,
        },
        {"type": "reach_accepted", "actor": 0},
        {"type": "end_kyoku"},
    ]


def dealer_double_riichi_result_kyoku(
    *result_events: dict[str, object],
) -> list[dict[str, object]]:
    kyoku = dealer_double_riichi_kyoku()
    return [*kyoku[:-1], *result_events, kyoku[-1]]


def test_is_dealer_double_riichi_accepts_first_tedashi() -> None:
    assert is_dealer_double_riichi(dealer_double_riichi_kyoku()) is True


def test_is_dealer_double_riichi_accepts_first_tsumogiri() -> None:
    kyoku = dealer_double_riichi_kyoku(tsumogiri=True)

    assert is_dealer_double_riichi(kyoku) is True


def test_is_dealer_double_riichi_rejects_dealer_later_reach() -> None:
    kyoku = [
        {"type": "start_kyoku", "oya": 0},
        {"type": "tsumo", "actor": 0, "pai": "1m"},
        {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "2m"},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "3m", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 0},
        {"type": "end_kyoku"},
    ]

    assert is_dealer_double_riichi(kyoku) is False


def test_is_dealer_double_riichi_rejects_child_first_reach() -> None:
    kyoku = [
        {"type": "start_kyoku", "oya": 0},
        {"type": "tsumo", "actor": 0, "pai": "1m"},
        {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "2m"},
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "3m", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 1},
        {"type": "end_kyoku"},
    ]

    assert is_dealer_double_riichi(kyoku) is False


def test_is_dealer_double_riichi_rejects_ankan_before_reach() -> None:
    kyoku = [
        {"type": "start_kyoku", "oya": 0},
        {"type": "tsumo", "actor": 0, "pai": "1m"},
        {"type": "ankan", "actor": 0, "consumed": ["9m"] * 4},
        {"type": "dora", "dora_marker": "1s"},
        {"type": "tsumo", "actor": 0, "pai": "2m"},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "2m", "tsumogiri": True},
        {"type": "reach_accepted", "actor": 0},
        {"type": "end_kyoku"},
    ]

    assert is_dealer_double_riichi(kyoku) is False


def test_is_dealer_double_riichi_rejects_unaccepted_reach_with_hora() -> None:
    kyoku = [
        {"type": "start_kyoku", "oya": 0},
        {"type": "tsumo", "actor": 0, "pai": "1m"},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "2m", "tsumogiri": False},
        {"type": "hora", "actor": 1, "target": 0},
        {"type": "end_kyoku"},
    ]

    assert is_dealer_double_riichi(kyoku) is False


def test_is_dealer_double_riichi_rejects_kyoku_without_dealer_reach() -> None:
    kyoku = [
        {"type": "start_kyoku", "oya": 0},
        {"type": "tsumo", "actor": 0, "pai": "1m"},
        {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": True},
        {"type": "end_kyoku"},
    ]

    assert is_dealer_double_riichi(kyoku) is False


def test_is_dealer_double_riichi_ignores_chi_after_acceptance() -> None:
    kyoku = dealer_double_riichi_kyoku()
    kyoku.insert(
        -1,
        {"type": "chi", "actor": 2, "target": 1, "pai": "3m"},
    )

    assert is_dealer_double_riichi(kyoku) is True


def test_is_dealer_double_riichi_ignores_pon_after_acceptance() -> None:
    kyoku = dealer_double_riichi_kyoku()
    kyoku.insert(
        -1,
        {"type": "pon", "actor": 2, "target": 1, "pai": "3m"},
    )

    assert is_dealer_double_riichi(kyoku) is True


def test_is_dealer_double_riichi_rejects_event_between_reach_and_dahai() -> None:
    kyoku = [
        {"type": "start_kyoku", "oya": 0},
        {"type": "reach", "actor": 0},
        {"type": "tsumo", "actor": 1, "pai": "1m"},
        {"type": "end_kyoku"},
    ]

    with pytest.raises(ValueError, match="followed by dahai"):
        is_dealer_double_riichi(kyoku)


def test_is_dealer_double_riichi_rejects_dahai_actor_mismatch() -> None:
    kyoku = [
        {"type": "start_kyoku", "oya": 0},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 1, "pai": "1m", "tsumogiri": True},
        {"type": "reach_accepted", "actor": 0},
        {"type": "end_kyoku"},
    ]

    with pytest.raises(ValueError, match="dahai actors"):
        is_dealer_double_riichi(kyoku)


def test_is_dealer_double_riichi_rejects_accepted_actor_mismatch() -> None:
    kyoku = [
        {"type": "start_kyoku", "oya": 0},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": True},
        {"type": "reach_accepted", "actor": 1},
        {"type": "end_kyoku"},
    ]

    with pytest.raises(ValueError, match="reach_accepted actors"):
        is_dealer_double_riichi(kyoku)


def test_is_dealer_double_riichi_returns_false_for_ryukyoku_after_later_reach() -> None:
    kyoku = [
        {"type": "start_kyoku", "oya": 0},
        {"type": "tsumo", "actor": 0, "pai": "1m"},
        {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "2m"},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "3m", "tsumogiri": False},
        {"type": "ryukyoku", "deltas": [0, 0, 0, 0]},
        {"type": "end_kyoku"},
    ]

    assert is_dealer_double_riichi(kyoku) is False


def test_is_dealer_double_riichi_does_not_filter_by_bakaze() -> None:
    kyoku = dealer_double_riichi_kyoku(bakaze="S")

    assert is_dealer_double_riichi(kyoku) is True


def test_is_dealer_double_riichi_rejects_tsumo_after_reach_dahai() -> None:
    kyoku = [
        {"type": "start_kyoku", "oya": 0},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "2m"},
        {"type": "end_kyoku"},
    ]

    with pytest.raises(ValueError, match="reach_accepted, hora, or ryukyoku"):
        is_dealer_double_riichi(kyoku)


def test_is_dealer_double_riichi_rejects_end_after_reach_dahai() -> None:
    kyoku = [
        {"type": "start_kyoku", "oya": 0},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": True},
    ]

    with pytest.raises(ValueError, match="followed by an event"):
        is_dealer_double_riichi(kyoku)


def test_classify_dealer_double_riichi_result_returns_dealer_win_for_tsumo(
) -> None:
    kyoku = dealer_double_riichi_result_kyoku(
        {
            "type": "hora",
            "actor": 0,
            "target": 0,
            "deltas": [12000, -4000, -4000, -4000],
        }
    )

    assert classify_dealer_double_riichi_result(kyoku) == "dealer_win"


def test_classify_dealer_double_riichi_result_returns_dealer_win_for_ron(
) -> None:
    kyoku = dealer_double_riichi_result_kyoku(
        {"type": "hora", "actor": 0, "target": 2}
    )

    assert classify_dealer_double_riichi_result(kyoku) == "dealer_win"


def test_classify_dealer_double_riichi_result_returns_other_win_for_tsumo(
) -> None:
    kyoku = dealer_double_riichi_result_kyoku(
        {"type": "hora", "actor": 1, "target": 1}
    )

    assert classify_dealer_double_riichi_result(kyoku) == "other_win"


def test_classify_dealer_double_riichi_result_returns_other_win_for_dealer_deal_in(
) -> None:
    kyoku = dealer_double_riichi_result_kyoku(
        {"type": "hora", "actor": 1, "target": 0}
    )

    assert classify_dealer_double_riichi_result(kyoku) == "other_win"


def test_classify_dealer_double_riichi_result_returns_other_win_for_other_deal_in(
) -> None:
    kyoku = dealer_double_riichi_result_kyoku(
        {
            "type": "hora",
            "actor": 1,
            "target": 2,
            "deltas": [0, 3900, -3900, 0],
        }
    )

    assert classify_dealer_double_riichi_result(kyoku) == "other_win"


def test_classify_dealer_double_riichi_result_returns_draw_for_ryukyoku() -> None:
    kyoku = dealer_double_riichi_result_kyoku(
        {"type": "ryukyoku", "deltas": [0, 0, 0, 0]}
    )

    assert classify_dealer_double_riichi_result(kyoku) == "draw"


def test_classify_dealer_double_riichi_result_returns_other_win_for_multiple_hora(
) -> None:
    kyoku = dealer_double_riichi_result_kyoku(
        {"type": "hora", "actor": 1, "target": 3},
        {"type": "hora", "actor": 2, "target": 3},
    )

    assert classify_dealer_double_riichi_result(kyoku) == "other_win"


def test_classify_dealer_double_riichi_result_checks_every_hora_for_dealer(
) -> None:
    kyoku = dealer_double_riichi_result_kyoku(
        {"type": "hora", "actor": 1, "target": 2},
        {"type": "hora", "actor": 0, "target": 2},
    )

    assert classify_dealer_double_riichi_result(kyoku) == "dealer_win"


def test_classify_dealer_double_riichi_result_rejects_missing_result() -> None:
    with pytest.raises(ValueError, match="no hora or ryukyoku"):
        classify_dealer_double_riichi_result(dealer_double_riichi_kyoku())


def test_classify_dealer_double_riichi_result_rejects_hora_and_ryukyoku() -> None:
    kyoku = dealer_double_riichi_result_kyoku(
        {"type": "hora", "actor": 0},
        {"type": "ryukyoku"},
    )

    with pytest.raises(ValueError, match="cannot coexist"):
        classify_dealer_double_riichi_result(kyoku)


def test_classify_dealer_double_riichi_result_rejects_multiple_ryukyoku() -> None:
    kyoku = dealer_double_riichi_result_kyoku(
        {"type": "ryukyoku"},
        {"type": "ryukyoku"},
    )

    with pytest.raises(ValueError, match="multiple ryukyoku"):
        classify_dealer_double_riichi_result(kyoku)


def test_classify_dealer_double_riichi_result_rejects_event_after_hora() -> None:
    kyoku = dealer_double_riichi_result_kyoku(
        {"type": "hora", "actor": 0},
        {"type": "tsumo", "actor": 1, "pai": "3m"},
    )

    with pytest.raises(ValueError, match="contiguous"):
        classify_dealer_double_riichi_result(kyoku)


def test_classify_dealer_double_riichi_result_rejects_event_after_ryukyoku() -> None:
    kyoku = dealer_double_riichi_result_kyoku(
        {"type": "ryukyoku"},
        {"type": "tsumo", "actor": 1, "pai": "3m"},
    )

    with pytest.raises(ValueError, match="contiguous"):
        classify_dealer_double_riichi_result(kyoku)


def test_classify_dealer_double_riichi_result_rejects_missing_end_kyoku() -> None:
    kyoku = dealer_double_riichi_result_kyoku(
        {"type": "hora", "actor": 0}
    )
    kyoku.pop()

    with pytest.raises(ValueError, match="end_kyoku"):
        classify_dealer_double_riichi_result(kyoku)


def test_classify_dealer_double_riichi_result_accepts_contiguous_multiple_hora(
) -> None:
    kyoku = dealer_double_riichi_result_kyoku(
        {"type": "hora", "actor": 2, "target": 3},
        {"type": "hora", "actor": 1, "target": 3},
    )

    assert [event["type"] for event in kyoku[-3:]] == [
        "hora",
        "hora",
        "end_kyoku",
    ]
    assert classify_dealer_double_riichi_result(kyoku) == "other_win"
