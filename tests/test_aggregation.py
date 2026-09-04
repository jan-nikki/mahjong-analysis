import json
from json import JSONDecodeError
from pathlib import Path

import pytest

from mahjong_analysis.aggregation import aggregate_dealer_double_riichi


def write_mjai(path: Path, kyokus: list[list[dict[str, object]]]) -> Path:
    events = [
        {"type": "start_game", "aka_flag": True},
        *(event for kyoku in kyokus for event in kyoku),
        {"type": "end_game"},
    ]
    path.write_text(
        "".join(f"{json.dumps(event)}\n" for event in events),
        encoding="utf-8",
    )
    return path


def dealer_double_riichi_kyoku(
    result: str,
    *,
    bakaze: str = "E",
    kyoku: int = 1,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = [
        {
            "type": "start_kyoku",
            "bakaze": bakaze,
            "kyoku": kyoku,
            "honba": 0,
            "oya": 0,
        },
        {"type": "tsumo", "actor": 0, "pai": "1m"},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "2m", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 0},
    ]

    if result == "dealer_win":
        events.append({"type": "hora", "actor": 0, "target": 0})
    elif result == "other_win":
        events.append({"type": "hora", "actor": 1, "target": 1})
    elif result == "draw":
        events.append({"type": "ryukyoku"})
    else:
        raise ValueError(f"unsupported artificial result: {result}")

    events.append({"type": "end_kyoku"})
    return events


def kyoku_without_dealer_double_riichi() -> list[dict[str, object]]:
    return [
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 3,
            "honba": 0,
            "oya": 0,
        },
        {"type": "tsumo", "actor": 0, "pai": "1m"},
        {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": True},
        {"type": "ryukyoku"},
        {"type": "end_kyoku"},
    ]


def test_aggregate_dealer_double_riichi_counts_mixed_games(
    tmp_path: Path,
) -> None:
    first = write_mjai(
        tmp_path / "2025010100gm-00a9-0000-00000001.mjson",
        [
            dealer_double_riichi_kyoku("dealer_win", kyoku=1),
            dealer_double_riichi_kyoku("other_win", kyoku=2),
            dealer_double_riichi_kyoku("dealer_win", bakaze="S"),
        ],
    )
    second = write_mjai(
        tmp_path / "2025010101gm-00a9-0000-00000002.mjson",
        [
            dealer_double_riichi_kyoku("dealer_win", kyoku=1),
            dealer_double_riichi_kyoku("draw", kyoku=2),
            kyoku_without_dealer_double_riichi(),
        ],
    )

    stats = aggregate_dealer_double_riichi(2025, [first, second])

    assert stats.year == 2025
    assert stats.target_games == 2
    assert stats.east_kyokus == 5
    assert stats.dealer_double_riichi == 4
    assert stats.dealer_win == 2
    assert stats.other_win == 1
    assert stats.draw == 1
    assert stats.win_rate == 0.5
    assert stats.dealer_win + stats.other_win + stats.draw == (
        stats.dealer_double_riichi
    )


def test_aggregate_dealer_double_riichi_excludes_non_target_game(
    tmp_path: Path,
) -> None:
    path = write_mjai(
        tmp_path / "2025010100gm-00e1-0000-00000001.mjson",
        [dealer_double_riichi_kyoku("dealer_win")],
    )

    stats = aggregate_dealer_double_riichi(2025, [path])

    assert stats.target_games == 0
    assert stats.east_kyokus == 0
    assert stats.dealer_double_riichi == 0
    assert stats.win_rate is None


def test_aggregate_dealer_double_riichi_returns_none_without_observations(
    tmp_path: Path,
) -> None:
    path = write_mjai(
        tmp_path / "2025010100gm-00a9-0000-00000001.mjson",
        [kyoku_without_dealer_double_riichi()],
    )

    stats = aggregate_dealer_double_riichi(2025, [path])

    assert stats.target_games == 1
    assert stats.east_kyokus == 1
    assert stats.dealer_double_riichi == 0
    assert stats.win_rate is None


def test_aggregate_dealer_double_riichi_accepts_generator(
    tmp_path: Path,
) -> None:
    paths = [
        write_mjai(
            tmp_path / f"202501010{i}gm-00a9-0000-0000000{i}.mjson",
            [kyoku_without_dealer_double_riichi()],
        )
        for i in range(2)
    ]

    stats = aggregate_dealer_double_riichi(2025, (path for path in paths))

    assert stats.target_games == 2
    assert stats.east_kyokus == 2


def test_aggregate_dealer_double_riichi_reports_invalid_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "2025010100gm-00a9-0000-00000001.mjson"
    path.write_text("not JSON\n", encoding="utf-8")

    with pytest.raises(RuntimeError) as error:
        aggregate_dealer_double_riichi(2025, [path])

    assert path.name in str(error.value)
    assert isinstance(error.value.__cause__, JSONDecodeError)
