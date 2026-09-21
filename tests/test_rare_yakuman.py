import gzip
import json
import zlib
from pathlib import Path

import pytest

import mahjong_analysis.rare_yakuman as rare_yakuman
from mahjong_analysis.rare_yakuman import (
    RareYakumanReplayError,
    analyze_rare_yakuman_game,
    extract_complete_log_id,
    get_detector,
    is_ryuuiisou,
    replay_kyoku_horas,
    result_to_dict,
    scan_rare_yakuman,
    tenhou_log_url,
)

_FILLER_HAND = [
    "1m",
    "1m",
    "1m",
    "2m",
    "2m",
    "2m",
    "3m",
    "3m",
    "3m",
    "4m",
    "4m",
    "4m",
    "5m",
]
_GREEN_WITH_F_WAIT = [
    "2s",
    "2s",
    "2s",
    "3s",
    "3s",
    "3s",
    "4s",
    "4s",
    "4s",
    "6s",
    "6s",
    "6s",
    "F",
]
_GREEN_WITHOUT_F_WAIT = [
    "2s",
    "2s",
    "2s",
    "3s",
    "3s",
    "3s",
    "4s",
    "4s",
    "4s",
    "6s",
    "6s",
    "6s",
    "8s",
]


def _start(tehais: list[list[str]]) -> dict[str, object]:
    return {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "dora_marker": "1p",
        "tehais": tehais,
    }


def _kyoku(
    actor_zero_hand: list[str], *events: dict[str, object]
) -> list[dict[str, object]]:
    return [
        _start(
            [
                actor_zero_hand,
                list(_FILLER_HAND),
                list(_FILLER_HAND),
                list(_FILLER_HAND),
            ]
        ),
        *events,
        {"type": "end_kyoku"},
    ]


def _tsumo_hora_kyoku(
    initial_hand: list[str], winning_tile: str
) -> list[dict[str, object]]:
    return _kyoku(
        initial_hand,
        {"type": "tsumo", "actor": 0, "pai": winning_tile},
        {"type": "hora", "actor": 0, "target": 0},
    )


def _ron_kyoku(
    actor_zero_hand: list[str],
    tile: str,
    *horas: dict[str, object],
    other_hands: dict[int, list[str]] | None = None,
) -> list[dict[str, object]]:
    tehais = [
        actor_zero_hand,
        list(_FILLER_HAND),
        list(_FILLER_HAND),
        list(_FILLER_HAND),
    ]
    if other_hands:
        for actor, hand in other_hands.items():
            tehais[actor] = hand
    discard_hand = list(tehais[1])
    discard_hand[-1] = tile
    tehais[1] = discard_hand
    return [
        _start(tehais),
        {"type": "tsumo", "actor": 1, "pai": "9p"},
        {"type": "dahai", "actor": 1, "pai": tile, "tsumogiri": False},
        *horas,
        {"type": "end_kyoku"},
    ]


def _game(*kyokus: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {"type": "start_game", "kyoku_first": 0, "aka_flag": True, "names": []},
        *(event for kyoku in kyokus for event in kyoku),
        {"type": "end_game"},
    ]


def _kakan_raw_anomaly_kyoku() -> list[dict[str, object]]:
    caller = [
        "5s",
        "5s",
        "1m",
        "1m",
        "1m",
        "2m",
        "2m",
        "2m",
        "3m",
        "3m",
        "3m",
        "4m",
        "4m",
    ]
    discarder = list(_FILLER_HAND)
    discarder[-1] = "5s"
    return [
        _start(
            [
                caller,
                list(_FILLER_HAND),
                list(_FILLER_HAND),
                discarder,
            ]
        ),
        {"type": "tsumo", "actor": 3, "pai": "9p"},
        {"type": "dahai", "actor": 3, "pai": "5s", "tsumogiri": False},
        {
            "type": "pon",
            "actor": 0,
            "target": 3,
            "pai": "5s",
            "consumed": ["5s", "5s"],
        },
        {"type": "dahai", "actor": 0, "pai": "4m", "tsumogiri": False},
        {"type": "tsumo", "actor": 0, "pai": "5sr"},
        {"type": "dahai", "actor": 0, "pai": "5sr", "tsumogiri": True},
        {
            "type": "kakan",
            "actor": 0,
            "pai": "5sr",
            "consumed": ["5s", "5s", "5s"],
        },
        {"type": "ryukyoku"},
        {"type": "end_kyoku"},
    ]


def _write_game(path: Path, events: list[dict[str, object]], *, gzip_: bool) -> None:
    payload = "".join(
        json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        for event in events
    )
    if gzip_:
        with gzip.open(path, "wt", encoding="utf-8") as file:
            file.write(payload)
    else:
        path.write_text(payload, encoding="utf-8")


def test_ryuuiisou_tsumo_with_green_dragon() -> None:
    horas = replay_kyoku_horas(_tsumo_hora_kyoku(_GREEN_WITH_F_WAIT, "F"))

    assert len(horas) == 1
    assert is_ryuuiisou(horas[0].owned_tiles)
    assert horas[0].owned_tiles.count("F") == 2


def test_ryuuiisou_ron_adds_the_claimable_tile() -> None:
    kyoku = _ron_kyoku(
        _GREEN_WITH_F_WAIT,
        "F",
        {"type": "hora", "actor": 0, "target": 1},
    )

    horas = replay_kyoku_horas(kyoku)

    assert is_ryuuiisou(horas[0].owned_tiles)
    assert horas[0].owned_tiles.count("F") == 2


def test_ryuuiisou_does_not_require_green_dragon() -> None:
    horas = replay_kyoku_horas(_tsumo_hora_kyoku(_GREEN_WITHOUT_F_WAIT, "8s"))

    assert is_ryuuiisou(horas[0].owned_tiles)
    assert "F" not in horas[0].owned_tiles


@pytest.mark.parametrize("forbidden_tile", ["5s", "5sr", "1m", "E"])
def test_ryuuiisou_rejects_any_forbidden_tile(forbidden_tile: str) -> None:
    initial = [*_GREEN_WITH_F_WAIT[:-1], forbidden_tile]
    horas = replay_kyoku_horas(_tsumo_hora_kyoku(initial, "F"))

    assert not is_ryuuiisou(horas[0].owned_tiles)


def test_ryuuiisou_with_open_pon_keeps_meld_ownership() -> None:
    actor_zero = [
        "2s",
        "2s",
        "3s",
        "3s",
        "3s",
        "4s",
        "4s",
        "4s",
        "6s",
        "6s",
        "6s",
        "8s",
        "F",
    ]
    actor_one = [*_FILLER_HAND[:-1], "2s"]
    kyoku = [
        _start([actor_zero, actor_one, list(_FILLER_HAND), list(_FILLER_HAND)]),
        {"type": "tsumo", "actor": 1, "pai": "9p"},
        {"type": "dahai", "actor": 1, "pai": "2s", "tsumogiri": False},
        {
            "type": "pon",
            "actor": 0,
            "target": 1,
            "pai": "2s",
            "consumed": ["2s", "2s"],
        },
        {"type": "dahai", "actor": 0, "pai": "F", "tsumogiri": False},
        {"type": "tsumo", "actor": 0, "pai": "F"},
        {"type": "hora", "actor": 0, "target": 0},
        {"type": "end_kyoku"},
    ]

    horas = replay_kyoku_horas(kyoku)

    assert is_ryuuiisou(horas[0].owned_tiles)
    assert horas[0].owned_tiles.count("2s") == 3


def test_replay_rejects_self_hora_immediately_after_pon_without_tsumo() -> None:
    actor_zero = [
        "2s",
        "2s",
        "3s",
        "3s",
        "3s",
        "4s",
        "4s",
        "4s",
        "6s",
        "6s",
        "6s",
        "F",
        "F",
    ]
    actor_one = [*_FILLER_HAND[:-1], "2s"]
    kyoku = [
        _start([actor_zero, actor_one, list(_FILLER_HAND), list(_FILLER_HAND)]),
        {"type": "tsumo", "actor": 1, "pai": "9p"},
        {"type": "dahai", "actor": 1, "pai": "2s", "tsumogiri": False},
        {
            "type": "pon",
            "actor": 0,
            "target": 1,
            "pai": "2s",
            "consumed": ["2s", "2s"],
        },
        {"type": "hora", "actor": 0, "target": 0},
        {"type": "end_kyoku"},
    ]

    with pytest.raises(ValueError, match="unconsumed tsumo event"):
        replay_kyoku_horas(kyoku)


def test_ryuuiisou_with_green_ankan_keeps_all_four_tiles() -> None:
    actor_zero = [
        "2s",
        "2s",
        "2s",
        "2s",
        "3s",
        "3s",
        "3s",
        "4s",
        "4s",
        "4s",
        "6s",
        "6s",
        "6s",
    ]
    kyoku = _kyoku(
        actor_zero,
        {"type": "tsumo", "actor": 0, "pai": "8s"},
        {"type": "ankan", "actor": 0, "consumed": ["2s"] * 4},
        {"type": "dora", "dora_marker": "1m"},
        {"type": "tsumo", "actor": 0, "pai": "8s"},
        {"type": "hora", "actor": 0, "target": 0},
    )

    horas = replay_kyoku_horas(kyoku)

    assert is_ryuuiisou(horas[0].owned_tiles)
    assert horas[0].owned_tiles.count("2s") == 4


def test_replay_rejects_self_hora_after_ankan_without_rinshan_tsumo() -> None:
    actor_zero = [
        "2s",
        "2s",
        "2s",
        "2s",
        "3s",
        "3s",
        "3s",
        "4s",
        "4s",
        "4s",
        "6s",
        "6s",
        "6s",
    ]
    kyoku = _kyoku(
        actor_zero,
        {"type": "tsumo", "actor": 0, "pai": "8s"},
        {"type": "ankan", "actor": 0, "consumed": ["2s"] * 4},
        {"type": "dora", "dora_marker": "1m"},
        {"type": "hora", "actor": 0, "target": 0},
    )

    with pytest.raises(ValueError, match="unconsumed tsumo event"):
        replay_kyoku_horas(kyoku)


def test_replay_rejects_reusing_one_tsumo_for_two_self_horas() -> None:
    kyoku = _tsumo_hora_kyoku(_GREEN_WITH_F_WAIT, "F")
    kyoku.insert(-1, {"type": "hora", "actor": 0, "target": 0})

    with pytest.raises(ValueError, match="unconsumed tsumo event"):
        replay_kyoku_horas(kyoku)


def test_non_green_chi_tile_remains_in_winner_ownership() -> None:
    actor_zero = [
        "3s",
        "4s",
        "2s",
        "2s",
        "2s",
        "6s",
        "6s",
        "6s",
        "8s",
        "8s",
        "F",
        "F",
        "F",
    ]
    actor_three = [*_FILLER_HAND[:-1], "5s"]
    kyoku = [
        _start([actor_zero, list(_FILLER_HAND), list(_FILLER_HAND), actor_three]),
        {"type": "tsumo", "actor": 3, "pai": "9p"},
        {"type": "dahai", "actor": 3, "pai": "5s", "tsumogiri": False},
        {
            "type": "chi",
            "actor": 0,
            "target": 3,
            "pai": "5s",
            "consumed": ["3s", "4s"],
        },
        {"type": "dahai", "actor": 0, "pai": "F", "tsumogiri": False},
        {"type": "tsumo", "actor": 0, "pai": "F"},
        {"type": "hora", "actor": 0, "target": 0},
        {"type": "end_kyoku"},
    ]

    horas = replay_kyoku_horas(kyoku)

    assert "5s" in horas[0].owned_tiles
    assert not is_ryuuiisou(horas[0].owned_tiles)


def test_non_green_daiminkan_tile_remains_in_winner_ownership() -> None:
    actor_zero = ["E", "E", "E", *_GREEN_WITHOUT_F_WAIT[:10]]
    actor_three = [*_FILLER_HAND[:-1], "E"]
    kyoku = [
        _start([actor_zero, list(_FILLER_HAND), list(_FILLER_HAND), actor_three]),
        {"type": "tsumo", "actor": 3, "pai": "9p"},
        {"type": "dahai", "actor": 3, "pai": "E", "tsumogiri": False},
        {
            "type": "daiminkan",
            "actor": 0,
            "target": 3,
            "pai": "E",
            "consumed": ["E", "E", "E"],
        },
        {"type": "dora", "dora_marker": "1m"},
        {"type": "tsumo", "actor": 0, "pai": "8s"},
        {"type": "hora", "actor": 0, "target": 0},
        {"type": "end_kyoku"},
    ]

    horas = replay_kyoku_horas(kyoku)

    assert horas[0].owned_tiles.count("E") == 4
    assert not is_ryuuiisou(horas[0].owned_tiles)


def test_kakan_chankan_uses_added_tile_as_claimable_tile() -> None:
    actor_zero = [
        "2s",
        "2s",
        "2s",
        "6s",
        "6s",
        "6s",
        "F",
        "F",
        "F",
        "8s",
        "8s",
        "2s",
        "4s",
    ]
    actor_one = ["3s", "3s", "1m", *_FILLER_HAND[:10]]
    actor_two = [*_FILLER_HAND[:-1], "3s"]
    kyoku = [
        _start([actor_zero, actor_one, actor_two, list(_FILLER_HAND)]),
        {"type": "tsumo", "actor": 2, "pai": "9p"},
        {"type": "dahai", "actor": 2, "pai": "3s", "tsumogiri": False},
        {
            "type": "pon",
            "actor": 1,
            "target": 2,
            "pai": "3s",
            "consumed": ["3s", "3s"],
        },
        {"type": "dahai", "actor": 1, "pai": "1m", "tsumogiri": False},
        {"type": "tsumo", "actor": 1, "pai": "3s"},
        {
            "type": "kakan",
            "actor": 1,
            "pai": "3s",
            "consumed": ["3s", "3s", "3s"],
        },
        {"type": "hora", "actor": 0, "target": 1},
        {"type": "end_kyoku"},
    ]

    horas = replay_kyoku_horas(kyoku)

    assert is_ryuuiisou(horas[0].owned_tiles)
    assert horas[0].owned_tiles[-1] == "3s"


def test_double_ron_counts_one_kyoku_and_two_matching_horas(tmp_path: Path) -> None:
    actor_two = list(_GREEN_WITH_F_WAIT)
    kyoku = _ron_kyoku(
        _GREEN_WITH_F_WAIT,
        "F",
        {"type": "hora", "actor": 0, "target": 1},
        {"type": "hora", "actor": 2, "target": 1},
        other_hands={2: actor_two},
    )
    path = tmp_path / "2025010100gm-00a9-0000-12345678.mjson"
    _write_game(path, _game(kyoku), gzip_=False)

    result = analyze_rare_yakuman_game(path, raw_root=tmp_path, yaku="緑一色")

    assert result.total_games == 1
    assert result.total_kyokus == 1
    assert result.analyzed_kyokus == 1
    assert result.anomaly_kyokus == 0
    assert result.matching_kyokus == 1
    assert result.matching_hora_events == 2
    assert result.probability_per_analyzed_kyoku == 1.0
    assert result.one_in_n_analyzed_kyokus == 1.0
    assert result.hits[0].matching_hora_event_indices == (3, 4)


def test_matching_and_nonmatching_hora_share_one_matching_kyoku(
    tmp_path: Path,
) -> None:
    non_green = [
        "1m",
        "1m",
        "1m",
        "2m",
        "2m",
        "2m",
        "3m",
        "3m",
        "3m",
        "4m",
        "4m",
        "4m",
        "F",
    ]
    kyoku = _ron_kyoku(
        _GREEN_WITH_F_WAIT,
        "F",
        {"type": "hora", "actor": 0, "target": 1},
        {"type": "hora", "actor": 2, "target": 1},
        other_hands={2: non_green},
    )
    path = tmp_path / "2025010100gm-00a9-0000-12345678.mjson"
    _write_game(path, _game(kyoku), gzip_=False)

    result = analyze_rare_yakuman_game(path, raw_root=tmp_path, yaku="緑一色")

    assert result.matching_kyokus == 1
    assert result.matching_hora_events == 1
    assert result.hits[0].matching_hora_event_indices == (3,)


def test_replay_rejects_fifth_copy_added_by_ron() -> None:
    kyoku = _ron_kyoku(
        ["2s"] * 4 + ["3s"] * 3 + ["4s"] * 3 + ["F"] * 3,
        "2s",
        {"type": "hora", "actor": 0, "target": 1},
    )

    with pytest.raises(ValueError, match="more than four owned copies of 2s"):
        replay_kyoku_horas(kyoku)


def test_replay_rejects_duplicate_raw_red_tile_added_by_ron() -> None:
    winner = [
        "5mr",
        "1m",
        "1m",
        "1m",
        "2m",
        "2m",
        "2m",
        "3m",
        "3m",
        "3m",
        "4m",
        "4m",
        "4m",
    ]
    kyoku = _ron_kyoku(
        winner,
        "5mr",
        {"type": "hora", "actor": 0, "target": 1},
    )

    with pytest.raises(ValueError, match="multiple physical copies of 5mr"):
        replay_kyoku_horas(kyoku)


def test_dahai_removes_exact_raw_five_without_removing_red_five() -> None:
    initial = ["5m", "5mr", *(["2s"] * 3), *(["3s"] * 3), *(["4s"] * 3), "F", "F"]
    kyoku = _kyoku(
        initial,
        {"type": "tsumo", "actor": 0, "pai": "6s"},
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
        {"type": "tsumo", "actor": 0, "pai": "6s"},
        {"type": "hora", "actor": 0, "target": 0},
    )

    owned = replay_kyoku_horas(kyoku)[0].owned_tiles

    assert "5m" not in owned
    assert owned.count("5mr") == 1


def test_zero_matches_has_safe_rates(tmp_path: Path) -> None:
    path = tmp_path / "2025010100gm-00a9-0000-12345678.mjson"
    _write_game(
        path,
        _game(_tsumo_hora_kyoku(_FILLER_HAND, "5m")),
        gzip_=False,
    )

    result = analyze_rare_yakuman_game(path, raw_root=tmp_path, yaku="緑一色")

    assert result.matching_kyokus == 0
    assert result.matching_hora_events == 0
    assert result.probability_per_analyzed_kyoku == 0.0
    assert result.one_in_n_analyzed_kyokus is None


def test_complete_log_id_and_tenhou_url_use_the_full_filename_stem() -> None:
    path = Path("nested/2025010100gm-00a9-0000-029e0f35.mjson")

    log_id = extract_complete_log_id(path)

    assert log_id == "2025010100gm-00a9-0000-029e0f35"
    assert tenhou_log_url(log_id) == (
        "https://tenhou.net/5/?log=2025010100gm-00a9-0000-029e0f35"
    )


def test_unsupported_yaku_lists_available_detectors() -> None:
    with pytest.raises(
        ValueError,
        match="unsupported yaku.*available yaku: 緑一色",
    ):
        get_detector("九蓮宝燈")


def test_plain_and_gzip_games_are_scanned_together(tmp_path: Path) -> None:
    for year in (2024, 2025):
        (tmp_path / str(year)).mkdir()
    gzip_path = tmp_path / "2024" / "2024010100gm-00a9-0000-12345678.mjson"
    plain_path = tmp_path / "2025" / "2025010100gm-00a9-0000-87654321.mjson"
    events = _game(_tsumo_hora_kyoku(_GREEN_WITH_F_WAIT, "F"))
    _write_game(gzip_path, events, gzip_=True)
    _write_game(plain_path, events, gzip_=False)

    result = scan_rare_yakuman(
        tmp_path,
        yaku="緑一色",
        years=(2024, 2025),
    )

    assert result.total_games == 2
    assert result.total_kyokus == 2
    assert result.matching_kyokus == 2
    assert result.matching_hora_events == 2
    assert [hit.source_path for hit in result.hits] == [
        "2024/2024010100gm-00a9-0000-12345678.mjson",
        "2025/2025010100gm-00a9-0000-87654321.mjson",
    ]


def test_parallel_year_scan_matches_serial_result(tmp_path: Path) -> None:
    for year in (2024, 2025):
        (tmp_path / str(year)).mkdir()
    events = _game(_tsumo_hora_kyoku(_GREEN_WITH_F_WAIT, "F"))
    _write_game(
        tmp_path / "2024" / "2024010100gm-00a9-0000-12345678.mjson",
        events,
        gzip_=True,
    )
    _write_game(
        tmp_path / "2025" / "2025010100gm-00a9-0000-87654321.mjson",
        events,
        gzip_=False,
    )

    serial = scan_rare_yakuman(tmp_path, yaku="緑一色", years=(2025, 2024), workers=1)
    parallel = scan_rare_yakuman(tmp_path, yaku="緑一色", years=(2025, 2024), workers=2)

    assert parallel == serial


def test_parallel_continue_mode_merges_anomalies_deterministically(
    tmp_path: Path,
) -> None:
    for year in (2024, 2025):
        (tmp_path / str(year)).mkdir()
    _write_game(
        tmp_path / "2024" / "2024010100gm-00a9-0000-12345678.mjson",
        _game(_kakan_raw_anomaly_kyoku()),
        gzip_=True,
    )
    _write_game(
        tmp_path / "2025" / "2025010100gm-00a9-0000-87654321.mjson",
        _game(
            _kakan_raw_anomaly_kyoku(),
            _tsumo_hora_kyoku(_GREEN_WITH_F_WAIT, "F"),
        ),
        gzip_=False,
    )

    serial = scan_rare_yakuman(
        tmp_path,
        yaku="緑一色",
        years=(2025, 2024),
        workers=1,
        continue_on_replay_error=True,
    )
    parallel = scan_rare_yakuman(
        tmp_path,
        yaku="緑一色",
        years=(2025, 2024),
        workers=2,
        continue_on_replay_error=True,
    )

    assert parallel == serial
    assert parallel.total_games == 2
    assert parallel.total_kyokus == 3
    assert parallel.analyzed_kyokus == 1
    assert parallel.anomaly_kyokus == 2
    assert parallel.matching_kyokus == 1
    assert parallel.matching_hora_events == 1
    assert [anomaly.source_path for anomaly in parallel.anomalies] == [
        "2024/2024010100gm-00a9-0000-12345678.mjson",
        "2025/2025010100gm-00a9-0000-87654321.mjson",
    ]
    assert [(hit.source_path, hit.kyoku_index) for hit in parallel.hits] == [
        ("2025/2025010100gm-00a9-0000-87654321.mjson", 1)
    ]


def test_replay_rejects_missing_discard_tile() -> None:
    kyoku = _kyoku(
        _GREEN_WITH_F_WAIT,
        {"type": "tsumo", "actor": 0, "pai": "8s"},
        {"type": "dahai", "actor": 0, "pai": "9s", "tsumogiri": False},
        {"type": "ryukyoku"},
    )

    with pytest.raises(ValueError, match="concealed hand lacks raw tiles"):
        replay_kyoku_horas(kyoku)


def test_replay_rejects_ron_without_claimable_tile() -> None:
    kyoku = _kyoku(
        _GREEN_WITH_F_WAIT,
        {"type": "hora", "actor": 0, "target": 1},
    )

    with pytest.raises(ValueError, match="claimable tile"):
        replay_kyoku_horas(kyoku)


def test_replay_rejects_kyoku_without_terminal_result() -> None:
    kyoku = _kyoku(
        _GREEN_WITH_F_WAIT,
        {"type": "dora", "dora_marker": "1m"},
    )

    with pytest.raises(ValueError, match="no hora or ryukyoku result"):
        replay_kyoku_horas(kyoku)


def test_game_replay_error_includes_source_and_kyoku_context(tmp_path: Path) -> None:
    source = tmp_path / "2025" / "2025010100gm-00a9-0000-12345678.mjson"
    source.parent.mkdir()
    kyoku = _kyoku(
        _GREEN_WITH_F_WAIT,
        {"type": "tsumo", "actor": 0, "pai": "8s"},
        {"type": "dahai", "actor": 0, "pai": "9s", "tsumogiri": False},
        {"type": "ryukyoku"},
    )
    _write_game(source, _game(kyoku), gzip_=False)

    with pytest.raises(ValueError) as caught:
        analyze_rare_yakuman_game(source, raw_root=tmp_path, yaku="緑一色")

    message = str(caught.value)
    assert "2025/2025010100gm-00a9-0000-12345678.mjson" in message
    assert "kyoku_index=0" in message
    assert "event 2, actor 0" in message
    assert caught.value.__cause__ is not None


def test_kakan_raw_anomaly_stops_in_strict_mode(tmp_path: Path) -> None:
    source = tmp_path / "2025" / "2025010100gm-00a9-0000-12345678.mjson"
    source.parent.mkdir()
    _write_game(
        source,
        _game(
            _kakan_raw_anomaly_kyoku(),
            _tsumo_hora_kyoku(_GREEN_WITH_F_WAIT, "F"),
        ),
        gzip_=False,
    )

    with pytest.raises(ValueError, match="kyoku_index=0") as caught:
        analyze_rare_yakuman_game(source, raw_root=tmp_path, yaku="緑一色")

    assert "concealed hand lacks raw tiles: {'5sr': 1}" in str(caught.value)
    assert isinstance(caught.value.__cause__, RareYakumanReplayError)


def test_continue_mode_collects_anomaly_and_analyzes_following_hit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "2025" / "2025010100gm-00a9-0000-12345678.mjson"
    source.parent.mkdir()
    _write_game(
        source,
        _game(
            _kakan_raw_anomaly_kyoku(),
            _tsumo_hora_kyoku(_GREEN_WITH_F_WAIT, "F"),
        ),
        gzip_=False,
    )

    result = analyze_rare_yakuman_game(
        source,
        raw_root=tmp_path,
        yaku="緑一色",
        continue_on_replay_error=True,
    )

    assert result.total_games == 1
    assert result.total_kyokus == 2
    assert result.analyzed_kyokus == 1
    assert result.anomaly_kyokus == 1
    assert result.analyzed_kyokus + result.anomaly_kyokus == result.total_kyokus
    assert result.matching_kyokus == 1
    assert result.matching_hora_events == 1
    assert result.probability_per_analyzed_kyoku == 1.0
    assert result.one_in_n_analyzed_kyokus == 1.0
    assert [
        (hit.kyoku_index, hit.matching_hora_event_indices) for hit in result.hits
    ] == [(1, (2,))]
    assert len(result.anomalies) == 1
    anomaly = result.anomalies[0]
    assert anomaly.source_path == "2025/2025010100gm-00a9-0000-12345678.mjson"
    assert anomaly.log_id == "2025010100gm-00a9-0000-12345678"
    assert anomaly.kyoku_index == 0
    assert "event 7, actor 0" in anomaly.error_message
    assert "concealed hand lacks raw tiles: {'5sr': 1}" in anomaly.error_message
    document = result_to_dict(result)
    assert document["total_kyokus"] == 2
    assert document["analyzed_kyokus"] == 1
    assert document["anomaly_kyokus"] == 1
    assert document["probability_per_analyzed_kyoku"] == 1.0
    assert document["one_in_n_analyzed_kyokus"] == 1.0
    assert document["anomalies"] == [
        {
            "source_path": "2025/2025010100gm-00a9-0000-12345678.mjson",
            "log_id": "2025010100gm-00a9-0000-12345678",
            "kyoku_index": 0,
            "error_message": anomaly.error_message,
        }
    ]


def test_continue_mode_discards_partial_hit_before_later_replay_error(
    tmp_path: Path,
) -> None:
    anomalous_kyoku = _tsumo_hora_kyoku(_GREEN_WITH_F_WAIT, "F")
    anomalous_kyoku.insert(-1, {"type": "hora", "actor": 0, "target": 0})
    source = tmp_path / "2025" / "2025010100gm-00a9-0000-12345678.mjson"
    source.parent.mkdir()
    _write_game(
        source,
        _game(
            anomalous_kyoku,
            _tsumo_hora_kyoku(_GREEN_WITH_F_WAIT, "F"),
        ),
        gzip_=False,
    )

    result = analyze_rare_yakuman_game(
        source,
        raw_root=tmp_path,
        yaku="緑一色",
        continue_on_replay_error=True,
    )

    assert result.total_kyokus == 2
    assert result.analyzed_kyokus == 1
    assert result.anomaly_kyokus == 1
    assert result.analyzed_kyokus + result.anomaly_kyokus == result.total_kyokus
    assert result.matching_kyokus == 1
    assert result.matching_hora_events == 1
    assert [
        (hit.kyoku_index, hit.matching_hora_event_indices) for hit in result.hits
    ] == [(1, (2,))]
    assert [anomaly.kyoku_index for anomaly in result.anomalies] == [0]
    assert "unconsumed tsumo event" in result.anomalies[0].error_message


def test_continue_mode_with_only_anomaly_has_safe_rates(tmp_path: Path) -> None:
    source = tmp_path / "2025" / "2025010100gm-00a9-0000-12345678.mjson"
    source.parent.mkdir()
    _write_game(source, _game(_kakan_raw_anomaly_kyoku()), gzip_=False)

    result = analyze_rare_yakuman_game(
        source,
        raw_root=tmp_path,
        yaku="緑一色",
        continue_on_replay_error=True,
    )

    assert result.total_kyokus == 1
    assert result.analyzed_kyokus == 0
    assert result.anomaly_kyokus == 1
    assert result.matching_kyokus == 0
    assert result.probability_per_analyzed_kyoku is None
    assert result.one_in_n_analyzed_kyokus is None


def test_continue_mode_does_not_collect_unexpected_runtime_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "2025" / "2025010100gm-00a9-0000-12345678.mjson"
    source.parent.mkdir()
    _write_game(
        source,
        _game(_tsumo_hora_kyoku(_GREEN_WITH_F_WAIT, "F")),
        gzip_=False,
    )
    monkeypatch.setattr(
        rare_yakuman,
        "replay_kyoku_horas",
        lambda events: (_ for _ in ()).throw(RuntimeError("injected bug")),
    )

    with pytest.raises(RuntimeError, match="injected bug"):
        analyze_rare_yakuman_game(
            source,
            raw_root=tmp_path,
            yaku="緑一色",
            continue_on_replay_error=True,
        )


def test_game_load_error_includes_source_context(tmp_path: Path) -> None:
    source = tmp_path / "2025" / "2025010100gm-00a9-0000-12345678.mjson"
    source.parent.mkdir()
    source.write_text("not JSON\n", encoding="utf-8")

    with pytest.raises(ValueError) as caught:
        analyze_rare_yakuman_game(
            source,
            raw_root=tmp_path,
            yaku="緑一色",
            continue_on_replay_error=True,
        )

    assert "2025/2025010100gm-00a9-0000-12345678.mjson" in str(caught.value)
    assert caught.value.__cause__ is not None


def test_continue_mode_does_not_collect_game_validation_error(tmp_path: Path) -> None:
    source = tmp_path / "2025" / "2025010100gm-00a9-0000-12345678.mjson"
    source.parent.mkdir()
    _write_game(
        source,
        [
            {"type": "start_game"},
            {"type": "start_game"},
            {"type": "end_game"},
        ],
        gzip_=False,
    )

    with pytest.raises(ValueError, match="exactly one start_game") as caught:
        analyze_rare_yakuman_game(
            source,
            raw_root=tmp_path,
            yaku="緑一色",
            continue_on_replay_error=True,
        )

    assert "2025/2025010100gm-00a9-0000-12345678.mjson" in str(caught.value)


def test_truncated_gzip_error_includes_source_context(tmp_path: Path) -> None:
    source = tmp_path / "2025" / "2025010100gm-00a9-0000-12345678.mjson"
    source.parent.mkdir()
    compressed = gzip.compress(b'{"type":"start_game"}\n{"type":"end_game"}\n')
    source.write_bytes(compressed[:-4])

    with pytest.raises(ValueError) as caught:
        analyze_rare_yakuman_game(
            source,
            raw_root=tmp_path,
            yaku="緑一色",
            continue_on_replay_error=True,
        )

    message = str(caught.value)
    assert "2025/2025010100gm-00a9-0000-12345678.mjson" in message
    assert "Compressed file ended before the end-of-stream marker" in message
    assert isinstance(caught.value.__cause__, EOFError)


def test_corrupt_deflate_error_includes_source_context(tmp_path: Path) -> None:
    source = tmp_path / "2025" / "2025010100gm-00a9-0000-12345678.mjson"
    source.parent.mkdir()
    compressed = bytearray(
        gzip.compress(b'{"type":"start_game"}\n{"type":"end_game"}\n')
    )
    compressed[10] = 0b111
    source.write_bytes(compressed)

    with pytest.raises(ValueError) as caught:
        analyze_rare_yakuman_game(source, raw_root=tmp_path, yaku="緑一色")

    message = str(caught.value)
    assert "2025/2025010100gm-00a9-0000-12345678.mjson" in message
    assert "invalid block type" in message
    assert isinstance(caught.value.__cause__, zlib.error)


def test_game_split_error_includes_source_context(tmp_path: Path) -> None:
    source = tmp_path / "2025" / "2025010100gm-00a9-0000-12345678.mjson"
    source.parent.mkdir()
    events = [
        {"type": "start_game"},
        _start([list(_FILLER_HAND) for _ in range(4)]),
        {"type": "end_game"},
    ]
    _write_game(source, events, gzip_=False)

    with pytest.raises(ValueError) as caught:
        analyze_rare_yakuman_game(
            source,
            raw_root=tmp_path,
            yaku="緑一色",
            continue_on_replay_error=True,
        )

    message = str(caught.value)
    assert "2025/2025010100gm-00a9-0000-12345678.mjson" in message
    assert "end_game encountered inside a kyoku" in message
    assert caught.value.__cause__ is not None


def test_parallel_scan_propagates_worker_error_with_source_context(
    tmp_path: Path,
) -> None:
    for year in (2024, 2025):
        (tmp_path / str(year)).mkdir()
    source = tmp_path / "2025" / "2025010100gm-00a9-0000-12345678.mjson"
    compressed = gzip.compress(b'{"type":"start_game"}\n{"type":"end_game"}\n')
    source.write_bytes(compressed[:-4])

    with pytest.raises(ValueError) as caught:
        scan_rare_yakuman(
            tmp_path,
            yaku="緑一色",
            workers=2,
            years=(2024, 2025),
            continue_on_replay_error=True,
        )

    message = str(caught.value)
    assert "2025/2025010100gm-00a9-0000-12345678.mjson" in message
    assert "Compressed file ended before the end-of-stream marker" in message
    assert caught.value.__cause__ is not None
    assert "EOFError" in str(caught.value.__cause__)
