import ast
import json
from dataclasses import replace
from pathlib import Path

import pytest

import mahjong_analysis.reference_validation as reference_validation
from mahjong_analysis.reference_validation import (
    CandidateRound,
    analyze_reference_mjai,
    compare_candidate_rounds,
)


def _start_kyoku(
    *,
    bakaze: str = "E",
    kyoku: int = 1,
    honba: int = 0,
    oya: int = 0,
) -> dict:
    return {
        "type": "start_kyoku",
        "bakaze": bakaze,
        "kyoku": kyoku,
        "honba": honba,
        "oya": oya,
    }


def _round(
    body: list[dict],
    *,
    bakaze: str = "E",
    kyoku: int = 1,
    honba: int = 0,
    oya: int = 0,
) -> list[dict]:
    return [
        _start_kyoku(
            bakaze=bakaze,
            kyoku=kyoku,
            honba=honba,
            oya=oya,
        ),
        *body,
        {"type": "end_kyoku"},
    ]


def _write_game(
    path: Path,
    rounds: list[list[dict]],
    *,
    aka_flag: object = True,
    include_end_game: bool = True,
) -> Path:
    events = [{"type": "start_game", "aka_flag": aka_flag}]
    for round_events in rounds:
        events.extend(round_events)
    if include_end_game:
        events.append({"type": "end_game"})

    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    return path


def _established_reach(*, tsumogiri: bool = False) -> list[dict]:
    return [
        {"type": "tsumo", "actor": 0, "pai": "1m"},
        {"type": "reach", "actor": 0},
        {
            "type": "dahai",
            "actor": 0,
            "pai": "1m",
            "tsumogiri": tsumogiri,
        },
        {"type": "reach_accepted", "actor": 0},
    ]


def _candidate(
    *,
    filename: str = "2025010100gm-00a9-0000-00000001.mjson",
    start_kyoku_line: int = 2,
    result: str = "dealer_win",
) -> CandidateRound:
    return CandidateRound(
        filename=filename,
        start_kyoku_line=start_kyoku_line,
        reach_line=4,
        bakaze="E",
        kyoku=1,
        honba=0,
        oya=0,
        result=result,
    )


def test_reference_detects_established_dealer_first_reach_with_line_numbers(
    tmp_path: Path,
) -> None:
    path = _write_game(
        tmp_path / "2025010100gm-00a9-0000-00000001.mjson",
        [
            _round(
                [
                    *_established_reach(),
                    {"type": "hora", "actor": 0, "target": 0},
                ]
            )
        ],
    )

    result = analyze_reference_mjai(path)

    assert result.target_game is True
    assert result.east_kyokus == 1
    assert len(result.candidates) == 1
    assert result.candidates[0].start_kyoku_line == 2
    assert result.candidates[0].reach_line == 4
    assert result.candidates[0].result == "dealer_win"


@pytest.mark.parametrize("terminal_type", ["hora", "ryukyoku"])
def test_reference_excludes_unaccepted_dealer_first_reach(
    tmp_path: Path,
    terminal_type: str,
) -> None:
    terminal = {"type": terminal_type}
    if terminal_type == "hora":
        terminal.update({"actor": 1, "target": 0})
    path = _write_game(
        tmp_path / "2025010100gm-00a9-0000-00000002.mjson",
        [
            _round(
                [
                    {"type": "tsumo", "actor": 0},
                    {"type": "reach", "actor": 0},
                    {"type": "dahai", "actor": 0},
                    terminal,
                ]
            )
        ],
    )

    result = analyze_reference_mjai(path)

    assert result.candidates == ()
    assert result.reach_audits[0].event_after_dahai == terminal_type
    assert result.reach_audits[0].established_dealer_double_riichi is False


def test_reference_excludes_dealer_later_reach(tmp_path: Path) -> None:
    path = _write_game(
        tmp_path / "2025010100gm-00a9-0000-00000003.mjson",
        [
            _round(
                [
                    {"type": "tsumo", "actor": 0},
                    {"type": "dahai", "actor": 0},
                    {"type": "tsumo", "actor": 0},
                    {"type": "reach", "actor": 0},
                    {"type": "dahai", "actor": 0},
                    {"type": "reach_accepted", "actor": 0},
                    {"type": "ryukyoku"},
                ]
            )
        ],
    )

    result = analyze_reference_mjai(path)

    assert result.candidates == ()
    assert result.reach_audits[0].actor_prior_dahai_count == 1


def test_reference_keeps_malformed_child_first_reach_as_nonfatal_audit(
    tmp_path: Path,
) -> None:
    path = _write_game(
        tmp_path / "2025010100gm-00a9-0000-00000004.mjson",
        [
            _round(
                [
                    {"type": "tsumo", "actor": 1},
                    {"type": "reach", "actor": 1},
                    {"type": "tsumo", "actor": 2},
                    {"type": "ryukyoku"},
                ]
            )
        ],
    )

    result = analyze_reference_mjai(path)

    assert result.candidates == ()
    assert result.reach_audits[0].reach_actor == 1
    assert result.reach_audits[0].actor_prior_dahai_count == 0
    assert result.reach_audits[0].event_after_dahai is None


def test_reference_excludes_dealer_reach_after_ankan(tmp_path: Path) -> None:
    path = _write_game(
        tmp_path / "2025010100gm-00a9-0000-00000005.mjson",
        [
            _round(
                [
                    {"type": "tsumo", "actor": 0},
                    {"type": "ankan", "actor": 0},
                    {"type": "dora", "dora_marker": "1p"},
                    {"type": "tsumo", "actor": 0},
                    {"type": "reach", "actor": 0},
                    {"type": "dahai", "actor": 0},
                    {"type": "reach_accepted", "actor": 0},
                    {"type": "ryukyoku"},
                ]
            )
        ],
    )

    result = analyze_reference_mjai(path)

    assert result.candidates == ()
    assert result.reach_audits[0].ankan_before_reach is True


@pytest.mark.parametrize(
    ("result_event", "expected"),
    [
        ({"type": "hora", "actor": 1, "target": 1}, "other_win"),
        ({"type": "ryukyoku", "reason": "fanpai"}, "draw"),
    ],
)
def test_reference_classifies_other_win_and_draw(
    tmp_path: Path,
    result_event: dict,
    expected: str,
) -> None:
    path = _write_game(
        tmp_path / "2025010100gm-00a9-0000-00000006.mjson",
        [_round([*_established_reach(), result_event])],
    )

    result = analyze_reference_mjai(path)

    assert result.candidates[0].result == expected


def test_reference_checks_all_hora_and_finds_dealer_second(tmp_path: Path) -> None:
    path = _write_game(
        tmp_path / "2025010100gm-00a9-0000-00000007.mjson",
        [
            _round(
                [
                    *_established_reach(),
                    {"type": "hora", "actor": 1, "target": 2},
                    {"type": "hora", "actor": 0, "target": 2},
                ]
            )
        ],
    )

    result = analyze_reference_mjai(path)

    assert result.candidates[0].result == "dealer_win"


@pytest.mark.parametrize("bakaze", ["S", "W"])
def test_reference_excludes_south_and_west_rounds(
    tmp_path: Path,
    bakaze: str,
) -> None:
    path = _write_game(
        tmp_path / "2025010100gm-00a9-0000-00000010.mjson",
        [
            _round(
                [
                    *_established_reach(),
                    {"type": "hora", "actor": 0, "target": 0},
                ],
                bakaze=bakaze,
            )
        ],
    )

    result = analyze_reference_mjai(path)

    assert result.target_game is True
    assert result.east_kyokus == 0
    assert result.candidates == ()


@pytest.mark.parametrize(
    ("rule_code", "aka_flag"),
    [("00e1", True), ("00a9", False)],
)
def test_reference_excludes_non_target_games(
    tmp_path: Path,
    rule_code: str,
    aka_flag: bool,
) -> None:
    path = _write_game(
        tmp_path / f"2025010100gm-{rule_code}-0000-00000008.mjson",
        [
            _round(
                [
                    *_established_reach(),
                    {"type": "hora", "actor": 0, "target": 0},
                ]
            )
        ],
        aka_flag=aka_flag,
    )

    result = analyze_reference_mjai(path)

    assert result.target_game is False
    assert result.east_kyokus == 0
    assert result.candidates == ()


def test_reference_rejects_non_bool_aka_flag(tmp_path: Path) -> None:
    path = _write_game(
        tmp_path / "2025010100gm-00a9-0000-00000009.mjson",
        [],
        aka_flag=1,
    )

    with pytest.raises(ValueError, match="aka_flag must be bool"):
        analyze_reference_mjai(path)


def test_reference_rejects_dealer_reach_dahai_actor_mismatch(
    tmp_path: Path,
) -> None:
    path = _write_game(
        tmp_path / "2025010100gm-00a9-0000-0000000a.mjson",
        [
            _round(
                [
                    {"type": "reach", "actor": 0},
                    {"type": "dahai", "actor": 1},
                    {"type": "reach_accepted", "actor": 0},
                    {"type": "ryukyoku"},
                ]
            )
        ],
    )

    with pytest.raises(ValueError, match="actors differ"):
        analyze_reference_mjai(path)


def test_reference_rejects_reach_accepted_actor_mismatch(
    tmp_path: Path,
) -> None:
    path = _write_game(
        tmp_path / "2025010100gm-00a9-0000-0000000b.mjson",
        [
            _round(
                [
                    {"type": "reach", "actor": 0},
                    {"type": "dahai", "actor": 0},
                    {"type": "reach_accepted", "actor": 1},
                    {"type": "ryukyoku"},
                ]
            )
        ],
    )

    with pytest.raises(ValueError, match="reach_accepted actors differ"):
        analyze_reference_mjai(path)


def test_reference_rejects_hora_and_ryukyoku_mixture(tmp_path: Path) -> None:
    path = _write_game(
        tmp_path / "2025010100gm-00a9-0000-0000000c.mjson",
        [
            _round(
                [
                    *_established_reach(),
                    {"type": "hora", "actor": 0, "target": 1},
                    {"type": "ryukyoku"},
                ]
            )
        ],
    )

    with pytest.raises(ValueError, match="coexist"):
        analyze_reference_mjai(path)


def test_reference_rejects_missing_end_kyoku(tmp_path: Path) -> None:
    path = _write_game(
        tmp_path / "2025010100gm-00a9-0000-0000000d.mjson",
        [
            [
                _start_kyoku(),
                *_established_reach(),
                {"type": "ryukyoku"},
            ]
        ],
        include_end_game=False,
    )

    with pytest.raises(ValueError, match="before end_kyoku"):
        analyze_reference_mjai(path)


def test_reference_rejects_unexpected_event_after_dealer_reach_dahai(
    tmp_path: Path,
) -> None:
    path = _write_game(
        tmp_path / "2025010100gm-00a9-0000-0000000e.mjson",
        [
            _round(
                [
                    {"type": "reach", "actor": 0},
                    {"type": "dahai", "actor": 0},
                    {"type": "tsumo", "actor": 1},
                    {"type": "ryukyoku"},
                ]
            )
        ],
    )

    with pytest.raises(ValueError, match="must be followed by"):
        analyze_reference_mjai(path)


def test_existing_side_uses_physical_event_lines(tmp_path: Path) -> None:
    from analysis.validate_dealer_double_riichi_2025 import (
        _analyze_with_existing_implementation,
    )

    path = _write_game(
        tmp_path / "2025010100gm-00a9-0000-0000000f.mjson",
        [
            _round(
                [
                    *_established_reach(),
                    {"type": "ryukyoku"},
                ]
            )
        ],
    )

    result = _analyze_with_existing_implementation(path)

    assert result.candidates[0].start_kyoku_line == 2
    assert result.candidates[0].reach_line == 4


def test_comparison_fails_when_counts_match_but_candidate_keys_differ() -> None:
    reference = _candidate()
    existing = replace(reference, start_kyoku_line=20)

    comparison = compare_candidate_rounds((reference,), (existing,))

    assert comparison.matches is False
    assert comparison.reference_only == (reference,)
    assert comparison.existing_only == (existing,)


def test_comparison_fails_when_candidate_result_differs() -> None:
    reference = _candidate(result="dealer_win")
    existing = replace(reference, result="other_win")

    comparison = compare_candidate_rounds((reference,), (existing,))

    assert comparison.matches is False
    assert comparison.reference_only == ()
    assert comparison.existing_only == ()
    assert comparison.result_mismatches == ((reference, existing),)


def test_reference_module_does_not_import_or_use_production_logic() -> None:
    source = Path(reference_validation.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_modules = {
        "mahjong_analysis.mjai",
        "mahjong_analysis.aggregation",
    }
    forbidden_names = {
        "load_mjai",
        "split_kyoku",
        "is_target_game",
        "filter_east_kyokus",
        "is_dealer_double_riichi",
        "classify_dealer_double_riichi_result",
        "aggregate_dealer_double_riichi",
    }

    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_from = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    used_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }

    assert forbidden_modules.isdisjoint(imported_modules | imported_from)
    assert forbidden_names.isdisjoint(used_names)
