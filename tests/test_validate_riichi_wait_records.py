import json
from pathlib import Path

import pytest

from analysis.validate_riichi_wait_records import (
    CATEGORY_ORDER,
    classify_audit_categories,
    collect_audit_samples,
    main,
    render_audit_report,
)
from mahjong_analysis.hand_waits import FixedMeld
from mahjong_analysis.riichi import ActorDiscard, EstablishedRiichi
from mahjong_analysis.riichi_wait_records import (
    RiichiWaitRecord,
    build_riichi_wait_record,
)
from mahjong_analysis.tiles import normalize_tile


def expand_hand(hand: str) -> list[str]:
    tiles: list[str] = []
    for group in hand.split():
        if group in {"5mr", "5pr", "5sr"}:
            tiles.append(group)
        elif group[-1] in "mps":
            tiles.extend(f"{rank}{group[-1]}" for rank in group[:-1])
        else:
            tiles.extend(group)
    return tiles


def make_record(
    hand: str,
    *,
    discard_number: int = 7,
    declaration_tile: str = "9s",
    fixed_melds: tuple[FixedMeld, ...] = (),
) -> RiichiWaitRecord:
    declaration_event_index = 101
    discards = tuple(
        ActorDiscard(
            discard_number=number,
            tile="9m",
            tile_kind="9m",
            tsumogiri=True,
            event_index=number,
            is_riichi_declaration=False,
        )
        for number in range(1, discard_number)
    ) + (
        ActorDiscard(
            discard_number=discard_number,
            tile=declaration_tile,
            tile_kind=normalize_tile(declaration_tile),
            tsumogiri=True,
            event_index=declaration_event_index,
            is_riichi_declaration=True,
        ),
    )
    established = EstablishedRiichi(
        actor=1,
        reach_event_index=100,
        declaration_dahai_event_index=declaration_event_index,
        reach_accepted_event_index=102,
        riichi_discard_number=discard_number,
        riichi_declaration_tile=declaration_tile,
        riichi_declaration_tile_kind=normalize_tile(declaration_tile),
        concealed_tiles_after_discard=tuple(expand_hand(hand)),
        fixed_melds=fixed_melds,
        actor_discards_before_riichi=discards,
    )
    return build_riichi_wait_record(established)


def write_game(path: Path, *, bakaze: str = "E") -> None:
    hand = expand_hand("123m 123p 789p EE 45s")
    events = [
        {"type": "start_game", "aka_flag": True},
        {
            "type": "start_kyoku",
            "bakaze": bakaze,
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "tehais": [list(hand) for _ in range(4)],
        },
        {"type": "tsumo", "actor": 1, "pai": "9s"},
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "9s", "tsumogiri": True},
        {"type": "reach_accepted", "actor": 1},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(event)}\n" for event in events),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    (
        "hand",
        "discard_number",
        "declaration_tile",
        "fixed_melds",
        "expected_categories",
    ),
    [
        (
            "123m 123p 789p EE 45s",
            3,
            "9s",
            (),
            ("pure_ryanmen", "early_riichi_1_to_6"),
        ),
        ("123m 123p 789p EE 46s", 7, "9s", (), ("kanchan",)),
        ("123m 456m 789p EE 12s", 7, "9s", (), ("penchan",)),
        ("123m 456m 789m 55p 77s", 7, "9s", (), ("shanpon",)),
        ("123m 456m 789m 123p 5s", 7, "9s", (), ("tanki",)),
        (
            "34567m 123p 789p EE",
            7,
            "9s",
            (),
            ("multiwait_3_plus", "contains_ryanmen_not_pure"),
        ),
        (
            "2345678m 123p 789p",
            7,
            "9s",
            (),
            ("tanki", "multiwait_3_plus"),
        ),
        (
            "2333456m 123p 789p",
            7,
            "9s",
            (),
            ("tanki", "multiwait_3_plus", "contains_ryanmen_not_pure"),
        ),
        (
            "123m 123p 789p EE 45s",
            7,
            "5mr",
            (),
            ("pure_ryanmen", "red_five"),
        ),
        (
            "123p 789p EE 45s",
            7,
            "9s",
            (FixedMeld(("9m", "9m", "9m", "9m")),),
            ("pure_ryanmen", "ankan"),
        ),
        (
            "123m 123p 789p EE 45s",
            13,
            "9s",
            (),
            ("pure_ryanmen", "late_riichi_13_plus"),
        ),
    ],
    ids=CATEGORY_ORDER,
)
def test_classifies_audit_categories_in_canonical_order(
    hand: str,
    discard_number: int,
    declaration_tile: str,
    fixed_melds: tuple[FixedMeld, ...],
    expected_categories: tuple[str, ...],
) -> None:
    record = make_record(
        hand,
        discard_number=discard_number,
        declaration_tile=declaration_tile,
        fixed_melds=fixed_melds,
    )

    assert classify_audit_categories(record) == expected_categories


def test_collects_source_locatable_sample_and_renders_complete_events(
    tmp_path: Path,
) -> None:
    path = tmp_path / "2025010100gm-00a9-0000-00000001.mjson"
    write_game(path)

    report = collect_audit_samples(
        [path],
        max_files=1,
        samples_per_category=1,
        context_before=2,
        context_after=1,
    )

    assert (
        report.scanned_files,
        report.target_games,
        report.east_kyokus,
        report.established_riichis,
    ) == (1, 1, 1, 1)
    sample = report.samples_for("pure_ryanmen")[0]
    assert sample.filename == path.name
    assert sample.start_kyoku_line == 2
    assert sample.is_dealer is False
    assert sample.reach_event.game_line_number == 4
    assert sample.declaration_dahai_event.game_line_number == 5
    assert sample.reach_accepted_event.game_line_number == 6
    assert sample.actor_discards[0].game_line_number == 5
    assert tuple(event.game_line_number for event in sample.context_events) == (
        2,
        3,
        4,
        5,
        6,
        7,
    )
    assert sample.record.concealed_tiles_after_discard == tuple(
        expand_hand("123m 123p 789p EE 45s")
    )
    assert sample.record.wait_tiles == ("3s", "6s")

    rendered = render_audit_report(report)
    assert f"source_path: {path.resolve()}" in rendered
    assert "reach_event: game_line=4" in rendered
    assert '{"type":"reach","actor":1}' in rendered
    assert "declaration_dahai_event: game_line=5" in rendered
    assert "reach_accepted_event: game_line=6" in rendered
    assert "riichi_declaration_tile_kind: 9s" in rendered
    assert 'wait_tiles: ["3s","6s"]' in rendered
    assert "wait_tile_count: 2" in rendered
    assert "## ankan (暗槓ありリーチ)\n\nsamples: 0" in rendered


def test_collection_sorts_paths_before_applying_file_limit(tmp_path: Path) -> None:
    first = tmp_path / "2025010100gm-00a9-0000-00000001.mjson"
    second = tmp_path / "2025010100gm-00a9-0000-00000002.mjson"
    write_game(first)
    write_game(second)

    report = collect_audit_samples([second, first], max_files=1)

    assert report.scanned_files == 1
    assert report.samples_for("pure_ryanmen")[0].filename == first.name


def test_collection_applies_target_game_and_east_kyoku_filters(
    tmp_path: Path,
) -> None:
    non_target = tmp_path / "2025010100gm-00e1-0000-00000001.mjson"
    south_target = tmp_path / "2025010100gm-00a9-0000-00000002.mjson"
    write_game(non_target)
    write_game(south_target, bakaze="S")

    report = collect_audit_samples([non_target, south_target], max_files=2)

    assert report.scanned_files == 2
    assert report.target_games == 1
    assert report.east_kyokus == 0
    assert report.established_riichis == 0
    assert all(not category.samples for category in report.categories)


@pytest.mark.parametrize(
    "arguments",
    [
        {"max_files": 0},
        {"samples_per_category": 4},
        {"context_before": -1},
        {"context_after": -1},
    ],
)
def test_collection_rejects_invalid_limits(arguments: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        collect_audit_samples([], **arguments)


def test_main_prints_report_without_writing_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "2025" / "2025010100gm-00a9-0000-00000001.mjson"
    write_game(path)

    result = main(
        [
            "--raw-root",
            str(tmp_path),
            "--max-files",
            "1",
            "--context-before",
            "1",
            "--context-after",
            "1",
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "scanned_files: 1" in output
    assert "source_path:" in output
    assert "mjai_event_context:" in output
