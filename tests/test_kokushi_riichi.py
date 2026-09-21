import gzip
import json
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from mahjong_analysis.kokushi_riichi import (
    KokushiRiichiParseError,
    _decode_concealed_meld_tiles,
    analyze_tenhou_log,
    classify_kokushi_wait,
    result_to_dict,
    scan_kokushi_riichi,
    tenhou_replay_url,
    write_result_json,
)

KOKUSHI_KINDS = (0, 8, 9, 17, 18, 26, *range(27, 34))


def _tile(kind: int, copy: int = 0) -> int:
    return kind * 4 + copy


def _kokushi_13men() -> list[int]:
    return [_tile(kind) for kind in KOKUSHI_KINDS]


def _kokushi_single_waiting_on_chun() -> list[int]:
    return [
        *(_tile(kind) for kind in KOKUSHI_KINDS if kind != 33),
        _tile(27, 1),
    ]


def _init(
    actor_zero: list[int],
    *,
    reserved: tuple[int, ...] = (),
    actor_one: list[int] | None = None,
) -> ET.Element:
    used = set(actor_zero) | set(reserved)
    hands = [actor_zero]
    if actor_one is not None:
        if used & set(actor_one):
            raise AssertionError("test fixture has duplicate physical tiles")
        hands.append(actor_one)
        used.update(actor_one)
    while len(hands) < 4:
        hand = [tile for tile in range(136) if tile not in used][:13]
        used.update(hand)
        hands.append(hand)
    return ET.Element(
        "INIT",
        {
            "seed": "0,0,0,1,0,0",
            "oya": "0",
            **{
                f"hai{actor}": ",".join(str(tile) for tile in hand)
                for actor, hand in enumerate(hands)
            },
        },
    )


def _payload(*events: ET.Element, gzip_: bool = True) -> bytes:
    root = ET.Element("mjloggm", {"ver": "2.3"})
    root.extend(events)
    xml = ET.tostring(root, encoding="utf-8")
    return gzip.compress(xml, mtime=0) if gzip_ else xml


def _event(tag: str, **attributes: object) -> ET.Element:
    return ET.Element(tag, {key: str(value) for key, value in attributes.items()})


def _analyze(payload: bytes):
    return analyze_tenhou_log(
        payload,
        year=2025,
        date="2025-01-02T03:04",
        log_id="2025010203gm-00a9-0000-12345678",
    )


def test_classify_kokushi_13men_and_single_wait() -> None:
    assert classify_kokushi_wait(_kokushi_13men()) == (
        "kokushi_13men",
        ("1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C"),
    )
    assert classify_kokushi_wait(_kokushi_single_waiting_on_chun()) == (
        "kokushi_single",
        ("C",),
    )


def test_classify_kokushi_rejects_non_tenpai_and_bad_physical_tiles() -> None:
    hand = _kokushi_13men()
    hand[-1] = _tile(1)

    assert classify_kokushi_wait(hand) is None
    with pytest.raises(ValueError, match="exactly 13"):
        classify_kokushi_wait(hand[:-1])
    with pytest.raises(ValueError, match="duplicate physical"):
        classify_kokushi_wait([hand[0], *hand[:-1]])


def test_established_first_turn_13men_is_double_riichi() -> None:
    draw = _tile(1)
    result = _analyze(
        _payload(
            _init(_kokushi_13men(), reserved=(draw,)),
            _event(f"T{draw}"),
            _event("REACH", who=0, step=1),
            _event(f"D{draw}"),
            _event("REACH", who=0, step=2, ten="240,250,250,250"),
            _event("RYUUKYOKU"),
        )
    )

    assert result.total_kyokus == 1
    assert result.total_riichis == 1
    assert len(result.records) == 1
    record = result.records[0]
    assert record.kyoku_index == 0
    assert record.who == 0
    assert record.turn == 1
    assert record.reach_type == "double_riichi"
    assert record.wait_type == "kokushi_13men"
    assert len(record.waits) == 13
    assert record.url.endswith("&tw=0&ts=0")


def test_second_discard_single_wait_is_ordinary_riichi() -> None:
    first_draw = _tile(1)
    second_draw = _tile(1, 1)
    result = _analyze(
        _payload(
            _init(
                _kokushi_single_waiting_on_chun(),
                reserved=(first_draw, second_draw),
            ),
            _event(f"T{first_draw}"),
            _event(f"D{first_draw}"),
            _event(f"T{second_draw}"),
            _event("REACH", who=0, step=1),
            _event(f"D{second_draw}"),
            _event("REACH", who=0, step=2),
            _event("RYUUKYOKU"),
        )
    )

    record = result.records[0]
    assert record.turn == 2
    assert record.reach_type == "riichi"
    assert record.wait_type == "kokushi_single"
    assert record.waits == ("C",)


def test_post_discard_reach_sequence_uses_current_thirteen_tiles() -> None:
    draw = _tile(1)
    result = _analyze(
        _payload(
            _init(_kokushi_13men(), reserved=(draw,)),
            _event(f"T{draw}"),
            _event(f"D{draw}"),
            _event("REACH", who=0, step=1),
            _event("REACH", who=0, step=2),
            _event("RYUUKYOKU"),
        )
    )

    assert result.total_riichis == 1
    assert result.post_discard_reach_sequences == 1
    assert result.records[0].turn == 1
    assert result.records[0].reach_type == "double_riichi"
    assert result.records[0].wait_type == "kokushi_13men"


def test_prior_call_cancels_double_riichi_classification() -> None:
    draw = _tile(2)
    actor_one = [_tile(1, copy) for copy in range(4)]
    excluded = set(_kokushi_13men()) | {draw, *actor_one}
    actor_one.extend(tile for tile in range(136) if tile not in excluded)
    actor_one = actor_one[:13]
    result = _analyze(
        _payload(
            _init(_kokushi_13men(), reserved=(draw,), actor_one=actor_one),
            _event("N", who=1, m=_tile(1) << 8),
            _event(f"T{draw}"),
            _event("REACH", who=0, step=1),
            _event(f"D{draw}"),
            _event("REACH", who=0, step=2),
            _event("RYUUKYOKU"),
        )
    )

    assert result.records[0].turn == 1
    assert result.records[0].reach_type == "riichi"


@pytest.mark.parametrize("result_tag", ["AGARI", "RYUUKYOKU"])
def test_unaccepted_declaration_is_not_counted(result_tag: str) -> None:
    draw = _tile(1)
    result = _analyze(
        _payload(
            _init(_kokushi_13men(), reserved=(draw,)),
            _event(f"T{draw}"),
            _event("REACH", who=0, step=1),
            _event(f"D{draw}"),
            _event(result_tag, who=1, fromWho=0),
        )
    )

    assert result.total_riichis == 0
    assert result.records == ()


def test_multiple_kyokus_preserve_zero_based_kyoku_index() -> None:
    draw_one = _tile(1)
    draw_two = _tile(1, 1)
    result = _analyze(
        _payload(
            _init(_kokushi_13men(), reserved=(draw_one,)),
            _event(f"T{draw_one}"),
            _event(f"D{draw_one}"),
            _event("RYUUKYOKU"),
            _init(_kokushi_13men(), reserved=(draw_two,)),
            _event(f"T{draw_two}"),
            _event("REACH", who=0, step=1),
            _event(f"D{draw_two}"),
            _event("REACH", who=0, step=2),
            _event("RYUUKYOKU"),
        )
    )

    assert result.total_kyokus == 2
    assert result.records[0].kyoku_index == 1


def test_malformed_reach_sequence_fails_closed() -> None:
    with pytest.raises(KokushiRiichiParseError, match="acceptance has no declaration"):
        _analyze(
            _payload(
                _init(_kokushi_13men()),
                _event("REACH", who=0, step=2),
            )
        )


def test_identical_duplicate_terminal_attribute_is_normalized_and_counted() -> None:
    xml = _payload(
        _init(_kokushi_13men()),
        _event("RYUUKYOKU"),
        gzip_=False,
    )
    malformed = xml.replace(
        b"<RYUUKYOKU />",
        b'<RYUUKYOKU owari="x" owari="x" />',
    )

    result = _analyze(gzip.compress(malformed, mtime=0))

    assert result.normalized_terminal_attributes is True
    assert result.total_kyokus == 1


def test_conflicting_or_state_bearing_duplicate_attributes_remain_fatal() -> None:
    terminal_xml = _payload(
        _init(_kokushi_13men()),
        _event("RYUUKYOKU"),
        gzip_=False,
    ).replace(
        b"<RYUUKYOKU />",
        b'<RYUUKYOKU owari="x" owari="y" />',
    )
    with pytest.raises(KokushiRiichiParseError, match="conflicting duplicate"):
        _analyze(terminal_xml)

    init_xml = _payload(_init(_kokushi_13men()), gzip_=False).replace(
        b'<INIT seed="0,0,0,1,0,0"',
        b'<INIT seed="0,0,0,1,0,0" seed="0,0,0,1,0,0"',
    )
    with pytest.raises(KokushiRiichiParseError, match="invalid XML"):
        _analyze(init_xml)


def _create_database(path: Path, rows: list[tuple[object, ...]]) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE logs (
            id TEXT PRIMARY KEY,
            date TEXT NOT NULL,
            num_players INTEGER NOT NULL,
            is_tonpu INTEGER NOT NULL,
            is_processed INTEGER NOT NULL,
            was_error INTEGER NOT NULL,
            log BLOB,
            game_type TEXT
        ) WITHOUT ROWID
        """
    )
    connection.executemany("INSERT INTO logs VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    connection.commit()
    connection.close()


def test_database_scan_filters_scope_and_serializes_result(tmp_path: Path) -> None:
    draw = _tile(1)
    payload = _payload(
        _init(_kokushi_13men(), reserved=(draw,)),
        _event(f"T{draw}"),
        _event("REACH", who=0, step=1),
        _event(f"D{draw}"),
        _event("REACH", who=0, step=2),
        _event("RYUUKYOKU"),
    )
    rows = [
        ("target", "2025-01-01T00:00", 4, 0, 1, 0, payload, "houou"),
        ("tonpu", "2025-01-01T00:01", 4, 1, 1, 0, payload, "houou"),
        ("sanma", "2025-01-01T00:02", 3, 0, 1, 0, payload, "houou"),
        ("other-room", "2025-01-01T00:03", 4, 0, 1, 0, payload, "joukyuu"),
    ]
    _create_database(tmp_path / "2025.db", rows)

    result = scan_kokushi_riichi(tmp_path, years=(2025,))
    document = result_to_dict(result)
    output = tmp_path / "nested" / "result.json"
    write_result_json(result, output)

    assert document["summary"] == {
        "available_logs": 1,
        "eligible_logs": 1,
        "kokushi_13men": 1,
        "kokushi_riichis": 1,
        "normalized_xml_logs": 0,
        "post_discard_reach_sequences": 0,
        "scanned_logs": 1,
        "total_kyokus": 1,
        "total_riichis": 1,
        "unavailable_logs": 0,
    }
    assert document["records"][0]["log_id"] == "target"
    assert json.loads(output.read_text(encoding="utf-8")) == document
    assert not output.with_name("result.json.part").exists()


def test_database_scan_reports_missing_target_payload(tmp_path: Path) -> None:
    _create_database(
        tmp_path / "2025.db",
        [("missing", "2025-01-01T00:00", 4, 0, 0, 1, None, "houou")],
    )

    result = scan_kokushi_riichi(tmp_path, years=(2025,))

    assert result.yearly[0].eligible_logs == 1
    assert result.yearly[0].available_logs == 0
    assert result.yearly[0].scanned_logs == 0
    assert result.yearly[0].unavailable_logs == 1


def test_tenhou_replay_url_validates_coordinates() -> None:
    assert tenhou_replay_url("log-id", 2, 7) == (
        "https://tenhou.net/5/?log=log-id&tw=2&ts=7"
    )
    with pytest.raises(ValueError, match="who"):
        tenhou_replay_url("log-id", 4, 0)


@pytest.mark.parametrize("missing", KOKUSHI_KINDS)
def test_all_single_waits_and_all_pair_choices(missing: int) -> None:
    names = ("1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C")
    for pair in KOKUSHI_KINDS:
        if pair != missing:
            tiles = [_tile(kind) for kind in KOKUSHI_KINDS if kind != missing]
            tiles.append(_tile(pair, 1))
            assert classify_kokushi_wait(tiles) == (
                "kokushi_single",
                (names[KOKUSHI_KINDS.index(missing)],),
            )


def test_fast_filter_matches_full_solver_for_random_and_near_kokushi_hands() -> None:
    import random

    from mahjong_analysis.hand_waits import calculate_hand_waits
    from mahjong_analysis.tiles import index_to_tile

    randomizer = random.Random(20260920)
    hands = [randomizer.sample(range(136), 13) for _ in range(100)]
    orphans = [kind * 4 + copy for kind in KOKUSHI_KINDS for copy in range(4)]
    hands += [randomizer.sample(orphans, 13) for _ in range(100)]
    for hand in hands:
        reference = calculate_hand_waits(index_to_tile(tile // 4) for tile in hand)
        details = [d for d in reference.wait_details if d.hand_type == "kokushi"]
        actual = classify_kokushi_wait(hand)
        if details:
            assert actual == (
                details[0].wait_shape,
                tuple(d.wait_tile for d in details),
            )
        else:
            assert actual is None


@pytest.mark.parametrize("discard_tag", ["D4", "d4", "d"])
def test_explicit_and_implicit_tsumogiri(discard_tag: str) -> None:
    result = _analyze(
        _payload(
            _init(_kokushi_13men(), reserved=(4,)),
            _event("T4"),
            _event("REACH", who=0, step=1),
            _event(discard_tag),
            _event("REACH", who=0, step=2),
            _event("RYUUKYOKU"),
        )
    )
    assert result.records[0].wait_type == "kokushi_13men"


def test_hand_discard_changes_thirteen_sided_to_single_wait() -> None:
    result = _analyze(
        _payload(
            _init(_kokushi_13men(), reserved=(1,)),
            _event("T1"),
            _event("REACH", who=0, step=1),
            _event("D132"),
            _event("REACH", who=0, step=2),
            _event("RYUUKYOKU"),
        )
    )
    assert result.records[0].wait_type == "kokushi_single"
    assert result.records[0].waits == ("C",)


def test_ordinary_riichi_in_denominator_and_two_players_in_one_kyoku() -> None:
    ordinary = [4, 8, 12, 16, 20, 24, 40, 44, 48, 52, 56, 60, 61]
    result = _analyze(
        _payload(
            _init(_kokushi_13men(), reserved=(5, 62), actor_one=ordinary),
            _event("T5"),
            _event("REACH", who=0, step=1),
            _event("D5"),
            _event("REACH", who=0, step=2),
            _event("U62"),
            _event("REACH", who=1, step=1),
            _event("E62"),
            _event("REACH", who=1, step=2),
            _event("RYUUKYOKU"),
        )
    )
    assert result.total_riichis == 2
    assert len(result.records) == 1


def test_closed_kan_riichi_counts_in_denominator_but_cannot_be_kokushi() -> None:
    hand = [4, 5, 6, 7, 40, 44, 48, 76, 80, 84, 108, 109, 110]
    result = _analyze(
        _payload(
            _init(hand, reserved=(16, 20)),
            _event("T16"),
            _event("N", who=0, m=1024),
            _event("DORA", hai=24),
            _event("T20"),
            _event("REACH", who=0, step=1),
            _event("D20"),
            _event("REACH", who=0, step=2),
            _event("RYUUKYOKU"),
        )
    )
    assert result.total_riichis == 1
    assert not result.records


@pytest.mark.parametrize(
    ("code", "consumed"),
    [
        (31, (4, 8)),  # chi 123m, called 1m copy 3
        (6763, (16, 18)),  # pon 5m, called copy 1, unused copy 3
        (6259, (19,)),  # added kan 5m copy 3
        (1024, (4, 5, 6, 7)),  # concealed kan 2m
        (1537, (4, 5, 7)),  # open kan 2m, called copy 2
    ],
)
def test_decode_consumed_physical_tiles_for_every_meld(
    code: int, consumed: tuple
) -> None:
    assert _decode_concealed_meld_tiles(code) == consumed


def test_post_discard_unaccepted_reach_does_not_increment_accepted_counter() -> None:
    result = _analyze(
        _payload(
            _init(_kokushi_13men(), reserved=(4,)),
            _event("T4"),
            _event("D4"),
            _event("REACH", who=0, step=1),
            _event("AGARI", who=1, fromWho=0),
        )
    )
    assert result.total_riichis == result.post_discard_reach_sequences == 0


def test_rejects_missing_result_and_intervening_draw_before_acceptance() -> None:
    with pytest.raises(KokushiRiichiParseError, match="before kyoku result"):
        _analyze(_payload(_init(_kokushi_13men())))
    with pytest.raises(KokushiRiichiParseError, match="before reach acceptance"):
        _analyze(
            _payload(
                _init(_kokushi_13men(), reserved=(4, 5)),
                _event("T4"),
                _event("REACH", who=0, step=1),
                _event("D4"),
                _event("U5"),
                _event("REACH", who=0, step=2),
                _event("RYUUKYOKU"),
            )
        )


def test_duplicate_same_actor_reach_is_rejected() -> None:
    sequence = [
        _event("T4"),
        _event("REACH", who=0, step=1),
        _event("D4"),
        _event("REACH", who=0, step=2),
    ]
    with pytest.raises(KokushiRiichiParseError, match="duplicate riichi"):
        _analyze(
            _payload(
                _init(_kokushi_13men(), reserved=(4,)),
                *sequence,
                *sequence,
                _event("RYUUKYOKU"),
            )
        )


def test_legacy_database_schema_and_parallel_results_match(tmp_path: Path) -> None:
    xml = _payload(_init(_kokushi_13men()), _event("RYUUKYOKU"))
    for year in (2009, 2025):
        _create_database(
            tmp_path / f"{year}.db",
            [
                ("b", f"{year}-01-01T00:00", 4, 0, 0, 1, xml, "houou"),
                ("a", f"{year}-01-01T00:00", 4, 0, 1, 0, xml, "houou"),
            ],
        )
    with sqlite3.connect(tmp_path / "2009.db") as connection:
        connection.execute("ALTER TABLE logs DROP COLUMN game_type")
    sequential = scan_kokushi_riichi(tmp_path, years=(2025, 2009), max_logs_per_year=1)
    completed = []
    parallel = scan_kokushi_riichi(
        tmp_path,
        years=(2009, 2025),
        workers=2,
        max_logs_per_year=1,
        on_year_complete=completed.append,
    )
    assert sequential == parallel
    assert len(completed) == 2
    assert all(item.scanned_logs == 1 for item in completed)


def test_malformed_database_payload_errors_include_year_and_id(tmp_path: Path) -> None:
    for year in (2009, 2025):
        _create_database(
            tmp_path / f"{year}.db",
            [
                ("broken", f"{year}-01-01T00:00", 4, 0, 1, 0, b"invalid", "houou"),
            ],
        )
    with pytest.raises(ValueError) as caught:
        scan_kokushi_riichi(tmp_path, years=(2009, 2025), workers=2)
    assert "2009.db, log_id=broken" in str(caught.value)
    assert "2025.db, log_id=broken" in str(caught.value)


def test_open_players_later_bad_kakan_does_not_hide_other_players_riichi() -> None:
    # Archive regression: U88 E88 N(kakan of 88). The player already had a pon.
    # Reproduce with 2m IDs; only the closed player's reconstruction matters here.
    actor_one = [4, 5, *range(12, 23)]
    result = _analyze(
        _payload(
            _init(_kokushi_13men(), reserved=(6, 7, 8), actor_one=actor_one),
            _event("N", who=1, m=2667),
            _event("U7"),
            _event("E7"),
            _event("N", who=1, m=2675),
            _event("T8"),
            _event("REACH", who=0, step=1),
            _event("D8"),
            _event("REACH", who=0, step=2),
            _event("RYUUKYOKU"),
        )
    )
    assert result.total_riichis == 1
    assert len(result.records) == 1
    assert result.records[0].reach_type == "riichi"


def test_acceptance_with_different_actor_is_rejected() -> None:
    with pytest.raises(KokushiRiichiParseError, match="acceptance actor"):
        _analyze(
            _payload(
                _init(_kokushi_13men(), reserved=(4,)),
                _event("T4"),
                _event("REACH", who=0, step=1),
                _event("D4"),
                _event("REACH", who=1, step=2),
                _event("RYUUKYOKU"),
            )
        )


def test_open_hand_reach_remains_an_error() -> None:
    actor_one = [4, 5, *range(12, 23)]
    with pytest.raises(KokushiRiichiParseError, match="open melds"):
        _analyze(
            _payload(
                _init(_kokushi_13men(), reserved=(6,), actor_one=actor_one),
                _event("N", who=1, m=2667),
                _event("REACH", who=1, step=1),
                _event("E12"),
                _event("REACH", who=1, step=2),
                _event("RYUUKYOKU"),
            )
        )


@pytest.mark.parametrize("hand", [_kokushi_13men(), _kokushi_single_waiting_on_chun()])
@pytest.mark.parametrize("post_discard", [False, True])
def test_independent_reference_recovers_identical_hit(
    hand: list[int], post_discard: bool
) -> None:
    from dataclasses import asdict

    from analysis.validate_kokushi_riichi import reference_log, targeted_reference_log

    reach = _event("REACH", who=0, step=1)
    discard = _event("D4")
    payload = _payload(
        _init(hand, reserved=(4,)),
        _event("T4"),
        *((discard, reach) if post_discard else (reach, discard)),
        _event("REACH", who=0, step=2),
        _event("RYUUKYOKU"),
    )
    production = _analyze(payload)
    reference = reference_log(
        payload, 2025, "2025-01-02T03:04", "2025010203gm-00a9-0000-12345678"
    )
    assert (
        targeted_reference_log(
            payload, 2025, "2025-01-02T03:04", "2025010203gm-00a9-0000-12345678"
        )
        == reference
    )
    expected = json.loads(json.dumps(asdict(production.records[0])))
    assert reference["records"] == [expected]
    assert reference["total_riichis"] == production.total_riichis
    assert reference["audit"][0]["concealed_tile_ids"] == sorted(hand)


def test_independent_comparison_detects_missing_hits_and_changed_metadata(
    tmp_path: Path,
) -> None:
    from copy import deepcopy

    from analysis.validate_kokushi_riichi import _year, compare_annual

    payload = _payload(
        _init(_kokushi_13men(), reserved=(4,)),
        _event("T4"),
        _event("REACH", who=0, step=1),
        _event("D4"),
        _event("REACH", who=0, step=2),
        _event("RYUUKYOKU"),
    )
    _create_database(
        tmp_path / "2025.db", [("test", "2025-01-01", 4, 0, 1, 0, payload, "houou")]
    )
    production = result_to_dict(scan_kokushi_riichi(tmp_path, years=(2025,)))
    independent = [_year((str(tmp_path), 2025, None))]
    assert (
        compare_annual(json.dumps(production).encode(), independent)["status"] == "PASS"
    )
    for field, value in (("waits", ["C"]), ("turn", 9), ("reach_type", "riichi")):
        changed = deepcopy(production)
        changed["records"][0][field] = value
        assert (
            compare_annual(json.dumps(changed).encode(), independent)["status"]
            == "FAIL"
        )
    changed = deepcopy(production)
    changed["records"] = []
    assert compare_annual(json.dumps(changed).encode(), independent)["status"] == "FAIL"
    changed = deepcopy(production)
    changed["yearly"][0]["total_riichis"] += 1
    assert compare_annual(json.dumps(changed).encode(), independent)["status"] == "FAIL"
