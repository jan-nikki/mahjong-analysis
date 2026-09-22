import gzip
import json
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

from mahjong_analysis.kokushi_tenpai_decisions import (
    analyze_tenhou_log,
    result_to_dict,
    scan_kokushi_tenpai_decisions,
    write_result_json,
)

KOKUSHI_KINDS = (0, 8, 9, 17, 18, 26, *range(27, 34))


def _tile(kind: int, copy: int = 0) -> int:
    return kind * 4 + copy


def _kokushi_13men() -> list[int]:
    return [_tile(kind) for kind in KOKUSHI_KINDS]


def _init(
    actor_zero: list[int],
    *,
    reserved: tuple[int, ...] = (),
    scores: str = "250,250,250,250",
) -> ET.Element:
    used = set(actor_zero) | set(reserved)
    hands = [actor_zero]
    while len(hands) < 4:
        hand = [tile for tile in range(136) if tile not in used][:13]
        used.update(hand)
        hands.append(hand)
    return ET.Element(
        "INIT",
        {
            "seed": "0,0,0,1,0,4",
            "oya": "0",
            "ten": scores,
            **{
                f"hai{actor}": ",".join(str(tile) for tile in hand)
                for actor, hand in enumerate(hands)
            },
        },
    )


def _event(tag: str, **attributes: object) -> ET.Element:
    return ET.Element(tag, {key: str(value) for key, value in attributes.items()})


def _payload(*events: ET.Element) -> bytes:
    root = ET.Element("mjloggm", {"ver": "2.3"})
    root.extend(events)
    return gzip.compress(ET.tostring(root, encoding="utf-8"), mtime=0)


def _analyze(payload: bytes):
    return analyze_tenhou_log(
        payload,
        year=2025,
        date="2025-01-02T03:04",
        log_id="2025010203gm-00a9-0000-12345678",
    )


def test_immediate_established_riichi_and_covariates() -> None:
    result = _analyze(
        _payload(
            _init(_kokushi_13men(), reserved=(5,)),
            _event("T5"),
            _event("REACH", who=0, step=1),
            _event("D5"),
            _event("REACH", who=0, step=2, ten="240,250,250,250"),
            _event("RYUUKYOKU", sc="240,0,250,0,250,0,250,0"),
        )
    )
    record = result.records[0]
    assert record.strategy == "immediate_riichi"
    assert record.riichi_declared is record.riichi_established is True
    assert record.riichi_reach_type == "double_riichi"
    assert record.wait_group == "thirteen_sided"
    assert record.scores == (25000, 25000, 25000, 25000)
    assert record.point_delta == -1000
    assert record.outcome == "draw"
    assert record.riichi_eligible is True
    assert record.url.endswith("&tj=2")


def test_dama_tsumo_and_delayed_riichi_are_separate_policies() -> None:
    dama = _analyze(
        _payload(
            _init(_kokushi_13men(), reserved=(5, 132)),
            _event("T5"),
            _event("D5"),
            _event("U132"),
            _event("E132"),
            _event("AGARI", who=0, fromWho=1),
        )
    ).records[0]
    assert dama.strategy == "dama"
    assert dama.won is True
    assert dama.win_method == "ron"

    delayed = _analyze(
        _payload(
            _init(_kokushi_13men(), reserved=(5, 6)),
            _event("T5"),
            _event("D5"),
            _event("T6"),
            _event("REACH", who=0, step=1),
            _event("D6"),
            _event("REACH", who=0, step=2),
            _event("RYUUKYOKU"),
        )
    ).records[0]
    assert delayed.strategy == "delayed_riichi"
    assert delayed.riichi_delay_turns == 1
    assert delayed.riichi_established is True


def test_failed_declaration_remains_in_immediate_riichi_choice() -> None:
    record = _analyze(
        _payload(
            _init(_kokushi_13men(), reserved=(5,)),
            _event("T5"),
            _event("REACH", who=0, step=1),
            _event("D5"),
            _event("AGARI", who=1, fromWho=0),
        )
    ).records[0]
    assert record.strategy == "immediate_riichi"
    assert record.riichi_declared is True
    assert record.riichi_established is False
    assert record.outcome == "deal_in"


def test_first_tenpai_only_and_wait_availability() -> None:
    record = _analyze(
        _payload(
            _init(_kokushi_13men(), reserved=(5, 6)),
            _event("T5"),
            _event("D5"),
            _event("T6"),
            _event("D6"),
            _event("RYUUKYOKU"),
        )
    ).records[0]
    assert (
        len(
            _analyze(
                _payload(
                    _init(_kokushi_13men(), reserved=(5,)),
                    _event("T5"),
                    _event("D5"),
                    _event("RYUUKYOKU"),
                )
            ).records
        )
        == 1
    )
    assert record.turn == 1
    assert record.visible_wait_counts[0] == 0
    assert record.unseen_wait_counts[0] == 3
    assert record.furiten is False


def _create_database(path: Path, rows: list[tuple[object, ...]]) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE logs (
            id TEXT PRIMARY KEY, date TEXT NOT NULL, num_players INTEGER NOT NULL,
            is_tonpu INTEGER NOT NULL, is_processed INTEGER NOT NULL,
            was_error INTEGER NOT NULL, log BLOB, game_type TEXT
        ) WITHOUT ROWID
        """
    )
    connection.executemany("INSERT INTO logs VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    connection.commit()
    connection.close()


def test_database_scope_summary_and_atomic_json(tmp_path: Path) -> None:
    payload = _payload(
        _init(_kokushi_13men(), reserved=(5,)),
        _event("T5"),
        _event("D5"),
        _event("RYUUKYOKU"),
    )
    _create_database(
        tmp_path / "2025.db",
        [
            ("target", "2025-01-01", 4, 0, 1, 0, payload, "houou"),
            ("tonpu", "2025-01-01", 4, 1, 1, 0, payload, "houou"),
            ("other", "2025-01-01", 4, 0, 1, 0, payload, "joukyuu"),
        ],
    )
    result = scan_kokushi_tenpai_decisions(tmp_path, years=(2025,))
    document = result_to_dict(result)
    assert document["summary"]["decisions"] == 1
    assert document["summary"]["strategies"]["dama"]["count"] == 1
    output = tmp_path / "nested" / "result.json"
    write_result_json(result, output)
    assert json.loads(output.read_text(encoding="utf-8")) == document
    assert not output.with_name("result.json.part").exists()
