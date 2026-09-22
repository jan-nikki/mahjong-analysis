from __future__ import annotations

import gzip
import sqlite3
from pathlib import Path

import pytest

from mahjong_analysis.kokushi_riichi_outcomes import (
    KyokuOutcome,
    classify_kyoku_outcome,
    enrich_result_document,
    find_accepted_riichi_tj,
)


def _payload(*events: str, compress: bool = False) -> bytes:
    xml = f"<mjloggm>{''.join(events)}</mjloggm>".encode()
    return gzip.compress(xml) if compress else xml


@pytest.mark.parametrize(
    ("terminal", "actor", "expected"),
    [
        (
            '<AGARI who="2" fromWho="2"/>',
            2,
            KyokuOutcome("kokushi_win", True, "tsumo", (2,), 2),
        ),
        (
            '<AGARI who="2" fromWho="0"/>',
            2,
            KyokuOutcome("kokushi_win", True, "ron", (2,), 0),
        ),
        (
            '<AGARI who="1" fromWho="2"/>',
            2,
            KyokuOutcome("other_win", False, None, (1,), None),
        ),
        ("<RYUUKYOKU/>", 2, KyokuOutcome("draw", False, None, (), None)),
    ],
)
def test_classify_kyoku_outcome(
    terminal: str,
    actor: int,
    expected: KyokuOutcome,
) -> None:
    payload = _payload("<INIT/>", terminal, compress=True)
    assert classify_kyoku_outcome(payload, kyoku_index=0, actor=actor) == expected


def test_classify_double_ron_including_target() -> None:
    payload = _payload(
        "<INIT/>",
        '<AGARI who="1" fromWho="3"/>',
        '<AGARI who="2" fromWho="3"/>',
    )
    assert classify_kyoku_outcome(payload, kyoku_index=0, actor=2) == KyokuOutcome(
        "kokushi_win", True, "ron", (1, 2), 3
    )


def test_classify_selects_requested_kyoku() -> None:
    payload = _payload(
        "<INIT/>",
        '<AGARI who="0" fromWho="0"/>',
        "<INIT/>",
        "<RYUUKYOKU/>",
    )
    assert classify_kyoku_outcome(payload, kyoku_index=1, actor=0).outcome == "draw"


def test_classify_normalizes_identical_duplicate_terminal_attribute() -> None:
    payload = _payload(
        "<INIT/>",
        '<RYUUKYOKU owari="x" owari="x"/>',
    )
    assert classify_kyoku_outcome(payload, kyoku_index=0, actor=0).outcome == "draw"


def test_classify_rejects_missing_kyoku() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        classify_kyoku_outcome(
            _payload("<INIT/>", "<RYUUKYOKU/>"),
            kyoku_index=1,
            actor=0,
        )


def test_find_accepted_riichi_tj_counts_events_after_init() -> None:
    payload = _payload(
        "<INIT/>",
        "<T0/>",
        '<REACH who="2" step="1"/>',
        "<G0/>",
        '<REACH who="2" step="2"/>',
        "<RYUUKYOKU/>",
    )
    assert find_accepted_riichi_tj(payload, kyoku_index=0, actor=2) == 3


def test_enrich_result_document_reads_only_target_logs(tmp_path: Path) -> None:
    database = tmp_path / "2020.db"
    connection = sqlite3.connect(database)
    try:
        connection.execute("CREATE TABLE logs (id TEXT PRIMARY KEY, log BLOB)")
        connection.executemany(
            "INSERT INTO logs VALUES (?, ?)",
            [
                (
                    "win",
                    _payload(
                        "<INIT/>",
                        '<REACH who="1" step="2"/>',
                        '<AGARI who="1" fromWho="0"/>',
                    ),
                ),
                (
                    "draw",
                    _payload(
                        "<INIT/>",
                        '<REACH who="3" step="2"/>',
                        "<RYUUKYOKU/>",
                    ),
                ),
                ("unused", b"not xml"),
            ],
        )
        connection.commit()
    finally:
        connection.close()

    source = {
        "summary": {"kokushi_riichis": 2},
        "records": [
            {"year": 2020, "log_id": "win", "kyoku_index": 0, "who": 1, "url": "u1"},
            {"year": 2020, "log_id": "draw", "kyoku_index": 0, "who": 3, "url": "u2"},
        ],
    }
    result = enrich_result_document(source, tmp_path)

    assert result["summary"] == {
        "total": 2,
        "kokushi_wins": 1,
        "not_won": 1,
        "other_wins": 0,
        "draws": 1,
        "ron_wins": 1,
        "tsumo_wins": 0,
        "win_rate": 0.5,
    }
    assert result["records"][0]["won"] is True
    assert result["records"][0]["riichi_tj"] == 0
    assert result["records"][1]["outcome"] == "draw"
