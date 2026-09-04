from pathlib import Path

from mahjong_analysis.mjai import load_mjai


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
