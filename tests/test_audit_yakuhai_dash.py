from __future__ import annotations

import gzip
from copy import deepcopy

import pytest

from analysis.analyze_yakuhai_dash import YearTask, build_document, scan_year
from analysis.audit_yakuhai_dash import (
    _deal_facts,
    _read_events,
    audit_blind_logs,
    audit_document,
    compare_v2_v3_aggregates,
    reference_shanten,
)
from test_analyze_yakuhai_dash import artificial_game, write_game
from test_yakuhai_dash import make_hands, make_kyoku


@pytest.fixture
def audit_input(tmp_path):
    data_root = tmp_path / "raw"
    path = data_root / "2025" / "2025010100gm-00a9-0000-1234abcd.mjson"
    write_game(path, artificial_game())
    return data_root, build_document((scan_year(YearTask(2025, str(data_root), None, 3)),))


@pytest.mark.parametrize("mutation", [
    "empty_years", "empty_samples", "missing_category", "duplicate_category",
    "missing_field", "duplicate_sample", "wrong_population", "disabled_samples",
])
def test_saved_audit_rejects_incomplete_coverage(audit_input, mutation):
    data_root, document = audit_input
    groups = document["audit_samples_by_year"][0]["categories"]
    populated = next(group for group in groups if group["samples"])
    if mutation == "empty_years":
        document["audit_samples_by_year"] = []
    elif mutation == "empty_samples":
        populated["samples"] = []
    elif mutation == "missing_category":
        groups.pop()
    elif mutation == "duplicate_category":
        groups.append(deepcopy(groups[0]))
    elif mutation == "missing_field":
        del populated["samples"][0]["initial_hand"]
    elif mutation == "duplicate_sample":
        populated["samples"].append(deepcopy(populated["samples"][0]))
    elif mutation == "wrong_population":
        populated["population_count"] += 1
    elif mutation == "disabled_samples":
        document["metadata"]["execution"]["audit_samples_per_category"] = 0
    result = audit_document(document, data_root)
    assert result["status"] == "fail"
    assert result["failures"][0]["field"] == "audit_coverage"


def test_empty_legacy_document_no_longer_passes(tmp_path):
    document = {"metadata": {"schema_version": 2}, "audit_samples_by_year": []}
    assert audit_document(document, tmp_path)["status"] == "fail"


@pytest.mark.parametrize("has_discard", [False, True])
def test_opening_audit_rejects_actor_without_initial_singletons(tmp_path, has_discard):
    hands = make_hands({0: ["P"]})
    events = []
    if has_discard:
        for actor in (0, 1):
            events.extend([
                {"type": "tsumo", "actor": actor, "pai": "9s"},
                {"type": "dahai", "actor": actor, "pai": "9s", "tsumogiri": True},
            ])
    kyoku = make_kyoku(hands, events)
    path = tmp_path / "2025" / "2025010100gm-00a9-0000-1234abcd.mjson"
    write_game(path, kyoku)
    document = build_document((scan_year(YearTask(2025, str(tmp_path), None, 3)),))
    category = "first_discard_without_dash" if has_discard else "no_first_discard"
    sample = next(g["samples"][0] for g in document["audit_samples_by_year"][0]["categories"]
                  if g["category"] == category)
    # All source facts are correct, but actor 1 is outside the opening cohort.
    other_deal = _deal_facts(kyoku[0], 1)
    assert other_deal["singleton_kind_count"] == 0
    sample["actor"] = 1
    for name in ("role", "initial_hand", "shanten", "dora_han"):
        sample[name] = other_deal[name]
    if has_discard:
        sample["first_discard_line"] = sample["start_kyoku_line"] + 4
    result = audit_document(document, tmp_path)
    assert result["status"] == "fail"
    assert any(f["field"] == "category_membership" for f in result["failures"])


def test_legacy_v2_audit_keeps_strict_field_and_year_checks(audit_input):
    data_root, document = audit_input
    document["metadata"]["schema_version"] = 2
    del document["metadata"]["execution"]["audit_samples_per_category"]
    groups = document["audit_samples_by_year"][0]["categories"]
    groups[:] = [g for g in groups if g["category"] not in {
        "self_pair_pon_shape", "self_pair_without_pon_shape",
    }]
    result = audit_document(document, data_root)
    assert result["status"] == "pass"
    assert result["coverage_count_policy"].startswith("legacy_v2")
    sample = next(g["samples"][0] for g in groups if g["category"] == "strict_dash")
    del sample["singleton_discard_event_index"]
    assert audit_document(document, data_root)["status"] == "fail"


@pytest.mark.parametrize("mode, retained, pon_shape", [
    ("retain", True, True), ("break", False, False),
    ("reach", True, False), ("no_discard", None, None),
])
def test_independent_audit_verifies_pair_followup(tmp_path, mode, retained, pon_shape):
    hands = make_hands({0: ["P"]})
    events = [{"type": "tsumo", "actor": 0, "pai": "P"}]
    if mode == "reach":
        events.append({"type": "reach", "actor": 0})
    if mode != "no_discard":
        events.append({
            "type": "dahai", "actor": 0,
            "pai": "P" if mode == "break" else hands[0][1],
            "tsumogiri": mode == "break",
        })
    path = tmp_path / "2025" / "2025010100gm-00a9-0000-1234abcd.mjson"
    write_game(path, make_kyoku(hands, events))
    document = build_document((scan_year(YearTask(2025, str(tmp_path), None, 3)),))
    group = next(g for g in document["audit_samples_by_year"][0]["categories"]
                 if g["category"] == "self_pair_draw")
    sample = group["samples"][0]
    assert sample["self_pair_retained_after_discard"] is retained
    assert sample["self_pair_pon_shape_after_discard"] is pon_shape
    assert audit_document(document, tmp_path)["status"] == "pass"
    assert audit_blind_logs(tmp_path, [2025], 1)["status"] == "pass"
    sample["self_pair_pon_shape_after_discard"] = not pon_shape
    result = audit_document(document, tmp_path)
    assert result["status"] == "fail"
    assert any(f["field"] == "self_pair_pon_shape_after_discard" for f in result["failures"])


def test_blind_audit_rejects_empty_years_and_no_east_rounds(tmp_path):
    with pytest.raises(ValueError, match="nonempty"):
        audit_blind_logs(tmp_path, [])
    events = artificial_game()
    for event in events:
        if event["type"] == "start_kyoku":
            event["bakaze"] = "S"
    path = tmp_path / "2025" / "2025010100gm-00a9-0000-1234abcd.mjson"
    write_game(path, events)
    result = audit_blind_logs(tmp_path, [2025], 1)
    assert result["status"] == "fail"
    assert result["east_kyokus"] == 0


def test_blind_audit_rejects_missing_production_fields(audit_input, monkeypatch):
    from mahjong_analysis import yakuhai_dash

    data_root, _ = audit_input
    original = yakuhai_dash.singleton_to_dict

    def incomplete(row):
        result = original(row)
        del result["self_pair_event_index"]
        return result

    monkeypatch.setattr(yakuhai_dash, "singleton_to_dict", incomplete)
    assert audit_blind_logs(data_root, [2025], 1)["status"] == "fail"


def test_legacy_aggregate_comparison_ignores_only_new_followup(audit_input):
    _, current = audit_input
    legacy = deepcopy(current)
    legacy["metadata"]["schema_version"] = 2
    for group in (legacy["overall"], *legacy["years"]):
        group["initial_singletons"]["all"].pop("self_pair_followup")
        for key in ("by_role", "by_exposure", "by_yakuhai_class", "by_tile"):
            for row in group["initial_singletons"][key]:
                row.pop("self_pair_followup")
    assert compare_v2_v3_aggregates(legacy, current)["status"] == "pass"
    changed = deepcopy(current)
    changed["overall"]["initial_singletons"]["all"]["self_pair_followup"][
        "pon_shape"
    ]["count"] += 1
    assert compare_v2_v3_aggregates(legacy, changed)["status"] == "pass"
    changed["overall"]["opening_behavior"]["all"]["strict_dash"]["count"] += 1
    comparison = compare_v2_v3_aggregates(legacy, changed)
    assert comparison["status"] == "fail"
    assert comparison["checks"]["overall_preexisting_aggregates"] is False


def test_independent_source_audit_passes_and_detects_tampering(tmp_path) -> None:
    data_root = tmp_path / "raw"
    path = data_root / "2025" / "2025010100gm-00a9-0000-1234abcd.mjson"
    write_game(path, artificial_game())
    document = build_document((scan_year(YearTask(2025, str(data_root), None, 3)),))

    result = audit_document(document, data_root)
    assert result["status"] == "pass"
    assert result["failure_count"] == 0
    assert result["source_files"] == 1
    blind = audit_blind_logs(data_root, [2025], logs_per_year=1)
    assert blind["status"] == "pass"
    assert blind["source_logs"] == 1
    assert blind["east_kyokus"] == 1
    assert blind["initial_singletons"] >= 1

    groups = document["audit_samples_by_year"][0]["categories"]
    strict = next(row for row in groups if row["category"] == "strict_dash")
    strict["samples"][0]["opponent_pon_capable_at_discard"] = False
    result = audit_document(document, data_root)
    assert result["status"] == "fail"
    assert any(
        failure["field"] == "opponent_pon_capable_at_discard"
        for failure in result["failures"]
    )
    strict["samples"][0]["opponent_pon_capable_at_discard"] = True
    deal = next(row for row in groups if row["category"] == "initial_yakuhai_pair")
    deal["samples"][0]["shanten"] += 1
    result = audit_document(document, data_root)
    assert any(failure["field"] == "shanten" for failure in result["failures"])


def test_source_reader_handles_gzipped_mjson(tmp_path) -> None:
    path = tmp_path / "compressed.mjson"
    path.write_bytes(gzip.compress(b'{"type":"start_game"}\n'))
    assert _read_events(path) == [{"type": "start_game"}]


def test_independent_shanten_reference_known_hands() -> None:
    assert reference_shanten(
        ["1m", "2m", "3m", "4m", "5m", "6m", "7p", "8p", "9p",
         "5s", "5s", "2s", "3s"]
    ) == 0
    assert reference_shanten(
        ["1m", "2m", "3m", "4m", "5m", "6m", "7p", "8p", "9p",
         "E", "E", "1s", "9s"]
    ) == 1
    assert reference_shanten(
        ["1m", "2m", "3m", "4m", "5m", "6m", "7p", "8p", "9p",
         "E", "S", "W", "N"]
    ) == 2
