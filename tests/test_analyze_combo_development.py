from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from analysis.analyze_combo_development import (
    _build_manifest,
    _canonical_json_sha256,
    _comparison_document,
    _samples_from_manifest,
    _turn_bin_documents,
    parse_args,
)
from mahjong_analysis.combo_dataset import DEVELOPMENT_YEARS, select_stable_year_sample
from mahjong_analysis.combo_prediction import (
    FEATURE_SETS,
    SparseExample,
    evaluate_binary_predictions,
)


def test_cli_has_no_option_to_include_validation_or_holdout_years() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--year", "2024"])
    with pytest.raises(SystemExit):
        parse_args(["--year", "2025"])
    with pytest.raises(SystemExit):
        parse_args(["--test-year", "2024"])
    with pytest.raises(SystemExit):
        parse_args(["--test-year", "2025"])


def test_cli_accepts_unique_development_prediction_years() -> None:
    args = parse_args(["--test-year", "2023", "--test-year", "2021"])

    assert args.test_years == [2023, 2021]

    with pytest.raises(SystemExit):
        parse_args(["--test-year", "2022", "--test-year", "2022"])


def test_cli_rejects_invalid_sample_and_optimizer_sizes() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--files-per-year", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--epochs", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--bootstrap-replicates", "-1"])


def test_manifest_digest_is_independent_of_dictionary_insertion_order() -> None:
    left = {"year": 2020, "sources": ["a", "b"]}
    right = {"sources": ["a", "b"], "year": 2020}

    assert _canonical_json_sha256(left) == _canonical_json_sha256(right)


def test_manifest_round_trip_locks_exact_sources_and_digests(tmp_path: Path) -> None:
    for year in DEVELOPMENT_YEARS:
        year_root = tmp_path / str(year)
        year_root.mkdir()
        (year_root / f"{year}.mjson").touch()
    args = parse_args(
        [
            "--raw-root",
            str(tmp_path),
            "--files-per-year",
            "1",
            "--sample-seed",
            "7",
        ]
    )
    selected = {
        year: select_stable_year_sample(tmp_path, year, 1, seed=7)
        for year in DEVELOPMENT_YEARS
    }
    manifest = _build_manifest(selected, args)

    loaded = _samples_from_manifest(manifest, args)

    assert manifest["schema_version"] == 2
    assert len(manifest["years"]["2020"]["candidate_population_sha256"]) == 64
    assert {year: sample.relative_sources for year, sample in loaded.items()} == {
        year: sample.relative_sources for year, sample in selected.items()
    }

    legacy = deepcopy(manifest)
    legacy["schema_version"] = 1
    for year_document in legacy["years"].values():
        year_document.pop("candidate_population_sha256")
        for entry in year_document["selected"]:
            entry.pop("content_sha256")
    legacy_loaded = _samples_from_manifest(legacy, args)
    assert legacy_loaded[2020].relative_sources == selected[2020].relative_sources

    (tmp_path / "2020" / "2020.mjson").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="content digest mismatch"):
        _samples_from_manifest(manifest, args)

    tampered = deepcopy(manifest)
    tampered["years"]["2020"]["selected"][0]["selection_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        _samples_from_manifest(tampered, args)


def test_turn_bin_documents_partition_complete_decisions_and_preserve_weights() -> None:
    turn_bins = ("1-6", "1-6", "7-9", "7-9", "10-12", "10-12", "13+", "13+")
    examples = tuple(
        SparseExample(
            row_id=f"row-{index}",
            group_id=f"decision-{index // 2}",
            cluster_id="game-a",
            label=bool(index % 2),
            weight=0.5,
            features=(),
        )
        for index in range(8)
    )
    common_predictions = tuple(0.8 if example.label else 0.2 for example in examples)
    predictions = {feature_set: common_predictions for feature_set in FEATURE_SETS}

    documents = _turn_bin_documents(
        examples,
        predictions,
        turn_bins,
        bootstrap_replicates=10,
        bootstrap_seed=5,
    )

    assert sum(document["candidate_rows"] for document in documents.values()) == 8
    for document in documents.values():
        assert document["candidate_rows"] == 2
        assert document["decision_count"] == 1
        assert document["game_count"] == 1
        metrics = document["models"]["conventional"]["metrics"]
        assert metrics["positive_rate"] == 0.5
        comparison = document["paired_comparisons"][
            "conventional_simple_minus_conventional"
        ]
        assert comparison["delta"]["log_loss"] == 0.0


def test_turn_bin_documents_report_empty_bins_explicitly() -> None:
    examples = (
        SparseExample("row-0", "decision", "game", False, 0.5, ()),
        SparseExample("row-1", "decision", "game", True, 0.5, ()),
    )
    predictions = {feature_set: (0.2, 0.8) for feature_set in FEATURE_SETS}

    documents = _turn_bin_documents(
        examples,
        predictions,
        ("1-6", "1-6"),
        bootstrap_replicates=0,
        bootstrap_seed=5,
    )

    assert documents["1-6"]["decision_count"] == 1
    for bin_name in ("7-9", "10-12", "13+"):
        assert documents[bin_name]["candidate_rows"] == 0
        assert documents[bin_name]["models"] is None


def test_comparison_document_uses_shared_draws_for_late_early_contrast() -> None:
    examples = tuple(
        SparseExample(
            row_id=f"row-{index}",
            group_id=f"decision-{index // 2}",
            cluster_id=f"game-{index // 4}",
            label=bool(index % 2),
            weight=0.5,
            features=(),
        )
        for index in range(8)
    )
    predictions = {
        feature_set: tuple(0.8 if example.label else 0.2 for example in examples)
        for feature_set in FEATURE_SETS
    }
    metrics = {
        feature_set: evaluate_binary_predictions(examples, values)
        for feature_set, values in predictions.items()
    }

    document = _comparison_document(
        examples,
        metrics,
        predictions,
        model_documents={feature_set: {} for feature_set in FEATURE_SETS},
        bootstrap_replicates=20,
        bootstrap_seed=9,
        turn_bins=("1-6", "1-6", "13+", "13+") * 2,
    )

    assert document["bootstrap_design"][
        "shared_across_comparisons_and_turn_bins"
    ]
    contrast = document["turn_bin_contrasts"]["late_minus_early"][
        "paired_comparisons"
    ]["conventional_simple_minus_conventional"]
    assert contrast["log_loss"]["point"] == pytest.approx(0.0)
    assert contrast["log_loss"]["lower"] == pytest.approx(0.0)
    assert contrast["log_loss"]["upper"] == pytest.approx(0.0)
