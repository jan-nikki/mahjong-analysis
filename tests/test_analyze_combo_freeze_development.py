from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from analysis.analyze_combo_freeze_development import (
    L2_GRID,
    _choose_lambda,
    _examples_from_artifact,
    _fit_is_accepted,
    _weighted_log_loss_sum,
    _write_json,
    parse_args,
)
from mahjong_analysis.combo_sparse_pipeline import (
    LbfgsDiagnostics,
    SparseLogisticModel,
    YearSparseArtifact,
)


def test_cli_exposes_only_fixed_development_prefixes() -> None:
    assert parse_args([]).source_prefix == 1_000
    assert parse_args(["--source-prefix", "250"]).source_prefix == 250

    with pytest.raises(SystemExit):
        parse_args(["--source-prefix", "999"])
    with pytest.raises(SystemExit):
        parse_args(["--test-year", "2024"])
    with pytest.raises(SystemExit):
        parse_args(["--max-iterations", "600", "--stability-max-iterations", "300"])


def test_lambda_selection_uses_fixed_grid_and_prefers_larger_within_tolerance() -> None:
    scores = {value: 0.25 + index * 0.01 for index, value in enumerate(L2_GRID)}
    scores[1e-3] = 0.2
    scores[3e-3] = 0.2 + 5e-7

    assert _choose_lambda(scores) == 3e-3

    with pytest.raises(ValueError, match="fixed grid"):
        _choose_lambda({1e-3: 0.2})


def test_weighted_log_loss_sum_matches_fractional_manual_value() -> None:
    labels = np.asarray([True, False])
    predictions = np.asarray([0.8, 0.3])
    weights = np.asarray([0.25, 0.75])

    result = _weighted_log_loss_sum(labels, predictions, weights)

    expected = 0.25 * -np.log(0.8) + 0.75 * -np.log(0.7)
    assert result == pytest.approx(expected)


def test_artifact_examples_preserve_compact_ids_turn_bins_and_prefix() -> None:
    artifact = YearSparseArtifact(
        year=2022,
        feature_names=("x",),
        matrix=csr_matrix(np.asarray([[1.0], [2.0], [3.0]])),
        y_structural=np.asarray([False, True, True]),
        y_ron=np.asarray([False, False, True]),
        weights=np.asarray([0.5, 0.5, 1.0]),
        group_indices=np.asarray([0, 0, 1], dtype=np.int32),
        group_ids=("d0", "d1"),
        cluster_indices=np.asarray([0, 0, 1], dtype=np.int32),
        cluster_ids=("2022/a", "2022/b"),
        turn_bins=np.asarray([0, 0, 3], dtype=np.uint8),
        source_ranks=np.asarray([0, 0, 50], dtype=np.int32),
        source_ids_by_rank=tuple(f"2022/{index}" for index in range(51)),
    )

    examples, turn_bins = _examples_from_artifact(
        artifact,
        source_prefix=50,
        label="ron_eligible",
    )

    assert [example.row_id for example in examples] == ["2022:0", "2022:1"]
    assert [example.group_id for example in examples] == ["d0", "d0"]
    assert [example.cluster_id for example in examples] == ["2022/a", "2022/a"]
    assert [example.label for example in examples] == [False, False]
    assert turn_bins == ("1-6", "1-6")


def test_fit_acceptance_requires_solver_success_and_gradient_gate() -> None:
    accepted = LbfgsDiagnostics(
        converged=True,
        status=0,
        message="ok",
        iterations=2,
        function_evaluations=3,
        gradient_evaluations=3,
        initial_objective=0.7,
        final_objective=0.6,
        initial_gradient_inf_norm=0.1,
        final_gradient_inf_norm=1e-8,
        row_count=10,
        total_weight=5.0,
        active_feature_count=2,
        l2=1e-3,
    )

    def model(diagnostics: LbfgsDiagnostics) -> SparseLogisticModel:
        return SparseLogisticModel(
            label="ron_eligible",
            feature_names=("x",),
            active_mask=np.asarray([True]),
            intercept=0.0,
            coefficients=np.asarray([0.5]),
            diagnostics=diagnostics,
        )

    assert _fit_is_accepted(model(accepted), 1e-7, max_iterations=300)
    assert type(_fit_is_accepted(model(accepted), 1e-7, max_iterations=300)) is bool
    assert not _fit_is_accepted(
        model(LbfgsDiagnostics(**{**accepted.__dict__, "converged": False})),
        1e-7,
        max_iterations=300,
    )
    assert not _fit_is_accepted(
        model(
            LbfgsDiagnostics(
                **{**accepted.__dict__, "final_gradient_inf_norm": 2e-7}
            )
        ),
        1e-7,
        max_iterations=300,
    )


def test_write_json_normalizes_numpy_scalars(tmp_path: Path) -> None:
    output = tmp_path / "report.json"

    _write_json(
        output,
        {
            "accepted": np.bool_(True),
            "count": np.int64(3),
            "score": np.float64(0.25),
        },
    )

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "accepted": True,
        "count": 3,
        "score": 0.25,
    }
