from __future__ import annotations

from dataclasses import dataclass
from math import exp
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import mahjong_analysis.combo_sparse_pipeline as pipeline
from mahjong_analysis.combo_sparse_pipeline import (
    YearSparseArtifact,
    build_year_sparse_artifact,
    conventional_feature_mask,
    fit_sparse_problem,
    global_feature_vocabulary,
    load_year_sparse_artifact,
    objective_and_gradient,
    prepare_logistic_problem,
    save_year_sparse_artifact,
)


def _artifact(
    *,
    year: int,
    feature_names: tuple[str, ...],
    values: list[list[float]],
    labels: list[bool],
    weights: list[float] | None = None,
    group_indices: list[int] | None = None,
    cluster_indices: list[int] | None = None,
    source_ranks: list[int] | None = None,
    source_ids: tuple[str, ...] | None = None,
) -> YearSparseArtifact:
    row_count = len(values)
    groups = group_indices or list(range(row_count))
    clusters = cluster_indices or list(groups)
    ranks = source_ranks or list(clusters)
    selected_sources = source_ids or tuple(
        f"{year}/source-{index}.mjson" for index in range(max(ranks) + 1)
    )
    group_count = max(groups) + 1
    cluster_count = max(clusters) + 1
    return YearSparseArtifact(
        year=year,
        feature_names=feature_names,
        matrix=csr_matrix(np.asarray(values, dtype=np.float64)),
        y_structural=np.asarray(labels, dtype=np.bool_),
        y_ron=np.asarray(labels, dtype=np.bool_),
        weights=np.asarray(weights or [1.0] * row_count, dtype=np.float64),
        group_indices=np.asarray(groups, dtype=np.int32),
        group_ids=tuple(f"group-{index}" for index in range(group_count)),
        cluster_indices=np.asarray(clusters, dtype=np.int32),
        cluster_ids=tuple(f"cluster-{index}" for index in range(cluster_count)),
        turn_bins=np.zeros(row_count, dtype=np.uint8),
        source_ranks=np.asarray(ranks, dtype=np.int32),
        source_ids_by_rank=selected_sources,
        provenance=(("sample_manifest_sha256", "abc123"),),
    )


def test_sharded_objective_gradient_matches_finite_difference() -> None:
    first = _artifact(
        year=2020,
        feature_names=("z", "x"),
        values=[[0.5, -1.0], [-0.25, 2.0], [1.5, 0.25]],
        labels=[False, True, True],
        weights=[0.7, 1.3, 0.4],
    )
    second = _artifact(
        year=2021,
        feature_names=("x", "a"),
        values=[[-0.5, 1.0], [1.25, -2.0]],
        labels=[False, True],
        weights=[2.0, 0.6],
    )
    vocabulary = global_feature_vocabulary((first, second))
    assert vocabulary == ("a", "x", "z")
    problem = prepare_logistic_problem(
        (second, first),
        label="ron_eligible",
        vocabulary=vocabulary,
    )
    parameters = np.asarray([0.2, -0.3, 0.4, 0.1], dtype=np.float64)
    l2 = 0.17

    _objective, analytic = objective_and_gradient(problem, parameters, l2=l2)
    finite_difference = np.zeros_like(parameters)
    step = 1e-6
    for index in range(len(parameters)):
        lower = parameters.copy()
        upper = parameters.copy()
        lower[index] -= step
        upper[index] += step
        lower_objective, _ = objective_and_gradient(problem, lower, l2=l2)
        upper_objective, _ = objective_and_gradient(problem, upper, l2=l2)
        finite_difference[index] = (upper_objective - lower_objective) / (2 * step)

    assert analytic == pytest.approx(finite_difference, rel=2e-7, abs=2e-9)


def test_lbfgsb_recovers_weighted_fractional_logistic_parameters() -> None:
    expected_intercept = -0.4
    expected_coefficient = 0.85
    values: list[list[float]] = []
    labels: list[bool] = []
    weights: list[float] = []
    groups: list[int] = []
    for group, x in enumerate((-2.0, -1.0, 0.0, 1.0, 2.0)):
        probability = 1.0 / (
            1.0 + exp(-(expected_intercept + expected_coefficient * x))
        )
        values.extend(([x], [x]))
        labels.extend((True, False))
        weights.extend((probability, 1.0 - probability))
        groups.extend((group, group))
    artifact = _artifact(
        year=2020,
        feature_names=("x",),
        values=values,
        labels=labels,
        weights=weights,
        group_indices=groups,
        cluster_indices=groups,
        source_ranks=groups,
    )
    problem = prepare_logistic_problem((artifact,), label="ron_eligible")

    model = fit_sparse_problem(
        problem,
        l2=0.0,
        max_iterations=200,
        gradient_tolerance=1e-10,
        function_tolerance=1e-15,
    )

    assert model.diagnostics.converged
    assert model.diagnostics.final_objective < model.diagnostics.initial_objective
    assert model.diagnostics.final_gradient_inf_norm < 1e-8
    assert model.intercept == pytest.approx(expected_intercept, abs=2e-7)
    assert model.coefficient("x") == pytest.approx(expected_coefficient, abs=2e-7)


def test_conventional_mask_keeps_combo_coefficient_exactly_zero() -> None:
    x_values = (-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0)
    artifact = _artifact(
        year=2020,
        feature_names=("tile:alternating", "combo:total"),
        values=[[1.0 if index % 2 else -1.0, x] for index, x in enumerate(x_values)],
        labels=[x > 0 for x in x_values],
    )
    vocabulary = global_feature_vocabulary((artifact,))
    mask = conventional_feature_mask(vocabulary)
    problem = prepare_logistic_problem(
        (artifact,), label="ron_eligible", vocabulary=vocabulary
    )

    base = fit_sparse_problem(problem, active_features=mask, l2=0.2)
    challenger = fit_sparse_problem(problem, l2=0.2)

    assert vocabulary == ("combo:total", "tile:alternating")
    assert mask.tolist() == [False, True]
    assert base.coefficient("combo:total") == 0.0
    assert not base.active_mask[0]
    assert challenger.coefficient("combo:total") > 0.5
    assert challenger.diagnostics.final_objective < base.diagnostics.final_objective


def test_artifact_round_trip_preserves_excluded_source_rank_prefix(
    tmp_path: Path,
) -> None:
    artifact = _artifact(
        year=2020,
        feature_names=("x", "combo:total"),
        values=[[1.0, 0.5], [-1.0, 0.25]],
        labels=[True, False],
        group_indices=[0, 1],
        cluster_indices=[0, 1],
        source_ranks=[1, 2],
        source_ids=(
            "2020/excluded.mjson",
            "2020/target-a.mjson",
            "2020/target-b.mjson",
        ),
    )
    path = tmp_path / "2020-combo-csr.npz"

    save_year_sparse_artifact(path, artifact)
    loaded = load_year_sparse_artifact(path)

    assert loaded.year == artifact.year
    assert loaded.feature_names == artifact.feature_names
    assert loaded.source_ids_by_rank == artifact.source_ids_by_rank
    assert loaded.provenance == artifact.provenance
    assert (loaded.matrix != artifact.matrix).nnz == 0
    np.testing.assert_array_equal(loaded.source_ranks, np.asarray([1, 2]))
    np.testing.assert_array_equal(loaded.source_prefix_mask(1), [False, False])
    np.testing.assert_array_equal(loaded.source_prefix_mask(2), [True, False])


@dataclass(frozen=True)
class _FakeDecision:
    decision_discard_number: int


@dataclass(frozen=True)
class _FakeCandidate:
    is_structural_wait: bool
    is_ron_eligible: bool


@dataclass(frozen=True)
class _FakeRow:
    year: int
    game_id: str
    decision_id: str
    candidate_count: int
    decision: _FakeDecision
    candidate: _FakeCandidate
    x: float
    combo: float


def test_builder_materializes_superset_once_and_preserves_manifest_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_feature_vector(
        row: _FakeRow, feature_set: str
    ) -> tuple[tuple[str, float], ...]:
        nonlocal calls
        calls += 1
        assert feature_set == "conventional_simple"
        return (("tile:test", row.x), ("combo:total", row.combo))

    monkeypatch.setattr(pipeline, "feature_vector", fake_feature_vector)
    rows = (
        _FakeRow(
            2020,
            "2020/target.mjson",
            "decision-a",
            2,
            _FakeDecision(7),
            _FakeCandidate(False, False),
            -1.0,
            0.25,
        ),
        _FakeRow(
            2020,
            "2020/target.mjson",
            "decision-a",
            2,
            _FakeDecision(7),
            _FakeCandidate(True, True),
            1.0,
            0.75,
        ),
    )

    artifact = build_year_sparse_artifact(
        rows,  # type: ignore[arg-type] - deliberately minimal row protocol
        source_rank_by_game={
            "2020/excluded.mjson": 0,
            "2020/target.mjson": 1,
        },
    )

    assert calls == len(rows)
    assert artifact.feature_names == ("tile:test", "combo:total")
    assert artifact.weights.tolist() == [0.5, 0.5]
    assert artifact.turn_bins.tolist() == [1, 1]
    assert artifact.source_ranks.tolist() == [1, 1]
    assert artifact.source_ids_by_rank == (
        "2020/excluded.mjson",
        "2020/target.mjson",
    )
    np.testing.assert_array_equal(artifact.source_prefix_mask(1), [False, False])
