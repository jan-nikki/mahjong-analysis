"""Memory-bounded sparse modeling for the combo-development analysis.

The raw combo extractor produces a rich Python object graph.  Keeping that graph
and a separate :class:`~mahjong_analysis.combo_prediction.SparseExample` graph
for every model is prohibitively expensive for the 1,000-file-per-year run.  This
module turns one year's ``ComboCandidateRow`` objects into a compact CSR artifact
once, then fits any nested model from the same artifacts.

The intended primary-analysis flow is::

    artifact = build_year_sparse_artifact(
        rows,
        source_rank_by_game={source: rank for rank, source in enumerate(sources)},
    )
    save_year_sparse_artifact(path, artifact)

    vocabulary = global_feature_vocabulary(all_development_artifacts)
    problem = prepare_logistic_problem(
        training_artifacts,
        label="ron_eligible",
        vocabulary=vocabulary,
    )
    base = fit_sparse_problem(
        problem,
        active_features=conventional_feature_mask(vocabulary),
    )
    challenger = fit_sparse_problem(problem)

``conventional_simple`` is materialized as the sole superset.  The conventional
model is therefore a coefficient mask, not a separately rebuilt feature table.
Loss is the weighted *mean* binary log loss plus ``l2 / 2 * ||beta||^2``;
the intercept is never penalized.
"""

from __future__ import annotations

import json
import os
from array import array
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, log
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import expit

from mahjong_analysis.combo_prediction import (
    DECISION_TURN_BINS,
    ComboCandidateRow,
    decision_turn_bin,
    feature_vector,
)

LabelName = Literal["structural_wait", "ron_eligible"]
type FeatureSelector = (
    Sequence[bool] | NDArray[np.bool_] | Callable[[str], bool] | None
)

ARTIFACT_SCHEMA_VERSION = 1
_TURN_BIN_TO_CODE = {name: index for index, name in enumerate(DECISION_TURN_BINS)}


@dataclass(frozen=True)
class YearSparseArtifact:
    """One year's model-ready ``conventional_simple`` candidate rows.

    ``source_ranks`` indexes ``source_ids_by_rank`` and deliberately refers to
    the selected-file rank, not the target-game cluster index.  Consequently a
    prefix such as ``source_ranks < 250`` is the exact lowest-hash 250-file
    sample even when some selected files yielded no candidate rows.
    """

    year: int
    feature_names: tuple[str, ...]
    matrix: csr_matrix
    y_structural: NDArray[np.bool_]
    y_ron: NDArray[np.bool_]
    weights: NDArray[np.float64]
    group_indices: NDArray[np.int32]
    group_ids: tuple[str, ...]
    cluster_indices: NDArray[np.int32]
    cluster_ids: tuple[str, ...]
    turn_bins: NDArray[np.uint8]
    source_ranks: NDArray[np.int32]
    source_ids_by_rank: tuple[str, ...]
    provenance: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _validate_year_artifact(self)

    @property
    def row_count(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def feature_count(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def selected_source_count(self) -> int:
        return len(self.source_ids_by_rank)

    def labels(self, label: LabelName) -> NDArray[np.bool_]:
        """Return the requested binary label vector without copying it."""

        if label == "structural_wait":
            return self.y_structural
        if label == "ron_eligible":
            return self.y_ron
        raise ValueError(f"unknown label: {label!r}")

    def source_prefix_mask(self, files_per_year: int) -> NDArray[np.bool_]:
        """Return rows belonging to the exact selected-file hash prefix."""

        if (
            isinstance(files_per_year, bool)
            or not isinstance(files_per_year, int)
            or not 1 <= files_per_year <= self.selected_source_count
        ):
            raise ValueError(
                "files_per_year must be between 1 and selected_source_count"
            )
        return self.source_ranks < files_per_year


@dataclass(frozen=True)
class _ProblemShard:
    year: int
    matrix: csr_matrix
    labels: NDArray[np.float64]
    weights: NDArray[np.float64]
    local_to_global: NDArray[np.int32]


@dataclass(frozen=True)
class SparseLogisticProblem:
    """Prepared, reusable training problem spanning one or more year shards."""

    label: LabelName
    vocabulary: tuple[str, ...]
    shards: tuple[_ProblemShard, ...]
    years: tuple[int, ...]
    row_count: int
    total_weight: float
    positive_weight: float
    source_prefix: int | None


@dataclass(frozen=True)
class LbfgsDiagnostics:
    """Stable optimizer and final objective/gradient diagnostics."""

    converged: bool
    status: int
    message: str
    iterations: int
    function_evaluations: int
    gradient_evaluations: int
    initial_objective: float
    final_objective: float
    initial_gradient_inf_norm: float
    final_gradient_inf_norm: float
    row_count: int
    total_weight: float
    active_feature_count: int
    l2: float


@dataclass(frozen=True)
class SparseLogisticModel:
    """A fitted model in one global, lexicographically sorted vocabulary."""

    label: LabelName
    feature_names: tuple[str, ...]
    active_mask: NDArray[np.bool_]
    intercept: float
    coefficients: NDArray[np.float64]
    diagnostics: LbfgsDiagnostics

    def coefficient(self, feature_name: str) -> float:
        """Return one coefficient, or zero for an unseen feature name."""

        position = _sorted_feature_position(self.feature_names, feature_name)
        return 0.0 if position is None else float(self.coefficients[position])

    def predict_artifact(self, artifact: YearSparseArtifact) -> NDArray[np.float64]:
        """Predict one artifact; artifact-only unknown features receive zero."""

        coefficient_by_name = dict(
            zip(self.feature_names, self.coefficients, strict=True)
        )
        local_coefficients = np.fromiter(
            (coefficient_by_name.get(name, 0.0) for name in artifact.feature_names),
            dtype=np.float64,
            count=artifact.feature_count,
        )
        predictions = expit(
            self.intercept + artifact.matrix.dot(local_coefficients)
        ).astype(np.float64, copy=False)
        predictions.setflags(write=False)
        return predictions


def build_year_sparse_artifact(
    rows: Iterable[ComboCandidateRow],
    *,
    year: int | None = None,
    source_rank_by_game: Mapping[str, int] | None = None,
    provenance: Mapping[str, str] | None = None,
) -> YearSparseArtifact:
    """Build a compact CSR artifact from one year's candidate rows.

    Feature extraction is performed exactly once with the
    ``conventional_simple`` superset.  Zero-valued features are omitted and
    duplicate feature names within a row are summed before insertion.

    ``source_rank_by_game`` should contain *every* selected source, including a
    selected source that later yields no target rows.  Its values must be the
    exact contiguous manifest ranks ``0..N-1``.
    """

    if year is not None and (isinstance(year, bool) or not isinstance(year, int)):
        raise TypeError("year must be an integer")
    rank_by_source, source_ids_by_rank = _normalize_source_ranks(source_rank_by_game)

    local_feature_index: dict[str, int] = {}
    local_feature_names: list[str] = []
    group_index: dict[str, int] = {}
    group_ids: list[str] = []
    group_expected_counts: list[int] = []
    group_observed_counts: list[int] = []
    group_cluster_indices: list[int] = []
    group_source_ranks: list[int] = []
    cluster_index: dict[str, int] = {}
    cluster_ids: list[str] = []

    indptr_buffer = array("q", [0])
    indices_buffer = array("i")
    data_buffer = array("d")
    y_structural_buffer = array("b")
    y_ron_buffer = array("b")
    weights_buffer = array("d")
    group_buffer = array("i")
    cluster_buffer = array("i")
    turn_buffer = array("B")
    source_rank_buffer = array("i")

    inferred_rank_by_source: dict[str, int] = {}
    inferred_sources: list[str] = []
    inferred_year = year
    row_count = 0
    for row in rows:
        if inferred_year is None:
            inferred_year = row.year
        if row.year != inferred_year:
            raise ValueError("all rows in a year artifact must have the same year")
        if row.candidate_count < 1:
            raise ValueError("candidate_count must be positive")

        cluster = cluster_index.get(row.game_id)
        if cluster is None:
            cluster = len(cluster_ids)
            _require_int32(cluster, "cluster index")
            cluster_index[row.game_id] = cluster
            cluster_ids.append(row.game_id)

        if rank_by_source is None:
            source_rank = inferred_rank_by_source.get(row.game_id)
            if source_rank is None:
                source_rank = len(inferred_sources)
                _require_int32(source_rank, "source rank")
                inferred_rank_by_source[row.game_id] = source_rank
                inferred_sources.append(row.game_id)
        else:
            try:
                source_rank = rank_by_source[row.game_id]
            except KeyError as error:
                raise ValueError(
                    f"row game_id is absent from source_rank_by_game: {row.game_id}"
                ) from error

        group = group_index.get(row.decision_id)
        if group is None:
            group = len(group_ids)
            _require_int32(group, "group index")
            group_index[row.decision_id] = group
            group_ids.append(row.decision_id)
            group_expected_counts.append(row.candidate_count)
            group_observed_counts.append(0)
            group_cluster_indices.append(cluster)
            group_source_ranks.append(source_rank)
        elif (
            group_expected_counts[group] != row.candidate_count
            or group_cluster_indices[group] != cluster
            or group_source_ranks[group] != source_rank
        ):
            raise ValueError("decision rows disagree on count, cluster, or source rank")
        group_observed_counts[group] += 1

        combined_features: dict[str, float] = {}
        for name, raw_value in feature_vector(row, "conventional_simple"):
            if not isinstance(name, str) or not name:
                raise ValueError("feature names must be non-empty strings")
            value = float(raw_value)
            if not isfinite(value):
                raise ValueError("feature values must be finite")
            combined_features[name] = combined_features.get(name, 0.0) + value
        for name, value in combined_features.items():
            if value == 0.0:
                continue
            feature = local_feature_index.get(name)
            if feature is None:
                feature = len(local_feature_names)
                _require_int32(feature, "feature index")
                local_feature_index[name] = feature
                local_feature_names.append(name)
            indices_buffer.append(feature)
            data_buffer.append(value)
        indptr_buffer.append(len(indices_buffer))

        y_structural_buffer.append(int(row.candidate.is_structural_wait))
        y_ron_buffer.append(int(row.candidate.is_ron_eligible))
        weights_buffer.append(1.0 / row.candidate_count)
        group_buffer.append(group)
        cluster_buffer.append(cluster)
        turn_name = decision_turn_bin(row.decision.decision_discard_number)
        turn_buffer.append(_TURN_BIN_TO_CODE[turn_name])
        source_rank_buffer.append(source_rank)
        row_count += 1

    if row_count == 0 or inferred_year is None:
        raise ValueError("at least one candidate row is required")
    for group, (observed, expected) in enumerate(
        zip(group_observed_counts, group_expected_counts, strict=True)
    ):
        if observed != expected:
            raise ValueError(
                f"decision {group_ids[group]!r} has {observed} rows, "
                f"expected {expected}"
            )
    if rank_by_source is None:
        source_ids_by_rank = tuple(inferred_sources)

    matrix = csr_matrix(
        (
            _array_view(data_buffer, np.float64),
            _array_view(indices_buffer, np.int32),
            _array_view(indptr_buffer, np.int64),
        ),
        shape=(row_count, len(local_feature_names)),
        copy=False,
    )
    matrix.check_format(full_check=True)
    normalized_provenance = tuple(sorted((provenance or {}).items()))
    artifact = YearSparseArtifact(
        year=inferred_year,
        feature_names=tuple(local_feature_names),
        matrix=matrix,
        y_structural=_array_view(y_structural_buffer, np.int8).astype(
            np.bool_, copy=False
        ),
        y_ron=_array_view(y_ron_buffer, np.int8).astype(np.bool_, copy=False),
        weights=_array_view(weights_buffer, np.float64),
        group_indices=_array_view(group_buffer, np.int32),
        group_ids=tuple(group_ids),
        cluster_indices=_array_view(cluster_buffer, np.int32),
        cluster_ids=tuple(cluster_ids),
        turn_bins=_array_view(turn_buffer, np.uint8),
        source_ranks=_array_view(source_rank_buffer, np.int32),
        source_ids_by_rank=source_ids_by_rank,
        provenance=normalized_provenance,
    )
    _make_artifact_arrays_read_only(artifact)
    return artifact


def global_feature_vocabulary(
    artifacts: Iterable[YearSparseArtifact],
) -> tuple[str, ...]:
    """Return the lexicographically sorted union of all local vocabularies."""

    artifacts_tuple = tuple(artifacts)
    if not artifacts_tuple:
        raise ValueError("at least one year artifact is required")
    return tuple(
        sorted(
            {name for artifact in artifacts_tuple for name in artifact.feature_names}
        )
    )


def conventional_feature_mask(
    vocabulary: Sequence[str],
) -> NDArray[np.bool_]:
    """Select the conventional base and exclude every ``combo:*`` feature."""

    names = _validate_vocabulary(vocabulary)
    mask = np.fromiter(
        (not name.startswith("combo:") for name in names),
        dtype=np.bool_,
        count=len(names),
    )
    mask.setflags(write=False)
    return mask


def prepare_logistic_problem(
    artifacts: Iterable[YearSparseArtifact],
    *,
    label: LabelName,
    vocabulary: Sequence[str] | None = None,
    source_prefix: int | None = None,
) -> SparseLogisticProblem:
    """Prepare reusable year shards for one label and optional hash prefix.

    Pass a vocabulary built from *all* development artifacts when base and
    challenger fits or expanding-window folds must share identical columns.
    Prefix slicing is performed once here, not once per optimizer evaluation.
    """

    if label not in {"structural_wait", "ron_eligible"}:
        raise ValueError(f"unknown label: {label!r}")
    ordered = tuple(sorted(artifacts, key=lambda artifact: artifact.year))
    if not ordered:
        raise ValueError("at least one year artifact is required")
    if len({artifact.year for artifact in ordered}) != len(ordered):
        raise ValueError("year artifacts must have unique years")
    names = (
        global_feature_vocabulary(ordered)
        if vocabulary is None
        else _validate_vocabulary(vocabulary)
    )
    global_index = {name: index for index, name in enumerate(names)}

    problem_shards: list[_ProblemShard] = []
    total_rows = 0
    total_weight = 0.0
    positive_weight = 0.0
    for artifact in ordered:
        try:
            local_to_global = np.fromiter(
                (global_index[name] for name in artifact.feature_names),
                dtype=np.int32,
                count=artifact.feature_count,
            )
        except KeyError as error:
            raise ValueError(
                f"vocabulary omits artifact feature: {error.args[0]!r}"
            ) from error
        labels = artifact.labels(label).astype(np.float64, copy=False)
        matrix = artifact.matrix
        weights = artifact.weights
        if source_prefix is not None:
            mask = artifact.source_prefix_mask(source_prefix)
            matrix = matrix[mask].tocsr()
            labels = labels[mask]
            weights = weights[mask]
        if matrix.shape[0] == 0:
            continue
        shard_weight = float(np.sum(weights, dtype=np.float64))
        problem_shards.append(
            _ProblemShard(
                year=artifact.year,
                matrix=matrix,
                labels=labels,
                weights=weights,
                local_to_global=local_to_global,
            )
        )
        total_rows += int(matrix.shape[0])
        total_weight += shard_weight
        positive_weight += float(np.dot(weights, labels))
    if not problem_shards or total_weight <= 0:
        raise ValueError("prepared problem contains no positive-weight rows")
    return SparseLogisticProblem(
        label=label,
        vocabulary=names,
        shards=tuple(problem_shards),
        years=tuple(artifact.year for artifact in ordered),
        row_count=total_rows,
        total_weight=total_weight,
        positive_weight=positive_weight,
        source_prefix=source_prefix,
    )


def objective_and_gradient(
    problem: SparseLogisticProblem,
    parameters: Sequence[float] | NDArray[np.float64],
    *,
    active_features: FeatureSelector = None,
    l2: float = 0.001,
) -> tuple[float, NDArray[np.float64]]:
    """Return weighted-mean penalized objective and its analytic gradient.

    Parameter zero is the unpenalized intercept.  Remaining parameters are in
    the order of ``problem.vocabulary[active_mask]``.
    """

    mask = _normalize_feature_mask(problem.vocabulary, active_features)
    active_indices = np.flatnonzero(mask)
    return _objective_and_gradient_active(
        problem,
        np.asarray(parameters, dtype=np.float64),
        active_indices,
        l2,
    )


def fit_sparse_problem(
    problem: SparseLogisticProblem,
    *,
    active_features: FeatureSelector = None,
    l2: float = 0.001,
    max_iterations: int = 500,
    gradient_tolerance: float = 1e-7,
    function_tolerance: float = 1e-12,
    max_line_search_steps: int = 50,
    history_size: int = 20,
) -> SparseLogisticModel:
    """Fit a prepared problem with SciPy L-BFGS-B.

    The same ``problem`` can be fitted twice with a conventional mask and with
    all features, avoiding extraction or CSR reconstruction for the nested
    base/challenger comparison.
    """

    _validate_optimizer_options(
        l2=l2,
        max_iterations=max_iterations,
        gradient_tolerance=gradient_tolerance,
        function_tolerance=function_tolerance,
        max_line_search_steps=max_line_search_steps,
        history_size=history_size,
    )
    mask = _normalize_feature_mask(problem.vocabulary, active_features)
    active_indices = np.flatnonzero(mask)
    prevalence = np.clip(
        problem.positive_weight / problem.total_weight,
        1e-12,
        1 - 1e-12,
    )
    initial = np.zeros(1 + len(active_indices), dtype=np.float64)
    initial[0] = log(prevalence / (1.0 - prevalence))

    def function(parameters: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        return _objective_and_gradient_active(problem, parameters, active_indices, l2)

    initial_objective, initial_gradient = function(initial)
    result = minimize(
        function,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": max_iterations,
            "gtol": gradient_tolerance,
            "ftol": function_tolerance,
            "maxls": max_line_search_steps,
            "maxcor": history_size,
        },
    )
    final_parameters = np.asarray(result.x, dtype=np.float64)
    final_objective, final_gradient = function(final_parameters)
    coefficients = np.zeros(len(problem.vocabulary), dtype=np.float64)
    coefficients[active_indices] = final_parameters[1:]
    normalized_mask = mask.copy()
    normalized_mask.setflags(write=False)
    coefficients.setflags(write=False)
    diagnostics = LbfgsDiagnostics(
        converged=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        iterations=int(result.nit),
        function_evaluations=int(result.nfev),
        gradient_evaluations=int(getattr(result, "njev", result.nfev)),
        initial_objective=float(initial_objective),
        final_objective=float(final_objective),
        initial_gradient_inf_norm=_infinity_norm(initial_gradient),
        final_gradient_inf_norm=_infinity_norm(final_gradient),
        row_count=problem.row_count,
        total_weight=problem.total_weight,
        active_feature_count=len(active_indices),
        l2=float(l2),
    )
    return SparseLogisticModel(
        label=problem.label,
        feature_names=problem.vocabulary,
        active_mask=normalized_mask,
        intercept=float(final_parameters[0]),
        coefficients=coefficients,
        diagnostics=diagnostics,
    )


def fit_sparse_years(
    artifacts: Iterable[YearSparseArtifact],
    *,
    label: LabelName,
    vocabulary: Sequence[str] | None = None,
    source_prefix: int | None = None,
    active_features: FeatureSelector = None,
    l2: float = 0.001,
    max_iterations: int = 500,
    gradient_tolerance: float = 1e-7,
    function_tolerance: float = 1e-12,
    max_line_search_steps: int = 50,
    history_size: int = 20,
) -> SparseLogisticModel:
    """Convenience wrapper around ``prepare_logistic_problem`` and fitting."""

    problem = prepare_logistic_problem(
        artifacts,
        label=label,
        vocabulary=vocabulary,
        source_prefix=source_prefix,
    )
    return fit_sparse_problem(
        problem,
        active_features=active_features,
        l2=l2,
        max_iterations=max_iterations,
        gradient_tolerance=gradient_tolerance,
        function_tolerance=function_tolerance,
        max_line_search_steps=max_line_search_steps,
        history_size=history_size,
    )


def save_year_sparse_artifact(
    path: Path,
    artifact: YearSparseArtifact,
    *,
    compressed: bool = True,
) -> None:
    """Atomically save an artifact as a pickle-free NumPy archive."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": np.asarray(ARTIFACT_SCHEMA_VERSION, dtype=np.int64),
        "year": np.asarray(artifact.year, dtype=np.int64),
        "shape": np.asarray(artifact.matrix.shape, dtype=np.int64),
        "feature_names": np.asarray(artifact.feature_names, dtype=np.str_),
        "data": artifact.matrix.data,
        "indices": artifact.matrix.indices,
        "indptr": artifact.matrix.indptr,
        "y_structural": artifact.y_structural,
        "y_ron": artifact.y_ron,
        "weights": artifact.weights,
        "group_indices": artifact.group_indices,
        "group_ids": np.asarray(artifact.group_ids, dtype=np.str_),
        "cluster_indices": artifact.cluster_indices,
        "cluster_ids": np.asarray(artifact.cluster_ids, dtype=np.str_),
        "turn_bins": artifact.turn_bins,
        "source_ranks": artifact.source_ranks,
        "source_ids_by_rank": np.asarray(artifact.source_ids_by_rank, dtype=np.str_),
        "provenance_json": np.asarray(
            json.dumps(artifact.provenance, ensure_ascii=False, separators=(",", ":")),
            dtype=np.str_,
        ),
    }
    temporary_name: str | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            saver = np.savez_compressed if compressed else np.savez
            saver(temporary, **payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        Path(temporary_name).replace(destination)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def load_year_sparse_artifact(path: Path) -> YearSparseArtifact:
    """Load and fully validate a pickle-free year artifact."""

    with np.load(Path(path), allow_pickle=False) as document:
        schema_version = int(document["schema_version"].item())
        if schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported artifact schema version: {schema_version}")
        shape_values = np.asarray(document["shape"], dtype=np.int64)
        if shape_values.shape != (2,):
            raise ValueError("artifact matrix shape is invalid")
        shape = (int(shape_values[0]), int(shape_values[1]))
        matrix = csr_matrix(
            (
                np.asarray(document["data"], dtype=np.float64).copy(),
                np.asarray(document["indices"], dtype=np.int32).copy(),
                np.asarray(document["indptr"], dtype=np.int64).copy(),
            ),
            shape=shape,
            copy=False,
        )
        provenance_value = json.loads(str(document["provenance_json"].item()))
        artifact = YearSparseArtifact(
            year=int(document["year"].item()),
            feature_names=tuple(str(value) for value in document["feature_names"]),
            matrix=matrix,
            y_structural=np.asarray(document["y_structural"], dtype=np.bool_).copy(),
            y_ron=np.asarray(document["y_ron"], dtype=np.bool_).copy(),
            weights=np.asarray(document["weights"], dtype=np.float64).copy(),
            group_indices=np.asarray(document["group_indices"], dtype=np.int32).copy(),
            group_ids=tuple(str(value) for value in document["group_ids"]),
            cluster_indices=np.asarray(
                document["cluster_indices"], dtype=np.int32
            ).copy(),
            cluster_ids=tuple(str(value) for value in document["cluster_ids"]),
            turn_bins=np.asarray(document["turn_bins"], dtype=np.uint8).copy(),
            source_ranks=np.asarray(document["source_ranks"], dtype=np.int32).copy(),
            source_ids_by_rank=tuple(
                str(value) for value in document["source_ids_by_rank"]
            ),
            provenance=tuple((str(key), str(value)) for key, value in provenance_value),
        )
    matrix.check_format(full_check=True)
    _make_artifact_arrays_read_only(artifact)
    return artifact


def _objective_and_gradient_active(
    problem: SparseLogisticProblem,
    parameters: NDArray[np.float64],
    active_indices: NDArray[np.int64],
    l2: float,
) -> tuple[float, NDArray[np.float64]]:
    if not isfinite(l2) or l2 < 0:
        raise ValueError("l2 must be finite and nonnegative")
    expected_size = 1 + len(active_indices)
    if parameters.ndim != 1 or len(parameters) != expected_size:
        raise ValueError(f"parameters must contain exactly {expected_size} values")
    if not np.all(np.isfinite(parameters)):
        raise ValueError("parameters must be finite")

    intercept = float(parameters[0])
    active_coefficients = parameters[1:]
    full_coefficients = np.zeros(len(problem.vocabulary), dtype=np.float64)
    full_coefficients[active_indices] = active_coefficients
    full_gradient = np.zeros(len(problem.vocabulary), dtype=np.float64)
    intercept_gradient = 0.0
    weighted_loss = 0.0
    for shard in problem.shards:
        local_coefficients = full_coefficients[shard.local_to_global]
        linear = intercept + shard.matrix.dot(local_coefficients)
        losses = np.logaddexp(0.0, linear) - shard.labels * linear
        weighted_loss += float(np.dot(shard.weights, losses))
        residual = shard.weights * (expit(linear) - shard.labels) / problem.total_weight
        intercept_gradient += float(np.sum(residual, dtype=np.float64))
        local_gradient = np.asarray(shard.matrix.T.dot(residual)).reshape(-1)
        full_gradient[shard.local_to_global] += local_gradient

    objective = weighted_loss / problem.total_weight
    objective += 0.5 * l2 * float(np.dot(active_coefficients, active_coefficients))
    gradient = np.empty(expected_size, dtype=np.float64)
    gradient[0] = intercept_gradient
    gradient[1:] = full_gradient[active_indices] + l2 * active_coefficients
    return float(objective), gradient


def _normalize_feature_mask(
    vocabulary: tuple[str, ...],
    selector: FeatureSelector,
) -> NDArray[np.bool_]:
    if selector is None:
        return np.ones(len(vocabulary), dtype=np.bool_)
    if callable(selector):
        return np.fromiter(
            (bool(selector(name)) for name in vocabulary),
            dtype=np.bool_,
            count=len(vocabulary),
        )
    mask = np.asarray(selector, dtype=np.bool_)
    if mask.ndim != 1 or len(mask) != len(vocabulary):
        raise ValueError("active feature mask must align with the vocabulary")
    return mask.copy()


def _normalize_source_ranks(
    source_rank_by_game: Mapping[str, int] | None,
) -> tuple[dict[str, int] | None, tuple[str, ...]]:
    if source_rank_by_game is None:
        return None, ()
    normalized: dict[str, int] = {}
    for source, rank in source_rank_by_game.items():
        if not isinstance(source, str) or not source:
            raise ValueError("source IDs must be non-empty strings")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError("source ranks must be nonnegative integers")
        _require_int32(rank, "source rank")
        normalized[source] = rank
    if not normalized:
        raise ValueError("source_rank_by_game must not be empty")
    ranks = sorted(normalized.values())
    if ranks != list(range(len(normalized))) or len(set(ranks)) != len(ranks):
        raise ValueError("source ranks must be unique and contiguous from zero")
    sources = [""] * len(normalized)
    for source, rank in normalized.items():
        sources[rank] = source
    return normalized, tuple(sources)


def _validate_year_artifact(artifact: YearSparseArtifact) -> None:
    if isinstance(artifact.year, bool) or not isinstance(artifact.year, int):
        raise TypeError("artifact year must be an integer")
    if not isinstance(artifact.matrix, csr_matrix):
        raise TypeError("artifact matrix must be scipy.sparse.csr_matrix")
    rows, columns = artifact.matrix.shape
    if rows < 1:
        raise ValueError("artifact must contain at least one row")
    if columns != len(artifact.feature_names):
        raise ValueError("feature names do not align with matrix columns")
    if len(set(artifact.feature_names)) != len(artifact.feature_names) or any(
        not isinstance(name, str) or not name for name in artifact.feature_names
    ):
        raise ValueError("feature names must be unique non-empty strings")
    if not np.all(np.isfinite(artifact.matrix.data)):
        raise ValueError("artifact feature values must be finite")
    _validate_row_array(artifact.y_structural, rows, "y_structural", np.bool_)
    _validate_row_array(artifact.y_ron, rows, "y_ron", np.bool_)
    _validate_row_array(artifact.weights, rows, "weights", np.float64)
    if not np.all(np.isfinite(artifact.weights)) or np.any(artifact.weights <= 0):
        raise ValueError("artifact weights must be finite and positive")
    _validate_row_array(artifact.group_indices, rows, "group_indices", np.int32)
    _validate_compact_indices(artifact.group_indices, artifact.group_ids, "group")
    _validate_row_array(artifact.cluster_indices, rows, "cluster_indices", np.int32)
    _validate_compact_indices(artifact.cluster_indices, artifact.cluster_ids, "cluster")
    _validate_row_array(artifact.turn_bins, rows, "turn_bins", np.uint8)
    if np.any(artifact.turn_bins >= len(DECISION_TURN_BINS)):
        raise ValueError("artifact contains an unknown turn-bin code")
    _validate_row_array(artifact.source_ranks, rows, "source_ranks", np.int32)
    if not artifact.source_ids_by_rank or len(set(artifact.source_ids_by_rank)) != len(
        artifact.source_ids_by_rank
    ):
        raise ValueError("source_ids_by_rank must be non-empty and unique")
    if any(
        not isinstance(source, str) or not source
        for source in artifact.source_ids_by_rank
    ):
        raise ValueError("source IDs must be non-empty strings")
    if np.any(artifact.source_ranks < 0) or np.any(
        artifact.source_ranks >= len(artifact.source_ids_by_rank)
    ):
        raise ValueError("source ranks are outside source_ids_by_rank")
    if len(set(key for key, _value in artifact.provenance)) != len(
        artifact.provenance
    ) or any(
        not key or not isinstance(key, str) for key, _value in artifact.provenance
    ):
        raise ValueError("provenance keys must be unique non-empty strings")
    if any(not isinstance(value, str) for _key, value in artifact.provenance):
        raise ValueError("provenance values must be strings")


def _validate_row_array(
    values: NDArray[np.generic],
    rows: int,
    name: str,
    dtype: np.dtype[object] | type[np.generic],
) -> None:
    if not isinstance(values, np.ndarray) or values.ndim != 1 or len(values) != rows:
        raise ValueError(f"{name} must be a one-dimensional row-aligned ndarray")
    if values.dtype != np.dtype(dtype):
        raise ValueError(f"{name} must have dtype {np.dtype(dtype)}")


def _validate_compact_indices(
    indices: NDArray[np.int32],
    identifiers: tuple[str, ...],
    name: str,
) -> None:
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{name} identifiers must be non-empty and unique")
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError(f"{name} identifiers must be non-empty strings")
    observed = np.unique(indices)
    expected = np.arange(len(identifiers), dtype=np.int32)
    if not np.array_equal(observed, expected):
        raise ValueError(f"{name} indices must be compact and cover every identifier")


def _validate_vocabulary(vocabulary: Sequence[str]) -> tuple[str, ...]:
    names = tuple(vocabulary)
    if tuple(sorted(set(names))) != names or any(
        not isinstance(name, str) or not name for name in names
    ):
        raise ValueError("global vocabulary must be sorted, unique, non-empty strings")
    return names


def _validate_optimizer_options(
    *,
    l2: float,
    max_iterations: int,
    gradient_tolerance: float,
    function_tolerance: float,
    max_line_search_steps: int,
    history_size: int,
) -> None:
    if not isfinite(l2) or l2 < 0:
        raise ValueError("l2 must be finite and nonnegative")
    for name, value in (
        ("max_iterations", max_iterations),
        ("max_line_search_steps", max_line_search_steps),
        ("history_size", history_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (
        ("gradient_tolerance", gradient_tolerance),
        ("function_tolerance", function_tolerance),
    ):
        if not isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")


def _make_artifact_arrays_read_only(artifact: YearSparseArtifact) -> None:
    for values in (
        artifact.matrix.data,
        artifact.matrix.indices,
        artifact.matrix.indptr,
        artifact.y_structural,
        artifact.y_ron,
        artifact.weights,
        artifact.group_indices,
        artifact.cluster_indices,
        artifact.turn_bins,
        artifact.source_ranks,
    ):
        values.setflags(write=False)


def _array_view(
    values: array[object], dtype: np.dtype[object] | type[np.generic]
) -> NDArray:
    return np.frombuffer(values, dtype=dtype)


def _require_int32(value: int, name: str) -> None:
    if value > np.iinfo(np.int32).max:
        raise OverflowError(f"{name} exceeds int32 range")


def _infinity_norm(values: NDArray[np.float64]) -> float:
    return 0.0 if len(values) == 0 else float(np.max(np.abs(values)))


def _sorted_feature_position(
    feature_names: tuple[str, ...], feature_name: str
) -> int | None:
    position = int(np.searchsorted(np.asarray(feature_names), feature_name))
    if position >= len(feature_names) or feature_names[position] != feature_name:
        return None
    return position
