"""Evaluate combo danger features on fixed 2020-2023 expanding windows."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from gc import collect as collect_garbage
from hashlib import sha256
from itertools import chain
from pathlib import Path
from typing import Any

from mahjong_analysis.combo_bootstrap import (
    GroupContrast,
    JointBootstrapResult,
    joint_paired_game_cluster_bootstrap,
)
from mahjong_analysis.combo_dataset import (
    DEVELOPMENT_YEARS,
    SELECTION_ALGORITHM,
    CandidateDataset,
    StableYearSample,
    collect_candidate_dataset,
    expanding_year_splits,
    file_sha256,
    select_stable_year_sample,
    selection_digest,
)
from mahjong_analysis.combo_prediction import (
    DECISION_TURN_BINS,
    FEATURE_SETS,
    BinaryMetrics,
    ComboCandidateRow,
    FeatureSet,
    LabelName,
    SparseExample,
    decision_turn_bin,
    evaluate_binary_predictions,
    fit_sparse_logistic,
    paired_game_cluster_bootstrap,
    sparse_examples,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw"
LABELS: tuple[LabelName, ...] = ("structural_wait", "ron_eligible")
COMPARISONS: tuple[tuple[str, FeatureSet, FeatureSet], ...] = (
    (
        "conventional_simple_minus_conventional",
        "conventional",
        "conventional_simple",
    ),
    (
        "conventional_simple_percentile_minus_conventional",
        "conventional",
        "conventional_simple_percentile",
    ),
    (
        "visibility_local_saturated_simple_minus_visibility_local_saturated",
        "visibility_local_saturated",
        "visibility_local_saturated_simple",
    ),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fixed-sample expanding-window combo evaluation using only 2020-2023."
        )
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--files-per-year", type=_positive_int, default=50)
    parser.add_argument("--sample-seed", type=int, default=20260923)
    parser.add_argument("--epochs", type=_positive_int, default=100)
    parser.add_argument("--learning-rate", type=_positive_float, default=0.05)
    parser.add_argument("--l2", type=_nonnegative_float, default=0.001)
    parser.add_argument("--bootstrap-replicates", type=_nonnegative_int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260923)
    parser.add_argument(
        "--test-year",
        type=int,
        choices=DEVELOPMENT_YEARS[1:],
        action="append",
        dest="test_years",
        help="Prediction year to run; repeat for multiple years (default: all).",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest-input", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Select or verify the sample manifest without extracting candidate rows.",
    )
    args = parser.parse_args(argv)
    if args.test_years is not None and len(set(args.test_years)) != len(
        args.test_years
    ):
        parser.error("--test-year values must be unique")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prediction_years = tuple(sorted(args.test_years or DEVELOPMENT_YEARS[1:]))
    if args.manifest_input is None:
        samples = _select_samples(args)
        manifest = _build_manifest(samples, args)
    else:
        manifest = _read_json_object(args.manifest_input)
        samples = _samples_from_manifest(manifest, args)
    if args.manifest_output is not None:
        _write_json(args.manifest_output, manifest)
    if args.manifest_only:
        if args.manifest_output is None:
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    datasets: dict[int, CandidateDataset] = {}
    required_years = tuple(
        year for year in DEVELOPMENT_YEARS if year <= max(prediction_years)
    )
    for year in required_years:
        sample = samples[year]
        print(
            f"[extract] {year}: {len(sample.paths)} files", file=sys.stderr, flush=True
        )
        datasets[year] = collect_candidate_dataset(
            args.raw_root,
            year,
            sample.paths,
        )

    manifest_digest = _canonical_json_sha256(manifest)
    pooled_examples: dict[LabelName, list[SparseExample]] = defaultdict(list)
    pooled_turn_bins: dict[LabelName, list[str]] = defaultdict(list)
    pooled_predictions: dict[LabelName, dict[FeatureSet, list[float]]] = {
        label: {feature_set: [] for feature_set in FEATURE_SETS} for label in LABELS
    }
    fold_documents: dict[str, Any] = {}
    for train_years, test_year in expanding_year_splits():
        if test_year not in prediction_years:
            continue
        train_rows = tuple(
            chain.from_iterable(datasets[year].rows for year in train_years)
        )
        test_rows = datasets[test_year].rows
        _assert_disjoint_games(train_rows, test_rows)
        test_turn_bins = tuple(
            decision_turn_bin(row.decision.decision_discard_number) for row in test_rows
        )
        print(
            f"[fold] train={','.join(map(str, train_years))} test={test_year}",
            file=sys.stderr,
            flush=True,
        )
        results: dict[str, Any] = {}
        for label in LABELS:
            label_document, evaluation_examples, predictions_by_model = _evaluate_label(
                train_rows,
                test_rows,
                label=label,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                l2=args.l2,
                bootstrap_replicates=args.bootstrap_replicates,
                bootstrap_seed=args.bootstrap_seed + test_year,
                progress_prefix=f"test={test_year}",
                turn_bins=test_turn_bins,
            )
            results[label] = label_document
            pooled_examples[label].extend(evaluation_examples)
            pooled_turn_bins[label].extend(test_turn_bins)
            for feature_set, predictions in predictions_by_model.items():
                pooled_predictions[label][feature_set].extend(predictions)
        fold_documents[str(test_year)] = {
            "train_years": list(train_years),
            "test_year": test_year,
            "train_counts": _aggregate_counts(datasets, train_years),
            "test_counts": datasets[test_year].counts,
            "results": results,
        }

    pooled_results = {
        label: _pooled_label_document(
            tuple(pooled_examples[label]),
            {
                feature_set: tuple(pooled_predictions[label][feature_set])
                for feature_set in FEATURE_SETS
            },
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            turn_bins=tuple(pooled_turn_bins[label]),
            stratum_by_cluster={
                example.cluster_id: example.cluster_id.split("/", 1)[0]
                for example in pooled_examples[label]
            },
        )
        for label in LABELS
    }
    report = {
        "scope": {
            "development_years": list(DEVELOPMENT_YEARS),
            "prediction_years": list(prediction_years),
            "files_per_year": args.files_per_year,
            "sample_seed": args.sample_seed,
            "selection_algorithm": SELECTION_ALGORITHM,
            "manifest_canonical_sha256": manifest_digest,
            "validation_2024_touched": False,
            "holdout_2025_touched": False,
            "status": ("development expanding-window evaluation; not final validation"),
            "turn_bin_definition": {
                "field": "decision_discard_number",
                "ordered_bins": list(DECISION_TURN_BINS),
                "bounds": "1-6, 7-9, 10-12, 13+ (inclusive)",
            },
        },
        "optimizer": {
            "type": "dependency-free deterministic full-batch Adam logistic regression",
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
            "candidate_weight": "1 / candidate_count_in_decision",
            "bootstrap_inference": {
                "cluster": "game",
                "paired": True,
                "model_refit_each_replicate": False,
                "interval": "pointwise percentile 95%",
                "turn_bin_intervals": "shared game draws across all fixed bins",
                "pooled_stratification": "prediction_year",
                "turn_bin_contrast": "delta(13+) - delta(1-6)",
            },
        },
        "data_by_year": {
            str(year): {
                "candidate_source_files": samples[year].candidate_count,
                "counts": datasets[year].counts,
            }
            for year in required_years
        },
        "folds": fold_documents,
        "pooled_selected_prediction_years": pooled_results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


def _evaluate_label(
    train_rows: Sequence[ComboCandidateRow],
    test_rows: Sequence[ComboCandidateRow],
    *,
    label: LabelName,
    epochs: int,
    learning_rate: float,
    l2: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    progress_prefix: str,
    turn_bins: Sequence[str],
) -> tuple[
    dict[str, Any], tuple[SparseExample, ...], dict[FeatureSet, tuple[float, ...]]
]:
    metrics_by_model: dict[FeatureSet, BinaryMetrics] = {}
    predictions_by_model: dict[FeatureSet, tuple[float, ...]] = {}
    model_documents: dict[str, Any] = {}
    evaluation_examples: tuple[SparseExample, ...] | None = None
    for feature_set in FEATURE_SETS:
        print(
            f"[fit] {progress_prefix} label={label} features={feature_set}",
            file=sys.stderr,
            flush=True,
        )
        train_examples = sparse_examples(train_rows, feature_set, label)
        test_examples = sparse_examples(test_rows, feature_set, label)
        if evaluation_examples is None:
            evaluation_examples = test_examples
        else:
            _assert_example_alignment(evaluation_examples, test_examples)
        model = fit_sparse_logistic(
            train_examples,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
        predictions = model.predict(test_examples)
        metrics = evaluate_binary_predictions(test_examples, predictions)
        metrics_by_model[feature_set] = metrics
        predictions_by_model[feature_set] = predictions
        model_documents[feature_set] = {
            "metrics": asdict(metrics),
            "feature_count": len(model.weights),
            "intercept": model.intercept,
            "epochs_run": model.epochs_run,
            "final_largest_gradient": model.final_largest_gradient,
            "largest_absolute_coefficients": _largest_coefficients(model.weights),
        }
        del train_examples, test_examples, model
        collect_garbage()
    assert evaluation_examples is not None
    return (
        _comparison_document(
            evaluation_examples,
            metrics_by_model,
            predictions_by_model,
            model_documents=model_documents,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
            turn_bins=turn_bins,
        ),
        evaluation_examples,
        predictions_by_model,
    )


def _pooled_label_document(
    examples: tuple[SparseExample, ...],
    predictions_by_model: dict[FeatureSet, tuple[float, ...]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    turn_bins: Sequence[str],
    stratum_by_cluster: dict[str, str],
) -> dict[str, Any]:
    metrics_by_model = {
        feature_set: evaluate_binary_predictions(examples, predictions)
        for feature_set, predictions in predictions_by_model.items()
    }
    return _comparison_document(
        examples,
        metrics_by_model,
        predictions_by_model,
        model_documents={
            feature_set: {"metrics": asdict(metrics)}
            for feature_set, metrics in metrics_by_model.items()
        },
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        turn_bins=turn_bins,
        stratum_by_cluster=stratum_by_cluster,
    )


def _comparison_document(
    examples: Sequence[SparseExample],
    metrics_by_model: dict[FeatureSet, BinaryMetrics],
    predictions_by_model: dict[FeatureSet, Sequence[float]],
    *,
    model_documents: dict[str, Any],
    bootstrap_replicates: int,
    bootstrap_seed: int,
    turn_bins: Sequence[str] | None = None,
    stratum_by_cluster: dict[str, str] | None = None,
) -> dict[str, Any]:
    joint_bootstrap = (
        joint_paired_game_cluster_bootstrap(
            examples,
            predictions_by_model,
            {
                comparison_name: (base, challenger)
                for comparison_name, base, challenger in COMPARISONS
            },
            group_ids=turn_bins,
            group_order=DECISION_TURN_BINS if turn_bins is not None else (),
            stratum_by_cluster=stratum_by_cluster,
            contrasts=(
                GroupContrast("late_minus_early", "13+", "1-6"),
            )
            if turn_bins is not None
            else (),
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
        if bootstrap_replicates
        else None
    )
    document = {
        "models": model_documents,
        "paired_comparisons": _paired_comparison_documents(
            examples,
            metrics_by_model,
            predictions_by_model,
            bootstrap_replicates=(
                0 if joint_bootstrap is not None else bootstrap_replicates
            ),
            bootstrap_seed=bootstrap_seed,
        ),
    }
    if joint_bootstrap is not None:
        _replace_overall_bootstraps(document["paired_comparisons"], joint_bootstrap)
        document["bootstrap_design"] = {
            "cluster": "game",
            "shared_across_comparisons_and_turn_bins": True,
            "stratified": joint_bootstrap.stratified,
            "stratum_cluster_counts": dict(joint_bootstrap.stratum_cluster_counts),
            "model_refit_each_replicate": False,
        }
    if turn_bins is not None:
        document["by_decision_turn_bin"] = _turn_bin_documents(
            examples,
            predictions_by_model,
            turn_bins,
            bootstrap_replicates=0,
            bootstrap_seed=bootstrap_seed,
        )
        if joint_bootstrap is not None:
            document["turn_bin_contrasts"] = _inject_joint_turn_bootstrap(
                document["by_decision_turn_bin"],
                joint_bootstrap,
            )
    return document


def _bootstrap_metric_document(
    metrics: Any,
    joint: JointBootstrapResult,
    *,
    cluster_count: int,
) -> dict[str, Any]:
    return {
        "cluster_count": cluster_count,
        "requested_replicates": joint.requested_replicates,
        "seed": joint.seed,
        **asdict(metrics),
    }


def _replace_overall_bootstraps(
    comparison_documents: dict[str, Any],
    joint: JointBootstrapResult,
) -> None:
    for comparison in joint.comparisons:
        comparison_documents[comparison.name]["game_cluster_bootstrap"] = (
            _bootstrap_metric_document(
                comparison.overall,
                joint,
                cluster_count=joint.cluster_count,
            )
        )


def _inject_joint_turn_bootstrap(
    turn_documents: dict[str, Any],
    joint: JointBootstrapResult,
) -> dict[str, Any]:
    contrasts: dict[str, Any] = {}
    for comparison in joint.comparisons:
        for group in comparison.by_group:
            if turn_documents[group.group]["paired_comparisons"] is not None:
                turn_documents[group.group]["paired_comparisons"][comparison.name][
                    "game_cluster_bootstrap"
                ] = _bootstrap_metric_document(
                    group.metrics,
                    joint,
                    cluster_count=group.cluster_count,
                )
        for contrast in comparison.contrasts:
            contrast_document = contrasts.setdefault(
                contrast.name,
                {
                    "minuend": contrast.left_group,
                    "subtrahend": contrast.right_group,
                    "definition": (
                        f"delta({contrast.left_group}) - "
                        f"delta({contrast.right_group})"
                    ),
                    "paired_comparisons": {},
                },
            )
            contrast_document["paired_comparisons"][comparison.name] = (
                _bootstrap_metric_document(
                    contrast.metrics,
                    joint,
                    cluster_count=joint.cluster_count,
                )
            )
    return contrasts


def _paired_comparison_documents(
    examples: Sequence[SparseExample],
    metrics_by_model: dict[FeatureSet, BinaryMetrics],
    predictions_by_model: dict[FeatureSet, Sequence[float]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    return {
        comparison_name: {
            "base": base,
            "challenger": challenger,
            "delta": _metric_delta(
                metrics_by_model[base],
                metrics_by_model[challenger],
            ),
            "game_cluster_bootstrap": asdict(
                paired_game_cluster_bootstrap(
                    examples,
                    predictions_by_model[base],
                    predictions_by_model[challenger],
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed,
                )
            )
            if bootstrap_replicates
            else None,
        }
        for comparison_name, base, challenger in COMPARISONS
    }


def _turn_bin_documents(
    examples: Sequence[SparseExample],
    predictions_by_model: dict[FeatureSet, Sequence[float]],
    turn_bins: Sequence[str],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if len(examples) != len(turn_bins):
        raise ValueError("turn bins must align with evaluation examples")
    if any(turn_bin not in DECISION_TURN_BINS for turn_bin in turn_bins):
        raise ValueError("unknown decision turn bin")
    documents: dict[str, Any] = {}
    for bin_index, bin_name in enumerate(DECISION_TURN_BINS):
        positions = tuple(
            index for index, value in enumerate(turn_bins) if value == bin_name
        )
        if not positions:
            documents[bin_name] = {
                "candidate_rows": 0,
                "decision_count": 0,
                "game_count": 0,
                "models": None,
                "paired_comparisons": None,
            }
            continue
        bin_examples = tuple(examples[index] for index in positions)
        bin_predictions = {
            feature_set: tuple(predictions[index] for index in positions)
            for feature_set, predictions in predictions_by_model.items()
        }
        bin_metrics = {
            feature_set: evaluate_binary_predictions(bin_examples, predictions)
            for feature_set, predictions in bin_predictions.items()
        }
        documents[bin_name] = {
            "candidate_rows": len(bin_examples),
            "decision_count": len({example.group_id for example in bin_examples}),
            "game_count": len({example.cluster_id for example in bin_examples}),
            "models": {
                feature_set: {"metrics": asdict(metrics)}
                for feature_set, metrics in bin_metrics.items()
            },
            "paired_comparisons": _paired_comparison_documents(
                bin_examples,
                bin_metrics,
                bin_predictions,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed + 10_000 * (bin_index + 1),
            ),
        }
    return documents


def _build_manifest(
    samples: dict[int, StableYearSample],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "selection_algorithm": SELECTION_ALGORITHM,
        "sample_seed": args.sample_seed,
        "files_per_year": args.files_per_year,
        "development_years": list(DEVELOPMENT_YEARS),
        "validation_2024_touched": False,
        "holdout_2025_touched": False,
        "years": {
            str(year): {
                "candidate_count": sample.candidate_count,
                "candidate_population_sha256": sample.candidate_population_sha256,
                "selected": [
                    {
                        "relative_source": source,
                        "selection_sha256": digest,
                        "content_sha256": content_digest,
                    }
                    for source, digest, content_digest in zip(
                        sample.relative_sources,
                        sample.selection_digests,
                        sample.source_content_sha256,
                        strict=True,
                    )
                ],
            }
            for year, sample in sorted(samples.items())
        },
    }


def _select_samples(args: argparse.Namespace) -> dict[int, StableYearSample]:
    samples: dict[int, StableYearSample] = {}
    for year in DEVELOPMENT_YEARS:
        print(f"[sample] {year}", file=sys.stderr, flush=True)
        samples[year] = select_stable_year_sample(
            args.raw_root,
            year,
            args.files_per_year,
            seed=args.sample_seed,
        )
    return samples


def _samples_from_manifest(
    manifest: dict[str, Any],
    args: argparse.Namespace,
) -> dict[int, StableYearSample]:
    schema_version = manifest.get("schema_version")
    if schema_version not in {1, 2}:
        raise ValueError(f"unsupported manifest schema_version: {schema_version!r}")
    expected_header = {
        "selection_algorithm": SELECTION_ALGORITHM,
        "sample_seed": args.sample_seed,
        "files_per_year": args.files_per_year,
        "development_years": list(DEVELOPMENT_YEARS),
        "validation_2024_touched": False,
        "holdout_2025_touched": False,
    }
    for key, expected in expected_header.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"manifest {key!r} mismatch: expected {expected!r}, "
                f"got {manifest.get(key)!r}"
            )
    years_document = manifest.get("years")
    if not isinstance(years_document, dict) or set(years_document) != {
        str(year) for year in DEVELOPMENT_YEARS
    }:
        raise ValueError("manifest years must exactly match development years")

    canonical_root = args.raw_root.resolve()
    samples: dict[int, StableYearSample] = {}
    for year in DEVELOPMENT_YEARS:
        year_document = years_document[str(year)]
        if not isinstance(year_document, dict):
            raise ValueError(f"manifest year {year} must be an object")
        candidate_count = year_document.get("candidate_count")
        population_digest = year_document.get("candidate_population_sha256")
        selected = year_document.get("selected")
        if (
            isinstance(candidate_count, bool)
            or not isinstance(candidate_count, int)
            or candidate_count < args.files_per_year
        ):
            raise ValueError(f"manifest candidate_count is invalid for {year}")
        if not isinstance(selected, list) or len(selected) != args.files_per_year:
            raise ValueError(f"manifest selected count is invalid for {year}")
        paths: list[Path] = []
        sources: list[str] = []
        digests: list[str] = []
        content_digests: list[str] = []
        if schema_version == 2 and (
            not isinstance(population_digest, str) or len(population_digest) != 64
        ):
            raise ValueError(f"manifest population digest is invalid for {year}")
        for entry in selected:
            if not isinstance(entry, dict):
                raise ValueError(f"manifest selected entry is invalid for {year}")
            source = entry.get("relative_source")
            digest = entry.get("selection_sha256")
            content_digest = entry.get("content_sha256")
            if not isinstance(source, str) or not isinstance(digest, str):
                raise ValueError(f"manifest selected entry is invalid for {year}")
            if schema_version == 2 and (
                not isinstance(content_digest, str) or len(content_digest) != 64
            ):
                raise ValueError(f"manifest content digest is invalid for {source}")
            if digest != selection_digest(source, args.sample_seed):
                raise ValueError(f"manifest selection digest mismatch: {source}")
            path = canonical_root.joinpath(*source.split("/")).resolve()
            try:
                path.relative_to(canonical_root / str(year))
            except ValueError as error:
                raise ValueError(
                    f"manifest source is outside year {year}: {source}"
                ) from error
            if not path.is_file():
                raise FileNotFoundError(f"manifest source does not exist: {path}")
            if schema_version == 2:
                actual_content_digest = file_sha256(path)
                if actual_content_digest != content_digest:
                    raise ValueError(f"manifest content digest mismatch: {source}")
                content_digests.append(content_digest)
            paths.append(path)
            sources.append(source)
            digests.append(digest)
        if len(set(sources)) != len(sources):
            raise ValueError(f"manifest contains duplicate sources for {year}")
        if tuple(digests) != tuple(sorted(digests)):
            raise ValueError(f"manifest selection digests are not ordered for {year}")
        if schema_version == 2:
            actual = select_stable_year_sample(
                args.raw_root,
                year,
                args.files_per_year,
                seed=args.sample_seed,
            )
            if actual.candidate_count != candidate_count:
                raise ValueError(f"manifest candidate_count mismatch for {year}")
            if actual.candidate_population_sha256 != population_digest:
                raise ValueError(f"manifest population digest mismatch for {year}")
            if actual.relative_sources != tuple(sources):
                raise ValueError(f"manifest does not contain lowest hashes for {year}")
            if actual.selection_digests != tuple(digests):
                raise ValueError(f"manifest selection order mismatch for {year}")
            if actual.source_content_sha256 != tuple(content_digests):
                raise ValueError(f"manifest selected content mismatch for {year}")
        samples[year] = StableYearSample(
            year=year,
            candidate_count=candidate_count,
            paths=tuple(paths),
            relative_sources=tuple(sources),
            selection_digests=tuple(digests),
            candidate_population_sha256=population_digest or "",
            source_content_sha256=tuple(content_digests),
        )
    return samples


def _read_json_object(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("manifest root must be an object")
    return document


def _canonical_json_sha256(document: dict[str, Any]) -> str:
    canonical = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256(canonical).hexdigest()


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _aggregate_counts(
    datasets: dict[int, CandidateDataset],
    years: Sequence[int],
) -> dict[str, int]:
    total: Counter[str] = Counter()
    for year in years:
        total.update(datasets[year].counts)
    return dict(sorted(total.items()))


def _assert_disjoint_games(
    train_rows: Sequence[ComboCandidateRow],
    test_rows: Sequence[ComboCandidateRow],
) -> None:
    train_games = {row.game_id for row in train_rows}
    test_games = {row.game_id for row in test_rows}
    overlap = train_games & test_games
    if overlap:
        example = min(overlap)
        raise ValueError(f"train/test game overlap: {example}")


def _assert_example_alignment(
    reference: Sequence[SparseExample],
    candidate: Sequence[SparseExample],
) -> None:
    if len(reference) != len(candidate):
        raise ValueError("feature sets produced different example counts")
    for left, right in zip(reference, candidate, strict=True):
        if (
            left.row_id,
            left.group_id,
            left.cluster_id,
            left.label,
            left.weight,
        ) != (
            right.row_id,
            right.group_id,
            right.cluster_id,
            right.label,
            right.weight,
        ):
            raise ValueError("feature sets produced misaligned examples")


def _metric_delta(
    base: BinaryMetrics, challenger: BinaryMetrics
) -> dict[str, float | None]:
    return {
        "auc": _optional_difference(challenger.auc, base.auc),
        "log_loss": challenger.log_loss - base.log_loss,
        "brier": challenger.brier - base.brier,
        "ece": challenger.ece - base.ece,
        "macro_concordance": _optional_difference(
            challenger.macro_concordance,
            base.macro_concordance,
        ),
    }


def _optional_difference(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else left - right


def _largest_coefficients(
    weights: dict[str, float],
    *,
    limit: int = 12,
) -> list[dict[str, float | str]]:
    return [
        {"feature": name, "coefficient": coefficient}
        for name, coefficient in sorted(
            weights.items(),
            key=lambda item: (-abs(item[1]), item[0]),
        )[:limit]
    ]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
