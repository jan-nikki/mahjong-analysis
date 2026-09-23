from dataclasses import asdict, replace
from random import Random

import pytest

from mahjong_analysis.combo_bootstrap import (
    GroupContrast,
    joint_paired_game_cluster_bootstrap,
)
from mahjong_analysis.combo_prediction import (
    SparseExample,
    evaluate_binary_predictions,
)


def _example(
    row_id: str,
    decision_id: str,
    cluster_id: str,
    label: bool,
) -> SparseExample:
    return SparseExample(
        row_id=row_id,
        group_id=decision_id,
        cluster_id=cluster_id,
        label=label,
        weight=0.5,
        features=(),
    )


def _parallel_group_data() -> tuple[
    tuple[SparseExample, ...], dict[str, tuple[float, ...]]
]:
    examples = []
    base = []
    challenger = []
    reverse = []
    for cluster_index, cluster in enumerate(("g1", "g2")):
        challenger_pair = (0.8, 0.2) if cluster_index == 0 else (0.4, 0.6)
        for turn_bin in ("early", "late"):
            decision = f"{cluster}:{turn_bin}"
            for label, base_value, challenger_value in zip(
                (True, False),
                (0.6, 0.4),
                challenger_pair,
                strict=True,
            ):
                examples.append(
                    _example(
                        f"{decision}:{int(label)}",
                        decision,
                        cluster,
                        label,
                    )
                )
                base.append(base_value)
                challenger.append(challenger_value)
                reverse.append(1 - challenger_value)
    return tuple(examples), {
        "base": tuple(base),
        "challenger": tuple(challenger),
        "reverse": tuple(reverse),
    }


def test_shared_draw_makes_parallel_group_contrast_exactly_zero() -> None:
    examples, predictions = _parallel_group_data()
    result = joint_paired_game_cluster_bootstrap(
        examples,
        predictions,
        {
            "main": ("base", "challenger"),
            "reverse": ("base", "reverse"),
        },
        group_ids=("early", "early", "late", "late") * 2,
        group_order=("early", "late"),
        contrasts=(GroupContrast("late_minus_early", "late", "early"),),
        replicates=40,
        seed=8,
    )

    assert len(result.comparisons) == 2
    for comparison in result.comparisons:
        contrast = comparison.contrasts[0]
        for interval in asdict(contrast.metrics).values():
            assert interval["point"] == pytest.approx(0.0)
            assert interval["lower"] == pytest.approx(0.0)
            assert interval["upper"] == pytest.approx(0.0)
            assert interval["valid_replicates"] == 40


def test_stratification_keeps_each_stratums_cluster_count_fixed() -> None:
    examples = (
        _example("a+", "da", "a", True),
        _example("a-", "da", "a", False),
        _example("b+", "db", "b", True),
        _example("b-", "db", "b", False),
    )
    result = joint_paired_game_cluster_bootstrap(
        examples,
        {
            "base": (0.55, 0.45, 0.55, 0.45),
            "challenger": (0.9, 0.1, 0.51, 0.49),
        },
        {"main": ("base", "challenger")},
        stratum_by_cluster={"a": "2021", "b": "2022"},
        replicates=30,
        seed=4,
    )

    assert result.stratified
    assert result.stratum_cluster_counts == (("2021", 1), ("2022", 1))
    log_loss = result.comparisons[0].overall.log_loss
    assert log_loss.valid_replicates == 30
    assert log_loss.lower == pytest.approx(log_loss.point)
    assert log_loss.upper == pytest.approx(log_loss.point)


def test_pooled_auc_is_valid_when_each_stratum_has_only_one_class() -> None:
    examples = (
        _example("p1", "p1", "positive-1", True),
        _example("p2", "p2", "positive-2", True),
        _example("n1", "n1", "negative-1", False),
        _example("n2", "n2", "negative-2", False),
    )
    strata = {
        "positive-1": "positive-only",
        "positive-2": "positive-only",
        "negative-1": "negative-only",
        "negative-2": "negative-only",
    }
    result = joint_paired_game_cluster_bootstrap(
        examples,
        {
            "base": (0.6, 0.4, 0.5, 0.3),
            "challenger": (0.8, 0.7, 0.2, 0.1),
        },
        {"main": ("base", "challenger")},
        stratum_by_cluster=strata,
        replicates=50,
        seed=9,
    )

    auc = result.comparisons[0].overall.auc
    assert auc.point is not None
    assert auc.valid_replicates == 50
    assert auc.lower is not None
    assert auc.upper is not None


def test_vectorized_bootstrap_matches_brute_force_duplicated_clusters() -> None:
    examples = tuple(
        example
        for cluster in ("g1", "g2", "g3")
        for example in (
            _example(f"{cluster}+", cluster, cluster, True),
            _example(f"{cluster}-", cluster, cluster, False),
        )
    )
    base = (0.7, 0.3, 0.4, 0.6, 0.5, 0.5)
    challenger = (0.8, 0.2, 0.6, 0.4, 0.45, 0.55)
    replicates = 30
    seed = 17
    result = joint_paired_game_cluster_bootstrap(
        examples,
        {"base": base, "challenger": challenger},
        {"main": ("base", "challenger")},
        replicates=replicates,
        seed=seed,
    )

    metric_names = ("auc", "log_loss", "brier", "ece", "macro_concordance")
    base_point = evaluate_binary_predictions(examples, base)
    challenger_point = evaluate_binary_predictions(examples, challenger)
    point_differences = {
        metric: getattr(challenger_point, metric) - getattr(base_point, metric)
        for metric in metric_names
    }
    samples = {metric: [] for metric in metric_names}
    random = Random(seed)
    saw_duplicate = False
    cluster_ids = ("g1", "g2", "g3")
    for replicate in range(replicates):
        draws = [random.randrange(len(cluster_ids)) for _ in cluster_ids]
        saw_duplicate |= len(set(draws)) < len(draws)
        expanded_examples = []
        expanded_base = []
        expanded_challenger = []
        for copy_number, cluster_index in enumerate(draws):
            cluster = cluster_ids[cluster_index]
            for row_index, example in enumerate(examples):
                if example.cluster_id != cluster:
                    continue
                suffix = f":bootstrap={replicate}:{copy_number}"
                expanded_examples.append(
                    replace(
                        example,
                        row_id=example.row_id + suffix,
                        group_id=example.group_id + suffix,
                        cluster_id=example.cluster_id + suffix,
                    )
                )
                expanded_base.append(base[row_index])
                expanded_challenger.append(challenger[row_index])
        base_metrics = evaluate_binary_predictions(expanded_examples, expanded_base)
        challenger_metrics = evaluate_binary_predictions(
            expanded_examples,
            expanded_challenger,
        )
        for metric in metric_names:
            samples[metric].append(
                getattr(challenger_metrics, metric) - getattr(base_metrics, metric)
            )

    assert saw_duplicate
    actual = result.comparisons[0].overall
    for metric in metric_names:
        interval = getattr(actual, metric)
        ordered = sorted(samples[metric])
        assert interval.point == pytest.approx(point_differences[metric], abs=1e-14)
        assert interval.lower == pytest.approx(
            _linear_percentile(ordered, 0.025),
            abs=1e-14,
        )
        assert interval.upper == pytest.approx(
            _linear_percentile(ordered, 0.975),
            abs=1e-14,
        )
        assert interval.valid_replicates == replicates


def test_results_are_deterministic_and_asdict_cleanly() -> None:
    examples, predictions = _parallel_group_data()
    kwargs = {
        "group_ids": ("early", "early", "late", "late") * 2,
        "group_order": ("early", "late", "empty"),
        "replicates": 25,
        "seed": 11,
    }
    first = joint_paired_game_cluster_bootstrap(
        examples,
        predictions,
        {"main": ("base", "challenger")},
        **kwargs,
    )
    second = joint_paired_game_cluster_bootstrap(
        examples,
        predictions,
        {"main": ("base", "challenger")},
        **kwargs,
    )

    assert first == second
    document = asdict(first)
    assert document["comparisons"][0]["name"] == "main"
    empty = document["comparisons"][0]["by_group"][2]
    assert empty["row_count"] == 0
    assert empty["metrics"]["auc"]["valid_replicates"] == 0


@pytest.mark.parametrize("replicates", [0, -1])
def test_replicates_must_be_positive(replicates: int) -> None:
    example = _example("r", "d", "g", True)
    with pytest.raises(ValueError, match="replicates must be positive"):
        joint_paired_game_cluster_bootstrap(
            (example,),
            {"base": (0.5,), "challenger": (0.6,)},
            {"main": ("base", "challenger")},
            replicates=replicates,
        )


def test_strict_input_validation() -> None:
    examples = (
        _example("r1", "d", "g1", True),
        _example("r2", "d", "g1", False),
    )
    predictions = {"base": (0.5, 0.5), "challenger": (0.6, 0.4)}
    comparisons = {"main": ("base", "challenger")}

    with pytest.raises(ValueError, match="do not align"):
        joint_paired_game_cluster_bootstrap(
            examples,
            {"base": (0.5,), "challenger": (0.6, 0.4)},
            comparisons,
            replicates=1,
        )
    with pytest.raises(ValueError, match="stratum cluster keys mismatch"):
        joint_paired_game_cluster_bootstrap(
            examples,
            predictions,
            comparisons,
            stratum_by_cluster={},
            replicates=1,
        )
    with pytest.raises(ValueError, match="spans multiple subgroups"):
        joint_paired_game_cluster_bootstrap(
            examples,
            predictions,
            comparisons,
            group_ids=("early", "late"),
            replicates=1,
        )
    with pytest.raises(ValueError, match="duplicate example row ID"):
        joint_paired_game_cluster_bootstrap(
            (examples[0], examples[0]),
            predictions,
            comparisons,
            replicates=1,
        )


def _linear_percentile(ordered: list[float], probability: float) -> float:
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction
