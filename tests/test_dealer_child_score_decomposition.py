import gzip
import hashlib
import json
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import pytest

import mahjong_analysis.dealer_child_score_decomposition as decomposition
from analysis.analyze_dealer_child_score_decomposition import (
    _load_and_validate_summary,
    iter_decomposition_records,
    main,
)
from mahjong_analysis.dealer_child_riichi_points import (
    PRIMARY_YEARS,
    SUPPORTED_YEARS,
    RiichiPointRecord,
)
from mahjong_analysis.dealer_child_score_decomposition import (
    CounterfactualState,
    DecompositionWinRecord,
    ScoreDecompositionBuilder,
    analyze_score_decomposition,
    decompose_score_gap,
    decomposition_document,
    decomposition_record_from_article_one,
    observed_group_means,
    render_decomposition_markdown,
    write_decomposition_outputs,
)
from mahjong_analysis.score_table import PaymentPattern, convert_payment_pattern


def _record(
    role: str,
    method: str,
    pattern: tuple[int, ...],
    *,
    year: int = 2025,
    index: int = 0,
    reach_type: str = "riichi",
) -> DecompositionWinRecord:
    converted = convert_payment_pattern(
        PaymentPattern(role, method, pattern)  # type: ignore[arg-type]
    )
    return DecompositionWinRecord(
        year=year,
        source_path=f"{year}/game-{index}.mjson",
        kyoku_index=index,
        actor=0 if role == "dealer" else 1,
        is_dealer=role == "dealer",
        reach_type=reach_type,  # type: ignore[arg-type]
        method=method,  # type: ignore[arg-type]
        observed_points=converted.observed_points,
        dealer_points=converted.dealer_points,
        nondealer_points=converted.nondealer_points,
    )


def test_known_shapley_example_and_efficiency() -> None:
    records = (
        *(
            _record("dealer", "tsumo", (4000, 4000, 4000), index=index)
            for index in range(3)
        ),
        _record("dealer", "ron", (2900,), index=3),
        _record("nondealer", "tsumo", (4000, 2000, 2000), index=4),
        *(_record("nondealer", "ron", (2000,), index=5 + index) for index in range(3)),
    )

    result = decompose_score_gap(records)

    assert result.observed_dealer_mean == 9725
    assert result.observed_nondealer_mean == 3500
    assert result.gap == 6225
    assert result.contribution_for("payment_rule") == 2450
    assert result.contribution_for("method_mix") == 3775
    assert result.contribution_for("within_method_score") == 0
    assert sum(value.points for value in result.contributions) == result.gap
    assert result.mean_for(CounterfactualState("dealer", "dealer", "dealer")) == 9725
    assert (
        result.mean_for(CounterfactualState("nondealer", "nondealer", "nondealer"))
        == 3500
    )


def test_identical_factor_distributions_have_zero_contribution() -> None:
    records = (
        _record("dealer", "tsumo", (4000, 4000, 4000), index=0),
        _record("dealer", "ron", (12_000,), index=1),
        _record("nondealer", "tsumo", (4000, 2000, 2000), index=2),
        _record("nondealer", "ron", (8000,), index=3),
    )

    result = decompose_score_gap(records)

    assert result.gap == 4000
    assert result.contribution_for("payment_rule") == 4000
    assert result.contribution_for("method_mix") == 0
    assert result.contribution_for("within_method_score") == 0


def test_decomposition_is_input_order_invariant() -> None:
    records = (
        _record("dealer", "tsumo", (4000, 4000, 4000), index=0),
        _record("dealer", "ron", (5800,), index=1),
        _record("nondealer", "tsumo", (2000, 1000, 1000), index=2),
        _record("nondealer", "ron", (2000,), index=3),
    )

    assert decompose_score_gap(records) == decompose_score_gap(reversed(records))


def test_streaming_builder_is_single_pass_order_invariant_and_multiyear() -> None:
    records = tuple(
        record
        for year in (2024, 2025)
        for record in (
            _record(
                "dealer",
                "tsumo",
                (4000, 4000, 4000),
                year=year,
                index=0,
            ),
            _record("dealer", "ron", (12_000,), year=year, index=1),
            _record(
                "nondealer",
                "tsumo",
                (4000, 2000, 2000),
                year=year,
                index=2,
            ),
            _record("nondealer", "ron", (8000,), year=year, index=3),
        )
    )

    class SinglePass:
        def __init__(self, values: tuple[DecompositionWinRecord, ...]) -> None:
            self.values = values
            self.iterations = 0

        def __iter__(self):  # type: ignore[no-untyped-def]
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("stream was consumed more than once")
            return iter(self.values)

    forward = SinglePass(records)
    forward_builder = ScoreDecompositionBuilder([2025, 2024])
    for record in forward:
        forward_builder.add(record)
    forward_analysis = forward_builder.freeze()

    backward = SinglePass(tuple(reversed(records)))
    backward_builder = ScoreDecompositionBuilder([2024, 2025])
    for record in backward:
        backward_builder.add(record)
    backward_analysis = backward_builder.freeze()

    assert forward.iterations == backward.iterations == 1
    assert forward_analysis == backward_analysis
    assert forward_analysis.selected_years == (2024, 2025)
    assert [year for year, _ in forward_analysis.years] == [2024, 2025]
    assert all(
        period.all_riichi.dealer_wins == 2 and period.all_riichi.nondealer_wins == 2
        for _, period in forward_analysis.years
    )


def test_rounding_effect_is_exact_for_each_score_source_and_pooled() -> None:
    records = (
        _record("dealer", "tsumo", (500, 500, 500), index=0),
        _record("dealer", "ron", (1500,), index=1),
        _record("nondealer", "tsumo", (500, 300, 300), index=2),
        _record("nondealer", "ron", (1000,), index=3),
    )

    result = decompose_score_gap(records)
    document = decomposition_document(analyze_score_decomposition(records, [2025]))

    assert result.rounding_effect_for("dealer") == Fraction(-75)
    assert result.rounding_effect_for("nondealer") == Fraction(-75)
    assert result.rounding_effect_for("pooled") == Fraction(-75)
    assert document["periods"]["selected"]["all_riichi"]["rounding_effect"][
        "pooled"
    ] == {"value": -75.0, "numerator": -75, "denominator": 1}


def test_missing_role_method_stratum_fails_closed() -> None:
    records = (
        _record("dealer", "tsumo", (4000, 4000, 4000), index=0),
        _record("dealer", "ron", (12_000,), index=1),
        _record("nondealer", "ron", (8000,), index=2),
    )

    with pytest.raises(ValueError, match="distribution is missing"):
        decompose_score_gap(records)


def _article_one_record(
    *,
    actor: int,
    oya: int,
    method: str,
    points: int,
    index: int,
    year: int = 2025,
    reach_type: str = "riichi",
    source_path: str | None = None,
    kyoku_index: int | None = None,
    reach_event_index: int = 1,
    honba_awarded: bool = True,
) -> RiichiPointRecord:
    return RiichiPointRecord(
        year=year,
        source_path=source_path or f"{year}/source-{index:03d}.mjson",
        kyoku_index=index if kyoku_index is None else kyoku_index,
        start_kyoku_line=1,
        bakaze="E",
        kyoku=1,
        honba=2,
        oya=oya,
        actor=actor,
        reach_type=reach_type,  # type: ignore[arg-type]
        reach_event_index=reach_event_index,
        reach_accepted_event_index=3,
        reach_line=2,
        reach_accepted_line=4,
        outcome="win",
        win_method=method,  # type: ignore[arg-type]
        hora_event_index=5,
        hora_line=6,
        honba_awarded=honba_awarded,
        hand_points=points,
        settlement_gain=points + (600 if honba_awarded else 0),
    )


def _summary_group(
    records: tuple[RiichiPointRecord, ...],
    *,
    is_dealer: bool,
    normal_only: bool = False,
) -> dict[str, object]:
    selected = tuple(
        record
        for record in records
        if (record.actor == record.oya) is is_dealer
        and (not normal_only or record.reach_type == "riichi")
    )
    wins = tuple(record for record in selected if record.outcome == "win")
    tsumo_wins = tuple(record for record in wins if record.win_method == "tsumo")
    ron_wins = tuple(record for record in wins if record.win_method == "ron")
    hand_points_sum = sum(record.hand_points or 0 for record in wins)
    settlement_gain_sum = sum(record.settlement_gain or 0 for record in wins)

    def mean(total: int, count: int) -> float | None:
        return None if count == 0 else total / count

    return {
        "riichis": len(selected),
        "wins": len(wins),
        "other_wins": sum(record.outcome == "other_win" for record in selected),
        "draws": sum(record.outcome == "draw" for record in selected),
        "tsumo_wins": len(tsumo_wins),
        "ron_wins": len(ron_wins),
        "win_methods": {
            "tsumo": {
                "wins": len(tsumo_wins),
                "hand_points_sum": sum(
                    record.hand_points or 0 for record in tsumo_wins
                ),
                "mean_hand_points": mean(
                    sum(record.hand_points or 0 for record in tsumo_wins),
                    len(tsumo_wins),
                ),
            },
            "ron": {
                "wins": len(ron_wins),
                "hand_points_sum": sum(record.hand_points or 0 for record in ron_wins),
                "mean_hand_points": mean(
                    sum(record.hand_points or 0 for record in ron_wins),
                    len(ron_wins),
                ),
            },
        },
        "hand_points": {
            "sum": hand_points_sum,
            "mean": mean(hand_points_sum, len(wins)),
        },
        "settlement_gain": {
            "sum": settlement_gain_sum,
            "mean": mean(settlement_gain_sum, len(wins)),
        },
    }


def _summary_period(
    records: tuple[RiichiPointRecord, ...],
    years: tuple[int, ...],
) -> dict[str, object]:
    selected = tuple(record for record in records if record.year in years)
    return {
        "years": list(years),
        "established_riichis": len(selected),
        "dealer": _summary_group(selected, is_dealer=True),
        "nondealer": _summary_group(selected, is_dealer=False),
        "sensitivity_normal_riichi_only": {
            "dealer": _summary_group(
                selected,
                is_dealer=True,
                normal_only=True,
            ),
            "nondealer": _summary_group(
                selected,
                is_dealer=False,
                normal_only=True,
            ),
        },
    }


def _article_one_summary(
    records: tuple[RiichiPointRecord, ...],
) -> dict[str, object]:
    years = []
    for year in SUPPORTED_YEARS:
        selected = tuple(record for record in records if record.year == year)
        years.append(
            {
                "year": year,
                "established_riichis": len(selected),
                "dealer": _summary_group(selected, is_dealer=True),
                "nondealer": _summary_group(selected, is_dealer=False),
                "sensitivity_normal_riichi_only": {
                    "dealer": _summary_group(
                        selected,
                        is_dealer=True,
                        normal_only=True,
                    ),
                    "nondealer": _summary_group(
                        selected,
                        is_dealer=False,
                        normal_only=True,
                    ),
                },
            }
        )
    return {
        "analysis_name": "dealer-child-riichi-points-v1",
        "schema_version": 2,
        "source": {
            "repository": "NikkeTryHard/tenhou-to-mjai",
            "release_tag": "v2.0.0",
            "format": "MJAI JSON Lines",
        },
        "scope": {
            "rule_code": "00a9",
            "aka_flag": True,
            "bakaze": "E",
            "riichi_population": "established",
            "selected_years": list(SUPPORTED_YEARS),
            "primary_years": list(PRIMARY_YEARS),
        },
        "periods": {
            "selected": _summary_period(records, SUPPORTED_YEARS),
            "primary_2020_2025": _summary_period(records, tuple(range(2020, 2026))),
            "long_term_2009_2025": _summary_period(records, SUPPORTED_YEARS),
        },
        "years": years,
    }


def _write_article_one_inputs(
    tmp_path: Path,
    records: tuple[RiichiPointRecord, ...],
) -> tuple[Path, Path]:
    summary_path = tmp_path / "article-one.json"
    summary_path.write_text(
        json.dumps(_article_one_summary(records)),
        encoding="utf-8",
    )
    records_path = tmp_path / "records.jsonl.gz"
    with gzip.open(records_path, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record)) + "\n")
    return summary_path, records_path


def _balanced_article_one_records(
    years: tuple[int, ...],
) -> tuple[RiichiPointRecord, ...]:
    return tuple(
        record
        for year in years
        for record in (
            _article_one_record(
                actor=0,
                oya=0,
                method="tsumo",
                points=12_000,
                index=0,
                year=year,
            ),
            _article_one_record(
                actor=0,
                oya=0,
                method="ron",
                points=12_000,
                index=1,
                year=year,
            ),
            _article_one_record(
                actor=1,
                oya=0,
                method="tsumo",
                points=8000,
                index=2,
                year=year,
            ),
            _article_one_record(
                actor=1,
                oya=0,
                method="ron",
                points=8000,
                index=3,
                year=year,
            ),
        )
    )


def test_article_one_adapter_reproduces_observed_means_and_second_ron() -> None:
    article_one = (
        _article_one_record(
            actor=0,
            oya=0,
            method="tsumo",
            points=12_000,
            index=0,
        ),
        _article_one_record(
            actor=1,
            oya=0,
            method="ron",
            points=8000,
            index=1,
            honba_awarded=False,
        ),
    )
    converted = (
        decomposition_record_from_article_one(
            article_one[0],
            payment_pattern=PaymentPattern("dealer", "tsumo", (4000, 4000, 4000)),
        ),
        decomposition_record_from_article_one(
            article_one[1],
            payment_pattern=PaymentPattern("nondealer", "ron", (8000,)),
        ),
    )

    assert converted[1].observed_points == 8000
    assert observed_group_means(converted) == (Fraction(12_000), Fraction(8000))


def test_analysis_recomputes_normal_riichi_sensitivity_and_outputs() -> None:
    records = (
        _record("dealer", "tsumo", (4000, 4000, 4000), index=0),
        _record("dealer", "ron", (12_000,), index=1),
        _record("nondealer", "tsumo", (4000, 2000, 2000), index=2),
        _record("nondealer", "ron", (8000,), index=3),
        _record(
            "dealer",
            "tsumo",
            (6000, 6000, 6000),
            index=4,
            reach_type="double_riichi",
        ),
        _record(
            "dealer",
            "ron",
            (18_000,),
            index=5,
            reach_type="double_riichi",
        ),
        _record(
            "nondealer",
            "tsumo",
            (6000, 3000, 3000),
            index=6,
            reach_type="double_riichi",
        ),
        _record(
            "nondealer",
            "ron",
            (12_000,),
            index=7,
            reach_type="double_riichi",
        ),
    )

    analysis = analyze_score_decomposition(records, [2025])
    document = decomposition_document(analysis)
    markdown = render_decomposition_markdown(analysis)

    assert analysis.selected.normal_riichi.dealer_wins == 2
    assert analysis.selected.all_riichi.dealer_wins == 4
    assert document["factors"] == [
        "payment_rule",
        "method_mix",
        "within_method_score",
    ]
    assert "normal_riichi" in markdown


def test_markdown_covers_selected_primary_long_term_and_all_eight_means() -> None:
    records = tuple(
        record
        for year in SUPPORTED_YEARS
        for record in (
            _record(
                "dealer",
                "tsumo",
                (4000, 4000, 4000),
                year=year,
                index=0,
            ),
            _record("dealer", "ron", (12_000,), year=year, index=1),
            _record(
                "nondealer",
                "tsumo",
                (4000, 2000, 2000),
                year=year,
                index=2,
            ),
            _record("nondealer", "ron", (8000,), year=year, index=3),
        )
    )

    markdown = render_decomposition_markdown(
        analyze_score_decomposition((record for record in records), SUPPORTED_YEARS)
    )

    for period in ("selected", "primary_2020_2025", "long_term_2009_2025"):
        for population in ("all_riichi", "normal_riichi"):
            heading = f"### {period} / {population}"
            assert heading in markdown
    lines = markdown.splitlines()
    heading_index = lines.index("### selected / all_riichi")
    mean_rows = lines[heading_index + 4 : heading_index + 12]
    assert len(mean_rows) == 8
    assert all(row.startswith("|") for row in mean_rows)
    assert "Rounding pooled" in markdown


def test_record_loader_rejects_duplicate_and_reverse_canonical_keys(
    tmp_path: Path,
) -> None:
    duplicate = _article_one_record(
        actor=0,
        oya=0,
        method="tsumo",
        points=12_000,
        index=0,
    )
    duplicate_identity_later_event = _article_one_record(
        actor=0,
        oya=0,
        method="ron",
        points=12_000,
        index=1,
        source_path=duplicate.source_path,
        kyoku_index=duplicate.kyoku_index,
        reach_event_index=2,
    )
    reverse_first = _article_one_record(
        actor=0,
        oya=0,
        method="tsumo",
        points=12_000,
        index=1,
    )
    reverse_second = _article_one_record(
        actor=1,
        oya=1,
        method="ron",
        points=12_000,
        index=0,
    )
    cases = {
        "duplicate": (duplicate, duplicate),
        "duplicate-identity": (duplicate, duplicate_identity_later_event),
        "reverse": (reverse_first, reverse_second),
    }

    for name, records in cases.items():
        records_path = tmp_path / f"{name}.jsonl.gz"
        with gzip.open(records_path, "wt", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(asdict(record)) + "\n")
        with pytest.raises(ValueError, match="(?i)canonical|order|duplicate"):
            tuple(iter_decomposition_records(records_path, (2025,)))


def test_cli_reads_article_one_records_and_writes_json_markdown(tmp_path) -> None:
    article_one = (
        _article_one_record(actor=0, oya=0, method="tsumo", points=12_000, index=0),
        _article_one_record(actor=0, oya=0, method="ron", points=12_000, index=1),
        _article_one_record(actor=1, oya=0, method="tsumo", points=8000, index=2),
        _article_one_record(actor=1, oya=0, method="ron", points=8000, index=3),
    )
    summary_path, records_path = _write_article_one_inputs(tmp_path, article_one)
    json_path = tmp_path / "summary.json"
    markdown_path = tmp_path / "summary.md"
    summary_logical_path = "fixtures/article-one/summary-v1.json"
    records_logical_path = "fixtures/article-one/records-v1.jsonl.gz"

    assert (
        main(
            [
                "--years",
                "2025",
                "--input-summary",
                str(summary_path),
                "--input-records",
                str(records_path),
                "--input-summary-logical-path",
                summary_logical_path,
                "--input-records-logical-path",
                records_logical_path,
                "--output-json",
                str(json_path),
                "--output-markdown",
                str(markdown_path),
            ]
        )
        == 0
    )

    document = json.loads(json_path.read_text(encoding="utf-8"))
    metadata = document["input_dataset"]
    assert document["selected_years"] == [2025]
    assert metadata["summary_logical_path"] == summary_logical_path
    assert metadata["records_logical_path"] == records_logical_path
    assert (
        metadata["summary_sha256"]
        == hashlib.sha256(summary_path.read_bytes()).hexdigest()
    )
    assert (
        metadata["records_sha256"]
        == hashlib.sha256(records_path.read_bytes()).hexdigest()
    )
    assert metadata["all_record_count"] == 4
    assert metadata["selected_record_count"] == 4
    assert metadata["selected_win_count"] == 4
    assert metadata["converted_win_count"] == 4
    assert metadata["unsupported_score_count"] == 0
    assert metadata["ambiguous_score_count"] == 0
    assert "rounding_effect" in document["periods"]["selected"]["all_riichi"]
    assert "Shapley" in markdown_path.read_text(encoding="utf-8")


def test_cli_rejects_full_summary_mismatch_before_publishing(tmp_path: Path) -> None:
    records = (
        _article_one_record(actor=0, oya=0, method="tsumo", points=12_000, index=0),
        _article_one_record(actor=0, oya=0, method="ron", points=12_000, index=1),
        _article_one_record(actor=1, oya=0, method="tsumo", points=8000, index=2),
        _article_one_record(actor=1, oya=0, method="ron", points=8000, index=3),
    )
    summary_path, records_path = _write_article_one_inputs(tmp_path, records)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    year_2025 = next(item for item in summary["years"] if item["year"] == 2025)
    year_2025["dealer"]["hand_points"]["sum"] += 100
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    json_path = tmp_path / "result.json"
    markdown_path = tmp_path / "result.md"

    with pytest.raises(
        (ValueError, RuntimeError),
        match="(?i)summary|mismatch|hand.points",
    ):
        main(
            [
                "--years",
                "2025",
                "--input-summary",
                str(summary_path),
                "--input-records",
                str(records_path),
                "--output-json",
                str(json_path),
                "--output-markdown",
                str(markdown_path),
            ]
        )

    assert not json_path.exists()
    assert not markdown_path.exists()


def test_output_bundle_rolls_back_both_artifacts_on_second_publish_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = (
        _record("dealer", "tsumo", (4000, 4000, 4000), index=0),
        _record("dealer", "ron", (12_000,), index=1),
        _record("nondealer", "tsumo", (4000, 2000, 2000), index=2),
        _record("nondealer", "ron", (8000,), index=3),
    )
    analysis = analyze_score_decomposition(records, [2025])
    json_path = tmp_path / "summary.json"
    markdown_path = tmp_path / "summary.md"
    json_path.write_bytes(b"old-json")
    markdown_path.write_bytes(b"old-markdown")
    real_replace = decomposition.os.replace
    replace_calls = 0

    def fail_second_publish(source: Path, target: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 4:
            raise OSError("injected second publish failure")
        real_replace(source, target)

    monkeypatch.setattr(decomposition.os, "replace", fail_second_publish)

    with pytest.raises(OSError, match="injected second publish failure"):
        write_decomposition_outputs(analysis, json_path, markdown_path)

    assert json_path.read_bytes() == b"old-json"
    assert markdown_path.read_bytes() == b"old-markdown"
    assert not tuple(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("interrupt_call", (1, 2, 3, 4))
def test_output_bundle_rolls_back_keyboard_interrupt_after_every_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_call: int,
) -> None:
    analysis = analyze_score_decomposition(
        (
            _record("dealer", "tsumo", (4000, 4000, 4000), index=0),
            _record("dealer", "ron", (12_000,), index=1),
            _record("nondealer", "tsumo", (4000, 2000, 2000), index=2),
            _record("nondealer", "ron", (8000,), index=3),
        ),
        [2025],
    )
    json_path = tmp_path / "summary.json"
    markdown_path = tmp_path / "summary.md"
    json_path.write_bytes(b"old-json")
    markdown_path.write_bytes(b"old-markdown")
    real_replace = decomposition.os.replace
    replace_calls = 0

    def interrupt_after_replace(source: Path, target: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        real_replace(source, target)
        if replace_calls == interrupt_call:
            raise KeyboardInterrupt("injected asynchronous interruption")

    monkeypatch.setattr(decomposition.os, "replace", interrupt_after_replace)

    with pytest.raises(KeyboardInterrupt, match="asynchronous interruption"):
        write_decomposition_outputs(analysis, json_path, markdown_path)

    assert json_path.read_bytes() == b"old-json"
    assert markdown_path.read_bytes() == b"old-markdown"
    assert not tuple(tmp_path.glob("*.tmp"))


def test_output_bundle_retains_original_backup_when_rollback_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = analyze_score_decomposition(
        (
            _record("dealer", "tsumo", (4000, 4000, 4000), index=0),
            _record("dealer", "ron", (12_000,), index=1),
            _record("nondealer", "tsumo", (4000, 2000, 2000), index=2),
            _record("nondealer", "ron", (8000,), index=3),
        ),
        [2025],
    )
    json_path = tmp_path / "summary.json"
    markdown_path = tmp_path / "summary.md"
    json_path.write_bytes(b"old-json")
    markdown_path.write_bytes(b"old-markdown")
    real_replace = decomposition.os.replace
    markdown_backup: Path | None = None
    publish_failed = False

    def fail_publish_and_rollback(source: Path, target: Path) -> None:
        nonlocal markdown_backup, publish_failed
        source = Path(source)
        target = Path(target)
        if source == markdown_path:
            real_replace(source, target)
            markdown_backup = target
            return
        if markdown_backup is not None and source == markdown_backup:
            raise OSError("injected rollback failure")
        if target == markdown_path and not publish_failed:
            publish_failed = True
            raise OSError("injected publish failure")
        real_replace(source, target)

    monkeypatch.setattr(decomposition.os, "replace", fail_publish_and_rollback)

    with pytest.raises(RuntimeError, match="retained backups") as caught:
        write_decomposition_outputs(analysis, json_path, markdown_path)

    assert json_path.read_bytes() == b"old-json"
    assert not markdown_path.exists()
    assert markdown_backup is not None
    assert markdown_backup.read_bytes() == b"old-markdown"
    assert str(markdown_backup) in str(caught.value)
    assert {path for path in tmp_path.iterdir() if path.suffix == ".tmp"} == {
        markdown_backup
    }


@pytest.mark.parametrize(
    "malformation",
    (
        "repository",
        "format",
        "scope_type",
        "rule_code",
        "aka_flag_type",
        "bakaze",
        "riichi_population",
        "selected_years_order",
        "primary_years_duplicate",
    ),
)
def test_summary_loader_rejects_malformed_identity_and_scope(
    tmp_path: Path,
    malformation: str,
) -> None:
    summary = _article_one_summary(())
    source = summary["source"]
    scope = summary["scope"]
    assert isinstance(source, dict)
    assert isinstance(scope, dict)
    if malformation == "repository":
        source["repository"] = "someone/else"
    elif malformation == "format":
        source["format"] = "CSV"
    elif malformation == "scope_type":
        summary["scope"] = []
    elif malformation == "rule_code":
        scope["rule_code"] = "0000"
    elif malformation == "aka_flag_type":
        scope["aka_flag"] = 1
    elif malformation == "bakaze":
        scope["bakaze"] = "S"
    elif malformation == "riichi_population":
        scope["riichi_population"] = "declared"
    elif malformation == "selected_years_order":
        scope["selected_years"] = list(reversed(SUPPORTED_YEARS))
    else:
        scope["primary_years"] = [*PRIMARY_YEARS, PRIMARY_YEARS[-1]]
    summary_path = tmp_path / f"summary-{malformation}.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(
        (TypeError, ValueError),
        match="(?i)source|scope|year|population",
    ):
        _load_and_validate_summary(summary_path)


@pytest.mark.parametrize(
    ("field_name", "malformed_value"),
    (
        ("bakaze", "S"),
        ("win_method", "chi"),
        ("hora_event_index", True),
        ("hora_line", -1),
        ("honba_awarded", 1),
        ("hand_points", 123),
        ("settlement_gain", 0),
    ),
)
def test_record_loader_validates_malformed_records_outside_selected_years(
    tmp_path: Path,
    field_name: str,
    malformed_value: object,
) -> None:
    record = asdict(
        _article_one_record(
            actor=0,
            oya=0,
            method="tsumo",
            points=12_000,
            index=0,
            year=2024,
        )
    )
    record[field_name] = malformed_value
    records_path = tmp_path / f"malformed-{field_name}.jsonl.gz"
    with gzip.open(records_path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")

    with pytest.raises(ValueError, match="records line 1"):
        tuple(iter_decomposition_records(records_path, (2025,)))


def test_cli_rejects_corrupt_primary_period_summary(tmp_path: Path) -> None:
    records = _balanced_article_one_records(PRIMARY_YEARS)
    summary_path, records_path = _write_article_one_inputs(tmp_path, records)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["periods"]["primary_2020_2025"]["dealer"]["wins"] += 1
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="(?i)period|primary|mismatch"):
        main(
            [
                "--years",
                *(str(year) for year in PRIMARY_YEARS),
                "--input-summary",
                str(summary_path),
                "--input-records",
                str(records_path),
                "--output-json",
                str(tmp_path / "result.json"),
                "--output-markdown",
                str(tmp_path / "result.md"),
            ]
        )


def test_cli_rejects_corrupt_displayed_mean(tmp_path: Path) -> None:
    records = _balanced_article_one_records((2025,))
    summary_path, records_path = _write_article_one_inputs(tmp_path, records)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    year_2025 = next(item for item in summary["years"] if item["year"] == 2025)
    year_2025["dealer"]["hand_points"]["mean"] += 0.5
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="(?i)mean|mismatch"):
        main(
            [
                "--years",
                "2025",
                "--input-summary",
                str(summary_path),
                "--input-records",
                str(records_path),
                "--output-json",
                str(tmp_path / "result.json"),
                "--output-markdown",
                str(tmp_path / "result.md"),
            ]
        )


def test_json_serializes_audit_distributions_shares_and_residual() -> None:
    records = (
        _record("dealer", "tsumo", (4000, 4000, 4000), index=0),
        _record("dealer", "ron", (12_000,), index=1),
        _record("nondealer", "tsumo", (4000, 2000, 2000), index=2),
        _record("nondealer", "ron", (8000,), index=3),
    )

    analysis = analyze_score_decomposition(records, [2025])
    document = decomposition_document(analysis)
    result = document["periods"]["selected"]["all_riichi"]

    assert result["method_counts"] == {
        "dealer": {"tsumo": 1, "ron": 1},
        "nondealer": {"tsumo": 1, "ron": 1},
    }
    assert {
        (
            row["score_source"],
            row["method"],
            tuple(row["candidate_basic_points"]),
            row["dealer_points"],
            row["nondealer_points"],
            row["wins"],
        )
        for row in result["score_distributions"]
    } == {
        ("dealer", "tsumo", (2000,), 12_000, 8000, 1),
        ("dealer", "ron", (2000,), 12_000, 8000, 1),
        ("nondealer", "tsumo", (2000,), 12_000, 8000, 1),
        ("nondealer", "ron", (2000,), 12_000, 8000, 1),
    }
    assert sum(row["wins"] for row in result["score_distributions"]) == (
        result["dealer_wins"] + result["nondealer_wins"]
    )
    assert result["shapley_share_of_gap"] == {
        "payment_rule": {"value": 1.0, "numerator": 1, "denominator": 1},
        "method_mix": {"value": 0.0, "numerator": 0, "denominator": 1},
        "within_method_score": {
            "value": 0.0,
            "numerator": 0,
            "denominator": 1,
        },
    }
    assert result["efficiency_residual"] == {
        "value": 0.0,
        "numerator": 0,
        "denominator": 1,
    }
    assert document["score_conversion"]["basic_point_candidates"]
    assert document["score_conversion"]["payment_formulas"]
    assert "tier" in document["counterfactual"]
    markdown = render_decomposition_markdown(analysis)
    assert "D T/R" in markdown
    assert "R share" in markdown
    assert "Residual" in markdown
    assert "Canonical conversion-tier distribution" in markdown


def test_json_serializes_null_shapley_shares_when_gap_is_zero() -> None:
    records = (
        _record("dealer", "tsumo", (4000, 4000, 4000), index=0),
        _record("dealer", "ron", (12_000,), index=1),
        _record("nondealer", "tsumo", (6000, 3000, 3000), index=2),
        _record("nondealer", "ron", (12_000,), index=3),
    )

    analysis = analyze_score_decomposition(records, [2025])
    result = decomposition_document(analysis)["periods"]["selected"]["all_riichi"]

    assert result["observed_gap"]["numerator"] == 0
    assert result["shapley_share_of_gap"] == {
        "payment_rule": None,
        "method_mix": None,
        "within_method_score": None,
    }
    assert result["efficiency_residual"]["numerator"] == 0
    assert "N/A" in render_decomposition_markdown(analysis)


def test_shapley_shares_preserve_negative_and_over_100_percent_values() -> None:
    records = (
        _record("dealer", "tsumo", (6000, 6000, 6000), index=0),
        _record("dealer", "tsumo", (500, 500, 500), index=1),
        _record("dealer", "ron", (1500,), index=2),
        _record("dealer", "ron", (12_000,), index=3),
        _record("dealer", "ron", (12_000,), index=4),
        _record("nondealer", "tsumo", (8000, 4000, 4000), index=5),
        _record("nondealer", "tsumo", (4000, 2000, 2000), index=6),
        _record("nondealer", "tsumo", (1000, 500, 500), index=7),
        _record("nondealer", "tsumo", (500, 300, 300), index=8),
        _record("nondealer", "ron", (1000,), index=9),
        _record("nondealer", "ron", (8000,), index=10),
        _record("nondealer", "ron", (8000,), index=11),
        _record("nondealer", "ron", (12_000,), index=12),
    )

    result = decompose_score_gap(records)
    shares = tuple(
        result.contribution_share(factor) for factor in decomposition.FACTORS
    )

    assert any(share is not None and share < 0 for share in shares)
    assert any(share is not None and share > 1 for share in shares)
    assert sum(share for share in shares if share is not None) == 1
    assert result.efficiency_residual == 0
    markdown = render_decomposition_markdown(
        analyze_score_decomposition(records, [2025])
    )
    assert "163.242%" in markdown
    assert "-1.412%" in markdown
