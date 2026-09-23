import gzip
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import analysis.analyze_dealer_child_riichi_points as cli
from analysis.analyze_dealer_child_riichi_points import (
    main,
    parse_args,
    selected_years_from_args,
)
from mahjong_analysis.dealer_child_riichi_points import (
    SUPPORTED_YEARS,
    CohortReference,
    CohortYearExpectation,
    RiichiPointAggregationCancelled,
    RiichiPointGroupStats,
    YearlyRiichiPointResult,
    aggregate_riichi_point_years,
    aggregate_riichi_points,
    analysis_document,
    combine_years,
    extract_established_riichi_actors,
    extract_established_riichis,
    load_cohort_reference,
    render_markdown,
    write_outputs,
)


def _filename(year: int, index: int, *, rule_code: str = "00a9") -> str:
    return f"{year}010100gm-{rule_code}-0000-{index:08x}.mjson"


def _write_game(
    root: Path,
    year: int,
    index: int,
    kyokus: list[list[dict[str, object]]],
    *,
    rule_code: str = "00a9",
    aka_flag: object = True,
) -> Path:
    path = root / str(year) / _filename(year, index, rule_code=rule_code)
    path.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {"type": "start_game", "aka_flag": aka_flag},
        *(event for kyoku in kyokus for event in kyoku),
        {"type": "end_game"},
    ]
    path.write_text(
        "".join(f"{json.dumps(event)}\n" for event in events),
        encoding="utf-8",
    )
    return path


def _kyoku(
    *,
    oya: int,
    riichi_actors: tuple[int, ...],
    result_events: list[dict[str, object]],
    honba: int = 0,
    bakaze: str = "E",
    kyoku_number: int = 1,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = [
        {
            "type": "start_kyoku",
            "bakaze": bakaze,
            "kyoku": kyoku_number,
            "honba": honba,
            "kyotaku": 0,
            "oya": oya,
        }
    ]
    for actor in riichi_actors:
        events.extend(
            [
                {"type": "reach", "actor": actor},
                {
                    "type": "dahai",
                    "actor": actor,
                    "pai": "1m",
                    "tsumogiri": False,
                },
                {"type": "reach_accepted", "actor": actor},
            ]
        )
    events.extend(result_events)
    events.append({"type": "end_kyoku"})
    return events


def _draw() -> list[dict[str, object]]:
    return [{"type": "ryukyoku", "deltas": [0, 0, 0, 0]}]


def _cohort(
    counts: dict[int, tuple[int, int, int, int]],
) -> CohortReference:
    return CohortReference(
        repository="example/tenhou-to-mjai",
        release_tag="v2.0.0",
        manifest_path="data/processed/riichi-waits-v1/manifest.json",
        years={
            year: CohortYearExpectation(
                year=year,
                scanned_files=values[0],
                target_games=values[1],
                east_kyokus=values[2],
                established_riichis=values[3],
            )
            for year, values in counts.items()
        },
    )


def _manifest(path: Path, years: list[dict[str, int]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "dataset_name": "riichi-waits-v1",
                "schema_version": 1,
                "generator": {"invocation": {"mode": "full", "max_files": None}},
                "source": {
                    "repository": "example/tenhou-to-mjai",
                    "release_tag": "v2.0.0",
                    "archive_hashes_verified": True,
                },
                "scope": {
                    "rule_code": "00a9",
                    "aka_flag": True,
                    "bakaze": "E",
                    "extraction_mode": "full",
                },
                "years": years,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_aggregate_raw_mjai_classifies_dealer_child_and_scores(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    path = _write_game(
        raw_root,
        2025,
        1,
        [
            _kyoku(
                oya=0,
                riichi_actors=(0,),
                honba=2,
                result_events=[
                    {
                        "type": "hora",
                        "actor": 0,
                        "target": 1,
                        "deltas": [14600, -12600, 0, 0],
                    }
                ],
            ),
            _kyoku(
                oya=0,
                riichi_actors=(1,),
                honba=1,
                kyoku_number=2,
                result_events=[
                    {
                        "type": "hora",
                        "actor": 1,
                        "target": 1,
                        "deltas": [-2100, 5300, -1100, -1100],
                    }
                ],
            ),
            _kyoku(
                oya=2,
                riichi_actors=(2,),
                kyoku_number=3,
                result_events=_draw(),
            ),
            _kyoku(
                oya=3,
                riichi_actors=(1,),
                kyoku_number=4,
                result_events=[
                    {
                        "type": "hora",
                        "actor": 3,
                        "target": 3,
                        "deltas": [-1000, -1000, -1000, 3000],
                    }
                ],
            ),
        ],
    )

    result = aggregate_riichi_points(2025, [path])

    assert result.scanned_files == 1
    assert result.target_games == 1
    assert result.east_kyokus == 4
    assert result.established_riichis == 4
    assert result.dealer.riichis == 2
    assert result.dealer.wins == 1
    assert result.dealer.ron_wins == 1
    assert result.dealer.draws == 1
    assert result.dealer.hand_points_sum == 12000
    assert result.dealer.mean_ron_hand_points == 12000
    assert result.dealer.ron_rate_per_riichi == 0.5
    assert result.dealer.settlement_gain_sum == 14600
    assert result.nondealer.riichis == 2
    assert result.nondealer.wins == 1
    assert result.nondealer.tsumo_wins == 1
    assert result.nondealer.other_wins == 1
    assert result.nondealer.hand_points_sum == 4000
    assert result.nondealer.mean_tsumo_hand_points == 4000
    assert result.nondealer.tsumo_rate_per_riichi == 0.5
    assert result.nondealer.settlement_gain_sum == 5300
    assert result.dealer.mean_hand_points_ci95 is None


def test_multiple_ron_counts_riichi_winner_and_removes_honba(tmp_path: Path) -> None:
    records = []
    path = _write_game(
        tmp_path / "raw",
        2025,
        1,
        [
            _kyoku(
                oya=0,
                riichi_actors=(1,),
                honba=1,
                result_events=[
                    {
                        "type": "hora",
                        "actor": 0,
                        "target": 2,
                        "deltas": [4200, 0, -3200, 0],
                    },
                    {
                        "type": "hora",
                        "actor": 1,
                        "target": 2,
                        "deltas": [0, 5200, -5200, 0],
                    },
                ],
            )
        ],
    )

    result = aggregate_riichi_points(2025, [path], record_callback=records.append)

    assert result.nondealer.wins == 1
    assert result.nondealer.ron_wins == 1
    assert result.nondealer.hand_points_sum == 5200
    assert records[0].honba_awarded is False
    assert records[0].source_path == f"2025/{path.name}"
    assert records[0].hora_event_index is not None
    assert records[0].start_kyoku_line == 2
    assert records[0].reach_line == 3
    assert records[0].reach_accepted_line == 5
    assert records[0].hora_line == 7


def test_dealer_riichi_tsumo_and_method_distribution(tmp_path: Path) -> None:
    path = _write_game(
        tmp_path / "raw",
        2025,
        1,
        [
            _kyoku(
                oya=2,
                riichi_actors=(2,),
                honba=1,
                result_events=[
                    {
                        "type": "hora",
                        "actor": 2,
                        "target": 2,
                        "deltas": [-2100, -2100, 7300, -2100],
                    }
                ],
            )
        ],
    )

    result = aggregate_riichi_points(2025, [path])

    assert result.dealer.tsumo_wins == 1
    assert result.dealer.mean_tsumo_hand_points == 6000
    assert result.dealer.tsumo_point_counts == ((6000, 1),)
    assert result.dealer.ron_point_counts == ()


def test_normal_riichi_sensitivity_excludes_double_riichi(tmp_path: Path) -> None:
    kyoku = [
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
        },
        {"type": "dahai", "actor": 1, "pai": "9m", "tsumogiri": False},
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "1m", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 1},
        {"type": "reach", "actor": 2},
        {"type": "dahai", "actor": 2, "pai": "1p", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 2},
        *_draw(),
        {"type": "end_kyoku"},
    ]
    path = _write_game(tmp_path / "raw", 2025, 1, [kyoku])

    reaches = extract_established_riichis(kyoku)
    result = aggregate_riichi_points(2025, [path])

    assert tuple(reach.reach_type for reach in reaches) == (
        "riichi",
        "double_riichi",
    )
    assert result.nondealer.riichis == 2
    assert result.normal_nondealer.riichis == 1


def test_clustered_mean_ci_is_exact_for_zero_residual_variance() -> None:
    group = RiichiPointGroupStats(
        riichis=6,
        wins=6,
        other_wins=0,
        draws=0,
        tsumo_wins=0,
        ron_wins=6,
        hand_points_sum=152_000,
        settlement_gain_sum=152_000,
        point_counts=((12_000, 2), (32_000, 4)),
        tsumo_hand_points_sum=0,
        ron_hand_points_sum=152_000,
        tsumo_point_counts=(),
        ron_point_counts=((12_000, 2), (32_000, 4)),
        cluster_games=2,
        cluster_wins_squared_sum=18,
        cluster_points_squared_sum=2 * 76_000**2,
        cluster_wins_points_sum=2 * 3 * 76_000,
    )

    assert group.mean_hand_points_ci95 == (
        group.mean_hand_points,
        group.mean_hand_points,
    )


def test_unestablished_declaration_is_not_counted(tmp_path: Path) -> None:
    kyoku = [
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
        },
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False},
        {
            "type": "hora",
            "actor": 1,
            "target": 0,
            "pai": "1m",
            "deltas": [-1000, 1000, 0, 0],
        },
        {"type": "end_kyoku"},
    ]
    path = _write_game(tmp_path / "raw", 2025, 1, [kyoku])

    result = aggregate_riichi_points(2025, [path])

    assert extract_established_riichi_actors(kyoku) == ()
    assert result.established_riichis == 0


def test_excludes_non_target_game_and_south_round(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    target = _write_game(
        raw_root,
        2025,
        1,
        [
            _kyoku(
                oya=0,
                riichi_actors=(0,),
                bakaze="S",
                result_events=_draw(),
            )
        ],
    )
    non_target = _write_game(
        raw_root,
        2025,
        2,
        [
            _kyoku(
                oya=0,
                riichi_actors=(0,),
                result_events=_draw(),
            )
        ],
        rule_code="00e1",
    )

    result = aggregate_riichi_points(2025, [target, non_target])

    assert result.scanned_files == 2
    assert result.target_games == 1
    assert result.east_kyokus == 0
    assert result.established_riichis == 0


def test_rejects_unmatched_reach_accepted(tmp_path: Path) -> None:
    path = _write_game(
        tmp_path / "raw",
        2025,
        1,
        [
            [
                {
                    "type": "start_kyoku",
                    "bakaze": "E",
                    "kyoku": 1,
                    "honba": 0,
                    "oya": 0,
                },
                {"type": "reach_accepted", "actor": 0},
                *_draw(),
                {"type": "end_kyoku"},
            ]
        ],
    )

    with pytest.raises(RuntimeError, match="failed to process") as error:
        aggregate_riichi_points(2025, [path])

    assert isinstance(error.value.__cause__, ValueError)
    assert "unmatched reach_accepted" in str(error.value.__cause__)


def test_rejects_noncontiguous_result_events(tmp_path: Path) -> None:
    kyoku = _kyoku(
        oya=0,
        riichi_actors=(0,),
        result_events=[
            {
                "type": "hora",
                "actor": 0,
                "target": 1,
                "deltas": [12000, -12000, 0, 0],
            },
            {"type": "dora", "dora_marker": "1p"},
        ],
    )
    path = _write_game(tmp_path / "raw", 2025, 1, [kyoku])

    with pytest.raises(RuntimeError, match="failed to process") as error:
        aggregate_riichi_points(2025, [path])

    assert "result events must be contiguous" in str(error.value.__cause__)


@pytest.mark.parametrize(
    "deltas",
    (
        [1000, -1000, 1, 0],
        [1000, -1000, 0],
        [1000, -1000, 0, True],
    ),
)
def test_rejects_invalid_hora_deltas(tmp_path: Path, deltas: list[object]) -> None:
    path = _write_game(
        tmp_path / "raw",
        2025,
        1,
        [
            _kyoku(
                oya=0,
                riichi_actors=(0,),
                result_events=[
                    {
                        "type": "hora",
                        "actor": 0,
                        "target": 1,
                        "deltas": deltas,
                    }
                ],
            )
        ],
    )

    with pytest.raises(RuntimeError, match="failed to process"):
        aggregate_riichi_points(2025, [path])


def test_load_cohort_reference_validates_identity_and_counts(tmp_path: Path) -> None:
    path = _manifest(
        tmp_path / "manifest.json",
        [
            {
                "year": 2025,
                "scanned_files": 10,
                "target_games": 9,
                "east_kyokus": 50,
                "established_riichis": 40,
            }
        ],
    )

    reference = load_cohort_reference(
        path,
        [2025],
        logical_path="data/processed/riichi-waits-v1/manifest.json",
    )

    assert reference.repository == "example/tenhou-to-mjai"
    assert reference.years[2025].established_riichis == 40

    document = json.loads(path.read_text(encoding="utf-8"))
    document["scope"]["bakaze"] = "S"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="scope"):
        load_cohort_reference(
            path,
            [2025],
            logical_path="data/processed/riichi-waits-v1/manifest.json",
        )


def test_year_orchestration_checks_reference_counts(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _write_game(
        raw_root,
        2025,
        1,
        [
            _kyoku(
                oya=0,
                riichi_actors=(1,),
                result_events=_draw(),
            )
        ],
    )
    reference = _cohort({2025: (1, 1, 1, 1)})

    analysis = aggregate_riichi_point_years([2025], raw_root, reference)

    assert analysis.selected_years == (2025,)
    assert analysis.years[0].nondealer.draws == 1

    wrong_reference = _cohort({2025: (1, 1, 1, 2)})
    with pytest.raises(RuntimeError, match="established_riichis"):
        aggregate_riichi_point_years([2025], raw_root, wrong_reference)


def test_year_orchestration_checks_cancellation_between_source_files(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    for index in (1, 2):
        _write_game(
            raw_root,
            2025,
            index,
            [_kyoku(oya=0, riichi_actors=(1,), result_events=_draw())],
        )
    checks = 0
    records: list[object] = []

    def cancellation_requested() -> bool:
        nonlocal checks
        checks += 1
        return checks > 1

    with pytest.raises(RiichiPointAggregationCancelled, match="source file 2"):
        aggregate_riichi_point_years(
            [2025],
            raw_root,
            _cohort({2025: (2, 2, 2, 2)}),
            record_callback=records.append,
            cancellation_callback=cancellation_requested,
        )

    assert checks == 2
    assert len(records) == 1


def test_combination_document_markdown_and_outputs(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    paths: list[Path] = []
    for index, points in enumerate((6000, 7000), start=1):
        paths.append(
            _write_game(
                raw_root,
                2025,
                index,
                [
                    _kyoku(
                        oya=0,
                        riichi_actors=(1,),
                        result_events=[
                            {
                                "type": "hora",
                                "actor": 1,
                                "target": 2,
                                "deltas": [0, points, -points, 0],
                            }
                        ],
                    )
                ],
            )
        )
    yearly = aggregate_riichi_points(2025, paths)
    analysis = aggregate_riichi_point_years(
        [2025],
        raw_root,
        _cohort({2025: (2, 2, 2, 2)}),
    )

    assert combine_years([yearly]).nondealer.median_hand_points == 6500
    assert yearly.nondealer.mean_hand_points_ci95 is not None
    document = analysis_document(analysis)
    selected = document["periods"]["selected"]
    assert selected["nondealer"]["hand_points"]["mean"] == 6500
    assert selected["nondealer"]["win_methods"]["ron"]["mean_hand_points"] == 6500
    assert (
        selected["nondealer"]["win_methods"]["ron"][
            "probability_per_established_riichi"
        ]
        == 1.0
    )
    assert selected["comparison"]["nondealer_mean_minus_6500"] == 0
    assert selected["comparison"]["nondealer_mean_ci95_contains_6500"] is True
    assert "primary_2020_2025" not in document["periods"]
    markdown = render_markdown(analysis)
    assert "6,500" in markdown
    assert "2025" in markdown

    json_path = tmp_path / "out" / "summary.json"
    markdown_path = tmp_path / "out" / "summary.md"
    write_outputs(analysis, json_path, markdown_path)
    assert json.loads(json_path.read_text(encoding="utf-8")) == document
    assert markdown_path.read_text(encoding="utf-8") == markdown


def test_multi_year_combination_and_period_markdown_tables(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    counts: dict[int, tuple[int, int, int, int]] = {}
    for year in SUPPORTED_YEARS:
        _write_game(
            raw_root,
            year,
            1,
            [
                _kyoku(
                    oya=0,
                    riichi_actors=(1,),
                    result_events=[
                        {
                            "type": "hora",
                            "actor": 1,
                            "target": 2,
                            "deltas": [0, 6500, -6500, 0],
                        }
                    ],
                )
            ],
        )
        counts[year] = (1, 1, 1, 1)

    analysis = aggregate_riichi_point_years(
        SUPPORTED_YEARS,
        raw_root,
        _cohort(counts),
    )
    combined = combine_years(analysis.years)
    markdown = render_markdown(analysis)
    period_table = markdown.split("### 期間別比較", maxsplit=1)[0]

    assert combined.nondealer.wins == len(SUPPORTED_YEARS)
    assert combined.nondealer.point_counts == ((6500, len(SUPPORTED_YEARS)),)
    assert combined.nondealer.cluster_games == len(SUPPORTED_YEARS)
    assert "| selected | nondealer |" in period_table
    assert "| primary_2020_2025 | nondealer |" in period_table
    assert "| long_term_2009_2025 | nondealer |" in period_table


def test_cli_year_selection() -> None:
    assert selected_years_from_args(parse_args(["--years", "2025"])) == (2025,)
    assert selected_years_from_args(parse_args(["--all"])) == SUPPORTED_YEARS
    assert parse_args(["--years", "2025"]).workers == 1
    assert parse_args(["--years", "2025", "--workers", "2"]).workers == 2
    with pytest.raises(SystemExit):
        parse_args([])
    with pytest.raises(SystemExit):
        parse_args(["--all", "--years", "2025"])
    with pytest.raises(SystemExit):
        parse_args(["--years", "2025", "--workers", "0"])


def test_cli_writes_auditable_gzip_records(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _write_game(
        raw_root,
        2025,
        1,
        [_kyoku(oya=0, riichi_actors=(1,), result_events=_draw())],
    )
    manifest = _manifest(
        tmp_path / "manifest.json",
        [
            {
                "year": 2025,
                "scanned_files": 1,
                "target_games": 1,
                "east_kyokus": 1,
                "established_riichis": 1,
            }
        ],
    )
    json_path = tmp_path / "summary.json"
    markdown_path = tmp_path / "summary.md"
    records_path = tmp_path / "records.jsonl.gz"

    assert (
        main(
            [
                "--years",
                "2025",
                "--raw-root",
                str(raw_root),
                "--cohort-manifest",
                str(manifest),
                "--output-json",
                str(json_path),
                "--output-markdown",
                str(markdown_path),
                "--output-records",
                str(records_path),
            ]
        )
        == 0
    )
    with gzip.open(records_path, "rt", encoding="utf-8") as file:
        records = [json.loads(line) for line in file]

    assert len(records) == 1
    assert records[0]["source_path"].endswith(".mjson")
    assert records[0]["outcome"] == "draw"


def test_cli_two_workers_writes_ordered_deterministic_multi_year_gzip(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    _write_game(
        raw_root,
        2024,
        1,
        [_kyoku(oya=0, riichi_actors=(2, 0), result_events=_draw())],
    )
    _write_game(
        raw_root,
        2025,
        1,
        [_kyoku(oya=0, riichi_actors=(1,), result_events=_draw())],
    )
    manifest = _manifest(
        tmp_path / "manifest.json",
        [
            {
                "year": 2024,
                "scanned_files": 1,
                "target_games": 1,
                "east_kyokus": 1,
                "established_riichis": 2,
            },
            {
                "year": 2025,
                "scanned_files": 1,
                "target_games": 1,
                "east_kyokus": 1,
                "established_riichis": 1,
            },
        ],
    )

    record_bytes: list[bytes] = []
    for run in (1, 2):
        json_path = tmp_path / f"summary-{run}.json"
        markdown_path = tmp_path / f"summary-{run}.md"
        records_path = tmp_path / f"records-{run}.jsonl.gz"
        assert (
            main(
                [
                    "--years",
                    "2025",
                    "2024",
                    "--workers",
                    "2",
                    "--raw-root",
                    str(raw_root),
                    "--cohort-manifest",
                    str(manifest),
                    "--output-json",
                    str(json_path),
                    "--output-markdown",
                    str(markdown_path),
                    "--output-records",
                    str(records_path),
                ]
            )
            == 0
        )
        with gzip.open(records_path, "rt", encoding="utf-8") as file:
            records = [json.loads(line) for line in file]

        assert [(record["year"], record["actor"]) for record in records] == [
            (2024, 2),
            (2024, 0),
            (2025, 1),
        ]
        assert json.loads(json_path.read_text(encoding="utf-8"))["scope"][
            "selected_years"
        ] == [2024, 2025]
        assert markdown_path.read_text(encoding="utf-8").index("| 2024 |") < (
            markdown_path.read_text(encoding="utf-8").index("| 2025 |")
        )
        record_bytes.append(records_path.read_bytes())

    assert record_bytes[0] == record_bytes[1]
    assert not tuple(tmp_path.glob(".records-*.jsonl.gz.years.*"))


def test_cli_failure_keeps_existing_records_and_cleans_temporaries(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    _write_game(
        raw_root,
        2025,
        1,
        [_kyoku(oya=0, riichi_actors=(1,), result_events=_draw())],
    )
    manifest = _manifest(
        tmp_path / "manifest.json",
        [
            {
                "year": 2025,
                "scanned_files": 1,
                "target_games": 1,
                "east_kyokus": 1,
                "established_riichis": 2,
            }
        ],
    )
    json_path = tmp_path / "summary.json"
    markdown_path = tmp_path / "summary.md"
    records_path = tmp_path / "records.jsonl.gz"
    json_path.write_bytes(b"old-json")
    markdown_path.write_bytes(b"old-markdown")
    records_path.write_bytes(b"old-records")

    with pytest.raises(RuntimeError, match="cohort count mismatch"):
        main(
            [
                "--years",
                "2025",
                "--raw-root",
                str(raw_root),
                "--cohort-manifest",
                str(manifest),
                "--output-json",
                str(json_path),
                "--output-markdown",
                str(markdown_path),
                "--output-records",
                str(records_path),
            ]
        )

    assert json_path.read_bytes() == b"old-json"
    assert markdown_path.read_bytes() == b"old-markdown"
    assert records_path.read_bytes() == b"old-records"
    assert not tuple(tmp_path.glob(".records.jsonl.gz.years.*"))
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_publish_failure_restores_all_three_previous_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = (
        tmp_path / "summary.json",
        tmp_path / "summary.md",
        tmp_path / "records.jsonl.gz",
    )
    staged = (
        tmp_path / "new-summary.json",
        tmp_path / "new-summary.md",
        tmp_path / "new-records.jsonl.gz",
    )
    old_contents = (b"old-json", b"old-markdown", b"old-records")
    new_contents = (b"new-json", b"new-markdown", b"new-records")
    for path, content in zip(targets, old_contents, strict=True):
        path.write_bytes(content)
    for path, content in zip(staged, new_contents, strict=True):
        path.write_bytes(content)

    real_replace = cli.os.replace
    failed = False

    def fail_markdown_publish(source: object, destination: object) -> None:
        nonlocal failed
        if not failed and Path(source) == staged[1] and Path(destination) == targets[1]:
            failed = True
            raise OSError("simulated Markdown replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr(cli.os, "replace", fail_markdown_publish)

    with pytest.raises(OSError, match="simulated Markdown replacement failure"):
        cli._publish_staged_outputs(staged, targets)

    assert tuple(path.read_bytes() for path in targets) == old_contents
    assert not any(path.exists() for path in staged)
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_rollback_failure_retains_recoverable_previous_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = (
        tmp_path / "summary.json",
        tmp_path / "summary.md",
        tmp_path / "records.jsonl.gz",
    )
    staged = (
        tmp_path / "new-summary.json",
        tmp_path / "new-summary.md",
        tmp_path / "new-records.jsonl.gz",
    )
    old_contents = (b"old-json", b"old-markdown", b"old-records")
    for path, content in zip(targets, old_contents, strict=True):
        path.write_bytes(content)
    for path in staged:
        path.write_bytes(b"new")

    real_replace = cli.os.replace
    publish_failed = False
    rollback_failed = False

    def fail_publish_and_rollback(source: object, destination: object) -> None:
        nonlocal publish_failed, rollback_failed
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            not publish_failed
            and source_path == staged[1]
            and destination_path == targets[1]
        ):
            publish_failed = True
            raise OSError("simulated publish failure")
        if (
            publish_failed
            and not rollback_failed
            and source_path != staged[0]
            and destination_path == targets[0]
        ):
            rollback_failed = True
            raise OSError("simulated rollback failure")
        real_replace(source, destination)

    monkeypatch.setattr(cli.os, "replace", fail_publish_and_rollback)

    with pytest.raises(RuntimeError, match="retained backups") as error:
        cli._publish_staged_outputs(staged, targets)

    retained = tuple(tmp_path.glob(".summary.json.*.tmp"))
    assert len(retained) == 1
    assert retained[0].read_bytes() == b"old-json"
    assert str(retained[0]) in str(error.value)
    assert targets[1].read_bytes() == b"old-markdown"
    assert targets[2].read_bytes() == b"old-records"


def test_parallel_failure_signals_running_worker_without_waiting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer_started = threading.Event()
    peer_cancelled = threading.Event()

    def thread_executor(**kwargs: object) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(
            max_workers=int(kwargs["max_workers"]),
            initializer=kwargs["initializer"],  # type: ignore[arg-type]
            initargs=kwargs["initargs"],  # type: ignore[arg-type]
        )

    def controlled_run(task: cli._YearTask) -> YearlyRiichiPointResult:
        cancellation_event = cli._WORKER_CANCELLATION_EVENT
        assert cancellation_event is not None
        if task.year == 2025:
            peer_started.set()
            if not cancellation_event.wait(timeout=2):
                raise AssertionError("parent did not signal cancellation")
            peer_cancelled.set()
            raise RiichiPointAggregationCancelled("controlled cancellation")
        if not peer_started.wait(timeout=2):
            raise AssertionError("peer worker did not start")
        raise ValueError("controlled annual failure")

    monkeypatch.setattr(cli, "ProcessPoolExecutor", thread_executor)
    monkeypatch.setattr(cli, "_run_year", controlled_run)
    monkeypatch.setattr(cli, "_WORKER_CANCELLATION_EVENT", None)

    with pytest.raises(RuntimeError, match="annual scan failed for 2024") as error:
        cli._run_years(
            (2024, 2025),
            tmp_path / "raw",
            _cohort({2024: (0, 0, 0, 0), 2025: (0, 0, 0, 0)}),
            tmp_path / "members",
            progress_interval=1,
            workers=2,
        )

    assert isinstance(error.value.__cause__, ValueError)
    assert "controlled annual failure" in str(error.value.__cause__)
    assert peer_cancelled.wait(timeout=0.1)
