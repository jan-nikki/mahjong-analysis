from pathlib import Path

import pytest

import analysis.benchmark_riichi_wait_summary as cli
from mahjong_analysis.riichi_wait_summary import (
    SummaryBenchmarkResult,
    SummaryBenchmarkYearResult,
)


def _result() -> SummaryBenchmarkResult:
    return SummaryBenchmarkResult(
        workers_requested=2,
        workers_used=2,
        years=(2024, 2025),
        record_limit_per_year=100,
        records_processed=200,
        elapsed_seconds=2.0,
        records_per_second=100.0,
        year_results=(
            SummaryBenchmarkYearResult(2024, 100, 1.5),
            SummaryBenchmarkYearResult(2025, 100, 1.25),
        ),
    )


def test_benchmark_cli_defaults_are_fixed_and_non_publishing() -> None:
    args = cli.parse_args([])

    assert args.dataset_root == cli.PROJECT_ROOT / "data/processed/riichi-waits-v1"
    assert args.years == list(range(2018, 2026))
    assert args.record_limit_per_year == 100_000
    assert args.workers == [1, 2, 4, 6]
    assert not hasattr(args, "output_root")


@pytest.mark.parametrize(
    "arguments",
    (("--record-limit-per-year", "0"), ("--workers", "0")),
)
def test_benchmark_cli_rejects_non_positive_values(arguments: tuple[str, str]) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(arguments)


def test_benchmark_cli_uses_one_sample_for_all_worker_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[object, ...]] = []

    def benchmark(root: Path, **kwargs: object) -> tuple[SummaryBenchmarkResult, ...]:
        calls.append((root, kwargs))
        return (_result(),)

    monkeypatch.setattr(cli, "benchmark_riichi_wait_dataset", benchmark)

    assert (
        cli.main(
            [
                "--dataset-root",
                str(tmp_path),
                "--years",
                "2024",
                "2025",
                "--record-limit-per-year",
                "100",
                "--workers",
                "1",
                "2",
                "4",
                "6",
            ]
        )
        == 0
    )

    assert calls == [
        (
            tmp_path,
            {
                "years": (2024, 2025),
                "record_limit_per_year": 100,
                "worker_counts": (1, 2, 4, 6),
            },
        )
    ]
    output = capsys.readouterr().out
    assert "workers_requested=2 workers_used=2" in output
    assert "records_processed=200" in output
    assert "records_per_second=100.00" in output
    assert "year=2024 records_processed=100" in output
    assert "filesystem cache" in output
