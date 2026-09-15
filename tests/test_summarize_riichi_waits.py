from pathlib import Path
from types import SimpleNamespace

import pytest

import analysis.summarize_riichi_waits as cli
from mahjong_analysis.riichi_wait_summary import AnalysisGitMetadata


def test_cli_defaults_to_canonical_paths() -> None:
    args = cli.parse_args([])

    assert args.dataset_root == cli.PROJECT_ROOT / "data/processed/riichi-waits-v1"
    assert args.output_root == (
        cli.PROJECT_ROOT / "research/results/riichi-wait-summary-v1"
    )
    assert args.workers == 1
    assert args.progress_interval == 100_000


def test_cli_rejects_non_positive_progress_interval() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--progress-interval", "0"])


def test_cli_rejects_non_positive_workers() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--workers", "0"])


def test_cli_collects_git_before_summary_and_writes_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "results"
    analysis = SimpleNamespace(name="analysis")
    calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        cli,
        "collect_analysis_git_metadata",
        lambda root: calls.append(("git", root)) or AnalysisGitMetadata("abc123", True),
    )
    monkeypatch.setattr(
        cli,
        "summarize_riichi_wait_dataset",
        lambda root, **kwargs: calls.append(("summarize", root, kwargs)) or analysis,
    )
    monkeypatch.setattr(
        cli,
        "write_summary_outputs",
        lambda value, root: (
            calls.append(("write", value, root)) or (root / "result.json",)
        ),
    )

    result = cli.main(
        [
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(output_root),
            "--workers",
            "4",
            "--progress-interval",
            "7",
        ]
    )

    assert result == 0
    assert calls[0] == ("git", cli.PROJECT_ROOT)
    assert calls[1][0:2] == ("summarize", dataset_root)
    assert calls[1][2]["analysis_git"] == AnalysisGitMetadata("abc123", True)
    assert calls[1][2]["project_root"] == cli.PROJECT_ROOT
    assert calls[1][2]["workers"] == 4
    assert calls[1][2]["progress_interval"] == 7
    assert calls[2] == ("write", analysis, output_root)


def test_cli_worker_failure_does_not_publish_or_change_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "results"
    output_root.mkdir()
    current_path = output_root / "current.json"
    current_before = b'{"generation_id":"existing"}\n'
    current_path.write_bytes(current_before)
    publication_calls: list[tuple[object, object]] = []

    monkeypatch.setattr(
        cli,
        "collect_analysis_git_metadata",
        lambda root: AnalysisGitMetadata("abc123", True),
    )

    def fail_summary(root: Path, **kwargs: object) -> object:
        raise RuntimeError("worker failed")

    monkeypatch.setattr(cli, "summarize_riichi_wait_dataset", fail_summary)
    monkeypatch.setattr(
        cli,
        "write_summary_outputs",
        lambda value, root: publication_calls.append((value, root)),
    )

    with pytest.raises(RuntimeError, match="worker failed"):
        cli.main(
            [
                "--dataset-root",
                str(tmp_path / "dataset"),
                "--output-root",
                str(output_root),
                "--workers",
                "4",
            ]
        )

    assert publication_calls == []
    assert current_path.read_bytes() == current_before
