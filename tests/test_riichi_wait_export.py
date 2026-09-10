import gzip
import hashlib
import json
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

import mahjong_analysis.riichi_wait_export as export_module
from mahjong_analysis.hand_waits import FixedMeld
from mahjong_analysis.riichi import ActorDiscard, EstablishedRiichi
from mahjong_analysis.riichi_wait_dataset import (
    DATASET_NAME,
    build_dataset_record,
    iter_dataset_records,
    iter_validated_year_records,
    validate_dataset_integrity,
)
from mahjong_analysis.riichi_wait_export import (
    FileExtractionCounts,
    GitMetadata,
    collect_git_metadata,
    export_riichi_wait_dataset,
    select_year_files,
)
from mahjong_analysis.riichi_wait_records import build_riichi_wait_record

_CLEAN_GIT = GitMetadata("abc123", True)


class StepClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 1.0
        return self.value


def sample_record(year: int = 2025, source: str = "2025/game.mjson"):
    established = EstablishedRiichi(
        actor=0,
        reach_event_index=1,
        declaration_dahai_event_index=2,
        reach_accepted_event_index=3,
        riichi_discard_number=1,
        riichi_declaration_tile="5mr",
        riichi_declaration_tile_kind="5m",
        concealed_tiles_after_discard=("1p", "2p", "3p", "E", "E", "4s", "5s"),
        fixed_melds=(
            FixedMeld(("1m", "1m", "1m", "1m")),
            FixedMeld(("9m", "9m", "9m", "9m")),
        ),
        actor_discards_before_riichi=(
            ActorDiscard(
                discard_number=1,
                tile="5mr",
                tile_kind="5m",
                tsumogiri=True,
                event_index=2,
                is_riichi_declaration=True,
            ),
        ),
    )
    return build_dataset_record(
        build_riichi_wait_record(established),
        year=year,
        relative_source_path=source,
        start_kyoku_line=1,
        reach_line=2,
        declaration_dahai_line=3,
        reach_accepted_line=4,
        bakaze="E",
        kyoku=1,
        honba=0,
        oya=0,
        scores_at_start=(25000, 25000, 25000, 25000),
        dora_marker="5pr",
    )


def write_summary(path: Path, counts: dict[int, int]) -> None:
    path.write_text(
        json.dumps(
            {
                "archive_hashes_verified": True,
                "release_tag": "v2.0.0",
                "repository": "example/source",
                "target_rule_code": "00a9",
                "validation_failed": False,
                "years": [
                    {"year": year, "raw": {"total_mjson_files": count}}
                    for year, count in sorted(counts.items())
                ],
            }
        ),
        encoding="utf-8",
    )


def make_raw_files(root: Path, year: int, names: tuple[str, ...]) -> None:
    for name in names:
        path = root / str(year) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n", encoding="utf-8")


def fake_extractor(calls: list[str] | None = None):
    def extract(path, raw_root, year, write_record):
        source = path.resolve().relative_to(raw_root.resolve()).as_posix()
        if calls is not None:
            calls.append(source)
        write_record(sample_record(year, source))
        return FileExtractionCounts(1, 1, 1)

    return extract


def run_export(
    raw_root: Path,
    output_root: Path,
    summary: Path,
    *,
    years=(2025,),
    max_files=2,
    extractor=None,
    force=False,
    resume=False,
    git=_CLEAN_GIT,
):
    return export_riichi_wait_dataset(
        raw_root=raw_root,
        output_root=output_root,
        years=years,
        validation_summary_path=summary,
        validation_summary_logical_path="data/validation/summary.json",
        git_metadata=git,
        max_files=max_files,
        force=force,
        resume=resume,
        allow_dirty=max_files is not None and not git.worktree_clean,
        progress_interval=1,
        file_extractor=extractor or fake_extractor(),
        utc_now=lambda: datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        monotonic=StepClock(),
    )


def test_input_files_are_relative_posix_sorted_before_limit(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    make_raw_files(raw, 2025, ("z.mjson", "sub/a.mjson", "a.mjson"))

    selected = select_year_files(raw, 2025, max_files=2)

    assert tuple(path.relative_to(raw).as_posix() for path in selected) == (
        "2025/a.mjson",
        "2025/sub/a.mjson",
    )


def test_export_is_streamed_in_input_order_with_exact_counters(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    summary = tmp_path / "summary.json"
    make_raw_files(raw, 2025, ("b.mjson", "a.mjson", "c.mjson"))
    write_summary(summary, {2025: 3})
    calls: list[str] = []

    manifest = run_export(
        raw,
        output,
        summary,
        extractor=fake_extractor(calls),
    )

    assert calls == ["2025/a.mjson", "2025/b.mjson"]
    assert (
        manifest["scope"]["input_selection"]["max_files_before_target_filtering"] == 2
    )
    assert manifest["totals"] == {
        "scanned_files": 2,
        "target_games": 2,
        "east_kyokus": 2,
        "established_riichis": 2,
        "output_records": 2,
    }
    year = manifest["years"][0]
    assert year["established_riichis"] == year["output_records"] == 2
    assert tuple(
        record.relative_source_path
        for record in iter_dataset_records(output / "2025.jsonl.gz")
    ) == (
        "2025/a.mjson",
        "2025/b.mjson",
    )
    assert not (output / "2025.jsonl.gz.part").exists()
    assert not (output / "manifest.json.part").exists()


def test_max_files_is_applied_before_target_filtering(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    summary = tmp_path / "summary.json"
    make_raw_files(raw, 2025, ("a.mjson", "b.mjson", "c.mjson"))
    write_summary(summary, {2025: 3})
    calls: list[str] = []

    def mixed_scope(path, root, year, write_record):
        source = path.relative_to(root).as_posix()
        calls.append(source)
        if source.endswith("a.mjson"):
            return FileExtractionCounts(0, 0, 0)
        write_record(sample_record(year, source))
        return FileExtractionCounts(1, 1, 1)

    manifest = run_export(raw, output, summary, extractor=mixed_scope)

    assert calls == ["2025/a.mjson", "2025/b.mjson"]
    assert manifest["totals"]["scanned_files"] == 2
    assert manifest["totals"]["target_games"] == 1
    assert manifest["totals"]["output_records"] == 1


def test_candidate_order_from_production_is_not_reordered(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    summary = tmp_path / "summary.json"
    make_raw_files(raw, 2025, ("a.mjson",))
    write_summary(summary, {2025: 1})

    def ordered(path, root, year, write_record):
        source = path.relative_to(root).as_posix()
        first = sample_record(year, source)
        write_record(first)
        write_record(replace(first, actor=1))
        return FileExtractionCounts(1, 1, 2)

    run_export(raw, output, summary, max_files=1, extractor=ordered)

    records = tuple(iter_dataset_records(output / "2025.jsonl.gz"))
    assert tuple(record.actor for record in records) == (0, 1)


def test_output_sha_size_and_manifest_have_no_local_absolute_paths(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "private-user" / "raw"
    output = tmp_path / "private-user" / "output"
    summary = tmp_path / "private-user" / "summary.json"
    make_raw_files(raw, 2025, ("a.mjson",))
    write_summary(summary, {2025: 1})

    manifest = run_export(raw, output, summary, max_files=1)

    year = manifest["years"][0]
    payload = (output / "2025.jsonl.gz").read_bytes()
    assert year["compressed_size_bytes"] == len(payload)
    assert year["sha256"] == hashlib.sha256(payload).hexdigest()
    manifest_text = (output / "manifest.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in manifest_text
    assert manifest["source"]["validation_summary_path"] == (
        "data/validation/summary.json"
    )
    assert manifest["generator"]["invocation"] == {
        "years": [2025],
        "mode": "sample",
        "max_files": 1,
    }


def test_validated_reader_checks_mode_hash_and_record_count(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    summary = tmp_path / "summary.json"
    make_raw_files(raw, 2025, ("a.mjson",))
    write_summary(summary, {2025: 1})
    run_export(raw, output, summary, max_files=1)

    records = tuple(iter_validated_year_records(output, 2025, expected_mode="sample"))
    assert len(records) == 1
    with pytest.raises(ValueError, match="mode mismatch"):
        tuple(iter_validated_year_records(output, 2025, expected_mode="full"))

    with open(output / "2025.jsonl.gz", "ab") as file:
        file.write(b"corrupt")
    with pytest.raises(ValueError, match="size mismatch"):
        tuple(iter_validated_year_records(output, 2025, expected_mode="sample"))


def test_gzip_bytes_and_header_are_deterministic(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    summary = tmp_path / "summary.json"
    make_raw_files(raw, 2025, ("a.mjson",))
    write_summary(summary, {2025: 1})

    run_export(raw, tmp_path / "one", summary, max_files=1)
    run_export(raw, tmp_path / "two", summary, max_files=1)

    first = (tmp_path / "one" / "2025.jsonl.gz").read_bytes()
    second = (tmp_path / "two" / "2025.jsonl.gz").read_bytes()
    assert first == second
    assert first[:2] == b"\x1f\x8b"
    assert first[3] & 0x08 == 0
    assert first[4:8] == b"\0\0\0\0"
    with gzip.open(tmp_path / "one" / "2025.jsonl.gz", "rb") as file:
        assert file.read().endswith(b"\n")


def test_failure_leaves_part_and_never_publishes_final(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    summary = tmp_path / "summary.json"
    make_raw_files(raw, 2025, ("a.mjson", "b.mjson"))
    write_summary(summary, {2025: 2})
    calls = 0

    def failing(path, root, year, write_record):
        nonlocal calls
        calls += 1
        write_record(sample_record(year, path.relative_to(root).as_posix()))
        if calls == 2:
            raise RuntimeError("synthetic failure")
        return FileExtractionCounts(1, 1, 1)

    with pytest.raises(RuntimeError, match="synthetic failure"):
        run_export(raw, output, summary, extractor=failing)

    assert (output / "2025.jsonl.gz.part").exists()
    assert (output / "manifest.json.part").exists()
    assert not (output / "2025.jsonl.gz").exists()
    assert not (output / "manifest.json").exists()
    with pytest.raises(FileExistsError, match="use --resume"):
        run_export(raw, output, summary)


def test_resume_skips_hash_verified_complete_year(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    summary = tmp_path / "summary.json"
    make_raw_files(raw, 2009, ("a.mjson",))
    make_raw_files(raw, 2010, ("a.mjson",))
    write_summary(summary, {2009: 1, 2010: 1})
    failed = False

    def fail_second_year(path, root, year, write_record):
        nonlocal failed
        if year == 2010 and not failed:
            failed = True
            raise RuntimeError("stop in 2010")
        return fake_extractor()(path, root, year, write_record)

    with pytest.raises(RuntimeError, match="stop in 2010"):
        run_export(
            raw,
            output,
            summary,
            years=(2009, 2010),
            max_files=None,
            extractor=fail_second_year,
        )
    first_hash = hashlib.sha256((output / "2009.jsonl.gz").read_bytes()).hexdigest()
    checkpoint_path = output / "manifest.json.part"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["years"][0]["status"] = "ready"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    calls: list[str] = []

    manifest = run_export(
        raw,
        output,
        summary,
        years=(2009, 2010),
        max_files=None,
        extractor=fake_extractor(calls),
        resume=True,
    )

    assert calls == ["2010/a.mjson"]
    assert (
        hashlib.sha256((output / "2009.jsonl.gz").read_bytes()).hexdigest()
        == first_hash
    )
    assert [entry["year"] for entry in manifest["years"]] == [2009, 2010]


def test_resume_restarts_failed_year_instead_of_appending(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    summary = tmp_path / "summary.json"
    make_raw_files(raw, 2025, ("a.mjson", "b.mjson"))
    write_summary(summary, {2025: 2})
    first_attempt: list[str] = []

    def fail_after_first(path, root, year, write_record):
        source = path.relative_to(root).as_posix()
        first_attempt.append(source)
        write_record(sample_record(year, source))
        if source.endswith("b.mjson"):
            raise RuntimeError("stop")
        return FileExtractionCounts(1, 1, 1)

    with pytest.raises(RuntimeError, match="stop"):
        run_export(raw, output, summary, extractor=fail_after_first)
    resumed: list[str] = []

    manifest = run_export(
        raw,
        output,
        summary,
        extractor=fake_extractor(resumed),
        resume=True,
    )

    assert first_attempt == ["2025/a.mjson", "2025/b.mjson"]
    assert resumed == ["2025/a.mjson", "2025/b.mjson"]
    assert manifest["totals"]["output_records"] == 2
    assert len(tuple(iter_dataset_records(output / "2025.jsonl.gz"))) == 2


def test_resume_rejects_changed_generator_identity(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    summary = tmp_path / "summary.json"
    make_raw_files(raw, 2025, ("a.mjson",))
    write_summary(summary, {2025: 1})

    def failing(_path, _root, _year, _write_record):
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError):
        run_export(raw, output, summary, max_files=1, extractor=failing)

    with pytest.raises(ValueError, match="configuration mismatch: generator"):
        run_export(
            raw,
            output,
            summary,
            max_files=1,
            resume=True,
            git=GitMetadata("different", True),
        )


def test_resume_rejects_corrupt_completed_year(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    summary = tmp_path / "summary.json"
    make_raw_files(raw, 2009, ("a.mjson",))
    make_raw_files(raw, 2010, ("a.mjson",))
    write_summary(summary, {2009: 1, 2010: 1})

    def fail_2010(path, root, year, write_record):
        if year == 2010:
            raise RuntimeError("stop")
        return fake_extractor()(path, root, year, write_record)

    with pytest.raises(RuntimeError):
        run_export(
            raw,
            output,
            summary,
            years=(2009, 2010),
            max_files=None,
            extractor=fail_2010,
        )
    with open(output / "2009.jsonl.gz", "ab") as file:
        file.write(b"corrupt")

    with pytest.raises(ValueError, match="size mismatch"):
        run_export(
            raw,
            output,
            summary,
            years=(2009, 2010),
            max_files=None,
            resume=True,
        )


def test_existing_output_requires_force_and_force_replaces_atomically(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    summary = tmp_path / "summary.json"
    make_raw_files(raw, 2025, ("a.mjson",))
    write_summary(summary, {2025: 1})
    run_export(raw, output, summary, max_files=1)

    with pytest.raises(FileExistsError, match="already exists"):
        run_export(raw, output, summary, max_files=1)

    manifest = run_export(raw, output, summary, max_files=1, force=True)
    assert manifest["totals"]["output_records"] == 1
    assert not (output / "manifest.json.previous").exists()


def test_failed_force_keeps_previous_annual_file_intact(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    summary = tmp_path / "summary.json"
    make_raw_files(raw, 2025, ("a.mjson",))
    write_summary(summary, {2025: 1})
    run_export(raw, output, summary, max_files=1)
    previous = (output / "2025.jsonl.gz").read_bytes()

    def failing(_path, _root, _year, _write_record):
        raise RuntimeError("replacement failed")

    with pytest.raises(RuntimeError, match="replacement failed"):
        run_export(
            raw,
            output,
            summary,
            max_files=1,
            force=True,
            extractor=failing,
        )

    assert (output / "2025.jsonl.gz").read_bytes() == previous
    assert not (output / "manifest.json.previous").exists()
    assert (output / "manifest.json").exists()


def test_sample_and_full_modes_cannot_share_output_root(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    summary = tmp_path / "summary.json"
    make_raw_files(raw, 2025, ("a.mjson",))
    write_summary(summary, {2025: 1})
    run_export(raw, output, summary, max_files=1)

    with pytest.raises(ValueError, match="separate roots"):
        run_export(raw, output, summary, max_files=None, force=True)


def test_full_mode_rejects_dirty_worktree_and_sample_can_opt_in(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    summary = tmp_path / "summary.json"
    make_raw_files(raw, 2025, ("a.mjson",))
    write_summary(summary, {2025: 1})
    dirty = GitMetadata("abc123", False)

    with pytest.raises(ValueError, match="clean Git"):
        run_export(raw, tmp_path / "full", summary, max_files=None, git=dirty)

    manifest = run_export(raw, tmp_path / "sample", summary, max_files=1, git=dirty)
    assert manifest["generator"]["worktree_clean"] is False


def test_zero_input_files_is_an_error(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    (raw / "2009").mkdir(parents=True)
    summary = tmp_path / "summary.json"
    write_summary(summary, {2009: 0})

    with pytest.raises(FileNotFoundError, match="no input"):
        run_export(
            raw,
            tmp_path / "output",
            summary,
            years=(2009,),
            max_files=None,
        )


def test_full_export_rejects_raw_count_mismatch(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    summary = tmp_path / "summary.json"
    make_raw_files(raw, 2009, ("a.mjson",))
    write_summary(summary, {2009: 2})

    with pytest.raises(ValueError, match="raw file count mismatch"):
        run_export(
            raw,
            output,
            summary,
            years=(2009,),
            max_files=None,
        )

    assert not (output / "2009.jsonl.gz").exists()


def test_year_with_input_but_zero_records_is_valid(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    summary = tmp_path / "summary.json"
    make_raw_files(raw, 2009, ("a.mjson",))
    write_summary(summary, {2009: 1})

    def no_records(_path, _root, _year, _write_record):
        return FileExtractionCounts(1, 0, 0)

    manifest = run_export(
        raw,
        output,
        summary,
        years=(2009,),
        max_files=None,
        extractor=no_records,
    )

    assert manifest["totals"] == {
        "scanned_files": 1,
        "target_games": 1,
        "east_kyokus": 0,
        "established_riichis": 0,
        "output_records": 0,
    }
    with gzip.open(output / "2009.jsonl.gz", "rb") as file:
        assert file.read() == b""


def test_max_files_is_rejected_for_multiple_years(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    write_summary(summary, {2009: 0, 2010: 0})
    with pytest.raises(ValueError, match="exactly one year"):
        run_export(
            tmp_path / "raw",
            tmp_path / "output",
            summary,
            years=(2009, 2010),
            max_files=1,
        )


def test_output_manifest_identity_is_fixed() -> None:
    assert DATASET_NAME == "riichi-waits-v1"


def test_resume_promotes_ready_part_when_final_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    summary = tmp_path / "summary.json"
    make_raw_files(raw, 2025, ("a.mjson",))
    write_summary(summary, {2025: 1})
    original_replace = export_module.os.replace
    crashed = False

    def crash_before_year_rename(source, target):
        nonlocal crashed
        if (
            not crashed
            and Path(source).name == "2025.jsonl.gz.part"
            and Path(target).name == "2025.jsonl.gz"
        ):
            crashed = True
            raise RuntimeError("crash D")
        return original_replace(source, target)

    monkeypatch.setattr(export_module.os, "replace", crash_before_year_rename)
    with pytest.raises(RuntimeError, match="crash D"):
        run_export(raw, output, summary, max_files=1)
    monkeypatch.setattr(export_module.os, "replace", original_replace)

    assert not (output / "2025.jsonl.gz").exists()
    assert (output / "2025.jsonl.gz.part").exists()
    checkpoint = json.loads((output / "manifest.json.part").read_text())
    assert checkpoint["years"][0]["status"] == "ready"
    manifest = run_export(raw, output, summary, max_files=1, resume=True)
    assert manifest["totals"]["output_records"] == 1
    assert not (output / "2025.jsonl.gz.part").exists()


def test_force_resume_prefers_valid_new_part_over_old_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    summary = tmp_path / "summary.json"
    make_raw_files(raw, 2025, ("a.mjson",))
    write_summary(summary, {2025: 1})
    run_export(raw, output, summary, max_files=1)
    old_bytes = (output / "2025.jsonl.gz").read_bytes()

    def changed(path, root, year, write_record):
        source = path.relative_to(root).as_posix()
        write_record(replace(sample_record(year, source), actor=1))
        return FileExtractionCounts(1, 1, 1)

    original_replace = export_module.os.replace
    crashed = False

    def crash_before_year_rename(source, target):
        nonlocal crashed
        if (
            not crashed
            and Path(source).name == "2025.jsonl.gz.part"
            and Path(target).name == "2025.jsonl.gz"
        ):
            crashed = True
            raise RuntimeError("force crash D")
        return original_replace(source, target)

    monkeypatch.setattr(export_module.os, "replace", crash_before_year_rename)
    with pytest.raises(RuntimeError, match="force crash D"):
        run_export(
            raw,
            output,
            summary,
            max_files=1,
            force=True,
            extractor=changed,
        )
    monkeypatch.setattr(export_module.os, "replace", original_replace)

    assert (output / "manifest.json.previous").exists()
    assert not (output / "manifest.json").exists()
    assert (output / "2025.jsonl.gz").read_bytes() == old_bytes
    assert (output / "2025.jsonl.gz.part").exists()
    manifest = run_export(
        raw,
        output,
        summary,
        max_files=1,
        resume=True,
        extractor=changed,
    )
    assert manifest["totals"]["output_records"] == 1
    records = tuple(iter_dataset_records(output / "2025.jsonl.gz"))
    assert tuple(record.actor for record in records) == (1,)


def test_final_ready_checkpoint_is_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    summary = tmp_path / "summary.json"
    make_raw_files(raw, 2025, ("a.mjson",))
    write_summary(summary, {2025: 1})
    original_publish = export_module._publish_final_ready_checkpoint

    def crash_before_manifest_publish(*_args, **_kwargs):
        raise RuntimeError("crash F")

    monkeypatch.setattr(
        export_module,
        "_publish_final_ready_checkpoint",
        crash_before_manifest_publish,
    )
    with pytest.raises(RuntimeError, match="crash F"):
        run_export(raw, output, summary, max_files=1)
    monkeypatch.setattr(
        export_module,
        "_publish_final_ready_checkpoint",
        original_publish,
    )

    checkpoint = json.loads((output / "manifest.json.part").read_text())
    assert checkpoint["checkpoint"] is True
    assert checkpoint["checkpoint_state"] == "final_ready"
    assert not (output / "manifest.json").exists()
    manifest = run_export(raw, output, summary, max_files=1, resume=True)
    assert manifest["totals"]["output_records"] == 1
    assert not (output / "manifest.json.part").exists()


def test_resume_cleans_final_ready_checkpoint_after_manifest_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    summary = tmp_path / "summary.json"
    make_raw_files(raw, 2025, ("a.mjson",))
    write_summary(summary, {2025: 1})
    original_write = export_module._write_checkpoint

    def crash_after_manifest_publish(path, value):
        original_write(path, value)
        if Path(path).name == "manifest.json":
            raise RuntimeError("post-publish crash")

    monkeypatch.setattr(
        export_module, "_write_checkpoint", crash_after_manifest_publish
    )
    with pytest.raises(RuntimeError, match="post-publish crash"):
        run_export(raw, output, summary, max_files=1)
    monkeypatch.setattr(export_module, "_write_checkpoint", original_write)

    assert (output / "manifest.json").exists()
    assert (output / "manifest.json.part").exists()
    manifest_before = (output / "manifest.json").read_bytes()
    checkpoint_before = (output / "manifest.json.part").read_bytes()
    with pytest.raises(FileExistsError, match="--force cannot reset"):
        run_export(raw, output, summary, max_files=1, force=True)
    assert (output / "manifest.json").read_bytes() == manifest_before
    assert (output / "manifest.json.part").read_bytes() == checkpoint_before

    manifest = run_export(raw, output, summary, max_files=1, resume=True)
    assert manifest["totals"]["output_records"] == 1
    assert not (output / "manifest.json.part").exists()


@pytest.mark.parametrize("max_files", [None, 1])
def test_unfinished_checkpoint_rejects_force_and_allows_resume(
    tmp_path: Path,
    max_files: int | None,
) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    summary = tmp_path / "summary.json"
    make_raw_files(raw, 2025, ("a.mjson",))
    write_summary(summary, {2025: 1})

    def failing(*_args):
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        run_export(
            raw,
            output,
            summary,
            max_files=max_files,
            extractor=failing,
        )
    checkpoint_before = (output / "manifest.json.part").read_bytes()
    part_before = (output / "2025.jsonl.gz.part").read_bytes()
    with pytest.raises(FileExistsError, match="--force cannot reset"):
        run_export(
            raw,
            output,
            summary,
            max_files=max_files,
            force=True,
        )
    assert (output / "manifest.json.part").read_bytes() == checkpoint_before
    assert (output / "2025.jsonl.gz.part").read_bytes() == part_before

    manifest = run_export(
        raw,
        output,
        summary,
        max_files=max_files,
        resume=True,
    )
    assert manifest["totals"]["output_records"] == 1
    assert not (output / "manifest.json.part").exists()


@pytest.mark.parametrize(
    ("initial_max_files", "replacement_max_files"),
    [(None, 1), (1, None)],
)
def test_unfinished_checkpoint_rejects_force_mode_change(
    tmp_path: Path,
    initial_max_files: int | None,
    replacement_max_files: int | None,
) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    summary = tmp_path / "summary.json"
    make_raw_files(raw, 2025, ("a.mjson",))
    write_summary(summary, {2025: 1})

    def failing(*_args):
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        run_export(
            raw,
            output,
            summary,
            max_files=initial_max_files,
            extractor=failing,
        )
    checkpoint_before = (output / "manifest.json.part").read_bytes()
    with pytest.raises(FileExistsError, match="--force cannot reset"):
        run_export(
            raw,
            output,
            summary,
            max_files=replacement_max_files,
            force=True,
        )
    assert (output / "manifest.json.part").read_bytes() == checkpoint_before


def test_completed_multi_year_full_rejects_single_year_force(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    summary = tmp_path / "summary.json"
    make_raw_files(raw, 2009, ("a.mjson",))
    make_raw_files(raw, 2010, ("a.mjson",))
    write_summary(summary, {2009: 1, 2010: 1})
    run_export(raw, output, summary, years=(2009, 2010), max_files=None)

    with pytest.raises(ValueError, match="same year selection"):
        run_export(
            raw,
            output,
            summary,
            years=(2009,),
            max_files=None,
            force=True,
        )
    assert (output / "manifest.json").exists()
    assert (output / "2010.jsonl.gz").exists()


@pytest.mark.parametrize(
    ("initial_years", "replacement_years"),
    [((2009, 2010), (2009,)), ((2009,), (2009, 2010))],
)
def test_incomplete_multi_year_rejects_force_scope_change(
    tmp_path: Path,
    initial_years: tuple[int, ...],
    replacement_years: tuple[int, ...],
) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    summary = tmp_path / "summary.json"
    make_raw_files(raw, 2009, ("a.mjson",))
    make_raw_files(raw, 2010, ("a.mjson",))
    write_summary(summary, {2009: 1, 2010: 1})

    def failing(*_args):
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        run_export(
            raw,
            output,
            summary,
            years=initial_years,
            max_files=None,
            extractor=failing,
        )
    checkpoint_before = (output / "manifest.json.part").read_bytes()
    with pytest.raises(FileExistsError, match="--force cannot reset"):
        run_export(
            raw,
            output,
            summary,
            years=replacement_years,
            max_files=None,
            force=True,
        )
    assert (output / "manifest.json.part").read_bytes() == checkpoint_before


def test_unexpected_annual_part_without_checkpoint_is_rejected(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    summary = tmp_path / "summary.json"
    make_raw_files(raw, 2025, ("a.mjson",))
    write_summary(summary, {2025: 1})
    output.mkdir()
    (output / "2025.jsonl.gz.part").write_bytes(b"unknown")

    with pytest.raises(ValueError, match="without a checkpoint"):
        run_export(raw, output, summary, max_files=1, force=True)


def test_completed_dataset_with_unexpected_part_is_rejected(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    summary = tmp_path / "summary.json"
    make_raw_files(raw, 2025, ("a.mjson",))
    write_summary(summary, {2025: 1})
    run_export(raw, output, summary, max_files=1)
    (output / "2025.jsonl.gz.part").write_bytes(b"unknown")

    with pytest.raises(ValueError, match="unexpected annual part"):
        run_export(raw, output, summary, max_files=1, force=True)


def test_resume_rejects_complete_year_with_unexpected_part(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    summary = tmp_path / "summary.json"
    make_raw_files(raw, 2009, ("a.mjson",))
    make_raw_files(raw, 2010, ("a.mjson",))
    write_summary(summary, {2009: 1, 2010: 1})

    def fail_2010(path, root, year, write_record):
        if year == 2010:
            raise RuntimeError("stop")
        return fake_extractor()(path, root, year, write_record)

    with pytest.raises(RuntimeError, match="stop"):
        run_export(
            raw,
            output,
            summary,
            years=(2009, 2010),
            max_files=None,
            extractor=fail_2010,
        )
    (output / "2009.jsonl.gz.part").write_bytes(b"unknown")
    with pytest.raises(ValueError, match="complete year has unexpected part"):
        run_export(
            raw,
            output,
            summary,
            years=(2009, 2010),
            max_files=None,
            resume=True,
        )


def test_resume_rejects_checkpoint_counter_inconsistency(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    summary = tmp_path / "summary.json"
    make_raw_files(raw, 2009, ("a.mjson",))
    make_raw_files(raw, 2010, ("a.mjson",))
    write_summary(summary, {2009: 1, 2010: 1})

    def fail_2010(path, root, year, write_record):
        if year == 2010:
            raise RuntimeError("stop")
        return fake_extractor()(path, root, year, write_record)

    with pytest.raises(RuntimeError, match="stop"):
        run_export(
            raw,
            output,
            summary,
            years=(2009, 2010),
            max_files=None,
            extractor=fail_2010,
        )
    checkpoint_path = output / "manifest.json.part"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["years"][0]["established_riichis"] = 2
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(ValueError, match="must equal"):
        run_export(
            raw,
            output,
            summary,
            years=(2009, 2010),
            max_files=None,
            resume=True,
        )


def test_dataset_integrity_checks_missing_size_sha_and_deep_count(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    summary = tmp_path / "summary.json"
    make_raw_files(raw, 2025, ("a.mjson",))
    write_summary(summary, {2025: 1})

    missing_output = tmp_path / "missing"
    run_export(raw, missing_output, summary, max_files=1)
    (missing_output / "2025.jsonl.gz").unlink()
    with pytest.raises(FileNotFoundError):
        validate_dataset_integrity(missing_output, expected_mode="sample")

    size_output = tmp_path / "size"
    run_export(raw, size_output, summary, max_files=1)
    with open(size_output / "2025.jsonl.gz", "ab") as file:
        file.write(b"x")
    with pytest.raises(ValueError, match="size mismatch"):
        validate_dataset_integrity(size_output, expected_mode="sample")

    sha_output = tmp_path / "sha"
    run_export(raw, sha_output, summary, max_files=1)
    annual = sha_output / "2025.jsonl.gz"
    payload = bytearray(annual.read_bytes())
    payload[-1] ^= 1
    annual.write_bytes(payload)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_dataset_integrity(sha_output, expected_mode="sample")

    count_output = tmp_path / "count"
    run_export(raw, count_output, summary, max_files=1)
    manifest_path = count_output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for container in (manifest["years"][0], manifest["totals"]):
        container["established_riichis"] += 1
        container["output_records"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    validate_dataset_integrity(count_output, expected_mode="sample", deep=False)
    with pytest.raises(ValueError, match="record count mismatch"):
        validate_dataset_integrity(count_output, expected_mode="sample", deep=True)


def test_collect_git_metadata_uses_head_and_ignores_ignored_files(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(("git", "init", "-q", str(repository)), check=True)
    subprocess.run(
        ("git", "-C", str(repository), "config", "user.email", "test@example.com"),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(repository), "config", "user.name", "Test"),
        check=True,
    )
    (repository / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    (repository / "tracked.txt").write_text("original\n", encoding="utf-8")
    subprocess.run(("git", "-C", str(repository), "add", "."), check=True)
    subprocess.run(
        ("git", "-C", str(repository), "commit", "-q", "-m", "initial"),
        check=True,
    )
    (repository / "ignored").mkdir()
    (repository / "ignored" / "output.bin").write_bytes(b"ignored")

    clean = collect_git_metadata(repository)
    assert clean.worktree_clean is True
    assert (
        clean.commit
        == subprocess.run(
            ("git", "-C", str(repository), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )

    (repository / "tracked.txt").write_text("changed\n", encoding="utf-8")
    assert collect_git_metadata(repository).worktree_clean is False


def test_collect_git_metadata_fails_closed_when_git_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def unavailable(*_args, **_kwargs):
        raise FileNotFoundError("git unavailable")

    monkeypatch.setattr(export_module.subprocess, "run", unavailable)
    with pytest.raises(FileNotFoundError, match="git unavailable"):
        collect_git_metadata(tmp_path)


def test_multi_year_force_failure_resumes_without_mixed_completion_marker(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    summary = tmp_path / "summary.json"
    make_raw_files(raw, 2009, ("a.mjson",))
    make_raw_files(raw, 2010, ("a.mjson",))
    write_summary(summary, {2009: 1, 2010: 1})
    run_export(raw, output, summary, years=(2009, 2010), max_files=None)

    def changed_then_fail(path, root, year, write_record):
        if year == 2010:
            raise RuntimeError("stop after first replacement")
        source = path.relative_to(root).as_posix()
        write_record(replace(sample_record(year, source), actor=1))
        return FileExtractionCounts(1, 1, 1)

    with pytest.raises(RuntimeError, match="stop after first replacement"):
        run_export(
            raw,
            output,
            summary,
            years=(2009, 2010),
            max_files=None,
            force=True,
            extractor=changed_then_fail,
        )
    assert not (output / "manifest.json").exists()
    assert (output / "manifest.json.previous").exists()

    manifest = run_export(
        raw,
        output,
        summary,
        years=(2009, 2010),
        max_files=None,
        resume=True,
        extractor=fake_extractor(),
    )
    assert manifest["totals"]["output_records"] == 2
    assert not (output / "manifest.json.previous").exists()
