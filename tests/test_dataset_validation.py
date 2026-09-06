import gzip
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from analysis.validate_dataset_v2_0_0 import main, parse_args
from mahjong_analysis.dataset_validation import (
    AKA_FLAG_STATUS_VALUES,
    COMPRESSION_VALUES,
    FILENAME_YEAR_STATUS_VALUES,
    FIRST_EVENT_STATUS_VALUES,
    ReleaseAsset,
    ReleaseAssetManifest,
    build_dataset_validation_summary,
    build_year_validation_summary,
    dataset_validation_summary_to_dict,
    inspect_mjai_file,
    load_release_asset_manifest,
    render_dataset_validation_markdown,
    validate_raw_year_directory,
    validate_release_archive,
    write_dataset_validation_outputs,
)


ASSET_MANIFEST_PATH = (
    Path(__file__).parents[1]
    / "data"
    / "validation"
    / "tenhou-to-mjai-v2.0.0-assets.json"
)


def _filename(
    *,
    year: int = 2020,
    rule_code: str = "00a9",
    game_id: str = "00000001",
) -> str:
    return f"{year}010100gm-{rule_code}-0000-{game_id}.mjson"


def _write_lines(
    path: Path,
    lines: list[bytes],
    *,
    compressed: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = b"".join(line + b"\n" for line in lines)
    if compressed:
        with gzip.open(path, mode="wb") as file:
            file.write(content)
    else:
        path.write_bytes(content)
    return path


def _write_events(
    path: Path,
    events: list[object],
    *,
    compressed: bool = False,
) -> Path:
    lines = [json.dumps(event, ensure_ascii=False).encode("utf-8") for event in events]
    return _write_lines(path, lines, compressed=compressed)


def _start_game(aka_flag: object = True) -> dict[str, object]:
    return {
        "type": "start_game",
        "aka_flag": aka_flag,
        "names": ["東家", "南家", "西家", "北家"],
    }


def _asset_for(path: Path, *, year: int = 2020) -> ReleaseAsset:
    return ReleaseAsset(
        year=year,
        filename=path.name,
        expected_size_bytes=path.stat().st_size,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _write_zip(
    path: Path,
    *,
    mjson_entries: int = 1,
    mjson_names: list[str] | None = None,
    include_non_mjson: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, mode="w") as archive:
        names = (
            [f"game-{index}.mjson" for index in range(mjson_entries)]
            if mjson_names is None
            else mjson_names
        )
        for name in names:
            archive.writestr(name, b"data")
        if include_non_mjson:
            archive.writestr("README.txt", "metadata")
    return path


def test_inspect_mjai_file_reads_plain_and_gzip_by_magic(tmp_path: Path) -> None:
    filename = _filename()
    plain_path = _write_events(
        tmp_path / "gzip-looking-year" / filename,
        [_start_game()],
    )
    gzip_path = _write_events(
        tmp_path / "plain-looking-year" / filename,
        [_start_game()],
        compressed=True,
    )

    plain = inspect_mjai_file(plain_path, 2020)
    compressed = inspect_mjai_file(gzip_path, 2020)

    assert plain_path.suffix == gzip_path.suffix == ".mjson"
    assert plain_path.read_bytes()[:2] != b"\x1f\x8b"
    assert gzip_path.read_bytes()[:2] == b"\x1f\x8b"
    assert plain.compression == "plain"
    assert compressed.compression == "gzip"
    assert plain.first_event_status == "readable_start_game"
    assert compressed.first_event_status == "readable_start_game"
    assert plain.aka_flag_status == compressed.aka_flag_status == "true"
    assert plain.target_game is compressed.target_game is True


@pytest.mark.parametrize(
    ("event", "expected_status", "target", "issue_category"),
    [
        (_start_game(True), "true", True, None),
        (_start_game(False), "false", False, None),
        ({"type": "start_game"}, "missing", False, "aka_flag_missing"),
        (_start_game("true"), "invalid", False, "aka_flag_invalid"),
        (_start_game(1), "invalid", False, "aka_flag_invalid"),
        (_start_game(0), "invalid", False, "aka_flag_invalid"),
    ],
)
def test_inspect_mjai_file_classifies_aka_flag_strictly(
    tmp_path: Path,
    event: dict[str, object],
    expected_status: str,
    target: bool,
    issue_category: str | None,
) -> None:
    path = _write_events(tmp_path / _filename(), [event])

    result = inspect_mjai_file(path, 2020)

    assert result.aka_flag_status == expected_status
    assert result.target_game is target
    assert [issue.category for issue in result.issues] == (
        [] if issue_category is None else [issue_category]
    )


def test_invalid_filename_still_inspects_first_event(tmp_path: Path) -> None:
    path = _write_events(tmp_path / "invalid.mjson", [_start_game()])

    result = inspect_mjai_file(path, 2020)

    assert result.filename_valid is False
    assert result.filename_year is None
    assert result.filename_year_status == "unavailable"
    assert result.rule_code is None
    assert result.compression == "plain"
    assert result.first_event_status == "readable_start_game"
    assert result.aka_flag_status == "true"
    assert result.target_game is False
    assert [issue.category for issue in result.issues] == ["invalid_filename"]


def test_inspect_mjai_file_records_year_mismatch(tmp_path: Path) -> None:
    path = _write_events(tmp_path / _filename(year=2019), [_start_game()])

    result = inspect_mjai_file(path, 2020)

    assert result.filename_year == 2019
    assert result.filename_year_status == "mismatch"
    assert [issue.category for issue in result.issues] == ["year_mismatch"]


@pytest.mark.parametrize(
    ("content", "status", "category"),
    [
        (b"{invalid}\n", "json_decode_error", "json_decode_error"),
        (b"\xff\n", "unreadable", "utf8_decode_error"),
        (b"[]\n", "not_object", "first_event_not_object"),
        (
            b'{"type":"start_kyoku"}\n',
            "not_start_game",
            "first_event_not_start_game",
        ),
        (b"", "empty", "empty_file"),
    ],
)
def test_inspect_mjai_file_classifies_first_event_errors(
    tmp_path: Path,
    content: bytes,
    status: str,
    category: str,
) -> None:
    path = tmp_path / _filename()
    path.write_bytes(content)

    result = inspect_mjai_file(path, 2020)

    assert result.first_event_status == status
    assert result.aka_flag_status == "unavailable"
    assert result.target_game is False
    assert [issue.category for issue in result.issues] == [category]


def test_inspect_mjai_file_rejects_broken_gzip(tmp_path: Path) -> None:
    path = tmp_path / _filename()
    path.write_bytes(b"\x1f\x8b\x00broken gzip")

    result = inspect_mjai_file(path, 2020)

    assert result.compression == "gzip"
    assert result.first_event_status == "unreadable"
    assert result.aka_flag_status == "unavailable"
    assert [issue.category for issue in result.issues] == ["gzip_first_line_read_error"]


def test_inspect_mjai_file_records_binary_open_error(tmp_path: Path) -> None:
    path = tmp_path / _filename()

    result = inspect_mjai_file(path, 2020)

    assert result.compression == "unknown"
    assert result.first_event_status == "unreadable"
    assert result.aka_flag_status == "unavailable"
    assert result.issues[0].category == "binary_open_error"
    assert result.issues[0].relative_path == f"2020/{path.name}"
    assert str(tmp_path) not in (result.issues[0].message or "")


@pytest.mark.parametrize("compressed", [False, True])
def test_inspect_mjai_file_does_not_read_invalid_second_line(
    tmp_path: Path,
    compressed: bool,
) -> None:
    path = _write_lines(
        tmp_path / _filename(),
        [json.dumps(_start_game()).encode("utf-8"), b"{invalid}"],
        compressed=compressed,
    )

    result = inspect_mjai_file(path, 2020)

    assert result.first_event_status == "readable_start_game"
    assert result.target_game is True
    assert result.issues == ()


def test_validate_raw_year_aggregates_rules_aka_and_errors(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "2020"
    _write_events(raw / _filename(game_id="00000001"), [_start_game(True)])
    _write_events(raw / _filename(game_id="00000002"), [_start_game(False)])
    _write_events(
        raw / _filename(rule_code="00e1", game_id="00000003"),
        [_start_game(True)],
    )
    _write_events(raw / "000-invalid.mjson", [_start_game(True)])
    _write_lines(
        raw / _filename(game_id="00000004"),
        [b"{invalid}"],
    )

    summary = validate_raw_year_directory(2020, raw)

    assert summary.total_mjson_files == 5
    assert summary.valid_filenames == 4
    assert summary.invalid_filenames == 1
    assert summary.rule_code_counts == {"00a9": 3, "00e1": 1}
    assert summary.aka_flag_counts == {
        "true": 3,
        "false": 1,
        "missing": 0,
        "invalid": 0,
        "unavailable": 1,
    }
    assert summary.aka_flag_by_rule_code["00a9"] == {
        "true": 1,
        "false": 1,
        "missing": 0,
        "invalid": 0,
        "unavailable": 1,
    }
    assert summary.aka_flag_by_rule_code["00e1"]["true"] == 1
    assert summary.target_games == 1
    assert summary.files_with_errors == 2
    assert summary.error_counts == {
        "invalid_filename": 1,
        "json_decode_error": 1,
    }
    assert sum(summary.filename_year_counts.values()) == 5
    assert sum(summary.compression_counts.values()) == 5
    assert sum(summary.first_event_status_counts.values()) == 5
    assert sum(summary.aka_flag_counts.values()) == 5


def test_validate_raw_year_limits_error_samples_deterministically(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "2020"
    for index in reversed(range(12)):
        _write_events(raw / f"invalid-{index:02}.mjson", [_start_game()])

    summary = validate_raw_year_directory(2020, raw)

    samples = summary.error_samples["invalid_filename"]
    assert summary.error_counts["invalid_filename"] == 12
    assert len(samples) == 10
    assert [sample.relative_path for sample in samples] == [
        f"2020/invalid-{index:02}.mjson" for index in range(10)
    ]


def test_validate_raw_year_records_unexpected_direct_entries(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "2020"
    _write_events(raw / _filename(), [_start_game()])
    (raw / "README.txt").write_text("metadata", encoding="utf-8")
    (raw / "nested").mkdir()
    _write_events(raw / "nested" / _filename(), [_start_game()])

    summary = validate_raw_year_directory(2020, raw)

    assert summary.total_mjson_files == 1
    assert summary.unexpected_raw_entries == {"files": 1, "subdirectories": 1}
    assert summary.unexpected_raw_entry_samples == {
        "files": ("2020/README.txt",),
        "subdirectories": ("2020/nested",),
    }
    assert summary.error_counts["unexpected_raw_file"] == 1
    assert summary.error_counts["unexpected_raw_subdirectory"] == 1


def test_validate_release_archive_observes_size_hash_and_entries(
    tmp_path: Path,
) -> None:
    path = _write_zip(
        tmp_path / "2020.zip",
        mjson_entries=2,
        include_non_mjson=True,
    )
    asset = _asset_for(path)

    result = validate_release_archive(asset, path)

    assert result.observed_size_bytes == asset.expected_size_bytes
    assert result.size_matches is True
    assert result.observed_sha256 == asset.expected_sha256
    assert result.sha256_matches is True
    assert result.observed_mjson_entries == 2
    assert result.unexpected_archive_entries == 1
    assert result.mjson_filenames == frozenset({"game-0.mjson", "game-1.mjson"})
    assert result.issues == ()


def test_validate_release_archive_records_size_mismatch(tmp_path: Path) -> None:
    path = _write_zip(tmp_path / "2020.zip")
    actual = _asset_for(path)
    asset = ReleaseAsset(
        year=actual.year,
        filename=actual.filename,
        expected_size_bytes=actual.expected_size_bytes + 1,
        expected_sha256=actual.expected_sha256,
    )

    result = validate_release_archive(asset, path)

    assert result.size_matches is False
    assert result.sha256_matches is True
    assert [issue.category for issue in result.issues] == ["archive_size_mismatch"]


def test_validate_release_archive_records_hash_mismatch(tmp_path: Path) -> None:
    path = _write_zip(tmp_path / "2020.zip")
    actual = _asset_for(path)
    asset = ReleaseAsset(
        year=actual.year,
        filename=actual.filename,
        expected_size_bytes=actual.expected_size_bytes,
        expected_sha256="0" * 64,
    )

    result = validate_release_archive(asset, path)

    assert result.size_matches is True
    assert result.sha256_matches is False
    assert [issue.category for issue in result.issues] == ["archive_sha256_mismatch"]


def test_build_year_summary_detects_raw_archive_count_mismatch(
    tmp_path: Path,
) -> None:
    archive_path = _write_zip(tmp_path / "2020.zip", mjson_entries=1)
    archive = validate_release_archive(_asset_for(archive_path), archive_path)
    raw_directory = tmp_path / "raw" / "2020"
    _write_events(raw_directory / _filename(game_id="00000001"), [_start_game()])
    _write_events(raw_directory / _filename(game_id="00000002"), [_start_game()])
    raw = validate_raw_year_directory(2020, raw_directory)

    summary = build_year_validation_summary(2020, archive, raw)

    assert summary.validation.failed is True
    assert summary.validation.error_counts == {
        "raw_archive_count_mismatch": 1,
        "raw_archive_filename_mismatch": 1,
    }
    assert summary.mjson_filename_set.matches is False
    assert summary.mjson_filename_set.archive_only_count == 1
    assert summary.mjson_filename_set.raw_only_count == 2


def test_build_year_summary_accepts_identical_filename_sets(
    tmp_path: Path,
) -> None:
    filenames = [
        _filename(game_id="00000001"),
        _filename(game_id="00000002"),
    ]
    archive_path = _write_zip(
        tmp_path / "2020.zip",
        mjson_names=list(reversed(filenames)),
    )
    asset = _asset_for(archive_path)
    archive = validate_release_archive(asset, archive_path)
    raw_directory = tmp_path / "raw" / "2020"
    for filename in filenames:
        _write_events(raw_directory / filename, [_start_game()])
    raw = validate_raw_year_directory(2020, raw_directory)

    year_summary = build_year_validation_summary(2020, archive, raw)
    manifest = ReleaseAssetManifest("example/repository", "v1", (asset,))
    dataset_summary = build_dataset_validation_summary(manifest, [year_summary])
    serialized_data = dataset_validation_summary_to_dict(dataset_summary)
    serialized = json.dumps(serialized_data)

    assert year_summary.validation.failed is False
    assert year_summary.mjson_filename_set.matches is True
    assert year_summary.mjson_filename_set.archive_only_count == 0
    assert year_summary.mjson_filename_set.raw_only_count == 0
    assert year_summary.mjson_filename_set.archive_only_samples == ()
    assert year_summary.mjson_filename_set.raw_only_samples == ()
    assert year_summary.archive.mjson_filenames is None
    assert year_summary.raw.mjson_filenames is None
    assert serialized_data["years"][0]["mjson_filename_set"] == {
        "matches": True,
        "archive_only_count": 0,
        "raw_only_count": 0,
        "archive_only_samples": [],
        "raw_only_samples": [],
    }
    assert "mjson_filenames" not in serialized
    assert all(filename not in serialized for filename in filenames)


def test_build_year_summary_detects_equal_count_different_filename_sets(
    tmp_path: Path,
) -> None:
    shared = _filename(game_id="00000001")
    archive_only = _filename(game_id="00000002")
    raw_only = _filename(game_id="00000003")
    archive_path = _write_zip(
        tmp_path / "2020.zip",
        mjson_names=[shared, archive_only],
    )
    archive = validate_release_archive(_asset_for(archive_path), archive_path)
    raw_directory = tmp_path / "raw" / "2020"
    for filename in (shared, raw_only):
        _write_events(raw_directory / filename, [_start_game()])
    raw = validate_raw_year_directory(2020, raw_directory)

    summary = build_year_validation_summary(2020, archive, raw)

    assert summary.validation.error_counts == {"raw_archive_filename_mismatch": 1}
    assert summary.mjson_filename_set.matches is False
    assert summary.mjson_filename_set.archive_only_count == 1
    assert summary.mjson_filename_set.raw_only_count == 1
    assert summary.mjson_filename_set.archive_only_samples == (archive_only,)
    assert summary.mjson_filename_set.raw_only_samples == (raw_only,)


def test_filename_set_samples_are_sorted_and_limited_to_ten(
    tmp_path: Path,
) -> None:
    archive_names = [
        _filename(game_id=f"{index:08x}") for index in reversed(range(100, 112))
    ]
    raw_names = [
        _filename(game_id=f"{index:08x}") for index in reversed(range(200, 212))
    ]
    archive_path = _write_zip(
        tmp_path / "2020.zip",
        mjson_names=archive_names,
    )
    archive = validate_release_archive(_asset_for(archive_path), archive_path)
    raw_directory = tmp_path / "raw" / "2020"
    for filename in raw_names:
        _write_events(raw_directory / filename, [_start_game()])
    raw = validate_raw_year_directory(2020, raw_directory)

    summary = build_year_validation_summary(
        2020,
        archive,
        raw,
        error_sample_limit=100,
    )

    assert summary.mjson_filename_set.archive_only_count == 12
    assert summary.mjson_filename_set.raw_only_count == 12
    assert summary.mjson_filename_set.archive_only_samples == tuple(
        sorted(archive_names)[:10]
    )
    assert summary.mjson_filename_set.raw_only_samples == tuple(sorted(raw_names)[:10])


def test_filename_set_comparison_is_unavailable_for_unreadable_archive(
    tmp_path: Path,
) -> None:
    asset = ReleaseAsset(2020, "missing.zip", 0, "0" * 64)
    archive = validate_release_archive(asset, tmp_path / asset.filename)
    raw_directory = tmp_path / "raw" / "2020"
    _write_events(raw_directory / _filename(), [_start_game()])
    raw = validate_raw_year_directory(2020, raw_directory)

    summary = build_year_validation_summary(2020, archive, raw)

    assert summary.mjson_filename_set.matches is None
    assert summary.mjson_filename_set.archive_only_count is None
    assert summary.mjson_filename_set.raw_only_count is None
    assert summary.mjson_filename_set.archive_only_samples == ()
    assert summary.mjson_filename_set.raw_only_samples == ()
    assert "raw_archive_filename_mismatch" not in summary.validation.error_counts


def test_asset_manifest_contains_only_release_expected_fields() -> None:
    data = json.loads(ASSET_MANIFEST_PATH.read_text(encoding="utf-8"))

    assert data["repository"] == "NikkeTryHard/tenhou-to-mjai"
    assert data["release_tag"] == "v2.0.0"
    assert len(data["assets"]) == 17
    assert all(
        set(asset)
        == {
            "year",
            "filename",
            "expected_size_bytes",
            "expected_sha256",
        }
        for asset in data["assets"]
    )
    assert all("mjson" not in key for asset in data["assets"] for key in asset)
    assert "total" not in data
    assert load_release_asset_manifest(ASSET_MANIFEST_PATH).assets[0].year == 2009


def test_dataset_summary_totals_are_calculated_from_observations(
    tmp_path: Path,
) -> None:
    archive_path = _write_zip(
        tmp_path / "2020.zip",
        mjson_names=[_filename()],
    )
    asset = _asset_for(archive_path)
    archive = validate_release_archive(asset, archive_path)
    raw_directory = tmp_path / "raw" / "2020"
    _write_events(raw_directory / _filename(), [_start_game()])
    raw = validate_raw_year_directory(2020, raw_directory)
    year_summary = build_year_validation_summary(2020, archive, raw)
    manifest = ReleaseAssetManifest("example/repository", "v1", (asset,))

    summary = build_dataset_validation_summary(manifest, [year_summary])

    assert summary.totals["total_mjson_files"] == 1
    assert summary.totals["archive_mjson_entries"] == 1
    assert summary.totals["target_games"] == 1
    assert summary.validation_failed is False
    assert summary.archive_hashes_verified is True


def test_summary_outputs_are_deterministic_and_contain_no_absolute_path(
    tmp_path: Path,
) -> None:
    archive_path = _write_zip(tmp_path / "2020.zip", mjson_entries=0)
    asset = _asset_for(archive_path)
    archive = validate_release_archive(asset, archive_path)
    raw = validate_raw_year_directory(2020, tmp_path / "missing-raw")
    year_summary = build_year_validation_summary(2020, archive, raw)
    manifest = ReleaseAssetManifest("example/repository", "v1", (asset,))
    summary = build_dataset_validation_summary(manifest, [year_summary])
    json_path = tmp_path / "out" / "summary.json"
    markdown_path = tmp_path / "out" / "summary.md"

    write_dataset_validation_outputs(summary, json_path, markdown_path)
    first_json = json_path.read_text(encoding="utf-8")
    first_markdown = markdown_path.read_text(encoding="utf-8")
    write_dataset_validation_outputs(summary, json_path, markdown_path)

    assert json_path.read_text(encoding="utf-8") == first_json
    assert markdown_path.read_text(encoding="utf-8") == first_markdown
    assert first_json.endswith("\n")
    assert first_markdown == render_dataset_validation_markdown(summary)
    assert str(tmp_path) not in first_json
    assert "generated_at" not in first_json
    assert dataset_validation_summary_to_dict(summary)["validation_failed"]


def test_parse_args_accepts_selected_years_and_rejects_out_of_range() -> None:
    assert parse_args(["--years", "2009", "2025"]).years == [2009, 2025]

    with pytest.raises(SystemExit):
        parse_args(["--years", "2008"])


def test_cli_writes_failure_summary_before_returning_nonzero(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archives"
    archive_path = _write_zip(archive_root / "2009.zip", mjson_entries=1)
    asset = _asset_for(archive_path, year=2009)
    manifest_path = tmp_path / "assets.json"
    manifest_path.write_text(
        json.dumps(
            {
                "repository": "example/repository",
                "release_tag": "v1",
                "assets": [
                    {
                        "year": asset.year,
                        "filename": asset.filename,
                        "expected_size_bytes": asset.expected_size_bytes,
                        "expected_sha256": asset.expected_sha256,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    raw_root = tmp_path / "raw"
    (raw_root / "2009").mkdir(parents=True)
    output_json = tmp_path / "summary.json"
    output_markdown = tmp_path / "summary.md"

    exit_code = main(
        [
            "--years",
            "2009",
            "--archive-root",
            str(archive_root),
            "--raw-root",
            str(raw_root),
            "--asset-manifest",
            str(manifest_path),
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_markdown),
            "--skip-archive-hash",
        ]
    )

    assert exit_code == 1
    assert output_json.exists()
    assert output_markdown.exists()
    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data["validation_failed"] is True
    assert data["years"][0]["validation"]["error_counts"] == {
        "raw_archive_count_mismatch": 1,
        "raw_archive_filename_mismatch": 1,
    }


def test_declared_distribution_values_are_complete() -> None:
    assert set(COMPRESSION_VALUES) == {"gzip", "plain", "unknown"}
    assert set(FILENAME_YEAR_STATUS_VALUES) == {
        "match",
        "mismatch",
        "unavailable",
    }
    assert set(FIRST_EVENT_STATUS_VALUES) == {
        "readable_start_game",
        "empty",
        "json_decode_error",
        "not_object",
        "not_start_game",
        "unreadable",
    }
    assert set(AKA_FLAG_STATUS_VALUES) == {
        "true",
        "false",
        "missing",
        "invalid",
        "unavailable",
    }
