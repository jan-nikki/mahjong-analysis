import gzip
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from mahjong_analysis.hand_waits import FixedMeld
from mahjong_analysis.riichi import ActorDiscard, EstablishedRiichi
from mahjong_analysis.riichi_wait_dataset import (
    DATASET_NAME,
    DatasetActorDiscard,
    DatasetFixedMeld,
    DatasetWaitDetail,
    build_dataset_record,
    dataset_record_from_dict,
    dataset_record_to_dict,
    iter_dataset_records,
    load_manifest,
    serialize_dataset_record,
    validate_manifest,
)
from mahjong_analysis.riichi_wait_records import build_riichi_wait_record


def sample_production_record():
    melds = (
        FixedMeld(("1m", "1m", "1m", "1m")),
        FixedMeld(("9m", "9m", "9m", "9m")),
    )
    discard = ActorDiscard(
        discard_number=1,
        tile="5mr",
        tile_kind="5m",
        tsumogiri=True,
        event_index=2,
        is_riichi_declaration=True,
        was_called=True,
        call_type="pon",
        called_by_actor=1,
        call_event_index=4,
    )
    established = EstablishedRiichi(
        actor=0,
        reach_event_index=1,
        declaration_dahai_event_index=2,
        reach_accepted_event_index=3,
        riichi_discard_number=1,
        riichi_declaration_tile="5mr",
        riichi_declaration_tile_kind="5m",
        concealed_tiles_after_discard=("1p", "2p", "3p", "E", "E", "4s", "5s"),
        fixed_melds=melds,
        actor_discards_before_riichi=(discard,),
    )
    return build_riichi_wait_record(established)


def sample_dataset_record():
    return build_dataset_record(
        sample_production_record(),
        year=2025,
        relative_source_path="2025/sub/game.mjson",
        start_kyoku_line=10,
        reach_line=11,
        declaration_dahai_line=12,
        reach_accepted_line=13,
        bakaze="E",
        kyoku=1,
        honba=0,
        oya=0,
        scores_at_start=(25000, 25000, 25000, 25000),
        dora_marker="5pr",
    )


def completed_manifest(*, mode: str = "sample", schema_version: int = 1):
    max_files = 1 if mode == "sample" else None
    year_entry = {
        "year": 2025,
        "scanned_files": 1,
        "target_games": 1,
        "east_kyokus": 1,
        "established_riichis": 1,
        "output_records": 1,
        "output_filename": "2025.jsonl.gz",
        "compressed_size_bytes": 1,
        "sha256": "0" * 64,
        "elapsed_seconds": 1.0,
    }
    return {
        "schema_version": schema_version,
        "dataset_name": DATASET_NAME,
        "created_at_utc": "2026-01-02T03:04:05Z",
        "source": {
            "repository": "example/source",
            "release_tag": "v2.0.0",
            "validation_summary_path": "data/validation/summary.json",
            "validation_summary_sha256": "1" * 64,
            "archive_hashes_verified": True,
        },
        "scope": {
            "years": [2025],
            "rule_code": "00a9",
            "aka_flag": True,
            "bakaze": "E",
            "input_selection": {
                "ordering": "raw-root-relative POSIX path lexicographic",
                "max_files_before_target_filtering": max_files,
            },
            "extraction_mode": mode,
        },
        "serialization": {
            "format": "JSON Lines",
            "encoding": "UTF-8",
            "json_options": {
                "ensure_ascii": False,
                "sort_keys": True,
                "separators": [",", ":"],
                "allow_nan": False,
                "line_terminator": "LF",
            },
            "compression": "gzip",
            "compression_level": 6,
            "gzip_mtime": 0,
            "gzip_header_filename": "",
        },
        "generator": {
            "git_commit": "abc123",
            "worktree_clean": True,
            "python_version": "3.12.10",
            "zlib_version": "1.3.1",
            "invocation": {
                "years": [2025],
                "mode": mode,
                "max_files": max_files,
            },
        },
        "years": [year_entry],
        "totals": {
            "scanned_files": 1,
            "target_games": 1,
            "east_kyokus": 1,
            "established_riichis": 1,
            "output_records": 1,
        },
    }


def test_schema_v1_round_trip_is_lossless() -> None:
    record = sample_dataset_record()

    value = dataset_record_to_dict(record)
    restored = dataset_record_from_dict(value)

    assert restored == record
    assert value["riichi"]["riichi_declaration_tile"] == "5mr"
    assert value["riichi"]["riichi_declaration_tile_kind"] == "5m"
    assert value["kyoku"]["dora_marker"] == "5pr"
    assert value["hand"]["fixed_melds"][0]["tiles"] == ["1m"] * 4
    assert value["actor_discards_before_riichi"] == [
        {
            "call_event_index": 4,
            "call_type": "pon",
            "called_by_actor": 1,
            "discard_number": 1,
            "event_index": 2,
            "is_riichi_declaration": True,
            "normalized_tile": "5m",
            "tile": "5mr",
            "tsumogiri": True,
            "was_called": True,
        }
    ]


def test_production_record_maps_every_field_without_reinterpretation() -> None:
    production = sample_production_record()
    dataset = sample_dataset_record()

    assert dataset.actor == production.actor
    assert dataset.riichi_discard_number == production.riichi_discard_number
    assert dataset.riichi_declaration_tile == production.riichi_declaration_tile
    assert (
        dataset.riichi_declaration_tile_kind == production.riichi_declaration_tile_kind
    )
    assert dataset.reach_event_index == production.reach_event_index
    assert dataset.declaration_dahai_event_index == (
        production.declaration_dahai_event_index
    )
    assert dataset.reach_accepted_event_index == production.reach_accepted_event_index
    assert dataset.concealed_tiles_after_discard == (
        production.concealed_tiles_after_discard
    )
    assert tuple((meld.meld_type, meld.tiles) for meld in dataset.fixed_melds) == tuple(
        (meld.meld_type, meld.tiles) for meld in production.fixed_melds
    )
    assert tuple(
        (
            discard.discard_number,
            discard.tile,
            discard.normalized_tile,
            discard.tsumogiri,
            discard.event_index,
            discard.is_riichi_declaration,
            discard.was_called,
            discard.call_type,
            discard.called_by_actor,
            discard.call_event_index,
        )
        for discard in dataset.actor_discards_before_riichi
    ) == tuple(
        (
            discard.discard_number,
            discard.tile,
            discard.tile_kind,
            discard.tsumogiri,
            discard.event_index,
            discard.is_riichi_declaration,
            discard.was_called,
            discard.call_type,
            discard.called_by_actor,
            discard.call_event_index,
        )
        for discard in production.actor_discards_before_riichi
    )
    assert dataset.wait_tiles == production.wait_tiles
    assert dataset.wait_tile_count == production.wait_tile_count
    assert tuple(
        (detail.tile, detail.hand_type, detail.wait_shape)
        for detail in dataset.wait_details
    ) == tuple(
        (detail.wait_tile, detail.hand_type, detail.wait_shape)
        for detail in production.wait_details
    )
    assert dataset.wait_shapes == production.wait_shapes
    assert dataset.contains_ryanmen is production.contains_ryanmen
    assert dataset.is_pure_ryanmen is production.is_pure_ryanmen
    assert dataset.is_multiwait is production.is_multiwait


def test_json_serialization_uses_fixed_compact_utf8_settings() -> None:
    record = sample_dataset_record()

    payload = serialize_dataset_record(record)

    assert payload == serialize_dataset_record(record)
    assert b"\n" not in payload
    assert b" " not in payload
    assert "5mr" in payload.decode("utf-8")
    assert payload.decode("utf-8") == json.dumps(
        dataset_record_to_dict(record),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def test_serializer_preserves_multiple_interpretations_for_one_tile() -> None:
    record = sample_dataset_record()
    altered = replace(
        record,
        wait_details=(
            DatasetWaitDetail("3s", "standard", "ryanmen"),
            DatasetWaitDetail("3s", "standard", "shanpon"),
            DatasetWaitDetail("6s", "standard", "ryanmen"),
        ),
        wait_shapes=("ryanmen", "shanpon"),
        is_pure_ryanmen=False,
    )

    value = dataset_record_to_dict(altered)

    assert value["waits"]["wait_details"] == [
        {"hand_type": "standard", "tile": "3s", "wait_shape": "ryanmen"},
        {"hand_type": "standard", "tile": "3s", "wait_shape": "shanpon"},
        {"hand_type": "standard", "tile": "6s", "wait_shape": "ryanmen"},
    ]


def test_serializer_preserves_concealed_and_river_order() -> None:
    record = sample_dataset_record()
    concealed = tuple(reversed(record.concealed_tiles_after_discard))
    altered = replace(record, concealed_tiles_after_discard=concealed)

    value = dataset_record_to_dict(altered)

    assert value["hand"]["concealed_tiles_after_discard"] == list(concealed)
    assert value["actor_discards_before_riichi"][0]["event_index"] == 2


def test_serializer_preserves_raw_red_tile_inside_fixed_meld() -> None:
    record = sample_dataset_record()
    red_ankan = DatasetFixedMeld("ankan", ("5m", "5m", "5m", "5mr"))
    altered = replace(record, fixed_melds=(red_ankan, *record.fixed_melds[1:]))

    value = dataset_record_to_dict(altered)

    assert value["hand"]["fixed_melds"][0] == {
        "meld_type": "ankan",
        "tiles": ["5m", "5m", "5m", "5mr"],
    }


def test_jsonl_reader_requires_lf_and_round_trips(tmp_path: Path) -> None:
    record = sample_dataset_record()
    path = tmp_path / "2025.jsonl.gz"
    with gzip.open(path, "wb") as file:
        file.write(serialize_dataset_record(record) + b"\n")

    assert tuple(iter_dataset_records(path)) == (record,)

    with gzip.open(path, "wb") as file:
        file.write(serialize_dataset_record(record))
    with pytest.raises(ValueError, match="missing LF"):
        tuple(iter_dataset_records(path))


@pytest.mark.parametrize("version", [0, 2, "1", True, False])
def test_manifest_rejects_unsupported_schema(version: object) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        validate_manifest(completed_manifest(schema_version=version))


def test_manifest_reader_rejects_sample_as_full(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(completed_manifest()), encoding="utf-8")

    with pytest.raises(ValueError, match="mode mismatch"):
        load_manifest(path, expected_mode="full")
    assert load_manifest(path, expected_mode="sample")["dataset_name"] == DATASET_NAME


def test_dataset_record_rejects_absolute_source_path() -> None:
    record = sample_dataset_record()

    with pytest.raises(ValueError, match="relative_source_path"):
        replace(record, relative_source_path="C:/private/game.mjson")


def test_two_entry_river_requires_sequence_order_and_final_declaration() -> None:
    record = sample_dataset_record()
    first = DatasetActorDiscard(1, "9s", "9s", False, 1, False, False, None, None, None)
    declaration = DatasetActorDiscard(
        2,
        "5mr",
        "5m",
        True,
        4,
        True,
        False,
        None,
        None,
        None,
    )
    valid = replace(
        record,
        reach_event_index=3,
        declaration_dahai_event_index=4,
        reach_accepted_event_index=5,
        reach_line=13,
        declaration_dahai_line=14,
        reach_accepted_line=15,
        riichi_discard_number=2,
        actor_discards_before_riichi=(first, declaration),
    )
    assert [
        item["discard_number"]
        for item in dataset_record_to_dict(valid)["actor_discards_before_riichi"]
    ] == [1, 2]

    with pytest.raises(ValueError, match="consecutive"):
        replace(
            valid,
            actor_discards_before_riichi=(
                replace(first, discard_number=2),
                declaration,
            ),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(
            valid,
            actor_discards_before_riichi=(replace(first, event_index=5), declaration),
        )
    with pytest.raises(ValueError, match="only the final"):
        replace(
            valid,
            actor_discards_before_riichi=(
                replace(first, is_riichi_declaration=True),
                declaration,
            ),
        )


def test_completed_manifest_rejects_invalid_source_and_serialization() -> None:
    source = completed_manifest()
    source["source"]["archive_hashes_verified"] = 1
    with pytest.raises(ValueError, match="SHA256-verified"):
        validate_manifest(source)

    serialization = completed_manifest()
    serialization["serialization"]["gzip_mtime"] = True
    with pytest.raises(TypeError, match="must be an integer"):
        validate_manifest(serialization)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rule_code", "rekt", "rule_code"),
        ("aka_flag", 1, "aka_flag"),
        ("bakaze", "S", "bakaze"),
    ],
)
def test_completed_manifest_rejects_invalid_scope(field, value, message) -> None:
    manifest = completed_manifest()
    manifest["scope"][field] = value
    with pytest.raises(ValueError, match=message):
        validate_manifest(manifest)


def test_completed_manifest_rejects_missing_required_field() -> None:
    manifest = completed_manifest()
    del manifest["source"]
    with pytest.raises(ValueError, match="missing required"):
        validate_manifest(manifest)


def test_completed_manifest_rejects_duplicate_years_and_bad_totals() -> None:
    duplicate = completed_manifest()
    duplicate["years"].append(deepcopy(duplicate["years"][0]))
    with pytest.raises(ValueError, match="duplicates"):
        validate_manifest(duplicate)

    mismatched = completed_manifest()
    mismatched["totals"]["output_records"] = 2
    with pytest.raises(ValueError, match="does not match"):
        validate_manifest(mismatched)


def test_completed_manifest_rejects_bad_year_counts_and_checkpoint_marker() -> None:
    mismatched = completed_manifest()
    mismatched["years"][0]["established_riichis"] = 2
    with pytest.raises(ValueError, match="must equal"):
        validate_manifest(mismatched)

    wrong_type = completed_manifest()
    wrong_type["years"][0]["scanned_files"] = True
    with pytest.raises(TypeError, match="integer"):
        validate_manifest(wrong_type)

    checkpoint = completed_manifest()
    checkpoint["checkpoint"] = True
    with pytest.raises(ValueError, match="must not contain checkpoint"):
        validate_manifest(checkpoint)
