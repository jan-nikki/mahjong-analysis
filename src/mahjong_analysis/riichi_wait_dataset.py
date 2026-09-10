"""Lossless, versioned DTOs for the established-riichi wait dataset."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from mahjong_analysis.hand_waits import FixedMeld, HandWaits, WaitDetail
from mahjong_analysis.riichi import ActorDiscard
from mahjong_analysis.riichi_wait_records import RiichiWaitRecord
from mahjong_analysis.tiles import normalize_tile

SCHEMA_VERSION = 1
DATASET_NAME = "riichi-waits-v1"
SUPPORTED_SCHEMA_VERSIONS = frozenset((SCHEMA_VERSION,))
SUPPORTED_YEARS = tuple(range(2009, 2026))
RULE_CODE = "00a9"
SOURCE_RELEASE_TAG = "v2.0.0"
COMPRESSION_LEVEL = 6
INPUT_ORDERING = "raw-root-relative POSIX path lexicographic"
_COUNT_FIELDS = (
    "scanned_files",
    "target_games",
    "east_kyokus",
    "established_riichis",
    "output_records",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class DatasetFixedMeld:
    """One fixed meld without reordering or normalizing its physical tiles."""

    meld_type: str
    tiles: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.meld_type, str):
            raise TypeError("meld_type must be a string")
        if not isinstance(self.tiles, tuple):
            raise TypeError("meld tiles must be a tuple")
        FixedMeld(tiles=self.tiles, meld_type=self.meld_type)  # type: ignore[arg-type]


@dataclass(frozen=True)
class DatasetActorDiscard:
    """One ordered river entry, including post-declaration call metadata."""

    discard_number: int
    tile: str
    normalized_tile: str
    tsumogiri: bool
    event_index: int
    is_riichi_declaration: bool
    was_called: bool
    call_type: str | None
    called_by_actor: int | None
    call_event_index: int | None

    def __post_init__(self) -> None:
        ActorDiscard(
            discard_number=self.discard_number,
            tile=self.tile,
            tile_kind=self.normalized_tile,
            tsumogiri=self.tsumogiri,
            event_index=self.event_index,
            is_riichi_declaration=self.is_riichi_declaration,
            was_called=self.was_called,
            call_type=self.call_type,  # type: ignore[arg-type]
            called_by_actor=self.called_by_actor,
            call_event_index=self.call_event_index,
        )


@dataclass(frozen=True)
class DatasetWaitDetail:
    """One canonical tile/hand-type/wait-shape relation."""

    tile: str
    hand_type: str
    wait_shape: str

    def __post_init__(self) -> None:
        WaitDetail(
            wait_tile=self.tile,
            hand_type=self.hand_type,  # type: ignore[arg-type]
            wait_shape=self.wait_shape,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class RiichiWaitDatasetRecord:
    """One source-locatable, lossless established-riichi record."""

    year: int
    relative_source_path: str
    start_kyoku_line: int
    reach_line: int
    declaration_dahai_line: int
    reach_accepted_line: int
    bakaze: str
    kyoku: int
    honba: int
    oya: int
    scores_at_start: tuple[int, int, int, int]
    dora_marker: str
    actor: int
    riichi_discard_number: int
    riichi_declaration_tile: str
    riichi_declaration_tile_kind: str
    reach_event_index: int
    declaration_dahai_event_index: int
    reach_accepted_event_index: int
    concealed_tiles_after_discard: tuple[str, ...]
    fixed_melds: tuple[DatasetFixedMeld, ...]
    actor_discards_before_riichi: tuple[DatasetActorDiscard, ...]
    wait_tiles: tuple[str, ...]
    wait_tile_count: int
    wait_details: tuple[DatasetWaitDetail, ...]
    wait_shapes: tuple[str, ...]
    contains_ryanmen: bool
    is_pure_ryanmen: bool
    is_multiwait: bool

    def __post_init__(self) -> None:
        if type(self.year) is not int or self.year < 1:
            raise ValueError("year must be a positive integer")
        if not self.relative_source_path or "\\" in self.relative_source_path:
            raise ValueError("relative_source_path must be non-empty and POSIX-style")
        if Path(self.relative_source_path).is_absolute():
            raise ValueError("relative_source_path must be relative")
        source_parts = self.relative_source_path.split("/")
        if ".." in source_parts or source_parts[0] != str(self.year):
            raise ValueError("relative_source_path must be rooted at its dataset year")
        line_values = (
            self.start_kyoku_line,
            self.reach_line,
            self.declaration_dahai_line,
            self.reach_accepted_line,
        )
        if any(type(value) is not int or value < 1 for value in line_values):
            raise ValueError("physical line numbers must be positive integers")
        if not (
            self.start_kyoku_line
            <= self.reach_line
            < self.declaration_dahai_line
            < self.reach_accepted_line
        ):
            raise ValueError("physical source lines are not in event order")
        event_indexes = (
            self.reach_event_index,
            self.declaration_dahai_event_index,
            self.reach_accepted_event_index,
        )
        if any(type(value) is not int or value < 0 for value in event_indexes):
            raise ValueError("event indexes must be non-negative integers")
        if not (
            self.reach_event_index
            < self.declaration_dahai_event_index
            < self.reach_accepted_event_index
        ):
            raise ValueError("riichi event indexes are not in event order")
        if (
            self.reach_line != self.start_kyoku_line + self.reach_event_index
            or self.declaration_dahai_line
            != self.start_kyoku_line + self.declaration_dahai_event_index
            or self.reach_accepted_line
            != self.start_kyoku_line + self.reach_accepted_event_index
        ):
            raise ValueError("physical lines must correspond to kyoku event indexes")
        if self.bakaze != "E":
            raise ValueError("dataset records must be from the east round")
        if type(self.kyoku) is not int or self.kyoku < 1:
            raise ValueError("kyoku must be a positive integer")
        if type(self.honba) is not int or self.honba < 0:
            raise ValueError("honba must be a non-negative integer")
        if type(self.oya) is not int or self.oya not in range(4):
            raise ValueError("oya must be an actor from 0 through 3")
        if (
            not isinstance(self.scores_at_start, tuple)
            or len(self.scores_at_start) != 4
            or any(type(score) is not int for score in self.scores_at_start)
        ):
            raise TypeError("scores_at_start must be a four-integer tuple")
        normalize_tile(self.dora_marker)
        if type(self.actor) is not int or self.actor not in range(4):
            raise ValueError("actor must be an integer from 0 through 3")
        if (
            type(self.riichi_discard_number) is not int
            or self.riichi_discard_number < 1
        ):
            raise ValueError("riichi_discard_number must be a positive integer")
        if normalize_tile(self.riichi_declaration_tile) != (
            self.riichi_declaration_tile_kind
        ):
            raise ValueError("riichi declaration raw and normalized tiles disagree")
        for name in (
            "concealed_tiles_after_discard",
            "fixed_melds",
            "actor_discards_before_riichi",
            "wait_tiles",
            "wait_details",
            "wait_shapes",
        ):
            if not isinstance(getattr(self, name), tuple):
                raise TypeError(f"{name} must be a tuple")
        if not all(isinstance(value, DatasetFixedMeld) for value in self.fixed_melds):
            raise TypeError("fixed_melds must contain DatasetFixedMeld")
        if not all(
            isinstance(value, DatasetActorDiscard)
            for value in self.actor_discards_before_riichi
        ):
            raise TypeError("river must contain DatasetActorDiscard")
        if not all(isinstance(value, DatasetWaitDetail) for value in self.wait_details):
            raise TypeError("wait_details must contain DatasetWaitDetail")
        for tile in self.concealed_tiles_after_discard:
            normalize_tile(tile)
        waits = HandWaits(
            wait_tiles=self.wait_tiles,
            wait_details=tuple(
                WaitDetail(
                    detail.tile,
                    detail.hand_type,  # type: ignore[arg-type]
                    detail.wait_shape,  # type: ignore[arg-type]
                )
                for detail in self.wait_details
            ),
        )
        derived_values = (
            ("wait_tile_count", self.wait_tile_count, waits.wait_tile_count),
            ("wait_shapes", self.wait_shapes, waits.wait_shapes),
            ("contains_ryanmen", self.contains_ryanmen, waits.contains_ryanmen),
            ("is_pure_ryanmen", self.is_pure_ryanmen, waits.is_pure_ryanmen),
            ("is_multiwait", self.is_multiwait, waits.is_multiwait),
        )
        if type(self.wait_tile_count) is not int:
            raise TypeError("wait_tile_count must be an integer")
        for name in ("contains_ryanmen", "is_pure_ryanmen", "is_multiwait"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        for name, actual, expected in derived_values:
            if actual != expected:
                raise ValueError(f"{name} must be derived from canonical waits")
        if len(self.actor_discards_before_riichi) != self.riichi_discard_number:
            raise ValueError("river length must equal riichi_discard_number")
        discard_numbers = tuple(
            discard.discard_number for discard in self.actor_discards_before_riichi
        )
        if discard_numbers != tuple(range(1, self.riichi_discard_number + 1)):
            raise ValueError("river discard numbers must be consecutive from 1")
        discard_event_indexes = tuple(
            discard.event_index for discard in self.actor_discards_before_riichi
        )
        if any(left >= right for left, right in pairwise(discard_event_indexes)):
            raise ValueError("river event indexes must be strictly increasing")
        if any(
            discard.is_riichi_declaration
            for discard in self.actor_discards_before_riichi[:-1]
        ):
            raise ValueError("only the final river entry may declare riichi")
        declaration = self.actor_discards_before_riichi[-1]
        if (
            not declaration.is_riichi_declaration
            or declaration.event_index != self.declaration_dahai_event_index
            or declaration.tile != self.riichi_declaration_tile
            or declaration.normalized_tile != self.riichi_declaration_tile_kind
        ):
            raise ValueError("final river entry must be the declaration discard")


def build_dataset_record(
    record: RiichiWaitRecord,
    *,
    year: int,
    relative_source_path: str,
    start_kyoku_line: int,
    reach_line: int,
    declaration_dahai_line: int,
    reach_accepted_line: int,
    bakaze: str,
    kyoku: int,
    honba: int,
    oya: int,
    scores_at_start: tuple[int, int, int, int],
    dora_marker: str,
) -> RiichiWaitDatasetRecord:
    """Map one production record losslessly without recomputing mahjong facts."""
    if not isinstance(record, RiichiWaitRecord):
        raise TypeError("record must be a RiichiWaitRecord")
    return RiichiWaitDatasetRecord(
        year=year,
        relative_source_path=relative_source_path,
        start_kyoku_line=start_kyoku_line,
        reach_line=reach_line,
        declaration_dahai_line=declaration_dahai_line,
        reach_accepted_line=reach_accepted_line,
        bakaze=bakaze,
        kyoku=kyoku,
        honba=honba,
        oya=oya,
        scores_at_start=scores_at_start,
        dora_marker=dora_marker,
        actor=record.actor,
        riichi_discard_number=record.riichi_discard_number,
        riichi_declaration_tile=record.riichi_declaration_tile,
        riichi_declaration_tile_kind=record.riichi_declaration_tile_kind,
        reach_event_index=record.reach_event_index,
        declaration_dahai_event_index=record.declaration_dahai_event_index,
        reach_accepted_event_index=record.reach_accepted_event_index,
        concealed_tiles_after_discard=record.concealed_tiles_after_discard,
        fixed_melds=tuple(
            DatasetFixedMeld(meld.meld_type, meld.tiles) for meld in record.fixed_melds
        ),
        actor_discards_before_riichi=tuple(
            DatasetActorDiscard(
                discard_number=discard.discard_number,
                tile=discard.tile,
                normalized_tile=discard.tile_kind,
                tsumogiri=discard.tsumogiri,
                event_index=discard.event_index,
                is_riichi_declaration=discard.is_riichi_declaration,
                was_called=discard.was_called,
                call_type=discard.call_type,
                called_by_actor=discard.called_by_actor,
                call_event_index=discard.call_event_index,
            )
            for discard in record.actor_discards_before_riichi
        ),
        wait_tiles=record.wait_tiles,
        wait_tile_count=record.wait_tile_count,
        wait_details=tuple(
            DatasetWaitDetail(detail.wait_tile, detail.hand_type, detail.wait_shape)
            for detail in record.wait_details
        ),
        wait_shapes=tuple(record.wait_shapes),
        contains_ryanmen=record.contains_ryanmen,
        is_pure_ryanmen=record.is_pure_ryanmen,
        is_multiwait=record.is_multiwait,
    )


def dataset_record_to_dict(record: RiichiWaitDatasetRecord) -> dict[str, Any]:
    """Return the schema-v1 JSON object for one record."""
    if not isinstance(record, RiichiWaitDatasetRecord):
        raise TypeError("record must be a RiichiWaitDatasetRecord")
    return {
        "source": {
            "relative_path": record.relative_source_path,
            "year": record.year,
        },
        "kyoku": {
            "bakaze": record.bakaze,
            "dora_marker": record.dora_marker,
            "honba": record.honba,
            "kyoku": record.kyoku,
            "oya": record.oya,
            "scores_at_start": list(record.scores_at_start),
            "start_kyoku_line": record.start_kyoku_line,
        },
        "riichi": {
            "actor": record.actor,
            "declaration_dahai_event_index": record.declaration_dahai_event_index,
            "declaration_dahai_line": record.declaration_dahai_line,
            "reach_accepted_event_index": record.reach_accepted_event_index,
            "reach_accepted_line": record.reach_accepted_line,
            "reach_event_index": record.reach_event_index,
            "reach_line": record.reach_line,
            "riichi_declaration_tile": record.riichi_declaration_tile,
            "riichi_declaration_tile_kind": record.riichi_declaration_tile_kind,
            "riichi_discard_number": record.riichi_discard_number,
        },
        "hand": {
            "concealed_tiles_after_discard": list(record.concealed_tiles_after_discard),
            "fixed_melds": [
                {"meld_type": meld.meld_type, "tiles": list(meld.tiles)}
                for meld in record.fixed_melds
            ],
        },
        "actor_discards_before_riichi": [
            {
                "call_event_index": discard.call_event_index,
                "call_type": discard.call_type,
                "called_by_actor": discard.called_by_actor,
                "discard_number": discard.discard_number,
                "event_index": discard.event_index,
                "is_riichi_declaration": discard.is_riichi_declaration,
                "normalized_tile": discard.normalized_tile,
                "tile": discard.tile,
                "tsumogiri": discard.tsumogiri,
                "was_called": discard.was_called,
            }
            for discard in record.actor_discards_before_riichi
        ],
        "waits": {
            "contains_ryanmen": record.contains_ryanmen,
            "is_multiwait": record.is_multiwait,
            "is_pure_ryanmen": record.is_pure_ryanmen,
            "wait_details": [
                {
                    "hand_type": detail.hand_type,
                    "tile": detail.tile,
                    "wait_shape": detail.wait_shape,
                }
                for detail in record.wait_details
            ],
            "wait_shapes": list(record.wait_shapes),
            "wait_tile_count": record.wait_tile_count,
            "wait_tiles": list(record.wait_tiles),
        },
    }


def serialize_dataset_record(record: RiichiWaitDatasetRecord) -> bytes:
    """Serialize one record as deterministic compact UTF-8 JSON (without LF)."""
    text = json.dumps(
        dataset_record_to_dict(record),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return text.encode("utf-8")


def dataset_record_from_dict(value: object) -> RiichiWaitDatasetRecord:
    """Parse and validate one schema-v1 record object."""
    data = _object(value, "record")
    source = _object(data.get("source"), "source")
    kyoku = _object(data.get("kyoku"), "kyoku")
    riichi = _object(data.get("riichi"), "riichi")
    hand = _object(data.get("hand"), "hand")
    waits = _object(data.get("waits"), "waits")
    return RiichiWaitDatasetRecord(
        year=source.get("year"),
        relative_source_path=source.get("relative_path"),
        start_kyoku_line=kyoku.get("start_kyoku_line"),
        reach_line=riichi.get("reach_line"),
        declaration_dahai_line=riichi.get("declaration_dahai_line"),
        reach_accepted_line=riichi.get("reach_accepted_line"),
        bakaze=kyoku.get("bakaze"),
        kyoku=kyoku.get("kyoku"),
        honba=kyoku.get("honba"),
        oya=kyoku.get("oya"),
        scores_at_start=_four_int_tuple(kyoku.get("scores_at_start")),
        dora_marker=kyoku.get("dora_marker"),
        actor=riichi.get("actor"),
        riichi_discard_number=riichi.get("riichi_discard_number"),
        riichi_declaration_tile=riichi.get("riichi_declaration_tile"),
        riichi_declaration_tile_kind=riichi.get("riichi_declaration_tile_kind"),
        reach_event_index=riichi.get("reach_event_index"),
        declaration_dahai_event_index=riichi.get("declaration_dahai_event_index"),
        reach_accepted_event_index=riichi.get("reach_accepted_event_index"),
        concealed_tiles_after_discard=_string_tuple(
            hand.get("concealed_tiles_after_discard"),
            "concealed_tiles_after_discard",
        ),
        fixed_melds=tuple(
            DatasetFixedMeld(
                _object(item, "fixed meld").get("meld_type"),
                _string_tuple(_object(item, "fixed meld").get("tiles"), "tiles"),
            )
            for item in _list(hand.get("fixed_melds"), "fixed_melds")
        ),
        actor_discards_before_riichi=tuple(
            _discard_from_dict(item)
            for item in _list(
                data.get("actor_discards_before_riichi"),
                "actor_discards_before_riichi",
            )
        ),
        wait_tiles=_string_tuple(waits.get("wait_tiles"), "wait_tiles"),
        wait_tile_count=waits.get("wait_tile_count"),
        wait_details=tuple(
            DatasetWaitDetail(
                _object(item, "wait detail").get("tile"),
                _object(item, "wait detail").get("hand_type"),
                _object(item, "wait detail").get("wait_shape"),
            )
            for item in _list(waits.get("wait_details"), "wait_details")
        ),
        wait_shapes=_string_tuple(waits.get("wait_shapes"), "wait_shapes"),
        contains_ryanmen=waits.get("contains_ryanmen"),
        is_pure_ryanmen=waits.get("is_pure_ryanmen"),
        is_multiwait=waits.get("is_multiwait"),
    )


def iter_dataset_records(path: str | Path) -> Iterator[RiichiWaitDatasetRecord]:
    """Read one JSONL stream after its manifest has been validated by the caller."""
    import gzip

    with gzip.open(path, mode="rt", encoding="utf-8", newline="") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.endswith("\n"):
                raise ValueError(
                    f"dataset line {line_number} is missing LF termination"
                )
            try:
                value = json.loads(line, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(
                    f"invalid dataset JSON at line {line_number}"
                ) from error
            yield dataset_record_from_dict(value)


def iter_validated_year_records(
    output_root: str | Path,
    year: int,
    *,
    expected_mode: str = "full",
) -> Iterator[RiichiWaitDatasetRecord]:
    """Verify a completed manifest and annual file before yielding records."""
    root = Path(output_root)
    manifest = load_manifest(root / "manifest.json", expected_mode=expected_mode)
    entries = manifest.get("years")
    if not isinstance(entries, list):
        raise TypeError("manifest years must be an array")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("year") == year
    ]
    if len(matches) != 1:
        raise ValueError(f"manifest must contain exactly one entry for year {year}")
    entry = matches[0]
    expected_filename = f"{year}.jsonl.gz"
    if entry.get("output_filename") != expected_filename:
        raise ValueError(f"unexpected output filename for year {year}")
    path = _verify_annual_file(root, entry)
    count = 0
    for record in iter_dataset_records(path):
        if record.year != year:
            raise ValueError(f"record year mismatch in {expected_filename}")
        count += 1
        yield record
    if count != entry.get("output_records"):
        raise ValueError(f"record count mismatch for year {year}")


def validate_manifest(
    value: object,
    *,
    expected_mode: str | None = None,
) -> Mapping[str, Any]:
    """Validate a completed schema-v1 manifest without reading annual payloads."""
    data = _object(value, "manifest")
    scope_years, _ = _validate_manifest_base(data, expected_mode=expected_mode)
    if "checkpoint" in data or "checkpoint_state" in data:
        raise ValueError("completed manifest must not contain checkpoint fields")
    _require_fields(data, ("years", "totals"), "completed manifest")
    entries = _list(data["years"], "manifest years")
    validated_entries = _validate_year_entries(entries, scope_years, checkpoint=False)
    totals = _object(data["totals"], "manifest totals")
    _require_fields(totals, _COUNT_FIELDS, "manifest totals")
    expected_totals = {
        field: sum(entry[field] for entry in validated_entries)
        for field in _COUNT_FIELDS
    }
    for field, expected in expected_totals.items():
        actual = _non_negative_int(totals.get(field), f"manifest totals {field}")
        if actual != expected:
            raise ValueError(f"manifest totals {field} does not match year entries")
    return data


def validate_checkpoint_manifest(
    value: object,
    *,
    expected_mode: str | None = None,
) -> Mapping[str, Any]:
    """Validate a resumable processing or final-ready checkpoint."""
    data = _object(value, "checkpoint manifest")
    scope_years, _ = _validate_manifest_base(data, expected_mode=expected_mode)
    if data.get("checkpoint") is not True:
        raise ValueError("checkpoint manifest must contain checkpoint=true")
    state = data.get("checkpoint_state")
    if state not in {"processing", "final_ready"}:
        raise ValueError("invalid checkpoint_state")
    if "totals" in data:
        raise ValueError("checkpoint manifest must not contain completed totals")
    _require_fields(data, ("years",), "checkpoint manifest")
    entries = _list(data["years"], "checkpoint years")
    validated_entries = _validate_year_entries(entries, scope_years, checkpoint=True)
    entry_years = tuple(entry["year"] for entry in validated_entries)
    if state == "final_ready" and (
        entry_years != scope_years
        or any(entry["status"] != "complete" for entry in validated_entries)
    ):
        raise ValueError("final_ready checkpoint requires every year complete")
    replacement = _object(data.get("replacement"), "checkpoint replacement")
    _require_fields(
        replacement,
        ("force", "had_completed_manifest"),
        "checkpoint replacement",
    )
    if type(replacement["force"]) is not bool:
        raise TypeError("checkpoint replacement force must be a bool")
    if type(replacement["had_completed_manifest"]) is not bool:
        raise TypeError("checkpoint replacement had_completed_manifest must be a bool")
    if replacement["had_completed_manifest"] and not replacement["force"]:
        raise ValueError("only a force checkpoint may replace a completed manifest")
    return data


def load_manifest(
    path: str | Path,
    *,
    expected_mode: str | None = None,
) -> Mapping[str, Any]:
    """Load and validate a completed dataset manifest."""
    with open(path, encoding="utf-8") as file:
        value = json.load(file, parse_constant=_reject_json_constant)
    return validate_manifest(value, expected_mode=expected_mode)


def validate_dataset_integrity(
    output_root: str | Path,
    *,
    expected_mode: str | None = None,
    deep: bool = False,
) -> Mapping[str, Any]:
    """Validate all annual files; optionally decode records and verify line counts."""
    if type(deep) is not bool:
        raise TypeError("deep must be a bool")
    root = Path(output_root)
    manifest = load_manifest(root / "manifest.json", expected_mode=expected_mode)
    entries = _list(manifest["years"], "manifest years")
    for entry_value in entries:
        entry = _object(entry_value, "manifest year entry")
        path = _verify_annual_file(root, entry)
        if not deep:
            continue
        count = 0
        for record in iter_dataset_records(path):
            if record.year != entry["year"]:
                raise ValueError(f"record year mismatch in {path.name}")
            count += 1
        if count != entry["output_records"]:
            raise ValueError(f"record count mismatch for year {entry['year']}")
    return manifest


def _validate_manifest_base(
    data: dict[str, Any],
    *,
    expected_mode: str | None,
) -> tuple[tuple[int, ...], str]:
    version = data.get("schema_version")
    if type(version) is not int or version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported dataset schema_version: {version!r}")
    if data.get("dataset_name") != DATASET_NAME:
        raise ValueError("unexpected dataset_name")
    _require_fields(
        data,
        (
            "schema_version",
            "dataset_name",
            "created_at_utc",
            "source",
            "scope",
            "serialization",
            "generator",
        ),
        "manifest",
    )
    _validate_created_at(data["created_at_utc"])
    _validate_source(_object(data["source"], "manifest source"))
    scope_years, mode, max_files = _validate_scope(
        _object(data["scope"], "manifest scope"),
        expected_mode=expected_mode,
    )
    _validate_serialization(_object(data["serialization"], "manifest serialization"))
    _validate_generator(
        _object(data["generator"], "manifest generator"),
        scope_years=scope_years,
        mode=mode,
        max_files=max_files,
    )
    return scope_years, mode


def _validate_created_at(value: object) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("created_at_utc must be an ISO-8601 UTC string ending in Z")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise ValueError("created_at_utc must be valid ISO-8601") from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("created_at_utc must be UTC")


def _validate_source(source: dict[str, Any]) -> None:
    fields = (
        "repository",
        "release_tag",
        "validation_summary_path",
        "validation_summary_sha256",
        "archive_hashes_verified",
    )
    _require_fields(source, fields, "manifest source")
    if not isinstance(source["repository"], str) or not source["repository"]:
        raise ValueError("manifest source repository must be non-empty text")
    if source["release_tag"] != SOURCE_RELEASE_TAG:
        raise ValueError(f"manifest source release_tag must be {SOURCE_RELEASE_TAG}")
    logical_path = source["validation_summary_path"]
    if (
        not isinstance(logical_path, str)
        or not logical_path
        or "\\" in logical_path
        or Path(logical_path).is_absolute()
        or ".." in logical_path.split("/")
    ):
        raise ValueError("validation_summary_path must be relative POSIX text")
    _validate_sha256(
        source["validation_summary_sha256"],
        "manifest source validation_summary_sha256",
    )
    if source["archive_hashes_verified"] is not True:
        raise ValueError("manifest source archives must be SHA256-verified")


def _validate_scope(
    scope: dict[str, Any],
    *,
    expected_mode: str | None,
) -> tuple[tuple[int, ...], str, int | None]:
    fields = (
        "years",
        "rule_code",
        "aka_flag",
        "bakaze",
        "input_selection",
        "extraction_mode",
    )
    _require_fields(scope, fields, "manifest scope")
    year_values = _list(scope["years"], "manifest scope years")
    years = tuple(year_values)
    if not years or any(type(year) is not int for year in years):
        raise TypeError("manifest scope years must contain integers")
    if any(year not in SUPPORTED_YEARS for year in years):
        raise ValueError("manifest scope contains an unsupported year")
    if years != tuple(sorted(set(years))):
        raise ValueError("manifest scope years must be unique and ascending")
    if scope["rule_code"] != RULE_CODE:
        raise ValueError(f"manifest scope rule_code must be {RULE_CODE}")
    if scope["aka_flag"] is not True:
        raise ValueError("manifest scope aka_flag must be true")
    if scope["bakaze"] != "E":
        raise ValueError("manifest scope bakaze must be E")
    mode = scope["extraction_mode"]
    if mode not in {"full", "sample"}:
        raise ValueError("manifest extraction_mode must be full or sample")
    if expected_mode is not None and mode != expected_mode:
        raise ValueError(
            f"dataset mode mismatch: expected {expected_mode!r}, found {mode!r}"
        )
    selection = _object(scope["input_selection"], "manifest input_selection")
    _require_fields(
        selection,
        ("ordering", "max_files_before_target_filtering"),
        "manifest input_selection",
    )
    if selection["ordering"] != INPUT_ORDERING:
        raise ValueError("manifest input ordering does not match schema v1")
    max_files = selection["max_files_before_target_filtering"]
    if mode == "full":
        if max_files is not None:
            raise ValueError("full manifest must not limit input files")
    elif type(max_files) is not int or max_files < 1:
        raise ValueError("sample manifest requires a positive max_files")
    return years, mode, max_files


def _validate_serialization(serialization: dict[str, Any]) -> None:
    fields = (
        "format",
        "encoding",
        "json_options",
        "compression",
        "compression_level",
        "gzip_mtime",
        "gzip_header_filename",
    )
    _require_fields(serialization, fields, "manifest serialization")
    expected_scalars = {
        "format": "JSON Lines",
        "encoding": "UTF-8",
        "compression": "gzip",
        "compression_level": COMPRESSION_LEVEL,
        "gzip_mtime": 0,
        "gzip_header_filename": "",
    }
    for field, expected in expected_scalars.items():
        actual = serialization[field]
        if type(expected) is int and type(actual) is not int:
            raise TypeError(f"manifest serialization {field} must be an integer")
        if actual != expected:
            raise ValueError(f"manifest serialization {field} does not match schema v1")
    options = _object(serialization["json_options"], "manifest json_options")
    expected_options = {
        "ensure_ascii": False,
        "sort_keys": True,
        "separators": [",", ":"],
        "allow_nan": False,
        "line_terminator": "LF",
    }
    _require_fields(options, tuple(expected_options), "manifest json_options")
    for field, expected in expected_options.items():
        actual = options[field]
        if type(expected) is bool and type(actual) is not bool:
            raise TypeError(f"manifest json option {field} must be a bool")
        if actual != expected:
            raise ValueError(f"manifest json option {field} does not match schema v1")


def _validate_generator(
    generator: dict[str, Any],
    *,
    scope_years: tuple[int, ...],
    mode: str,
    max_files: int | None,
) -> None:
    fields = (
        "git_commit",
        "worktree_clean",
        "python_version",
        "zlib_version",
        "invocation",
    )
    _require_fields(generator, fields, "manifest generator")
    for field in ("git_commit", "python_version", "zlib_version"):
        if not isinstance(generator[field], str) or not generator[field]:
            raise ValueError(f"manifest generator {field} must be non-empty text")
    if type(generator["worktree_clean"]) is not bool:
        raise TypeError("manifest generator worktree_clean must be a bool")
    if mode == "full" and generator["worktree_clean"] is not True:
        raise ValueError("full manifest requires a clean generator worktree")
    invocation = _object(generator["invocation"], "manifest invocation")
    _require_fields(invocation, ("years", "mode", "max_files"), "manifest invocation")
    invocation_years = _list(invocation["years"], "manifest invocation years")
    if tuple(invocation_years) != scope_years:
        raise ValueError("manifest invocation years must match scope years")
    if invocation["mode"] != mode or invocation["max_files"] != max_files:
        raise ValueError("manifest invocation must match scope selection")


def _validate_year_entries(
    entries: list[Any],
    scope_years: tuple[int, ...],
    *,
    checkpoint: bool,
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for entry_value in entries:
        entry = _object(entry_value, "manifest year entry")
        fields = (
            "year",
            *_COUNT_FIELDS,
            "output_filename",
            "compressed_size_bytes",
            "sha256",
            "elapsed_seconds",
        )
        if checkpoint:
            fields = (*fields, "status")
        _require_fields(entry, fields, "manifest year entry")
        if not checkpoint and "status" in entry:
            raise ValueError("completed manifest year must not contain status")
        year = entry["year"]
        if type(year) is not int or year not in scope_years:
            raise ValueError("manifest year entry has invalid year")
        for field in _COUNT_FIELDS:
            _non_negative_int(entry[field], f"manifest year {year} {field}")
        if entry["target_games"] > entry["scanned_files"]:
            raise ValueError("manifest target_games cannot exceed scanned_files")
        if entry["established_riichis"] != entry["output_records"]:
            raise ValueError("established_riichis must equal output_records")
        if entry["output_filename"] != f"{year}.jsonl.gz":
            raise ValueError(f"invalid output filename for year {year}")
        _non_negative_int(
            entry["compressed_size_bytes"],
            f"manifest year {year} compressed_size_bytes",
        )
        _validate_sha256(entry["sha256"], f"manifest year {year} sha256")
        elapsed = entry["elapsed_seconds"]
        if (
            type(elapsed) not in {int, float}
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise ValueError(
                "manifest elapsed_seconds must be a finite non-negative number"
            )
        if checkpoint and entry["status"] not in {"ready", "complete"}:
            raise ValueError("checkpoint year status must be ready or complete")
        validated.append(entry)
    years = tuple(entry["year"] for entry in validated)
    if len(set(years)) != len(years):
        raise ValueError("manifest year entries must not contain duplicates")
    if years != tuple(sorted(years)):
        raise ValueError("manifest year entries must be in ascending order")
    if checkpoint:
        if any(year not in scope_years for year in years):
            raise ValueError("checkpoint contains a year outside its scope")
    elif years != scope_years:
        raise ValueError("completed manifest years must match scope years")
    return validated


def _verify_annual_file(root: Path, entry: Mapping[str, Any]) -> Path:
    year = entry["year"]
    path = root / entry["output_filename"]
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != entry["compressed_size_bytes"]:
        raise ValueError(f"annual file size mismatch for year {year}")
    if _sha256_file(path) != entry["sha256"]:
        raise ValueError(f"annual file SHA256 mismatch for year {year}")
    return path


def _require_fields(
    value: Mapping[str, Any],
    fields: tuple[str, ...],
    context: str,
) -> None:
    missing = tuple(field for field in fields if field not in value)
    if missing:
        raise ValueError(f"{context} is missing required fields: {missing}")


def _non_negative_int(value: object, context: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{context} must be an integer")
    if value < 0:
        raise ValueError(f"{context} must be non-negative")
    return value


def _validate_sha256(value: object, context: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA256 digest")


def _discard_from_dict(value: object) -> DatasetActorDiscard:
    data = _object(value, "discard")
    return DatasetActorDiscard(
        discard_number=data.get("discard_number"),
        tile=data.get("tile"),
        normalized_tile=data.get("normalized_tile"),
        tsumogiri=data.get("tsumogiri"),
        event_index=data.get("event_index"),
        is_riichi_declaration=data.get("is_riichi_declaration"),
        was_called=data.get("was_called"),
        call_type=data.get("call_type"),
        called_by_actor=data.get("called_by_actor"),
        call_event_index=data.get("call_event_index"),
    )


def _object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array")
    return value


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    values = _list(value, name)
    if not all(isinstance(item, str) for item in values):
        raise TypeError(f"{name} must contain strings")
    return tuple(values)


def _four_int_tuple(value: object) -> tuple[int, int, int, int]:
    values = _list(value, "scores_at_start")
    if len(values) != 4 or any(type(item) is not int for item in values):
        raise TypeError("scores_at_start must contain four integers")
    return values[0], values[1], values[2], values[3]


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
