"""Compare production and independent-reference riichi wait candidates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from mahjong_analysis.hand_waits import HAND_TYPE_ORDER, WAIT_SHAPE_ORDER
from mahjong_analysis.mjai import (
    filter_east_kyokus,
    is_target_game,
    load_mjai,
    split_kyoku,
)
from mahjong_analysis.riichi import extract_established_riichis
from mahjong_analysis.riichi_wait_records import (
    RiichiWaitRecord,
    build_riichi_wait_records,
)
from mahjong_analysis.riichi_wait_reference import (
    ReferenceKyoku,
    ReferenceRiichiCandidate,
    extract_reference_riichi_candidates,
    filter_reference_east_kyokus,
    is_reference_target_game,
    read_reference_mjai,
    split_reference_kyokus,
)
from mahjong_analysis.tiles import tile_to_index

CandidateKey = tuple[str, int, int]
ComparisonSide = Literal["production", "reference"]

_HAND_TYPE_INDEX = {hand_type: index for index, hand_type in enumerate(HAND_TYPE_ORDER)}
_WAIT_SHAPE_INDEX = {
    wait_shape: index for index, wait_shape in enumerate(WAIT_SHAPE_ORDER)
}
_COMPARISON_FIELDS = (
    "actor",
    "bakaze",
    "kyoku",
    "honba",
    "oya",
    "declaration_dahai_line",
    "reach_accepted_line",
    "riichi_discard_number",
    "riichi_declaration_tile",
    "riichi_declaration_tile_kind",
    "concealed_tiles_after_discard",
    "fixed_melds",
    "actor_discards_before_riichi",
    "wait_tiles",
    "wait_tile_count",
    "wait_details",
    "wait_shapes",
    "contains_ryanmen",
    "is_pure_ryanmen",
    "is_multiwait",
)


@dataclass(frozen=True)
class ComparableMeld:
    """A fixed meld normalized only for order-insensitive comparison."""

    meld_type: str
    tiles: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.meld_type, str):
            raise TypeError("meld_type must be a string")
        if not isinstance(self.tiles, tuple) or not all(
            isinstance(tile, str) for tile in self.tiles
        ):
            raise TypeError("meld tiles must be a tuple of strings")
        object.__setattr__(self, "tiles", _sort_raw_tiles(self.tiles))


@dataclass(frozen=True)
class ComparableDiscard:
    """One ordered river entry with all formally compared metadata."""

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


@dataclass(frozen=True)
class ComparableWaitDetail:
    """One normalized wait detail."""

    wait_tile: str
    hand_type: str
    wait_shape: str


@dataclass(frozen=True)
class ComparableRiichiCandidate:
    """Common, lossless comparison representation for one candidate."""

    relative_source_path: str
    start_kyoku_line: int
    reach_line: int
    actor: int
    bakaze: str
    kyoku: int
    honba: int
    oya: int
    declaration_dahai_line: int
    reach_accepted_line: int
    riichi_discard_number: int
    riichi_declaration_tile: str
    riichi_declaration_tile_kind: str
    concealed_tiles_after_discard: tuple[str, ...]
    fixed_melds: tuple[ComparableMeld, ...]
    actor_discards_before_riichi: tuple[ComparableDiscard, ...]
    wait_tiles: tuple[str, ...]
    wait_tile_count: int
    wait_details: tuple[ComparableWaitDetail, ...]
    wait_shapes: tuple[str, ...]
    contains_ryanmen: bool
    is_pure_ryanmen: bool
    is_multiwait: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.relative_source_path, str)
            or not self.relative_source_path
        ):
            raise ValueError("relative_source_path must be a non-empty string")
        if "\\" in self.relative_source_path:
            raise ValueError("relative_source_path must use '/' separators")
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

        object.__setattr__(
            self,
            "concealed_tiles_after_discard",
            _sort_raw_tiles(self.concealed_tiles_after_discard),
        )
        object.__setattr__(
            self,
            "fixed_melds",
            tuple(
                sorted(self.fixed_melds, key=lambda meld: (meld.meld_type, meld.tiles))
            ),
        )
        object.__setattr__(
            self,
            "wait_tiles",
            tuple(sorted(self.wait_tiles, key=tile_to_index)),
        )
        object.__setattr__(
            self,
            "wait_details",
            tuple(sorted(self.wait_details, key=_wait_detail_sort_key)),
        )
        object.__setattr__(
            self,
            "wait_shapes",
            tuple(sorted(self.wait_shapes, key=_wait_shape_sort_key)),
        )

    @property
    def candidate_key(self) -> CandidateKey:
        """Return the specification candidate key."""
        return self.relative_source_path, self.start_kyoku_line, self.reach_line

    @property
    def related_lines(self) -> tuple[int, int, int, int]:
        """Return source lines useful for mismatch diagnosis."""
        return (
            self.start_kyoku_line,
            self.reach_line,
            self.declaration_dahai_line,
            self.reach_accepted_line,
        )


@dataclass(frozen=True)
class ProductionOnly:
    """A candidate key found only by production."""

    key: CandidateKey
    candidate: ComparableRiichiCandidate


@dataclass(frozen=True)
class ReferenceOnly:
    """A candidate key found only by reference."""

    key: CandidateKey
    candidate: ComparableRiichiCandidate


@dataclass(frozen=True)
class FieldMismatch:
    """One differing field for a candidate key shared by both sides."""

    key: CandidateKey
    field_name: str
    production_value: object
    reference_value: object
    production_actor: int
    reference_actor: int
    source_path: str
    production_lines: tuple[int, int, int, int]
    reference_lines: tuple[int, int, int, int]


@dataclass(frozen=True)
class ScopeMismatch:
    """A target-game or east-kyoku scope decision mismatch."""

    source_path: str
    field_name: Literal["target_game", "east_kyoku_start_lines"]
    production_value: object
    reference_value: object


@dataclass(frozen=True)
class ProcessingError:
    """One input file for which at least one side raised an exception."""

    source_path: str
    production_error: str | None
    reference_error: str | None

    def __post_init__(self) -> None:
        if self.production_error is None and self.reference_error is None:
            raise ValueError("processing_error must contain at least one exception")


@dataclass(frozen=True)
class CandidateComparison:
    """Pure candidate-key and field comparison result."""

    production_candidates: int
    reference_candidates: int
    production_only: tuple[ProductionOnly, ...]
    reference_only: tuple[ReferenceOnly, ...]
    field_mismatches: tuple[FieldMismatch, ...]

    @property
    def is_pass(self) -> bool:
        """Return whether no candidate-level difference exists."""
        return not (
            self.production_only or self.reference_only or self.field_mismatches
        )


@dataclass(frozen=True)
class RiichiWaitComparisonReport:
    """Complete multi-file comparison counts and classified differences."""

    scanned_files: int
    production_target_games: int
    reference_target_games: int
    production_east_kyokus: int
    reference_east_kyokus: int
    production_candidates: int
    reference_candidates: int
    production_only: tuple[ProductionOnly, ...]
    reference_only: tuple[ReferenceOnly, ...]
    field_mismatches: tuple[FieldMismatch, ...]
    scope_mismatches: tuple[ScopeMismatch, ...]
    processing_errors: tuple[ProcessingError, ...]

    @property
    def is_pass(self) -> bool:
        """Return the strict no-difference, no-error PASS decision."""
        return not (
            self.production_only
            or self.reference_only
            or self.field_mismatches
            or self.scope_mismatches
            or self.processing_errors
        )


@dataclass(frozen=True)
class _SideFileResult:
    source_path: str
    target_game: bool
    east_kyoku_start_lines: tuple[int, ...]
    candidates: tuple[ComparableRiichiCandidate, ...]


def compare_candidates(
    production_candidates: Iterable[ComparableRiichiCandidate],
    reference_candidates: Iterable[ComparableRiichiCandidate],
) -> CandidateComparison:
    """Compare two candidate collections by key and then by every formal field."""
    production = tuple(production_candidates)
    reference = tuple(reference_candidates)
    production_by_key = _index_candidates(production, "production")
    reference_by_key = _index_candidates(reference, "reference")

    production_keys = set(production_by_key)
    reference_keys = set(reference_by_key)
    production_only = tuple(
        ProductionOnly(key, production_by_key[key])
        for key in sorted(production_keys - reference_keys)
    )
    reference_only = tuple(
        ReferenceOnly(key, reference_by_key[key])
        for key in sorted(reference_keys - production_keys)
    )

    field_mismatches: list[FieldMismatch] = []
    for key in sorted(production_keys & reference_keys):
        production_candidate = production_by_key[key]
        reference_candidate = reference_by_key[key]
        for field_name in _COMPARISON_FIELDS:
            production_value = getattr(production_candidate, field_name)
            reference_value = getattr(reference_candidate, field_name)
            if production_value != reference_value:
                field_mismatches.append(
                    FieldMismatch(
                        key=key,
                        field_name=field_name,
                        production_value=production_value,
                        reference_value=reference_value,
                        production_actor=production_candidate.actor,
                        reference_actor=reference_candidate.actor,
                        source_path=key[0],
                        production_lines=production_candidate.related_lines,
                        reference_lines=reference_candidate.related_lines,
                    )
                )

    return CandidateComparison(
        production_candidates=len(production),
        reference_candidates=len(reference),
        production_only=production_only,
        reference_only=reference_only,
        field_mismatches=tuple(field_mismatches),
    )


def compare_riichi_wait_files(
    paths: Iterable[str | Path],
    *,
    raw_root: str | Path,
) -> RiichiWaitComparisonReport:
    """Process identical files independently and return a strict comparison."""
    root = Path(raw_root).resolve()
    ordered_paths = _ordered_paths(paths, root)
    production_candidates: list[ComparableRiichiCandidate] = []
    reference_candidates: list[ComparableRiichiCandidate] = []
    scope_mismatches: list[ScopeMismatch] = []
    processing_errors: list[ProcessingError] = []
    production_target_games = 0
    reference_target_games = 0
    production_east_kyokus = 0
    reference_east_kyokus = 0
    production_candidate_count = 0
    reference_candidate_count = 0

    for path in ordered_paths:
        source_path = _relative_source_path(path, root)
        production_result, production_error = _capture_side_processing(
            _process_production_file, path, root
        )
        reference_result, reference_error = _capture_side_processing(
            _process_reference_file, path, root
        )

        if production_result is not None:
            _index_candidates(production_result.candidates, "production")
            production_target_games += int(production_result.target_game)
            production_east_kyokus += len(production_result.east_kyoku_start_lines)
            production_candidate_count += len(production_result.candidates)
        if reference_result is not None:
            _index_candidates(reference_result.candidates, "reference")
            reference_target_games += int(reference_result.target_game)
            reference_east_kyokus += len(reference_result.east_kyoku_start_lines)
            reference_candidate_count += len(reference_result.candidates)

        if production_error is not None or reference_error is not None:
            processing_errors.append(
                ProcessingError(source_path, production_error, reference_error)
            )
            continue
        if production_result is None or reference_result is None:
            raise RuntimeError("side processing returned neither a result nor an error")

        file_scope_mismatches = _compare_file_scope(production_result, reference_result)
        scope_mismatches.extend(file_scope_mismatches)
        if file_scope_mismatches:
            continue
        production_candidates.extend(production_result.candidates)
        reference_candidates.extend(reference_result.candidates)

    candidate_comparison = compare_candidates(
        production_candidates,
        reference_candidates,
    )
    return RiichiWaitComparisonReport(
        scanned_files=len(ordered_paths),
        production_target_games=production_target_games,
        reference_target_games=reference_target_games,
        production_east_kyokus=production_east_kyokus,
        reference_east_kyokus=reference_east_kyokus,
        production_candidates=production_candidate_count,
        reference_candidates=reference_candidate_count,
        production_only=candidate_comparison.production_only,
        reference_only=candidate_comparison.reference_only,
        field_mismatches=candidate_comparison.field_mismatches,
        scope_mismatches=tuple(scope_mismatches),
        processing_errors=tuple(processing_errors),
    )


def _process_production_file(path: Path, raw_root: Path) -> _SideFileResult:
    source_path = _relative_source_path(path, raw_root)
    events = load_mjai(path)
    target_game = is_target_game(path, events)
    if not target_game:
        return _SideFileResult(source_path, False, (), ())

    line_by_event_id = {
        id(event): line_number for line_number, event in enumerate(events, start=1)
    }
    east_kyokus = filter_east_kyokus(split_kyoku(events))
    start_lines = tuple(line_by_event_id[id(kyoku[0])] for kyoku in east_kyokus)
    candidates: list[ComparableRiichiCandidate] = []
    for kyoku, start_line in zip(east_kyokus, start_lines, strict=True):
        records = build_riichi_wait_records(extract_established_riichis(kyoku))
        candidates.extend(
            _production_candidate(
                record,
                kyoku,
                line_by_event_id,
                source_path,
                start_line,
            )
            for record in records
        )
    return _SideFileResult(source_path, True, start_lines, tuple(candidates))


def _process_reference_file(path: Path, raw_root: Path) -> _SideFileResult:
    log = read_reference_mjai(path, source_root=raw_root)
    target_game = is_reference_target_game(log)
    if not target_game:
        return _SideFileResult(log.source_path, False, (), ())

    east_kyokus = filter_reference_east_kyokus(split_reference_kyokus(log))
    start_lines = tuple(kyoku.start_kyoku_line for kyoku in east_kyokus)
    candidates = tuple(
        _reference_candidate(candidate, kyoku)
        for kyoku in east_kyokus
        for candidate in extract_reference_riichi_candidates(kyoku)
    )
    return _SideFileResult(log.source_path, True, start_lines, candidates)


def _production_candidate(
    record: RiichiWaitRecord,
    kyoku_events: list[dict[str, Any]],
    line_by_event_id: Mapping[int, int],
    source_path: str,
    start_line: int,
) -> ComparableRiichiCandidate:
    start = kyoku_events[0]
    return ComparableRiichiCandidate(
        relative_source_path=source_path,
        start_kyoku_line=start_line,
        reach_line=_production_event_line(
            kyoku_events, record.reach_event_index, line_by_event_id
        ),
        actor=record.actor,
        bakaze=_required_start_string(start, "bakaze"),
        kyoku=_required_start_int(start, "kyoku"),
        honba=_required_start_int(start, "honba"),
        oya=_required_start_int(start, "oya"),
        declaration_dahai_line=_production_event_line(
            kyoku_events, record.declaration_dahai_event_index, line_by_event_id
        ),
        reach_accepted_line=_production_event_line(
            kyoku_events, record.reach_accepted_event_index, line_by_event_id
        ),
        riichi_discard_number=record.riichi_discard_number,
        riichi_declaration_tile=record.riichi_declaration_tile,
        riichi_declaration_tile_kind=record.riichi_declaration_tile_kind,
        concealed_tiles_after_discard=record.concealed_tiles_after_discard,
        fixed_melds=tuple(
            ComparableMeld(meld.meld_type, tuple(meld.tiles))
            for meld in record.fixed_melds
        ),
        actor_discards_before_riichi=tuple(
            _comparable_discard(discard)
            for discard in record.actor_discards_before_riichi
        ),
        wait_tiles=record.wait_tiles,
        wait_tile_count=record.wait_tile_count,
        wait_details=tuple(
            ComparableWaitDetail(detail.wait_tile, detail.hand_type, detail.wait_shape)
            for detail in record.wait_details
        ),
        wait_shapes=tuple(record.wait_shapes),
        contains_ryanmen=record.contains_ryanmen,
        is_pure_ryanmen=record.is_pure_ryanmen,
        is_multiwait=record.is_multiwait,
    )


def _reference_candidate(
    candidate: ReferenceRiichiCandidate,
    kyoku: ReferenceKyoku,
) -> ComparableRiichiCandidate:
    start = kyoku.events[0].data
    return ComparableRiichiCandidate(
        relative_source_path=candidate.source_path,
        start_kyoku_line=candidate.start_kyoku_line,
        reach_line=candidate.reach_line,
        actor=candidate.actor,
        bakaze=_required_start_string(start, "bakaze"),
        kyoku=_required_start_int(start, "kyoku"),
        honba=_required_start_int(start, "honba"),
        oya=_required_start_int(start, "oya"),
        declaration_dahai_line=candidate.declaration_dahai_line,
        reach_accepted_line=candidate.reach_accepted_line,
        riichi_discard_number=candidate.riichi_discard_number,
        riichi_declaration_tile=candidate.riichi_declaration_tile,
        riichi_declaration_tile_kind=candidate.riichi_declaration_tile_kind,
        concealed_tiles_after_discard=candidate.concealed_tiles_after_discard,
        fixed_melds=tuple(
            ComparableMeld(meld.meld_type, tuple(meld.tiles))
            for meld in candidate.fixed_melds
        ),
        actor_discards_before_riichi=tuple(
            _comparable_discard(discard)
            for discard in candidate.actor_discards_before_riichi
        ),
        wait_tiles=candidate.wait_tiles,
        wait_tile_count=candidate.wait_tile_count,
        wait_details=tuple(
            ComparableWaitDetail(detail.wait_tile, detail.hand_type, detail.wait_shape)
            for detail in candidate.wait_details
        ),
        wait_shapes=tuple(candidate.wait_shapes),
        contains_ryanmen=candidate.contains_ryanmen,
        is_pure_ryanmen=candidate.is_pure_ryanmen,
        is_multiwait=candidate.is_multiwait,
    )


def _comparable_discard(discard: Any) -> ComparableDiscard:
    return ComparableDiscard(
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


def _compare_file_scope(
    production: _SideFileResult,
    reference: _SideFileResult,
) -> tuple[ScopeMismatch, ...]:
    if production.source_path != reference.source_path:
        raise ValueError("production/reference source paths do not match")
    if production.target_game != reference.target_game:
        return (
            ScopeMismatch(
                production.source_path,
                "target_game",
                production.target_game,
                reference.target_game,
            ),
        )
    if (
        production.target_game
        and production.east_kyoku_start_lines != reference.east_kyoku_start_lines
    ):
        return (
            ScopeMismatch(
                production.source_path,
                "east_kyoku_start_lines",
                production.east_kyoku_start_lines,
                reference.east_kyoku_start_lines,
            ),
        )
    return ()


def _capture_side_processing(
    processor: Any,
    path: Path,
    raw_root: Path,
) -> tuple[_SideFileResult | None, str | None]:
    try:
        return processor(path, raw_root), None
    except Exception as error:  # noqa: BLE001 - comparison preserves side failures
        return None, f"{type(error).__name__}: {error}"


def _index_candidates(
    candidates: tuple[ComparableRiichiCandidate, ...],
    side: ComparisonSide,
) -> dict[CandidateKey, ComparableRiichiCandidate]:
    result: dict[CandidateKey, ComparableRiichiCandidate] = {}
    for candidate in candidates:
        if not isinstance(candidate, ComparableRiichiCandidate):
            raise TypeError(f"{side} candidates must be ComparableRiichiCandidate")
        key = candidate.candidate_key
        if key in result:
            raise ValueError(f"duplicate {side} candidate key: {key!r}")
        result[key] = candidate
    return result


def _ordered_paths(paths: Iterable[str | Path], raw_root: Path) -> tuple[Path, ...]:
    by_source: dict[str, Path] = {}
    for value in paths:
        path = Path(value).resolve()
        source_path = _relative_source_path(path, raw_root)
        if source_path in by_source:
            raise ValueError(f"duplicate input path: {source_path}")
        by_source[source_path] = path
    return tuple(by_source[source] for source in sorted(by_source))


def _relative_source_path(path: Path, raw_root: Path) -> str:
    try:
        return path.resolve().relative_to(raw_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"input path is outside raw_root: {path}") from error


def _production_event_line(
    kyoku_events: Sequence[dict[str, Any]],
    event_index: int,
    line_by_event_id: Mapping[int, int],
) -> int:
    try:
        return line_by_event_id[id(kyoku_events[event_index])]
    except (IndexError, KeyError) as error:
        raise ValueError(
            f"kyoku event index {event_index} has no physical source line"
        ) from error


def _required_start_string(start: Mapping[str, Any], field_name: str) -> str:
    value = start.get(field_name)
    if not isinstance(value, str):
        raise TypeError(f"start_kyoku {field_name} must be a string")
    return value


def _required_start_int(start: Mapping[str, Any], field_name: str) -> int:
    value = start.get(field_name)
    if type(value) is not int:
        raise ValueError(f"start_kyoku {field_name} must be an integer")
    return value


def _sort_raw_tiles(tiles: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(tiles, key=lambda tile: (tile_to_index(tile), tile)))


def _wait_detail_sort_key(detail: ComparableWaitDetail) -> tuple[int, int, int]:
    try:
        return (
            tile_to_index(detail.wait_tile),
            _HAND_TYPE_INDEX[detail.hand_type],
            _WAIT_SHAPE_INDEX[detail.wait_shape],
        )
    except KeyError as error:
        raise ValueError(f"invalid wait detail: {detail!r}") from error


def _wait_shape_sort_key(wait_shape: str) -> int:
    try:
        return _WAIT_SHAPE_INDEX[wait_shape]
    except KeyError as error:
        raise ValueError(f"invalid wait shape: {wait_shape!r}") from error
