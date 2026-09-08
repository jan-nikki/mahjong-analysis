"""Print deterministic, human-readable audit samples for 2025 riichi waits."""

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mahjong_analysis.mjai import (
    filter_east_kyokus,
    is_target_game,
    load_mjai,
    split_kyoku,
)
from mahjong_analysis.riichi import ActorDiscard, extract_established_riichis
from mahjong_analysis.riichi_wait_records import (
    RiichiWaitRecord,
    build_riichi_wait_records,
)
from mahjong_analysis.tiles import RED_FIVE_NORMALIZATION

YEAR = 2025
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_MAX_FILES = 100
DEFAULT_SAMPLES_PER_CATEGORY = 1
DEFAULT_CONTEXT_BEFORE = 3
DEFAULT_CONTEXT_AFTER = 3
EARLY_RIICHI_MAX_DISCARD = 6
LATE_RIICHI_MIN_DISCARD = 13

CATEGORY_ORDER = (
    "pure_ryanmen",
    "kanchan",
    "penchan",
    "shanpon",
    "tanki",
    "multiwait_3_plus",
    "contains_ryanmen_not_pure",
    "red_five",
    "ankan",
    "early_riichi_1_to_6",
    "late_riichi_13_plus",
)

CATEGORY_LABELS = {
    "pure_ryanmen": "純両面",
    "kanchan": "カンチャン",
    "penchan": "ペンチャン",
    "shanpon": "シャンポン",
    "tanki": "単騎",
    "multiwait_3_plus": "3面張以上",
    "contains_ryanmen_not_pure": "両面を含むが純両面ではない待ち",
    "red_five": "赤5を含むリーチ",
    "ankan": "暗槓ありリーチ",
    "early_riichi_1_to_6": "早巡リーチ（本人第1〜6打）",
    "late_riichi_13_plus": "遅巡リーチ（本人第13打以降）",
}


@dataclass(frozen=True)
class AuditEvent:
    """One complete MJAI event with both local and physical positions."""

    kyoku_event_index: int
    game_line_number: int
    event_json: str


@dataclass(frozen=True)
class AuditDiscard:
    """One actor discard paired with its physical source line."""

    game_line_number: int
    discard: ActorDiscard


@dataclass(frozen=True)
class RiichiWaitAuditSample:
    """One source-locatable riichi wait record for manual inspection."""

    source_path: str
    filename: str
    start_kyoku_line: int
    bakaze: str
    kyoku: int
    honba: int
    kyotaku: int
    oya: int
    is_dealer: bool
    matched_categories: tuple[str, ...]
    record: RiichiWaitRecord
    reach_event: AuditEvent
    declaration_dahai_event: AuditEvent
    reach_accepted_event: AuditEvent
    actor_discards: tuple[AuditDiscard, ...]
    context_events: tuple[AuditEvent, ...]


@dataclass(frozen=True)
class CategorySamples:
    """Selected samples for one audit category."""

    category: str
    samples: tuple[RiichiWaitAuditSample, ...]


@dataclass(frozen=True)
class AuditReport:
    """A bounded scan result and its category samples."""

    scanned_files: int
    target_games: int
    east_kyokus: int
    established_riichis: int
    samples_per_category: int
    categories: tuple[CategorySamples, ...]

    def samples_for(self, category: str) -> tuple[RiichiWaitAuditSample, ...]:
        """Return samples for one known category."""
        for category_samples in self.categories:
            if category_samples.category == category:
                return category_samples.samples
        raise KeyError(category)


def classify_audit_categories(record: RiichiWaitRecord) -> tuple[str, ...]:
    """Return audit categories matched by one record in canonical order."""
    shapes = {detail.wait_shape for detail in record.wait_details}
    structural_tiles = (
        record.riichi_declaration_tile,
        *record.concealed_tiles_after_discard,
        *(tile for meld in record.fixed_melds for tile in meld.tiles),
    )
    matches = {
        "pure_ryanmen": record.is_pure_ryanmen,
        "kanchan": "kanchan" in shapes,
        "penchan": "penchan" in shapes,
        "shanpon": "shanpon" in shapes,
        "tanki": "tanki" in shapes,
        "multiwait_3_plus": record.is_multiwait,
        "contains_ryanmen_not_pure": (
            record.contains_ryanmen and not record.is_pure_ryanmen
        ),
        "red_five": any(tile in RED_FIVE_NORMALIZATION for tile in structural_tiles),
        "ankan": bool(record.fixed_melds),
        "early_riichi_1_to_6": (
            record.riichi_discard_number <= EARLY_RIICHI_MAX_DISCARD
        ),
        "late_riichi_13_plus": (
            record.riichi_discard_number >= LATE_RIICHI_MIN_DISCARD
        ),
    }
    return tuple(category for category in CATEGORY_ORDER if matches[category])


def collect_audit_samples(
    paths: Iterable[str | Path],
    *,
    max_files: int = DEFAULT_MAX_FILES,
    samples_per_category: int = DEFAULT_SAMPLES_PER_CATEGORY,
    context_before: int = DEFAULT_CONTEXT_BEFORE,
    context_after: int = DEFAULT_CONTEXT_AFTER,
) -> AuditReport:
    """Read a bounded number of sorted logs and select deterministic samples."""
    _validate_limits(max_files, samples_per_category, context_before, context_after)
    ordered_paths = sorted(
        (Path(path) for path in paths),
        key=lambda path: (path.name, str(path)),
    )
    selected: dict[str, list[RiichiWaitAuditSample]] = {
        category: [] for category in CATEGORY_ORDER
    }
    scanned_files = 0
    target_games = 0
    east_kyoku_count = 0
    established_riichi_count = 0

    for path in ordered_paths[:max_files]:
        scanned_files += 1
        try:
            events = load_mjai(path)
            if not is_target_game(path, events):
                continue
            target_games += 1

            line_by_event_id = {
                id(event): line_number
                for line_number, event in enumerate(events, start=1)
            }
            east_kyokus = filter_east_kyokus(split_kyoku(events))
            east_kyoku_count += len(east_kyokus)
            for kyoku_events in east_kyokus:
                records = build_riichi_wait_records(
                    extract_established_riichis(kyoku_events)
                )
                established_riichi_count += len(records)
                for record in records:
                    categories = classify_audit_categories(record)
                    wanted = tuple(
                        category
                        for category in categories
                        if len(selected[category]) < samples_per_category
                    )
                    if not wanted:
                        continue
                    sample = _build_sample(
                        path,
                        kyoku_events,
                        line_by_event_id,
                        record,
                        categories,
                        context_before,
                        context_after,
                    )
                    for category in wanted:
                        selected[category].append(sample)
        except Exception as error:
            raise RuntimeError(f"failed to audit MJAI file: {path}") from error

        if all(
            len(selected[category]) >= samples_per_category
            for category in CATEGORY_ORDER
        ):
            break

    return AuditReport(
        scanned_files=scanned_files,
        target_games=target_games,
        east_kyokus=east_kyoku_count,
        established_riichis=established_riichi_count,
        samples_per_category=samples_per_category,
        categories=tuple(
            CategorySamples(category, tuple(selected[category]))
            for category in CATEGORY_ORDER
        ),
    )


def render_audit_report(report: AuditReport) -> str:
    """Render a deterministic, human-readable manual audit report."""
    lines = [
        "# Riichi wait record audit",
        "",
        f"scanned_files: {report.scanned_files}",
        f"target_games: {report.target_games}",
        f"east_kyokus: {report.east_kyokus}",
        f"established_riichis: {report.established_riichis}",
        f"samples_per_category: {report.samples_per_category}",
        f"early_riichi: discard_number <= {EARLY_RIICHI_MAX_DISCARD}",
        f"late_riichi: discard_number >= {LATE_RIICHI_MIN_DISCARD}",
        "",
        "## Category counts",
        "",
    ]
    for category_samples in report.categories:
        label = CATEGORY_LABELS[category_samples.category]
        lines.append(
            f"- {category_samples.category} ({label}): {len(category_samples.samples)}"
        )

    for category_samples in report.categories:
        category = category_samples.category
        lines.extend(("", f"## {category} ({CATEGORY_LABELS[category]})", ""))
        if not category_samples.samples:
            lines.append("samples: 0")
            continue
        for sample_number, sample in enumerate(category_samples.samples, start=1):
            _append_sample(lines, sample_number, sample)

    return "\n".join(lines) + "\n"


def _build_sample(
    path: Path,
    kyoku_events: list[dict[str, Any]],
    line_by_event_id: dict[int, int],
    record: RiichiWaitRecord,
    categories: tuple[str, ...],
    context_before: int,
    context_after: int,
) -> RiichiWaitAuditSample:
    start = kyoku_events[0]
    start_line = _source_line(kyoku_events, 0, line_by_event_id)
    reach = _audit_event(
        kyoku_events,
        record.reach_event_index,
        line_by_event_id,
    )
    declaration = _audit_event(
        kyoku_events,
        record.declaration_dahai_event_index,
        line_by_event_id,
    )
    accepted = _audit_event(
        kyoku_events,
        record.reach_accepted_event_index,
        line_by_event_id,
    )
    context_start = max(0, record.reach_event_index - context_before)
    context_stop = min(
        len(kyoku_events),
        record.reach_accepted_event_index + context_after + 1,
    )
    return RiichiWaitAuditSample(
        source_path=str(path.resolve()),
        filename=path.name,
        start_kyoku_line=start_line,
        bakaze=start["bakaze"],
        kyoku=start["kyoku"],
        honba=start["honba"],
        kyotaku=start["kyotaku"],
        oya=start["oya"],
        is_dealer=record.actor == start["oya"],
        matched_categories=categories,
        record=record,
        reach_event=reach,
        declaration_dahai_event=declaration,
        reach_accepted_event=accepted,
        actor_discards=tuple(
            AuditDiscard(
                game_line_number=_source_line(
                    kyoku_events,
                    discard.event_index,
                    line_by_event_id,
                ),
                discard=discard,
            )
            for discard in record.actor_discards_before_riichi
        ),
        context_events=tuple(
            _audit_event(kyoku_events, event_index, line_by_event_id)
            for event_index in range(context_start, context_stop)
        ),
    )


def _audit_event(
    kyoku_events: list[dict[str, Any]],
    event_index: int,
    line_by_event_id: dict[int, int],
) -> AuditEvent:
    event = kyoku_events[event_index]
    return AuditEvent(
        kyoku_event_index=event_index,
        game_line_number=_source_line(kyoku_events, event_index, line_by_event_id),
        event_json=json.dumps(event, ensure_ascii=False, separators=(",", ":")),
    )


def _source_line(
    kyoku_events: list[dict[str, Any]],
    event_index: int,
    line_by_event_id: dict[int, int],
) -> int:
    try:
        event = kyoku_events[event_index]
        return line_by_event_id[id(event)]
    except (IndexError, KeyError) as error:
        raise ValueError(
            f"kyoku event index {event_index} has no physical source line"
        ) from error


def _append_sample(
    lines: list[str],
    sample_number: int,
    sample: RiichiWaitAuditSample,
) -> None:
    record = sample.record
    role = "dealer" if sample.is_dealer else "child"
    lines.extend(
        (
            f"### Sample {sample_number}",
            "",
            f"source_path: {sample.source_path}",
            f"filename: {sample.filename}",
            f"start_kyoku_line: {sample.start_kyoku_line}",
            (
                "kyoku: "
                f"bakaze={sample.bakaze} kyoku={sample.kyoku} "
                f"honba={sample.honba} kyotaku={sample.kyotaku} oya={sample.oya}"
            ),
            f"actor: {record.actor} ({role})",
            f"matched_categories: {', '.join(sample.matched_categories)}",
            f"riichi_discard_number: {record.riichi_discard_number}",
            f"riichi_declaration_tile: {record.riichi_declaration_tile}",
            (f"riichi_declaration_tile_kind: {record.riichi_declaration_tile_kind}"),
            _format_named_event("reach_event", sample.reach_event),
            _format_named_event(
                "declaration_dahai_event",
                sample.declaration_dahai_event,
            ),
            _format_named_event("reach_accepted_event", sample.reach_accepted_event),
            (
                "concealed_tiles_after_discard: "
                f"{_json(record.concealed_tiles_after_discard)}"
            ),
            f"fixed_melds: {_json(record.fixed_melds)}",
            "actor_discards_before_riichi:",
        )
    )
    for audit_discard in sample.actor_discards:
        lines.append(
            f"  game_line={audit_discard.game_line_number} "
            f"{_json(audit_discard.discard)}"
        )
    lines.extend(
        (
            f"wait_tiles: {_json(record.wait_tiles)}",
            f"wait_tile_count: {record.wait_tile_count}",
            f"wait_details: {_json(record.wait_details)}",
            f"wait_shapes: {_json(record.wait_shapes)}",
            f"contains_ryanmen: {record.contains_ryanmen}",
            f"is_pure_ryanmen: {record.is_pure_ryanmen}",
            f"is_multiwait: {record.is_multiwait}",
            "mjai_event_context:",
        )
    )
    for event in sample.context_events:
        lines.append(
            f"  game_line={event.game_line_number} "
            f"kyoku_event_index={event.kyoku_event_index} {event.event_json}"
        )
    lines.append("")


def _format_named_event(name: str, event: AuditEvent) -> str:
    return (
        f"{name}: game_line={event.game_line_number} "
        f"kyoku_event_index={event.kyoku_event_index} {event.event_json}"
    )


def _json(value: object) -> str:
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    elif isinstance(value, tuple):
        value = [
            asdict(item) if hasattr(item, "__dataclass_fields__") else item
            for item in value
        ]
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _validate_limits(
    max_files: int,
    samples_per_category: int,
    context_before: int,
    context_after: int,
) -> None:
    if type(max_files) is not int or max_files < 1:
        raise ValueError("max_files must be a positive integer")
    if type(samples_per_category) is not int or not 1 <= samples_per_category <= 3:
        raise ValueError("samples_per_category must be an integer from 1 to 3")
    if type(context_before) is not int or context_before < 0:
        raise ValueError("context_before must be a non-negative integer")
    if type(context_after) is not int or context_after < 0:
        raise ValueError("context_after must be a non-negative integer")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse bounded 2025 audit options."""
    parser = argparse.ArgumentParser(
        description="Print deterministic audit samples from a few 2025 MJAI logs."
    )
    parser.add_argument("--year", type=int, choices=(YEAR,), default=YEAR)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--max-files",
        type=_positive_int,
        default=DEFAULT_MAX_FILES,
        help="maximum number of sorted MJAI files to read (default: 100)",
    )
    parser.add_argument(
        "--samples-per-category",
        type=int,
        choices=(1, 2, 3),
        default=DEFAULT_SAMPLES_PER_CATEGORY,
    )
    parser.add_argument(
        "--context-before",
        type=_non_negative_int,
        default=DEFAULT_CONTEXT_BEFORE,
    )
    parser.add_argument(
        "--context-after",
        type=_non_negative_int,
        default=DEFAULT_CONTEXT_AFTER,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Collect and print a bounded 2025 audit report without saving it."""
    args = parse_args(argv)
    data_directory = args.raw_root / str(args.year)
    if not data_directory.is_dir():
        raise FileNotFoundError(f"raw data directory does not exist: {data_directory}")

    paths = sorted(data_directory.glob("*.mjson"), key=lambda path: path.name)
    if not paths:
        raise FileNotFoundError(f"no MJAI files found: {data_directory}")
    report = collect_audit_samples(
        paths,
        max_files=args.max_files,
        samples_per_category=args.samples_per_category,
        context_before=args.context_before,
        context_after=args.context_after,
    )
    print(render_audit_report(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
