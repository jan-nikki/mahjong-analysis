"""Analyze opening yakuhai discards and pair-formation races in raw MJAI."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from mahjong_analysis.mjai import (
    filter_east_kyokus,
    is_target_game,
    load_mjai,
    split_kyoku,
)
from mahjong_analysis.yakuhai_dash import (
    SingletonObservation,
    YakuhaiDashAccumulator,
    analyze_yakuhai_dash_kyoku,
    deal_to_dict,
    singleton_to_dict,
    theoretical_baselines,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "yakuhai-dash" / "summary-v3.json"
LEGACY_DEFAULT_OUTPUT = DEFAULT_OUTPUT.with_name("summary-v2.json")
PROGRESS_LOG_INTERVAL = 1000
PROGRESS_SECONDS_INTERVAL = 30.0
ANALYSIS_YEARS = tuple(range(2020, 2026))
AUDIT_CATEGORIES = (
    "initial_yakuhai_pair",
    "initial_no_yakuhai_pair",
    "strict_dash",
    "first_discard_without_dash",
    "no_first_discard",
    "self_pair_draw",
    "no_self_pair_draw",
    "self_pair_pon_shape",
    "self_pair_without_pon_shape",
    "opponent_ready_at_deal",
    "opponent_not_ready_at_deal",
    "opponent_became_ready_after_deal",
    "opponent_pon_capable_at_discard",
    "opponent_not_pon_capable_at_discard",
    "only_riichi_pair_holders_at_discard",
    "actual_pon",
    "post_discard_self_draw",
    "post_discard_opponent_pair",
    "own_only_seat_wind",
)


@dataclass(frozen=True)
class YearTask:
    year: int
    data_root: str
    max_logs: int | None
    audit_samples_per_category: int
    progress: bool = False


@dataclass
class YearResult:
    year: int
    data_root: str
    max_logs: int | None
    audit_samples_per_category: int
    available_logs: int
    scanned_logs: int
    target_logs: int
    accumulator: YakuhaiDashAccumulator
    audit_samples: dict[str, list[dict[str, Any]]]
    audit_category_counts: dict[str, int]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure initial yakuhai pairs, early singleton discards, self-pair "
            "draws, and opponent pon capability."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--years",
        type=_analysis_year,
        nargs="+",
        default=list(ANALYSIS_YEARS),
    )
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument("--max-logs-per-year", type=_positive_int)
    parser.add_argument(
        "--audit-samples-per-category", type=_nonnegative_int, default=3
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="preserve each completed annual result as a separate JSON file",
    )
    return parser.parse_args(argv)


def scan_year(task: YearTask) -> YearResult:
    root = Path(task.data_root).resolve()
    year_directory = root / str(task.year)
    if not year_directory.is_dir():
        raise FileNotFoundError(f"year directory does not exist: {year_directory}")
    paths = sorted(year_directory.glob("*.mjson"))
    available_logs = len(paths)
    if task.max_logs is not None:
        paths = paths[: task.max_logs]
    if not paths:
        raise FileNotFoundError(f"no .mjson logs found: {year_directory}")

    accumulator = YakuhaiDashAccumulator()
    audits: dict[str, list[dict[str, Any]]] = {
        category: [] for category in AUDIT_CATEGORIES
    }
    audit_category_counts = {category: 0 for category in AUDIT_CATEGORIES}
    target_logs = 0
    last_progress_at = time.monotonic()
    if task.progress:
        _report_progress(task.year, "start", 0, len(paths), 0, 0)

    for scanned_count, path in enumerate(paths, start=1):
        events = load_mjai(path)
        if not is_target_game(path, events):
            if task.progress:
                now = time.monotonic()
                if (
                    scanned_count % PROGRESS_LOG_INTERVAL == 0
                    or now - last_progress_at >= PROGRESS_SECONDS_INTERVAL
                ) and scanned_count < len(paths):
                    _report_progress(
                        task.year,
                        "progress",
                        scanned_count,
                        len(paths),
                        target_logs,
                        accumulator.east_kyokus,
                    )
                    last_progress_at = now
            continue
        target_logs += 1
        source_line_by_event = {
            id(event): line_number for line_number, event in enumerate(events, start=1)
        }
        for east_ordinal, kyoku in enumerate(
            filter_east_kyokus(split_kyoku(events)), start=1
        ):
            start = kyoku[0]
            start_line = source_line_by_event[id(start)]
            try:
                result = analyze_yakuhai_dash_kyoku(kyoku)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{path}: east kyoku ordinal {east_ordinal}: {error}"
                ) from error
            accumulator.add(result)
            context = {
                "relative_source_path": path.relative_to(root).as_posix(),
                "start_kyoku_line": start_line,
                "east_kyoku_ordinal_in_game": east_ordinal,
                "bakaze": start.get("bakaze"),
                "kyoku": start.get("kyoku"),
                "honba": start.get("honba"),
                "oya": start.get("oya"),
            }
            for deal in result.deals:
                category = (
                    "initial_yakuhai_pair"
                    if deal.has_yakuhai_pair
                    else "initial_no_yakuhai_pair"
                )
                _append_audit(
                    audits,
                    audit_category_counts,
                    category,
                    {**context, **deal_to_dict(deal)},
                    task.audit_samples_per_category,
                )
            for opening in result.openings:
                if opening.first_discard_event_index is None:
                    _append_audit(
                        audits,
                        audit_category_counts,
                        "no_first_discard",
                        {
                            **context,
                            "actor": opening.actor,
                            "role": opening.role,
                            "initial_hand": list(result.deals[opening.actor].initial_hand),
                            "shanten": opening.shanten,
                            "dora_han": opening.dora_han,
                        },
                        task.audit_samples_per_category,
                    )
                elif not opening.strict_dash:
                    _append_audit(
                        audits,
                        audit_category_counts,
                        "first_discard_without_dash",
                        {
                            **context,
                            "actor": opening.actor,
                            "role": opening.role,
                            "initial_hand": list(result.deals[opening.actor].initial_hand),
                            "first_discard_line": _source_line(
                                start_line, opening.first_discard_event_index
                            ),
                            "shanten": opening.shanten,
                            "dora_han": opening.dora_han,
                        },
                        task.audit_samples_per_category,
                    )
            for observation in result.singletons:
                sample = {
                    **context,
                    **singleton_to_dict(observation),
                    "first_opponent_pair_line": _source_line(
                        start_line, observation.first_opponent_pair_event_index
                    ),
                    "self_pair_line": _source_line(
                        start_line, observation.self_pair_event_index
                    ),
                    "self_pair_followup_discard_line": _source_line(
                        start_line,
                        observation.self_pair_followup_discard_event_index,
                    ),
                    "singleton_discard_line": _source_line(
                        start_line, observation.singleton_discard_event_index
                    ),
                    "discard_pon_line": _source_line(
                        start_line, observation.discard_pon_event_index
                    ),
                    "focal_pon_line": _source_line(
                        start_line, observation.focal_pon_event_index
                    ),
                    "post_discard_self_draw_line": _source_line(
                        start_line, observation.post_discard_self_draw_event_index
                    ),
                    "post_discard_opponent_pair_line": _source_line(
                        start_line,
                        observation.post_discard_opponent_pair_event_index,
                    ),
                }
                for category in _audit_categories(observation):
                    _append_audit(
                        audits,
                        audit_category_counts,
                        category,
                        sample,
                        task.audit_samples_per_category,
                    )
        if task.progress:
            now = time.monotonic()
            if (
                scanned_count % PROGRESS_LOG_INTERVAL == 0
                or now - last_progress_at >= PROGRESS_SECONDS_INTERVAL
            ) and scanned_count < len(paths):
                _report_progress(
                    task.year,
                    "progress",
                    scanned_count,
                    len(paths),
                    target_logs,
                    accumulator.east_kyokus,
                )
                last_progress_at = now

    if task.progress:
        _report_progress(
            task.year,
            "complete",
            len(paths),
            len(paths),
            target_logs,
            accumulator.east_kyokus,
        )

    return YearResult(
        year=task.year,
        data_root=str(root),
        max_logs=task.max_logs,
        audit_samples_per_category=task.audit_samples_per_category,
        available_logs=available_logs,
        scanned_logs=len(paths),
        target_logs=target_logs,
        accumulator=accumulator,
        audit_samples=audits,
        audit_category_counts=audit_category_counts,
    )


def build_document(results: tuple[YearResult, ...]) -> dict[str, Any]:
    if not results:
        raise ValueError("at least one annual result is required")
    ordered = tuple(sorted(results, key=lambda result: result.year))
    if len({result.year for result in ordered}) != len(ordered):
        raise ValueError("annual results contain duplicate years")
    for field_name in (
        "data_root",
        "max_logs",
        "audit_samples_per_category",
    ):
        if len({getattr(result, field_name) for result in ordered}) != 1:
            raise ValueError(f"annual results contain different {field_name} values")

    overall = YakuhaiDashAccumulator()
    audit_limit = ordered[0].audit_samples_per_category
    audits: dict[str, list[dict[str, Any]]] = {
        category: [] for category in AUDIT_CATEGORIES
    }
    for result in ordered:
        overall.merge(result.accumulator)
        for category in AUDIT_CATEGORIES:
            remaining = audit_limit - len(audits[category])
            if remaining > 0:
                audits[category].extend(result.audit_samples[category][:remaining])

    max_logs = ordered[0].max_logs
    return {
        "metadata": {
            "analysis_name": "yakuhai-dash-v3",
            "schema_version": 3,
            "years": [result.year for result in ordered],
            "execution": {
                "run_mode": "sample" if max_logs is not None else "full",
                "max_logs_per_year": max_logs,
                "data_root": ordered[0].data_root,
                "sampling_order": "filename_ascending_per_year",
                "audit_samples_per_category": audit_limit,
                "limit_applied_before_target_filter": True,
                "per_year_log_inventory": [
                    {
                        "year": result.year,
                        "available_logs": result.available_logs,
                        "scanned_logs": result.scanned_logs,
                    }
                    for result in ordered
                ],
            },
            "scope": {
                "game": "four-player Houou hanchan with red fives (00a9)",
                "rounds": "east rounds only",
                "initial_hand": "13 tiles in start_kyoku.tehais for every actor",
            },
            "definitions": {
                "yakuhai": "three dragons, round wind, and seat wind, deduplicated by tile kind",
                "initial_pair_or_more": "two through four copies in the 13-tile initial hand",
                "initial_singleton": "exactly one copy in the 13-tile initial hand",
                "strict_dash": "an initial singleton yakuhai discarded while still singleton as actor discard 1",
                "dash_by_2": "at least one such singleton discard by actor discard 2",
                "dash_by_3": "at least one such singleton discard by actor discard 3",
                "self_pair_draw": "the focal actor draws a second copy before singleton discard",
                "self_pair_retained_after_discard": "at least two concealed copies remain immediately after the first own discard following self_pair_draw; null if no such discard occurs",
                "self_pair_pon_shape_after_discard": "retained after that discard with neither declared nor accepted riichi; a shape proxy, not a guarantee of a legal or actual later pon; null if no such discard occurs",
                "self_pair_followup_draw_axis": "original pair-formation draw number; follow-up discard can occur after extra replacement draws from kan",
                "opponent_pon_capable": "at least one non-riichi opponent holds two or more copies immediately before discard; other situational call restrictions are not evaluated",
                "first_pair_race": "exclusive first arrival before the focal singleton is discarded or the round ends",
                "post_discard_events": "actual-log matching draws after the focal discard; not counterfactual results of holding the tile",
                "opening_shanten": "minimum standard, seven-pairs, or thirteen-orphans shanten on the 13-tile initial hand",
                "opening_dora_han": "initial visible dora copies plus red-five han; a red five can contribute twice",
                "public_copies_before_first_discard": "initial and later dora markers, discards, and exposed meld tiles before the actor's first discard",
                "complete_information": "opponent concealed hands are used for research classification",
                "causal_scope": "descriptive mechanism analysis; not a causal estimate of point or win-rate advantage",
            },
            "theoretical_baselines": theoretical_baselines(),
        },
        "input_counts": {
            "available_logs": sum(result.available_logs for result in ordered),
            "scanned_logs": sum(result.scanned_logs for result in ordered),
            "target_logs": sum(result.target_logs for result in ordered),
            "east_kyokus": overall.east_kyokus,
        },
        "overall": overall.to_dict(),
        "years": [
            {
                "year": result.year,
                "input_counts": {
                    "available_logs": result.available_logs,
                    "scanned_logs": result.scanned_logs,
                    "target_logs": result.target_logs,
                    "east_kyokus": result.accumulator.east_kyokus,
                },
                **result.accumulator.to_dict(),
            }
            for result in ordered
        ],
        "audit_samples": [
            {"category": category, "samples": audits[category]}
            for category in AUDIT_CATEGORIES
        ],
        "audit_samples_by_year": [
            {
                "year": result.year,
                "categories": [
                    {
                        "category": category,
                        "population_count": result.audit_category_counts[category],
                        "samples": result.audit_samples[category],
                    }
                    for category in AUDIT_CATEGORIES
                ],
            }
            for result in ordered
        ],
    }


def run_tasks(
    tasks: tuple[YearTask, ...],
    workers: int,
    on_result: Callable[[YearResult], None] | None = None,
) -> tuple[YearResult, ...]:
    if workers == 1:
        results: list[YearResult] = []
        for task in tasks:
            result = scan_year(task)
            if on_result is not None:
                on_result(result)
            results.append(result)
        return tuple(sorted(results, key=lambda item: item.year))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(scan_year, task): task.year for task in tasks}
        results = []
        first_error: Exception | None = None
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as error:
                if first_error is None:
                    first_error = error
                continue
            if on_result is not None:
                on_result(result)
            results.append(result)
        if first_error is not None:
            raise first_error
        return tuple(sorted(results, key=lambda item: item.year))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    years = tuple(sorted(set(args.years)))
    output = args.output or (
        DEFAULT_OUTPUT.with_name(f"summary-v3-sample-{args.max_logs_per_year}.json")
        if args.max_logs_per_year is not None
        else DEFAULT_OUTPUT
    )
    tasks = tuple(
        YearTask(
            year=year,
            data_root=str(args.data_root.resolve()),
            max_logs=args.max_logs_per_year,
            audit_samples_per_category=args.audit_samples_per_category,
            progress=args.progress,
        )
        for year in years
    )
    try:
        _validate_output_destination(output, sample=args.max_logs_per_year is not None)
        if args.checkpoint_dir is not None:
            checkpoint_paths = {
                year: args.checkpoint_dir / f"year-{year}-v3.json"
                for year in years
            }
            for checkpoint_path in checkpoint_paths.values():
                if checkpoint_path.resolve() == output.resolve():
                    raise ValueError("checkpoint destination is also the summary output")
                _validate_output_destination(checkpoint_path, sample=False)

            def save_year(result: YearResult) -> None:
                _write_json_atomic(
                    checkpoint_paths[result.year], build_document((result,))
                )

        else:
            save_year = None
        results = run_tasks(tasks, min(args.workers, len(tasks)), on_result=save_year)
        document = build_document(results)
        _write_json_atomic(output, document)
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "input_counts": document["input_counts"],
                "initial_pair_or_more": document["overall"]["initial_deals"]["all"][
                    "any_yakuhai_pair_or_more"
                ],
                "strict_dash": document["overall"]["opening_behavior"]["all"][
                    "strict_dash"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _source_line(start_line: int, event_index: int | None) -> int | None:
    return None if event_index is None else start_line + event_index


def _append_audit(
    audits: dict[str, list[dict[str, Any]]],
    counts: dict[str, int],
    category: str,
    sample: dict[str, Any],
    limit: int,
) -> None:
    counts[category] += 1
    if len(audits[category]) < limit:
        audits[category].append(sample)


def _audit_categories(observation: SingletonObservation) -> tuple[str, ...]:
    categories: list[str] = []
    if observation.singleton_discard_number == 1:
        categories.append("strict_dash")
    if observation.outcome == "self_pair_draw":
        categories.append("self_pair_draw")
        if observation.self_pair_pon_shape_after_discard is True:
            categories.append("self_pair_pon_shape")
        else:
            categories.append("self_pair_without_pon_shape")
    else:
        categories.append("no_self_pair_draw")
    if observation.opponent_ready_at_deal:
        categories.append("opponent_ready_at_deal")
    else:
        categories.append("opponent_not_ready_at_deal")
    if (
        observation.first_opponent_pair_event_index is not None
        and observation.first_opponent_pair_event_index > 0
    ):
        categories.append("opponent_became_ready_after_deal")
    if observation.opponent_pon_capable_at_discard:
        categories.append("opponent_pon_capable_at_discard")
    elif observation.opponent_pon_capable_at_discard is False:
        categories.append("opponent_not_pon_capable_at_discard")
    if (
        observation.opponent_pair_holder_actor_count
        and not observation.opponent_pon_capable_at_discard
    ):
        categories.append("only_riichi_pair_holders_at_discard")
    if observation.discard_was_ponned:
        categories.append("actual_pon")
    if observation.post_discard_self_draw_event_index is not None:
        categories.append("post_discard_self_draw")
    if observation.post_discard_opponent_pair_event_index is not None:
        categories.append("post_discard_opponent_pair")
    if observation.yakuhai_class == "own_seat_wind":
        categories.append("own_only_seat_wind")
    return tuple(categories)


def _validate_output_destination(path: Path, *, sample: bool) -> None:
    if sample and path.resolve() in {
        DEFAULT_OUTPUT.resolve(),
        LEGACY_DEFAULT_OUTPUT.resolve(),
    }:
        raise ValueError(f"sample output cannot use the full-run destination: {path}")
    if os.path.lexists(path):
        raise FileExistsError(f"output already exists: {path}")


def _write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                document,
                stream,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _analysis_year(value: str) -> int:
    year = int(value)
    if year not in ANALYSIS_YEARS:
        raise argparse.ArgumentTypeError("must be a year from 2020 through 2025")
    return year


def _report_progress(
    year: int,
    phase: str,
    scanned_logs: int,
    total_logs: int,
    target_logs: int,
    east_kyokus: int,
) -> None:
    print(
        f"{year} {phase}: scanned {scanned_logs:,}/{total_logs:,} logs, "
        f"{target_logs:,} target logs, {east_kyokus:,} east kyokus",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
