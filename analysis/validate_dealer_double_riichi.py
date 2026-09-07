"""Validate a yearly dealer double-riichi aggregate independently."""

import argparse
import json
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from typing import Any

from mahjong_analysis.mjai import (
    classify_dealer_double_riichi_result,
    filter_east_kyokus,
    is_dealer_double_riichi,
    is_target_game,
    load_mjai,
    split_kyoku,
)
from mahjong_analysis.reference_validation import (
    CandidateComparison,
    CandidateRound,
    ReachAuditRecord,
    analyze_reference_mjai,
    compare_candidate_rounds,
)
from mahjong_analysis.yearly_aggregation import (
    YearlyDealerDoubleRiichiResult,
    load_known_results,
)


FIRST_YEAR = 2009
LAST_YEAR = 2025
SUPPORTED_YEARS = tuple(range(FIRST_YEAR, LAST_YEAR + 1))
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "validation" / "dealer-double-riichi"
DEFAULT_KNOWN_RESULTS = (
    PROJECT_ROOT
    / "data"
    / "validation"
    / "dealer-double-riichi-v2.0.0-known-results.json"
)
PROGRESS_INTERVAL = 10_000
AUDIT_SAMPLES_PER_CATEGORY = 10
_COUNT_FIELDS = (
    "target_games",
    "east_kyokus",
    "dealer_double_riichi",
    "dealer_win",
    "other_win",
    "draw",
)

ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class ValidationStats:
    """Counts produced by one side of the validation."""

    target_games: int
    east_kyokus: int
    dealer_double_riichi: int
    dealer_win: int
    other_win: int
    draw: int

    @property
    def win_rate(self) -> float | None:
        """Return the dealer win rate, or None when there are no candidates."""
        if self.dealer_double_riichi == 0:
            return None
        return self.dealer_win / self.dealer_double_riichi


@dataclass(frozen=True)
class _ExistingGameResult:
    target_game: bool
    east_kyokus: int
    candidates: tuple[CandidateRound, ...]


@dataclass(frozen=True)
class ValidationReport:
    """Complete comparison result for one selected year."""

    year: int
    scanned_files: int
    reference_stats: ValidationStats
    existing_stats: ValidationStats
    reference_candidates: tuple[CandidateRound, ...]
    existing_candidates: tuple[CandidateRound, ...]
    candidate_comparison: CandidateComparison
    file_count_mismatches: tuple[dict[str, Any], ...]
    audit_counts: dict[str, int]
    audit_samples: dict[str, tuple[ReachAuditRecord, ...]]
    known_result_checked: bool
    known_value_mismatches: dict[str, dict[str, int]]

    @property
    def passed(self) -> bool:
        """Return whether all production/reference and known checks passed."""
        return (
            not self.file_count_mismatches
            and self.candidate_comparison.matches
            and self.reference_stats == self.existing_stats
            and not self.known_value_mismatches
        )


def _analyze_with_existing_implementation(path: Path) -> _ExistingGameResult:
    events = load_mjai(path)

    # split_kyoku() keeps the original event dicts.  Make that relationship
    # explicit so physical line numbers do not depend on value equality.
    line_by_event_id = {
        id(event): line_number for line_number, event in enumerate(events, 1)
    }

    if not is_target_game(path, events):
        return _ExistingGameResult(False, 0, ())

    east_kyokus = filter_east_kyokus(split_kyoku(events))
    candidates: list[CandidateRound] = []
    for kyoku in east_kyokus:
        if not is_dealer_double_riichi(kyoku):
            continue

        start = kyoku[0]
        dealer = start["oya"]
        reach = next(
            event
            for event in kyoku
            if event["type"] == "reach" and event["actor"] == dealer
        )
        try:
            start_line = line_by_event_id[id(start)]
            reach_line = line_by_event_id[id(reach)]
        except KeyError as error:
            raise ValueError(
                f"split event identity was not found in source events: {path}"
            ) from error

        candidates.append(
            CandidateRound(
                filename=path.name,
                start_kyoku_line=start_line,
                reach_line=reach_line,
                bakaze=start["bakaze"],
                kyoku=start["kyoku"],
                honba=start["honba"],
                oya=dealer,
                result=classify_dealer_double_riichi_result(kyoku),
            )
        )

    return _ExistingGameResult(True, len(east_kyokus), tuple(candidates))


def _summarize(
    target_games: int,
    east_kyokus: int,
    candidates: list[CandidateRound],
) -> ValidationStats:
    results = Counter(candidate.result for candidate in candidates)
    classified = sum(results[result] for result in ("dealer_win", "other_win", "draw"))
    if classified != len(candidates):
        raise RuntimeError("candidate result counts are inconsistent")
    return ValidationStats(
        target_games=target_games,
        east_kyokus=east_kyokus,
        dealer_double_riichi=len(candidates),
        dealer_win=results["dealer_win"],
        other_win=results["other_win"],
        draw=results["draw"],
    )


def _known_value_mismatches(
    scanned_files: int,
    reference: ValidationStats,
    existing: ValidationStats,
    known_result: YearlyDealerDoubleRiichiResult | None,
) -> dict[str, dict[str, int]]:
    if known_result is None:
        return {}

    reference_values = {
        "scanned_files": scanned_files,
        **asdict(reference),
    }
    existing_values = {
        "scanned_files": scanned_files,
        **asdict(existing),
    }
    return {
        field: {
            "expected": getattr(known_result, field),
            "reference": reference_values[field],
            "existing": existing_values[field],
        }
        for field in ("scanned_files", *_COUNT_FIELDS)
        if reference_values[field] != getattr(known_result, field)
        or existing_values[field] != getattr(known_result, field)
    }


def _audit_category(record: ReachAuditRecord) -> str | None:
    if record.established_dealer_double_riichi:
        return "established"
    if record.reach_actor != record.oya:
        if record.actor_prior_dahai_count == 0:
            return "child_first_reach"
        return None
    if record.actor_prior_dahai_count > 0:
        return "dealer_later_reach"
    if record.ankan_before_reach:
        return "dealer_after_ankan"
    if record.event_after_dahai in {"hora", "ryukyoku"}:
        return "dealer_first_reach_unaccepted"
    return None


def validate_paths(
    year: int,
    paths: Iterable[str | PathLike[str]],
    *,
    known_result: YearlyDealerDoubleRiichiResult | None = None,
    progress_interval: int = PROGRESS_INTERVAL,
    progress_callback: ProgressCallback | None = None,
) -> ValidationReport:
    """Run production and independent analysis over one year's paths."""
    if year not in SUPPORTED_YEARS:
        raise ValueError(f"unsupported year: {year}")
    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive")
    if known_result is not None and known_result.year != year:
        raise ValueError(f"known result year {known_result.year} does not match {year}")

    path_values = tuple(sorted(map(Path, paths), key=lambda path: path.name))
    reference_target_games = 0
    reference_east_kyokus = 0
    reference_candidates: list[CandidateRound] = []
    existing_target_games = 0
    existing_east_kyokus = 0
    existing_candidates: list[CandidateRound] = []
    file_count_mismatches: list[dict[str, Any]] = []
    audit_counts: Counter[str] = Counter()
    audit_samples: dict[str, list[ReachAuditRecord]] = {}

    for index, path in enumerate(path_values, 1):
        try:
            reference = analyze_reference_mjai(path)
        except Exception as error:
            raise RuntimeError(
                f"reference validation failed for year {year}, file {path.name}"
            ) from error
        try:
            existing = _analyze_with_existing_implementation(path)
        except Exception as error:
            raise RuntimeError(
                f"production validation failed for year {year}, file {path.name}"
            ) from error

        reference_target_games += int(reference.target_game)
        reference_east_kyokus += reference.east_kyokus
        reference_candidates.extend(reference.candidates)
        existing_target_games += int(existing.target_game)
        existing_east_kyokus += existing.east_kyokus
        existing_candidates.extend(existing.candidates)

        if (
            reference.target_game != existing.target_game
            or reference.east_kyokus != existing.east_kyokus
        ):
            file_count_mismatches.append(
                {
                    "filename": path.name,
                    "reference_target_game": reference.target_game,
                    "existing_target_game": existing.target_game,
                    "reference_east_kyokus": reference.east_kyokus,
                    "existing_east_kyokus": existing.east_kyokus,
                }
            )

        for record in reference.reach_audits:
            category = _audit_category(record)
            if category is None:
                continue
            audit_counts[category] += 1
            samples = audit_samples.setdefault(category, [])
            if len(samples) < AUDIT_SAMPLES_PER_CATEGORY:
                samples.append(record)

        if progress_callback is not None and index % progress_interval == 0:
            progress_callback(year, index)

    reference_stats = _summarize(
        reference_target_games,
        reference_east_kyokus,
        reference_candidates,
    )
    existing_stats = _summarize(
        existing_target_games,
        existing_east_kyokus,
        existing_candidates,
    )
    comparison = compare_candidate_rounds(
        tuple(reference_candidates),
        tuple(existing_candidates),
    )
    return ValidationReport(
        year=year,
        scanned_files=len(path_values),
        reference_stats=reference_stats,
        existing_stats=existing_stats,
        reference_candidates=tuple(reference_candidates),
        existing_candidates=tuple(existing_candidates),
        candidate_comparison=comparison,
        file_count_mismatches=tuple(file_count_mismatches),
        audit_counts=dict(sorted(audit_counts.items())),
        audit_samples={
            category: tuple(audit_samples[category])
            for category in sorted(audit_samples)
        },
        known_result_checked=known_result is not None,
        known_value_mismatches=_known_value_mismatches(
            len(path_values),
            reference_stats,
            existing_stats,
            known_result,
        ),
    )


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file, ensure_ascii=False, sort_keys=True)
            file.write("\n")


def _comparison_details(comparison: CandidateComparison) -> dict[str, Any]:
    def pairs(
        values: tuple[tuple[CandidateRound, CandidateRound], ...],
    ) -> list[dict[str, Any]]:
        return [
            {"reference": asdict(reference), "existing": asdict(existing)}
            for reference, existing in values
        ]

    return {
        "reference_only": [asdict(record) for record in comparison.reference_only],
        "existing_only": [asdict(record) for record in comparison.existing_only],
        "metadata_mismatches": pairs(comparison.metadata_mismatches),
        "result_mismatches": pairs(comparison.result_mismatches),
    }


def _print_stats(label: str, stats: ValidationStats) -> None:
    win_rate = "N/A" if stats.win_rate is None else f"{stats.win_rate:.2%}"
    print(f"{label}:")
    for field, value in asdict(stats).items():
        print(f"  {field}: {value}")
    print(f"  win_rate: {win_rate}")


def write_validation_outputs(
    report: ValidationReport,
    output_root: str | PathLike[str],
) -> Path:
    """Write traceable candidates, audits, and comparison for one year."""
    output_directory = Path(output_root) / str(report.year)
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        output_directory / "reference_candidates.jsonl",
        (asdict(record) for record in report.reference_candidates),
    )
    _write_jsonl(
        output_directory / "existing_candidates.jsonl",
        (asdict(record) for record in report.existing_candidates),
    )
    _write_jsonl(
        output_directory / "reach_audit_samples.jsonl",
        (
            {"category": category, **asdict(record)}
            for category in sorted(report.audit_samples)
            for record in report.audit_samples[category]
        ),
    )

    differences = {
        "year": report.year,
        "scanned_files": report.scanned_files,
        "reference_stats": asdict(report.reference_stats),
        "existing_stats": asdict(report.existing_stats),
        "counts_match": report.reference_stats == report.existing_stats,
        "file_count_mismatches": list(report.file_count_mismatches),
        "candidate_comparison": _comparison_details(report.candidate_comparison),
        "known_result_checked": report.known_result_checked,
        "known_value_mismatches": report.known_value_mismatches,
        "audit_counts": report.audit_counts,
        "passed": report.passed,
    }
    with (output_directory / "comparison.json").open("w", encoding="utf-8") as file:
        json.dump(differences, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    return output_directory


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse a single supported validation year and path overrides."""
    parser = argparse.ArgumentParser(
        description="Independently validate dealer double-riichi results."
    )
    parser.add_argument(
        "--year",
        type=int,
        choices=SUPPORTED_YEARS,
        required=True,
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--known-results",
        type=Path,
        default=DEFAULT_KNOWN_RESULTS,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run both implementations for one year and compare their output."""
    args = parse_args(argv)
    data_directory = args.raw_root / str(args.year)
    if not data_directory.is_dir():
        raise FileNotFoundError(
            f"raw directory does not exist for year {args.year}: {data_directory}"
        )
    try:
        paths = sorted(
            data_directory.glob("*.mjson"),
            key=lambda path: path.name,
        )
    except OSError as error:
        raise RuntimeError(
            f"failed to enumerate raw files for year {args.year}"
        ) from error

    known_results = load_known_results(args.known_results)

    def report_progress(year: int, processed: int) -> None:
        print(f"{year}: processed {processed:,} files", flush=True)

    report = validate_paths(
        args.year,
        paths,
        known_result=known_results.get(args.year),
        progress_interval=PROGRESS_INTERVAL,
        progress_callback=report_progress,
    )
    output_directory = write_validation_outputs(report, args.output_root)

    _print_stats("reference", report.reference_stats)
    _print_stats("existing", report.existing_stats)
    print("audit_counts:")
    for category, count in report.audit_counts.items():
        print(f"  {category}: {count}")

    if not report.passed:
        print(f"validation mismatch; details: {output_directory / 'comparison.json'}")
        return 1

    print(
        f"validation passed for {args.year}: "
        "candidate keys, results, and counts all match"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
