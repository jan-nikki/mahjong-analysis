"""Validate the 2025 dealer double-riichi aggregate independently."""

import json
from collections import Counter
from dataclasses import asdict, dataclass
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


YEAR = 2025
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / str(YEAR)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "validation" / "issue15"
PROGRESS_INTERVAL = 10_000
AUDIT_SAMPLES_PER_CATEGORY = 10

EXPECTED_2025 = {
    "target_games": 178_887,
    "east_kyokus": 1_028_072,
    "dealer_double_riichi": 702,
    "dealer_win": 507,
    "other_win": 87,
    "draw": 108,
}


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
    classified = sum(
        results[result] for result in ("dealer_win", "other_win", "draw")
    )
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


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
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


def main() -> int:
    """Run both implementations over all 2025 files and compare their output."""
    paths = sorted(DATA_DIR.glob("*.mjson"), key=lambda path: path.name)
    reference_target_games = 0
    reference_east_kyokus = 0
    reference_candidates: list[CandidateRound] = []
    existing_target_games = 0
    existing_east_kyokus = 0
    existing_candidates: list[CandidateRound] = []
    file_count_mismatches: list[dict[str, Any]] = []
    audit_counts: Counter[str] = Counter()
    audit_samples: dict[str, list[ReachAuditRecord]] = {}

    for index, path in enumerate(paths, 1):
        try:
            reference = analyze_reference_mjai(path)
            existing = _analyze_with_existing_implementation(path)
        except Exception as error:
            raise RuntimeError(f"validation failed while reading {path}") from error

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

        if index % PROGRESS_INTERVAL == 0:
            print(f"processed: {index:,} files", flush=True)

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
        tuple(reference_candidates), tuple(existing_candidates)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        OUTPUT_DIR / "reference_candidates.jsonl",
        [asdict(record) for record in reference_candidates],
    )
    _write_jsonl(
        OUTPUT_DIR / "existing_candidates.jsonl",
        [asdict(record) for record in existing_candidates],
    )
    _write_jsonl(
        OUTPUT_DIR / "reach_audit_samples.jsonl",
        [
            {"category": category, **asdict(record)}
            for category in sorted(audit_samples)
            for record in audit_samples[category]
        ],
    )

    actual_reference = asdict(reference_stats)
    actual_existing = asdict(existing_stats)
    known_value_mismatches = {
        key: {
            "expected": expected,
            "reference": actual_reference[key],
            "existing": actual_existing[key],
        }
        for key, expected in EXPECTED_2025.items()
        if actual_reference[key] != expected or actual_existing[key] != expected
    }
    differences = {
        "file_count_mismatches": file_count_mismatches,
        "candidate_comparison": _comparison_details(comparison),
        "known_value_mismatches": known_value_mismatches,
    }
    with (OUTPUT_DIR / "comparison.json").open("w", encoding="utf-8") as file:
        json.dump(differences, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")

    _print_stats("reference", reference_stats)
    _print_stats("existing", existing_stats)
    print("audit_counts:")
    for category in sorted(audit_counts):
        print(f"  {category}: {audit_counts[category]}")

    counts_match = reference_stats == existing_stats
    if (
        file_count_mismatches
        or not comparison.matches
        or not counts_match
        or known_value_mismatches
    ):
        print(f"validation mismatch; details: {OUTPUT_DIR / 'comparison.json'}")
        return 1

    print("validation passed: candidate keys, results, and counts all match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
