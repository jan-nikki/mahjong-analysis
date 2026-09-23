"""Decompose the dealer/nondealer winning-riichi score gap."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

from mahjong_analysis.dealer_child_riichi_points import (
    ANALYSIS_NAME as ARTICLE_ONE_ANALYSIS_NAME,
)
from mahjong_analysis.dealer_child_riichi_points import (
    SCHEMA_VERSION as ARTICLE_ONE_SCHEMA_VERSION,
)
from mahjong_analysis.dealer_child_riichi_points import (
    SUPPORTED_YEARS,
    RiichiPointRecord,
    normalize_years,
)
from mahjong_analysis.dealer_child_score_decomposition import (
    DecompositionInputMetadata,
    DecompositionWinRecord,
    ScoreDecompositionBuilder,
    decomposition_record_from_article_one,
    write_decomposition_outputs,
)
from mahjong_analysis.score_table import (
    AmbiguousScorePatternError,
    UnmatchedScorePatternError,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_SUMMARY = (
    PROJECT_ROOT / "outputs" / "dealer-child-riichi-points" / "summary-v1.json"
)
DEFAULT_INPUT_RECORDS = (
    PROJECT_ROOT / "outputs" / "dealer-child-riichi-points" / "records-v1.jsonl.gz"
)
DEFAULT_OUTPUT_JSON = (
    PROJECT_ROOT / "outputs" / "dealer-child-score-decomposition" / "summary-v1.json"
)
DEFAULT_OUTPUT_MARKDOWN = (
    PROJECT_ROOT / "outputs" / "dealer-child-score-decomposition" / "summary-v1.md"
)
DEFAULT_SUMMARY_LOGICAL_PATH = "outputs/dealer-child-riichi-points/summary-v1.json"
DEFAULT_RECORDS_LOGICAL_PATH = "outputs/dealer-child-riichi-points/records-v1.jsonl.gz"
EXPECTED_REPOSITORY = "NikkeTryHard/tenhou-to-mjai"
EXPECTED_RELEASE_TAG = "v2.0.0"
EXPECTED_FORMAT = "MJAI JSON Lines"
EXPECTED_RULE_CODE = "00a9"
PRIMARY_YEARS = tuple(range(2020, 2026))

Role = Literal["dealer", "nondealer"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose the dealer/nondealer winning-riichi score gap."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--years", nargs="+", type=int, choices=SUPPORTED_YEARS)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--input-summary", type=Path, default=DEFAULT_INPUT_SUMMARY)
    parser.add_argument("--input-records", type=Path, default=DEFAULT_INPUT_RECORDS)
    parser.add_argument(
        "--input-summary-logical-path",
        default=DEFAULT_SUMMARY_LOGICAL_PATH,
    )
    parser.add_argument(
        "--input-records-logical-path",
        default=DEFAULT_RECORDS_LOGICAL_PATH,
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    return parser.parse_args(argv)


def selected_years_from_args(args: argparse.Namespace) -> tuple[int, ...]:
    return normalize_years(SUPPORTED_YEARS if args.all else args.years)


def _load_and_validate_summary(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("cannot read Article 1 summary JSON") from error
    if not isinstance(document, dict):
        raise TypeError("Article 1 summary must be a JSON object")
    if document.get("analysis_name") != ARTICLE_ONE_ANALYSIS_NAME:
        raise ValueError("unexpected Article 1 analysis name")
    if (
        type(document.get("schema_version")) is not int
        or document.get("schema_version") != ARTICLE_ONE_SCHEMA_VERSION
    ):
        raise ValueError("unexpected Article 1 schema version")
    source = _object(document.get("source"), "source")
    if source.get("repository") != EXPECTED_REPOSITORY:
        raise ValueError("unexpected Article 1 source repository")
    if source.get("release_tag") != EXPECTED_RELEASE_TAG:
        raise ValueError("unexpected Article 1 source release")
    if source.get("format") != EXPECTED_FORMAT:
        raise ValueError("unexpected Article 1 source format")
    scope = _object(document.get("scope"), "scope")
    if scope.get("rule_code") != EXPECTED_RULE_CODE:
        raise ValueError("unexpected Article 1 scope rule_code")
    if scope.get("aka_flag") is not True:
        raise ValueError("unexpected Article 1 scope aka_flag")
    if scope.get("bakaze") != "E":
        raise ValueError("unexpected Article 1 scope bakaze")
    if scope.get("riichi_population") != "established":
        raise ValueError("unexpected Article 1 riichi population")
    _validate_year_array(
        scope.get("primary_years"),
        PRIMARY_YEARS,
        "scope.primary_years",
    )
    _validate_year_array(
        scope.get("selected_years"),
        SUPPORTED_YEARS,
        "scope.selected_years",
    )
    raw_years = document.get("years")
    if not isinstance(raw_years, list):
        raise TypeError("Article 1 summary years must be an array")
    summary_years = tuple(
        _integer(_object(value, f"years[{index}]").get("year"), f"years[{index}].year")
        for index, value in enumerate(raw_years)
    )
    if summary_years != SUPPORTED_YEARS:
        raise ValueError(
            "Article 1 summary years must be the unique ascending 2009-2025 sequence"
        )
    return document


def _article_one_record(document: object, line_number: int) -> RiichiPointRecord:
    if not isinstance(document, dict):
        raise TypeError(f"records line {line_number}: expected a JSON object")
    expected_fields = {value.name for value in fields(RiichiPointRecord)}
    actual_fields = set(document)
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        extra = sorted(actual_fields - expected_fields)
        raise ValueError(
            f"records line {line_number}: record fields differ from schema; "
            f"missing={missing}, extra={extra}"
        )
    integer_ranges: dict[str, tuple[int, int | None]] = {
        "year": (SUPPORTED_YEARS[0], SUPPORTED_YEARS[-1]),
        "kyoku_index": (0, None),
        "start_kyoku_line": (1, None),
        "kyoku": (1, 4),
        "honba": (0, None),
        "oya": (0, 3),
        "actor": (0, 3),
        "reach_event_index": (0, None),
        "reach_accepted_event_index": (0, None),
        "reach_line": (1, None),
        "reach_accepted_line": (1, None),
    }
    for name, (minimum, maximum) in integer_ranges.items():
        value = document[name]
        if (
            type(value) is not int
            or value < minimum
            or (maximum is not None and value > maximum)
        ):
            raise ValueError(
                f"records line {line_number}: {name} is outside its integer range"
            )
    if document["reach_accepted_event_index"] <= document["reach_event_index"]:
        raise ValueError(
            f"records line {line_number}: reach_accepted_event_index must follow reach"
        )
    if not (
        document["start_kyoku_line"]
        <= document["reach_line"]
        < document["reach_accepted_line"]
    ):
        raise ValueError(
            f"records line {line_number}: reach source lines are out of order"
        )
    source_path = document["source_path"]
    if (
        type(source_path) is not str
        or not source_path
        or "\\" in source_path
        or Path(source_path).is_absolute()
    ):
        raise ValueError(f"records line {line_number}: invalid source_path")
    if document["bakaze"] != "E":
        raise ValueError(f"records line {line_number}: bakaze must be E")
    if document["reach_type"] not in ("riichi", "double_riichi"):
        raise ValueError(f"records line {line_number}: invalid reach_type")
    outcome = document["outcome"]
    if outcome not in ("win", "other_win", "draw"):
        raise ValueError(f"records line {line_number}: invalid outcome")
    win_fields = (
        "win_method",
        "hora_event_index",
        "hora_line",
        "honba_awarded",
        "hand_points",
        "settlement_gain",
    )
    if outcome == "win":
        if document["win_method"] not in ("tsumo", "ron"):
            raise ValueError(f"records line {line_number}: invalid win_method")
        for name in ("hora_event_index", "hora_line"):
            value = document[name]
            minimum = 0 if name == "hora_event_index" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"records line {line_number}: invalid {name} for win")
        if document["hora_event_index"] <= document["reach_accepted_event_index"]:
            raise ValueError(
                f"records line {line_number}: hora_event_index must follow reach"
            )
        if document["hora_line"] <= document["reach_accepted_line"]:
            raise ValueError(f"records line {line_number}: hora_line must follow reach")
        if type(document["honba_awarded"]) is not bool:
            raise ValueError(
                f"records line {line_number}: honba_awarded must be boolean for win"
            )
        for name in ("hand_points", "settlement_gain"):
            value = document[name]
            if type(value) is not int or value <= 0 or value % 100:
                raise ValueError(
                    f"records line {line_number}: {name} must be a positive "
                    "multiple of 100 for win"
                )
        if document["settlement_gain"] < document["hand_points"]:
            raise ValueError(
                f"records line {line_number}: settlement_gain is below hand_points"
            )
    elif any(document[name] is not None for name in win_fields):
        raise ValueError(
            f"records line {line_number}: non-win record has win-only fields"
        )
    try:
        return RiichiPointRecord(**document)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"records line {line_number}: invalid Article 1 record"
        ) from error


@dataclass
class _GroupAuditCounts:
    riichis: int = 0
    wins: int = 0
    other_wins: int = 0
    draws: int = 0
    tsumo_wins: int = 0
    ron_wins: int = 0
    hand_points_sum: int = 0
    settlement_gain_sum: int = 0
    tsumo_hand_points_sum: int = 0
    ron_hand_points_sum: int = 0

    def add(self, record: RiichiPointRecord) -> None:
        self.riichis += 1
        if record.outcome == "win":
            self.wins += 1
            assert record.win_method is not None
            assert record.hand_points is not None
            assert record.settlement_gain is not None
            self.hand_points_sum += record.hand_points
            self.settlement_gain_sum += record.settlement_gain
            if record.win_method == "tsumo":
                self.tsumo_wins += 1
                self.tsumo_hand_points_sum += record.hand_points
            else:
                self.ron_wins += 1
                self.ron_hand_points_sum += record.hand_points
        elif record.outcome == "other_win":
            self.other_wins += 1
        else:
            self.draws += 1

    def merge(self, other: _GroupAuditCounts) -> None:
        for name in (
            "riichis",
            "wins",
            "other_wins",
            "draws",
            "tsumo_wins",
            "ron_wins",
            "hand_points_sum",
            "settlement_gain_sum",
            "tsumo_hand_points_sum",
            "ron_hand_points_sum",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))


def _new_role_counts() -> dict[Role, _GroupAuditCounts]:
    return {"dealer": _GroupAuditCounts(), "nondealer": _GroupAuditCounts()}


@dataclass
class _YearAuditCounts:
    all_riichi: dict[Role, _GroupAuditCounts] = field(default_factory=_new_role_counts)
    normal_riichi: dict[Role, _GroupAuditCounts] = field(
        default_factory=_new_role_counts
    )

    def add(self, record: RiichiPointRecord) -> None:
        role: Role = "dealer" if record.actor == record.oya else "nondealer"
        self.all_riichi[role].add(record)
        if record.reach_type == "riichi":
            self.normal_riichi[role].add(record)

    def merge(self, other: _YearAuditCounts) -> None:
        for role in ("dealer", "nondealer"):
            self.all_riichi[role].merge(other.all_riichi[role])
            self.normal_riichi[role].merge(other.normal_riichi[role])

    @property
    def record_count(self) -> int:
        return sum(group.riichis for group in self.all_riichi.values())


def _combine_audit_counts(
    values: Sequence[_YearAuditCounts],
) -> _YearAuditCounts:
    combined = _YearAuditCounts()
    for value in values:
        combined.merge(value)
    return combined


def _object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"Article 1 summary {context} must be an object")
    return value


def _integer(value: object, context: str) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"Article 1 summary {context} must be a non-negative integer")
    return value


def _validate_year_array(
    value: object,
    expected: tuple[int, ...],
    context: str,
) -> None:
    if not isinstance(value, list):
        raise TypeError(f"Article 1 summary {context} must be an array")
    if any(type(year) is not int for year in value) or tuple(value) != expected:
        raise ValueError(
            f"Article 1 summary {context} must equal the unique ascending "
            f"sequence {expected[0]}-{expected[-1]}"
        )


class _ArticleOneSummaryValidator:
    """Fixed-memory reconciliation of records against Article 1's summary."""

    def __init__(
        self,
        summary: Mapping[str, object],
        selected_years: tuple[int, ...],
    ) -> None:
        raw_years = summary.get("years")
        if not isinstance(raw_years, list):
            raise TypeError("Article 1 summary years must be an array")
        self._summary_years: dict[int, Mapping[str, object]] = {}
        for index, value in enumerate(raw_years):
            year_summary = _object(value, f"years[{index}]")
            year = _integer(year_summary.get("year"), f"years[{index}].year")
            if year in self._summary_years:
                raise ValueError(f"duplicate Article 1 summary year: {year}")
            self._summary_years[year] = year_summary
        if tuple(self._summary_years) != SUPPORTED_YEARS:
            raise ValueError(
                "Article 1 summary years must be the unique ascending "
                "2009-2025 sequence"
            )
        self._periods = _object(summary.get("periods"), "periods")
        self.selected_years = selected_years
        self._selected_set = frozenset(selected_years)
        self.year_counts = {year: _YearAuditCounts() for year in SUPPORTED_YEARS}
        self.all_record_count = 0
        self.selected_record_count = 0
        self.selected_win_count = 0
        self.converted_win_count = 0
        self.unsupported_score_count = 0
        self.ambiguous_score_count = 0

    def add_record(self, record: RiichiPointRecord) -> None:
        self.all_record_count += 1
        self.year_counts[record.year].add(record)
        if record.year not in self._selected_set:
            return
        self.selected_record_count += 1
        if record.outcome == "win":
            self.selected_win_count += 1

    def converted(self) -> None:
        self.converted_win_count += 1

    def unsupported(self) -> None:
        self.unsupported_score_count += 1

    def ambiguous(self) -> None:
        self.ambiguous_score_count += 1

    @staticmethod
    def _check(*, actual: int, expected: object, context: str) -> None:
        expected_integer = _integer(expected, context)
        if actual != expected_integer:
            raise ValueError(
                f"Article 1 summary mismatch at {context}: "
                f"records={actual}, summary={expected_integer}"
            )

    @staticmethod
    def _check_mean(
        *,
        numerator: int,
        denominator: int,
        expected: object,
        context: str,
    ) -> None:
        actual = None if denominator == 0 else numerator / denominator
        if expected is not None and type(expected) not in (int, float):
            raise TypeError(f"Article 1 summary {context} must be numeric or null")
        if expected != actual:
            raise ValueError(
                f"Article 1 summary mismatch at {context}: "
                f"records={actual}, summary={expected}"
            )

    def _validate_group(
        self,
        actual: _GroupAuditCounts,
        expected_value: object,
        context: str,
    ) -> None:
        expected = _object(expected_value, context)
        for field_name in (
            "riichis",
            "wins",
            "other_wins",
            "draws",
            "tsumo_wins",
            "ron_wins",
        ):
            self._check(
                actual=getattr(actual, field_name),
                expected=expected.get(field_name),
                context=f"{context}.{field_name}",
            )
        hand_points = _object(expected.get("hand_points"), f"{context}.hand_points")
        settlement = _object(
            expected.get("settlement_gain"),
            f"{context}.settlement_gain",
        )
        self._check(
            actual=actual.hand_points_sum,
            expected=hand_points.get("sum"),
            context=f"{context}.hand_points.sum",
        )
        self._check_mean(
            numerator=actual.hand_points_sum,
            denominator=actual.wins,
            expected=hand_points.get("mean"),
            context=f"{context}.hand_points.mean",
        )
        self._check(
            actual=actual.settlement_gain_sum,
            expected=settlement.get("sum"),
            context=f"{context}.settlement_gain.sum",
        )
        self._check_mean(
            numerator=actual.settlement_gain_sum,
            denominator=actual.wins,
            expected=settlement.get("mean"),
            context=f"{context}.settlement_gain.mean",
        )
        methods = _object(expected.get("win_methods"), f"{context}.win_methods")
        for method, wins, point_sum in (
            ("tsumo", actual.tsumo_wins, actual.tsumo_hand_points_sum),
            ("ron", actual.ron_wins, actual.ron_hand_points_sum),
        ):
            method_summary = _object(
                methods.get(method),
                f"{context}.win_methods.{method}",
            )
            self._check(
                actual=wins,
                expected=method_summary.get("wins"),
                context=f"{context}.win_methods.{method}.wins",
            )
            self._check(
                actual=point_sum,
                expected=method_summary.get("hand_points_sum"),
                context=f"{context}.win_methods.{method}.hand_points_sum",
            )
            self._check_mean(
                numerator=point_sum,
                denominator=wins,
                expected=method_summary.get("mean_hand_points"),
                context=f"{context}.win_methods.{method}.mean_hand_points",
            )

    def _validate_aggregate(
        self,
        actual: _YearAuditCounts,
        expected: Mapping[str, object],
        context: str,
    ) -> None:
        self._check(
            actual=actual.record_count,
            expected=expected.get("established_riichis"),
            context=f"{context}.established_riichis",
        )
        sensitivity = _object(
            expected.get("sensitivity_normal_riichi_only"),
            f"{context}.sensitivity_normal_riichi_only",
        )
        for role in ("dealer", "nondealer"):
            self._validate_group(
                actual.all_riichi[role],
                expected.get(role),
                f"{context}.{role}",
            )
            self._validate_group(
                actual.normal_riichi[role],
                sensitivity.get(role),
                f"{context}.sensitivity_normal_riichi_only.{role}",
            )

    def _validate_period(
        self,
        name: str,
        years: tuple[int, ...],
    ) -> None:
        context = f"periods.{name}"
        expected = _object(self._periods.get(name), context)
        _validate_year_array(expected.get("years"), years, f"{context}.years")
        actual = _combine_audit_counts(tuple(self.year_counts[year] for year in years))
        self._validate_aggregate(actual, expected, context)

    def finish(self) -> None:
        if self.selected_win_count != self.converted_win_count:
            raise ValueError(
                "not every selected Article 1 win was converted: "
                f"wins={self.selected_win_count}, converted={self.converted_win_count}, "
                f"unsupported={self.unsupported_score_count}, "
                f"ambiguous={self.ambiguous_score_count}"
            )
        for year in SUPPORTED_YEARS:
            actual = self.year_counts[year]
            expected = self._summary_years[year]
            self._validate_aggregate(actual, expected, f"year {year}")
        self._validate_period("selected", SUPPORTED_YEARS)
        self._validate_period("primary_2020_2025", PRIMARY_YEARS)
        self._validate_period("long_term_2009_2025", SUPPORTED_YEARS)
        selected = _combine_audit_counts(
            tuple(self.year_counts[year] for year in self.selected_years)
        )
        selected_wins = sum(group.wins for group in selected.all_riichi.values())
        if selected.record_count != self.selected_record_count:
            raise RuntimeError("selected record accumulator is inconsistent")
        if selected_wins != self.selected_win_count:
            raise RuntimeError("selected win accumulator is inconsistent")

    def metadata(
        self,
        *,
        summary_logical_path: str,
        records_logical_path: str,
        summary_sha256: str,
        records_sha256: str,
    ) -> DecompositionInputMetadata:
        return DecompositionInputMetadata(
            summary_logical_path=summary_logical_path,
            records_logical_path=records_logical_path,
            summary_sha256=summary_sha256,
            records_sha256=records_sha256,
            all_record_count=self.all_record_count,
            selected_record_count=self.selected_record_count,
            selected_win_count=self.selected_win_count,
            converted_win_count=self.converted_win_count,
            unsupported_score_count=self.unsupported_score_count,
            ambiguous_score_count=self.ambiguous_score_count,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"cannot hash input file: {path}") from error
    return digest.hexdigest()


def iter_decomposition_records(
    path: Path,
    years: tuple[int, ...],
    *,
    validator: _ArticleOneSummaryValidator | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> Iterator[DecompositionWinRecord]:
    """Read concatenated Article 1 gzip members and yield selected wins.

    Canonical order is validated with one previous key, so memory stays fixed
    even for the full 10-million-record audit stream.
    """
    selected_years = frozenset(years)
    previous_key: tuple[int, str, int, int, int] | None = None
    previous_hand: tuple[int, str, int] | None = None
    actors_seen = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    document = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"records line {line_number}: invalid JSON"
                    ) from error
                record = _article_one_record(document, line_number)
                if not record.source_path.startswith(f"{record.year}/"):
                    raise ValueError(
                        f"records line {line_number}: source_path year does not "
                        f"match record year: {record.source_path!r}"
                    )
                key = (
                    record.year,
                    record.source_path,
                    record.kyoku_index,
                    record.reach_event_index,
                    record.actor,
                )
                if previous_key is not None and key <= previous_key:
                    raise ValueError(
                        "Article 1 records contain a duplicate or are not in "
                        f"canonical order at line {line_number}: "
                        f"previous={previous_key}, current={key}"
                    )
                previous_key = key
                hand = (record.year, record.source_path, record.kyoku_index)
                if hand != previous_hand:
                    previous_hand = hand
                    actors_seen = 0
                actor_bit = 1 << record.actor
                if actors_seen & actor_bit:
                    identity = (
                        record.year,
                        record.source_path,
                        record.kyoku_index,
                        record.actor,
                    )
                    raise ValueError(
                        "duplicate Article 1 record identity at line "
                        f"{line_number}: {identity}"
                    )
                actors_seen |= actor_bit
                if validator is not None:
                    validator.add_record(record)
                if progress_callback is not None and line_number % 1_000_000 == 0:
                    progress_callback(line_number)
                if record.year not in selected_years or record.outcome != "win":
                    continue
                try:
                    converted = decomposition_record_from_article_one(record)
                except UnmatchedScorePatternError as error:
                    if validator is not None:
                        validator.unsupported()
                    raise ValueError(
                        "unsupported Article 1 win score at "
                        f"{record.source_path}, kyoku_index={record.kyoku_index}, "
                        f"actor={record.actor}, method={record.win_method}, "
                        f"hand_points={record.hand_points}"
                    ) from error
                except AmbiguousScorePatternError as error:
                    if validator is not None:
                        validator.ambiguous()
                    raise ValueError(
                        "ambiguous Article 1 win score at "
                        f"{record.source_path}, kyoku_index={record.kyoku_index}, "
                        f"actor={record.actor}, method={record.win_method}, "
                        f"hand_points={record.hand_points}"
                    ) from error
                except ValueError as error:
                    raise ValueError(
                        "cannot convert Article 1 win at "
                        f"{record.source_path}, kyoku_index={record.kyoku_index}, "
                        f"actor={record.actor}, method={record.win_method}, "
                        f"hand_points={record.hand_points}"
                    ) from error
                if validator is not None:
                    validator.converted()
                yield converted
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError("cannot read Article 1 gzip records") from error


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    years = selected_years_from_args(args)
    summary = _load_and_validate_summary(args.input_summary)
    validator = _ArticleOneSummaryValidator(summary, years)
    builder = ScoreDecompositionBuilder(years)
    print(
        "scanning Article 1 records "
        f"for {years[0]}-{years[-1]} (fixed-memory streaming)",
        flush=True,
    )
    for record in iter_decomposition_records(
        args.input_records,
        years,
        validator=validator,
        progress_callback=lambda count: print(
            f"records scanned: {count:,}",
            flush=True,
        ),
    ):
        builder.add(record)
    validator.finish()
    metadata = validator.metadata(
        summary_logical_path=args.input_summary_logical_path,
        records_logical_path=args.input_records_logical_path,
        summary_sha256=_sha256(args.input_summary),
        records_sha256=_sha256(args.input_records),
    )
    analysis = builder.freeze(input_metadata=metadata)
    write_decomposition_outputs(
        analysis,
        args.output_json,
        args.output_markdown,
    )
    print(f"records scanned: {metadata.all_record_count:,}", flush=True)
    print(f"selected wins converted: {metadata.converted_win_count:,}", flush=True)
    print(f"wrote: {args.output_json}", flush=True)
    print(f"wrote: {args.output_markdown}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
