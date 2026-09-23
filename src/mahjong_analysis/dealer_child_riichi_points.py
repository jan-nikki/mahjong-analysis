"""Dealer/nondealer riichi point aggregation from raw MJAI logs."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from statistics import NormalDist
from typing import Any, Literal

from mahjong_analysis.mjai import (
    filter_east_kyokus,
    is_target_game,
    load_mjai,
    match_reach_sequence,
    split_kyoku,
)
from mahjong_analysis.yearly_inference import WilsonInterval, wilson_score_interval

FIRST_YEAR = 2009
LAST_YEAR = 2025
SUPPORTED_YEARS = tuple(range(FIRST_YEAR, LAST_YEAR + 1))
PRIMARY_YEARS = tuple(range(2020, 2026))
ANALYSIS_NAME = "dealer-child-riichi-points-v1"
SCHEMA_VERSION = 2
EXPECTED_DATASET_NAME = "riichi-waits-v1"
EXPECTED_DATASET_SCHEMA_VERSION = 1
EXPECTED_RELEASE_TAG = "v2.0.0"
TARGET_RULE_CODE = "00a9"
NONDEALER_BENCHMARK_POINTS = 6_500
CONFIDENCE_LEVEL = 0.95

WinMethod = Literal["tsumo", "ron"]
ReachType = Literal["riichi", "double_riichi"]
RiichiOutcome = Literal["win", "other_win", "draw"]
ProgressCallback = Callable[[int, int], None]
CancellationCallback = Callable[[], bool]
_CALL_TYPES = frozenset(("chi", "pon", "daiminkan", "ankan", "kakan"))


class RiichiPointAggregationCancelled(RuntimeError):
    """Raised between source files after a parallel peer has failed."""


@dataclass(frozen=True)
class EstablishedRiichi:
    """One validated reach sequence and its first-go-around classification."""

    actor: int
    reach_event_index: int
    reach_accepted_event_index: int
    reach_type: ReachType


@dataclass(frozen=True)
class RiichiPointRecord:
    """Auditable row for one established riichi in the analysis population."""

    year: int
    source_path: str
    kyoku_index: int
    start_kyoku_line: int
    bakaze: str
    kyoku: int
    honba: int
    oya: int
    actor: int
    reach_type: ReachType
    reach_event_index: int
    reach_accepted_event_index: int
    reach_line: int
    reach_accepted_line: int
    outcome: RiichiOutcome
    win_method: WinMethod | None
    hora_event_index: int | None
    hora_line: int | None
    honba_awarded: bool | None
    hand_points: int | None
    settlement_gain: int | None

    def __post_init__(self) -> None:
        if self.year not in SUPPORTED_YEARS:
            raise ValueError("unsupported record year")
        if not self.source_path or "\\" in self.source_path:
            raise ValueError("source_path must be relative POSIX text")
        for name in (
            "kyoku_index",
            "start_kyoku_line",
            "kyoku",
            "honba",
            "oya",
            "actor",
            "reach_event_index",
            "reach_accepted_event_index",
            "reach_line",
            "reach_accepted_line",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.oya not in range(4) or self.actor not in range(4):
            raise ValueError("oya and actor must be between 0 and 3")
        if self.reach_type not in ("riichi", "double_riichi"):
            raise ValueError("unsupported reach_type")
        win_fields = (
            self.win_method,
            self.hora_event_index,
            self.hora_line,
            self.honba_awarded,
            self.hand_points,
            self.settlement_gain,
        )
        if self.outcome == "win":
            if any(value is None for value in win_fields):
                raise ValueError("win record is missing hora fields")
        elif self.outcome in ("other_win", "draw"):
            if any(value is not None for value in win_fields):
                raise ValueError("non-win record cannot contain hora fields")
        else:
            raise ValueError("unsupported riichi outcome")


RecordCallback = Callable[[RiichiPointRecord], None]


@dataclass(frozen=True)
class CohortYearExpectation:
    """Expected raw and established-riichi counts for one year."""

    year: int
    scanned_files: int
    target_games: int
    east_kyokus: int
    established_riichis: int

    def __post_init__(self) -> None:
        values = (
            self.year,
            self.scanned_files,
            self.target_games,
            self.east_kyokus,
            self.established_riichis,
        )
        if any(type(value) is not int for value in values):
            raise TypeError("cohort expectation values must be integers")
        if self.year not in SUPPORTED_YEARS:
            raise ValueError(f"unsupported year: {self.year}")
        if any(value < 0 for value in values[1:]):
            raise ValueError("cohort expectation counts must be non-negative")


@dataclass(frozen=True)
class CohortReference:
    """Validated canonical cohort counts used only as an independent oracle."""

    repository: str
    release_tag: str
    manifest_path: str
    years: Mapping[int, CohortYearExpectation]


@dataclass(frozen=True)
class RiichiPointGroupStats:
    """Sufficient statistics for one dealer/nondealer group."""

    riichis: int
    wins: int
    other_wins: int
    draws: int
    tsumo_wins: int
    ron_wins: int
    hand_points_sum: int
    settlement_gain_sum: int
    point_counts: tuple[tuple[int, int], ...]
    tsumo_hand_points_sum: int
    ron_hand_points_sum: int
    tsumo_point_counts: tuple[tuple[int, int], ...]
    ron_point_counts: tuple[tuple[int, int], ...]
    cluster_games: int
    cluster_wins_squared_sum: int
    cluster_points_squared_sum: int
    cluster_wins_points_sum: int

    def __post_init__(self) -> None:
        scalar_values = (
            self.riichis,
            self.wins,
            self.other_wins,
            self.draws,
            self.tsumo_wins,
            self.ron_wins,
            self.hand_points_sum,
            self.settlement_gain_sum,
            self.tsumo_hand_points_sum,
            self.ron_hand_points_sum,
            self.cluster_games,
            self.cluster_wins_squared_sum,
            self.cluster_points_squared_sum,
            self.cluster_wins_points_sum,
        )
        if any(type(value) is not int for value in scalar_values):
            raise TypeError("group statistics must be integers")
        if any(value < 0 for value in scalar_values):
            raise ValueError("group statistics must be non-negative")
        if self.riichis != self.wins + self.other_wins + self.draws:
            raise ValueError("riichi outcome counts are inconsistent")
        if self.wins != self.tsumo_wins + self.ron_wins:
            raise ValueError("riichi win-method counts are inconsistent")
        distributions = (
            ("point_counts", self.point_counts, self.wins, self.hand_points_sum),
            (
                "tsumo_point_counts",
                self.tsumo_point_counts,
                self.tsumo_wins,
                self.tsumo_hand_points_sum,
            ),
            (
                "ron_point_counts",
                self.ron_point_counts,
                self.ron_wins,
                self.ron_hand_points_sum,
            ),
        )
        for name, distribution, expected_wins, expected_points in distributions:
            counted_wins, counted_points = _validate_distribution(distribution, name)
            if counted_wins != expected_wins:
                raise ValueError(f"{name} count does not match wins")
            if counted_points != expected_points:
                raise ValueError(f"{name} sum does not match hand points")
        if self.hand_points_sum != (
            self.tsumo_hand_points_sum + self.ron_hand_points_sum
        ):
            raise ValueError("method hand-point sums do not match total")
        if self.settlement_gain_sum < self.hand_points_sum:
            raise ValueError("settlement gain cannot be below hand points")

    @property
    def win_rate(self) -> float | None:
        return _ratio(self.wins, self.riichis)

    @property
    def tsumo_share(self) -> float | None:
        return _ratio(self.tsumo_wins, self.wins)

    @property
    def ron_share(self) -> float | None:
        return _ratio(self.ron_wins, self.wins)

    @property
    def tsumo_rate_per_riichi(self) -> float | None:
        return _ratio(self.tsumo_wins, self.riichis)

    @property
    def ron_rate_per_riichi(self) -> float | None:
        return _ratio(self.ron_wins, self.riichis)

    @property
    def mean_hand_points(self) -> float | None:
        return _ratio(self.hand_points_sum, self.wins)

    @property
    def mean_settlement_gain(self) -> float | None:
        return _ratio(self.settlement_gain_sum, self.wins)

    @property
    def mean_tsumo_hand_points(self) -> float | None:
        return _ratio(self.tsumo_hand_points_sum, self.tsumo_wins)

    @property
    def mean_ron_hand_points(self) -> float | None:
        return _ratio(self.ron_hand_points_sum, self.ron_wins)

    @property
    def hand_points_per_riichi(self) -> float | None:
        return _ratio(self.hand_points_sum, self.riichis)

    @property
    def minimum_hand_points(self) -> int | None:
        return None if not self.point_counts else self.point_counts[0][0]

    @property
    def maximum_hand_points(self) -> int | None:
        return None if not self.point_counts else self.point_counts[-1][0]

    @property
    def median_hand_points(self) -> float | None:
        if self.wins == 0:
            return None
        lower_rank = (self.wins - 1) // 2
        upper_rank = self.wins // 2
        lower_value = _value_at_rank(self.point_counts, lower_rank)
        upper_value = _value_at_rank(self.point_counts, upper_rank)
        return (lower_value + upper_value) / 2.0

    @property
    def mean_hand_points_ci95(self) -> tuple[float, float] | None:
        """Return a game-clustered normal interval for the mean hand points."""
        mean = self.mean_hand_points
        if mean is None or self.cluster_games < 2:
            return None
        # Keep the cancellation-prone residual sum exact until the final division.
        numerator = (
            self.wins * self.wins * self.cluster_points_squared_sum
            - 2 * self.hand_points_sum * self.wins * self.cluster_wins_points_sum
            + self.hand_points_sum
            * self.hand_points_sum
            * self.cluster_wins_squared_sum
        )
        if numerator < 0:
            raise ValueError("cluster variance components are inconsistent")
        variance = (
            self.cluster_games / (self.cluster_games - 1) * numerator / (self.wins**4)
        )
        half_width = NormalDist().inv_cdf(0.975) * math.sqrt(variance)
        return max(0.0, mean - half_width), mean + half_width


@dataclass(frozen=True)
class YearlyRiichiPointResult:
    """One year's source counts and dealer/nondealer statistics."""

    year: int
    scanned_files: int
    target_games: int
    east_kyokus: int
    dealer: RiichiPointGroupStats
    nondealer: RiichiPointGroupStats
    normal_dealer: RiichiPointGroupStats
    normal_nondealer: RiichiPointGroupStats

    def __post_init__(self) -> None:
        values = (self.year, self.scanned_files, self.target_games, self.east_kyokus)
        if any(type(value) is not int for value in values):
            raise TypeError("yearly source counts must be integers")
        if self.year not in SUPPORTED_YEARS:
            raise ValueError(f"unsupported year: {self.year}")
        if any(value < 0 for value in values[1:]):
            raise ValueError("yearly source counts must be non-negative")
        for name in ("dealer", "nondealer", "normal_dealer", "normal_nondealer"):
            if getattr(self, name).cluster_games != self.target_games:
                raise ValueError(f"{name} cluster count must equal target games")

    @property
    def established_riichis(self) -> int:
        return self.dealer.riichis + self.nondealer.riichis


@dataclass(frozen=True)
class RiichiPointPeriodResult:
    """Combined result for a deterministic set of years."""

    years: tuple[int, ...]
    scanned_files: int
    target_games: int
    east_kyokus: int
    dealer: RiichiPointGroupStats
    nondealer: RiichiPointGroupStats
    normal_dealer: RiichiPointGroupStats
    normal_nondealer: RiichiPointGroupStats

    @property
    def established_riichis(self) -> int:
        return self.dealer.riichis + self.nondealer.riichis


@dataclass(frozen=True)
class RiichiPointAnalysis:
    """Complete analysis over selected years."""

    repository: str
    release_tag: str
    cohort_manifest_path: str
    selected_years: tuple[int, ...]
    years: tuple[YearlyRiichiPointResult, ...]

    def __post_init__(self) -> None:
        if tuple(result.year for result in self.years) != self.selected_years:
            raise ValueError("yearly results do not match selected years")


@dataclass
class _GroupCounts:
    riichis: int = 0
    wins: int = 0
    other_wins: int = 0
    draws: int = 0
    tsumo_wins: int = 0
    ron_wins: int = 0
    hand_points_sum: int = 0
    settlement_gain_sum: int = 0
    point_counts: Counter[int] | None = None
    tsumo_hand_points_sum: int = 0
    ron_hand_points_sum: int = 0
    tsumo_point_counts: Counter[int] | None = None
    ron_point_counts: Counter[int] | None = None

    def __post_init__(self) -> None:
        if self.point_counts is None:
            self.point_counts = Counter()
        if self.tsumo_point_counts is None:
            self.tsumo_point_counts = Counter()
        if self.ron_point_counts is None:
            self.ron_point_counts = Counter()

    def add_win(
        self,
        method: WinMethod,
        hand_points: int,
        settlement_gain: int,
    ) -> None:
        self.riichis += 1
        self.wins += 1
        if method == "tsumo":
            self.tsumo_wins += 1
            self.tsumo_hand_points_sum += hand_points
            assert self.tsumo_point_counts is not None
            self.tsumo_point_counts[hand_points] += 1
        elif method == "ron":
            self.ron_wins += 1
            self.ron_hand_points_sum += hand_points
            assert self.ron_point_counts is not None
            self.ron_point_counts[hand_points] += 1
        else:
            raise ValueError(f"unexpected win method: {method}")
        self.hand_points_sum += hand_points
        self.settlement_gain_sum += settlement_gain
        assert self.point_counts is not None
        self.point_counts[hand_points] += 1

    def add_other_win(self) -> None:
        self.riichis += 1
        self.other_wins += 1

    def add_draw(self) -> None:
        self.riichis += 1
        self.draws += 1

    def merge(self, other: _GroupCounts) -> None:
        self.riichis += other.riichis
        self.wins += other.wins
        self.other_wins += other.other_wins
        self.draws += other.draws
        self.tsumo_wins += other.tsumo_wins
        self.ron_wins += other.ron_wins
        self.hand_points_sum += other.hand_points_sum
        self.settlement_gain_sum += other.settlement_gain_sum
        self.tsumo_hand_points_sum += other.tsumo_hand_points_sum
        self.ron_hand_points_sum += other.ron_hand_points_sum
        assert self.point_counts is not None
        assert other.point_counts is not None
        self.point_counts.update(other.point_counts)
        assert self.tsumo_point_counts is not None
        assert other.tsumo_point_counts is not None
        self.tsumo_point_counts.update(other.tsumo_point_counts)
        assert self.ron_point_counts is not None
        assert other.ron_point_counts is not None
        self.ron_point_counts.update(other.ron_point_counts)


@dataclass
class _GroupAccumulator:
    counts: _GroupCounts
    cluster_games: int = 0
    cluster_wins_squared_sum: int = 0
    cluster_points_squared_sum: int = 0
    cluster_wins_points_sum: int = 0

    @classmethod
    def empty(cls) -> _GroupAccumulator:
        return cls(counts=_GroupCounts())

    def add_game(self, game: _GroupCounts) -> None:
        self.counts.merge(game)
        self.cluster_games += 1
        self.cluster_wins_squared_sum += game.wins * game.wins
        self.cluster_points_squared_sum += game.hand_points_sum**2
        self.cluster_wins_points_sum += game.wins * game.hand_points_sum

    def finish(self) -> RiichiPointGroupStats:
        assert self.counts.point_counts is not None
        assert self.counts.tsumo_point_counts is not None
        assert self.counts.ron_point_counts is not None
        return RiichiPointGroupStats(
            riichis=self.counts.riichis,
            wins=self.counts.wins,
            other_wins=self.counts.other_wins,
            draws=self.counts.draws,
            tsumo_wins=self.counts.tsumo_wins,
            ron_wins=self.counts.ron_wins,
            hand_points_sum=self.counts.hand_points_sum,
            settlement_gain_sum=self.counts.settlement_gain_sum,
            point_counts=tuple(sorted(self.counts.point_counts.items())),
            tsumo_hand_points_sum=self.counts.tsumo_hand_points_sum,
            ron_hand_points_sum=self.counts.ron_hand_points_sum,
            tsumo_point_counts=tuple(sorted(self.counts.tsumo_point_counts.items())),
            ron_point_counts=tuple(sorted(self.counts.ron_point_counts.items())),
            cluster_games=self.cluster_games,
            cluster_wins_squared_sum=self.cluster_wins_squared_sum,
            cluster_points_squared_sum=self.cluster_points_squared_sum,
            cluster_wins_points_sum=self.cluster_wins_points_sum,
        )


def normalize_years(years: Iterable[int]) -> tuple[int, ...]:
    """Return unique supported years in ascending order."""
    values = tuple(years)
    if not values:
        raise ValueError("at least one year must be selected")
    if any(type(year) is not int for year in values):
        raise TypeError("years must be integers")
    if len(set(values)) != len(values):
        raise ValueError("duplicate years are not allowed")
    invalid = tuple(sorted(year for year in values if year not in SUPPORTED_YEARS))
    if invalid:
        raise ValueError(f"unsupported years: {invalid}")
    return tuple(sorted(values))


def load_cohort_reference(
    path: str | PathLike[str],
    years: Iterable[int],
    *,
    logical_path: str,
) -> CohortReference:
    """Load and validate canonical counts without using records as input."""
    selected_years = normalize_years(years)
    if not logical_path or "\\" in logical_path or Path(logical_path).is_absolute():
        raise ValueError("cohort manifest logical_path must be relative POSIX text")
    try:
        data = json.loads(Path(path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("cannot read cohort manifest JSON") from error
    if not isinstance(data, dict):
        raise TypeError("cohort manifest must be an object")
    if data.get("dataset_name") != EXPECTED_DATASET_NAME:
        raise ValueError("unexpected cohort dataset name")
    if data.get("schema_version") != EXPECTED_DATASET_SCHEMA_VERSION:
        raise ValueError("unexpected cohort schema version")

    generator = _mapping(data.get("generator"), "generator")
    invocation = _mapping(generator.get("invocation"), "generator.invocation")
    if invocation.get("mode") != "full" or invocation.get("max_files") is not None:
        raise ValueError("cohort manifest must describe a full export")

    source = _mapping(data.get("source"), "source")
    repository = _nonempty_string(source.get("repository"), "source.repository")
    if source.get("release_tag") != EXPECTED_RELEASE_TAG:
        raise ValueError("unexpected cohort source release")
    if source.get("archive_hashes_verified") is not True:
        raise ValueError("cohort source archives were not verified")

    scope = _mapping(data.get("scope"), "scope")
    if scope.get("rule_code") != TARGET_RULE_CODE:
        raise ValueError("unexpected cohort rule code")
    if scope.get("aka_flag") is not True or scope.get("bakaze") != "E":
        raise ValueError("unexpected cohort scope")
    if scope.get("extraction_mode") != "full":
        raise ValueError("cohort extraction mode must be full")

    raw_years = data.get("years")
    if not isinstance(raw_years, list):
        raise TypeError("cohort years must be an array")
    expectations: dict[int, CohortYearExpectation] = {}
    for raw_year in raw_years:
        value = _mapping(raw_year, "cohort year")
        expectation = CohortYearExpectation(
            year=_integer(value.get("year"), "year"),
            scanned_files=_integer(value.get("scanned_files"), "scanned_files"),
            target_games=_integer(value.get("target_games"), "target_games"),
            east_kyokus=_integer(value.get("east_kyokus"), "east_kyokus"),
            established_riichis=_integer(
                value.get("established_riichis"),
                "established_riichis",
            ),
        )
        if expectation.year in expectations:
            raise ValueError(f"duplicate cohort year: {expectation.year}")
        expectations[expectation.year] = expectation
    missing = tuple(year for year in selected_years if year not in expectations)
    if missing:
        raise ValueError(f"cohort manifest is missing years: {missing}")
    return CohortReference(
        repository=repository,
        release_tag=EXPECTED_RELEASE_TAG,
        manifest_path=logical_path,
        years={year: expectations[year] for year in selected_years},
    )


def extract_established_riichi_actors(
    kyoku: Sequence[dict[str, Any]],
) -> tuple[int, ...]:
    """Return actors whose reach declaration reached ``reach_accepted``."""
    return tuple(record.actor for record in extract_established_riichis(kyoku))


def extract_established_riichis(
    kyoku: Sequence[dict[str, Any]],
) -> tuple[EstablishedRiichi, ...]:
    """Return established riichis with source indexes and reach type."""
    events = list(kyoku)
    accepted_indexes: set[int] = set()
    records: list[EstablishedRiichi] = []
    for event_index, event in enumerate(events):
        if not isinstance(event, dict):
            raise TypeError(f"event {event_index}: MJAI event must be an object")
        if event.get("type") != "reach":
            continue
        sequence = match_reach_sequence(
            events,
            event_index,
            context="riichi-point analysis reach",
        )
        if sequence is None:
            continue
        actor, _, _, accepted_index = sequence
        if any(record.actor == actor for record in records):
            raise ValueError(f"event {event_index}: actor {actor} reached twice")
        if accepted_index in accepted_indexes:
            raise ValueError(f"event {accepted_index}: duplicate reach_accepted")
        had_own_discard = any(
            prior.get("type") == "dahai" and prior.get("actor") == actor
            for prior in events[:event_index]
        )
        had_call = any(
            prior.get("type") in _CALL_TYPES for prior in events[:event_index]
        )
        records.append(
            EstablishedRiichi(
                actor=actor,
                reach_event_index=event_index,
                reach_accepted_event_index=accepted_index,
                reach_type=(
                    "double_riichi"
                    if not had_own_discard and not had_call
                    else "riichi"
                ),
            )
        )
        accepted_indexes.add(accepted_index)

    for event_index, event in enumerate(events):
        if (
            event.get("type") == "reach_accepted"
            and event_index not in accepted_indexes
        ):
            raise ValueError(f"event {event_index}: unmatched reach_accepted")
    return tuple(records)


def aggregate_riichi_points(
    year: int,
    paths: Iterable[str | PathLike[str]],
    *,
    record_callback: RecordCallback | None = None,
) -> YearlyRiichiPointResult:
    """Aggregate one year from raw MJAI paths in a single pass."""
    if year not in SUPPORTED_YEARS:
        raise ValueError(f"unsupported year: {year}")
    scanned_files = 0
    target_games = 0
    east_kyokus = 0
    dealer = _GroupAccumulator.empty()
    nondealer = _GroupAccumulator.empty()
    normal_dealer = _GroupAccumulator.empty()
    normal_nondealer = _GroupAccumulator.empty()

    for path_value in paths:
        path = Path(path_value)
        scanned_files += 1
        try:
            events = load_mjai(path)
            if not is_target_game(path, events):
                continue
            target_games += 1
            game_dealer = _GroupCounts()
            game_nondealer = _GroupCounts()
            game_normal_dealer = _GroupCounts()
            game_normal_nondealer = _GroupCounts()
            all_kyokus = split_kyoku(events)
            kyoku_indexes = {id(kyoku): index for index, kyoku in enumerate(all_kyokus)}
            source_lines = {
                id(event): line_number
                for line_number, event in enumerate(events, start=1)
            }
            kyokus = filter_east_kyokus(all_kyokus)
            east_kyokus += len(kyokus)
            for kyoku in kyokus:
                _aggregate_kyoku(
                    kyoku,
                    game_dealer,
                    game_nondealer,
                    game_normal_dealer,
                    game_normal_nondealer,
                    year=year,
                    source_path=f"{year}/{path.name}",
                    kyoku_index=kyoku_indexes[id(kyoku)],
                    source_lines=source_lines,
                    record_callback=record_callback,
                )
            dealer.add_game(game_dealer)
            nondealer.add_game(game_nondealer)
            normal_dealer.add_game(game_normal_dealer)
            normal_nondealer.add_game(game_normal_nondealer)
        except Exception as error:
            raise RuntimeError(f"failed to process MJAI file: {path}") from error

    return YearlyRiichiPointResult(
        year=year,
        scanned_files=scanned_files,
        target_games=target_games,
        east_kyokus=east_kyokus,
        dealer=dealer.finish(),
        nondealer=nondealer.finish(),
        normal_dealer=normal_dealer.finish(),
        normal_nondealer=normal_nondealer.finish(),
    )


def aggregate_riichi_point_years(
    years: Iterable[int],
    raw_root: str | PathLike[str],
    cohort: CohortReference,
    *,
    progress_interval: int = 10_000,
    progress_callback: ProgressCallback | None = None,
    record_callback: RecordCallback | None = None,
    cancellation_callback: CancellationCallback | None = None,
) -> RiichiPointAnalysis:
    """Aggregate selected complete years and validate canonical cohort counts."""
    selected_years = normalize_years(years)
    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive")
    missing = tuple(year for year in selected_years if year not in cohort.years)
    if missing:
        raise ValueError(f"cohort reference is missing years: {missing}")
    root = Path(raw_root)
    results: list[YearlyRiichiPointResult] = []
    for year in selected_years:
        year_root = root / str(year)
        if not year_root.is_dir():
            raise FileNotFoundError(f"raw year directory does not exist: {year_root}")
        paths = sorted(
            (
                path
                for path in year_root.iterdir()
                if path.is_file() and path.suffix == ".mjson"
            ),
            key=lambda path: path.name,
        )
        expected = cohort.years[year]
        if len(paths) != expected.scanned_files:
            raise RuntimeError(
                f"raw file count mismatch for {year}: "
                f"expected {expected.scanned_files}, found {len(paths)}"
            )
        result = aggregate_riichi_points(
            year,
            _paths_with_progress(
                year,
                paths,
                progress_interval,
                progress_callback,
                cancellation_callback,
            ),
            record_callback=record_callback,
        )
        _validate_cohort_counts(result, expected)
        results.append(result)
        del paths
    return RiichiPointAnalysis(
        repository=cohort.repository,
        release_tag=cohort.release_tag,
        cohort_manifest_path=cohort.manifest_path,
        selected_years=selected_years,
        years=tuple(results),
    )


def combine_years(
    results: Iterable[YearlyRiichiPointResult],
) -> RiichiPointPeriodResult:
    """Combine yearly sufficient statistics without losing distributions."""
    values = tuple(results)
    if not values:
        raise ValueError("at least one yearly result is required")
    years = tuple(value.year for value in values)
    if years != tuple(sorted(years)) or len(set(years)) != len(years):
        raise ValueError("yearly results must be unique and sorted")
    return RiichiPointPeriodResult(
        years=years,
        scanned_files=sum(value.scanned_files for value in values),
        target_games=sum(value.target_games for value in values),
        east_kyokus=sum(value.east_kyokus for value in values),
        dealer=_combine_groups(value.dealer for value in values),
        nondealer=_combine_groups(value.nondealer for value in values),
        normal_dealer=_combine_groups(value.normal_dealer for value in values),
        normal_nondealer=_combine_groups(value.normal_nondealer for value in values),
    )


def analysis_document(analysis: RiichiPointAnalysis) -> dict[str, Any]:
    """Return the deterministic JSON-compatible analysis document."""
    by_year = {result.year: result for result in analysis.years}
    periods: dict[str, Any] = {
        "selected": _period_to_dict(combine_years(analysis.years)),
    }
    if all(year in by_year for year in PRIMARY_YEARS):
        periods["primary_2020_2025"] = _period_to_dict(
            combine_years(by_year[year] for year in PRIMARY_YEARS)
        )
    if all(year in by_year for year in SUPPORTED_YEARS):
        periods["long_term_2009_2025"] = _period_to_dict(
            combine_years(by_year[year] for year in SUPPORTED_YEARS)
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_name": ANALYSIS_NAME,
        "source": {
            "repository": analysis.repository,
            "release_tag": analysis.release_tag,
            "format": "MJAI JSON Lines",
            "raw_root_pattern": "data/raw/YYYY/*.mjson",
            "cohort_validation_manifest": analysis.cohort_manifest_path,
            "cohort_manifest_role": "count validation only",
        },
        "scope": {
            "rule_code": TARGET_RULE_CODE,
            "aka_flag": True,
            "bakaze": "E",
            "riichi_population": "established",
            "selected_years": list(analysis.selected_years),
            "primary_years": list(PRIMARY_YEARS),
        },
        "definitions": {
            "dealer": "riichi actor equals start_kyoku.oya",
            "nondealer": "riichi actor differs from start_kyoku.oya",
            "win_method": "tsumo when hora.actor == hora.target; otherwise ron",
            "hand_points": (
                "sum of negative non-winner hora deltas, less 300 points per honba "
                "only for the first (honba-awarded) hora in a multi-ron result"
            ),
            "hand_points_excludes": ["honba", "kyotaku"],
            "tsumo_points": "total paid by all three opponents",
            "settlement_gain": (
                "hora.deltas[winner], including honba and awarded kyotaku, "
                "excluding the earlier riichi payment"
            ),
            "mean_hand_points_ci95": "game-clustered normal sandwich interval",
            "rate_ci95": "Wilson score interval",
            "normal_riichi_sensitivity": (
                "same metrics after excluding first-go-around double riichi"
            ),
        },
        "benchmark": {
            "nondealer_points": NONDEALER_BENCHMARK_POINTS,
            "provenance_status": "external source and definition not yet verified",
        },
        "periods": periods,
        "years": [_year_to_dict(result) for result in analysis.years],
    }


def render_markdown(analysis: RiichiPointAnalysis) -> str:
    """Render a compact human-readable report from the canonical statistics."""
    document = analysis_document(analysis)
    lines = [
        "# 親子別・成立リーチ平均打点",
        "",
        f"- Source: `{analysis.repository}` `{analysis.release_tag}` raw MJAI",
        f"- Selected years: {', '.join(map(str, analysis.selected_years))}",
        "- Scope: 四人打ち鳳凰卓・赤あり・東場・成立リーチ",
        "- 主打点: 本場・供託を除く和了点（ツモは3人の支払い合計）",
        "",
        "## 期間別",
        "",
        (
            "| Period | Group | Riichis | Wins | Win rate | Tsumo share | "
            "Mean points | 95% CI | Median | Mangan+ |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for period_name, period in document["periods"].items():
        for group_name in ("dealer", "nondealer"):
            group = period[group_name]
            ci = group["hand_points"]["mean_ci95"]
            lines.append(
                f"| {period_name} | {group_name} | {group['riichis']:,} | "
                f"{group['wins']:,} | {_format_rate(group['win_rate'])} | "
                f"{_format_rate(group['tsumo_share_among_wins'])} | "
                f"{_format_points(group['hand_points']['mean'])} | "
                f"{_format_interval(ci)} | "
                f"{_format_points(group['hand_points']['median'])} | "
                f"{_format_rate(group['hand_points']['mangan_or_higher_rate'])} |"
            )
    lines.extend(["", "### 期間別比較", ""])
    for period_name, period in document["periods"].items():
        comparison = period["comparison"]
        lines.extend(
            [
                (
                    f"- `{period_name}` 親平均÷子平均: "
                    f"{_format_decimal(comparison['dealer_to_nondealer_mean_ratio'])}"
                ),
                (
                    f"- `{period_name}` 子平均−6,500点: "
                    f"{_format_signed_points(comparison['nondealer_mean_minus_6500'])}"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## 期間別ツモ・ロン",
            "",
            (
                "| Period | Group | Tsumo/riichi | Ron/riichi | "
                "Mean tsumo points | Mean ron points |"
            ),
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for period_name, period in document["periods"].items():
        for group_name in ("dealer", "nondealer"):
            methods = period[group_name]["win_methods"]
            lines.append(
                f"| {period_name} | {group_name} | "
                f"{_format_rate(methods['tsumo']['probability_per_established_riichi'])} | "
                f"{_format_rate(methods['ron']['probability_per_established_riichi'])} | "
                f"{_format_points(methods['tsumo']['mean_hand_points'])} | "
                f"{_format_points(methods['ron']['mean_hand_points'])} |"
            )

    lines.extend(
        [
            "",
            "## 年別",
            "",
            (
                "| Year | Riichis | Dealer mean | Nondealer mean | Ratio | "
                "Nondealer − 6,500 | Dealer tsumo share | Nondealer tsumo share |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for yearly in document["years"]:
        comparison = yearly["comparison"]
        lines.append(
            f"| {yearly['year']} | {yearly['established_riichis']:,} | "
            f"{_format_points(yearly['dealer']['hand_points']['mean'])} | "
            f"{_format_points(yearly['nondealer']['hand_points']['mean'])} | "
            f"{_format_decimal(comparison['dealer_to_nondealer_mean_ratio'])} | "
            f"{_format_signed_points(comparison['nondealer_mean_minus_6500'])} | "
            f"{_format_rate(yearly['dealer']['tsumo_share_among_wins'])} | "
            f"{_format_rate(yearly['nondealer']['tsumo_share_among_wins'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    analysis: RiichiPointAnalysis,
    json_path: str | PathLike[str],
    markdown_path: str | PathLike[str],
) -> None:
    """Write JSON and Markdown via temporary files after a complete run."""
    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    if json_target.resolve() == markdown_target.resolve():
        raise ValueError("JSON and Markdown output paths must differ")
    json_bytes = (
        json.dumps(
            analysis_document(analysis),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    markdown_bytes = render_markdown(analysis).encode("utf-8")
    _atomic_write(json_target, json_bytes)
    _atomic_write(markdown_target, markdown_bytes)


def riichi_point_record_document(record: RiichiPointRecord) -> dict[str, Any]:
    """Return one audit record as a stable JSON-compatible mapping."""
    if not isinstance(record, RiichiPointRecord):
        raise TypeError("record must be a RiichiPointRecord")
    return asdict(record)


def _aggregate_kyoku(
    kyoku: list[dict[str, Any]],
    dealer: _GroupCounts,
    nondealer: _GroupCounts,
    normal_dealer: _GroupCounts,
    normal_nondealer: _GroupCounts,
    *,
    year: int,
    source_path: str,
    kyoku_index: int,
    source_lines: Mapping[int, int],
    record_callback: RecordCallback | None,
) -> None:
    start = _mapping(kyoku[0] if kyoku else None, "start_kyoku")
    if start.get("type") != "start_kyoku":
        raise ValueError("kyoku must start with start_kyoku")
    oya = _actor(start.get("oya"), "start_kyoku.oya")
    honba = _integer(start.get("honba"), "start_kyoku.honba")
    if honba < 0:
        raise ValueError("start_kyoku.honba must be non-negative")
    bakaze = _nonempty_string(start.get("bakaze"), "start_kyoku.bakaze")
    kyoku_number = _integer(start.get("kyoku"), "start_kyoku.kyoku")
    riichis = extract_established_riichis(kyoku)
    result_kind, horas = _result_events(kyoku)
    horas_by_actor = {_actor(hora.get("actor"), "hora.actor"): hora for hora in horas}
    if len(horas_by_actor) != len(horas):
        raise ValueError("one actor cannot have multiple hora events")
    hora_indexes = {id(event): index for index, event in enumerate(kyoku)}
    first_hora = horas[0] if horas else None

    for riichi in riichis:
        actor = riichi.actor
        group = dealer if actor == oya else nondealer
        normal_group = normal_dealer if actor == oya else normal_nondealer
        hora = horas_by_actor.get(actor)
        if hora is not None:
            honba_awarded = hora is first_hora
            method, hand_points, settlement_gain = _score_hora(
                hora,
                honba,
                honba_awarded=honba_awarded,
            )
            group.add_win(method, hand_points, settlement_gain)
            if riichi.reach_type == "riichi":
                normal_group.add_win(method, hand_points, settlement_gain)
            outcome: RiichiOutcome = "win"
            hora_event_index = hora_indexes[id(hora)]
            hora_line = source_lines[id(hora)]
        elif result_kind == "draw":
            group.add_draw()
            if riichi.reach_type == "riichi":
                normal_group.add_draw()
            outcome = "draw"
            method = None
            hand_points = None
            settlement_gain = None
            hora_event_index = None
            hora_line = None
            honba_awarded = None
        else:
            group.add_other_win()
            if riichi.reach_type == "riichi":
                normal_group.add_other_win()
            outcome = "other_win"
            method = None
            hand_points = None
            settlement_gain = None
            hora_event_index = None
            hora_line = None
            honba_awarded = None
        if record_callback is not None:
            record_callback(
                RiichiPointRecord(
                    year=year,
                    source_path=source_path,
                    kyoku_index=kyoku_index,
                    start_kyoku_line=source_lines[id(kyoku[0])],
                    bakaze=bakaze,
                    kyoku=kyoku_number,
                    honba=honba,
                    oya=oya,
                    actor=actor,
                    reach_type=riichi.reach_type,
                    reach_event_index=riichi.reach_event_index,
                    reach_accepted_event_index=riichi.reach_accepted_event_index,
                    reach_line=source_lines[id(kyoku[riichi.reach_event_index])],
                    reach_accepted_line=source_lines[
                        id(kyoku[riichi.reach_accepted_event_index])
                    ],
                    outcome=outcome,
                    win_method=method,
                    hora_event_index=hora_event_index,
                    hora_line=hora_line,
                    honba_awarded=honba_awarded,
                    hand_points=hand_points,
                    settlement_gain=settlement_gain,
                )
            )


def _result_events(
    kyoku: Sequence[dict[str, Any]],
) -> tuple[Literal["hora", "draw"], tuple[dict[str, Any], ...]]:
    if not kyoku or not isinstance(kyoku[-1], dict):
        raise ValueError("kyoku is empty or malformed")
    if kyoku[-1].get("type") != "end_kyoku":
        raise ValueError("kyoku must end with end_kyoku")
    results: list[dict[str, Any]] = []
    started = False
    for event_index, raw_event in enumerate(kyoku[:-1]):
        event = _mapping(raw_event, f"event {event_index}")
        event_type = event.get("type")
        if event_type in {"hora", "ryukyoku"}:
            results.append(event)
            started = True
        elif started:
            raise ValueError("result events must be contiguous before end_kyoku")
    if not results:
        raise ValueError("kyoku has no hora or ryukyoku result")
    horas = tuple(event for event in results if event.get("type") == "hora")
    draws = tuple(event for event in results if event.get("type") == "ryukyoku")
    if horas and draws:
        raise ValueError("hora and ryukyoku cannot coexist")
    if draws:
        if len(draws) != 1:
            raise ValueError("kyoku cannot contain multiple ryukyoku events")
        return "draw", ()
    targets: set[int] = set()
    for hora in horas:
        actor = _actor(hora.get("actor"), "hora.actor")
        target = _actor(hora.get("target"), "hora.target")
        if len(horas) > 1:
            if actor == target:
                raise ValueError("multiple hora events cannot include tsumo")
            targets.add(target)
    if len(targets) > 1:
        raise ValueError("multiple ron events must have the same target")
    return "hora", horas


def _score_hora(
    hora: Mapping[str, Any],
    honba: int,
    *,
    honba_awarded: bool,
) -> tuple[WinMethod, int, int]:
    actor = _actor(hora.get("actor"), "hora.actor")
    target = _actor(hora.get("target"), "hora.target")
    raw_deltas = hora.get("deltas")
    if not isinstance(raw_deltas, list) or len(raw_deltas) != 4:
        raise ValueError("hora.deltas must contain four integers")
    if any(type(delta) is not int for delta in raw_deltas):
        raise TypeError("hora.deltas must contain four integers")
    if raw_deltas[actor] <= 0:
        raise ValueError("hora winner delta must be positive")
    other_deltas = [delta for index, delta in enumerate(raw_deltas) if index != actor]
    if any(delta > 0 for delta in other_deltas):
        raise ValueError("non-winner hora delta cannot be positive")
    payer_loss = -sum(delta for delta in other_deltas if delta < 0)
    hand_points = payer_loss - (300 * honba if honba_awarded else 0)
    settlement_gain = raw_deltas[actor]
    if hand_points <= 0 or hand_points % 100 != 0:
        raise ValueError("derived hand points must be a positive multiple of 100")
    if settlement_gain < payer_loss or settlement_gain % 100 != 0:
        raise ValueError("hora settlement gain is inconsistent with payer losses")
    if actor != target and raw_deltas[target] >= 0:
        raise ValueError("ron target delta must be negative")
    method: WinMethod = "tsumo" if actor == target else "ron"
    return method, hand_points, settlement_gain


def _validate_cohort_counts(
    result: YearlyRiichiPointResult,
    expected: CohortYearExpectation,
) -> None:
    actual = {
        "scanned_files": result.scanned_files,
        "target_games": result.target_games,
        "east_kyokus": result.east_kyokus,
        "established_riichis": result.established_riichis,
    }
    expected_values = {
        "scanned_files": expected.scanned_files,
        "target_games": expected.target_games,
        "east_kyokus": expected.east_kyokus,
        "established_riichis": expected.established_riichis,
    }
    differences = [
        f"{field}=expected {expected_values[field]}, actual {actual[field]}"
        for field in actual
        if actual[field] != expected_values[field]
    ]
    if differences:
        raise RuntimeError(
            f"cohort count mismatch for {result.year}: {', '.join(differences)}"
        )


def _combine_groups(
    groups: Iterable[RiichiPointGroupStats],
) -> RiichiPointGroupStats:
    values = tuple(groups)
    if not values:
        raise ValueError("at least one group is required")
    distribution: Counter[int] = Counter()
    tsumo_distribution: Counter[int] = Counter()
    ron_distribution: Counter[int] = Counter()
    for value in values:
        distribution.update(dict(value.point_counts))
        tsumo_distribution.update(dict(value.tsumo_point_counts))
        ron_distribution.update(dict(value.ron_point_counts))
    return RiichiPointGroupStats(
        riichis=sum(value.riichis for value in values),
        wins=sum(value.wins for value in values),
        other_wins=sum(value.other_wins for value in values),
        draws=sum(value.draws for value in values),
        tsumo_wins=sum(value.tsumo_wins for value in values),
        ron_wins=sum(value.ron_wins for value in values),
        hand_points_sum=sum(value.hand_points_sum for value in values),
        settlement_gain_sum=sum(value.settlement_gain_sum for value in values),
        point_counts=tuple(sorted(distribution.items())),
        tsumo_hand_points_sum=sum(value.tsumo_hand_points_sum for value in values),
        ron_hand_points_sum=sum(value.ron_hand_points_sum for value in values),
        tsumo_point_counts=tuple(sorted(tsumo_distribution.items())),
        ron_point_counts=tuple(sorted(ron_distribution.items())),
        cluster_games=sum(value.cluster_games for value in values),
        cluster_wins_squared_sum=sum(
            value.cluster_wins_squared_sum for value in values
        ),
        cluster_points_squared_sum=sum(
            value.cluster_points_squared_sum for value in values
        ),
        cluster_wins_points_sum=sum(value.cluster_wins_points_sum for value in values),
    )


def _year_to_dict(result: YearlyRiichiPointResult) -> dict[str, Any]:
    comparison = _comparison(result.dealer, result.nondealer)
    return {
        "year": result.year,
        "scanned_files": result.scanned_files,
        "target_games": result.target_games,
        "east_kyokus": result.east_kyokus,
        "established_riichis": result.established_riichis,
        "dealer": _group_to_dict(result.dealer, is_dealer=True),
        "nondealer": _group_to_dict(result.nondealer, is_dealer=False),
        "comparison": comparison,
        "sensitivity_normal_riichi_only": _sensitivity_to_dict(
            result.normal_dealer,
            result.normal_nondealer,
        ),
    }


def _period_to_dict(result: RiichiPointPeriodResult) -> dict[str, Any]:
    return {
        "years": list(result.years),
        "scanned_files": result.scanned_files,
        "target_games": result.target_games,
        "east_kyokus": result.east_kyokus,
        "established_riichis": result.established_riichis,
        "dealer": _group_to_dict(result.dealer, is_dealer=True),
        "nondealer": _group_to_dict(result.nondealer, is_dealer=False),
        "comparison": _comparison(result.dealer, result.nondealer),
        "sensitivity_normal_riichi_only": _sensitivity_to_dict(
            result.normal_dealer,
            result.normal_nondealer,
        ),
    }


def _sensitivity_to_dict(
    dealer: RiichiPointGroupStats,
    nondealer: RiichiPointGroupStats,
) -> dict[str, Any]:
    return {
        "dealer": _group_to_dict(dealer, is_dealer=True),
        "nondealer": _group_to_dict(nondealer, is_dealer=False),
        "comparison": _comparison(dealer, nondealer),
    }


def _group_to_dict(
    group: RiichiPointGroupStats,
    *,
    is_dealer: bool,
) -> dict[str, Any]:
    win_interval = wilson_score_interval(group.wins, group.riichis)
    tsumo_interval = wilson_score_interval(group.tsumo_wins, group.wins)
    mean_interval = group.mean_hand_points_ci95
    mangan_threshold = 12_000 if is_dealer else 8_000
    haneman_threshold = 18_000 if is_dealer else 12_000
    mangan_count = _count_at_least(group.point_counts, mangan_threshold)
    haneman_count = _count_at_least(group.point_counts, haneman_threshold)
    return {
        "riichis": group.riichis,
        "wins": group.wins,
        "other_wins": group.other_wins,
        "draws": group.draws,
        "win_rate": group.win_rate,
        "win_rate_ci95": _interval_to_list(win_interval),
        "tsumo_wins": group.tsumo_wins,
        "ron_wins": group.ron_wins,
        "tsumo_share_among_wins": group.tsumo_share,
        "tsumo_share_ci95": _interval_to_list(tsumo_interval),
        "ron_share_among_wins": group.ron_share,
        "win_methods": {
            "tsumo": {
                "wins": group.tsumo_wins,
                "probability_per_established_riichi": group.tsumo_rate_per_riichi,
                "share_among_wins": group.tsumo_share,
                "hand_points_sum": group.tsumo_hand_points_sum,
                "mean_hand_points": group.mean_tsumo_hand_points,
                "distribution": [
                    {"points": points, "wins": count}
                    for points, count in group.tsumo_point_counts
                ],
            },
            "ron": {
                "wins": group.ron_wins,
                "probability_per_established_riichi": group.ron_rate_per_riichi,
                "share_among_wins": group.ron_share,
                "hand_points_sum": group.ron_hand_points_sum,
                "mean_hand_points": group.mean_ron_hand_points,
                "distribution": [
                    {"points": points, "wins": count}
                    for points, count in group.ron_point_counts
                ],
            },
        },
        "hand_points": {
            "sum": group.hand_points_sum,
            "mean": group.mean_hand_points,
            "mean_ci95": None if mean_interval is None else list(mean_interval),
            "median": group.median_hand_points,
            "minimum": group.minimum_hand_points,
            "maximum": group.maximum_hand_points,
            "per_established_riichi": group.hand_points_per_riichi,
            "mangan_or_higher_count": mangan_count,
            "mangan_or_higher_rate": _ratio(mangan_count, group.wins),
            "haneman_or_higher_count": haneman_count,
            "haneman_or_higher_rate": _ratio(haneman_count, group.wins),
            "distribution": [
                {"points": points, "wins": count}
                for points, count in group.point_counts
            ],
        },
        "settlement_gain": {
            "sum": group.settlement_gain_sum,
            "mean": group.mean_settlement_gain,
        },
        "cluster_games": group.cluster_games,
    }


def _comparison(
    dealer: RiichiPointGroupStats,
    nondealer: RiichiPointGroupStats,
) -> dict[str, Any]:
    dealer_mean = dealer.mean_hand_points
    nondealer_mean = nondealer.mean_hand_points
    ratio = (
        None
        if dealer_mean is None or nondealer_mean in {None, 0.0}
        else dealer_mean / nondealer_mean
    )
    difference = (
        None if nondealer_mean is None else nondealer_mean - NONDEALER_BENCHMARK_POINTS
    )
    interval = nondealer.mean_hand_points_ci95
    return {
        "dealer_to_nondealer_mean_ratio": ratio,
        "dealer_to_nondealer_ratio_minus_1_5": (None if ratio is None else ratio - 1.5),
        "nondealer_benchmark_points": NONDEALER_BENCHMARK_POINTS,
        "nondealer_mean_minus_6500": difference,
        "nondealer_mean_ci95_contains_6500": (
            None
            if interval is None
            else interval[0] <= NONDEALER_BENCHMARK_POINTS <= interval[1]
        ),
    }


def _paths_with_progress(
    year: int,
    paths: Iterable[Path],
    interval: int,
    callback: ProgressCallback | None,
    cancellation_callback: CancellationCallback | None,
) -> Iterable[Path]:
    for processed, path in enumerate(paths, start=1):
        if cancellation_callback is not None and cancellation_callback():
            raise RiichiPointAggregationCancelled(
                f"aggregation cancelled for {year} before source file {processed}"
            )
        yield path
        if callback is not None and processed % interval == 0:
            callback(year, processed)


def _atomic_write(target: Path, content: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary = Path(file.name)
            file.write(content)
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _validate_distribution(
    distribution: tuple[tuple[int, int], ...],
    name: str,
) -> tuple[int, int]:
    if not isinstance(distribution, tuple):
        raise TypeError(f"{name} must be a tuple")
    previous_points = -1
    counted_wins = 0
    counted_points = 0
    for points, count in distribution:
        if type(points) is not int or type(count) is not int:
            raise TypeError(f"{name} values must be integers")
        if points <= 0 or points % 100 != 0 or count <= 0:
            raise ValueError(f"{name} values are invalid")
        if points <= previous_points:
            raise ValueError(f"{name} must be strictly sorted")
        previous_points = points
        counted_wins += count
        counted_points += points * count
    return counted_wins, counted_points


def _value_at_rank(distribution: tuple[tuple[int, int], ...], rank: int) -> int:
    cumulative = 0
    for points, count in distribution:
        cumulative += count
        if rank < cumulative:
            return points
    raise IndexError("distribution rank is out of bounds")


def _count_at_least(
    distribution: tuple[tuple[int, int], ...],
    threshold: int,
) -> int:
    return sum(count for points, count in distribution if points >= threshold)


def _interval_to_list(interval: WilsonInterval | None) -> list[float] | None:
    if interval is None:
        return None
    return [interval.lower, interval.upper]


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be an integer")
    return value


def _actor(value: object, field: str) -> int:
    actor = _integer(value, field)
    if actor not in range(4):
        raise ValueError(f"{field} must be between 0 and 3")
    return actor


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field} must be a non-empty string")
    return value


def _format_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def _format_points(value: float | None) -> str:
    return "N/A" if value is None else f"{value:,.1f}"


def _format_interval(value: list[float] | None) -> str:
    if value is None:
        return "N/A"
    return f"{value[0]:,.1f}–{value[1]:,.1f}"


def _format_decimal(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _format_signed_points(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+,.1f}点"
