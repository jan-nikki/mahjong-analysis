"""Lightweight rare-yakuman search over complete MJAI games."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast
from urllib.parse import urlencode
import zlib

from mahjong_analysis.mjai import extract_rule_code, load_mjai, split_kyoku
from mahjong_analysis.tiles import RED_FIVE_NORMALIZATION, normalize_tile

SUPPORTED_YEARS: tuple[int, ...] = tuple(range(2009, 2026))
_GREEN_TILE_KINDS = frozenset({"2s", "3s", "4s", "6s", "8s", "F"})

RareYakumanDetector = Callable[[tuple[str, ...]], bool]


class RareYakumanReplayError(ValueError):
    """Expected validation failure while replaying one MJAI kyoku."""


@dataclass(frozen=True)
class HoraOwnership:
    """One hora event and all physical tiles composing the winner's hand."""

    actor: int
    target: int
    event_index: int
    owned_tiles: tuple[str, ...]


@dataclass(frozen=True)
class RareYakumanHit:
    """One matching kyoku, with every matching hora in that kyoku."""

    source_path: str
    log_id: str
    tenhou_url: str
    kyoku_index: int
    bakaze: str
    kyoku: int
    honba: int
    matching_hora_event_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.source_path:
            raise ValueError("source_path must not be empty")
        if not self.log_id:
            raise ValueError("log_id must not be empty")
        if self.tenhou_url != tenhou_log_url(self.log_id):
            raise ValueError("tenhou_url does not match log_id")
        for name, value in (
            ("kyoku_index", self.kyoku_index),
            ("kyoku", self.kyoku),
            ("honba", self.honba),
        ):
            _require_non_negative_int(value, name)
        if not isinstance(self.bakaze, str) or not self.bakaze:
            raise TypeError("bakaze must be a non-empty string")
        if not isinstance(self.matching_hora_event_indices, tuple) or not (
            self.matching_hora_event_indices
        ):
            raise TypeError("matching_hora_event_indices must be a non-empty tuple")
        previous = -1
        for event_index in self.matching_hora_event_indices:
            _require_non_negative_int(event_index, "matching hora event index")
            if event_index <= previous:
                raise ValueError(
                    "matching hora event indices must be strictly increasing"
                )
            previous = event_index


@dataclass(frozen=True)
class RareYakumanAnomaly:
    """One kyoku that strict replay validation could not analyze."""

    source_path: str
    log_id: str
    kyoku_index: int
    error_message: str

    def __post_init__(self) -> None:
        if not self.source_path:
            raise ValueError("source_path must not be empty")
        if not self.log_id:
            raise ValueError("log_id must not be empty")
        _require_non_negative_int(self.kyoku_index, "kyoku_index")
        if not isinstance(self.error_message, str) or not self.error_message:
            raise TypeError("error_message must be a non-empty string")


@dataclass(frozen=True)
class RareYakumanResult:
    """Aggregate result whose primary observation unit is one kyoku."""

    yaku: str
    total_games: int
    total_kyokus: int
    analyzed_kyokus: int
    anomaly_kyokus: int
    matching_kyokus: int
    matching_hora_events: int
    anomalies: tuple[RareYakumanAnomaly, ...]
    hits: tuple[RareYakumanHit, ...]

    def __post_init__(self) -> None:
        if self.yaku not in DETECTORS:
            raise ValueError(f"unsupported yaku in result: {self.yaku!r}")
        for name, value in (
            ("total_games", self.total_games),
            ("total_kyokus", self.total_kyokus),
            ("analyzed_kyokus", self.analyzed_kyokus),
            ("anomaly_kyokus", self.anomaly_kyokus),
            ("matching_kyokus", self.matching_kyokus),
            ("matching_hora_events", self.matching_hora_events),
        ):
            _require_non_negative_int(value, name)
        if self.analyzed_kyokus + self.anomaly_kyokus != self.total_kyokus:
            raise ValueError("analyzed_kyokus + anomaly_kyokus must equal total_kyokus")
        if self.matching_kyokus > self.analyzed_kyokus:
            raise ValueError("matching_kyokus cannot exceed analyzed_kyokus")
        if not isinstance(self.anomalies, tuple) or not all(
            isinstance(anomaly, RareYakumanAnomaly) for anomaly in self.anomalies
        ):
            raise TypeError("anomalies must be a tuple of RareYakumanAnomaly")
        if self.anomaly_kyokus != len(self.anomalies):
            raise ValueError("anomaly_kyokus must equal the number of anomalies")
        anomaly_keys = [
            (anomaly.source_path, anomaly.kyoku_index) for anomaly in self.anomalies
        ]
        if len(anomaly_keys) != len(set(anomaly_keys)):
            raise ValueError("anomalies must contain at most one entry per kyoku")
        if not isinstance(self.hits, tuple) or not all(
            isinstance(hit, RareYakumanHit) for hit in self.hits
        ):
            raise TypeError("hits must be a tuple of RareYakumanHit")
        if self.matching_kyokus != len(self.hits):
            raise ValueError("matching_kyokus must equal the number of hits")
        if self.matching_hora_events != sum(
            len(hit.matching_hora_event_indices) for hit in self.hits
        ):
            raise ValueError("matching_hora_events does not match hit details")
        hit_keys = [(hit.source_path, hit.kyoku_index) for hit in self.hits]
        if len(hit_keys) != len(set(hit_keys)):
            raise ValueError("hits must contain at most one entry per kyoku")
        if set(hit_keys) & set(anomaly_keys):
            raise ValueError("a kyoku cannot be both a hit and an anomaly")

    @property
    def probability_per_analyzed_kyoku(self) -> float | None:
        """Return matching kyokus divided by successfully analyzed kyokus."""
        if self.analyzed_kyokus == 0:
            return None
        return self.matching_kyokus / self.analyzed_kyokus

    @property
    def one_in_n_analyzed_kyokus(self) -> float | None:
        """Return reciprocal frequency among analyzed kyokus, or None for no hits."""
        if self.matching_kyokus == 0:
            return None
        return self.analyzed_kyokus / self.matching_kyokus


@dataclass(frozen=True)
class _Meld:
    meld_type: str
    tiles: tuple[str, ...]


@dataclass
class _PlayerState:
    concealed: list[str]
    melds: list[_Meld]


@dataclass(frozen=True)
class _ClaimableTile:
    actor: int
    tile: str
    event_index: int
    source: Literal["dahai", "kakan"]


def is_ryuuiisou(owned_tiles: tuple[str, ...]) -> bool:
    """Return whether every physical tile is one of the all-green tile kinds."""
    if not isinstance(owned_tiles, tuple) or not owned_tiles:
        return False
    return all(normalize_tile(tile) in _GREEN_TILE_KINDS for tile in owned_tiles)


DETECTORS: MappingProxyType[str, RareYakumanDetector] = MappingProxyType(
    {"緑一色": is_ryuuiisou}
)


def get_detector(yaku: str) -> RareYakumanDetector:
    """Return a registered detector or reject an unsupported yaku explicitly."""
    try:
        return DETECTORS[yaku]
    except (KeyError, TypeError) as error:
        available = ", ".join(DETECTORS)
        raise ValueError(
            f"unsupported yaku: {yaku!r}; available yaku: {available}"
        ) from error


def extract_complete_log_id(path: str | Path) -> str:
    """Validate an MJAI filename and return its complete Tenhou log ID."""
    source = Path(path)
    extract_rule_code(source)
    return source.name.removesuffix(".mjson")


def tenhou_log_url(log_id: str) -> str:
    """Build the official HTML5 Tenhou viewer URL for a complete log ID."""
    if not isinstance(log_id, str) or not log_id:
        raise TypeError("log_id must be a non-empty string")
    validated = extract_complete_log_id(f"{log_id}.mjson")
    if validated != log_id:
        raise ValueError("log_id is not a complete Tenhou log ID")
    return f"https://tenhou.net/5/?{urlencode({'log': log_id})}"


def replay_kyoku_horas(
    kyoku: Sequence[dict[str, Any]],
) -> tuple[HoraOwnership, ...]:
    """Replay one complete kyoku and snapshot ownership at each hora event."""
    states = _initial_states(kyoku)
    claimable: _ClaimableTile | None = None
    self_draw_ready = [False] * 4
    horas: list[HoraOwnership] = []
    result_type: str | None = None
    hora_actors: set[int] = set()

    for event_index, event in enumerate(kyoku[1:-1], start=1):
        event_type = _event_type(event, event_index)
        if result_type is not None and event_type != "hora":
            raise RareYakumanReplayError(
                f"event {event_index}: {event_type} follows a terminal result"
            )

        if event_type == "tsumo":
            actor = _require_actor(event, event_index)
            tile = _require_tile(event, "pai", event_index, actor)
            states[actor].concealed.append(tile)
            _validate_owned_tiles(states[actor], event_index, actor)
            self_draw_ready[:] = [False] * 4
            self_draw_ready[actor] = True
            claimable = None
        elif event_type == "dahai":
            actor = _require_actor(event, event_index)
            tile = _require_tile(event, "pai", event_index, actor)
            if type(event.get("tsumogiri")) is not bool:
                raise _event_error(event_index, actor, "dahai tsumogiri must be bool")
            _remove_concealed(states[actor], (tile,), event_index, actor)
            _validate_all_post_discard_counts(states, event_index)
            self_draw_ready[actor] = False
            claimable = _ClaimableTile(actor, tile, event_index, "dahai")
        elif event_type in {"chi", "pon", "daiminkan"}:
            actor = _require_actor(event, event_index)
            target = _require_target(event, event_index, actor)
            tile = _require_tile(event, "pai", event_index, actor)
            if event_type == "chi" and actor != (target + 1) % 4:
                raise _event_error(
                    event_index, actor, "chi actor must immediately follow target"
                )
            consumed = _require_consumed(
                event,
                {"chi": 2, "pon": 2, "daiminkan": 3}[event_type],
                event_index,
                actor,
            )
            _require_claimable(claimable, target, tile, event_index, actor)
            meld_tiles = (*consumed, tile)
            _validate_meld(event_type, meld_tiles, event_index, actor)
            _remove_concealed(states[actor], consumed, event_index, actor)
            states[actor].melds.append(_Meld(event_type, meld_tiles))
            _validate_owned_tiles(states[actor], event_index, actor)
            self_draw_ready[actor] = False
            claimable = None
        elif event_type == "ankan":
            actor = _require_actor(event, event_index)
            consumed = _require_consumed(event, 4, event_index, actor)
            _validate_same_kind("ankan", consumed, event_index, actor)
            _remove_concealed(states[actor], consumed, event_index, actor)
            states[actor].melds.append(_Meld("ankan", consumed))
            _validate_owned_tiles(states[actor], event_index, actor)
            self_draw_ready[actor] = False
            claimable = None
        elif event_type == "kakan":
            actor = _require_actor(event, event_index)
            tile = _require_tile(event, "pai", event_index, actor)
            consumed = _require_consumed(event, 3, event_index, actor)
            _apply_kakan(states[actor], consumed, tile, event_index, actor)
            self_draw_ready[actor] = False
            claimable = _ClaimableTile(actor, tile, event_index, "kakan")
        elif event_type == "hora":
            if result_type == "ryukyoku":
                raise RareYakumanReplayError(
                    f"event {event_index}: hora follows ryukyoku"
                )
            result_type = "hora"
            ownership = _hora_ownership(
                states,
                event,
                event_index,
                claimable,
                self_draw_ready,
            )
            if ownership.actor == ownership.target:
                self_draw_ready[ownership.actor] = False
            if ownership.actor in hora_actors:
                raise _event_error(
                    event_index, ownership.actor, "actor has duplicate hora events"
                )
            hora_actors.add(ownership.actor)
            horas.append(ownership)
        elif event_type == "ryukyoku":
            if result_type is not None:
                raise RareYakumanReplayError(
                    f"event {event_index}: duplicate terminal result"
                )
            result_type = "ryukyoku"
            claimable = None
        elif event_type == "reach":
            _require_actor(event, event_index)
            claimable = None
        elif event_type == "reach_accepted":
            _require_actor(event, event_index)
            # The declaration discard remains callable after acceptance.
        elif event_type == "dora":
            claimable = None
        elif event_type in {"start_kyoku", "end_kyoku"}:
            raise RareYakumanReplayError(
                f"event {event_index}: unexpected {event_type}"
            )
        else:
            raise RareYakumanReplayError(
                f"event {event_index}: unsupported event type {event_type!r}"
            )

    if result_type is None:
        raise RareYakumanReplayError("kyoku has no hora or ryukyoku result")
    return tuple(horas)


def analyze_rare_yakuman_game(
    path: str | Path,
    *,
    raw_root: str | Path,
    yaku: str,
    continue_on_replay_error: bool = False,
) -> RareYakumanResult:
    """Analyze one source game without filtering by wind, rule code, or aka flag."""
    detector = get_detector(yaku)
    source = Path(path).resolve()
    root = Path(raw_root).resolve()
    try:
        relative_source = source.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("source path must be below raw_root") from error

    log_id = extract_complete_log_id(source)
    try:
        events = load_mjai(source)
        _validate_game_events(events)
        kyokus = split_kyoku(events)
    except (
        OSError,
        EOFError,
        zlib.error,
        UnicodeError,
        ValueError,
        TypeError,
        KeyError,
    ) as error:
        raise ValueError(f"{relative_source}: {error}") from error
    if type(continue_on_replay_error) is not bool:
        raise TypeError("continue_on_replay_error must be bool")
    anomalies: list[RareYakumanAnomaly] = []
    hits: list[RareYakumanHit] = []
    for kyoku_index, kyoku_events in enumerate(kyokus):
        try:
            start = kyoku_events[0]
            bakaze = _require_non_empty_string(start.get("bakaze"), "bakaze")
            kyoku_number = _require_event_non_negative_int(start.get("kyoku"), "kyoku")
            honba = _require_event_non_negative_int(start.get("honba"), "honba")
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(
                f"{relative_source}, kyoku_index={kyoku_index}: {error}"
            ) from error
        try:
            horas = replay_kyoku_horas(kyoku_events)
        except RareYakumanReplayError as error:
            if continue_on_replay_error:
                anomalies.append(
                    RareYakumanAnomaly(
                        source_path=relative_source,
                        log_id=log_id,
                        kyoku_index=kyoku_index,
                        error_message=str(error),
                    )
                )
                continue
            raise ValueError(
                f"{relative_source}, kyoku_index={kyoku_index}: {error}"
            ) from error
        try:
            matching = tuple(hora for hora in horas if detector(hora.owned_tiles))
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(
                f"{relative_source}, kyoku_index={kyoku_index}: {error}"
            ) from error
        if not matching:
            continue
        hits.append(
            RareYakumanHit(
                source_path=relative_source,
                log_id=log_id,
                tenhou_url=tenhou_log_url(log_id),
                kyoku_index=kyoku_index,
                bakaze=bakaze,
                kyoku=kyoku_number,
                honba=honba,
                matching_hora_event_indices=tuple(
                    hora.event_index for hora in matching
                ),
            )
        )
    return RareYakumanResult(
        yaku=yaku,
        total_games=1,
        total_kyokus=len(kyokus),
        analyzed_kyokus=len(kyokus) - len(anomalies),
        anomaly_kyokus=len(anomalies),
        matching_kyokus=len(hits),
        matching_hora_events=sum(len(hit.matching_hora_event_indices) for hit in hits),
        anomalies=tuple(anomalies),
        hits=tuple(hits),
    )


def scan_rare_yakuman(
    raw_root: str | Path,
    *,
    yaku: str,
    workers: int = 1,
    years: Sequence[int] = SUPPORTED_YEARS,
    continue_on_replay_error: bool = False,
) -> RareYakumanResult:
    """Scan selected year directories and merge deterministic annual results."""
    get_detector(yaku)
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if type(continue_on_replay_error) is not bool:
        raise TypeError("continue_on_replay_error must be bool")
    selected_years = tuple(years)
    if not selected_years:
        raise ValueError("years must not be empty")
    if any(type(year) is not int for year in selected_years):
        raise TypeError("years must contain integers")
    if len(set(selected_years)) != len(selected_years):
        raise ValueError("years must not contain duplicates")
    root = Path(raw_root).resolve()
    missing = [year for year in selected_years if not (root / str(year)).is_dir()]
    if missing:
        raise FileNotFoundError(f"raw year directories do not exist: {missing}")

    tasks = tuple(
        (str(root), year, yaku, continue_on_replay_error)
        for year in sorted(selected_years)
    )
    if workers == 1:
        annual = tuple(_scan_year(task) for task in tasks)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            annual = tuple(executor.map(_scan_year, tasks))
    return _merge_results(yaku, annual)


def result_to_dict(result: RareYakumanResult) -> dict[str, Any]:
    """Return the compact JSON-compatible public result."""
    return {
        "yaku": result.yaku,
        "total_games": result.total_games,
        "total_kyokus": result.total_kyokus,
        "analyzed_kyokus": result.analyzed_kyokus,
        "anomaly_kyokus": result.anomaly_kyokus,
        "matching_kyokus": result.matching_kyokus,
        "matching_hora_events": result.matching_hora_events,
        "probability_per_analyzed_kyoku": result.probability_per_analyzed_kyoku,
        "one_in_n_analyzed_kyokus": result.one_in_n_analyzed_kyokus,
        "anomalies": [
            {
                "source_path": anomaly.source_path,
                "log_id": anomaly.log_id,
                "kyoku_index": anomaly.kyoku_index,
                "error_message": anomaly.error_message,
            }
            for anomaly in result.anomalies
        ],
        "hits": [
            {
                "source_path": hit.source_path,
                "log_id": hit.log_id,
                "tenhou_url": hit.tenhou_url,
                "kyoku_index": hit.kyoku_index,
                "start_kyoku": {
                    "bakaze": hit.bakaze,
                    "kyoku": hit.kyoku,
                    "honba": hit.honba,
                },
                "matching_hora_event_indices": list(hit.matching_hora_event_indices),
            }
            for hit in result.hits
        ],
    }


def _scan_year(task: tuple[str, int, str, bool]) -> RareYakumanResult:
    raw_root_text, year, yaku, continue_on_replay_error = task
    raw_root = Path(raw_root_text)
    year_root = raw_root / str(year)
    paths = sorted(
        year_root.rglob("*.mjson"),
        key=lambda path: path.resolve().relative_to(raw_root).as_posix(),
    )
    total_kyokus = 0
    analyzed_kyokus = 0
    anomalies: list[RareYakumanAnomaly] = []
    hits: list[RareYakumanHit] = []
    for path in paths:
        result = analyze_rare_yakuman_game(
            path,
            raw_root=raw_root,
            yaku=yaku,
            continue_on_replay_error=continue_on_replay_error,
        )
        total_kyokus += result.total_kyokus
        analyzed_kyokus += result.analyzed_kyokus
        anomalies.extend(result.anomalies)
        hits.extend(result.hits)
    return RareYakumanResult(
        yaku=yaku,
        total_games=len(paths),
        total_kyokus=total_kyokus,
        analyzed_kyokus=analyzed_kyokus,
        anomaly_kyokus=len(anomalies),
        matching_kyokus=len(hits),
        matching_hora_events=sum(len(hit.matching_hora_event_indices) for hit in hits),
        anomalies=tuple(anomalies),
        hits=tuple(hits),
    )


def _merge_results(
    yaku: str, results: Iterable[RareYakumanResult]
) -> RareYakumanResult:
    materialized = tuple(results)
    if any(result.yaku != yaku for result in materialized):
        raise ValueError("cannot merge results for different yaku")
    hits = tuple(
        sorted(
            (hit for result in materialized for hit in result.hits),
            key=lambda hit: (
                hit.source_path,
                hit.kyoku_index,
                hit.matching_hora_event_indices,
            ),
        )
    )
    anomalies = tuple(
        sorted(
            (anomaly for result in materialized for anomaly in result.anomalies),
            key=lambda anomaly: (
                anomaly.source_path,
                anomaly.kyoku_index,
                anomaly.error_message,
            ),
        )
    )
    return RareYakumanResult(
        yaku=yaku,
        total_games=sum(result.total_games for result in materialized),
        total_kyokus=sum(result.total_kyokus for result in materialized),
        analyzed_kyokus=sum(result.analyzed_kyokus for result in materialized),
        anomaly_kyokus=len(anomalies),
        matching_kyokus=len(hits),
        matching_hora_events=sum(len(hit.matching_hora_event_indices) for hit in hits),
        anomalies=anomalies,
        hits=hits,
    )


def _initial_states(kyoku: Sequence[dict[str, Any]]) -> list[_PlayerState]:
    if not isinstance(kyoku, Sequence) or isinstance(kyoku, (str, bytes)):
        raise RareYakumanReplayError("kyoku must be a sequence of events")
    if len(kyoku) < 3:
        raise RareYakumanReplayError("kyoku must contain start, result, and end events")
    if not isinstance(kyoku[0], dict) or kyoku[0].get("type") != "start_kyoku":
        raise RareYakumanReplayError("kyoku must start with start_kyoku")
    if not isinstance(kyoku[-1], dict) or kyoku[-1].get("type") != "end_kyoku":
        raise RareYakumanReplayError("kyoku must end with end_kyoku")
    tehais = kyoku[0].get("tehais")
    if not isinstance(tehais, list) or len(tehais) != 4:
        raise RareYakumanReplayError("start_kyoku tehais must contain four hands")
    states: list[_PlayerState] = []
    for actor, tehai in enumerate(tehais):
        if not isinstance(tehai, list) or len(tehai) != 13:
            raise RareYakumanReplayError(
                f"event 0, actor {actor}: initial hand must have 13 tiles"
            )
        state = _PlayerState(list(tehai), [])
        _validate_owned_tiles(state, 0, actor)
        states.append(state)
    return states


def _validate_game_events(events: list[dict[str, Any]]) -> None:
    if len(events) < 2:
        raise ValueError("game must contain start_game and end_game")
    event_types = tuple(_event_type(event, index) for index, event in enumerate(events))
    if event_types[0] != "start_game":
        raise ValueError("game must start with start_game")
    if event_types[-1] != "end_game":
        raise ValueError("game must end with end_game")
    if event_types.count("start_game") != 1 or event_types.count("end_game") != 1:
        raise ValueError("game must contain exactly one start_game and one end_game")


def _hora_ownership(
    states: list[_PlayerState],
    event: dict[str, Any],
    event_index: int,
    claimable: _ClaimableTile | None,
    self_draw_ready: Sequence[bool],
) -> HoraOwnership:
    actor = _require_actor(event, event_index)
    target = _require_target(event, event_index, actor, allow_self=True)
    state = states[actor]
    if actor == target:
        if not self_draw_ready[actor]:
            raise _event_error(
                event_index, actor, "tsumo hora requires an unconsumed tsumo event"
            )
        expected = 14 - 3 * len(state.melds)
        if len(state.concealed) != expected:
            raise _event_error(
                event_index,
                actor,
                f"tsumo hora concealed count must be {expected}, "
                f"got {len(state.concealed)}",
            )
        winning_tiles: tuple[str, ...] = ()
    else:
        expected = 13 - 3 * len(state.melds)
        if len(state.concealed) != expected:
            raise _event_error(
                event_index,
                actor,
                f"ron hora concealed count must be {expected}, "
                f"got {len(state.concealed)}",
            )
        if claimable is None or claimable.actor != target:
            raise _event_error(
                event_index, actor, "ron target does not match the claimable tile"
            )
        winning_tiles = (claimable.tile,)
    owned = (
        *state.concealed,
        *(tile for meld in state.melds for tile in meld.tiles),
        *winning_tiles,
    )
    _validate_physical_tiles(owned, event_index, actor)
    return HoraOwnership(actor, target, event_index, tuple(owned))


def _apply_kakan(
    state: _PlayerState,
    consumed: tuple[str, ...],
    added_tile: str,
    event_index: int,
    actor: int,
) -> None:
    matches = [
        index
        for index, meld in enumerate(state.melds)
        if meld.meld_type == "pon" and Counter(meld.tiles) == Counter(consumed)
    ]
    if len(matches) != 1:
        raise _event_error(event_index, actor, "kakan has no unique matching pon")
    index = matches[0]
    pon = state.melds[index]
    if normalize_tile(pon.tiles[0]) != normalize_tile(added_tile):
        raise _event_error(event_index, actor, "kakan pai does not match pon")
    _remove_concealed(state, (added_tile,), event_index, actor)
    state.melds[index] = _Meld("kakan", (*pon.tiles, added_tile))
    _validate_owned_tiles(state, event_index, actor)


def _validate_meld(
    meld_type: str,
    tiles: tuple[str, ...],
    event_index: int,
    actor: int,
) -> None:
    if meld_type == "chi":
        unsorted_indices = tuple(_suited_index(tile) for tile in tiles)
        if any(index is None for index in unsorted_indices):
            raise _event_error(event_index, actor, "chi tiles do not form a sequence")
        indices = sorted(cast(int, index) for index in unsorted_indices)
        if (
            len(set(indices)) != 3
            or indices[0] // 9 != indices[-1] // 9
            or indices != list(range(indices[0], indices[0] + 3))
        ):
            raise _event_error(event_index, actor, "chi tiles do not form a sequence")
    else:
        _validate_same_kind(meld_type, tiles, event_index, actor)


def _suited_index(tile: str) -> int | None:
    normalized = normalize_tile(tile)
    if normalized[-1] not in "mps":
        return None
    return {"m": 0, "p": 9, "s": 18}[normalized[-1]] + int(normalized[0]) - 1


def _validate_same_kind(
    meld_type: str,
    tiles: tuple[str, ...],
    event_index: int,
    actor: int,
) -> None:
    if len({normalize_tile(tile) for tile in tiles}) != 1:
        raise _event_error(
            event_index, actor, f"{meld_type} tiles must have one tile kind"
        )


def _remove_concealed(
    state: _PlayerState,
    tiles: tuple[str, ...],
    event_index: int,
    actor: int,
) -> None:
    missing = Counter(tiles) - Counter(state.concealed)
    if missing:
        raise _event_error(
            event_index,
            actor,
            f"concealed hand lacks raw tiles: {dict(missing)}",
        )
    for tile in tiles:
        state.concealed.remove(tile)


def _validate_owned_tiles(state: _PlayerState, event_index: int, actor: int) -> None:
    _validate_physical_tiles(
        (
            *state.concealed,
            *(tile for meld in state.melds for tile in meld.tiles),
        ),
        event_index,
        actor,
    )


def _validate_physical_tiles(
    tiles: Iterable[object], event_index: int, actor: int
) -> None:
    normalized: Counter[str] = Counter()
    reds: Counter[str] = Counter()
    for tile in tiles:
        if not isinstance(tile, str):
            raise _event_error(event_index, actor, f"invalid physical tile: {tile!r}")
        try:
            tile_kind = normalize_tile(tile)
        except ValueError as error:
            raise _event_error(
                event_index, actor, f"invalid physical tile: {tile!r}"
            ) from error
        normalized[tile_kind] += 1
        if tile in RED_FIVE_NORMALIZATION:
            reds[tile] += 1
    overfull = next((tile for tile, count in normalized.items() if count > 4), None)
    if overfull is not None:
        raise _event_error(
            event_index, actor, f"more than four owned copies of {overfull}"
        )
    duplicate_red = next((tile for tile, count in reds.items() if count > 1), None)
    if duplicate_red is not None:
        raise _event_error(
            event_index, actor, f"multiple physical copies of {duplicate_red}"
        )


def _validate_all_post_discard_counts(
    states: list[_PlayerState], event_index: int
) -> None:
    for actor, state in enumerate(states):
        expected = 13 - 3 * len(state.melds)
        if len(state.concealed) != expected:
            raise _event_error(
                event_index,
                actor,
                f"post-discard concealed count must be {expected}, "
                f"got {len(state.concealed)}",
            )


def _require_claimable(
    claimable: _ClaimableTile | None,
    target: int,
    tile: str,
    event_index: int,
    actor: int,
) -> None:
    if (
        claimable is None
        or claimable.source != "dahai"
        or claimable.actor != target
        or claimable.tile != tile
    ):
        raise _event_error(
            event_index, actor, "call does not match the latest callable discard"
        )


def _require_consumed(
    event: dict[str, Any],
    expected: int,
    event_index: int,
    actor: int,
) -> tuple[str, ...]:
    value = event.get("consumed")
    if not isinstance(value, list) or len(value) != expected:
        raise _event_error(
            event_index,
            actor,
            f"consumed must contain exactly {expected} tiles",
        )
    consumed = tuple(value)
    for tile in consumed:
        if not isinstance(tile, str):
            raise _event_error(event_index, actor, "consumed must contain strings")
        try:
            normalize_tile(tile)
        except ValueError as error:
            raise _event_error(
                event_index, actor, f"invalid consumed tile: {tile!r}"
            ) from error
    return consumed


def _require_tile(
    event: dict[str, Any], field: str, event_index: int, actor: int
) -> str:
    value = event.get(field)
    if not isinstance(value, str):
        raise _event_error(event_index, actor, f"{field} must be a tile string")
    try:
        normalize_tile(value)
    except ValueError as error:
        raise _event_error(event_index, actor, f"invalid {field}: {value!r}") from error
    return value


def _require_actor(event: dict[str, Any], event_index: int) -> int:
    value = event.get("actor")
    if type(value) is not int or not 0 <= value <= 3:
        raise RareYakumanReplayError(
            f"event {event_index}: actor must be an integer from 0 to 3"
        )
    return value


def _require_target(
    event: dict[str, Any],
    event_index: int,
    actor: int,
    *,
    allow_self: bool = False,
) -> int:
    value = event.get("target")
    if type(value) is not int or not 0 <= value <= 3:
        raise _event_error(event_index, actor, "target must be an integer from 0 to 3")
    if not allow_self and value == actor:
        raise _event_error(event_index, actor, "call target must differ from actor")
    return value


def _event_type(event: dict[str, Any], event_index: int) -> str:
    if not isinstance(event, dict):
        raise RareYakumanReplayError(f"event {event_index}: event must be a dict")
    value = event.get("type")
    if not isinstance(value, str) or not value:
        raise RareYakumanReplayError(
            f"event {event_index}: type must be a non-empty string"
        )
    return value


def _event_error(event_index: int, actor: int, message: str) -> RareYakumanReplayError:
    return RareYakumanReplayError(f"event {event_index}, actor {actor}: {message}")


def _require_non_negative_int(value: object, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_event_non_negative_int(value: object, name: str) -> int:
    _require_non_negative_int(value, name)
    return cast(int, value)


def _require_non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value
