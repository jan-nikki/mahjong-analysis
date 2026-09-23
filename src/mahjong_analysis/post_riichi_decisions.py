"""Observer-safe extraction of ordinary draw decisions facing one riichi."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from mahjong_analysis.combo_theory import (
    SimpleComboBreakdown,
    calculate_simple_combo,
)
from mahjong_analysis.danger_features import (
    ConventionalDangerFeatures,
    calculate_conventional_danger_features,
)
from mahjong_analysis.hand_waits import FixedMeld, calculate_hand_waits
from mahjong_analysis.riichi import EstablishedRiichi, extract_established_riichis
from mahjong_analysis.tiles import (
    TILE_KINDS,
    normalize_tile,
    tile_to_index,
    tiles_to_counts,
)


@dataclass(frozen=True)
class CandidateDanger:
    """One normalized discard candidate and its target-riichi labels."""

    tile: str
    simple_combo: SimpleComboBreakdown
    conventional: ConventionalDangerFeatures
    is_structural_wait: bool
    is_ron_eligible: bool


@dataclass(frozen=True)
class PostRiichiDrawDecision:
    """One ordinary draw decision observed before reach or discard is declared."""

    observer_actor: int
    target_actor: int
    bakaze: str
    kyoku_number: int
    honba: int
    kyotaku: int
    oya: int
    scores: tuple[int, int, int, int]
    target_reach_accepted_event_index: int
    target_riichi_discard_number: int
    target_riichi_declaration_tile_kind: str
    target_concealed_tile_count: int
    tsumo_event_index: int
    dahai_event_index: int
    decision_discard_number: int
    post_riichi_decision_number: int
    actual_discard_tile: str
    actual_discard_tile_kind: str
    actual_declared_reach: bool
    observer_concealed_tiles: tuple[str, ...]
    public_visible_counts: tuple[int, ...]
    visible_counts: tuple[int, ...]
    unseen_counts: tuple[int, ...]
    target_own_discard_kinds: tuple[str, ...]
    post_riichi_passed_tile_kinds: tuple[str, ...]
    target_safe_tile_kinds: tuple[str, ...]
    target_wait_tiles: tuple[str, ...]
    target_is_ron_furiten: bool
    candidates: tuple[CandidateDanger, ...]


@dataclass
class _ReplayPlayer:
    concealed_tiles: list[str]
    fixed_melds: list[tuple[str, tuple[str, ...]]]
    discard_kinds: list[str]


@dataclass(frozen=True)
class _TargetInfo:
    riichi: EstablishedRiichi
    wait_tiles: tuple[str, ...]


@dataclass
class _RoundState:
    bakaze: str
    kyoku_number: int
    honba: int
    kyotaku: int
    oya: int
    scores: list[int]


def visible_tile_counts(
    observer_concealed_tiles: tuple[str, ...] | list[str],
    public_visible_tiles: tuple[str, ...] | list[str],
) -> tuple[int, ...]:
    """Count only the observer's hand and public physical tiles."""
    return tiles_to_counts((*observer_concealed_tiles, *public_visible_tiles))


def unseen_tile_counts(visible_counts: tuple[int, ...]) -> tuple[int, ...]:
    """Convert a validated 34-kind visible count vector to unseen counts."""
    if len(visible_counts) != len(TILE_KINDS):
        raise ValueError("visible_counts must contain exactly 34 values")
    result: list[int] = []
    for tile, count in zip(TILE_KINDS, visible_counts, strict=True):
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"visible count for {tile} must be an integer")
        if not 0 <= count <= 4:
            raise ValueError(f"visible count for {tile} must be between 0 and 4")
        result.append(4 - count)
    return tuple(result)


def extract_post_riichi_draw_decisions(
    kyoku: list[dict[str, Any]],
) -> tuple[PostRiichiDrawDecision, ...]:
    """Extract ordinary tsumo decisions facing exactly one accepted riichi.

    Both ``tsumo -> dahai`` and ``tsumo -> reach -> dahai`` are included. Calls
    and kan decisions are outside this first extraction boundary.
    """
    riichis = extract_established_riichis(kyoku)
    riichi_by_accepted = {
        riichi.reach_accepted_event_index: riichi for riichi in riichis
    }
    target_info_by_actor: dict[int, _TargetInfo] = {}
    accepted_actors: set[int] = set()
    passed_wait_after_riichi: dict[int, bool] = {}
    passed_tile_kinds_after_riichi: dict[int, set[str]] = {}
    decision_counts_by_target_observer: Counter[tuple[int, int]] = Counter()
    rinshan_pending_actors: set[int] = set()
    states = _initial_states(kyoku)
    round_state = _initial_round_state(kyoku[0])
    public_visible_tiles = _initial_public_tiles(kyoku[0])
    decisions: list[PostRiichiDrawDecision] = []

    for event_index, event in enumerate(kyoku):
        event_type = _event_type(event, event_index)
        if event_type in {"start_kyoku", "end_kyoku"}:
            continue

        if event_type == "tsumo":
            actor = _actor(event, event_index)
            tile = _tile(event, "pai", event_index)
            states[actor].concealed_tiles.append(tile)
            is_rinshan = actor in rinshan_pending_actors
            rinshan_pending_actors.discard(actor)
            decision_end = _ordinary_draw_decision_end(kyoku, event_index, actor)
            if (
                not is_rinshan
                and decision_end is not None
                and len(accepted_actors) == 1
                and actor not in accepted_actors
            ):
                target_actor = next(iter(accepted_actors))
                target_info = target_info_by_actor[target_actor]
                decision_key = (target_actor, actor)
                decision_counts_by_target_observer[decision_key] += 1
                decisions.append(
                    _build_decision(
                        kyoku=kyoku,
                        observer_actor=actor,
                        target_actor=target_actor,
                        round_state=round_state,
                        target_info=target_info,
                        tsumo_event_index=event_index,
                        dahai_event_index=decision_end[0],
                        post_riichi_decision_number=(
                            decision_counts_by_target_observer[decision_key]
                        ),
                        actual_declared_reach=decision_end[1],
                        observer_state=states[actor],
                        target_state=states[target_actor],
                        public_visible_tiles=public_visible_tiles,
                        target_passed_wait=passed_wait_after_riichi[target_actor],
                        post_riichi_passed_tile_kinds=(
                            passed_tile_kinds_after_riichi[target_actor]
                        ),
                    )
                )
            continue

        if event_type == "dahai":
            actor = _actor(event, event_index)
            tile = _tile(event, "pai", event_index)
            _remove_tiles(states[actor].concealed_tiles, (tile,), event_index, actor)
            tile_kind = normalize_tile(tile)
            states[actor].discard_kinds.append(tile_kind)
            public_visible_tiles.append(tile)

            for target_actor, target_info in target_info_by_actor.items():
                if actor != target_actor:
                    passed_tile_kinds_after_riichi[target_actor].add(tile_kind)
                    if tile_kind in target_info.wait_tiles:
                        passed_wait_after_riichi[target_actor] = True
            if actor in accepted_actors:
                _validate_target_waits(
                    states[actor],
                    target_info_by_actor[actor],
                    event_index,
                )
            continue

        if event_type in {"chi", "pon", "daiminkan"}:
            actor = _actor(event, event_index)
            expected = {"chi": 2, "pon": 2, "daiminkan": 3}[event_type]
            consumed = _consumed(event, expected, event_index)
            _remove_tiles(states[actor].concealed_tiles, consumed, event_index, actor)
            called_tile = _tile(event, "pai", event_index)
            states[actor].fixed_melds.append((event_type, (*consumed, called_tile)))
            public_visible_tiles.extend(consumed)
            if event_type == "daiminkan":
                rinshan_pending_actors.add(actor)
            continue

        if event_type == "ankan":
            actor = _actor(event, event_index)
            consumed = _consumed(event, 4, event_index)
            _remove_tiles(states[actor].concealed_tiles, consumed, event_index, actor)
            states[actor].fixed_melds.append(("ankan", consumed))
            public_visible_tiles.extend(consumed)
            rinshan_pending_actors.add(actor)
            continue

        if event_type == "kakan":
            actor = _actor(event, event_index)
            added_tile = _tile(event, "pai", event_index)
            _remove_tiles(
                states[actor].concealed_tiles,
                (added_tile,),
                event_index,
                actor,
            )
            public_visible_tiles.append(added_tile)
            rinshan_pending_actors.add(actor)
            continue

        if event_type == "dora":
            public_visible_tiles.append(_tile(event, "dora_marker", event_index))
            continue

        if event_type == "reach_accepted":
            actor = _actor(event, event_index)
            riichi = riichi_by_accepted.get(event_index)
            if riichi is None or riichi.actor != actor:
                raise ValueError(
                    f"event {event_index}: accepted riichi snapshot is missing"
                )
            waits = calculate_hand_waits(
                riichi.concealed_tiles_after_discard,
                riichi.fixed_melds,
            )
            target_info_by_actor[actor] = _TargetInfo(riichi, waits.wait_tiles)
            round_state.scores[actor] -= 1000
            round_state.kyotaku += 1
            passed_wait_after_riichi[actor] = False
            passed_tile_kinds_after_riichi[actor] = set()
            accepted_actors.add(actor)

    return tuple(decisions)


def _build_decision(
    *,
    kyoku: list[dict[str, Any]],
    observer_actor: int,
    target_actor: int,
    round_state: _RoundState,
    target_info: _TargetInfo,
    tsumo_event_index: int,
    dahai_event_index: int,
    post_riichi_decision_number: int,
    actual_declared_reach: bool,
    observer_state: _ReplayPlayer,
    target_state: _ReplayPlayer,
    public_visible_tiles: list[str],
    target_passed_wait: bool,
    post_riichi_passed_tile_kinds: set[str],
) -> PostRiichiDrawDecision:
    observer_hand = tuple(observer_state.concealed_tiles)
    public_counts = tiles_to_counts(public_visible_tiles)
    visible_counts = visible_tile_counts(observer_hand, public_visible_tiles)
    unseen_counts = unseen_tile_counts(visible_counts)
    target_wait_set = frozenset(target_info.wait_tiles)
    target_discard_furiten = bool(
        target_wait_set.intersection(target_state.discard_kinds)
    )
    target_is_ron_furiten = target_discard_furiten or target_passed_wait
    target_own_discards = tuple(target_state.discard_kinds)
    passed_tile_kinds = tuple(
        tile for tile in TILE_KINDS if tile in post_riichi_passed_tile_kinds
    )
    target_safe_tiles = tuple(
        tile
        for tile in TILE_KINDS
        if tile in target_state.discard_kinds or tile in post_riichi_passed_tile_kinds
    )
    candidate_tiles = tuple(
        sorted({normalize_tile(tile) for tile in observer_hand}, key=tile_to_index)
    )
    candidates = tuple(
        CandidateDanger(
            tile=tile,
            simple_combo=calculate_simple_combo(tile, unseen_counts),
            conventional=calculate_conventional_danger_features(
                tile,
                visible_counts,
                target_own_discards=target_own_discards,
                post_riichi_passed_tiles=passed_tile_kinds,
                riichi_declaration_tile=(
                    target_info.riichi.riichi_declaration_tile_kind
                ),
            ),
            is_structural_wait=tile in target_wait_set,
            is_ron_eligible=(tile in target_wait_set and not target_is_ron_furiten),
        )
        for tile in candidate_tiles
    )
    actual_dahai = kyoku[dahai_event_index]
    actual_tile = _tile(actual_dahai, "pai", dahai_event_index)
    return PostRiichiDrawDecision(
        observer_actor=observer_actor,
        target_actor=target_actor,
        bakaze=round_state.bakaze,
        kyoku_number=round_state.kyoku_number,
        honba=round_state.honba,
        kyotaku=round_state.kyotaku,
        oya=round_state.oya,
        scores=tuple(round_state.scores),
        target_reach_accepted_event_index=(
            target_info.riichi.reach_accepted_event_index
        ),
        target_riichi_discard_number=target_info.riichi.riichi_discard_number,
        target_riichi_declaration_tile_kind=(
            target_info.riichi.riichi_declaration_tile_kind
        ),
        target_concealed_tile_count=13 - 3 * len(target_state.fixed_melds),
        tsumo_event_index=tsumo_event_index,
        dahai_event_index=dahai_event_index,
        decision_discard_number=len(observer_state.discard_kinds) + 1,
        post_riichi_decision_number=post_riichi_decision_number,
        actual_discard_tile=actual_tile,
        actual_discard_tile_kind=normalize_tile(actual_tile),
        actual_declared_reach=actual_declared_reach,
        observer_concealed_tiles=observer_hand,
        public_visible_counts=public_counts,
        visible_counts=visible_counts,
        unseen_counts=unseen_counts,
        target_own_discard_kinds=target_own_discards,
        post_riichi_passed_tile_kinds=passed_tile_kinds,
        target_safe_tile_kinds=target_safe_tiles,
        target_wait_tiles=target_info.wait_tiles,
        target_is_ron_furiten=target_is_ron_furiten,
        candidates=candidates,
    )


def _ordinary_draw_decision_end(
    kyoku: list[dict[str, Any]],
    tsumo_event_index: int,
    actor: int,
) -> tuple[int, bool] | None:
    next_index = tsumo_event_index + 1
    if next_index >= len(kyoku):
        return None
    next_event = kyoku[next_index]
    if _event_type(next_event, next_index) == "dahai":
        return (next_index, False) if _actor(next_event, next_index) == actor else None
    if _event_type(next_event, next_index) != "reach":
        return None
    if _actor(next_event, next_index) != actor:
        return None
    dahai_index = next_index + 1
    if dahai_index >= len(kyoku):
        return None
    dahai = kyoku[dahai_index]
    if _event_type(dahai, dahai_index) != "dahai":
        return None
    return (dahai_index, True) if _actor(dahai, dahai_index) == actor else None


def _validate_target_waits(
    state: _ReplayPlayer,
    target_info: _TargetInfo,
    event_index: int,
) -> None:
    if any(meld_type != "ankan" for meld_type, _ in state.fixed_melds):
        raise ValueError(f"event {event_index}: accepted riichi actor has open meld")
    fixed_melds = tuple(
        FixedMeld(tiles)
        for meld_type, tiles in state.fixed_melds
        if meld_type == "ankan"
    )
    waits = calculate_hand_waits(state.concealed_tiles, fixed_melds)
    if waits.wait_tiles != target_info.wait_tiles:
        raise ValueError(
            f"event {event_index}: accepted riichi wait changed from "
            f"{target_info.wait_tiles} to {waits.wait_tiles}"
        )


def _initial_states(kyoku: list[dict[str, Any]]) -> list[_ReplayPlayer]:
    if not kyoku or not isinstance(kyoku[0], dict):
        raise ValueError("kyoku must begin with start_kyoku")
    start = kyoku[0]
    if start.get("type") != "start_kyoku":
        raise ValueError("kyoku must begin with start_kyoku")
    tehais = start.get("tehais")
    if not isinstance(tehais, list) or len(tehais) != 4:
        raise ValueError("start_kyoku tehais must contain four hands")
    states: list[_ReplayPlayer] = []
    for actor, hand in enumerate(tehais):
        if not isinstance(hand, list) or len(hand) != 13:
            raise ValueError(f"actor {actor}: initial hand must contain 13 tiles")
        for tile in hand:
            if not isinstance(tile, str):
                raise TypeError(f"actor {actor}: initial hand tiles must be strings")
            normalize_tile(tile)
        states.append(_ReplayPlayer(list(hand), [], []))
    return states


def _initial_public_tiles(start_kyoku: dict[str, Any]) -> list[str]:
    if "dora_marker" not in start_kyoku:
        return []
    marker = start_kyoku["dora_marker"]
    if not isinstance(marker, str):
        raise TypeError("start_kyoku dora_marker must be a tile string")
    normalize_tile(marker)
    return [marker]


def _initial_round_state(start_kyoku: dict[str, Any]) -> _RoundState:
    bakaze = start_kyoku.get("bakaze")
    if not isinstance(bakaze, str):
        raise TypeError("start_kyoku bakaze must be a string")
    kyoku_number = _nonnegative_event_int(start_kyoku, "kyoku")
    if kyoku_number < 1:
        raise ValueError("start_kyoku kyoku must be at least 1")
    honba = _nonnegative_event_int(start_kyoku, "honba")
    kyotaku = _nonnegative_event_int(start_kyoku, "kyotaku")
    oya = start_kyoku.get("oya")
    if type(oya) is not int or oya not in range(4):
        raise ValueError("start_kyoku oya must be from 0 through 3")
    scores = start_kyoku.get("scores")
    if not isinstance(scores, list) or len(scores) != 4:
        raise ValueError("start_kyoku scores must contain four values")
    if any(type(score) is not int for score in scores):
        raise TypeError("start_kyoku scores must contain integers")
    return _RoundState(bakaze, kyoku_number, honba, kyotaku, oya, list(scores))


def _nonnegative_event_int(event: dict[str, Any], field: str) -> int:
    value = event.get(field)
    if type(value) is not int or value < 0:
        raise ValueError(f"start_kyoku {field} must be a non-negative integer")
    return value


def _event_type(event: object, event_index: int) -> str:
    if not isinstance(event, dict):
        raise TypeError(f"event {event_index}: event must be an object")
    event_type = event.get("type")
    if not isinstance(event_type, str):
        raise TypeError(f"event {event_index}: event type must be a string")
    return event_type


def _actor(event: dict[str, Any], event_index: int) -> int:
    actor = event.get("actor")
    if type(actor) is not int or actor not in range(4):
        raise ValueError(f"event {event_index}: actor must be from 0 through 3")
    return actor


def _tile(event: dict[str, Any], field: str, event_index: int) -> str:
    tile = event.get(field)
    if not isinstance(tile, str):
        raise TypeError(f"event {event_index}: {field} must be a tile string")
    normalize_tile(tile)
    return tile


def _consumed(
    event: dict[str, Any],
    expected: int,
    event_index: int,
) -> tuple[str, ...]:
    consumed = event.get("consumed")
    if not isinstance(consumed, list) or len(consumed) != expected:
        raise ValueError(f"event {event_index}: consumed must contain {expected} tiles")
    result: list[str] = []
    for tile in consumed:
        if not isinstance(tile, str):
            raise TypeError(f"event {event_index}: consumed tiles must be strings")
        normalize_tile(tile)
        result.append(tile)
    return tuple(result)


def _remove_tiles(
    concealed_tiles: list[str],
    tiles: tuple[str, ...],
    event_index: int,
    actor: int,
) -> None:
    missing = Counter(tiles) - Counter(concealed_tiles)
    if missing:
        raise ValueError(
            f"event {event_index}, actor {actor}: concealed hand lacks {dict(missing)}"
        )
    for tile in tiles:
        concealed_tiles.remove(tile)
