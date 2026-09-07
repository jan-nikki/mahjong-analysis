"""MJAI hand replay and established-riichi snapshot extraction."""

from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Literal

from mahjong_analysis.hand_waits import FixedMeld, calculate_hand_waits
from mahjong_analysis.mjai import match_reach_sequence
from mahjong_analysis.tiles import RED_FIVE_NORMALIZATION, normalize_tile

CallType = Literal["chi", "pon", "daiminkan"]
MeldType = Literal["chi", "pon", "daiminkan", "ankan", "kakan"]

_OPEN_MELD_TYPES = frozenset(("chi", "pon", "daiminkan", "kakan"))


@dataclass(frozen=True)
class ActorDiscard:
    """One raw and normalized discard by an actor."""

    discard_number: int
    tile: str
    tile_kind: str
    tsumogiri: bool
    event_index: int
    is_riichi_declaration: bool
    was_called: bool = False
    call_type: CallType | None = None
    called_by_actor: int | None = None
    call_event_index: int | None = None


@dataclass(frozen=True)
class EstablishedRiichi:
    """An established riichi and its declaration-post hand snapshot.

    Event indexes are zero-based indexes within the supplied kyoku list.
    ``fixed_melds`` contains only ankan and can be passed directly to
    ``calculate_hand_waits`` with ``concealed_tiles_after_discard``.
    """

    actor: int
    reach_event_index: int
    declaration_dahai_event_index: int
    reach_accepted_event_index: int
    riichi_discard_number: int
    riichi_declaration_tile: str
    riichi_declaration_tile_kind: str
    concealed_tiles_after_discard: tuple[str, ...]
    fixed_melds: tuple[FixedMeld, ...]
    actor_discards_before_riichi: tuple[ActorDiscard, ...]


@dataclass(frozen=True)
class _Meld:
    meld_type: MeldType
    tiles: tuple[str, ...]
    event_index: int


@dataclass
class _PlayerState:
    concealed_tiles: list[str]
    fixed_melds: list[_Meld]
    discards: list[ActorDiscard]


@dataclass(frozen=True)
class _AcceptedSnapshot:
    actor: int
    reach_event_index: int
    declaration_dahai_event_index: int
    reach_accepted_event_index: int
    riichi_discard_number: int
    riichi_declaration_tile: str
    riichi_declaration_tile_kind: str
    concealed_tiles_after_discard: tuple[str, ...]
    fixed_melds: tuple[FixedMeld, ...]
    discard_count: int


def extract_established_riichis(
    kyoku: list[dict[str, Any]],
) -> tuple[EstablishedRiichi, ...]:
    """Replay one kyoku and return every valid established riichi.

    This boundary validates replay consistency needed by downstream analysis,
    including declaration-post tenpai, but is not a complete action-legality
    validator.
    """
    states = _initial_player_states(kyoku)
    accepted_by_dahai, accepted_by_final = _find_accepted_sequences(kyoku)
    snapshots_by_final: dict[int, _AcceptedSnapshot] = {}
    accepted_snapshots: list[_AcceptedSnapshot] = []
    callable_discard: tuple[int, int, str] | None = None

    for event_index, event in enumerate(kyoku):
        event_type = _event_type(event, event_index)
        if event_type in {"start_kyoku", "end_kyoku"}:
            if event_index not in {0, len(kyoku) - 1}:
                raise ValueError(
                    f"event {event_index}: unexpected {event_type} inside kyoku"
                )
            callable_discard = None
            continue

        if "actor" in event:
            _require_actor(event, event_index)

        if event_type == "tsumo":
            actor = _require_actor(event, event_index)
            tile = _require_tile(event, "pai", event_index, actor)
            states[actor].concealed_tiles.append(tile)
            _validate_owned_tiles(states[actor], event_index, actor)
            callable_discard = None
        elif event_type == "dahai":
            actor = _require_actor(event, event_index)
            tile = _require_tile(event, "pai", event_index, actor)
            tsumogiri = event.get("tsumogiri")
            if type(tsumogiri) is not bool:
                raise _state_error(event_index, actor, "dahai tsumogiri must be bool")
            _remove_concealed_tiles(states[actor], (tile,), event_index, actor)
            _validate_post_discard_count(states[actor], event_index, actor)

            sequence = accepted_by_dahai.get(event_index)
            discard = ActorDiscard(
                discard_number=len(states[actor].discards) + 1,
                tile=tile,
                tile_kind=normalize_tile(tile),
                tsumogiri=tsumogiri,
                event_index=event_index,
                is_riichi_declaration=sequence is not None,
            )
            states[actor].discards.append(discard)
            callable_discard = (actor, event_index, tile)

            if sequence is not None:
                snapshot = _snapshot_after_declaration(
                    states[actor], sequence, discard, event_index
                )
                snapshots_by_final[sequence[3]] = snapshot
        elif event_type in {"chi", "pon", "daiminkan"}:
            actor = _require_actor(event, event_index)
            target = _require_target(event, event_index, actor)
            if event_type == "chi" and actor != (target + 1) % 4:
                raise _state_error(
                    event_index,
                    actor,
                    "chi actor must be the next actor after target",
                )
            called_tile = _require_tile(event, "pai", event_index, actor)
            consumed = _require_consumed(
                event,
                {"chi": 2, "pon": 2, "daiminkan": 3}[event_type],
                event_index,
                actor,
            )
            meld_tiles = (*consumed, called_tile)
            _validate_open_meld(event_type, meld_tiles, event_index, actor)
            _require_callable_discard(
                callable_discard,
                target,
                called_tile,
                event_index,
                actor,
            )
            _remove_concealed_tiles(states[actor], consumed, event_index, actor)
            states[actor].fixed_melds.append(_Meld(event_type, meld_tiles, event_index))
            _validate_owned_tiles(states[actor], event_index, actor)
            _mark_discard_called(
                states[target],
                callable_discard[1],
                event_type,
                actor,
                event_index,
            )
            callable_discard = None
        elif event_type == "ankan":
            actor = _require_actor(event, event_index)
            consumed = _require_consumed(event, 4, event_index, actor)
            _validate_same_kind_meld("ankan", consumed, event_index, actor)
            _remove_concealed_tiles(states[actor], consumed, event_index, actor)
            states[actor].fixed_melds.append(_Meld("ankan", consumed, event_index))
            _validate_owned_tiles(states[actor], event_index, actor)
            callable_discard = None
        elif event_type == "kakan":
            actor = _require_actor(event, event_index)
            added_tile = _require_tile(event, "pai", event_index, actor)
            consumed = _require_consumed(event, 3, event_index, actor)
            _apply_kakan(states[actor], consumed, added_tile, event_index, actor)
            callable_discard = None
        elif event_type == "reach_accepted":
            if event_index not in accepted_by_final:
                actor = event.get("actor")
                raise ValueError(
                    f"event {event_index}, actor {actor}: "
                    "reach_accepted has no matching reach"
                )
            snapshot = snapshots_by_final.get(event_index)
            if snapshot is None:
                raise ValueError(
                    f"event {event_index}: declaration snapshot was not created"
                )
            accepted_snapshots.append(snapshot)
        elif event_type == "reach":
            callable_discard = None
        else:
            callable_discard = None

    return tuple(
        EstablishedRiichi(
            actor=snapshot.actor,
            reach_event_index=snapshot.reach_event_index,
            declaration_dahai_event_index=snapshot.declaration_dahai_event_index,
            reach_accepted_event_index=snapshot.reach_accepted_event_index,
            riichi_discard_number=snapshot.riichi_discard_number,
            riichi_declaration_tile=snapshot.riichi_declaration_tile,
            riichi_declaration_tile_kind=snapshot.riichi_declaration_tile_kind,
            concealed_tiles_after_discard=snapshot.concealed_tiles_after_discard,
            fixed_melds=snapshot.fixed_melds,
            actor_discards_before_riichi=tuple(
                states[snapshot.actor].discards[: snapshot.discard_count]
            ),
        )
        for snapshot in accepted_snapshots
    )


def _initial_player_states(
    kyoku: list[dict[str, Any]],
) -> list[_PlayerState]:
    if not kyoku:
        raise ValueError("kyoku is empty")
    if not isinstance(kyoku[0], dict) or kyoku[0].get("type") != "start_kyoku":
        raise ValueError("kyoku must start with start_kyoku")
    if not isinstance(kyoku[-1], dict) or kyoku[-1].get("type") != "end_kyoku":
        raise ValueError("kyoku must end with end_kyoku")

    tehais = kyoku[0].get("tehais")
    if not isinstance(tehais, list) or len(tehais) != 4:
        raise ValueError("start_kyoku tehais must contain four hands")

    states: list[_PlayerState] = []
    for actor, tehai in enumerate(tehais):
        if not isinstance(tehai, list) or len(tehai) != 13:
            raise ValueError(
                f"event 0, actor {actor}: initial tehai must contain 13 tiles"
            )
        state = _PlayerState(list(tehai), [], [])
        _validate_owned_tiles(state, 0, actor)
        states.append(state)
    return states


def _find_accepted_sequences(
    kyoku: list[dict[str, Any]],
) -> tuple[
    dict[int, tuple[int, int, int, int]],
    dict[int, tuple[int, int, int, int]],
]:
    by_dahai: dict[int, tuple[int, int, int, int]] = {}
    by_final: dict[int, tuple[int, int, int, int]] = {}
    for event_index, event in enumerate(kyoku):
        if not isinstance(event, dict):
            raise TypeError(f"event {event_index}: MJAI event must be an object")
        if event.get("type") != "reach":
            continue
        actor = _require_actor(event, event_index)
        sequence = match_reach_sequence(kyoku, event_index)
        if sequence is None:
            continue
        if sequence[0] != actor:
            raise ValueError(f"event {event_index}: inconsistent reach actor")
        by_dahai[sequence[2]] = sequence
        by_final[sequence[3]] = sequence
    return by_dahai, by_final


def _snapshot_after_declaration(
    state: _PlayerState,
    sequence: tuple[int, int, int, int],
    discard: ActorDiscard,
    event_index: int,
) -> _AcceptedSnapshot:
    actor = sequence[0]
    open_melds = [
        meld.meld_type
        for meld in state.fixed_melds
        if meld.meld_type in _OPEN_MELD_TYPES
    ]
    if open_melds:
        raise _state_error(
            event_index,
            actor,
            f"established riichi actor has open melds: {open_melds}",
        )

    expected_count = 13 - 3 * len(state.fixed_melds)
    if len(state.concealed_tiles) != expected_count:
        raise _state_error(
            event_index,
            actor,
            "declaration-post concealed tile count is invalid: "
            f"expected {expected_count}, got {len(state.concealed_tiles)}",
        )

    fixed_melds = tuple(
        FixedMeld(meld.tiles, meld_type="ankan") for meld in state.fixed_melds
    )
    waits = calculate_hand_waits(state.concealed_tiles, fixed_melds)
    if not waits.wait_tiles:
        raise _state_error(
            event_index,
            actor,
            "established riichi declaration-post hand has no waits",
        )
    return _AcceptedSnapshot(
        actor=actor,
        reach_event_index=sequence[1],
        declaration_dahai_event_index=sequence[2],
        reach_accepted_event_index=sequence[3],
        riichi_discard_number=discard.discard_number,
        riichi_declaration_tile=discard.tile,
        riichi_declaration_tile_kind=discard.tile_kind,
        concealed_tiles_after_discard=tuple(state.concealed_tiles),
        fixed_melds=fixed_melds,
        discard_count=len(state.discards),
    )


def _event_type(event: object, event_index: int) -> str:
    if not isinstance(event, dict):
        raise TypeError(f"event {event_index}: MJAI event must be an object")
    event_type = event.get("type")
    if not isinstance(event_type, str):
        raise TypeError(f"event {event_index}: event type must be a string")
    return event_type


def _require_actor(event: dict[str, Any], event_index: int) -> int:
    actor = event.get("actor")
    if type(actor) is not int or not 0 <= actor <= 3:
        raise ValueError(f"event {event_index}: actor must be an integer from 0 to 3")
    return actor


def _require_target(
    event: dict[str, Any],
    event_index: int,
    actor: int,
) -> int:
    target = event.get("target")
    if type(target) is not int or not 0 <= target <= 3 or target == actor:
        raise _state_error(event_index, actor, "call target is invalid")
    return target


def _require_tile(
    event: dict[str, Any],
    field: str,
    event_index: int,
    actor: int,
) -> str:
    tile = event.get(field)
    if not isinstance(tile, str):
        raise _state_error(event_index, actor, f"{field} must be a tile string")
    try:
        normalize_tile(tile)
    except ValueError as error:
        raise _state_error(event_index, actor, f"invalid {field}: {tile!r}") from error
    return tile


def _require_consumed(
    event: dict[str, Any],
    expected_count: int,
    event_index: int,
    actor: int,
) -> tuple[str, ...]:
    value = event.get("consumed")
    if not isinstance(value, list) or len(value) != expected_count:
        raise _state_error(
            event_index,
            actor,
            f"consumed must contain exactly {expected_count} tiles",
        )
    consumed = tuple(value)
    for tile in consumed:
        if not isinstance(tile, str):
            raise _state_error(event_index, actor, "consumed must contain tile strings")
        try:
            normalize_tile(tile)
        except ValueError as error:
            raise _state_error(
                event_index, actor, f"invalid consumed tile: {tile!r}"
            ) from error
    return consumed


def _remove_concealed_tiles(
    state: _PlayerState,
    tiles: tuple[str, ...],
    event_index: int,
    actor: int,
) -> None:
    available = Counter(state.concealed_tiles)
    needed = Counter(tiles)
    missing = needed - available
    if missing:
        raise _state_error(
            event_index,
            actor,
            f"concealed hand does not contain raw tiles: {dict(missing)}",
        )
    for tile in tiles:
        state.concealed_tiles.remove(tile)


def _validate_owned_tiles(
    state: _PlayerState,
    event_index: int,
    actor: int,
) -> None:
    owned_tiles = [
        *state.concealed_tiles,
        *(tile for meld in state.fixed_melds for tile in meld.tiles),
    ]
    normalized_counts: Counter[str] = Counter()
    red_counts: Counter[str] = Counter()
    for tile in owned_tiles:
        if not isinstance(tile, str):
            raise _state_error(event_index, actor, f"invalid physical tile: {tile!r}")
        try:
            tile_kind = normalize_tile(tile)
        except ValueError as error:
            raise _state_error(
                event_index, actor, f"invalid physical tile: {tile!r}"
            ) from error
        normalized_counts[tile_kind] += 1
        if tile in RED_FIVE_NORMALIZATION:
            red_counts[tile] += 1

    overfull = [tile for tile, count in normalized_counts.items() if count > 4]
    if overfull:
        raise _state_error(
            event_index,
            actor,
            f"more than four owned copies of tile kind: {overfull[0]}",
        )
    duplicate_red = [tile for tile, count in red_counts.items() if count > 1]
    if duplicate_red:
        raise _state_error(
            event_index,
            actor,
            f"multiple physical copies of red five: {duplicate_red[0]}",
        )


def _validate_post_discard_count(
    state: _PlayerState,
    event_index: int,
    actor: int,
) -> None:
    expected_count = 13 - 3 * len(state.fixed_melds)
    if len(state.concealed_tiles) != expected_count:
        raise _state_error(
            event_index,
            actor,
            "post-discard concealed tile count is invalid: "
            f"expected {expected_count}, got {len(state.concealed_tiles)}",
        )


def _validate_open_meld(
    meld_type: str,
    tiles: tuple[str, ...],
    event_index: int,
    actor: int,
) -> None:
    if meld_type == "chi":
        unsorted_indices = tuple(_suited_tile_index(tile) for tile in tiles)
        if any(index is None for index in unsorted_indices):
            raise _state_error(event_index, actor, "chi tiles do not form a sequence")
        indices = sorted(index for index in unsorted_indices if index is not None)
        if (
            len(set(indices)) != 3
            or indices[0] // 9 != indices[-1] // 9
            or indices != list(range(indices[0], indices[0] + 3))
        ):
            raise _state_error(event_index, actor, "chi tiles do not form a sequence")
    else:
        _validate_same_kind_meld(meld_type, tiles, event_index, actor)


def _suited_tile_index(tile: str) -> int | None:
    tile_kind = normalize_tile(tile)
    if tile_kind[-1] not in "mps":
        return None
    suit_offset = {"m": 0, "p": 9, "s": 18}[tile_kind[-1]]
    return suit_offset + int(tile_kind[0]) - 1


def _validate_same_kind_meld(
    meld_type: str,
    tiles: tuple[str, ...],
    event_index: int,
    actor: int,
) -> None:
    if len({normalize_tile(tile) for tile in tiles}) != 1:
        raise _state_error(
            event_index, actor, f"{meld_type} tiles must have one tile kind"
        )


def _apply_kakan(
    state: _PlayerState,
    consumed: tuple[str, ...],
    added_tile: str,
    event_index: int,
    actor: int,
) -> None:
    matching_pon_indices = [
        index
        for index, meld in enumerate(state.fixed_melds)
        if meld.meld_type == "pon" and Counter(meld.tiles) == Counter(consumed)
    ]
    if len(matching_pon_indices) != 1:
        raise _state_error(
            event_index, actor, "kakan has no unique matching pon fixed meld"
        )

    pon_index = matching_pon_indices[0]
    pon = state.fixed_melds[pon_index]
    if normalize_tile(added_tile) != normalize_tile(pon.tiles[0]):
        raise _state_error(event_index, actor, "kakan pai does not match pon")
    _remove_concealed_tiles(state, (added_tile,), event_index, actor)
    state.fixed_melds[pon_index] = _Meld("kakan", (*pon.tiles, added_tile), event_index)
    _validate_owned_tiles(state, event_index, actor)


def _require_callable_discard(
    callable_discard: tuple[int, int, str] | None,
    target: int,
    called_tile: str,
    event_index: int,
    actor: int,
) -> None:
    if (
        callable_discard is None
        or callable_discard[0] != target
        or callable_discard[2] != called_tile
    ):
        raise _state_error(
            event_index,
            actor,
            "call does not match the latest callable raw discard",
        )


def _mark_discard_called(
    target_state: _PlayerState,
    discard_event_index: int,
    call_type: str,
    called_by_actor: int,
    call_event_index: int,
) -> None:
    for index in range(len(target_state.discards) - 1, -1, -1):
        discard = target_state.discards[index]
        if discard.event_index != discard_event_index:
            continue
        target_state.discards[index] = replace(
            discard,
            was_called=True,
            call_type=call_type,
            called_by_actor=called_by_actor,
            call_event_index=call_event_index,
        )
        return
    raise ValueError(f"event {call_event_index}: called discard record was not found")


def _state_error(event_index: int, actor: int, message: str) -> ValueError:
    return ValueError(f"event {event_index}, actor {actor}: {message}")
