"""Observed child-responder behavior after the first established riichi."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from mahjong_analysis.dealer_child_riichi_points import (
    ReachType,
    extract_established_riichis,
)
from mahjong_analysis.mjai import match_reach_sequence
from mahjong_analysis.riichi_wait_dataset import RiichiWaitDatasetRecord
from mahjong_analysis.tiles import TILE_KINDS, normalize_tile

FocalRole = Literal["dealer", "nondealer"]
SafetyClass = Literal[
    "genbutsu",
    "full_suji",
    "partial_suji",
    "unsuji_numbered",
    "honor_non_genbutsu",
]
SAFETY_CLASSES: tuple[SafetyClass, ...] = (
    "genbutsu",
    "full_suji",
    "partial_suji",
    "unsuji_numbered",
    "honor_non_genbutsu",
)
WaitClass = Literal[
    "pure_ryanmen",
    "contains_ryanmen_other",
    "multiwait_other",
    "single_other",
]
Outcome = Literal[
    "focal_tsumo",
    "focal_ron",
    "other_win",
    "draw",
    "censored_second_reach",
]
WindowEndReason = Literal["hand_end", "second_reach_accepted"]
TerminalOutcome = Literal["focal_tsumo", "focal_ron", "other_win", "draw"]

ROLES: tuple[FocalRole, ...] = ("dealer", "nondealer")
DECISION_METRICS = (
    "genbutsu_share",
    "unsuji_numbered_share",
    "unsuji_among_numbered_non_genbutsu",
    "tsumogiri_share",
    "focal_ron_per_discard",
)
RESPONDER_METRICS = ("any_call_share", "later_riichi_share")
DECISION_STRATUM_FIELDS = (
    "year",
    "focal_turn_bin",
    "wait_class",
    "wait_count_bin",
    "relative_seat",
    "responder_open_at_focal",
    "responder_open_at_decision",
    "decision_ordinal_bin",
    "safe_kind_count_bin",
)
RESPONDER_STRATUM_FIELDS = (
    "year",
    "focal_turn_bin",
    "wait_class",
    "wait_count_bin",
    "relative_seat",
    "responder_open_at_focal",
)
OPEN_CALL_TYPES = frozenset(("chi", "pon", "daiminkan", "kakan"))
ALL_CALL_TYPES = OPEN_CALL_TYPES | {"ankan"}
AUDIT_LIMIT = 32
DECIMAL_PRECISION_DIGITS = 60
DECIMAL_ROUNDING = ROUND_HALF_EVEN
DECIMAL_MIN_EXPONENT = -999_999
DECIMAL_MAX_EXPONENT = 999_999
WEIGHTED_SUM_REPRESENTATION = "weighted_sum_of_exact_stratum_counts"


@dataclass(frozen=True)
class BinaryCount:
    """Exact numerator/denominator pair."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if type(self.numerator) is not int or type(self.denominator) is not int:
            raise TypeError("binary counts must be integers")
        if not 0 <= self.numerator <= self.denominator:
            raise ValueError("binary count must satisfy 0 <= numerator <= denominator")

    @property
    def rate(self) -> Fraction | None:
        return (
            None
            if self.denominator == 0
            else Fraction(self.numerator, self.denominator)
        )


@dataclass(frozen=True)
class StandardizedRates:
    """Pooled-denominator direct standardization on exact common support."""

    dealer_rate: Decimal | None
    nondealer_rate: Decimal | None
    difference: Decimal | None
    common_strata: int
    union_strata: int
    dealer_coverage: Fraction | None
    nondealer_coverage: Fraction | None
    dealer_total_eligible_denominator: int
    nondealer_total_eligible_denominator: int
    dealer_common_denominator: int
    nondealer_common_denominator: int
    dealer_outside_denominator: int
    nondealer_outside_denominator: int
    dealer_nonempty_strata: int
    nondealer_nonempty_strata: int
    common_pooled_denominator: int
    status: Literal["established", "not_estimable"]


@dataclass(frozen=True)
class DiscardDecision:
    """One observed opponent discard after the focal riichi was accepted."""

    year: int
    source_path: str
    start_kyoku_line: int
    event_index: int
    focal_actor: int
    responder: int
    focal_role: FocalRole
    reach_type: ReachType
    primary_child_responder: bool
    tile: str
    tile_kind: str
    safety_class: SafetyClass
    tsumogiri: bool
    focal_ron_on_discard: bool
    relative_seat: int
    focal_turn_bin: str
    wait_class: WaitClass
    wait_count_bin: str
    focal_furiten: bool
    responder_open_at_focal: bool
    responder_open_at_decision: bool
    decision_ordinal_bin: str
    decision_ordinal: int
    safe_kind_count_bin: str
    safe_kinds_before: tuple[str, ...]


@dataclass(frozen=True)
class ResponderObservation:
    """One responder followed until censoring or the hand result."""

    year: int
    focal_actor: int
    responder: int
    focal_role: FocalRole
    reach_type: ReachType
    primary_child_responder: bool
    relative_seat: int
    focal_turn_bin: str
    wait_class: WaitClass
    wait_count_bin: str
    responder_open_at_focal: bool
    any_call: bool
    later_established_riichi: bool
    discard_decisions: int


@dataclass(frozen=True)
class RonDetail:
    """Observed discard on which the focal actor won by ron."""

    responder: int
    relative_seat: int
    responder_open: bool
    safety_class: SafetyClass
    tsumogiri: bool
    event_index: int


@dataclass(frozen=True)
class _TerminalSummary:
    outcome: TerminalOutcome
    hora_events: int
    multi_ron: bool
    focal_ron_without_discard: bool


@dataclass(frozen=True)
class _SecondReachDeclaration:
    established: bool = False
    unestablished: bool = False
    ron_events: int = 0
    focal_ron: bool = False


@dataclass(frozen=True)
class FocalObservation:
    """Bounded per-hand output consumed immediately by the streaming builder."""

    year: int
    source_path: str
    start_kyoku_line: int
    kyoku: int
    honba: int
    oya: int
    focal_actor: int
    reach_accepted_event_index: int
    focal_role: FocalRole
    reach_type: ReachType
    focal_turn_bin: str
    wait_class: WaitClass
    wait_count_bin: str
    focal_furiten: bool
    decisions: tuple[DiscardDecision, ...]
    responders: tuple[ResponderObservation, ...]
    focal_draw_opportunities: int
    primary_child_discard_exposures: int
    all_responder_discard_exposures: int
    window_end_reason: WindowEndReason
    terminal_outcome: TerminalOutcome
    censor_actor: int | None
    ron_details: tuple[RonDetail, ...]
    terminal_hora_events: int
    terminal_multi_ron: bool
    second_reach_declaration_established: bool
    second_reach_declaration_unestablished: bool
    ron_events_on_second_reach_declaration: int
    focal_ron_on_second_reach_declaration: bool
    focal_ron_without_discard: bool

    @property
    def outcome(self) -> Outcome:
        """Backward-compatible window outcome used by the initial API."""
        if self.window_end_reason == "second_reach_accepted":
            return "censored_second_reach"
        return self.terminal_outcome


def relative_seat(focal_actor: int, responder: int) -> int:
    """Return responder position clockwise from the focal actor."""
    _validate_actor(focal_actor, "focal_actor")
    _validate_actor(responder, "responder")
    value = (responder - focal_actor) % 4
    if value == 0:
        raise ValueError("focal actor cannot respond to itself")
    return value


def focal_turn_bin(discard_number: int) -> str:
    if type(discard_number) is not int or discard_number < 1:
        raise ValueError("riichi discard number must be a positive integer")
    if discard_number <= 5:
        return "1-5"
    if discard_number <= 8:
        return "6-8"
    if discard_number <= 11:
        return "9-11"
    return "12+"


def wait_count_bin(wait_count: int) -> str:
    if type(wait_count) is not int or wait_count < 1:
        raise ValueError("wait count must be a positive integer")
    return "1" if wait_count == 1 else "2" if wait_count == 2 else "3+"


def classify_wait(record: RiichiWaitDatasetRecord) -> WaitClass:
    if record.is_pure_ryanmen:
        return "pure_ryanmen"
    if record.contains_ryanmen:
        return "contains_ryanmen_other"
    if record.wait_tile_count > 1:
        return "multiwait_other"
    return "single_other"


def safe_kind_count_bin(count: int) -> str:
    if type(count) is not int or count < 0:
        raise ValueError("safe kind count must be a non-negative integer")
    if count <= 3:
        return "0-3"
    if count <= 7:
        return "4-7"
    if count <= 11:
        return "8-11"
    return "12+"


def decision_ordinal_bin(ordinal: int) -> str:
    if type(ordinal) is not int or ordinal < 1:
        raise ValueError("decision ordinal must be a positive integer")
    return str(ordinal) if ordinal <= 2 else "3+"


def classify_discard(tile: str, safe_kinds: Iterable[str]) -> SafetyClass:
    """Classify a discard against the safe set existing before that discard."""
    tile_kind = normalize_tile(tile)
    safe = {normalize_tile(value) for value in safe_kinds}
    if tile_kind in safe:
        return "genbutsu"
    if tile_kind in TILE_KINDS[27:]:
        return "honor_non_genbutsu"
    suit = tile_kind[-1]
    rank = int(tile_kind[0])
    related_ranks: tuple[int, ...]
    if rank <= 3:
        related_ranks = (rank + 3,)
    elif rank >= 7:
        related_ranks = (rank - 3,)
    else:
        related_ranks = (rank - 3, rank + 3)
    related = {f"{value}{suit}" for value in related_ranks}
    passed = len(related & safe)
    if passed == len(related):
        return "full_suji"
    if passed:
        return "partial_suji"
    return "unsuji_numbered"


def validate_focal_wait_record(
    record: RiichiWaitDatasetRecord,
    events: Sequence[dict[str, Any]],
) -> None:
    """Reconstruct the focal river from raw events and match the wait record."""
    declaration_index = record.declaration_dahai_event_index
    if declaration_index != record.reach_event_index + 1:
        raise ValueError("wait record declaration event index is inconsistent")
    if record.reach_accepted_event_index != declaration_index + 1:
        raise ValueError("wait record reach_accepted event index is inconsistent")
    expected_lines = (
        record.start_kyoku_line + record.reach_event_index,
        record.start_kyoku_line + declaration_index,
        record.start_kyoku_line + record.reach_accepted_event_index,
    )
    if expected_lines != (
        record.reach_line,
        record.declaration_dahai_line,
        record.reach_accepted_line,
    ):
        raise ValueError("wait record event indexes and physical lines disagree")
    if declaration_index >= len(events):
        raise ValueError("wait record declaration event index is out of range")
    declaration = events[declaration_index]
    if _event_type(declaration, declaration_index) != "dahai":
        raise ValueError("wait record declaration index does not point to dahai")
    if _event_actor(declaration, declaration_index) != record.actor:
        raise ValueError("wait record declaration actor disagrees with raw MJAI")
    declaration_tile = _event_tile(declaration, declaration_index)
    if (
        declaration_tile != record.riichi_declaration_tile
        or normalize_tile(declaration_tile) != record.riichi_declaration_tile_kind
    ):
        raise ValueError("wait record declaration tile disagrees with raw MJAI")

    raw_discards: list[dict[str, Any]] = []
    for event_index, event in enumerate(events[: declaration_index + 1]):
        event_type = _event_type(event, event_index)
        if event_type == "dahai" and _event_actor(event, event_index) == record.actor:
            tile = _event_tile(event, event_index)
            tsumogiri = event.get("tsumogiri")
            if type(tsumogiri) is not bool:
                raise ValueError(
                    f"event {event_index}: dahai tsumogiri must be boolean"
                )
            raw_discards.append(
                {
                    "discard_number": len(raw_discards) + 1,
                    "tile": tile,
                    "normalized_tile": normalize_tile(tile),
                    "tsumogiri": tsumogiri,
                    "event_index": event_index,
                    "is_riichi_declaration": event_index == declaration_index,
                    "was_called": False,
                    "call_type": None,
                    "called_by_actor": None,
                    "call_event_index": None,
                }
            )
        elif event_type in {"chi", "pon", "daiminkan"}:
            if "target" not in event:
                continue
            target = _validate_actor(event.get("target"), f"event {event_index} target")
            if target != record.actor:
                continue
            called_tile = _event_tile(event, event_index)
            if (
                not raw_discards
                or raw_discards[-1]["event_index"] != event_index - 1
                or raw_discards[-1]["tile"] != called_tile
                or raw_discards[-1]["was_called"]
            ):
                raise ValueError(
                    f"event {event_index}: call does not match focal raw river"
                )
            raw_discards[-1].update(
                was_called=True,
                call_type=event_type,
                called_by_actor=_event_actor(event, event_index),
                call_event_index=event_index,
            )

    # An accepted riichi declaration discard can still be called immediately
    # after the intervening reach_accepted event. The wait-dataset replay keeps
    # that call annotation on the declaration river entry, so reproduce it here
    # before comparing every saved field.
    call_index = record.reach_accepted_event_index + 1
    if call_index < len(events):
        event = events[call_index]
        event_type = _event_type(event, call_index)
        if event_type in {"chi", "pon", "daiminkan"} and "target" in event:
            target = _validate_actor(event.get("target"), f"event {call_index} target")
            if target == record.actor:
                called_tile = _event_tile(event, call_index)
                if (
                    not raw_discards
                    or raw_discards[-1]["event_index"] != declaration_index
                    or raw_discards[-1]["tile"] != called_tile
                    or raw_discards[-1]["was_called"]
                ):
                    raise ValueError(
                        f"event {call_index}: call does not match focal declaration"
                    )
                raw_discards[-1].update(
                    was_called=True,
                    call_type=event_type,
                    called_by_actor=_event_actor(event, call_index),
                    call_event_index=call_index,
                )

    if len(raw_discards) != record.riichi_discard_number:
        raise ValueError("wait record riichi discard number disagrees with raw river")
    recorded = record.actor_discards_before_riichi
    if len(recorded) != len(raw_discards):
        raise ValueError("wait record focal river length disagrees with raw MJAI")
    fields = (
        "discard_number",
        "tile",
        "normalized_tile",
        "tsumogiri",
        "event_index",
        "is_riichi_declaration",
        "was_called",
        "call_type",
        "called_by_actor",
        "call_event_index",
    )
    for raw, saved in zip(raw_discards, recorded, strict=True):
        mismatched = tuple(
            field_name
            for field_name in fields
            if raw[field_name] != getattr(saved, field_name)
        )
        if mismatched:
            raise ValueError(
                f"{record.relative_source_path}:{record.start_kyoku_line}: "
                "wait record focal river entry disagrees with raw MJAI for "
                f"actor {record.actor}, discard {raw['discard_number']}, fields "
                f"{mismatched}"
            )


def analyze_focal_kyoku(
    record: RiichiWaitDatasetRecord,
    kyoku: Sequence[dict[str, Any]],
) -> FocalObservation:
    """Analyze one first-established-riichi interval from source MJAI events."""
    events = list(kyoku)
    if not events or not isinstance(events[0], dict):
        raise ValueError("kyoku must begin with an event object")
    start = events[0]
    if start.get("type") != "start_kyoku" or start.get("bakaze") != "E":
        raise ValueError("focal kyoku must be an east-round start_kyoku group")
    _validate_actor(start.get("oya"), "start_kyoku oya")
    for name in ("kyoku", "honba"):
        value = start.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(f"start_kyoku {name} must be a non-negative integer")
    established = extract_established_riichis(events)
    if not established:
        raise ValueError("focal kyoku has no established riichi")
    focal = established[0]
    if (
        focal.actor != record.actor
        or focal.reach_event_index != record.reach_event_index
        or focal.reach_accepted_event_index != record.reach_accepted_event_index
    ):
        raise ValueError("wait record is not the first established riichi in the kyoku")
    if start.get("oya") != record.oya or start.get("kyoku") != record.kyoku:
        raise ValueError("wait record and source start_kyoku disagree")
    if start.get("honba") != record.honba or record.bakaze != "E":
        raise ValueError("wait record and source round identity disagree")
    validate_focal_wait_record(record, events)
    accepted_index = focal.reach_accepted_event_index
    if accepted_index >= len(events):
        raise ValueError("focal reach_accepted index is out of range")

    focal_actor = record.actor
    role: FocalRole = "dealer" if focal_actor == record.oya else "nondealer"
    turn_bin = focal_turn_bin(record.riichi_discard_number)
    wait_shape = classify_wait(record)
    count_bin = wait_count_bin(record.wait_tile_count)
    river_kinds = {
        normalize_tile(discard.tile) for discard in record.actor_discards_before_riichi
    }
    if normalize_tile(record.riichi_declaration_tile) not in river_kinds:
        raise ValueError("focal declaration tile is absent from the recorded river")
    wait_kinds = {normalize_tile(tile) for tile in record.wait_tiles}
    furiten = bool(river_kinds & wait_kinds)

    open_state = [False, False, False, False]
    for event_index, event in enumerate(events[: accepted_index + 1]):
        event_type = _event_type(event, event_index)
        if event_type in OPEN_CALL_TYPES:
            open_state[_event_actor(event, event_index)] = True
        elif event_type == "ankan":
            _event_actor(event, event_index)
    open_at_focal = tuple(open_state)
    responder_ids = tuple(actor for actor in range(4) if actor != focal_actor)
    safe_kinds = set(river_kinds)
    decision_counts = Counter[int]()
    any_call = {actor: False for actor in responder_ids}
    later_reach = {actor: False for actor in responder_ids}
    decisions: list[DiscardDecision] = []
    ron_details: list[RonDetail] = []
    focal_draws = 0
    censor_actor: int | None = None

    for event_index in range(accepted_index + 1, len(events)):
        event = events[event_index]
        event_type = _event_type(event, event_index)
        if event_type == "reach_accepted":
            censor_actor = _event_actor(event, event_index)
            if censor_actor == focal_actor:
                raise ValueError("focal actor reached twice")
            later_reach[censor_actor] = True
            break
        if event_type in ALL_CALL_TYPES:
            actor = _event_actor(event, event_index)
            if event_type in OPEN_CALL_TYPES:
                open_state[actor] = True
            if actor in any_call:
                any_call[actor] = True
            continue
        if event_type == "tsumo":
            if _event_actor(event, event_index) == focal_actor:
                focal_draws += 1
            continue
        if event_type != "dahai":
            continue
        actor = _event_actor(event, event_index)
        tile = _event_tile(event, event_index)
        tile_kind = normalize_tile(tile)
        tsumogiri = event.get("tsumogiri")
        if type(tsumogiri) is not bool:
            raise ValueError(f"event {event_index}: dahai tsumogiri must be boolean")
        if actor == focal_actor:
            safe_kinds.add(tile_kind)
            continue
        if actor not in any_call:
            raise ValueError(f"event {event_index}: invalid responder actor")
        decision_counts[actor] += 1
        safety = classify_discard(tile, safe_kinds)
        focal_ron = _has_immediate_focal_ron(events, event_index, focal_actor, actor)
        if focal_ron and tile_kind not in wait_kinds:
            raise ValueError(
                f"event {event_index}: focal ron discard is absent from wait tiles"
            )
        primary = actor != record.oya
        ordered_safe_kinds = tuple(tile for tile in TILE_KINDS if tile in safe_kinds)
        decision = DiscardDecision(
            year=record.year,
            source_path=record.relative_source_path,
            start_kyoku_line=record.start_kyoku_line,
            event_index=event_index,
            focal_actor=focal_actor,
            responder=actor,
            focal_role=role,
            reach_type=focal.reach_type,
            primary_child_responder=primary,
            tile=tile,
            tile_kind=tile_kind,
            safety_class=safety,
            tsumogiri=tsumogiri,
            focal_ron_on_discard=focal_ron,
            relative_seat=relative_seat(focal_actor, actor),
            focal_turn_bin=turn_bin,
            wait_class=wait_shape,
            wait_count_bin=count_bin,
            focal_furiten=furiten,
            responder_open_at_focal=open_at_focal[actor],
            responder_open_at_decision=open_state[actor],
            decision_ordinal_bin=decision_ordinal_bin(decision_counts[actor]),
            decision_ordinal=decision_counts[actor],
            safe_kind_count_bin=safe_kind_count_bin(len(safe_kinds)),
            safe_kinds_before=ordered_safe_kinds,
        )
        decisions.append(decision)
        if focal_ron:
            ron_details.append(
                RonDetail(
                    responder=actor,
                    relative_seat=decision.relative_seat,
                    responder_open=open_state[actor],
                    safety_class=safety,
                    tsumogiri=tsumogiri,
                    event_index=event_index,
                )
            )
        # The current tile was not safe when classified. If play continues, it
        # has now passed the focal hand and is safe for later decisions.
        safe_kinds.add(tile_kind)

    terminal = _terminal_summary(events, focal_actor)
    second_declaration = _second_reach_declaration(events, accepted_index, focal_actor)
    responders = tuple(
        ResponderObservation(
            year=record.year,
            focal_actor=focal_actor,
            responder=actor,
            focal_role=role,
            reach_type=focal.reach_type,
            primary_child_responder=actor != record.oya,
            relative_seat=relative_seat(focal_actor, actor),
            focal_turn_bin=turn_bin,
            wait_class=wait_shape,
            wait_count_bin=count_bin,
            responder_open_at_focal=open_at_focal[actor],
            any_call=any_call[actor],
            later_established_riichi=later_reach[actor],
            discard_decisions=decision_counts[actor],
        )
        for actor in responder_ids
    )
    primary_exposure = sum(decision.primary_child_responder for decision in decisions)
    return FocalObservation(
        year=record.year,
        source_path=record.relative_source_path,
        start_kyoku_line=record.start_kyoku_line,
        kyoku=record.kyoku,
        honba=record.honba,
        oya=record.oya,
        focal_actor=focal_actor,
        reach_accepted_event_index=accepted_index,
        focal_role=role,
        reach_type=focal.reach_type,
        focal_turn_bin=turn_bin,
        wait_class=wait_shape,
        wait_count_bin=count_bin,
        focal_furiten=furiten,
        decisions=tuple(decisions),
        responders=responders,
        focal_draw_opportunities=focal_draws,
        primary_child_discard_exposures=primary_exposure,
        all_responder_discard_exposures=len(decisions),
        window_end_reason=(
            "second_reach_accepted" if censor_actor is not None else "hand_end"
        ),
        terminal_outcome=terminal.outcome,
        censor_actor=censor_actor,
        ron_details=tuple(ron_details),
        terminal_hora_events=terminal.hora_events,
        terminal_multi_ron=terminal.multi_ron,
        second_reach_declaration_established=second_declaration.established,
        second_reach_declaration_unestablished=second_declaration.unestablished,
        ron_events_on_second_reach_declaration=second_declaration.ron_events,
        focal_ron_on_second_reach_declaration=second_declaration.focal_ron,
        focal_ron_without_discard=terminal.focal_ron_without_discard,
    )


def standardize_binary_counts(
    counts: Mapping[tuple[FocalRole, tuple[Any, ...]], BinaryCount],
) -> StandardizedRates:
    """Standardize common strata without constructing an aggregate Fraction.

    Each stratum's integer numerator and denominator remains the exact source of
    truth. Only the deterministic display aggregate is evaluated with the fixed
    Decimal context below, avoiding an LCM across potentially thousands of
    mutually prime denominators.
    """
    by_role: dict[FocalRole, dict[tuple[Any, ...], BinaryCount]] = {
        "dealer": {},
        "nondealer": {},
    }
    for (role, stratum), count in counts.items():
        if role not in ROLES:
            raise ValueError(f"unsupported focal role: {role}")
        by_role[role][stratum] = count
    role_strata = {
        role: {key for key, value in values.items() if value.denominator > 0}
        for role, values in by_role.items()
    }
    common = role_strata["dealer"] & role_strata["nondealer"]
    union = role_strata["dealer"] | role_strata["nondealer"]
    totals = {
        role: sum(value.denominator for value in by_role[role].values())
        for role in ROLES
    }
    common_denominators = {
        role: sum(by_role[role][key].denominator for key in common) for role in ROLES
    }
    pooled_total = sum(
        by_role[role][key].denominator for key in common for role in ROLES
    )
    ordered_common = tuple(sorted(common, key=_stable_stratum_sort_key))
    if not ordered_common or pooled_total == 0:
        dealer_rate = nondealer_rate = difference = None
    else:
        with localcontext(_new_decimal_context()):
            adjusted: dict[FocalRole, Decimal] = {}
            for role in ROLES:
                total = Decimal(0)
                for key in ordered_common:
                    role_count = by_role[role][key]
                    pooled_stratum = (
                        by_role["dealer"][key].denominator
                        + by_role["nondealer"][key].denominator
                    )
                    term = _decimal_weighted_term(
                        pooled_stratum,
                        role_count.numerator,
                        pooled_total,
                        role_count.denominator,
                    )
                    total += term
                adjusted[role] = +total
            dealer_rate = adjusted["dealer"]
            nondealer_rate = adjusted["nondealer"]
            difference = +(dealer_rate - nondealer_rate)
    coverage = {
        role: (
            None
            if totals[role] == 0
            else Fraction(common_denominators[role], totals[role])
        )
        for role in ROLES
    }
    return StandardizedRates(
        dealer_rate=dealer_rate,
        nondealer_rate=nondealer_rate,
        difference=difference,
        common_strata=len(common),
        union_strata=len(union),
        dealer_coverage=coverage["dealer"],
        nondealer_coverage=coverage["nondealer"],
        dealer_total_eligible_denominator=totals["dealer"],
        nondealer_total_eligible_denominator=totals["nondealer"],
        dealer_common_denominator=common_denominators["dealer"],
        nondealer_common_denominator=common_denominators["nondealer"],
        dealer_outside_denominator=(totals["dealer"] - common_denominators["dealer"]),
        nondealer_outside_denominator=(
            totals["nondealer"] - common_denominators["nondealer"]
        ),
        dealer_nonempty_strata=len(role_strata["dealer"]),
        nondealer_nonempty_strata=len(role_strata["nondealer"]),
        common_pooled_denominator=pooled_total,
        status=("established" if ordered_common and pooled_total else "not_estimable"),
    )


@dataclass
class _CountPair:
    numerator: int = 0
    denominator: int = 0

    def add(self, success: bool) -> None:
        self.denominator += 1
        self.numerator += int(success)

    def freeze(self) -> BinaryCount:
        return BinaryCount(self.numerator, self.denominator)


@dataclass
class DefenseAccumulator:
    """Fixed-cardinality sufficient statistics for one period/sensitivity."""

    focal_counts: Counter[FocalRole] = field(default_factory=Counter)
    focal_furiten_counts: Counter[FocalRole] = field(default_factory=Counter)
    focal_draws: Counter[FocalRole] = field(default_factory=Counter)
    primary_discard_exposures: Counter[FocalRole] = field(default_factory=Counter)
    all_discard_exposures: Counter[FocalRole] = field(default_factory=Counter)
    outcomes: Counter[tuple[FocalRole, Outcome]] = field(default_factory=Counter)
    window_end_reasons: Counter[tuple[FocalRole, WindowEndReason]] = field(
        default_factory=Counter
    )
    terminal_outcomes: Counter[tuple[FocalRole, TerminalOutcome]] = field(
        default_factory=Counter
    )
    terminal_auxiliary: Counter[tuple[FocalRole, str]] = field(default_factory=Counter)
    primary_decisions: dict[tuple[str, FocalRole, tuple[Any, ...]], _CountPair] = field(
        default_factory=lambda: defaultdict(_CountPair)
    )
    secondary_decisions: dict[tuple[str, FocalRole], _CountPair] = field(
        default_factory=lambda: defaultdict(_CountPair)
    )
    primary_responders: dict[tuple[str, FocalRole, tuple[Any, ...]], _CountPair] = (
        field(default_factory=lambda: defaultdict(_CountPair))
    )
    secondary_responders: dict[tuple[str, FocalRole], _CountPair] = field(
        default_factory=lambda: defaultdict(_CountPair)
    )
    breakdowns: dict[tuple[str, str, str, str, FocalRole], _CountPair] = field(
        default_factory=lambda: defaultdict(_CountPair)
    )
    safety_classes: Counter[tuple[FocalRole, SafetyClass]] = field(
        default_factory=Counter
    )
    ron_details: Counter[tuple[FocalRole, str, str, str, str]] = field(
        default_factory=Counter
    )
    audits: list[dict[str, Any]] = field(default_factory=list)

    def add(self, focal: FocalObservation) -> None:
        role = focal.focal_role
        self.focal_counts[role] += 1
        self.focal_furiten_counts[role] += int(focal.focal_furiten)
        self.focal_draws[role] += focal.focal_draw_opportunities
        self.primary_discard_exposures[role] += focal.primary_child_discard_exposures
        self.all_discard_exposures[role] += focal.all_responder_discard_exposures
        self.outcomes[(role, focal.outcome)] += 1
        self.window_end_reasons[(role, focal.window_end_reason)] += 1
        self.terminal_outcomes[(role, focal.terminal_outcome)] += 1
        self.terminal_auxiliary[(role, "terminal_hora_events")] += (
            focal.terminal_hora_events
        )
        self.terminal_auxiliary[(role, "terminal_multi_ron_hands")] += int(
            focal.terminal_multi_ron
        )
        self.terminal_auxiliary[
            (role, "established_second_reach_declaration_discards")
        ] += int(focal.second_reach_declaration_established)
        self.terminal_auxiliary[
            (role, "unestablished_second_reach_declaration_discards")
        ] += int(focal.second_reach_declaration_unestablished)
        self.terminal_auxiliary[(role, "ron_events_on_second_reach_declaration")] += (
            focal.ron_events_on_second_reach_declaration
        )
        self.terminal_auxiliary[(role, "focal_rons_on_second_reach_declaration")] += (
            int(focal.focal_ron_on_second_reach_declaration)
        )
        self.terminal_auxiliary[(role, "focal_rons_without_discard")] += int(
            focal.focal_ron_without_discard
        )
        for decision in focal.decisions:
            self._add_decision(decision, primary=False)
            if decision.primary_child_responder:
                self._add_decision(decision, primary=True)
        for responder in focal.responders:
            self._add_responder(responder, primary=False)
            if responder.primary_child_responder:
                self._add_responder(responder, primary=True)
        for detail in focal.ron_details:
            self.ron_details[
                (
                    role,
                    str(detail.relative_seat),
                    "open" if detail.responder_open else "closed",
                    detail.safety_class,
                    "tsumogiri" if detail.tsumogiri else "tedashi",
                )
            ] += 1
        if len(self.audits) < AUDIT_LIMIT:
            self.audits.append(
                {
                    "year": focal.year,
                    "source_path": focal.source_path,
                    "start_kyoku_line": focal.start_kyoku_line,
                    "kyoku": focal.kyoku,
                    "honba": focal.honba,
                    "focal_actor": focal.focal_actor,
                    "reach_accepted_event_index": (focal.reach_accepted_event_index),
                    "reach_accepted_line": (
                        focal.start_kyoku_line + focal.reach_accepted_event_index
                    ),
                    "focal_role": role,
                    "reach_type": focal.reach_type,
                    "furiten": focal.focal_furiten,
                    "window_end_reason": focal.window_end_reason,
                    "terminal_outcome": focal.terminal_outcome,
                    "censor_actor": focal.censor_actor,
                    "terminal_hora_events": focal.terminal_hora_events,
                    "terminal_multi_ron": focal.terminal_multi_ron,
                    "second_reach_declaration_established": (
                        focal.second_reach_declaration_established
                    ),
                    "second_reach_declaration_unestablished": (
                        focal.second_reach_declaration_unestablished
                    ),
                    "ron_events_on_second_reach_declaration": (
                        focal.ron_events_on_second_reach_declaration
                    ),
                    "focal_ron_without_discard": (focal.focal_ron_without_discard),
                    "primary_child_discard_exposures": (
                        focal.primary_child_discard_exposures
                    ),
                    "focal_draw_opportunities": focal.focal_draw_opportunities,
                    "decision_samples": [
                        {
                            "event_index": decision.event_index,
                            "event_line": (
                                focal.start_kyoku_line + decision.event_index
                            ),
                            "responder": decision.responder,
                            "primary_child_responder": (
                                decision.primary_child_responder
                            ),
                            "tile": decision.tile,
                            "tile_kind": decision.tile_kind,
                            "safety_class": decision.safety_class,
                            "tsumogiri": decision.tsumogiri,
                            "relative_seat": decision.relative_seat,
                            "responder_open_at_focal": (
                                decision.responder_open_at_focal
                            ),
                            "responder_open_at_decision": (
                                decision.responder_open_at_decision
                            ),
                            "decision_ordinal": decision.decision_ordinal,
                            "safe_kinds_before": list(decision.safe_kinds_before),
                            "safe_kind_count": len(decision.safe_kinds_before),
                            "focal_ron_on_discard": (decision.focal_ron_on_discard),
                        }
                        for decision in focal.decisions[:8]
                    ],
                }
            )

    def _add_decision(self, decision: DiscardDecision, *, primary: bool) -> None:
        successes = {
            "genbutsu_share": decision.safety_class == "genbutsu",
            "unsuji_numbered_share": decision.safety_class == "unsuji_numbered",
            "unsuji_among_numbered_non_genbutsu": (
                decision.safety_class == "unsuji_numbered"
            ),
            "tsumogiri_share": decision.tsumogiri,
            "focal_ron_per_discard": decision.focal_ron_on_discard,
        }
        if not primary:
            for metric, success in successes.items():
                if metric == "focal_ron_per_discard" and decision.focal_furiten:
                    continue
                if metric == "unsuji_among_numbered_non_genbutsu" and (
                    decision.safety_class
                    not in {"full_suji", "partial_suji", "unsuji_numbered"}
                ):
                    continue
                self.secondary_decisions[(metric, decision.focal_role)].add(success)
            return
        self.safety_classes[(decision.focal_role, decision.safety_class)] += 1
        stratum = _decision_stratum(decision)
        for metric, success in successes.items():
            if metric == "focal_ron_per_discard" and decision.focal_furiten:
                continue
            if metric == "unsuji_among_numbered_non_genbutsu" and (
                decision.safety_class
                not in {"full_suji", "partial_suji", "unsuji_numbered"}
            ):
                continue
            self.primary_decisions[(metric, decision.focal_role, stratum)].add(success)
            for dimension, value in (
                ("decision_ordinal", decision.decision_ordinal_bin),
                ("relative_seat", str(decision.relative_seat)),
                (
                    "responder_open_at_decision",
                    "open" if decision.responder_open_at_decision else "closed",
                ),
                ("focal_turn", decision.focal_turn_bin),
                ("wait_class", decision.wait_class),
                ("wait_count", decision.wait_count_bin),
            ):
                self.breakdowns[
                    ("decision", metric, dimension, value, decision.focal_role)
                ].add(success)

    def _add_responder(self, responder: ResponderObservation, *, primary: bool) -> None:
        successes = {
            "any_call_share": responder.any_call,
            "later_riichi_share": responder.later_established_riichi,
        }
        if not primary:
            for metric, success in successes.items():
                self.secondary_responders[(metric, responder.focal_role)].add(success)
            return
        stratum = _responder_stratum(responder)
        for metric, success in successes.items():
            self.primary_responders[(metric, responder.focal_role, stratum)].add(
                success
            )
            for dimension, value in (
                ("relative_seat", str(responder.relative_seat)),
                (
                    "responder_open_at_focal",
                    "open" if responder.responder_open_at_focal else "closed",
                ),
                ("focal_turn", responder.focal_turn_bin),
                ("wait_class", responder.wait_class),
                ("wait_count", responder.wait_count_bin),
            ):
                self.breakdowns[
                    ("responder", metric, dimension, value, responder.focal_role)
                ].add(success)

    def standardization_rows(
        self,
        unit: Literal["decision", "responder"],
        *,
        year: int,
    ) -> list[list[Any]]:
        """Return one compact row per role/stratum with every metric count."""
        self._validate_internal_counts()
        if unit == "decision":
            values = self.primary_decisions
            metrics = DECISION_METRICS
        else:
            values = self.primary_responders
            metrics = RESPONDER_METRICS
        return _combined_stratum_rows(values, metrics, expected_year=year)

    def to_dict(self) -> dict[str, Any]:
        self._validate_internal_counts()
        primary_metrics = {}
        for metric in DECISION_METRICS:
            counts = _collapse_strata(self.primary_decisions, metric)
            primary_metrics[metric] = _metric_document(
                counts,
                _standardize_from_pairs(self.primary_decisions, metric),
            )
        for metric in RESPONDER_METRICS:
            counts = _collapse_strata(self.primary_responders, metric)
            primary_metrics[metric] = _metric_document(
                counts,
                _standardize_from_pairs(self.primary_responders, metric),
            )
        secondary_metrics = {}
        for metric in DECISION_METRICS:
            secondary_metrics[metric] = _raw_role_document(
                {
                    role: self.secondary_decisions[(metric, role)].freeze()
                    for role in ROLES
                }
            )
        for metric in RESPONDER_METRICS:
            secondary_metrics[metric] = _raw_role_document(
                {
                    role: self.secondary_responders[(metric, role)].freeze()
                    for role in ROLES
                }
            )
        return {
            "focals": {role: self.focal_counts[role] for role in ROLES},
            "furiten_focals": {role: self.focal_furiten_counts[role] for role in ROLES},
            "exposures": {
                role: {
                    "focal_draw_opportunities": self.focal_draws[role],
                    "primary_child_discard_opportunities": (
                        self.primary_discard_exposures[role]
                    ),
                    "all_responder_discard_opportunities": (
                        self.all_discard_exposures[role]
                    ),
                    "mean_focal_draws_per_focal": _ratio_document(
                        self.focal_draws[role], self.focal_counts[role]
                    ),
                    "mean_primary_child_discards_per_focal": _ratio_document(
                        self.primary_discard_exposures[role],
                        self.focal_counts[role],
                    ),
                }
                for role in ROLES
            },
            "legacy_window_outcomes": {
                role: {
                    outcome: self.outcomes[(role, outcome)]
                    for outcome in (
                        "focal_tsumo",
                        "focal_ron",
                        "other_win",
                        "draw",
                        "censored_second_reach",
                    )
                }
                for role in ROLES
            },
            "window_end_reasons": {
                role: {
                    reason: self.window_end_reasons[(role, reason)]
                    for reason in ("hand_end", "second_reach_accepted")
                }
                for role in ROLES
            },
            "terminal_outcomes": {
                role: {
                    outcome: self.terminal_outcomes[(role, outcome)]
                    for outcome in (
                        "focal_tsumo",
                        "focal_ron",
                        "other_win",
                        "draw",
                    )
                }
                for role in ROLES
            },
            "terminal_auxiliary": {
                role: {
                    name: self.terminal_auxiliary[(role, name)]
                    for name in (
                        "terminal_hora_events",
                        "terminal_multi_ron_hands",
                        "established_second_reach_declaration_discards",
                        "unestablished_second_reach_declaration_discards",
                        "ron_events_on_second_reach_declaration",
                        "focal_rons_on_second_reach_declaration",
                        "focal_rons_without_discard",
                    )
                }
                for role in ROLES
            },
            "primary_child_responders": primary_metrics,
            "secondary_all_responders": secondary_metrics,
            "safety_class_distribution": [
                {
                    "focal_role": role,
                    "safety_class": safety,
                    "decisions": self.safety_classes[(role, safety)],
                }
                for role in ROLES
                for safety in SAFETY_CLASSES
            ],
            "breakdowns": _breakdown_document(self.breakdowns),
            "focal_ron_details": [
                {
                    "focal_role": key[0],
                    "relative_seat": key[1],
                    "responder_open": key[2],
                    "safety_class": key[3],
                    "discard_origin": key[4],
                    "rons": count,
                }
                for key, count in sorted(self.ron_details.items())
            ],
            "audit_samples": list(self.audits),
        }

    def _validate_internal_counts(self) -> None:
        for role in ROLES:
            outcomes = sum(
                self.outcomes[(role, outcome)]
                for outcome in (
                    "focal_tsumo",
                    "focal_ron",
                    "other_win",
                    "draw",
                    "censored_second_reach",
                )
            )
            if outcomes != self.focal_counts[role]:
                raise RuntimeError("focal outcomes do not sum to focal count")
            window_ends = sum(
                self.window_end_reasons[(role, reason)]
                for reason in ("hand_end", "second_reach_accepted")
            )
            if window_ends != self.focal_counts[role]:
                raise RuntimeError("window end reasons do not sum to focal count")
            terminal_outcomes = sum(
                self.terminal_outcomes[(role, outcome)]
                for outcome in ("focal_tsumo", "focal_ron", "other_win", "draw")
            )
            if terminal_outcomes != self.focal_counts[role]:
                raise RuntimeError("terminal outcomes do not sum to focal count")
            safety_total = sum(
                self.safety_classes[(role, safety)] for safety in SAFETY_CLASSES
            )
            if safety_total != self.primary_discard_exposures[role]:
                raise RuntimeError("safety classes do not sum to child exposure")
            expected_responders = self.focal_counts[role] * (
                3 if role == "dealer" else 2
            )
            responder_total = sum(
                value.denominator
                for (metric, key_role, _), value in self.primary_responders.items()
                if metric == "any_call_share" and key_role == role
            )
            if responder_total != expected_responders:
                raise RuntimeError("responder counts do not match focal population")


@dataclass
class DefenseAnalysisBuilder:
    """Streaming period/sensitivity builder with fixed-size statistics."""

    selected_years: tuple[int, ...]
    periods: dict[tuple[str, str], DefenseAccumulator] = field(init=False)
    period_years: dict[str, tuple[int, ...]] = field(init=False)

    def __post_init__(self) -> None:
        years = tuple(sorted(set(self.selected_years)))
        if not years or years != self.selected_years:
            raise ValueError("selected years must be non-empty, unique, and sorted")
        self.period_years = {
            "selected": years,
            **{f"year_{year}": (year,) for year in years},
        }
        if set(range(2020, 2026)).issubset(years):
            self.period_years["primary_2020_2025"] = tuple(range(2020, 2026))
        if set(range(2009, 2026)).issubset(years):
            self.period_years["long_term_2009_2025"] = tuple(range(2009, 2026))
        self.periods = {
            (name, sensitivity): DefenseAccumulator()
            for name in self.period_years
            for sensitivity in ("all_riichi", "normal_riichi")
        }

    def add(self, focal: FocalObservation) -> None:
        if focal.year not in self.selected_years:
            raise ValueError("focal year is outside selected years")
        names = ["selected", f"year_{focal.year}"]
        if (
            2020 <= focal.year <= 2025
            and (
                "primary_2020_2025",
                "all_riichi",
            )
            in self.periods
        ):
            names.append("primary_2020_2025")
        if ("long_term_2009_2025", "all_riichi") in self.periods:
            names.append("long_term_2009_2025")
        for name in names:
            self.periods[(name, "all_riichi")].add(focal)
            if focal.reach_type == "riichi":
                self.periods[(name, "normal_riichi")].add(focal)

    def to_dict(self) -> dict[str, Any]:
        return {
            name: {
                sensitivity: accumulator.to_dict()
                for sensitivity in ("all_riichi", "normal_riichi")
                for accumulator in (self.periods[(name, sensitivity)],)
            }
            for name in dict.fromkeys(name for name, _ in self.periods)
        }

    def period_definitions(self) -> dict[str, list[int]]:
        """Return the exact annual membership used by every period summary."""
        return {name: list(years) for name, years in self.period_years.items()}

    def standardization_sufficient_statistics(self) -> dict[str, Any]:
        """Emit annual integer rows once for independent period reconstruction."""
        units = {
            "decision": {
                "dimensions": list(DECISION_STRATUM_FIELDS[1:]),
                "metric_columns": list(DECISION_METRICS),
                "metric_value_layout": ["numerator", "denominator"],
                "row_layout": _statistics_row_layout(
                    DECISION_STRATUM_FIELDS[1:], DECISION_METRICS
                ),
            },
            "responder": {
                "dimensions": list(RESPONDER_STRATUM_FIELDS[1:]),
                "metric_columns": list(RESPONDER_METRICS),
                "metric_value_layout": ["numerator", "denominator"],
                "row_layout": _statistics_row_layout(
                    RESPONDER_STRATUM_FIELDS[1:], RESPONDER_METRICS
                ),
            },
        }
        tables: list[dict[str, Any]] = []
        for year in self.selected_years:
            for sensitivity in ("all_riichi", "normal_riichi"):
                accumulator = self.periods[(f"year_{year}", sensitivity)]
                tables.append(
                    {
                        "year": year,
                        "sensitivity": sensitivity,
                        "decision_rows": accumulator.standardization_rows(
                            "decision", year=year
                        ),
                        "responder_rows": accumulator.standardization_rows(
                            "responder", year=year
                        ),
                    }
                )
        return {
            "representation": "annual_role_stratum_integer_rows",
            "table_dimensions": ["year", "sensitivity"],
            "role_values": list(ROLES),
            "zero_denominator_policy": "explicit_0_0",
            "units": units,
            "tables": tables,
        }


@dataclass
class _PublishState:
    target: Path
    staged: Path
    originally_existed: bool
    backup: Path | None = None


def publish_output_bundle(outputs: Mapping[Path, str]) -> None:
    """Atomically replace a small same-filesystem bundle with rollback."""
    if not outputs:
        raise ValueError("output bundle must not be empty")
    targets = tuple(outputs)
    if len(set(targets)) != len(targets):
        raise ValueError("output targets must be unique")
    states: list[_PublishState] = []
    retained: list[Path] = []
    try:
        for target, content in outputs.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
            os.close(descriptor)
            temporary = Path(name)
            temporary.write_text(content, encoding="utf-8", newline="\n")
            states.append(
                _PublishState(
                    target=target,
                    staged=temporary,
                    originally_existed=target.exists(),
                )
            )
        for state in states:
            if state.originally_existed:
                descriptor, name = tempfile.mkstemp(
                    prefix=f".{state.target.name}.",
                    suffix=".backup",
                    dir=state.target.parent,
                )
                os.close(descriptor)
                backup = Path(name)
                backup.unlink()
                # Register the recovery path before moving the original. An
                # interrupt raised just after os.replace returns must still
                # leave enough state to restore it.
                state.backup = backup
                os.replace(state.target, backup)
            os.replace(state.staged, state.target)
    except BaseException as error:
        rollback_errors: list[str] = []
        for state in reversed(states):
            backup = state.backup
            try:
                if backup is not None and backup.exists():
                    os.replace(backup, state.target)
                elif (
                    not state.originally_existed
                    and not state.staged.exists()
                    and state.target.exists()
                ):
                    # A missing staged file proves the publish replace
                    # completed even if interruption happened before the next
                    # Python statement.
                    state.target.unlink()
            except BaseException as rollback_error:  # noqa: BLE001
                # Rollback must continue even when cleanup is interrupted.
                if backup is not None and backup.exists():
                    retained.append(backup)
                rollback_errors.append(f"{state.target}: {rollback_error}")
        if rollback_errors:
            paths = ", ".join(str(path) for path in retained)
            raise RuntimeError(
                "output publication failed and rollback was incomplete; "
                f"retained backups: {paths}; errors: {'; '.join(rollback_errors)}"
            ) from error
        raise
    finally:
        for state in states:
            if state.staged.exists():
                state.staged.unlink()
            if (
                state.backup is not None
                and state.backup.exists()
                and state.backup not in retained
            ):
                state.backup.unlink()


def canonical_json(value: object) -> str:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def _validate_actor(value: object, name: str) -> int:
    if type(value) is not int or value not in range(4):
        raise ValueError(f"{name} must be an integer from 0 through 3")
    return value


def _event_type(event: object, index: int) -> str:
    if not isinstance(event, dict) or not isinstance(event.get("type"), str):
        raise ValueError(  # noqa: TRY004 - malformed MJAI is a data-value error
            f"event {index}: MJAI event must have a string type"
        )
    return event["type"]


def _event_actor(event: Mapping[str, Any], index: int) -> int:
    return _validate_actor(event.get("actor"), f"event {index} actor")


def _event_tile(event: Mapping[str, Any], index: int) -> str:
    tile = event.get("pai")
    if not isinstance(tile, str):
        raise ValueError(  # noqa: TRY004 - malformed MJAI is a data-value error
            f"event {index}: dahai pai must be a string"
        )
    normalize_tile(tile)
    return tile


def _has_immediate_focal_ron(
    events: Sequence[dict[str, Any]],
    discard_index: int,
    focal_actor: int,
    responder: int,
) -> bool:
    found = False
    for index in range(discard_index + 1, len(events)):
        event = events[index]
        if _event_type(event, index) != "hora":
            break
        actor = _event_actor(event, index)
        target = _validate_actor(event.get("target"), f"event {index} target")
        if actor == focal_actor and target == responder:
            found = True
    return found


def _terminal_summary(
    events: Sequence[dict[str, Any]], focal_actor: int
) -> _TerminalSummary:
    """Classify the actual hand result, independently of the response window."""
    if not events or _event_type(events[-1], len(events) - 1) != "end_kyoku":
        raise ValueError("kyoku must end with end_kyoku")
    result_indexes: list[int] = []
    result_started = False
    for index, event in enumerate(events[:-1]):
        event_type = _event_type(event, index)
        if event_type in {"hora", "ryukyoku"}:
            result_indexes.append(index)
            result_started = True
        elif result_started:
            raise ValueError("result events must be contiguous before end_kyoku")
    if not result_indexes:
        raise ValueError("kyoku has no terminal result")
    horas = [index for index in result_indexes if events[index]["type"] == "hora"]
    draws = [index for index in result_indexes if events[index]["type"] == "ryukyoku"]
    if (horas and draws) or len(draws) > 1:
        raise ValueError("kyoku has inconsistent terminal results")
    if draws:
        return _TerminalSummary("draw", 0, False, False)

    winners: set[int] = set()
    focal_method: Literal["tsumo", "ron"] | None = None
    focal_target: int | None = None
    for index in horas:
        event = events[index]
        actor = _event_actor(event, index)
        target = _validate_actor(event.get("target"), f"event {index} target")
        if actor in winners:
            raise ValueError("one actor cannot have multiple hora events")
        winners.add(actor)
        if actor == focal_actor:
            focal_target = target
            focal_method = "tsumo" if target == actor else "ron"
    if focal_method == "tsumo":
        outcome: TerminalOutcome = "focal_tsumo"
    elif focal_method == "ron":
        outcome = "focal_ron"
    else:
        outcome = "other_win"
    without_discard = False
    if focal_method == "ron":
        source_index = result_indexes[0] - 1
        if source_index < 0:
            without_discard = True
        else:
            source = events[source_index]
            without_discard = not (
                _event_type(source, source_index) == "dahai"
                and _event_actor(source, source_index) == focal_target
            )
    return _TerminalSummary(
        outcome=outcome,
        hora_events=len(horas),
        multi_ron=len(horas) > 1,
        focal_ron_without_discard=without_discard,
    )


def _second_reach_declaration(
    events: Sequence[dict[str, Any]],
    focal_accepted_index: int,
    focal_actor: int,
) -> _SecondReachDeclaration:
    """Describe the first later reach declaration, whether accepted or not."""
    for reach_index in range(focal_accepted_index + 1, len(events)):
        event = events[reach_index]
        if _event_type(event, reach_index) != "reach":
            continue
        actor = _event_actor(event, reach_index)
        if actor == focal_actor:
            raise ValueError("focal actor declared reach twice")
        sequence = match_reach_sequence(
            list(events), reach_index, context="second reach declaration"
        )
        declaration_index = reach_index + 1
        ron_events = 0
        focal_ron = False
        for index in range(declaration_index + 1, len(events)):
            result = events[index]
            if _event_type(result, index) != "hora":
                break
            ron_events += 1
            if (
                _event_actor(result, index) == focal_actor
                and _validate_actor(result.get("target"), f"event {index} target")
                == actor
            ):
                focal_ron = True
        return _SecondReachDeclaration(
            established=sequence is not None,
            unestablished=sequence is None,
            ron_events=ron_events,
            focal_ron=focal_ron,
        )
    return _SecondReachDeclaration()


def _decision_stratum(decision: DiscardDecision) -> tuple[Any, ...]:
    return (
        decision.year,
        decision.focal_turn_bin,
        decision.wait_class,
        decision.wait_count_bin,
        decision.relative_seat,
        decision.responder_open_at_focal,
        decision.responder_open_at_decision,
        decision.decision_ordinal_bin,
        decision.safe_kind_count_bin,
    )


def _responder_stratum(responder: ResponderObservation) -> tuple[Any, ...]:
    return (
        responder.year,
        responder.focal_turn_bin,
        responder.wait_class,
        responder.wait_count_bin,
        responder.relative_seat,
        responder.responder_open_at_focal,
    )


def _collapse_strata(
    values: Mapping[tuple[str, FocalRole, tuple[Any, ...]], _CountPair],
    metric: str,
) -> dict[FocalRole, BinaryCount]:
    result = {}
    for role in ROLES:
        numerator = sum(
            value.numerator
            for (name, key_role, _), value in values.items()
            if name == metric and key_role == role
        )
        denominator = sum(
            value.denominator
            for (name, key_role, _), value in values.items()
            if name == metric and key_role == role
        )
        result[role] = BinaryCount(numerator, denominator)
    return result


def _standardize_from_pairs(
    values: Mapping[tuple[str, FocalRole, tuple[Any, ...]], _CountPair],
    metric: str,
) -> StandardizedRates:
    return standardize_binary_counts(
        {
            (role, stratum): value.freeze()
            for (name, role, stratum), value in values.items()
            if name == metric
        }
    )


def _combined_stratum_rows(
    values: Mapping[tuple[str, FocalRole, tuple[Any, ...]], _CountPair],
    metrics: tuple[str, ...],
    *,
    expected_year: int,
) -> list[list[Any]]:
    keys = {(role, stratum) for metric, role, stratum in values if metric in metrics}
    ordered = sorted(
        keys,
        key=lambda item: (_stable_stratum_sort_key(item[1]), ROLES.index(item[0])),
    )
    rows: list[list[Any]] = []
    for role, stratum in ordered:
        if not stratum or stratum[0] != expected_year:
            raise RuntimeError(
                "annual standardization stratum year does not match its table"
            )
        row: list[Any] = [*stratum[1:], role]
        for metric in metrics:
            value = values.get((metric, role, stratum))
            row.extend(
                (0, 0) if value is None else (value.numerator, value.denominator)
            )
        rows.append(row)
    return rows


def _statistics_row_layout(
    dimensions: tuple[str, ...], metrics: tuple[str, ...]
) -> list[str]:
    return [
        *dimensions,
        "focal_role",
        *(
            value
            for metric in metrics
            for value in (f"{metric}.numerator", f"{metric}.denominator")
        ),
    ]


def _stable_stratum_sort_key(stratum: tuple[Any, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((type(value).__name__, repr(value)) for value in stratum)


def _decimal_ratio(numerator: int, denominator: int) -> Decimal | None:
    if denominator == 0:
        return None
    with localcontext(_new_decimal_context()):
        return +(Decimal(numerator) / Decimal(denominator))


def _decimal_weighted_term(
    pooled_stratum: int,
    numerator: int,
    pooled_total: int,
    denominator: int,
) -> Decimal:
    # Form each rational term with Python integers first. Decimal multiplication
    # inside the finite context would round an integer product longer than the
    # configured precision before division.
    term_numerator = pooled_stratum * numerator
    term_denominator = pooled_total * denominator
    with localcontext(_new_decimal_context()):
        return +(Decimal(term_numerator) / Decimal(term_denominator))


def _new_decimal_context() -> Context:
    return Context(
        prec=DECIMAL_PRECISION_DIGITS,
        rounding=DECIMAL_ROUNDING,
        Emin=DECIMAL_MIN_EXPONENT,
        Emax=DECIMAL_MAX_EXPONENT,
    )


def _decimal_string(value: Decimal) -> str:
    return format(value, f".{DECIMAL_PRECISION_DIGITS}g")


def _ratio_document(numerator: int, denominator: int) -> dict[str, Any] | None:
    value = _decimal_ratio(numerator, denominator)
    if value is None:
        return None
    return {
        "numerator": numerator,
        "denominator": denominator,
        "decimal": _decimal_string(value),
        "precision_digits": DECIMAL_PRECISION_DIGITS,
        "rounding": DECIMAL_ROUNDING,
    }


def _weighted_decimal_document(value: Decimal | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "representation": WEIGHTED_SUM_REPRESENTATION,
        "decimal": _decimal_string(value),
        "precision_digits": DECIMAL_PRECISION_DIGITS,
        "rounding": DECIMAL_ROUNDING,
    }


def _raw_role_document(counts: Mapping[FocalRole, BinaryCount]) -> dict[str, Any]:
    return {
        role: {
            "numerator": counts[role].numerator,
            "denominator": counts[role].denominator,
            "rate": _ratio_document(counts[role].numerator, counts[role].denominator),
        }
        for role in ROLES
    }


def _metric_document(
    counts: Mapping[FocalRole, BinaryCount],
    adjustment: StandardizedRates,
) -> dict[str, Any]:
    return {
        "raw": _raw_role_document(counts),
        "common_support_standardized": {
            "status": adjustment.status,
            "sufficient_statistics_reference": (
                "standardization_sufficient_statistics"
            ),
            "dealer_rate": _weighted_decimal_document(adjustment.dealer_rate),
            "nondealer_rate": _weighted_decimal_document(adjustment.nondealer_rate),
            "dealer_minus_nondealer": _weighted_decimal_document(adjustment.difference),
            "common_strata": adjustment.common_strata,
            "union_strata": adjustment.union_strata,
            "nonempty_strata": {
                "dealer": adjustment.dealer_nonempty_strata,
                "nondealer": adjustment.nondealer_nonempty_strata,
            },
            "common_pooled_denominator": adjustment.common_pooled_denominator,
            "eligible_denominators": {
                "dealer": {
                    "total": adjustment.dealer_total_eligible_denominator,
                    "common": adjustment.dealer_common_denominator,
                    "outside_common_support": adjustment.dealer_outside_denominator,
                },
                "nondealer": {
                    "total": adjustment.nondealer_total_eligible_denominator,
                    "common": adjustment.nondealer_common_denominator,
                    "outside_common_support": (
                        adjustment.nondealer_outside_denominator
                    ),
                },
            },
            "coverage": {
                "dealer": _ratio_document(
                    adjustment.dealer_common_denominator,
                    adjustment.dealer_total_eligible_denominator,
                ),
                "nondealer": _ratio_document(
                    adjustment.nondealer_common_denominator,
                    adjustment.nondealer_total_eligible_denominator,
                ),
            },
        },
    }


def _breakdown_document(
    values: Mapping[tuple[str, str, str, str, FocalRole], _CountPair],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[FocalRole, BinaryCount]] = {}
    for unit, metric, dimension, value, role in values:
        key = (unit, metric, dimension, value)
        if key not in grouped:
            grouped[key] = {role_name: BinaryCount(0, 0) for role_name in ROLES}
        grouped[key][role] = values[(unit, metric, dimension, value, role)].freeze()
    return [
        {
            "unit": key[0],
            "metric": key[1],
            "dimension": key[2],
            "value": key[3],
            "raw": _raw_role_document(counts),
        }
        for key, counts in sorted(grouped.items())
    ]
