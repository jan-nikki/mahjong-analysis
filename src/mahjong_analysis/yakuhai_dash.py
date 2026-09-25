"""Opening yakuhai-discard behavior and pair-formation analysis."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from math import comb
from typing import Any, Literal

from mahjong_analysis.riichi_declaration_tile_analysis import dora_from_marker
from mahjong_analysis.shanten import initial_shanten
from mahjong_analysis.tiles import normalize_tile, tile_to_index, tiles_to_counts

DRAGONS: tuple[str, ...] = ("P", "F", "C")
WINDS: tuple[str, ...] = ("E", "S", "W", "N")
HONORS: frozenset[str] = frozenset((*WINDS, *DRAGONS))
YAKUHAI_TILES: tuple[str, ...] = ("E", "S", "W", "N", "P", "F", "C")
YAKUHAI_CLASSES: tuple[str, ...] = (
    "dealer_double_wind",
    "shared_round_wind",
    "shared_dragon",
    "own_seat_wind",
)

YakuhaiClass = Literal[
    "dealer_double_wind",
    "shared_round_wind",
    "shared_dragon",
    "own_seat_wind",
]
SingletonOutcome = Literal["self_pair_draw", "singleton_discard", "round_end"]


@dataclass(frozen=True)
class DealObservation:
    """One actor's 13-tile initial hand and its yakuhai composition."""

    actor: int
    dealer: bool
    seat_wind: str
    initial_hand: tuple[str, ...]
    yakuhai_counts: tuple[tuple[str, int], ...]
    max_suit_count: int
    honor_count: int
    terminal_honor_count: int
    all_pair_kind_count: int
    shanten: int
    dora_han: int

    @property
    def role(self) -> str:
        return "dealer" if self.dealer else "nondealer"

    @property
    def yakuhai_pair_kind_count(self) -> int:
        return sum(count >= 2 for _, count in self.yakuhai_counts)

    @property
    def singleton_kind_count(self) -> int:
        return sum(count == 1 for _, count in self.yakuhai_counts)

    @property
    def has_yakuhai_pair(self) -> bool:
        return self.yakuhai_pair_kind_count > 0


@dataclass(frozen=True)
class OpeningObservation:
    """Hand-level adoption of strict and relaxed yakuhai dash definitions."""

    actor: int
    dealer: bool
    initial_singleton_count: int
    strict_dash: bool
    dash_by_2: bool
    dash_by_3: bool
    max_suit_count: int
    honor_count: int
    all_pair_kind_count: int
    shanten: int
    dora_han: int
    first_discard_event_index: int | None
    opponent_discards_before_first_discard: int | None
    actor_discard_count: int

    @property
    def role(self) -> str:
        return "dealer" if self.dealer else "nondealer"


@dataclass(frozen=True)
class SingletonObservation:
    """One initially singleton yakuhai followed until pair, discard, or end."""

    actor: int
    dealer: bool
    seat_wind: str
    tile: str
    yakuhai_class: YakuhaiClass
    initial_hand: tuple[str, ...]
    opponent_ready_at_deal: bool
    opponent_ready_actor_count_at_deal: int
    first_opponent_pair_event_index: int | None
    first_opponent_pair_actor: int | None
    focal_draws_completed_at_opponent_pair: int | None
    outcome: SingletonOutcome
    self_pair_draw_number: int | None
    self_pair_event_index: int | None
    self_pair_followup_discard_event_index: int | None
    self_pair_retained_after_discard: bool | None
    self_pair_pon_shape_after_discard: bool | None
    singleton_discard_number: int | None
    singleton_discard_event_index: int | None
    opponent_pon_capable_at_discard: bool | None
    opponent_pon_capable_actor_count: int | None
    opponent_pair_holder_actor_count: int | None
    discard_was_ponned: bool
    discard_pon_actor: int | None
    discard_pon_event_index: int | None
    focal_pon_event_index: int | None
    public_same_tile_before_first_discard: int | None
    post_discard_self_draw_number: int | None
    post_discard_self_draw_event_index: int | None
    post_discard_opponent_pair_event_index: int | None
    post_discard_opponent_pair_actor: int | None

    @property
    def role(self) -> str:
        return "dealer" if self.dealer else "nondealer"

    @property
    def exposure(self) -> str:
        return "own_only" if self.yakuhai_class == "own_seat_wind" else "shared"

    @property
    def pair_race(self) -> str:
        """Exclusive first-arrival classification before singleton resolution."""
        if self.opponent_ready_at_deal:
            return "opponent_at_deal"
        opponent_event = self.first_opponent_pair_event_index
        self_event = self.self_pair_event_index
        if opponent_event is not None and (
            self_event is None or opponent_event < self_event
        ):
            return "opponent_first_after_deal"
        if self_event is not None:
            return "self_first"
        if self.outcome == "singleton_discard":
            return "discard_before_either_pair"
        return "round_end_before_either_pair"


@dataclass(frozen=True)
class KyokuYakuhaiAnalysis:
    deals: tuple[DealObservation, ...]
    openings: tuple[OpeningObservation, ...]
    singletons: tuple[SingletonObservation, ...]


@dataclass
class _PlayerState:
    concealed_tiles: list[str]
    draw_count: int = 0
    discard_count: int = 0
    riichi_accepted: bool = False
    reach_declared: bool = False
    first_discard_event_index: int | None = None
    opponent_discards_before_first_discard: int | None = None


@dataclass
class _SingletonTracker:
    actor: int
    dealer: bool
    seat_wind: str
    tile: str
    yakuhai_class: YakuhaiClass
    initial_hand: tuple[str, ...]
    opponent_ready_at_deal: bool
    opponent_ready_actor_count_at_deal: int
    first_opponent_pair_event_index: int | None = None
    first_opponent_pair_actor: int | None = None
    focal_draws_completed_at_opponent_pair: int | None = None
    outcome: SingletonOutcome | None = None
    self_pair_draw_number: int | None = None
    self_pair_event_index: int | None = None
    self_pair_followup_discard_event_index: int | None = None
    self_pair_retained_after_discard: bool | None = None
    self_pair_pon_shape_after_discard: bool | None = None
    singleton_discard_number: int | None = None
    singleton_discard_event_index: int | None = None
    opponent_pon_capable_at_discard: bool | None = None
    opponent_pon_capable_actor_count: int | None = None
    opponent_pair_holder_actor_count: int | None = None
    discard_was_ponned: bool = False
    discard_pon_actor: int | None = None
    discard_pon_event_index: int | None = None
    focal_pon_event_index: int | None = None
    public_same_tile_before_first_discard: int | None = None
    post_discard_self_draw_number: int | None = None
    post_discard_self_draw_event_index: int | None = None
    post_discard_opponent_pair_event_index: int | None = None
    post_discard_opponent_pair_actor: int | None = None

    def freeze(self) -> SingletonObservation:
        outcome: SingletonOutcome = self.outcome or "round_end"
        return SingletonObservation(
            actor=self.actor,
            dealer=self.dealer,
            seat_wind=self.seat_wind,
            tile=self.tile,
            yakuhai_class=self.yakuhai_class,
            initial_hand=self.initial_hand,
            opponent_ready_at_deal=self.opponent_ready_at_deal,
            opponent_ready_actor_count_at_deal=(
                self.opponent_ready_actor_count_at_deal
            ),
            first_opponent_pair_event_index=self.first_opponent_pair_event_index,
            first_opponent_pair_actor=self.first_opponent_pair_actor,
            focal_draws_completed_at_opponent_pair=(
                self.focal_draws_completed_at_opponent_pair
            ),
            outcome=outcome,
            self_pair_draw_number=self.self_pair_draw_number,
            self_pair_event_index=self.self_pair_event_index,
            self_pair_followup_discard_event_index=(
                self.self_pair_followup_discard_event_index
            ),
            self_pair_retained_after_discard=self.self_pair_retained_after_discard,
            self_pair_pon_shape_after_discard=self.self_pair_pon_shape_after_discard,
            singleton_discard_number=self.singleton_discard_number,
            singleton_discard_event_index=self.singleton_discard_event_index,
            opponent_pon_capable_at_discard=(self.opponent_pon_capable_at_discard),
            opponent_pon_capable_actor_count=(self.opponent_pon_capable_actor_count),
            opponent_pair_holder_actor_count=(
                self.opponent_pair_holder_actor_count
            ),
            discard_was_ponned=self.discard_was_ponned,
            discard_pon_actor=self.discard_pon_actor,
            discard_pon_event_index=self.discard_pon_event_index,
            focal_pon_event_index=self.focal_pon_event_index,
            public_same_tile_before_first_discard=(
                self.public_same_tile_before_first_discard
            ),
            post_discard_self_draw_number=self.post_discard_self_draw_number,
            post_discard_self_draw_event_index=(
                self.post_discard_self_draw_event_index
            ),
            post_discard_opponent_pair_event_index=(
                self.post_discard_opponent_pair_event_index
            ),
            post_discard_opponent_pair_actor=self.post_discard_opponent_pair_actor,
        )


def seat_wind(actor: int, dealer: int) -> str:
    """Return an actor's seat wind from the dealer actor index."""
    _validate_actor_value(actor, "actor")
    _validate_actor_value(dealer, "dealer")
    return WINDS[(actor - dealer) % 4]


def yakuhai_kinds(actor: int, dealer: int, round_wind: str = "E") -> tuple[str, ...]:
    """Return unique yakuhai kinds for one actor in stable tile order."""
    wind = normalize_tile(round_wind)
    if wind not in WINDS:
        raise ValueError("round wind must be a wind tile")
    kinds = {wind, seat_wind(actor, dealer), *DRAGONS}
    return tuple(sorted(kinds, key=tile_to_index))


def classify_yakuhai(
    tile: str,
    actor: int,
    dealer: int,
    round_wind: str = "E",
) -> YakuhaiClass:
    """Classify one of the actor's yakuhai kinds."""
    normalized = normalize_tile(tile)
    wind = normalize_tile(round_wind)
    actor_wind = seat_wind(actor, dealer)
    if normalized in DRAGONS:
        return "shared_dragon"
    if normalized == wind == actor_wind:
        return "dealer_double_wind"
    if normalized == wind:
        return "shared_round_wind"
    if normalized == actor_wind:
        return "own_seat_wind"
    raise ValueError(f"tile {tile!r} is not yakuhai for actor {actor}")


def theoretical_specific_pair_probability(hand_size: int = 13) -> float:
    """Probability that one specified four-copy kind appears at least twice."""
    if type(hand_size) is not int or not 0 <= hand_size <= 136:
        raise ValueError("hand_size must be an integer from 0 through 136")
    total = comb(136, hand_size)
    zero = comb(132, hand_size) if hand_size <= 132 else 0
    one = 4 * comb(132, hand_size - 1) if 1 <= hand_size <= 133 else 0
    return 1.0 - (zero + one) / total


def theoretical_any_pair_probability(kind_count: int, hand_size: int = 13) -> float:
    """Probability that any of ``kind_count`` specified kinds appears twice."""
    if type(kind_count) is not int or not 1 <= kind_count <= 34:
        raise ValueError("kind_count must be an integer from 1 through 34")
    if type(hand_size) is not int or not 0 <= hand_size <= 136:
        raise ValueError("hand_size must be an integer from 0 through 136")
    other_tiles = 136 - 4 * kind_count
    no_pair = 0
    for singleton_kinds in range(0, min(kind_count, hand_size) + 1):
        remaining = hand_size - singleton_kinds
        if 0 <= remaining <= other_tiles:
            no_pair += (
                comb(kind_count, singleton_kinds)
                * (4**singleton_kinds)
                * comb(other_tiles, remaining)
            )
    return 1.0 - no_pair / comb(136, hand_size)


def theoretical_singleton_pair_by_draw(draw_count: int) -> float:
    """Simple cumulative chance to draw one of three matches from 123 unknowns."""
    if type(draw_count) is not int or not 0 <= draw_count <= 123:
        raise ValueError("draw_count must be an integer from 0 through 123")
    misses = comb(120, draw_count) if draw_count <= 120 else 0
    return 1.0 - misses / comb(123, draw_count)


def analyze_yakuhai_dash_kyoku(
    kyoku: list[dict[str, Any]],
) -> KyokuYakuhaiAnalysis:
    """Replay one east-round kyoku and analyze every actor's initial yakuhai."""
    start, states, dealer, round_wind = _initial_state(kyoku)
    marker = start.get("dora_marker")
    if not isinstance(marker, str):
        raise ValueError("start_kyoku dora_marker must be a tile string")
    dora_kind = dora_from_marker(marker)
    public_visible_counts: Counter[str] = Counter((normalize_tile(marker),))
    total_discards = 0
    deals = tuple(
        _deal_observation(
            actor, dealer, round_wind, dora_kind, state.concealed_tiles
        )
        for actor, state in enumerate(states)
    )
    trackers: dict[tuple[int, str], _SingletonTracker] = {}
    for deal in deals:
        for tile, count in deal.yakuhai_counts:
            if count != 1:
                continue
            ready_actors = _ready_opponents(states, deal.actor, tile)
            tracker = _SingletonTracker(
                actor=deal.actor,
                dealer=deal.dealer,
                seat_wind=deal.seat_wind,
                tile=tile,
                yakuhai_class=classify_yakuhai(tile, deal.actor, dealer, round_wind),
                initial_hand=deal.initial_hand,
                opponent_ready_at_deal=bool(ready_actors),
                opponent_ready_actor_count_at_deal=len(ready_actors),
            )
            if ready_actors:
                tracker.first_opponent_pair_event_index = 0
                tracker.first_opponent_pair_actor = ready_actors[0]
                tracker.focal_draws_completed_at_opponent_pair = 0
            trackers[(deal.actor, tile)] = tracker

    last_discard: tuple[int, str, int] | None = None
    for event_index, raw_event in enumerate(kyoku[1:], start=1):
        event = _mapping(raw_event, event_index)
        event_type = _event_type(event, event_index)

        if event_type == "end_kyoku":
            last_discard = None
            continue
        if event_type == "tsumo":
            actor = _actor(event, event_index)
            tile = _tile(event, "pai", event_index, actor)
            normalized = normalize_tile(tile)
            state = states[actor]
            state.draw_count += 1
            own_tracker = trackers.get((actor, normalized))
            if own_tracker is not None and own_tracker.outcome is None:
                if _count_kind(state.concealed_tiles, normalized) != 1:
                    raise _error(
                        event_index,
                        actor,
                        "active singleton tracker does not own exactly one tile",
                    )
                own_tracker.outcome = "self_pair_draw"
                own_tracker.self_pair_draw_number = state.draw_count
                own_tracker.self_pair_event_index = event_index
            elif (
                own_tracker is not None
                and own_tracker.outcome == "singleton_discard"
                and own_tracker.post_discard_self_draw_event_index is None
            ):
                own_tracker.post_discard_self_draw_number = state.draw_count
                own_tracker.post_discard_self_draw_event_index = event_index
            state.concealed_tiles.append(tile)
            _validate_owned_tiles(state, event_index, actor)
            for tracker in trackers.values():
                if (
                    tracker.actor != actor
                    and tracker.tile == normalized
                    and tracker.outcome is None
                    and tracker.first_opponent_pair_event_index is None
                    and _count_kind(state.concealed_tiles, normalized) >= 2
                ):
                    tracker.first_opponent_pair_event_index = event_index
                    tracker.first_opponent_pair_actor = actor
                    tracker.focal_draws_completed_at_opponent_pair = states[
                        tracker.actor
                    ].draw_count
                if (
                    tracker.actor != actor
                    and tracker.tile == normalized
                    and tracker.outcome == "singleton_discard"
                    and tracker.post_discard_opponent_pair_event_index is None
                    and _count_kind(state.concealed_tiles, normalized) == 2
                ):
                    tracker.post_discard_opponent_pair_event_index = event_index
                    tracker.post_discard_opponent_pair_actor = actor
            last_discard = None
            continue

        if event_type == "dahai":
            actor = _actor(event, event_index)
            tile = _tile(event, "pai", event_index, actor)
            if type(event.get("tsumogiri")) is not bool:
                raise _error(event_index, actor, "dahai tsumogiri must be bool")
            normalized = normalize_tile(tile)
            state = states[actor]
            if state.discard_count == 0:
                state.first_discard_event_index = event_index
                state.opponent_discards_before_first_discard = total_discards
                for owned_tracker in trackers.values():
                    if owned_tracker.actor == actor:
                        owned_tracker.public_same_tile_before_first_discard = (
                            public_visible_counts[owned_tracker.tile]
                        )
            state.discard_count += 1
            tracker = trackers.get((actor, normalized))
            if tracker is not None and tracker.outcome is None:
                if _count_kind(state.concealed_tiles, normalized) != 1:
                    raise _error(
                        event_index,
                        actor,
                        "active singleton discard does not own exactly one tile",
                    )
                pair_holders = _pair_holding_opponents(states, actor, normalized)
                ready_actors = tuple(
                    opponent
                    for opponent in pair_holders
                    if not states[opponent].riichi_accepted
                )
                tracker.outcome = "singleton_discard"
                tracker.singleton_discard_number = state.discard_count
                tracker.singleton_discard_event_index = event_index
                tracker.opponent_pon_capable_at_discard = bool(ready_actors)
                tracker.opponent_pon_capable_actor_count = len(ready_actors)
                tracker.opponent_pair_holder_actor_count = len(pair_holders)
            _remove_tiles(state, (tile,), event_index, actor)
            for owned_tracker in trackers.values():
                if (
                    owned_tracker.actor == actor
                    and owned_tracker.outcome == "self_pair_draw"
                    and owned_tracker.self_pair_followup_discard_event_index is None
                ):
                    retained = _count_kind(state.concealed_tiles, owned_tracker.tile) >= 2
                    owned_tracker.self_pair_followup_discard_event_index = event_index
                    owned_tracker.self_pair_retained_after_discard = retained
                    owned_tracker.self_pair_pon_shape_after_discard = (
                        retained and not state.riichi_accepted and not state.reach_declared
                    )
            public_visible_counts[normalized] += 1
            total_discards += 1
            last_discard = (actor, normalized, event_index)
            continue

        if event_type in {"chi", "pon", "daiminkan"}:
            actor = _actor(event, event_index)
            target = _target(event, event_index, actor)
            called_tile = _tile(event, "pai", event_index, actor)
            normalized = normalize_tile(called_tile)
            if last_discard is None or last_discard[:2] != (target, normalized):
                raise _error(
                    event_index,
                    actor,
                    f"{event_type} does not match the preceding callable discard",
                )
            expected = 2 if event_type in {"chi", "pon"} else 3
            consumed = _consumed(event, expected, event_index, actor)
            if event_type == "chi" and actor != (target + 1) % 4:
                raise _error(event_index, actor, "chi actor must follow target")
            if event_type == "pon":
                discarded_tracker = trackers.get((target, normalized))
                if (
                    discarded_tracker is not None
                    and discarded_tracker.singleton_discard_event_index
                    == last_discard[2]
                ):
                    if not discarded_tracker.opponent_pon_capable_at_discard:
                        raise _error(
                            event_index,
                            actor,
                            "pon occurred without a pon-capable opponent",
                        )
                    discarded_tracker.discard_was_ponned = True
                    discarded_tracker.discard_pon_actor = actor
                    discarded_tracker.discard_pon_event_index = event_index
                caller_tracker = trackers.get((actor, normalized))
                if (
                    caller_tracker is not None
                    and caller_tracker.outcome == "self_pair_draw"
                    and caller_tracker.focal_pon_event_index is None
                ):
                    caller_tracker.focal_pon_event_index = event_index
            _remove_tiles(states[actor], consumed, event_index, actor)
            public_visible_counts.update(normalize_tile(tile) for tile in consumed)
            last_discard = None
            continue

        if event_type == "ankan":
            actor = _actor(event, event_index)
            consumed = _consumed(event, 4, event_index, actor)
            _remove_tiles(states[actor], consumed, event_index, actor)
            public_visible_counts.update(normalize_tile(tile) for tile in consumed)
            last_discard = None
            continue

        if event_type == "kakan":
            actor = _actor(event, event_index)
            added = _tile(event, "pai", event_index, actor)
            _consumed(event, 3, event_index, actor)
            _remove_tiles(states[actor], (added,), event_index, actor)
            public_visible_counts[normalize_tile(added)] += 1
            last_discard = None
            continue

        if event_type == "reach_accepted":
            actor = _actor(event, event_index)
            states[actor].riichi_accepted = True
            continue
        if event_type == "reach":
            actor = _actor(event, event_index)
            states[actor].reach_declared = True
            last_discard = None
            continue
        if event_type == "dora":
            marker = _tile(event, "dora_marker", event_index, -1)
            public_visible_counts[normalize_tile(marker)] += 1
            last_discard = None
            continue
        if event_type in {"hora", "ryukyoku"}:
            last_discard = None
            continue
        raise ValueError(f"event {event_index}: unsupported event type {event_type!r}")

    singleton_observations = tuple(
        trackers[key].freeze()
        for key in sorted(trackers, key=lambda item: (item[0], tile_to_index(item[1])))
    )
    singleton_by_actor: dict[int, list[SingletonObservation]] = {
        actor: [] for actor in range(4)
    }
    for observation in singleton_observations:
        singleton_by_actor[observation.actor].append(observation)
    openings = tuple(
        _opening_observation(deal, singleton_by_actor[deal.actor], states[deal.actor])
        for deal in deals
        if deal.singleton_kind_count
    )
    _validate_kyoku_analysis(deals, openings, singleton_observations)
    return KyokuYakuhaiAnalysis(deals, openings, singleton_observations)


@dataclass
class DealAccumulator:
    player_deals: int = 0
    yakuhai_kind_opportunities: int = 0
    any_pair_deals: int = 0
    copy_count_distribution: Counter[int] = field(default_factory=Counter)
    pair_kind_count_distribution: Counter[int] = field(default_factory=Counter)
    singleton_kind_count_distribution: Counter[int] = field(default_factory=Counter)

    def add(self, observation: DealObservation) -> None:
        self.player_deals += 1
        self.any_pair_deals += observation.has_yakuhai_pair
        self.pair_kind_count_distribution[observation.yakuhai_pair_kind_count] += 1
        self.singleton_kind_count_distribution[observation.singleton_kind_count] += 1
        for _, count in observation.yakuhai_counts:
            self.yakuhai_kind_opportunities += 1
            self.copy_count_distribution[count] += 1

    def merge(self, other: DealAccumulator) -> None:
        self.player_deals += other.player_deals
        self.yakuhai_kind_opportunities += other.yakuhai_kind_opportunities
        self.any_pair_deals += other.any_pair_deals
        self.copy_count_distribution.update(other.copy_count_distribution)
        self.pair_kind_count_distribution.update(other.pair_kind_count_distribution)
        self.singleton_kind_count_distribution.update(
            other.singleton_kind_count_distribution
        )

    def validate(self) -> None:
        if (
            sum(self.copy_count_distribution.values())
            != self.yakuhai_kind_opportunities
        ):
            raise RuntimeError("yakuhai copy-count distribution does not sum")
        if sum(self.pair_kind_count_distribution.values()) != self.player_deals:
            raise RuntimeError("pair-kind distribution does not sum")
        if sum(self.singleton_kind_count_distribution.values()) != self.player_deals:
            raise RuntimeError("singleton-kind distribution does not sum")
        if self.any_pair_deals != sum(
            count
            for pair_kinds, count in self.pair_kind_count_distribution.items()
            if pair_kinds > 0
        ):
            raise RuntimeError("any-pair count disagrees with pair-kind distribution")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "player_deals": self.player_deals,
            "yakuhai_kind_opportunities": self.yakuhai_kind_opportunities,
            "any_yakuhai_pair_or_more": _rate(self.any_pair_deals, self.player_deals),
            "copy_count_distribution": [
                {"copies": copies, "count": self.copy_count_distribution[copies]}
                for copies in range(5)
            ],
            "yakuhai_pair_kind_count_distribution": _counter_rows(
                self.pair_kind_count_distribution, "pair_kind_count"
            ),
            "singleton_kind_count_distribution": _counter_rows(
                self.singleton_kind_count_distribution, "singleton_kind_count"
            ),
        }


@dataclass
class OpeningAccumulator:
    eligible_deals: int = 0
    strict_dash: int = 0
    dash_by_2: int = 0
    dash_by_3: int = 0
    first_discard_observed: int = 0
    by_max_suit_count: Counter[tuple[int, str]] = field(default_factory=Counter)
    by_honor_count: Counter[tuple[int, str]] = field(default_factory=Counter)
    by_shanten: Counter[tuple[int, str]] = field(default_factory=Counter)
    by_dora_han: Counter[tuple[int, str]] = field(default_factory=Counter)
    by_all_pair_kind_count: Counter[tuple[int, str]] = field(default_factory=Counter)
    by_opponent_discards_before_first: Counter[tuple[int, str]] = field(
        default_factory=Counter
    )

    def add(self, observation: OpeningObservation) -> None:
        self.eligible_deals += 1
        self.strict_dash += observation.strict_dash
        self.dash_by_2 += observation.dash_by_2
        self.dash_by_3 += observation.dash_by_3
        self.first_discard_observed += observation.actor_discard_count >= 1
        self._add_stratum(
            self.by_max_suit_count, observation.max_suit_count, observation
        )
        self._add_stratum(self.by_honor_count, observation.honor_count, observation)
        self._add_stratum(self.by_shanten, observation.shanten, observation)
        self._add_stratum(self.by_dora_han, observation.dora_han, observation)
        self._add_stratum(
            self.by_all_pair_kind_count, observation.all_pair_kind_count, observation
        )
        if observation.opponent_discards_before_first_discard is not None:
            self._add_stratum(
                self.by_opponent_discards_before_first,
                observation.opponent_discards_before_first_discard,
                observation,
            )

    @staticmethod
    def _add_stratum(
        counter: Counter[tuple[int, str]],
        value: int,
        observation: OpeningObservation,
    ) -> None:
        counter[(value, "eligible")] += 1
        counter[(value, "strict")] += observation.strict_dash
        counter[(value, "by_2")] += observation.dash_by_2
        counter[(value, "by_3")] += observation.dash_by_3

    def merge(self, other: OpeningAccumulator) -> None:
        self.eligible_deals += other.eligible_deals
        self.strict_dash += other.strict_dash
        self.dash_by_2 += other.dash_by_2
        self.dash_by_3 += other.dash_by_3
        self.first_discard_observed += other.first_discard_observed
        self.by_max_suit_count.update(other.by_max_suit_count)
        self.by_honor_count.update(other.by_honor_count)
        self.by_shanten.update(other.by_shanten)
        self.by_dora_han.update(other.by_dora_han)
        self.by_all_pair_kind_count.update(other.by_all_pair_kind_count)
        self.by_opponent_discards_before_first.update(
            other.by_opponent_discards_before_first
        )

    def validate(self) -> None:
        if not 0 <= self.strict_dash <= self.dash_by_2 <= self.dash_by_3:
            raise RuntimeError("opening dash counts are not nested")
        if self.dash_by_3 > self.eligible_deals:
            raise RuntimeError("opening dash count exceeds eligible deals")
        if self.strict_dash > self.first_discard_observed:
            raise RuntimeError("strict dash exceeds observed first discards")
        if self.first_discard_observed > self.eligible_deals:
            raise RuntimeError("observed first discards exceed eligible deals")
        for name, counter in (
            ("max suit", self.by_max_suit_count),
            ("honor", self.by_honor_count),
            ("shanten", self.by_shanten),
            ("dora", self.by_dora_han),
            ("all pairs", self.by_all_pair_kind_count),
            ("first-discard exposure", self.by_opponent_discards_before_first),
        ):
            expected = (
                self.first_discard_observed
                if name == "first-discard exposure"
                else self.eligible_deals
            )
            for field, total in (
                ("eligible", expected),
                ("strict", self.strict_dash),
                ("by_2", self.dash_by_2),
                ("by_3", self.dash_by_3),
            ):
                if sum(
                    count
                    for (_, metric), count in counter.items()
                    if metric == field
                ) != total:
                    raise RuntimeError(f"{name} {field} strata do not sum")
            for value, metric in counter:
                if metric != "eligible":
                    continue
                if not (
                    0
                    <= counter[(value, "strict")]
                    <= counter[(value, "by_2")]
                    <= counter[(value, "by_3")]
                    <= counter[(value, "eligible")]
                ):
                    raise RuntimeError(f"{name} dash stratum is not nested")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "eligible_player_deals": self.eligible_deals,
            "strict_dash": _rate(self.strict_dash, self.eligible_deals),
            "first_discard_observed": _rate(
                self.first_discard_observed, self.eligible_deals
            ),
            "strict_dash_given_first_discard": _rate(
                self.strict_dash, self.first_discard_observed
            ),
            "dash_by_2": _rate(self.dash_by_2, self.eligible_deals),
            "dash_by_3": _rate(self.dash_by_3, self.eligible_deals),
            "by_max_suit_count": _stratum_rows(self.by_max_suit_count),
            "by_honor_count": _stratum_rows(self.by_honor_count),
            "by_shanten": _stratum_rows(self.by_shanten),
            "by_dora_han": _stratum_rows(self.by_dora_han),
            "by_all_pair_kind_count": _stratum_rows(self.by_all_pair_kind_count),
            "by_opponent_discards_before_first_discard": _stratum_rows(
                self.by_opponent_discards_before_first
            ),
        }


@dataclass
class SingletonAccumulator:
    observations: int = 0
    outcomes: Counter[str] = field(default_factory=Counter)
    opponent_ready_at_deal: int = 0
    self_pair_by_draw: Counter[int] = field(default_factory=Counter)
    self_pair_followup: Counter[str] = field(default_factory=Counter)
    self_pair_followup_by_draw: Counter[tuple[int, str]] = field(
        default_factory=Counter
    )
    opponent_pair_by_focal_draws: Counter[int] = field(default_factory=Counter)
    singleton_discard_by_number: Counter[int] = field(default_factory=Counter)
    capable_discard_by_number: Counter[int] = field(default_factory=Counter)
    actual_pon_by_number: Counter[int] = field(default_factory=Counter)
    focal_pon_after_pair: int = 0
    pair_holder_but_riichi_blocked: int = 0
    pair_races: Counter[str] = field(default_factory=Counter)
    race_by_outcome: Counter[tuple[str, str]] = field(default_factory=Counter)
    post_discard_self_draw_by_number: Counter[int] = field(default_factory=Counter)
    post_discard_opponent_pair: int = 0
    post_discard_opponent_pair_after_no_capability: int = 0
    by_public_same_tile_before_first: Counter[tuple[int, str]] = field(
        default_factory=Counter
    )

    def add(self, observation: SingletonObservation) -> None:
        self.observations += 1
        self.outcomes[observation.outcome] += 1
        self.pair_races[observation.pair_race] += 1
        self.race_by_outcome[(observation.pair_race, observation.outcome)] += 1
        self.opponent_ready_at_deal += observation.opponent_ready_at_deal
        if observation.self_pair_draw_number is not None:
            draw_number = observation.self_pair_draw_number
            self.self_pair_by_draw[draw_number] += 1
            if observation.self_pair_followup_discard_event_index is None:
                followup_category = "no_followup_discard"
            elif not observation.self_pair_retained_after_discard:
                followup_category = "pair_broken"
            elif observation.self_pair_pon_shape_after_discard:
                followup_category = "pon_shape"
            else:
                followup_category = "riichi_blocked"
            self.self_pair_followup[followup_category] += 1
            self.self_pair_followup_by_draw[(draw_number, followup_category)] += 1
            if observation.self_pair_retained_after_discard:
                self.self_pair_followup["retained_pair"] += 1
                self.self_pair_followup_by_draw[(draw_number, "retained_pair")] += 1
        if observation.focal_draws_completed_at_opponent_pair is not None:
            self.opponent_pair_by_focal_draws[
                observation.focal_draws_completed_at_opponent_pair
            ] += 1
        if observation.singleton_discard_number is not None:
            number = observation.singleton_discard_number
            self.singleton_discard_by_number[number] += 1
            if observation.opponent_pon_capable_at_discard:
                self.capable_discard_by_number[number] += 1
            if observation.discard_was_ponned:
                self.actual_pon_by_number[number] += 1
            if (
                observation.opponent_pair_holder_actor_count
                and not observation.opponent_pon_capable_at_discard
            ):
                self.pair_holder_but_riichi_blocked += 1
        self.focal_pon_after_pair += observation.focal_pon_event_index is not None
        if observation.post_discard_self_draw_number is not None:
            self.post_discard_self_draw_by_number[
                observation.post_discard_self_draw_number
            ] += 1
        if observation.post_discard_opponent_pair_event_index is not None:
            self.post_discard_opponent_pair += 1
            if not observation.opponent_pon_capable_at_discard:
                self.post_discard_opponent_pair_after_no_capability += 1
        visibility = observation.public_same_tile_before_first_discard
        if visibility is not None:
            self.by_public_same_tile_before_first[(visibility, "observed")] += 1
            self.by_public_same_tile_before_first[
                (visibility, observation.outcome)
            ] += 1

    def merge(self, other: SingletonAccumulator) -> None:
        self.observations += other.observations
        self.outcomes.update(other.outcomes)
        self.opponent_ready_at_deal += other.opponent_ready_at_deal
        self.self_pair_by_draw.update(other.self_pair_by_draw)
        self.self_pair_followup.update(other.self_pair_followup)
        self.self_pair_followup_by_draw.update(other.self_pair_followup_by_draw)
        self.opponent_pair_by_focal_draws.update(other.opponent_pair_by_focal_draws)
        self.singleton_discard_by_number.update(other.singleton_discard_by_number)
        self.capable_discard_by_number.update(other.capable_discard_by_number)
        self.actual_pon_by_number.update(other.actual_pon_by_number)
        self.focal_pon_after_pair += other.focal_pon_after_pair
        self.pair_holder_but_riichi_blocked += other.pair_holder_but_riichi_blocked
        self.pair_races.update(other.pair_races)
        self.race_by_outcome.update(other.race_by_outcome)
        self.post_discard_self_draw_by_number.update(
            other.post_discard_self_draw_by_number
        )
        self.post_discard_opponent_pair += other.post_discard_opponent_pair
        self.post_discard_opponent_pair_after_no_capability += (
            other.post_discard_opponent_pair_after_no_capability
        )
        self.by_public_same_tile_before_first.update(
            other.by_public_same_tile_before_first
        )

    def validate(self) -> None:
        if sum(self.outcomes.values()) != self.observations:
            raise RuntimeError("singleton outcomes do not sum")
        if sum(self.self_pair_by_draw.values()) != self.outcomes["self_pair_draw"]:
            raise RuntimeError("self-pair draw distribution does not sum")
        followup_partition = (
            "pon_shape",
            "pair_broken",
            "riichi_blocked",
            "no_followup_discard",
        )
        if sum(self.self_pair_followup[name] for name in followup_partition) != self.outcomes[
            "self_pair_draw"
        ]:
            raise RuntimeError("self-pair follow-up categories do not sum")
        if self.self_pair_followup["retained_pair"] != (
            self.self_pair_followup["pon_shape"]
            + self.self_pair_followup["riichi_blocked"]
        ):
            raise RuntimeError("retained self-pair count does not sum")
        for name in (*followup_partition, "retained_pair"):
            if sum(
                count
                for (_, category), count in self.self_pair_followup_by_draw.items()
                if category == name
            ) != self.self_pair_followup[name]:
                raise RuntimeError("self-pair follow-up draw totals do not sum")
        if set(self.self_pair_followup) - {*followup_partition, "retained_pair"}:
            raise RuntimeError("unknown self-pair follow-up category")
        if any(
            draw_number not in self.self_pair_by_draw
            or category not in {*followup_partition, "retained_pair"}
            for draw_number, category in self.self_pair_followup_by_draw
        ):
            raise RuntimeError("unknown self-pair follow-up draw cohort")
        for draw_number, count in self.self_pair_by_draw.items():
            if sum(
                self.self_pair_followup_by_draw[(draw_number, name)]
                for name in followup_partition
            ) != count:
                raise RuntimeError("self-pair follow-up draw cohort does not sum")
            if self.self_pair_followup_by_draw[(draw_number, "retained_pair")] != (
                self.self_pair_followup_by_draw[(draw_number, "pon_shape")]
                + self.self_pair_followup_by_draw[(draw_number, "riichi_blocked")]
            ):
                raise RuntimeError("retained self-pair draw cohort does not sum")
        if (
            sum(self.singleton_discard_by_number.values())
            != self.outcomes["singleton_discard"]
        ):
            raise RuntimeError("singleton-discard distribution does not sum")
        capable = sum(self.capable_discard_by_number.values())
        actual = sum(self.actual_pon_by_number.values())
        if actual > capable or capable > self.outcomes["singleton_discard"]:
            raise RuntimeError("pon counts exceed their denominators")
        if self.pair_holder_but_riichi_blocked > self.outcomes[
            "singleton_discard"
        ] - capable:
            raise RuntimeError("riichi-blocked pair holders exceed non-capable discards")
        if self.focal_pon_after_pair > self.outcomes["self_pair_draw"]:
            raise RuntimeError("focal pon count exceeds self-pair outcomes")
        if sum(self.opponent_pair_by_focal_draws.values()) > self.observations:
            raise RuntimeError("opponent pair arrivals exceed observations")
        if sum(self.pair_races.values()) != self.observations:
            raise RuntimeError("first-pair race classes do not sum")
        if sum(self.race_by_outcome.values()) != self.observations:
            raise RuntimeError("race/outcome cross table does not sum")
        for race, count in self.pair_races.items():
            if sum(
                cross_count
                for (cross_race, _), cross_count in self.race_by_outcome.items()
                if cross_race == race
            ) != count:
                raise RuntimeError(f"{race} race/outcome row does not sum")
        if sum(self.post_discard_self_draw_by_number.values()) > self.outcomes[
            "singleton_discard"
        ]:
            raise RuntimeError("post-discard self draws exceed discarded singletons")
        if not (
            0
            <= self.post_discard_opponent_pair_after_no_capability
            <= self.post_discard_opponent_pair
            <= self.outcomes["singleton_discard"]
        ):
            raise RuntimeError("post-discard opponent pair counts are invalid")
        if sum(
            count
            for (_, label), count in self.by_public_same_tile_before_first.items()
            if label == "observed"
        ) > self.observations:
            raise RuntimeError("visibility strata exceed observations")
        for (copies, label), count in self.by_public_same_tile_before_first.items():
            if label != "observed":
                continue
            if count != sum(
                outcome_count
                for (other_copies, outcome), outcome_count in (
                    self.by_public_same_tile_before_first.items()
                )
                if other_copies == copies and outcome != "observed"
            ):
                raise RuntimeError("visibility outcome stratum does not sum")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        pair_count = self.outcomes["self_pair_draw"]
        discard_count = self.outcomes["singleton_discard"]
        capable = sum(self.capable_discard_by_number.values())
        actual = sum(self.actual_pon_by_number.values())
        race_order = (
            "opponent_at_deal",
            "self_first",
            "opponent_first_after_deal",
            "discard_before_either_pair",
            "round_end_before_either_pair",
        )
        outcome_order = ("self_pair_draw", "singleton_discard", "round_end")
        post_discard_self_draws = sum(self.post_discard_self_draw_by_number.values())
        followup_names = (
            "retained_pair",
            "pon_shape",
            "pair_broken",
            "riichi_blocked",
            "no_followup_discard",
        )
        followup_by_draw = {
            name: Counter(
                {
                    draw_number: count
                    for (draw_number, category), count in (
                        self.self_pair_followup_by_draw.items()
                    )
                    if category == name
                }
            )
            for name in ("retained_pair", "pon_shape")
        }
        return {
            "initial_singleton_observations": self.observations,
            "outcomes": [
                {
                    "outcome": outcome,
                    **_rate(self.outcomes[outcome], self.observations),
                }
                for outcome in outcome_order
            ],
            "first_pair_race": [
                {
                    "race": race,
                    **_rate(self.pair_races[race], self.observations),
                    "by_outcome": [
                        {
                            "outcome": outcome,
                            "count": self.race_by_outcome[(race, outcome)],
                        }
                        for outcome in outcome_order
                    ],
                }
                for race in race_order
            ],
            "opponent_ready_at_deal": _rate(
                self.opponent_ready_at_deal, self.observations
            ),
            "self_pair_formation_by_draw": _cumulative_rows(
                self.self_pair_by_draw,
                self.observations,
                start=1,
                minimum_end=18,
                key_name="self_draw_number",
            ),
            "self_pair_followup": {
                **{
                    name: _rate(self.self_pair_followup[name], pair_count)
                    for name in followup_names
                },
                "by_self_pair_draw_number": [
                    {
                        "pair_formation_self_draw_number": draw_number,
                        "self_pair_draw": _rate(
                            self.self_pair_by_draw[draw_number], self.observations
                        ),
                        **{
                            name: _rate(
                                self.self_pair_followup_by_draw[(draw_number, name)],
                                self.observations,
                            )
                            for name in followup_names
                        },
                    }
                    for draw_number in range(
                        1, max(18, max(self.self_pair_by_draw, default=0)) + 1
                    )
                ],
                "retained_pair_cumulative_by_pair_formation_draw": _cumulative_rows(
                    followup_by_draw["retained_pair"],
                    self.observations,
                    start=1,
                    minimum_end=18,
                    key_name="pair_formation_self_draw_number",
                ),
                "pon_shape_cumulative_by_pair_formation_draw": _cumulative_rows(
                    followup_by_draw["pon_shape"],
                    self.observations,
                    start=1,
                    minimum_end=18,
                    key_name="pair_formation_self_draw_number",
                ),
            },
            "opponent_first_pair_by_focal_draws_completed": _cumulative_rows(
                self.opponent_pair_by_focal_draws,
                self.observations,
                start=0,
                minimum_end=18,
                key_name="focal_draws_completed",
            ),
            "singleton_discards_by_actor_discard_number": [
                {
                    "actor_discard_number": number,
                    "singleton_discards": self.singleton_discard_by_number[number],
                    "share_of_all_initial_singletons": _rate(
                        self.singleton_discard_by_number[number], self.observations
                    ),
                    "opponent_pon_capable": _rate(
                        self.capable_discard_by_number[number],
                        self.singleton_discard_by_number[number],
                    ),
                    "actual_pon": _rate(
                        self.actual_pon_by_number[number],
                        self.singleton_discard_by_number[number],
                    ),
                }
                for number in range(
                    1,
                    max(18, max(self.singleton_discard_by_number, default=0)) + 1,
                )
            ],
            "opponent_pon_capable_at_singleton_discard": _rate(capable, discard_count),
            "actual_pon_at_singleton_discard": _rate(actual, discard_count),
            "actual_pon_given_capable": _rate(actual, capable),
            "only_riichi_pair_holders_at_discard": _rate(
                self.pair_holder_but_riichi_blocked, discard_count
            ),
            "focal_pon_after_self_pair": _rate(self.focal_pon_after_pair, pair_count),
            "post_discard_self_draw_in_actual_log": _rate(
                post_discard_self_draws, discard_count
            ),
            "post_discard_self_draw_by_actor_draw_number": [
                {
                    "self_draw_number": number,
                    "count": self.post_discard_self_draw_by_number[number],
                }
                for number in sorted(self.post_discard_self_draw_by_number)
            ],
            "post_discard_opponent_pair_in_actual_log": _rate(
                self.post_discard_opponent_pair, discard_count
            ),
            "post_discard_opponent_pair_given_no_capability_at_discard": _rate(
                self.post_discard_opponent_pair_after_no_capability,
                discard_count - capable,
            ),
            "by_public_same_tile_before_first_discard": [
                {
                    "public_copies": copies,
                    "observed_singletons": self.by_public_same_tile_before_first[
                        (copies, "observed")
                    ],
                    "outcomes": [
                        {
                            "outcome": outcome,
                            "count": self.by_public_same_tile_before_first[
                                (copies, outcome)
                            ],
                        }
                        for outcome in outcome_order
                    ],
                }
                for copies in sorted(
                    {
                        copies
                        for copies, label in self.by_public_same_tile_before_first
                        if label == "observed"
                    }
                )
            ],
        }


@dataclass
class YakuhaiDashAccumulator:
    """Mergeable complete aggregate for one period."""

    east_kyokus: int = 0
    deals: dict[str, DealAccumulator] = field(default_factory=dict)
    openings: dict[str, OpeningAccumulator] = field(default_factory=dict)
    singletons: dict[str, SingletonAccumulator] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.deals:
            self.deals = {
                key: DealAccumulator() for key in ("all", "dealer", "nondealer")
            }
        if not self.openings:
            self.openings = {
                key: OpeningAccumulator() for key in ("all", "dealer", "nondealer")
            }
        if not self.singletons:
            keys = (
                "all",
                "role:dealer",
                "role:nondealer",
                "exposure:shared",
                "exposure:own_only",
                *(f"class:{name}" for name in YAKUHAI_CLASSES),
                *(f"tile:{tile}" for tile in YAKUHAI_TILES),
            )
            self.singletons = {key: SingletonAccumulator() for key in keys}

    def add(self, analysis: KyokuYakuhaiAnalysis) -> None:
        self.east_kyokus += 1
        for observation in analysis.deals:
            self.deals["all"].add(observation)
            self.deals[observation.role].add(observation)
        for observation in analysis.openings:
            self.openings["all"].add(observation)
            self.openings[observation.role].add(observation)
        for observation in analysis.singletons:
            for key in (
                "all",
                f"role:{observation.role}",
                f"exposure:{observation.exposure}",
                f"class:{observation.yakuhai_class}",
                f"tile:{observation.tile}",
            ):
                self.singletons[key].add(observation)

    def merge(self, other: YakuhaiDashAccumulator) -> None:
        self.east_kyokus += other.east_kyokus
        for key in self.deals:
            self.deals[key].merge(other.deals[key])
        for key in self.openings:
            self.openings[key].merge(other.openings[key])
        for key in self.singletons:
            self.singletons[key].merge(other.singletons[key])

    def validate(self) -> None:
        if self.deals["all"].player_deals != self.east_kyokus * 4:
            raise RuntimeError("player-deal count is not four times east kyokus")
        if (
            self.deals["dealer"].player_deals + self.deals["nondealer"].player_deals
            != self.deals["all"].player_deals
        ):
            raise RuntimeError("dealer and nondealer deals do not partition all deals")
        if self.deals["dealer"].player_deals != self.east_kyokus:
            raise RuntimeError("dealer deal count differs from east kyokus")
        if self.deals["nondealer"].player_deals != self.east_kyokus * 3:
            raise RuntimeError(
                "nondealer deal count differs from three times east kyokus"
            )
        for accumulator in self.deals.values():
            accumulator.validate()
        for accumulator in self.openings.values():
            accumulator.validate()
        for accumulator in self.singletons.values():
            accumulator.validate()
        if (
            self.singletons["role:dealer"].observations
            + self.singletons["role:nondealer"].observations
            != self.singletons["all"].observations
        ):
            raise RuntimeError("singleton roles do not partition all observations")
        if (
            self.singletons["exposure:shared"].observations
            + self.singletons["exposure:own_only"].observations
            != self.singletons["all"].observations
        ):
            raise RuntimeError("singleton exposures do not partition all observations")
        if (
            self.openings["dealer"].eligible_deals
            + self.openings["nondealer"].eligible_deals
            != self.openings["all"].eligible_deals
        ):
            raise RuntimeError("opening roles do not partition all eligible deals")
        if (
            sum(self.singletons[f"class:{name}"].observations for name in YAKUHAI_CLASSES)
            != self.singletons["all"].observations
        ):
            raise RuntimeError("yakuhai classes do not partition all singletons")
        if (
            sum(self.singletons[f"tile:{tile}"].observations for tile in YAKUHAI_TILES)
            != self.singletons["all"].observations
        ):
            raise RuntimeError("tile kinds do not partition all singletons")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "east_kyokus": self.east_kyokus,
            "initial_deals": {
                "all": self.deals["all"].to_dict(),
                "by_role": [
                    {"role": role, **self.deals[role].to_dict()}
                    for role in ("dealer", "nondealer")
                ],
            },
            "opening_behavior": {
                "all": self.openings["all"].to_dict(),
                "by_role": [
                    {"role": role, **self.openings[role].to_dict()}
                    for role in ("dealer", "nondealer")
                ],
            },
            "initial_singletons": {
                "all": self.singletons["all"].to_dict(),
                "by_role": [
                    {
                        "role": role,
                        **self.singletons[f"role:{role}"].to_dict(),
                    }
                    for role in ("dealer", "nondealer")
                ],
                "by_exposure": [
                    {
                        "exposure": exposure,
                        **self.singletons[f"exposure:{exposure}"].to_dict(),
                    }
                    for exposure in ("shared", "own_only")
                ],
                "by_yakuhai_class": [
                    {
                        "yakuhai_class": yakuhai_class,
                        **self.singletons[f"class:{yakuhai_class}"].to_dict(),
                    }
                    for yakuhai_class in YAKUHAI_CLASSES
                ],
                "by_tile": [
                    {"tile": tile, **self.singletons[f"tile:{tile}"].to_dict()}
                    for tile in YAKUHAI_TILES
                ],
            },
        }


def deal_to_dict(observation: DealObservation) -> dict[str, Any]:
    return {
        "actor": observation.actor,
        "role": observation.role,
        "seat_wind": observation.seat_wind,
        "initial_hand": list(observation.initial_hand),
        "yakuhai_counts": [
            {"tile": tile, "copies": count}
            for tile, count in observation.yakuhai_counts
        ],
        "yakuhai_pair_kind_count": observation.yakuhai_pair_kind_count,
        "singleton_kind_count": observation.singleton_kind_count,
        "max_suit_count": observation.max_suit_count,
        "honor_count": observation.honor_count,
        "terminal_honor_count": observation.terminal_honor_count,
        "all_pair_kind_count": observation.all_pair_kind_count,
        "shanten": observation.shanten,
        "dora_han": observation.dora_han,
    }


def singleton_to_dict(observation: SingletonObservation) -> dict[str, Any]:
    return {
        "actor": observation.actor,
        "role": observation.role,
        "seat_wind": observation.seat_wind,
        "tile": observation.tile,
        "yakuhai_class": observation.yakuhai_class,
        "exposure": observation.exposure,
        "initial_hand": list(observation.initial_hand),
        "opponent_ready_at_deal": observation.opponent_ready_at_deal,
        "opponent_ready_actor_count_at_deal": (
            observation.opponent_ready_actor_count_at_deal
        ),
        "first_opponent_pair_event_index": (
            observation.first_opponent_pair_event_index
        ),
        "first_opponent_pair_actor": observation.first_opponent_pair_actor,
        "focal_draws_completed_at_opponent_pair": (
            observation.focal_draws_completed_at_opponent_pair
        ),
        "outcome": observation.outcome,
        "self_pair_draw_number": observation.self_pair_draw_number,
        "self_pair_event_index": observation.self_pair_event_index,
        "self_pair_followup_discard_event_index": (
            observation.self_pair_followup_discard_event_index
        ),
        "self_pair_retained_after_discard": (
            observation.self_pair_retained_after_discard
        ),
        "self_pair_pon_shape_after_discard": (
            observation.self_pair_pon_shape_after_discard
        ),
        "singleton_discard_number": observation.singleton_discard_number,
        "singleton_discard_event_index": observation.singleton_discard_event_index,
        "opponent_pon_capable_at_discard": (
            observation.opponent_pon_capable_at_discard
        ),
        "opponent_pon_capable_actor_count": (
            observation.opponent_pon_capable_actor_count
        ),
        "opponent_pair_holder_actor_count": (
            observation.opponent_pair_holder_actor_count
        ),
        "discard_was_ponned": observation.discard_was_ponned,
        "discard_pon_actor": observation.discard_pon_actor,
        "discard_pon_event_index": observation.discard_pon_event_index,
        "focal_pon_event_index": observation.focal_pon_event_index,
        "pair_race": observation.pair_race,
        "public_same_tile_before_first_discard": (
            observation.public_same_tile_before_first_discard
        ),
        "post_discard_self_draw_number": observation.post_discard_self_draw_number,
        "post_discard_self_draw_event_index": (
            observation.post_discard_self_draw_event_index
        ),
        "post_discard_opponent_pair_event_index": (
            observation.post_discard_opponent_pair_event_index
        ),
        "post_discard_opponent_pair_actor": (
            observation.post_discard_opponent_pair_actor
        ),
    }


def theoretical_baselines() -> dict[str, Any]:
    """Return fixed combinatorial baselines used by the result document."""
    dealer = theoretical_any_pair_probability(4)
    nondealer = theoretical_any_pair_probability(5)
    return {
        "model": "uniform 13-tile deal from 136 tiles; draw curve samples from 123 unknown tiles",
        "specific_kind_pair_or_more": theoretical_specific_pair_probability(),
        "any_yakuhai_pair_or_more": {
            "dealer_four_unique_kinds": dealer,
            "nondealer_five_unique_kinds": nondealer,
            "four_player_role_weighted_average": (dealer + 3 * nondealer) / 4,
        },
        "singleton_pair_cumulative_by_self_draw": [
            {
                "self_draw_number": draw,
                "cumulative_probability": theoretical_singleton_pair_by_draw(draw),
            }
            for draw in range(1, 19)
        ],
    }


def _initial_state(
    kyoku: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[_PlayerState], int, str]:
    if not kyoku:
        raise ValueError("kyoku is empty")
    start = _mapping(kyoku[0], 0)
    if start.get("type") != "start_kyoku":
        raise ValueError("kyoku must start with start_kyoku")
    if not isinstance(kyoku[-1], Mapping) or kyoku[-1].get("type") != "end_kyoku":
        raise ValueError("kyoku must end with end_kyoku")
    round_wind = start.get("bakaze")
    if round_wind != "E":
        raise ValueError("yakuhai dash analysis requires an east-round kyoku")
    dealer = start.get("oya")
    _validate_actor_value(dealer, "start_kyoku oya")
    tehais = start.get("tehais")
    if not isinstance(tehais, list) or len(tehais) != 4:
        raise ValueError("start_kyoku tehais must contain four hands")
    states: list[_PlayerState] = []
    all_tiles: list[str] = []
    for actor, hand in enumerate(tehais):
        if not isinstance(hand, list) or len(hand) != 13:
            raise ValueError(
                f"event 0, actor {actor}: initial tehai must contain 13 tiles"
            )
        if not all(isinstance(tile, str) for tile in hand):
            raise TypeError(
                f"event 0, actor {actor}: initial tehai must contain tile strings"
            )
        tiles_to_counts(hand)
        states.append(_PlayerState(list(hand)))
        all_tiles.extend(hand)
    tiles_to_counts(all_tiles)
    return start, states, dealer, round_wind


def _deal_observation(
    actor: int,
    dealer: int,
    round_wind: str,
    dora_kind: str,
    hand: list[str],
) -> DealObservation:
    normalized = tuple(normalize_tile(tile) for tile in hand)
    counts = Counter(normalized)
    suits = Counter(tile[-1] for tile in normalized if tile[-1:] in "mps")
    yakuhai_counts = tuple(
        (tile, counts[tile]) for tile in yakuhai_kinds(actor, dealer, round_wind)
    )
    return DealObservation(
        actor=actor,
        dealer=actor == dealer,
        seat_wind=seat_wind(actor, dealer),
        initial_hand=tuple(hand),
        yakuhai_counts=yakuhai_counts,
        max_suit_count=max(suits.values(), default=0),
        honor_count=sum(tile in HONORS for tile in normalized),
        terminal_honor_count=sum(
            tile in HONORS or (tile[-1:] in "mps" and tile[0] in "19")
            for tile in normalized
        ),
        all_pair_kind_count=sum(value >= 2 for value in counts.values()),
        shanten=initial_shanten(hand),
        dora_han=sum(tile == dora_kind for tile in normalized)
        + sum(tile in {"5mr", "5pr", "5sr"} for tile in hand),
    )


def _opening_observation(
    deal: DealObservation,
    observations: list[SingletonObservation],
    state: _PlayerState,
) -> OpeningObservation:
    discard_numbers = tuple(
        observation.singleton_discard_number
        for observation in observations
        if observation.singleton_discard_number is not None
    )
    return OpeningObservation(
        actor=deal.actor,
        dealer=deal.dealer,
        initial_singleton_count=deal.singleton_kind_count,
        strict_dash=1 in discard_numbers,
        dash_by_2=any(number <= 2 for number in discard_numbers),
        dash_by_3=any(number <= 3 for number in discard_numbers),
        max_suit_count=deal.max_suit_count,
        honor_count=deal.honor_count,
        all_pair_kind_count=deal.all_pair_kind_count,
        shanten=deal.shanten,
        dora_han=deal.dora_han,
        first_discard_event_index=state.first_discard_event_index,
        opponent_discards_before_first_discard=(
            state.opponent_discards_before_first_discard
        ),
        actor_discard_count=state.discard_count,
    )


def _validate_kyoku_analysis(
    deals: tuple[DealObservation, ...],
    openings: tuple[OpeningObservation, ...],
    singletons: tuple[SingletonObservation, ...],
) -> None:
    if len(deals) != 4:
        raise RuntimeError("kyoku analysis must contain four deal observations")
    expected_singletons = sum(deal.singleton_kind_count for deal in deals)
    if len(singletons) != expected_singletons:
        raise RuntimeError("singleton observation count disagrees with deals")
    if len(openings) != sum(deal.singleton_kind_count > 0 for deal in deals):
        raise RuntimeError("opening observation count disagrees with eligible deals")
    for observation in singletons:
        if observation.outcome == "self_pair_draw":
            if (
                observation.self_pair_draw_number is None
                or observation.self_pair_event_index is None
            ):
                raise RuntimeError("self-pair outcome lacks draw metadata")
            followup_event = observation.self_pair_followup_discard_event_index
            retained = observation.self_pair_retained_after_discard
            pon_shape = observation.self_pair_pon_shape_after_discard
            if followup_event is None:
                if retained is not None or pon_shape is not None:
                    raise RuntimeError("self-pair follow-up flags lack discard event")
            elif (
                followup_event <= observation.self_pair_event_index
                or type(retained) is not bool
                or type(pon_shape) is not bool
                or (pon_shape and not retained)
            ):
                raise RuntimeError("self-pair follow-up metadata is inconsistent")
        elif observation.outcome == "singleton_discard":
            if (
                observation.singleton_discard_number is None
                or observation.singleton_discard_event_index is None
                or observation.opponent_pon_capable_at_discard is None
            ):
                raise RuntimeError("singleton-discard outcome lacks discard metadata")
        if observation.outcome != "self_pair_draw" and any(
            value is not None
            for value in (
                observation.self_pair_followup_discard_event_index,
                observation.self_pair_retained_after_discard,
                observation.self_pair_pon_shape_after_discard,
            )
        ):
            raise RuntimeError("non-self-pair outcome has follow-up metadata")
        if (
            observation.discard_was_ponned
            and not observation.opponent_pon_capable_at_discard
        ):
            raise RuntimeError("actual pon lacks pre-discard capability")


def _ready_opponents(
    states: list[_PlayerState], focal_actor: int, tile: str
) -> tuple[int, ...]:
    return tuple(
        actor
        for actor in _pair_holding_opponents(states, focal_actor, tile)
        if not states[actor].riichi_accepted
    )


def _pair_holding_opponents(
    states: list[_PlayerState], focal_actor: int, tile: str
) -> tuple[int, ...]:
    return tuple(
        actor
        for actor, state in enumerate(states)
        if actor != focal_actor and _count_kind(state.concealed_tiles, tile) >= 2
    )


def _count_kind(tiles: Iterable[str], tile: str) -> int:
    normalized = normalize_tile(tile)
    return sum(normalize_tile(owned) == normalized for owned in tiles)


def _mapping(value: object, event_index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"event {event_index}: MJAI event must be an object")
    return value


def _event_type(event: dict[str, Any], event_index: int) -> str:
    event_type = event.get("type")
    if not isinstance(event_type, str):
        raise TypeError(f"event {event_index}: event type must be a string")
    return event_type


def _actor(event: dict[str, Any], event_index: int) -> int:
    actor = event.get("actor")
    _validate_actor_value(actor, f"event {event_index} actor")
    return actor


def _target(event: dict[str, Any], event_index: int, actor: int) -> int:
    target = event.get("target")
    _validate_actor_value(target, f"event {event_index} target")
    if target == actor:
        raise _error(event_index, actor, "call target must differ from actor")
    return target


def _validate_actor_value(value: object, name: str) -> None:
    if type(value) is not int or value not in range(4):
        raise ValueError(f"{name} must be an integer from 0 through 3")


def _tile(event: dict[str, Any], field: str, event_index: int, actor: int) -> str:
    tile = event.get(field)
    if not isinstance(tile, str):
        raise _error(event_index, actor, f"{field} must be a tile string")
    try:
        normalize_tile(tile)
    except ValueError as error:
        raise _error(event_index, actor, f"invalid {field}: {tile!r}") from error
    return tile


def _consumed(
    event: dict[str, Any], expected: int, event_index: int, actor: int
) -> tuple[str, ...]:
    consumed = event.get("consumed")
    if not isinstance(consumed, list) or len(consumed) != expected:
        raise _error(
            event_index, actor, f"consumed must contain exactly {expected} tiles"
        )
    result: list[str] = []
    for tile in consumed:
        if not isinstance(tile, str):
            raise _error(event_index, actor, "consumed must contain tile strings")
        try:
            normalize_tile(tile)
        except ValueError as error:
            raise _error(
                event_index, actor, f"invalid consumed tile: {tile!r}"
            ) from error
        result.append(tile)
    return tuple(result)


def _remove_tiles(
    state: _PlayerState,
    tiles: tuple[str, ...],
    event_index: int,
    actor: int,
) -> None:
    available = Counter(state.concealed_tiles)
    missing = Counter(tiles) - available
    if missing:
        raise _error(
            event_index,
            actor,
            f"concealed hand does not contain raw tiles: {dict(missing)}",
        )
    for tile in tiles:
        state.concealed_tiles.remove(tile)
    _validate_owned_tiles(state, event_index, actor)


def _validate_owned_tiles(state: _PlayerState, event_index: int, actor: int) -> None:
    try:
        tiles_to_counts(state.concealed_tiles)
    except ValueError as error:
        raise _error(event_index, actor, str(error)) from error


def _error(event_index: int, actor: int, message: str) -> ValueError:
    return ValueError(f"event {event_index}, actor {actor}: {message}")


def _rate(count: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "count": count,
        "denominator": denominator,
        "rate": count / denominator if denominator else None,
    }


def _counter_rows(counter: Counter[int], key_name: str) -> list[dict[str, int]]:
    if not counter:
        return []
    return [
        {key_name: key, "count": counter[key]} for key in range(0, max(counter) + 1)
    ]


def _stratum_rows(counter: Counter[tuple[int, str]]) -> list[dict[str, Any]]:
    values = sorted({value for value, name in counter if name == "eligible"})
    return [
        {
            "value": value,
            "eligible_player_deals": counter[(value, "eligible")],
            "strict_dash": _rate(
                counter[(value, "strict")], counter[(value, "eligible")]
            ),
            "dash_by_2": _rate(counter[(value, "by_2")], counter[(value, "eligible")]),
            "dash_by_3": _rate(counter[(value, "by_3")], counter[(value, "eligible")]),
        }
        for value in values
    ]


def _cumulative_rows(
    counter: Counter[int],
    denominator: int,
    *,
    start: int,
    minimum_end: int,
    key_name: str,
) -> list[dict[str, Any]]:
    end = max(minimum_end, max(counter, default=start))
    cumulative = 0
    rows: list[dict[str, Any]] = []
    for key in range(start, end + 1):
        exact = counter[key]
        cumulative += exact
        rows.append(
            {
                key_name: key,
                "exact_count": exact,
                "cumulative": _rate(cumulative, denominator),
            }
        )
    return rows
