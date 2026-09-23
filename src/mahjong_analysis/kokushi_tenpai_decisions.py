"""Extract first kokushi-tenpai decisions and their outcomes from Tenhou DBs."""

from __future__ import annotations

import gzip
import json
import sqlite3
import xml.etree.ElementTree as ET
import zlib
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from mahjong_analysis.kokushi_riichi import (
    SUPPORTED_YEARS,
    KokushiRiichiParseError,
    KokushiWaitType,
    ReachType,
    _decode_concealed_meld_tiles,
    _discard_actor,
    _draw_actor,
    _parse_actor_attribute,
    _parse_initial_hand,
    _remove_exact_tiles,
    _remove_identical_terminal_attributes,
    _tile_id_from_tag,
    _validate_unique_initial_tiles,
    classify_kokushi_wait,
    tenhou_replay_url,
)
from mahjong_analysis.tiles import tile_to_index

DecisionStrategy = Literal["immediate_riichi", "delayed_riichi", "dama"]
WaitGroup = Literal["honor_single", "terminal_single", "thirteen_sided"]
Outcome = Literal[
    "win",
    "deal_in",
    "opponent_tsumo",
    "other_ron",
    "draw",
]
WinMethod = Literal["ron", "tsumo"]

_REQUIRED_COLUMNS = frozenset(
    {"id", "date", "num_players", "is_tonpu", "is_processed", "was_error", "log"}
)
_WIND_NAMES = ("E", "S", "W", "N")


@dataclass(frozen=True)
class KokushiTenpaiDecisionRecord:
    """The first post-discard kokushi tenpai by one player in one kyoku."""

    year: int
    date: str
    log_id: str
    kyoku_index: int
    who: int
    decision_tj: int
    turn: int
    bakaze: str
    kyoku_number: int
    honba: int
    dealer: bool
    scores: tuple[int, int, int, int] | None
    actor_score: int | None
    actor_rank: int | None
    wait_type: KokushiWaitType
    wait_group: WaitGroup
    waits: tuple[str, ...]
    visible_wait_counts: tuple[int, ...]
    unseen_wait_counts: tuple[int, ...]
    furiten: bool
    prior_riichis: int
    any_call: bool
    opponent_open_hands: int
    opponent_called_hands: int
    remaining_draws_estimate: int
    riichi_eligible: bool | None
    strategy: DecisionStrategy
    riichi_declared: bool
    riichi_established: bool
    riichi_turn: int | None
    riichi_delay_turns: int | None
    riichi_reach_type: ReachType | None
    riichi_wait_type: KokushiWaitType | None
    riichi_waits: tuple[str, ...]
    outcome: Outcome
    won: bool
    win_method: WinMethod | None
    winners: tuple[int, ...]
    from_who: int | None
    draw_type: str | None
    point_delta: int | None
    url: str

    def __post_init__(self) -> None:
        if self.strategy == "dama" and self.riichi_declared:
            raise ValueError("dama decision cannot contain a later kokushi riichi")
        if self.riichi_established and not self.riichi_declared:
            raise ValueError("an established riichi must have been declared")
        if self.riichi_declared != (self.riichi_turn is not None):
            raise ValueError("riichi_turn must be present exactly for riichi decisions")
        if len(self.waits) != len(self.visible_wait_counts) or len(self.waits) != len(
            self.unseen_wait_counts
        ):
            raise ValueError("wait availability counts must align with waits")
        if self.won != (self.outcome == "win"):
            raise ValueError("won must match outcome")
        if self.won != (self.win_method is not None):
            raise ValueError("win_method must be present exactly for wins")


@dataclass(frozen=True)
class KokushiTenpaiYearSummary:
    year: int
    eligible_logs: int
    available_logs: int
    scanned_logs: int
    unavailable_logs: int
    normalized_xml_logs: int
    total_kyokus: int
    decisions: int
    riichi_eligible_decisions: int
    immediate_riichi: int
    delayed_riichi: int
    dama: int
    declared_riichis: int
    established_riichis: int


@dataclass(frozen=True)
class KokushiTenpaiDecisionResult:
    years: tuple[int, ...]
    max_logs_per_year: int | None
    yearly: tuple[KokushiTenpaiYearSummary, ...]
    records: tuple[KokushiTenpaiDecisionRecord, ...]


@dataclass
class _Decision:
    year: int
    date: str
    log_id: str
    kyoku_index: int
    who: int
    decision_tj: int
    turn: int
    bakaze: str
    kyoku_number: int
    honba: int
    dealer: bool
    scores: tuple[int, int, int, int] | None
    actor_score: int | None
    actor_rank: int | None
    wait_type: KokushiWaitType
    wait_group: WaitGroup
    waits: tuple[str, ...]
    visible_wait_counts: tuple[int, ...]
    unseen_wait_counts: tuple[int, ...]
    furiten: bool
    prior_riichis: int
    any_call: bool
    opponent_open_hands: int
    opponent_called_hands: int
    remaining_draws_estimate: int
    riichi_eligible: bool | None
    strategy: DecisionStrategy = "dama"
    riichi_declared: bool = False
    riichi_established: bool = False
    riichi_turn: int | None = None
    riichi_reach_type: ReachType | None = None
    riichi_wait_type: KokushiWaitType | None = None
    riichi_waits: tuple[str, ...] = ()


@dataclass
class _PendingReach:
    actor: int
    reach_type: ReachType
    turn: int | None = None
    kokushi: tuple[KokushiWaitType, tuple[str, ...]] | None = None
    decision: _Decision | None = None


@dataclass(frozen=True)
class _LogResult:
    total_kyokus: int
    normalized_xml_logs: int
    records: tuple[KokushiTenpaiDecisionRecord, ...]


def analyze_tenhou_log(
    payload: bytes, *, year: int, date: str, log_id: str
) -> _LogResult:
    """Replay one log and retain the first kokushi-tenpai choice per actor/kyoku."""
    root, normalized = _parse_root(payload)
    hands: list[list[int]] | None = None
    track_hand = [True] * 4
    last_drawn: list[int | None] = [None] * 4
    discards = [0] * 4
    river_kinds: list[set[int]] = [set() for _ in range(4)]
    open_hands = [False] * 4
    called_hands = [False] * 4
    accepted_actors: set[int] = set()
    visible_ids: set[int] = set()
    draw_count = 0
    any_call = False
    pending: _PendingReach | None = None
    current_scores: list[int] | None = None
    round_meta: tuple[str, int, int, int] | None = None
    round_ended = True
    terminal_events: list[ET.Element] = []
    decisions: dict[int, _Decision] = {}
    all_records: list[KokushiTenpaiDecisionRecord] = []
    kyoku_index = -1
    viewer_tj = -1

    def finish_round() -> None:
        nonlocal terminal_events
        if not decisions:
            terminal_events = []
            return
        all_records.extend(
            _finalize_decision(decision, terminal_events)
            for decision in decisions.values()
        )
        terminal_events = []

    for event_index, event in enumerate(root):
        tag = event.tag
        if tag == "INIT":
            if pending is not None:
                raise _error(event_index, "new INIT before reach sequence ended")
            if kyoku_index >= 0:
                if not round_ended:
                    raise _error(event_index, "new INIT before previous kyoku result")
                finish_round()
            kyoku_index += 1
            viewer_tj = -1
            hands = [
                _parse_initial_hand(event, actor, event_index) for actor in range(4)
            ]
            _validate_unique_initial_tiles(hands, event_index)
            track_hand = [True] * 4
            last_drawn = [None] * 4
            discards = [0] * 4
            river_kinds = [set() for _ in range(4)]
            open_hands = [False] * 4
            called_hands = [False] * 4
            accepted_actors = set()
            visible_ids = set()
            draw_count = 0
            any_call = False
            decisions = {}
            current_scores = _parse_scores(event, event_index)
            round_meta = _parse_round_meta(event, event_index)
            dora_id = _parse_initial_dora(event, event_index)
            if dora_id is not None:
                visible_ids.add(dora_id)
            round_ended = False
            continue

        if hands is None:
            if (
                tag in {"REACH", "N", "AGARI", "RYUUKYOKU"}
                or _draw_actor(tag) is not None
                or _discard_actor(tag) is not None
            ):
                raise _error(event_index, "state event before INIT")
            continue
        viewer_tj += 1
        if tag in {"SHUFFLE", "GO", "UN", "TAIKYOKU", "BYE"}:
            continue
        if round_ended:
            if tag == "AGARI":
                terminal_events.append(event)
                continue
            raise _error(event_index, "state event after kyoku result")

        if tag == "DORA":
            visible_ids.add(_required_tile_attribute(event, "hai", event_index))
            continue
        if pending is not None:
            if pending.turn is None and _discard_actor(tag) != pending.actor:
                raise _error(event_index, "reach must be followed by actor discard")
            if pending.turn is not None and (
                _draw_actor(tag) is not None or _discard_actor(tag) is not None
            ):
                raise _error(event_index, "draw/discard before reach acceptance")

        draw_actor = _draw_actor(tag)
        if draw_actor is not None:
            tile_id = _tile_id_from_tag(tag, event_index, allow_missing=False)
            assert tile_id is not None
            draw_count += 1
            last_drawn[draw_actor] = tile_id
            if track_hand[draw_actor]:
                if tile_id in hands[draw_actor]:
                    raise _error(event_index, "draw duplicates a concealed tile")
                hands[draw_actor].append(tile_id)
            continue

        discard_actor = _discard_actor(tag)
        if discard_actor is not None:
            tile_id = _tile_id_from_tag(tag, event_index, allow_missing=True)
            if tile_id is None:
                tile_id = last_drawn[discard_actor]
                if tile_id is None:
                    raise _error(event_index, "implicit tsumogiri has no draw")
            if track_hand[discard_actor]:
                _remove_exact_tiles(hands[discard_actor], (tile_id,), event_index)
            discards[discard_actor] += 1
            last_drawn[discard_actor] = None
            visible_ids.add(tile_id)
            river_kinds[discard_actor].add(tile_id // 4)

            kokushi = None
            if track_hand[discard_actor] and len(hands[discard_actor]) == 13:
                kokushi = classify_kokushi_wait(hands[discard_actor])
            if kokushi is not None and discard_actor not in decisions:
                assert round_meta is not None
                decisions[discard_actor] = _new_decision(
                    year=year,
                    date=date,
                    log_id=log_id,
                    kyoku_index=kyoku_index,
                    viewer_tj=viewer_tj,
                    actor=discard_actor,
                    turn=discards[discard_actor],
                    round_meta=round_meta,
                    current_scores=current_scores,
                    kokushi=kokushi,
                    hand=hands[discard_actor],
                    visible_ids=visible_ids,
                    river_kinds=river_kinds[discard_actor],
                    accepted_actors=accepted_actors,
                    any_call=any_call,
                    open_hands=open_hands,
                    called_hands=called_hands,
                    draw_count=draw_count,
                )
            if pending is not None and pending.turn is None:
                pending.turn = discards[discard_actor]
                pending.kokushi = kokushi
                if kokushi is not None:
                    pending.decision = decisions[discard_actor]
                    _mark_riichi(pending.decision, pending, kokushi)
            continue

        if tag == "N":
            actor = _parse_actor_attribute(event, event_index)
            try:
                meld_code = int(event.attrib["m"])
            except (KeyError, ValueError) as error:
                raise _error(event_index, "N has an invalid m attribute") from error
            any_call = True
            called_hands[actor] = True
            full_meld = _decode_full_meld_tiles(meld_code)
            visible_ids.update(full_meld)
            if track_hand[actor]:
                consumed = _decode_concealed_meld_tiles(meld_code)
                _remove_exact_tiles(hands[actor], consumed, event_index)
                track_hand[actor] = False
            if meld_code & 0x3F:
                open_hands[actor] = True
            last_drawn[actor] = None
            continue

        if tag == "REACH":
            actor = _parse_actor_attribute(event, event_index)
            step = event.attrib.get("step")
            if step == "1":
                if pending is not None:
                    raise _error(event_index, "overlapping reach declarations")
                if actor in accepted_actors:
                    raise _error(event_index, "duplicate riichi by the same actor")
                if open_hands[actor]:
                    raise _error(event_index, "riichi actor has open melds")
                reach_type: ReachType = (
                    "double_riichi"
                    if discards[actor] == 0 and not any_call
                    else "riichi"
                )
                if last_drawn[actor] is not None:
                    pending = _PendingReach(actor=actor, reach_type=reach_type)
                else:
                    if (
                        event_index == 0
                        or _discard_actor(root[event_index - 1].tag) != actor
                    ):
                        raise _error(
                            event_index, "post-discard reach lacks actor discard"
                        )
                    kokushi = (
                        classify_kokushi_wait(hands[actor])
                        if track_hand[actor] and len(hands[actor]) == 13
                        else None
                    )
                    pending = _PendingReach(
                        actor=actor,
                        reach_type=(
                            "double_riichi"
                            if discards[actor] == 1 and not any_call
                            else "riichi"
                        ),
                        turn=discards[actor],
                        kokushi=kokushi,
                    )
                    if kokushi is not None and actor in decisions:
                        pending.decision = decisions[actor]
                        _mark_riichi(pending.decision, pending, kokushi)
            elif step == "2":
                if pending is None or pending.turn is None or pending.actor != actor:
                    raise _error(
                        event_index, "reach acceptance has no matching declaration"
                    )
                accepted_actors.add(actor)
                if pending.decision is not None:
                    pending.decision.riichi_established = True
                current_scores = _scores_after_reach(
                    event, current_scores, actor, event_index
                )
                pending = None
            else:
                raise _error(event_index, "REACH has an invalid step")
            continue

        if tag in {"AGARI", "RYUUKYOKU"}:
            pending = None
            terminal_events.append(event)
            round_ended = True
            continue
        raise _error(event_index, f"unsupported event {tag!r}")

    if kyoku_index < 0:
        raise KokushiRiichiParseError("log contains no INIT events")
    if not round_ended:
        raise KokushiRiichiParseError("log ended before kyoku result")
    finish_round()
    return _LogResult(kyoku_index + 1, int(normalized), tuple(all_records))


def scan_kokushi_tenpai_decisions(
    db_root: str | Path,
    *,
    years: Sequence[int] = SUPPORTED_YEARS,
    workers: int = 1,
    max_logs_per_year: int | None = None,
    on_year_complete: Callable[[KokushiTenpaiYearSummary], None] | None = None,
) -> KokushiTenpaiDecisionResult:
    """Scan annual four-player Houou hanchan databases."""
    selected = tuple(sorted(set(years)))
    if not selected or len(selected) != len(tuple(years)):
        raise ValueError("years must be non-empty and contain no duplicates")
    if any(type(year) is not int or year not in SUPPORTED_YEARS for year in selected):
        raise ValueError("years must be integers from 2009 through 2025")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if max_logs_per_year is not None and (
        type(max_logs_per_year) is not int or max_logs_per_year < 1
    ):
        raise ValueError("max_logs_per_year must be positive or None")
    root = Path(db_root).resolve()
    tasks = []
    for year in selected:
        database = root / f"{year}.db"
        if not database.is_file():
            raise FileNotFoundError(f"annual database does not exist: {database}")
        tasks.append((str(database), year, max_logs_per_year))
    outcomes = []
    if workers == 1:
        for task in tasks:
            outcome = _scan_year_safe(task)
            outcomes.append(outcome)
            if on_year_complete is not None and outcome[0] is not None:
                on_year_complete(outcome[0])
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            futures = [executor.submit(_scan_year_safe, task) for task in tasks]
            for future in as_completed(futures):
                outcome = future.result()
                outcomes.append(outcome)
                if on_year_complete is not None and outcome[0] is not None:
                    on_year_complete(outcome[0])
    errors = [error for _, _, error in outcomes if error]
    if errors:
        raise ValueError("annual scan errors:\n" + "\n".join(errors))
    annual = sorted(
        ((summary, records) for summary, records, _ in outcomes if summary is not None),
        key=lambda item: item[0].year,
    )
    records = tuple(
        sorted(
            (record for _, rows in annual for record in rows),
            key=lambda row: (row.year, row.date, row.log_id, row.kyoku_index, row.who),
        )
    )
    return KokushiTenpaiDecisionResult(
        years=selected,
        max_logs_per_year=max_logs_per_year,
        yearly=tuple(summary for summary, _ in annual),
        records=records,
    )


def result_to_dict(result: KokushiTenpaiDecisionResult) -> dict[str, Any]:
    """Return a stable JSON document with unadjusted descriptive summaries."""
    strategy_counts = Counter(row.strategy for row in result.records)
    strategy_wins = Counter(row.strategy for row in result.records if row.won)
    wait_counts = Counter(row.wait_group for row in result.records)
    return {
        "scope": {
            "archive_format": "tenhou_xml_sqlite_v1.2.0",
            "decision_unit": "first_post_discard_kokushi_tenpai_per_actor_kyoku",
            "game_type": "houou",
            "is_tonpu": False,
            "max_logs_per_year": result.max_logs_per_year,
            "num_players": 4,
            "years": list(result.years),
        },
        "summary": {
            "decisions": len(result.records),
            "riichi_eligible_decisions": sum(
                row.riichi_eligible is True for row in result.records
            ),
            "strategies": {
                strategy: {
                    "count": strategy_counts[strategy],
                    "wins": strategy_wins[strategy],
                    "win_rate": (
                        strategy_wins[strategy] / strategy_counts[strategy]
                        if strategy_counts[strategy]
                        else None
                    ),
                }
                for strategy in ("immediate_riichi", "delayed_riichi", "dama")
            },
            "declared_riichis": sum(row.riichi_declared for row in result.records),
            "established_riichis": sum(
                row.riichi_established for row in result.records
            ),
            "wait_groups": dict(sorted(wait_counts.items())),
            "wins": sum(row.won for row in result.records),
        },
        "yearly": [asdict(item) for item in result.yearly],
        "records": [_record_to_dict(row) for row in result.records],
    }


def write_result_json(result: KokushiTenpaiDecisionResult, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.part")
    text = json.dumps(
        result_to_dict(result),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    temporary.write_text(f"{text}\n", encoding="utf-8", newline="\n")
    temporary.replace(output)


def _new_decision(
    *,
    year: int,
    date: str,
    log_id: str,
    kyoku_index: int,
    viewer_tj: int,
    actor: int,
    turn: int,
    round_meta: tuple[str, int, int, int],
    current_scores: list[int] | None,
    kokushi: tuple[KokushiWaitType, tuple[str, ...]],
    hand: Sequence[int],
    visible_ids: set[int],
    river_kinds: set[int],
    accepted_actors: set[int],
    any_call: bool,
    open_hands: Sequence[bool],
    called_hands: Sequence[bool],
    draw_count: int,
) -> _Decision:
    bakaze, kyoku_number, honba, dealer = round_meta
    wait_type, waits = kokushi
    wait_kinds = tuple(tile_to_index(tile) for tile in waits)
    own_counts = Counter(tile // 4 for tile in hand)
    visible_counts = Counter(tile // 4 for tile in visible_ids)
    visible_wait_counts = tuple(visible_counts[kind] for kind in wait_kinds)
    unseen_wait_counts = tuple(
        4 - own_counts[kind] - visible_counts[kind] for kind in wait_kinds
    )
    scores = tuple(current_scores) if current_scores is not None else None
    actor_score = scores[actor] if scores is not None else None
    remaining = max(0, 70 - draw_count)
    eligible = None if actor_score is None else actor_score >= 1000 and remaining >= 4
    wait_group: WaitGroup
    if wait_type == "kokushi_13men":
        wait_group = "thirteen_sided"
    elif wait_kinds[0] >= 27:
        wait_group = "honor_single"
    else:
        wait_group = "terminal_single"
    return _Decision(
        year=year,
        date=date,
        log_id=log_id,
        kyoku_index=kyoku_index,
        who=actor,
        decision_tj=viewer_tj,
        turn=turn,
        bakaze=bakaze,
        kyoku_number=kyoku_number,
        honba=honba,
        dealer=actor == dealer,
        scores=scores,
        actor_score=actor_score,
        actor_rank=_rank(scores, actor),
        wait_type=wait_type,
        wait_group=wait_group,
        waits=waits,
        visible_wait_counts=visible_wait_counts,
        unseen_wait_counts=unseen_wait_counts,
        furiten=bool(set(wait_kinds) & river_kinds),
        prior_riichis=len(accepted_actors),
        any_call=any_call,
        opponent_open_hands=sum(open_hands[seat] for seat in range(4) if seat != actor),
        opponent_called_hands=sum(
            called_hands[seat] for seat in range(4) if seat != actor
        ),
        remaining_draws_estimate=remaining,
        riichi_eligible=eligible,
    )


def _mark_riichi(
    decision: _Decision,
    pending: _PendingReach,
    kokushi: tuple[KokushiWaitType, tuple[str, ...]],
) -> None:
    assert pending.turn is not None
    decision.strategy = (
        "immediate_riichi" if pending.turn == decision.turn else "delayed_riichi"
    )
    decision.riichi_declared = True
    decision.riichi_turn = pending.turn
    decision.riichi_reach_type = pending.reach_type
    decision.riichi_wait_type, decision.riichi_waits = kokushi


def _finalize_decision(
    decision: _Decision, terminal_events: Sequence[ET.Element]
) -> KokushiTenpaiDecisionRecord:
    if not terminal_events:
        raise KokushiRiichiParseError("decision kyoku has no terminal event")
    agari = [event for event in terminal_events if event.tag == "AGARI"]
    ryukyoku = [event for event in terminal_events if event.tag == "RYUUKYOKU"]
    if agari and ryukyoku:
        raise KokushiRiichiParseError("kyoku mixes AGARI and RYUUKYOKU")
    winners = tuple(int(event.attrib["who"]) for event in agari)
    from_whos = tuple(int(event.attrib["fromWho"]) for event in agari)
    win_method: WinMethod | None = None
    from_who: int | None = None
    draw_type: str | None = None
    if decision.who in winners:
        index = winners.index(decision.who)
        from_who = from_whos[index]
        win_method = "tsumo" if from_who == decision.who else "ron"
        outcome: Outcome = "win"
    elif agari and decision.who in from_whos:
        outcome = "deal_in"
    elif agari and any(
        winner == target for winner, target in zip(winners, from_whos, strict=True)
    ):
        outcome = "opponent_tsumo"
    elif agari:
        outcome = "other_ron"
    else:
        outcome = "draw"
        draw_type = ryukyoku[0].attrib.get("type") if ryukyoku else None
    final_scores = _terminal_scores(terminal_events)
    point_delta = (
        final_scores[decision.who] - decision.actor_score
        if final_scores is not None and decision.actor_score is not None
        else None
    )
    delay = (
        decision.riichi_turn - decision.turn
        if decision.riichi_turn is not None
        else None
    )
    url = f"{tenhou_replay_url(decision.log_id, decision.who, decision.kyoku_index)}&tj={decision.decision_tj}"
    return KokushiTenpaiDecisionRecord(
        **asdict(decision),
        riichi_delay_turns=delay,
        outcome=outcome,
        won=outcome == "win",
        win_method=win_method,
        winners=winners,
        from_who=from_who,
        draw_type=draw_type,
        point_delta=point_delta,
        url=url,
    )


def _scan_year(
    task: tuple[str, int, int | None],
) -> tuple[KokushiTenpaiYearSummary, tuple[KokushiTenpaiDecisionRecord, ...]]:
    database_text, year, max_logs = task
    database = Path(database_text)
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(logs)")}
        missing = sorted(_REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(
                f"{database.name}: logs table is missing columns {missing}"
            )
        scope = "num_players = 4 AND is_tonpu = 0"
        parameters: tuple[object, ...] = ()
        if "game_type" in columns:
            scope += " AND game_type = ?"
            parameters = ("houou",)
        eligible, available = connection.execute(
            f"SELECT COUNT(*), COUNT(log) FROM logs WHERE {scope}", parameters
        ).fetchone()
        sql = f"SELECT id, date, log FROM logs WHERE {scope} AND log IS NOT NULL ORDER BY id"
        query_parameters = parameters
        if max_logs is not None:
            sql += " LIMIT ?"
            query_parameters = (*parameters, max_logs)
        scanned = total_kyokus = normalized = 0
        records: list[KokushiTenpaiDecisionRecord] = []
        for log_id, date, payload in connection.execute(sql, query_parameters):
            scanned += 1
            try:
                result = analyze_tenhou_log(
                    bytes(payload), year=year, date=date, log_id=log_id
                )
            except (KokushiRiichiParseError, TypeError, ValueError) as error:
                raise ValueError(
                    f"{database.name}, log_id={log_id}: {error}"
                ) from error
            total_kyokus += result.total_kyokus
            normalized += result.normalized_xml_logs
            records.extend(result.records)
    finally:
        connection.close()
    counts = Counter(row.strategy for row in records)
    summary = KokushiTenpaiYearSummary(
        year=year,
        eligible_logs=eligible,
        available_logs=available,
        scanned_logs=scanned,
        unavailable_logs=eligible - available,
        normalized_xml_logs=normalized,
        total_kyokus=total_kyokus,
        decisions=len(records),
        riichi_eligible_decisions=sum(row.riichi_eligible is True for row in records),
        immediate_riichi=counts["immediate_riichi"],
        delayed_riichi=counts["delayed_riichi"],
        dama=counts["dama"],
        declared_riichis=sum(row.riichi_declared for row in records),
        established_riichis=sum(row.riichi_established for row in records),
    )
    return summary, tuple(records)


def _scan_year_safe(task: tuple[str, int, int | None]):
    try:
        summary, records = _scan_year(task)
    except (OSError, ValueError, sqlite3.Error) as error:
        return None, (), str(error)
    return summary, records, None


def _parse_root(payload: bytes) -> tuple[ET.Element, bool]:
    if not isinstance(payload, bytes) or not payload:
        raise TypeError("payload must be non-empty bytes")
    try:
        xml = gzip.decompress(payload) if payload.startswith(b"\x1f\x8b") else payload
    except (gzip.BadGzipFile, EOFError, OSError, zlib.error) as error:
        raise KokushiRiichiParseError(f"invalid gzip payload: {error}") from error
    normalized = False
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as error:
        if "duplicate attribute" not in str(error):
            raise KokushiRiichiParseError(f"invalid XML: {error}") from error
        xml, removed = _remove_identical_terminal_attributes(xml)
        if not removed:
            raise KokushiRiichiParseError(f"invalid XML: {error}") from error
        root = ET.fromstring(xml)
        normalized = True
    if root.tag != "mjloggm":
        raise KokushiRiichiParseError("root element must be mjloggm")
    return root, normalized


def _parse_scores(event: ET.Element, event_index: int) -> list[int] | None:
    raw = event.attrib.get("ten")
    if raw is None:
        return None
    try:
        values = [int(value) * 100 for value in raw.split(",")]
    except ValueError as error:
        raise _error(event_index, "INIT ten is invalid") from error
    if len(values) != 4:
        raise _error(event_index, "INIT ten must contain four scores")
    return values


def _parse_round_meta(event: ET.Element, event_index: int) -> tuple[str, int, int, int]:
    try:
        seed = [int(value) for value in event.attrib["seed"].split(",")]
        dealer = int(event.attrib["oya"])
    except (KeyError, ValueError) as error:
        raise _error(event_index, "INIT round metadata is invalid") from error
    if len(seed) < 6 or dealer not in range(4):
        raise _error(event_index, "INIT round metadata is invalid")
    round_index = seed[0]
    wind_index = round_index // 4
    bakaze = _WIND_NAMES[wind_index] if wind_index < 4 else f"round_{wind_index}"
    return bakaze, round_index % 4 + 1, seed[1], dealer


def _parse_initial_dora(event: ET.Element, event_index: int) -> int | None:
    try:
        seed = [int(value) for value in event.attrib["seed"].split(",")]
    except (KeyError, ValueError) as error:
        raise _error(event_index, "INIT seed is invalid") from error
    if len(seed) < 6:
        raise _error(event_index, "INIT seed must contain a dora indicator")
    dora = seed[5]
    if not 0 <= dora < 136:
        raise _error(event_index, "INIT dora indicator is invalid")
    return dora


def _required_tile_attribute(event: ET.Element, name: str, event_index: int) -> int:
    try:
        tile = int(event.attrib[name])
    except (KeyError, ValueError) as error:
        raise _error(event_index, f"{event.tag} {name} is invalid") from error
    if not 0 <= tile < 136:
        raise _error(event_index, f"{event.tag} {name} is invalid")
    return tile


def _scores_after_reach(
    event: ET.Element,
    scores: list[int] | None,
    actor: int,
    event_index: int,
) -> list[int] | None:
    raw = event.attrib.get("ten")
    if raw is not None:
        try:
            values = [int(value) * 100 for value in raw.split(",")]
        except ValueError as error:
            raise _error(event_index, "REACH ten is invalid") from error
        if len(values) != 4:
            raise _error(event_index, "REACH ten must contain four scores")
        return values
    if scores is not None:
        values = list(scores)
        values[actor] -= 1000
        return values
    return None


def _terminal_scores(events: Sequence[ET.Element]) -> tuple[int, int, int, int] | None:
    for event in reversed(events):
        raw = event.attrib.get("sc")
        if raw is None:
            continue
        try:
            values = [int(value) for value in raw.split(",")]
        except ValueError as error:
            raise KokushiRiichiParseError("terminal sc is invalid") from error
        if len(values) != 8:
            raise KokushiRiichiParseError("terminal sc must contain eight values")
        return tuple(
            (values[index] + values[index + 1]) * 100 for index in range(0, 8, 2)
        )  # type: ignore[return-value]
    return None


def _decode_full_meld_tiles(meld_code: int) -> tuple[int, ...]:
    if meld_code & 0x4:
        encoded = meld_code >> 10
        base = encoded // 3
        base = ((base // 7) * 9 + base % 7) * 4
        return (
            base + ((meld_code >> 3) & 0x3),
            base + 4 + ((meld_code >> 5) & 0x3),
            base + 8 + ((meld_code >> 7) & 0x3),
        )
    if meld_code & 0x8:
        unused = (meld_code >> 5) & 0x3
        base = ((meld_code >> 9) // 3) * 4
        return tuple(base + copy for copy in range(4) if copy != unused)
    if meld_code & 0x10:
        base = ((meld_code >> 9) // 3) * 4
        return tuple(base + copy for copy in range(4))
    if meld_code & 0x20:
        raise KokushiRiichiParseError(
            "north extraction is invalid in a four-player log"
        )
    called = meld_code >> 8
    base = (called // 4) * 4
    return tuple(base + copy for copy in range(4))


def _rank(scores: tuple[int, int, int, int] | None, actor: int) -> int | None:
    if scores is None:
        return None
    return sorted(range(4), key=lambda seat: (-scores[seat], seat)).index(actor) + 1


def _record_to_dict(record: KokushiTenpaiDecisionRecord) -> dict[str, Any]:
    value = asdict(record)
    for field in (
        "scores",
        "waits",
        "visible_wait_counts",
        "unseen_wait_counts",
        "riichi_waits",
        "winners",
    ):
        if value[field] is not None:
            value[field] = list(value[field])
    return value


def _error(event_index: int, message: str) -> KokushiRiichiParseError:
    return KokushiRiichiParseError(f"event {event_index}: {message}")
