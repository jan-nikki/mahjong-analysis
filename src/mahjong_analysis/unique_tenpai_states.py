"""Exact, bounded enumeration of unique tenpai count-vector states."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from math import comb

from mahjong_analysis.combo_theory import UnseenCountsInput, normalize_unseen_counts
from mahjong_analysis.hand_waits import (
    HAND_TYPE_ORDER,
    WAIT_SHAPE_ORDER,
    FixedMeld,
    HandType,
    WaitShape,
    calculate_hand_waits,
)
from mahjong_analysis.tiles import TILE_KINDS, index_to_tile, tile_to_index


@dataclass(frozen=True)
class InterpretationStateCount:
    """Hands having at least one canonical interpretation of this type."""

    hand_type: HandType
    wait_shape: WaitShape
    state_count: int
    physical_weight: int


@dataclass(frozen=True)
class UniqueWaitStateCount:
    """Unique and physical-allocation-weighted counts for one wait tile."""

    wait_tile: str
    state_count: int
    physical_weight: int
    interpretation_counts: tuple[InterpretationStateCount, ...]


@dataclass(frozen=True)
class UniqueTenpaiSummary:
    """Exact counts over every bounded concealed count vector considered."""

    unseen_counts: tuple[int, ...]
    concealed_tile_count: int
    candidate_vector_count: int
    tenpai_state_count: int
    tenpai_physical_weight: int
    wait_counts: tuple[UniqueWaitStateCount, ...]

    def for_tile(self, tile: str) -> UniqueWaitStateCount:
        index = tile_to_index(tile)
        return self.wait_counts[index]


@dataclass(frozen=True)
class CandidateWaitStateCount:
    """Exact unique-hand numerator for one externally supplied winning tile."""

    candidate_tile: str
    state_count: int
    physical_weight: int
    standard_state_count: int
    standard_physical_weight: int
    chiitoitsu_state_count: int
    chiitoitsu_physical_weight: int
    kokushi_state_count: int
    kokushi_physical_weight: int
    multi_hand_type_state_count: int
    multi_hand_type_physical_weight: int


@dataclass(frozen=True)
class _BlockSignature:
    standard_mask: int
    chiitoitsu_pairs: int
    kokushi_present: int
    kokushi_pairs: int


_SUITED_MELDS = tuple(
    tuple(3 if index == rank else 0 for index in range(9)) for rank in range(9)
) + tuple(
    tuple(1 if start <= index < start + 3 else 0 for index in range(9))
    for start in range(7)
)
_HONOR_MELDS = tuple(
    tuple(3 if index == rank else 0 for index in range(7)) for rank in range(7)
)


def _standard_bit(meld_count: int, pair_count: int) -> int:
    return 1 << (meld_count * 2 + pair_count)


def _standard_block_vectors(
    size: int,
    melds: tuple[tuple[int, ...], ...],
) -> dict[tuple[int, ...], int]:
    by_meld_count: list[set[tuple[int, ...]]] = [{(0,) * size}]
    for _ in range(4):
        next_vectors: set[tuple[int, ...]] = set()
        for current in by_meld_count[-1]:
            for meld in melds:
                combined = tuple(current[index] + meld[index] for index in range(size))
                if max(combined) <= 4:
                    next_vectors.add(combined)
        by_meld_count.append(next_vectors)

    result: dict[tuple[int, ...], int] = {}
    for meld_count, vectors in enumerate(by_meld_count):
        for vector in vectors:
            result[vector] = result.get(vector, 0) | _standard_bit(meld_count, 0)
            for pair_index in range(size):
                paired = list(vector)
                paired[pair_index] += 2
                if paired[pair_index] <= 4:
                    paired_tuple = tuple(paired)
                    result[paired_tuple] = result.get(paired_tuple, 0) | _standard_bit(
                        meld_count, 1
                    )
    return result


_SUITED_STANDARD_VECTORS = _standard_block_vectors(9, _SUITED_MELDS)
_HONOR_STANDARD_VECTORS = _standard_block_vectors(7, _HONOR_MELDS)


def _relevant_vectors(
    standard_vectors: dict[tuple[int, ...], int],
    *,
    suited: bool,
) -> tuple[tuple[int, ...], ...]:
    vectors = set(standard_vectors)
    size = 9 if suited else 7

    def add_chiitoitsu(index: int, current: list[int]) -> None:
        if index == size:
            vectors.add(tuple(current))
            return
        for value in (0, 2):
            current[index] = value
            add_chiitoitsu(index + 1, current)

    add_chiitoitsu(0, [0] * size)
    if suited:
        for first in (1, 2):
            for last in (1, 2):
                vector = [0] * size
                vector[0] = first
                vector[8] = last
                vectors.add(tuple(vector))
    else:

        def add_kokushi(index: int, current: list[int]) -> None:
            if index == size:
                vectors.add(tuple(current))
                return
            for value in (1, 2):
                current[index] = value
                add_kokushi(index + 1, current)

        add_kokushi(0, [0] * size)
    return tuple(sorted(vectors))


_SUITED_RELEVANT_VECTORS = _relevant_vectors(
    _SUITED_STANDARD_VECTORS,
    suited=True,
)
_HONOR_RELEVANT_VECTORS = _relevant_vectors(
    _HONOR_STANDARD_VECTORS,
    suited=False,
)


def count_candidate_wait_states(
    unseen_counts: UnseenCountsInput,
    candidate_tile: str,
    *,
    fixed_meld_count: int = 0,
) -> CandidateWaitStateCount:
    """Count unique hands completed by one visible candidate tile.

    The candidate is external to the unseen pool. Standard, chiitoitsu, and
    kokushi complete count vectors are unioned before counting, so hands with
    multiple hand-type interpretations are not duplicated.
    """
    counts = normalize_unseen_counts(unseen_counts)
    candidate_index = tile_to_index(candidate_tile)
    candidate = index_to_tile(candidate_index)
    if isinstance(fixed_meld_count, bool) or not isinstance(fixed_meld_count, int):
        raise TypeError("fixed_meld_count must be an integer")
    if not 0 <= fixed_meld_count <= 4:
        raise ValueError("fixed_meld_count must be between 0 and 4")

    block_specs = (
        (0, 9, True),
        (9, 18, True),
        (18, 27, True),
        (27, 34, False),
    )
    dynamic: dict[_BlockSignature, tuple[int, int]] = {
        _BlockSignature(_standard_bit(0, 0), 0, 0, 0): (1, 1)
    }
    for start, end, suited in block_specs:
        local_candidate = (
            candidate_index - start if start <= candidate_index < end else None
        )
        local_groups = _block_groups(
            counts[start:end],
            local_candidate,
            suited=suited,
        )
        next_dynamic: dict[_BlockSignature, list[int]] = {}
        for left_signature, (left_states, left_weight) in dynamic.items():
            for right_signature, (right_states, right_weight) in local_groups.items():
                signature = _combine_signatures(left_signature, right_signature)
                aggregate = next_dynamic.setdefault(signature, [0, 0])
                aggregate[0] += left_states * right_states
                aggregate[1] += left_weight * right_weight
        dynamic = {
            signature: (aggregate[0], aggregate[1])
            for signature, aggregate in next_dynamic.items()
        }

    melds_needed = 4 - fixed_meld_count
    union_states = union_weight = 0
    standard_states = standard_weight = 0
    chiitoitsu_states = chiitoitsu_weight = 0
    kokushi_states = kokushi_weight = 0
    multi_states = multi_weight = 0
    for signature, (states, weight) in dynamic.items():
        is_standard = bool(signature.standard_mask & _standard_bit(melds_needed, 1))
        is_chiitoitsu = fixed_meld_count == 0 and signature.chiitoitsu_pairs == 7
        is_kokushi = (
            fixed_meld_count == 0
            and signature.kokushi_present == 13
            and signature.kokushi_pairs == 1
        )
        matched_types = sum((is_standard, is_chiitoitsu, is_kokushi))
        if not matched_types:
            continue
        union_states += states
        union_weight += weight
        if is_standard:
            standard_states += states
            standard_weight += weight
        if is_chiitoitsu:
            chiitoitsu_states += states
            chiitoitsu_weight += weight
        if is_kokushi:
            kokushi_states += states
            kokushi_weight += weight
        if matched_types > 1:
            multi_states += states
            multi_weight += weight

    return CandidateWaitStateCount(
        candidate_tile=candidate,
        state_count=union_states,
        physical_weight=union_weight,
        standard_state_count=standard_states,
        standard_physical_weight=standard_weight,
        chiitoitsu_state_count=chiitoitsu_states,
        chiitoitsu_physical_weight=chiitoitsu_weight,
        kokushi_state_count=kokushi_states,
        kokushi_physical_weight=kokushi_weight,
        multi_hand_type_state_count=multi_states,
        multi_hand_type_physical_weight=multi_weight,
    )


def _block_groups(
    unseen_counts: tuple[int, ...],
    candidate_index: int | None,
    *,
    suited: bool,
) -> dict[_BlockSignature, tuple[int, int]]:
    standard_vectors = _SUITED_STANDARD_VECTORS if suited else _HONOR_STANDARD_VECTORS
    relevant_vectors = _SUITED_RELEVANT_VECTORS if suited else _HONOR_RELEVANT_VECTORS
    groups: dict[_BlockSignature, list[int]] = {}
    for completed in relevant_vectors:
        held = list(completed)
        if candidate_index is not None:
            held[candidate_index] -= 1
        if any(value < 0 for value in held):
            continue
        if any(
            value > available
            for value, available in zip(held, unseen_counts, strict=True)
        ):
            continue
        signature = _block_signature(
            completed,
            standard_vectors.get(completed, 0),
            suited=suited,
        )
        weight = 1
        for available, value in zip(unseen_counts, held, strict=True):
            weight *= comb(available, value)
        aggregate = groups.setdefault(signature, [0, 0])
        aggregate[0] += 1
        aggregate[1] += weight
    return {
        signature: (aggregate[0], aggregate[1])
        for signature, aggregate in groups.items()
    }


def _block_signature(
    vector: tuple[int, ...],
    standard_mask: int,
    *,
    suited: bool,
) -> _BlockSignature:
    chiitoitsu_pairs = (
        sum(value == 2 for value in vector)
        if all(value in (0, 2) for value in vector)
        else -1
    )
    if suited:
        kokushi_valid = (
            vector[0] in (1, 2)
            and vector[8] in (1, 2)
            and all(value == 0 for value in vector[1:8])
        )
    else:
        kokushi_valid = all(value in (1, 2) for value in vector)
    kokushi_present = sum(value > 0 for value in vector) if kokushi_valid else -1
    kokushi_pairs = sum(value == 2 for value in vector) if kokushi_valid else -1
    return _BlockSignature(
        standard_mask=standard_mask,
        chiitoitsu_pairs=chiitoitsu_pairs,
        kokushi_present=kokushi_present,
        kokushi_pairs=kokushi_pairs,
    )


def _combine_signatures(
    left: _BlockSignature,
    right: _BlockSignature,
) -> _BlockSignature:
    return _BlockSignature(
        standard_mask=_combine_standard_masks(
            left.standard_mask,
            right.standard_mask,
        ),
        chiitoitsu_pairs=(
            left.chiitoitsu_pairs + right.chiitoitsu_pairs
            if left.chiitoitsu_pairs >= 0 and right.chiitoitsu_pairs >= 0
            else -1
        ),
        kokushi_present=(
            left.kokushi_present + right.kokushi_present
            if left.kokushi_present >= 0 and right.kokushi_present >= 0
            else -1
        ),
        kokushi_pairs=(
            left.kokushi_pairs + right.kokushi_pairs
            if left.kokushi_pairs >= 0 and right.kokushi_pairs >= 0
            else -1
        ),
    )


def _combine_standard_masks(left: int, right: int) -> int:
    combined = 0
    for left_melds in range(5):
        for left_pairs in range(2):
            if not left & _standard_bit(left_melds, left_pairs):
                continue
            for right_melds in range(5):
                for right_pairs in range(2):
                    if not right & _standard_bit(right_melds, right_pairs):
                        continue
                    melds = left_melds + right_melds
                    pairs = left_pairs + right_pairs
                    if melds <= 4 and pairs <= 1:
                        combined |= _standard_bit(melds, pairs)
    return combined


def bounded_count_vector_count(bounds: tuple[int, ...], total: int) -> int:
    """Count bounded nonnegative integer vectors having the requested sum."""
    if isinstance(total, bool) or not isinstance(total, int):
        raise TypeError("total must be an integer")
    if total < 0:
        return 0
    ways = [0] * (total + 1)
    ways[0] = 1
    for bound in bounds:
        if isinstance(bound, bool) or not isinstance(bound, int):
            raise TypeError("bounds must contain integers")
        if bound < 0:
            raise ValueError("bounds must be non-negative")
        next_ways = [0] * (total + 1)
        for subtotal, ways_to_subtotal in enumerate(ways):
            if not ways_to_subtotal:
                continue
            for value in range(min(bound, total - subtotal) + 1):
                next_ways[subtotal + value] += ways_to_subtotal
        ways = next_ways
    return ways[total]


def count_unique_tenpai_states(
    unseen_counts: UnseenCountsInput,
    fixed_melds: tuple[FixedMeld, ...] = (),
    *,
    max_candidate_vectors: int | None = 1_000_000,
) -> UniqueTenpaiSummary:
    """Exactly count unique tenpai hands bounded by an unseen tile pool.

    This audit implementation intentionally refuses unexpectedly large state
    spaces by default. It is suitable for reduced universes and small exact
    checks, not yet the production algorithm for ordinary 34-kind positions.
    """
    counts = normalize_unseen_counts(unseen_counts)
    fixed = tuple(fixed_melds)
    if not all(isinstance(meld, FixedMeld) for meld in fixed):
        raise TypeError("fixed_melds must contain FixedMeld instances")
    if len(fixed) > 4:
        raise ValueError("a hand cannot contain more than four fixed melds")
    if max_candidate_vectors is not None and (
        isinstance(max_candidate_vectors, bool)
        or not isinstance(max_candidate_vectors, int)
        or max_candidate_vectors < 1
    ):
        raise ValueError("max_candidate_vectors must be positive or None")

    fixed_counts = [0] * len(TILE_KINDS)
    for meld in fixed:
        for tile in meld.tiles:
            fixed_counts[tile_to_index(tile)] += 1
    for index, (unseen, visible_fixed) in enumerate(
        zip(counts, fixed_counts, strict=True)
    ):
        if unseen + visible_fixed > 4:
            raise ValueError(
                "unseen pool and fixed meld exceed four copies of "
                f"{index_to_tile(index)}"
            )

    concealed_tile_count = 13 - 3 * len(fixed)
    candidate_vector_count = bounded_count_vector_count(counts, concealed_tile_count)
    if max_candidate_vectors is not None and (
        candidate_vector_count > max_candidate_vectors
    ):
        raise ValueError(
            "exact enumeration would inspect "
            f"{candidate_vector_count} count vectors, exceeding limit "
            f"{max_candidate_vectors}"
        )

    state_counts = [0] * len(TILE_KINDS)
    physical_weights = [0] * len(TILE_KINDS)
    interpretation_counts: list[dict[tuple[HandType, WaitShape], list[int]]] = [
        {} for _ in TILE_KINDS
    ]
    tenpai_state_count = 0
    tenpai_physical_weight = 0

    for hand_counts in _bounded_count_vectors(counts, concealed_tile_count):
        hand = tuple(
            index_to_tile(index)
            for index, count in enumerate(hand_counts)
            for _ in range(count)
        )
        waits = calculate_hand_waits(hand, fixed)
        if not waits.wait_tiles:
            continue

        weight = _physical_weight(counts, hand_counts)
        tenpai_state_count += 1
        tenpai_physical_weight += weight
        for wait_tile in waits.wait_tiles:
            wait_index = tile_to_index(wait_tile)
            state_counts[wait_index] += 1
            physical_weights[wait_index] += weight
        for detail in waits.wait_details:
            wait_index = tile_to_index(detail.wait_tile)
            key = (detail.hand_type, detail.wait_shape)
            aggregate = interpretation_counts[wait_index].setdefault(key, [0, 0])
            aggregate[0] += 1
            aggregate[1] += weight

    hand_type_order = {value: index for index, value in enumerate(HAND_TYPE_ORDER)}
    wait_shape_order = {value: index for index, value in enumerate(WAIT_SHAPE_ORDER)}
    wait_counts = tuple(
        UniqueWaitStateCount(
            wait_tile=index_to_tile(index),
            state_count=state_counts[index],
            physical_weight=physical_weights[index],
            interpretation_counts=tuple(
                InterpretationStateCount(
                    hand_type=hand_type,
                    wait_shape=wait_shape,
                    state_count=aggregate[0],
                    physical_weight=aggregate[1],
                )
                for (hand_type, wait_shape), aggregate in sorted(
                    interpretation_counts[index].items(),
                    key=lambda item: (
                        hand_type_order[item[0][0]],
                        wait_shape_order[item[0][1]],
                    ),
                )
            ),
        )
        for index in range(len(TILE_KINDS))
    )
    return UniqueTenpaiSummary(
        unseen_counts=counts,
        concealed_tile_count=concealed_tile_count,
        candidate_vector_count=candidate_vector_count,
        tenpai_state_count=tenpai_state_count,
        tenpai_physical_weight=tenpai_physical_weight,
        wait_counts=wait_counts,
    )


def _bounded_count_vectors(
    bounds: tuple[int, ...],
    total: int,
) -> Iterator[tuple[int, ...]]:
    values = [0] * len(bounds)
    suffix_capacity = [0] * (len(bounds) + 1)
    for index in range(len(bounds) - 1, -1, -1):
        suffix_capacity[index] = suffix_capacity[index + 1] + bounds[index]

    def visit(index: int, remaining: int) -> Iterator[tuple[int, ...]]:
        if index == len(bounds):
            if remaining == 0:
                yield tuple(values)
            return
        minimum = max(0, remaining - suffix_capacity[index + 1])
        maximum = min(bounds[index], remaining)
        for value in range(minimum, maximum + 1):
            values[index] = value
            yield from visit(index + 1, remaining - value)
        values[index] = 0

    yield from visit(0, total)


def _physical_weight(
    unseen_counts: tuple[int, ...],
    hand_counts: tuple[int, ...],
) -> int:
    weight = 1
    for unseen, held in zip(unseen_counts, hand_counts, strict=True):
        weight *= comb(unseen, held)
    return weight
