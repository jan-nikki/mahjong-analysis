"""Reference waits found by reserving incomplete 13-tile fragments first."""

from collections.abc import Iterable, Iterator

from .model import (
    REFERENCE_TILE_KINDS,
    ReferenceHandWaits,
    ReferenceMeld,
    ReferenceWaitDetail,
    ReferenceWaitShape,
    reference_tile_to_index,
    reference_wait_detail_sort_key,
)

_EMPTY_SUIT_VECTOR = (0, 0, 0, 0, 0, 0, 0, 0, 0)
_SUITED_MELD_VECTORS = tuple(
    tuple(3 if position == rank else 0 for position in range(9)) for rank in range(9)
) + tuple(
    tuple(1 if start <= position < start + 3 else 0 for position in range(9))
    for start in range(7)
)
_ORPHAN_INDICES = (
    0,
    8,
    9,
    17,
    18,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
)


def _generate_complete_suit_vectors() -> tuple[frozenset[tuple[int, ...]], ...]:
    by_meld_count: list[set[tuple[int, ...]]] = [{_EMPTY_SUIT_VECTOR}]
    for _ in range(4):
        next_vectors: set[tuple[int, ...]] = set()
        for existing in by_meld_count[-1]:
            for meld in _SUITED_MELD_VECTORS:
                combined = tuple(
                    existing[position] + meld[position] for position in range(9)
                )
                if max(combined) <= 4:
                    next_vectors.add(combined)
        by_meld_count.append(next_vectors)
    return tuple(frozenset(vectors) for vectors in by_meld_count)


_COMPLETE_SUIT_VECTORS = _generate_complete_suit_vectors()


def calculate_reference_hand_waits(
    concealed_tiles: Iterable[str],
    fixed_melds: Iterable[ReferenceMeld] = (),
) -> ReferenceHandWaits:
    """Calculate waits without adding a candidate to a completed 14-tile hand."""
    concealed = tuple(concealed_tiles)
    fixed = tuple(fixed_melds)
    _validate_reference_input(concealed, fixed)

    concealed_counts = _tile_counts(concealed)
    required_melds = 4 - len(fixed)
    details = _standard_wait_details(
        concealed_counts,
        required_melds,
    )
    if not fixed:
        details.update(_chiitoitsu_wait_details(concealed_counts))
        details.update(_kokushi_wait_details(concealed_counts))

    ordered_details = tuple(sorted(details, key=reference_wait_detail_sort_key))
    detail_tile_indices = {
        reference_tile_to_index(detail.wait_tile) for detail in ordered_details
    }
    wait_tiles = tuple(
        tile
        for index, tile in enumerate(REFERENCE_TILE_KINDS)
        if index in detail_tile_indices
    )
    return ReferenceHandWaits(wait_tiles, ordered_details)


def _validate_reference_input(
    concealed_tiles: tuple[str, ...],
    fixed_melds: tuple[ReferenceMeld, ...],
) -> None:
    if not all(isinstance(meld, ReferenceMeld) for meld in fixed_melds):
        raise TypeError("fixed_melds must contain ReferenceMeld instances")
    if len(fixed_melds) > 4:
        raise ValueError("at most four reference fixed melds are allowed")

    expected_concealed_count = 13 - 3 * len(fixed_melds)
    if len(concealed_tiles) != expected_concealed_count:
        raise ValueError(
            "reference concealed physical tile count must equal "
            f"{expected_concealed_count} with {len(fixed_melds)} fixed melds"
        )

    owned_counts = list(_tile_counts(concealed_tiles))
    for meld in fixed_melds:
        for tile in meld.tiles:
            owned_counts[reference_tile_to_index(tile)] += 1
    if any(count > 4 for count in owned_counts):
        raise ValueError(
            "reference concealed and fixed melds cannot own five of one tile kind"
        )


def _tile_counts(tiles: tuple[str, ...]) -> tuple[int, ...]:
    counts = [0] * 34
    for tile in tiles:
        counts[reference_tile_to_index(tile)] += 1
    return tuple(counts)


def _standard_wait_details(
    concealed_counts: tuple[int, ...],
    required_melds: int,
) -> set[ReferenceWaitDetail]:
    details: set[ReferenceWaitDetail] = set()

    for singleton_index, count in enumerate(concealed_counts):
        if count == 0:
            continue
        residue = list(concealed_counts)
        residue[singleton_index] -= 1
        if _is_complete_meld_residue(tuple(residue), required_melds):
            _add_standard_detail(
                details,
                singleton_index,
                "tanki",
                concealed_counts,
            )

    if required_melds == 0:
        return details

    for pair_index, count in enumerate(concealed_counts):
        if count < 2:
            continue
        after_pair = list(concealed_counts)
        after_pair[pair_index] -= 2

        for first, second in _two_tile_reservations(tuple(after_pair)):
            waits = _waits_from_reserved_fragment(first, second)
            if not waits:
                continue
            residue = list(after_pair)
            residue[first] -= 1
            residue[second] -= 1
            if not _is_complete_meld_residue(tuple(residue), required_melds - 1):
                continue
            for wait_index, wait_shape in waits:
                _add_standard_detail(
                    details,
                    wait_index,
                    wait_shape,
                    concealed_counts,
                )

    return details


def _two_tile_reservations(
    counts: tuple[int, ...],
) -> Iterator[tuple[int, int]]:
    present = tuple(index for index, count in enumerate(counts) if count)
    for offset, first in enumerate(present):
        if counts[first] >= 2:
            yield first, first
        for second in present[offset + 1 :]:
            yield first, second


def _waits_from_reserved_fragment(
    first: int,
    second: int,
) -> tuple[tuple[int, ReferenceWaitShape], ...]:
    if first == second:
        return ((first, "shanpon"),)
    if first >= 27 or second >= 27 or first // 9 != second // 9:
        return ()

    distance = second - first
    first_rank = first % 9 + 1
    if distance == 1:
        if first_rank == 1:
            return ((second + 1, "penchan"),)
        if first_rank == 8:
            return ((first - 1, "penchan"),)
        return (
            (first - 1, "ryanmen"),
            (second + 1, "ryanmen"),
        )
    if distance == 2:
        return ((first + 1, "kanchan"),)
    return ()


def _is_complete_meld_residue(
    counts: tuple[int, ...],
    required_melds: int,
) -> bool:
    if required_melds not in range(5):
        return False
    if sum(counts) != required_melds * 3:
        return False

    honor_counts = counts[27:]
    if any(count not in (0, 3) for count in honor_counts):
        return False
    completed_melds = sum(count // 3 for count in honor_counts)

    for suit_start in (0, 9, 18):
        suit_vector = counts[suit_start : suit_start + 9]
        suit_tile_count = sum(suit_vector)
        if suit_tile_count % 3:
            return False
        suit_meld_count = suit_tile_count // 3
        if suit_meld_count > 4:
            return False
        if suit_vector not in _COMPLETE_SUIT_VECTORS[suit_meld_count]:
            return False
        completed_melds += suit_meld_count

    return completed_melds == required_melds


def _add_standard_detail(
    details: set[ReferenceWaitDetail],
    wait_index: int,
    wait_shape: ReferenceWaitShape,
    concealed_counts: tuple[int, ...],
) -> None:
    # Tenhou permits fifth-copy tenpai through fixed melds, even closed kans.
    # The restriction concerns the original pure hand, before reservations.
    if concealed_counts[wait_index] >= 4:
        return
    details.add(
        ReferenceWaitDetail(
            REFERENCE_TILE_KINDS[wait_index],
            "standard",
            wait_shape,
        )
    )


def _chiitoitsu_wait_details(
    counts: tuple[int, ...],
) -> set[ReferenceWaitDetail]:
    pair_indices = [index for index, count in enumerate(counts) if count == 2]
    singleton_indices = [index for index, count in enumerate(counts) if count == 1]
    if (
        len(pair_indices) != 6
        or len(singleton_indices) != 1
        or any(count not in (0, 1, 2) for count in counts)
    ):
        return set()
    return {
        ReferenceWaitDetail(
            REFERENCE_TILE_KINDS[singleton_indices[0]],
            "chiitoitsu",
            "tanki",
        )
    }


def _kokushi_wait_details(
    counts: tuple[int, ...],
) -> set[ReferenceWaitDetail]:
    orphan_set = frozenset(_ORPHAN_INDICES)
    if any(counts[index] for index in range(34) if index not in orphan_set):
        return set()

    orphan_counts = tuple(counts[index] for index in _ORPHAN_INDICES)
    if all(count == 1 for count in orphan_counts):
        return {
            ReferenceWaitDetail(
                REFERENCE_TILE_KINDS[index],
                "kokushi",
                "kokushi_13men",
            )
            for index in _ORPHAN_INDICES
        }

    missing = [index for index in _ORPHAN_INDICES if counts[index] == 0]
    pairs = [index for index in _ORPHAN_INDICES if counts[index] == 2]
    if (
        len(missing) == 1
        and len(pairs) == 1
        and all(count in (0, 1, 2) for count in orphan_counts)
    ):
        return {
            ReferenceWaitDetail(
                REFERENCE_TILE_KINDS[missing[0]],
                "kokushi",
                "kokushi_single",
            )
        }
    return set()
