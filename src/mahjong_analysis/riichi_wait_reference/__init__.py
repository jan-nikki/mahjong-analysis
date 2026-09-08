"""Independent reference implementation for structural hand waits."""

from .model import (
    REFERENCE_TILE_KINDS,
    ReferenceHandType,
    ReferenceHandWaits,
    ReferenceMeld,
    ReferenceWaitDetail,
    ReferenceWaitShape,
    normalize_reference_tile,
    reference_tile_to_index,
)
from .waits import calculate_reference_hand_waits

__all__ = [
    "REFERENCE_TILE_KINDS",
    "ReferenceHandType",
    "ReferenceHandWaits",
    "ReferenceMeld",
    "ReferenceWaitDetail",
    "ReferenceWaitShape",
    "calculate_reference_hand_waits",
    "normalize_reference_tile",
    "reference_tile_to_index",
]
