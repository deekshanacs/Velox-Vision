import numpy as np
from dataclasses import dataclass
from typing import Optional
from src.core.entities import BoundingBox
from src.engines.tracking.entities.track_state import TrackState
from src.engines.tracking.value_objects.point import Point

@dataclass(frozen=True)
class MemorySnapshot:
    """Immutable state snapshot of a tracked vehicle at a specific frame index."""
    frame_number: int
    timestamp: float
    bbox: BoundingBox
    center: Point
    area: float
    confidence: float
    state: TrackState
    # Raw thumbnail frame snippet is optional to prevent excessive system memory usage
    frame_thumbnail: Optional[np.ndarray] = None
