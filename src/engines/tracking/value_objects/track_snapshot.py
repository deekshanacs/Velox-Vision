from dataclasses import dataclass
from src.core.entities import BoundingBox
from src.engines.tracking.value_objects.point import Point
from src.engines.tracking.entities.track_state import TrackState

@dataclass(frozen=True)
class TrackSnapshot:
    """Immutable value object representing a single historical point in a vehicle's track.
    
    Contains coordinate centers, bounding boxes, confidences, and tracking state.
    """
    frame_number: int
    timestamp: float
    center: Point
    bbox: BoundingBox
    confidence: float
    state: TrackState
