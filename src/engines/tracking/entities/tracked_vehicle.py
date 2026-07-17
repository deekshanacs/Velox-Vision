from dataclasses import dataclass, field
from typing import Optional, Tuple
from src.core.entities import BoundingBox
from src.engines.tracking.entities.track_state import TrackState
from src.engines.tracking.entities.track_history import TrackHistory
from src.engines.tracking.value_objects.point import Point
from src.engines.tracking.value_objects.velocity import Velocity

@dataclass
class TrackedVehicle:
    """Represents a unique tracked vehicle entity across multiple frames.
    
    Contains current coordinates, motion metadata, historic snapshots,
    and placeholders for license plate and speed estimations.
    """
    track_id: int
    class_name: str
    class_id: int
    bbox: BoundingBox
    confidence: float
    state: TrackState
    first_seen_frame: int
    last_seen_frame: int
    track_age: int = 1
    is_occluded: bool = False
    
    # History and Motion Attributes
    history: TrackHistory = field(default_factory=TrackHistory)
    velocity: Optional[Velocity] = None
    
    # Domain Specific Attributes Placeholders
    estimated_speed_kmh: Optional[float] = None
    license_plate_text: Optional[str] = None
    
    # Custom Extensible Metadata Channel
    attributes: dict = field(default_factory=dict)

    @property
    def center(self) -> Point:
        """Dynamically computes the center point of the current bounding box."""
        return Point(x=self.bbox.center_x, y=self.bbox.center_y)

    @property
    def previous_center(self) -> Optional[Point]:
        """Returns the center point of the previous snapshot from history."""
        if self.history.length > 1:
            return self.history.snapshots[-2].center
        return None
