from dataclasses import dataclass, field
from typing import List
from src.engines.tracking.entities.tracked_vehicle import TrackedVehicle

@dataclass
class TrackingResult:
    """Represents the output state of the Multi-Object Tracking Engine for a single frame.
    
    Acts as a domain-centric payload containing currently tracked active vehicle entities.
    """
    frame_number: int
    timestamp: float
    tracked_vehicles: List[TrackedVehicle]
    tracking_latency_ms: float
    tracking_statistics: dict = field(default_factory=dict)
    attributes: dict = field(default_factory=dict)

    @property
    def active_tracks_count(self) -> int:
        """Returns the count of verified vehicles tracked in the active frame."""
        return len(self.tracked_vehicles)
