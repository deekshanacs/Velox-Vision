import time
from typing import List, Optional
from dataclasses import dataclass, field
from src.core.entities import BoundingBox
from src.engines.tracking.entities.track_state import TrackState
from src.engines.tracking.value_objects.point import Point
from src.engines.tracking.value_objects.velocity import Velocity
from src.engines.tracking.memory.memory_snapshot import MemorySnapshot

@dataclass
class VehicleMemory:
    """Persistent domain-centric memory accumulated over a vehicle's lifetime.
    
    Acts as the single source of truth for temporal, spatial, and confidence history.
    """
    track_id: int
    vehicle_class: str
    class_id: int
    creation_timestamp: float
    
    first_seen_frame: int
    last_seen_frame: int
    observation_count: int = 1
    
    # Latest active tracking coordinates
    state: TrackState = TrackState.TENTATIVE
    bbox: BoundingBox = field(default=None, repr=False)
    confidence: float = 0.0
    is_occluded: bool = False
    
    # Snapshots list
    snapshots: List[MemorySnapshot] = field(default_factory=list, repr=False)
    
    # Internal confidence metrics
    _highest_confidence: float = 0.0
    _lowest_confidence: float = 1.0
    _confidence_sum: float = 0.0
    _confidence_variance: float = 0.0
    
    # Spatial metrics
    path_length_accumulator: float = 0.0
    frames_lost: int = 0
    recovery_count: int = 0
    occlusion_count: int = 0

    # Downstream placeholders
    estimated_speed_kmh: Optional[float] = None
    license_plate_text: Optional[str] = None
    ocr_results: Optional[str] = None
    violation_events: List[dict] = field(default_factory=list, repr=False)
    current_lane: Optional[int] = None
    movement_direction: Optional[str] = None
    behavior_label: Optional[str] = None
    risk_score: Optional[float] = None
    prediction_trajectory: List[Point] = field(default_factory=list, repr=False)
    
    initial_center: Optional[Point] = None

    def __init__(
        self,
        track_id: int,
        vehicle_class: str,
        class_id: int,
        creation_timestamp: float,
        first_seen_frame: int,
        last_seen_frame: int,
        observation_count: int = 1,
        state: TrackState = TrackState.TENTATIVE,
        bbox: BoundingBox = None,
        confidence: float = 0.0,
        is_occluded: bool = False,
        snapshots: List[MemorySnapshot] = None,
        highest_confidence: float = 0.0,
        lowest_confidence: float = 1.0,
        confidence_sum: float = 0.0,
        confidence_variance: float = 0.0,
        path_length_accumulator: float = 0.0,
        frames_lost: int = 0,
        recovery_count: int = 0,
        occlusion_count: int = 0,
        estimated_speed_kmh: Optional[float] = None,
        license_plate_text: Optional[str] = None,
        ocr_results: Optional[str] = None,
        violation_events: List[dict] = None,
        current_lane: Optional[int] = None,
        movement_direction: Optional[str] = None,
        behavior_label: Optional[str] = None,
        risk_score: Optional[float] = None,
        prediction_trajectory: List[Point] = None,
        initial_center: Optional[Point] = None,
    ):
        self.track_id = track_id
        self.vehicle_class = vehicle_class
        self.class_id = class_id
        self.creation_timestamp = creation_timestamp
        self.first_seen_frame = first_seen_frame
        self.last_seen_frame = last_seen_frame
        self.observation_count = observation_count
        self.state = state
        self.bbox = bbox
        self.confidence = confidence
        self.is_occluded = is_occluded
        self.snapshots = snapshots if snapshots is not None else []
        self._highest_confidence = highest_confidence
        self._lowest_confidence = lowest_confidence
        self._confidence_sum = confidence_sum
        self._confidence_variance = confidence_variance
        self.path_length_accumulator = path_length_accumulator
        self.frames_lost = frames_lost
        self.recovery_count = recovery_count
        self.occlusion_count = occlusion_count
        self.estimated_speed_kmh = estimated_speed_kmh
        self.license_plate_text = license_plate_text
        self.ocr_results = ocr_results
        self.violation_events = violation_events if violation_events is not None else []
        self.current_lane = current_lane
        self.movement_direction = movement_direction
        self.behavior_label = behavior_label
        self.risk_score = risk_score
        self.prediction_trajectory = prediction_trajectory if prediction_trajectory is not None else []
        self.initial_center = initial_center
        
        if self.bbox is not None and self.initial_center is None:
            self.initial_center = Point(x=self.bbox.center_x, y=self.bbox.center_y)

    @property
    def track_age(self) -> int:
        """Returns the total number of frames since creation."""
        return self.last_seen_frame - self.first_seen_frame + 1

    @property
    def center(self) -> Point:
        """Dynamically computes the center point of the latest bounding box."""
        if self.bbox is not None:
            return Point(x=self.bbox.center_x, y=self.bbox.center_y)
        return Point(x=0.0, y=0.0)

    def average_confidence(self) -> float:
        """Calculates average detection confidence over all observations."""
        return (self._confidence_sum / self.observation_count) if self.observation_count > 0 else 0.0

    def highest_confidence(self) -> float:
        """Calculates highest detection confidence observed."""
        return self._highest_confidence

    def lowest_confidence(self) -> float:
        """Calculates lowest detection confidence observed."""
        return self._lowest_confidence

    def confidence_stability(self) -> float:
        """Calculates confidence stability (1.0 - standard deviation of confidence history).
        
        Returns 1.0 if there are fewer than 2 snapshots.
        """
        if len(self.snapshots) < 2:
            return 1.0
        confidences = [s.confidence for s in self.snapshots]
        mean = sum(confidences) / len(confidences)
        variance = sum((c - mean) ** 2 for c in confidences) / len(confidences)
        std_dev = variance ** 0.5
        return max(0.0, 1.0 - std_dev)

    @property
    def visibility_ratio(self) -> float:
        """Calculates track active visibility ratio over its lifetime."""
        total_frames = self.track_age
        return (self.observation_count / total_frames) if total_frames > 0 else 0.0

    def total_displacement(self) -> float:
        """Returns the total accumulated displacement (path length) of the vehicle."""
        return self.path_length_accumulator

    def net_displacement(self) -> float:
        """Calculates straight-line displacement between the first observed position and the latest position."""
        if self.initial_center is not None and self.bbox is not None:
            return self.initial_center.distance_to(self.center)
        return 0.0

    def path_efficiency(self) -> float:
        """Calculates path efficiency (net displacement / total displacement).
        
        Returns 1.0 if total displacement is 0.
        """
        total = self.total_displacement()
        if total > 0.0:
            return self.net_displacement() / total
        return 1.0

