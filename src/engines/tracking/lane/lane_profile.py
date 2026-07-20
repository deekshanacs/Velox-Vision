from dataclasses import dataclass, field
from typing import List, Optional
from src.engines.tracking.lane.lane_assignment import LaneAssignmentStatus
from src.engines.tracking.lane.lane_geometry import LaneGeometryMetrics
from src.engines.tracking.lane.lane_statistics import LaneStatistics, LaneTransitionHistory, LaneOccupancy


@dataclass(frozen=True)
class LaneObservation:
    """Rich snapshot of lane assignment observation at a specific frame."""
    frame_number: int
    timestamp: float
    lane_id: Optional[int]
    confidence: float
    assignment_status: LaneAssignmentStatus
    lateral_offset: float


@dataclass(frozen=True)
class LaneProfile:
    """Comprehensive lane intelligence profile for a tracked vehicle."""
    current_lane_id: Optional[int]
    previous_lane_id: Optional[int]
    assignment_status: LaneAssignmentStatus
    lane_confidence: float
    
    geometry_metrics: Optional[LaneGeometryMetrics]
    
    occupancy_seconds: float
    occupancy_frames: int
    transition_count: int
    
    observations: List[LaneObservation] = field(default_factory=list)
    statistics: Optional[LaneStatistics] = None
