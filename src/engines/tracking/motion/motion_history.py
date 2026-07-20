from dataclasses import dataclass
from typing import Optional
from src.engines.tracking.value_objects.point import Point
from src.engines.tracking.motion.motion_state import MotionState
from src.engines.tracking.motion.motion_vector import MotionVector
from src.engines.tracking.motion.motion_metrics import BoundingBoxStability, ConfidenceAnalysis
from src.engines.tracking.motion.trajectory import Trajectory
from src.engines.tracking.motion.trajectory_statistics import TrajectoryStatistics

# Downstream analysis placeholders
@dataclass(frozen=True)
class SpeedProfile:
    """Placeholder for Phase 4.2 speed metrics."""
    pass

@dataclass(frozen=True)
class LaneAssignment:
    """Placeholder for Phase 4.3 lane tracking metrics."""
    pass

@dataclass(frozen=True)
class BehaviorProfile:
    """Placeholder for Phase 4.4 behavior metrics."""
    pass

@dataclass(frozen=True)
class ViolationProfile:
    """Placeholder for Phase 4.5 violation event metrics."""
    pass

@dataclass(frozen=True)
class PredictionProfile:
    """Placeholder for Phase 4.6 motion trajectory prediction."""
    pass


@dataclass(frozen=True)
class MotionProfile:
    """Comprehensive motion statistics accumulated over a vehicle track's lifetime."""
    current_heading: float
    average_heading: float
    motion_state: MotionState
    current_motion_vector: MotionVector
    average_motion_vector: MotionVector
    
    net_displacement: float
    total_travelled_distance: float
    current_center: Point
    average_center: Point
    path_efficiency: float
    
    bbox_stability: BoundingBoxStability
    confidence_analysis: ConfidenceAnalysis
    trajectory: Trajectory
    trajectory_stats: TrajectoryStatistics
    
    stationary_duration_frames: int
    moving_duration_frames: int

    # Future integration profiles
    speed_profile: Optional[SpeedProfile] = None
    lane_assignment: Optional[LaneAssignment] = None
    behavior_profile: Optional[BehaviorProfile] = None
    violation_profile: Optional[ViolationProfile] = None
    prediction_profile: Optional[PredictionProfile] = None
