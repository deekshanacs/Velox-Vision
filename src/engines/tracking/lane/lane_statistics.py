from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class LaneTransition:
    """Record of a lane change transition event."""
    from_lane_id: Optional[int]
    to_lane_id: int
    timestamp: float
    frame_index: int
    duration_seconds: float = 0.0


@dataclass(frozen=True)
class LaneTransitionHistory:
    """Historical log of all lane change transitions for a vehicle track."""
    transitions: List[LaneTransition] = field(default_factory=list)

    @property
    def total_transitions(self) -> int:
        return len(self.transitions)


@dataclass(frozen=True)
class LaneOccupancy:
    """Accumulated occupancy duration per lane ID."""
    occupancy_seconds: Dict[int, float] = field(default_factory=dict)
    occupancy_frames: Dict[int, int] = field(default_factory=dict)

    def get_time_in_lane(self, lane_id: int) -> float:
        return self.occupancy_seconds.get(lane_id, 0.0)


@dataclass(frozen=True)
class LaneStatistics:
    """Combined statistics container for lane tracking."""
    current_lane_occupancy_seconds: float
    current_lane_occupancy_frames: int
    transition_history: LaneTransitionHistory
    occupancy: LaneOccupancy
