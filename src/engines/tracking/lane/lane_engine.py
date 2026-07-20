import time
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from src.engines.tracking.memory.vehicle_memory import VehicleMemory
from src.engines.tracking.motion.motion_history import MotionProfile
from src.engines.tracking.speed.speed_profile import SpeedProfile
from src.engines.tracking.configuration.tracking_config import TrackingConfiguration
from src.engines.tracking.lane.lane_configuration import LaneConfiguration
from src.engines.tracking.lane.lane_assignment import LaneAssignmentStatus
from src.engines.tracking.lane.lane_model import RoadModel, Lane
from src.engines.tracking.lane.lane_validator import LaneValidator
from src.engines.tracking.lane.lane_geometry import LaneGeometryCalculator, LaneGeometryMetrics
from src.engines.tracking.lane.lane_statistics import (
    LaneStatistics,
    LaneTransition,
    LaneTransitionHistory,
    LaneOccupancy
)
from src.engines.tracking.lane.lane_profile import LaneProfile, LaneObservation


class TrackLaneState:

    """Internal mutable tracking state for a vehicle track."""
    current_lane_id: Optional[int] = None
    previous_lane_id: Optional[int] = None
    status: LaneAssignmentStatus = LaneAssignmentStatus.UNKNOWN
    pending_lane_id: Optional[int] = None
    pending_count: int = 0
    occupancy_seconds: Dict[int, float] = None
    occupancy_frames: Dict[int, int] = None
    transitions: List[LaneTransition] = None
    observations: List[LaneObservation] = None

    def __init__(self):
        self.current_lane_id = None
        self.previous_lane_id = None
        self.status = LaneAssignmentStatus.UNKNOWN
        self.pending_lane_id = None
        self.pending_count = 0
        self.occupancy_seconds = {}
        self.occupancy_frames = {}
        self.transitions = []
        self.observations = []


class LaneEngine:
    """Enterprise-grade Lane Intelligence Engine (Phase 4.3).
    
    Determines lane relationships, lateral offsets, lane change transitions,
    and occupancy statistics for tracked vehicles using VehicleMemory, MotionProfile, and SpeedProfile.
    Strictly architecturally isolated: Never consumes raw object detections.
    Operates incrementally (<1 ms runtime per vehicle).
    """

    def __init__(
        self,
        road_model: Optional[RoadModel] = None,
        config: Optional[LaneConfiguration] = None
    ):
        self.config = config or LaneConfiguration()
        self.road_model = road_model or RoadModel.create_from_config(self.config)
        LaneValidator.validate_road_model(self.road_model)
        self._states: Dict[int, TrackLaneState] = {}

    def compute_lane_profile(
        self,
        memory: VehicleMemory,
        motion_profile: MotionProfile,
        speed_profile: Optional[SpeedProfile] = None,
        tracking_config: Optional[TrackingConfiguration] = None
    ) -> Optional[LaneProfile]:
        """Computes or incrementally updates the LaneProfile for a vehicle track.
        
        Args:
            memory: Valid VehicleMemory instance.
            motion_profile: Valid MotionProfile from Phase 4.1.
            speed_profile: Optional SpeedProfile from Phase 4.2.
            tracking_config: System tracking configuration container.
            
        Returns:
            Constructed LaneProfile, or None if insufficient snapshot data.
        """
        # Architectural rule validation: Never consume raw detections
        if not isinstance(memory, VehicleMemory):
            raise TypeError("LaneEngine must consume VehicleMemory instance, not raw detections.")
        if not isinstance(motion_profile, MotionProfile):
            raise TypeError("LaneEngine must consume MotionProfile instance, not raw detections.")
        if speed_profile is not None and not isinstance(speed_profile, SpeedProfile):
            raise TypeError("Speed profile input must be a valid SpeedProfile instance.")

        if not memory.snapshots:
            return None

        track_id = memory.track_id
        latest_snap = memory.snapshots[-1]
        center = latest_snap.center
        heading = motion_profile.current_heading
        frame_number = latest_snap.frame_number
        timestamp = latest_snap.timestamp

        # Determine frame delta time dt
        if len(memory.snapshots) >= 2:
            dt = latest_snap.timestamp - memory.snapshots[-2].timestamp
            if dt <= 0:
                dt = 0.04
        else:
            dt = 0.04

        # Retrieve or initialize track state
        state = self._states.get(track_id)
        if state is None:
            state = TrackLaneState()
            self._states[track_id] = state

        # 1. Match vehicle center to road model lanes
        raw_lane_id, matched_lane = self._find_matching_lane(center.x)
        
        # Determine geometric metrics
        if matched_lane is not None:
            geom_metrics = LaneGeometryCalculator.calculate(center, matched_lane, heading)
        else:
            geom_metrics = None

        # 2. Determine assignment status & handle hysteresis
        margin = self.config.boundary_margin
        is_near_boundary = False
        if matched_lane is not None:
            left_dist = abs(center.x - matched_lane.left_boundary.position_x)
            right_dist = abs(center.x - matched_lane.right_boundary.position_x)
            if left_dist <= margin or right_dist <= margin:
                is_near_boundary = True

        if raw_lane_id is None:
            target_status = LaneAssignmentStatus.OUTSIDE_LANE
            target_lane_id = None
        elif len(memory.snapshots) < self.config.hysteresis_frames:
            target_status = LaneAssignmentStatus.TEMPORARY_ASSIGNMENT
            target_lane_id = raw_lane_id
        elif is_near_boundary:
            target_status = LaneAssignmentStatus.TRANSITIONING
            target_lane_id = raw_lane_id
        else:
            target_status = LaneAssignmentStatus.ASSIGNED
            target_lane_id = raw_lane_id

        # Hysteresis state filtering (prevent rapid toggling near boundary)
        if state.current_lane_id is not None and target_lane_id != state.current_lane_id:
            if state.pending_lane_id == target_lane_id:
                state.pending_count += 1
            else:
                state.pending_lane_id = target_lane_id
                state.pending_count = 1

            if state.pending_count >= self.config.hysteresis_frames:
                # Confirm lane transition
                state.previous_lane_id = state.current_lane_id
                state.current_lane_id = target_lane_id
                state.status = target_status
                state.transitions.append(
                    LaneTransition(
                        from_lane_id=state.previous_lane_id,
                        to_lane_id=target_lane_id,
                        timestamp=timestamp,
                        frame_index=frame_number
                    )
                )
                state.pending_lane_id = None
                state.pending_count = 0
            else:
                # Remain in current lane during hysteresis window
                pass
        else:
            state.pending_lane_id = None
            state.pending_count = 0
            if state.current_lane_id is None:
                state.current_lane_id = target_lane_id
            state.status = target_status

        # 3. Update occupancy statistics
        curr_id = state.current_lane_id
        if curr_id is not None:
            state.occupancy_seconds[curr_id] = state.occupancy_seconds.get(curr_id, 0.0) + dt
            state.occupancy_frames[curr_id] = state.occupancy_frames.get(curr_id, 0) + 1

        # 4. Calculate lane confidence score
        obs_conf = motion_profile.confidence_analysis.observation_confidence
        if state.status == LaneAssignmentStatus.ASSIGNED:
            lane_conf = round(0.5 * obs_conf + 0.5, 3)
        elif state.status == LaneAssignmentStatus.TRANSITIONING:
            lane_conf = round(0.5 * obs_conf + 0.25, 3)
        elif state.status == LaneAssignmentStatus.TEMPORARY_ASSIGNMENT:
            lane_conf = round(0.4 * obs_conf + 0.2, 3)
        else:
            lane_conf = 0.2

        # Record rich observation
        obs = LaneObservation(
            frame_number=frame_number,
            timestamp=timestamp,
            lane_id=curr_id,
            confidence=lane_conf,
            assignment_status=state.status,
            lateral_offset=geom_metrics.normalized_lateral_offset if geom_metrics else 0.0
        )
        state.observations.append(obs)
        if len(state.observations) > 300:
            state.observations.pop(0)

        # Build statistics object
        occupancy_obj = LaneOccupancy(
            occupancy_seconds=dict(state.occupancy_seconds),
            occupancy_frames=dict(state.occupancy_frames)
        )
        trans_history = LaneTransitionHistory(transitions=list(state.transitions))
        
        curr_sec = state.occupancy_seconds.get(curr_id, 0.0) if curr_id else 0.0
        curr_frames = state.occupancy_frames.get(curr_id, 0) if curr_id else 0
        
        stats = LaneStatistics(
            current_lane_occupancy_seconds=round(curr_sec, 2),
            current_lane_occupancy_frames=curr_frames,
            transition_history=trans_history,
            occupancy=occupancy_obj
        )

        return LaneProfile(
            current_lane_id=curr_id,
            previous_lane_id=state.previous_lane_id,
            assignment_status=state.status,
            lane_confidence=lane_conf,
            geometry_metrics=geom_metrics,
            occupancy_seconds=round(curr_sec, 2),
            occupancy_frames=curr_frames,
            transition_count=trans_history.total_transitions,
            observations=list(state.observations),
            statistics=stats
        )

    def _find_matching_lane(self, x: float) -> Tuple[Optional[int], Optional[Lane]]:
        """Finds which lane in RoadModel contains the x coordinate."""
        for lane_id, lane in self.road_model.lanes.items():
            if lane.contains_x(x):
                return lane_id, lane
        return None, None

    def clear_track(self, track_id: int) -> None:
        """Cleans up internal track state upon track removal."""
        self._states.pop(track_id, None)
