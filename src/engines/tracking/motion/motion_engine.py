import math
from typing import List, Optional
from src.engines.tracking.memory.vehicle_memory import VehicleMemory
from src.engines.tracking.value_objects.point import Point
from src.engines.tracking.motion.motion_state import MotionState
from src.engines.tracking.motion.motion_vector import MotionVector
from src.engines.tracking.motion.motion_metrics import BoundingBoxStability, ConfidenceAnalysis
from src.engines.tracking.motion.trajectory import Trajectory
from src.engines.tracking.motion.trajectory_statistics import TrajectoryStatistics
from src.engines.tracking.motion.motion_history import MotionProfile

class MotionEngine:
    """Computes, updates, and tracks the MotionProfile of a vehicle from VehicleMemory.
    
    Contains no speed estimation algorithms.
    """

    def __init__(
        self,
        minimum_snapshots: int = 5,
        stationary_threshold: float = 2.5,
        heading_window: int = 10,
        smoothing_window: int = 8,
        confidence_threshold: float = 0.5
    ):
        self.minimum_snapshots = minimum_snapshots
        self.stationary_threshold = stationary_threshold
        self.heading_window = heading_window
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold

    def generate_profile(self, memory: VehicleMemory, current_profile: Optional[MotionProfile] = None) -> Optional[MotionProfile]:
        """Generates or incrementally updates the MotionProfile for the given vehicle memory."""
        snapshots = memory.snapshots
        if len(snapshots) < self.minimum_snapshots:
            return None

        # 1. Retrieve coordinates & centers
        centers = [s.center for s in snapshots]
        areas = []
        aspect_ratios = []
        for s in snapshots:
            if s.bbox is not None:
                areas.append(s.bbox.area)
                aspect_ratios.append(s.bbox.width / s.bbox.height if s.bbox.height > 0 else 1.0)
            else:
                areas.append(0.0)
                aspect_ratios.append(1.0)

        confidences = [s.confidence for s in snapshots]

        current_center = centers[-1]
        first_center = centers[0]

        # Calculate displacements
        frame_disps = []
        for i in range(len(centers) - 1):
            frame_disps.append(centers[i].distance_to(centers[i + 1]))

        total_dist = sum(frame_disps)
        net_disp = first_center.distance_to(current_center)
        path_eff = (net_disp / total_dist) if total_dist > 0 else 1.0

        # Current center / Avg center
        avg_cx = sum(c.x for c in centers) / len(centers)
        avg_cy = sum(c.y for c in centers) / len(centers)
        average_center = Point(avg_cx, avg_cy)

        # 2. Motion vectors
        # Frame-to-frame vector
        dx = current_center.x - centers[-2].x
        dy = current_center.y - centers[-2].y
        current_vector = MotionVector(dx, dy)

        # Average motion vector
        avg_dx = sum(centers[i+1].x - centers[i].x for i in range(len(centers)-1)) / (len(centers)-1)
        avg_dy = sum(centers[i+1].y - centers[i].y for i in range(len(centers)-1)) / (len(centers)-1)
        average_vector = MotionVector(avg_dx, avg_dy)

        # 3. Headings
        h_points = centers[-self.heading_window:]
        if len(h_points) >= 2:
            current_heading = math.atan2(h_points[-1].y - h_points[-2].y, h_points[-1].x - h_points[-2].x)
        else:
            current_heading = 0.0

        headings = []
        for i in range(len(centers) - 1):
            headings.append(math.atan2(centers[i+1].y - centers[i].y, centers[i+1].x - centers[i].x))
        average_heading = sum(headings) / len(headings) if headings else 0.0

        # Curvature & heading/direction changes
        direction_changes = 0
        heading_changes = 0
        turn_angles = []
        
        for i in range(len(headings) - 1):
            diff = math.atan2(math.sin(headings[i+1] - headings[i]), math.cos(headings[i+1] - headings[i]))
            turn_angles.append(abs(diff))
            if abs(diff) > 0.1:  # 0.1 rad turn threshold
                heading_changes += 1
            # Check sign changes in dx/dy to detect direction changes
            v1_dx = centers[i+1].x - centers[i].x
            v2_dx = centers[i+2].x - centers[i+1].x
            v1_dy = centers[i+1].y - centers[i].y
            v2_dy = centers[i+2].y - centers[i+1].y
            if (v1_dx * v2_dx < 0) or (v1_dy * v2_dy < 0):
                direction_changes += 1

        max_turn_angle = max(turn_angles) if turn_angles else 0.0
        total_turn = sum(turn_angles)
        trajectory_curvature = (total_turn / total_dist) if total_dist > 0 else 0.0
        
        mean_turn = sum(turn_angles) / len(turn_angles) if turn_angles else 0.0
        trajectory_smoothness = max(0.0, 1.0 - mean_turn / math.pi)

        # 4. Stationary vs Moving state durations
        if current_profile is not None:
            stationary_frames = current_profile.stationary_duration_frames
            moving_frames = current_profile.moving_duration_frames
            
            recent_displacement = current_vector.magnitude
            is_stationary = recent_displacement < self.stationary_threshold
            
            if is_stationary:
                stationary_frames += 1
                moving_frames = 0
            else:
                moving_frames += 1
                stationary_frames = 0
        else:
            consec_stationary = 0
            consec_moving = 0
            for i in range(len(centers) - 1):
                disp = centers[i].distance_to(centers[i+1])
                if disp < self.stationary_threshold:
                    consec_stationary += 1
                    consec_moving = 0
                else:
                    consec_moving += 1
                    consec_stationary = 0
            
            stationary_frames = consec_stationary
            moving_frames = consec_moving

        recent_displacement = current_vector.magnitude
        is_stationary = recent_displacement < self.stationary_threshold

        # Motion State classification
        if is_stationary:
            motion_state = MotionState.STATIONARY
        else:
            if recent_displacement < self.stationary_threshold * 2:
                motion_state = MotionState.SLOW_MOVEMENT
            else:
                motion_state = MotionState.MOVING

            if moving_frames >= 10:
                motion_state = MotionState.CONTINUOUS_MOVEMENT
            if direction_changes > 3:
                motion_state = MotionState.INTERMITTENT_MOVEMENT

        if memory.recovery_count > 0:
            motion_state = MotionState.RECOVERED_MOTION

        # 5. Aspect Ratio & Area Stability
        mean_area = sum(areas) / len(areas)
        var_area = sum((a - mean_area) ** 2 for a in areas) / len(areas)
        area_stability = var_area ** 0.5

        mean_ar = sum(aspect_ratios) / len(aspect_ratios)
        var_ar = sum((ar - mean_ar) ** 2 for ar in aspect_ratios) / len(aspect_ratios)
        ar_stability = var_ar ** 0.5

        center_jitter = sum(frame_disps) / len(frame_disps) if frame_disps else 0.0
        detection_stability = memory.observation_count / memory.track_age if memory.track_age > 0 else 0.0

        bbox_stability = BoundingBoxStability(
            area_stability=area_stability,
            aspect_ratio_stability=ar_stability,
            center_jitter=center_jitter,
            detection_stability=detection_stability
        )

        # 6. Confidence Analysis
        observation_confidence = sum(confidences) / len(confidences)
        motion_conf = 1.0 - (center_jitter / self.stationary_threshold) if center_jitter < self.stationary_threshold else 0.2
        motion_conf = min(1.0, max(0.0, motion_conf * detection_stability))

        trajectory_conf = trajectory_smoothness * path_eff
        
        confidence_analysis = ConfidenceAnalysis(
            motion_confidence=motion_conf,
            observation_confidence=observation_confidence,
            trajectory_confidence=trajectory_conf,
            tracking_confidence=memory.confidence
        )

        trajectory_entity = Trajectory(points=centers)
        trajectory_stats = TrajectoryStatistics(
            path_length=total_dist,
            trajectory_curvature=trajectory_curvature,
            direction_changes=direction_changes,
            heading_changes=heading_changes,
            average_heading=average_heading,
            maximum_turn_angle=max_turn_angle,
            trajectory_smoothness=trajectory_smoothness
        )

        return MotionProfile(
            current_heading=current_heading,
            average_heading=average_heading,
            motion_state=motion_state,
            current_motion_vector=current_vector,
            average_motion_vector=average_vector,
            net_displacement=net_disp,
            total_travelled_distance=total_dist,
            current_center=current_center,
            average_center=average_center,
            path_efficiency=path_eff,
            bbox_stability=bbox_stability,
            confidence_analysis=confidence_analysis,
            trajectory=trajectory_entity,
            trajectory_stats=trajectory_stats,
            stationary_duration_frames=stationary_frames,
            moving_duration_frames=moving_frames
        )
