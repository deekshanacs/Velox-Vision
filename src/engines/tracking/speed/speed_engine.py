import time
from typing import Dict, List, Optional, Union, TYPE_CHECKING
from src.engines.tracking.memory.vehicle_memory import VehicleMemory
from src.engines.tracking.motion.motion_state import MotionState
from src.engines.tracking.configuration.tracking_config import TrackingConfiguration
from src.engines.tracking.speed.speed_profile import SpeedProfile, CalibrationMode
from src.engines.tracking.speed.speed_statistics import SpeedStatistics
from src.engines.tracking.speed.calibration import SpeedCalibration
from src.engines.tracking.speed.perspective import PerspectiveTransformer
from src.engines.tracking.speed.smoothing import SpeedSmoother, SmoothingMethod

if TYPE_CHECKING:
    from src.engines.tracking.motion.motion_history import MotionProfile



class SpeedEngine:
    """Enterprise-grade Speed Estimation Engine (Phase 4.2).
    
    Estimates real-world vehicle speed using MotionProfile and VehicleMemory.
    Architecturally isolated: Never directly consumes raw detections.
    Operates incrementally (<1 ms runtime per vehicle).
    """

    def __init__(
        self,
        calibration: Optional[SpeedCalibration] = None,
        perspective: Optional[PerspectiveTransformer] = None,
        smoother: Optional[SpeedSmoother] = None
    ):
        self.calibration = calibration or SpeedCalibration()
        self.perspective = perspective or PerspectiveTransformer()
        self.smoother = smoother or SpeedSmoother()
        self._speed_histories: Dict[int, List[float]] = {}

    def compute_speed(
        self,
        motion_profile: 'MotionProfile',
        memory: VehicleMemory,
        config: Optional[TrackingConfiguration] = None
    ) -> Optional[SpeedProfile]:
        """Computes or incrementally updates the SpeedProfile for a vehicle track.
        
        Args:
            motion_profile: Valid MotionProfile from Phase 4.1.
            memory: Valid VehicleMemory snapshot sequence for the track.
            config: Optional system tracking configuration container.
            
        Returns:
            Constructed SpeedProfile, or None if insufficient history.
        """
        from src.engines.tracking.motion.motion_history import MotionProfile

        # Architectural constraint validation: Never consume raw detections
        if not isinstance(motion_profile, MotionProfile):
            raise TypeError("SpeedEngine must consume MotionProfile instance, not raw detections or raw boxes.")

        if not isinstance(memory, VehicleMemory):
            raise TypeError("SpeedEngine must consume VehicleMemory instance, not raw detections.")

        # Configuration parameters
        fps = getattr(config, 'frame_rate', 25.0) if config else 25.0
        min_dist = getattr(config, 'speed_minimum_motion_distance', 5.0) if config else 5.0
        smoothing_window = getattr(config, 'speed_smoothing_window', 8) if config else 8

        if len(memory.snapshots) < 2:
            return None

        track_id = memory.track_id
        snapshots = memory.snapshots
        curr_snap = snapshots[-1]
        prev_snap = snapshots[-2]

        dt = curr_snap.timestamp - prev_snap.timestamp
        if dt <= 0.0:
            dt = 1.0 / fps if fps > 0 else 0.04

        # Compute displacement
        p1 = prev_snap.center
        p2 = curr_snap.center
        px_displacement = p1.distance_to(p2)

        # Apply perspective / calibration conversion
        ground_displacement_meters, calib_mode, calib_conf = self.perspective.compute_ground_displacement(
            p1, p2, self.calibration
        )

        # Check stationary state or minimum motion distance
        is_stationary = (
            motion_profile.motion_state == MotionState.STATIONARY or
            px_displacement < min_dist
        )

        if is_stationary:
            raw_speed_kmh = 0.0
        else:
            raw_speed_mps = ground_displacement_meters / dt
            raw_speed_kmh = raw_speed_mps * 3.6

        # Retrieve vehicle speed history
        history = self._speed_histories.get(track_id, [])
        smoothed_speed_kmh = self.smoother.smooth(raw_speed_kmh, history)
        
        # Incremental history maintenance
        history.append(smoothed_speed_kmh)
        if len(history) > max(smoothing_window, 300):
            history.pop(0)
        self._speed_histories[track_id] = history

        # Compute speed statistics
        heading = motion_profile.current_heading
        stats = SpeedStatistics.compute(history, dt, heading)

        # Compute confidence metrics
        obs_conf = motion_profile.confidence_analysis.observation_confidence
        
        # Speed sequence variance confidence (lower variance = higher confidence)
        if stats.speed_variance < 5.0:
            speed_conf = 1.0
        elif stats.speed_variance < 25.0:
            speed_conf = 0.85
        else:
            speed_conf = max(0.2, 1.0 - (stats.speed_variance / 100.0))

        # Overall estimation quality weighted score
        overall_conf = round(
            0.35 * speed_conf + 0.35 * obs_conf + 0.30 * calib_conf, 3
        )
        overall_conf = min(1.0, max(0.0, overall_conf))

        current_speed_mps = round(stats.current_speed_kmh / 3.6, 2)

        profile = SpeedProfile(
            current_speed_mps=current_speed_mps,
            current_speed_kmh=stats.current_speed_kmh,
            average_speed_kmh=stats.average_speed_kmh,
            peak_speed_kmh=stats.peak_speed_kmh,
            min_speed_kmh=stats.min_speed_kmh,
            speed_history=list(history),
            acceleration_mps2=stats.acceleration_mps2,
            deceleration_mps2=stats.deceleration_mps2,
            is_accelerating=stats.is_accelerating,
            is_decelerating=stats.is_decelerating,
            velocity_vector_mps=stats.velocity_vector_mps,
            velocity_vector_kmh=stats.velocity_vector_kmh,
            timestamp=time.time(),
            speed_confidence=round(speed_conf, 3),
            calibration_confidence=round(calib_conf, 3),
            observation_confidence=round(obs_conf, 3),
            overall_confidence=overall_conf,
            calibration_mode=calib_mode
        )

        return profile

    def clear_track(self, track_id: int) -> None:
        """Cleans up cached speed history when a track is removed."""
        self._speed_histories.pop(track_id, None)
