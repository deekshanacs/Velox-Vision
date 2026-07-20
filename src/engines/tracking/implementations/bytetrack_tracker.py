import time
import logging
from typing import Dict, List, Optional

from src.engines.tracking.interfaces.tracker import Tracker
from src.engines.tracking.configuration.tracking_config import TrackingConfiguration
from src.engines.tracking.entities.tracking_context import TrackingContext
from src.engines.tracking.entities.tracking_result import TrackingResult
from src.engines.tracking.entities.tracked_vehicle import TrackedVehicle
from src.engines.tracking.entities.track_state import TrackState
from src.engines.tracking.memory.memory_manager import MemoryManager
from src.engines.tracking.metrics.tracking_statistics import TrackingStatistics
from src.engines.tracking.metrics.performance_metrics import TrackingPerformance
from src.engines.tracking.adapters.bytetrack_adapter import ByteTrackAdapter
from src.engines.tracking.exceptions import TrackerInitializationError, TrackingRuntimeError

logger = logging.getLogger(__name__)

class ByteTrackTracker(Tracker):
    """Concrete Multi-Object Tracker implementation orchestrating ByteTrackAdapter.
    
    This class maintains custom tracking metadata mappings (e.g. states, ages,
    and historical track snapshot queues) which are not supported natively by ByteTrack.
    """

    def __init__(self):
        self._configs: Optional[TrackingConfiguration] = None
        self._adapter: Optional[ByteTrackAdapter] = None
        self._memory_manager: Optional[MemoryManager] = None
        
        # Local state cache maps track IDs to TrackedVehicle domain entities.
        # This is maintained locally to retain historic trajectories and placeholders
        # for downstream modules (e.g., speed estimation and ANPR characters).
        self._active_tracks: Dict[int, TrackedVehicle] = {}
        self._motion_engine = None
        
        self._statistics = TrackingStatistics()
        self._performance = TrackingPerformance()
        self._is_initialized = False

    def initialize(self, configs: TrackingConfiguration) -> None:
        """Initializes the tracker adapter, local state repositories, and telemetry containers."""
        try:
            logger.info("Initializing ByteTrack tracker subsystem...")
            self._configs = configs
            self._adapter = ByteTrackAdapter(configs)
            self._memory_manager = MemoryManager(configs)
            self._active_tracks.clear()
            self._statistics.reset()
            self._performance.reset()
            
            # Instantiate Motion Analytics Engine
            self._motion_engine = None
            if configs.motion_enabled:
                from src.engines.tracking.motion.motion_engine import MotionEngine
                self._motion_engine = MotionEngine(
                    minimum_snapshots=configs.motion_minimum_snapshots,
                    stationary_threshold=configs.motion_stationary_threshold,
                    heading_window=configs.motion_heading_window,
                    smoothing_window=configs.motion_smoothing_window,
                    confidence_threshold=configs.motion_confidence_threshold
                )

            # Instantiate Speed Estimation Engine (Phase 4.2)
            self._speed_engine = None
            if getattr(configs, 'speed_enabled', True):
                from src.engines.tracking.speed import SpeedEngine, SpeedCalibration, SpeedSmoother, SmoothingMethod
                calib = SpeedCalibration(pixel_to_meter_ratio=configs.speed_pixel_to_meter_ratio)
                smoother = SpeedSmoother(
                    method=SmoothingMethod(configs.speed_smoothing_method),
                    window_size=configs.speed_smoothing_window,
                    max_speed_jump_kmh=configs.speed_max_speed_jump_kmh
                )
                self._speed_engine = SpeedEngine(calibration=calib, smoother=smoother)

            # Instantiate Lane Intelligence Engine (Phase 4.3)
            self._lane_engine = None
            if getattr(configs, 'lane_enabled', True):
                from src.engines.tracking.lane import LaneEngine, LaneConfiguration
                lane_cfg = LaneConfiguration(
                    enabled=configs.lane_enabled,
                    lane_count=configs.lane_count,
                    lane_width=configs.lane_width,
                    boundary_margin=configs.lane_boundary_margin,
                    confidence_threshold=configs.lane_confidence_threshold,
                    road_start_x=configs.lane_road_start_x,
                    coordinate_system=configs.lane_coordinate_system,
                    lane_origin=configs.lane_origin,
                    orientation=configs.lane_orientation,
                    hysteresis_frames=configs.lane_hysteresis_frames
                )
                self._lane_engine = LaneEngine(config=lane_cfg)

            self._is_initialized = True
            logger.info("ByteTrack tracker successfully initialized with Motion, Speed, and Lane engines.")

        except Exception as e:
            err_msg = f"Failed to initialize ByteTrack tracker: {e}"
            logger.error(err_msg, exc_info=True)
            raise TrackerInitializationError(err_msg) from e

    def track(self, context: TrackingContext) -> TrackingResult:
        """Processes detections inside the tracking context and updates vehicle trajectories.
        
        Args:
            context: Context details grouping frames, resolutions, and detections.
            
        Returns:
            TrackingResult listing active tracked vehicles.
        """
        if not self._is_initialized or self._adapter is None or self._configs is None:
            raise TrackingRuntimeError("Tracker not initialized. Call initialize() first.")

        frame_num = context.metadata.frame_number
        timestamp = context.metadata.timestamp

        try:
            # 1. Update adapter to run inference
            raw_vehicles, adapter_latency_ms = self._adapter.update(context)
            
            start_tracking_time = time.perf_counter()
            matched_ids = set()

            # 2. Iterate and reconcile active matches
            tracked_vehicles_output = []
            for raw_veh in raw_vehicles:
                tid = raw_veh.track_id
                bbox = raw_veh.bbox
                conf = raw_veh.confidence
                cid = raw_veh.class_id
                cname = raw_veh.class_name
                
                matched_ids.add(tid)
                
                if tid in self._active_tracks:
                    # Retrieve existing domain entity
                    vehicle = self._active_tracks[tid]
                    
                    # State transitions machine logic
                    state = TrackState.TENTATIVE
                    if vehicle.state == TrackState.TEMPORARILY_LOST:
                        state = TrackState.RECOVERED
                        self._statistics.recovered_tracks += 1
                        logger.debug(f"Track {tid} RECOVERED at frame {frame_num}")
                    elif vehicle.state == TrackState.RECOVERED:
                        state = TrackState.TRACKED
                    elif vehicle.state == TrackState.TENTATIVE:
                        if (vehicle.track_age + 1) >= self._configs.tentative_frames:
                            state = TrackState.CONFIRMED
                            self._statistics.total_track_age += (vehicle.track_age + 1)
                            logger.info(f"Track {tid} CONFIRMED at frame {frame_num} after {vehicle.track_age + 1} frames.")
                        else:
                            state = TrackState.TENTATIVE
                    elif vehicle.state == TrackState.CONFIRMED:
                        state = TrackState.TRACKED
                    else:
                        state = vehicle.state
                    
                    # Delegate update to memory manager
                    self._memory_manager.update_memory(tid, frame_num, timestamp, bbox, conf, state)
                else:
                    # Instantiate new vehicle trajectory with memory
                    memory = self._memory_manager.create_memory(tid, cname, cid, frame_num, timestamp, bbox, conf)
                    vehicle = TrackedVehicle(track_id=tid, memory=memory)
                    
                    # Store first observation snapshot
                    self._memory_manager.update_memory(tid, frame_num, timestamp, bbox, conf, TrackState.TENTATIVE)
                    
                    self._active_tracks[tid] = vehicle
                    self._statistics.tracks_created += 1
                    logger.debug(f"Track {tid} CREATED as tentative at frame {frame_num}")

                # Compute motion profile analytics (Phase 4.1)
                if self._motion_engine is not None:
                    profile = self._motion_engine.generate_profile(vehicle.memory, vehicle.motion_profile)
                    vehicle.motion_profile = profile

                # Compute speed profile analytics (Phase 4.2)
                if self._speed_engine is not None and vehicle.motion_profile is not None:
                    s_profile = self._speed_engine.compute_speed(vehicle.motion_profile, vehicle.memory, self._configs)
                    vehicle.speed_profile = s_profile

                # Compute lane profile analytics (Phase 4.3)
                if self._lane_engine is not None and vehicle.motion_profile is not None:
                    l_profile = self._lane_engine.compute_lane_profile(
                        vehicle.memory, vehicle.motion_profile, vehicle.speed_profile, self._configs
                    )
                    vehicle.lane_profile = l_profile

                tracked_vehicles_output.append(vehicle)

            # 3. Handle missing track frames (Temporarily Lost or Exited Removal)
            lost_ids = []
            for tid, vehicle in list(self._active_tracks.items()):
                if tid not in matched_ids:
                    frames_lost = frame_num - vehicle.last_seen_frame
                    if frames_lost > self._configs.max_lost_frames:
                        # Exceeded lost buffer retention threshold, delete track
                        state = TrackState.REMOVED
                        self._statistics.tracks_removed += 1
                        lost_ids.append(tid)
                        logger.info(f"Track {tid} REMOVED after exceeding {self._configs.max_lost_frames} lost frames.")
                    else:
                        # Transition to Temporarily Lost status
                        state = TrackState.TEMPORARILY_LOST
                        if vehicle.state != TrackState.TEMPORARILY_LOST:
                            self._statistics.tracks_lost += 1
                            logger.debug(f"Track {tid} marked as TEMPORARILY_LOST at frame {frame_num}")
                        
                    self._memory_manager.update_memory(tid, frame_num, timestamp, vehicle.bbox, vehicle.confidence, state)

            for tid in lost_ids:
                del self._active_tracks[tid]
                self._memory_manager.repository.remove(tid)
                if self._speed_engine is not None:
                    self._speed_engine.clear_track(tid)
                if self._lane_engine is not None:
                    self._lane_engine.clear_track(tid)


            # 4. Telemetry tracking latencies and output packing
            reconcile_latency = (time.perf_counter() - start_tracking_time) * 1000.0
            total_latency = adapter_latency_ms + reconcile_latency

            self._performance.processed_frames += 1
            self._performance.total_latency_ms += total_latency
            self._performance.current_active_tracks = len(self._active_tracks)
            if len(self._active_tracks) > self._performance.peak_active_tracks:
                self._performance.peak_active_tracks = len(self._active_tracks)

            result = TrackingResult(
                frame_number=frame_num,
                timestamp=timestamp,
                tracked_vehicles=tracked_vehicles_output,
                tracking_latency_ms=total_latency,
                tracking_statistics={
                    "total_tracks": self._statistics.tracks_created,
                    "active_tracks": len(self._active_tracks)
                }
            )
            return result

        except Exception as e:
            err_msg = f"Runtime tracking failure in ByteTrackTracker: {e}"
            logger.error(err_msg, exc_info=True)
            raise TrackingRuntimeError(err_msg) from e

    def reset(self) -> None:
        """Resets the tracker adapter, local state repositories, and statistics counters."""
        if self._adapter is not None:
            self._adapter.reset()
        if self._memory_manager is not None:
            self._memory_manager.repository.clear()
        self._active_tracks.clear()
        self._statistics.reset()
        self._performance.reset()
        logger.info("ByteTrack tracker state successfully reset.")

    def shutdown(self) -> None:
        """Cleans up resources and resets tracker initialization status flags."""
        self.reset()
        self._is_initialized = False
        logger.info("ByteTrack tracker successfully shut down.")

    def get_statistics(self) -> TrackingStatistics:
        """Returns statistics metrics container."""
        return self._statistics

    def get_performance(self) -> TrackingPerformance:
        """Returns performance metrics container."""
        return self._performance

    @property
    def is_initialized(self) -> bool:
        """Checks initialization status flag."""
        return self._is_initialized
