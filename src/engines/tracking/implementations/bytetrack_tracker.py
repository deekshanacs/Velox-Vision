import time
import logging
from typing import Dict, List, Optional

from src.engines.tracking.interfaces.tracker import Tracker
from src.engines.tracking.configuration.tracking_config import TrackingConfiguration
from src.engines.tracking.entities.tracking_context import TrackingContext
from src.engines.tracking.entities.tracking_result import TrackingResult
from src.engines.tracking.entities.tracked_vehicle import TrackedVehicle
from src.engines.tracking.entities.track_state import TrackState
from src.engines.tracking.entities.track_history import TrackSnapshot, TrackHistory
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
        
        # Local state cache maps track IDs to TrackedVehicle domain entities.
        # This is maintained locally to retain historic trajectories and placeholders
        # for downstream modules (e.g., speed estimation and ANPR characters).
        self._active_tracks: Dict[int, TrackedVehicle] = {}
        
        self._statistics = TrackingStatistics()
        self._performance = TrackingPerformance()
        self._is_initialized = False

    def initialize(self, configs: TrackingConfiguration) -> None:
        """Initializes the tracker adapter, local state repositories, and telemetry containers."""
        try:
            logger.info("Initializing ByteTrack tracker subsystem...")
            self._configs = configs
            self._adapter = ByteTrackAdapter(configs)
            self._active_tracks.clear()
            self._statistics.reset()
            self._performance.reset()
            self._is_initialized = True
            logger.info("ByteTrack tracker successfully initialized.")
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
                    vehicle.bbox = bbox
                    vehicle.confidence = conf
                    vehicle.last_seen_frame = frame_num
                    vehicle.track_age += 1
                    
                    # State transitions machine logic
                    if vehicle.state == TrackState.TEMPORARILY_LOST:
                        vehicle.state = TrackState.RECOVERED
                        self._statistics.recovered_tracks += 1
                        logger.debug(f"Track {tid} RECOVERED at frame {frame_num}")
                    elif vehicle.state == TrackState.RECOVERED:
                        vehicle.state = TrackState.TRACKED
                    elif vehicle.state == TrackState.TENTATIVE:
                        if vehicle.track_age >= self._configs.tentative_frames:
                            vehicle.state = TrackState.CONFIRMED
                            self._statistics.total_track_age += vehicle.track_age
                            logger.info(f"Track {tid} CONFIRMED at frame {frame_num} after {vehicle.track_age} frames.")
                    elif vehicle.state == TrackState.CONFIRMED:
                        vehicle.state = TrackState.TRACKED
                    
                    # Store track snapshot (no velocity/speed calculations)
                    snap = TrackSnapshot(
                        frame_number=frame_num,
                        timestamp=timestamp,
                        center=vehicle.center,
                        bbox=bbox,
                        confidence=conf,
                        state=vehicle.state
                    )
                    vehicle.history.append(snap)
                else:
                    # Instantiated new vehicle trajectory in tentative state
                    history = TrackHistory(max_size=self._configs.history_size)
                    vehicle = TrackedVehicle(
                        track_id=tid,
                        class_name=cname,
                        class_id=cid,
                        bbox=bbox,
                        confidence=conf,
                        state=TrackState.TENTATIVE,
                        first_seen_frame=frame_num,
                        last_seen_frame=frame_num,
                        track_age=1,
                        history=history
                    )
                    
                    snap = TrackSnapshot(
                        frame_number=frame_num,
                        timestamp=timestamp,
                        center=vehicle.center,
                        bbox=bbox,
                        confidence=conf,
                        state=TrackState.TENTATIVE
                    )
                    vehicle.history.append(snap)
                    
                    self._active_tracks[tid] = vehicle
                    self._statistics.tracks_created += 1
                    logger.debug(f"Track {tid} CREATED as tentative at frame {frame_num}")

                tracked_vehicles_output.append(vehicle)

            # 3. Handle missing track frames (Temporarily Lost or Exited Removal)
            lost_ids = []
            for tid, vehicle in list(self._active_tracks.items()):
                if tid not in matched_ids:
                    frames_lost = frame_num - vehicle.last_seen_frame
                    if frames_lost > self._configs.max_lost_frames:
                        # Exceeded lost buffer retention threshold, delete track
                        vehicle.state = TrackState.REMOVED
                        self._statistics.tracks_removed += 1
                        lost_ids.append(tid)
                        logger.info(f"Track {tid} REMOVED after exceeding {self._configs.max_lost_frames} lost frames.")
                    else:
                        # Transition to Temporarily Lost status
                        if vehicle.state != TrackState.TEMPORARILY_LOST:
                            vehicle.state = TrackState.TEMPORARILY_LOST
                            self._statistics.tracks_lost += 1
                            logger.debug(f"Track {tid} marked as TEMPORARILY_LOST at frame {frame_num}")
                        
                        snap = TrackSnapshot(
                            frame_number=frame_num,
                            timestamp=timestamp,
                            center=vehicle.center,
                            bbox=vehicle.bbox,
                            confidence=vehicle.confidence,
                            state=TrackState.TEMPORARILY_LOST
                        )
                        vehicle.history.append(snap)

            for tid in lost_ids:
                del self._active_tracks[tid]

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
