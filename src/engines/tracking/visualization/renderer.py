import time
from typing import List, Optional
import numpy as np

from src.engines.tracking.configuration.tracking_config import TrackingConfiguration
from src.engines.tracking.entities.tracked_vehicle import TrackedVehicle
from src.engines.tracking.visualization.trail_renderer import TrailRenderer
from src.engines.tracking.visualization.tracking_overlay import TrackingOverlay
from src.engines.tracking.visualization.telemetry_overlay import TelemetryOverlay
from src.engines.tracking.visualization.debug_overlay import DebugOverlay

class TrackingRenderer:
    """Master visual coordinator responsible for rendering vehicle overlays, trails, telemetry, and debugging inspect sidebar."""

    def __init__(self, configs: TrackingConfiguration):
        self.configs = configs
        self.trail_renderer = TrailRenderer(trail_length=configs.viz_trail_length)
        self.tracking_overlay = TrackingOverlay()
        self.telemetry_overlay = TelemetryOverlay()
        self.debug_overlay = DebugOverlay()
        
        # Internal cache of event messages for debug overlay
        self._debug_events: List[str] = []

    def log_debug_event(self, message: str) -> None:
        """Stores a debug event message to show in debug overlay."""
        self._debug_events.append(message)
        if len(self._debug_events) > 20:
            self._debug_events.pop(0)

    def render(
        self,
        frame: np.ndarray,
        tracked_vehicles: List[TrackedVehicle],
        telemetry_data: dict,
        current_frame: int,
        total_frames: int,
        selected_track_id: Optional[int] = None
    ) -> float:
        """Renders all active overlays onto the frame.
        
        Returns the rendering latency in milliseconds.
        """
        if not self.configs.viz_enabled:
            return 0.0

        start_time = time.perf_counter()

        # 1. Draw trails first so they render underneath bounding boxes
        if self.configs.viz_show_trails:
            for vehicle in tracked_vehicles:
                from src.engines.tracking.entities.track_state import TrackState
                if vehicle.state == TrackState.REMOVED:
                    continue
                self.trail_renderer.render_trail(frame, vehicle)

        # 2. Draw vehicle bounding boxes & detail cards
        for vehicle in tracked_vehicles:
            from src.engines.tracking.entities.track_state import TrackState
            if vehicle.state == TrackState.REMOVED:
                continue
            self.tracking_overlay.render_vehicle(
                frame,
                vehicle,
                show_states=self.configs.viz_show_states,
                show_memory=self.configs.viz_show_memory
            )

        # 3. Draw live telemetry dashboard panel
        if self.configs.viz_show_dashboard:
            self.telemetry_overlay.render_panel(frame, telemetry_data, current_frame, total_frames)

        # 4. Draw debug information & memory inspector
        render_latency_ms = (time.perf_counter() - start_time) * 1000.0

        if self.configs.viz_debug:
            debug_data = {
                "events": self._debug_events,
                "det_latency_ms": telemetry_data.get("det_latency_ms", 0.0),
                "track_latency_ms": telemetry_data.get("track_latency_ms", 0.0),
                "render_latency_ms": render_latency_ms
            }
            self.debug_overlay.render_debug_info(frame, debug_data)

        # Draw the Memory Inspector sidebar if a vehicle is selected
        if selected_track_id is not None:
            # Find the vehicle
            target_vehicle = next((v for v in tracked_vehicles if v.track_id == selected_track_id), None)
            if target_vehicle is not None:
                self.debug_overlay.render_memory_inspector(frame, target_vehicle)

        return (time.perf_counter() - start_time) * 1000.0
