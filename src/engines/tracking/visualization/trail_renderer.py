import cv2
import numpy as np
from src.engines.tracking.entities.tracked_vehicle import TrackedVehicle
from src.engines.tracking.visualization.color_palette import get_state_color

class TrailRenderer:
    """Renders fading historical movement trails for tracked vehicles based on memory snapshots."""

    def __init__(self, trail_length: int = 40):
        self.trail_length = trail_length

    def render_trail(self, frame: np.ndarray, vehicle: TrackedVehicle, base_thickness: int = 2) -> None:
        """Draws a fading movement trail for a tracked vehicle."""
        snapshots = vehicle.memory.snapshots
        if len(snapshots) < 2:
            return

        # Take only the last trail_length snapshots
        active_snaps = snapshots[-self.trail_length:]
        num_points = len(active_snaps)
        if num_points < 2:
            return

        state_color = get_state_color(vehicle.state)

        for i in range(num_points - 1):
            pt1 = (int(active_snaps[i].center.x), int(active_snaps[i].center.y))
            pt2 = (int(active_snaps[i + 1].center.x), int(active_snaps[i + 1].center.y))

            # Fading factor (0.0 to 1.0)
            factor = (i + 1) / num_points
            
            # Fade thickness
            thickness = max(1, int(round(base_thickness * factor)))
            
            # Fade color (neons glow effect)
            b, g, r = state_color
            fade_color = (
                int(b * factor),
                int(g * factor),
                int(r * factor)
            )

            cv2.line(frame, pt1, pt2, fade_color, thickness, cv2.LINE_AA)
