import cv2
import numpy as np
from src.engines.tracking.entities.tracked_vehicle import TrackedVehicle
from src.utils.visualization import get_resolution_scaling

class DebugOverlay:
    """Renders debug metadata and a detailed interactive vehicle memory inspector sidebar on demand."""

    def render_debug_info(self, frame: np.ndarray, debug_data: dict) -> None:
        """Renders dynamic debug indicators (log event queues, latency meters)."""
        h, w, _ = frame.shape
        _, font_scale, scale_val = get_resolution_scaling(h)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Draw small bottom-left debug overlay listing recent tracker event details
        x_pos = int(15 * scale_val)
        y_pos = int(h - 15 * scale_val)
        
        events = debug_data.get("events", [])
        for evt in reversed(events[-4:]):  # Display last 4 events
            cv2.putText(frame, f"[DBG EVENT] {evt}", (x_pos, y_pos), font, 0.35 * scale_val, (0, 255, 255), 1, cv2.LINE_AA)
            y_pos -= int(18 * scale_val)

        # Show active latency details at the bottom-right corner
        latency_text = f"Det Latency: {debug_data.get('det_latency_ms', 0.0):.1f}ms | Track: {debug_data.get('track_latency_ms', 0.0):.1f}ms | Render: {debug_data.get('render_latency_ms', 0.0):.1f}ms"
        y_pos_lat = int(h - 15 * scale_val)
        cv2.putText(frame, latency_text, (int(w - 450 * scale_val), y_pos_lat), font, 0.35 * scale_val, (200, 200, 200), 1, cv2.LINE_AA)

    def render_memory_inspector(self, frame: np.ndarray, vehicle: TrackedVehicle) -> None:
        """Renders the detailed memory inspector panel on the right side of the screen."""
        h, w, _ = frame.shape
        _, font_scale, scale_val = get_resolution_scaling(h)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Sidebar width: 350px, scales proportionally
        sidebar_w = int(350 * scale_val)
        sidebar_x1 = w - sidebar_w
        
        # Transparent sidebar mask
        overlay = frame.copy()
        cv2.rectangle(overlay, (sidebar_x1, 0), (w, h), (20, 20, 20), cv2.FILLED)
        cv2.line(overlay, (sidebar_x1, 0), (sidebar_x1, h), (80, 80, 80), 2)
        
        alpha = 0.90
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Drawing title & metrics
        color_orange = (235, 135, 25)
        color_white = (242, 242, 242)
        color_green = (95, 220, 95)
        color_gray = (170, 170, 170)

        x_margin = sidebar_x1 + int(15 * scale_val)
        y_pos = int(45 * scale_val)

        # Section 1: Header Info
        cv2.putText(frame, "MEMORY INSPECTOR", (x_margin, y_pos), font, 0.55 * scale_val, color_orange, max(1, int(round(2 * scale_val))), cv2.LINE_AA)
        y_pos += int(25 * scale_val)
        cv2.putText(frame, f"Identity: {vehicle.class_name.upper()} #{vehicle.track_id}", (x_margin, y_pos), font, 0.42 * scale_val, color_white, 1, cv2.LINE_AA)
        y_pos += int(20 * scale_val)
        cv2.putText(frame, f"State: {vehicle.state.name}", (x_margin, y_pos), font, 0.38 * scale_val, color_green, 1, cv2.LINE_AA)
        y_pos += int(30 * scale_val)

        # Section 2: Lifetime & Counts
        cv2.putText(frame, "LIFETIME & OBSERVATIONS", (x_margin, y_pos), font, 0.38 * scale_val, color_orange, 1, cv2.LINE_AA)
        y_pos += int(18 * scale_val)
        cv2.putText(frame, f"Lifetime: {vehicle.track_age} frames", (x_margin, y_pos), font, 0.35 * scale_val, color_white, 1, cv2.LINE_AA)
        y_pos += int(18 * scale_val)
        cv2.putText(frame, f"Observation Count: {vehicle.memory.observation_count}", (x_margin, y_pos), font, 0.35 * scale_val, color_white, 1, cv2.LINE_AA)
        y_pos += int(30 * scale_val)

        # Section 3: Confidence stats
        cv2.putText(frame, "CONFIDENCE STATISTICS", (x_margin, y_pos), font, 0.38 * scale_val, color_orange, 1, cv2.LINE_AA)
        y_pos += int(18 * scale_val)
        cv2.putText(frame, f"Avg Confidence: {vehicle.memory.average_confidence() * 100:.1f}%", (x_margin, y_pos), font, 0.35 * scale_val, color_white, 1, cv2.LINE_AA)
        y_pos += int(18 * scale_val)
        cv2.putText(frame, f"Highest Confidence: {vehicle.memory.highest_confidence() * 100:.1f}%", (x_margin, y_pos), font, 0.35 * scale_val, color_white, 1, cv2.LINE_AA)
        y_pos += int(18 * scale_val)
        cv2.putText(frame, f"Lowest Confidence: {vehicle.memory.lowest_confidence() * 100:.1f}%", (x_margin, y_pos), font, 0.35 * scale_val, color_white, 1, cv2.LINE_AA)
        y_pos += int(18 * scale_val)
        cv2.putText(frame, f"Stability index: {vehicle.memory.confidence_stability():.4f}", (x_margin, y_pos), font, 0.35 * scale_val, color_white, 1, cv2.LINE_AA)
        y_pos += int(30 * scale_val)

        # Section 4: Movement descriptors
        cv2.putText(frame, "MOVEMENT DESCRIPTORS", (x_margin, y_pos), font, 0.38 * scale_val, color_orange, 1, cv2.LINE_AA)
        y_pos += int(18 * scale_val)
        cv2.putText(frame, f"Total Displacement: {vehicle.memory.total_displacement():.1f} px", (x_margin, y_pos), font, 0.35 * scale_val, color_white, 1, cv2.LINE_AA)
        y_pos += int(18 * scale_val)
        cv2.putText(frame, f"Net Displacement: {vehicle.memory.net_displacement():.1f} px", (x_margin, y_pos), font, 0.35 * scale_val, color_white, 1, cv2.LINE_AA)
        y_pos += int(18 * scale_val)
        cv2.putText(frame, f"Path Efficiency: {vehicle.memory.path_efficiency():.4f}", (x_margin, y_pos), font, 0.35 * scale_val, color_white, 1, cv2.LINE_AA)
        y_pos += int(30 * scale_val)

        # Section 5: Last Snapshot details
        cv2.putText(frame, "LAST OBSERVATION", (x_margin, y_pos), font, 0.38 * scale_val, color_orange, 1, cv2.LINE_AA)
        snapshots = vehicle.memory.snapshots
        if snapshots:
            last_snap = snapshots[-1]
            y_pos += int(18 * scale_val)
            cv2.putText(frame, f"Frame: {last_snap.frame_number} | Conf: {last_snap.confidence * 100:.0f}%", (x_margin, y_pos), font, 0.35 * scale_val, color_white, 1, cv2.LINE_AA)
            y_pos += int(18 * scale_val)
            cv2.putText(frame, f"Center: ({int(last_snap.center.x)}, {int(last_snap.center.y)})", (x_margin, y_pos), font, 0.35 * scale_val, color_white, 1, cv2.LINE_AA)
            y_pos += int(18 * scale_val)
            cv2.putText(frame, f"Area: {int(last_snap.area)} px^2", (x_margin, y_pos), font, 0.35 * scale_val, color_white, 1, cv2.LINE_AA)
        else:
            y_pos += int(18 * scale_val)
            cv2.putText(frame, "No snapshots recorded", (x_margin, y_pos), font, 0.35 * scale_val, color_gray, 1, cv2.LINE_AA)
        y_pos += int(30 * scale_val)

        # Section 6: Future placeholders (Phase 4 integration)
        cv2.putText(frame, "PHASE 4 PLACEHOLDERS", (x_margin, y_pos), font, 0.38 * scale_val, color_orange, 1, cv2.LINE_AA)
        y_pos += int(18 * scale_val)
        speed = vehicle.memory.estimated_speed_kmh if vehicle.memory.estimated_speed_kmh is not None else "N/A"
        cv2.putText(frame, f"Est. Speed: {speed} km/h", (x_margin, y_pos), font, 0.35 * scale_val, color_gray, 1, cv2.LINE_AA)
        y_pos += int(18 * scale_val)
        plate = vehicle.memory.license_plate_text if vehicle.memory.license_plate_text is not None else "N/A"
        cv2.putText(frame, f"License Plate: {plate}", (x_margin, y_pos), font, 0.35 * scale_val, color_gray, 1, cv2.LINE_AA)
        y_pos += int(18 * scale_val)
        risk = vehicle.memory.risk_score if vehicle.memory.risk_score is not None else "N/A"
        cv2.putText(frame, f"Risk Score: {risk}", (x_margin, y_pos), font, 0.35 * scale_val, color_gray, 1, cv2.LINE_AA)
