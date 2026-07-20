import cv2
import numpy as np
from src.utils.visualization import get_resolution_scaling

class TelemetryOverlay:
    """Renders a comprehensive live system telemetry panel at the top of the frame."""

    def render_panel(self, frame: np.ndarray, telemetry_data: dict, current_frame: int, total_frames: int) -> None:
        h, w, _ = frame.shape
        thickness, font_scale, scale_val = get_resolution_scaling(h)

        # Height of telemetry HUD: 85px on standard 1080p, scales proportionally
        hud_h = int(85 * scale_val)
        if hud_h < 75:
            hud_h = 75

        # Create transparent background overlay banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (18, 18, 18), cv2.FILLED)
        cv2.line(overlay, (0, hud_h), (w, hud_h), (55, 55, 55), 1)
        
        alpha = 0.82
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Fonts configuration
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_title = 0.50 * scale_val
        font_scale_sub = 0.30 * scale_val
        font_scale_main = 0.40 * scale_val
        
        color_white = (242, 242, 242)
        color_green = (95, 220, 95)
        color_orange = (235, 135, 25)
        color_gray = (150, 150, 150)

        # Column positions (X-coordinates)
        col0_x = int(15 * scale_val)
        col1_x = int(w * 0.18)
        col2_x = int(w * 0.38)
        col3_x = int(w * 0.58)
        col4_x = int(w * 0.78)

        # Col 0: Brand Title
        cv2.putText(frame, "VELOX ANALYTICS", (col0_x, int(25 * scale_val)), font, font_scale_title, color_orange, max(1, int(round(2 * scale_val))), cv2.LINE_AA)
        cv2.putText(frame, f"v0.3.3 | {w}x{h}", (col0_x, int(45 * scale_val)), font, font_scale_sub, color_gray, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Frame: {current_frame}", (col0_x, int(65 * scale_val)), font, font_scale_sub, color_gray, 1, cv2.LINE_AA)

        # Col 1: System Speeds (FPS)
        cv2.putText(frame, "PROCESSING SPEED", (col1_x, int(20 * scale_val)), font, font_scale_sub, color_gray, 1, cv2.LINE_AA)
        fps = telemetry_data.get("fps", 0.0)
        det_fps = telemetry_data.get("det_fps", 0.0)
        track_fps = telemetry_data.get("track_fps", 0.0)
        cv2.putText(frame, f"Pipeline FPS: {fps:.1f}", (col1_x, int(38 * scale_val)), font, font_scale_main, color_green, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Det FPS: {det_fps:.1f} | Track: {track_fps:.1f}", (col1_x, int(58 * scale_val)), font, font_scale_sub, color_gray, 1, cv2.LINE_AA)

        # Col 2: Active Tracks (Stats)
        cv2.putText(frame, "TRACK TELEMETRY", (col2_x, int(20 * scale_val)), font, font_scale_sub, color_gray, 1, cv2.LINE_AA)
        active = telemetry_data.get("active_tracks", 0)
        lost = telemetry_data.get("lost_tracks", 0)
        recovered = telemetry_data.get("recovered_tracks", 0)
        cv2.putText(frame, f"Active Tracks: {active}", (col2_x, int(38 * scale_val)), font, font_scale_main, color_white, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Lost: {lost} | Recovered: {recovered}", (col2_x, int(58 * scale_val)), font, font_scale_sub, color_gray, 1, cv2.LINE_AA)

        # Col 3: Lifetime & Averages
        cv2.putText(frame, "HISTORICAL METRICS", (col3_x, int(20 * scale_val)), font, font_scale_sub, color_gray, 1, cv2.LINE_AA)
        total = telemetry_data.get("total_tracks", 0)
        longest = telemetry_data.get("longest_lifetime", 0)
        cv2.putText(frame, f"Total Created: {total}", (col3_x, int(38 * scale_val)), font, font_scale_main, color_white, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Max Lifetime: {longest} frames", (col3_x, int(58 * scale_val)), font, font_scale_sub, color_gray, 1, cv2.LINE_AA)

        # Col 4: Quality Index & Memory usage
        cv2.putText(frame, "QUALITY & FOOTPRINT", (col4_x, int(20 * scale_val)), font, font_scale_sub, color_gray, 1, cv2.LINE_AA)
        avg_conf = telemetry_data.get("average_confidence", 0.0)
        mem_kb = telemetry_data.get("memory_usage_kb", 0.0)
        cv2.putText(frame, f"Avg Confidence: {avg_conf:.1f}%", (col4_x, int(38 * scale_val)), font, font_scale_main, color_green if avg_conf > 75 else color_orange, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Memory: {mem_kb:.1f} KB", (col4_x, int(58 * scale_val)), font, font_scale_sub, color_gray, 1, cv2.LINE_AA)

        # Progress timeline bar at the very base of the banner
        if total_frames > 0:
            progress_ratio = min(1.0, current_frame / total_frames)
            bar_w = int(w * progress_ratio)
            cv2.line(frame, (0, hud_h - 1), (bar_w, hud_h - 1), color_orange, max(2, int(round(3 * scale_val))))
