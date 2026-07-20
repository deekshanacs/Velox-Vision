import cv2
import numpy as np
from src.engines.tracking.entities.tracked_vehicle import TrackedVehicle
from src.engines.tracking.visualization.color_palette import get_state_color
from src.utils.visualization import draw_rounded_rectangle, get_resolution_scaling

class TrackingOverlay:
    """Renders professional bounding boxes and multi-line detailed cards for vehicles."""

    def render_vehicle(self, frame: np.ndarray, vehicle: TrackedVehicle, show_states: bool = True, show_memory: bool = True) -> None:
        """Renders bounding box and metadata overlay for a tracked vehicle."""
        if vehicle.bbox is None:
            return

        h, w, _ = frame.shape
        thickness, font_scale, scale_val = get_resolution_scaling(h)

        state_color = get_state_color(vehicle.state)
        x1, y1, x2, y2 = vehicle.bbox.x1, vehicle.bbox.y1, vehicle.bbox.x2, vehicle.bbox.y2
        
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))

        # Draw rounded bounding box
        draw_rounded_rectangle(frame, pt1, pt2, state_color, thickness=thickness, r=8)

        # 2. Build multi-line text lines
        # Line 1: Class + ID (e.g. "CAR #14")
        line1 = f"{vehicle.class_name.upper()} #{vehicle.track_id}"
        
        # Line 2: State and Confidence (e.g. "Tracked | 96%")
        line2 = ""
        if show_states:
            line2 += f"{vehicle.state.name.capitalize()} "
        line2 += f"{int(vehicle.confidence * 100)}%"

        # Line 3: Age & Obs (e.g. "Age: 62 | Obs: 61")
        line3 = ""
        if show_memory:
            line3 = f"Age:{vehicle.track_age} Obs:{vehicle.memory.observation_count}"

        lines = [line1, line2]
        if line3:
            lines.append(line3)

        # Render detail card
        self._draw_detail_card(frame, lines, pt1, state_color, thickness, font_scale, scale_val)

    def _draw_detail_card(self, img: np.ndarray, lines: list, position: tuple, bg_color: tuple, thickness: int, font_scale: float, scale_val: float) -> None:
        """Draws a multi-line detail card with padded background."""
        x, y = position
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Calculate size for all lines
        text_sizes = []
        for line in lines:
            (w_txt, h_txt), _ = cv2.getTextSize(line, font, font_scale, max(1, thickness - 1))
            text_sizes.append((w_txt, h_txt))

        max_w = max(sz[0] for sz in text_sizes)
        line_spacing = int(4 * (font_scale / 0.4))
        
        # Calculate total height of the card
        total_h = sum(sz[1] for sz in text_sizes) + (len(lines) - 1) * line_spacing + 12
        
        # Position label cleanly on bounding box top border or inside if outside boundary
        rect_pt1 = (int(x), int(y - total_h))
        # Clamp to frame coordinates
        if rect_pt1[1] < 0:
            rect_pt1 = (int(x), int(y))
            rect_pt2 = (int(x + max_w + 12), int(y + total_h))
            draw_y = y + text_sizes[0][1] + 6
        else:
            rect_pt2 = (int(x + max_w + 12), int(y))
            draw_y = y - total_h + text_sizes[0][1] + 6

        # Draw transparent filled background card
        overlay = img.copy()
        cv2.rectangle(overlay, rect_pt1, rect_pt2, bg_color, cv2.FILLED)
        cv2.rectangle(overlay, rect_pt1, rect_pt2, (255, 255, 255), max(1, int(round(1 * scale_val))))
        
        alpha = 0.85
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # Draw lines of text
        text_color = (255, 255, 255)
        curr_y = draw_y
        for i, line in enumerate(lines):
            cv2.putText(
                img,
                line,
                (int(x + 6), int(curr_y)),
                font,
                font_scale,
                text_color,
                max(1, thickness - 1),
                cv2.LINE_AA
            )
            curr_y += text_sizes[i][1] + line_spacing
