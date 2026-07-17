import cv2
import numpy as np
from src.core.entities import DetectionMetrics

def draw_rounded_rectangle(
    img: np.ndarray,
    pt1: tuple,
    pt2: tuple,
    color: tuple,
    thickness: int = 2,
    r: int = 8
) -> None:
    """Draws a bounding box with rounded corners for a clean, modern HUD look.
    
    Args:
        img: Image frame to draw on.
        pt1: Top-left coordinate (x1, y1).
        pt2: Bottom-right coordinate (x2, y2).
        color: RGB/BGR tuple color of the box.
        thickness: Border thickness.
        r: Corner radius in pixels.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Ensure correct ordering
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    w = x2 - x1
    h = y2 - y1
    
    # Bound corner radius by half the width and height
    if r > w // 2:
        r = w // 2
    if r > h // 2:
        r = h // 2

    # Draw straight borders
    cv2.line(img, (int(x1 + r), int(y1)), (int(x2 - r), int(y1)), color, thickness)
    cv2.line(img, (int(x1 + r), int(y2)), (int(x2 - r), int(y2)), color, thickness)
    cv2.line(img, (int(x1), int(y1 + r)), (int(x1), int(y2 - r)), color, thickness)
    cv2.line(img, (int(x2), int(y1 + r)), (int(x2), int(y2 - r)), color, thickness)
    
    # Draw arcs at corners
    cv2.ellipse(img, (int(x1 + r), int(y1 + r)), (int(r), int(r)), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (int(x2 - r), int(y1 + r)), (int(r), int(r)), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (int(x2 - r), int(y2 - r)), (int(r), int(r)), 0, 0, 90, color, thickness)
    cv2.ellipse(img, (int(x1 + r), int(y2 - r)), (int(r), int(r)), 90, 0, 90, color, thickness)


def draw_metadata_label(
    img: np.ndarray,
    text: str,
    position: tuple,
    bg_color: tuple,
    text_color: tuple = (255, 255, 255)
) -> None:
    """Draws a clean metadata label with a solid background and anti-aliased text.
    
    Args:
        img: Frame to draw label on.
        text: Label string (e.g. 'CAR 98%').
        position: Top-left coordinate (x, y) where label starts.
        bg_color: BGR tuple color for background fill.
        text_color: BGR tuple color for text. Defaults to white.
    """
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    
    # Calculate text dimensions
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Define background border coordinates
    rect_pt1 = (int(x), int(y - h - 6))
    rect_pt2 = (int(x + w + 10), int(y + baseline))
    
    # Draw solid label background
    cv2.rectangle(img, rect_pt1, rect_pt2, bg_color, cv2.FILLED)
    
    # Overlay text
    cv2.putText(
        img,
        text,
        (int(x + 5), int(y - 3)),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )


def draw_hud_dashboard(
    img: np.ndarray,
    metrics: DetectionMetrics,
    model_name: str,
    frame_number: int,
    total_frames: int,
    detected_count: int,
    avg_inference_ms: float
) -> None:
    """Draws a professional, transparent HUD dashboard at the top of the video frame.
    
    Args:
        img: Frame to overlay HUD on.
        metrics: Active frame's latency and performance telemetry.
        model_name: Identifier weights name of active model.
        frame_number: Sequential frame pointer index.
        total_frames: Total video frame count.
        detected_count: Active count of vehicles detected in frame.
        avg_inference_ms: Cumulative average inference execution duration.
    """
    h, w, _ = img.shape
    
    # Establish transparent overlay region at top 60px
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (18, 18, 18), cv2.FILLED)
    
    # Blend overlay back to produce transparency
    alpha = 0.8
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Add a thin gray bounding divider line
    cv2.line(img, (0, 60), (w, 60), (50, 50, 50), 1)

    # Label positioning and layout setup
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_title = 0.55
    font_scale_main = 0.45
    color_white = (240, 240, 240)
    color_green = (90, 220, 90)
    color_orange = (235, 135, 25)
    color_gray = (160, 160, 160)
    
    # 1. Platform Info (Left)
    cv2.putText(img, "VELOX VISION", (15, 25), font, font_scale_title, color_orange, 2, cv2.LINE_AA)
    cv2.putText(img, "AI ENFORCEMENT RUNNER", (15, 45), font, 0.32, color_gray, 1, cv2.LINE_AA)

    # Split remaining layout width into columns
    col1_x = int(w * 0.25)
    col2_x = int(w * 0.44)
    col3_x = int(w * 0.63)
    col4_x = int(w * 0.80)

    # Column 1: Model Name
    cv2.putText(img, "ACTIVE MODEL", (col1_x, 20), font, 0.32, color_gray, 1, cv2.LINE_AA)
    model_short = model_name.split("/")[-1].split("\\")[-1]
    cv2.putText(img, model_short, (col1_x, 42), font, font_scale_main, color_white, 1, cv2.LINE_AA)

    # Column 2: Latency & Speed
    cv2.putText(img, "PERFORMANCE", (col2_x, 20), font, 0.32, color_gray, 1, cv2.LINE_AA)
    perf_text = f"{metrics.fps:.1f} FPS ({metrics.total_latency_ms:.1f} ms)"
    cv2.putText(img, perf_text, (col2_x, 42), font, font_scale_main, color_green, 1, cv2.LINE_AA)

    # Column 3: Frame Progress
    cv2.putText(img, "PROGRESS", (col3_x, 20), font, 0.32, color_gray, 1, cv2.LINE_AA)
    progress_text = f"{frame_number}"
    if total_frames > 0:
        progress_text += f" / {total_frames}"
    cv2.putText(img, progress_text, (col3_x, 42), font, font_scale_main, color_white, 1, cv2.LINE_AA)

    # Column 4: Detections Metrics
    cv2.putText(img, "METRICS", (col4_x, 20), font, 0.32, color_gray, 1, cv2.LINE_AA)
    metrics_text = f"Count: {detected_count} | Inf: {avg_inference_ms:.1f}ms"
    cv2.putText(img, metrics_text, (col4_x, 42), font, font_scale_main, color_orange, 1, cv2.LINE_AA)
