import cv2
import numpy as np
from typing import Dict
from src.core.entities import DetectionMetrics

# Centralized professional BGR color system
CLASS_COLORS = {
    "car": (220, 100, 30),         # Professional Slate Blue
    "truck": (180, 50, 180),       # Deep Purple
    "bus": (30, 120, 240),         # Vibrant Orange
    "motorcycle": (70, 200, 70)    # Soft Emerald Green
}
DEFAULT_COLOR = (120, 120, 120)    # Cool Slate Gray for unknown classes

# Friendly model name mappings
MODEL_NAME_MAP = {
    "yolo11n.pt": "YOLO11 Nano",
    "yolov8n.pt": "YOLOv8 Nano",
    "yolo11s.pt": "YOLO11 Small",
    "yolo11m.pt": "YOLO11 Medium",
    "yolov8s.pt": "YOLOv8 Small"
}


def get_resolution_scaling(h: int) -> tuple:
    """Computes adaptive line thickness and font scale based on frame height.
    
    This ensures that HUD elements and bounding boxes scale proportionally
    across 720p, 1080p, 1440p, and 4K resolutions.
    """
    scale = max(0.5, h / 1080.0)
    thickness = max(1, int(round(2 * scale)))
    font_scale = 0.40 * scale
    return thickness, font_scale, scale


def get_friendly_model_name(model_path: str) -> str:
    """Translates model weight filename into a polished presentation name."""
    filename = model_path.split("/")[-1].split("\\")[-1]
    return MODEL_NAME_MAP.get(filename.lower(), filename.split(".")[0].upper())


def draw_rounded_rectangle(
    img: np.ndarray,
    pt1: tuple,
    pt2: tuple,
    color: tuple,
    thickness: int = 2,
    r: int = 8
) -> None:
    """Draws a bounding box with rounded corners for a clean, modern HUD look."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Ensure correct coordinate ordering
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
    class_name: str,
    confidence: float,
    position: tuple,
    bg_color: tuple,
    thickness: int = 1,
    font_scale: float = 0.4,
    track_id: int = None
) -> None:
    """Draws a multi-line class label (Line 1: Class + ID, Line 2: Confidence) with a padded background."""
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    if track_id is not None:
        line1 = f"{class_name.upper()} #{track_id}"
    else:
        line1 = class_name.upper()
    line2 = f"{int(confidence * 100)}%"
    
    # Calculate sizes for both text lines
    (w1, h1), b1 = cv2.getTextSize(line1, font, font_scale, thickness)
    (w2, h2), b2 = cv2.getTextSize(line2, font, font_scale, thickness)
    
    max_w = max(w1, w2)
    line_spacing = int(4 * (font_scale / 0.4))
    total_h = h1 + h2 + line_spacing + 8
    
    # Define label coordinates
    rect_pt1 = (int(x), int(y - total_h))
    rect_pt2 = (int(x + max_w + 12), int(y))
    
    # Draw filled background
    cv2.rectangle(img, rect_pt1, rect_pt2, bg_color, cv2.FILLED)
    
    # Draw text lines
    text_color = (255, 255, 255)
    cv2.putText(
        img,
        line1,
        (int(x + 6), int(y - h2 - line_spacing - 4)),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )
    cv2.putText(
        img,
        line2,
        (int(x + 6), int(y - 4)),
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
    device_name: str,
    frame_number: int,
    total_frames: int,
    active_breakdown: Dict[str, int],
    avg_fps: float,
    avg_inference_ms: float,
    elapsed_sec: float,
    eta_sec: float,
    video_fps: float
) -> None:
    """Draws a professional, transparent HUD dashboard at the top of the video frame."""
    h, w, _ = img.shape
    
    # Height of HUD: 75px on standard 1080p, scales proportionally
    hud_h = int(75 * (h / 1080.0))
    if hud_h < 65:
        hud_h = 65

    # Establish transparent overlay region at top of the frame
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (18, 18, 18), cv2.FILLED)
    
    # Blend overlay back to produce transparency
    alpha = 0.82
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Add a thin gray bounding divider line
    cv2.line(img, (0, hud_h), (w, hud_h), (55, 55, 55), 1)

    # Resolution scaling for HUD text
    thickness, font_scale, scale_val = get_resolution_scaling(h)
    
    # Text sizing adjustments
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_title = 0.55 * scale_val
    font_scale_sub = 0.32 * scale_val
    font_scale_main = 0.45 * scale_val
    
    color_white = (242, 242, 242)
    color_green = (95, 220, 95)
    color_orange = (235, 135, 25)
    color_gray = (150, 150, 150)
    
    # 1. Platform Info (Left)
    cv2.putText(img, "VELOX VISION", (int(15 * scale_val), int(25 * scale_val)), font, font_scale_title, color_orange, max(1, int(round(2 * scale_val))), cv2.LINE_AA)
    cv2.putText(img, f"v0.2.0 | {w}x{h} @ {video_fps:.1f} FPS", (int(15 * scale_val), int(45 * scale_val)), font, font_scale_sub, color_gray, 1, cv2.LINE_AA)

    # Column positioning layout (fraction of width)
    col1_x = int(w * 0.20)
    col2_x = int(w * 0.40)
    col3_x = int(w * 0.61)
    col4_x = int(w * 0.77)

    # Column 1: Model & Hardware Device Info
    cv2.putText(img, "SYSTEM ENVIRONMENT", (col1_x, int(20 * scale_val)), font, font_scale_sub, color_gray, 1, cv2.LINE_AA)
    model_friendly = get_friendly_model_name(model_name)
    env_text = f"{model_friendly} ({device_name.upper()})"
    cv2.putText(img, env_text, (col1_x, int(42 * scale_val)), font, font_scale_main, color_white, 1, cv2.LINE_AA)

    # Column 2: Running Performance (Averages & Latencies)
    cv2.putText(img, "PERFORMANCE METRICS", (col2_x, int(20 * scale_val)), font, font_scale_sub, color_gray, 1, cv2.LINE_AA)
    perf_text = f"FPS: {metrics.fps:.1f} (Avg: {avg_fps:.1f}) | Latency: {metrics.total_latency_ms:.1f}ms"
    cv2.putText(img, perf_text, (col2_x, int(42 * scale_val)), font, font_scale_main, color_green, 1, cv2.LINE_AA)

    # Column 3: Processing Progress & ETA
    cv2.putText(img, "BENCHMARK PROGRESS", (col3_x, int(20 * scale_val)), font, font_scale_sub, color_gray, 1, cv2.LINE_AA)
    progress_text = f"{frame_number}"
    if total_frames > 0:
        progress_text += f" / {total_frames}"
    time_text = f"Elapsed: {elapsed_sec:.1f}s"
    if eta_sec >= 0:
        time_text += f" | ETA: {eta_sec:.1f}s"
    cv2.putText(img, f"{progress_text} ({time_text})", (col3_x, int(42 * scale_val)), font, font_scale_main, color_white, 1, cv2.LINE_AA)

    # Column 4: Live Vehicle breakdown stats
    cv2.putText(img, "VEHICLE DISTRIBUTION", (col4_x, int(20 * scale_val)), font, font_scale_sub, color_gray, 1, cv2.LINE_AA)
    breakdown_text = f"Car:{active_breakdown.get('car', 0)} Bus:{active_breakdown.get('bus', 0)} Truck:{active_breakdown.get('truck', 0)} Moto:{active_breakdown.get('motorcycle', 0)}"
    cv2.putText(img, breakdown_text, (col4_x, int(42 * scale_val)), font, font_scale_main, color_orange, 1, cv2.LINE_AA)

    # Visual progress bar (drawn as a colored line at the base of the HUD banner)
    if total_frames > 0:
        progress_ratio = frame_number / total_frames
        bar_w = int(w * progress_ratio)
        # Dynamic colored bar transition from orange to green
        bar_color = (
            int(25 + 70 * progress_ratio),
            int(135 + 85 * progress_ratio),
            int(235 - 140 * progress_ratio)
        )
        cv2.line(img, (0, hud_h - 1), (bar_w, hud_h - 1), bar_color, max(2, int(round(3 * scale_val))))
