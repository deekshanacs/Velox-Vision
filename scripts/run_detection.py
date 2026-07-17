import os
import sys
import time
import glob
import logging
import cv2
import numpy as np
from typing import List, Optional

# Add project root to sys.path to resolve imports correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.settings import settings
from src.engines.vehicle_detection.detector_factory import VehicleDetectorFactory
from src.engines.vehicle_detection.exceptions import DetectionError
from src.core.entities import DetectionResult
from src.utils.visualization import draw_rounded_rectangle, draw_metadata_label, draw_hud_dashboard

# Configure logging style
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("run_detection")

# Class-specific colors for drawing annotations (BGR format)
CLASS_COLORS = {
    "car": (225, 140, 20),         # Azure / Bright Blue
    "motorcycle": (75, 215, 75),   # Lime Green
    "bus": (20, 110, 230),         # Warm Orange
    "truck": (210, 40, 210)        # Violet / Purple
}
DEFAULT_COLOR = (130, 130, 130)    # Gray for unidentified objects


class DetectionRunner:
    """Production-grade verification runner to benchmark and visualize vehicle detections."""

    def __init__(self):
        """Initializes configs, logs directories, and constructs the vehicle detector."""
        # Resolve configurations
        self.input_config = settings.get("runner.input_video", "data/videos/traffic.mp4")
        self.output_dir = settings.get("runner.output_dir", "outputs/processed_videos/")
        self.show_preview = settings.get("runner.show_preview", True)
        self.save_output = settings.get("runner.save_output", True)

        # Resolve video path with folder fallbacks if missing
        self.video_path = self._resolve_video_path(self.input_config)
        
        # Detector configuration parameters
        self.model_path = settings.get("detection.model_path", "models/weights/yolo11n.pt")
        self.confidence_threshold = settings.get("detection.confidence_threshold", 0.30)
        self.device = settings.get("detection.device", "auto")
        self.warmup = settings.get("detection.warmup", True)
        self.classes = settings.get("detection.classes", ["car", "motorcycle", "bus", "truck"])

        # Instantiate target detector engine
        logger.info("Initializing Vehicle Detection Engine via factory...")
        self.detector = VehicleDetectorFactory.create_detector(
            detector_type="yolo",
            model_path=self.model_path,
            confidence_threshold=self.confidence_threshold,
            device=self.device,
            warmup=self.warmup,
            classes=self.classes
        )
        logger.info("Detector engine initialized successfully.")

    def _resolve_video_path(self, requested_path: str) -> str:
        """Validates input path or searches default folders for video fallbacks."""
        if os.path.exists(requested_path):
            return requested_path

        # If not found, look for any video file inside data/videos
        search_dir = "data/videos"
        video_extensions = ["*.mp4", "*.avi", "*.mov"]
        found_videos = []
        
        if os.path.exists(search_dir):
            for ext in video_extensions:
                found_videos.extend(glob.glob(os.path.join(search_dir, ext)))
                
        if found_videos:
            fallback_video = found_videos[0]
            logger.warning(
                f"Requested video '{requested_path}' not found. "
                f"Auto-detected fallback video: '{fallback_video}'"
            )
            return fallback_video

        return requested_path  # Bubbles up path issue to standard OpenCV validator

    def _generate_output_path(self, input_path: str) -> str:
        """Generates processed output path based on base input video name."""
        os.makedirs(self.output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        ext = os.path.splitext(input_path)[1]
        if not ext:
            ext = ".mp4" # Fallback extension
        return os.path.join(self.output_dir, f"{base_name}_detected{ext}")

    def run(self) -> None:
        """Runs the main frame processing execution loop."""
        logger.info(f"Opening target video stream: '{self.video_path}'")
        cap = cv2.VideoCapture(self.video_path)
        
        # Verify capture source opens
        if not cap.isOpened():
            logger.error(f"Failed to open video file: '{self.video_path}'")
            return

        # Fetch video telemetry properties
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Guard against zero division/properties errors
        if vid_fps <= 0:
            vid_fps = 30.0

        logger.info(f"Video Loaded - Resolution: {vid_width}x{vid_height} | FPS: {vid_fps} | Frames: {total_frames}")

        # Setup Video Writer
        writer = None
        output_path = "N/A"
        if self.save_output:
            output_path = self._generate_output_path(self.video_path)
            logger.info(f"Annotated video will be saved to: '{output_path}'")
            # Use standard MP4V codec for portability
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, vid_fps, (vid_width, vid_height))

        # Runtime telemetry variables
        processed_frames = 0
        total_vehicles_seen = 0
        sum_inference_ms = 0.0
        start_runtime = time.perf_counter()

        logger.info("Starting frame processing loop. Press 'q' or 'Ctrl+C' to exit.")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logger.info("Reached end of video stream or unreadable frame.")
                    break

                # Execute detection
                try:
                    result: DetectionResult = self.detector.detect(frame, frame_number=processed_frames)
                except DetectionError as de:
                    logger.error(f"Detector error at frame {processed_frames}: {de}")
                    continue

                processed_frames += 1
                total_vehicles_seen += len(result.detections)
                sum_inference_ms += result.metrics.inference_time_ms

                # Render annotations
                avg_inf_time = sum_inference_ms / processed_frames
                self._render_frame_annotations(frame, result, total_frames, avg_inf_time)

                # Save Frame
                if writer is not None:
                    writer.write(frame)

                # Show Preview Window
                if self.show_preview:
                    window_name = "Velox Vision - Detection Preview"
                    cv2.imshow(window_name, frame)
                    # Check key press for interrupt ('q')
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Execution interrupted by user key press ('q').")
                        break

                # Log progress periodic checkpoints
                if processed_frames % 50 == 0:
                    percent = (processed_frames / total_frames * 100) if total_frames > 0 else 0
                    logger.info(
                        f"Processed {processed_frames}/{total_frames} frames "
                        f"({percent:.1f}%) | Active Detections: {len(result.detections)}"
                    )

        except KeyboardInterrupt:
            logger.info("Execution interrupted by keyboard command (Ctrl+C).")
        finally:
            # Release resources
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
            logger.info("Video streams and OpenCV window resources successfully released.")

        # Compute summary metrics
        total_runtime = time.perf_counter() - start_runtime
        average_fps = processed_frames / total_runtime if total_runtime > 0 else 0.0
        avg_inf_ms = sum_inference_ms / processed_frames if processed_frames > 0 else 0.0

        # Print final report
        self._print_performance_report(
            input_video=self.video_path,
            output_video=output_path,
            resolution=f"{vid_width}x{vid_height}",
            fps=vid_fps,
            frames_processed=processed_frames,
            vehicles_detected=total_vehicles_seen,
            avg_fps=average_fps,
            avg_inf_time=avg_inf_ms,
            total_runtime=total_runtime,
            model_name=self.model_path
        )

    def _render_frame_annotations(
        self,
        frame: np.ndarray,
        result: DetectionResult,
        total_frames: int,
        avg_inference_ms: float
    ) -> None:
        """Renders bounding boxes, labels, and the dashboard HUD on the active frame."""
        # 1. Draw vehicle bounding boxes and class metadata labels
        for det in result.detections:
            color = CLASS_COLORS.get(det.class_name, DEFAULT_COLOR)
            
            # Extract box corners
            x1, y1, x2, y2 = det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            
            # Draw rounded box
            draw_rounded_rectangle(frame, pt1, pt2, color, thickness=2, r=10)
            
            # Create label string: e.g. "CAR 89%"
            label_text = f"{det.class_name.upper()} {int(det.confidence * 100)}%"
            label_pos = (pt1[0], pt1[1] - 5 if pt1[1] > 20 else pt1[1] + 15)
            
            # Draw metadata box label
            draw_metadata_label(frame, label_text, label_pos, bg_color=color)

        # 2. Draw transparency HUD Dashboard at top of the frame
        draw_hud_dashboard(
            img=frame,
            metrics=result.metrics,
            model_name=result.model_name,
            frame_number=result.frame_number,
            total_frames=total_frames,
            detected_count=len(result.detections),
            avg_inference_ms=avg_inference_ms
        )

    def _print_performance_report(
        self,
        input_video: str,
        output_video: str,
        resolution: str,
        fps: float,
        frames_processed: int,
        vehicles_detected: int,
        avg_fps: float,
        avg_inf_time: float,
        total_runtime: float,
        model_name: str
    ) -> None:
        """Prints a structured execution summary to stdout."""
        border = "=" * 45
        print(f"\n{border}")
        print("        Velox Vision Detection Report")
        print(border)
        print(f"Input Video:            {input_video}")
        print(f"Output Video:           {output_video}")
        print(f"Resolution:             {resolution}")
        print(f"Video Frame Rate (FPS): {fps:.1f}")
        print(f"Frames Processed:       {frames_processed}")
        print(f"Total Detections Logged:{vehicles_detected}")
        print(f"Average Execution FPS:  {avg_fps:.1f}")
        print(f"Average Inference Time: {avg_inf_time:.1f} ms")
        print(f"Total Runtime:          {total_runtime:.2f} seconds")
        print(f"Model File:             {model_name}")
        print(f"{border}\n")


if __name__ == "__main__":
    runner = DetectionRunner()
    runner.run()
