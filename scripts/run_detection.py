import os
import sys
import time
import glob
import uuid
import json
import logging
import cv2
import numpy as np
from typing import Dict, List, Optional

# Add project root to sys.path to resolve imports correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.settings import settings
from src.engines.vehicle_detection.detector_factory import VehicleDetectorFactory
from src.engines.vehicle_detection.exceptions import DetectionError
from src.core.entities import DetectionResult, BoundingBox, VehicleDetection
from src.engines.tracking.factories.tracker_factory import TrackerFactory
from src.engines.tracking.configuration.tracking_config import TrackingConfiguration
from src.engines.tracking.entities.tracking_context import TrackingContext
from src.engines.tracking.value_objects.frame_metadata import FrameMetadata
from src.engines.tracking.entities.tracked_vehicle import TrackedVehicle
from src.engines.tracking.entities.track_state import TrackState
from src.engines.tracking import (
    TrackingRenderer,
    TrackingDashboard,
    TrackingProfiler,
    TrackingReportGenerator
)
from src.utils.visualization import (
    draw_rounded_rectangle,
    draw_metadata_label,
    draw_hud_dashboard,
    get_friendly_model_name,
    get_resolution_scaling,
    CLASS_COLORS,
    DEFAULT_COLOR
)

# Configure logging style
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("run_detection")


class DetectionRunner:
    """Polished production-grade verification runner to benchmark vehicle detections (v0.2.0)."""

    def __init__(self):
        """Initializes configs, logs directories, and constructs the vehicle detector."""
        # Resolve configurations
        self.input_config = settings.get("runner.input_video", "data/videos/traffic.mp4")
        self.output_dir = settings.get("runner.output_dir", "outputs/processed_videos/")
        self.show_preview = settings.get("runner.show_preview", True)
        self.save_output = settings.get("runner.save_output", True)
        
        # New Phase 2 Polish Configurations
        self.frame_skip = int(settings.get("runner.frame_skip", 1))
        self.benchmark_mode = settings.get("runner.benchmark", False)
        self.min_height_ratio = float(settings.get("filtering.min_height_ratio", 0.005))
        self.reports_dir = settings.get("reports.output_dir", "outputs/reports/")
        self.save_summary = settings.get("reports.save_summary", True)
        self.snapshots_dir = settings.get("snapshots.output_dir", "outputs/snapshots/")

        # Resolve video path with folder fallbacks if missing
        self.video_path = self._resolve_video_path(self.input_config)
        
        # Detector configuration parameters
        self.model_path = settings.get("detection.model_path", "models/weights/yolo11n.pt")
        self.confidence_threshold = float(settings.get("detection.confidence_threshold", 0.50))
        self.iou_threshold = float(settings.get("detection.iou_threshold", 0.45))
        self.device = settings.get("detection.device", "auto")
        self.warmup = settings.get("detection.warmup", True)
        self.classes = settings.get("detection.classes", ["car", "motorcycle", "bus", "truck"])

        # Instantiate target detector engine and profile warm-up duration
        logger.info("Initializing Vehicle Detection Engine via factory...")
        start_warmup = time.perf_counter()
        
        self.detector = VehicleDetectorFactory.create_detector(
            detector_type="yolo",
            model_path=self.model_path,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            device=self.device,
            warmup=self.warmup,
            classes=self.classes
        )
        
        self.warmup_time = time.perf_counter() - start_warmup
        logger.info(f"Detector engine initialized. Warmup duration: {self.warmup_time:.2f} seconds.")

        # Tracker initialization parameters
        self.tracking_enabled = settings.get("tracking.enabled", True)
        self.tracker = None
        self.tracking_renderer = None
        self.tracking_dashboard = None
        self.tracking_profiler = None
        self.tracking_report_gen = None
        self.selected_track_id = None
        self._prev_track_states = {}
        self._active_tracked_vehicles_for_click = []

        if self.tracking_enabled:
            logger.info("Initializing Object Tracking Engine via factory...")
            self.tracking_cfg = TrackingConfiguration.from_settings()
            self.tracker = TrackerFactory.create_tracker(
                tracker_type=self.tracking_cfg.tracker_type,
                configs=self.tracking_cfg
            )
            logger.info(f"Tracking engine '{self.tracking_cfg.tracker_type}' initialized successfully.")
            self.tracking_renderer = TrackingRenderer(self.tracking_cfg)
            self.tracking_dashboard = TrackingDashboard()
            self.tracking_profiler = TrackingProfiler()
            self.tracking_report_gen = TrackingReportGenerator(self.reports_dir)

        # Centralize metrics storage
        self.peak_fps = 0.0
        self.confidence_totals = {"high_gt_90": 0, "medium_70_90": 0, "low_50_70": 0}
        self.vehicles_breakdown = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0}
        self.class_conf_sums = {"car": 0.0, "bus": 0.0, "truck": 0.0, "motorcycle": 0.0}
        
        # Friendly device mapping
        self.device_presentation = self.detector.device.upper() if hasattr(self.detector, 'device') else "CPU"

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

        return requested_path

    def _generate_output_path(self, input_path: str) -> str:
        """Generates processed output path based on base input video name."""
        os.makedirs(self.output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        ext = os.path.splitext(input_path)[1]
        if not ext:
            ext = ".mp4"
        return os.path.join(self.output_dir, f"{base_name}_detected{ext}")

    def _get_git_info(self) -> Dict[str, str]:
        """Retrieves active Git commit hash and branch name dynamically."""
        import subprocess
        git_info = {"commit": "N/A", "branch": "N/A"}
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            git_info["commit"] = commit
            git_info["branch"] = branch
        except Exception:
            pass
        return git_info

    def _get_hardware_info(self) -> Dict[str, str]:
        """Queries CPU specs, memory capacity, and logical cores via system shell."""
        import platform
        import os
        import subprocess
        
        hardware = {
            "processor": platform.processor() or "Unknown Processor",
            "threads": str(os.cpu_count() or 1),
            "ram": "Unknown RAM"
        }
        
        if platform.system() == "Windows":
            try:
                # CPU Name query
                cpu_out = subprocess.check_output(
                    "wmic cpu get name",
                    shell=True,
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                cpu_lines = [l.strip() for l in cpu_out.split('\n') if l.strip()]
                if len(cpu_lines) > 1:
                    hardware["processor"] = cpu_lines[1]
                    
                # RAM Capacity query
                ram_out = subprocess.check_output(
                    "wmic computersystem get totalphysicalmemory",
                    shell=True,
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                ram_lines = [l.strip() for l in ram_out.split('\n') if l.strip()]
                if len(ram_lines) > 1:
                    ram_bytes = int(ram_lines[1])
                    hardware["ram"] = f"{ram_bytes / (1024**3):.1f} GB"
            except Exception:
                pass
        return hardware

    def _get_friendly_resolution(self, w: int, h: int) -> str:
        """Returns standard resolution strings mapped to their display tags."""
        res_map = {
            (1280, 720): "HD",
            (1920, 1080): "Full HD",
            (2560, 1440): "QHD",
            (3840, 2160): "4K UHD"
        }
        friendly = res_map.get((w, h), "")
        if friendly:
            return f"{w} \u00d7 {h} ({friendly})"
        return f"{w} \u00d7 {h}"

    def run(self) -> None:
        """Runs the main frame processing execution loop."""
        logger.info(f"Opening target video stream: '{self.video_path}'")
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video file: '{self.video_path}'")
            return

        # Fetch video telemetry properties
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if vid_fps <= 0:
            vid_fps = 30.0

        resolution_str = self._get_friendly_resolution(vid_width, vid_height)
        logger.info(f"Video Loaded - Resolution: {resolution_str} | FPS: {vid_fps} | Frames: {total_frames}")

        # Setup Video Writer
        writer = None
        output_path = "N/A"
        if self.save_output:
            output_path = self._generate_output_path(self.video_path)
            logger.info(f"Annotated video will be saved to: '{output_path}'")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, vid_fps / self.frame_skip, (vid_width, vid_height))

        # Register Mouse Callbacks for Memory Inspector
        if self.show_preview and not self.benchmark_mode:
            try:
                cv2.namedWindow("Velox Vision - Detection Preview")
                def on_mouse(event, x, y, flags, param):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        clicked_id = None
                        active_vehicles = getattr(self, "_active_tracked_vehicles_for_click", [])
                        for vehicle in active_vehicles:
                            bbox = vehicle.bbox
                            if bbox and bbox.x1 <= x <= bbox.x2 and bbox.y1 <= y <= bbox.y2:
                                clicked_id = vehicle.track_id
                                break
                        
                        if clicked_id is not None and clicked_id == self.selected_track_id:
                            self.selected_track_id = None
                            logger.info("Interactive Inspector: Cleared selection")
                        elif clicked_id is not None:
                            self.selected_track_id = clicked_id
                            logger.info(f"Interactive Inspector: Selected track ID {clicked_id}")
                        else:
                            self.selected_track_id = None
                            logger.info("Interactive Inspector: Cleared selection")

                cv2.setMouseCallback("Velox Vision - Detection Preview", on_mouse)
            except Exception as e:
                logger.warning(f"Could not bind mouse callback (running headless/no GUI): {e}")

        # Runtime telemetry variables
        processed_frames = 0
        total_detections_count = 0
        sum_inference_ms = 0.0
        start_runtime = time.perf_counter()
        
        last_frame = None

        logger.info("Starting frame processing loop. Press 'q' or 'Ctrl+C' to exit.")

        try:
            while cap.isOpened():
                frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                ret, frame = cap.read()
                if not ret:
                    break

                # Frame skipping logic
                if frame_id % self.frame_skip != 0:
                    continue

                # Execute detection
                det_start = self.tracking_profiler.start_segment("detection") if self.tracking_profiler else 0
                try:
                    result: DetectionResult = self.detector.detect(frame, frame_number=processed_frames)
                except DetectionError as de:
                    logger.error(f"Detector error at frame {processed_frames}: {de}")
                    continue
                if self.tracking_profiler:
                    self.tracking_profiler.end_segment("detection", det_start)

                processed_frames += 1

                # Filter detections based on height ratio and track statistics
                filtered_detections: List[VehicleDetection] = []
                active_breakdown = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0}
                
                for det in result.detections:
                    h_px = det.bbox.height
                    # Min height check: scale dynamically based on frame size
                    if h_px < self.min_height_ratio * vid_height:
                        continue
                        
                    filtered_detections.append(det)
                    
                    # Accumulate class breakdowns
                    c_name = det.class_name.lower()
                    if c_name in active_breakdown:
                        active_breakdown[c_name] += 1
                        self.vehicles_breakdown[c_name] += 1
                        self.class_conf_sums[c_name] += det.confidence

                    # Accumulate confidence distributions
                    conf = det.confidence
                    if conf > 0.90:
                        self.confidence_totals["high_gt_90"] += 1
                    elif conf >= 0.70:
                        self.confidence_totals["medium_70_90"] += 1
                    else:
                        self.confidence_totals["low_50_70"] += 1

                total_detections_count += len(filtered_detections)
                sum_inference_ms += result.metrics.inference_time_ms

                # Peak FPS calculations
                if result.metrics.fps > self.peak_fps:
                    self.peak_fps = result.metrics.fps

                # Track detections
                tracked_vehicles = None
                if self.tracker is not None:
                    temp_det_result = DetectionResult(
                        detections=filtered_detections,
                        metrics=result.metrics,
                        frame_number=processed_frames * self.frame_skip,
                        timestamp=processed_frames * self.frame_skip / vid_fps,
                        model_name=self.model_path
                    )
                    metadata = FrameMetadata(
                        frame_number=processed_frames * self.frame_skip,
                        timestamp=processed_frames * self.frame_skip / vid_fps,
                        width=vid_width,
                        height=vid_height,
                        fps=vid_fps
                    )
                    context = TrackingContext(
                        frame=frame,
                        detections=temp_det_result,
                        metadata=metadata
                    )
                    track_start = self.tracking_profiler.start_segment("tracking") if self.tracking_profiler else 0
                    try:
                        tracking_res = self.tracker.track(context)
                        tracked_vehicles = tracking_res.tracked_vehicles
                    except Exception as te:
                        logger.error(f"Tracking error at frame {processed_frames}: {te}")
                    if self.tracking_profiler:
                        self.tracking_profiler.end_segment("tracking", track_start)

                # Render annotations
                avg_inf_time = sum_inference_ms / processed_frames
                elapsed_sec = time.perf_counter() - start_runtime
                avg_fps = processed_frames / elapsed_sec if elapsed_sec > 0 else 0.0
                
                # Calculate ETA based on average processing rate
                eta_sec = -1.0
                if avg_fps > 0 and total_frames > 0:
                    remaining_frames = (total_frames / self.frame_skip) - processed_frames
                    eta_sec = max(0.0, remaining_frames / avg_fps)

                # Set current tracked vehicles for mouse listener
                self._active_tracked_vehicles_for_click = tracked_vehicles if tracked_vehicles else []

                # Overlay drawings on frame
                if self.tracking_renderer is not None:
                    # Compile debug events
                    current_states = {}
                    for vehicle in self._active_tracked_vehicles_for_click:
                        tid = vehicle.track_id
                        state = vehicle.state
                        current_states[tid] = state
                        
                        if tid not in self._prev_track_states:
                            self.tracking_renderer.log_debug_event(f"Track #{tid} ({vehicle.class_name}) created as {state.name}")
                        else:
                            prev_state = self._prev_track_states[tid]
                            if prev_state != state:
                                self.tracking_renderer.log_debug_event(f"Track #{tid} transitioned {prev_state.name} -> {state.name}")
                    
                    for tid in self._prev_track_states:
                        if tid not in current_states:
                            self.tracking_renderer.log_debug_event(f"Track #{tid} removed")
                    self._prev_track_states = current_states

                    # Update class breakdown
                    self.tracking_dashboard.update_class_breakdown(self._active_tracked_vehicles_for_click)
                    
                    # Compile telemetry dictionary
                    telemetry_dict = self.tracking_dashboard.get_telemetry_dict(
                        overall_fps=avg_fps,
                        det_fps=1000.0 / result.metrics.inference_time_ms if result.metrics.inference_time_ms > 0 else 0.0,
                        perf=self.tracker.get_performance() if self.tracker else None,
                        stats=self.tracker.get_statistics() if self.tracker else None,
                        memory_mgr=self.tracker._memory_manager if self.tracker and hasattr(self.tracker, "_memory_manager") else None
                    )
                    telemetry_dict["det_latency_ms"] = result.metrics.inference_time_ms
                    telemetry_dict["track_latency_ms"] = self.tracking_profiler.latencies.get("tracking", 0.0) if self.tracking_profiler else 0.0

                    # Render using tracking renderer (profiled)
                    render_start = self.tracking_profiler.start_segment("visualization") if self.tracking_profiler else 0
                    self.tracking_renderer.render(
                        frame=frame,
                        tracked_vehicles=self._active_tracked_vehicles_for_click,
                        telemetry_data=telemetry_dict,
                        current_frame=processed_frames * self.frame_skip,
                        total_frames=total_frames,
                        selected_track_id=self.selected_track_id
                    )
                    if self.tracking_profiler:
                        self.tracking_profiler.end_segment("visualization", render_start)
                else:
                    self._render_frame_annotations(
                        frame=frame,
                        detections=filtered_detections,
                        metrics=result.metrics,
                        frame_number=processed_frames * self.frame_skip,
                        total_frames=total_frames,
                        active_breakdown=active_breakdown,
                        avg_fps=avg_fps,
                        avg_inf_time=avg_inf_time,
                        elapsed_sec=elapsed_sec,
                        eta_sec=eta_sec,
                        video_fps=vid_fps,
                        tracked_vehicles=tracked_vehicles
                    )

                last_frame = frame.copy()

                # Save Frame
                if writer is not None:
                    writer.write(frame)

                # Show Preview Window (skipped if benchmark_mode is enabled)
                if self.show_preview and not self.benchmark_mode:
                    try:
                        cv2.imshow("Velox Vision - Detection Preview", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            logger.info("Execution interrupted by user key press ('q').")
                            break
                    except Exception as e:
                        logger.warning(f"Could not display preview window (running headless/no GUI): {e}")
                        self.show_preview = False  # Disable for remainder of run

                # Progress checkpoints logging
                if processed_frames % 50 == 0:
                    percent = (processed_frames * self.frame_skip / total_frames * 100) if total_frames > 0 else 0
                    logger.info(
                        f"Processed {processed_frames}/{int(total_frames/self.frame_skip)} frames "
                        f"({percent:.1f}%) | Active Detections: {len(filtered_detections)}"
                    )

        except KeyboardInterrupt:
            logger.info("Execution interrupted by keyboard command (Ctrl+C).")
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
            logger.info("Video streams and OpenCV window resources successfully released.")

        # Compute summary metrics
        total_runtime = time.perf_counter() - start_runtime
        average_fps = processed_frames / total_runtime if total_runtime > 0 else 0.0
        avg_inf_ms = sum_inference_ms / processed_frames if processed_frames > 0 else 0.0
        
        # Save snapshot of last frame
        if last_frame is not None:
            os.makedirs(self.snapshots_dir, exist_ok=True)
            snapshot_path = os.path.join(self.snapshots_dir, "final_frame.png")
            cv2.imwrite(snapshot_path, last_frame)
            logger.info(f"Final preview frame snapshot exported to: '{snapshot_path}'")
            if self.tracking_enabled and self.tracking_report_gen is not None:
                self.tracking_report_gen.save_final_snapshot(last_frame, self.snapshots_dir)

        # Generate summary report exports
        self._generate_execution_reports(
            vid_fps=vid_fps,
            resolution_str=resolution_str,
            frames_processed=processed_frames,
            vehicles_detected=total_detections_count,
            avg_fps=average_fps,
            avg_inf_time=avg_inf_ms,
            total_runtime=total_runtime
        )

    def _render_frame_annotations(
        self,
        frame: np.ndarray,
        detections: List[VehicleDetection],
        metrics,
        frame_number: int,
        total_frames: int,
        active_breakdown: Dict[str, int],
        avg_fps: float,
        avg_inf_time: float,
        elapsed_sec: float,
        eta_sec: float,
        video_fps: float,
        tracked_vehicles: List[TrackedVehicle] = None
    ) -> None:
        """Renders bounding boxes, labels, and the dashboard HUD on the active frame."""
        h, w, _ = frame.shape
        thickness, font_scale, _ = get_resolution_scaling(h)

        # 1. Draw vehicle bounding boxes and class metadata labels
        if tracked_vehicles is not None and self.tracking_enabled:
            for vehicle in tracked_vehicles:
                if vehicle.state == TrackState.TEMPORARILY_LOST or vehicle.state == TrackState.REMOVED:
                    continue
                color = CLASS_COLORS.get(vehicle.class_name, DEFAULT_COLOR)
                
                x1, y1, x2, y2 = vehicle.bbox.x1, vehicle.bbox.y1, vehicle.bbox.x2, vehicle.bbox.y2
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))
                
                # Draw rounded box
                draw_rounded_rectangle(frame, pt1, pt2, color, thickness=thickness, r=8)
                
                # Position label cleanly on bounding box top border
                label_pos = (pt1[0], pt1[1] - 5 if pt1[1] > 25 else pt1[1] + 15)
                
                # Draw metadata box label with track_id
                draw_metadata_label(
                    frame,
                    class_name=vehicle.class_name,
                    confidence=vehicle.confidence,
                    position=label_pos,
                    bg_color=color,
                    thickness=max(1, thickness - 1),
                    font_scale=font_scale,
                    track_id=vehicle.track_id
                )
        else:
            for det in detections:
                color = CLASS_COLORS.get(det.class_name, DEFAULT_COLOR)
                
                x1, y1, x2, y2 = det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))
                
                # Draw rounded box
                draw_rounded_rectangle(frame, pt1, pt2, color, thickness=thickness, r=8)
                
                # Position label cleanly on bounding box top border
                label_pos = (pt1[0], pt1[1] - 5 if pt1[1] > 25 else pt1[1] + 15)
                
                # Draw metadata box label
                draw_metadata_label(
                    frame,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    position=label_pos,
                    bg_color=color,
                    thickness=max(1, thickness - 1),
                    font_scale=font_scale
                )

        # 2. Draw transparency HUD Dashboard at top of the frame
        draw_hud_dashboard(
            img=frame,
            metrics=metrics,
            model_name=self.model_path,
            device_name=self.device_presentation,
            frame_number=frame_number,
            total_frames=total_frames,
            active_breakdown=active_breakdown,
            avg_fps=avg_fps,
            avg_inference_ms=avg_inf_time,
            elapsed_sec=elapsed_sec,
            eta_sec=eta_sec,
            video_fps=video_fps
        )

    def _generate_execution_reports(
        self,
        vid_fps: float,
        resolution_str: str,
        frames_processed: int,
        vehicles_detected: int,
        avg_fps: float,
        avg_inf_time: float,
        total_runtime: float
    ) -> None:
        """Computes summary files, logs JSON reports, and prints final benchmark stdout."""
        # Calculate class-wise confidence averages
        class_conf_averages = {}
        for c, s_val in self.class_conf_sums.items():
            cnt = self.vehicles_breakdown[c]
            class_conf_averages[c] = f"{int((s_val / cnt) * 100)}%" if cnt > 0 else "0%"

        total_breakdown_count = sum(self.vehicles_breakdown.values())
        overall_avg_conf = (
            int((sum(self.class_conf_sums.values()) / total_breakdown_count) * 100)
            if total_breakdown_count > 0 else 0
        )

        avg_vehicles_per_frame = vehicles_detected / frames_processed if frames_processed > 0 else 0.0

        # Retrieve Git metadata
        git_info = self._get_git_info()
        
        # Retrieve Hardware specifications
        hw_info = self._get_hardware_info()

        # Build execution report dict
        report_id = uuid.uuid4().hex[:8]
        timestamp_str = time.strftime("%Y-%m-%d_%H-%M-%S")
        
        report_data = {
            "benchmark_id": report_id,
            "timestamp": timestamp_str,
            "git": git_info,
            "hardware": hw_info,
            "settings": {
                "confidence_threshold": self.confidence_threshold,
                "iou_threshold": self.iou_threshold,
                "frame_skip": self.frame_skip,
                "min_height_ratio": self.min_height_ratio
            },
            "metrics": {
                "runtime_seconds": round(total_runtime, 2),
                "avg_fps": round(avg_fps, 1),
                "peak_fps": round(self.peak_fps, 1),
                "avg_latency_ms": round(avg_inf_time, 1),
                "warmup_seconds": round(self.warmup_time, 2)
            },
            "vehicles": self.vehicles_breakdown,
            "avg_vehicles_per_frame": round(avg_vehicles_per_frame, 1),
            "confidence_distribution": self.confidence_totals,
            "average_confidence": f"{overall_avg_conf}%",
            "confidence_by_class": class_conf_averages
        }

        # Save summary report to timestamped JSON file
        report_path = "N/A"
        if self.save_summary:
            os.makedirs(self.reports_dir, exist_ok=True)
            report_path = os.path.join(self.reports_dir, f"{timestamp_str}_summary.json")
            try:
                with open(report_path, "w") as f:
                    json.dump(report_data, f, indent=4)
                logger.info(f"Execution summary report successfully saved to: '{report_path}'")
            except Exception as e:
                logger.error(f"Failed to write execution report file: {e}")

        # Save tracking visualization JSON report
        if self.tracking_enabled and self.tracking_report_gen is not None:
            perf = self.tracker.get_performance() if self.tracker else None
            stats = self.tracker.get_statistics() if self.tracker else None
            memory_mgr = self.tracker._memory_manager if self.tracker and hasattr(self.tracker, "_memory_manager") else None
            
            perf_stats = {
                "total_latency_ms": perf.total_latency_ms if perf else 0.0,
                "avg_latency_ms": perf.average_latency_ms if perf else 0.0,
                "tracking_fps": perf.tracking_fps if perf else 0.0,
                "pipeline_fps": avg_fps,
                "processed_frames": frames_processed
            }
            track_stats = {
                "tracks_created": stats.tracks_created if stats else 0,
                "tracks_removed": stats.tracks_removed if stats else 0,
                "tracks_lost": stats.tracks_lost if stats else 0,
                "recovered_tracks": stats.recovered_tracks if stats else 0,
                "average_track_age": stats.average_track_age if stats else 0.0
            }
            mem_stats = {
                "total_memories": 0,
                "average_lifetime_frames": 0.0,
                "estimated_memory_bytes": 0
            }
            if memory_mgr:
                m_stats = memory_mgr.get_statistics()
                mem_stats = {
                    "total_memories": m_stats.total_memories,
                    "average_lifetime_frames": m_stats.average_lifetime_frames,
                    "estimated_memory_bytes": m_stats.estimated_memory_bytes
                }
                
            self.tracking_report_gen.generate_json_report(
                report_id=report_id,
                configs=self.tracking_cfg,
                perf_stats=perf_stats,
                track_stats=track_stats,
                memory_stats=mem_stats,
                git_info=git_info,
                hardware_info=hw_info
            )

        # Print structured final benchmark table to stdout
        model_friendly = get_friendly_model_name(self.model_path)
        border = "=" * 41
        print(f"\n{border}")
        print("        VELOX VISION v0.2.0")
        print(border)
        print(f"Model                    {model_friendly}")
        print(f"Device                   {self.device_presentation}")
        print(f"Resolution               {resolution_str}")
        print(f"Frames                   {frames_processed}")
        print(f"Runtime                  {total_runtime:.1f} s")
        print(f"Average FPS              {avg_fps:.1f}")
        print(f"Peak FPS                 {self.peak_fps:.1f}")
        print(f"Average Latency          {avg_inf_time:.1f} ms")
        print(f"Warmup                   {self.warmup_time:.2f} s")
        print(f"\nVehicles")
        print(f"  Cars                   {self.vehicles_breakdown.get('car', 0)}")
        print(f"  Bus                    {self.vehicles_breakdown.get('bus', 0)}")
        print(f"  Truck                  {self.vehicles_breakdown.get('truck', 0)}")
        print(f"  Motorcycle             {self.vehicles_breakdown.get('motorcycle', 0)}")
        print(f"\nAverage Vehicles/Frame   {avg_vehicles_per_frame:.1f}")
        print(f"\nConfidence")
        print(f"  Cars                   {class_conf_averages.get('car', '0%')}")
        print(f"  Bus                    {class_conf_averages.get('bus', '0%')}")
        print(f"  Truck                  {class_conf_averages.get('truck', '0%')}")
        print(f"  Motorcycle             {class_conf_averages.get('motorcycle', '0%')}")
        print(f"\nReport                   {report_path}")
        print(border)


if __name__ == "__main__":
    runner = DetectionRunner()
    runner.run()
