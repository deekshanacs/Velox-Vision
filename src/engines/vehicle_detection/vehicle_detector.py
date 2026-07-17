import time
import logging
from typing import List, Optional
import numpy as np
from ultralytics import YOLO

from config.settings import settings
from src.core.interfaces import IVehicleDetector
from src.core.entities import BoundingBox, VehicleDetection, DetectionMetrics, DetectionResult
from src.engines.vehicle_detection.constants import TARGET_CLASS_IDS, CLASS_MAPPING
from src.engines.vehicle_detection.exceptions import ModelLoadError, InvalidFrameError, InferenceFailedError
from src.utils.metrics import PerformanceTracker

logger = logging.getLogger(__name__)

# Map class names to their COCO class IDs
COCO_VEHICLE_MAP = {
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7
}

class YOLOVehicleDetector(IVehicleDetector):
    """Production-grade YOLO vehicle detector implementation."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        device: Optional[str] = None,
        warmup: Optional[bool] = None,
        classes: Optional[List[str]] = None
    ):
        """Initializes the vehicle detector with configuration parameters.
        
        Args:
            model_path: Path to model weights file. Resolves from settings if None.
            confidence_threshold: Confidence boundary value. Resolves from settings if None.
            device: Computing hardware target ('auto', 'cpu', 'cuda', 'mps'). Resolves from settings if None.
            warmup: Enable/disable warm-up pass on load. Resolves from settings if None.
            classes: Target vehicle classes to filter. Resolves from settings if None.
        """
        # Resolve configurations
        self.model_path = model_path or settings.get("detection.model_path", "models/weights/yolo11n.pt")
        self.fallback_path = settings.get("detection.fallback_path", "models/weights/yolov8n.pt")
        self.confidence_threshold = confidence_threshold or settings.get("detection.confidence_threshold", 0.30)
        self.warmup_enabled = warmup if warmup is not None else settings.get("detection.warmup", True)
        
        # Configure target classes & IDs dynamically
        self.classes_to_detect = classes or settings.get("detection.classes", ["car", "motorcycle", "bus", "truck"])
        self.target_class_ids = [COCO_VEHICLE_MAP[c] for c in self.classes_to_detect if c in COCO_VEHICLE_MAP]
        if not self.target_class_ids:
            self.target_class_ids = TARGET_CLASS_IDS  # Default back to all vehicle types if none match

        # Manage device selection
        config_device = device or settings.get("detection.device", "auto")
        self.device = self._determine_device(config_device)

        # Log banner initialization
        self._log_initialization_banner()

        # Load weights and warmup
        self.model = None
        self._load_model()
        
        if self.warmup_enabled:
            self._warmup_model()

    def _determine_device(self, config_device: str) -> str:
        """Determines the hardware device target, catching security DLL errors."""
        device_lower = config_device.lower()
        if device_lower != "auto":
            return device_lower

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except Exception as e:
            # Fallback to cpu if torch library load fails due to system AppControl policy DLL locks
            logger.debug(f"Device auto-detect fallback to CPU due to import restrictions: {e}")
            return "cpu"

    def _log_initialization_banner(self) -> None:
        """Prints structured startup logs."""
        border = "=" * 45
        logger.info(f"\n{border}\nVelox Vision Detection Engine Initializing\n{border}")
        logger.info(f"Model Path:         {self.model_path}")
        logger.info(f"Target Device:      {self.device.upper()}")
        logger.info(f"Confidence Limit:   {self.confidence_threshold:.2f}")
        logger.info(f"Filtering Classes:  {', '.join(self.classes_to_detect)}")
        logger.info(f"Warm-Up Enabled:    {self.warmup_enabled}")
        logger.info(f"{border}")

    def _load_model(self) -> None:
        """Loads model weights, falling back to backup weights if primary fails."""
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"Successfully loaded primary model weights: '{self.model_path}'")
        except Exception as e:
            logger.warning(f"Could not load primary model '{self.model_path}' ({e}). Attempting fallback...")
            try:
                self.model = YOLO(self.fallback_path)
                self.model_path = self.fallback_path
                logger.info(f"Successfully loaded fallback model weights: '{self.fallback_path}'")
            except Exception as fe:
                err_msg = f"Fatal model loading failure for primary and fallback weights: {fe}"
                logger.critical(err_msg, exc_info=True)
                raise ModelLoadError(err_msg) from fe

    def _warmup_model(self) -> None:
        """Warms up the model graph via dummy frame inference."""
        try:
            logger.info("Performing warm-up forward pass on graph...")
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(
                source=dummy_frame,
                classes=self.target_class_ids,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False
            )
            logger.info("Warm-up complete. Detection engine ready.")
        except Exception as e:
            logger.warning(f"Engine warm-up pass failed: {e}. Proceeding to execution.")

    def detect(self, frame: np.ndarray, frame_number: int = 0) -> DetectionResult:
        """Executes vehicle detection over an input frame.
        
        Args:
            frame: Input image array in BGR format.
            frame_number: Current frame sequential counter.
            
        Returns:
            DetectionResult package container.
            
        Raises:
            InvalidFrameError: If frame shape is empty or invalid.
            InferenceFailedError: If execution throws error.
        """
        timestamp = time.time()
        
        # Guard clause check
        if frame is None or frame.size == 0:
            raise InvalidFrameError("Frame cannot be None or empty.")
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise InvalidFrameError(f"Unsupported image shape {frame.shape}. Frame must be a 3-channel BGR image.")

        start_time = time.perf_counter()
        
        try:
            # Predict
            results = self.model.predict(
                source=frame,
                classes=self.target_class_ids,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False
            )
        except Exception as e:
            err_msg = f"YOLO model execution failed: {e}"
            logger.error(err_msg, exc_info=True)
            raise InferenceFailedError(err_msg) from e

        end_time = time.perf_counter()
        total_latency_ms = (end_time - start_time) * 1000.0

        preprocess_time_ms = 0.0
        inference_time_ms = 0.0
        postprocess_time_ms = 0.0

        # Extract timing metrics directly from model speed profiling logs if available
        if results and hasattr(results[0], 'speed'):
            speed = results[0].speed
            preprocess_time_ms = speed.get('preprocess', 0.0)
            inference_time_ms = speed.get('inference', 0.0)
            postprocess_time_ms = speed.get('postprocess', 0.0)
        else:
            inference_time_ms = total_latency_ms

        fps = PerformanceTracker.calculate_fps(total_latency_ms)

        metrics = DetectionMetrics(
            preprocess_time_ms=preprocess_time_ms,
            inference_time_ms=inference_time_ms,
            postprocess_time_ms=postprocess_time_ms,
            total_latency_ms=total_latency_ms,
            fps=fps
        )

        detections: List[VehicleDetection] = []
        if results:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    # Get box coordinates (xyxy), confidence, and class
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = CLASS_MAPPING.get(class_id, "unknown")

                    bbox = BoundingBox(
                        x1=xyxy[0],
                        y1=xyxy[1],
                        x2=xyxy[2],
                        y2=xyxy[3]
                    )

                    detections.append(VehicleDetection(
                        bbox=bbox,
                        confidence=confidence,
                        class_name=class_name,
                        class_id=class_id
                    ))

        return DetectionResult(
            detections=detections,
            metrics=metrics,
            frame_number=frame_number,
            timestamp=timestamp,
            model_name=self.model_path
        )
