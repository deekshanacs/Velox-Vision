import time
import logging
import numpy as np
from typing import List

from src.core.interfaces import IVehicleDetector
from src.core.entities import BoundingBox, VehicleDetection, DetectionMetrics, DetectionResult
from src.engines.vehicle_detection.exceptions import InvalidFrameError

logger = logging.getLogger(__name__)

class SimulatedVehicleDetector(IVehicleDetector):
    """Simulates moving vehicles on input video streams for sandbox validation and testing."""

    def __init__(self, confidence_threshold: float = 0.25):
        """Initializes the simulated engine."""
        self.confidence_threshold = confidence_threshold
        logger.info("Initializing SimulatedVehicleDetector fallback...")
        # Brief sleep to simulate warm-up initialization time
        time.sleep(0.1)
        logger.info("SimulatedVehicleDetector fallback engine ready.")

    def detect(self, frame: np.ndarray, frame_number: int = 0) -> DetectionResult:
        """Simulates bounding box detections on OpenCV frames.
        
        Args:
            frame: BGR OpenCV frame.
            frame_number: Sequential frame count index.
            
        Returns:
            DetectionResult package containing moving coordinates and mocked latency metrics.
        """
        timestamp = time.time()
        
        # Frame check guard clauses
        if frame is None or frame.size == 0:
            raise InvalidFrameError("Frame cannot be None or empty.")
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise InvalidFrameError(f"Unsupported image shape {frame.shape}.")
            
        h, w, _ = frame.shape
        start_time = time.perf_counter()
        
        # Mock processing processing latency (approx 6-12ms)
        time.sleep(0.008)
        
        end_time = time.perf_counter()
        total_latency_ms = (end_time - start_time) * 1000.0
        
        detections: List[VehicleDetection] = []
        
        # Simulate moving vehicle coordinates relative to frame resolution
        # 1. Car moving down center-left lane
        car_y = int((150 + frame_number * 3.5) % (h - 100))
        car_x = int(w * 0.35)
        if car_y + 80 < h:
            bbox = BoundingBox(x1=car_x, y1=car_y, x2=car_x + 95, y2=car_y + 75)
            detections.append(VehicleDetection(
                bbox=bbox,
                confidence=0.91,
                class_name="car",
                class_id=2
            ))
            
        # 2. Truck moving down center-right lane
        truck_y = int((50 + frame_number * 2.2) % (h - 150))
        truck_x = int(w * 0.58)
        if truck_y + 120 < h:
            bbox = BoundingBox(x1=truck_x, y1=truck_y, x2=truck_x + 125, y2=truck_y + 115)
            detections.append(VehicleDetection(
                bbox=bbox,
                confidence=0.87,
                class_name="truck",
                class_id=7
            ))

        # 3. Motorcycle moving down left lane
        moto_y = int((280 + frame_number * 4.8) % (h - 80))
        moto_x = int(w * 0.15)
        if moto_y + 50 < h:
            bbox = BoundingBox(x1=moto_x, y1=moto_y, x2=moto_x + 35, y2=moto_y + 48)
            detections.append(VehicleDetection(
                bbox=bbox,
                confidence=0.83,
                class_name="motorcycle",
                class_id=3
            ))

        metrics = DetectionMetrics(
            preprocess_time_ms=0.4,
            inference_time_ms=total_latency_ms - 0.6,
            postprocess_time_ms=0.2,
            total_latency_ms=total_latency_ms,
            fps=1000.0 / total_latency_ms if total_latency_ms > 0 else 30.0
        )
        
        return DetectionResult(
            detections=detections,
            metrics=metrics,
            frame_number=frame_number,
            timestamp=timestamp,
            model_name="simulated_model.pt"
        )
