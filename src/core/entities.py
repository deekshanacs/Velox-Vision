from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class BoundingBox:
    """Represents a 2D bounding box with absolute coordinates.
    
    Attributes:
        x1 (float): Horizontal start coordinate (left edge)
        y1 (float): Vertical start coordinate (top edge)
        x2 (float): Horizontal end coordinate (right edge)
        y2 (float): Vertical end coordinate (bottom edge)
    """
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Calculates the width of the bounding box."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Calculates the height of the bounding box."""
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        """Calculates the horizontal center coordinate."""
        return self.x1 + self.width / 2.0

    @property
    def center_y(self) -> float:
        """Calculates the vertical center coordinate."""
        return self.y1 + self.height / 2.0

    @property
    def area(self) -> float:
        """Calculates the total pixel area of the box."""
        return self.width * self.height


@dataclass(frozen=True)
class VehicleDetection:
    """Represents a single structured vehicle detection.
    
    Attributes:
        bbox (BoundingBox): The structured bounding box coordinates and dimensions.
        confidence (float): Model confidence score (between 0.0 and 1.0).
        class_name (str): Label of the vehicle ('car', 'motorcycle', 'bus', 'truck').
        class_id (int): Model class index.
    """
    bbox: BoundingBox
    confidence: float
    class_name: str
    class_id: int


@dataclass(frozen=True)
class DetectionMetrics:
    """Performance metrics captured during frame processing.
    
    Attributes:
        preprocess_time_ms (float): Frame resizing and tensor conversion latency in milliseconds.
        inference_time_ms (float): Model neural network execution latency in milliseconds.
        postprocess_time_ms (float): Non-maximum suppression and filtering latency in milliseconds.
        total_latency_ms (float): Total processing time for this frame in milliseconds.
        fps (float): Frame processing rate (frames per second).
    """
    preprocess_time_ms: float
    inference_time_ms: float
    postprocess_time_ms: float
    total_latency_ms: float
    fps: float


@dataclass(frozen=True)
class DetectionResult:
    """Comprehensive output packet containing detection data and execution telemetry.
    
    Attributes:
        detections (List[VehicleDetection]): Structured list of detected vehicles.
        metrics (DetectionMetrics): Latency and FPS metrics.
        frame_number (int): Sequential index of the processed frame.
        timestamp (float): UNIX epoch timestamp when the frame was processed.
        model_name (str): Name of the model architecture used (e.g. 'yolo11n.pt').
    """
    detections: List[VehicleDetection]
    metrics: DetectionMetrics
    frame_number: int
    timestamp: float
    model_name: str
