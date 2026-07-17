from abc import ABC, abstractmethod
import numpy as np
from src.core.entities import DetectionResult

class IVehicleDetector(ABC):
    """Abstract interface contract for any vehicle detector implementation."""

    @abstractmethod
    def detect(self, frame: np.ndarray, frame_number: int = 0) -> DetectionResult:
        """Detects vehicles in a given OpenCV image frame.
        
        Args:
            frame: OpenCV image frame (BGR format).
            frame_number: Optional sequential frame index. Defaults to 0.
            
        Returns:
            Structured DetectionResult object containing detections and metrics.
        """
        pass
