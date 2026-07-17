import logging
from typing import List, Optional
from src.core.interfaces import IVehicleDetector
from src.engines.vehicle_detection.vehicle_detector import YOLOVehicleDetector

logger = logging.getLogger(__name__)

class VehicleDetectorFactory:
    """Factory class to create and retrieve vehicle detector instances."""

    @staticmethod
    def create_detector(
        detector_type: str = "yolo",
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        device: Optional[str] = None,
        warmup: Optional[bool] = None,
        classes: Optional[List[str]] = None
    ) -> IVehicleDetector:
        """Instantiates a vehicle detector engine according to type and configs.
        
        Args:
            detector_type: Class string type ('yolo' or custom implementations).
            model_path: Model weights file path override.
            confidence_threshold: Confidence minimum filter boundary override.
            device: Computing device override.
            warmup: Warm-up override.
            classes: Target classes list override.
            
        Returns:
            An instance of IVehicleDetector.
            
        Raises:
            ValueError: If the requested detector_type is not supported.
        """
        detector_type_lower = detector_type.lower()
        
        if detector_type_lower == "yolo":
            return YOLOVehicleDetector(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                device=device,
                warmup=warmup,
                classes=classes
            )
        elif detector_type_lower in ("simulated", "mock"):
            from src.engines.vehicle_detection.simulated_detector import SimulatedVehicleDetector
            return SimulatedVehicleDetector(
                confidence_threshold=confidence_threshold or 0.25
            )
        else:
            error_msg = f"Unsupported vehicle detector type: '{detector_type}'"
            logger.error(error_msg)
            raise ValueError(error_msg)
