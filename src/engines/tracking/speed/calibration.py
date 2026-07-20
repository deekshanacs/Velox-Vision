from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from src.engines.tracking.value_objects.point import Point
from src.engines.tracking.speed.speed_profile import CalibrationMode


@dataclass
class SpeedCalibration:
    """Manages spatial calibration parameters for converting pixel measurements to ground-truth meters.
    
    Supports manual ratio configuration, reference distance calibration, reference point pairs,
    and perspective transform integration.
    """
    pixel_to_meter_ratio: float = 0.05
    reference_distance_meters: Optional[float] = None
    reference_distance_pixels: Optional[float] = None
    reference_points: List[Tuple[float, float]] = field(default_factory=list)
    mode: CalibrationMode = CalibrationMode.DEFAULT
    calibration_confidence: float = 0.8

    def __post_init__(self):
        self.recalculate_scale()

    def recalculate_scale(self) -> None:
        """Recalculate the effective pixel to meter scale ratio based on reference parameters."""
        if (
            self.reference_distance_meters is not None
            and self.reference_distance_pixels is not None
            and self.reference_distance_pixels > 0
        ):
            self.pixel_to_meter_ratio = self.reference_distance_meters / self.reference_distance_pixels
            self.mode = CalibrationMode.MANUAL
            self.calibration_confidence = 1.0
        elif len(self.reference_points) >= 2:
            p1 = Point(self.reference_points[0][0], self.reference_points[0][1])
            p2 = Point(self.reference_points[1][0], self.reference_points[1][1])
            px_dist = p1.distance_to(p2)
            if px_dist > 0 and self.reference_distance_meters is not None:
                self.reference_distance_pixels = px_dist
                self.pixel_to_meter_ratio = self.reference_distance_meters / px_dist
                self.mode = CalibrationMode.MANUAL
                self.calibration_confidence = 1.0

    def pixels_to_meters(self, pixel_distance: float) -> float:
        """Converts distance in pixels to ground meters using active calibration ratio."""
        return pixel_distance * self.pixel_to_meter_ratio

    def set_manual_ratio(self, ratio: float) -> None:
        """Directly sets manual pixel-to-meter ratio."""
        if ratio <= 0:
            raise ValueError("Pixel-to-meter ratio must be strictly positive.")
        self.pixel_to_meter_ratio = ratio
        self.mode = CalibrationMode.MANUAL
        self.calibration_confidence = 0.95
