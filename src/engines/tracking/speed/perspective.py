import math
from typing import List, Optional, Tuple
from src.engines.tracking.value_objects.point import Point
from src.engines.tracking.speed.calibration import SpeedCalibration
from src.engines.tracking.speed.speed_profile import CalibrationMode


class PerspectiveTransformer:
    """Handles 2D/3D perspective transformations and ground plane projections.
    
    Transforms image space coordinates into ground plane coordinates using optional
    homography matrix mapping or trapezoidal foreshortening models.
    """

    def __init__(
        self,
        enabled: bool = False,
        homography_matrix: Optional[List[List[float]]] = None,
        source_points: Optional[List[Tuple[float, float]]] = None,
        destination_points: Optional[List[Tuple[float, float]]] = None
    ):
        self.enabled = enabled
        self.homography_matrix = homography_matrix
        self.source_points = source_points or []
        self.destination_points = destination_points or []

    def set_homography_matrix(self, matrix_3x3: List[List[float]]) -> None:
        """Sets the 3x3 homography transformation matrix."""
        if len(matrix_3x3) != 3 or any(len(row) != 3 for row in matrix_3x3):
            raise ValueError("Homography matrix must be a 3x3 array.")
        self.homography_matrix = matrix_3x3
        self.enabled = True

    def transform_point(self, point: Point) -> Point:
        """Transforms a 2D image point to perspective-corrected ground plane coordinates."""
        if not self.enabled or self.homography_matrix is None:
            return point

        H = self.homography_matrix
        x, y = point.x, point.y

        # Homography multiplication [u, v, w]^T = H * [x, y, 1]^T
        u = H[0][0] * x + H[0][1] * y + H[0][2]
        v = H[1][0] * x + H[1][1] * y + H[1][2]
        w = H[2][0] * x + H[2][1] * y + H[2][2]

        if abs(w) < 1e-9:
            return point

        return Point(u / w, v / w)

    def compute_ground_displacement(
        self,
        p1: Point,
        p2: Point,
        calibration: SpeedCalibration
    ) -> Tuple[float, CalibrationMode, float]:
        """Computes the true ground-plane displacement in meters between two image points.
        
        Returns:
            Tuple of (displacement_meters, calibration_mode, calibration_confidence)
        """
        if self.enabled and self.homography_matrix is not None:
            gp1 = self.transform_point(p1)
            gp2 = self.transform_point(p2)
            ground_dist_px = gp1.distance_to(gp2)
            meters = ground_dist_px * calibration.pixel_to_meter_ratio
            return meters, CalibrationMode.PERSPECTIVE, 0.95

        # Standard flat-plane Euclidean displacement fallback
        pixel_dist = p1.distance_to(p2)
        meters = calibration.pixels_to_meters(pixel_dist)
        return meters, calibration.mode, calibration.calibration_confidence
