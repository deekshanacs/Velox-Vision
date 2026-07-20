import math
from dataclasses import dataclass
from src.engines.tracking.value_objects.point import Point
from src.engines.tracking.lane.lane_model import Lane


@dataclass(frozen=True)
class LaneGeometryMetrics:
    """Detailed geometric relationship metrics of a vehicle relative to a lane."""
    distance_to_centerline: float
    distance_to_left_boundary: float
    distance_to_right_boundary: float
    normalized_lateral_offset: float  # -1.0 (left boundary) to +1.0 (right boundary), 0.0 at center
    relative_heading_rad: float
    relative_heading_deg: float
    longitudinal_position: float
    nearest_boundary_name: str


class LaneGeometryCalculator:
    """Computes high-precision spatial and angular geometry of vehicle relative to lane model."""

    @staticmethod
    def calculate(
        vehicle_center: Point,
        lane: Lane,
        vehicle_heading_rad: float = 0.0
    ) -> LaneGeometryMetrics:
        """Calculates distance to centerline, boundaries, lateral offset, and relative heading."""
        vx = vehicle_center.x
        vy = vehicle_center.y

        cx = lane.centerline.center_x
        lx = lane.left_boundary.position_x
        rx = lane.right_boundary.position_x

        half_w = lane.width / 2.0 if lane.width > 0 else 1.0

        # Lateral distances
        dist_center = vx - cx
        dist_left = abs(vx - lx)
        dist_right = abs(vx - rx)

        # Normalized lateral offset (-1.0 left edge, 0.0 center, +1.0 right edge)
        norm_offset = dist_center / half_w if half_w > 0 else 0.0
        norm_offset = max(-2.0, min(2.0, norm_offset))

        # Nearest boundary
        nearest_boundary = "LEFT" if dist_left < dist_right else "RIGHT"

        # Relative heading angle (assuming vertical lane orientation, lane direction is along -y or +y)
        # Standard lane direction is north (rad = -pi/2 or 0 depending on convention)
        rel_heading_rad = math.atan2(math.sin(vehicle_heading_rad), math.cos(vehicle_heading_rad))
        rel_heading_deg = math.degrees(rel_heading_rad)

        return LaneGeometryMetrics(
            distance_to_centerline=round(dist_center, 2),
            distance_to_left_boundary=round(dist_left, 2),
            distance_to_right_boundary=round(dist_right, 2),
            normalized_lateral_offset=round(norm_offset, 3),
            relative_heading_rad=round(rel_heading_rad, 4),
            relative_heading_deg=round(rel_heading_deg, 2),
            longitudinal_position=round(vy, 2),
            nearest_boundary_name=nearest_boundary
        )
