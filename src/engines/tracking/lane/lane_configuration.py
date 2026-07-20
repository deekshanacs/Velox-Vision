from dataclasses import dataclass


@dataclass(frozen=True)
class LaneConfiguration:
    """Strongly-typed parameters container for the Lane Intelligence Engine."""
    enabled: bool = True
    lane_count: int = 3
    lane_width: float = 120.0
    boundary_margin: float = 15.0
    confidence_threshold: float = 0.6
    road_start_x: float = 0.0
    coordinate_system: str = "image"
    lane_origin: str = "left"
    orientation: str = "vertical"
    hysteresis_frames: int = 3
