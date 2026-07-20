from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from src.engines.tracking.lane.lane_configuration import LaneConfiguration


class BoundaryType(str, Enum):
    """Types of lane boundary markings."""
    SOLID = "SOLID"
    DASHED = "DASHED"
    DOUBLE_SOLID = "DOUBLE_SOLID"
    SHOULDER = "SHOULDER"
    DIVIDER = "DIVIDER"


@dataclass(frozen=True)
class LaneBoundary:
    """Represents a lateral boundary delimiting a lane."""
    boundary_id: str
    position_x: float
    boundary_type: BoundaryType = BoundaryType.DASHED


@dataclass(frozen=True)
class LaneCenterline:
    """Represents the central reference path of a lane."""
    center_x: float


@dataclass(frozen=True)
class Lane:
    """Domain model representing a single physical or virtual traffic lane."""
    lane_id: int
    name: str
    left_boundary: LaneBoundary
    right_boundary: LaneBoundary
    centerline: LaneCenterline
    width: float

    def contains_x(self, x: float) -> bool:
        """Returns True if the given x coordinate lies within lane boundaries."""
        return self.left_boundary.position_x <= x <= self.right_boundary.position_x


@dataclass
class RoadModel:
    """Hierarchical structural model of a roadway containing multiple lanes."""
    lanes: Dict[int, Lane] = field(default_factory=dict)
    total_width: float = 0.0
    orientation: str = "vertical"

    def get_lane(self, lane_id: int) -> Optional[Lane]:
        """Retrieve lane model by ID."""
        return self.lanes.get(lane_id)

    @classmethod
    def create_from_config(cls, config: LaneConfiguration) -> 'RoadModel':
        """Constructs a RoadModel instance from configuration parameters."""
        lanes = {}
        curr_x = config.road_start_x

        for i in range(1, config.lane_count + 1):
            left_x = curr_x
            right_x = curr_x + config.lane_width
            center_x = (left_x + right_x) / 2.0

            l_type = BoundaryType.SHOULDER if i == 1 else BoundaryType.DASHED
            r_type = BoundaryType.SHOULDER if i == config.lane_count else BoundaryType.DASHED

            left_b = LaneBoundary(f"b_{i-1}", left_x, l_type)
            right_b = LaneBoundary(f"b_{i}", right_x, r_type)
            center_l = LaneCenterline(center_x)

            lanes[i] = Lane(
                lane_id=i,
                name=f"Lane {i}",
                left_boundary=left_b,
                right_boundary=right_b,
                centerline=center_l,
                width=config.lane_width
            )
            curr_x = right_x

        total_w = config.lane_width * config.lane_count
        return cls(lanes=lanes, total_width=total_w, orientation=config.orientation)
