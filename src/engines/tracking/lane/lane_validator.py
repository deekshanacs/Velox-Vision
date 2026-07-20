from typing import List
from src.engines.tracking.lane.lane_model import RoadModel


class LaneValidationError(ValueError):
    """Exception raised when road or lane geometry fails structural validation."""
    pass


class LaneValidator:
    """Validates structural integrity and geometric consistency of RoadModel configurations."""

    @staticmethod
    def validate_road_model(road_model: RoadModel) -> bool:
        """Validates that a RoadModel contains non-overlapping, positive-width lanes.
        
        Raises:
            LaneValidationError: If invalid lane parameters or overlaps are detected.
        """
        if not road_model.lanes:
            raise LaneValidationError("RoadModel contains no defined lanes.")

        sorted_lanes = sorted(road_model.lanes.values(), key=lambda l: l.lane_id)

        for lane in sorted_lanes:
            if lane.width <= 0:
                raise LaneValidationError(f"Lane {lane.lane_id} has invalid width: {lane.width}")
            if lane.left_boundary.position_x >= lane.right_boundary.position_x:
                raise LaneValidationError(
                    f"Lane {lane.lane_id} has inverted boundaries: left={lane.left_boundary.position_x}, right={lane.right_boundary.position_x}"
                )

        # Check for overlaps or gap inconsistencies
        for i in range(len(sorted_lanes) - 1):
            curr_l = sorted_lanes[i]
            next_l = sorted_lanes[i + 1]

            if curr_l.right_boundary.position_x > next_l.left_boundary.position_x + 1e-5:
                raise LaneValidationError(
                    f"Overlapping lanes detected between Lane {curr_l.lane_id} and Lane {next_l.lane_id}"
                )

        return True
