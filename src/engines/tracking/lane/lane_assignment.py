from enum import Enum


class LaneAssignmentStatus(str, Enum):
    """Assignment status of a tracked vehicle relative to road lanes."""
    UNKNOWN = "UNKNOWN"
    ASSIGNED = "ASSIGNED"
    TRANSITIONING = "TRANSITIONING"
    OUTSIDE_LANE = "OUTSIDE_LANE"
    TEMPORARY_ASSIGNMENT = "TEMPORARY_ASSIGNMENT"
