from dataclasses import dataclass

@dataclass(frozen=True)
class TrajectoryStatistics:
    """Detailed geometric path characteristics."""
    path_length: float           # Total distance travelled
    trajectory_curvature: float  # Sum of absolute heading turn angles divided by length
    direction_changes: int       # Number of axis sign changes in displacement
    heading_changes: int         # Count of heading transitions above threshold
    average_heading: float       # Mean path direction in radians
    maximum_turn_angle: float    # Largest turn angle recorded in radians
    trajectory_smoothness: float # Index indicating path stability (1.0 = straight line)
