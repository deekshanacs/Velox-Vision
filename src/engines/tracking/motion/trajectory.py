from dataclasses import dataclass
from typing import List
from src.engines.tracking.value_objects.point import Point

@dataclass(frozen=True)
class Trajectory:
    """Immutable sequence of spatial coordinate points visited by the vehicle."""
    points: List[Point]
