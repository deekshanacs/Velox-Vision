import math
from dataclasses import dataclass

@dataclass(frozen=True)
class MotionVector:
    """Represents a spatial geometric displacement vector.
    
    Contains no speed estimation properties.
    """
    dx: float
    dy: float

    @property
    def magnitude(self) -> float:
        """Calculates the Euclidean distance displacement."""
        return math.sqrt(self.dx ** 2 + self.dy ** 2)

    @property
    def heading_rad(self) -> float:
        """Calculates direction angle in radians (range: -pi to pi)."""
        return math.atan2(self.dy, self.dx)
