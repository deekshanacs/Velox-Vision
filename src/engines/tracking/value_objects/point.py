from dataclasses import dataclass

@dataclass(frozen=True)
class Point:
    """Immutable value object representing a 2D spatial coordinate (x, y) in pixel space.
    
    Provides foundational geometry properties.
    """
    x: float
    y: float

    def distance_to(self, other: 'Point') -> float:
        """Calculates Euclidean distance between this point and another point."""
        import math
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def to_tuple(self) -> tuple:
        """Converts coordinate values to a standard tuple (x, y)."""
        return self.x, self.y
