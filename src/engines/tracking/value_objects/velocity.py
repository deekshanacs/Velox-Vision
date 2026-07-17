from dataclasses import dataclass

@dataclass(frozen=True)
class Velocity:
    """Immutable value object representing motion speed vectors (vx, vy) in pixels.
    
    Used to track trajectory direction and estimate physical speeds.
    """
    vx: float  # Horizontal velocity component
    vy: float  # Vertical velocity component

    @property
    def speed(self) -> float:
        """Computes the magnitude of the velocity vector."""
        import math
        return math.sqrt(self.vx ** 2 + self.vy ** 2)

    def to_tuple(self) -> tuple:
        """Converts velocity coordinates to a tuple (vx, vy)."""
        return self.vx, self.vy
