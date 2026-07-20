from enum import Enum, auto

class MotionState(Enum):
    """Descriptive-only representation of a vehicle's motion state lifecycle.
    
    Contains no velocity/speed metrics.
    """
    UNKNOWN = auto()
    STATIONARY = auto()
    MOVING = auto()
    SLOW_MOVEMENT = auto()
    CONTINUOUS_MOVEMENT = auto()
    INTERMITTENT_MOVEMENT = auto()
    RECOVERED_MOTION = auto()
