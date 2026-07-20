from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
from src.engines.tracking.motion.motion_vector import MotionVector


class CalibrationMode(str, Enum):
    """Modes of spatial calibration used for speed calculation."""
    DEFAULT = "DEFAULT"
    MANUAL = "MANUAL"
    PERSPECTIVE = "PERSPECTIVE"


@dataclass(frozen=True)
class SpeedProfile:
    """Comprehensive speed and velocity profile for a tracked vehicle.
    
    Contains all physical speed metrics (m/s and km/h), velocity vectors,
    acceleration/deceleration statistics, and multi-factor confidence scores.
    """
    current_speed_mps: float
    current_speed_kmh: float
    average_speed_kmh: float
    peak_speed_kmh: float
    min_speed_kmh: float
    
    speed_history: List[float] = field(default_factory=list)
    
    acceleration_mps2: float = 0.0
    deceleration_mps2: float = 0.0
    is_accelerating: bool = False
    is_decelerating: bool = False
    
    velocity_vector_mps: MotionVector = field(default_factory=lambda: MotionVector(0.0, 0.0))
    velocity_vector_kmh: MotionVector = field(default_factory=lambda: MotionVector(0.0, 0.0))
    
    timestamp: float = 0.0
    
    speed_confidence: float = 1.0
    calibration_confidence: float = 1.0
    observation_confidence: float = 1.0
    overall_confidence: float = 1.0
    
    calibration_mode: CalibrationMode = CalibrationMode.DEFAULT
