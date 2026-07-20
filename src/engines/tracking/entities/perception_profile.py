from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class PerceptionProfile:
    """Unified container aggregating all perception engine output profiles for a vehicle track.
    
    Prevents entity property bloat on TrackedVehicle as higher-level perception subsystems accumulate.
    """
    motion: Optional[Any] = None
    speed: Optional[Any] = None
    lane: Optional[Any] = None
    behavior: Optional[Any] = None
    violation: Optional[Any] = None
    prediction: Optional[Any] = None
