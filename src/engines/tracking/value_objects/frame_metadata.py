from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass(frozen=True)
class FrameMetadata:
    """Immutable DTO containing physical and runtime metadata for a single frame.
    
    Decoupled container representing resolution, timestamps, frame rates, and latency.
    """
    frame_number: int
    timestamp: float
    width: int
    height: int
    fps: float
    processing_time_ms: Optional[float] = None

    @property
    def resolution(self) -> Tuple[int, int]:
        """Returns the frame dimensions tuple (width, height)."""
        return self.width, self.height
