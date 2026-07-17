import numpy as np
from dataclasses import dataclass
from src.core.entities import DetectionResult
from src.engines.tracking.value_objects.frame_metadata import FrameMetadata

@dataclass(frozen=True)
class TrackingContext:
    """Immutable context containing everything necessary to update the tracker state.
    
    Combines the raw BGR image array, detection metrics, and frame DTO metadata.
    """
    frame: np.ndarray
    detections: DetectionResult
    metadata: FrameMetadata
