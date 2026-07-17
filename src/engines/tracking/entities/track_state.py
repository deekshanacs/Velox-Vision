from enum import Enum, auto

class TrackState(Enum):
    """Enumeration representing the lifecycle state transitions of a tracked object.
    
    Adheres to standard Multi-Object Tracking (MOT) state machines.
    """
    
    TENTATIVE = auto()
    """Newly spawned track that has not yet been confirmed by consecutive observations."""
    
    CONFIRMED = auto()
    """Track has been active and observed for enough frames to be considered stable."""
    
    TRACKED = auto()
    """Track successfully matched in the current frame (synonymous to CONFIRMED)."""
    
    TEMPORARILY_LOST = auto()
    """Track is missing matching detections in the active frame, but is retained for recovery."""
    
    RECOVERED = auto()
    """A previously lost track that has successfully re-associated with a new detection."""
    
    EXITED = auto()
    """The tracked vehicle has crossed the boundaries of the field of view."""
    
    REMOVED = auto()
    """The track has exceeded the frame retention threshold and is permanently closed."""
