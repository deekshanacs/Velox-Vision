class TrackingError(Exception):
    """Base exception class for all tracking engine related exceptions."""
    pass

class TrackerInitializationError(TrackingError):
    """Raised when tracker configurations or weights fail to initialize."""
    pass

class TrackingConfigurationError(TrackingError):
    """Raised when parameters values fail boundary checks or are missing."""
    pass

class TrackingRuntimeError(TrackingError):
    """Raised when execution or matrix computations fail during processing."""
    pass

class InvalidTrackStateError(TrackingError):
    """Raised when state lifecycle transactions are illegal."""
    pass
