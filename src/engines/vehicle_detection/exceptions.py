class DetectionError(Exception):
    """Base exception class for all vehicle detection engine errors."""
    pass


class ModelLoadError(DetectionError):
    """Raised when the YOLO or target neural model fails to load weights from disk."""
    pass


class InvalidFrameError(DetectionError):
    """Raised when an empty, corrupted, or unsupported frame shape is provided to the detector."""
    pass


class InferenceFailedError(DetectionError):
    """Raised when model forward propagation or postprocessing encounters critical execution errors."""
    pass
