from dataclasses import dataclass

@dataclass(frozen=True)
class BoundingBoxStability:
    """Evaluates spatial dimensions and jitter stability of tracking boxes."""
    area_stability: float        # Standard deviation or variance of box area
    aspect_ratio_stability: float # Standard deviation or variance of width/height ratio
    center_jitter: float          # Average centroid frame-to-frame shift
    detection_stability: float     # Ratio of active observations vs. total track age

@dataclass(frozen=True)
class ConfidenceAnalysis:
    """Profiles motion and trajectory consistency indexes."""
    motion_confidence: float      # Score based on jitter and frame gaps consistency
    observation_confidence: float # Mean of detector confidence values
    trajectory_confidence: float  # Score based on trail curvature consistency
    tracking_confidence: float   # Current frame tracking confidence
