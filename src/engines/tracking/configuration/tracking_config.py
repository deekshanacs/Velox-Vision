from dataclasses import dataclass

@dataclass(frozen=True)
class TrackingConfiguration:
    """Strongly-typed configurations container for the Multi-Object Tracking Engine.
    
    Prevents passing untyped dictionaries throughout internal tracking subsystems.
    """
    enabled: bool = True
    tracker_type: str = "bytetrack"
    history_size: int = 30
    max_lost_frames: int = 30
    tentative_frames: int = 3
    metrics_enabled: bool = True
    track_activation_threshold: float = 0.25
    minimum_matching_threshold: float = 0.8
    track_buffer: int = 30
    min_box_area: float = 10.0
    frame_rate: float = 25.0
    show_trails: bool = False
    show_ids: bool = True

    @classmethod
    def from_settings(cls) -> 'TrackingConfiguration':
        """Constructs tracking configurations using current system settings values."""
        from config.settings import settings
        return cls(
            enabled=settings.get("tracking.enabled", True),
            tracker_type=settings.get("tracking.tracker_type", "bytetrack"),
            history_size=settings.get("tracking.history_size", 30),
            max_lost_frames=settings.get("tracking.max_lost_frames", 30),
            tentative_frames=settings.get("tracking.tentative_frames", 3),
            metrics_enabled=settings.get("tracking.metrics_enabled", True),
            track_activation_threshold=settings.get("tracking.track_activation_threshold", 0.25),
            minimum_matching_threshold=settings.get("tracking.minimum_matching_threshold", 0.8),
            track_buffer=settings.get("tracking.track_buffer", 30),
            min_box_area=settings.get("tracking.min_box_area", 10.0),
            frame_rate=settings.get("tracking.frame_rate", 25.0),
            show_trails=settings.get("tracking.visualization.show_trails", False),
            show_ids=settings.get("tracking.visualization.show_ids", True)
        )
