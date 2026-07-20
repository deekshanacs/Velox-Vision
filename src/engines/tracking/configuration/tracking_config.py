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
    
    # Memory Subsystem Parameters
    memory_enabled: bool = True
    memory_max_snapshots: int = 300
    memory_cleanup_interval: int = 500
    memory_statistics_enabled: bool = True
    memory_confidence_history: bool = True

    # Visualization Subsystem Parameters
    viz_enabled: bool = True
    viz_show_trails: bool = True
    viz_trail_length: int = 40
    viz_show_memory: bool = True
    viz_show_states: bool = True
    viz_show_profiler: bool = True
    viz_show_dashboard: bool = True
    viz_debug: bool = False

    # Motion Analytics Parameters
    motion_enabled: bool = True
    motion_minimum_snapshots: int = 5
    motion_stationary_threshold: float = 2.5
    motion_heading_window: int = 10
    motion_smoothing_window: int = 8
    motion_confidence_threshold: float = 0.5

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
            show_ids=settings.get("tracking.visualization.show_ids", True),
            
            # Memory Loaders
            memory_enabled=settings.get("memory.enabled", True),
            memory_max_snapshots=settings.get("memory.max_snapshots", 300),
            memory_cleanup_interval=settings.get("memory.cleanup_interval", 500),
            memory_statistics_enabled=settings.get("memory.statistics_enabled", True),
            memory_confidence_history=settings.get("memory.confidence_history", True),

            # Visualization Loaders
            viz_enabled=settings.get("tracking_visualization.enabled", True),
            viz_show_trails=settings.get("tracking_visualization.show_trails", True),
            viz_trail_length=settings.get("tracking_visualization.trail_length", 40),
            viz_show_memory=settings.get("tracking_visualization.show_memory", True),
            viz_show_states=settings.get("tracking_visualization.show_states", True),
            viz_show_profiler=settings.get("tracking_visualization.show_profiler", True),
            viz_show_dashboard=settings.get("tracking_visualization.show_dashboard", True),
            viz_debug=settings.get("tracking_visualization.debug", False),

            # Motion Loaders
            motion_enabled=settings.get("motion.enabled", True),
            motion_minimum_snapshots=settings.get("motion.minimum_snapshots", 5),
            motion_stationary_threshold=settings.get("motion.stationary_threshold", 2.5),
            motion_heading_window=settings.get("motion.heading_window", 10),
            motion_smoothing_window=settings.get("motion.smoothing_window", 8),
            motion_confidence_threshold=settings.get("motion.confidence_threshold", 0.5)
        )
