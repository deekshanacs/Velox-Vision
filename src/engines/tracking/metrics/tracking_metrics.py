from dataclasses import dataclass

@dataclass
class TrackingMetrics:
    """Aggregated metrics container for tracking engine quality and performance."""
    tracks_created: int = 0
    tracks_removed: int = 0
    tracks_lost: int = 0
    recovered_tracks: int = 0
    longest_track_age: int = 0
    total_track_age: int = 0
    current_active_tracks: int = 0
    peak_active_tracks: int = 0
    total_latency_ms: float = 0.0
    processed_frames: int = 0
    id_switch_count: int = 0
    fragmentation_count: int = 0

    @property
    def average_track_age(self) -> float:
        """Computes average age duration of tracks."""
        return (self.total_track_age / self.tracks_created) if self.tracks_created > 0 else 0.0

    @property
    def average_tracking_latency_ms(self) -> float:
        """Computes mean processing execution duration."""
        return (self.total_latency_ms / self.processed_frames) if self.processed_frames > 0 else 0.0

    @property
    def tracking_fps(self) -> float:
        """Calculates effective tracking execution rate."""
        avg_latency = self.average_tracking_latency_ms
        return (1000.0 / avg_latency) if avg_latency > 0 else 0.0

    def reset(self) -> None:
        """Resets all metrics values to their defaults."""
        self.tracks_created = 0
        self.tracks_removed = 0
        self.tracks_lost = 0
        self.recovered_tracks = 0
        self.longest_track_age = 0
        self.total_track_age = 0
        self.current_active_tracks = 0
        self.peak_active_tracks = 0
        self.total_latency_ms = 0.0
        self.processed_frames = 0
        self.id_switch_count = 0
        self.fragmentation_count = 0
