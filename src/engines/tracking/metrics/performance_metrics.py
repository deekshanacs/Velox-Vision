from dataclasses import dataclass

@dataclass
class TrackingPerformance:
    """Telemetric performance metrics profiling throughput and runtime latency.
    
    Used to benchmark processing speed and detect runtime performance degradation.
    """
    total_latency_ms: float = 0.0
    processed_frames: int = 0
    peak_active_tracks: int = 0
    current_active_tracks: int = 0

    @property
    def average_latency_ms(self) -> float:
        """Computes the average tracking execution latency."""
        return (self.total_latency_ms / self.processed_frames) if self.processed_frames > 0 else 0.0

    @property
    def tracking_fps(self) -> float:
        """Calculates tracking execution throughput rate in Frames Per Second."""
        avg_lat = self.average_latency_ms
        return (1000.0 / avg_lat) if avg_lat > 0 else 0.0

    def reset(self) -> None:
        """Resets all performance metrics counters to default values."""
        self.total_latency_ms = 0.0
        self.processed_frames = 0
        self.peak_active_tracks = 0
        self.current_active_tracks = 0
