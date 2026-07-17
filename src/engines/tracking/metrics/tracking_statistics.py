from dataclasses import dataclass

@dataclass
class TrackingStatistics:
    """Statistical tracking metrics profiling track lifecycle quality.
    
    Provides insights into AI model tracking accuracy (ID switches, fragmentations).
    """
    tracks_created: int = 0
    tracks_removed: int = 0
    tracks_lost: int = 0
    recovered_tracks: int = 0
    total_track_age: int = 0
    id_switch_count: int = 0
    fragmentation_count: int = 0

    @property
    def average_track_age(self) -> float:
        """Computes the average age duration of spawned tracks."""
        return (self.total_track_age / self.tracks_created) if self.tracks_created > 0 else 0.0

    def reset(self) -> None:
        """Resets all statistical metrics counters to default values."""
        self.tracks_created = 0
        self.tracks_removed = 0
        self.tracks_lost = 0
        self.recovered_tracks = 0
        self.total_track_age = 0
        self.id_switch_count = 0
        self.fragmentation_count = 0
