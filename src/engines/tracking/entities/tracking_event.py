from dataclasses import dataclass
from src.engines.tracking.value_objects.track_snapshot import TrackSnapshot

@dataclass(frozen=True)
class TrackingEvent:
    """Base tracking event value object representing a track state transition."""
    track_id: int
    frame_number: int
    timestamp: float
    snapshot: TrackSnapshot


@dataclass(frozen=True)
class TrackCreated(TrackingEvent):
    """Fired when a new track is initialized in tentative state."""
    pass


@dataclass(frozen=True)
class TrackConfirmed(TrackingEvent):
    """Fired when a tentative track satisfies minimum observation counts and is confirmed."""
    pass


@dataclass(frozen=True)
class TrackLost(TrackingEvent):
    """Fired when an active confirmed track fails to match in the current frame."""
    pass


@dataclass(frozen=True)
class TrackRecovered(TrackingEvent):
    """Fired when a lost track is successfully re-associated with a detection."""
    pass


@dataclass(frozen=True)
class TrackExited(TrackingEvent):
    """Fired when a tracked vehicle crosses the field of view boundary constraints."""
    pass
