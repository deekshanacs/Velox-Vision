from collections import deque
from typing import List
from src.engines.tracking.value_objects.track_snapshot import TrackSnapshot

class TrackHistory:
    """Manages the historical list of movement snapshots for a tracked vehicle.
    
    Provides bounded memory usage via collections.deque, append-only actions,
    and copy-on-read accessors.
    """
    
    def __init__(self, max_size: int = 30):
        """Initializes track history with capacity boundaries."""
        self._max_size = max_size
        self._snapshots = deque(maxlen=max_size)

    def append(self, snapshot: TrackSnapshot) -> None:
        """Appends a new track state snapshot to the end of the history queue."""
        self._snapshots.append(snapshot)

    @property
    def snapshots(self) -> List[TrackSnapshot]:
        """Returns a read-only list representation of the snapshots history."""
        return list(self._snapshots)

    @property
    def max_size(self) -> int:
        """Returns the maximum historical capacity limit of this track queue."""
        return self._max_size

    @property
    def length(self) -> int:
        """Returns the current number of snapshots recorded in the queue."""
        return len(self._snapshots)

    def clear(self) -> None:
        """Clears all historical snapshots from the queue."""
        self._snapshots.clear()
