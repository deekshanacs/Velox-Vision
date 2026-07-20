import logging
from typing import Dict, Tuple
from src.engines.tracking.entities.track_state import TrackState

logger = logging.getLogger(__name__)

# Configurable BGR colors for each track state
STATE_COLORS: Dict[TrackState, Tuple[int, int, int]] = {
    TrackState.TENTATIVE: (0, 255, 255),          # Yellow
    TrackState.CONFIRMED: (0, 255, 0),            # Green
    TrackState.TRACKED: (255, 0, 0),              # Blue
    TrackState.RECOVERED: (255, 0, 255),          # Purple
    TrackState.TEMPORARILY_LOST: (0, 165, 255),   # Orange
    TrackState.EXITED: (128, 128, 128),           # Gray
    TrackState.REMOVED: (128, 128, 128)           # Gray
}

def get_state_color(state: TrackState, overrides: Dict[TrackState, Tuple[int, int, int]] = None) -> Tuple[int, int, int]:
    """Returns the BGR color for a given TrackState, supporting overrides."""
    if overrides and state in overrides:
        return overrides[state]
    return STATE_COLORS.get(state, (128, 128, 128))
