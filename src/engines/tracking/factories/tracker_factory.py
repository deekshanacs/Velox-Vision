import logging
from typing import List, Optional
from src.engines.tracking.interfaces.tracker import Tracker
from src.engines.tracking.configuration.tracking_config import TrackingConfiguration

logger = logging.getLogger(__name__)

class TrackerFactory:
    """Factory module to instantiate trackers adhering to the Tracker ABC."""

    @staticmethod
    def create_tracker(
        tracker_type: str = "bytetrack",
        configs: Optional[TrackingConfiguration] = None
    ) -> Tracker:
        """Instantiates a tracking engine concrete implementation.
        
        Args:
            tracker_type: Key identifier of tracker implementation ('bytetrack', 'deepsort').
            configs: Strongly-typed tracking configurations override.
            
        Returns:
            An instance implementing the Tracker interface.
            
        Raises:
            ValueError: If tracker_type is unrecognized.
            NotImplementedError: For architecture-only validation phases.
        """
        tracker_type_lower = tracker_type.lower()
        logger.info(f"Instantiating object tracker of type: '{tracker_type_lower}'")

        if tracker_type_lower == "bytetrack":
            from src.engines.tracking.implementations.bytetrack_tracker import ByteTrackTracker
            tracker = ByteTrackTracker()
            if configs is not None:
                tracker.initialize(configs)
            return tracker
        elif tracker_type_lower in ("deepsort", "ocsort", "botsort"):
            error_msg = f"Concrete tracker '{tracker_type_lower}' is not yet implemented. Please complete Phase 3.2 architectural setup."
            logger.error(error_msg)
            raise NotImplementedError(error_msg)
        else:
            error_msg = f"Unsupported tracker type: '{tracker_type}'"
            logger.error(error_msg)
            raise ValueError(error_msg)

    @staticmethod
    def available_trackers() -> List[str]:
        """Returns the list of available tracking implementation keys."""
        return ["bytetrack", "deepsort", "ocsort", "botsort"]
