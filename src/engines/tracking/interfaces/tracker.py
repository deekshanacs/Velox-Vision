from abc import ABC, abstractmethod
from src.engines.tracking.configuration.tracking_config import TrackingConfiguration
from src.engines.tracking.entities.tracking_context import TrackingContext
from src.engines.tracking.entities.tracking_result import TrackingResult
from src.engines.tracking.metrics.tracking_statistics import TrackingStatistics
from src.engines.tracking.metrics.performance_metrics import TrackingPerformance

class Tracker(ABC):
    """Abstract Base Class defining the contract for all Multi-Object Trackers.
    
    Decoupled interface shielding client modules from external tracking implementations
    such as ByteTrack or DeepSORT.
    """

    @abstractmethod
    def initialize(self, configs: TrackingConfiguration) -> None:
        """Initializes the tracker weights and internal structures.
        
        Args:
            configs: Strongly-typed tracking configurations container.
            
        Raises:
            TrackerInitializationError: If configurations parsing or hardware setup fails.
        """
        pass

    @abstractmethod
    def track(self, context: TrackingContext) -> TrackingResult:
        """Associates new bounding box detections to tracked identities.
        
        Args:
            context: Context wrapper grouping frames, metadata, and detections result.
            
        Returns:
            TrackingResult object containing updated object coordinates and IDs.
            
        Raises:
            TrackingRuntimeError: If tracker state calculations throw runtime errors.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Clears all active tracks and resets internal frame counters."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Releases hardware resources or file streams cleanly."""
        pass

    @abstractmethod
    def get_statistics(self) -> TrackingStatistics:
        """Returns the tracking statistics metrics container."""
        pass

    @abstractmethod
    def get_performance(self) -> TrackingPerformance:
        """Returns the tracking performance metrics container."""
        pass

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Indicates whether the tracker has been successfully initialized."""
        pass
