from abc import ABC, abstractmethod
from typing import List, Optional
from src.engines.tracking.entities.tracked_vehicle import TrackedVehicle

class TrackRepository(ABC):
    """Abstract Base Class defining the contract for tracking persistence.
    
    Acts as a repository boundary shielding tracking domains from database layers.
    """

    @abstractmethod
    def add(self, vehicle: TrackedVehicle) -> None:
        """Adds a new tracked vehicle to the repository store.
        
        Args:
            vehicle: TrackedVehicle instance to add.
        """
        pass

    @abstractmethod
    def get(self, track_id: int) -> Optional[TrackedVehicle]:
        """Retrieves a tracked vehicle by its unique ID.
        
        Args:
            track_id: Unique tracking pointer index.
            
        Returns:
            The TrackedVehicle instance if active, else None.
        """
        pass

    @abstractmethod
    def get_all_active(self) -> List[TrackedVehicle]:
        """Returns all currently active tracked vehicles."""
        pass

    @abstractmethod
    def remove(self, track_id: int) -> None:
        """Removes a tracked vehicle from repository storage.
        
        Args:
            track_id: Unique tracking pointer index.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clears all stored tracked vehicles."""
        pass
