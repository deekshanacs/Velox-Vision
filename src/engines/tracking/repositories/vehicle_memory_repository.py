from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Iterator
from src.engines.tracking.memory.vehicle_memory import VehicleMemory

class VehicleMemoryRepository(ABC):
    """Abstract Base Class specifying persistent operations for VehicleMemory."""

    @abstractmethod
    def add(self, memory: VehicleMemory) -> None:
        """Persists a new vehicle memory record.
        
        Args:
            memory: The VehicleMemory entity to add.
        """
        pass

    @abstractmethod
    def get(self, track_id: int) -> Optional[VehicleMemory]:
        """Retrieves a memory record by ID.
        
        Args:
            track_id: Unique tracking pointer index.
            
        Returns:
            The VehicleMemory record if active, else None.
        """
        pass

    @abstractmethod
    def exists(self, track_id: int) -> bool:
        """Checks if a record exists in the repository.
        
        Args:
            track_id: Unique tracking pointer index.
            
        Returns:
            True if exists, else False.
        """
        pass

    @abstractmethod
    def remove(self, track_id: int) -> None:
        """Deletes a record.
        
        Args:
            track_id: Unique tracking pointer index.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clears all records from repository storage."""
        pass

    @abstractmethod
    def get_all(self) -> List[VehicleMemory]:
        """Returns all persisted records in the store."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[VehicleMemory]:
        """Supports iterating over records."""
        pass


class InMemoryVehicleMemoryRepository(VehicleMemoryRepository):
    """Concrete thread-safe in-memory store for vehicle memories."""
    
    def __init__(self):
        self._store: Dict[int, VehicleMemory] = {}

    def add(self, memory: VehicleMemory) -> None:
        self._store[memory.track_id] = memory

    def get(self, track_id: int) -> Optional[VehicleMemory]:
        return self._store.get(track_id)

    def exists(self, track_id: int) -> bool:
        return track_id in self._store

    def remove(self, track_id: int) -> None:
        if track_id in self._store:
            del self._store[track_id]

    def clear(self) -> None:
        self._store.clear()

    def get_all(self) -> List[VehicleMemory]:
        return list(self._store.values())

    def __iter__(self) -> Iterator[VehicleMemory]:
        return iter(self._store.values())
