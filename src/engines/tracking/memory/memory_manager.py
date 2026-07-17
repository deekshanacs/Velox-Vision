import logging
import sys
from typing import Optional, List
from src.engines.tracking.memory.vehicle_memory import VehicleMemory
from src.engines.tracking.memory.memory_snapshot import MemorySnapshot
from src.engines.tracking.memory.memory_statistics import MemoryStatistics
from src.engines.tracking.repositories.vehicle_memory_repository import VehicleMemoryRepository, InMemoryVehicleMemoryRepository
from src.engines.tracking.entities.track_state import TrackState
from src.engines.tracking.configuration.tracking_config import TrackingConfiguration

logger = logging.getLogger(__name__)

class MemoryManager:
    """Orchestrates vehicle memory creation, updates, and limits sweeps."""

    def __init__(self, configs: TrackingConfiguration, repository: Optional[VehicleMemoryRepository] = None):
        self.configs = configs
        self.repository = repository or InMemoryVehicleMemoryRepository()
        self.max_snapshots = configs.memory_max_snapshots
        self.cleanup_interval = configs.memory_cleanup_interval
        self.frame_counter = 0

    def create_memory(self, track_id: int, vehicle_class: str, class_id: int, frame_number: int, timestamp: float, bbox, confidence: float) -> VehicleMemory:
        """Initializes a new persistent memory space."""
        if self.repository.exists(track_id):
            raise ValueError(f"Memory with Track ID {track_id} already exists.")

        memory = VehicleMemory(
            track_id=track_id,
            vehicle_class=vehicle_class,
            class_id=class_id,
            creation_timestamp=timestamp,
            first_seen_frame=frame_number,
            last_seen_frame=frame_number,
            confidence=confidence,
            confidence_sum=confidence,
            highest_confidence=confidence,
            lowest_confidence=confidence,
            bbox=bbox,
            state=TrackState.TENTATIVE
        )
        self.repository.add(memory)
        return memory

    def update_memory(self, track_id: int, frame_number: int, timestamp: float, bbox, confidence: float, state: TrackState, frame_thumbnail = None) -> VehicleMemory:
        """Appends new tracking measurements, updates statistics, and limits snapshot lengths."""
        memory = self.repository.get(track_id)
        if memory is None:
            raise ValueError(f"Memory with Track ID {track_id} not found.")

        # Update temporal only if the vehicle is actively observed
        is_observed = state not in (TrackState.TEMPORARILY_LOST, TrackState.REMOVED)
        if is_observed:
            memory.last_seen_frame = frame_number
            memory.observation_count += 1
        
        # Calculate spatial displacements
        curr_center = memory.center
        if is_observed:
            prev_center = memory.center
            memory.bbox = bbox
            curr_center = memory.center
            if memory.initial_center is None:
                memory.initial_center = curr_center
            disp = prev_center.distance_to(curr_center)
            memory.path_length_accumulator += disp

        # Update state transitions
        if memory.state == TrackState.TEMPORARILY_LOST and state == TrackState.RECOVERED:
            memory.recovery_count += 1
        elif state == TrackState.TEMPORARILY_LOST:
            memory.occlusion_count += 1
            memory.frames_lost += 1
            
        memory.state = state
        memory.confidence = confidence

        # Confidence stats
        if is_observed:
            memory._confidence_sum += confidence
            if confidence > memory._highest_confidence:
                memory._highest_confidence = confidence
            if confidence < memory._lowest_confidence:
                memory._lowest_confidence = confidence

        # Create snapshot
        snapshot = MemorySnapshot(
            frame_number=frame_number,
            timestamp=timestamp,
            bbox=bbox if is_observed else memory.bbox,
            center=curr_center,
            area=bbox.area if is_observed else memory.bbox.area,
            confidence=confidence,
            state=state,
            frame_thumbnail=frame_thumbnail
        )
        memory.snapshots.append(snapshot)

        # Enforce memory capacity bounds
        if len(memory.snapshots) > self.max_snapshots:
            memory.snapshots.pop(0)

        # Cleanups scheduler
        self.frame_counter += 1
        if self.frame_counter % self.cleanup_interval == 0:
            self.cleanup_expired_memories()

        return memory

    def cleanup_expired_memories(self) -> None:
        """Triggers scans to remove inactive memory records."""
        # For this phase, cleanup is a lifecycle hook, which will be populated in downstream phases.
        pass

    def get_statistics(self) -> MemoryStatistics:
        """Calculates aggregate memory statistics."""
        memories = self.repository.get_all()
        total = len(memories)
        if total == 0:
            return MemoryStatistics()

        longest_life = 0
        total_life = 0
        total_obs = 0
        total_conf = 0.0
        total_path = 0.0
        recovered_count = 0

        for m in memories:
            age = m.track_age
            total_life += age
            if age > longest_life:
                longest_life = age
            total_obs += m.observation_count
            total_conf += m.average_confidence()
            total_path += m.path_length_accumulator
            recovered_count += m.recovery_count

        # Rough memory usage estimate
        est_bytes = sum(sys.getsizeof(m) for m in memories)

        return MemoryStatistics(
            total_memories=total,
            average_lifetime_frames=total_life / total,
            longest_lifetime_frames=longest_life,
            recovered_memories_count=recovered_count,
            average_observations=total_obs / total,
            average_confidence=total_conf / total,
            average_path_length=total_path / total,
            estimated_memory_bytes=est_bytes
        )
