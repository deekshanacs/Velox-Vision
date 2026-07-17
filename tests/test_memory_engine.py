import unittest
import time
from src.core.entities import BoundingBox
from src.engines.tracking.entities.track_state import TrackState
from src.engines.tracking.value_objects.point import Point
from src.engines.tracking.configuration.tracking_config import TrackingConfiguration
from src.engines.tracking.memory.vehicle_memory import VehicleMemory
from src.engines.tracking.memory.memory_snapshot import MemorySnapshot
from src.engines.tracking.memory.memory_manager import MemoryManager
from src.engines.tracking.repositories.vehicle_memory_repository import InMemoryVehicleMemoryRepository
from src.engines.tracking.entities.tracked_vehicle import TrackedVehicle


class TestVehicleMemoryEngine(unittest.TestCase):
    """Exhaustively verifies operations of the Vehicle Memory Subsystem."""

    def setUp(self):
        self.configs = TrackingConfiguration(
            memory_enabled=True,
            memory_max_snapshots=5,
            memory_cleanup_interval=10
        )
        self.repository = InMemoryVehicleMemoryRepository()
        self.manager = MemoryManager(configs=self.configs, repository=self.repository)

    def test_memory_creation_and_retrieval(self):
        """Verifies new persistent memories are created and retrieved correctly."""
        bbox = BoundingBox(x1=10, y1=10, x2=50, y2=50)
        
        # Test creation
        memory = self.manager.create_memory(
            track_id=1,
            vehicle_class="car",
            class_id=2,
            frame_number=10,
            timestamp=0.4,
            bbox=bbox,
            confidence=0.92
        )
        
        self.assertEqual(memory.track_id, 1)
        self.assertEqual(memory.vehicle_class, "car")
        self.assertEqual(memory.first_seen_frame, 10)
        self.assertEqual(memory.confidence, 0.92)
        
        # Test repository states
        self.assertTrue(self.repository.exists(1))
        retrieved = self.repository.get(1)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.track_id, 1)
        
        # Test exception on duplicate creation
        with self.assertRaises(ValueError):
            self.manager.create_memory(1, "car", 2, 10, 0.4, bbox, 0.92)

    def test_memory_updates_and_metrics(self):
        """Verifies spatial displacements, state changes, and confidence limits updates."""
        bbox1 = BoundingBox(x1=10, y1=10, x2=30, y2=30)
        memory = self.manager.create_memory(1, "car", 2, 1, 0.04, bbox1, 0.85)
        
        # Apply initial update
        self.manager.update_memory(1, 1, 0.04, bbox1, 0.85, TrackState.TENTATIVE)
        self.assertEqual(memory.observation_count, 2)  # initial create + first update
        
        # Center: (20, 20)
        self.assertEqual(memory.center.x, 20.0)
        
        # Update 2: Move to center (25, 20), displacement = 5.0
        bbox2 = BoundingBox(x1=15, y1=10, x2=35, y2=30)
        self.manager.update_memory(1, 2, 0.08, bbox2, 0.95, TrackState.CONFIRMED)
        
        self.assertAlmostEqual(memory.total_displacement(), 5.0)
        self.assertAlmostEqual(memory.net_displacement(), 5.0)
        self.assertAlmostEqual(memory.path_efficiency(), 1.0)
        
        self.assertEqual(memory.highest_confidence(), 0.95)
        self.assertEqual(memory.lowest_confidence(), 0.85)
        self.assertAlmostEqual(memory.average_confidence(), (0.85 + 0.85 + 0.95)/3)
        
        # Update 3: Move to (25, 25), displacement = 5.0 (total = 10.0)
        # Net displacement from (20, 20) to (25, 25) is sqrt(50) = 7.0711
        bbox3 = BoundingBox(x1=15, y1=15, x2=35, y2=35)
        self.manager.update_memory(1, 3, 0.12, bbox3, 0.90, TrackState.TRACKED)
        
        self.assertAlmostEqual(memory.total_displacement(), 10.0)
        self.assertAlmostEqual(memory.net_displacement(), 50 ** 0.5)
        self.assertAlmostEqual(memory.path_efficiency(), (50 ** 0.5) / 10.0)
        self.assertAlmostEqual(memory.confidence_stability(), 1.0 - ((0.005 / 3.0) ** 0.5), places=5)
        
        # Update 4: Transition to TEMPORARILY_LOST
        self.manager.update_memory(1, 4, 0.16, bbox3, 0.70, TrackState.TEMPORARILY_LOST)
        self.assertEqual(memory.occlusion_count, 1)
        self.assertEqual(memory.frames_lost, 1)
        
        # Update 5: RECOVERED transition
        self.manager.update_memory(1, 5, 0.20, bbox3, 0.90, TrackState.RECOVERED)
        self.assertEqual(memory.recovery_count, 1)

    def test_snapshots_bounds_capping(self):
        """Verifies snapshot list length is capped according to max configs."""
        bbox = BoundingBox(x1=10, y1=10, x2=30, y2=30)
        memory = self.manager.create_memory(1, "car", 2, 1, 0.04, bbox, 0.85)
        
        # Apply 8 updates (config limit is set to 5)
        for i in range(1, 9):
            self.manager.update_memory(1, i, i * 0.04, bbox, 0.85, TrackState.TRACKED)
            
        self.assertEqual(len(memory.snapshots), 5)
        # Verify oldest snapshots were popped (leftmost element should be frame 4)
        self.assertEqual(memory.snapshots[0].frame_number, 4)
        self.assertEqual(memory.snapshots[-1].frame_number, 8)

    def test_global_statistics(self):
        """Verifies global aggregation statistics calculations."""
        bbox = BoundingBox(x1=10, y1=10, x2=30, y2=30)
        
        # Vehicle 1: 5 frames age
        self.manager.create_memory(1, "car", 2, 1, 0.04, bbox, 0.80)
        self.manager.update_memory(1, 1, 0.04, bbox, 0.80, TrackState.TRACKED)
        self.manager.update_memory(1, 5, 0.20, bbox, 0.90, TrackState.TRACKED)
        
        # Vehicle 2: 10 frames age
        self.manager.create_memory(2, "truck", 7, 1, 0.04, bbox, 0.70)
        self.manager.update_memory(2, 10, 0.40, bbox, 0.70, TrackState.TRACKED)
        
        stats = self.manager.get_statistics()
        self.assertEqual(stats.total_memories, 2)
        self.assertEqual(stats.longest_lifetime_frames, 10)
        self.assertAlmostEqual(stats.average_lifetime_frames, 7.5)

    def test_tracked_vehicle_memory_integration(self):
        """Verifies TrackedVehicle delegates properties correctly to VehicleMemory."""
        bbox = BoundingBox(x1=10, y1=10, x2=30, y2=30)
        memory = self.manager.create_memory(1, "car", 2, 1, 0.04, bbox, 0.95)
        self.manager.update_memory(1, 1, 0.04, bbox, 0.95, TrackState.CONFIRMED)
        
        vehicle = TrackedVehicle(track_id=1, memory=memory)
        
        self.assertEqual(vehicle.class_name, "car")
        self.assertEqual(vehicle.class_id, 2)
        self.assertEqual(vehicle.confidence, 0.95)
        self.assertEqual(vehicle.state, TrackState.CONFIRMED)
        
        # Test setter updates
        new_bbox = BoundingBox(x1=15, y1=15, x2=35, y2=35)
        vehicle.bbox = new_bbox
        self.assertEqual(memory.bbox, new_bbox)
        self.assertEqual(vehicle.center.x, 25.0)


if __name__ == "__main__":
    unittest.main()
