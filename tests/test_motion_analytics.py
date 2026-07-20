import unittest
import numpy as np
import math
from src.core.entities import BoundingBox
from src.engines.tracking.entities.track_state import TrackState
from src.engines.tracking.entities.tracked_vehicle import TrackedVehicle
from src.engines.tracking.memory.vehicle_memory import VehicleMemory
from src.engines.tracking.memory.memory_snapshot import MemorySnapshot
from src.engines.tracking.value_objects.point import Point
from src.engines.tracking.motion.motion_state import MotionState
from src.engines.tracking.motion.motion_vector import MotionVector
from src.engines.tracking.motion.motion_engine import MotionEngine

class TestMotionAnalytics(unittest.TestCase):
    """Unit test suite verifying Phase 4.1 Motion Analytics Foundation layer."""

    def setUp(self):
        self.engine = MotionEngine(
            minimum_snapshots=5,
            stationary_threshold=2.5,
            heading_window=10,
            smoothing_window=8,
            confidence_threshold=0.5
        )

    def _create_mock_memory(self, centers, bboxes=None) -> VehicleMemory:
        memory = VehicleMemory(
            track_id=1,
            vehicle_class="car",
            class_id=2,
            creation_timestamp=1.0,
            first_seen_frame=1,
            last_seen_frame=len(centers),
            bbox=BoundingBox(10, 10, 30, 30),
            state=TrackState.TRACKED
        )
        
        snapshots = []
        for i, center in enumerate(centers):
            bbox = bboxes[i] if bboxes else BoundingBox(center.x - 10, center.y - 10, center.x + 10, center.y + 10)
            snapshots.append(
                MemorySnapshot(
                    frame_number=i + 1,
                    timestamp=(i + 1) * 0.04,
                    bbox=bbox,
                    center=center,
                    area=bbox.area,
                    confidence=0.85,
                    state=TrackState.TRACKED
                )
            )
        memory.snapshots = snapshots
        return memory

    def test_stationary_vehicle(self):
        """Verifies profile stats for a vehicle that remains in a static location."""
        centers = [Point(100.0, 200.0) for _ in range(10)]
        memory = self._create_mock_memory(centers)
        
        profile = self.engine.generate_profile(memory)
        self.assertIsNotNone(profile)
        self.assertEqual(profile.motion_state, MotionState.STATIONARY)
        self.assertEqual(profile.total_travelled_distance, 0.0)
        self.assertEqual(profile.net_displacement, 0.0)
        self.assertEqual(profile.current_motion_vector.magnitude, 0.0)
        self.assertEqual(profile.stationary_duration_frames, 9)

    def test_straight_path(self):
        """Verifies profile calculation for a vehicle traversing in a straight line."""
        # 12 frames moving 10 pixels dx per frame
        centers = [Point(100.0 + i * 10.0, 200.0) for i in range(12)]
        memory = self._create_mock_memory(centers)
        
        profile = self.engine.generate_profile(memory)
        self.assertIsNotNone(profile)
        self.assertEqual(profile.motion_state, MotionState.CONTINUOUS_MOVEMENT)
        self.assertAlmostEqual(profile.total_travelled_distance, 110.0)
        self.assertAlmostEqual(profile.net_displacement, 110.0)
        self.assertAlmostEqual(profile.path_efficiency, 1.0)
        self.assertAlmostEqual(profile.current_heading, 0.0)
        self.assertEqual(profile.moving_duration_frames, 11)

    def test_curved_path(self):
        """Verifies profile trajectory curvature statistics on a curved path."""
        # Vehicle traversing a quarter circle
        centers = []
        r = 100.0
        for i in range(10):
            angle = (i / 9.0) * (math.pi / 2.0)
            centers.append(Point(r * math.cos(angle), r * math.sin(angle)))
        memory = self._create_mock_memory(centers)
        
        profile = self.engine.generate_profile(memory)
        self.assertIsNotNone(profile)
        self.assertGreater(profile.trajectory_stats.trajectory_curvature, 0.0)
        self.assertLess(profile.path_efficiency, 1.0)
        self.assertGreater(profile.trajectory_stats.maximum_turn_angle, 0.0)

    def test_motion_vector_correctness(self):
        """Checks dx and dy vector magnitude and heading properties."""
        v = MotionVector(3.0, 4.0)
        self.assertEqual(v.magnitude, 5.0)
        self.assertAlmostEqual(v.heading_rad, math.atan2(4.0, 3.0))

    def test_state_transitions(self):
        """Verifies moving and stationary sequence durations accumulator."""
        # Profile begins with stationary frames
        centers1 = [Point(10.0, 10.0) for _ in range(5)]
        memory = self._create_mock_memory(centers1)
        profile = self.engine.generate_profile(memory)
        
        self.assertEqual(profile.motion_state, MotionState.STATIONARY)
        self.assertEqual(profile.stationary_duration_frames, 4)
        self.assertEqual(profile.moving_duration_frames, 0)
        
        # Next update makes it move
        centers2 = centers1 + [Point(20.0, 10.0)]
        memory2 = self._create_mock_memory(centers2)
        profile2 = self.engine.generate_profile(memory2, current_profile=profile)
        self.assertEqual(profile2.motion_state, MotionState.MOVING)
        self.assertEqual(profile2.stationary_duration_frames, 0)
        self.assertEqual(profile2.moving_duration_frames, 1)

    def test_config_parsing(self):
        """Verifies that tracking visualizer parses settings successfully."""
        from src.engines.tracking import TrackingConfiguration
        cfg = TrackingConfiguration.from_settings()
        self.assertTrue(cfg.motion_enabled)
        self.assertEqual(cfg.motion_minimum_snapshots, 5)
        self.assertEqual(cfg.motion_stationary_threshold, 2.5)
