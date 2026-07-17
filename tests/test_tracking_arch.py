import unittest
import numpy as np

from src.core.entities import BoundingBox, DetectionResult
from src.engines.tracking import (
    Tracker,
    TrackerFactory,
    TrackState,
    TrackHistory,
    TrackedVehicle,
    TrackingContext,
    TrackingResult,
    TrackingConfiguration,
    TrackCreated,
    TrackingError,
    TrackerInitializationError
)
from src.engines.tracking.value_objects.point import Point
from src.engines.tracking.value_objects.velocity import Velocity
from src.engines.tracking.value_objects.frame_metadata import FrameMetadata
from src.engines.tracking.value_objects.track_snapshot import TrackSnapshot
from src.engines.tracking.metrics.tracking_statistics import TrackingStatistics
from src.engines.tracking.metrics.performance_metrics import TrackingPerformance

class TestTrackingArchitecture(unittest.TestCase):
    """Unit test suite verifying the Multi-Object Tracking Engine Architectural Foundation."""

    def test_value_objects_instantiation(self):
        """Verify Point, Velocity, FrameMetadata, and TrackSnapshot instantiate correctly."""
        # 1. Point
        pt1 = Point(x=10.0, y=20.0)
        pt2 = Point(x=13.0, y=24.0)
        self.assertAlmostEqual(pt1.distance_to(pt2), 5.0)
        self.assertEqual(pt1.to_tuple(), (10.0, 20.0))

        # 2. Velocity
        vel = Velocity(vx=3.0, vy=4.0)
        self.assertAlmostEqual(vel.speed, 5.0)
        self.assertEqual(vel.to_tuple(), (3.0, 4.0))

        # 3. FrameMetadata
        meta = FrameMetadata(
            frame_number=1,
            timestamp=12345.67,
            width=1280,
            height=720,
            fps=30.0
        )
        self.assertEqual(meta.resolution, (1280, 720))

        # 4. TrackSnapshot
        bbox = BoundingBox(x1=10, y1=20, x2=50, y2=80)
        snap = TrackSnapshot(
            frame_number=1,
            timestamp=12345.67,
            center=pt1,
            bbox=bbox,
            confidence=0.95,
            state=TrackState.TENTATIVE
        )
        self.assertEqual(snap.state, TrackState.TENTATIVE)

    def test_tracking_configuration_resolver(self):
        """Verify TrackingConfiguration resolves parameters from system configs correctly."""
        config = TrackingConfiguration.from_settings()
        self.assertTrue(config.enabled)
        self.assertEqual(config.tracker_type, "bytetrack")
        self.assertEqual(config.history_size, 30)

    def test_track_history_capacity_constraints(self):
        """Verify TrackHistory queue implements max capacity queue limits."""
        history = TrackHistory(max_size=3)
        bbox = BoundingBox(x1=10, y1=20, x2=50, y2=80)
        
        for i in range(5):
            snap = TrackSnapshot(
                frame_number=i,
                timestamp=float(i),
                center=Point(x=float(i), y=float(i)),
                bbox=bbox,
                confidence=0.90,
                state=TrackState.TRACKED
            )
            history.append(snap)

        self.assertEqual(history.length, 3)
        self.assertEqual(history.snapshots[0].frame_number, 2)
        self.assertEqual(history.snapshots[-1].frame_number, 4)

    def test_tracked_vehicle_centers(self):
        """Verify TrackedVehicle dynamically computes current and previous centers."""
        bbox1 = BoundingBox(x1=10, y1=20, x2=50, y2=80)
        vehicle = TrackedVehicle(
            track_id=1,
            class_name="car",
            class_id=2,
            bbox=bbox1,
            confidence=0.90,
            state=TrackState.CONFIRMED,
            first_seen_frame=1,
            last_seen_frame=1
        )
        
        # Test current center
        self.assertAlmostEqual(vehicle.center.x, 30.0)
        self.assertAlmostEqual(vehicle.center.y, 50.0)
        self.assertIsNone(vehicle.previous_center)

        # Append snapshots to history
        snap1 = TrackSnapshot(1, 1.0, vehicle.center, bbox1, 0.90, TrackState.CONFIRMED)
        vehicle.history.append(snap1)
        
        bbox2 = BoundingBox(x1=20, y1=30, x2=60, y2=90)
        vehicle.bbox = bbox2
        snap2 = TrackSnapshot(2, 2.0, vehicle.center, bbox2, 0.92, TrackState.CONFIRMED)
        vehicle.history.append(snap2)

        self.assertAlmostEqual(vehicle.center.x, 40.0)
        self.assertAlmostEqual(vehicle.previous_center.x, 30.0)

    def test_tracking_context_and_result(self):
        """Verify that context inputs and results structures map values cleanly."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        det_result = DetectionResult(frame_number=1, timestamp=1.0, detections=[], metrics=None, model_name="")
        metadata = FrameMetadata(1, 1.0, 100, 100, 30.0)
        
        context = TrackingContext(frame=frame, detections=det_result, metadata=metadata)
        self.assertIs(context.frame, frame)
        self.assertIs(context.detections, det_result)

        tracking_res = TrackingResult(frame_number=1, timestamp=1.0, tracked_vehicles=[], tracking_latency_ms=5.0)
        self.assertEqual(tracking_res.active_tracks_count, 0)

    def test_tracking_statistics_resets(self):
        """Verify TrackingStatistics counters increment and reset."""
        stats = TrackingStatistics(tracks_created=10, tracks_removed=2, total_track_age=100)
        self.assertAlmostEqual(stats.average_track_age, 10.0)
        
        stats.reset()
        self.assertEqual(stats.tracks_created, 0)
        self.assertEqual(stats.average_track_age, 0.0)

    def test_tracking_performance_fps(self):
        """Verify TrackingPerformance FPS calculations."""
        perf = TrackingPerformance(total_latency_ms=50.0, processed_frames=5)
        self.assertAlmostEqual(perf.average_latency_ms, 10.0)
        self.assertAlmostEqual(perf.tracking_fps, 100.0)

    def test_tracker_factory_methods(self):
        """Verify factory returns list of supported trackers and throws error on implementation calls."""
        available = TrackerFactory.available_trackers()
        self.assertIn("bytetrack", available)
        self.assertIn("deepsort", available)

        # Confirm ValueError on invalid tracker types
        with self.assertRaises(ValueError):
            TrackerFactory.create_tracker(tracker_type="invalid_tracker")

        # Confirm NotImplementedError on unimplemented valid tracker types
        with self.assertRaises(NotImplementedError):
            TrackerFactory.create_tracker(tracker_type="deepsort")

    def test_custom_exceptions_routing(self):
        """Verify custom tracking exceptions correctly subclass TrackingError."""
        with self.assertRaises(TrackingError):
            raise TrackerInitializationError("Failed to initialize tracker")

if __name__ == "__main__":
    unittest.main()
