import unittest
import numpy as np

from src.core.entities import BoundingBox, DetectionResult, VehicleDetection
from src.engines.tracking import (
    TrackerFactory,
    TrackState,
    TrackingContext,
    TrackingConfiguration
)
from src.engines.tracking.value_objects.frame_metadata import FrameMetadata
from src.engines.tracking.implementations.bytetrack_tracker import ByteTrackTracker

class TestByteTrackIntegration(unittest.TestCase):
    """Integration test suite for ByteTrack tracker lifecycle operations."""

    def setUp(self):
        # Strongly-typed tracking config setup
        self.configs = TrackingConfiguration(
            enabled=True,
            tracker_type="bytetrack",
            history_size=30,
            max_lost_frames=3,      # Shortened for quick test exits
            tentative_frames=2,     # Shortened for confirmation tests
            track_activation_threshold=0.10,
            minimum_matching_threshold=0.5,
            track_buffer=5,
            min_box_area=5.0,
            frame_rate=25.0
        )
        self.tracker = TrackerFactory.create_tracker("bytetrack", self.configs)

    def _create_context(self, frame_num: int, boxes: list) -> TrackingContext:
        """Helper to create TrackingContext from a list of bounding boxes."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detections = []
        for box in boxes:
            x1, y1, x2, y2 = box
            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            det = VehicleDetection(bbox=bbox, confidence=0.90, class_name="car", class_id=2)
            detections.append(det)
            
        det_result = DetectionResult(
            detections=detections,
            metrics=None,
            frame_number=frame_num,
            timestamp=float(frame_num) * 0.04,
            model_name="yolo11n.pt"
        )
        metadata = FrameMetadata(
            frame_number=frame_num,
            timestamp=float(frame_num) * 0.04,
            width=1280,
            height=720,
            fps=25.0
        )
        return TrackingContext(frame=frame, detections=det_result, metadata=metadata)

    def test_factory_creation(self):
        """Verify factory compiles and returns ByteTrackTracker."""
        self.assertIsInstance(self.tracker, ByteTrackTracker)
        self.assertTrue(self.tracker.is_initialized)

    def test_id_persistence_and_entries(self):
        """Verify that stable IDs persist across frames for moving vehicles."""
        # Frame 1: Two separate cars
        ctx1 = self._create_context(1, [[100, 100, 150, 150], [400, 400, 450, 450]])
        res1 = self.tracker.track(ctx1)
        self.assertEqual(len(res1.tracked_vehicles), 2)
        
        # Extract initial IDs
        id_car1 = res1.tracked_vehicles[0].track_id
        id_car2 = res1.tracked_vehicles[1].track_id
        self.assertNotEqual(id_car1, id_car2)
        self.assertEqual(res1.tracked_vehicles[0].state, TrackState.TENTATIVE)

        # Frame 2: Slightly shifted boxes (simulating motion)
        ctx2 = self._create_context(2, [[102, 102, 152, 152], [402, 402, 452, 452]])
        res2 = self.tracker.track(ctx2)
        self.assertEqual(len(res2.tracked_vehicles), 2)
        
        # Verify IDs persisted stable
        id_car1_f2 = res2.tracked_vehicles[0].track_id
        id_car2_f2 = res2.tracked_vehicles[1].track_id
        self.assertEqual(id_car1, id_car1_f2)
        self.assertEqual(id_car2, id_car2_f2)
        
        # Verify confirmed transition state after tentative_frames=2 limit
        self.assertEqual(res2.tracked_vehicles[0].state, TrackState.CONFIRMED)

    def test_occlusion_and_recovery(self):
        """Verify that lost tracks are held and recovered under their original ID."""
        # Frame 1: Vehicle detected
        ctx1 = self._create_context(1, [[100, 100, 150, 150]])
        res1 = self.tracker.track(ctx1)
        self.assertEqual(len(res1.tracked_vehicles), 1)
        tid = res1.tracked_vehicles[0].track_id

        # Frame 2: Vehicle gets occluded (no detections matching)
        ctx2 = self._create_context(2, [])
        res2 = self.tracker.track(ctx2)
        
        # Active tracked vehicles should be empty in tracking result output,
        # but the tracker should cache it locally in lost state
        self.assertEqual(len(res2.tracked_vehicles), 0)
        self.assertEqual(len(self.tracker._active_tracks), 1)
        self.assertEqual(self.tracker._active_tracks[tid].state, TrackState.TEMPORARILY_LOST)
        self.assertEqual(self.tracker.get_statistics().tracks_lost, 1)

        # Frame 3: Vehicle reappears near previous location
        ctx3 = self._create_context(3, [[102, 102, 152, 152]])
        res3 = self.tracker.track(ctx3)
        self.assertEqual(len(res3.tracked_vehicles), 1)
        self.assertEqual(res3.tracked_vehicles[0].track_id, tid)
        self.assertEqual(res3.tracked_vehicles[0].state, TrackState.RECOVERED)
        self.assertEqual(self.tracker.get_statistics().recovered_tracks, 1)

    def test_exits_handling(self):
        """Verify that lost tracks are removed after exceeding max_lost_frames."""
        # Frame 1: Vehicle detected
        ctx1 = self._create_context(1, [[100, 100, 150, 150]])
        res1 = self.tracker.track(ctx1)
        self.assertEqual(len(res1.tracked_vehicles), 1)
        tid = res1.tracked_vehicles[0].track_id

        # Feed frame context gaps exceeding max_lost_frames = 3
        self.tracker.track(self._create_context(2, []))
        self.tracker.track(self._create_context(3, []))
        self.tracker.track(self._create_context(4, []))
        self.tracker.track(self._create_context(5, []))

        # Check repository cleanups
        self.assertNotIn(tid, self.tracker._active_tracks)
        self.assertEqual(self.tracker.get_statistics().tracks_removed, 1)

    def test_reset_behavior(self):
        """Verify resetting tracker clears active maps and stats."""
        ctx = self._create_context(1, [[100, 100, 150, 150]])
        self.tracker.track(ctx)
        self.assertEqual(len(self.tracker._active_tracks), 1)

        self.tracker.reset()
        self.assertEqual(len(self.tracker._active_tracks), 0)
        self.assertEqual(self.tracker.get_statistics().tracks_created, 0)

if __name__ == "__main__":
    unittest.main()
