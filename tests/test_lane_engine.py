import time
import pytest
from src.core.entities import BoundingBox
from src.engines.tracking.entities.track_state import TrackState
from src.engines.tracking.memory.vehicle_memory import VehicleMemory
from src.engines.tracking.memory.memory_snapshot import MemorySnapshot
from src.engines.tracking.value_objects.point import Point
from src.engines.tracking.motion.motion_engine import MotionEngine
from src.engines.tracking.speed.speed_engine import SpeedEngine
from src.engines.tracking.speed.calibration import SpeedCalibration
from src.engines.tracking.lane.lane_configuration import LaneConfiguration
from src.engines.tracking.lane.lane_assignment import LaneAssignmentStatus
from src.engines.tracking.lane.lane_model import RoadModel, Lane, LaneBoundary, LaneCenterline
from src.engines.tracking.lane.lane_validator import LaneValidator, LaneValidationError
from src.engines.tracking.lane.lane_geometry import LaneGeometryCalculator
from src.engines.tracking.lane.lane_engine import LaneEngine
from src.engines.tracking.lane.lane_profile import LaneProfile


def create_mock_memory(centers: list, track_id: int = 1, fps: float = 25.0) -> VehicleMemory:
    """Helper to construct VehicleMemory for testing."""
    memory = VehicleMemory(
        track_id=track_id,
        vehicle_class="car",
        class_id=2,
        creation_timestamp=1000.0,
        first_seen_frame=1,
        last_seen_frame=len(centers),
        confidence=0.9,
        bbox=BoundingBox(100.0, 100.0, 150.0, 150.0),
        state=TrackState.CONFIRMED
    )

    dt = 1.0 / fps
    for idx, (cx, cy) in enumerate(centers):
        bbox = BoundingBox(cx - 25.0, cy - 25.0, cx + 25.0, cy + 25.0)
        snap = MemorySnapshot(
            frame_number=idx + 1,
            timestamp=1000.0 + idx * dt,
            bbox=bbox,
            center=Point(cx, cy),
            area=bbox.area,
            confidence=0.9,
            state=TrackState.CONFIRMED
        )
        memory.snapshots.append(snap)

    return memory


class TestLaneEngine:

    def test_single_lane_assignment(self):
        """Verify vehicle center inside single lane gets ASSIGNED status."""
        config = LaneConfiguration(lane_count=1, lane_width=100.0, road_start_x=0.0, hysteresis_frames=1)
        lane_engine = LaneEngine(config=config)
        motion_engine = MotionEngine(minimum_snapshots=3)
        speed_engine = SpeedEngine()

        centers = [(50.0, 100 + i * 10) for i in range(5)]
        memory = create_mock_memory(centers)
        motion_profile = motion_engine.generate_profile(memory)
        speed_profile = speed_engine.compute_speed(motion_profile, memory)

        lane_profile = lane_engine.compute_lane_profile(memory, motion_profile, speed_profile)

        assert lane_profile is not None
        assert lane_profile.current_lane_id == 1
        assert lane_profile.assignment_status == LaneAssignmentStatus.ASSIGNED
        assert lane_profile.geometry_metrics.distance_to_centerline == 0.0

    def test_multiple_lanes_assignment(self):
        """Verify vehicles in different lanes are assigned to respective lane IDs."""
        config = LaneConfiguration(lane_count=3, lane_width=100.0, road_start_x=0.0, hysteresis_frames=1)
        lane_engine = LaneEngine(config=config)
        motion_engine = MotionEngine(minimum_snapshots=3)

        # Vehicle in Lane 2 (x=150 is between 100 and 200)
        centers = [(150.0, 100 + i * 10) for i in range(5)]
        memory = create_mock_memory(centers)
        motion_profile = motion_engine.generate_profile(memory)

        profile = lane_engine.compute_lane_profile(memory, motion_profile)
        assert profile is not None
        assert profile.current_lane_id == 2
        assert profile.assignment_status == LaneAssignmentStatus.ASSIGNED

    def test_lane_transition(self):
        """Verify vehicle moving from Lane 1 to Lane 2 triggers TRANSITIONING status and transition event."""
        config = LaneConfiguration(lane_count=2, lane_width=100.0, road_start_x=0.0, hysteresis_frames=2, boundary_margin=10.0)
        lane_engine = LaneEngine(config=config)
        motion_engine = MotionEngine(minimum_snapshots=3)

        # Start in Lane 1 (x=50), move to Lane 2 (x=150)
        centers_l1 = [(50.0, 100 + i * 10) for i in range(5)]
        centers_l2 = centers_l1 + [(150.0, 150 + i * 10) for i in range(5)]

        profile = None
        for count in range(3, len(centers_l2) + 1):
            sub_mem = create_mock_memory(centers_l2[:count])
            mp = motion_engine.generate_profile(sub_mem)
            profile = lane_engine.compute_lane_profile(sub_mem, mp)

        assert profile is not None
        assert profile.current_lane_id == 2
        assert profile.previous_lane_id == 1
        assert profile.transition_count >= 1

    def test_boundary_conditions(self):
        """Verify vehicle straddling near a lane boundary is classified as TRANSITIONING."""
        # Lane 1 (0 to 100), boundary at 100. Margin = 15. x = 98 is near boundary
        config = LaneConfiguration(lane_count=2, lane_width=100.0, road_start_x=0.0, boundary_margin=15.0, hysteresis_frames=1)
        lane_engine = LaneEngine(config=config)
        motion_engine = MotionEngine(minimum_snapshots=3)

        centers = [(98.0, 100 + i * 10) for i in range(5)]
        memory = create_mock_memory(centers)
        mp = motion_engine.generate_profile(memory)

        profile = lane_engine.compute_lane_profile(memory, mp)
        assert profile is not None
        assert profile.assignment_status == LaneAssignmentStatus.TRANSITIONING

    def test_outside_lane(self):
        """Verify vehicle center outside defined road boundaries gets OUTSIDE_LANE status."""
        config = LaneConfiguration(lane_count=2, lane_width=100.0, road_start_x=0.0, hysteresis_frames=1)
        lane_engine = LaneEngine(config=config)
        motion_engine = MotionEngine(minimum_snapshots=3)

        # x = 500 is far outside road (0 to 200)
        centers = [(500.0, 100 + i * 10) for i in range(5)]
        memory = create_mock_memory(centers)
        mp = motion_engine.generate_profile(memory)

        profile = lane_engine.compute_lane_profile(memory, mp)
        assert profile is not None
        assert profile.current_lane_id is None
        assert profile.assignment_status == LaneAssignmentStatus.OUTSIDE_LANE

    def test_static_vehicle_stability(self):
        """Verify parked static vehicle maintains stable lane assignment without spurious transitions."""
        config = LaneConfiguration(lane_count=2, lane_width=100.0, road_start_x=0.0, hysteresis_frames=2)
        lane_engine = LaneEngine(config=config)
        motion_engine = MotionEngine(minimum_snapshots=3)

        centers = [(50.0, 100.0) for _ in range(10)]
        memory = create_mock_memory(centers)
        mp = motion_engine.generate_profile(memory)

        profile = lane_engine.compute_lane_profile(memory, mp)
        assert profile is not None
        assert profile.current_lane_id == 1
        assert profile.transition_count == 0

    def test_oscillation_filtering(self):
        """Verify hysteresis prevents rapid lane toggling when jittering near boundary."""
        config = LaneConfiguration(lane_count=2, lane_width=100.0, road_start_x=0.0, hysteresis_frames=3, boundary_margin=5.0)
        lane_engine = LaneEngine(config=config)
        motion_engine = MotionEngine(minimum_snapshots=3)

        # Vehicle starts at x=50 (Lane 1), then jitters to x=105 (Lane 2) for only 1 frame
        centers = [(50.0, 100 + i * 10) for i in range(5)]
        memory1 = create_mock_memory(centers)
        mp1 = motion_engine.generate_profile(memory1)
        lane_engine.compute_lane_profile(memory1, mp1)

        # Single jitter frame into Lane 2
        centers.append((105.0, 160.0))
        memory2 = create_mock_memory(centers)
        mp2 = motion_engine.generate_profile(memory2)
        profile = lane_engine.compute_lane_profile(memory2, mp2)

        # Should remain in Lane 1 due to 3-frame hysteresis
        assert profile.current_lane_id == 1

    def test_multi_vehicle_scale_performance(self):
        """Verify processing 100 parallel vehicles runs under 1 ms per vehicle without cross-track contamination."""
        config = LaneConfiguration(lane_count=4, lane_width=100.0, road_start_x=0.0)
        lane_engine = LaneEngine(config=config)
        motion_engine = MotionEngine(minimum_snapshots=3)

        # Create 100 vehicles across 4 lanes
        memories = []
        motion_profiles = []
        for v_id in range(1, 101):
            lane_idx = (v_id % 4) + 1
            start_x = (lane_idx - 1) * 100.0 + 50.0
            centers = [(start_x, 100 + i * 10) for i in range(5)]
            mem = create_mock_memory(centers, track_id=v_id)
            mp = motion_engine.generate_profile(mem)
            memories.append(mem)
            motion_profiles.append(mp)

        t0 = time.perf_counter()
        profiles = []
        for mem, mp in zip(memories, motion_profiles):
            lp = lane_engine.compute_lane_profile(mem, mp)
            profiles.append(lp)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / 100.0

        assert elapsed_ms < 1.0
        assert len(profiles) == 100
        # Verify no cross-track state contamination
        for idx, lp in enumerate(profiles):
            expected_lane = ((idx + 1) % 4) + 1
            assert lp.current_lane_id == expected_lane

    def test_road_validator(self):
        """Verify LaneValidator rejects invalid or overlapping RoadModels."""
        config = LaneConfiguration(lane_count=2, lane_width=-50.0)
        with pytest.raises(LaneValidationError):
            LaneEngine(config=config)


    def test_input_isolation_rule(self):
        """Verify engine rejects raw detections or invalid inputs."""
        lane_engine = LaneEngine()
        raw_detection = {"bbox": [100, 100, 200, 200], "confidence": 0.9}

        with pytest.raises(TypeError):
            lane_engine.compute_lane_profile(raw_detection, raw_detection)
