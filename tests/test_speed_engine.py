import time
import pytest
from src.core.entities import BoundingBox
from src.engines.tracking.entities.track_state import TrackState
from src.engines.tracking.entities.tracked_vehicle import TrackedVehicle
from src.engines.tracking.memory.vehicle_memory import VehicleMemory
from src.engines.tracking.memory.memory_snapshot import MemorySnapshot
from src.engines.tracking.value_objects.point import Point
from src.engines.tracking.motion.motion_engine import MotionEngine
from src.engines.tracking.speed.speed_engine import SpeedEngine
from src.engines.tracking.speed.speed_profile import SpeedProfile, CalibrationMode
from src.engines.tracking.speed.calibration import SpeedCalibration
from src.engines.tracking.speed.perspective import PerspectiveTransformer
from src.engines.tracking.speed.smoothing import SpeedSmoother, SmoothingMethod
from src.engines.tracking.configuration.tracking_config import TrackingConfiguration


def create_mock_memory(centers: list, fps: float = 25.0) -> VehicleMemory:
    """Helper to construct VehicleMemory with specified center points at uniform time intervals."""
    memory = VehicleMemory(
        track_id=1,
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






class TestSpeedEngine:

    def test_constant_speed(self):
        """Verify constant movement generates consistent speed values in m/s and km/h."""
        # 10 frames moving 10 pixels right each frame (dt = 0.04s, ratio = 0.05 m/px -> 0.5m / 0.04s = 12.5 m/s = 45 km/h)
        centers = [(100 + i * 10, 200) for i in range(10)]
        memory = create_mock_memory(centers, fps=25.0)

        motion_engine = MotionEngine(minimum_snapshots=3)
        motion_profile = motion_engine.generate_profile(memory)
        assert motion_profile is not None

        calib = SpeedCalibration(pixel_to_meter_ratio=0.05)
        speed_engine = SpeedEngine(calibration=calib)
        config = TrackingConfiguration(frame_rate=25.0)

        profile = speed_engine.compute_speed(motion_profile, memory, config)
        assert profile is not None
        assert isinstance(profile, SpeedProfile)
        assert pytest.approx(profile.current_speed_kmh, abs=2.0) == 45.0
        assert pytest.approx(profile.current_speed_mps, abs=0.5) == 12.5

    def test_stationary_vehicle(self):
        """Verify stationary vehicle (motion below threshold) reports 0 km/h."""
        centers = [(100, 200) for _ in range(10)]
        memory = create_mock_memory(centers, fps=25.0)

        motion_engine = MotionEngine(minimum_snapshots=3)
        motion_profile = motion_engine.generate_profile(memory)
        assert motion_profile is not None

        speed_engine = SpeedEngine()
        profile = speed_engine.compute_speed(motion_profile, memory)

        assert profile is not None
        assert profile.current_speed_kmh == 0.0
        assert profile.current_speed_mps == 0.0
        assert profile.is_accelerating is False
        assert profile.is_decelerating is False

    def test_acceleration(self):
        """Verify increasing displacement generates positive acceleration."""
        centers = [(100, 100)]
        curr_x = 100
        for step in range(1, 10):
            curr_x += step * 5
            centers.append((curr_x, 100))

        speed_engine = SpeedEngine()
        motion_engine = MotionEngine(minimum_snapshots=3)
        profile = None

        for count in range(3, len(centers) + 1):
            sub_memory = create_mock_memory(centers[:count], fps=25.0)
            motion_profile = motion_engine.generate_profile(sub_memory)
            profile = speed_engine.compute_speed(motion_profile, sub_memory)

        assert profile is not None
        assert profile.acceleration_mps2 > 0.0
        assert profile.is_accelerating is True

    def test_deceleration(self):
        """Verify decreasing displacement generates positive deceleration."""
        centers = [(100, 100)]
        curr_x = 100
        for step in range(8, 0, -1):
            curr_x += step * 5
            centers.append((curr_x, 100))

        speed_engine = SpeedEngine()
        motion_engine = MotionEngine(minimum_snapshots=3)
        profile = None

        for count in range(3, len(centers) + 1):
            sub_memory = create_mock_memory(centers[:count], fps=25.0)
            motion_profile = motion_engine.generate_profile(sub_memory)
            profile = speed_engine.compute_speed(motion_profile, sub_memory)

        assert profile is not None
        assert profile.deceleration_mps2 > 0.0
        assert profile.is_decelerating is True

    def test_calibration_scaling(self):
        """Verify changing calibration pixel_to_meter_ratio scales speed output linearly."""
        centers = [(100 + i * 10, 200) for i in range(10)]
        memory = create_mock_memory(centers, fps=25.0)

        motion_engine = MotionEngine(minimum_snapshots=3)
        motion_profile = motion_engine.generate_profile(memory)

        # Scale 1: 0.05
        engine_1 = SpeedEngine(calibration=SpeedCalibration(pixel_to_meter_ratio=0.05))
        p1 = engine_1.compute_speed(motion_profile, memory)

        # Scale 2: 0.10 (double ratio = double speed)
        engine_2 = SpeedEngine(calibration=SpeedCalibration(pixel_to_meter_ratio=0.10))
        p2 = engine_2.compute_speed(motion_profile, memory)

        assert p1 is not None and p2 is not None
        assert pytest.approx(p2.current_speed_kmh, rel=1e-2) == p1.current_speed_kmh * 2.0

    def test_noise_and_outlier_filtering(self):
        """Verify outlier rejection prevents extreme position jitter spikes from corrupting speed."""
        centers = [(100 + i * 10, 200) for i in range(8)]
        centers.append((1000, 200))

        smoother = SpeedSmoother(max_speed_jump_kmh=40.0)
        speed_engine = SpeedEngine(smoother=smoother)
        motion_engine = MotionEngine(minimum_snapshots=3)
        profile = None

        for count in range(3, len(centers) + 1):
            sub_memory = create_mock_memory(centers[:count], fps=25.0)
            motion_profile = motion_engine.generate_profile(sub_memory)
            profile = speed_engine.compute_speed(motion_profile, sub_memory)

        assert profile is not None
        # Outlier is capped relative to previous normal frames
        assert profile.current_speed_kmh < 300.0


    def test_confidence_calculations(self):
        """Verify multi-factor confidence score calculation."""
        centers = [(100 + i * 10, 200) for i in range(10)]
        memory = create_mock_memory(centers, fps=25.0)

        motion_engine = MotionEngine(minimum_snapshots=3)
        motion_profile = motion_engine.generate_profile(memory)

        speed_engine = SpeedEngine()
        profile = speed_engine.compute_speed(motion_profile, memory)

        assert profile is not None
        assert 0.0 <= profile.speed_confidence <= 1.0
        assert 0.0 <= profile.calibration_confidence <= 1.0
        assert 0.0 <= profile.observation_confidence <= 1.0
        assert 0.0 <= profile.overall_confidence <= 1.0

    def test_performance_under_1ms(self):
        """Verify incremental speed update runs in under 1 millisecond per vehicle."""
        centers = [(100 + i * 10, 200) for i in range(30)]
        memory = create_mock_memory(centers, fps=25.0)

        motion_engine = MotionEngine(minimum_snapshots=3)
        motion_profile = motion_engine.generate_profile(memory)

        speed_engine = SpeedEngine()

        t0 = time.perf_counter()
        for _ in range(100):
            speed_engine.compute_speed(motion_profile, memory)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / 100.0

        assert elapsed_ms < 1.0

    def test_input_isolation_rule(self):
        """Verify engine rejects raw detections or invalid inputs."""
        speed_engine = SpeedEngine()
        raw_detection = {"bbox": [100, 100, 200, 200], "confidence": 0.9}

        with pytest.raises(TypeError):
            speed_engine.compute_speed(raw_detection, raw_detection)
