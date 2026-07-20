import unittest
import numpy as np
import os
import shutil
import tempfile
import json
from src.core.entities import BoundingBox
from src.engines.tracking.entities.track_state import TrackState
from src.engines.tracking.entities.tracked_vehicle import TrackedVehicle
from src.engines.tracking.memory.vehicle_memory import VehicleMemory
from src.engines.tracking.memory.memory_snapshot import MemorySnapshot
from src.engines.tracking.value_objects.point import Point
from src.engines.tracking.configuration.tracking_config import TrackingConfiguration
from src.engines.tracking.visualization.color_palette import get_state_color, STATE_COLORS
from src.engines.tracking.visualization.trail_renderer import TrailRenderer
from src.engines.tracking.telemetry.tracking_dashboard import TrackingDashboard
from src.engines.tracking.telemetry.tracking_profiler import TrackingProfiler
from src.engines.tracking.telemetry.tracking_report import TrackingReportGenerator
from src.engines.tracking.metrics.performance_metrics import TrackingPerformance
from src.engines.tracking.metrics.tracking_statistics import TrackingStatistics

class TestTrackingVisualization(unittest.TestCase):
    """Verifies Phase 3.4 Tracking Visualization and Telemetry aggregate layer logic."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.configs = TrackingConfiguration(
            viz_enabled=True,
            viz_show_trails=True,
            viz_trail_length=5,
            viz_show_memory=True,
            viz_show_states=True,
            viz_show_profiler=True,
            viz_show_dashboard=True,
            viz_debug=True
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_configuration_parsing(self):
        """Verifies visual config options exist on TrackingConfiguration."""
        self.assertTrue(self.configs.viz_enabled)
        self.assertTrue(self.configs.viz_show_trails)
        self.assertEqual(self.configs.viz_trail_length, 5)
        self.assertTrue(self.configs.viz_show_memory)
        self.assertTrue(self.configs.viz_show_states)
        self.assertTrue(self.configs.viz_show_profiler)
        self.assertTrue(self.configs.viz_show_dashboard)
        self.assertTrue(self.configs.viz_debug)

    def test_color_palette(self):
        """Verifies state color mapping and override mechanisms."""
        yellow = get_state_color(TrackState.TENTATIVE)
        self.assertEqual(yellow, (0, 255, 255))
        
        # Test override
        overrides = {TrackState.TENTATIVE: (1, 2, 3)}
        custom_color = get_state_color(TrackState.TENTATIVE, overrides=overrides)
        self.assertEqual(custom_color, (1, 2, 3))

    def test_trail_renderer(self):
        """Verifies trail generation loop runs safely without errors."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock a TrackedVehicle with snapshot centers
        memory = VehicleMemory(
            track_id=1,
            vehicle_class="car",
            class_id=2,
            creation_timestamp=1.0,
            first_seen_frame=1,
            last_seen_frame=3,
            bbox=BoundingBox(10, 10, 30, 30),
            state=TrackState.TRACKED
        )
        
        memory.snapshots = [
            MemorySnapshot(frame_number=1, timestamp=0.1, bbox=BoundingBox(10, 10, 30, 30), center=Point(20, 20), area=400, confidence=0.9, state=TrackState.TENTATIVE),
            MemorySnapshot(frame_number=2, timestamp=0.2, bbox=BoundingBox(12, 10, 32, 30), center=Point(22, 20), area=400, confidence=0.92, state=TrackState.CONFIRMED),
            MemorySnapshot(frame_number=3, timestamp=0.3, bbox=BoundingBox(14, 10, 34, 30), center=Point(24, 20), area=400, confidence=0.95, state=TrackState.TRACKED)
        ]
        
        vehicle = TrackedVehicle(track_id=1, memory=memory)
        
        renderer = TrailRenderer(trail_length=5)
        try:
            renderer.render_trail(frame, vehicle)
            success = True
        except Exception:
            success = False
        self.assertTrue(success)

    def test_tracking_dashboard(self):
        """Verifies telemetry calculations and dashboard output structures."""
        dashboard = TrackingDashboard()
        
        # Mock tracked vehicle list
        memory = VehicleMemory(1, "car", 2, 1.0, 1, 1, bbox=BoundingBox(10, 10, 30, 30))
        vehicle1 = TrackedVehicle(track_id=1, memory=memory)
        
        dashboard.update_class_breakdown([vehicle1])
        self.assertEqual(dashboard.class_breakdown.get("car"), 1)
        
        perf = TrackingPerformance(total_latency_ms=10.0, processed_frames=5, peak_active_tracks=2, current_active_tracks=1)
        stats = TrackingStatistics(tracks_created=3, tracks_removed=1, tracks_lost=1, recovered_tracks=1)
        
        telemetry = dashboard.get_telemetry_dict(
            overall_fps=30.0,
            det_fps=25.0,
            perf=perf,
            stats=stats,
            memory_mgr=None
        )
        
        self.assertEqual(telemetry["fps"], 30.0)
        self.assertEqual(telemetry["det_fps"], 25.0)
        self.assertEqual(telemetry["active_tracks"], 1)
        self.assertEqual(telemetry["lost_tracks"], 1)
        self.assertEqual(telemetry["recovered_tracks"], 1)
        self.assertEqual(telemetry["total_tracks"], 3)
        self.assertEqual(telemetry["class_breakdown"]["car"], 1)

    def test_tracking_profiler(self):
        """Verifies latency profiling and budget warning thresholds."""
        profiler = TrackingProfiler(budget_ms=1.0)
        start = profiler.start_segment("visualization")
        import time
        time.sleep(0.002) # 2ms
        profiler.end_segment("visualization", start)
        
        self.assertGreater(profiler.latencies.get("visualization", 0.0), 0.0)
        self.assertEqual(profiler.violating_frames_count, 1)
        self.assertEqual(profiler.get_violation_rate(), 1.0)

    def test_report_generator(self):
        """Verifies JSON summary write calls and frame snapshot exports."""
        gen = TrackingReportGenerator(output_dir=self.temp_dir)
        
        configs = self.configs
        perf_stats = {"pipeline_fps": 30.0}
        track_stats = {"tracks_created": 3}
        memory_stats = {"total_memories": 2}
        git_info = {"commit": "abcdefg", "branch": "main"}
        hardware_info = {"processor": "CPU Spec", "ram": "16 GB"}
        
        report_path = gen.generate_json_report(
            report_id="test_run",
            configs=configs,
            perf_stats=perf_stats,
            track_stats=track_stats,
            memory_stats=memory_stats,
            git_info=git_info,
            hardware_info=hardware_info
        )
        
        self.assertTrue(os.path.exists(report_path))
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["report_id"], "test_run")
        self.assertEqual(data["git"]["commit"], "abcdefg")
        self.assertEqual(data["performance"]["pipeline_fps"], 30.0)
        
        # Test snapshot saving
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        snapshot_path = gen.save_final_snapshot(frame, snapshot_dir=self.temp_dir)
        self.assertTrue(os.path.exists(snapshot_path))
