import logging
from typing import Dict, List, Any
from src.engines.tracking.metrics.performance_metrics import TrackingPerformance
from src.engines.tracking.entities.tracked_vehicle import TrackedVehicle

logger = logging.getLogger(__name__)

class TrackingDashboard:
    """Aggregates and formats live tracking telemetry metrics and breakdowns."""

    def __init__(self):
        self.class_breakdown: Dict[str, int] = {}

    def update_class_breakdown(self, tracked_vehicles: List[TrackedVehicle]) -> None:
        """Updates the active class-wise vehicle counts."""
        self.class_breakdown.clear()
        for vehicle in tracked_vehicles:
            cname = vehicle.class_name.lower()
            self.class_breakdown[cname] = self.class_breakdown.get(cname, 0) + 1

    def get_telemetry_dict(
        self,
        overall_fps: float,
        det_fps: float,
        perf: TrackingPerformance,
        stats: Any,
        memory_mgr: Any
    ) -> Dict[str, Any]:
        """Compiles a flat dictionary of system telemetry metrics."""
        avg_conf = 0.0
        longest_age = 0
        memory_size_kb = 0.0

        if memory_mgr is not None:
            mem_stats = memory_mgr.get_statistics()
            avg_conf = mem_stats.average_confidence * 100.0
            longest_age = mem_stats.longest_lifetime_frames
            memory_size_kb = mem_stats.estimated_memory_bytes / 1024.0

        lost_count = 0
        if stats and hasattr(stats, "tracks_lost"):
            lost_count = stats.tracks_lost

        active_count = perf.current_active_tracks if perf else 0
        total_created = stats.tracks_created if stats else 0
        recovered_count = stats.recovered_tracks if stats else 0

        return {
            "fps": overall_fps,
            "det_fps": det_fps,
            "track_fps": perf.tracking_fps if perf else 0.0,
            "active_tracks": active_count,
            "lost_tracks": lost_count,
            "recovered_tracks": recovered_count,
            "total_tracks": total_created,
            "longest_lifetime": longest_age,
            "average_confidence": avg_conf,
            "memory_usage_kb": memory_size_kb,
            "class_breakdown": self.class_breakdown.copy()
        }

from typing import Any
