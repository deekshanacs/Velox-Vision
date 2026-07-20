import time
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class TrackingProfiler:
    """Profiles system execution latencies (detection, tracking, visualization rendering) and monitors latency budgets."""

    def __init__(self, budget_ms: float = 2.0):
        self.budget_ms = budget_ms
        self.latencies: Dict[str, float] = {}
        self.violating_frames_count = 0
        self.total_profiled_frames = 0

    def start_segment(self, name: str) -> float:
        """Helper to start timing a segment."""
        return time.perf_counter()

    def end_segment(self, name: str, start_time: float) -> float:
        """Ends timing and logs the latency in milliseconds."""
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        self.latencies[name] = duration_ms
        
        if name == "visualization":
            self.total_profiled_frames += 1
            if duration_ms > self.budget_ms:
                self.violating_frames_count += 1
                logger.warning(
                    f"[LATENCY BUDGET VIOLATION] Visualization took {duration_ms:.2f}ms "
                    f"(Budget: {self.budget_ms:.2f}ms)"
                )
        return duration_ms

    def get_latencies(self) -> Dict[str, float]:
        """Returns the current latency profile metrics."""
        return self.latencies.copy()

    def get_violation_rate(self) -> float:
        """Returns the ratio of frames violating the latency budget."""
        if self.total_profiled_frames == 0:
            return 0.0
        return self.violating_frames_count / self.total_profiled_frames
