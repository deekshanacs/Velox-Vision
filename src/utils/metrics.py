import time

class PerformanceTracker:
    """Performance tracker utility for computing real-time latency and frame rates."""

    @staticmethod
    def calculate_fps(total_latency_ms: float) -> float:
        """Computes frames per second (FPS) from total processing latency in milliseconds.
        
        Args:
            total_latency_ms: Execution duration in milliseconds.
            
        Returns:
            Computed frame rate (FPS).
        """
        if total_latency_ms <= 0:
            return 0.0
        return 1000.0 / total_latency_ms
