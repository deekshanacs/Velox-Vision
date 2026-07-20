import statistics
from enum import Enum
from typing import List


class SmoothingMethod(str, Enum):
    """Supported noise filtering and smoothing algorithms."""
    MOVING_AVERAGE = "MOVING_AVERAGE"
    EXPONENTIAL = "EXPONENTIAL"
    MEDIAN = "MEDIAN"


class SpeedSmoother:
    """Applies noise filtering, outlier rejection, and temporal smoothing to speed measurements."""

    def __init__(
        self,
        method: SmoothingMethod = SmoothingMethod.MOVING_AVERAGE,
        window_size: int = 8,
        alpha: float = 0.3,
        max_speed_jump_kmh: float = 60.0
    ):
        self.method = method
        self.window_size = max(1, window_size)
        self.alpha = max(0.01, min(1.0, alpha))
        self.max_speed_jump_kmh = max_speed_jump_kmh

    def filter_outliers(self, raw_speed_kmh: float, speed_history: List[float]) -> float:
        """Rejects unnatural speed jumps or tracking jitter spikes."""
        if not speed_history:
            return raw_speed_kmh

        recent_prev = speed_history[-1]
        delta = abs(raw_speed_kmh - recent_prev)

        if delta > self.max_speed_jump_kmh:
            # Clamp or cap the sudden jump to maximum realistic change
            sign = 1.0 if raw_speed_kmh > recent_prev else -1.0
            return recent_prev + sign * self.max_speed_jump_kmh

        return raw_speed_kmh

    def smooth(self, raw_speed_kmh: float, speed_history: List[float]) -> float:
        """Applies outlier rejection and the selected smoothing filter algorithm."""
        cleaned_speed = self.filter_outliers(raw_speed_kmh, speed_history)
        working_series = speed_history + [cleaned_speed]
        recent_window = working_series[-self.window_size:]

        if self.method == SmoothingMethod.MEDIAN:
            return statistics.median(recent_window)
        elif self.method == SmoothingMethod.EXPONENTIAL:
            if len(speed_history) == 0:
                return cleaned_speed
            smoothed = speed_history[0]
            for s in working_series[1:]:
                smoothed = self.alpha * s + (1.0 - self.alpha) * smoothed
            return smoothed
        else:  # MOVING_AVERAGE
            return sum(recent_window) / len(recent_window)
