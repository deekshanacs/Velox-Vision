import math
from dataclasses import dataclass
from typing import List, Tuple
from src.engines.tracking.motion.motion_vector import MotionVector


@dataclass(frozen=True)
class SpeedStatistics:
    """Calculated statistical metrics for vehicle speed over history."""
    current_speed_kmh: float
    average_speed_kmh: float
    peak_speed_kmh: float
    min_speed_kmh: float
    speed_variance: float
    acceleration_mps2: float
    deceleration_mps2: float
    is_accelerating: bool
    is_decelerating: bool
    velocity_vector_mps: MotionVector
    velocity_vector_kmh: MotionVector

    @classmethod
    def compute(
        cls,
        speed_history_kmh: List[float],
        dt: float,
        heading_rad: float
    ) -> 'SpeedStatistics':
        """Computes statistical metrics incrementally from speed history and current frame delta time."""
        if not speed_history_kmh:
            zero_vec = MotionVector(0.0, 0.0)
            return cls(
                current_speed_kmh=0.0,
                average_speed_kmh=0.0,
                peak_speed_kmh=0.0,
                min_speed_kmh=0.0,
                speed_variance=0.0,
                acceleration_mps2=0.0,
                deceleration_mps2=0.0,
                is_accelerating=False,
                is_decelerating=False,
                velocity_vector_mps=zero_vec,
                velocity_vector_kmh=zero_vec
            )

        current_kmh = speed_history_kmh[-1]
        avg_kmh = sum(speed_history_kmh) / len(speed_history_kmh)
        peak_kmh = max(speed_history_kmh)
        min_kmh = min(speed_history_kmh)

        if len(speed_history_kmh) > 1:
            variance = sum((s - avg_kmh) ** 2 for s in speed_history_kmh) / len(speed_history_kmh)
        else:
            variance = 0.0

        # Compute acceleration (m/s^2)
        if len(speed_history_kmh) >= 2 and dt > 0:
            prev_mps = (speed_history_kmh[-2] / 3.6)
            curr_mps = (current_kmh / 3.6)
            accel = (curr_mps - prev_mps) / dt
        else:
            accel = 0.0

        is_accel = accel > 0.05
        is_decel = accel < -0.05
        accel_mps2 = max(0.0, accel)
        decel_mps2 = max(0.0, -accel)

        # Velocity vectors using current speed and heading angle
        curr_mps = current_kmh / 3.6
        vx_mps = curr_mps * math.cos(heading_rad)
        vy_mps = curr_mps * math.sin(heading_rad)
        vx_kmh = current_kmh * math.cos(heading_rad)
        vy_kmh = current_kmh * math.sin(heading_rad)

        return cls(
            current_speed_kmh=round(current_kmh, 2),
            average_speed_kmh=round(avg_kmh, 2),
            peak_speed_kmh=round(peak_kmh, 2),
            min_speed_kmh=round(min_kmh, 2),
            speed_variance=round(variance, 4),
            acceleration_mps2=round(accel_mps2, 3),
            deceleration_mps2=round(decel_mps2, 3),
            is_accelerating=is_accel,
            is_decelerating=is_decel,
            velocity_vector_mps=MotionVector(round(vx_mps, 3), round(vy_mps, 3)),
            velocity_vector_kmh=MotionVector(round(vx_kmh, 2), round(vy_kmh, 2))
        )
