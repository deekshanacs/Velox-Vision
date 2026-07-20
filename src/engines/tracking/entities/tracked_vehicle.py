import time
from dataclasses import dataclass, field
from typing import Optional, List
from src.core.entities import BoundingBox
from src.engines.tracking.entities.track_state import TrackState
from src.engines.tracking.value_objects.velocity import Velocity
from src.engines.tracking.value_objects.point import Point
from src.engines.tracking.memory.vehicle_memory import VehicleMemory
from src.engines.tracking.motion.motion_history import MotionProfile
from src.engines.tracking.entities.perception_profile import PerceptionProfile


@dataclass
class TrackedVehicle:
    track_id: int
    memory: VehicleMemory = field(default=None)

    def __init__(
        self,
        track_id: int,
        memory: Optional[VehicleMemory] = None,
        class_name: Optional[str] = None,
        class_id: Optional[int] = None,
        bbox: Optional[BoundingBox] = None,
        confidence: Optional[float] = 0.0,
        state: Optional[TrackState] = TrackState.TENTATIVE,
        first_seen_frame: Optional[int] = 0,
        last_seen_frame: Optional[int] = 0,
        track_age: Optional[int] = 1,
        is_occluded: bool = False,
        **kwargs
    ):
        self.track_id = track_id
        self._motion_profile = None
        self._speed_profile = None
        self._perception_profile = PerceptionProfile()

        if memory is not None:
            self.memory = memory
        else:
            self.memory = VehicleMemory(
                track_id=track_id,
                vehicle_class=class_name or "car",
                class_id=class_id or 2,
                creation_timestamp=time.time(),
                first_seen_frame=first_seen_frame,
                last_seen_frame=last_seen_frame,
                confidence=confidence,
                confidence_sum=confidence,
                highest_confidence=confidence,
                lowest_confidence=confidence,
                bbox=bbox,
                state=state,
                is_occluded=is_occluded
            )

    @property
    def class_name(self) -> str:
        return self.memory.vehicle_class

    @property
    def class_id(self) -> int:
        return self.memory.class_id

    @property
    def bbox(self) -> BoundingBox:
        return self.memory.bbox

    @bbox.setter
    def bbox(self, value: BoundingBox) -> None:
        self.memory.bbox = value

    @property
    def confidence(self) -> float:
        return self.memory.confidence

    @confidence.setter
    def confidence(self, value: float) -> None:
        self.memory.confidence = value

    @property
    def state(self) -> TrackState:
        return self.memory.state

    @state.setter
    def state(self, value: TrackState) -> None:
        self.memory.state = value

    @property
    def first_seen_frame(self) -> int:
        return self.memory.first_seen_frame

    @property
    def last_seen_frame(self) -> int:
        return self.memory.last_seen_frame

    @last_seen_frame.setter
    def last_seen_frame(self, value: int) -> None:
        self.memory.last_seen_frame = value

    @property
    def track_age(self) -> int:
        return self.memory.track_age

    @track_age.setter
    def track_age(self, value: int) -> None:
        pass

    @property
    def is_occluded(self) -> bool:
        return self.memory.is_occluded

    @is_occluded.setter
    def is_occluded(self, value: bool) -> None:
        self.memory.is_occluded = value

    @property
    def estimated_speed_kmh(self) -> Optional[float]:
        return self.memory.estimated_speed_kmh

    @estimated_speed_kmh.setter
    def estimated_speed_kmh(self, value: Optional[float]) -> None:
        self.memory.estimated_speed_kmh = value

    @property
    def license_plate_text(self) -> Optional[str]:
        return self.memory.license_plate_text

    @license_plate_text.setter
    def license_plate_text(self, value: Optional[str]) -> None:
        self.memory.license_plate_text = value

    @property
    def center(self) -> Point:
        return self.memory.center

    @property
    def history(self):
        class HistoryWrapper:
            def __init__(self, memory: VehicleMemory):
                self._memory = memory

            @property
            def snapshots(self):
                return self._memory.snapshots

            @property
            def length(self) -> int:
                return len(self._memory.snapshots)

            def append(self, snapshot) -> None:
                self._memory.snapshots.append(snapshot)
        return HistoryWrapper(self.memory)

    @property
    def previous_center(self) -> Optional[Point]:
        snaps = self.memory.snapshots
        if len(snaps) > 1:
            return snaps[-2].center
        return None

    @property
    def velocity(self) -> Optional[Velocity]:
        return None

    @property
    def attributes(self) -> dict:
        return {}

    @property
    def perception(self) -> PerceptionProfile:
        return self._perception_profile

    @property
    def perception_profile(self) -> PerceptionProfile:
        return self._perception_profile


    @property
    def motion_profile(self) -> Optional[MotionProfile]:
        return self._perception_profile.motion

    @motion_profile.setter
    def motion_profile(self, value: Optional[MotionProfile]) -> None:
        self._perception_profile.motion = value
        self._motion_profile = value

    @property
    def speed_profile(self):
        return self._perception_profile.speed

    @speed_profile.setter
    def speed_profile(self, value) -> None:
        self._perception_profile.speed = value
        self._speed_profile = value
        if value is not None and hasattr(value, 'current_speed_kmh'):
            self.memory.estimated_speed_kmh = value.current_speed_kmh

    @property
    def lane_profile(self):
        return self._perception_profile.lane

    @lane_profile.setter
    def lane_profile(self, value) -> None:
        self._perception_profile.lane = value


