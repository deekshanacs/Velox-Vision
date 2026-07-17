import time
from typing import List, Tuple
import numpy as np
import supervision as sv

from src.core.entities import BoundingBox
from src.engines.tracking.configuration.tracking_config import TrackingConfiguration
from src.engines.tracking.entities.tracking_context import TrackingContext
from src.engines.tracking.entities.tracked_vehicle import TrackedVehicle
from src.engines.tracking.entities.track_state import TrackState

# Target class ID mapping for vehicle naming
CLASS_ID_TO_NAME = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

class ByteTrackAdapter:
    """Translates between Velox Vision domain models and supervision.ByteTrack.
    
    CRITICAL BOUNDARY: The supervision library is strictly isolated inside this adapter.
    """

    def __init__(self, configs: TrackingConfiguration):
        self._configs = configs
        self._init_tracker()

    def _init_tracker(self) -> None:
        """Helper to initialize the ByteTrack instance."""
        self._tracker = sv.ByteTrack(
            track_activation_threshold=self._configs.track_activation_threshold,
            lost_track_buffer=self._configs.track_buffer,
            minimum_matching_threshold=self._configs.minimum_matching_threshold,
            frame_rate=self._configs.frame_rate,
            minimum_consecutive_frames=1  # Rely on the core tracker state machine for tentative frames
        )

    def update(self, context: TrackingContext) -> Tuple[List[TrackedVehicle], float]:
        """Runs supervision ByteTrack update on the context and translates the output.
        
        Args:
            context: The current frame tracking context containing detections.
            
        Returns:
            Tuple of:
              - A list of TrackedVehicle domain entities containing raw bounding box
                coordinates and tracker IDs.
              - The execution time in milliseconds.
        """
        start_time = time.perf_counter()
        
        detections_result = context.detections
        xyxy_list = []
        confidence_list = []
        class_id_list = []

        for det in detections_result.detections:
            # Enforce bounding area threshold
            if det.bbox.area >= self._configs.min_box_area:
                xyxy_list.append([det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2])
                confidence_list.append(det.confidence)
                class_id_list.append(det.class_id)

        if not xyxy_list:
            xyxy_array = np.empty((0, 4), dtype=np.float32)
            confidence_array = np.empty((0,), dtype=np.float32)
            class_id_array = np.empty((0,), dtype=np.int32)
        else:
            xyxy_array = np.array(xyxy_list, dtype=np.float32)
            confidence_array = np.array(confidence_list, dtype=np.float32)
            class_id_array = np.array(class_id_list, dtype=np.int32)

        sv_detections = sv.Detections(
            xyxy=xyxy_array,
            confidence=confidence_array,
            class_id=class_id_array
        )

        # Execute ByteTrack
        tracked_detections = self._tracker.update_with_detections(sv_detections)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        tracked_vehicles = []
        frame_num = context.metadata.frame_number

        if tracked_detections.tracker_id is not None:
            for idx in range(len(tracked_detections)):
                tid = int(tracked_detections.tracker_id[idx])
                cid = int(tracked_detections.class_id[idx])
                conf = float(tracked_detections.confidence[idx])
                
                bbox = BoundingBox(
                    x1=float(tracked_detections.xyxy[idx][0]),
                    y1=float(tracked_detections.xyxy[idx][1]),
                    x2=float(tracked_detections.xyxy[idx][2]),
                    y2=float(tracked_detections.xyxy[idx][3])
                )
                
                # Instantiating domain entity representation
                vehicle = TrackedVehicle(
                    track_id=tid,
                    class_name=CLASS_ID_TO_NAME.get(cid, "car"),
                    class_id=cid,
                    bbox=bbox,
                    confidence=conf,
                    state=TrackState.TENTATIVE, # Default state; tracker will reconcile
                    first_seen_frame=frame_num,
                    last_seen_frame=frame_num
                )
                tracked_vehicles.append(vehicle)

        return tracked_vehicles, latency_ms

    def reset(self) -> None:
        """Resets tracking by re-initializing the supervision tracker."""
        self._init_tracker()
