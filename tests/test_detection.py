import sys
from unittest.mock import MagicMock

# Mock the ultralytics module to prevent loading torch and its blocked DLLs
mock_ultralytics = MagicMock()
mock_yolo_class = MagicMock()
mock_ultralytics.YOLO = mock_yolo_class
sys.modules['ultralytics'] = mock_ultralytics

import unittest
import numpy as np

from src.core.entities import BoundingBox, VehicleDetection, DetectionResult, DetectionMetrics
from src.engines.vehicle_detection.detector_factory import VehicleDetectorFactory
from src.engines.vehicle_detection.vehicle_detector import YOLOVehicleDetector
from src.engines.vehicle_detection.exceptions import ModelLoadError, InvalidFrameError, InferenceFailedError

class TestVehicleDetection(unittest.TestCase):
    """Rigorous unit tests for the production-grade Vehicle Detection Engine."""

    def setUp(self):
        # Reset mock parameters to isolate tests
        mock_yolo_class.reset_mock()
        mock_yolo_class.side_effect = None
        mock_yolo_class.return_value = MagicMock()

    def test_bounding_box_properties(self):
        """Test BoundingBox dynamically computes properties (width, height, area, center)."""
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=80.0)
        self.assertAlmostEqual(bbox.width, 40.0)
        self.assertAlmostEqual(bbox.height, 60.0)
        self.assertAlmostEqual(bbox.area, 2400.0)
        self.assertAlmostEqual(bbox.center_x, 30.0)
        self.assertAlmostEqual(bbox.center_y, 50.0)

    def test_factory_creates_yolo_detector(self):
        """Test factory instantiates YOLOVehicleDetector with parameters."""
        detector = VehicleDetectorFactory.create_detector(
            detector_type="yolo",
            model_path="yolo11n.pt",
            confidence_threshold=0.35,
            warmup=False
        )
        self.assertIsInstance(detector, YOLOVehicleDetector)
        mock_yolo_class.assert_called_once_with("yolo11n.pt")

    def test_detector_filters_vehicle_classes(self):
        """Test target class mapping and DetectionResult wrappers."""
        # Setup mock predict output
        mock_box_car = MagicMock()
        mock_box_car.xyxy = [MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=np.array([10.0, 20.0, 100.0, 200.0])))))]
        mock_box_car.conf = [MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=np.array(0.85)))))]
        mock_box_car.cls = [MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=np.array(2)))))] # COCO 2: car

        mock_result = MagicMock()
        mock_result.boxes = [mock_box_car]
        mock_result.speed = {'preprocess': 1.5, 'inference': 10.0, 'postprocess': 2.0}
        
        mock_yolo_instance = mock_yolo_class.return_value
        mock_yolo_instance.predict.return_value = [mock_result]

        detector = YOLOVehicleDetector(model_path="yolo11n.pt", confidence_threshold=0.25, warmup=False)
        
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(dummy_frame, frame_number=42)

        self.assertIsInstance(result, DetectionResult)
        self.assertEqual(result.frame_number, 42)
        self.assertEqual(len(result.detections), 1)
        self.assertEqual(result.detections[0].class_name, "car")
        self.assertEqual(result.detections[0].bbox.area, 90.0 * 180.0)
        self.assertAlmostEqual(result.metrics.inference_time_ms, 10.0)

    def test_detector_handles_empty_or_invalid_frame(self):
        """Test InvalidFrameError is raised on empty, None or wrong shape inputs."""
        detector = YOLOVehicleDetector(warmup=False)
        
        # Test None input
        with self.assertRaises(InvalidFrameError):
            detector.detect(None)
            
        # Test empty numpy array
        with self.assertRaises(InvalidFrameError):
            detector.detect(np.empty((0, 0, 3)))
            
        # Test wrong dimensional channels (e.g. 2D grayscale)
        grayscale_frame = np.zeros((480, 640), dtype=np.uint8)
        with self.assertRaises(InvalidFrameError):
            detector.detect(grayscale_frame)

    def test_detector_handles_prediction_exception(self):
        """Test InferenceFailedError is raised when model throws errors."""
        mock_yolo_instance = mock_yolo_class.return_value
        mock_yolo_instance.predict.side_effect = RuntimeError("GPU Out of Memory")

        detector = YOLOVehicleDetector(warmup=False)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        with self.assertRaises(InferenceFailedError):
            detector.detect(dummy_frame)

    def test_detector_no_vehicles(self):
        """Test behavior when frame contains no target vehicles."""
        mock_result = MagicMock()
        mock_result.boxes = []
        mock_result.speed = {'preprocess': 0.5, 'inference': 5.0, 'postprocess': 0.5}
        
        mock_yolo_instance = mock_yolo_class.return_value
        mock_yolo_instance.predict.return_value = [mock_result]

        detector = YOLOVehicleDetector(warmup=False)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(dummy_frame)

        self.assertEqual(len(result.detections), 0)
        self.assertAlmostEqual(result.metrics.inference_time_ms, 5.0)

    def test_detector_hundred_vehicles(self):
        """Test that the engine successfully extracts and returns 100 boxes."""
        mock_boxes = []
        for i in range(100):
            mock_box = MagicMock()
            mock_box.xyxy = [MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=np.array([float(i), float(i), float(i+10), float(i+10)])))))]
            mock_box.conf = [MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=np.array(0.95)))))]
            mock_box.cls = [MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=np.array(2)))))] # COCO 2: car
            mock_boxes.append(mock_box)
            
        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        
        mock_yolo_instance = mock_yolo_class.return_value
        mock_yolo_instance.predict.return_value = [mock_result]

        detector = YOLOVehicleDetector(warmup=False)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(dummy_frame)

        self.assertEqual(len(result.detections), 100)
        self.assertEqual(result.detections[99].bbox.x1, 99.0)

    def test_detector_invalid_model(self):
        """Test ModelLoadError is raised when weight files are corrupted/unreadable."""
        # Cause model load to fail on primary and fallback weights
        mock_yolo_class.side_effect = RuntimeError("Weights file is corrupted")

        with self.assertRaises(ModelLoadError):
            YOLOVehicleDetector(model_path="corrupted.pt", warmup=False)

if __name__ == "__main__":
    unittest.main()
