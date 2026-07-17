import sys
from unittest.mock import MagicMock, patch

# Mock the ultralytics module to prevent loading torch and its blocked DLLs
mock_ultralytics = MagicMock()
sys.modules['ultralytics'] = mock_ultralytics

import unittest
import numpy as np

from src.core.entities import DetectionMetrics
from src.utils.visualization import draw_rounded_rectangle, draw_metadata_label, draw_hud_dashboard
from scripts.run_detection import DetectionRunner

class TestPipelineAndVisualization(unittest.TestCase):
    """Unit tests for the visualization library and execution runner class."""

    def test_drawing_rounded_rectangle(self):
        """Verify draw_rounded_rectangle executes without errors on dummy frame."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        color = (255, 0, 0)
        draw_rounded_rectangle(frame, (10, 10), (90, 90), color, thickness=1, r=5)
        
        # Verify drawing occurred by checking pixels changed from black
        self.assertTrue(np.any(frame > 0))

    def test_drawing_metadata_label(self):
        """Verify draw_metadata_label executes without errors on dummy frame."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        color = (0, 255, 0)
        draw_metadata_label(frame, "TEST LABEL", (10, 50), bg_color=color)
        
        self.assertTrue(np.any(frame > 0))

    def test_drawing_hud_dashboard(self):
        """Verify draw_hud_dashboard renders text elements and overlays on dummy frame."""
        frame = np.zeros((200, 640, 3), dtype=np.uint8)
        metrics = DetectionMetrics(
            preprocess_time_ms=1.0,
            inference_time_ms=10.0,
            postprocess_time_ms=1.0,
            total_latency_ms=12.0,
            fps=83.3
        )
        
        draw_hud_dashboard(
            img=frame,
            metrics=metrics,
            model_name="yolo11n.pt",
            frame_number=100,
            total_frames=300,
            detected_count=5,
            avg_inference_ms=10.0
        )
        
        self.assertTrue(np.any(frame > 0))

    @patch("scripts.run_detection.VehicleDetectorFactory")
    @patch("scripts.run_detection.cv2.VideoCapture")
    def test_runner_initialization_configs(self, mock_capture, mock_factory):
        """Verify that DetectionRunner initializes with correct configs and mock detector."""
        mock_detector_instance = MagicMock()
        mock_factory.create_detector.return_value = mock_detector_instance
        
        runner = DetectionRunner()
        
        # Check properties are correctly bound
        self.assertEqual(runner.model_path, "models/weights/yolo11n.pt")
        self.assertAlmostEqual(runner.confidence_threshold, 0.30)
        self.assertEqual(runner.classes, ["car", "motorcycle", "bus", "truck"])
        self.assertIs(runner.detector, mock_detector_instance)

if __name__ == "__main__":
    unittest.main()
