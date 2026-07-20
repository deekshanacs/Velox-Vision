import os
import json
import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class TrackingReportGenerator:
    """Generates benchmark execution report logs in JSON format and saves final visual snapshots."""

    def __init__(self, output_dir: str = "outputs/reports/"):
        self.output_dir = output_dir

    def generate_json_report(
        self,
        report_id: str,
        configs: Any,
        perf_stats: Dict[str, Any],
        track_stats: Dict[str, Any],
        memory_stats: Dict[str, Any],
        git_info: Dict[str, str],
        hardware_info: Dict[str, str]
    ) -> str:
        """Saves a detailed JSON benchmark execution file."""
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        report_data = {
            "report_id": report_id,
            "timestamp": timestamp,
            "git": git_info,
            "hardware": hardware_info,
            "configuration": {
                "tracker_type": configs.tracker_type if hasattr(configs, "tracker_type") else "N/A",
                "history_size": configs.history_size if hasattr(configs, "history_size") else 30,
                "max_lost_frames": configs.max_lost_frames if hasattr(configs, "max_lost_frames") else 30,
                "tentative_frames": configs.tentative_frames if hasattr(configs, "tentative_frames") else 3,
                "memory_max_snapshots": configs.memory_max_snapshots if hasattr(configs, "memory_max_snapshots") else 300,
                "viz_enabled": configs.viz_enabled if hasattr(configs, "viz_enabled") else True,
                "viz_trail_length": configs.viz_trail_length if hasattr(configs, "viz_trail_length") else 40
            },
            "performance": perf_stats,
            "tracking": track_stats,
            "memory": memory_stats
        }

        report_path = os.path.join(self.output_dir, f"tracking_report_{timestamp}_{report_id}.json")
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=4)
            logger.info(f"Tracking benchmark execution report saved to: '{report_path}'")
        except Exception as e:
            logger.error(f"Failed to save tracking JSON report: {e}", exc_info=True)
        return report_path

    def save_final_snapshot(self, frame: Any, snapshot_dir: str = "outputs/snapshots/") -> str:
        """Saves a PNG preview snapshot of the final frame containing the HUD dashboard."""
        import cv2
        os.makedirs(snapshot_dir, exist_ok=True)
        snapshot_path = os.path.join(snapshot_dir, "tracking_final_frame.png")
        try:
            cv2.imwrite(snapshot_path, frame)
            logger.info(f"Final preview frame snapshot exported to: '{snapshot_path}'")
        except Exception as e:
            logger.error(f"Failed to export frame preview snapshot: {e}", exc_info=True)
        return snapshot_path
