# Velox-Vision

Velox-Vision is a clean, scalable, production-grade AI-powered intelligent traffic monitoring and speed enforcement platform.

## Features
- **Vehicle Detection Engine**: Identifies vehicles in real-time video frames.
- **Multi-Object Tracking Engine**: Tracks multiple vehicles across frames sequentially.
- **Speed Estimation Engine**: Estimates vehicle speed using pixel-to-meter and camera perspective calculations.
- **License Plate Detection & OCR Engine**: Detects license plates and performs OCR to capture plate texts.
- **Violation Detection Engine**: Evaluates tracking history to flags overspeeding and other traffic violations.
- **Evidence Capture Service**: Records snapshots and footage clips of violations.
- **Streamlit Dashboard**: A dashboard to review real-time logs, analytics, and violation records.

## Project Structure

```
Velox-Vision/
├── config/                     # Configuration schemas & default values
│   ├── settings.py             # Config settings loader
│   ├── default.yaml            # Default system & model parameters
│   └── logging.yaml            # Logging config schema
├── dashboard/                  # Streamlit dashboard interface
│   ├── app.py                  # Entrypoint for the web GUI
│   ├── pages/                  # Multipage analytics and settings views
│   ├── components/             # Reusable UI widgets and graphs
│   └── assets/                 # Web GUI icons & logos
├── data/                       # Local storage for assets & annotations
│   ├── videos/                 # Input video sequences
│   ├── frames/                 # Extracted frame sequences
│   ├── calibration/            # Camera perspective calibration files
│   └── annotations/            # Visual model ground truths
├── demo/                       # Media/recordings showing working demos
├── docs/                       # Architecture, Installation, API, and Roadmap details
├── models/                     # Weight files & architecture configs for neural nets
│   ├── configs/                # Hyperparameter/layer configurations
│   └── weights/                # Local serialized neural net files (.pt, .onnx)
├── outputs/                    # Exported outputs
│   ├── csv/                    # Recorded traffic logs
│   ├── snapshots/              # Violation evidence images
│   ├── processed_video/        # Visualized annotated video files
│   └── logs/                   # System runtime logs
├── scripts/                    # Maintenance & executable entrypoints
│   ├── download_models.py      # Weights downloader script
│   ├── setup.py                # Setup/install helper
│   └── run_pipeline.py         # Pipeline manual launch script
├── src/                        # Core codebase package
│   ├── core/                   # Entities and module contracts/interfaces
│   ├── engines/                # AI processing units (detection, tracking, speed, plate, OCR, violations)
│   ├── services/               # Integrations (csv reporting, evidence storage)
│   ├── pipeline/               # Unified orchestration pipeline
│   └── utils/                  # Coordinate/math & video file manipulation utility functions
└── tests/                      # Unified testing suites
```
