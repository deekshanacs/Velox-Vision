# Velox Vision

Velox Vision is a production-grade, AI-powered intelligent traffic monitoring and speed enforcement platform built using Clean Architecture principles.

## Features
- **Vehicle Detection**: Real-time object detection optimized for vehicular traffic (cars, trucks, motorcycles, buses).
- **Multi Object Tracking**: Seamless vehicle tracking across frames with unique ID assignment.
- **Speed Estimation**: Mathematical speed calculation based on pixel displacement and camera perspective calibration.
- **License Plate Detection**: Automated detection of vehicle license plate areas.
- **OCR (Optical Character Recognition)**: Extracting alphanumeric text characters from detected license plates.
- **Overspeed Detection**: Automatic speed threshold checking to flag speeding violations.
- **Evidence Capture**: Automated visual snapshot capture and logging of violating vehicles.
- **CSV Logging**: Continuous logging of all tracked vehicle metrics, speeds, and violations.
- **Streamlit Dashboard**: A web-based graphical interface for real-time telemetry, analytics, and violation reviews.

## Architecture
This project is built using **Clean Architecture** principles to separate core domain business logic from external frameworks, model models, and visualization tools:
- **Core (Domain)**: Independent core entities and contract interfaces defining engine capabilities.
- **Engines (Application Logic)**: Specialized AI engines implementing specific tasks (vehicle detection, tracking, speed, OCR, etc.).
- **Services (Infrastructure)**: Outward integrations for logging, telemetry storage, and evidence files.
- **Pipeline (Orchestration)**: Integrator module that processes video streams, runs engines sequentially, and pushes metrics to services.
- **Presentation (Dashboard)**: Streamlit application displaying real-time statistics, violation records, and reports.

## Folder Structure
```
Velox-Vision/
├── config/                     # Configuration files & setting loaders
│   ├── settings.py             # Config settings loader
│   ├── default.yaml            # Default system parameters
│   └── logging.yaml            # Logging configurations
├── dashboard/                  # Streamlit application UI
│   ├── app.py                  # Streamlit web GUI entrypoint
│   ├── pages/                  # Multipage dashboard scripts
│   ├── components/             # Reusable UI widgets and graphs
│   └── assets/                 # App icons, logos, static images
├── data/                       # Local storage for datasets & calibration
│   ├── videos/                 # Input traffic videos
│   ├── frames/                 # Extracted frame sequences
│   ├── calibration/            # Camera perspective matrices
│   └── annotations/            # Visual model ground truths
├── demo/                       # Media showing working demos
├── docs/                       # Project documentation
│   ├── architecture.md
│   ├── api.md
│   ├── installation.md
│   └── roadmap.md
├── models/                     # Weight files & architecture configs for nets
│   ├── configs/                # Model parameter/layer files
│   └── weights/                # Local serialized neural net files (.pt, .onnx)
├── outputs/                    # Exported outputs
│   ├── csv/                    # Recorded traffic logs
│   ├── snapshots/              # Violation evidence images
│   ├── processed_videos/       # Annotated processed videos
│   └── logs/                   # System runtime logs
├── scripts/                    # Maintenance & executable entrypoints
│   ├── download_models.py      # Weights downloader script
│   ├── setup.py                # Setup/install helper
│   └── run_pipeline.py         # Pipeline manual launch script
├── src/                        # Core codebase package
│   ├── core/                   # Shared data objects & interface contracts
│   ├── engines/                # AI processing units (detection, tracking, speed, plate, OCR, violations)
│   ├── services/               # Integrations (csv reporting, evidence storage)
│   ├── pipeline/               # Unified orchestration pipeline
│   └── utils/                  # Coordinate/math & video file utility functions
└── tests/                      # Testing suites
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/deekshanacs/Velox-Vision.git
   cd Velox-Vision
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```

## Usage
To test the environment and ensure the setup is functional, run:
```bash
python app.py
```
This should print:
```
Welcome to Velox Vision 🏎️
```

## Development Progress
- [x] Project Architecture
- [x] Vehicle Detection Engine
- [ ] Multi-Object Tracking
- [ ] Speed Estimation
- [ ] ANPR
- [ ] OCR
- [ ] Violation Detection
- [ ] Dashboard

## Future Roadmap
- [ ] Implement vehicle detection using custom trained YOLOv8 model weights.
- [ ] Add ByteTrack/DeepSORT integration for tracking unique vehicle IDs.
- [ ] Integrate pixel-to-meter camera calibration for speed estimation.
- [ ] Incorporate YOLO-based license plate detection.
- [ ] Connect EasyOCR to perform character recognition on plates.
- [ ] Build the overspeed enforcement engine and violation CSV logger.
- [ ] Create the interactive Streamlit dashboard.
- [ ] Optimize the pipeline for real-time frame rates and CPU/GPU deployment.

## Tech Stack
- **Languages**: Python 3.11+
- **Deep Learning**: Ultralytics (YOLOv8), EasyOCR
- **Computer Vision Utilities**: OpenCV, Supervision
- **UI & Web Presentation**: Streamlit
- **Data Analytics**: Pandas, NumPy, SciPy
- **Data Formats**: YAML, CSV, JSON

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
