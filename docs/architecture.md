# Velox Vision — System Architecture & FAQ

This document outlines the architectural decisions, design paradigms, and operational strategies of the Velox Vision platform.

---

## 1. Why Did You Choose These Tech Stack Components?
We selected a modern, modular Python stack designed for high-throughput computer vision:
- **YOLOv8 (via `ultralytics`)**: Industry-standard object detection model that runs in real-time on edge GPUs and CPUs, offering a strong trade-off between speed and mean Average Precision (mAP).
- **`supervision` (Roboflow)**: Simplifies drawing detections, managing tracking states, and working with frame zones, avoiding verbose boilerplate OpenCV drawing logic.
- **EasyOCR**: A flexible, PyTorch-based OCR library that provides robust text extraction out-of-the-box without requiring complex installation of external engines (like Tesseract).
- **Streamlit**: Enables fast creation of responsive, interactive frontend dashboards using pure Python, simplifying internal and client-facing demos.
- **Clean Architecture Guidelines**: Code is decoupled. The domain core defines *contracts* (interfaces), whereas third-party libraries (like PyTorch, OpenCV, Streamlit) are treated as pluggable details.

---

## 2. What Happens When Detection Fails?
Traffic environments are dynamic, introducing occlusion, lighting variation, and temporary obstructions:
- **Kalman Filtering in Tracking**: When the vehicle detector fails to output a bounding box for a few frames (false negatives), the tracking engine (e.g. ByteTrack) uses linear prediction (Kalman filters) to predict the vehicle's trajectory, preserving its tracking ID.
- **Speed Estimation Smoothing**: Centroid jumping can lead to sudden speed estimation spikes. We apply temporal smoothing (moving average or Savitzky-Golay filters) across a window of tracked coordinates to isolate and discard noise.
- **Graceful Plate OCR Degradation**: If plate detection or OCR fails to resolve text, the system does not crash. It records the incident as `UNKNOWN` plate text, writes the tracking ID and speed statistics to logs, captures a visual snapshot of the vehicle, and continues processing.

---

## 3. How Is Your Code Organized?
Following Clean Architecture guidelines, the code is split into independent layers:
- **`src/core/`**: Contains domain entities (e.g. `Vehicle`, `Violation`, `DetectionResult`) and abstract base classes (interfaces like `IDetector`, `ITracker`, `ISpeedEstimator`). This folder contains no external machine learning library imports.
- **`src/engines/`**: Houses independent implementations of core engines (e.g., vehicle detection, tracking, speed estimation, OCR). These inherit from the core interfaces.
- **`src/services/`**: Handles I/O and communication with external resources, such as writing CSV logs or exporting snapshot files.
- **`src/pipeline/`**: The orchestration layer. It reads incoming video feeds, runs frames through the engines sequentially, manages track state, and dispatches events to services.
- **`dashboard/`**: Presentation layer for end-users to interact with analytics, CSV databases, and evidence snapshots.

---

## 4. Can Another Engineer Extend Your System?
**Yes, easily.** Because of interface-driven design, engines and frameworks are decoupled:
- If a developer wants to replace `EasyOCR` with a cloud-based API or a custom OCR model, they only need to create a class implementing the `IOCR` interface defined in `src/core/interfaces.py` and register it in `config/settings.py`.
- No modification is needed in the orchestration logic (`src/pipeline/traffic_pipeline.py`) or database logic to perform this replacement.

---

## 5. How Would You Deploy It to 100 Cameras?
Running 100 concurrent neural net pipelines on a single server is not feasible due to CPU/GPU bottlenecking. We would scale using a **hybrid edge-cloud architecture**:
1. **Edge Processing (NVIDIA Jetson / Intel NUC)**: 
   - Deploy lightweight vehicle detection and tracking engines directly onto edge processing boxes situated near the cameras.
   - These edge devices process raw video frames locally, keeping high-bandwidth video traffic local.
2. **Event Broker (Apache Kafka / AWS Kinesis)**:
   - When a violation is detected locally, the edge device transmits metadata (e.g., speed, time, relative coordinates) and compressed evidence images to a centralized message broker.
3. **Centralized Cloud (Kubernetes / Docker Containers)**:
   - Central consumer nodes scale out to ingest incoming messages.
   - Run heavier, non-real-time engines (such as high-fidelity Plate OCR and database writes) on these cloud containers.
4. **Streamlit UI Scalability**:
   - Streamlit dashboards query from a replication-configured PostgreSQL database rather than reading raw CSV files locally, ensuring fast load times for multiple concurrent dashboard users.
