# Phase 4.2 Speed Estimation Engine Documentation

## Overview & Architecture

The **Speed Estimation Engine** (`src/engines/tracking/speed/`) is a high-performance, enterprise-grade subsystem responsible for computing real-world vehicle speed, velocity vectors, acceleration/deceleration, and estimation confidence scores for Velox Vision v0.4.1.

### Architectural Guarantees & Constraints
- **Strict Input Isolation**: The engine **never directly consumes raw object detections**. It consumes only `MotionProfile`, `VehicleMemory`, and `TrackingConfiguration`.
- **Downstream Ownership**: Every `TrackedVehicle` owns a `SpeedProfile` attached to both `vehicle.speed_profile` and `vehicle.motion_profile.speed_profile`. Future analytics modules (lane tracking, behavior analysis, enforcement) consume `SpeedProfile` directly without recalculating speed.
- **Incremental Real-Time Computation**: Evaluates speed incrementally on frame updates (< 1ms per vehicle execution time) without replaying history.

```
+------------------+     +---------------+
|  MotionProfile   | --> |               |
|   (Phase 4.1)    |     |  SpeedEngine  |     +-------------------+
+------------------+     |  (Phase 4.2)  | --> |   SpeedProfile    |
                         |               |     |  (TrackedVehicle) |
+------------------+     |  - Calib      |     +-------------------+
|  VehicleMemory   | --> |  - Persp      |
|   (Snapshots)    |     |  - Smooth     |
+------------------+     +---------------+
```

---

## Speed Mathematics & Formulation

### 1. Displacement & Spatial Calibration
Given consecutive center locations $\mathbf{p}_{t-1} = (x_{t-1}, y_{t-1})$ and $\mathbf{p}_t = (x_t, y_t)$ on the image plane:

$$\Delta d_{\text{px}} = \sqrt{(x_t - x_{t-1})^2 + (y_t - y_{t-1})^2}$$

Using spatial scale ratio $s = \text{pixel\_to\_meter\_ratio}$ (in meters per pixel):

$$\Delta d_{\text{meters}} = \Delta d_{\text{px}} \times s$$

### 2. Time Normalization & Raw Speed
Using frame duration $\Delta t = t_t - t_{t-1}$ (or $1/\text{fps}$):

$$v_{\text{raw, m/s}} = \frac{\Delta d_{\text{meters}}}{\Delta t}, \quad v_{\text{raw, km/h}} = v_{\text{raw, m/s}} \times 3.6$$

### 3. Outlier Rejection & Temporal Filtering
To suppress camera shake and tracking jitter spikes, instantaneous speed $v_{\text{raw}}$ is passed through an outlier filter and configured smoothing filter (Moving Average, EWMA, or Median):

$$\hat{v}_t = \text{SpeedSmoother}(v_{\text{raw}}, \{v_{t-k}, \dots, v_{t-1}\})$$

### 4. Acceleration & Deceleration
Acceleration $a_t$ ($m/s^2$) is computed over smoothed speed series:

$$a_t = \frac{\hat{v}_{t, \text{m/s}} - \hat{v}_{t-1, \text{m/s}}}{\Delta t}$$

If $a_t > 0$, $\text{acceleration} = a_t$; if $a_t < 0$, $\text{deceleration} = |a_t|$.

### 5. Multi-Factor Confidence Scoring
Overall estimation confidence $C_{\text{overall}} \in [0.0, 1.0]$ is computed as a weighted composite score:

$$C_{\text{overall}} = 0.35 \cdot C_{\text{speed}} + 0.35 \cdot C_{\text{observation}} + 0.30 \cdot C_{\text{calibration}}$$

Where:
- $C_{\text{speed}}$: Sequence stability score based on speed variance over history.
- $C_{\text{observation}}$: Tracking observation ratio from `MotionProfile`.
- $C_{\text{calibration}}$: Precision rating of active spatial calibration mode.

---

## Calibration Guide

### 1. Manual Ratio Calibration
Set `pixel_to_meter_ratio` directly when pixel-to-meter scale is known:
```python
from src.engines.tracking.speed import SpeedCalibration

calib = SpeedCalibration()
calib.set_manual_ratio(0.05)  # 0.05 meters per pixel
```

### 2. Reference Line / Points Calibration
Calibrate using a known physical ground distance (e.g., lane line marking of 3.0 meters):
```python
calib = SpeedCalibration(
    reference_distance_meters=3.0,
    reference_points=[(100.0, 500.0), (100.0, 600.0)]  # 100 pixels apart -> 0.03 m/px
)
```

---

## Perspective Correction & Homography Integration

When `perspective_enabled: true`, `PerspectiveTransformer` projects image coordinates onto ground plane coordinates via a $3 \times 3$ Homography matrix $\mathbf{H}$:

$$\begin{bmatrix} u \\ v \\ w \end{bmatrix} = \mathbf{H} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}, \quad (x_{\text{ground}}, y_{\text{ground}}) = \left( \frac{u}{w}, \frac{v}{w} \right)$$

### Future Homography Extension Hook
```python
from src.engines.tracking.speed import PerspectiveTransformer

perspective = PerspectiveTransformer(enabled=True)
perspective.set_homography_matrix([
    [1.2, 0.1, -50.0],
    [0.0, 1.5, -120.0],
    [0.0, 0.001, 1.0]
])
```

---

## Configuration Reference

Add or customize parameters in `config/default.yaml`:

```yaml
speed:
  enabled: true
  smoothing_window: 8
  minimum_motion_distance: 5.0
  pixel_to_meter_ratio: 0.05
  perspective_enabled: false
  confidence_threshold: 0.6
  smoothing_method: "MOVING_AVERAGE"
  max_speed_jump_kmh: 60.0
```
