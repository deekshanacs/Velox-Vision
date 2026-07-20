# Phase 4.3 Lane Intelligence Engine Documentation

## Overview & Architecture

The **Lane Intelligence Engine** (`src/engines/tracking/lane/`) is a high-performance, enterprise perception subsystem responsible for determining lane relationships, lateral offsets, boundary distances, relative headings, lane change transitions, and per-lane occupancy statistics for Velox Vision v0.4.2.

### Architectural Guarantees
- **Strict Input Isolation**: The engine **never directly consumes raw object detections**. It consumes `VehicleMemory`, `MotionProfile`, and `SpeedProfile`.
- **PerceptionProfile Container**: Avoids entity property bloat on `TrackedVehicle` by aggregating all perception outputs under `vehicle.perception_profile` (`motion`, `speed`, `lane`, `behavior`, `violation`, `prediction`).
- **Downstream Module Support**: Downstream analytics (behavior classification, traffic violations) consume `LaneProfile` directly without recalculating geometry.
- **Incremental Real-Time Computation**: Real-time evaluation (< 1 ms per vehicle).

```
+-------------------+     +------------------+     +-------------------+
|   VehicleMemory   |     |  MotionProfile   |     |   SpeedProfile    |
|    (Snapshots)    |     |   (Phase 4.1)    |     |    (Phase 4.2)    |
+-------------------+     +------------------+     +-------------------+
          \                        |                        /
           \                       |                       /
            +---------------------------------------------+
                                   |
                                   v
                         +-------------------+
                         |    LaneEngine     |
                         |   (Phase 4.3)     |
                         |  - RoadModel      |
                         |  - LaneValidator  |
                         |  - LaneGeometry   |
                         |  - Hysteresis     |
                         +-------------------+
                                   |
                                   v
                         +-------------------+
                         |    LaneProfile    |
                         | (TrackedVehicle)  |
                         +-------------------+
```

---

## Lane Geometry Formulations

Given a vehicle center position $\mathbf{p} = (x_v, y_v)$, motion heading angle $\theta_{\text{heading}}$, and a matching lane $L_i$ with left boundary $x_{\text{left}}$, right boundary $x_{\text{right}}$, and centerline $x_{\text{center}}$:

### 1. Distance to Centerline & Boundaries
$$\text{dist}_{\text{center}} = x_v - x_{\text{center}}$$

$$\text{dist}_{\text{left}} = |x_v - x_{\text{left}}|, \quad \text{dist}_{\text{right}} = |x_v - x_{\text{right}}|$$

### 2. Normalized Lateral Offset
$$\text{lateral\_offset} = \frac{x_v - x_{\text{center}}}{w_{\text{lane}} / 2} \in [-1.0, 1.0]$$

- $-1.0$: Vehicle center at left boundary edge.
- $0.0$: Vehicle centered perfectly in lane.
- $+1.0$: Vehicle center at right boundary edge.

### 3. Relative Heading
Angle $\theta_{\text{relative}}$ relative to road orientation vector:

$$\theta_{\text{relative}} = \text{atan2}(\sin \theta_{\text{heading}}, \cos \theta_{\text{heading}})$$

---

## Assignment State Machine & Hysteresis

Every vehicle is classified into one of 5 assignment states:

```
[ UNKNOWN ] ---> [ TEMPORARY_ASSIGNMENT ] ---> [ ASSIGNED ]
                         |                         |
                         v                         v
                 [ TRANSITIONING ] <-------> [ OUTSIDE_LANE ]
```

- **`UNKNOWN`**: Uninitialized track or missing snapshots.
- **`TEMPORARY_ASSIGNMENT`**: Track age is less than configured hysteresis window.
- **`ASSIGNED`**: Vehicle is firmly centered inside lane boundaries.
- **`TRANSITIONING`**: Vehicle center is within `boundary_margin` pixels of a lane boundary or changing lanes.
- **`OUTSIDE_LANE`**: Vehicle is off-road or outside mapped road boundaries.

### Boundary Oscillation Hysteresis
To prevent rapid toggling when a vehicle jitters near a lane line, lane transitions require $N = \text{hysteresis\_frames}$ consecutive observations in the target lane before confirming the transition.

---

## Configuration Reference

Add or customize parameters in `config/default.yaml`:

```yaml
lane:
  enabled: true
  lane_count: 3
  lane_width: 120.0
  boundary_margin: 15.0
  confidence_threshold: 0.6
  road_start_x: 0.0
  coordinate_system: "image"
  lane_origin: "left"
  orientation: "vertical"
  hysteresis_frames: 3
```

---

## Future Traffic Violation Integration (Phase 4.5)

The `LaneProfile` provides direct data hooks for Phase 4.5 Enforcement Engines:
- **Illegal Lane Change**: Consumes `LaneTransitionHistory` and `boundary_type` (e.g. crossing `DOUBLE_SOLID` boundary).
- **Shoulder Driving**: Detects occupancy in lanes with `BoundaryType.SHOULDER`.
- **Lane Straddling / Weaving**: Monitors persistent `TRANSITIONING` state or rapid transition counts over time.
