# Distance Estimation in `camera_inference.py`

This document explains exactly how the system measures the distance to each detected object, step by step.

---

## Overview — Two-Layer Architecture

Every detected object goes through **two independent distance estimation layers** that are then fused together:

```
Frame
  │
  ├─ Layer 1 (every frame, ~0 ms extra)
  │       Pinhole Camera Model
  │       distance = (H_real × f) / h_px
  │
  └─ Layer 2 (background thread, every N frames)
          ML Depth Map
          ZoeDepth (desktop)  or  MiDaS Small (Jetson)
                │
                ▼
          DepthCalibrator → metric scale
                │
                ▼
          ADAS Pipeline → blend + smooth
                │
                ▼
          Final distance per object (metres)
```

---

## Layer 1 — Pinhole Camera Model

### Formula

$$d = \frac{H_{\text{real}} \times f}{h_{\text{px}}}$$

| Symbol | Meaning |
|--------|---------|
| $d$ | Distance to object (metres) |
| $H_{\text{real}}$ | Known real-world height of the object class (metres) |
| $f$ | Camera focal length (pixels) |
| $h_{\text{px}}$ | Corrected pixel height of object's bounding box |

### Known Object Heights (`REAL_HEIGHTS` dict)

| Class | Height (m) |
|-------|-----------|
| Person | 1.7 |
| Bicycle | 1.2 |
| Two-wheeler | 1.3 |
| Sedan / Hatchback | 1.5 |
| SUV | 1.8 |
| Bus | 3.2 |
| Truck | 3.0 |
| Three-wheeler | 1.6 |
| LCV | 2.2 |
| Van | 2.0 |
| Others | 1.5 |

### Focal Length (`FOCAL_LENGTH`)

- Loaded from `calibration_data/camera_matrix.npy` if available:
  `f = (fx + fy) / 2`
- Falls back to **1000.0 pixels** if no calibration file is found.

### Pose-Aware Height for Persons

Before applying the formula, if the class is `Person`, the real height is adjusted based on the aspect ratio of the bounding box:

| Aspect ratio (W/H) | Assumed posture | Real height used |
|--------------------|-----------------|-----------------|
| > 0.65 | Sitting | 0.85 m |
| 0.55 – 0.65 | Crouching | 1.10 m |
| < 0.55 | Standing | 1.70 m |

### Class-Specific Empirical Multipliers

Some classes are systematically under-estimated (due to partial occlusion or datasets bias). These are compensated with:

```python
CLASS_DEPTH_MULTIPLIERS = {
    'Person':                3.5,
    'Bicycle':               3.5,
    'Two-wheeler':           4.5,
    'Person + Two-wheeler':  4.0,
    'Person + Bicycle':      3.5,
    'Person + Three-wheeler':4.0,
}
```

Final distance = pinhole distance × multiplier.

### Fisheye Radial Correction (Rear Camera)

The rear camera has a ~170° fisheye lens that compresses objects at the frame edges, making them appear shorter than they really are. This makes the pinhole formula over-estimate distance at the edges.

**Correction model** — equidistant fisheye projection:

$$\text{factor}(x, y) = \frac{\tan(\theta)}{\theta}$$

where $\theta$ is the angle from the optical axis at pixel $(x, y)$.  
At the centre, $\text{factor} = 1.0$.  At the extreme edges, $\text{factor} > 1.0$.

A full $H \times W$ lookup table (LUT) is pre-computed at startup using:

```python
r_norm  = r_px / r_max            # normalised radial distance
theta   = r_norm × half_fov_rad   # angle from axis
factor  = tan(theta) / theta      # correction factor
```

Applied in `estimate_distance()`:

```python
corrected_height = raw_bbox_height × lut[cy, cx]
```

---

## Layer 2 — ML Depth Map

### Why ML Depth?

The pinhole formula degrades for objects that are:
- Very far away (small bounding boxes, pixel noise dominates)
- Partially occluded (pixel height is truncated)
- Non-rigid (crouching people, tipping trucks)

The ML depth network produces a **dense per-pixel depth map** for the whole frame, not just points where there are detections.

### Which Model Runs When

| Mode | Depth Model | Output type |
|------|-------------|-------------|
| Default (GPU desktop) | **ZoeDepth** (`ZoeD_K`) | Metric depth (0.1 – 100 m) directly |
| `--jetson` flag | **MiDaS Small** (ONNX/CUDAExecutionProvider) | Relative inverse depth (needs calibration) |

### Async Background Thread

Neither model runs every frame. Instead:

1. `detect_frame()` calls `async_zoedepth.request_depth(frame)` — this only submits the frame if enough time has elapsed (controlled by `update_interval_frames`, default = 30 frames).
2. The worker thread runs **ZoeDepth or MiDaS** on the GPU and puts results into an output queue.
3. `get_depth_map()` calls `async_zoedepth.get_depth(wait=False)` — non-blocking retrieval. If no new result is ready, the last known depth map is returned.

This keeps the main detection loop at full FPS while depth updates arrive at ~5–10 Hz.

---

## DepthCalibrator — Converting Relative to Metric Depth

MiDaS outputs **relative inverse depth** (closer objects have larger values, units are arbitrary). The `DepthCalibrator` class converts this to metres dynamically.

### Scale Recovery Formula

For each detected anchor object (Sedan / Hatchback / SUV with confidence > 0.6, not touching frame edges):

$$\text{scale} = d_{\text{pinhole}} \times D_{\text{relative}}$$

$$d_{\text{metric}} = \frac{\text{scale}}{D_{\text{relative}}}$$

where $D_{\text{relative}}$ is the median relative depth value inside the bounding box crop.

### Temporal Smoothing

Scales across frames are managed with:
- A history window of 60 frames (simple moving average)
- Sanity check: a new frame scale is accepted only if it is within 50% of the running average, to reject outliers caused by bad detections.

```python
if 0.5 * avg_scale < frame_scale < 2.0 * avg_scale:
    scale_history.append(frame_scale)
current_scale = mean(scale_history)
```

---

## ADAS Distance Pipeline — Depth Map Sampling

Once a depth map (either ZoeDepth metric or MiDaS calibrated) is available, the `AdasDistancePipeline.estimate_distance()` method replaces the raw pinhole calculation.

### Sampling Strategy

**With segmentation mask** (YOLO x-seg model):
- Samples pixels from the ground-contact region of the mask (`focus_ground=True`)
- Uses the lower portion of the mask which corresponds to the feet/tyres — the closest actual surface point
- Reports higher confidence than bbox fallback

**Without mask** (YOLO11n detection-only):
- Samples at the bottom-centre of the bounding box:
  ```python
  cy = y2 - (y2 - y1) × 0.1    # 10% up from the bottom edge
  raw_distance = depth_map[cy, cx]
  ```
- This approximates the ground-contact point

### Per-Class Scale Correction

A loaded or default class-scale factor is applied to the raw sampled depth:

```python
calibrated = raw_distance × class_scales.get(class_name, 1.0)
```

Default scales: Sedan=1.0, Bus=1.08, Person=0.90, Truck=1.05, etc.

---

## Depth Blending — Pinhole vs ML

The `AccurateHybridDepth._blend_depths()` method fuses the two depth maps:

1. A soft binary mask is created for detected objects (filled bounding boxes, Gaussian-blurred with 31×31 kernel).
2. Blending weights:
   - **Inside objects**: 30% ML + 70% pinhole
   - **Background (ground, sky)**: 100% ML

```python
ml_weight      = 1.0 - object_mask × 0.7
pinhole_weight = object_mask × 0.7
blended        = ml_depth × ml_weight + pinhole_depth × pinhole_weight
```

The rationale: the ML model is better at backgrounds and scene-level geometry, but the pinhole formula (anchored to known real heights) is more accurate for the actual object surface.

---

## Temporal Filtering — Smoothing Across Frames

Raw per-frame distances are noisy. Two filters are applied in sequence:

### 1. Exponential Moving Average (EMA)

$$d_{\text{EMA}}[t] = \alpha \cdot d[t] + (1-\alpha) \cdot d_{\text{EMA}}[t-1]$$

With $\alpha = 0.25$ — biased toward history to suppress single-frame spikes.

### 2. Kalman Filter (1D)

A simple 1D Kalman filter with:
- Process noise variance: 0.01
- Measurement noise variance: 0.10

This further smooths the EMA output, particularly useful when a vehicle accelerates/decelerates.

Both filters operate **per track ID**, so each independently tracked object has its own filter state.

---

## Rear-Camera Domain Corrections

### Depth Clipping

All distances are hard-clipped to the reverse-ADAS relevant range:

| Limit | Value |
|-------|-------|
| Minimum | **0.3 m** (below this = own bumper) |
| Maximum | **15.0 m** (beyond this = not a reverse hazard) |

### Bumper Exclusion Zone

The bottom **8% of the frame** is the vehicle's own bumper.  
Any detection where > 60% of its bounding box height falls in this strip is **suppressed** (distance set to `None`) to prevent false alarms at 0 m.

### Ground-Contact Distance Fallback

When ML depth is unreliable (e.g. very large objects, or depth map not yet available), the system uses a ground-plane geometry estimate:

$$d = \frac{h_{\text{mount}} \times f}{y_{\text{bottom}} - y_{\text{horizon}}}$$

where:
- $h_{\text{mount}}$ = camera mounting height = **0.75 m**
- $y_{\text{horizon}}$ = estimated horizon = `H × (1 − ground_plane_ratio)` = top 35% of frame
- $y_{\text{bottom}}$ = y-coordinate of the bottom edge of the bounding box (ground contact point)

---

## Three-Zone Alert System

Once a final `distance` value is assigned to a detection, `RearADASAlert.classify_zone()` maps it to an alert zone:

| Zone | Distance range | Colour (BGR) | Meaning |
|------|---------------|-------------|---------|
| **CRITICAL** | < 2 m | Red `(0,0,255)` | Stop immediately |
| **DANGER** | 2 – 5 m | Orange-red `(0,80,255)` | Slow down |
| **CAUTION** | 5 – 10 m | Orange `(0,165,255)` | Be aware |
| **SAFE** | > 10 m | Green `(0,200,0)` | No action needed |

Alerts have a **0.5 s per-track cooldown** to avoid repeated triggering.

---

## Complete Data-Flow Diagram

```
Camera Frame
      │
      ▼
RearCameraDomainAdapter.preprocess_frame()
  • CLAHE on bottom 70% (ground enhancement)
      │
      ▼
YOLO11n Detection  ──────────────────────────────────────────────────►
  • 6 COCO classes mapped to project classes                          │
  • Fine-tuned classifier refines vehicle sub-type                   │
      │                                              Background Thread
      ▼                                              ┌──────────────────────┐
estimate_distance()  ← Layer 1                      │  AsyncZoeDepth /      │
  (1) corrected_height = h_px × fisheye_lut[cy,cx]  │  AsyncDepthLite       │
  (2) d = H_real × focal_length / corrected_height   │  (runs every N frames)│
  (3) d *= CLASS_DEPTH_MULTIPLIERS[class]            │                      │
  (4) clip to [0.3 m, 15.0 m]                        │  Output: depth_map   │
      │                                              └──────────┬───────────┘
      │                ◄─────────────────────────────────────────┘
      ▼           get_depth_map() retrieves latest async result
AdasDistancePipeline.estimate_distance()  ← Layer 2 fusion
  (1) sample depth_map at ground-contact point of bbox/mask
  (2) apply per-class scale correction
  (3) EMA smoothing (α=0.25)
  (4) Kalman 1D smoothing
      │
      ▼
RearADASAlert.classify_zone(distance)
  CRITICAL / DANGER / CAUTION / SAFE
      │
      ▼
draw_detections()  →  Overlay on frame with coloured bounding box + distance label
```

---

## Key Source Locations

| Component | File | Lines |
|-----------|------|-------|
| `REAL_HEIGHTS` dict | [inference/camera_inference.py](inference/camera_inference.py#L741) | ~741 |
| `FOCAL_LENGTH` + calibration load | [inference/camera_inference.py](inference/camera_inference.py#L95) | ~95–117 |
| `CLASS_DEPTH_MULTIPLIERS` | [inference/camera_inference.py](inference/camera_inference.py#L82) | ~82 |
| `REAR_CAMERA_CONFIG` | [inference/camera_inference.py](inference/camera_inference.py#L136) | ~136 |
| `AsyncZoeDepth` class | [inference/camera_inference.py](inference/camera_inference.py#L202) | ~202 |
| `AccurateHybridDepth._compute_pinhole_depth()` | [inference/camera_inference.py](inference/camera_inference.py#L484) | ~484 |
| `AccurateHybridDepth._blend_depths()` | [inference/camera_inference.py](inference/camera_inference.py#L522) | ~522 |
| `AdasDistancePipeline.estimate_distance()` | [inference/camera_inference.py](inference/camera_inference.py#L702) | ~702 |
| `RearCameraDomainAdapter` class | [inference/camera_inference.py](inference/camera_inference.py#L816) | ~816 |
| `RearCameraDomainAdapter.fisheye_correct_bbox_height()` | [inference/camera_inference.py](inference/camera_inference.py#L953) | ~953 |
| `RearCameraDomainAdapter.ground_contact_distance()` | [inference/camera_inference.py](inference/camera_inference.py#L992) | ~992 |
| `DepthCalibrator` class | [inference/camera_inference.py](inference/camera_inference.py#L1203) | ~1203 |
| `CameraVehicleDetector.estimate_distance()` | [inference/camera_inference.py](inference/camera_inference.py#L1816) | ~1816 |
| `CameraVehicleDetector.get_depth_map()` | [inference/camera_inference.py](inference/camera_inference.py#L1874) | ~1874 |
| `CameraVehicleDetector.detect_frame()` | [inference/camera_inference.py](inference/camera_inference.py#L1929) | ~1929 |
| `AsyncDepthLite` (Jetson MiDaS) | [inference/jetson_depth_lite.py](inference/jetson_depth_lite.py) | — |
