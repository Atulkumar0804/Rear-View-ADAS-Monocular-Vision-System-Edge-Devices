# Rear-View ADAS Monocular System

Advanced AI-powered rear-view detection and collision avoidance system with real-time vehicle detection, safety metrics, and network streaming.

## Key Features

- **Real-time Vehicle Detection** — YOLO11n at 16-22 FPS
- **YOLO Caching Optimization** — 7-12x faster than baseline (2.9 → 16-22 FPS)
- **TCP Streaming Server** — Low-latency remote camera access (Raspberry Pi & Desktop)
- **Web Dashboard** — Real-time FPS, safety metrics, distance, lane detection
- **Advanced ADAS Metrics** — TTC, DRAC, MTTC, PET, lane-aware collision warnings
- **ByteTracker** — Motion-aware vehicle tracking between detection frames
- **Raspberry Pi Optimized** — Picamera2/libcamera support with CPU-friendly settings
- **Safety Assessment** — Automatic risk classification (CRITICAL / WARNING / CAUTION / SAFE)
- **Mobile Responsive** — Access from phone, tablet, or desktop browser

## Performance

| Metric | Before | After | Improvement |
|---|---|---|---|
| FPS | 2.9 | **16-22 FPS** | **7-12x** |
| Detection Time | 300-500 ms | **0.7-2 ms** | **200-700x** |
| Total Frame Time | 330-500 ms | **2-5 ms** | **100-200x** |

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Atulkumar0804/Rear-View-ADAS-Monocular-Vision-System-Edge-Devices.git
cd Rear-View-ADAS-Monocular-Vision-System-Edge-Devices
```

### 2. Create Virtual Environment

```bash
python3 -m venv inference/venv
source inference/venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r inference/web_requirements.txt
```

### 4. Run

Choose one of the modes below depending on your hardware.

---

## Running Modes

### Option A — TCP Streaming (Raspberry Pi 5 + Picamera2) — Recommended

**On the Raspberry Pi (camera device):**
```bash
cd inference
python adas_tcp_stream_wrapper.py \
  --width 480 --height 360 \
  --detect-width 320 --detect-height 240 \
  --jpeg-quality 80
```

**On a Laptop / Desktop (viewer):**
```bash
cd inference
python tcp_stream_viewer.py --host <pi-ip> --port 5001 --web-port 8000
```

Then open your browser at `http://localhost:8000/stream`

Expected: **16-22 FPS** over LAN

---

### Option B — Web Server (PC / Desktop with USB Camera)

```bash
cd inference
python web_server.py --port 5000
```

Open your browser at `http://localhost:5000`

Expected: **16-22 FPS**

---

### Option C — Direct Camera (Local Display)

```bash
cd inference
python camera_inference.py --camera 0 --rear-camera
```

Expected: **16-22 FPS**

---

### Option D — Video File

```bash
cd inference
python camera_inference.py --camera /path/to/video.mp4
```

---

## TCP Server Configuration

```bash
# Balanced (Recommended)
python adas_tcp_stream_wrapper.py \
  --width 480 --height 360 \
  --detect-width 320 --detect-height 240 \
  --fps 30 \
  --jpeg-quality 80
# Expected: 16-22 FPS with safety metrics

# Maximum FPS (raw stream, no detection)
python adas_tcp_stream_wrapper.py \
  --width 320 --height 240 \
  --raw-stream
# Expected: 40-60 FPS (no detection overhead)

# Full Resolution (GPU required)
python adas_tcp_stream_wrapper.py \
  --width 1280 --height 720 \
  --detect-width 640 --detect-height 360
# Expected: 16-22 FPS (GPU)
```

### TCP Server Arguments

| Argument | Default | Description |
|---|---|---|
| `--bind` | `0.0.0.0` | Server bind address |
| `--port` | `5001` | Server port |
| `--width` | `640` | Stream frame width |
| `--height` | `480` | Stream frame height |
| `--fps` | `30` | Capture FPS |
| `--jpeg-quality` | `80` | JPEG compression (1-100) |
| `--detect-width` | — | Detection resolution width |
| `--detect-height` | — | Detection resolution height |
| `--raw-stream` | — | Skip detection for max speed |
| `--max-failures` | `30` | Max consecutive failures |

---

## Project Structure

```
Rear-View-ADAS/
├── README.md
├── requirements.txt
├── main.sh                             # Main launcher
├── start_web_server.sh                 # Web server launcher
├── Dockerfile / Dockerfile.web         # Docker support
├── docker-compose.web.yml
├── inference/
│   ├── camera_inference.py             # Main ADAS detection engine
│   ├── adas_tcp_stream_wrapper.py      # TCP server (16-22 FPS)
│   ├── adas_tcp_stream.py              # TCP stream core
│   ├── camera_capture_wrapper.py       # Picamera2 capture wrapper
│   ├── tcp_stream_viewer.py            # Web viewer for TCP stream
│   ├── video_inference.py              # Video file processing
│   ├── web_server.py                   # Flask web server
│   ├── byte_tracker.py                 # Motion-aware tracking
│   ├── jetson_depth_lite.py            # Lightweight depth (Jetson/Pi)
│   ├── gpu_config.py                   # GPU profile manager
│   ├── model_optimizer.py              # Inference optimizer
│   └── templates/index.html            # Web dashboard UI
├── models/
│   ├── depth_anything_v2/              # Depth Anything V2 base
│   ├── depth_anything_v2_finetuned/    # Fine-tuned depth model
│   ├── depth_lite/                     # ONNX MiDaS models
│   └── zoedepth/                       # ZoeDepth + MiDaS reference
└── scripts/
    ├── calibrate_camera.py
    ├── train_depth_da2_kitti.py
    ├── export_jetson.py
    └── ...
```

---

## How the Optimization Works

Instead of running YOLO on every frame (expensive), the system runs YOLO every 10 frames and caches results:

```
OLD (2.9 FPS):                    NEW (16-22 FPS):
Frame 1: YOLO (300-500ms)        Frame 1: YOLO (300-500ms) → cache
Frame 2: YOLO (300-500ms)        Frame 2: Reuse cache (0.7ms)
Frame 3: YOLO (300-500ms)        Frame 3: Reuse cache (0.7ms)
...                              ...
                                 Frame 10: YOLO → update cache
Result: 2-3 FPS                  Result: 16-22 FPS
```

| Component | Frequency | Notes |
|---|---|---|
| YOLO Detection | Every 10 frames | Fresh detection every ~500ms |
| ByteTracker | Every frame | Smooth tracking between YOLO runs |
| Horizon Estimation | Every 30 frames | Cached Hough-based update |
| Safety Assessment | Every frame | TTC/DRAC updated with real-time data |
| Classifier | Disabled | Skipped for speed (YOLO sufficient) |

---

## Web Dashboard

The browser dashboard shows:

- **Live video** — MJPEG stream with bounding boxes, class labels, distances
- **Risk Level** — CRITICAL / WARNING / CAUTION / SAFE (color-coded)
- **Safety Metrics** — TTC, DRAC, lane position, motion state, estimated speed
- **Performance Stats** — FPS, detection time, total detections

---

## Prerequisites

- Python 3.8+
- PyTorch (CPU or CUDA)
- OpenCV
- Picamera2 (Raspberry Pi only) or USB camera

---

## Configuration

Edit `inference/camera_inference.py` to tune behavior:

```python
skip_yolo_frames = 10        # Run YOLO every N frames (default 10)
skip_horizon_frames = 30     # Update horizon every N frames
CONFIDENCE_THRESHOLD = 0.4   # Detection confidence cutoff
```

---

## Supported Vehicle Classes

Hatchback, Sedan, SUV, MUV, Bus, Truck, Three-wheeler, Two-wheeler,
LCV, Van, Mini-bus, Tempo-traveller, Bicycle, Person

---

## Troubleshooting

### Low FPS on Raspberry Pi
```bash
python adas_tcp_stream_wrapper.py \
  --width 320 --height 240 \
  --detect-width 240 --detect-height 180
```

### Camera Not Found
```bash
ls /dev/video*
python3 -c "from picamera2 import Picamera2; Picamera2()"
```

### TCP Connection Issues
```bash
telnet <pi-ip> 5001
ss -tuln | grep 5001
```

### Memory Issues
- Use `--raw-stream` to disable detection
- Reduce resolution with `--width` and `--height`

---

## Mobile Access

```bash
python web_server.py --port 5000
# On mobile: http://<your-pc-ip>:5000
```

---

## Training (Optional)

```bash
cd scripts
python train_depth_da2_kitti.py   # Fine-tune depth model on KITTI
python export_jetson.py           # Export optimized model for Jetson/Pi
```

---

## References

- [YOLO11n — Ultralytics](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [ZoeDepth](https://github.com/isl-org/ZoeDepth)

---

## Documentation

- [inference/README_WEB.md](inference/README_WEB.md) — TCP streaming & web server guide
- [WEB_SERVER_IMPLEMENTATION.md](WEB_SERVER_IMPLEMENTATION.md) — server implementation details
- [inference/GPU_CONFIG_GUIDE.md](inference/GPU_CONFIG_GUIDE.md) — GPU optimization guide
