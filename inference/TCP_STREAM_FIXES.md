# Fixed TCP Stream Server for ADAS on Raspberry Pi

## Issues Found & Fixed

 **Issue 1: V4L2 backend failing**
- **Problem**: OpenCV with V4L2 backend couldn't access `/dev/video*` devices
- **Fix**: Modified to detect Raspberry Pi and force Picamera2 backend automatically

 **Issue 2: Picamera2 not in venv**
- **Problem**: `pip install picamera2` doesn't work in venv (system package only)
- **Fix**: Added graceful fallback and improved error messages

 **Issue 3: Missing diagnostic output**
- **Problem**: Couldn't tell where frame capture was failing
- **Fix**: Added detailed logging at each step (camera opening, resolution, FPS, frame reading)

 **Issue 4: Poor error handling**
- **Problem**: Script would exit silently on camera errors
- **Fix**: Added verbose error messages and automatic camera reopening

## How to Use on Raspberry Pi

### Step 1: Ensure Picamera2 is installed at system level
```bash
sudo apt update
sudo apt install -y python3-picamera2 libopencv-dev python3-opencv
```

### Step 2: Run the troubleshooting script
```bash
chmod +x troubleshoot_camera.sh
./troubleshoot_camera.sh
```

This will show you:
- Camera availability via libcamera
- /dev/video* device status
- Picamera2 installation status

### Step 3: Run the TCP stream server

**Option A: Using Picamera2 (Recommended)**
```bash
cd /home/spidey/Atul/Rear-View-ADAS/CNN/inference
./venv/bin/python adas_tcp_stream.py \
  --camera-backend picamera2 \
  --bind 0.0.0.0 \
  --port 5001 \
  --width 640 \
  --height 480 \
  --no-adas
```

**Option B: Using OpenCV (Auto-detect Picamera2)**
```bash
./venv/bin/python adas_tcp_stream.py \
  --camera-backend auto \
  --bind 0.0.0.0 \
  --port 5001 \
  --width 640 \
  --height 480 \
  --no-adas
```

### Step 4: On your laptop, view the stream
```bash
cd /home/spidey/Atul/Rear-View-ADAS/CNN/inference
./venv/bin/python tcp_stream_viewer.py \
  --host <pi-ip> \
  --port 5001 \
  --web-port 8000
```

Then open browser: `http://localhost:8000`

## Diagnostics

The improved server now prints:
- Listening on address:port
- Client connected info
- Camera opened successfully
- Resolution got/set
- ⏱ FPS got/set
- Frame capture status every 30 frames
- Detailed error messages if something fails

## Known Issues & Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Camera not found | "Cannot open camera" at startup | Run troubleshoot_camera.sh, check libcamera-still --list-cameras |
| Picamera2 import fails | "Picamera2 unavailable" | Install: `sudo apt install python3-picamera2` |
| Permission denied | "/dev/video0: Permission denied" | Add user to video group: `sudo usermod -aG video $USER` |
| VIDIOC_STREAMON error | Still can't open after permissions fixed | Camera may be in use, check with `lsof /dev/video*` |
| Frames received but display broken | Stream connects but blank | Try `--no-adas` flag first, check JPEG encoding |

## Files Changed

- **adas_tcp_stream.py**: Enhanced diagnostics, auto Picamera2 detection, better error handling
- **troubleshoot_camera.sh**: NEW - Camera system diagnostics
- **test_camera_device.py**: NEW - Test each /dev/video* device

## Testing Path

1. Run troubleshoot_camera.sh to verify camera works
2. Run adas_tcp_stream.py with --no-adas (pure streaming)
3. Run tcp_stream_viewer.py on laptop
4. Open browser and verify stream appears
5. (Optional) Enable ADAS by removing --no-adas flag