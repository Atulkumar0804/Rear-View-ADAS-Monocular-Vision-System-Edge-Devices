# Raspberry Pi 5 ADAS TCP Streaming Setup

## Problem Solved

- Picamera2 works when using **system Python** (not venv)
- Created camera capture wrapper that uses system Python subprocess
- TCP server now works with the wrapper for frame streaming
- Ready to view camera stream in web browser

## Quick Start (3 Steps)

### Step 1: Start the TCP Stream Server on Raspberry Pi
```bash
cd /home/spidey/Atul/Rear-View-ADAS/CNN/inference
python3 adas_tcp_stream_wrapper.py \
  --bind 0.0.0.0 \
  --port 5001 \
  --width 640 \
  --height 480 \
  --no-adas
```

Wait for output showing " Listening on 0.0.0.0:5001"

### Step 2: Start the Web Viewer on Your Laptop
(In another terminal)
```bash
cd /home/spidey/Atul/Rear-View-ADAS/CNN/inference
./venv/bin/python tcp_stream_viewer.py \
  --host <PI_IP> \
  --port 5001 \
  --web-port 8000
```

Replace `<PI_IP>` with Raspberry Pi's IP address (e.g., `10.72.241.199`)

Wait for output showing " Web viewer on http://0.0.0.0:8000"

### Step 3: Open Browser and View
Navigate to: `http://localhost:8000`

You should see the **live camera feed**!

---

## Architecture

```
Raspberry Pi:

 System Python (has Picamera2)
   camera_capture_wrapper.py
     Reads from /dev/media3 (libcamera)
     Outputs JPEG frames to stdout

 venv Python
   adas_tcp_stream_wrapper.py
     Reads subprocess output
     Sends over TCP socket :5001
     (Optional: runs ADAS detection)

                TCP port 5001

Your Laptop:

 venv Python
   tcp_stream_viewer.py
     Connects to TCP :5001
     Flask server on :8000

 Browser
   http://localhost:8000
     Displays MJPEG stream

```

---

## Files

| File | Purpose |
|------|---------|
| `camera_capture_wrapper.py` | System Python script that captures from Picamera2 |
| `adas_tcp_stream_wrapper.py` | Main TCP server, runs wrapper as subprocess |
| `tcp_stream_viewer.py` | Flask web server for browser viewing |
| `adas_tcp_stream.py` | Legacy version (may not work, kept for reference) |

---

## Options

### Enable ADAS Detection
Remove `--no-adas` flag:
```bash
python3 adas_tcp_stream_wrapper.py \
  --bind 0.0.0.0 \
  --port 5001 \
  --width 640 \
  --height 480
```
(Note: This will be slower as ADAS runs on CPU)

### Adjust Quality/FPS
```bash
python3 adas_tcp_stream_wrapper.py \
  --bind 0.0.0.0 \
  --port 5001 \
  --width 1280 \
  --height 720 \
  --fps 15 \
  --jpeg-quality 75 \
  --no-adas
```

---

## Troubleshooting

### "Connection refused" on laptop
- Check Pi IP: `hostname -I`
- Check Pi server is running and listening: `lsof -i :5001` on Pi
- Check network connectivity: `ping <PI_IP>` from laptop

### "Camera subprocess failed"
- Verify Picamera2 is installed: `python3 -c "import picamera2; print('OK')"`
- Check camera: `python3 camera_capture_wrapper.py --duration 3`

### "No image in browser"
- Check web viewer is running
- Check laptop can access `http://localhost:8000`
- Check browser console for JavaScript errors

### Frame rate is slow
- Reduce resolution: `--width 320 --height 240`
- Reduce JPEG quality: `--jpeg-quality 60`
- Disable ADAS: `--no-adas`

---

## Performance Notes

- **Raw streaming** (~30 FPS @ 640x480): ~2 Mbps bandwidth
- **With ADAS detection** (~5-10 FPS @ 640x480): CPU-intensive, should run in background
- **Resolution vs Quality**: Larger frames = more bandwidth but more detail
- **JPEG Quality**: 60-80 is good balance between size and quality

---

## Testing

Individual component tests:

```bash
# Test 1: Verify camera works
python3 camera_capture_wrapper.py --duration 5

# Test 2: Start TCP server (wait for client)
python3 adas_tcp_stream_wrapper.py --no-adas

# Test 3: Test TCP connection from laptop
nc -v <PI_IP> 5001
(Should connect without error)

# Test 4: Full setup
# Terminal 1 on Pi:
python3 adas_tcp_stream_wrapper.py --no-adas

# Terminal 2 on laptop:
./venv/bin/python tcp_stream_viewer.py --host <PI_IP> --port 5001 --web-port 8000

# Browser:
http://localhost:8000
```

---

## Next Steps

1. Stream is stable? → Enable ADAS detection by removing `--no-adas`
2. Want persistence? → Create systemd service or cron job
3. Want to record? → Modify wrapper to simultaneously save video file
4. Want multiple streams? → Modify server to handle multiple clients

---

## Technical Details

### Why the wrapper approach?

Picamera2 requires:
- System libraries (libcamera, libpisp) → only available system-wide
- System Python packages → can't install in venv due to conflicts

Solution:
- Run capture in **system Python subprocess**
- Communicate via **length-prefixed JPEG protocol** through stdout/stdin pipe
- Keeps venv clean, avoids version conflicts

### Frame Format

Each frame is sent as:
```
[4 bytes: JPEG length (big-endian)] [JPEG data]
```

This allows fast, length-aware reading without scanning for frame boundaries.

### ADAS Integration

When ADAS is enabled:
1. Decode JPEG → numpy array
2. Run detector.detect_frame()
3. Draw bounding boxes
4. Re-encode to JPEG
5. Send over network

Adds ~50-100ms latency but provides safety metrics in stream.