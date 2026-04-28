# Web Server for Real-time Camera Inference

This directory contains a Flask-based web server that exposes the camera inference functionality via REST API and a mobile-friendly web dashboard.

## Features

✅ **Real-time Camera Streaming** - Live MJPEG stream to browsers
✅ **REST API Endpoints** - Process images and get detection results
✅ **Mobile Dashboard** - Responsive web interface for smartphones
✅ **Video Upload** - Upload and process video files
✅ **Live Statistics** - FPS, inference time, detection counts
✅ **Safety Assessment** - Real-time safety level indicators
✅ **Lane Detection** - Multi-lane positioning awareness

## Installation

### Prerequisites
- Python 3.8+
- GPU/CUDA (recommended for real-time performance)
- Camera connected to PC

### Setup

1. Install web server dependencies:
```bash
pip install -r web_requirements.txt
```

2. Main inference dependencies are already in the project requirements.txt

## Usage

### Start Web Server

```bash
# Default: localhost:5000
python web_server.py

# Custom port and auto-start camera
python web_server.py --port 8080 --auto-start

# Enable debug mode
python web_server.py --debug

# Specify camera device
python web_server.py --camera 0  # Use camera 0
```

### Access from Mobile

1. **Find your PC IP address:**
   - Linux: `hostname -I`
   - Windows: `ipconfig` (look for IPv4)
   - Mac: `ifconfig`

2. **Open browser on mobile:**
   ```
   http://<your-pc-ip>:5000
   ```

3. **Or use your network:**
   ```
   http://192.168.1.100:5000  (example IP)
   ```

## API Endpoints

### Camera Control
- `POST /api/camera/start` - Start camera and inference
- `POST /api/camera/stop` - Stop camera

### Video Stream
- `GET /api/video/stream` - MJPEG video stream for display

### Detections
- `GET /api/detections` - Get latest detections
- `GET /api/detections/history?limit=10` - Get detection history

### Image Processing
- `POST /api/image/process` - Upload and process single image
- `POST /api/upload/video` - Upload video file

### Info
- `GET /api/status` - Server and camera status
- `GET /api/config` - Configuration (classes, thresholds)
- `GET /api/health` - Health check

## Web Dashboard Features

### Camera Feed
- Live video streaming (MJPEG)
- Real-time bounding boxes
- Safety level indicators (Red/Orange/Yellow/Green)

### Statistics
- Current FPS
- Total detections
- Average inference time
- Total frames processed

### Detections Panel
- Class name
- Confidence score
- Distance estimate
- Safety level badge

### Upload & Process
- Drag-and-drop video/image upload
- Supported formats: MP4, AVI, JPG, PNG
- Process in background

## Mobile Optimization

✅ **Responsive Design** - Works on phones, tablets, desktop
✅ **Touch-Friendly** - Large buttons for mobile
✅ **Low Bandwidth** - MJPEG with 80% JPEG quality
✅ **Auto-Refresh** - Updates every 500ms
✅ **Error Handling** - User-friendly error messages

## Performance Tips

1. **Reduce Video Quality:**
   - Modify JPEG quality in `generate_frames()` function
   - Current: 80% quality (good balance)

2. **Lower Resolution:**
   - Edit `camera_capture_thread()` to set lower resolution
   - Try 640x480 instead of 1280x720

3. **GPU Acceleration:**
   - Server automatically uses CUDA if available
   - Falls back to CPU otherwise

4. **FPS Limiting:**
   - Use `--max-fps` argument (if supported)
   - Reduces bandwidth and CPU usage

## Troubleshooting

### Camera Not Found
```
Error: Failed to open camera 0
Solution: Check camera_index with: python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

### Mobile Can't Connect
```
Error: Connection refused
Solution: 
1. Check firewall is open for port 5000
2. Verify PC and mobile are on same network
3. Use actual IP address, not localhost
4. Try: ipconfig /all (Windows) or hostname -I (Linux)
```

### Poor Performance
```
Solution:
1. Reduce resolution in web_server.py
2. Lower JPEG quality (from 80 to 60)
3. Increase update interval (from 500ms to 1000ms)
4. Check GPU memory: nvidia-smi
```

## Example cURL Commands

```bash
# Start camera
curl -X POST http://localhost:5000/api/camera/start

# Get status
curl http://localhost:5000/api/status

# Get detections
curl http://localhost:5000/api/detections

# Process image
curl -F "image=@image.jpg" http://localhost:5000/api/image/process

# Stop camera
curl -X POST http://localhost:5000/api/camera/stop
```

## Configuration

Edit constants in `web_server.py`:

```python
# Frame queue size (lower = more CPU friendly)
frame_queue = queue.Queue(maxsize=2)

# Detection history size
detection_queue = queue.Queue(maxsize=10)

# JPEG quality (80% recommended)
cv2.IMWRITE_JPEG_QUALITY, 80
```

## File Structure

```
inference/
├── web_server.py           # Main Flask server
├── web_requirements.txt    # Dependencies
├── camera_inference.py     # Inference logic
├── templates/
│   └── index.html         # Web dashboard
└── README_WEB.md          # This file
```

## Future Enhancements

🔧 **Planned Features:**
- WebSocket for real-time alerts
- Video recording with detections
- Multi-camera support
- Authentication/API keys
- Database logging
- Real-time metrics dashboard

## Security Notes

⚠️ **For Local Network Only**
- No authentication by default
- Run behind firewall for security
- Consider adding API key protection for public deployment
- Use HTTPS in production

## Support

For issues or questions:
1. Check logs in `CNN/logs/adas.log`
2. Enable debug mode: `python web_server.py --debug`
3. Check camera connectivity first

## License

Part of Rear-View ADAS Project
