# Quick Start: Web-Based ADAS on Mobile

This guide shows you how to access the rear-view ADAS system from any mobile device or computer on your network.

## 🚀 Setup (5 minutes)

### Step 1: Install Dependencies
```bash
cd /home/atul/Desktop/atul/rear_view_adas_monocular/CNN
pip install -r inference/web_requirements.txt
```

### Step 2: Start the Web Server
```bash
python3 inference/web_server.py --port 5000
```

**Output should show:**
```
 * Running on http://0.0.0.0:5000
 * Press CTRL+C to quit
```

### Step 3: Access from Mobile

#### Option A: Same WiFi Network (Recommended)
1. **Find your PC's IP address:**
   ```bash
   hostname -I
   ```
   You'll see something like: `192.168.1.100`

2. **Open browser on mobile:**
   - Type: `http://192.168.1.100:5000`
   - Press Enter

#### Option B: USB Tethering / Hotspot
- Connect your PC to your phone's hotspot
- Use `hostname -I` to find PC IP
- Access same URL on mobile

#### Option C: Local Network (LAN)
- Both devices must be on same WiFi/LAN
- Use the IP address from `hostname -I`

## 📱 Mobile Dashboard Features

### Camera Controls
- **▶ Start Camera** - Begin live detection
- **⏹ Stop Camera** - Stop inference

### Live Video Feed
- Real-time camera stream with bounding boxes
- Color-coded safety levels:
  - 🟢 **GREEN** - Safe
  - 🟡 **YELLOW** - Caution
  - 🟠 **ORANGE** - Warning
  - 🔴 **RED** - Critical

### Real-Time Statistics
- **FPS** - Frames per second
- **Total Detections** - Cumulative count
- **Avg Inference Time** - Processing time per frame
- **Detections** - Current frame detections

### Detected Vehicles
- Class name (e.g., Sedan, Truck, Bus)
- Confidence score
- Distance estimate (meters)
- Safety assessment level

## 🔧 Advanced Options

### Custom Port
```bash
python3 inference/web_server.py --port 8080
```

### Allow External Access (Non-Local)
```bash
python3 inference/web_server.py --host 0.0.0.0 --port 5000
```

### Debug Mode
```bash
python3 inference/web_server.py --debug
```

## 🎬 Example Workflow

### 1. Start Server
```bash
cd /home/atul/Desktop/atul/rear_view_adas_monocular/CNN
python3 inference/web_server.py
```

### 2. On Mobile (Same WiFi)
```
Open Browser → Type: http://192.168.1.X:5000
```

### 3. Click "Start Camera"
- System initializes YOLOv11 + Depth models
- Real-time detection begins
- Boxes appear around vehicles

### 4. Monitor Statistics
- Watch FPS, detections, safety levels
- View distance estimates
- Check inference time

## 🛠️ Troubleshooting

### "Can't connect to server"
```
✓ Check: PC and mobile on same WiFi
✓ Check: Firewall not blocking port 5000
✓ Check: Use correct IP (from hostname -I)
✓ Try: http://192.168.1.X:5000 (not localhost)
```

### "Camera not found"
```
✓ Check: Camera connected to PC
✓ Check: No other app using camera
✓ Try: python3 -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

### "Slow/Laggy Stream"
```
✓ Reduce resolution in web_server.py (1280x720 → 640x480)
✓ Lower JPEG quality (80 → 60)
✓ Increase refresh interval (500ms → 1000ms)
✓ Check WiFi signal strength
```

### "GPU Memory Error"
```
✓ System will automatically fall back to CPU
✓ Check GPU: nvidia-smi
```

## 📊 API Endpoints

All available via curl or API clients:

```bash
# Start camera
curl -X POST http://localhost:5000/api/camera/start

# Get current status
curl http://localhost:5000/api/status

# Get latest detections
curl http://localhost:5000/api/detections

# Process single image
curl -F "image=@image.jpg" http://localhost:5000/api/image/process

# Health check
curl http://localhost:5000/api/health
```

## 📈 Performance Expectations

**On Desktop/Laptop:**
- FPS: 20-30
- Inference Time: 30-50ms
- Latency: <500ms

**On Mobile (4G/5G):**
- FPS: 5-15 (depends on bandwidth)
- Video Quality: 480p-720p

## 🔒 Security Notes

⚠️ **Current Setup:**
- No authentication
- No encryption
- Local network only

**For Public Deployment:**
- Use HTTPS (SSL certificate)
- Add API key authentication
- Use behind reverse proxy (nginx)
- Restrict IP ranges

## 🎯 Next Steps

1. ✅ Try mobile access
2. 📹 Test with different camera angles
3. 📊 Monitor performance metrics
4. 🚗 Test with real-world traffic scenarios
5. 🐳 Deploy with Docker (advanced)

## 📚 Related Documentation

- [Full Web Server README](README_WEB.md)
- [Camera Inference Details](../QUICK_START.md)
- [Deployment Guide](../README_DOCUMENTATION.md)

## 💡 Pro Tips

- **Bookmark the page:** Save URL for quick access
- **Fullscreen:** Press F11 for fullscreen mode
- **Mobile shortcut:** Add to home screen for app-like experience
- **Record results:** Use mobile screenshot for safety documentation

---

**Questions?** Check the logs:
```bash
tail -f /home/atul/Desktop/atul/rear_view_adas_monocular/CNN/logs/adas.log
```

Enjoy mobile-based ADAS! 🚀
