#!/bin/bash
# Troubleshooting script for Raspberry Pi camera on ADAS

echo "🔍 TROUBLESHOOTING RASPBERRY PI CAMERA SETUP"
echo "==========================================="
echo

echo "1️⃣ Checking system info..."
uname -m
echo

echo "2️⃣ Checking libcamera installation..."
libcamera-hello --help 2>&1 | head -5
echo

echo "3️⃣ Listing available cameras..."
libcamera-still --list-cameras
echo

echo "4️⃣ Checking /dev/video* devices..."
ls -la /dev/video* 2>/dev/null || echo "No /dev/video* devices found"
echo

echo "5️⃣ Checking v4l2-ctl devices..."
v4l2-ctl --list-devices 2>/dev/null || echo "v4l2-ctl not available"
echo

echo "6️⃣ Checking picamera2 Python package..."
python3 -c "import picamera2; print('✅ Picamera2 installed')" 2>/dev/null || echo "❌ Picamera2 not available"
echo

echo "7️⃣ Testing libcamera viewer (30 sec)..."
timeout 30 libcamera-hello --qt-sink /dev/null 2>&1 | head -10
echo

echo "If Picamera2 is not installed, run:"
echo "  sudo apt update && sudo apt install -y python3-picamera2"
echo ""
echo "To use the TCP stream server, run:"
echo "  ./venv/bin/python adas_tcp_stream.py --camera-backend picamera2 --bind 0.0.0.0 --port 5001 --no-adas"
