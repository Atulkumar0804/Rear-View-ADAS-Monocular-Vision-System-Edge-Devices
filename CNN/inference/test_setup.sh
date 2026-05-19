#!/bin/bash
# Quick test of camera streaming setup

echo "🧪 Testing Raspberry Pi Camera Streaming Setup"
echo "=============================================="
echo ""

# Test 1: Check if we're on Raspberry Pi
echo "1️⃣ Checking system..."
if [[ $(uname -m) == "aarch64" || $(uname -m) == "armv7l" ]]; then
    echo "✅ Raspberry Pi detected"
else
    echo "⚠️ Not a Raspberry Pi ($(uname -m)); will skip Pi-specific tests"
fi
echo ""

# Test 2: Check Picamera2 system package
echo "2️⃣ Checking Picamera2 (system Python)..."
python3 -c "import picamera2; print('✅ Picamera2 available')" 2>/dev/null || echo "❌ Picamera2 NOT available"
echo ""

# Test 3: Check Python packages in venv
echo "3️⃣ Checking venv packages..."
if [ -d "./venv" ]; then
    echo "✅ venv exists"
    ./venv/bin/python -c "import cv2, flask; print('✅ cv2 and flask available')" 2>/dev/null || echo "⚠️ Missing packages in venv"
else
    echo "❌ venv not found"
fi
echo ""

# Test 4: Check camera devices
echo "4️⃣ Checking camera devices..."
if ls /dev/video* 2>/dev/null | head -3; then
    echo "✅ Video devices found"
else
    echo "❌ No /dev/video* devices found"
fi
echo ""

# Test 5: Test camera capture wrapper
echo "5️⃣ Testing camera capture wrapper (5 sec)..."
if timeout 6 python3 camera_capture_wrapper.py --duration 5 > /dev/null 2>&1; then
    echo "✅ Camera wrapper works"
else
    echo "❌ Camera wrapper failed"
fi
echo ""

# Test 6: Check if TCP ports are free
echo "6️⃣ Checking TCP ports..."
if ! lsof -i :5001 >/dev/null 2>&1; then
    echo "✅ Port 5001 is free"
else
    echo "⚠️ Port 5001 is in use"
fi
if ! lsof -i :8000 >/dev/null 2>&1; then
    echo "✅ Port 8000 is free"
else
    echo "⚠️ Port 8000 is in use"
fi
echo ""

# Test 7: Check network connectivity
echo "7️⃣ Checking network..."
ip_addr=$(hostname -I | awk '{print $1}')
echo "✅ This Pi's IP: $ip_addr"
echo ""

echo "=============================================="
echo "✅ Ready to run:"
echo ""
echo "  Terminal 1 (on Pi):"
echo "    python3 adas_tcp_stream_wrapper.py --no-adas --bind 0.0.0.0 --port 5001"
echo ""
echo "  Terminal 2 (on laptop):"
echo "    ./venv/bin/python tcp_stream_viewer.py --host $ip_addr --port 5001 --web-port 8000"
echo ""
echo "  Browser:"
echo "    http://localhost:8000"
echo ""
