#!/usr/bin/env python3
"""Test V4L2 camera device with various approaches"""
import cv2
import time
import subprocess

print("🔍 Testing V4L2 camera /dev/video0")
print("=" * 50)

# Test 1: Simple OpenCV read
print("\n1️⃣ Approach: Simple OpenCV read (no warmup)")
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()
if ret and frame is not None:
    print(f"✅ SUCCESS: {frame.shape}")
else:
    print(f"❌ FAILED: ret={ret}, frame={'None' if frame is None else 'exists'}")


# Test 2: OpenCV with warmup
print("\n2️⃣ Approach: OpenCV with 1 second warmup")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(1.0)  # Warmup
ret, frame = cap.read()
cap.release()
if ret and frame is not None:
    print(f"✅ SUCCESS: {frame.shape}")
else:
    print(f"❌ FAILED: ret={ret}")


# Test 3: OpenCV with multiple attempts
print("\n3️⃣ Approach: OpenCV with 10 frame attempts")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
for i in range(10):
    ret, frame = cap.read()
    if ret and frame is not None and frame.size > 0:
        print(f"✅ SUCCESS on attempt {i+1}: {frame.shape}")
        break
    else:
        print(f"  attempt {i+1}: ret={ret}, size={frame.size if frame is not None else 0}")
    time.sleep(0.1)
else:
    print(f"❌ FAILED after 10 attempts")
cap.release()


# Test 4: Using libcamera directly
print("\n4️⃣ Approach: Test libcamera-hello (camera availability)")
try:
    result = subprocess.run(['libcamera-hello', '--version'], capture_output=True, text=True, timeout=2)
    if result.returncode == 0:
        print(f"✅ libcamera available")
    else:
        print(f"❌ libcamera failed: {result.stderr}")
except subprocess.TimeoutExpired:
    print(f"⚠️ libcamera-hello timed out")
except FileNotFoundError:
    print(f"❌ libcamera-hello not found")


# Test 5: Check device capabilities
print("\n5️⃣ Approach: V4L2 device info")
try:
    result = subprocess.run(['v4l2-ctl', '-d', '/dev/video0', '--all'], capture_output=True, text=True, timeout=2)
    if result.returncode == 0:
        lines = result.stdout.split('\n')[:15]  # First 15 lines
        print("✅ Device info:")
        for line in lines:
            if line.strip():
                print(f"  {line}")
    else:
        print(f"❌ v4l2-ctl failed")
except Exception as e:
    print(f"❌ Error: {e}")


print("\n" + "=" * 50)
print("📊 Summary: Check above for which approach works")
print("   Use that approach in adas_tcp_stream.py")
