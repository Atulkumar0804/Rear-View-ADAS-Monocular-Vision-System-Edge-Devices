#!/usr/bin/env python3
"""Direct camera frame test without network"""
import sys
sys.path.insert(0, '/usr/lib/python3/dist-packages')

from camera_inference import setup_picamera2, is_raspberry_pi, get_picamera2_frame
import cv2
import time

print(f"Raspberry Pi: {is_raspberry_pi()}")

# Try Picamera2 first
print("\n🔍 Testing Picamera2...")
try:
    cam = setup_picamera2(640, 480, fps=30)
    if cam:
        print("✅ Picamera2 initialized")
        for i in range(5):
            frame = get_picamera2_frame(cam)
            if frame is not None:
                print(f"  ✅ Frame {i+1}: {frame.shape}")
            else:
                print(f"  ⚠️ Frame {i+1}: None")
            time.sleep(0.1)
        cam.stop()
        cam.close()
    else:
        print("❌ Picamera2 init returned None")
except Exception as e:
    print(f"❌ Picamera2 failed: {e}")
    import traceback
    traceback.print_exc()

# Try OpenCV fallback
print("\n🔍 Testing OpenCV /dev/video0...")
try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("✅ OpenCV camera opened")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        for i in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"  ✅ Frame {i+1}: {frame.shape}")
            else:
                print(f"  ⚠️ Frame {i+1}: read failed")
            time.sleep(0.1)
        cap.release()
    else:
        print("❌ OpenCV camera open failed")
except Exception as e:
    print(f"❌ OpenCV failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Test complete")
