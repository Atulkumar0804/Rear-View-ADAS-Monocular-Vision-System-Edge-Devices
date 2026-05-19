#!/usr/bin/env python3
"""
Test which /dev/video* devices actually work for frame capture
"""
import cv2
import sys

def test_device(device_path, width=640, height=480):
    """Try to open and read a frame from a device"""
    print(f"\n🔍 Testing {device_path}...")
    
    cap = cv2.VideoCapture(device_path)
    if not cap.isOpened():
        print(f"  ❌ Cannot open {device_path}")
        return False
    
    print(f"  ✅ Device opened")
    
    # Try to set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"  📐 Resolution: {int(actual_w)}x{int(actual_h)}")
    
    # Try to read frames
    for attempt in range(3):
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"  ✅ Frame read successful! Shape: {frame.shape}")
            cap.release()
            return True
        else:
            print(f"  ⚠️  Attempt {attempt+1}/3: read failed")
    
    cap.release()
    print(f"  ❌ Cannot read frames from {device_path}")
    return False

if __name__ == "__main__":
    print("Testing /dev/video* devices on this system...")
    
    found = False
    for i in range(0, 20):
        device = f"/dev/video{i}"
        if test_device(device):
            print(f"\n✅✅✅ WORKING DEVICE FOUND: {device}")
            found = True
    
    if not found:
        print("\n❌ No working video device found")
        sys.exit(1)
