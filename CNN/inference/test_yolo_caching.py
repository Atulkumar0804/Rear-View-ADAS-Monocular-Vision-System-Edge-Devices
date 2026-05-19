#!/usr/bin/env python3
"""Test YOLO caching optimization"""

import numpy as np
import time
from camera_inference import CameraVehicleDetector

print("=" * 60)
print("Testing YOLO Caching Optimization")
print("=" * 60)

# Create detector
detector = CameraVehicleDetector(device="cpu", frame_width=320, frame_height=240, fps=30)

# Create dummy frame
frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

print("\n📊 Measuring YOLO call frequency...")
print(f"   skip_yolo_frames = {detector.skip_yolo_frames}")
print(f"   This means YOLO runs every {detector.skip_yolo_frames} frames\n")

# Track YOLO calls
yolo_call_count = 0
original_yolo_call = detector.yolo.__call__

call_log = []

def tracked_yolo_call(*args, **kwargs):
    global yolo_call_count
    yolo_call_count += 1
    call_log.append(detector.frame_counter)
    return original_yolo_call(*args, **kwargs)

detector.yolo.__call__ = tracked_yolo_call

# Run 30 frames
num_frames = 30
print(f"Running {num_frames} detect_frame calls...\n")

frame_times = []
for i in range(num_frames):
    start = time.time()
    detections = detector.detect_frame(frame)
    frame_time = (time.time() - start) * 1000
    frame_times.append(frame_time)

print(f"Results:")
print(f"  YOLO calls made: {yolo_call_count} out of {num_frames}")
print(f"  YOLO frames: {call_log}")
print(f"  Expected (every 10 frames): frames 0, 10, 20, etc.")
print()

avg_time = np.mean(frame_times)
min_time = np.min(frame_times)
max_time = np.max(frame_times)

print(f"Timing:")
print(f"  Avg frame time: {avg_time:.1f}ms")
print(f"  Min frame time: {min_time:.1f}ms (cached YOLO)")
print(f"  Max frame time: {max_time:.1f}ms (running YOLO)")
print(f"  Theoretical FPS (avg): {1000/avg_time:.1f}")
print(f"  Theoretical FPS (min): {1000/min_time:.1f} (cached frames)")
print()

# Check if optimization is working
expected_yolo_count = (num_frames + detector.skip_yolo_frames - 1) // detector.skip_yolo_frames
if 1 <= yolo_call_count <= expected_yolo_count + 1:
    print("✅ YOLO caching working correctly!")
    print(f"   {yolo_call_count} YOLO calls for {num_frames} frames")
    print(f"   Reduction factor: {num_frames/yolo_call_count:.1f}x")
else:
    print(f"⚠️  Unexpected YOLO call count: {yolo_call_count}")

if avg_time < 150:
    print("✅ Average frame time is FAST (<150ms)")
    print(f"   Should achieve 15-20 FPS in production")
else:
    print(f"⚠️  Average frame time is {avg_time:.0f}ms (might not reach 20 FPS)")

print("\n" + "=" * 60)
