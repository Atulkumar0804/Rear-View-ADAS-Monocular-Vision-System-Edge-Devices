#!/usr/bin/env python3
"""
Camera frame reader wrapper for Raspberry Pi with Picamera2

This script runs with system Python (which has Picamera2) and streams frames
via stdout or network socket to the main application.
"""
import sys
import os

# Use system Python's packages
print("Importing Picamera2 from system packages...", file=sys.stderr)

try:
    from picamera2 import Picamera2
    print("✅ Picamera2 imported successfully", file=sys.stderr)
except ImportError as e:
    print(f"❌ Failed to import Picamera2: {e}", file=sys.stderr)
    sys.exit(1)

import numpy as np
import cv2
import struct
import time
import argparse
import signal


shutdown_requested = False


def handle_shutdown(signum, _frame):
    global shutdown_requested
    shutdown_requested = signum


signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

def capture_frames(width=640, height=480, fps=30, duration=None):
    """Capture frames from Picamera2 and write to stdout"""
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"format":'XRGB8888',"size":(width,height)}))
    picam2.start()
    time.sleep(0.5)  # Warmup
    
    print(f"📷 Camera started: {width}x{height} @ {fps} FPS", file=sys.stderr)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while not shutdown_requested:
            frame = picam2.capture_array()
            if frame is None:
                continue
            
            # Convert XRGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Encode as JPEG
            ok, jpg = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue
            
            frame_count += 1
            
            # Write length-prefixed JPEG to stdout
            data = jpg.tobytes()
            header = struct.pack("!I", len(data))
            try:
                sys.stdout.buffer.write(header + data)
                sys.stdout.buffer.flush()
            except BrokenPipeError:
                print("🛑 Output pipe closed", file=sys.stderr)
                break
            
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                actual_fps = frame_count / elapsed
                print(f"📊 Frame {frame_count}: {actual_fps:.1f} FPS", file=sys.stderr)
            
            if duration and (time.time() - start_time) > duration:
                break
            
            time.sleep(max(0, 1.0 / fps))
    except KeyboardInterrupt as e:
        print(f"🛑 Capture shutdown requested ({e})", file=sys.stderr)
    finally:
        if shutdown_requested:
            print(f"🛑 Capture shutdown requested (received signal {shutdown_requested})", file=sys.stderr)
        try:
            picam2.stop()
        except Exception:
            pass
        try:
            picam2.close()
        except Exception:
            pass
        print(f"✅ Captured {frame_count} frames", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture frames from Picamera2")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=int, default=None)
    args = parser.parse_args()
    
    capture_frames(args.width, args.height, args.fps, args.duration)
