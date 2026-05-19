#!/usr/bin/env python3
"""
TCP streaming server for ADAS camera inference using Picamera2 wrapper.
Uses subprocess to run camera capture with system Python (which has Picamera2).

Run on the camera device (Pi):
  python adas_tcp_stream_wrapper.py --bind 0.0.0.0 --port 5001

Run the viewer on the laptop:
  python tcp_stream_viewer.py --host <pi-ip> --port 5001
"""

import argparse
import socket
import struct
import time
import subprocess
import sys
import os
import signal
import threading

import cv2
import numpy as np

from camera_inference import (
    CameraVehicleDetector,
    is_raspberry_pi,
)


def drain_subprocess_stderr(process):
    """Forward camera subprocess diagnostics without letting the stderr pipe fill."""
    if process.stderr is None:
        return

    for line in iter(process.stderr.readline, b""):
        if not line:
            break
        print(line.decode(errors="replace").rstrip(), file=sys.stderr)


def start_camera_process(args):
    print("🚀 Starting camera capture subprocess...")
    process = subprocess.Popen(
        ["python3", "camera_capture_wrapper.py",
         "--width", str(args.width),
         "--height", str(args.height),
         "--fps", str(args.fps)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0
    )
    threading.Thread(target=drain_subprocess_stderr, args=(process,), daemon=True).start()
    print("✅ Camera capture subprocess started")

    time.sleep(2)  # Let subprocess initialize

    if process.poll() is not None:
        print(f"❌ Camera subprocess failed with exit code {process.returncode}")
        sys.exit(1)

    return process


def stop_camera_process(process):
    if process is None:
        return

    try:
        process.terminate()
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="ADAS TCP stream server with vehicle detection")
    parser.add_argument("--bind", type=str, default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=5001, help="Bind port")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Capture FPS")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality (1-100)")
    parser.add_argument("--raw-stream", action="store_true", help="Stream raw frames without ADAS detection")
    parser.add_argument("--max-failures", type=int, default=30, help="Max consecutive frame read failures")
    parser.add_argument("--detect-width", type=int, default=None, help="Resolution for detection (lower = faster)")
    parser.add_argument("--detect-height", type=int, default=None, help="Resolution for detection")
    args = parser.parse_args()

    if not is_raspberry_pi():
        print("❌ This server should run on Raspberry Pi")
        sys.exit(1)

    camera_process = None
    client = None
    server = None

    def handle_shutdown(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    # Initialize ADAS vehicle detector (from camera_inference.py)
    detector = None
    if not args.raw_stream:
        print("📦 Initializing Vehicle Detector (CameraVehicleDetector)...")
        detector = CameraVehicleDetector(device="cpu", frame_width=args.width, frame_height=args.height)
        print("✅ Vehicle Detector ready for real-time detection")

    consecutive_failures = 0
    frame_count = 0
    first_success = False
    start_time = time.time()
    last_fps_time = time.time()
    
    # Determine detection resolution
    detect_width = args.detect_width if args.detect_width else args.width
    detect_height = args.detect_height if args.detect_height else args.height
    
    if detect_width != args.width or detect_height != args.height:
        print(f"📐 Stream: {args.width}x{args.height}, Detection: {detect_width}x{detect_height}")
    print("✅ Camera Vehicle Detector ready with aggressive YOLO caching")
    print("   (YOLO runs every ~10 frames, cached detections used otherwise)")
    
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.bind, args.port))
        server.listen(1)
        print(f"✅ Listening on {args.bind}:{args.port}")
        print("⏳ Waiting for viewer before opening the camera...")

        client, addr = server.accept()
        print(f"✅ Client connected: {addr[0]}:{addr[1]}")

        # Start Picamera2 only after a client connects so idle/test runs do not
        # keep the camera pipeline busy.
        camera_process = start_camera_process(args)

        while True:
            frame_start = time.time()
            
            # Read length-prefixed JPEG from subprocess
            try:
                header_data = camera_process.stdout.read(4)
                if not header_data or len(header_data) < 4:
                    print("⚠️ Failed to read frame header from subprocess")
                    consecutive_failures += 1
                    if consecutive_failures >= args.max_failures:
                        print(f"❌ Subprocess not responding after {args.max_failures} attempts")
                        break
                    time.sleep(0.1)
                    continue
                
                frame_len = struct.unpack("!I", header_data)[0]
                if frame_len <= 0 or frame_len > 10_000_000:  # Max 10MB
                    print(f"⚠️ Invalid frame size: {frame_len}")
                    consecutive_failures += 1
                    continue
                
                frame_data = camera_process.stdout.read(frame_len)
                if len(frame_data) < frame_len:
                    print(f"⚠️ Incomplete frame: got {len(frame_data)}/{frame_len}")
                    consecutive_failures += 1
                    continue
                
                consecutive_failures = 0
                frame_count += 1
                
                if not first_success:
                    print(f"✅ First frame received! Starting detection stream\n")
                    first_success = True
                
                # Measure time to decode
                decode_start = time.time()
                jpg_array = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)
                decode_time = (time.time() - decode_start) * 1000
                
                if frame is None:
                    if frame_count < 5:
                        print(f"⚠️ Failed to decode JPEG from subprocess")
                    continue
                
                # Measure detection time
                detect_start = time.time()
                
                # Optionally resize frame for detection (faster on lower res)
                frame_for_detection = frame
                if detect_width != args.width or detect_height != args.height:
                    frame_for_detection = cv2.resize(frame, (detect_width, detect_height))
                
                # Run detection - internally uses caching to skip YOLO on most frames
                detections = detector.detect_frame(frame_for_detection) if detector is not None else []
                
                # Get safety assessments
                safety_assessments, scenario_info = detector.validate_and_assess_rear_scenario(detections) if detector is not None else ({}, {})
                
                for det in detections:
                    track_id = det.get("track_id", -1)
                    if track_id in safety_assessments:
                        det["safety_assessment"] = safety_assessments[track_id]
                
                detect_time = (time.time() - detect_start) * 1000
                
                # Draw detections (whether new or cached)
                if detector is not None and detections:
                    frame = detector.draw_detections(frame, detections, fps=None)
                
                # Measure encoding time
                encode_start = time.time()
                ok, jpg_enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
                if not ok:
                    if frame_count < 5:
                        print(f"⚠️ JPEG encoding failed for frame {frame_count}")
                    continue
                encode_time = (time.time() - encode_start) * 1000
                jpg = jpg_enc.tobytes()
                
                # Measure send time
                send_start = time.time()
                header = struct.pack("!I", len(jpg))
                try:
                    client.sendall(header + jpg)
                except Exception as e:
                    print(f"❌ Failed to send frame: {e}")
                    break
                send_time = (time.time() - send_start) * 1000
                
                frame_total = (time.time() - frame_start) * 1000
                
                # Calculate and display FPS every 30 frames
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    actual_fps = frame_count / elapsed
                    print(f"\n📊 Frame {frame_count}")
                    print(f"   🎬 Actual FPS: {actual_fps:.2f}")
                    print(f"   ⏱️  Timings (ms):")
                    print(f"      Decode:    {decode_time:.2f}ms")
                    print(f"      Detect:    {detect_time:.2f}ms ({len(detections)} detections)")
                    print(f"      Encode:    {encode_time:.2f}ms")
                    print(f"      Send:      {send_time:.2f}ms")
                    print(f"      Total:     {frame_total:.2f}ms")
                    print(f"      Theory max FPS: {1000/frame_total:.2f}")
                    print()
            
            except BrokenPipeError:
                print(f"❌ Subprocess pipe broken")
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"❌ Error: {e}")
                break
    except KeyboardInterrupt as e:
        print(f"\n🛑 Shutdown requested ({e})")
    
    finally:
        print("🛑 Shutting down...")
        try:
            if client is not None:
                client.close()
        except Exception:
            pass
        try:
            if server is not None:
                server.close()
        except Exception:
            pass

        stop_camera_process(camera_process)
        
        print("✅ Cleanup complete")


if __name__ == "__main__":
    main()
