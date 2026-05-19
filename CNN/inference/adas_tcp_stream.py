#!/usr/bin/env python3
"""
TCP streaming server for ADAS camera inference.
Sends length-prefixed JPEG frames over a TCP socket.

Run on the camera device (Pi):
  python adas_tcp_stream.py --bind 0.0.0.0 --port 5001

Run the viewer on the laptop:
  python tcp_stream_viewer.py --host <pi-ip> --port 5001
"""

import argparse
import socket
import struct
import time
import sys
import os

import cv2

# Don't add system path globally, handle Picamera2 imports specially
from camera_inference import (
    CameraVehicleDetector,
    get_picamera2_frame,
    get_realsense_frame,
    is_raspberry_pi,
    setup_picamera2,
    setup_realsense_camera,
)


def open_camera(args):
    cap = None
    realsense_pipeline = None
    use_realsense = False
    picamera2_camera = None
    use_picamera2 = False

    if args.realsense:
        realsense_pipeline, _ = setup_realsense_camera(args.width, args.height, fps=args.fps)
        if realsense_pipeline:
            use_realsense = True

    if not use_realsense and args.camera_backend in ["picamera2", "auto"]:
        if is_raspberry_pi():
            try:
                # Try direct import first (from system or venv)
                import picamera2
                picamera2_camera = setup_picamera2(args.width, args.height, fps=args.fps)
                if picamera2_camera is not None:
                    use_picamera2 = True
                    print("✅ Picamera2 initialized successfully")
                else:
                    print("⚠️ Picamera2 available but initialization failed, falling back to OpenCV")
            except ImportError as e:
                print(f"⚠️ Picamera2 not importable: {e}")
                print("   Attempting to use OpenCV with libcamera device...")
        else:
            print("⚠️ Picamera2 requested but not on Raspberry Pi, falling back to OpenCV camera")

    if not use_realsense and not use_picamera2:
        if isinstance(args.camera, str) and args.camera.isdigit():
            camera_source = int(args.camera)
        else:
            camera_source = args.camera

        print(f"📷 Attempting to open camera: {camera_source}")
        
        # Try to open camera without forcing V4L2 backend
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera: {camera_source}")
        
        print(f"✅ Camera device opened: {camera_source}")
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"📐 Resolution: requested {args.width}x{args.height}, got {int(actual_width)}x{int(actual_height)}")
        
        # Set FPS if possible (may fail on some devices)
        try:
            cap.set(cv2.CAP_PROP_FPS, args.fps)
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            if actual_fps > 0:
                print(f"⏱️  FPS: requested {args.fps}, got {actual_fps:.1f}")
        except:
            pass
        
        # Try different formats if MJPG doesn't work
        if args.fourcc:
            fourcc_list = [args.fourcc, "MJPG", "YUYV", "H264"]
            for fourcc_str in fourcc_list:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                    result = cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                    current_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                    current_fourcc_str = "".join([chr((current_fourcc >> 8*i) & 0xFF) for i in range(4)])
                    print(f"🎬 FOURCC tested {fourcc_str}: got {current_fourcc_str}")
                    if current_fourcc_str.strip('\x00'):
                        break
                except:
                    pass
        
        # Set buffer size
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Enable CAP_PROP_AUTOFOCUS if available
        try:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        except:
            pass
        
        # Test frame reading with multiple attempts and formats
        print(f"🔍 Testing frame capture (trying up to 10 times)...")
        frame_read = False
        for attempt in range(10):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print(f"✅ Test read succeeded on attempt {attempt+1}: shape {frame.shape}")
                frame_read = True
                break
            elif attempt < 3:
                print(f"  ⚠️  Attempt {attempt+1}/10: no frame data")
            time.sleep(0.1)
        
        if not frame_read:
            print(f"⚠️ Camera opened but may not be streaming properly")
            print(f"   Continuing anyway - may recover after first client connects")
            # Don't raise here, let streaming loop handle it

    return cap, realsense_pipeline, use_realsense, picamera2_camera, use_picamera2


def read_frame(cap, realsense_pipeline, use_realsense, picamera2_camera, use_picamera2):
    if use_realsense:
        frame = get_realsense_frame(realsense_pipeline)
        return frame
    if use_picamera2:
        return get_picamera2_frame(picamera2_camera)
    ret, frame = cap.read()
    return frame if ret else None


def main():
    parser = argparse.ArgumentParser(description="ADAS TCP stream server")
    parser.add_argument("--bind", type=str, default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=5001, help="Bind port")
    parser.add_argument("--camera", type=str, default="0", help="Camera index or device path")
    parser.add_argument("--camera-backend", type=str, default="auto",
                        choices=["auto", "opencv", "picamera2"],
                        help="Camera backend to use")
    parser.add_argument("--realsense", action="store_true", help="Use RealSense D455")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Capture FPS")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality (1-100)")
    parser.add_argument("--fourcc", type=str, default="MJPG",
                        help="Camera fourcc (e.g., MJPG, YUYV)")
    parser.add_argument("--no-adas", action="store_true", help="Stream raw frames only")
    parser.add_argument("--max-failures", type=int, default=20,
                        help="Max consecutive frame capture failures")
    parser.add_argument("--retry-delay", type=float, default=0.05,
                        help="Delay between frame capture retries (seconds)")
    parser.add_argument("--reopen-every", type=int, default=10,
                        help="Reopen OpenCV camera after N failures")
    args = parser.parse_args()
    
    # On Raspberry Pi with auto mode, force Picamera2
    if args.camera_backend == "auto" and is_raspberry_pi():
        print("🟠 Raspberry Pi detected, forcing Picamera2 backend")
        args.camera_backend = "picamera2"

    cap, realsense_pipeline, use_realsense, picamera2_camera, use_picamera2 = open_camera(args)

    detector = None
    if not args.no_adas:
        detector = CameraVehicleDetector(device="cpu", frame_width=args.width, frame_height=args.height)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.bind, args.port))
    server.listen(1)
    print(f"✅ Listening on {args.bind}:{args.port}")

    client, addr = server.accept()
    print(f"✅ Client connected: {addr[0]}:{addr[1]}")

    consecutive_failures = 0
    frame_count = 0
    first_success = False
    try:
        while True:
            frame = read_frame(cap, realsense_pipeline, use_realsense, picamera2_camera, use_picamera2)
            if frame is None:
                consecutive_failures += 1
                if consecutive_failures == 1:
                    print(f"⚠️ Frame capture started failing")
                    
                if cap and args.reopen_every > 0 and consecutive_failures % args.reopen_every == 0:
                    print(f"🔄 Reopening camera after {consecutive_failures} failures...")
                    try:
                        cap.release()
                    except Exception as e:
                        print(f"   Error releasing: {e}")
                    try:
                        cap, realsense_pipeline, use_realsense, picamera2_camera, use_picamera2 = open_camera(args)
                    except Exception as e:
                        print(f"❌ Failed to reopen camera: {e}")
                        break
                
                if consecutive_failures >= args.max_failures:
                    print(f"❌ Frame capture failed {args.max_failures} times consecutively, giving up")
                    break
                
                time.sleep(args.retry_delay)
                continue

            consecutive_failures = 0
            frame_count += 1
            
            if not first_success:
                print(f"✅ First frame received! Resolution: {frame.shape}, starting stream")
                first_success = True
            
            if frame_count % 60 == 0:  # Log every 60 frames (~2 sec at 30 fps)
                print(f"📊 Frames sent: {frame_count}")

            if detector is not None:
                detections = detector.detect_frame(frame)
                safety_assessments, _ = detector.validate_and_assess_rear_scenario(detections)
                for det in detections:
                    track_id = det.get("track_id", -1)
                    if track_id in safety_assessments:
                        det["safety_assessment"] = safety_assessments[track_id]
                frame = detector.draw_detections(frame, detections, fps=None)

            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            if not ok:
                if frame_count < 5:
                    print(f"⚠️  JPEG encoding failed for frame {frame_count}")
                continue

            data = jpg.tobytes()
            header = struct.pack("!I", len(data))
            try:
                client.sendall(header + data)
            except Exception as e:
                print(f"❌ Socket send failed: {e}")
                break

            if args.fps > 0:
                time.sleep(max(0.0, 1.0 / args.fps))
    finally:
        print("🛑 Shutting down...")
        try:
            client.close()
        except Exception:
            pass
        server.close()
        if use_realsense and realsense_pipeline:
            realsense_pipeline.stop()
        if use_picamera2 and picamera2_camera:
            try:
                picamera2_camera.stop()
                picamera2_camera.close()
            except Exception:
                pass
        if cap:
            cap.release()
        print("✅ Cleanup complete")


if __name__ == "__main__":
    main()
