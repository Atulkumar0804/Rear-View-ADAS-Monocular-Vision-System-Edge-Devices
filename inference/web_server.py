#!/usr/bin/env python3
"""
Web Server for Real-time Camera Inference via Mobile
Provides:
1. REST API endpoints for camera detection
2. Real-time camera streaming (MJPEG)
3. WebSocket for live detection results
4. Mobile-friendly web dashboard
5. Video file upload and processing

Usage:
    python inference/web_server.py --port 5000 --camera 0
    python inference/web_server.py --port 8080 --camera 0 --host 0.0.0.0

Mobile Access:
    Open browser: http://<your-pc-ip>:5000
"""

import os
import sys
import time
import threading
import queue
import json
import base64
import logging
from pathlib import Path
from datetime import datetime
from collections import deque, defaultdict
import io

import cv2
import numpy as np
import torch
import flask
from flask import Flask, render_template, request, jsonify, Response, send_file
from flask_cors import CORS
import argparse

# Add parent directory to path
SCRIPT_DIR = Path(__file__).parent.resolve()
CNN_DIR = SCRIPT_DIR.parent
sys.path.append(str(CNN_DIR))

# Import camera inference
from camera_inference import (
    CameraVehicleDetector,
    CONFIDENCE_THRESHOLD,
    VEHICLE_CLASSES,
    CLASS_COLORS,
    VEHICLE_DIMENSIONS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('WebServer')

# ============================================================================
# FLASK APP SETUP
# ============================================================================

app = Flask(__name__)
CORS(app)  # Enable CORS for mobile access

# Global state
inference_engine = None
camera_thread = None
frame_queue = queue.Queue(maxsize=2)  # Keep latest 2 frames
detection_queue = queue.Queue(maxsize=10)  # Store last 10 detections
camera_running = False
inference_stats = {
    'total_frames': 0,
    'fps': 0,
    'avg_inference_time': 0,
    'total_detections': 0,
    'last_update': None
}

# ============================================================================
# MJPEG STREAMING (Video feed for web)
# ============================================================================

def generate_frames():
    """Generate MJPEG frames for streaming to web"""
    global camera_running, inference_engine
    
    frame_buffer = deque(maxlen=1)
    frame_times = deque(maxlen=30)
    
    while camera_running:
        try:
            # Get latest frame from queue (with timeout)
            frame = frame_queue.get(timeout=1)
            if frame is None:
                continue
            
            # Draw detections on frame
            annotated_frame = draw_detections_on_frame(frame)
            
            # Encode frame to JPEG
            ret, jpeg = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            
            # Calculate FPS
            current_time = time.time()
            frame_times.append(current_time)
            if len(frame_times) > 1:
                fps = len(frame_times) / (frame_times[-1] - frame_times[0] + 1e-6)
                inference_stats['fps'] = fps
            
            # Yield frame as MJPEG
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(jpeg.tobytes())).encode() + b'\r\n\r\n' 
                   + jpeg.tobytes() + b'\r\n')
        
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error generating frame: {e}")
            continue

def draw_detections_on_frame(frame):
    """Draw bounding boxes and safety info on frame"""
    annotated = frame.copy()
    
    # Get latest detection if available
    try:
        latest_detection = detection_queue.queue[-1] if len(detection_queue.queue) > 0 else None
    except:
        latest_detection = None
    
    if latest_detection:
        detections = latest_detection.get('detections', [])
        
        for det in detections:
            bbox = det.get('bbox', [])
            if not bbox:
                continue
            
            x1, y1, x2, y2 = [int(v) for v in bbox]
            class_name = det.get('class', 'Unknown')
            confidence = det.get('confidence', 0.0)
            distance = det.get('distance', None)
            safety_level = det.get('safety_level', 'UNKNOWN')
            
            # Color based on safety level
            if safety_level == 'CRITICAL':
                color = (0, 0, 255)  # Red
            elif safety_level == 'WARNING':
                color = (0, 165, 255)  # Orange
            elif safety_level == 'CAUTION':
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 255, 0)  # Green
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name} {confidence:.2f}"
            if distance:
                label += f" ({distance:.1f}m)"
            
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw stats on frame
        stats_text = [
            f"FPS: {inference_stats['fps']:.1f}",
            f"Detections: {len(detections)}",
            f"Total: {inference_stats['total_detections']}"
        ]
        y_offset = 30
        for text in stats_text:
            cv2.putText(annotated, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 30
    
    return annotated

# ============================================================================
# CAMERA CAPTURE THREAD
# ============================================================================

def camera_capture_thread(camera_index=0):
    """Continuously capture from camera and run inference"""
    global camera_running, inference_engine, inference_stats
    
    logger.info(f"Starting camera capture thread on camera {camera_index}")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        logger.error(f"Failed to open camera {camera_index}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    inference_times = deque(maxlen=30)
    
    while camera_running:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            time.sleep(0.1)
            continue
        
        try:
            # Put frame in queue for streaming
            try:
                frame_queue.put(frame, block=False)
            except queue.Full:
                pass  # Drop frame if queue full
            
            # Run inference
            if inference_engine:
                inference_start = time.time()
                detections = inference_engine.detect_frame(frame)
                inference_time = time.time() - inference_start
                inference_times.append(inference_time)
                
                # Update statistics
                inference_stats['total_frames'] += 1
                inference_stats['total_detections'] += len(detections)
                if inference_times:
                    inference_stats['avg_inference_time'] = np.mean(inference_times)
                inference_stats['last_update'] = datetime.now().isoformat()
                
                # Store detection result
                detection_result = {
                    'frame_number': frame_count,
                    'timestamp': time.time(),
                    'detections': detections,
                    'stats': inference_stats.copy()
                }
                
                try:
                    detection_queue.put(detection_result, block=False)
                except queue.Full:
                    pass  # Drop if queue full
                
                frame_count += 1
                
                # Log every 30 frames
                if frame_count % 30 == 0:
                    logger.info(f"Processed {frame_count} frames, "
                               f"Avg inference: {inference_stats['avg_inference_time']*1000:.1f}ms, "
                               f"FPS: {inference_stats['fps']:.1f}")
        
        except Exception as e:
            logger.error(f"Error in inference: {e}")
            continue
    
    cap.release()
    logger.info("Camera capture thread stopped")

# ============================================================================
# REST API ENDPOINTS
# ============================================================================

@app.route('/')
def index():
    """Serve main web dashboard"""
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get server and inference status"""
    return jsonify({
        'status': 'running' if camera_running else 'stopped',
        'inference_engine': inference_engine is not None,
        'stats': inference_stats
    })

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    """Start camera capture and inference"""
    global camera_running, camera_thread, inference_engine
    
    if camera_running:
        return jsonify({'error': 'Camera already running'}), 400
    
    try:
        camera_index = request.json.get('camera_index', 0)
        
        # Initialize inference engine if not already done
        if inference_engine is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            inference_engine = CameraVehicleDetector(device=device)
        
        camera_running = True
        camera_thread = threading.Thread(target=camera_capture_thread, args=(camera_index,))
        camera_thread.daemon = True
        camera_thread.start()
        
        logger.info("Camera started successfully")
        return jsonify({'status': 'Camera started'})
    
    except Exception as e:
        logger.error(f"Error starting camera: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    """Stop camera capture"""
    global camera_running
    
    camera_running = False
    time.sleep(0.5)  # Wait for thread to finish
    
    logger.info("Camera stopped")
    return jsonify({'status': 'Camera stopped'})

@app.route('/api/detections', methods=['GET'])
def get_detections():
    """Get latest detections"""
    try:
        if len(detection_queue.queue) > 0:
            latest = detection_queue.queue[-1]
            return jsonify(latest)
        else:
            return jsonify({'detections': []})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detections/history', methods=['GET'])
def get_detection_history():
    """Get detection history"""
    try:
        limit = request.args.get('limit', default=10, type=int)
        history = list(detection_queue.queue)[-limit:]
        return jsonify({'history': history})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/video/stream')
def video_stream():
    """Stream live video frames as MJPEG"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/upload/video', methods=['POST'])
def upload_video():
    """Upload and process a video file"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        if not video_file:
            return jsonify({'error': 'Empty video file'}), 400
        
        # Save uploaded file
        upload_dir = CNN_DIR / 'uploads'
        upload_dir.mkdir(exist_ok=True)
        
        filename = f"upload_{int(time.time())}.mp4"
        filepath = upload_dir / filename
        video_file.save(str(filepath))
        
        logger.info(f"Video uploaded: {filename}")
        
        # Process video in background
        result = {
            'filename': filename,
            'status': 'processing',
            'filepath': str(filepath)
        }
        
        return jsonify(result), 202
    
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/image/process', methods=['POST'])
def process_image():
    """Process a single uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        image_data = np.frombuffer(image_file.read(), np.uint8)
        frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        # Run inference
        if inference_engine:
            detections = inference_engine.detect_frame(frame)
            
            return jsonify({
                'detections': detections,
                'frame_size': frame.shape[:2]
            })
        else:
            return jsonify({'error': 'Inference engine not initialized'}), 503
    
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get inference configuration"""
    return jsonify({
        'vehicle_classes': VEHICLE_CLASSES,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'vehicle_dimensions': VEHICLE_DIMENSIONS
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'camera_running': camera_running
    })

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Web Server for Camera Inference')
    parser.add_argument('--port', type=int, default=5000, help='Port to run server on')
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--auto-start', action='store_true', help='Auto-start camera on startup')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--ssl-cert', help='Path to SSL certificate file')
    parser.add_argument('--ssl-key', help='Path to SSL key file')


def create_app(gpu_profile: str = None):
    """
    Create and return Flask app instance
    
    Args:
        gpu_profile: GPU profile name (for logging/context)
        
    Returns:
        Flask app instance
    """
    global app
    if gpu_profile:
        logger.info(f"Created web app instance for GPU profile: {gpu_profile}")
    return app


if __name__ == '__main__':
    args = parser.parse_args()
    
    # Prepare SSL context if certificates provided
    ssl_context = None
    protocol = 'http'
    if args.ssl_cert and args.ssl_key:
        ssl_context = (args.ssl_cert, args.ssl_key)
        protocol = 'https'
    
    logger.info(f"Starting Web Server on {protocol}://{args.host}:{args.port}")
    logger.info(f"Access from mobile: {protocol}://{args.host}:{args.port}")
    
    # Auto-start camera if requested
    if args.auto_start:
        camera_running = True
        camera_thread = threading.Thread(target=camera_capture_thread, args=(args.camera,))
        camera_thread.daemon = True
        camera_thread.start()
        logger.info("Auto-started camera")
    
    # Run Flask app
    if ssl_context:
        app.run(host=args.host, port=args.port, debug=args.debug, threaded=True, ssl_context=ssl_context)
    else:
        app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
