#!/usr/bin/env python3
"""
Video Inference Script - Process video file and save output
No GUI, just processes video once and saves result

Usage:
    python video_inference.py --input video.mp4 --output result.mp4
"""

import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import argparse
from pathlib import Path
import time
import sys
import threading
from queue import Queue
from collections import deque, defaultdict

from ultralytics import YOLO

# Get absolute path to YOLO model
SCRIPT_DIR = Path(__file__).parent.resolve()
CNN_DIR = SCRIPT_DIR.parent
sys.path.append(str(CNN_DIR))

# Import ZoeDepth loader
from scripts.zoedepth_loader import load_zoedepth_model

# Use YOLOv11 from models folder
YOLO_MODEL_PATH = str(CNN_DIR / "models/yolo/yolo11x-seg.pt")


# ============================================================================
# KALMAN FILTER FOR DISTANCE SMOOTHING
# ============================================================================

class KalmanFilter1D:
    """1D Kalman Filter for distance smoothing"""
    
    def __init__(self, process_variance=0.01, measurement_variance=0.1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = None
        self.error_estimate = 1.0
    
    def update(self, measurement):
        if self.estimate is None:
            self.estimate = measurement
            return measurement
        
        prediction = self.estimate
        error_prediction = self.error_estimate + self.process_variance
        kalman_gain = error_prediction / (error_prediction + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.error_estimate = (1 - kalman_gain) * error_prediction
        
        return self.estimate


# ============================================================================
# CONFIGURATION
# ============================================================================
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.4

# Camera parameters (typical rear-view camera)
FOCAL_LENGTH = 1000  # pixels (approximate)

# Real-world heights for distance estimation (in meters)
REAL_HEIGHTS = {
    'Hatchback': 1.5,        # Average hatchback height
    'Sedan': 1.5,            # Average sedan height
    'SUV': 1.8,              # SUVs are taller
    'MUV': 1.9,              # Multi-utility vehicles
    'Bus': 3.2,              # Standard bus height
    'Truck': 3.0,            # Commercial truck height
    'Three-wheeler': 1.6,    # Auto-rickshaw height
    'Two-wheeler': 1.3,      # Motorcycle/scooter with rider
    'LCV': 2.2,              # Light commercial vehicle
    'Mini-bus': 2.5,         # Compact bus
    'Tempo-traveller': 2.4,  # Passenger van
    'Bicycle': 1.2,          # Bicycle with rider
    'Van': 2.0,              # Delivery van
    'Others': 1.5,           # Default estimate
    'Person': 1.7,           # Average human height
    'Person + Two-wheeler': 1.6,
    'Person + Bicycle': 1.6,
    'Person + Three-wheeler': 1.6
}

# Vehicle classes - 12 classes from UVH-26 Filtered (Alphabetical order)
# Excluded: Person, Bicycle, Two-wheeler (handled by YOLO)
VEHICLE_CLASSES = [
    'Bus', 'Hatchback', 'LCV', 'MUV', 'Mini-bus', 
    'Others', 'SUV', 'Sedan', 'Tempo-traveller', 'Three-wheeler', 
    'Truck', 'Van'
]

# YOLO class mapping (map YOLO COCO classes to our CNN classes)
YOLO_CLASS_MAPPING = {
    0: 'Person',         # YOLO person -> Person
    1: 'Bicycle',        # YOLO bicycle -> Bicycle
    2: 'Sedan',          # YOLO car -> Sedan (Generic Car, will be refined)
    3: 'Two-wheeler',    # YOLO motorcycle -> Two-wheeler
    5: 'Bus',            # YOLO bus -> Bus
    7: 'Truck',          # YOLO truck -> Truck
}

# Color mapping for all classes
CLASS_COLORS = {
    'Hatchback': (0, 255, 0),        # Green
    'Sedan': (0, 255, 127),          # Spring Green
    'SUV': (0, 255, 255),            # Cyan
    'MUV': (127, 255, 0),            # Chartreuse
    'Bus': (0, 165, 255),            # Orange-Blue
    'Truck': (255, 165, 0),          # Orange
    'Three-wheeler': (255, 255, 0),  # Yellow
    'Two-wheeler': (255, 0, 127),    # Deep Pink
    'LCV': (255, 127, 80),           # Coral
    'Mini-bus': (138, 43, 226),      # Blue Violet
    'Tempo-traveller': (147, 112, 219), # Medium Purple
    'Bicycle': (255, 192, 203),      # Pink
    'Van': (255, 140, 0),            # Dark Orange
    'Others': (128, 128, 128),       # Gray
    'Person': (255, 0, 255),         # Magenta
    'Person + Two-wheeler': (255, 0, 127),
    'Person + Bicycle': (255, 192, 203),
    'Person + Three-wheeler': (255, 255, 0),
}

# CNN Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])


class VideoDetector:
    def __init__(self, device='cuda', zoedepth_interval=30):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🔥 Device: {self.device}")
        # Load YOLO
        print("📦 Loading YOLO...")
        self.yolo = YOLO(YOLO_MODEL_PATH)
        print("✅ YOLO loaded")
        # Load Fine-tuned Classifier
        CLASSIFIER_PATH = str(CNN_DIR / "models/classifier/weights/best.pt")
        print(f"📦 Loading Classifier: {CLASSIFIER_PATH}")
        self.classifier = YOLO(CLASSIFIER_PATH)
        print("✅ Classifier loaded")
        # Tracking history for velocity estimation
        self.prev_distances = {}  # track_id -> [distances over time]
        self.track_id_counter = 0
        self.prev_boxes = []
        self.track_classes = defaultdict(lambda: deque(maxlen=5))
        self.IOU_THRESHOLD = 0.45
        self.MIN_CROP_AREA = 64 * 64  # pixels
        # Dual-depth system
        print("📦 Loading ZoeDepth (ZoeD_K)...")
        try:
            self.zoedepth_model = load_zoedepth_model("ZoeD_K", device=str(self.device))
            print("✅ ZoeDepth (ZoeD_K) loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load ZoeDepth: {e}")
            self.zoedepth_model = None
        self.zoedepth_interval = zoedepth_interval
        self.zoedepth_frame_counter = 0
        self.last_zoedepth_depth = None
        self.zoedepth_corrections = {}
        self.classical_depth_scale = 1.0
        self.ema_alpha = 0.3
        self.kalman_filters = defaultdict(lambda: KalmanFilter1D())
    
    def match_detections(self, current_boxes, prev_boxes, iou_threshold=0.3):
        """Match current detections with previous ones using IoU"""
        if not prev_boxes:
            return [-1] * len(current_boxes)
        
        matches = []
        for curr_box in current_boxes:
            best_iou = 0
            best_idx = -1
            
            for i, prev_box in enumerate(prev_boxes):
                iou = self.calculate_iou(curr_box, prev_box['bbox'])
                if iou > best_iou and iou > iou_threshold:
                    best_iou = iou
                    best_idx = prev_box.get('track_id', -1)
            
            matches.append(best_idx)
        
        return matches
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def estimate_motion(self, track_id, current_distance):
        """Estimate if object is approaching, stable, or receding"""
        if track_id not in self.prev_distances:
            self.prev_distances[track_id] = []
        
        self.prev_distances[track_id].append(current_distance)
        
        # Keep last 30 frames (approx 1 second) for smoother estimation
        if len(self.prev_distances[track_id]) > 30:
            self.prev_distances[track_id].pop(0)
        
        # Need at least 15 frames to estimate reliably
        if len(self.prev_distances[track_id]) < 15:
            return "stable"
        
        # Calculate trend using robust averaging
        distances = self.prev_distances[track_id]
        time_steps = len(distances) - 1
        if time_steps == 0: return "stable"
        
        # Use average of recent vs old frames to reduce noise
        # Compare average of last 5 frames vs average of first 5 frames
        window = 5
        if len(distances) < window * 2:
            # Fallback for shorter history
            avg_change = (distances[-1] - distances[0]) / time_steps
        else:
            recent_avg = sum(distances[-window:]) / window
            old_avg = sum(distances[:window]) / window
            # Effective time difference is total length minus average offset of the two windows
            effective_steps = len(distances) - window 
            avg_change = (recent_avg - old_avg) / effective_steps
        
        # Threshold for motion detection (meters per frame)
        # 0.03 m/frame * 30 fps = ~1 m/s = 3.6 km/h
        threshold = 0.03
        
        if avg_change < -threshold:  # Getting closer
            return "approaching"
        elif avg_change > threshold:  # Getting farther
            return "receding"
        else:
            return "stable"
    
    def detect_frame(self, frame):
        """Detect vehicles in a single frame with tracking and dual-depth correction"""
        results = []
        current_boxes_raw = []
        # YOLO detection
        yolo_results = self.yolo(frame, verbose=False)[0]
        for i, detection in enumerate(yolo_results.boxes.data):
            x1, y1, x2, y2, conf, cls_id = detection.cpu().numpy()
            cls_id = int(cls_id)
            mask = None
            if yolo_results.masks is not None:
                if i < len(yolo_results.masks.xy):
                    mask = yolo_results.masks.xy[i]
            if cls_id not in YOLO_CLASS_MAPPING:
                continue
            yolo_class = YOLO_CLASS_MAPPING[cls_id]
            if yolo_class in ['Person', 'Bicycle', 'Two-wheeler']:
                if conf >= CONFIDENCE_THRESHOLD:
                    current_boxes_raw.append([int(x1), int(y1), int(x2), int(y2)])
                    results.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'class': yolo_class,
                        'confidence': float(conf),
                        'source': 'YOLO',
                        'mask': mask
                    })
                continue
            if conf >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                final_class = yolo_class
                final_conf = float(conf)
                source = 'YOLO'
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_area = crop.shape[0] * crop.shape[1]
                    if crop_area >= self.MIN_CROP_AREA:
                        try:
                            cls_results = self.classifier(crop, verbose=False)
                            if cls_results and len(cls_results) > 0:
                                top1 = cls_results[0].probs.top1
                                top1_conf = cls_results[0].probs.top1conf.item()
                                pred_class = cls_results[0].names[top1]
                                if top1_conf > 0.4:
                                    final_class = pred_class
                                    final_conf = top1_conf
                                    source = 'YOLO_CLS'
                        except Exception as e:
                            print(f"Classifier error: {e}")
                current_boxes_raw.append([x1, y1, x2, y2])
                results.append({
                    'bbox': [x1, y1, x2, y2],
                    'class': final_class,
                    'confidence': final_conf,
                    'source': source,
                    'mask': mask
                })
        results = self.merge_rider_and_vehicle(results)
        current_boxes_raw = [r['bbox'] for r in results]
        matches = self.match_detections(current_boxes_raw, self.prev_boxes, iou_threshold=self.IOU_THRESHOLD)
        # --- Dual-depth correction logic ---
        self.zoedepth_frame_counter += 1
        run_zoedepth = (self.zoedepth_frame_counter % self.zoedepth_interval == 0)
        if run_zoedepth:
            with torch.no_grad():
                h, w = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = torch.from_numpy(rgb_frame).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
                depth_output = self.zoedepth_model.infer(input_tensor)
                if isinstance(depth_output, dict):
                    depth_map = depth_output.get('metric_depth', depth_output.get('depth'))
                else:
                    depth_map = depth_output
                if depth_map.shape[-2:] != (h, w):
                    depth_map = torch.nn.functional.interpolate(
                        depth_map.unsqueeze(0) if depth_map.dim() == 2 else depth_map,
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze()
                self.last_zoedepth_depth = depth_map.cpu().numpy()
        # Update correction factors if ZoeDepth ran
        if self.last_zoedepth_depth is not None:
            for det in results:
                bbox = det['bbox']
                class_name = det['class']
                h_pixel = bbox[3] - bbox[1]
                real_height = REAL_HEIGHTS.get(class_name, 1.5)
                pinhole_depth = (real_height * FOCAL_LENGTH) / max(h_pixel, 1)
                y1, y2 = max(0, int(bbox[1])), min(self.last_zoedepth_depth.shape[0], int(bbox[3]))
                x1, x2 = max(0, int(bbox[0])), min(self.last_zoedepth_depth.shape[1], int(bbox[2]))
                region = self.last_zoedepth_depth[y1:y2, x1:x2]
                valid_depths = region[region > 0]
                if len(valid_depths) > 10:
                    ml_depth = np.median(valid_depths)
                    if pinhole_depth > 0.1:
                        correction = ml_depth / pinhole_depth
                        if class_name in self.zoedepth_corrections:
                            self.zoedepth_corrections[class_name] = self.ema_alpha * correction + (1 - self.ema_alpha) * self.zoedepth_corrections[class_name]
                        else:
                            self.zoedepth_corrections[class_name] = correction
                        self.classical_depth_scale = self.ema_alpha * correction + (1 - self.ema_alpha) * self.classical_depth_scale
        # Assign track IDs and estimate motion
        for i, result in enumerate(results):
            bbox_height = result['bbox'][3] - result['bbox'][1]
            class_name = result['class']
            real_height = REAL_HEIGHTS.get(class_name, 1.5)
            pinhole_depth = (real_height * FOCAL_LENGTH) / max(bbox_height, 1)
            correction_factor = self.zoedepth_corrections.get(class_name, self.classical_depth_scale)
            corrected_depth = pinhole_depth * correction_factor
            # Kalman filter smoothing
            track_id = matches[i] if matches[i] != -1 else self.track_id_counter
            if matches[i] == -1:
                self.track_id_counter += 1
            smoothed_depth = self.kalman_filters[track_id].update(corrected_depth)
            result['track_id'] = track_id
            result['distance'] = smoothed_depth
            # Estimate motion
            if smoothed_depth:
                motion = self.estimate_motion(track_id, smoothed_depth)
                result['motion'] = motion
            else:
                result['motion'] = "unknown"
            # --- Per-track class smoothing ---
            cls = result['class']
            conf_val = float(result.get('confidence', 0.0))
            self.track_classes[track_id].append((cls, conf_val))
            votes = {}
            for c, cconf in self.track_classes[track_id]:
                votes.setdefault(c, 0.0)
                votes[c] += cconf
            if votes:
                stable_class = max(votes.items(), key=lambda x: x[1])[0]
                stable_conf = votes[stable_class] / len(self.track_classes[track_id])
                if 'Person' in cls:
                    result['class'] = cls
                else:
                    result['class'] = stable_class
                    result['confidence'] = float(min(1.0, stable_conf))
        self.prev_boxes = [{'bbox': r['bbox'], 'track_id': r['track_id']} for r in results]
        return results
    
    def merge_rider_and_vehicle(self, results):
        """
        Merge overlapping 'Person' and 'Two-wheeler'/'Bicycle' detections.
        Returns a filtered list of results.
        """
        final_results = []
        persons = []
        vehicles = []
        others = []
        
        ridable_classes = ['Two-wheeler', 'Bicycle', 'Three-wheeler']
        
        for r in results:
            if r['class'] == 'Person':
                persons.append(r)
            elif r['class'] in ridable_classes:
                vehicles.append(r)
            else:
                others.append(r)
        
        used_persons = set()
        
        for v in vehicles:
            v_box = v['bbox']
            # Start merged box with vehicle box
            mx1, my1, mx2, my2 = v_box
            
            # Check for overlapping persons
            merged_any = False
            for i, p in enumerate(persons):
                if i in used_persons:
                    continue
                
                p_box = p['bbox']
                
                # Check overlap
                xA = max(mx1, p_box[0])
                yA = max(my1, p_box[1])
                xB = min(mx2, p_box[2])
                yB = min(my2, p_box[3])
                
                interArea = max(0, xB - xA) * max(0, yB - yA)
                p_area = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])
                
                # Intersection over Person Area (IoP)
                # If a significant portion of the person overlaps with the vehicle
                iop = interArea / p_area if p_area > 0 else 0
                
                if iop > 0.2: # 20% overlap is enough to consider them related in 2D view
                    # Merge
                    mx1 = min(mx1, p_box[0])
                    my1 = min(my1, p_box[1])
                    mx2 = max(mx2, p_box[2])
                    my2 = max(my2, p_box[3])
                    used_persons.add(i)
                    merged_any = True
            
            # Update vehicle box
            v['bbox'] = [mx1, my1, mx2, my2]
            if merged_any:
                v['class'] = f"Person + {v['class']}"
            final_results.append(v)
            
        # Add remaining persons
        for i, p in enumerate(persons):
            if i not in used_persons:
                final_results.append(p)
                
        # Add others
        final_results.extend(others)
        
        return final_results

    def estimate_distance(self, bbox_height, class_name):
        """Estimate distance based on bounding box height"""
        if bbox_height <= 0:
            return None
        
        # Get real-world height based on class
        real_height = REAL_HEIGHTS.get(class_name, 1.5)  # Default to car height
        
        # Distance = (Real Height × Focal Length) / Pixel Height
        distance = (real_height * FOCAL_LENGTH) / bbox_height
        
        return distance
    
    def draw_detections(self, frame, detections, fps=None):
        """Draw bounding boxes and labels with distance and motion state"""
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            distance = det.get('distance', None)
            motion = det.get('motion', 'unknown')
            
            # Get base color for class
            color = CLASS_COLORS.get(class_name, (255, 255, 255))
            
            # Choose motion-based color for box border
            motion_colors = {
                'approaching': (0, 0, 255),    # Red - Warning!
                'receding': (0, 255, 255),     # Yellow - Moving away
                'stable': (0, 255, 0),         # Green - Safe
                'unknown': color                # Default class color
            }
            box_color = motion_colors.get(motion, color)
            
            # Draw mask if available
            mask = det.get('mask')
            if mask is not None:
                # Create overlay
                overlay = annotated.copy()
                # Convert mask points to int32
                pts = mask.astype(np.int32)
                cv2.fillPoly(overlay, [pts], color)
                # Apply transparency
                alpha = 0.4
                cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)
            
            # Draw box with motion-based color
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw label with distance
            if distance:
                label = f"{class_name}: {confidence:.2f} | {distance:.1f}m"
            else:
                label = f"{class_name}: {confidence:.2f}"
            
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), box_color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Draw distance below box (larger text)
            if distance:
                dist_text = f"{distance:.1f}m"
                dist_size, _ = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(annotated, (x1, y2), 
                             (x1 + dist_size[0] + 10, y2 + dist_size[1] + 10), box_color, -1)
                cv2.putText(annotated, dist_text, (x1 + 5, y2 + dist_size[1] + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # Draw motion state on the right side of box
            if motion != 'unknown':
                motion_text = motion.upper()
                motion_size, _ = cv2.getTextSize(motion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                motion_x = x2 - motion_size[0] - 10
                motion_y = y1 + 25
                cv2.rectangle(annotated, (motion_x - 5, motion_y - motion_size[1] - 5), 
                             (x2 - 5, motion_y + 5), box_color, -1)
                cv2.putText(annotated, motion_text, (motion_x, motion_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw FPS if provided
        if fps:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(annotated, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated


def process_video(input_path, output_path, device='cuda'):
    """Process video file"""
    
    # Check input
    if not Path(input_path).exists():
        print(f"❌ Input video not found: {input_path}")
        return False
    
    # Initialize detector
    detector = VideoDetector(device)
    
    # Open video
    print(f"📹 Opening: {input_path}")
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"❌ Failed to open video")
        return False
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✅ Video: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")
    
    # Setup writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        print(f"❌ Failed to create output video")
        cap.release()
        return False
    
    print(f"💾 Saving to: {output_path}")
    print(f"\n🚀 Processing...\n")
    
    # Process frames
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect
        detections = detector.detect_frame(frame)
        
        # Calculate FPS
        elapsed = time.time() - start_time
        processing_fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Annotate
        # Use video FPS for display so it matches the video speed, not processing speed
        annotated = detector.draw_detections(frame, detections, fps)
        
        # Write
        writer.write(annotated)
        
        # Progress
        if frame_count % 10 == 0 or frame_count == total_frames:
            progress = frame_count / total_frames * 100
            print(f"   Frame {frame_count}/{total_frames} ({progress:.1f}%) - "
                  f"{processing_fps:.1f} FPS", end='\r')
    
    print()
    
    # Cleanup
    cap.release()
    writer.release()
    
    # Stats
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed
    
    print(f"\n✅ Processing complete!")
    print(f"   Frames: {frame_count}")
    print(f"   Time: {elapsed:.1f}s")
    print(f"   Avg FPS: {avg_fps:.1f}")
    print(f"   Output: {output_path}")
    
    # Verify output
    output_file = Path(output_path)
    if output_file.exists():
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"   Size: {size_mb:.2f} MB")
        return True
    else:
        print(f"❌ Output file not created")
        return False


def main():
    parser = argparse.ArgumentParser(description='Video Detection Processor')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input video file')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output video file')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🎬 VIDEO DETECTION PROCESSOR")
    print("="*60 + "\n")
    
    success = process_video(args.input, args.output, args.device)
    
    if success:
        print("\n" + "="*60)
        print("🎉 Done!")
        print("="*60 + "\n")
        return 0
    else:
        print("\n" + "="*60)
        print("❌ Processing failed")
        print("="*60 + "\n")
        return 1


if __name__ == "__main__":
    exit(main())
