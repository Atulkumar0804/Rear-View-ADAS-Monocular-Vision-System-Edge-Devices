#!/usr/bin/env python3
"""
Distance Prediction Using Calibrated System
============================================

Use the calibrated focal length from interactive calibration
to predict accurate distances in real-time.

Usage:
    python3 scripts/predict_distance_calibrated.py --camera 4

Author: ADAS Research Team
Date: 2026-02-02
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
import sys

# Add parent directory to path
CNN_DIR = Path(__file__).parent.parent
sys.path.append(str(CNN_DIR))

from ultralytics import YOLO


class CalibratedDistancePredictor:
    """
    Predict distances using calibrated focal length
    """
    
    def __init__(self, camera_index=4, calibration_dir='calibration_data/distance_samples'):
        self.camera_index = camera_index
        self.calibration_dir = Path(calibration_dir)
        
        # Load calibrated focal length
        self.focal_length = self.load_focal_length()
        
        # Load YOLO
        print("📦 Loading YOLO...")
        yolo_path = CNN_DIR / "models/yolo/yolo11x-seg.pt"
        if not yolo_path.exists():
            yolo_path = "yolo11n.pt"
        
        self.yolo = YOLO(str(yolo_path))
        print("✅ YOLO loaded")
        
        # Person height (adjustable)
        self.person_height = 1.7  # meters
        
    def load_focal_length(self):
        """Load calibrated focal length"""
        config_path = self.calibration_dir / 'focal_length_calibrated.json'
        
        if not config_path.exists():
            print("⚠️ No calibration found. Using default focal length: 1000.0")
            print("   Run: python3 scripts/calibrate_distance_interactive.py")
            return 1000.0
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        focal_length = config['focal_length']
        print(f"✅ Loaded calibrated focal length: {focal_length:.2f} pixels")
        print(f"   Based on {config['num_samples']} calibration samples")
        print(f"   Distance range: {config['distance_range'][0]:.1f}m - {config['distance_range'][1]:.1f}m")
        
        return focal_length
    
    def predict_distance(self, pixel_height):
        """
        Predict distance using pinhole camera formula
        
        distance = (real_height * focal_length) / pixel_height
        """
        if pixel_height <= 0:
            return None
        
        distance = (self.person_height * self.focal_length) / pixel_height
        return distance
    
    def detect_and_measure(self, frame):
        """Detect people and measure distances"""
        results = self.yolo(frame, verbose=False, conf=0.3)
        
        detections = []
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return detections
        
        # Find person detections
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            if class_id == 0:  # Person class
                bbox = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                
                pixel_height = bbox[3] - bbox[1]
                distance = self.predict_distance(pixel_height)
                
                detections.append({
                    'bbox': bbox,
                    'confidence': conf,
                    'pixel_height': pixel_height,
                    'distance': distance
                })
        
        return detections
    
    def run(self):
        """Run real-time distance prediction"""
        print("\n" + "=" * 70)
        print("CALIBRATED DISTANCE PREDICTION")
        print("=" * 70)
        print(f"Focal Length: {self.focal_length:.2f} pixels")
        print(f"Person Height: {self.person_height}m")
        print()
        print("Press 'q' to quit")
        print("Press '+' to increase person height")
        print("Press '-' to decrease person height")
        print("=" * 70)
        print()
        
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"❌ Cannot open camera {self.camera_index}")
            return
        
        print(f"📷 Camera {self.camera_index} opened\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect and measure
            detections = self.detect_and_measure(frame)
            
            # Create display
            display = frame.copy()
            
            if detections:
                for det in detections:
                    bbox = det['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    distance = det['distance']
                    
                    # Choose color based on distance
                    if distance < 5:
                        color = (0, 0, 255)  # Red - Very close
                    elif distance < 10:
                        color = (0, 165, 255)  # Orange - Close
                    elif distance < 20:
                        color = (0, 255, 255)  # Yellow - Medium
                    else:
                        color = (0, 255, 0)  # Green - Far
                    
                    # Draw bounding box
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw distance
                    distance_text = f"{distance:.1f}m"
                    cv2.putText(display, distance_text, 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                    
                    # Draw pixel height
                    cv2.putText(display, f"{det['pixel_height']:.0f}px", 
                               (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show info
            cv2.putText(display, f"Focal Length: {self.focal_length:.1f}px | Person Height: {self.person_height:.2f}m", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(display, f"People detected: {len(detections)}", 
                       (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Calibrated Distance Prediction', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                self.person_height += 0.05
                print(f"Person height: {self.person_height:.2f}m")
            elif key == ord('-') or key == ord('_'):
                self.person_height = max(0.5, self.person_height - 0.05)
                print(f"Person height: {self.person_height:.2f}m")
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Calibrated Distance Prediction')
    parser.add_argument('--camera', type=int, default=4, help='Camera index')
    parser.add_argument('--calibration_dir', type=str, default='calibration_data/distance_samples',
                       help='Directory with calibration data')
    
    args = parser.parse_args()
    
    predictor = CalibratedDistancePredictor(
        camera_index=args.camera,
        calibration_dir=args.calibration_dir
    )
    
    predictor.run()


if __name__ == "__main__":
    main()
