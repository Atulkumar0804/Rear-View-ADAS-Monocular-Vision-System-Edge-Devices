#!/usr/bin/env python3
"""
Interactive Distance Calibration System
========================================

Capture images of objects at known distances to train accurate depth estimation.

This system:
1. Captures images of people/objects at distances you specify
2. Saves images with ground truth distance labels
3. Trains/updates the depth model based on real-world measurements
4. Improves distance prediction accuracy over time

Usage:
    python3 scripts/calibrate_distance_interactive.py --camera 4

Author: ADAS Research Team
Date: 2026-02-02
"""

import cv2
import numpy as np
import torch
import json
import argparse
from pathlib import Path
from datetime import datetime
import sys
from PIL import Image

# Add parent directory to path
CNN_DIR = Path(__file__).parent.parent
sys.path.append(str(CNN_DIR))

from ultralytics import YOLO


class DistanceCalibrationSystem:
    """
    Interactive system to collect distance calibration data
    """
    
    def __init__(self, camera_index=4, save_dir='calibration_data/distance_samples'):
        self.camera_index = camera_index
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load YOLO for person detection
        print("📦 Loading YOLO for person detection...")
        yolo_path = CNN_DIR / "models/yolo/yolo11x-seg.pt"
        if not yolo_path.exists():
            yolo_path = "yolo11n.pt"  # Fallback to small model
        
        self.yolo = YOLO(str(yolo_path))
        print("✅ YOLO loaded")
        
        # Load existing calibration data
        self.calibration_file = self.save_dir / 'distance_calibration.json'
        self.samples = self.load_calibration_data()
        
        # Statistics
        self.focal_length_estimates = []
        
    def load_calibration_data(self):
        """Load existing calibration samples"""
        if self.calibration_file.exists():
            with open(self.calibration_file, 'r') as f:
                data = json.load(f)
            print(f"📂 Loaded {len(data['samples'])} existing calibration samples")
            return data['samples']
        return []
    
    def save_calibration_data(self):
        """Save calibration samples to disk"""
        data = {
            'samples': self.samples,
            'num_samples': len(self.samples),
            'last_updated': datetime.now().isoformat(),
            'focal_length_estimates': self.focal_length_estimates
        }
        
        with open(self.calibration_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Saved {len(self.samples)} calibration samples")
    
    def detect_person(self, frame):
        """Detect person in frame and return bbox"""
        results = self.yolo(frame, verbose=False, conf=0.3)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None
        
        # Find person detections (class 0 in COCO)
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            if class_id == 0:  # Person class
                bbox = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                return {
                    'bbox': bbox,
                    'confidence': conf,
                    'pixel_height': bbox[3] - bbox[1]
                }
        
        return None
    
    def estimate_focal_length(self, pixel_height, real_distance, person_height=1.7):
        """
        Estimate camera focal length from known measurements
        
        Formula: focal_length = (pixel_height * distance) / real_height
        """
        focal_length = (pixel_height * real_distance) / person_height
        return focal_length
    
    def run_interactive_calibration(self):
        """Run interactive calibration session"""
        print("\n" + "=" * 70)
        print("INTERACTIVE DISTANCE CALIBRATION")
        print("=" * 70)
        print()
        print("INSTRUCTIONS:")
        print("  1. Position person at a known distance")
        print("  2. Measure actual distance with tape measure/laser")
        print("  3. Enter distance when prompted")
        print("  4. Press 'c' to capture calibration sample")
        print("  5. Repeat for different distances (2m, 5m, 10m, 15m, 20m, etc.)")
        print("  6. Press 'q' to finish and train model")
        print()
        print("TIP: Capture at least 10 different distances for best accuracy")
        print("=" * 70)
        print()
        
        # Open camera
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"❌ Cannot open camera {self.camera_index}")
            return
        
        print(f"📷 Camera {self.camera_index} opened")
        print()
        
        current_distance = None
        waiting_for_distance = True
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Failed to read frame")
                break
            
            # Detect person
            detection = self.detect_person(frame)
            
            # Create display
            display = frame.copy()
            
            if detection:
                # Draw bounding box
                bbox = detection['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Show pixel height
                pixel_height = detection['pixel_height']
                cv2.putText(display, f"Height: {pixel_height:.1f}px", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # If we have a distance entered, show estimated focal length
                if current_distance is not None:
                    estimated_fl = self.estimate_focal_length(pixel_height, current_distance)
                    cv2.putText(display, f"Distance: {current_distance:.1f}m", 
                               (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(display, f"Est. Focal Length: {estimated_fl:.1f}px", 
                               (x1, y2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(display, "Press 'c' to CAPTURE this sample", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(display, "Enter distance in terminal, then press 'c'", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            else:
                cv2.putText(display, "NO PERSON DETECTED - Position person in view", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Show sample count
            cv2.putText(display, f"Samples collected: {len(self.samples)}", 
                       (10, display.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show instructions
            cv2.putText(display, "Press 'q' to finish | 'c' to capture | 'd' to enter distance", 
                       (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Distance Calibration', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == ord('d'):
                # Enter new distance
                print("\n" + "-" * 50)
                try:
                    dist_input = input("Enter actual distance to person (in meters): ")
                    current_distance = float(dist_input)
                    print(f"✅ Distance set to: {current_distance:.2f}m")
                    print("   Position person at this distance, then press 'c' to capture")
                    print("-" * 50 + "\n")
                except ValueError:
                    print("❌ Invalid distance. Please enter a number.")
                    current_distance = None
            
            elif key == ord('c'):
                if detection and current_distance is not None:
                    # Capture sample
                    sample = {
                        'timestamp': datetime.now().isoformat(),
                        'distance_meters': current_distance,
                        'pixel_height': float(detection['pixel_height']),
                        'bbox': detection['bbox'].tolist(),
                        'confidence': detection['confidence'],
                        'image_shape': frame.shape[:2]
                    }
                    
                    # Calculate focal length estimate
                    focal_estimate = self.estimate_focal_length(
                        detection['pixel_height'], current_distance
                    )
                    sample['focal_length_estimate'] = focal_estimate
                    self.focal_length_estimates.append(focal_estimate)
                    
                    # Save image
                    sample_id = len(self.samples) + 1
                    img_filename = f"sample_{sample_id:03d}_dist_{current_distance:.1f}m.jpg"
                    img_path = self.save_dir / img_filename
                    cv2.imwrite(str(img_path), frame)
                    sample['image_file'] = img_filename
                    
                    # Add to samples
                    self.samples.append(sample)
                    
                    print(f"✅ Captured sample #{sample_id}:")
                    print(f"   Distance: {current_distance:.2f}m")
                    print(f"   Pixel height: {detection['pixel_height']:.1f}px")
                    print(f"   Focal length estimate: {focal_estimate:.1f}px")
                    print(f"   Saved: {img_filename}")
                    
                    # Reset for next capture
                    current_distance = None
                    print("\n   Ready for next sample. Press 'd' to enter new distance.")
                    
                elif detection is None:
                    print("❌ No person detected. Position person in view.")
                else:
                    print("❌ Distance not set. Press 'd' to enter distance first.")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save all calibration data
        if self.samples:
            self.save_calibration_data()
            self.analyze_calibration()
    
    def analyze_calibration(self):
        """Analyze collected calibration data"""
        if not self.samples:
            print("\n⚠️ No calibration samples collected")
            return
        
        print("\n" + "=" * 70)
        print("CALIBRATION ANALYSIS")
        print("=" * 70)
        
        # Calculate average focal length
        focal_lengths = [s['focal_length_estimate'] for s in self.samples]
        avg_focal = np.mean(focal_lengths)
        std_focal = np.std(focal_lengths)
        
        print(f"\nTotal samples: {len(self.samples)}")
        print(f"\nFocal Length Analysis:")
        print(f"  Average: {avg_focal:.2f} pixels")
        print(f"  Std Dev: {std_focal:.2f} pixels")
        print(f"  Min: {min(focal_lengths):.2f} pixels")
        print(f"  Max: {max(focal_lengths):.2f} pixels")
        
        # Distance distribution
        distances = [s['distance_meters'] for s in self.samples]
        print(f"\nDistance Range:")
        print(f"  Min: {min(distances):.1f}m")
        print(f"  Max: {max(distances):.1f}m")
        print(f"  Average: {np.mean(distances):.1f}m")
        
        # Save optimized focal length
        focal_config = {
            'focal_length': float(avg_focal),
            'focal_length_std': float(std_focal),
            'num_samples': len(self.samples),
            'distance_range': [float(min(distances)), float(max(distances))],
            'calibration_date': datetime.now().isoformat()
        }
        
        config_path = self.save_dir / 'focal_length_calibrated.json'
        with open(config_path, 'w') as f:
            json.dump(focal_config, f, indent=2)
        
        print(f"\n💾 Saved calibrated focal length: {config_path}")
        print(f"\n✅ Use focal length {avg_focal:.1f} for accurate distance estimation")
        
        # Test accuracy on samples
        print("\n" + "-" * 70)
        print("ACCURACY TEST ON CALIBRATION DATA")
        print("-" * 70)
        
        errors = []
        for sample in self.samples:
            # Predict distance using average focal length
            predicted_dist = (1.7 * avg_focal) / sample['pixel_height']
            actual_dist = sample['distance_meters']
            error = abs(predicted_dist - actual_dist)
            error_pct = (error / actual_dist) * 100
            errors.append(error)
            
            print(f"Distance {actual_dist:.1f}m: Predicted {predicted_dist:.2f}m, Error {error:.2f}m ({error_pct:.1f}%)")
        
        print("-" * 70)
        print(f"Average Error: {np.mean(errors):.2f}m")
        print(f"Max Error: {max(errors):.2f}m")
        print(f"RMSE: {np.sqrt(np.mean([e**2 for e in errors])):.2f}m")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Interactive Distance Calibration')
    parser.add_argument('--camera', type=int, default=4, help='Camera index')
    parser.add_argument('--save_dir', type=str, default='calibration_data/distance_samples',
                       help='Directory to save calibration data')
    
    args = parser.parse_args()
    
    system = DistanceCalibrationSystem(
        camera_index=args.camera,
        save_dir=args.save_dir
    )
    
    system.run_interactive_calibration()


if __name__ == "__main__":
    main()
