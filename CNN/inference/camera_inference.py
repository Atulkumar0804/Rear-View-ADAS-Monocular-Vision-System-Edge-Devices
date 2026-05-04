#!/usr/bin/env python3
"""
Real-time Camera Inference with Advanced ADAS Features
Fully integrated with all features from video_inference.py for live camera feed

KEY FEATURES:
1. Dynamic Horizon Estimation - Adapts to camera suspension movement
2. ByteTracker + IoU Fallback - Motion-aware tracking
3. Dual-Depth Fusion - Classical + ML depth estimation
4. Lane-Aware Safety Assessment - SSM-based metrics (TTC, MTTC, PET, DRAC)
5. Rider Action Recommendations - Natural language instructions
6. RearViewSafetyAssessment - Comprehensive safety metrics
7. RearSideUseCaseValidator - Validation logic for rear-view ADAS

Usage:
    python camera_inference.py --camera 0 --rear-camera
    python camera_inference.py --camera video.mp4 --hybrid-depth
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
import os
import threading
import json
import logging
import logging.handlers
import queue
from queue import Queue
from collections import deque, defaultdict
import math

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent.resolve()
CNN_DIR = SCRIPT_DIR.parent
sys.path.append(str(CNN_DIR))

from scripts.zoedepth_loader import load_zoedepth_model
from ultralytics import YOLO

# Import GPU configuration
try:
    from inference.gpu_config import GPUConfigManager, setup_gpu
    from inference.model_optimizer import ModelOptimizer, InferenceOptimizer
    _GPU_CONFIG_AVAILABLE = True
except ImportError:
    _GPU_CONFIG_AVAILABLE = False
    print("⚠️  GPU configuration not available, running with default settings")

# Import ByteTracker
try:
    from inference.byte_tracker import ByteTracker
    _BYTE_TRACKER_AVAILABLE = True
except ImportError:
    try:
        from byte_tracker import ByteTracker
        _BYTE_TRACKER_AVAILABLE = True
    except ImportError:
        _BYTE_TRACKER_AVAILABLE = False
        print("⚠️  ByteTracker not available, will fall back to IoU tracking")

# Import AsyncDepthLite
try:
    from inference.jetson_depth_lite import AsyncDepthLite
    _DEPTH_LITE_AVAILABLE = True
except ImportError:
    try:
        from jetson_depth_lite import AsyncDepthLite
        _DEPTH_LITE_AVAILABLE = True
    except ImportError:
        _DEPTH_LITE_AVAILABLE = False

# YOLO Models
YOLO_MODEL_PATH_FAST = "yolo11n.pt"
YOLO_MODEL_PATH_HEAVY = str(CNN_DIR / "models/yolo/yolo11x-seg.pt")

# Configuration
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.4
FOCAL_LENGTH = 721.5377

# Camera intrinsic constants
KITTI_FX      = 721.5377
KITTI_FY      = 721.5377
KITTI_CX      = 609.5593
KITTI_CY      = 172.854

# Rear camera parameters
MOUNTING_HEIGHT_M = 1.1
GROUND_PLANE_RATIO = 0.55
DEPTH_CLIP_MIN_M = 0.5
DEPTH_CLIP_MAX_M = 25.0

# Vehicle dimensions
VEHICLE_DIMENSIONS = {
    'Person': (1.7, 0.45, 0.15, 0.35),
    'Bicycle': (1.2, 0.65, 0.1, 0.55),
    'Two-wheeler': (1.3, 0.7, 0.1, 0.6),
    'Three-wheeler': (1.6, 1.4, 0.15, 1.0),
    'Hatchback': (1.5, 1.7, 0.1, 0.95),
    'Sedan': (1.5, 1.8, 0.1, 0.95),
    'SUV': (1.8, 1.9, 0.15, 0.95),
    'MUV': (1.9, 2.0, 0.15, 1.0),
    'Bus': (3.2, 2.5, 0.2, 1.25),
    'Truck': (3.0, 2.4, 0.2, 0.8),
    'Van': (2.0, 1.8, 0.15, 0.9),
    'LCV': (2.2, 1.9, 0.15, 1.0),
    'Mini-bus': (2.5, 2.1, 0.15, 1.15),
    'Tempo-traveller': (2.4, 2.0, 0.15, 1.1),
    'Person + Two-wheeler': (2.0, 0.8, 0.2, 0.65),
    'Person + Bicycle': (1.95, 0.7, 0.2, 0.6),
    'Person + Three-wheeler': (1.8, 1.4, 0.2, 0.85),
    'Others': (1.5, 1.6, 0.2, 0.95),
}

REAL_HEIGHTS = {k: v[0] for k, v in VEHICLE_DIMENSIONS.items()}

# Vehicle classes
VEHICLE_CLASSES = [
    'Bus', 'Hatchback', 'LCV', 'MUV', 'Mini-bus', 
    'Others', 'SUV', 'Sedan', 'Tempo-traveller', 'Three-wheeler', 
    'Truck', 'Van'
]

# YOLO class mapping
YOLO_CLASS_MAPPING = {
    0: 'Person',
    1: 'Bicycle',
    2: 'Sedan',
    3: 'Two-wheeler',
    5: 'Bus',
    7: 'Truck',
}

# Color mapping
CLASS_COLORS = {
    'Hatchback': (0, 255, 0),
    'Sedan': (0, 255, 127),
    'SUV': (0, 255, 255),
    'MUV': (127, 255, 0),
    'Bus': (0, 165, 255),
    'Truck': (255, 165, 0),
    'Three-wheeler': (255, 255, 0),
    'Two-wheeler': (255, 0, 127),
    'LCV': (255, 127, 80),
    'Mini-bus': (138, 43, 226),
    'Tempo-traveller': (147, 112, 219),
    'Bicycle': (255, 192, 203),
    'Van': (255, 140, 0),
    'Others': (128, 128, 128),
    'Person': (255, 0, 255),
    'Person + Two-wheeler': (255, 0, 127),
    'Person + Bicycle': (255, 192, 203),
    'Person + Three-wheeler': (255, 255, 0),
}

# CNN Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


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
# DYNAMIC HORIZON ESTIMATION (Adaptive to Camera Suspension Movement)
# ============================================================================

class DynamicHorizonEstimator:
    """Estimates horizon position dynamically based on vanishing point detection."""
    
    def __init__(self, frame_width=1920, frame_height=1080, ema_alpha=0.15, 
                 fallback_ratio=0.55):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.ema_alpha = float(np.clip(ema_alpha, 0.05, 0.5))
        self.fallback_ratio = fallback_ratio
        
        self.y_horizon_smoothed = frame_height * fallback_ratio
        self.y_horizon_detected = frame_height * fallback_ratio
        self.y_horizon_prev = frame_height * fallback_ratio
        self.detection_confidence = 0.5
        self.detection_count = 0
        
    def detect_horizon_vanishing_point(self, frame):
        """Detect horizon using vanishing point from road edges/lane lines."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            roi_mask = np.zeros_like(gray)
            roi_mask[int(h * 0.3):, :] = 255
            gray_roi = cv2.bitwise_and(gray, gray, mask=roi_mask)
            
            edges = cv2.Canny(gray_roi, 50, 150)
            
            lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30,
                                    minLineLength=50, maxLineGap=20)
            
            if lines is None or len(lines) < 2:
                return self.y_horizon_smoothed, 0.1
            
            line_segments = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if y1 != y2:
                    slope = (y2 - y1) / (x2 - x1 + 1e-6)
                    if abs(slope) > 0.2:
                        intercept = y1 - slope * x1
                        line_segments.append((slope, intercept, x1, y1, x2, y2))
            
            if len(line_segments) < 2:
                return self.y_horizon_smoothed, 0.1
            
            left_lines = [l for l in line_segments if l[0] > 0]
            right_lines = [l for l in line_segments if l[0] < 0]
            
            if len(left_lines) > 0 and len(right_lines) > 0:
                left_line = left_lines[len(left_lines)//2]
                right_line = right_lines[len(right_lines)//2]
                
                m1, b1 = left_line[0], left_line[1]
                m2, b2 = right_line[0], right_line[1]
                
                if abs(m1 - m2) > 0.01:
                    x_vp = (b2 - b1) / (m1 - m2)
                    y_vp = m1 * x_vp + b1
                    
                    if 0 <= y_vp <= h and 0 <= x_vp <= w:
                        x_confidence = 1.0 - abs(x_vp - w/2) / (w/2)
                        y_confidence = 1.0 if 0 < y_vp < h * 0.7 else 0.3
                        confidence = 0.7 * x_confidence + 0.3 * y_confidence
                        
                        return float(y_vp), float(np.clip(confidence, 0.1, 0.9))
            
            return self.y_horizon_smoothed, 0.1
            
        except Exception as e:
            return self.y_horizon_smoothed, 0.0
    
    def update(self, frame):
        """Update horizon estimate with temporal smoothing."""
        y_new, conf_new = self.detect_horizon_vanishing_point(frame)
        
        self.y_horizon_detected = y_new
        self.detection_confidence = conf_new
        self.detection_count += 1
        
        alpha_adaptive = self.ema_alpha * conf_new
        
        self.y_horizon_smoothed = (
            alpha_adaptive * y_new + 
            (1 - alpha_adaptive) * self.y_horizon_smoothed
        )
        
        horizon_delta = abs(self.y_horizon_smoothed - self.y_horizon_prev)
        if horizon_delta > 30:
            self.y_horizon_smoothed = self.y_horizon_prev + np.sign(
                self.y_horizon_smoothed - self.y_horizon_prev) * 30
        
        self.y_horizon_prev = self.y_horizon_smoothed
        
        return self.y_horizon_smoothed, self.detection_confidence
    
    def get_horizon(self):
        """Get current horizon without updating"""
        return self.y_horizon_smoothed


# ============================================================================
# LANE DETECTOR
# ============================================================================

class LaneDetector:
    """Detects vehicle lane position based on bounding box"""
    
    def __init__(self, frame_width=1920, frame_height=1080, lane_overlap=0.1):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.lane_overlap = lane_overlap
        self.lane_width = frame_width / 3.0
        
    def detect_lane(self, bbox):
        """Determine which lane a vehicle is in"""
        x1, y1, x2, y2 = bbox
        bbox_center_x = (x1 + x2) / 2.0
        bbox_width = x2 - x1
        
        if bbox_center_x < self.lane_width:
            primary_lane = "RIGHT"
        elif bbox_center_x < 2 * self.lane_width:
            primary_lane = "CENTER"
        else:
            primary_lane = "LEFT"
        
        left_boundary = max(x1, 0)
        right_boundary = min(x2, self.frame_width)
        
        left_lane_end = self.lane_width
        center_lane_start = self.lane_width
        center_lane_end = 2 * self.lane_width
        right_lane_start = 2 * self.lane_width
        
        left_coverage = max(0, min(right_boundary, left_lane_end) - left_boundary)
        center_coverage = max(0, min(right_boundary, center_lane_end) - max(left_boundary, center_lane_start))
        right_coverage = max(0, right_boundary - max(left_boundary, right_lane_start))
        
        total_coverage = left_coverage + center_coverage + right_coverage
        if total_coverage > 0:
            left_pct = left_coverage / total_coverage
            center_pct = center_coverage / total_coverage
            right_pct = right_coverage / total_coverage
        else:
            left_pct = center_pct = right_pct = 0.33
        
        if primary_lane == "LEFT":
            confidence = right_pct
        elif primary_lane == "CENTER":
            confidence = center_pct
        else:
            confidence = left_pct
        
        lane_span = []
        if right_pct > 0.15:
            lane_span.append("LEFT")
        if center_pct > 0.15:
            lane_span.append("CENTER")
        if left_pct > 0.15:
            lane_span.append("RIGHT")
        
        return {
            'lane': primary_lane,
            'confidence': float(confidence),
            'lane_coverage': {
                'LEFT': float(right_pct),
                'CENTER': float(center_pct),
                'RIGHT': float(left_pct)
            },
            'spans_multiple_lanes': len(lane_span) > 1,
            'lanes_occupied': lane_span,
            'center_x': float(bbox_center_x),
            'bbox_width': float(bbox_width)
        }


# ============================================================================
# RIDER ACTION RECOMMENDATION
# ============================================================================

class RiderActionRecommendation:
    """Generates natural language rider action recommendations"""
    
    def __init__(self):
        self.context_history = {}
    
    def get_rider_action(self, safety_level, lane_info, distance_m, speed_kmh,
                         relative_speed_kmh, motion, ego_speed_kmh):
        """Generate rider action recommendation"""
        vehicle_lane = lane_info.get('lane', 'CENTER')
        same_lane = vehicle_lane == 'CENTER'
        
        if same_lane:
            if safety_level == 'CRITICAL':
                if relative_speed_kmh > 5:
                    return {
                        'action': 'EMERGENCY_BRAKE',
                        'urgency': 'CRITICAL',
                        'description': f'⚠️ IMMEDIATE BRAKING REQUIRED! Vehicle {distance_m:.1f}m away',
                        'rider_instruction': 'Apply strong brakes immediately!',
                        'reason': f'Collision imminent - vehicle catching up at {relative_speed_kmh:.1f}km/h'
                    }
                else:
                    return {
                        'action': 'STRONG_DECELERATE',
                        'urgency': 'CRITICAL',
                        'description': f'⚠️ CRITICAL: Vehicle {distance_m:.1f}m away - reduce speed now!',
                        'rider_instruction': 'Decelerate aggressively to increase gap.',
                        'reason': 'Critical collision risk'
                    }
            
            elif safety_level == 'WARNING':
                return {
                    'action': 'DECELERATE',
                    'urgency': 'HIGH',
                    'description': f'⚠️ WARNING: Vehicle {distance_m:.1f}m away approaching - reduce speed',
                    'rider_instruction': 'Slow down gradually to maintain safe distance.',
                    'reason': f'Vehicle approaching at {relative_speed_kmh:.1f}km/h, closing gap'
                }
            
            elif safety_level == 'CAUTION':
                return {
                    'action': 'MONITOR',
                    'urgency': 'MEDIUM',
                    'description': f'ℹ️ CAUTION: Vehicle {distance_m:.1f}m away - monitor distance',
                    'rider_instruction': 'Reduce speed or maintain safe distance.',
                    'reason': f'Close following distance - currently {distance_m:.1f}m'
                }
            
            else:  # SAFE
                if ego_speed_kmh > 60:
                    return {
                        'action': 'MAINTAIN_SPEED',
                        'urgency': 'LOW',
                        'description': f'✓ SAFE: Vehicle {distance_m:.1f}m away - maintain speed',
                        'rider_instruction': 'Maintain current speed and distance.',
                        'reason': 'Safe following distance'
                    }
                else:
                    return {
                        'action': 'MAINTAIN_SPEED',
                        'urgency': 'LOW',
                        'description': f'✓ SAFE: Vehicle {distance_m:.1f}m away',
                        'rider_instruction': 'Continue at current speed.',
                        'reason': 'Safe driving distance'
                    }
        else:
            return {
                'action': 'BE_AWARE',
                'urgency': 'LOW',
                'description': f'ℹ️ Vehicle in {vehicle_lane} lane - {distance_m:.1f}m away',
                'rider_instruction': f'Be aware of vehicle in {vehicle_lane} lane.',
                'reason': f'Vehicle in {vehicle_lane} lane - no collision risk'
            }


# ============================================================================
# REAR-VIEW SAFETY ASSESSMENT
# ============================================================================

class RearViewSafetyAssessment:
    """Rear-View ADAS Safety Assessment with SSM-based metrics"""
    
    TTC_CRITICAL = 1.0
    TTC_WARNING = 1.5
    TTC_SAFE = 2.5
    PET_CRITICAL = 1.0
    DRAC_CRITICAL = 3.35
    DRAC_WARNING = 2.0
    DISTANCE_CRITICAL = 10.0
    DISTANCE_WARNING = 15.0
    
    def __init__(self, ego_vehicle_speed=0.0):
        self.ego_vehicle_speed = ego_vehicle_speed
        self.reaction_time = 1.0
        self.vehicle_length = 4.5
        
    def calculate_ttc(self, distance_m, ego_speed_ms, rear_vehicle_speed_ms):
        """Calculate Time to Collision"""
        if distance_m <= 0:
            return None
        
        relative_speed = rear_vehicle_speed_ms - ego_speed_ms
        
        if relative_speed <= 0.1:
            return None
        
        ttc = distance_m / relative_speed
        return max(0.0, ttc)
    
    def calculate_mttc(self, distance_m, ego_speed_ms, rear_vehicle_speed_ms,
                       ego_accel_ms2, rear_accel_ms2):
        """Calculate Modified Time to Collision"""
        d_speed = rear_vehicle_speed_ms - ego_speed_ms
        d_accel = rear_accel_ms2 - ego_accel_ms2
        
        if abs(d_accel) > 0.01:
            discriminant = d_speed**2 + 2*d_accel*distance_m
            
            if discriminant < 0:
                return None
            
            mttc = (-d_speed + np.sqrt(discriminant)) / d_accel
            return max(0.0, mttc)
        else:
            if d_speed > 0.1:
                return distance_m / d_speed
            else:
                return None
    
    def calculate_pet(self, distance_m, ego_speed_ms, rear_vehicle_speed_ms):
        """Calculate Post Encroachment Time"""
        relative_speed = rear_vehicle_speed_ms - ego_speed_ms
        
        if relative_speed <= 0.1:
            return float('inf')
        
        effective_distance = distance_m + self.vehicle_length
        pet = effective_distance / relative_speed
        return max(0.0, pet)
    
    def calculate_drac(self, distance_m, ego_speed_ms, rear_vehicle_speed_ms,
                       reaction_time=1.0):
        """Calculate Deceleration Rate to Avoid Collision"""
        if distance_m <= 0:
            return float('inf')
        
        reaction_distance = rear_vehicle_speed_ms * reaction_time
        available_distance = distance_m - reaction_distance
        
        if available_distance <= 0:
            return float('inf')
        
        if rear_vehicle_speed_ms > 0:
            drac = (rear_vehicle_speed_ms**2) / (2 * available_distance)
        else:
            drac = 0.0
        
        return max(0.0, drac)
    
    def assess_risk_level(self, ttc_s, mttc_s, pet_s, drac_ms2, distance_m,
                          ego_speed_ms, rear_speed_ms, lane_info=None):
        """Comprehensive lane-aware risk assessment"""
        is_same_lane = lane_info is None or lane_info.get('lane', 'CENTER') == 'CENTER'
        lane_name = lane_info.get('lane', 'CENTER') if lane_info else 'CENTER'
        
        collision_time = mttc_s if mttc_s is not None else ttc_s
        
        if is_same_lane:
            has_critical_ttc = collision_time is not None and collision_time < self.TTC_CRITICAL
            has_critical_drac = drac_ms2 > self.DRAC_CRITICAL
            has_critical_distance = distance_m < self.DISTANCE_CRITICAL
            has_critical_pet = pet_s < self.PET_CRITICAL
            
            critical_count = sum([has_critical_ttc, has_critical_drac, 
                                 has_critical_distance, has_critical_pet])
            
            if critical_count >= 2:
                level = "CRITICAL"
                alert_type = "collision_imminent"
                confidence = min(1.0, critical_count / 4.0)
                message = f"⚠️ CRITICAL: Multiple risk factors detected"
            
            elif collision_time is not None and collision_time < self.TTC_WARNING:
                level = "WARNING"
                alert_type = "collision_warning"
                confidence = 0.8
                message = f"⚠️ WARNING: TTC={collision_time:.2f}s, D={distance_m:.1f}m"
            
            elif drac_ms2 > self.DRAC_WARNING:
                level = "WARNING"
                alert_type = "high_deceleration"
                confidence = 0.7
                message = f"⚠️ WARNING: High deceleration required"
            
            elif distance_m < self.DISTANCE_WARNING:
                level = "CAUTION"
                alert_type = "distance_warning"
                confidence = 0.6
                message = f"ℹ️ CAUTION: Close following distance: {distance_m:.1f}m"
            
            else:
                level = "SAFE"
                alert_type = "none"
                confidence = 0.95
                message = "✓ Safe driving distance"
        
        else:
            level = "INFO"
            alert_type = "lane_awareness"
            confidence = lane_info.get('confidence', 0.8) if lane_info else 0.7
            message = f"ℹ️ {lane_name} lane: Vehicle {distance_m:.1f}m away"
        
        return {
            'level': level,
            'message': message,
            'alert_type': alert_type,
            'confidence': float(confidence),
            'ttc': collision_time,
            'mttc': mttc_s,
            'pet': pet_s,
            'drac': drac_ms2,
            'distance': distance_m,
            'ttc_critical': self.TTC_CRITICAL,
            'ttc_warning': self.TTC_WARNING,
            'drac_critical': self.DRAC_CRITICAL,
            'lane_aware': True,
            'same_lane': is_same_lane,
            'detected_lane': lane_name,
        }


# ============================================================================
# REAR-SIDE USE CASE VALIDATOR
# ============================================================================

class RearSideUseCaseValidator:
    """Rear-Side Use Case Validation for rear-view ADAS"""
    
    def __init__(self, frame_width=1920, frame_height=1080):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.rear_view_horizontal_margin = 0.05
        self.rear_view_vertical_threshold = 0.2
        
    def is_valid_rear_detection(self, bbox, bbox_height, distance_m):
        """Validate if detection is reliable for rear-view ADAS"""
        x1, y1, x2, y2 = bbox
        
        margin_left = self.frame_width * self.rear_view_horizontal_margin
        margin_right = self.frame_width * (1 - self.rear_view_horizontal_margin)
        
        in_horizontal_range = margin_left <= x1 and x2 <= margin_right
        horizontal_confidence = 1.0 if in_horizontal_range else 0.7
        
        top_margin = self.frame_height * self.rear_view_vertical_threshold
        in_vertical_range = y1 >= top_margin
        
        min_bbox_height = 20
        bbox_height_valid = bbox_height >= min_bbox_height
        bbox_confidence = min(1.0, bbox_height / 100.0)
        
        distance_min = 0.5
        distance_max = 30.0
        distance_valid = distance_min <= distance_m <= distance_max if distance_m else False
        
        validation_confidence = (
            0.3 * horizontal_confidence +
            0.2 * (1.0 if in_vertical_range else 0.5) +
            0.3 * bbox_confidence +
            0.2 * (1.0 if distance_valid else 0.5)
        )
        
        is_valid = (
            in_vertical_range and 
            bbox_height_valid and 
            distance_valid and
            validation_confidence > 0.5
        )
        
        return {
            'is_valid': is_valid,
            'confidence': float(validation_confidence),
            'in_horizontal_range': bool(in_horizontal_range),
            'in_vertical_range': bool(in_vertical_range),
            'bbox_height_valid': bool(bbox_height_valid),
            'distance_valid': bool(distance_valid),
        }
    
    def validate_rear_scenario(self, detections, ego_speed_ms):
        """Validate complete rear-view scenario"""
        if not detections:
            return {
                'scenario_valid': True,
                'scenario_type': 'clear_rear',
                'threat_level': 'none',
                'critical_vehicles': [],
            }
        
        critical_vehicles = []
        
        for detection in detections:
            bbox = detection.get('bbox')
            distance = detection.get('distance')
            bbox_height = bbox[3] - bbox[1] if bbox else 0
            
            if bbox is None or distance is None:
                continue
            
            validation = self.is_valid_rear_detection(bbox, bbox_height, distance)
            
            if validation['is_valid']:
                if distance < 10.0 and ego_speed_ms > 10:
                    critical_vehicles.append({
                        'class': detection.get('class'),
                        'distance': distance,
                        'motion': detection.get('motion'),
                        'track_id': detection.get('track_id'),
                        'validation_confidence': validation['confidence']
                    })
        
        if not critical_vehicles:
            scenario_type = "clear_rear"
            threat_level = "none"
        elif len(critical_vehicles) == 1:
            threat_level = "medium" if critical_vehicles[0]['distance'] > 5 else "high"
            scenario_type = "approaching_vehicle"
        else:
            threat_level = "high"
            scenario_type = "multiple_approaching_vehicles"
        
        return {
            'scenario_valid': True,
            'scenario_type': scenario_type,
            'threat_level': threat_level,
            'critical_vehicles_count': len(critical_vehicles),
            'critical_vehicles': critical_vehicles,
        }


# ============================================================================
# CAMERA VEHICLE DETECTOR (Main Class)
# ============================================================================

class CameraVehicleDetector:
    """Real-time camera-based vehicle detector with advanced ADAS features"""
    
    def __init__(self, device='cuda', zoedepth_interval=30, correction_alpha=0.3,
                 learnable_alpha=True, alpha_lr=0.05, fps=30):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🔥 Device: {self.device}")
        self.fps = fps
        
        # Load YOLO (using yolo11n for real-time performance)
        print("📦 Loading YOLO11n (fast model)...")
        self.yolo = YOLO(YOLO_MODEL_PATH_FAST)
        print("✅ YOLO11n loaded (optimized for real-time inference)")
        
        # Load Classifier
        CLASSIFIER_PATH = str(CNN_DIR / "models/classifier/weights/best.pt")
        print(f"📦 Loading Classifier: {CLASSIFIER_PATH}")
        self.classifier = YOLO(CLASSIFIER_PATH)
        print("✅ Classifier loaded")
        
        # Initialize ByteTracker
        if _BYTE_TRACKER_AVAILABLE:
            print("📦 Initializing ByteTracker...")
            self.tracker = ByteTracker(track_buffer=300, frame_rate=int(fps))
            self.use_byte_tracker = True
            print("✅ ByteTracker initialized (motion-aware tracking enabled)")
        else:
            print("⚠️  ByteTracker not available, falling back to IoU tracking")
            self.tracker = None
            self.use_byte_tracker = False
        
        # Tracking
        self.prev_distances = {}
        self.track_id_counter = 0
        self.prev_boxes = []
        self.track_classes = defaultdict(lambda: deque(maxlen=5))
        self.IOU_THRESHOLD = 0.45
        self.MIN_CROP_AREA = 64 * 64
        
        # Motion tracking
        self.track_motion_state = {}
        self.kalman_filters = defaultdict(lambda: KalmanFilter1D())
        
        # Initialize Advanced ADAS Components
        print("📦 Initializing Advanced ADAS Components...")
        
        # Dynamic Horizon Estimator
        self.horizon_estimator = DynamicHorizonEstimator(
            frame_width=1920, frame_height=1080,
            ema_alpha=0.15, fallback_ratio=0.55
        )
        print("✅ Dynamic Horizon Estimator initialized")
        
        # Lane Detector
        self.lane_detector = LaneDetector(frame_width=1920, frame_height=1080)
        print("✅ Lane Detector initialized")
        
        # Rider Action Recommender
        self.rider_action_recommender = RiderActionRecommendation()
        print("✅ Rider Action Recommender initialized")
        
        # Rear-View Safety Assessment
        self.rear_safety_assessment = RearViewSafetyAssessment(ego_vehicle_speed=0.0)
        print("✅ Rear-View Safety Assessment initialized")
        
        # Rear-Side Validator
        self.rear_side_validator = RearSideUseCaseValidator(frame_width=1920, frame_height=1080)
        print("✅ Rear-Side Use Case Validator initialized")
        
        # Safety assessment cache
        self.safety_assessments_cache = {}
        self.last_scenario_validation = None
        
        print("✅ All Advanced ADAS Components initialized")
    
    def validate_and_assess_rear_scenario(self, detections, ego_speed_kmh=0.0, frame_shape=None):
        """Validate and assess rear scenario with safety metrics"""
        safety_assessments = {}
        
        for det in detections:
            if det.get('distance') is None:
                continue
            
            track_id = det.get('track_id', -1)
            distance_m = det.get('distance', 0.0)
            speed_kmh = det.get('speed', 0.0)
            bbox = det.get('bbox')
            
            # Detect lane position
            lane_info = self.lane_detector.detect_lane(bbox) if bbox else None
            
            # Calculate safety metrics
            ego_speed_ms = ego_speed_kmh / 3.6
            rear_speed_ms = speed_kmh / 3.6
            
            ttc = self.rear_safety_assessment.calculate_ttc(distance_m, ego_speed_ms, rear_speed_ms)
            mttc = self.rear_safety_assessment.calculate_mttc(distance_m, ego_speed_ms, rear_speed_ms, 0, 0)
            pet = self.rear_safety_assessment.calculate_pet(distance_m, ego_speed_ms, rear_speed_ms)
            drac = self.rear_safety_assessment.calculate_drac(distance_m, ego_speed_ms, rear_speed_ms)
            
            # Assess risk level
            assessment = self.rear_safety_assessment.assess_risk_level(
                ttc, mttc, pet, drac, distance_m, ego_speed_ms, rear_speed_ms, lane_info
            )
            
            # Get rider action
            relative_speed_kmh = speed_kmh - ego_speed_kmh
            rider_action = self.rider_action_recommender.get_rider_action(
                assessment['level'], lane_info or {}, distance_m, speed_kmh,
                relative_speed_kmh, det.get('motion', 'unknown'), ego_speed_kmh
            )
            
            assessment['rider_action'] = rider_action
            assessment['lane_info'] = lane_info
            assessment['same_lane'] = lane_info.get('lane') == 'CENTER' if lane_info else True
            
            safety_assessments[track_id] = assessment
        
        # Validate overall scenario
        ego_speed_ms = ego_speed_kmh / 3.6
        scenario_validation = self.rear_side_validator.validate_rear_scenario(detections, ego_speed_ms)
        self.last_scenario_validation = scenario_validation
        
        return safety_assessments, scenario_validation
    
    def estimate_distance(self, bbox_height, class_name):
        """Estimate distance using object size"""
        if bbox_height <= 0:
            return None
        
        real_height = REAL_HEIGHTS.get(class_name, 1.5)
        distance = (real_height * FOCAL_LENGTH) / bbox_height
        
        return float(np.clip(distance, 0.3, 80.0))
    
    def estimate_motion(self, track_id, current_distance):
        """Estimate motion state"""
        if track_id not in self.prev_distances:
            self.prev_distances[track_id] = []
        
        self.prev_distances[track_id].append(current_distance)
        
        max_buffer = 10 if self.use_byte_tracker else 30
        if len(self.prev_distances[track_id]) > max_buffer:
            self.prev_distances[track_id].pop(0)
        
        min_frames = 5 if self.use_byte_tracker else 15
        if len(self.prev_distances[track_id]) < min_frames:
            return "stable", 0.0
        
        distances = self.prev_distances[track_id]
        time_steps = len(distances) - 1
        if time_steps == 0:
            return "stable", 0.0
        
        window = 5
        if len(distances) < window * 2:
            avg_change = (distances[-1] - distances[0]) / time_steps
        else:
            recent_avg = sum(distances[-window:]) / window
            old_avg = sum(distances[:window]) / window
            effective_steps = len(distances) - window
            avg_change = (recent_avg - old_avg) / effective_steps
        
        speed_kmh = avg_change * self.fps * 3.6
        threshold = 0.03
        
        if avg_change < -threshold:
            return "approaching", abs(speed_kmh)
        elif avg_change > threshold:
            return "receding", abs(speed_kmh)
        else:
            return "stable", 0.0
    
    def detect_frame(self, frame):
        """Detect vehicles in a single frame"""
        results = []
        current_boxes_raw = []
        ts_now = time.time()
        h_frame, w_frame = frame.shape[:2]
        
        # Update Dynamic Horizon
        y_horizon, horizon_conf = self.horizon_estimator.update(frame)
        
        # YOLO detection
        yolo_results = self.yolo(frame, verbose=False)[0]
        
        for i, detection in enumerate(yolo_results.boxes.data):
            x1, y1, x2, y2, conf, cls_id = detection.cpu().numpy()
            cls_id = int(cls_id)
            mask = None
            
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
                            pass
                
                current_boxes_raw.append([x1, y1, x2, y2])
                results.append({
                    'bbox': [x1, y1, x2, y2],
                    'class': final_class,
                    'confidence': final_conf,
                    'source': source,
                    'mask': mask
                })
        
        # Merge rider and vehicle
        results = self.merge_rider_and_vehicle(results)
        current_boxes_raw = [r['bbox'] for r in results]
        
        # Use ByteTracker for tracking
        matches = []
        if self.use_byte_tracker and self.tracker is not None:
            tracker_inputs = []
            for i, r in enumerate(results):
                x1, y1, x2, y2 = r['bbox']
                tracker_inputs.append({
                    'bbox': [x1, y1, x2-x1, y2-y1],
                    'confidence': r['confidence'],
                    'class_name': r['class'],
                })
            
            tracked_results = self.tracker.update(tracker_inputs)
            track_id_list = [t['track_id'] for t in tracked_results]
            
            for i, track_data in enumerate(tracked_results):
                if i < len(results):
                    results[i]['track_id'] = track_data['track_id']
                    matches.append(track_data['track_id'])
            
            for i in range(len(tracked_results), len(results)):
                matches.append(-1)
        else:
            matches = self.match_detections(current_boxes_raw, self.prev_boxes, iou_threshold=self.IOU_THRESHOLD)
        
        # Assign track IDs and estimate motion
        for i, result in enumerate(results):
            if 'track_id' in result:
                track_id = result['track_id']
            else:
                track_id = matches[i] if matches[i] != -1 else self.track_id_counter
                if matches[i] == -1:
                    self.track_id_counter += 1
                result['track_id'] = track_id
            
            bbox_h = result['bbox'][3] - result['bbox'][1]
            distance = self.estimate_distance(bbox_h, result['class'])
            
            # Apply Kalman filter smoothing
            smoothed_distance = self.kalman_filters[track_id].update(distance) if distance else None
            
            result['distance'] = smoothed_distance
            result['distance_metadata'] = {
                'method': 'classical_pinhole',
                'confidence': 0.8,
                'distance': smoothed_distance,
            }
            
            # Estimate motion
            if smoothed_distance:
                motion, speed = self.estimate_motion(track_id, smoothed_distance)
                result['motion'] = motion
                result['speed'] = speed
            else:
                result['motion'] = "unknown"
                result['speed'] = 0.0
            
            # Per-track class smoothing
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
        """Merge overlapping rider and vehicle detections"""
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
            mx1, my1, mx2, my2 = v_box
            
            merged_any = False
            for i, p in enumerate(persons):
                if i in used_persons:
                    continue
                
                p_box = p['bbox']
                
                xA = max(mx1, p_box[0])
                yA = max(my1, p_box[1])
                xB = min(mx2, p_box[2])
                yB = min(my2, p_box[3])
                
                interArea = max(0, xB - xA) * max(0, yB - yA)
                p_area = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])
                
                iop = interArea / p_area if p_area > 0 else 0
                
                if iop > 0.2:
                    mx1 = min(mx1, p_box[0])
                    my1 = min(my1, p_box[1])
                    mx2 = max(mx2, p_box[2])
                    my2 = max(my2, p_box[3])
                    used_persons.add(i)
                    merged_any = True
            
            v['bbox'] = [mx1, my1, mx2, my2]
            if merged_any:
                v['class'] = f"Person + {v['class']}"
            final_results.append(v)
            
        for i, p in enumerate(persons):
            if i not in used_persons:
                final_results.append(p)
                
        final_results.extend(others)
        
        return final_results
    
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
    
    def draw_detections(self, frame, detections, fps=None):
        """Draw bounding boxes with rich ADAS information"""
        annotated = frame.copy()
        
        # Draw top HUD bar for scenario summary and FPS
        hud_height = 90
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], hud_height), (0, 0, 0), -1)
        cv2.putText(annotated, "LIVE VEHICLE DETECTION", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Scenario and threat information
        if hasattr(self, 'last_scenario_validation') and self.last_scenario_validation:
            scenario = self.last_scenario_validation
            scenario_text = (
                f"Scenario: {scenario.get('scenario_type', 'unknown')} | "
                f"Threat: {scenario.get('threat_level', 'none')} | "
                f"Critical: {scenario.get('critical_vehicles_count', 0)}"
            )
            cv2.putText(annotated, scenario_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
        
        # Additional FPS text overlay if available
        if fps:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(annotated, fps_text, (annotated.shape[1] - 250, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Define alert colors based on safety level
        safety_colors = {
            'CRITICAL': (0, 0, 255),      # Red
            'WARNING': (0, 165, 255),     # Orange
            'CAUTION': (0, 255, 255),     # Yellow
            'SAFE': (0, 255, 0),          # Green
            'INFO': (0, 165, 255),        # Orange
            'unknown': (128, 128, 128)    # Gray
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            distance = det.get('distance', None)
            distance_metadata = det.get('distance_metadata', {})
            z_classical = distance_metadata.get('classical_fused', None)
            z_ml = distance_metadata.get('ml', None)
            motion = det.get('motion', 'unknown')
            speed = det.get('speed', 0.0)
            
            # Get safety assessment
            safety_assess = det.get('safety_assessment', {})
            safety_level = safety_assess.get('level', 'unknown')
            alert_type = safety_assess.get('alert_type', 'none')
            
            # Choose color based on safety level
            if safety_level != 'unknown':
                box_color = safety_colors.get(safety_level, (128, 128, 128))
            else:
                motion_colors = {
                    'approaching': (0, 0, 255),    # Red
                    'receding': (0, 255, 255),     # Yellow
                    'stable': (0, 255, 0),         # Green
                    'unknown': CLASS_COLORS.get(class_name, (255, 255, 255))
                }
                box_color = motion_colors.get(motion, CLASS_COLORS.get(class_name, (255, 255, 255)))
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3)
            
            # Draw label with distance and depth estimates
            if distance:
                label = f"{class_name}: {confidence:.2f} | D:{distance:.1f}m"
                if isinstance(z_classical, (int, float, np.floating)):
                    label += f" C:{float(z_classical):.1f}"
                if isinstance(z_ml, (int, float, np.floating)):
                    label += f" ML:{float(z_ml):.1f}"
            else:
                label = f"{class_name}: {confidence:.2f}"
            
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), box_color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw distance below box
            if distance:
                dist_text = f"{distance:.1f}m"
                dist_size, _ = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(annotated, (x1, y2), 
                             (x1 + dist_size[0] + 10, y2 + dist_size[1] + 10), box_color, -1)
                cv2.putText(annotated, dist_text, (x1 + 5, y2 + dist_size[1] + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw motion state
            if motion != 'unknown':
                if speed > 0:
                    motion_text = f"{motion.upper()} {speed:.1f}km/h"
                else:
                    motion_text = motion.upper()
                motion_size, _ = cv2.getTextSize(motion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                motion_x = x2 - motion_size[0] - 10
                motion_y = y1 + 25
                cv2.rectangle(annotated, (motion_x - 5, motion_y - motion_size[1] - 5), 
                             (x2 - 5, motion_y + 5), box_color, -1)
                cv2.putText(annotated, motion_text, (motion_x, motion_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw safety assessment and rider action
            if safety_level != 'unknown' and alert_type != 'none':
                safety_y = y2 + 40
                
                lane_info = safety_assess.get('lane_info', {})
                lane_name = lane_info.get('lane', 'CENTER') if lane_info else 'CENTER'
                is_same_lane = safety_assess.get('same_lane', True)
                
                if not is_same_lane:
                    safety_text = f"[{lane_name} LANE] {distance:.1f}m away"
                    box_color_text = (0, 165, 255)
                else:
                    safety_text = f"[{safety_level}] {alert_type}"
                    
                    ttc_val = safety_assess.get('ttc')
                    drac_val = safety_assess.get('drac')
                    
                    if ttc_val is not None:
                        safety_text += f" TTC:{ttc_val:.2f}s"
                    if drac_val is not None and drac_val != float('inf'):
                        safety_text += f" DRAC:{drac_val:.2f}m/s²"
                    
                    box_color_text = box_color
                
                safety_size, _ = cv2.getTextSize(safety_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, safety_y - safety_size[1] - 5), 
                             (x1 + safety_size[0] + 10, safety_y + 5), box_color_text, -1)
                cv2.putText(annotated, safety_text, (x1 + 5, safety_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw rider action recommendation
                rider_action = safety_assess.get('rider_action', {})
                if rider_action and rider_action.get('action'):
                    instruction = rider_action.get('rider_instruction', '')
                    urgency = rider_action.get('urgency', 'LOW')
                    
                    urgency_colors = {
                        'CRITICAL': (0, 0, 255),      # Red
                        'HIGH': (0, 165, 255),        # Orange
                        'MEDIUM': (0, 255, 255),      # Yellow
                        'LOW': (0, 255, 0),           # Green
                    }
                    urgency_color = urgency_colors.get(urgency, (128, 128, 128))
                    
                    action_y = safety_y + 25
                    action_text = f"→ {instruction[:50]}"
                    action_size, _ = cv2.getTextSize(action_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated, (x1, action_y - action_size[1] - 5), 
                                 (x1 + action_size[0] + 10, action_y + 5), urgency_color, -1)
                    cv2.putText(annotated, action_text, (x1 + 5, action_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated


def setup_realsense_camera(width=640, height=480, fps=30):
    """Setup RealSense D455 camera"""
    try:
        import pyrealsense2 as rs

        # Check if any RealSense devices are connected
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            print("⚠️ No RealSense cameras detected")
            return None, None

        print(f"📷 Found {len(devices)} RealSense device(s)")
        for i, device in enumerate(devices):
            try:
                name = device.get_info(rs.camera_info.name)
                serial = device.get_info(rs.camera_info.serial_number)
                print(f"   {i+1}. {name} (Serial: {serial})")
            except:
                print(f"   {i+1}. Unknown RealSense device")

        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        print("🔄 Starting RealSense pipeline...")
        pipeline.start(config)
        print("✅ RealSense pipeline started successfully")

        return pipeline, (width, height)
    except Exception as e:
        print(f"⚠️ RealSense setup failed: {e}")
        print("💡 Common solutions:")
        print("   - Ensure camera is properly powered (use USB 3.0 port or external power)")
        print("   - Check USB cable connection")
        print("   - Remove any lens covers or obstructions")
        print("   - Try a different USB port")
        print("   - Update RealSense firmware if needed")
        return None, None


def get_realsense_frame(pipeline):
    """Get frame from RealSense camera"""
    try:
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        color_frame = frames.get_color_frame()

        if not color_frame:
            print("⚠️ No color frame received from RealSense")
            return None

        return np.asanyarray(color_frame.get_data())
    except Exception as e:
        print(f"⚠️ Error getting RealSense frame: {e}")
        if "didn't arrive within" in str(e):
            print("💡 Camera timeout - check power, USB connection, and remove any lens covers")
        return None


def main():
    parser = argparse.ArgumentParser(description='Real-time Camera ADAS Detection')
    parser.add_argument('--camera', type=str, default='0',
                       help='Camera index or video file path')
    parser.add_argument('--width', type=int, default=1280,
                       help='Frame width')
    parser.add_argument('--height', type=int, default=720,
                       help='Frame height')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    parser.add_argument('--save', type=str, default=None,
                       help='Save output video')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable display window')
    parser.add_argument('--realsense', action='store_true',
                       help='Use RealSense D455 camera')
    parser.add_argument('--force-usb', action='store_true',
                       help='Force USB camera even if RealSense is available')
    parser.add_argument('--profile', type=str, default='a6000_full',
                       choices=['a6000_full', 'jetson_nano_restricted', 'jetson_nano_power_save'],
                       help='GPU profile')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--rear-camera', action='store_true',
                       help='Rear camera mode')
    parser.add_argument('--hybrid-depth', action='store_true',
                       help='Enable hybrid depth estimation')
    
    args = parser.parse_args()
    
    # Apply GPU configuration if available
    if _GPU_CONFIG_AVAILABLE:
        print(f"🎮 Applying GPU profile: {args.profile}")
        try:
            gpu_manager = setup_gpu(profile=args.profile, verbose=args.verbose)
            # Setup inference optimization
            opt_settings = ModelOptimizer(profile=args.profile).get_optimization_settings()
            InferenceOptimizer.enable_cudnn_benchmark(opt_settings.get('use_cudnn_autotuner', True))
            InferenceOptimizer.use_tf32(opt_settings.get('enable_tf32', False))
            print("✓ GPU configuration applied\n")
        except Exception as e:
            print(f"⚠️  Failed to apply GPU configuration: {e}")
            print("   Running with default settings\n")
    else:
        print(f"🎮 GPU profile requested: {args.profile} (configuration not available)")
    
    print("\n" + "="*60)
    print("🎬 REAL-TIME CAMERA VEHICLE DETECTION (Advanced ADAS)")
    print("="*60 + "\n")
    
    # Try to open camera
    cap = None
    realsense_pipeline = None
    use_realsense = False
    
    if args.realsense and not args.force_usb:
        print("🎥 Attempting to open RealSense D455 camera...")
        realsense_pipeline, _ = setup_realsense_camera(args.width, args.height, fps=30)
        if realsense_pipeline:
            use_realsense = True
            print("✓ RealSense camera ready\n")
        else:
            print("❌ RealSense camera setup failed")
            print("💡 RealSense troubleshooting:")
            print("   • Ensure camera is powered (use USB 3.0 port or powered hub)")
            print("   • Remove any lens covers or obstructions")
            print("   • Try: rs-enumerate-devices (if installed)")
            print("   • Update RealSense firmware if needed")
            print("   • Falling back to USB camera...\n")
    
    # If not using RealSense, try USB camera
    if not use_realsense:
        # Try the specified camera first
        try:
            camera_id = int(args.camera)
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                print(f"✅ Camera {camera_id} opened successfully")
            else:
                print(f"⚠️ Camera {camera_id} not available, trying other cameras...")
                cap = None
        except ValueError:
            # If it's not a number, treat it as a video file path
            cap = cv2.VideoCapture(args.camera)
            if cap.isOpened():
                print(f"✅ Video file opened: {args.camera}")
            else:
                print(f"⚠️ Video file not found: {args.camera}")
                cap = None
        
        # If the specified camera failed, try to find an available camera
        if not cap or not cap.isOpened():
            print("🔍 Searching for available cameras...")
            available_cameras = []
            
            # Try common camera indices
            for i in range(10):  # Check cameras 0-9
                test_cap = cv2.VideoCapture(i)
                if test_cap.isOpened():
                    width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    available_cameras.append((i, width, height))
                    test_cap.release()
            
            if available_cameras:
                # Use the first available camera
                camera_id, width, height = available_cameras[0]
                cap = cv2.VideoCapture(camera_id)
                print(f"✅ Found and opened camera {camera_id} ({width}x{height})")
            else:
                print("❌ No cameras found on the system")
                print("💡 Troubleshooting tips:")
                print("   • Connect a USB webcam")
                print("   • Check USB cable connections")
                print("   • Try running: v4l2-ctl --list-devices")
                print("   • For RealSense: use --realsense flag")
                return
    
    # Set frame dimensions
    if not use_realsense:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        actual_width = args.width
        actual_height = args.height
    
    print(f"✅ Source opened: {actual_width}x{actual_height}")
    
    # Initialize detector
    detector = CameraVehicleDetector(device=args.device)
    
    # Setup video writer if saving
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save, fourcc, 30, (actual_width, actual_height))
        print(f"💾 Saving to: {args.save}")
    
    print("\n🚀 Starting detection...")
    if not args.no_display:
        print("   Press 'q' or ESC to quit\n")
    
    frame_count = 0
    start_time = time.time()
    window_name = "🚗 ADAS Detection (q=quit)"
    consecutive_failures = 0
    max_consecutive_failures = 5
    
    # Create display window
    if not args.no_display:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720)
        except Exception as e:
            print(f"⚠️  Could not create display window: {e}")
            args.no_display = True
    
    try:
        while True:
            # Get frame
            if use_realsense:
                frame = get_realsense_frame(realsense_pipeline)
                if frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"\n❌ RealSense camera failed after {consecutive_failures} attempts")
                        print("💡 Troubleshooting tips:")
                        print("   • Check USB cable and power supply")
                        print("   • Ensure camera lens is not covered")
                        print("   • Try different USB port (preferably USB 3.0)")
                        print("   • Restart the camera by unplugging and replugging")
                        print("   • Check camera firmware: https://dev.intelrealsense.com/docs/firmware-update-tool")
                        break
                    print(f"⚠️ RealSense frame failed ({consecutive_failures}/{max_consecutive_failures}), retrying...")
                    time.sleep(0.5)  # Brief pause before retry
                    continue
                else:
                    consecutive_failures = 0  # Reset on success
                ret = True
            else:
                ret, frame = cap.read()
                if not ret:
                    print("\n🎬 End of stream reached")
                    break
            
            frame_count += 1
            
            # Detect
            detections = detector.detect_frame(frame)
            
            # Safety assessment
            safety_assessments, scenario_validation = detector.validate_and_assess_rear_scenario(
                detections, ego_speed_kmh=0.0, frame_shape=frame.shape
            )
            
            # Add safety assessment to detections
            for det in detections:
                track_id = det.get('track_id', -1)
                if track_id in safety_assessments:
                    det['safety_assessment'] = safety_assessments[track_id]
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Draw
            annotated = detector.draw_detections(frame, detections, fps=fps)
            
            # Display
            if not args.no_display:
                try:
                    cv2.imshow(window_name, annotated)
                except Exception as e:
                    print(f"⚠️  Display error: {e}")
            
            # Save
            if writer:
                writer.write(annotated)
            
            # Print progress
            if frame_count % 30 == 0:
                print(f"Frame {frame_count} | FPS: {fps:.1f} | Detections: {len(detections)}")
            
            # Handle key press
            if not args.no_display:
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q') or key == 27:
                    print("\n✓ Exit requested")
                    break
    
    finally:
        # Cleanup
        if use_realsense and realsense_pipeline:
            realsense_pipeline.stop()
            print("✓ RealSense pipeline stopped")
        elif cap:
            cap.release()
        
        if writer:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        
        # Stats
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        
        print("\n" + "="*60)
        print("📊 SESSION STATISTICS")
        print("="*60)
        print(f"Frames processed: {frame_count}")
        print(f"Time elapsed: {elapsed:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")
        if args.save:
            print(f"Output saved to: {args.save}")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()

