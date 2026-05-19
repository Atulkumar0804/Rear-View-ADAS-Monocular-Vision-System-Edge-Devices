#!/usr/bin/env python3
"""
Video Inference Script - Process video file and save output
No GUI, just processes video once and saves result

Usage:
    python video_inference.py --input video.mp4 --output result.mp4

KEY FEATURES:
1. Dynamic Horizon Estimation (NEW)
   - Adapts to camera suspension movement (pitch angle changes)
   - Detects vanishing point from lane lines
   - Uses EMA temporal smoothing (15% new, 85% previous)
   - Falls back to fixed horizon (55%) when lanes not visible
   - Fixes instability in ground plane distance estimation
   
   Why this matters:
   - Without dynamic horizon: ±10px error → noticeable distance errors
   - With dynamic horizon: Adapts per-frame to suspension movement
   - Critical for ADAS safety: accurate distance = accurate collision warnings
   
2. ByteTracker + IoU Fallback (Motion-aware tracking)
   - Handles occlusion and temporary track loss
   - Track buffer = 300 frames (~10 seconds at 30 FPS)
   
3. Dual-Depth Fusion (Classical + ML)
   - Ground plane projection (adaptive horizon)
   - Object size-based estimation
   - Motion parallax
   - Learnable correction factor with adaptive EMA alpha

4. Lane-Aware Safety Assessment
   - Same lane: strict collision thresholds
   - Adjacent lanes: informational only
   - TTC, MTTC, PET, DRAC safety metrics
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
import csv
from queue import Queue
from collections import deque, defaultdict

from ultralytics import YOLO

# Get absolute path to YOLO model
SCRIPT_DIR = Path(__file__).parent.resolve()
CNN_DIR = SCRIPT_DIR.parent
sys.path.append(str(CNN_DIR))

# Import ByteTracker (improved tracking over IoU)
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

try:
    from inference.jetson_depth_lite import AsyncDepthLite
    _DEPTH_LITE_AVAILABLE = True
except ImportError:
    try:
        from jetson_depth_lite import AsyncDepthLite
        _DEPTH_LITE_AVAILABLE = True
    except ImportError:
        _DEPTH_LITE_AVAILABLE = False

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
# DYNAMIC HORIZON ESTIMATION (Adaptive to Camera Suspension Movement)
# ============================================================================

class DynamicHorizonEstimator:
    """
    Estimates horizon position dynamically based on vanishing point detection.
    Adapts to camera suspension movement (pitch angle changes) without IMU.
    
    Methods:
    1. Lane-based vanishing point (primary)
    2. Edge-based vanishing point (fallback)
    3. EMA temporal smoothing
    4. Fixed horizon fallback when detection fails
    """
    
    def __init__(self, frame_width=1920, frame_height=1080, ema_alpha=0.15, 
                 fallback_ratio=0.55):
        """
        Initialize horizon estimator
        
        Args:
            frame_width: Video frame width
            frame_height: Video frame height  
            ema_alpha: EMA smoothing factor [0, 1] (higher = faster adaptation)
            fallback_ratio: Fixed horizon ratio when detection fails (default 0.55)
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.ema_alpha = float(np.clip(ema_alpha, 0.05, 0.5))
        self.fallback_ratio = fallback_ratio
        
        # State
        self.y_horizon_smoothed = frame_height * fallback_ratio  # Initial value
        self.y_horizon_detected = frame_height * fallback_ratio
        self.y_horizon_prev = frame_height * fallback_ratio
        self.detection_confidence = 0.5
        self.detection_count = 0
        
    def detect_horizon_vanishing_point(self, frame):
        """
        Detect horizon using vanishing point from road edges/lane lines.
        
        Approach:
        1. Edge detection (Canny)
        2. Line detection (Hough)
        3. Find lane lines in lower half of frame
        4. Extend lines → vanishing point
        5. Horizon = horizontal line through VP
        
        Args:
            frame: BGR image
            
        Returns:
            y_horizon: Estimated horizon pixel y-coordinate
            confidence: Confidence [0, 1]
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Focus on lower 70% of frame (road region for rear view)
            roi_mask = np.zeros_like(gray)
            roi_mask[int(h * 0.3):, :] = 255
            gray_roi = cv2.bitwise_and(gray, gray, mask=roi_mask)
            
            # Edge detection
            edges = cv2.Canny(gray_roi, 50, 150)
            
            # Hough line detection
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=30,
                minLineLength=50,
                maxLineGap=20
            )
            
            if lines is None or len(lines) < 2:
                return self.y_horizon_smoothed, 0.1  # Low confidence, use previous
            
            # Extract line endpoints
            line_segments = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if y1 != y2:  # Avoid horizontal lines
                    # Calculate slope and intercept
                    slope = (y2 - y1) / (x2 - x1 + 1e-6)
                    # Only consider lines with reasonable slope (not too horizontal)
                    if abs(slope) > 0.2:
                        intercept = y1 - slope * x1
                        line_segments.append((slope, intercept, x1, y1, x2, y2))
            
            if len(line_segments) < 2:
                return self.y_horizon_smoothed, 0.1
            
            # Find vanishing point (intersection of two most prominent lines)
            # Group lines by approximate slope direction
            left_lines = [l for l in line_segments if l[0] > 0]  # Positive slope
            right_lines = [l for l in line_segments if l[0] < 0]  # Negative slope
            
            if len(left_lines) > 0 and len(right_lines) > 0:
                # Use average of each group
                left_line = left_lines[len(left_lines)//2]  # Median line
                right_line = right_lines[len(right_lines)//2]
                
                # Calculate intersection (vanishing point)
                # Line 1: y = m1*x + b1
                # Line 2: y = m2*x + b2
                # Intersection: m1*x + b1 = m2*x + b2
                m1, b1 = left_line[0], left_line[1]
                m2, b2 = right_line[0], right_line[1]
                
                if abs(m1 - m2) > 0.01:  # Lines not parallel
                    x_vp = (b2 - b1) / (m1 - m2)
                    y_vp = m1 * x_vp + b1
                    
                    # Check if vanishing point is reasonable (within frame)
                    if 0 <= y_vp <= h and 0 <= x_vp <= w:
                        # Confidence increases if VP is near center horizontally
                        x_confidence = 1.0 - abs(x_vp - w/2) / (w/2)
                        y_confidence = 1.0 if 0 < y_vp < h * 0.7 else 0.3
                        confidence = 0.7 * x_confidence + 0.3 * y_confidence
                        
                        return float(y_vp), float(np.clip(confidence, 0.1, 0.9))
            
            # Fallback: use edge-based approach
            return self._estimate_horizon_from_edges(edges)
            
        except Exception as e:
            print(f"⚠️  Horizon detection error: {e}")
            return self.y_horizon_smoothed, 0.0
    
    def _estimate_horizon_from_edges(self, edges):
        """
        Fallback: Estimate horizon as the horizontal level with most edge activity.
        Assumes road edges cluster around horizon line.
        """
        try:
            # Count edge pixels per row in middle section
            h, w = edges.shape
            row_activity = np.sum(edges, axis=1)
            
            # Look for peak in upper-middle region (horizon typically 40-60% down)
            search_region = row_activity[int(h*0.3):int(h*0.65)]
            if len(search_region) > 0:
                peak_idx = np.argmax(search_region)
                y_horizon_est = int(h * 0.3) + peak_idx
                confidence = 0.4
                return float(y_horizon_est), confidence
        except Exception as e:
            print(f"⚠️  Edge-based horizon estimation failed: {e}")
        
        return self.y_horizon_smoothed, 0.0
    
    def update(self, frame):
        """
        Update horizon estimate with temporal smoothing (EMA).
        
        Args:
            frame: Current video frame (BGR)
            
        Returns:
            y_horizon: Smoothed horizon y-coordinate
            confidence: Overall confidence [0, 1]
        """
        # Detect new horizon
        y_new, conf_new = self.detect_horizon_vanishing_point(frame)
        
        self.y_horizon_detected = y_new
        self.detection_confidence = conf_new
        self.detection_count += 1
        
        # Adaptive EMA: higher confidence → faster learning
        alpha_adaptive = self.ema_alpha * conf_new
        
        # Update smoothed horizon
        self.y_horizon_smoothed = (
            alpha_adaptive * y_new + 
            (1 - alpha_adaptive) * self.y_horizon_smoothed
        )
        
        # Sanity check: horizon shouldn't move more than 30px per frame
        horizon_delta = abs(self.y_horizon_smoothed - self.y_horizon_prev)
        if horizon_delta > 30:
            # Too much change, revert to previous
            self.y_horizon_smoothed = self.y_horizon_prev + np.sign(
                self.y_horizon_smoothed - self.y_horizon_prev) * 30
        
        self.y_horizon_prev = self.y_horizon_smoothed
        
        return self.y_horizon_smoothed, self.detection_confidence
    
    def get_horizon(self):
        """Get current horizon without updating"""
        return self.y_horizon_smoothed


# ============================================================================
# REAR-VIEW SAFETY ASSESSMENT (Based on Traffic Safety Paper - SSMs)
# ============================================================================

# ============================================================================
# LANE DETECTION AND RIDER ACTION LOGIC
# ============================================================================
class LaneDetector:
    """
    Detects vehicle lane position (left, center/same, right) based on bounding box
    Enables lane-aware safety assessment and decision logic
    """
    
    def __init__(self, frame_width=1920, frame_height=1080, lane_overlap=0.1):
        """
        Initialize lane detector
        
        Args:
            frame_width: Video frame width (default 1920)
            frame_height: Video frame height (default 1080)
            lane_overlap: Overlap percentage between lanes (0-1) for smooth transitions
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.lane_overlap = lane_overlap  # Allow 10% overlap between lane boundaries
        
        # Divide frame into 3 lanes horizontally
        # NOTE: Camera perspective is mirrored for rear-view
        # Camera view:  0-33% (LEFT) → 33%-66% (CENTER) → 66%-100% (RIGHT)
        # Rider view:   0-33% (RIGHT) ← 33%-66% (CENTER) ← 66%-100% (LEFT)
        self.lane_width = frame_width / 3.0
        
    def detect_lane(self, bbox):
        """
        Determine which lane a vehicle is in based on bounding box center
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box coordinates
            
        Returns:
            dict with lane info: 'lane', 'position', 'confidence', 'center_x'
        """
        x1, y1, x2, y2 = bbox
        bbox_center_x = (x1 + x2) / 2.0
        bbox_width = x2 - x1
        
        # Detect primary lane based on center
        # NOTE: Rear-view camera flips perspective - camera LEFT = rider RIGHT and vice versa
        if bbox_center_x < self.lane_width:
            primary_lane = "RIGHT"  # Camera LEFT = Rider RIGHT
        elif bbox_center_x < 2 * self.lane_width:
            primary_lane = "CENTER"  # Camera CENTER = Rider CENTER
        else:
            primary_lane = "LEFT"  # Camera RIGHT = Rider LEFT
        
        # Calculate confidence (how centered in the lane)
        # Full width vehicle might span multiple lanes
        left_boundary = max(x1, 0)
        right_boundary = min(x2, self.frame_width)
        
        # Check lane coverage
        left_lane_end = self.lane_width
        center_lane_start = self.lane_width
        center_lane_end = 2 * self.lane_width
        right_lane_start = 2 * self.lane_width
        
        # Percentage of vehicle in each lane
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
        
        # Confidence is how much of vehicle is in primary lane
        # After flipping for rear-view perspective (LEFT <-> RIGHT)
        if primary_lane == "LEFT":
            confidence = right_pct  # Camera RIGHT = Rider LEFT
        elif primary_lane == "CENTER":
            confidence = center_pct
        else:
            confidence = left_pct  # Camera LEFT = Rider RIGHT
        
        # Detect if vehicle spans multiple lanes
        lane_span = []
        if right_pct > 0.15:  # Camera RIGHT = Rider LEFT
            lane_span.append("LEFT")
        if center_pct > 0.15:
            lane_span.append("CENTER")
        if left_pct > 0.15:  # Camera LEFT = Rider RIGHT
            lane_span.append("RIGHT")
        
        return {
            'lane': primary_lane,
            'confidence': float(confidence),
            'lane_coverage': {
                'LEFT': float(right_pct),    # Camera RIGHT = Rider LEFT
                'CENTER': float(center_pct),
                'RIGHT': float(left_pct)     # Camera LEFT = Rider RIGHT
            },
            'spans_multiple_lanes': len(lane_span) > 1,
            'lanes_occupied': lane_span,
            'center_x': float(bbox_center_x),
            'bbox_width': float(bbox_width)
        }


class RiderActionRecommendation:
    """
    Generates natural language rider action recommendations based on safety level,
    lane position, and vehicle behavior
    
    Recommendations are easy for rider to understand and actionable:
    - Maintain speed
    - Decelerate / Slow down
    - Accelerate / Speed up
    - Change lane (left/right)
    """
    
    def __init__(self):
        self.context_history = {}  # Track recommendations over time
    
    def get_rider_action(self, safety_level, lane_info, distance_m, speed_kmh,
                         relative_speed_kmh, motion, ego_speed_kmh):
        """
        Generate rider action recommendation
        
        Args:
            safety_level: 'CRITICAL', 'WARNING', 'CAUTION', 'SAFE'
            lane_info: dict from LaneDetector with 'lane', 'confidence'
            distance_m: Distance to vehicle in meters
            speed_kmh: Vehicle speed in km/h
            relative_speed_kmh: Speed difference (vehicle_speed - ego_speed)
            motion: 'approaching', 'receding', 'stable'
            ego_speed_kmh: Ego vehicle speed in km/h
            
        Returns:
            dict with 'action', 'reason', 'urgency', 'description'
        """
        vehicle_lane = lane_info.get('lane', 'CENTER')
        same_lane = vehicle_lane == 'CENTER'
        
        # SAME LANE VEHICLES - Collision risk
        if same_lane:
            if safety_level == 'CRITICAL':
                if relative_speed_kmh > 5:
                    return {
                        'action': 'EMERGENCY_BRAKE',
                        'urgency': 'CRITICAL',
                        'description': f'⚠️ IMMEDIATE BRAKING REQUIRED! Vehicle {distance_m:.1f}m away approaching at {speed_kmh:.0f}km/h',
                        'rider_instruction': 'Apply strong brakes immediately!',
                        'reason': f'Collision imminent - vehicle catching up at {relative_speed_kmh:.1f}km/h relative speed'
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
                        'description': f'✓ Safe: Vehicle {distance_m:.1f}m away at {speed_kmh:.0f}km/h',
                        'rider_instruction': 'Maintain current speed and lane position.',
                        'reason': 'Safe following distance maintained'
                    }
                else:
                    return {
                        'action': 'MAINTAIN_SPEED',
                        'urgency': 'LOW',
                        'description': f'✓ Safe: Vehicle {distance_m:.1f}m away, all clear',
                        'rider_instruction': 'Continue normal driving.',
                        'reason': 'Adequate safety margins'
                    }
        
        # ADJACENT LANE VEHICLES - No collision risk, just awareness
        else:
            if motion == 'approaching' and distance_m < 15:
                return {
                    'action': 'BE_AWARE',
                    'urgency': 'LOW',
                    'description': f'ℹ️ Vehicle in {vehicle_lane} lane approaching - {distance_m:.1f}m away',
                    'rider_instruction': f'Be aware of vehicle in {vehicle_lane} lane. Stay in your lane.',
                    'reason': f'Vehicle in {vehicle_lane} lane approaching but in different lane - no collision risk'
                }
            elif motion == 'approaching':
                return {
                    'action': 'MONITOR',
                    'urgency': 'LOW',
                    'description': f'ℹ️ Vehicle in {vehicle_lane} lane - {distance_m:.1f}m away',
                    'rider_instruction': f'Monitor vehicle in {vehicle_lane} lane.',
                    'reason': f'Vehicle in adjacent {vehicle_lane} lane'
                }
            else:
                return {
                    'action': 'MONITOR',
                    'urgency': 'LOW',
                    'description': f'ℹ️ Vehicle in {vehicle_lane} lane - {distance_m:.1f}m away',
                    'rider_instruction': f'Monitor vehicle in {vehicle_lane} lane.',
                    'reason': f'Vehicle in adjacent {vehicle_lane} lane'
                }


class RearViewSafetyAssessment:
    """
    Rear-View ADAS Safety Assessment based on Surrogate Safety Measures (SSMs)
    reference: "Traffic safety evaluation using surrogate safety measures in 
    the context of Indian mixed traffic" (2025)
    
    Implements SSMs for rear-end collision detection:
    - Time to Collision (TTC)
    - Post Encroachment Time (PET)
    - Time Exposed Time-to-Collision (TET)
    - Deceleration Rate to Avoid Collision (DRAC)
    - Modified TTC (MTTC) for mixed traffic
    """
    
    # Critical thresholds from paper for rear-end conflicts
    # SAME LANE thresholds (strict)
    TTC_CRITICAL = 1.0      # seconds - critical threshold
    TTC_WARNING = 1.5       # seconds - warning threshold
    TTC_SAFE = 2.5          # seconds - safe threshold
    
    # ADJACENT LANE thresholds (relaxed - no collision risk)
    TTC_CRITICAL_ADJACENT = 0.5   # Won't collide, just informational
    TTC_WARNING_ADJACENT = 1.0    # Won't collide, just informational
    
    PET_CRITICAL = 1.0      # seconds - Post Encroachment Time critical
    
    DRAC_CRITICAL = 3.35    # m/s² - Deceleration Rate to Avoid Collision
    DRAC_WARNING = 2.0      # m/s²
    
    DISTANCE_CRITICAL = 10.0  # meters - minimum safe distance
    DISTANCE_WARNING = 15.0   # meters
    
    def __init__(self, ego_vehicle_speed=0.0):
        """
        Initialize rear-view safety assessment
        
        Args:
            ego_vehicle_speed: Ego vehicle speed in m/s
        """
        self.ego_vehicle_speed = ego_vehicle_speed
        self.reaction_time = 1.0  # sec - human reaction time per paper
        self.vehicle_length = 4.5  # meters - typical car length
        
    def calculate_ttc(self, distance_m, ego_speed_ms, rear_vehicle_speed_ms):
        """
        Calculate Time to Collision (TTC) for rear-end collision
        TTC = Distance / (Rear_Speed - Ego_Speed)
        Based on: Hayward (1972) formula for rear-end conflicts
        
        Args:
            distance_m: Distance from rear vehicle to ego vehicle (meters)
            ego_speed_ms: Ego vehicle speed (m/s)
            rear_vehicle_speed_ms: Following vehicle speed (m/s)
            
        Returns:
            ttc_s: Time to Collision in seconds (None if vehicles moving away)
        """
        if distance_m <= 0:
            return None
        
        relative_speed = rear_vehicle_speed_ms - ego_speed_ms
        
        # If rear vehicle slower or same speed -> no collision risk
        if relative_speed <= 0.1:  # 0.1 m/s threshold to avoid division issues
            return None
        
        ttc = distance_m / relative_speed
        return max(0.0, ttc)
    
    def calculate_mttc(self, distance_m, ego_speed_ms, rear_vehicle_speed_ms,
                       ego_accel_ms2, rear_accel_ms2):
        """
        Calculate Modified Time to Collision (MTTC) accounting for acceleration
        Suitable for mixed traffic with lane violations
        
        Args:
            distance_m: Distance between vehicles
            ego_speed_ms: Ego vehicle speed
            rear_vehicle_speed_ms: Rear vehicle speed
            ego_accel_ms2: Ego vehicle acceleration
            rear_accel_ms2: Rear vehicle acceleration
            
        Returns:
            mttc_s: Modified TTC in seconds
        """
        d_speed = rear_vehicle_speed_ms - ego_speed_ms
        d_accel = rear_accel_ms2 - ego_accel_ms2
        
        # Case 1: Acceleration difference exists
        if abs(d_accel) > 0.01:
            discriminant = d_speed**2 + 2*d_accel*distance_m
            
            if discriminant < 0:
                return None  # No collision
            
            mttc = (-d_speed + np.sqrt(discriminant)) / d_accel
            return max(0.0, mttc)
        
        # Case 2: No acceleration, use standard TTC
        else:
            if d_speed > 0.1:
                return distance_m / d_speed
            else:
                return None
    
    def calculate_pet(self, distance_m, ego_speed_ms, rear_vehicle_speed_ms):
        """
        Calculate Post Encroachment Time (PET)
        PET = Time for rear vehicle to reach ego's position after ego moves away
        
        Args:
            distance_m: Current distance between vehicles
            ego_speed_ms: Ego vehicle speed
            rear_vehicle_speed_ms: Rear vehicle speed
            
        Returns:
            pet_s: Post Encroachment Time in seconds
        """
        relative_speed = rear_vehicle_speed_ms - ego_speed_ms
        
        if relative_speed <= 0.1:
            return float('inf')  # No threat
        
        # Account for vehicle length
        effective_distance = distance_m + self.vehicle_length
        pet = effective_distance / relative_speed
        return max(0.0, pet)
    
    def calculate_drac(self, distance_m, ego_speed_ms, rear_vehicle_speed_ms,
                       reaction_time=1.0):
        """
        Calculate Deceleration Rate to Avoid Collision (DRAC)
        Minimum deceleration needed by rear vehicle to avoid collision
        
        Args:
            distance_m: Current distance
            ego_speed_ms: Ego vehicle speed
            rear_vehicle_speed_ms: Rear vehicle speed
            reaction_time: Driver reaction time in seconds
            
        Returns:
            drac_ms2: Required deceleration in m/s²
        """
        if distance_m <= 0:
            return float('inf')
        
        # Distance traveled by rear vehicle during reaction time
        reaction_distance = rear_vehicle_speed_ms * reaction_time
        
        # Available distance for deceleration
        available_distance = distance_m - reaction_distance
        
        if available_distance <= 0:
            return float('inf')  # Collision unavoidable
        
        # v² = u² - 2*a*s => a = (u² - v²) / (2*s)
        # Assuming full stop: v = 0
        if rear_vehicle_speed_ms > 0:
            drac = (rear_vehicle_speed_ms**2) / (2 * available_distance)
        else:
            drac = 0.0
        
        return max(0.0, drac)
    
    def calculate_tet(self, distance_m, ego_speed_ms, rear_vehicle_speed_ms,
                      ttc_threshold=1.5):
        """
        Calculate Time Exposed Time-to-Collision (TET)
        Aggregates duration when TTC is below critical threshold
        
        Args:
            distance_m: Current distance
            ego_speed_ms: Ego speed
            rear_vehicle_speed_ms: Rear vehicle speed
            ttc_threshold: Critical TTC threshold
            
        Returns:
            tet_s: Exposure time below threshold
        """
        ttc = self.calculate_ttc(distance_m, ego_speed_ms, rear_vehicle_speed_ms)
        
        if ttc is None or ttc >= ttc_threshold:
            return 0.0
        
        return ttc_threshold - ttc
    
    def assess_risk_level(self, ttc_s, mttc_s, pet_s, drac_ms2, distance_m,
                          ego_speed_ms, rear_speed_ms, lane_info=None):
        """
        Comprehensive lane-aware risk assessment based on multiple SSMs
        
        Args:
            lane_info: dict from LaneDetector with 'lane' and 'confidence'
                      If None, assumes same lane (safe default)
        
        Returns:
            dict with keys: 'level', 'message', 'alert_type', 'confidence', 'lane_aware'
        """
        # Determine if vehicle is in same lane
        is_same_lane = lane_info is None or lane_info.get('lane', 'CENTER') == 'CENTER'
        lane_name = lane_info.get('lane', 'CENTER') if lane_info else 'CENTER'
        
        # Priority: Use MTTC if available (better for mixed traffic)
        collision_time = mttc_s if mttc_s is not None else ttc_s
        
        # SAME LANE ASSESSMENT (strict thresholds)
        if is_same_lane:
            # Critical condition indicators
            has_critical_ttc = collision_time is not None and collision_time < self.TTC_CRITICAL
            has_critical_drac = drac_ms2 > self.DRAC_CRITICAL
            has_critical_distance = distance_m < self.DISTANCE_CRITICAL
            has_critical_pet = pet_s < self.PET_CRITICAL
            
            # Count critical indicators
            critical_count = sum([has_critical_ttc, has_critical_drac, 
                                 has_critical_distance, has_critical_pet])
            
            if critical_count >= 2:
                level = "CRITICAL"
                alert_type = "collision_imminent"
                confidence = min(1.0, critical_count / 4.0)
                
                if has_critical_ttc and has_critical_drac:
                    message = f"⚠️ CRITICAL: TTC={collision_time:.2f}s, DRAC={drac_ms2:.2f}m/s²"
                elif has_critical_pet:
                    message = f"⚠️ CRITICAL: PET={pet_s:.2f}s < {self.PET_CRITICAL}s"
                else:
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
                message = f"⚠️ WARNING: High deceleration required: {drac_ms2:.2f}m/s²"
            
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
        
        # ADJACENT LANE ASSESSMENT (relaxed/informational only)
        else:
            # Only provide awareness alerts, never collision warnings
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


class RearSideUseCaseValidator:
    """
    Rear-Side Use Case Validation for rear-view ADAS
    Validates that detected vehicles are within rear-view camera FOV
    and that safety assessments are reliable
    """
    
    def __init__(self, frame_width=1920, frame_height=1080):
        """
        Initialize rear-side validator
        
        Args:
            frame_width: Video frame width
            frame_height: Video frame height
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Rear-view camera typically covers ~170° horizontal FOV
        # Center portion is 80-90% of frame horizontally
        self.rear_view_horizontal_margin = 0.05  # 5% margin on each side
        self.rear_view_vertical_threshold = 0.2  # Top 20% is typically mirror/sky
        
    def is_valid_rear_detection(self, bbox, bbox_height, distance_m):
        """
        Validate if detection is reliable for rear-view ADAS
        
        Checks:
        1. Bounding box is within valid rear-view region
        2. Bounding box size indicates reliable distance estimate
        3. Distance is within rear-view monitoring range
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            bbox_height: Height of bounding box in pixels
            distance_m: Estimated distance in meters
            
        Returns:
            dict with validation status and details
        """
        x1, y1, x2, y2 = bbox
        
        # Check 1: Horizontal position (should be centered rear-view region)
        margin_left = self.frame_width * self.rear_view_horizontal_margin
        margin_right = self.frame_width * (1 - self.rear_view_horizontal_margin)
        
        in_horizontal_range = margin_left <= x1 and x2 <= margin_right
        # Allow some vehicles on edges but mark as lower confidence
        horizontal_confidence = 1.0 if in_horizontal_range else 0.7
        
        # Check 2: Vertical position (avoid mirror frame and sky)
        top_margin = self.frame_height * self.rear_view_vertical_threshold
        in_vertical_range = y1 >= top_margin
        
        # Check 3: Bounding box height (vehicle must be visible enough)
        min_bbox_height = 20  # pixels
        bbox_height_valid = bbox_height >= min_bbox_height
        bbox_confidence = min(1.0, bbox_height / 100.0)  # Normalize to 100px
        
        # Check 4: Distance range (rear-view typically monitors up to 25-30m)
        distance_min = 0.5  # meters
        distance_max = 30.0  # meters
        distance_valid = distance_min <= distance_m <= distance_max if distance_m else False
        
        # Calculate overall validation score
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
            'issues': [
                "Outside horizontal FOV" if not in_horizontal_range else None,
                "In mirror/sky region" if not in_vertical_range else None,
                "Bounding box too small" if not bbox_height_valid else None,
                "Distance out of range" if not distance_valid else None,
            ]
        }
    
    def validate_rear_scenario(self, detections, ego_speed_ms):
        """
        Validate complete rear-view scenario
        
        Args:
            detections: List of detection dictionaries with bbox, distance, etc.
            ego_speed_ms: Ego vehicle speed in m/s
            
        Returns:
            dict with scenario validation and criticality assessment
        """
        if not detections:
            return {
                'scenario_valid': True,
                'scenario_type': 'clear_rear',
                'threat_level': 'none',
                'critical_vehicles': [],
                'validation_details': []
            }
        
        critical_vehicles = []
        
        for detection in detections:
            bbox = detection.get('bbox')
            distance = detection.get('distance')
            bbox_height = bbox[3] - bbox[1] if bbox else 0
            
            if bbox is None or distance is None:
                continue
            
            # Validate detection
            validation = self.is_valid_rear_detection(bbox, bbox_height, distance)
            
            if validation['is_valid']:
                # Assessment based on distance and speed
                if distance < 10.0 and ego_speed_ms > 10:  # High speed, small distance
                    critical_vehicles.append({
                        'class': detection.get('class'),
                        'distance': distance,
                        'motion': detection.get('motion'),
                        'track_id': detection.get('track_id'),
                        'validation_confidence': validation['confidence']
                    })
        
        # Determine scenario type
        if not critical_vehicles:
            scenario_type = "clear_rear" if not [d for d in detections 
                                                 if d.get('distance', float('inf')) < 30] else "vehicles_monitored"
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
            'total_vehicles_detected': len([d for d in detections if d.get('distance')]),
        }


# ============================================================================
# CSV LOGGER FOR VEHICLE DETECTIONS
# ============================================================================

class DetectionLogger:
    """CSV logger for vehicle detection data with rear-view safety assessment"""
    
    def __init__(self, log_file):
        """
        Initialize CSV logger
        
        Args:
            log_file: Path to CSV file for logging
        """
        self.log_file = log_file
        self.file_handle = open(log_file, 'w', newline='')
        self.writer = csv.DictWriter(
            self.file_handle,
            fieldnames=[
                'frame_number',
                'track_id',
                'vehicle_class',
                'confidence',
                'distance_m',
                'speed_kmh',
                'motion_state',
                'bbox_x1',
                'bbox_y1',
                'bbox_x2',
                'bbox_y2',
                'bbox_width',
                'bbox_height',
                'classical_depth',
                'ml_depth',
                'timestamp_s',
                # Rear-view safety assessment fields
                'safety_level',
                'alert_type',
                'ttc_s',
                'mttc_s',
                'pet_s',
                'drac_ms2',
                'rear_validation_score',
                'scenario_type',
            ]
        )
        self.writer.writeheader()
    
    def log_detections(self, frame_number, detections, timestamp_s=0.0, 
                       safety_assessments=None, scenario_validation=None):
        """
        Log all detections for a frame with safety assessment data
        
        Args:
            frame_number: Current frame number
            detections: List of detection dictionaries
            timestamp_s: Timestamp in seconds
            safety_assessments: Dict of track_id -> safety assessment
            scenario_validation: Scenario validation result
        """
        if safety_assessments is None:
            safety_assessments = {}
        if scenario_validation is None:
            scenario_validation = {'scenario_type': 'unknown'}
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            distance_metadata = det.get('distance_metadata', {})
            track_id = det.get('track_id', -1)
            safety_assess = safety_assessments.get(track_id, {})
            
            row = {
                'frame_number': frame_number,
                'track_id': track_id,
                'vehicle_class': det.get('class', 'unknown'),
                'confidence': f"{det.get('confidence', 0.0):.4f}",
                'distance_m': f"{det.get('distance', 0.0):.2f}" if det.get('distance') else '',
                'speed_kmh': f"{det.get('speed', 0.0):.2f}",
                'motion_state': det.get('motion', 'unknown'),
                'bbox_x1': int(x1),
                'bbox_y1': int(y1),
                'bbox_x2': int(x2),
                'bbox_y2': int(y2),
                'bbox_width': int(bbox_width),
                'bbox_height': int(bbox_height),
                'classical_depth': f"{distance_metadata.get('classical_fused', 0.0):.2f}" if distance_metadata.get('classical_fused') else '',
                'ml_depth': f"{distance_metadata.get('ml', 0.0):.2f}" if distance_metadata.get('ml') else '',
                'timestamp_s': f"{timestamp_s:.4f}",
                # Safety assessment fields
                'safety_level': safety_assess.get('level', 'unknown'),
                'alert_type': safety_assess.get('alert_type', 'none'),
                'ttc_s': f"{safety_assess.get('ttc', 0.0):.3f}" if safety_assess.get('ttc') else '',
                'mttc_s': f"{safety_assess.get('mttc', 0.0):.3f}" if safety_assess.get('mttc') else '',
                'pet_s': f"{safety_assess.get('pet', 0.0):.3f}" if safety_assess.get('pet') else '',
                'drac_ms2': f"{safety_assess.get('drac', 0.0):.3f}" if safety_assess.get('drac') else '',
                'rear_validation_score': '',
                'scenario_type': scenario_validation.get('scenario_type', 'unknown'),
            }
            self.writer.writerow(row)
    
    def flush(self):
        """Flush data to disk"""
        self.file_handle.flush()
    
    def close(self):
        """Close the CSV file"""
        if self.file_handle:
            self.file_handle.close()


# ============================================================================
# CONFIGURATION
# ============================================================================
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.4

# Camera parameters (typical rear-view camera)
FOCAL_LENGTH = 1000  # pixels (approximate)
MOUNTING_HEIGHT_M = 1.1
GROUND_PLANE_RATIO = 0.55
DEPTH_CLIP_MIN_M = 0.5
DEPTH_CLIP_MAX_M = 25.0

    # Vehicle dimensions for size-based depth estimation (NEW: improved with widths)
# Format: (height_m, width_m, height_uncertainty_m, typical_rear_aspect_ratio)
VEHICLE_DIMENSIONS = {
    'Person': (1.7, 0.45, 0.15, 0.35),
    'Bicycle': (1.2, 0.65, 0.1, 0.55),
    'Two-wheeler': (1.3, 0.7, 0.1, 0.6),                # Motorcycle with rider
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
    'Auto-rickshaw': (2.0, 1.6, 0.15, 0.85),
    
    # Merged entities (rider + vehicle)
    'Person + Two-wheeler': (2.0, 0.8, 0.2, 0.65),
    'Person + Bicycle': (1.95, 0.7, 0.2, 0.6),
    'Person + Three-wheeler': (1.8, 1.4, 0.2, 0.85),
    
    'Others': (1.5, 1.6, 0.2, 0.95),
}

# Legacy REAL_HEIGHTS for backward compatibility
REAL_HEIGHTS = {k: v[0] for k, v in VEHICLE_DIMENSIONS.items()}

# Distance range validation for rear-view camera
TYPICAL_DISTANCE_RANGES = {
    'Person': (0.5, 50),        # Can see people up to ~50m
    'Bicycle': (0.5, 50),
    'Two-wheeler': (0.5, 60),
    'Auto-rickshaw': (0.5, 70),
    'Sedan': (0.5, 100),
    'SUV': (0.5, 110),
    'Bus': (0.5, 150),
    'Truck': (0.5, 150),
    'Others': (0.5, 100),
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


def load_depth_with_fallback(device: str, interval: int):
    """Load best available depth backend with DA2 priority."""
    if not _DEPTH_LITE_AVAILABLE:
        return None

    candidates = [
        ("onnx", CNN_DIR / "models" / "depth_lite" / "da2_kitti_metric.onnx", True),
        ("pytorch", CNN_DIR / "models" / "depth_lite" / "da2_kitti_metric.pt", True),
        ("onnx", CNN_DIR / "models" / "depth_lite" / "midas_kitti_metric.onnx", True),
    ]

    for backend, model_path, metric in candidates:
        if model_path.exists():
            try:
                model = AsyncDepthLite(
                    backend=backend,
                    device=device,
                    update_interval_frames=interval,
                    model_path=str(model_path),
                    metric_output=metric,
                )
                print(f"✅ Depth loaded [{backend}]: {model_path.name}")
                return model
            except Exception as exc:
                print(f"⚠️ Depth backend failed for {model_path.name}: {exc}")

    for backend in ("trt_fp16", "onnx", "pytorch"):
        try:
            model = AsyncDepthLite(
                backend=backend,
                device=device,
                update_interval_frames=interval,
            )
            print(f"✅ Depth loaded fallback backend: {backend}")
            return model
        except Exception:
            pass
    return None


class VideoDetector:
    def __init__(self, device='cuda', zoedepth_interval=30,
                 correction_alpha=0.3, learnable_alpha=True,
                 alpha_lr=0.05, classical_depth_weight=0.80, fps=30):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🔥 Device: {self.device}")
        self.fps = fps
        # Load YOLO
        print("📦 Loading YOLO...")
        self.yolo = YOLO(YOLO_MODEL_PATH)
        print("✅ YOLO loaded")
        # Load Fine-tuned Classifier
        CLASSIFIER_PATH = str(CNN_DIR / "models/classifier/weights/best.pt")
        print(f"📦 Loading Classifier: {CLASSIFIER_PATH}")
        self.classifier = YOLO(CLASSIFIER_PATH)
        print("✅ Classifier loaded")
        
        # Initialize ByteTracker (replaces IoU-based tracking)
        if _BYTE_TRACKER_AVAILABLE:
            print("📦 Initializing ByteTracker...")
            self.tracker = ByteTracker(track_buffer=300, frame_rate=int(fps))
            self.use_byte_tracker = True
            print("✅ ByteTracker initialized (motion-aware tracking enabled)")
        else:
            print("⚠️  ByteTracker not available, falling back to IoU tracking")
            self.tracker = None
            self.use_byte_tracker = False
        
        # Keep legacy tracking for fallback
        self.prev_distances = {}  # track_id -> [distances over time]
        self.track_id_counter = 0
        self.prev_boxes = []
        self.track_classes = defaultdict(lambda: deque(maxlen=5))
        self.IOU_THRESHOLD = 0.45
        self.MIN_CROP_AREA = 64 * 64  # pixels
        # Depth system (DA2/MIDAS - no ZoeDepth fallback)
        self.depth_model = load_depth_with_fallback(str(self.device), zoedepth_interval)
        if self.depth_model is None:
            print("⚠️  No depth model available (DA2/MIDAS). Using classical depth only.")
        self.zoedepth_interval = zoedepth_interval
        self.zoedepth_frame_counter = 0
        self.last_zoedepth_depth = None
        self._ml_depth_fresh = False

        self.track_correction_factors = {}
        self.class_correction_factors = {}
        self.correction_ema_alpha = float(np.clip(correction_alpha, 0.02, 0.98))
        self.learnable_alpha = bool(learnable_alpha)
        self.alpha_lr = float(np.clip(alpha_lr, 0.001, 0.5))
        self.alpha_min = 0.05
        self.alpha_max = 0.95
        self.correction_clip_min = 0.75
        self.correction_clip_max = 1.35
        self.classical_depth_weight = classical_depth_weight

        self.track_motion_state = {}
        self.kalman_filters = defaultdict(lambda: KalmanFilter1D())
        
        # Kalman filters for size-based depth smoothing (NEW: temporal smoothing)
        self.size_depth_filters = defaultdict(lambda: KalmanFilter1D(
            process_variance=0.1,
            measurement_variance=0.5
        ))
        
        # Initialize Rear-View Safety Assessment
        self.rear_safety_assessment = RearViewSafetyAssessment(ego_vehicle_speed=0.0)
        self.rear_side_validator = RearSideUseCaseValidator(frame_width=1920, frame_height=1080)
        self.lane_detector = LaneDetector(frame_width=1920, frame_height=1080)
        
        # Initialize Dynamic Horizon Estimator (adapts to camera suspension movement)
        self.horizon_estimator = DynamicHorizonEstimator(
            frame_width=1920, 
            frame_height=1080,
            ema_alpha=0.15,  # Smooth EMA (15% new, 85% previous)
            fallback_ratio=0.55  # Fallback to fixed horizon
        )
        print("✅ Dynamic Horizon Estimator initialized (adapts to suspension movement)")
        
        self.rider_action_recommender = RiderActionRecommendation()
        self.safety_assessments_cache = {}  # track_id -> safety assessment dict
        self.last_scenario_validation = None

    def _estimate_ground_plane_depth(self, bbox, frame_h, y_horizon=None):
        """
        Estimate distance using ground plane projection with adaptive horizon.
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            frame_h: Frame height in pixels
            y_horizon: Dynamic horizon y-coordinate (if None, uses fallback)
            
        Returns:
            distance, confidence
        """
        y_bottom = float(bbox[3])
        
        # Use dynamic horizon if provided, otherwise use fallback
        if y_horizon is None:
            y_horizon = frame_h * (1.0 - GROUND_PLANE_RATIO)
        else:
            y_horizon = float(y_horizon)
        
        dy = y_bottom - y_horizon
        if dy <= 1.0:
            return None, 0.0
        
        z = (MOUNTING_HEIGHT_M * FOCAL_LENGTH) / dy
        z = float(np.clip(z, DEPTH_CLIP_MIN_M, DEPTH_CLIP_MAX_M))
        conf = float(np.clip(dy / max(frame_h * 0.35, 1.0), 0.2, 1.0))
        return z, conf

    def _estimate_size_depth(self, bbox_height, class_name):
        """Estimate depth using object size with improved confidence (NEW: multi-factor)"""
        if bbox_height <= 0:
            return None, 0.0
        real_height = REAL_HEIGHTS.get(class_name, 1.5)
        z = (real_height * FOCAL_LENGTH) / max(float(bbox_height), 1.0)
        z = float(np.clip(z, DEPTH_CLIP_MIN_M, DEPTH_CLIP_MAX_M))
        
        # OLD confidence (single factor): conf = bbox_height / 180.0
        # NEW confidence (multi-factor): pixel height ranges
        if bbox_height < 30:
            pixel_conf = 0.2  # Too small, too far
        elif bbox_height < 100:
            pixel_conf = 0.4 + (bbox_height - 30) / 700  # Ramping up
        elif bbox_height <= 300:
            pixel_conf = 0.95  # Optimal range
        else:
            pixel_conf = 0.85  # Too close, perspective distortion
        
        conf = float(np.clip(pixel_conf, 0.2, 0.95))
        return z, conf
    
    def detect_occlusion_level(self, bbox, frame_shape):
        """
        Detect if bounding box is occluded or cut off by frame boundaries.
        Returns occlusion ratio 0.0 (fully visible) to 1.0 (mostly hidden)
        """
        x1, y1, x2, y2 = [float(v) for v in bbox]
        frame_h, frame_w = frame_shape[:2]
        margin = 5  # pixels from edge
        
        # Vehicle cut off at sides
        if x1 < margin or x2 > (frame_w - margin):
            return 0.3  # Likely partially occluded laterally
        
        # Vehicle cut off at top (common in rear view)
        if y1 < margin:
            return 0.4
        
        # Vehicle cut off at bottom (less common but possible)
        if y2 > (frame_h - margin):
            return 0.25
        
        # Fully visible
        return 0.0
    
    def calculate_size_confidence_multifactor(self, bbox, class_name, frame_shape):
        """
        (NEW) Calculate multi-factor confidence for size-based estimation.
        Combines pixel height, aspect ratio, and frame position.
        """
        x1, y1, x2, y2 = [float(v) for v in bbox]
        frame_h, frame_w = frame_shape[:2]
        
        H_pixel = y2 - y1
        W_pixel = x2 - x1
        
        # Factor 1: Pixel height reliability (optimized ranges)
        if H_pixel < 30:
            pixel_conf = 0.2
        elif H_pixel < 100:
            pixel_conf = 0.4 + (H_pixel - 30) / 700
        elif H_pixel <= 300:
            pixel_conf = 0.95
        else:
            pixel_conf = 0.85  # Too close
        
        # Factor 2: Aspect ratio validation
        aspect_ratio = W_pixel / max(H_pixel, 1.0)
        
        if class_name in VEHICLE_DIMENSIONS:
            expected_aspect = VEHICLE_DIMENSIONS[class_name][3]
            aspect_norm = aspect_ratio / expected_aspect
            
            if 0.7 <= aspect_norm <= 1.3:
                aspect_conf = 1.0
            else:
                aspect_conf = max(0.3, 1.0 - abs(1.0 - aspect_norm) * 0.5)
        else:
            aspect_conf = 0.8
        
        # Factor 3: Frame position (occlusion)
        margin = 10
        is_cut_top = y1 < margin
        is_cut_side = (x1 < margin) or (x2 > (frame_w - margin))
        is_cut_bottom = y2 > (frame_h - margin)
        
        if is_cut_top or is_cut_side:
            frame_pos_conf = 0.4  # Cut off at top/sides = unreliable
        elif is_cut_bottom:
            frame_pos_conf = 0.6  # Possibly cut bottom (common in rear view)
        else:
            frame_pos_conf = 1.0  # Fully visible
        
        # Combine with weights
        combined_conf = (0.5 * pixel_conf + 0.3 * aspect_conf + 0.2 * frame_pos_conf)
        return float(np.clip(combined_conf, 0.15, 0.95))
    
    def smooth_size_based_distance(self, distance_raw, track_id):
        """Apply Kalman filter to smooth size-based distance estimates"""
        if distance_raw is None or distance_raw <= 0:
            return distance_raw
        
        distance_smooth = self.size_depth_filters[track_id].update(distance_raw)
        return float(np.clip(distance_smooth, DEPTH_CLIP_MIN_M, DEPTH_CLIP_MAX_M))
    
    def validate_distance_range(self, distance_m, class_name):
        """(NEW) Check if distance is within realistic range for rear-view camera"""
        min_range, max_range = TYPICAL_DISTANCE_RANGES.get(class_name, (0.5, 100))
        
        if distance_m < min_range:
            print(f"⚠️  {class_name} at {distance_m:.1f}m below typical range")
            return min_range, 0.7  # Likely detection artifact
        
        if distance_m > max_range:
            print(f"⚠️  {class_name} at {distance_m:.1f}m beyond typical range {max_range}m")
            return max_range, 0.5  # Extrapolation unreliable
        
        return distance_m, 1.0  # Within valid range

    def _estimate_motion_depth(self, track_id, bbox, timestamp_s, frame_shape):
        prev = self.track_motion_state.get(track_id)
        if prev is None:
            return None, 0.0
        x1, y1, x2, y2 = [float(v) for v in bbox]
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        bbox_h = max(1.0, y2 - y1)
        dt = max(1e-3, timestamp_s - prev['ts'])
        disparity = float(np.hypot(cx - prev['cx'], cy - prev['cy']))
        if disparity < 0.75:
            return None, 0.0
        baseline = float(np.clip(0.03 * (dt / (1.0 / 30.0)), 0.01, 0.25))
        z = (FOCAL_LENGTH * baseline) / float(np.clip(disparity, 0.75, 80.0))
        z_prev = prev.get('distance')
        if z_prev is not None and z_prev > 0:
            z_scale = float(z_prev) * (max(1.0, prev['bbox_h']) / bbox_h)
            z = 0.65 * z + 0.35 * z_scale
        z = float(np.clip(z, DEPTH_CLIP_MIN_M, DEPTH_CLIP_MAX_M))
        h, w = frame_shape[:2]
        conf = float(np.clip(disparity / max(np.hypot(w, h) * 0.025, 1.0), 0.1, 0.9))
        return z, conf

    def _update_motion_state(self, track_id, bbox, fused_distance, timestamp_s):
        x1, y1, x2, y2 = [float(v) for v in bbox]
        self.track_motion_state[track_id] = {
            'cx': 0.5 * (x1 + x2),
            'cy': 0.5 * (y1 + y2),
            'bbox_h': max(1.0, y2 - y1),
            'distance': float(fused_distance) if fused_distance is not None else None,
            'ts': float(timestamp_s),
        }

    def estimate_classical_distance(self, bbox, class_name, track_id, frame_shape, timestamp_s, y_horizon=None):
        bbox_h = max(1, int(bbox[3] - bbox[1]))
        z_ground, c_ground = self._estimate_ground_plane_depth(bbox, frame_shape[0], y_horizon=y_horizon)
        z_size, c_size = self._estimate_size_depth(bbox_h, class_name)
        z_motion, c_motion = self._estimate_motion_depth(track_id, bbox, timestamp_s, frame_shape)
        
        # (NEW) Enhanced size-based depth with multi-factor confidence
        # Apply occlusion detection
        occlusion_level = self.detect_occlusion_level(bbox, frame_shape)
        c_size *= (1.0 - occlusion_level)  # Reduce confidence for occluded vehicles
        
        # Apply multi-factor confidence calculation
        c_size_multifactor = self.calculate_size_confidence_multifactor(bbox, class_name, frame_shape)
        c_size = (c_size + c_size_multifactor) / 2.0  # Blend with original
        
        # Apply temporal smoothing (Kalman filter) to size-based depth
        if z_size is not None:
            z_size = self.smooth_size_based_distance(z_size, track_id)
        
        # For merged detections (Person + vehicle), use special handling
        if "Person + " in class_name:
            # Merged entities get penalty on confidence
            c_size *= 0.8  # 20% penalty for merged estimates
        
        # Validate distance range
        if z_size is not None:
            z_size, range_conf = self.validate_distance_range(z_size, class_name)
            c_size *= range_conf  # Apply range validation penalty if needed

        cues = []
        if z_ground is not None:
            cues.append((z_ground, 0.55 * c_ground))
        if z_size is not None:
            cues.append((z_size, 0.30 * c_size))
        if z_motion is not None:
            cues.append((z_motion, 0.15 * c_motion))

        if not cues:
            return None, {'ground': None, 'size': None, 'motion': None, 'classical_fused': None}

        num = sum(z * w for z, w in cues)
        den = sum(w for _, w in cues)
        fused = float(np.clip(num / max(den, 1e-6), DEPTH_CLIP_MIN_M, DEPTH_CLIP_MAX_M))
        self._update_motion_state(track_id, bbox, fused, timestamp_s)
        return fused, {
            'ground': z_ground,
            'size': z_size,
            'motion': z_motion,
            'classical_fused': fused,
        }

    def _sample_ml_depth_for_bbox(self, depth_map, bbox):
        h_map, w_map = depth_map.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(x1, w_map - 1))
        y1 = max(0, min(y1, h_map - 1))
        x2 = max(x1 + 1, min(x2, w_map))
        y2 = max(y1 + 1, min(y2, h_map))
        strip_h = max(2, int((y2 - y1) * 0.05))
        gy1 = max(0, y2 - strip_h)
        valid = depth_map[gy1:y2, x1:x2].flatten()
        valid = valid[valid > 0]
        if len(valid) > 10:
            return float(np.percentile(valid, 20)), float(min(1.0, len(valid) / 50.0)), 'ml_bbox_ground_strip'
        cx, cy = (x1 + x2) // 2, min(y2 - 1, h_map - 1)
        d = float(depth_map[cy, cx])
        return (d, 0.2, 'ml_bbox_centre_fallback') if d > 0 else (None, 0.0, 'ml_failed')

    def _compute_dual_depth(self, classical_depth, ml_depth, class_name, track_id, ml_fresh):
        if classical_depth is None or classical_depth <= 0:
            return classical_depth, 1.0, False
        c_prev = self.track_correction_factors.get(track_id, self.class_correction_factors.get(class_name, 1.0))
        c_new = c_prev
        updated = False
        if ml_fresh and ml_depth is not None and ml_depth > 0:
            alpha_prev = float(self.correction_ema_alpha)
            c_current = float(np.clip(ml_depth / max(classical_depth, 1e-6), self.correction_clip_min, self.correction_clip_max))
            if self.learnable_alpha:
                z_prev = float(classical_depth) * float(c_prev)
                rel_err = float(np.clip(abs(z_prev - ml_depth) / max(ml_depth, 1e-6), 0.0, 1.0))
                alpha_target = float(np.clip(self.alpha_min + (self.alpha_max - self.alpha_min) * rel_err,
                                             self.alpha_min, self.alpha_max))
                self.correction_ema_alpha = float(np.clip(
                    (1.0 - self.alpha_lr) * self.correction_ema_alpha + self.alpha_lr * alpha_target,
                    self.alpha_min, self.alpha_max,
                ))
                if abs(self.correction_ema_alpha - alpha_prev) > 1e-6:
                    print(f"🧠 Alpha update [track={track_id} class={class_name}] "
                          f"{alpha_prev:.3f}->{self.correction_ema_alpha:.3f}")
            c_new = self.correction_ema_alpha * c_current + (1.0 - self.correction_ema_alpha) * c_prev
            c_new = float(np.clip(c_new, self.correction_clip_min, self.correction_clip_max))
            self.track_correction_factors[track_id] = c_new
            self.class_correction_factors[class_name] = c_new
            updated = True
        z_corr = float(classical_depth) * float(c_new)
        z_final = self.classical_depth_weight * float(classical_depth) + (1.0 - self.classical_depth_weight) * z_corr
        return z_final, c_new, updated
    
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
        """
        Estimate if object is approaching, stable, or receding and calculate speed
        
        Optimized for ByteTracker: Uses fewer frames for faster response while maintaining accuracy
        thanks to ByteTracker's consistent IDs and smoother trajectory prediction
        """
        if track_id not in self.prev_distances:
            self.prev_distances[track_id] = []
        
        self.prev_distances[track_id].append(current_distance)
        
        # With ByteTracker's improved tracking, we can use fewer frames (10 instead of 30)
        # for faster motion detection while maintaining accuracy
        max_buffer = 10 if self.use_byte_tracker else 30
        if len(self.prev_distances[track_id]) > max_buffer:
            self.prev_distances[track_id].pop(0)
        
        # Need at least 5 frames to estimate reliably with ByteTracker (3 otherwise)
        min_frames = 5 if self.use_byte_tracker else 15
        if len(self.prev_distances[track_id]) < min_frames:
            return "stable", 0.0
        
        # Calculate trend using robust averaging
        distances = self.prev_distances[track_id]
        time_steps = len(distances) - 1
        if time_steps == 0: 
            return "stable", 0.0
        
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
        
        # Calculate speed in km/h (negative for receding, positive for approaching)
        speed_kmh = avg_change * self.fps * 3.6  # 3.6 to convert m/s to km/h
        
        # Threshold for motion detection (meters per frame)
        # 0.03 m/frame * 30 fps = ~1 m/s = 3.6 km/h
        threshold = 0.03
        
        if avg_change < -threshold:  # Getting closer
            return "approaching", abs(speed_kmh)
        elif avg_change > threshold:  # Getting farther
            return "receding", abs(speed_kmh)
        else:
            return "stable", 0.0
    
    def calculate_rear_safety_assessment(self, detection, ego_speed_kmh=0.0):
        """
        Calculate rear-view safety assessment for a single detection
        Based on Surrogate Safety Measures (SSMs) from traffic safety paper
        
        Args:
            detection: Detection dictionary with bbox, distance, speed, motion
            ego_speed_kmh: Ego vehicle speed in km/h
            
        Returns:
            dict with safety assessment results including lane-aware alerts and rider actions
        """
        distance_m = detection.get('distance')
        rear_speed_kmh = detection.get('speed', 0.0)
        motion = detection.get('motion', 'unknown')
        bbox = detection.get('bbox')
        
        if distance_m is None or distance_m <= 0:
            return {
                'level': 'unknown',
                'message': 'Unable to assess - no distance',
                'alert_type': 'none',
                'confidence': 0.0,
                'ttc': None,
                'mttc': None,
                'pet': None,
                'drac': None,
                'lane': 'unknown',
                'rider_action': None,
            }
        
        # --- LANE DETECTION ---
        lane_info = None
        if bbox is not None:
            lane_info = self.lane_detector.detect_lane(bbox)
        
        # Convert speeds to m/s
        ego_speed_ms = ego_speed_kmh / 3.6
        rear_speed_ms = rear_speed_kmh / 3.6
        
        # Calculate SSMs
        ttc = self.rear_safety_assessment.calculate_ttc(
            distance_m, ego_speed_ms, rear_speed_ms
        )
        
        mttc = self.rear_safety_assessment.calculate_mttc(
            distance_m, ego_speed_ms, rear_speed_ms,
            ego_accel_ms2=0.0,  # Assume constant ego speed
            rear_accel_ms2=0.1 if motion == 'approaching' else (-0.1 if motion == 'receding' else 0.0)
        )
        
        pet = self.rear_safety_assessment.calculate_pet(
            distance_m, ego_speed_ms, rear_speed_ms
        )
        
        drac = self.rear_safety_assessment.calculate_drac(
            distance_m, ego_speed_ms, rear_speed_ms,
            reaction_time=1.0
        )
        
        # --- LANE-AWARE RISK ASSESSMENT ---
        safety_result = self.rear_safety_assessment.assess_risk_level(
            ttc, mttc, pet, drac, distance_m, ego_speed_ms, rear_speed_ms,
            lane_info=lane_info  # Pass lane information
        )
        
        # --- RIDER ACTION RECOMMENDATION ---
        rider_action = self.rider_action_recommender.get_rider_action(
            safety_level=safety_result.get('level'),
            lane_info=lane_info if lane_info else {'lane': 'CENTER', 'confidence': 0.0},
            distance_m=distance_m,
            speed_kmh=rear_speed_kmh,
            relative_speed_kmh=rear_speed_kmh - ego_speed_kmh,
            motion=motion,
            ego_speed_kmh=ego_speed_kmh
        )
        
        safety_result['rider_action'] = rider_action
        safety_result['lane_info'] = lane_info
        
        return safety_result
    
    def validate_and_assess_rear_scenario(self, detections, ego_speed_kmh=0.0, frame_shape=None):
        """
        Validate rear-view scenario and perform safety assessment for all detections
        
        Args:
            detections: List of detection dictionaries
            ego_speed_kmh: Ego vehicle speed in km/h
            frame_shape: Frame shape for validator initialization
            
        Returns:
            tuple: (assessments_dict, scenario_validation)
        """
        # Update validator with frame dimensions if provided
        if frame_shape is not None:
            h, w = frame_shape[:2]
            self.rear_side_validator.frame_width = w
            self.rear_side_validator.frame_height = h
        
        # Calculate safety assessment for each detection
        safety_assessments = {}
        for det in detections:
            track_id = det.get('track_id', -1)
            if track_id == -1:
                continue
            
            assessment = self.calculate_rear_safety_assessment(det, ego_speed_kmh)
            safety_assessments[track_id] = assessment
            
            # Store in detection for drawing and logging
            det['safety_assessment'] = assessment
        
        # Validate overall rear-view scenario
        ego_speed_ms = ego_speed_kmh / 3.6
        scenario_validation = self.rear_side_validator.validate_rear_scenario(
            detections, ego_speed_ms
        )
        self.last_scenario_validation = scenario_validation
        
        return safety_assessments, scenario_validation
    
    def detect_frame(self, frame):
        """Detect vehicles in a single frame with tracking and dual-depth correction"""
        results = []
        current_boxes_raw = []
        ts_now = time.time()
        
        # Horizon update every 15 frames (Hough transform is expensive)
        if not hasattr(self, '_horizon_frame_ctr'):
            self._horizon_frame_ctr = 0
        self._horizon_frame_ctr += 1
        if self._horizon_frame_ctr % 15 == 0:
            y_horizon, _ = self.horizon_estimator.update(frame)
        else:
            y_horizon = self.horizon_estimator.get_horizon()
        
        # YOLO detection
        yolo_results = self.yolo(frame, imgsz=320, verbose=False)[0]
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
                            cls_results = self.classifier(crop, imgsz=224, verbose=False)
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
        
        # Use ByteTracker for improved tracking (motion-aware, occlusion-robust)
        matches = []
        if self.use_byte_tracker and self.tracker is not None:
            # Prepare detections for ByteTracker [x, y, w, h] format
            tracker_inputs = []
            for i, r in enumerate(results):
                x1, y1, x2, y2 = r['bbox']
                tracker_inputs.append({
                    'bbox': [x1, y1, x2-x1, y2-y1],  # Convert to [x, y, w, h]
                    'confidence': r['confidence'],
                    'class_name': r['class'],
                })
            
            # Update tracker and get tracked results
            tracked_results = self.tracker.update(tracker_inputs)
            
            # Build match list with track IDs from ByteTracker
            # tracked_results contains the matched detections with track_ids
            track_id_list = [t['track_id'] for t in tracked_results]
            
            # Ensure we have a track_id for each result
            # Use the order from tracker which should match the input order
            for i, track_data in enumerate(tracked_results):
                if i < len(results):
                    results[i]['track_id'] = track_data['track_id']
                    matches.append(track_data['track_id'])
            
            # Handle any unmatched detections (shouldn't happen with ByteTracker)
            for i in range(len(tracked_results), len(results)):
                matches.append(-1)
        else:
            # Fallback to IoU-based matching (legacy)
            matches = self.match_detections(current_boxes_raw, self.prev_boxes, iou_threshold=self.IOU_THRESHOLD)

        # --- ML Depth correction (DA2/MIDAS async update) ---
        self._ml_depth_fresh = False
        depth_map = None
        if self.depth_model is not None:
            self.depth_model.request_depth(frame)
            result = self.depth_model.get_depth(wait=False)
            if result is not None:
                depth_map, _ = result
                self.last_zoedepth_depth = depth_map
                self._ml_depth_fresh = True
            elif self.last_zoedepth_depth is not None:
                depth_map = self.last_zoedepth_depth

        # Assign track IDs and estimate motion
        for i, result in enumerate(results):
            # Get track_id from ByteTracker (already assigned) or from matches (legacy)
            if 'track_id' in result:
                track_id = result['track_id']
            else:
                # Legacy IoU-based tracking
                track_id = matches[i] if matches[i] != -1 else self.track_id_counter
                if matches[i] == -1:
                    self.track_id_counter += 1
                result['track_id'] = track_id

            classical_depth, classical_meta = self.estimate_classical_distance(
                bbox=result['bbox'],
                class_name=result['class'],
                track_id=track_id,
                frame_shape=frame.shape,
                timestamp_s=ts_now,
                y_horizon=y_horizon,  # Use dynamic horizon (adapts to suspension)
            )

            ml_depth, ml_conf, ml_method = (None, 0.0, "ml_none")
            corr_c = self.track_correction_factors.get(track_id, self.class_correction_factors.get(result['class'], 1.0))
            c_updated = False

            if depth_map is not None:
                if self._ml_depth_fresh:
                    ml_depth, ml_conf, ml_method = self._sample_ml_depth_for_bbox(depth_map, result['bbox'])
                    corrected_depth, corr_c, c_updated = self._compute_dual_depth(
                        classical_depth=classical_depth,
                        ml_depth=ml_depth,
                        class_name=result['class'],
                        track_id=track_id,
                        ml_fresh=True,
                    )
                else:
                    ml_method = "ml_skip_stale"
                    corrected_depth = (
                        self.classical_depth_weight * float(classical_depth)
                        + (1.0 - self.classical_depth_weight) * float(classical_depth) * float(corr_c)
                    )
            else:
                corrected_depth = classical_depth

            smoothed_depth = self.kalman_filters[track_id].update(corrected_depth) if corrected_depth is not None else None
            result['track_id'] = track_id
            result['distance'] = smoothed_depth
            result['distance_metadata'] = {
                'method': f"dual_depth_{ml_method}",
                'ground': classical_meta.get('ground'),
                'size': classical_meta.get('size'),
                'motion': classical_meta.get('motion'),
                'classical_fused': classical_meta.get('classical_fused', classical_depth),
                'ml': ml_depth,
                'correction_factor': corr_c,
                'alpha': float(self.correction_ema_alpha),
                'correction_updated': c_updated,
                'ml_fresh': self._ml_depth_fresh,
                'confidence': float(ml_conf),
            }
            # Estimate motion
            if smoothed_depth:
                motion, speed = self.estimate_motion(track_id, smoothed_depth)
                result['motion'] = motion
                result['speed'] = speed
            else:
                result['motion'] = "unknown"
                result['speed'] = 0.0
            
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
        active_track_ids = {r['track_id'] for r in results}
        self.track_motion_state = {k: v for k, v in self.track_motion_state.items() if k in active_track_ids}
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
        """
        Draw bounding boxes, labels, distance, motion state, and rear-view safety assessment
        Enhanced with SSM-based safety indicators
        """
        annotated = frame.copy()
        
        # Define alert colors based on safety level
        safety_colors = {
            'CRITICAL': (0, 0, 255),      # Red - Critical danger
            'WARNING': (0, 165, 255),     # Orange - Warning
            'CAUTION': (0, 255, 255),     # Yellow - Caution
            'SAFE': (0, 255, 0),          # Green - Safe
            'unknown': (128, 128, 128)    # Gray - Unknown
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            distance = det.get('distance', None)
            motion = det.get('motion', 'unknown')
            distance_metadata = det.get('distance_metadata', {})
            z_classical = distance_metadata.get('classical_fused', None)
            z_ml = distance_metadata.get('ml', None)
            
            # Get safety assessment
            safety_assess = det.get('safety_assessment', {})
            safety_level = safety_assess.get('level', 'unknown')
            alert_type = safety_assess.get('alert_type', 'none')
            
            # Choose color based on safety level (primary) or motion state (fallback)
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
            
            # Draw box with safety-based color
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3)
            
            # Draw label with distance
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
            
            # Draw distance below box (larger text)
            if distance:
                dist_text = f"{distance:.1f}m"
                dist_size, _ = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(annotated, (x1, y2), 
                             (x1 + dist_size[0] + 10, y2 + dist_size[1] + 10), box_color, -1)
                cv2.putText(annotated, dist_text, (x1 + 5, y2 + dist_size[1] + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw motion state on the right side of box
            if motion != 'unknown':
                speed = det.get('speed', 0.0)
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
            
            # Draw safety level with SSM indicators and LANE-AWARE information
            if safety_level != 'unknown' and alert_type != 'none':
                safety_y = y2 + 40
                
                # Lane-aware message
                lane_info = safety_assess.get('lane_info', {})
                lane_name = lane_info.get('lane', 'UNKNOWN') if lane_info else 'CENTER'
                is_same_lane = safety_assess.get('same_lane', True)
                
                # Build safety text
                if not is_same_lane:
                    # Different lane - only informational
                    safety_text = f"[{lane_name} LANE] {distance:.1f}m away"
                    box_color = (0, 165, 255)  # Orange for adjacent lane
                else:
                    # Same lane - use collision thresholds
                    safety_text = f"[{safety_level}] {alert_type}"
                    
                    # Display key SSM metrics
                    ttc_val = safety_assess.get('ttc')
                    drac_val = safety_assess.get('drac')
                    
                    if ttc_val is not None:
                        safety_text += f" TTC:{ttc_val:.2f}s"
                    if drac_val is not None and drac_val != float('inf'):
                        safety_text += f" DRAC:{drac_val:.2f}m/s²"
                
                safety_size, _ = cv2.getTextSize(safety_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, safety_y - safety_size[1] - 5), 
                             (x1 + safety_size[0] + 10, safety_y + 5), box_color, -1)
                cv2.putText(annotated, safety_text, (x1 + 5, safety_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw RIDER ACTION RECOMMENDATION
                rider_action = safety_assess.get('rider_action', {})
                if rider_action and rider_action.get('action'):
                    action = rider_action.get('action')
                    instruction = rider_action.get('rider_instruction', '')
                    reason = rider_action.get('reason', '')
                    urgency = rider_action.get('urgency', 'LOW')
                    
                    # Color based on urgency
                    urgency_colors = {
                        'CRITICAL': (0, 0, 255),      # Red
                        'HIGH': (0, 165, 255),        # Orange
                        'MEDIUM': (0, 255, 255),      # Yellow
                        'LOW': (0, 255, 0),           # Green
                    }
                    urgency_color = urgency_colors.get(urgency, (128, 128, 128))
                    
                    # Draw action instruction
                    action_y = safety_y + 25
                    action_text = f"→ {instruction[:50]}"  # Truncate long text
                    action_size, _ = cv2.getTextSize(action_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated, (x1, action_y - action_size[1] - 5), 
                                 (x1 + action_size[0] + 10, action_y + 5), urgency_color, -1)
                    cv2.putText(annotated, action_text, (x1 + 5, action_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        
        # Draw FPS if provided
        if fps:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(annotated, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw scenario information if available
        if hasattr(self, 'last_scenario_validation') and self.last_scenario_validation:
            scenario = self.last_scenario_validation
            scenario_text = (
                f"Scenario: {scenario.get('scenario_type', 'unknown')} | "
                f"Threat: {scenario.get('threat_level', 'none')} | "
                f"Critical Vehicles: {scenario.get('critical_vehicles_count', 0)}"
            )
            cv2.putText(annotated, scenario_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return annotated


def process_video(input_path, output_path, device='cuda',
                  zoedepth_interval=30, correction_alpha=0.3,
                  alpha_lr=0.05, freeze_alpha=False):
    """Process video file"""
    
    # Open video
    print(f"📹 Opening: {input_path}")
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"❌ Failed to open video")
        return False
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✅ Video: {width}x{height} @ {video_fps:.2f} FPS, {total_frames} frames")
    
    # Initialize detector
    detector = VideoDetector(
        device=device,
        zoedepth_interval=zoedepth_interval,
        correction_alpha=correction_alpha,
        learnable_alpha=not freeze_alpha,
        alpha_lr=alpha_lr,
        fps=video_fps,
    )
    
    # Setup writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
    
    if not writer.isOpened():
        print(f"❌ Failed to create output video")
        cap.release()
        return False
    
    # Setup CSV logger
    output_path_obj = Path(output_path)
    csv_log_file = output_path_obj.parent / f"{output_path_obj.stem}_detections.csv"
    logger = DetectionLogger(str(csv_log_file))
    print(f"📊 Logging to: {csv_log_file}")
    
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
        
        # Rear-view safety assessment and scenario validation
        safety_assessments, scenario_validation = detector.validate_and_assess_rear_scenario(
            detections, ego_speed_kmh=0.0, frame_shape=frame.shape
        )
        
        # Calculate FPS
        elapsed = time.time() - start_time
        processing_fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Log detections to CSV with safety assessment
        frame_timestamp = frame_count / video_fps
        logger.log_detections(frame_count, detections, frame_timestamp, 
                             safety_assessments, scenario_validation)
        
        # Annotate
        # Use video FPS for display so it matches the video speed, not processing speed
        annotated = detector.draw_detections(frame, detections, video_fps)
        
        # Write
        writer.write(annotated)
        
        # Progress
        if frame_count % 10 == 0 or frame_count == total_frames:
            progress = frame_count / total_frames * 100
            print(f"   Frame {frame_count}/{total_frames} ({progress:.1f}%) - "
                  f"{processing_fps:.1f} FPS", end='\r')
        if frame_count % 30 == 0 and detections:
            d0 = detections[0]
            md = d0.get('distance_metadata', {})
            safety_assess = d0.get('safety_assessment', {})
            print(
                f"\n[Frame {frame_count}] {d0.get('class','?')} "
                f"D={d0.get('distance',0):.2f}m "
                f"C={md.get('classical_fused', None)} "
                f"ML={md.get('ml', None)} "
                f"k={md.get('correction_factor',1.0):.3f} "
                f"alpha={md.get('alpha',0.0):.3f}"
            )
    
    print()
    
    # Cleanup
    cap.release()
    writer.release()
    logger.close()
    
    # Stats
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed
    
    print(f"\n✅ Processing complete!")
    print(f"   Frames: {frame_count}")
    print(f"   Time: {elapsed:.1f}s")
    print(f"   Avg FPS: {avg_fps:.1f}")
    print(f"   Video Output: {output_path}")
    
    # Verify video output
    output_file = Path(output_path)
    if output_file.exists():
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"   Video Size: {size_mb:.2f} MB")
    else:
        print(f"❌ Video file not created")
        return False
    
    # Verify CSV output
    if csv_log_file.exists():
        csv_size_kb = csv_log_file.stat().st_size / 1024
        print(f"\n   CSV Detections: {csv_log_file}")
        print(f"   CSV Size: {csv_size_kb:.2f} KB")
        print(f"   Total Detections Logged: {frame_count} frames")
    else:
        print(f"❌ CSV file not created")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Video Detection Processor')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input video file')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output video file')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    parser.add_argument('--zoedepth-interval', type=int, default=30,
                       help='Run ML depth update every N frames (default: 30)')
    parser.add_argument('--correction-alpha', type=float, default=0.3,
                       help='Initial correction EMA alpha (default: 0.3)')
    parser.add_argument('--alpha-lr', type=float, default=0.05,
                       help='Online alpha learning rate (default: 0.05)')
    parser.add_argument('--freeze-alpha', action='store_true',
                       help='Keep alpha fixed (disable online adaptation)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🎬 VIDEO DETECTION PROCESSOR")
    print("="*60 + "\n")
    
    success = process_video(
        args.input,
        args.output,
        args.device,
        zoedepth_interval=args.zoedepth_interval,
        correction_alpha=args.correction_alpha,
        alpha_lr=args.alpha_lr,
        freeze_alpha=args.freeze_alpha,
    )
    
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
