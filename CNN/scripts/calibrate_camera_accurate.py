#!/usr/bin/env python3
"""
Enhanced Camera Calibration with Checkerboard Pattern
======================================================

Advanced calibration for accurate depth measurement:
- Intrinsic calibration (focal length, principal point, distortion)
- Extrinsic calibration (camera pose)
- Focal length optimization for depth estimation
- Save calibration parameters for inference

Usage:
    python scripts/calibrate_camera_accurate.py --camera 4 --square_size 0.025 --width 9 --height 6
    
    Press 'c' to capture a frame with detected corners
    Press 'q' to finish and run calibration
    Capture at least 20 frames from different angles and distances

Author: ADAS Research Team  
Date: 2026-02-02
"""

import cv2
import numpy as np
import argparse
import os
import json
from datetime import datetime
from pathlib import Path


def calibrate_camera_accurate(camera_source=0, square_size=0.025, width=9, height=6, 
                               save_dir="calibration_data", min_captures=20):
    """
    Perform accurate camera calibration using checkerboard pattern
    
    Args:
        camera_source: Camera index (0, 2, 4, etc.)
        square_size: Size of checkerboard square in meters (e.g., 25mm = 0.025m)
        width: Number of inner corners along width
        height: Number of inner corners along height
        save_dir: Directory to save calibration results
        min_captures: Minimum number of captures recommended
    """
    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare 3D object points
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp = objp * square_size  # Convert to real-world coordinates
    
    # Storage for calibration data
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    captured_frames = []
    
    # Open camera
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {camera_source}")
        print("Available devices: /dev/video0, /dev/video2, /dev/video4, etc.")
        return None
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 70)
    print("📷 ENHANCED CAMERA CALIBRATION")
    print("=" * 70)
    print(f"Camera: {camera_source}")
    print(f"Checkerboard: {width}x{height} corners")
    print(f"Square size: {square_size * 1000:.1f}mm")
    print(f"Minimum captures: {min_captures}")
    print()
    print("INSTRUCTIONS:")
    print("  1. Move checkerboard to different positions and angles")
    print("  2. Keep checkerboard flat and fully visible")
    print("  3. Vary distances (close, medium, far)")
    print("  4. Include corners and center of frame")
    print()
    print("CONTROLS:")
    print("  'c' - Capture frame (when corners detected)")
    print("  'q' - Finish and calibrate")
    print("=" * 70)
    
    capture_count = 0
    
    # Add flags for better corner detection
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply image preprocessing for better detection
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Find checkerboard corners with adaptive flags
        found, corners = cv2.findChessboardCorners(gray, (width, height), flags)
        
        # Display frame
        display_frame = frame.copy()
        
        if found:
            # Refine corner locations
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Draw corners
            cv2.drawChessboardCorners(display_frame, (width, height), corners_refined, found)
            
            # Show status
            cv2.putText(display_frame, "✓ Corners detected - Press 'c' to capture", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "✗ No corners - Adjust checkerboard position", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show helpful tips
            cv2.putText(display_frame, "Tips: Good lighting, flat board, fill 40-60% of frame", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show capture count
        cv2.putText(display_frame, f"Captures: {capture_count}/{min_captures}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Show expected pattern
        cv2.putText(display_frame, f"Looking for: {width}x{height} corners ({square_size*1000:.0f}mm squares)", 
                   (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Camera Calibration', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c') and found:
            # Capture frame
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            captured_frames.append(frame.copy())
            capture_count += 1
            
            # Save capture
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_dir, f"capture_{capture_count:03d}_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            
            print(f"✅ Captured {capture_count}/{min_captures}: {filename}")
            
        elif key == ord('q'):
            if capture_count < 10:
                print(f"⚠️ Warning: Only {capture_count} captures. Recommended: {min_captures}+")
                print("Press 'q' again to proceed anyway, or continue capturing...")
                if cv2.waitKey(3000) & 0xFF == ord('q'):
                    break
                else:
                    continue
            else:
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if capture_count < 5:
        print(f"❌ Insufficient captures ({capture_count}). Need at least 5.")
        return None
    
    print()
    print("=" * 70)
    print("🔧 RUNNING CALIBRATION...")
    print("=" * 70)
    
    # Get image size
    img_size = gray.shape[::-1]
    
    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )
    
    if not ret:
        print("❌ Calibration failed")
        return None
    
    # Calculate reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints_reprojected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                                      camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints_reprojected, cv2.NORM_L2) / len(imgpoints_reprojected)
        total_error += error
    
    mean_error = total_error / len(objpoints)
    
    # Extract focal lengths
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    print()
    print("=" * 70)
    print("✅ CALIBRATION RESULTS")
    print("=" * 70)
    print(f"Captures used: {capture_count}")
    print(f"Reprojection error: {mean_error:.4f} pixels")
    print()
    print("Camera Matrix:")
    print(f"  fx (focal length x): {fx:.2f} pixels")
    print(f"  fy (focal length y): {fy:.2f} pixels")
    print(f"  cx (principal point x): {cx:.2f} pixels")
    print(f"  cy (principal point y): {cy:.2f} pixels")
    print()
    print("Distortion Coefficients:")
    print(f"  k1: {dist_coeffs[0][0]:.6f}")
    print(f"  k2: {dist_coeffs[0][1]:.6f}")
    print(f"  p1: {dist_coeffs[0][2]:.6f}")
    print(f"  p2: {dist_coeffs[0][3]:.6f}")
    print(f"  k3: {dist_coeffs[0][4]:.6f}")
    print("=" * 70)
    
    # Save results
    results = {
        'camera_matrix': camera_matrix.tolist(),
        'distortion_coefficients': dist_coeffs.tolist(),
        'focal_length_x': float(fx),
        'focal_length_y': float(fy),
        'focal_length_avg': float((fx + fy) / 2),
        'principal_point_x': float(cx),
        'principal_point_y': float(cy),
        'reprojection_error': float(mean_error),
        'image_size': list(img_size),
        'num_captures': capture_count,
        'square_size': square_size,
        'board_size': [width, height],
        'calibration_date': datetime.now().isoformat()
    }
    
    # Save JSON
    json_path = os.path.join(save_dir, 'calibration_accurate.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Saved calibration: {json_path}")
    
    # Save numpy arrays
    np.save(os.path.join(save_dir, 'camera_matrix_accurate.npy'), camera_matrix)
    np.save(os.path.join(save_dir, 'dist_coeffs_accurate.npy'), dist_coeffs)
    print(f"💾 Saved arrays: camera_matrix_accurate.npy, dist_coeffs_accurate.npy")
    
    # Quality assessment
    print()
    print("📊 QUALITY ASSESSMENT:")
    if mean_error < 0.5:
        print("  ✅ EXCELLENT (error < 0.5 pixels)")
    elif mean_error < 1.0:
        print("  ✅ GOOD (error < 1.0 pixels)")
    elif mean_error < 2.0:
        print("  ⚠️ ACCEPTABLE (error < 2.0 pixels)")
    else:
        print("  ❌ POOR (error >= 2.0 pixels) - Consider recalibrating")
    
    if capture_count >= min_captures:
        print(f"  ✅ Sufficient captures ({capture_count} >= {min_captures})")
    else:
        print(f"  ⚠️ Few captures ({capture_count} < {min_captures}) - More is better")
    
    print()
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced Camera Calibration')
    parser.add_argument('--camera', type=int, default=4, help='Camera index (default: 4)')
    parser.add_argument('--square_size', type=float, default=0.055, 
                       help='Checkerboard square size in meters (default: 0.055 = 55mm)')
    parser.add_argument('--width', type=int, default=7, 
                       help='Number of inner corners along width (default: 7)')
    parser.add_argument('--height', type=int, default=5, 
                       help='Number of inner corners along height (default: 5)')
    parser.add_argument('--min_captures', type=int, default=20,
                       help='Minimum recommended captures (default: 20)')
    parser.add_argument('--save_dir', type=str, default='calibration_data',
                       help='Directory to save results (default: calibration_data)')
    
    args = parser.parse_args()
    
    calibrate_camera_accurate(
        camera_source=args.camera,
        square_size=args.square_size,
        width=args.width,
        height=args.height,
        save_dir=args.save_dir,
        min_captures=args.min_captures
    )
