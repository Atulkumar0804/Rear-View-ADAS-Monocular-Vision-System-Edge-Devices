#!/usr/bin/env python3
"""
Extract Camera Intrinsics from Checkerboard Images
===================================================

Analyze saved checkerboard images to calculate camera intrinsic parameters.

Usage:
    python3 scripts/extract_camera_intrinsics.py --images "checkerboard_*.jpg"

Author: ADAS Research Team
Date: 2026-02-02
"""

import cv2
import numpy as np
import glob
import json
import argparse
from pathlib import Path
from datetime import datetime


def extract_intrinsics(image_pattern, width=7, height=5, square_size=0.055, 
                       output_dir='calibration_data'):
    """
    Extract camera intrinsics from checkerboard images
    
    Args:
        image_pattern: Glob pattern for images (e.g., "checkerboard_*.jpg")
        width: Number of inner corners along width
        height: Number of inner corners along height
        square_size: Size of squares in meters
        output_dir: Where to save results
    """
    print("=" * 70)
    print("CAMERA INTRINSICS EXTRACTION FROM IMAGES")
    print("=" * 70)
    print(f"Pattern: {width}x{height} inner corners")
    print(f"Square size: {square_size * 1000:.1f}mm")
    print()
    
    # Find images
    image_files = sorted(glob.glob(image_pattern))
    
    if not image_files:
        print(f"❌ No images found matching: {image_pattern}")
        print("\nTry:")
        print("  python3 scripts/extract_camera_intrinsics.py --images 'checkerboard*.jpg'")
        print("  python3 scripts/extract_camera_intrinsics.py --images '*.jpg'")
        print("  python3 scripts/extract_camera_intrinsics.py --images 'calibration_data/*.jpg'")
        return None
    
    print(f"📂 Found {len(image_files)} images:")
    for img_file in image_files[:10]:  # Show first 10
        print(f"   • {img_file}")
    if len(image_files) > 10:
        print(f"   ... and {len(image_files) - 10} more")
    print()
    
    # Prepare object points
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp = objp * square_size
    
    # Arrays to store points
    objpoints = []  # 3D points in real world
    imgpoints = []  # 2D points in image plane
    successful_images = []
    
    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Detection flags
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    
    print("🔍 Processing images...")
    
    for i, img_file in enumerate(image_files):
        img = cv2.imread(img_file)
        if img is None:
            print(f"   ⚠️ Could not read: {img_file}")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Find checkerboard corners
        found, corners = cv2.findChessboardCorners(enhanced, (width, height), flags)
        
        if found:
            # Refine corner positions
            corners_refined = cv2.cornerSubPix(enhanced, corners, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            successful_images.append(img_file)
            
            if (i + 1) % 10 == 0 or i < 5:
                print(f"   ✅ {Path(img_file).name}")
        else:
            if i < 5:
                print(f"   ❌ {Path(img_file).name} - No corners detected")
    
    print()
    
    if len(successful_images) < 3:
        print(f"❌ Not enough images with detected corners ({len(successful_images)}/3 minimum)")
        print("\nTips:")
        print("  • Ensure checkerboard is clearly visible")
        print("  • Good lighting, no shadows")
        print("  • All corners must be visible")
        print("  • Try different --width and --height values")
        return None
    
    print(f"✅ Successfully detected corners in {len(successful_images)}/{len(image_files)} images")
    print()
    
    # Get image size from first successful image
    img = cv2.imread(successful_images[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_shape = gray.shape[::-1]
    
    # Calibrate camera
    print("🔧 Running camera calibration...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )
    
    if not ret:
        print("❌ Calibration failed")
        return None
    
    # Calculate reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints_reprojected, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(imgpoints[i], imgpoints_reprojected, cv2.NORM_L2) / len(imgpoints_reprojected)
        total_error += error
    
    mean_error = total_error / len(objpoints)
    
    # Extract intrinsic parameters
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    focal_avg = (fx + fy) / 2.0
    
    # Display results
    print()
    print("=" * 70)
    print("✅ CAMERA INTRINSIC PARAMETERS")
    print("=" * 70)
    print(f"Images used: {len(successful_images)}")
    print(f"Image size: {img_shape[0]}x{img_shape[1]}")
    print(f"Reprojection error: {mean_error:.4f} pixels")
    print()
    print("Camera Matrix:")
    print(f"  fx (focal length x): {fx:.2f} pixels")
    print(f"  fy (focal length y): {fy:.2f} pixels")
    print(f"  cx (principal point x): {cx:.2f} pixels")
    print(f"  cy (principal point y): {cy:.2f} pixels")
    print(f"  Focal length (average): {focal_avg:.2f} pixels")
    print()
    print("Distortion Coefficients:")
    print(f"  k1: {dist_coeffs[0][0]:.6f}")
    print(f"  k2: {dist_coeffs[0][1]:.6f}")
    print(f"  p1: {dist_coeffs[0][2]:.6f}")
    print(f"  p2: {dist_coeffs[0][3]:.6f}")
    print(f"  k3: {dist_coeffs[0][4]:.6f}")
    print()
    
    # Quality assessment
    print("Quality Assessment:")
    if mean_error < 0.5:
        print("  ✅ EXCELLENT (error < 0.5 pixels)")
    elif mean_error < 1.0:
        print("  ✅ GOOD (error < 1.0 pixels)")
    elif mean_error < 2.0:
        print("  ⚠️ ACCEPTABLE (error < 2.0 pixels)")
    else:
        print("  ❌ POOR (error >= 2.0 pixels)")
        print("  Consider capturing more images with better lighting/angles")
    
    print("=" * 70)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    results = {
        'camera_matrix': camera_matrix.tolist(),
        'distortion_coefficients': dist_coeffs.tolist(),
        'focal_length_x': float(fx),
        'focal_length_y': float(fy),
        'focal_length_avg': float(focal_avg),
        'principal_point_x': float(cx),
        'principal_point_y': float(cy),
        'reprojection_error': float(mean_error),
        'image_size': list(img_shape),
        'num_images': len(successful_images),
        'square_size': square_size,
        'board_size': [width, height],
        'calibration_date': datetime.now().isoformat(),
        'images_used': successful_images[:20]  # Save first 20 to avoid huge file
    }
    
    json_path = output_path / 'camera_intrinsics_from_images.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved calibration: {json_path}")
    
    # Save numpy arrays
    np.save(output_path / 'camera_matrix_from_images.npy', camera_matrix)
    np.save(output_path / 'dist_coeffs_from_images.npy', dist_coeffs)
    print(f"💾 Saved arrays: camera_matrix_from_images.npy, dist_coeffs_from_images.npy")
    
    # Save readable text summary
    txt_path = output_path / 'camera_intrinsics_summary.txt'
    with open(txt_path, 'w') as f:
        f.write("CAMERA INTRINSIC PARAMETERS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Calibration Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Images Used: {len(successful_images)}\n")
        f.write(f"Image Size: {img_shape[0]}x{img_shape[1]}\n")
        f.write(f"Reprojection Error: {mean_error:.4f} pixels\n\n")
        
        f.write("FOCAL LENGTH:\n")
        f.write(f"  fx: {fx:.2f} pixels\n")
        f.write(f"  fy: {fy:.2f} pixels\n")
        f.write(f"  Average: {focal_avg:.2f} pixels\n\n")
        
        f.write("PRINCIPAL POINT:\n")
        f.write(f"  cx: {cx:.2f} pixels\n")
        f.write(f"  cy: {cy:.2f} pixels\n\n")
        
        f.write("DISTORTION COEFFICIENTS:\n")
        f.write(f"  k1: {dist_coeffs[0][0]:.6f}\n")
        f.write(f"  k2: {dist_coeffs[0][1]:.6f}\n")
        f.write(f"  p1: {dist_coeffs[0][2]:.6f}\n")
        f.write(f"  p2: {dist_coeffs[0][3]:.6f}\n")
        f.write(f"  k3: {dist_coeffs[0][4]:.6f}\n\n")
        
        f.write("CAMERA MATRIX:\n")
        f.write(f"  [{fx:.2f}    0.00    {cx:.2f}]\n")
        f.write(f"  [  0.00  {fy:.2f}    {cy:.2f}]\n")
        f.write(f"  [  0.00    0.00      1.00]\n\n")
        
        f.write("FIRST 10 IMAGES USED:\n")
        for img_file in successful_images[:10]:
            f.write(f"  • {img_file}\n")
        if len(successful_images) > 10:
            f.write(f"  ... and {len(successful_images) - 10} more\n")
    
    print(f"💾 Saved summary: {txt_path}")
    
    print()
    print("=" * 70)
    print(f"🎯 USE THIS FOCAL LENGTH: {focal_avg:.1f} pixels")
    print("=" * 70)
    print()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Extract camera intrinsics from checkerboard images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Current directory images
  python3 scripts/extract_camera_intrinsics.py --images "checkerboard*.jpg"
  
  # Specific directory
  python3 scripts/extract_camera_intrinsics.py --images "calibration_data/*.jpg"
  
  # All JPG files
  python3 scripts/extract_camera_intrinsics.py --images "*.jpg"
  
  # Custom board size (9x6 board, 25mm squares)
  python3 scripts/extract_camera_intrinsics.py --images "*.jpg" --width 9 --height 6 --square_size 0.025
        """
    )
    
    parser.add_argument('--images', type=str, required=True,
                       help='Glob pattern for checkerboard images (e.g., "checkerboard*.jpg")')
    parser.add_argument('--width', type=int, default=7,
                       help='Number of inner corners along width (default: 7)')
    parser.add_argument('--height', type=int, default=5,
                       help='Number of inner corners along height (default: 5)')
    parser.add_argument('--square_size', type=float, default=0.055,
                       help='Checkerboard square size in meters (default: 0.055 = 55mm)')
    parser.add_argument('--output_dir', type=str, default='calibration_data',
                       help='Output directory for calibration files (default: calibration_data)')
    
    args = parser.parse_args()
    
    extract_intrinsics(
        image_pattern=args.images,
        width=args.width,
        height=args.height,
        square_size=args.square_size,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
