#!/usr/bin/env python3
"""
Estimate Focal Length from a Single Image of a Checkerboard at a Known Distance.

Usage:
    python scripts/estimate_focal_length.py --image path/to/image.jpg --distance 1.0 --square_size 0.055 --width 4 --height 6

    - distance: Distance from camera to pattern in meters.
    - square_size: Size of one square in meters.
    - width: Number of inner corners horizontally.
    - height: Number of inner corners vertically.
"""

import cv2
import numpy as np
import argparse
import os

def estimate_focal_length(image_path, distance, square_size, width, height):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Could not load image.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    print(f"Analyzing image: {image_path}")
    print(f"Pattern: {width}x{height} inner corners")
    print(f"Square size: {square_size} m")
    print(f"Known Distance: {distance} m")

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

    if ret:
        # Refine corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Draw and display (optional, saves to file)
        debug_img = frame.copy()
        cv2.drawChessboardCorners(debug_img, (width, height), corners, ret)
        cv2.imwrite("focal_length_debug.jpg", debug_img)
        print("Saved debug image to focal_length_debug.jpg")

        # Calculate Focal Length
        # We can use the width of the board in pixels vs real meters.
        
        # Get top-left and top-right corners (assuming standard ordering)
        # Corners are usually returned row by row.
        # Top-left: index 0
        # Top-right: index width-1
        # Bottom-left: index (height-1)*width
        # Bottom-right: index width*height - 1
        
        # Let's use the widest span to be robust (e.g. top-left to top-right)
        # Real width of the inner corner grid = (width - 1) * square_size? 
        # No, 'width' is number of points.
        # The distance between point 0 and point (width-1) is (width-1) squares.
        
        # Let's verify:
        # If we have 4 corners: C1 - C2 - C3 - C4
        # There are 3 intervals (squares).
        # So real distance = (width - 1) * square_size
        
        p1 = corners[0][0]
        p2 = corners[width-1][0]
        
        pixel_dist_x = np.linalg.norm(p1 - p2)
        real_dist_x = (width - 1) * square_size
        
        focal_x = (pixel_dist_x * distance) / real_dist_x
        
        # Let's do vertical too
        p3 = corners[0][0]
        p4 = corners[(height-1)*width][0] # Bottom-left of the grid
        
        pixel_dist_y = np.linalg.norm(p3 - p4)
        real_dist_y = (height - 1) * square_size
        
        focal_y = (pixel_dist_y * distance) / real_dist_y
        
        focal_avg = (focal_x + focal_y) / 2
        
        print("\n--- Results ---")
        print(f"Detected Board Pixel Width: {pixel_dist_x:.2f} px")
        print(f"Real Board Width (Inner): {real_dist_x:.4f} m")
        print(f"Estimated Focal Length (X): {focal_x:.2f} pixels")
        
        print(f"Detected Board Pixel Height: {pixel_dist_y:.2f} px")
        print(f"Real Board Height (Inner): {real_dist_y:.4f} m")
        print(f"Estimated Focal Length (Y): {focal_y:.2f} pixels")
        
        print(f"\n✅ Recommended Focal Length: {focal_avg:.2f} pixels")
        
        # Calculate Field of View (FOV) assuming image size
        h, w = frame.shape[:2]
        fov_x = 2 * np.arctan(w / (2 * focal_avg)) * 180 / np.pi
        fov_y = 2 * np.arctan(h / (2 * focal_avg)) * 180 / np.pi
        print(f"Estimated FOV: {fov_x:.1f}° (H) x {fov_y:.1f}° (V)")
        
    else:
        print("❌ Checkerboard corners not found. Please check:")
        print("1. The --width and --height parameters match the INNER corners (not squares).")
        print("2. The image is clear and the board is fully visible.")
        print("3. Lighting is sufficient.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estimate Focal Length')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--distance', type=float, default=1.0, help='Distance to pattern in meters')
    parser.add_argument('--square_size', type=float, default=0.055, help='Size of one square in meters')
    parser.add_argument('--width', type=int, default=4, help='Number of inner corners along width')
    parser.add_argument('--height', type=int, default=6, help='Number of inner corners along height')
    
    args = parser.parse_args()
    
    estimate_focal_length(args.image, args.distance, args.square_size, args.width, args.height)
