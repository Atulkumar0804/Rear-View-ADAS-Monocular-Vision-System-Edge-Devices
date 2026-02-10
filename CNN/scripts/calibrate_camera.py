#!/usr/bin/env python3
"""
Camera Calibration Script using Chessboard Pattern
Usage:
    python scripts/calibrate_camera.py --square_size 0.055 --width 9 --height 6

    - Press 'c' to capture a frame (if corners are found).
    - Press 'q' to finish capturing and run calibration.
"""

import cv2
import numpy as np
import argparse
import glob
import os
import json
from datetime import datetime

def calibrate_camera(square_size, width, height, camera_source=0, save_dir="calibration_data"):
    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:,:2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_source}")
        print("Try specifying a different camera index using --camera <index>")
        print("Available devices likely include: /dev/video0, /dev/video2, /dev/video4 etc.")
        return

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Starting calibration with board size {width}x{height} and square size {square_size}m")
    print("Press 'c' to capture a valid frame.")
    print("Press 'q' to finish and calibrate.")

    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret_corners, corners = cv2.findChessboardCorners(gray, (width, height), None)

        display_frame = frame.copy()

        # If found, add object points, image points (after refining them)
        if ret_corners:
            cv2.drawChessboardCorners(display_frame, (width, height), corners, ret_corners)
            cv2.putText(display_frame, "Corners Found! Press 'c' to capture", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Searching for chessboard...", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Camera Calibration', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if ret_corners:
                objpoints.append(objp)
                
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                
                count += 1
                print(f"Captured image {count}")
                
                # Save the image for reference
                img_name = os.path.join(save_dir, f"calib_{count}.jpg")
                cv2.imwrite(img_name, frame)
                
                # Flash effect
                cv2.imshow('Camera Calibration', np.ones_like(frame)*255)
                cv2.waitKey(50)
            else:
                print("Corners not found, cannot capture.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if count > 0:
        print(f"\nCalibrating with {count} images...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        print("\nCalibration successful!")
        print(f"Reprojection Error: {ret}")
        print("\nCamera Matrix:\n", mtx)
        
        fx = mtx[0,0]
        fy = mtx[1,1]
        print(f"\n✅ Estimated Focal Length (pixels): fx={fx:.2f}, fy={fy:.2f}")
        print(f"   Use the average ({ (fx+fy)/2:.2f} ) in your inference scripts.")
        
        print("\nDistortion Coefficients:\n", dist)

        # Save results
        calib_data = {
            "camera_matrix": mtx.tolist(),
            "dist_coeff": dist.tolist(),
            "reprojection_error": ret,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        json_path = os.path.join(save_dir, "calibration_result.json")
        with open(json_path, "w") as f:
            json.dump(calib_data, f, indent=4)
            
        np_path = os.path.join(save_dir, "calibration_matrix.npy")
        np.save(np_path, mtx)
        
        dist_path = os.path.join(save_dir, "distortion_coefficients.npy")
        np.save(dist_path, dist)

        print(f"\nResults saved to {save_dir}")
    else:
        print("No images captured. Calibration aborted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Camera Calibration')
    parser.add_argument('--square_size', type=float, default=0.055, help='Size of one square in meters')
    parser.add_argument('--width', type=int, default=9, help='Number of inner corners along width')
    parser.add_argument('--height', type=int, default=6, help='Number of inner corners along height')
    parser.add_argument('--camera', type=int, default=0, help='Camera source index')
    
    args = parser.parse_args()
    
    calibrate_camera(args.square_size, args.width, args.height, args.camera)
