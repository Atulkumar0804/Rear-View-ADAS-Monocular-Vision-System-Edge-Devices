#!/usr/bin/env python3
"""
Simple Distance Dataset Preparation
====================================

Organizes distance-labeled images for easy model training.

Directory structure created:
    distance_dataset/
        2m/
            image_001.jpg
            image_002.jpg
        5m/
            image_001.jpg
        10m/
            image_001.jpg
        metadata.csv  (image_path, distance_meters, pixel_height, focal_length)

Usage:
    python3 scripts/prepare_distance_dataset.py

Author: ADAS Research Team
Date: 2026-02-02
"""

import cv2
import numpy as np
import json
import shutil
import csv
from pathlib import Path
from datetime import datetime


class DistanceDatasetPreparer:
    """
    Prepare distance dataset for model training
    """
    
    def __init__(self, source_dir='calibration_data/distance_samples', 
                 output_dir='distance_dataset'):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def organize_by_distance(self):
        """
        Organize images into distance-based folders
        """
        print("=" * 70)
        print("DISTANCE DATASET PREPARATION")
        print("=" * 70)
        
        # Load calibration data
        calib_file = self.source_dir / 'distance_calibration.json'
        if not calib_file.exists():
            print(f"❌ No calibration data found at: {calib_file}")
            print("   Run: python3 scripts/calibrate_distance_interactive.py")
            return
        
        with open(calib_file, 'r') as f:
            data = json.load(f)
        
        samples = data['samples']
        print(f"\n📂 Found {len(samples)} calibration samples")
        
        # Create metadata
        metadata = []
        
        # Group by distance (rounded to nearest meter)
        distance_groups = {}
        for sample in samples:
            distance = sample['distance_meters']
            distance_rounded = round(distance)
            
            if distance_rounded not in distance_groups:
                distance_groups[distance_rounded] = []
            distance_groups[distance_rounded].append(sample)
        
        print(f"\n📊 Distance distribution:")
        for dist in sorted(distance_groups.keys()):
            count = len(distance_groups[dist])
            print(f"   {dist}m: {count} images")
        
        # Copy images to organized structure
        print("\n📁 Organizing images...")
        
        for dist, samples_list in distance_groups.items():
            # Create distance folder
            dist_folder = self.output_dir / f"{dist}m"
            dist_folder.mkdir(exist_ok=True)
            
            for idx, sample in enumerate(samples_list, 1):
                # Copy image
                src_img = self.source_dir / sample['image_file']
                if src_img.exists():
                    dst_img = dist_folder / f"image_{idx:03d}.jpg"
                    shutil.copy2(src_img, dst_img)
                    
                    # Add to metadata
                    metadata.append({
                        'image_path': str(dst_img.relative_to(self.output_dir)),
                        'distance_meters': sample['distance_meters'],
                        'distance_rounded': dist,
                        'pixel_height': sample['pixel_height'],
                        'focal_length_estimate': sample.get('focal_length_estimate', 0),
                        'timestamp': sample['timestamp']
                    })
                    
                    print(f"   ✅ {dst_img.relative_to(self.output_dir)}")
        
        # Save metadata CSV
        csv_path = self.output_dir / 'metadata.csv'
        with open(csv_path, 'w', newline='') as f:
            if metadata:
                writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
                writer.writeheader()
                writer.writerows(metadata)
        
        print(f"\n💾 Saved metadata: {csv_path}")
        
        # Save simplified training file (image, distance)
        simple_csv = self.output_dir / 'training_data.csv'
        with open(simple_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'distance_meters', 'pixel_height'])
            for item in metadata:
                writer.writerow([
                    item['image_path'],
                    item['distance_meters'],
                    item['pixel_height']
                ])
        
        print(f"💾 Saved training data: {simple_csv}")
        
        # Create summary
        self.create_summary(metadata, distance_groups)
        
        print("\n" + "=" * 70)
        print("✅ DATASET PREPARATION COMPLETE")
        print("=" * 70)
        print(f"\nDataset location: {self.output_dir.absolute()}")
        print(f"Total images: {len(metadata)}")
        print(f"Distance categories: {len(distance_groups)}")
        print("\nFiles created:")
        print(f"  • metadata.csv - Complete metadata")
        print(f"  • training_data.csv - Simple (image, distance, pixel_height)")
        print(f"  • dataset_summary.txt - Overview")
        print(f"  • {len(distance_groups)} distance folders (2m/, 5m/, etc.)")
        
    def create_summary(self, metadata, distance_groups):
        """Create dataset summary text file"""
        summary_path = self.output_dir / 'dataset_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("DISTANCE DATASET SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Created: {datetime.now().isoformat()}\n")
            f.write(f"Total images: {len(metadata)}\n")
            f.write(f"Distance categories: {len(distance_groups)}\n\n")
            
            f.write("DISTANCE DISTRIBUTION:\n")
            f.write("-" * 70 + "\n")
            for dist in sorted(distance_groups.keys()):
                count = len(distance_groups[dist])
                f.write(f"{dist}m: {count} images\n")
            
            f.write("\n" + "=" * 70 + "\n\n")
            
            f.write("STATISTICS:\n")
            f.write("-" * 70 + "\n")
            distances = [m['distance_meters'] for m in metadata]
            pixel_heights = [m['pixel_height'] for m in metadata]
            focal_lengths = [m['focal_length_estimate'] for m in metadata]
            
            f.write(f"Distance range: {min(distances):.1f}m - {max(distances):.1f}m\n")
            f.write(f"Average distance: {np.mean(distances):.1f}m\n")
            f.write(f"Pixel height range: {min(pixel_heights):.1f}px - {max(pixel_heights):.1f}px\n")
            f.write(f"Average focal length: {np.mean(focal_lengths):.1f}px\n")
            
            f.write("\n" + "=" * 70 + "\n\n")
            
            f.write("USAGE:\n")
            f.write("-" * 70 + "\n")
            f.write("1. Training CSV: training_data.csv\n")
            f.write("   Format: image_path, distance_meters, pixel_height\n\n")
            f.write("2. Full metadata: metadata.csv\n")
            f.write("   Includes: focal_length_estimate, timestamp, etc.\n\n")
            f.write("3. Images organized by distance:\n")
            for dist in sorted(distance_groups.keys()):
                f.write(f"   - {dist}m/ ({len(distance_groups[dist])} images)\n")
        
        print(f"💾 Saved summary: {summary_path}")
    
    def create_visualization(self):
        """Create visualization of dataset"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️ Matplotlib not available, skipping visualization")
            return
        
        # Load metadata
        csv_path = self.output_dir / 'metadata.csv'
        if not csv_path.exists():
            return
        
        data = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        distances = [float(d['distance_meters']) for d in data]
        pixel_heights = [float(d['pixel_height']) for d in data]
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Distance distribution
        ax1.hist(distances, bins=20, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Distance (meters)')
        ax1.set_ylabel('Number of Images')
        ax1.set_title('Distance Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Distance vs Pixel Height
        ax2.scatter(distances, pixel_heights, alpha=0.6, s=100)
        ax2.set_xlabel('Distance (meters)')
        ax2.set_ylabel('Pixel Height')
        ax2.set_title('Distance vs Pixel Height')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        viz_path = self.output_dir / 'dataset_visualization.png'
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved visualization: {viz_path}")
        plt.close()


def main():
    preparer = DistanceDatasetPreparer()
    preparer.organize_by_distance()
    preparer.create_visualization()


if __name__ == "__main__":
    main()
