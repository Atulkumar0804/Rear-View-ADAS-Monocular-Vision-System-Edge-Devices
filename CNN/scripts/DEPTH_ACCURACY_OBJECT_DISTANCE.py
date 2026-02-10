#!/usr/bin/env python3
"""
Object Distance Accuracy Evaluation

Evaluates the accuracy of distance estimation for detected objects
(vehicles, pedestrians, cyclists) using monocular depth and object detection.
"""

import os
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ObjectDistanceEvaluator:
    """Evaluates object distance accuracy"""
    
    # KITTI object classes
    CLASSES = ['car', 'pedestrian', 'cyclist', 'truck', 'misc']
    
    def __init__(self, output_dir: str = "results/object_distance"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = defaultdict(list)
    
    def extract_ground_truth_boxes(self, label_file: str) -> List[Dict]:
        """
        Extract KITTI ground truth bounding boxes
        
        Format: type truncated occluded alpha bbox_left bbox_top bbox_right bbox_bottom
                height width length x y z rotation_y [score]
        """
        boxes = []
        
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 15:
                        continue
                    
                    obj_type = parts[0]
                    if obj_type not in self.CLASSES:
                        continue
                    
                    bbox = {
                        'class': obj_type,
                        'truncated': float(parts[1]),
                        'occluded': int(parts[2]),
                        'alpha': float(parts[3]),
                        'bbox': [float(parts[4]), float(parts[5]), 
                                float(parts[6]), float(parts[7])],
                        'dimensions': [float(parts[8]), float(parts[9]), float(parts[10])],
                        'location': [float(parts[11]), float(parts[12]), float(parts[13])],
                        'rotation_y': float(parts[14])
                    }
                    
                    # Ground truth depth in camera frame
                    bbox['gt_distance'] = bbox['location'][2]  # Z coordinate in camera frame
                    boxes.append(bbox)
        
        except Exception as e:
            logger.warning(f"Error reading {label_file}: {e}")
        
        return boxes
    
    def estimate_bbox_distance(self, pred_depth: np.ndarray, bbox: List[float],
                              camera_matrix: np.ndarray) -> float:
        """
        Estimate distance to object using bounding box and depth map
        
        Args:
            pred_depth: Predicted depth map
            bbox: [x1, y1, x2, y2] bounding box coordinates
            camera_matrix: Camera intrinsics (3x3)
        
        Returns:
            Estimated distance in meters
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Clamp to image bounds
        h, w = pred_depth.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        
        # Extract depth in bounding box
        region_depth = pred_depth[y1:y2+1, x1:x2+1]
        valid_depth = region_depth[region_depth > 0]
        
        if len(valid_depth) == 0:
            return -1
        
        # Use median depth in bounding box
        distance = np.median(valid_depth)
        return distance
    
    def evaluate_object_distances(self, pred_depth: np.ndarray, 
                                 gt_boxes: List[Dict],
                                 camera_matrix: Optional[np.ndarray] = None) -> Dict:
        """
        Evaluate distance predictions for ground truth boxes
        """
        results = {
            'per_object': [],
            'per_class': defaultdict(lambda: {'errors': [], 'distances': []})
        }
        
        for gt_box in gt_boxes:
            # Skip truncated/occluded objects for now
            if gt_box['truncated'] > 0.5 or gt_box['occluded'] > 0:
                continue
            
            gt_distance = gt_box['gt_distance']
            pred_bbox = gt_box['bbox']
            
            pred_distance = self.estimate_bbox_distance(pred_depth, pred_bbox, camera_matrix)
            
            if pred_distance < 0:
                continue
            
            error = abs(pred_distance - gt_distance)
            rel_error = error / (gt_distance + 1e-8)
            
            obj_result = {
                'class': gt_box['class'],
                'gt_distance': float(gt_distance),
                'pred_distance': float(pred_distance),
                'error': float(error),
                'rel_error': float(rel_error),
                'occluded': int(gt_box['occluded'])
            }
            
            results['per_object'].append(obj_result)
            results['per_class'][gt_box['class']]['errors'].append(error)
            results['per_class'][gt_box['class']]['distances'].append(gt_distance)
        
        # Compute per-class statistics
        class_stats = {}
        for class_name, data in results['per_class'].items():
            if not data['errors']:
                continue
            
            errors = np.array(data['errors'])
            distances = np.array(data['distances'])
            
            class_stats[class_name] = {
                'MAE': float(np.mean(errors)),
                'RMSE': float(np.sqrt(np.mean(errors**2))),
                'Median_AE': float(np.median(errors)),
                'Rel_Error_Mean': float(np.mean(np.array(data['errors']) / (np.array(data['distances']) + 1e-8))),
                'count': len(errors),
                'distance_range': [float(distances.min()), float(distances.max())]
            }
        
        # Overall statistics
        all_errors = np.array([obj['error'] for obj in results['per_object']])
        all_distances = np.array([obj['gt_distance'] for obj in results['per_object']])
        
        results['overall'] = {
            'MAE': float(np.mean(all_errors)),
            'RMSE': float(np.sqrt(np.mean(all_errors**2))),
            'Median_AE': float(np.median(all_errors)),
            'Rel_Error_Mean': float(np.mean(all_errors / (all_distances + 1e-8))),
            'total_objects': len(results['per_object']),
            'distance_range': [float(all_distances.min()), float(all_distances.max())]
        }
        
        results['per_class_stats'] = class_stats
        
        return results
    
    def analyze_occlusion_impact(self, results: Dict) -> Dict:
        """Analyze how occlusion affects distance accuracy"""
        
        visible_errors = []
        partially_occluded_errors = []
        heavily_occluded_errors = []
        
        for obj in results['per_object']:
            error = obj['error']
            occluded = obj['occluded']
            
            if occluded == 0:
                visible_errors.append(error)
            elif occluded <= 1:
                partially_occluded_errors.append(error)
            else:
                heavily_occluded_errors.append(error)
        
        occlusion_analysis = {}
        
        if visible_errors:
            occlusion_analysis['not_occluded'] = {
                'MAE': float(np.mean(visible_errors)),
                'RMSE': float(np.sqrt(np.mean(np.array(visible_errors)**2))),
                'count': len(visible_errors)
            }
        
        if partially_occluded_errors:
            occlusion_analysis['partially_occluded'] = {
                'MAE': float(np.mean(partially_occluded_errors)),
                'RMSE': float(np.sqrt(np.mean(np.array(partially_occluded_errors)**2))),
                'count': len(partially_occluded_errors)
            }
        
        if heavily_occluded_errors:
            occlusion_analysis['heavily_occluded'] = {
                'MAE': float(np.mean(heavily_occluded_errors)),
                'RMSE': float(np.sqrt(np.mean(np.array(heavily_occluded_errors)**2))),
                'count': len(heavily_occluded_errors)
            }
        
        return occlusion_analysis
    
    def visualize_results(self, results: Dict, output_prefix: str = "object_distance"):
        """Generate visualization plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Predicted vs Ground Truth
        objects = results['per_object']
        gt_distances = [obj['gt_distance'] for obj in objects]
        pred_distances = [obj['pred_distance'] for obj in objects]
        
        axes[0, 0].scatter(gt_distances, pred_distances, alpha=0.6)
        max_dist = max(max(gt_distances), max(pred_distances))
        axes[0, 0].plot([0, max_dist], [0, max_dist], 'r--', label='Perfect Prediction')
        axes[0, 0].set_xlabel('Ground Truth Distance (m)')
        axes[0, 0].set_ylabel('Predicted Distance (m)')
        axes[0, 0].set_title('Distance Prediction Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Error distribution
        errors = [obj['error'] for obj in objects]
        axes[0, 1].hist(errors, bins=30, edgecolor='black')
        axes[0, 1].set_xlabel('Absolute Error (m)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Error Distribution')
        axes[0, 1].axvline(np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.2f}m')
        axes[0, 1].legend()
        
        # Plot 3: Error vs Distance
        axes[1, 0].scatter(gt_distances, errors, alpha=0.6, c=errors, cmap='viridis')
        axes[1, 0].set_xlabel('Ground Truth Distance (m)')
        axes[1, 0].set_ylabel('Absolute Error (m)')
        axes[1, 0].set_title('Error vs Distance')
        cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
        cbar.set_label('Error (m)')
        
        # Plot 4: Per-class performance
        class_stats = results['per_class_stats']
        class_names = list(class_stats.keys())
        class_maes = [class_stats[c]['MAE'] for c in class_names]
        class_counts = [class_stats[c]['count'] for c in class_names]
        
        bars = axes[1, 1].bar(class_names, class_maes)
        axes[1, 1].set_ylabel('Mean Absolute Error (m)')
        axes[1, 1].set_title('Per-Class Distance Error')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, class_counts):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'n={count}', ha='center', va='bottom')
        
        plt.tight_layout()
        output_path = self.output_dir / f'{output_prefix}_visualization.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization to {output_path}")
    
    def save_results(self, results: Dict, filename: str = "object_distance_results.json"):
        """Save results to JSON"""
        output_path = self.output_dir / filename
        
        # Convert numpy types
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj
        
        results_converted = convert_types(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_converted, f, indent=2)
        
        logger.info(f"Saved results to {output_path}")
    
    def print_summary(self, results: Dict):
        """Print summary statistics"""
        logger.info("\n" + "="*60)
        logger.info("OBJECT DISTANCE EVALUATION SUMMARY")
        logger.info("="*60)
        
        if 'overall' in results:
            overall = results['overall']
            logger.info(f"\nOVERALL STATISTICS:")
            logger.info(f"  Mean Absolute Error: {overall['MAE']:.3f} m")
            logger.info(f"  RMSE: {overall['RMSE']:.3f} m")
            logger.info(f"  Median Absolute Error: {overall['Median_AE']:.3f} m")
            logger.info(f"  Mean Relative Error: {overall['Rel_Error_Mean']:.3f}")
            logger.info(f"  Total Objects Evaluated: {overall['total_objects']}")
            logger.info(f"  Distance Range: [{overall['distance_range'][0]:.1f}, {overall['distance_range'][1]:.1f}] m")
        
        if 'per_class_stats' in results:
            logger.info(f"\nPER-CLASS STATISTICS:")
            for class_name, stats in results['per_class_stats'].items():
                logger.info(f"\n  {class_name.upper()}:")
                logger.info(f"    MAE: {stats['MAE']:.3f} m")
                logger.info(f"    RMSE: {stats['RMSE']:.3f} m")
                logger.info(f"    Count: {stats['count']}")
                logger.info(f"    Distance Range: [{stats['distance_range'][0]:.1f}, {stats['distance_range'][1]:.1f}] m")

def main():
    parser = argparse.ArgumentParser(description='Object Distance Accuracy Evaluation')
    parser.add_argument('--model', default='zoedepth', help='Depth model to use')
    parser.add_argument('--detector', default='yolo11n', help='Object detector to use')
    parser.add_argument('--dataset', default='kitti', help='Dataset to evaluate on')
    parser.add_argument('--output-dir', default='results/object_distance', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    
    args = parser.parse_args()
    
    evaluator = ObjectDistanceEvaluator(args.output_dir)
    
    logger.info(f"Object Distance Evaluation Configuration:")
    logger.info(f"  Depth Model: {args.model}")
    logger.info(f"  Detector: {args.detector}")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Output Directory: {args.output_dir}")

if __name__ == '__main__':
    main()
