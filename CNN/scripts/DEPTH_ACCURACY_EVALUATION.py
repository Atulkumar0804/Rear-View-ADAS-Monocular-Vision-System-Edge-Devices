#!/usr/bin/env python3
"""
Comprehensive Depth Accuracy Evaluation Script

Evaluates monocular depth estimation accuracy on KITTI and other datasets.
Provides metrics for:
- Per-frame depth accuracy
- Object distance accuracy
- Occlusion robustness
- Cross-dataset generalization
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
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DepthEvaluator:
    """Comprehensive depth accuracy evaluator"""
    
    def __init__(self, dataset_path: str = "dataset/kitti_depth", output_dir: str = "results/depth_eval"):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = defaultdict(list)
    
    def compute_depth_metrics(self, pred_depth: np.ndarray, gt_depth: np.ndarray, 
                            mask: Optional[np.ndarray] = None) -> Dict:
        """
        Compute standard depth evaluation metrics
        
        Args:
            pred_depth: Predicted depth map (H, W)
            gt_depth: Ground truth depth map (H, W)
            mask: Valid pixel mask (H, W, bool)
        
        Returns:
            Dictionary of metrics
        """
        if mask is None:
            mask = gt_depth > 0
        
        pred = pred_depth[mask]
        gt = gt_depth[mask]
        
        # Avoid division by zero
        if len(pred) == 0 or len(gt) == 0:
            return {}
        
        # Relative errors
        rel_error = np.abs(pred - gt) / (gt + 1e-8)
        sq_rel_error = ((pred - gt) ** 2) / (gt ** 2 + 1e-8)
        
        # Absolute error
        abs_error = np.abs(pred - gt)
        
        # Log errors
        log_error = np.abs(np.log(pred + 1e-8) - np.log(gt + 1e-8))
        
        # Accuracy
        thresholds = [1.25, 1.5, 1.75]
        delta_accs = {}
        for t in thresholds:
            delta_accs[f'delta_{t}'] = np.mean(np.maximum(pred / (gt + 1e-8), gt / (pred + 1e-8)) < t)
        
        metrics = {
            'AbsRel': np.mean(rel_error),
            'SqRel': np.mean(sq_rel_error),
            'RMSE': np.sqrt(np.mean(abs_error ** 2)),
            'RMSE_log': np.sqrt(np.mean(log_error ** 2)),
            'MAE': np.mean(abs_error),
            'Median_AE': np.median(abs_error),
            'Median_RE': np.median(rel_error),
            **delta_accs
        }
        
        return metrics
    
    def evaluate_object_distances(self, pred_depth: np.ndarray, gt_depth: np.ndarray,
                                 bboxes: List[Dict], classes: List[str]) -> Dict:
        """
        Evaluate distance accuracy for detected objects
        
        Args:
            pred_depth: Predicted depth map
            gt_depth: Ground truth depth map
            bboxes: List of bounding boxes [{x1, y1, x2, y2, class_id}, ...]
            classes: Class names
        
        Returns:
            Dictionary of per-object and per-class metrics
        """
        object_metrics = defaultdict(list)
        class_metrics = defaultdict(lambda: defaultdict(list))
        
        for bbox in bboxes:
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
            class_id = bbox.get('class_id', 0)
            class_name = classes[class_id] if class_id < len(classes) else f'class_{class_id}'
            
            # Extract region
            region_pred = pred_depth[y1:y2, x1:x2]
            region_gt = gt_depth[y1:y2, x1:x2]
            
            # Use centroid depth
            valid_pred = region_pred[region_pred > 0]
            valid_gt = region_gt[region_gt > 0]
            
            if len(valid_pred) == 0 or len(valid_gt) == 0:
                continue
            
            dist_pred = np.median(valid_pred)
            dist_gt = np.median(valid_gt)
            
            error = abs(dist_pred - dist_gt)
            rel_error = error / (dist_gt + 1e-8)
            
            object_metrics['MAE'].append(error)
            object_metrics['Rel_Error'].append(rel_error)
            
            class_metrics[class_name]['MAE'].append(error)
            class_metrics[class_name]['Rel_Error'].append(rel_error)
            class_metrics[class_name]['Count'].append(1)
        
        # Aggregate metrics
        aggregated = {}
        aggregated['overall'] = {
            'MAE': np.mean(object_metrics['MAE']) if object_metrics['MAE'] else 0,
            'Rel_Error': np.mean(object_metrics['Rel_Error']) if object_metrics['Rel_Error'] else 0,
            'Count': len(object_metrics['MAE'])
        }
        
        for class_name, metrics in class_metrics.items():
            aggregated[class_name] = {
                'MAE': np.mean(metrics['MAE']) if metrics['MAE'] else 0,
                'Rel_Error': np.mean(metrics['Rel_Error']) if metrics['Rel_Error'] else 0,
                'Count': int(np.sum(metrics['Count']))
            }
        
        return aggregated
    
    def analyze_occlusion_impact(self, pred_depth: np.ndarray, gt_depth: np.ndarray,
                                bbox_mask: np.ndarray) -> Dict:
        """
        Analyze depth accuracy as a function of occlusion
        
        Args:
            pred_depth: Predicted depth map
            gt_depth: Ground truth depth map
            bbox_mask: Binary mask of object regions
        
        Returns:
            Occlusion impact metrics
        """
        occlusion_levels = {
            '0-25%': (0.0, 0.25),
            '25-50%': (0.25, 0.50),
            '50-75%': (0.50, 0.75),
            '75-100%': (0.75, 1.0)
        }
        
        results = {}
        
        for level_name, (occ_min, occ_max) in occlusion_levels.items():
            # Create occlusion mask for this range
            occ_ratio = np.sum(bbox_mask) / (np.sum(bbox_mask > 0) + 1e-8)
            
            if occ_min <= occ_ratio <= occ_max:
                mask = bbox_mask > 0
                metrics = self.compute_depth_metrics(pred_depth, gt_depth, mask)
                results[level_name] = metrics
        
        return results
    
    def generate_visualizations(self, pred_depth: np.ndarray, gt_depth: np.ndarray, 
                               image: np.ndarray, output_prefix: str = "eval"):
        """Generate visualization plots"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Predicted depth
        im1 = axes[0, 0].imshow(pred_depth, cmap='viridis')
        axes[0, 0].set_title('Predicted Depth')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Ground truth depth
        im2 = axes[0, 1].imshow(gt_depth, cmap='viridis')
        axes[0, 1].set_title('Ground Truth Depth')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Error map
        error = np.abs(pred_depth - gt_depth)
        im3 = axes[0, 2].imshow(error, cmap='hot')
        axes[0, 2].set_title('Absolute Error')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # RGB image
        axes[1, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('RGB Image')
        
        # Histogram of errors
        valid_error = error[error > 0]
        axes[1, 1].hist(valid_error, bins=50, edgecolor='black')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].set_xlabel('Error (m)')
        axes[1, 1].set_ylabel('Count')
        
        # Scatter: pred vs gt
        valid_mask = gt_depth > 0
        axes[1, 2].scatter(gt_depth[valid_mask], pred_depth[valid_mask], alpha=0.1, s=1)
        max_depth = max(gt_depth[valid_mask].max(), pred_depth[valid_mask].max())
        axes[1, 2].plot([0, max_depth], [0, max_depth], 'r--', label='Perfect')
        axes[1, 2].set_xlabel('Ground Truth Depth (m)')
        axes[1, 2].set_ylabel('Predicted Depth (m)')
        axes[1, 2].set_title('Depth Prediction Accuracy')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{output_prefix}_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization to {output_prefix}_visualization.png")
    
    def save_results(self, results: Dict, filename: str = "evaluation_results.json"):
        """Save evaluation results to JSON"""
        output_path = self.output_dir / filename
        
        # Convert numpy types to native Python types
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
        logger.info("DEPTH EVALUATION SUMMARY")
        logger.info("="*60)
        
        for dataset_name, metrics in results.items():
            logger.info(f"\n{dataset_name.upper()}")
            logger.info("-" * 40)
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"  {key}: {value:.4f}")
                    elif isinstance(value, dict):
                        logger.info(f"  {key}:")
                        for k, v in value.items():
                            logger.info(f"    {k}: {v:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Depth Accuracy Evaluation')
    parser.add_argument('--dataset-path', default='dataset/kitti_depth', help='Path to dataset')
    parser.add_argument('--output-dir', default='results/depth_eval', help='Output directory')
    parser.add_argument('--model', default='zoedepth', help='Model to evaluate')
    parser.add_argument('--analyze-datasets', action='store_true', help='Analyze available datasets')
    parser.add_argument('--evaluate-kitti', action='store_true', help='Evaluate on KITTI')
    parser.add_argument('--occlusion-analysis', action='store_true', help='Analyze occlusion impact')
    parser.add_argument('--generate-report', action='store_true', help='Generate comprehensive report')
    
    args = parser.parse_args()
    
    evaluator = DepthEvaluator(args.dataset_path, args.output_dir)
    
    if args.analyze_datasets:
        logger.info("Analyzing available datasets...")
        logger.info(f"KITTI Depth path: {evaluator.dataset_path}")
        if evaluator.dataset_path.exists():
            logger.info("  ✓ KITTI dataset found")
            training_path = evaluator.dataset_path / 'training'
            if training_path.exists():
                images = list(training_path.glob('image_2/*'))
                depths = list(training_path.glob('proj_depth/*/*'))
                logger.info(f"    - Training images: {len(images)}")
                logger.info(f"    - Depth maps: {len(depths)}")
        else:
            logger.warning("  ✗ KITTI dataset not found")
    
    logger.info(f"\nEvaluation configuration:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Dataset: {args.dataset_path}")
    logger.info(f"  Output: {args.output_dir}")

if __name__ == '__main__':
    main()
