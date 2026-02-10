#!/usr/bin/env python3
"""
Metric Scale Optimization for Depth Models

Analyzes KITTI ground truth vs predictions to compute optimal scale factors
for converting relative depth to metric depth.
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import json
import logging

sys.path.insert(0, 'scripts')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_kitti_depth(depth_path):
    """Load KITTI depth ground truth"""
    depth_png = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
    if depth_png is None:
        return None
    depth = depth_png.astype(np.float32) / 256.0
    return depth

def compute_scale_factor(pred_depth, gt_depth, min_depth=0.5, max_depth=80.0):
    """Compute optimal scale factor using median scaling"""
    mask = (gt_depth > min_depth) & (gt_depth < max_depth)
    
    if np.sum(mask) == 0:
        return None
    
    pred = pred_depth[mask]
    gt = gt_depth[mask]
    
    # Compute scale: scale = median(gt) / median(pred)
    scale = np.median(gt) / (np.median(pred) + 1e-8)
    
    return scale

def optimize_scale_kitti(kitti_path="dataset/kitti_depth", num_samples=200):
    """Optimize scale factor using KITTI dataset"""
    
    logger.info("="*60)
    logger.info("DEPTH SCALE OPTIMIZATION - KITTI Dataset")
    logger.info("="*60)
    
    kitti_path = Path(kitti_path)
    
    # Load depth model
    logger.info("Loading ZoeDepth model...")
    from zoedepth_loader import load_zoedepth_model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_model = load_zoedepth_model("ZoeD_NK", device=device)
    
    if depth_model is None:
        logger.error("Failed to load depth model!")
        return None
    
    depth_model.eval()
    logger.info(f"✓ Model loaded on {device}")
    
    # Find sample pairs
    logger.info(f"Finding RGB-Depth pairs (target: {num_samples} samples)...")
    pairs = []
    train_dir = kitti_path / "train"
    
    if not train_dir.exists():
        train_dir = kitti_path / "training"
    
    if not train_dir.exists():
        logger.error(f"Training directory not found: {train_dir}")
        return None
    
    sequences = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    
    for seq in sequences:
        rgb_dir = seq / "image_02" / "data"
        depth_dir = seq / "proj_depth" / "groundtruth" / "image_02"
        
        if not rgb_dir.exists():
            rgb_dir = seq / "image_02"
        
        if rgb_dir.exists() and depth_dir.exists():
            rgb_files = sorted(rgb_dir.glob("*.png"))
            
            for rgb_file in rgb_files:
                depth_file = depth_dir / rgb_file.name
                
                if depth_file.exists():
                    pairs.append((rgb_file, depth_file))
                    
                if len(pairs) >= num_samples:
                    break
        
        if len(pairs) >= num_samples:
            break
    
    logger.info(f"Found {len(pairs)} RGB-Depth pairs")
    
    if len(pairs) == 0:
        logger.error("No valid pairs found!")
        return None
    
    # Compute scale factors
    logger.info("Computing optimal scale factors...")
    scale_factors = []
    
    for rgb_path, depth_path in tqdm(pairs[:num_samples], desc="Processing"):
        # Load RGB
        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            continue
        
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        
        # Predict depth
        with torch.no_grad():
            img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img_tensor = img_tensor.to(device)
            depth_pred = depth_model(img_tensor)
            
            if isinstance(depth_pred, dict) and 'metric_depth' in depth_pred:
                depth_pred = depth_pred['metric_depth'].squeeze().cpu().numpy()
            elif isinstance(depth_pred, torch.Tensor):
                depth_pred = depth_pred.squeeze().cpu().numpy()
            else:
                depth_pred = np.array(depth_pred)
        
        # Load ground truth
        gt_depth = load_kitti_depth(depth_path)
        if gt_depth is None:
            continue
        
        # Resize prediction
        if depth_pred.shape != gt_depth.shape:
            depth_pred = cv2.resize(depth_pred, (gt_depth.shape[1], gt_depth.shape[0]))
        
        # Compute scale factor
        scale = compute_scale_factor(depth_pred, gt_depth)
        if scale is not None and 0.1 < scale < 10.0:  # Filter outliers
            scale_factors.append(scale)
    
    if len(scale_factors) == 0:
        logger.error("No valid scale factors computed!")
        return None
    
    # Compute statistics
    scale_factors = np.array(scale_factors)
    
    median_scale = np.median(scale_factors)
    mean_scale = np.mean(scale_factors)
    std_scale = np.std(scale_factors)
    
    logger.info("\n" + "="*60)
    logger.info("SCALE OPTIMIZATION RESULTS")
    logger.info("="*60)
    logger.info(f"Samples analyzed: {len(scale_factors)}")
    logger.info(f"Median scale factor: {median_scale:.4f}")
    logger.info(f"Mean scale factor:   {mean_scale:.4f}")
    logger.info(f"Std deviation:       {std_scale:.4f}")
    logger.info(f"Min scale:           {scale_factors.min():.4f}")
    logger.info(f"Max scale:           {scale_factors.max():.4f}")
    
    # Current scale factor check
    current_scale = 1.21  # From evaluation results
    corrected_scale = current_scale / median_scale
    
    logger.info("\n" + "="*60)
    logger.info("RECOMMENDED CORRECTION")
    logger.info("="*60)
    logger.info(f"Current METRIC_DEPTH_SCALE: {current_scale:.4f}")
    logger.info(f"Optimal scale factor:       {median_scale:.4f}")
    logger.info(f"NEW METRIC_DEPTH_SCALE:     {corrected_scale:.4f}")
    
    logger.info("\n" + "="*60)
    logger.info("ACTION REQUIRED")
    logger.info("="*60)
    logger.info("Update the following files:")
    logger.info("1. inference/camera_inference.py")
    logger.info("   Line ~80-85: METRIC_DEPTH_SCALE = {:.4f}".format(corrected_scale))
    logger.info("\n2. inference/video_inference.py")
    logger.info("   Line ~25-30: METRIC_DEPTH_SCALE = {:.4f}".format(corrected_scale))
    
    # Expected improvement
    logger.info("\n" + "="*60)
    logger.info("EXPECTED IMPROVEMENT")
    logger.info("="*60)
    current_rmse = 7.01
    current_mae = 3.67
    expected_rmse = current_rmse / median_scale
    expected_mae = current_mae / median_scale
    
    logger.info(f"RMSE: {current_rmse:.2f}m → {expected_rmse:.2f}m ({(1-expected_rmse/current_rmse)*100:.1f}% reduction)")
    logger.info(f"MAE:  {current_mae:.2f}m → {expected_mae:.2f}m ({(1-expected_mae/current_mae)*100:.1f}% reduction)")
    
    # Save results
    results = {
        'samples_analyzed': int(len(scale_factors)),
        'median_scale': float(median_scale),
        'mean_scale': float(mean_scale),
        'std_scale': float(std_scale),
        'current_scale': float(current_scale),
        'recommended_scale': float(corrected_scale),
        'expected_rmse_improvement': float((1-expected_rmse/current_rmse)*100),
        'expected_mae_improvement': float((1-expected_mae/current_mae)*100)
    }
    
    output_file = Path("calibration_data/optimized_scale_factors.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to: {output_file}")
    
    return corrected_scale

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize depth scale factor using KITTI')
    parser.add_argument('--kitti-path', default='dataset/kitti_depth', help='Path to KITTI dataset')
    parser.add_argument('--num-samples', type=int, default=200, help='Number of samples to analyze')
    
    args = parser.parse_args()
    
    optimized_scale = optimize_scale_kitti(args.kitti_path, args.num_samples)
    
    if optimized_scale is not None:
        logger.info("\n✅ Scale optimization completed successfully!")
        logger.info(f"📊 Use METRIC_DEPTH_SCALE = {optimized_scale:.4f} in your inference scripts")
    else:
        logger.error("\n❌ Scale optimization failed!")
        sys.exit(1)
