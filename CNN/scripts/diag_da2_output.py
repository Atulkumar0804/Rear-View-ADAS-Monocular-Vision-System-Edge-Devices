#!/usr/bin/env python3
"""Diagnose raw DA2 output: range, direction, correlation with KITTI GT."""
import torch, numpy as np
from transformers import AutoModelForDepthEstimation
import cv2
from pathlib import Path
import torch.nn.functional as F

CNN_DIR = Path(__file__).parent.parent

da2 = AutoModelForDepthEstimation.from_pretrained(str(CNN_DIR/"models/depth_anything_v2")).cuda().eval()
MEAN = torch.tensor([0.485,0.456,0.406]).view(3,1,1).cuda()
STD  = torch.tensor([0.229,0.224,0.225]).view(3,1,1).cuda()

kitti_rgb = sorted((CNN_DIR/"dataset/kitti_depth/val").rglob("image_02/*.png"))[:5]
kitti_gt  = sorted((CNN_DIR/"dataset/kitti_depth/val").rglob("groundtruth/*.png"))[:5]

print("=== Raw DA2 output stats on KITTI val images ===")
print(f"{'Image':<30} {'rel_min':>8} {'rel_max':>8} {'rel_mean':>9}  {'GT_mean':>8}  {'corr(rel,GT)':>13}  {'corr(1/rel,GT)':>15}")
print("-"*105)

for rgb_p, gt_p in zip(kitti_rgb, kitti_gt):
    rgb = cv2.cvtColor(cv2.imread(str(rgb_p)), cv2.COLOR_BGR2RGB)
    rgb_t = torch.from_numpy(rgb.astype(np.float32)/255.).permute(2,0,1).unsqueeze(0).cuda()
    rgb_t = (rgb_t - MEAN) / STD
    gt_raw = cv2.imread(str(gt_p), cv2.IMREAD_ANYDEPTH).astype(np.float32)/256.
    valid = gt_raw > 0.1

    with torch.no_grad():
        rel = da2(rgb_t).predicted_depth.float()

    rel_up = F.interpolate(rel.unsqueeze(1), size=gt_raw.shape, mode='bilinear', align_corners=True).squeeze().cpu().numpy()
    r = rel_up[valid]
    g = gt_raw[valid]
    corr_direct  = np.corrcoef(r, g)[0,1]
    corr_inverse = np.corrcoef(1.0/(r+1e-6), g)[0,1]
    print(f"{rgb_p.name:<30} {r.min():>8.2f} {r.max():>8.2f} {r.mean():>9.2f}  {g.mean():>8.1f}m  {corr_direct:>+13.3f}  {corr_inverse:>+15.3f}")

print("\nConclusion:")
print("  corr(rel, GT) > 0  → rel is depth (larger = farther): use scale * rel")
print("  corr(1/rel, GT) > corr(rel, GT)  → rel is disparity: use scale / rel")
