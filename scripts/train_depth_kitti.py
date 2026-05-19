#!/usr/bin/env python3
"""
train_depth_kitti.py
────────────────────────────────────────────────────────────────────────────────
Fine-tune MiDaS v2.1 Small (the edge-device depth backend) on the KITTI Eigen
depth subset that ships with this project.

Dataset layout expected
───────────────────────
  dataset/kitti_depth/
    train/<drive>/
      image_02/data/<frame>.png          ← RGB  1242×375
      proj_depth/groundtruth/image_02/<frame>.png   ← uint16 depth (depth_m = px/256)
    val/<drive>/
      image_02/data/<frame>.png
      proj_depth/groundtruth/image_02/<frame>.png

Training strategy
─────────────────
  1. Load pretrained MiDaS v2.1 Small backbone (EfficientNet-Lite0 encoder).
  2. Replace the prediction head with a lightweight metric-depth head that
     maps to [0.1 m, 80 m] via a sigmoid-scaled output.
  3. Loss  = Scale-Invariant Log (SILog) + 0.15 × L1-gradient smoothness
     SILog is the standard KITTI benchmark loss [Eigen et al. 2014].
  4. Cosine-LR schedule, AdamW, early stopping on δ1 metric.
  5. Save best checkpoint to  models/depth_lite/midas_kitti_metric.onnx
     (ONNX export for deployment via AsyncDepthLite).

Usage
─────
  python scripts/train_depth_kitti.py
  python scripts/train_depth_kitti.py --epochs 30 --batch 8 --lr 3e-4
  python scripts/train_depth_kitti.py --resume models/depth_lite/midas_kitti_metric.pt

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

# ── Project root ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
CNN_DIR    = SCRIPT_DIR.parent
DATASET_ROOT = CNN_DIR / "dataset" / "kitti_depth"
MODEL_OUT_DIR = CNN_DIR / "models" / "depth_lite"
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("DepthTrain")

# ── KITTI camera intrinsics (standard Eigen split cam2) ──────────────────────
#   fx = 721.5377, fy = 721.5377, cx = 609.5593, cy = 172.854
#   Sensor resolution: 1242×375  (W×H)
KITTI_FX = 721.5377
KITTI_FY = 721.5377
KITTI_CX = 609.5593
KITTI_CY = 172.854
KITTI_W  = 1242
KITTI_H  = 375

# Training input resolution (downsampled for speed while preserving structure)
TRAIN_H = 192
TRAIN_W = 640
DEPTH_MIN = 0.1   # metres
DEPTH_MAX = 80.0  # metres


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class KITTIDepthDataset(Dataset):
    """
    KITTI Depth Completion dataset loader (projected LiDAR groundtruth).

    depth_m = uint16_pixel / 256.0
    """

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, split: str = "train", augment: bool = True):
        assert split in ("train", "val")
        self.split   = split
        self.augment = augment and (split == "train")
        self.pairs   = self._scan(DATASET_ROOT / split)
        log.info("KITTIDepth  split=%s  pairs=%d", split, len(self.pairs))

    # ---- internal helpers ---------------------------------------------------

    def _scan(self, root: Path):
        """Return [(rgb_path, depth_path)] for all matching pairs."""
        pairs = []
        for depth_path in sorted(root.rglob("proj_depth/groundtruth/image_02/*.png")):
            # depth:  <root>/<drive>/proj_depth/groundtruth/image_02/<frame>.png
            # rgb  :  <root>/<drive>/image_02/data/<frame>.png
            # parents[0]=image_02  [1]=groundtruth  [2]=proj_depth  [3]=<drive>
            drive_dir  = depth_path.parents[3]
            frame_name = depth_path.name
            rgb_path   = drive_dir / "image_02" / "data" / frame_name
            if rgb_path.exists():
                pairs.append((str(rgb_path), str(depth_path)))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rgb_path, depth_path = self.pairs[idx]

        # ── Load RGB ─────────────────────────────────────────────────────────
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            rgb = np.zeros((KITTI_H, KITTI_W, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # ── Load depth (uint16, depth_m = px/256) ────────────────────────────
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            depth_raw = np.zeros((KITTI_H, KITTI_W), dtype=np.uint16)
        depth_m = depth_raw.astype(np.float32) / 256.0

        # ── Resize ───────────────────────────────────────────────────────────
        rgb    = cv2.resize(rgb,    (TRAIN_W, TRAIN_H), interpolation=cv2.INTER_LINEAR)
        depth_m = cv2.resize(depth_m, (TRAIN_W, TRAIN_H), interpolation=cv2.INTER_NEAREST)

        # ── Augmentation (training only) ──────────────────────────────────────
        if self.augment:
            # Random horizontal flip
            if np.random.rand() < 0.5:
                rgb     = cv2.flip(rgb, 1)
                depth_m = cv2.flip(depth_m, 1)

            # Colour jitter (brightness / contrast / saturation)
            from PIL import Image as PILImage
            pil = PILImage.fromarray(rgb)
            brightness = np.random.uniform(0.8, 1.2)
            contrast   = np.random.uniform(0.8, 1.2)
            saturation = np.random.uniform(0.8, 1.2)
            pil = TF.adjust_brightness(pil, brightness)
            pil = TF.adjust_contrast(pil,   contrast)
            pil = TF.adjust_saturation(pil, saturation)
            rgb = np.array(pil)

        # ── To tensor & normalise RGB ─────────────────────────────────────────
        rgb_t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor(self.MEAN, dtype=torch.float32).view(3, 1, 1)
        std  = torch.tensor(self.STD,  dtype=torch.float32).view(3, 1, 1)
        rgb_t = (rgb_t - mean) / std

        # ── Depth mask (valid LiDAR returns only) ─────────────────────────────
        valid_mask = torch.from_numpy((depth_m > DEPTH_MIN) & (depth_m <= DEPTH_MAX))
        depth_t    = torch.from_numpy(
            np.clip(depth_m, DEPTH_MIN, DEPTH_MAX).astype(np.float32)
        )

        return rgb_t, depth_t, valid_mask


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL — MiDaS Small + Metric Head
# ═══════════════════════════════════════════════════════════════════════════════

class MetricDepthHead(nn.Module):
    """
    Thin metric-depth head that plugs on top of MiDaS v2.1 Small.

    MiDaS normally outputs relative *inverse* depth (disparity).
    This head:
        1. Receives the 1-channel disparity map from MiDaS.
        2. Passes it through two conv layers to learn the metric mapping.
        3. Applies a sigmoid to bound output, then scales to [DEPTH_MIN, DEPTH_MAX].
    """

    def __init__(self, depth_min=DEPTH_MIN, depth_max=DEPTH_MAX):
        super().__init__()
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.log_depth_min = math.log(depth_min)
        self.log_depth_max = math.log(depth_max)

        # Deeper head with BatchNorm for stable training.
        # Input: per-sample-normalised disparity in roughly N(0,1)
        # Output: log-depth (unbounded); clipped to [log_min, log_max]
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),   # 1×1 projection to log-depth
        )
        # Learnable global scale + shift in log-space.
        # Initialised so that output ≈ log(10 m) = 2.3 at start.
        self.log_scale = nn.Parameter(torch.tensor(1.0))
        self.log_shift = nn.Parameter(torch.tensor(2.3))   # log(10 m)

    def forward(self, disp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            disp: (B, 1, H, W) – raw MiDaS disparity output (relative,
                  range approximately [0, 600] depending on scene)
        Returns:
            depth: (B, 1, H, W) – metric depth in metres [depth_min, depth_max]

        Key insight
        ───────────
        MiDaS outputs RELATIVE disparity (large = close).  We cannot feed
        raw disparity into a sigmoid because the range is unbounded and shifts
        with every image.  The fix is to normalise per sample to N(0,1) first,
        then learn a log-affine mapping:  log(d) = scale × conv(norm_disp) + shift
        This is the same approach used by AdaBins and ZoeDepth.
        """
        B = disp.shape[0]

        # Per-sample normalise to N(0,1) — removes absolute scale ambiguity
        mu  = disp.view(B, -1).mean(dim=1).view(B, 1, 1, 1)
        sig = disp.view(B, -1).std(dim=1).clamp(min=1e-6).view(B, 1, 1, 1)
        d_norm = (disp - mu) / sig

        # Log-depth prediction
        log_depth = self.log_scale * self.conv(d_norm) + self.log_shift

        # Clamp to valid log-depth range, then exponentiate
        log_depth = log_depth.clamp(self.log_depth_min, self.log_depth_max)
        depth = torch.exp(log_depth)
        return depth


class MiDaSMetricDepth(nn.Module):
    """
    Full trainable model: MiDaS-Small backbone (frozen BN) + MetricDepthHead.

    The MiDaS encoder is kept entirely frozen for the first `warmup_epochs`
    epochs (backbone_frozen=True).  After that the entire model is fine-tuned
    end-to-end with a 10× lower LR for the backbone.
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # Load MiDaS v2.1 Small via torch.hub
        log.info("Loading MiDaS v2.1 Small …")
        try:
            self.midas = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small",
                pretrained=pretrained, trust_repo=True,
            )
        except Exception:
            # Offline fallback: load from cached zoedepth intel-isl directory
            local_path = CNN_DIR / "models" / "zoedepth" / "intel-isl_MiDaS_master"
            self.midas = torch.hub.load(
                str(local_path), "MiDaS_small",
                source="local", pretrained=pretrained,
            )

        # Freeze BN statistics throughout (they were calibrated on ImageNet)
        for m in self.midas.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)

        self.head = MetricDepthHead()

        # Transform for MiDaS (will NOT be applied inside the model; caller
        # is responsible for normalising with ImageNet mean/std at 384×384)
        # We keep the attribute for reference.
        self.midas_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) normalised with ImageNet mean/std
        Returns:
            depth: (B, 1, H, W) metric depth in metres
        """
        B, C, H, W = x.shape

        # MiDaS expects any resolution ≥ 32×32 (internally resamples)
        disp = self.midas(x)          # (B, H', W') – 2-D output from MiDaS

        # Ensure (B, 1, H, W)
        if disp.dim() == 2:
            disp = disp.unsqueeze(0).unsqueeze(0)
        elif disp.dim() == 3:
            disp = disp.unsqueeze(1)

        # Upsample back to input resolution
        if disp.shape[-2:] != (H, W):
            disp = F.interpolate(disp, size=(H, W), mode="bilinear",
                                 align_corners=False)

        depth = self.head(disp)
        return depth

    def backbone_params(self):
        return self.midas.parameters()

    def head_params(self):
        return self.head.parameters()


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def silog_loss(pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor, variance_focus: float = 0.85) -> torch.Tensor:
    """
    Scale-Invariant Logarithmic loss (Eigen et al. 2014).

        L = sqrt( mean(d²) - variance_focus × mean(d)² )
        where d = log(pred) - log(target)

    Args:
        pred, target: (B, 1, H, W) metric depth
        mask        : (B, H, W) or (B, 1, H, W) boolean, True = valid pixel
        variance_focus: 0.85 is standard (0 = pure MSE in log space)
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    eps      = 1e-6
    log_pred = torch.log(pred.clamp(min=eps))
    log_tgt  = torch.log(target.clamp(min=eps))
    d        = (log_pred - log_tgt) * mask.float()

    n = mask.float().sum().clamp(min=1)
    d_mean   = d.sum() / n
    d_sq_mean = (d ** 2).sum() / n

    loss = torch.sqrt((d_sq_mean - variance_focus * d_mean ** 2).clamp(min=0))
    return loss


def gradient_loss(pred: torch.Tensor, target: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
    """
    First-order depth gradient smoothness loss.
    Penalises discontinuities in predicted depth that are absent from GT.
    All inputs must be (B, 1, H, W).  mask is broadcast to that shape.
    """
    # Ensure all tensors have the same 4-D shape
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    # Ensure spatial size matches (pred may be upsampled to input res,
    # target is at training res — they should be identical after dataset)
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred.float(), size=target.shape[-2:],
                             mode="bilinear", align_corners=False)

    mask_f = mask.float()

    def grad(x):
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        return dy, dx

    pred_dy, pred_dx = grad(pred)
    tgt_dy,  tgt_dx  = grad(target)

    m_dy = mask_f[:, :, 1:, :] * mask_f[:, :, :-1, :]
    m_dx = mask_f[:, :, :, 1:] * mask_f[:, :, :, :-1]

    n_dy = m_dy.sum().clamp(min=1)
    n_dx = m_dx.sum().clamp(min=1)
    loss = (pred_dy - tgt_dy).abs().mul(m_dy).sum() / n_dy \
         + (pred_dx - tgt_dx).abs().mul(m_dx).sum() / n_dx
    return loss


def total_loss(pred, target, mask, lambda_grad=0.15):
    sl = silog_loss(pred, target, mask)
    gl = gradient_loss(pred, target, mask)
    return sl + lambda_grad * gl


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_metrics(pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor) -> dict:
    """
    Standard KITTI monocular depth evaluation metrics:
        δ1   : % pixels with max(pred/gt, gt/pred) < 1.25
        δ2   : < 1.25²
        δ3   : < 1.25³
        AbsRel: mean |pred−gt| / gt
        SqRel : mean (pred−gt)² / gt
        RMSE  : RMS error in metres
        log10 : mean |log10(pred) − log10(gt)|
    """
    if mask.dim() == 4:
        mask = mask.squeeze(1)
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)

    eps   = 1e-6
    p     = pred[mask].clamp(min=eps)
    t     = target[mask].clamp(min=eps)
    n     = p.numel()

    if n == 0:
        return {k: float("nan") for k in
                ("delta1", "delta2", "delta3", "abs_rel",
                 "sq_rel", "rmse", "log10")}

    thresh = torch.max(p / t, t / p)
    d1 = (thresh < 1.25  ).float().mean().item()
    d2 = (thresh < 1.25**2).float().mean().item()
    d3 = (thresh < 1.25**3).float().mean().item()

    abs_rel = ((p - t).abs() / t).mean().item()
    sq_rel  = (((p - t) ** 2) / t).mean().item()
    rmse    = ((p - t) ** 2).mean().sqrt().item()
    log10   = (torch.log10(p) - torch.log10(t)).abs().mean().item()

    return dict(delta1=d1, delta2=d2, delta3=d3,
                abs_rel=abs_rel, sq_rel=sq_rel, rmse=rmse, log10=log10)


# ═══════════════════════════════════════════════════════════════════════════════
# ONNX EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def export_onnx(model: nn.Module, out_path: Path, device: torch.device):
    """Export the trained model to ONNX with dynamic batch + spatial dims."""
    model.eval()
    dummy = torch.randn(1, 3, TRAIN_H, TRAIN_W, device=device)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["input"],
        output_names=["depth"],
        dynamic_axes={
            "input":  {0: "batch", 2: "height", 3: "width"},
            "depth":  {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=12,
        do_constant_folding=True,
    )
    log.info("ONNX exported → %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = KITTIDepthDataset("train", augment=True)
    val_ds   = KITTIDepthDataset("val",   augment=False)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True,
                          drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MiDaSMetricDepth(pretrained=True).to(device)

    # Stage 1 (warmup): freeze backbone, train head at 5× the base LR.
    #   Head needs to learn the disparity→metric mapping from scratch.
    #   At 5× LR it converges within the first few epochs.
    # Stage 2: unfreeze backbone at 0.05× LR (very conservative to avoid
    #   destroying ImageNet features that anchor spatial structure).
    warmup_epochs = max(2, args.epochs * 3 // 10)  # 30% warmup

    def make_optimizer(unfreeze_backbone=False):
        if unfreeze_backbone:
            return torch.optim.AdamW([
                {"params": model.head_params(),     "lr": args.lr},
                {"params": model.backbone_params(), "lr": args.lr * 0.05},
            ], weight_decay=1e-4)
        else:
            # Freeze backbone — only train the new head
            for p in model.midas.parameters():
                p.requires_grad_(False)
            # Head LR is 5× the base LR: head starts with random weights and
            # must learn quickly while the backbone is still frozen.
            return torch.optim.AdamW(model.head_params(), lr=args.lr * 5,
                                     weight_decay=1e-4)

    optimizer = make_optimizer(unfreeze_backbone=False)
    # Cosine schedule from warm LR → 1e-6.  Warm-up phase uses head-only LR
    # which is set to args.lr * 5 (head needs to learn fast while backbone frozen).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Optional resume ───────────────────────────────────────────────────────
    start_epoch = 0
    best_d1 = 0.0
    ckpt_path = MODEL_OUT_DIR / "midas_kitti_metric.pt"

    if args.resume and Path(args.resume).exists():
        log.info("Resuming from %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_d1     = ckpt.get("best_d1", 0.0)
        log.info("Resumed epoch=%d  best_δ1=%.4f", start_epoch, best_d1)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    for epoch in range(start_epoch, args.epochs):

        # Unfreeze backbone after warmup
        if epoch == warmup_epochs:
            log.info("Epoch %d: unfreezing backbone (stage 2)", epoch)
            optimizer = make_optimizer(unfreeze_backbone=True)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch, eta_min=args.lr * 1e-4
            )

        # ─── Train ─────────────────────────────────────────────────────────
        model.train()
        # Keep BN in eval mode (frozen stats)
        for m in model.midas.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        epoch_loss = 0.0
        t0 = time.time()
        for step, (rgb, depth, mask) in enumerate(train_dl):
            rgb   = rgb.to(device,  non_blocking=True)
            depth = depth.to(device, non_blocking=True).unsqueeze(1)
            mask  = mask.to(device,  non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(rgb)
                loss = total_loss(pred, depth, mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if step % 100 == 0:
                log.info(
                    "Epoch %d/%d  step %d/%d  loss=%.4f  LR=%.2e",
                    epoch + 1, args.epochs,
                    step, len(train_dl),
                    loss.item(),
                    optimizer.param_groups[0]["lr"],
                )

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_dl), 1)
        log.info("Epoch %d done in %.1f s  avg_loss=%.4f",
                 epoch + 1, time.time() - t0, avg_loss)

        # ─── Validate ──────────────────────────────────────────────────────
        model.eval()
        all_metrics = []
        with torch.no_grad():
            for rgb, depth, mask in val_dl:
                rgb   = rgb.to(device)
                depth = depth.to(device).unsqueeze(1)
                mask  = mask.to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    pred = model(rgb)
                m = compute_metrics(pred, depth, mask)
                all_metrics.append(m)

        # Average metrics across batches
        keys = all_metrics[0].keys()
        avg_m = {k: float(np.mean([m[k] for m in all_metrics
                                    if not math.isnan(m[k])])) for k in keys}

        log.info(
            "Val  δ1=%.4f  δ2=%.4f  δ3=%.4f  "
            "AbsRel=%.4f  RMSE=%.3f  log10=%.4f",
            avg_m["delta1"], avg_m["delta2"], avg_m["delta3"],
            avg_m["abs_rel"], avg_m["rmse"], avg_m["log10"],
        )

        # ─── Checkpoint ────────────────────────────────────────────────────
        is_best = avg_m["delta1"] > best_d1
        if is_best:
            best_d1 = avg_m["delta1"]
            torch.save(
                {"epoch": epoch, "model": model.state_dict(),
                 "best_d1": best_d1, "metrics": avg_m},
                ckpt_path,
            )
            log.info("✅ New best δ1=%.4f  → %s", best_d1, ckpt_path)

            # Also export ONNX on each new best
            onnx_path = MODEL_OUT_DIR / "midas_kitti_metric.onnx"
            try:
                export_onnx(model, onnx_path, device)
            except Exception as e:
                log.warning("ONNX export failed: %s", e)

    log.info("Training complete.  Best δ1=%.4f", best_d1)
    log.info("Best checkpoint : %s", ckpt_path)
    log.info("ONNX model      : %s", MODEL_OUT_DIR / 'midas_kitti_metric.onnx')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Fine-tune MiDaS Small on KITTI Depth for metric output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--epochs",  type=int,   default=40,
                    help="Total training epochs (30-40 needed for good metric depth)")
    ap.add_argument("--batch",   type=int,   default=16,
                    help="Batch size — 16 gives better gradient signal on sparse KITTI depth")
    ap.add_argument("--lr",      type=float, default=3e-4,
                    help="Base LR (head gets 5× this during warmup, backbone 0.05×)")
    ap.add_argument("--workers", type=int,   default=4,
                    help="DataLoader worker processes")
    ap.add_argument("--resume",  type=str,   default=None,
                    help="Path to checkpoint .pt to resume from")
    args = ap.parse_args()

    log.info("=== KITTI Depth Fine-tuning ===")
    log.info("  Dataset     : %s", DATASET_ROOT)
    log.info("  Output dir  : %s", MODEL_OUT_DIR)
    log.info("  Train size  : %dx%d", TRAIN_W, TRAIN_H)
    log.info("  Epochs      : %d", args.epochs)
    log.info("  Batch       : %d", args.batch)
    log.info("  LR          : %.2e", args.lr)

    train(args)


if __name__ == "__main__":
    main()
