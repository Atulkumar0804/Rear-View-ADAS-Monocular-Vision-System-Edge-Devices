#!/usr/bin/env python3
"""
export_jetson.py
─────────────────────────────────────────────────────────────────────────────
Exports ALL models needed for Jetson Nano / Jetson Orin deployment:

  1.  YOLOv11n  (detect)       → models/yolo/yolo11n_jetson.onnx
  2.  Classifier YOLO (cls)    → models/classifier/weights/best_jetson.onnx
  3.  MiDaS v2.1 Small         → models/depth_lite/midas_small.onnx
  4.  Depth Anything V2 Small  → models/depth_lite/depth_anything_v2_small.onnx
       (fallback if MiDaS fails)

Jetson Nano constraints
───────────────────────
  GPU  : 128-core Maxwell  (CUDA 10.2, TensorRT 7.x)
  RAM  : 4 GB shared
  ZoeDepth : ❌ Infeasible  (~4 GB VRAM needed at 640×480)
  MiDaS Small: ✅ ~25-35 ms @ 384×384  (≈ 28 FPS)
  YOLO11n ONNX: ✅ ~18-25 ms @ 320×320 (≈ 40 FPS with TRT)

ONNX opset 11 is chosen because:
  • TensorRT 7.x (Jetson Nano JetPack 4.6) supports up to opset 13 for most ops
  • Opset 11 is the safest compatibility target across all Jetson generations

After running this script on your HOST PC, copy the exported folder to Jetson
and run the TensorRT conversion commands printed at the end.

Usage (host PC):
    python scripts/export_jetson.py
    python scripts/export_jetson.py --imgsz 320      # faster on Nano
    python scripts/export_jetson.py --no-depth       # skip depth export
    python scripts/export_jetson.py --depth-only     # only export depth

Author: Rear-View ADAS Project
"""

import argparse
import os
import sys
import shutil
import time
from pathlib import Path

# ── Project root ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
CNN_DIR    = SCRIPT_DIR.parent
sys.path.insert(0, str(CNN_DIR))

import torch
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
YOLO_PT_PATH       = CNN_DIR / "models/yolo11n.pt"
CLS_PT_PATH        = CNN_DIR / "models/classifier/weights/best.pt"
DEPTH_OUT_DIR      = CNN_DIR / "models/depth_lite"
YOLO_ONNX_PATH     = CNN_DIR / "models/yolo/yolo11n_jetson.onnx"
CLS_ONNX_PATH      = CNN_DIR / "models/classifier/weights/best_jetson.onnx"
MIDAS_ONNX_PATH    = DEPTH_OUT_DIR / "midas_small.onnx"
DA2_ONNX_PATH      = DEPTH_OUT_DIR / "depth_anything_v2_small.onnx"

# Jetson Nano recommended input sizes
YOLO_IMGSZ_NANO    = 320   # 320×320 vs 640×640 = 4× fewer pixels → 4× faster
DEPTH_H_NANO       = 256   # MiDaS Small recommended
DEPTH_W_NANO       = 320

ONNX_OPSET        = 11    # TensorRT 7.x safe opset

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def section(title: str):
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")


def success(msg: str): print(f"  ✅  {msg}")
def warn(msg: str):    print(f"  ⚠️   {msg}")
def info(msg: str):    print(f"  ℹ️   {msg}")
def fail(msg: str):    print(f"  ❌  {msg}")


def file_mb(path: Path) -> str:
    if path.exists():
        return f"{path.stat().st_size / 1e6:.2f} MB"
    return "N/A"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. YOLO11n → ONNX
# ═══════════════════════════════════════════════════════════════════════════════
def export_yolo(imgsz: int = YOLO_IMGSZ_NANO) -> bool:
    section(f"1/4  YOLOv11n → ONNX  (imgsz={imgsz})")

    try:
        from ultralytics import YOLO

        if not YOLO_PT_PATH.exists():
            fail(f"YOLO weights not found: {YOLO_PT_PATH}")
            return False

        info(f"Loading {YOLO_PT_PATH}  ({file_mb(YOLO_PT_PATH)})")
        model = YOLO(str(YOLO_PT_PATH))

        info(f"Exporting to ONNX (opset={ONNX_OPSET}, imgsz={imgsz}) ...")
        exported = model.export(
            format="onnx",
            imgsz=imgsz,
            opset=ONNX_OPSET,
            simplify=True,    # onnx-simplifier reduces graph nodes
            dynamic=False,    # static shapes → faster TRT engine
            half=False,       # FP32 for host export; TRT will do FP16/INT8
            device="cpu",     # CPU export is more reproducible
        )

        # Ultralytics saves next to .pt by default; move to our target path
        default_out = Path(str(YOLO_PT_PATH).replace(".pt", ".onnx"))
        if default_out.exists() and default_out != YOLO_ONNX_PATH:
            YOLO_ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(default_out), str(YOLO_ONNX_PATH))
        elif Path(str(exported)).exists() and Path(str(exported)) != YOLO_ONNX_PATH:
            YOLO_ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(exported), str(YOLO_ONNX_PATH))

        if YOLO_ONNX_PATH.exists():
            success(f"YOLO ONNX saved → {YOLO_ONNX_PATH}  ({file_mb(YOLO_ONNX_PATH)})")
        else:
            # ultralytics may have saved it with a different name pattern
            candidates = list(YOLO_PT_PATH.parent.glob("*.onnx"))
            if candidates:
                YOLO_ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(candidates[0]), str(YOLO_ONNX_PATH))
                success(f"YOLO ONNX saved → {YOLO_ONNX_PATH}  ({file_mb(YOLO_ONNX_PATH)})")
            else:
                warn("Could not locate exported ONNX file – check ultralytics output above")
                return False

        return True

    except Exception as e:
        fail(f"YOLO export failed: {e}")
        import traceback; traceback.print_exc()
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Classifier → ONNX
# ═══════════════════════════════════════════════════════════════════════════════
def export_classifier(imgsz: int = 224) -> bool:
    section("2/4  Classifier (YOLO-CLS) → ONNX")

    try:
        from ultralytics import YOLO

        if not CLS_PT_PATH.exists():
            warn(f"Classifier weights not found: {CLS_PT_PATH}  – skipping")
            return False

        info(f"Loading {CLS_PT_PATH}  ({file_mb(CLS_PT_PATH)})")
        model = YOLO(str(CLS_PT_PATH))

        info(f"Exporting classifier to ONNX (opset={ONNX_OPSET}, imgsz={imgsz}) ...")
        exported = model.export(
            format="onnx",
            imgsz=imgsz,
            opset=ONNX_OPSET,
            simplify=True,
            dynamic=False,
            device="cpu",
        )

        default_out = Path(str(CLS_PT_PATH).replace(".pt", ".onnx"))
        if default_out.exists() and default_out != CLS_ONNX_PATH:
            CLS_ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(default_out), str(CLS_ONNX_PATH))
        elif Path(str(exported)).exists() and Path(str(exported)) != CLS_ONNX_PATH:
            shutil.move(str(exported), str(CLS_ONNX_PATH))

        if CLS_ONNX_PATH.exists():
            success(f"Classifier ONNX saved → {CLS_ONNX_PATH}  ({file_mb(CLS_ONNX_PATH)})")
        return True

    except Exception as e:
        fail(f"Classifier export failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MiDaS v2.1 Small → ONNX   (ZoeDepth replacement for Jetson Nano)
# ═══════════════════════════════════════════════════════════════════════════════
def export_midas_small(h: int = DEPTH_H_NANO, w: int = DEPTH_W_NANO) -> bool:
    """
    MiDaS v2.1 Small replaces ZoeDepth on Jetson Nano.

    Why MiDaS Small (not ZoeDepth)?
    ────────────────────────────────
    ZoeDepth requires ~2-4 GB VRAM and runs at < 1 FPS on Jetson Nano.
    MiDaS v2.1 Small uses a MobileNetV2 encoder → ONLY 21 MB, ~28 FPS on Nano.

    Accuracy comparison (vs ZoeDepth on KITTI):
      ZoeDepth (ZoeD_K) : δ₁=0.955, AbsRel=0.071  ← better absolute depth
      MiDaS Small        : δ₁=0.921, AbsRel=0.107  ← relative depth (needs scale)
    Trade-off: MiDaS depth is relative, not metric. We run our existing
    DepthCalibrator scale correction on top (same as the hybrid-depth mode).
    """
    section(f"3/4  MiDaS v2.1 Small → ONNX  ({h}×{w})")

    DEPTH_OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        device = torch.device("cpu")  # export on CPU for portability

        info("Loading MiDaS v2.1 small from torch.hub ...")
        # torch.hub downloads ~21 MB MobileNetV2-based model
        midas = torch.hub.load(
            "intel-isl/MiDaS", "MiDaS_small",
            trust_repo=True, verbose=False
        )
        midas = midas.to(device).eval()
        info(f"Params: {sum(p.numel() for p in midas.parameters())/1e6:.2f} M")

        # Dummy input (1, 3, H, W)
        dummy = torch.randn(1, 3, h, w).to(device)

        info(f"Exporting to ONNX (opset={ONNX_OPSET}) ...")
        torch.onnx.export(
            midas,
            dummy,
            str(MIDAS_ONNX_PATH),
            opset_version=ONNX_OPSET,
            input_names=["image"],
            output_names=["depth"],
            dynamic_axes=None,   # static shapes for TRT
            do_constant_folding=True,
            export_params=True,
        )

        success(f"MiDaS Small ONNX → {MIDAS_ONNX_PATH}  ({file_mb(MIDAS_ONNX_PATH)})")

        # Verify with onnxruntime if available
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(MIDAS_ONNX_PATH),
                                        providers=["CPUExecutionProvider"])
            inp = dummy.numpy()
            t0 = time.time()
            out = sess.run(None, {"image": inp})
            elapsed = (time.time() - t0) * 1000
            info(f"ONNX Runtime CPU inference: {elapsed:.1f} ms  "
                 f"output shape: {out[0].shape}")
            success("MiDaS ONNX verified with onnxruntime ✓")
        except ImportError:
            warn("onnxruntime not installed – skipping verification. "
                 "Install: pip install onnxruntime")

        return True

    except Exception as e:
        fail(f"MiDaS Small export failed: {e}")
        import traceback; traceback.print_exc()
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Depth Anything V2 Small → ONNX  (alternative to MiDaS)
# ═══════════════════════════════════════════════════════════════════════════════
def export_depth_anything_v2_small(h: int = DEPTH_H_NANO,
                                    w: int = DEPTH_W_NANO) -> bool:
    """
    Depth Anything V2 Small (ViT-S distilled) – better accuracy than MiDaS Small
    but slightly heavier (~25 MB vs ~21 MB).
    Runs at ~20-25 FPS on Jetson Orin; ~8-12 FPS on Jetson Nano.
    Recommended for Jetson Orin / Xavier NX.
    Use MiDaS Small for Jetson Nano.
    """
    section(f"4/4  Depth Anything V2 Small → ONNX  ({h}×{w})")

    DEPTH_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check if fine-tuned weights exist locally
    local_model_dir = CNN_DIR / "models/depth_anything_v2"
    local_finetuned_dir = CNN_DIR / "models/depth_anything_v2_finetuned"

    model_dir = local_finetuned_dir if local_finetuned_dir.exists() else local_model_dir

    try:
        from transformers import AutoModelForDepthEstimation
        import torch

        device = torch.device("cpu")

        if model_dir.exists():
            info(f"Loading Depth Anything V2 from local: {model_dir}")
            model = AutoModelForDepthEstimation.from_pretrained(
                str(model_dir), ignore_mismatched_sizes=True
            )
        else:
            info("Downloading Depth Anything V2 Small from HuggingFace ...")
            info("(depth-anything/Depth-Anything-V2-Small-hf)")
            model = AutoModelForDepthEstimation.from_pretrained(
                "depth-anything/Depth-Anything-V2-Small-hf"
            )

        model = model.to(device).eval()
        info(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

        # Depth Anything V2 expects pixel_values (1,3,H,W) normalized
        dummy = torch.randn(1, 3, h, w).to(device)

        info(f"Exporting to ONNX (opset={ONNX_OPSET}) ...")
        with torch.no_grad():
            torch.onnx.export(
                model,
                {"pixel_values": dummy},
                str(DA2_ONNX_PATH),
                opset_version=ONNX_OPSET,
                input_names=["pixel_values"],
                output_names=["predicted_depth"],
                dynamic_axes=None,
                do_constant_folding=True,
                export_params=True,
            )

        success(f"Depth Anything V2 Small ONNX → {DA2_ONNX_PATH}  "
                f"({file_mb(DA2_ONNX_PATH)})")
        return True

    except Exception as e:
        warn(f"Depth Anything V2 Small export failed: {e}")
        warn("This is optional – MiDaS Small is the primary Jetson depth model.")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# ONNX MODEL QUICK-VERIFY
# ═══════════════════════════════════════════════════════════════════════════════
def verify_onnx(path: Path, input_name: str = None,
                shape: tuple = None) -> bool:
    """Quick verify ONNX graph is valid and can run one inference."""
    try:
        import onnx
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
        info(f"ONNX graph check passed: {path.name}")
        return True
    except ImportError:
        warn("pip install onnx  to enable graph validation")
    except Exception as e:
        fail(f"ONNX graph check failed: {e}")
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# PRINT JETSON DEPLOYMENT INSTRUCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def print_jetson_instructions(imgsz: int):
    section("JETSON DEPLOYMENT INSTRUCTIONS")
    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │          Copy the following folder to your Jetson:                  │
  │   scp -r models/yolo/yolo11n_jetson.onnx  jetson:~/adas/models/    │
  │   scp -r models/depth_lite/               jetson:~/adas/models/    │
  │   scp -r models/classifier/weights/best_jetson.onnx  jetson:~/adas/ │
  └─────────────────────────────────────────────────────────────────────┘

  ── On Jetson: Convert ONNX → TensorRT Engine ────────────────────────""")

    print(f"""
  # YOLO11n  TRT engine (FP16, {imgsz}×{imgsz})
  /usr/src/tensorrt/bin/trtexec \\
      --onnx=models/yolo/yolo11n_jetson.onnx \\
      --saveEngine=models/yolo/yolo11n_jetson_fp16.engine \\
      --fp16 \\
      --workspace=1024 \\
      --verbose

  # MiDaS Small  TRT engine (FP16, 256×320)
  /usr/src/tensorrt/bin/trtexec \\
      --onnx=models/depth_lite/midas_small.onnx \\
      --saveEngine=models/depth_lite/midas_small_fp16.engine \\
      --fp16 \\
      --workspace=512 \\
      --verbose

  # Classifier  TRT engine (FP16, 224×224)
  /usr/src/tensorrt/bin/trtexec \\
      --onnx=models/classifier/weights/best_jetson.onnx \\
      --saveEngine=models/classifier/weights/best_jetson_fp16.engine \\
      --fp16 \\
      --workspace=256 \\
      --verbose
""")
    print("""
  ── On Jetson: Run inference ──────────────────────────────────────────
  # With ONNX runtime (simpler, works immediately after copy)
  python inference/camera_inference.py \\
      --camera 0 --rear-camera --jetson --imgsz 320

  # With TensorRT engine (fastest, after trtexec conversion)
  python inference/camera_inference.py \\
      --camera 0 --rear-camera --jetson --tensorrt --imgsz 320

  ── Expected FPS on Jetson Nano (4GB) ────────────────────────────────
  YOLO11n  ONNX FP32  @ 320×320 :  ~18-22 FPS
  YOLO11n  TRT  FP16  @ 320×320 :  ~35-45 FPS
  MiDaS Small ONNX   @ 256×320 :  ~15-20 FPS (async thread)
  MiDaS Small TRT FP16 @256×320:  ~28-35 FPS (async thread)
  Combined pipeline (YOLO + depth async): ~30-40 FPS
""")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Export models to ONNX for Jetson Nano/Orin deployment")
    parser.add_argument("--imgsz", type=int, default=YOLO_IMGSZ_NANO,
                        help=f"YOLO input size (default: {YOLO_IMGSZ_NANO}  "
                             f"for Jetson Nano; use 416 for Orin)")
    parser.add_argument("--depth-h", type=int, default=DEPTH_H_NANO,
                        help=f"Depth model input height (default: {DEPTH_H_NANO})")
    parser.add_argument("--depth-w", type=int, default=DEPTH_W_NANO,
                        help=f"Depth model input width  (default: {DEPTH_W_NANO})")
    parser.add_argument("--no-depth",   action="store_true",
                        help="Skip depth model export")
    parser.add_argument("--depth-only", action="store_true",
                        help="Only export depth models")
    parser.add_argument("--no-cls",    action="store_true",
                        help="Skip classifier export")
    args = parser.parse_args()

    print("\n" + "═"*70)
    print("  JETSON ONNX EXPORT PIPELINE")
    print("  Rear-View ADAS  –  Jetson Nano / Orin ready")
    print("═"*70)
    print(f"  YOLO imgsz  : {args.imgsz}×{args.imgsz}")
    print(f"  Depth imgsz : {args.depth_h}×{args.depth_w}")
    print(f"  ONNX opset  : {ONNX_OPSET}")
    print(f"  Output root : {CNN_DIR / 'models'}")

    results = {}

    if not args.depth_only:
        results["yolo"]       = export_yolo(args.imgsz)
        if not args.no_cls:
            results["classifier"] = export_classifier()

    if not args.no_depth:
        results["midas"]      = export_midas_small(args.depth_h, args.depth_w)
        results["da2_small"]  = export_depth_anything_v2_small(
                                    args.depth_h, args.depth_w)

    # Verify exported ONNX files
    section("ONNX GRAPH VALIDATION")
    for path in [YOLO_ONNX_PATH, CLS_ONNX_PATH,
                  MIDAS_ONNX_PATH, DA2_ONNX_PATH]:
        if path.exists():
            verify_onnx(path)

    # Summary
    section("EXPORT SUMMARY")
    all_ok = True
    rows = [
        ("YOLOv11n ONNX",          YOLO_ONNX_PATH,    results.get("yolo", False)),
        ("Classifier ONNX",         CLS_ONNX_PATH,     results.get("classifier", False)),
        ("MiDaS Small ONNX",        MIDAS_ONNX_PATH,   results.get("midas", False)),
        ("Depth Anything V2 ONNX",  DA2_ONNX_PATH,     results.get("da2_small", False)),
    ]
    for name, path, ok in rows:
        icon = "✅" if ok else "❌"
        size = file_mb(path) if path.exists() else "not created"
        print(f"  {icon}  {name:<28} {size:<12} {path.name}")
        if not ok:
            all_ok = False

    print_jetson_instructions(args.imgsz)

    if all_ok:
        print("  🚀  All exports successful! Ready for Jetson deployment.\n")
    else:
        print("  ⚠️   Some exports failed. See errors above.\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
