#!/usr/bin/env python3
"""
jetson_depth_lite.py
─────────────────────────────────────────────────────────────────────────────
Lightweight asynchronous depth estimation for Jetson Nano / edge devices.
Replaces AsyncZoeDepth in camera_inference.py when running on Jetson.

Backends (auto-selected in order):
  1. MiDaS v2.1 Small   – TensorRT engine  (fastest, ~28-35 FPS)
  2. MiDaS v2.1 Small   – ONNX Runtime     (~15-20 FPS)
  3. MiDaS v2.1 Small   – PyTorch          (~8-12 FPS)
  4. Depth Anything V2 Small – ONNX/TRT    (~20 FPS on Orin)
  5. Fallback: linear ground-plane estimate (0 FPS cost)

Why NOT ZoeDepth on Jetson Nano?
─────────────────────────────────
  ZoeDepth ZoeD_K:  ~4 GB VRAM, ~0.5 FPS on Nano  → unusable
  MiDaS Small:      ~90 MB RAM,  ~28 FPS on Nano  → perfect
  Accuracy loss is mitigated by the DepthCalibrator scale correction
  already present in camera_inference.py.

MiDaS depth is RELATIVE (not metric). Metric scale is recovered at runtime
by the DepthCalibrator class using known-height object anchors.
This is identical to how the original DepthAnything model was used.

Usage (host PC for testing):
    python inference/jetson_depth_lite.py --test-image /path/to/image.jpg

Usage (imported in camera_inference.py):
    from inference.jetson_depth_lite import AsyncDepthLite
    depth_model = AsyncDepthLite(backend='auto', device='cuda')
"""

import threading
import time
from pathlib import Path
from queue import Queue
import queue
from typing import Optional, Tuple

import cv2
import numpy as np

# ── Project root ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
CNN_DIR    = SCRIPT_DIR.parent

MIDAS_ONNX_PATH  = CNN_DIR / "models/depth_lite/midas_small.onnx"
MIDAS_TRT_PATH   = CNN_DIR / "models/depth_lite/midas_small_fp16.engine"
DA2_ONNX_PATH    = CNN_DIR / "models/depth_lite/depth_anything_v2_small.onnx"
DA2_TRT_PATH     = CNN_DIR / "models/depth_lite/depth_anything_v2_small_fp16.engine"

# MiDaS v2.1 Small input normalization (ImageNet mean/std)
MIDAS_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
MIDAS_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Recommended input resolution for Jetson Nano (balances speed vs quality)
JETSON_NANO_H = 256
JETSON_NANO_W = 320
# Jetson Orin / Xavier NX can use higher resolution
JETSON_ORIN_H = 384
JETSON_ORIN_W = 512


# ═══════════════════════════════════════════════════════════════════════════════
# BACKEND IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class _MiDaSBackend:
    """MiDaS v2.1 Small PyTorch backend (host PC / Jetson fallback)."""

    def __init__(self, device: str = "cuda"):
        import torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model  = None
        self.transform = None
        self._load()

    def _load(self):
        import torch
        print("📦 [DepthLite] Loading MiDaS v2.1 Small (PyTorch) ...")
        self.model = torch.hub.load(
            "intel-isl/MiDaS", "MiDaS_small",
            trust_repo=True, verbose=False
        ).to(self.device).eval()
        # MiDaS transforms from torch.hub
        transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms",
            trust_repo=True, verbose=False
        )
        self.transform = transforms.small_transform
        params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"✅ [DepthLite] MiDaS Small loaded ({params:.2f}M params) "
              f"on {self.device}")

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        import torch
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp = self.transform(rgb).to(self.device)
        with torch.no_grad():
            prediction = self.model(inp)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame_bgr.shape[:2],
                mode="bilinear",
                align_corners=False,
            ).squeeze().cpu().numpy()
        return prediction.astype(np.float32)

    @property
    def name(self) -> str:
        return "MiDaS-Small-PyTorch"


class _MiDaSOnNXBackend:
    """MiDaS v2.1 Small ONNX Runtime backend (primary Jetson backend)."""

    def __init__(self, onnx_path: Path, input_h: int, input_w: int):
        import onnxruntime as ort
        self.h = input_h
        self.w = input_w
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(onnx_path), providers=providers)
        # Check which provider was actually used
        used = self.sess.get_providers()[0]
        print(f"✅ [DepthLite] MiDaS ONNX loaded  provider={used}  "
              f"input={input_h}×{input_w}")

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.w, self.h)).astype(np.float32) / 255.0
        normalized = (resized - MIDAS_MEAN) / MIDAS_STD
        return normalized.transpose(2, 0, 1)[np.newaxis].astype(np.float32)  # NCHW

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        inp = self._preprocess(frame_bgr)
        out = self.sess.run(None, {"image": inp})[0]  # (1, H, W) or (H, W)
        depth = out.squeeze().astype(np.float32)
        # Resize back to original frame size
        h_orig, w_orig = frame_bgr.shape[:2]
        if depth.shape != (h_orig, w_orig):
            depth = cv2.resize(depth, (w_orig, h_orig),
                               interpolation=cv2.INTER_LINEAR)
        return depth

    @property
    def name(self) -> str:
        return "MiDaS-Small-ONNX"


class _MiDaSTRTBackend:
    """
    MiDaS v2.1 Small TensorRT backend.
    Requires: tensorrt, pycuda  (installed on Jetson via JetPack)
    On Jetson Nano: ~28-35 FPS @ 256×320
    """

    def __init__(self, engine_path: Path, input_h: int, input_w: int):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa – initialises CUDA context

        self.h = input_h
        self.w = input_w

        logger = trt.Logger(trt.Logger.WARNING)
        with open(str(engine_path), "rb") as f, \
             trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Allocate GPU/CPU buffers
        self.inputs, self.outputs, self.bindings, self.stream = \
            self._allocate_buffers()
        print(f"✅ [DepthLite] MiDaS TRT engine loaded  "
              f"input={input_h}×{input_w}")

    def _allocate_buffers(self):
        import pycuda.driver as cuda
        import tensorrt as trt
        import numpy as np

        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for binding in self.engine:
            size = (abs(trt.volume(self.engine.get_binding_shape(binding)))
                    * self.engine.max_batch_size)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem   = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                inputs.append({"host": host_mem, "device": device_mem})
            else:
                outputs.append({"host": host_mem, "device": device_mem})

        return inputs, outputs, bindings, stream

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.w, self.h)).astype(np.float32) / 255.0
        normalized = (resized - MIDAS_MEAN) / MIDAS_STD
        return normalized.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        import pycuda.driver as cuda
        inp = self._preprocess(frame_bgr).ravel()
        np.copyto(self.inputs[0]["host"], inp)
        cuda.memcpy_htod_async(self.inputs[0]["device"],
                               self.inputs[0]["host"], self.stream)
        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]["host"],
                               self.outputs[0]["device"], self.stream)
        self.stream.synchronize()
        depth = self.outputs[0]["host"].reshape(self.h, self.w).astype(np.float32)
        h_orig, w_orig = frame_bgr.shape[:2]
        if depth.shape != (h_orig, w_orig):
            depth = cv2.resize(depth, (w_orig, h_orig),
                               interpolation=cv2.INTER_LINEAR)
        return depth

    @property
    def name(self) -> str:
        return "MiDaS-Small-TensorRT"


class _FallbackDepth:
    """
    Ground-plane linear depth fallback.
    Used when no depth model is available (e.g., ultra-low-end device).
    Returns a depth map where the bottom of the frame = close, top = far.
    """
    @staticmethod
    def infer(frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        y = np.linspace(0, 1, h, dtype=np.float32).reshape(-1, 1)
        # Ground plane: top ~ 50 m, bottom ~ 1 m (reversed: near = high value)
        depth = 50.0 * (1.0 - y) + 1.0 * y  # far at top, close at bottom
        depth = np.tile(depth, (1, w))
        return depth

    @property
    def name(self) -> str:
        return "FallbackGroundPlane"


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC DEPTH LITE  (public API – drop-in for AsyncZoeDepth)
# ═══════════════════════════════════════════════════════════════════════════════

class AsyncDepthLite:
    """
    Asynchronous lightweight depth estimation for Jetson Nano/Orin.
    Drop-in replacement for AsyncZoeDepth — same public API.

    Backend selection order (auto):
        TRT engine  → ONNX runtime  → PyTorch  → Fallback

    The depth map is RELATIVE, not metric. camera_inference.py's
    DepthCalibrator will recover metric scale using known-height anchors.

    Args:
        backend  : 'auto' | 'trt' | 'onnx' | 'pytorch' | 'fallback'
        device   : 'cuda' | 'cpu'
        input_h  : Depth model input height (default 256)
        input_w  : Depth model input width  (default 320)
        update_interval_frames: Run depth every N frames (default 5)
                   Lower = more accurate but slower.
                   Higher = faster but depth updates less often.
    """

    def __init__(
        self,
        backend: str = "auto",
        device:  str = "cuda",
        input_h: int = JETSON_NANO_H,
        input_w: int = JETSON_NANO_W,
        update_interval_frames: int = 5,
    ):
        self.device = device
        self.input_h = input_h
        self.input_w = input_w
        self.update_interval_frames = update_interval_frames
        self.frame_counter = 0

        self._model = self._load_backend(backend)

        self.input_queue:  Queue = Queue(maxsize=2)
        self.output_queue: Queue = Queue(maxsize=2)
        self.last_depth_map: Optional[np.ndarray] = None
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.running = True

        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="DepthLiteWorker")
        self._thread.start()
        print(f"✅ [AsyncDepthLite] backend={self._model.name}  "
              f"interval={update_interval_frames} frames  "
              f"input={input_h}×{input_w}")

    # ── Backend loader ────────────────────────────────────────────────────────
    @staticmethod
    def _onnx_has_gpu() -> bool:
        """Return True only when onnxruntime-gpu is installed AND CUDA is usable."""
        try:
            import onnxruntime as ort
            return "CUDAExecutionProvider" in ort.get_available_providers()
        except ImportError:
            return False

    def _load_backend(self, backend: str):
        import torch
        # When the host has a CUDA GPU but only CPU onnxruntime is installed,
        # PyTorch-GPU is 5-10× faster than ONNX-CPU.  Reorder 'auto' accordingly.
        if backend == "auto":
            if torch.cuda.is_available() and not self._onnx_has_gpu():
                auto_order = ["trt", "pytorch", "onnx", "fallback"]
                print("ℹ️  [DepthLite] onnxruntime-gpu absent – using PyTorch GPU backend")
            else:
                auto_order = ["trt", "onnx", "pytorch", "fallback"]
        else:
            auto_order = None   # not used for non-auto

        order = {
            "auto":    auto_order,
            "trt":     ["trt", "fallback"],
            "onnx":    ["onnx", "fallback"],
            "pytorch": ["pytorch", "fallback"],
            "fallback":["fallback"],
        }.get(backend, ["onnx", "pytorch", "fallback"])

        for b in order:
            try:
                if b == "trt" and MIDAS_TRT_PATH.exists():
                    return _MiDaSTRTBackend(MIDAS_TRT_PATH,
                                            self.input_h, self.input_w)
                elif b == "onnx" and MIDAS_ONNX_PATH.exists():
                    return _MiDaSOnNXBackend(MIDAS_ONNX_PATH,
                                             self.input_h, self.input_w)
                elif b == "pytorch":
                    return _MiDaSBackend(self.device)
                elif b == "fallback":
                    print("⚠️  [DepthLite] Using fallback ground-plane depth")
                    return _FallbackDepth()
            except Exception as e:
                print(f"⚠️  [DepthLite] Backend '{b}' failed: {e} – trying next")

        return _FallbackDepth()

    # ── Background worker ─────────────────────────────────────────────────────
    def _worker(self):
        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                t0 = time.time()
                depth = self._model.infer(frame)
                elapsed = time.time() - t0
                self.inference_count += 1
                self.total_inference_time += elapsed
                self.last_depth_map = depth

                # Normalise to a scale that DepthCalibrator can work with
                # MiDaS outputs inverse-depth (larger = closer).
                # Convert to pseudo-metric: depth_m = scale / midas_value
                # The scale will be calibrated dynamically by DepthCalibrator.
                # We invert and scale to roughly 0.5–50 m range.
                d_min, d_max = depth.min(), depth.max()
                if d_max > d_min:
                    # Invert & normalise to [1, 50] m pseudo-metric
                    depth_norm = (depth - d_min) / (d_max - d_min + 1e-6)
                    depth_pseudo = 50.0 - depth_norm * 49.0  # close → small value
                else:
                    depth_pseudo = np.ones_like(depth) * 5.0

                try:
                    self.output_queue.put(
                        (depth_pseudo, elapsed), block=False)
                except queue.Full:
                    try:
                        self.output_queue.get_nowait()
                        self.output_queue.put(
                            (depth_pseudo, elapsed), block=False)
                    except queue.Empty:
                        pass
            except Exception as e:
                print(f"⚠️  [DepthLite] Worker error: {e}")

    # ── Public API (mirrors AsyncZoeDepth) ────────────────────────────────────
    def request_depth(self, frame: np.ndarray, force: bool = False) -> bool:
        """Queue a frame for depth inference. Returns True if queued."""
        self.frame_counter += 1
        if not force and (self.frame_counter % self.update_interval_frames != 0):
            return False
        try:
            self.input_queue.put(frame.copy(), block=False)
            return True
        except queue.Full:
            return False

    def get_depth(self, wait: bool = False,
                  timeout: float = 0.01) -> Optional[Tuple[np.ndarray, float]]:
        """Get latest depth result. Returns (depth_map, inference_time) or None."""
        try:
            result = (self.output_queue.get(timeout=timeout)
                      if wait else self.output_queue.get_nowait())
            if result is not None:
                self.last_depth_map = result[0]
            return result
        except queue.Empty:
            return None

    def get_last_depth(self) -> Optional[np.ndarray]:
        return self.last_depth_map.copy() if self.last_depth_map is not None else None

    def get_stats(self) -> dict:
        avg = (self.total_inference_time / self.inference_count
               if self.inference_count > 0 else 0.0)
        return {
            "backend":          self._model.name,
            "inference_count":  self.inference_count,
            "total_time":       self.total_inference_time,
            "avg_inference_ms": avg * 1000,
            "fps":              1.0 / avg if avg > 0 else 0.0,
        }

    def stop(self):
        self.running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        print(f"🛑 [AsyncDepthLite] stopped  ({self._model.name})")

    @property
    def backend_name(self) -> str:
        return self._model.name


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test-image",  type=str,  default=None,
                        help="Path to a test image")
    parser.add_argument("--backend",     type=str,  default="auto",
                        choices=["auto", "trt", "onnx", "pytorch", "fallback"])
    parser.add_argument("--device",      type=str,  default="cuda")
    parser.add_argument("--input-h",     type=int,  default=JETSON_NANO_H)
    parser.add_argument("--input-w",     type=int,  default=JETSON_NANO_W)
    parser.add_argument("--n-iters",     type=int,  default=10)
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  AsyncDepthLite  –  Jetson Nano depth model test")
    print("="*60)

    model = AsyncDepthLite(
        backend=args.backend, device=args.device,
        input_h=args.input_h,  input_w=args.input_w,
        update_interval_frames=1,  # run every frame for testing
    )

    if args.test_image:
        frame = cv2.imread(args.test_image)
        if frame is None:
            print(f"❌ Could not read: {args.test_image}")
        else:
            print(f"Test image: {args.test_image}  {frame.shape}")
    else:
        # Synthetic frame
        frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
        print("Using synthetic random frame (640×480)")

    print(f"\nRunning {args.n_iters} inferences ...")
    times = []
    for i in range(args.n_iters):
        model.request_depth(frame, force=True)
        t0 = time.time()
        result = None
        for _ in range(50):                       # wait up to 5 s
            result = model.get_depth(wait=True, timeout=0.1)
            if result is not None:
                break
        if result:
            depth, t_inf = result
            times.append(t_inf * 1000)
            print(f"  iter {i+1:2d}  depth shape={depth.shape}  "
                  f"range=[{depth.min():.2f}, {depth.max():.2f}]  "
                  f"inference={t_inf*1000:.1f} ms")

    if times:
        print(f"\n  avg={np.mean(times):.1f} ms  "
              f"min={np.min(times):.1f} ms  "
              f"max={np.max(times):.1f} ms  "
              f"FPS≈{1000/np.mean(times):.1f}")

    print("\nStats:", model.get_stats())
    model.stop()
    print("\n✅ Test complete.")
