# DEPTH PRO DISTANCE MEASUREMENT - IMPROVEMENTS NEEDED

## Problem Summary

Currently, **Depth Pro is NOT actually measuring distances** in camera_inference.py. The system always shows `[pinhole]` method, which means:
- ❌ Depth Pro model is **failing silently** 
- ❌ Fallback to pinhole camera model is being used
- ❌ Distances are purely calculated from object height + focal length, not from actual depth

## Why It's Not Working

The code has **3 main issues**:

### Issue #1: Depth Pro Model Not Properly Loaded
**Location:** Lines 820-860 in camera_inference.py

**Problem:** 
```python
checkpoint = torch.load(checkpoint_path, map_location=self.device)
# Directly passing checkpoint as model, but checkpoint is just weights!
```

The code loads a `.pt` file that contains only weights (state_dict), not a model object. When `AsyncDepthPro` tries to call `model.infer()` or `model()`, it fails because the raw weights don't have these methods.

**Fix:**
- You need the actual **Depth Pro model architecture** code
- The `depth_pro.pt` file must be compatible with a model class
- Currently, the file might be just a state dict, not a full model

### Issue #2: AsyncDepthPro Inference Fails Silently
**Location:** Lines 147-200 in camera_inference.py

**Problem:**
```python
def _run_inference(self, frame: np.ndarray) -> np.ndarray:
    try:
        prediction = self.model.infer(image_tensor)  # ← This fails!
    except Exception as e:
        print(f"⚠️ Depth Pro inference failed: {e}")
        # Returns default 5m depth - system doesn't realize it failed
```

When the inference fails, it silently returns a default depth map. The main system never knows the ML model failed.

### Issue #3: Hybrid Depth Blending But With Zero ML Input
**Location:** Lines 280-320 in camera_inference.py

**Problem:**
```python
if self.ml_depth_map is not None and self.ml_depth_map.shape == (h, w):
    depth_map = self._blend_depths(depth_map, self.ml_depth_map, detections)
```

Since Depth Pro inference fails, `ml_depth_map` is always None, so blending never happens. Distances are 100% from pinhole camera model.

---

## What You Need To Do (To Improve)

### Step 1: Verify Depth Pro Model File Format
```bash
cd /home/atul/Desktop/atul/rear_view_adas_monocular/CNN

python3 << 'EOF'
import torch

checkpoint = torch.load("models/depth_pro/checkpoints/depth_pro.pt", map_location='cpu')
print(f"Type: {type(checkpoint)}")
print(f"Keys: {list(checkpoint.keys())[:10]}")

# If it's a state_dict (weights only):
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    print("✓ Has model_state_dict - need architecture")
elif isinstance(checkpoint, dict) and any(k.startswith('model.') for k in checkpoint.keys()):
    print("✓ Has model. prefix - need architecture")
else:
    print("? Unknown format - might be full model")
EOF
```

### Step 2: Get Proper Depth Pro Model Architecture

If the checkpoint is just weights, you need to:
1. **Install depth_pro package:**
   ```bash
   pip install depth-pro
   # OR
   pip install git+https://github.com/apple/ml-depth-pro.git
   ```

2. **Load model properly:**
   ```python
   from depth_pro import create_model
   
   model = create_model()  # Initialize model architecture
   checkpoint = torch.load("depth_pro.pt")
   model.load_state_dict(checkpoint['model_state_dict'])  # Load weights
   model = model.to(device).eval()
   ```

### Step 3: Add Better Error Handling and Debugging

Create a diagnostic script:

```bash
cat > /home/atul/Desktop/atul/rear_view_adas_monocular/debug_depthpro.py << 'SCRIPT'
#!/usr/bin/env python3
import torch
import cv2
import numpy as np
from pathlib import Path

def debug_depth_pro():
    checkpoint_path = Path("CNN/models/depth_pro/checkpoints/depth_pro.pt")
    
    print("\n" + "="*70)
    print("DEPTH PRO DIAGNOSTIC")
    print("="*70)
    
    # 1. Check file
    if not checkpoint_path.exists():
        print(f"❌ File not found: {checkpoint_path}")
        return False
    
    print(f"✓ File found: {checkpoint_path}")
    print(f"  Size: {checkpoint_path.stat().st_size / 1e9:.2f} GB")
    
    # 2. Load checkpoint
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"✓ Checkpoint loaded")
        print(f"  Type: {type(checkpoint)}")
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return False
    
    # 3. Analyze structure
    if isinstance(checkpoint, dict):
        keys = list(checkpoint.keys())
        print(f"  Keys ({len(keys)}): {keys[:5]}...")
        
        # Check for state dict indicators
        if 'model_state_dict' in keys:
            print(f"  → Contains model_state_dict (weights only)")
        elif any(k.startswith('model.') for k in keys):
            print(f"  → Contains model. prefix (weights only)")
        elif any('weight' in k or 'bias' in k for k in keys):
            print(f"  → Contains weight/bias (state dict)")
        elif callable(checkpoint):
            print(f"  → Is callable (likely model class)")
    
    # 4. Try to use it
    print(f"\n✓ Trying inference...")
    dummy_input = torch.randn(1, 3, 512, 768).to(device)
    
    try:
        # Try different inference methods
        if hasattr(checkpoint, 'infer'):
            output = checkpoint.infer(dummy_input)
            print(f"✓ model.infer() works!")
        elif hasattr(checkpoint, '__call__'):
            output = checkpoint(dummy_input)
            print(f"✓ model() works!")
        else:
            print(f"❌ No infer or __call__ method")
            return False
        
        print(f"  Output type: {type(output)}")
        if isinstance(output, dict):
            print(f"  Output keys: {list(output.keys())}")
            if 'depth' in output:
                print(f"  ✓ Has 'depth' key")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70 + "\n")
    return True

if __name__ == "__main__":
    debug_depth_pro()
SCRIPT
python3 /home/atul/Desktop/atul/rear_view_adas_monocular/debug_depthpro.py
```

### Step 4: Enable Logging to See Actual Errors

Add verbose logging:

```python
# In AsyncDepthPro._run_inference(), change the exception handler to:
except Exception as e:
    print(f"\n⚠️ DEPTH PRO INFERENCE FAILED:")
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()
    print(f"\n   (This is expected if depth_pro module is not installed)")
    print(f"   Install with: pip install depth-pro\n")
```

---

## Quick Wins to Improve Distance Accuracy

Even if Depth Pro inference fails, you can improve pinhole camera accuracy:

### 1. Better Focal Length Calibration
The pinhole model uses focal length to estimate distance. Calibrate it:

```python
# Current: FOCAL_LENGTH = 435.75 (default)
# Better: Get from actual camera calibration
python3 << 'EOF'
import numpy as np
calib_path = "CNN/calibration_data/camera_matrix.npy"
if Path(calib_path).exists():
    K = np.load(calib_path)
    fx = K[0, 0]
    fy = K[1, 1]
    focal_length_avg = (fx + fy) / 2
    print(f"Calibrated focal length: {focal_length_avg:.2f}")
else:
    print("No calibration found - using default 435.75")
EOF
```

### 2. Better Height Estimates Per Class
```python
# Current: Using generic heights like Person=1.7m
# Better: Collect actual height statistics from your videos
# Update REAL_HEIGHTS dictionary with accurate measurements
```

### 3. Temporal Smoothing
The system already does this with Kalman filters. Make sure it's enabled:
```python
smoothed = self.apply_temporal_filter(calibrated, track_id, use_kalman=True)
```

---

## Next Steps

1. **Run the diagnostic script** to understand the Depth Pro checkpoint format
2. **Install depth_pro if not present:** `pip install depth-pro`
3. **Enable verbose logging** to see actual errors
4. **Fix the model loading** based on diagnostic output
5. **Test with small script** before running full inference

---

## For Better Depth Pro Performance

If you get Depth Pro working, optimize like this:

```python
# Reduce update interval for faster depth refreshes
ml_update_interval=2.0,  # Instead of 5.0

# Increase ML weight for detected objects
ml_weight = 1.0 - object_mask * 0.3  # Instead of 0.7
pinhole_weight = object_mask * 0.3

# This means: 70% ML + 30% Pinhole for detected objects
```

---

## Summary

| Component | Status | Issue |
|-----------|--------|-------|
| Depth Pro Model | ❌ Broken | Inference fails silently |
| Async Loading | ✓ Working | Properly async, non-blocking |
| Hybrid Blending | ✓ Ready | Just needs actual ML input |
| Pinhole Fallback | ✓ Working | Gives reasonable distance estimates |
| Kalman Filter | ✓ Working | Smooths temporal variations |

**Bottom line:** Get the Depth Pro model inference working properly, then distances will be measured from actual depth data instead of just geometric estimates.
