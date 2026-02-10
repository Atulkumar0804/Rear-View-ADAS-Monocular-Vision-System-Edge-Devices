# Code Changes Summary - Hybrid Depth Async Integration

## File: `inference/camera_inference.py`

### Change 1: Added Threading Imports (Lines 12-16)
```python
# ADDED:
import threading
from queue import Queue
from collections import deque
import json
import time
```

### Change 2: Initialize Async Infrastructure in __init__ (Lines 241-246)
```python
# ADDED in __init__ (after other initializations):
self.depth_queue = Queue(maxsize=2)  # Only keep latest 2 frames
self.ml_depth_thread = None
self.depth_result_cache = {}  # {frame_id: depth_result}
self.ml_depth_running = False
self.ml_depth_confidence = 0.6  # 60% ML + 40% pinhole
```

### Change 3: Start ML Thread When Hybrid Mode Enabled (Lines 408-413)
```python
# ADDED new method:
def _start_ml_depth_thread(self):
    """Start background thread for ML Depth Pro inference"""
    if not self.use_hybrid:
        return
    self.ml_depth_running = True
    self.ml_depth_thread = threading.Thread(target=self._ml_depth_worker, daemon=True)
    self.ml_depth_thread.start()
```

### Change 4: ML Depth Worker Background Thread (Lines 415-450)
```python
# ADDED new method:
def _ml_depth_worker(self):
    """Background thread: Continuously processes frames with Depth Pro"""
    while self.ml_depth_running:
        try:
            frame, detections, frame_id = self.depth_queue.get(timeout=1)
            
            # Run Depth Pro (runs in background, no impact on main thread)
            try:
                depth_map = self.hybrid_depth.get_ml_depth(frame)
                
                # Cache result by frame_id
                self.depth_result_cache[frame_id] = {
                    'depth_map': depth_map,
                    'timestamp': time.time()
                }
                
                # Clean old cached results (keep only last 10)
                if len(self.depth_result_cache) > 10:
                    oldest_id = min(self.depth_result_cache.keys())
                    del self.depth_result_cache[oldest_id]
            except Exception as e:
                pass  # Silently fail, main thread continues
        except:
            pass  # Queue timeout, continue polling
```

### Change 5: Stop ML Thread Gracefully (Lines 452-457)
```python
# ADDED new method:
def stop_ml_depth_thread(self):
    """Stop background thread gracefully"""
    if self.ml_depth_thread:
        self.ml_depth_running = False
        self.ml_depth_thread.join(timeout=2)
        self.ml_depth_thread = None
```

### Change 6: Hybrid Depth Blending Method (Lines 459-510)
```python
# ADDED new method:
def get_hybrid_depth_for_detections(self, detections, track_id):
    """Blend ML depth (background) with pinhole camera (instant)"""
    if not detections:
        return {'distance': None, 'method': 'unknown', 'confidence': 0.0}
    
    bbox = detections[0]['bbox']
    bbox_height = bbox[3] - bbox[1]
    class_name = detections[0].get('class', 'unknown')
    
    # Get fast pinhole camera depth (always available, instant)
    pinhole_distance = self.estimate_distance(bbox_height, class_name)
    
    # Check if ML depth is cached and ready
    ml_distance = None
    ml_confidence = 0.0
    
    if track_id in self.depth_result_cache:
        cache_entry = self.depth_result_cache[track_id]
        
        # Extract depth at bounding box center
        depth_map = cache_entry['depth_map']
        if depth_map is not None:
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            
            # Safe pixel access
            cx = max(0, min(cx, depth_map.shape[1] - 1))
            cy = max(0, min(cy, depth_map.shape[0] - 1))
            
            ml_distance = float(depth_map[cy, cx])
            ml_confidence = self.ml_depth_confidence
    
    # Blending logic
    if ml_distance is not None and ml_distance > 0:
        # Blend: 60% ML (accurate) + 40% pinhole (fast)
        blended_distance = (0.6 * ml_distance + 0.4 * pinhole_distance)
        return {
            'distance': blended_distance,
            'method': 'blend',
            'confidence': ml_confidence,
            'ml_distance': ml_distance,
            'pinhole_distance': pinhole_distance
        }
    else:
        # ML not ready yet, use pinhole only
        return {
            'distance': pinhole_distance,
            'method': 'pinhole',
            'confidence': 1.0,
            'pinhole_distance': pinhole_distance
        }
```

### Change 7: Queue Frames in detect_frame() (Lines ~730, in detect_frame method)
```python
# ADDED after YOLO detection, before depth estimation:

# Queue current frame for async ML Depth Pro (non-blocking)
# This runs in background while we process detections
if self.use_hybrid and self.ml_depth_thread and self.ml_depth_thread.is_alive():
    frame_id = int(time.time() * 1000) % 10000  # Frame timestamp ID
    try:
        self.depth_queue.put_nowait((frame.copy(), detections_for_depth, frame_id))
    except:
        pass  # Queue full, skip this frame (older frames will be used)
```

### Change 8: Use Hybrid Depth in Distance Calculation (Lines ~800, in detect_frame)
```python
# REPLACED:
# Default to pinhole (fallback)
distance = self.estimate_distance(bbox_height, result['class'])
distance_metadata = {}

# REFINE with Depth Map if available using ADAS Pipeline
if depth_map is not None:
    # ... ADAS pipeline code ...

# WITH:
# Assign track ID first
if matches[i] == -1:
    track_id = self.track_id_counter
    self.track_id_counter += 1
else:
    track_id = matches[i]

result['track_id'] = track_id

# Default to pinhole (fallback)
distance = self.estimate_distance(bbox_height, result['class'])
distance_metadata = {'method': 'pinhole', 'confidence': 1.0}

# Use hybrid depth blending if available
if self.use_hybrid and self.ml_depth_thread and self.ml_depth_thread.is_alive():
    hybrid_result = self.get_hybrid_depth_for_detections([result], track_id)
    if hybrid_result['distance'] is not None:
        distance = hybrid_result['distance']
        distance_metadata = hybrid_result
# Refine with Depth Map if available using ADAS Pipeline
elif depth_map is not None:
    # ... existing ADAS pipeline code ...
```

### Change 9: Store Distance Metadata (Lines ~832, in detect_frame)
```python
# ADDED after distance calculation:
result['track_id'] = track_id
result['distance'] = distance
result['distance_metadata'] = distance_metadata  # NEW
```

### Change 10: Enhanced draw_detections() for Visual Feedback (Lines 945-1050)
```python
# MODIFIED method signature to add parameters to labels and display:

def draw_detections(self, frame, detections, fps=None, debug=False):
    """Draw bounding boxes and labels with distance and motion state"""
    annotated = frame.copy()
    
    for det in detections:
        # ... existing code ...
        
        # ADDED: Extract and use distance metadata
        distance_metadata = det.get('distance_metadata', {})
        
        # MODIFIED: Label to show depth method
        depth_method = distance_metadata.get('method', 'unknown')
        if distance:
            label = f"{class_name}: {confidence:.2f} | {distance:.1f}m [{depth_method}]"
        else:
            label = f"{class_name}: {confidence:.2f}"
        
        # ... existing drawing code ...
        
        # ADDED: Show depth confidence
        if distance:
            depth_conf = distance_metadata.get('confidence', 1.0)
            if depth_conf < 1.0:
                conf_text = f"conf:{depth_conf:.2f}"
                cv2.putText(annotated, conf_text, (x1 + 5, conf_y), ...)
        
        # ... existing motion state drawing ...
    
    # ADDED: Show depth processing status
    if self.use_hybrid and self.ml_depth_thread:
        if self.ml_depth_thread.is_alive():
            status = f"Depth: Hybrid (ML async: ON)"
            cv2.putText(annotated, status, (10, 70), ...)
        else:
            status = f"Depth: Fallback (pinhole only)"
            cv2.putText(annotated, status, (10, 70), ...)
    
    return annotated
```

---

## Summary of Changes

| Change | Type | Lines | Purpose |
|--------|------|-------|---------|
| Threading imports | Added | 12-16 | Enable async processing |
| Async init fields | Added | 241-246 | Queue, thread, cache setup |
| _start_ml_depth_thread() | New method | 408-413 | Initialize background worker |
| _ml_depth_worker() | New method | 415-450 | Background ML depth inference |
| stop_ml_depth_thread() | New method | 452-457 | Graceful thread shutdown |
| get_hybrid_depth_for_detections() | New method | 459-510 | Blending algorithm (60% ML + 40% pinhole) |
| Frame queuing | Added | ~730 | Queue frames for async ML |
| Distance calculation | Modified | ~800 | Use hybrid depth instead of sync |
| Distance metadata storage | Added | ~832 | Store method + confidence |
| Enhanced draw_detections() | Modified | 945-1050 | Show depth method & confidence |

**Total additions**: ~120 lines of code
**Total modifications**: 3 existing methods enhanced
**New methods**: 4 async-related methods
**Files created**: 3 (test script + 2 documentation files)

---

## Backward Compatibility

All changes are:
- ✅ Behind `--hybrid-depth` flag (optional)
- ✅ Non-breaking to existing API
- ✅ Graceful fallback if threading fails
- ✅ Conditional (checks `self.use_hybrid` before using new code)

Default behavior (without `--hybrid-depth` flag) is unchanged.
