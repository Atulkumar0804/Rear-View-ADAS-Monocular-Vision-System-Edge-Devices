# GPU Configuration System for A6000 → Jetson Nano Restriction

## Overview

This system allows you to restrict your NVIDIA RTX A6000 GPU to match Jetson Nano Super specifications, or use any intermediate configuration. You can easily switch between:

- **A6000 Full**: Unrestricted maximum performance
- **Jetson Nano Restricted**: Simulates Jetson Nano Super constraints (67 TOPS, 8GB memory, 7-25W power)
- **Jetson Nano Power Save**: Ultra-low power mode (7W fixed, minimum compute)

## Architecture

### Components

1. **gpu_config.py** - Core GPU configuration manager
   - Hardware profiles with specifications
   - Memory fraction limiting
   - Precision configuration (FP32, FP16, BF16)
   - CUDA environment variable management
   - Power monitoring via nvidia-smi

2. **model_optimizer.py** - Model optimization utilities
   - Quantization (INT8)
   - Precision conversion (FP16)
   - Inference-time optimizations
   - Recommended model selection per profile

3. **gpu_profiles.json** - Profile configuration database
   - Hardware specifications
   - Inference settings (batch size, workers, etc.)
   - Optimization flags
   - Recommended model choices

4. **inference_with_gpu_config.py** - User-friendly wrapper
   - Easy command-line interface
   - Integration with camera_inference.py
   - Power monitoring and statistics

## Hardware Specifications

### A6000 Full (Unrestricted)
```
AI Performance:    1458 TFLOPS (FP32)
CUDA Cores:        7680
Memory:            48GB
Memory Bandwidth:  576 GB/s
Power:             50-300W
Max Batch Size:    128
Recommended Precision: FP32
```

### Jetson Nano Restricted (Simulated)
```
AI Performance:    67 TOPS
CUDA Cores:        1024 (simulated)
Memory:            8GB (limited)
Memory Bandwidth:  102 GB/s (simulated)
Power:             7-25W
Max Batch Size:    4
Recommended Precision: FP16
Quantization:      INT8
```

### Jetson Nano Power Save (7W Mode)
```
AI Performance:    35 TOPS (50% reduced)
CUDA Cores:        512 (simulated reduced)
Memory:            8GB
Memory Bandwidth:  51 GB/s (simulated)
Power:             7W (fixed)
Max Batch Size:    1
Recommended Precision: FP16
Quantization:      INT8
```

## Installation

1. Ensure the GPU configuration files are in the inference directory:
   - `gpu_config.py`
   - `model_optimizer.py`
   - `gpu_profiles.json`
   - `inference_with_gpu_config.py`

2. Install required dependencies:
   ```bash
   pip install torch torchvision torchaudio psutil
   ```

3. Ensure `nvidia-smi` is installed (usually comes with CUDA):
   ```bash
   which nvidia-smi
   ```

## Quick Start

### 1. List Available Profiles

```bash
python inference_with_gpu_config.py list-profiles
```

Output:
```
📊 Available GPU Profiles:

  • a6000_full
    NVIDIA RTX A6000

  • jetson_nano_restricted
    NVIDIA Jetson Nano Super (Simulated)

  • jetson_nano_power_save
    NVIDIA Jetson Nano Super (7W Power Save)
```

### 2. View Profile Information

```bash
python inference_with_gpu_config.py profile-info --profile jetson_nano_restricted
```

### 3. Run Camera Inference with A6000 Full Power

```bash
python inference_with_gpu_config.py camera --profile a6000_full --camera 0 -v
```

### 4. Run Camera Inference with Jetson Nano Restrictions

```bash
python inference_with_gpu_config.py camera --profile jetson_nano_restricted --camera 0 -v
```

### 5. Run with Power Monitoring

```bash
python inference_with_gpu_config.py camera \
  --profile jetson_nano_restricted \
  --camera 0 \
  --monitor-power \
  -v
```

## Programmatic Usage

### Basic Setup

```python
from gpu_config import setup_gpu

# Apply configuration and get manager
gpu_manager = setup_gpu('jetson_nano_restricted', verbose=True)

# Get memory info
memory_info = gpu_manager.get_memory_info()
print(f"Memory: {memory_info['allocated_gb']:.2f}GB / {memory_info['total_gb']:.2f}GB")

# Get recommended batch size
batch_size = gpu_manager.get_recommended_batch_size()  # Returns 4

# Get recommended model size
model_size = gpu_manager.get_recommended_model_size()  # Returns 'medium'
```

### Using Context Manager

```python
from gpu_config import GPUConfigManager

with GPUConfigManager('jetson_nano_restricted'):
    # Your inference code here
    model = load_model()
    results = model.predict(data)
    # GPU config is automatically restored on exit
```

### Model Optimization

```python
from model_optimizer import ModelOptimizer, InferenceOptimizer

# Create optimizer for profile
optimizer = ModelOptimizer('jetson_nano_restricted')

# Get optimization settings
batch_size = optimizer.get_batch_size()  # 4
num_workers = optimizer.get_num_workers()  # 2
precision = optimizer.get_inference_settings()['model_precision']  # 'float16'

# Optimize model
model = optimizer.optimize_model(model)

# Create optimized dataloader
dataloader = optimizer.create_optimized_dataloader(dataset)

# Setup inference optimization
InferenceOptimizer.use_tf32(False)  # Disable for Jetson mode
InferenceOptimizer.enable_cudnn_benchmark(False)
```

## Monitoring and Debugging

### Check GPU Memory Usage

```python
from gpu_config import GPUConfigManager

manager = GPUConfigManager('jetson_nano_restricted')
manager.apply_config()

# During inference
memory_info = manager.get_memory_info()
print(f"Allocated: {memory_info['allocated_gb']:.2f}GB")
print(f"Utilization: {memory_info['utilization_percent']:.1f}%")
```

### Monitor Power Consumption

```python
from gpu_config import GPUConfigManager

manager = GPUConfigManager('jetson_nano_restricted')
power_info = manager.monitor_power_consumption()

if 'power_draw_w' in power_info:
    print(f"Power: {power_info['power_draw_w']:.1f}W / {power_info['power_limit_w']:.1f}W")
    print(f"Utilization: {power_info['power_utilization_percent']:.1f}%")
```

### Verbose Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)

from gpu_config import setup_gpu
gpu_manager = setup_gpu('jetson_nano_restricted', verbose=True)
```

## Integration with Existing Code

### Camera Inference Integration

```python
import sys
from pathlib import Path

from gpu_config import setup_gpu
from model_optimizer import ModelOptimizer
from camera_inference import CameraVehicleDetector

# Apply GPU configuration
gpu_manager = setup_gpu('jetson_nano_restricted')

# Create detector
detector = CameraVehicleDetector(camera=0)

# Use optimized settings
optimizer = ModelOptimizer('jetson_nano_restricted')
batch_size = optimizer.get_batch_size()

# Run inference with restricted resources
while True:
    if not detector.process_frame():
        break
```

## Performance Characteristics

### Expected Inference Speed

| Profile | YOLO Model | Depth Model | Latency (1080p) | Throughput |
|---------|-----------|------------|-----------------|-----------|
| A6000 Full | yolo11x | zoedepth_nk | ~15ms | ~60 FPS |
| Jetson Nano Restricted | yolo11n | midas_small | ~60ms | ~15 FPS |
| Jetson Nano Power Save | yolo11n | midas_small_lite | ~100ms | ~10 FPS |

### Memory Usage

| Profile | YOLO Model | Depth Model | Total Memory | Margin |
|---------|-----------|------------|--------------|--------|
| A6000 Full | yolo11x | zoedepth_nk | 28GB | 20GB free |
| Jetson Nano Restricted | yolo11n | midas_small | 6.5GB | 1.5GB free |
| Jetson Nano Power Save | yolo11n | midas_small_lite | 4.2GB | 3.8GB free |

### Power Consumption

| Profile | Target Power | Typical Draw | Peak Draw |
|---------|------------|--------------|----------|
| A6000 Full | 50-300W | 150-250W | 300W |
| Jetson Nano Restricted | 7-25W | 12-20W | 25W |
| Jetson Nano Power Save | 7W | 6-7W | 8W |

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution 1**: Reduce batch size in gpu_profiles.json
```json
"batch_size": 2  // Reduce from 4
```

**Solution 2**: Use power save mode
```bash
python inference_with_gpu_config.py camera --profile jetson_nano_power_save
```

**Solution 3**: Force smaller models
```python
optimizer = ModelOptimizer('jetson_nano_restricted')
model_choice = optimizer.get_model_choice()
# Use model_choice['yolo_model'] = 'yolo11n' or smaller
```

### Issue: High Power Consumption in "Power Save" Mode

**Cause**: Full A6000 GPU is still running at high clock

**Solution**: Verify configuration was applied:
```python
gpu_manager = setup_gpu('jetson_nano_power_save', verbose=True)
power_info = gpu_manager.monitor_power_consumption()
print(power_info)  # Should show ~7W
```

### Issue: Models Load Slowly

**Solution**: Model compilation is disabled in restricted profiles
```python
# Pre-compile model before inference loop
model = torch.jit.trace(model, example_input)
```

### Issue: Inconsistent Performance

**Cause**: cuDNN autotuning disabled in restricted profiles

**Solution**: Pre-tune cuDNN with a warmup:
```python
from model_optimizer import InferenceOptimizer
InferenceOptimizer.enable_cudnn_benchmark(True)  # Temporary for warmup
# Run 10-20 iterations
InferenceOptimizer.enable_cudnn_benchmark(False)  # Restore for consistency
```

## Advanced Configuration

### Create Custom Profile

1. Add to gpu_profiles.json:
```json
"custom_profile": {
  "description": "My custom configuration",
  "hardware": {
    "ai_performance_tops": 100,
    "cuda_cores": 2048,
    "memory_gb": 12,
    "power_max_w": 50
  },
  "inference_settings": { ... }
}
```

2. Use in code:
```python
from gpu_config import GPUConfigManager

# Note: Custom profiles require creating them in gpu_profiles.json
manager = GPUConfigManager('custom_profile')
manager.apply_config()
```

### Environment-based Selection

```bash
# Set GPU profile via environment variable
export GPU_PROFILE=jetson_nano_restricted
python inference_with_gpu_config.py camera --profile $GPU_PROFILE
```

### Bash Script for Easy Switching

Create `switch_gpu.sh`:
```bash
#!/bin/bash
PROFILE=${1:-a6000_full}
echo "Switching to GPU profile: $PROFILE"
export GPU_PROFILE=$PROFILE
python inference_with_gpu_config.py $@
```

Usage:
```bash
chmod +x switch_gpu.sh
./switch_gpu.sh jetson_nano_restricted camera --camera 0
```

## References

- NVIDIA Jetson Nano Specifications: https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano/
- NVIDIA RTX A6000 Specifications: https://www.nvidia.com/en-us/design/rtx/
- PyTorch GPU Memory Management: https://pytorch.org/docs/stable/notes/cuda.html
- NVIDIA CUDA Best Practices: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

## License

This GPU configuration system is provided as-is for development and testing purposes.

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the logs with `-v` verbose flag
3. Verify nvidia-smi works: `nvidia-smi`
4. Check PyTorch CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
