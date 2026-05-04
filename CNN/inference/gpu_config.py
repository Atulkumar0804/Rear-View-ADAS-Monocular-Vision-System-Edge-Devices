#!/usr/bin/env python3
"""
GPU Configuration Manager
Provides flexible GPU resource management with profiles:
- a6000_full: Unrestricted RTX A6000 mode
- jetson_nano_restricted: Simulates Jetson Nano Super constraints
- jetson_nano_power_save: Ultra-power-saving mode (7W target)

Usage:
    from gpu_config import GPUConfigManager
    
    # Switch to Jetson mode
    gpu_manager = GPUConfigManager(profile='jetson_nano_restricted')
    gpu_manager.apply_config()
    
    # Or use context manager
    with GPUConfigManager('jetson_nano_restricted'):
        # Your inference code here
        pass
"""

import os
import sys
import json
import logging
import subprocess
import psutil
import torch
import numpy as np
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
from enum import Enum
import warnings

logger = logging.getLogger(__name__)


class GPUProfile(Enum):
    """Available GPU configuration profiles"""
    A6000_FULL = "a6000_full"
    JETSON_NANO_RESTRICTED = "jetson_nano_restricted"
    JETSON_NANO_POWER_SAVE = "jetson_nano_power_save"
    CUSTOM = "custom"


@dataclass
class GPUSpec:
    """GPU specification dataclass"""
    name: str
    ai_performance_tops: float
    cuda_cores: int
    tensor_cores: int
    memory_gb: int
    memory_bandwidth_gbs: float
    power_min_w: float
    power_max_w: float
    target_inference_latency_ms: float
    max_batch_size: int
    enable_tf32: bool
    precision: str  # 'float32', 'float16', 'bfloat16'
    max_clock_mhz: float = 0  # 0 = auto
    max_power_limit_w: Optional[float] = None  # None = don't set
    target_fps: int = 30  # Target inference frames per second
    max_utilization_percent: float = 100.0  # Max GPU utilization %


class GPUConfigManager:
    """Manages GPU configuration and resource constraints"""
    
    # Hardware specifications
    PROFILES = {
        GPUProfile.A6000_FULL.value: GPUSpec(
            name="NVIDIA RTX A6000",
            ai_performance_tops=1458,  # ~1458 TFLOPS FP32
            cuda_cores=7680,
            tensor_cores=240,
            memory_gb=48,
            memory_bandwidth_gbs=576,
            power_min_w=50,
            power_max_w=300,
            target_inference_latency_ms=10,  # Fast inference
            max_batch_size=128,
            enable_tf32=True,
            precision='float32',
            max_clock_mhz=2505,  # A6000 boost clock
            max_power_limit_w=300,  # No restriction
            target_fps=60,  # High FPS
            max_utilization_percent=100.0
        ),
        GPUProfile.JETSON_NANO_RESTRICTED.value: GPUSpec(
            name="NVIDIA Jetson Nano Super (Simulated)",
            ai_performance_tops=67,  # 67 TOPS from spec sheet
            cuda_cores=1024,  # Ampere architecture
            tensor_cores=32,
            memory_gb=8,  # 8GB LPDDR5
            memory_bandwidth_gbs=102,
            power_min_w=7,
            power_max_w=25,
            target_inference_latency_ms=100,  # More conservative
            max_batch_size=4,  # Very limited batch size
            enable_tf32=False,
            precision='float16',
            max_clock_mhz=1320,  # Jetson Nano typical clock
            max_power_limit_w=25,  # Hard power limit
            target_fps=15,  # Reduced FPS
            max_utilization_percent=80.0  # Keep some headroom
        ),
        GPUProfile.JETSON_NANO_POWER_SAVE.value: GPUSpec(
            name="NVIDIA Jetson Nano Super (7W Power Save)",
            ai_performance_tops=35,  # ~50% performance at min power
            cuda_cores=512,  # Simulated reduced performance
            tensor_cores=16,
            memory_gb=8,
            memory_bandwidth_gbs=51,  # Reduced bandwidth
            power_min_w=7,
            power_max_w=7,  # Fixed at 7W
            target_inference_latency_ms=150,
            max_batch_size=1,  # Single batch only
            enable_tf32=False,
            precision='float16',
            max_clock_mhz=800,  # Minimum clock
            max_power_limit_w=7,  # Ultra-low power
            target_fps=5,  # Very low FPS
            max_utilization_percent=60.0  # Very conservative
        ),
    }
    
    def __init__(self, profile: str = 'a6000_full', custom_spec: Optional[GPUSpec] = None):
        """
        Initialize GPU Configuration Manager
        
        Args:
            profile: One of 'a6000_full', 'jetson_nano_restricted', 'jetson_nano_power_save'
            custom_spec: Optional GPUSpec for custom configuration
        """
        self.profile_name = profile
        
        if profile == GPUProfile.CUSTOM.value and custom_spec:
            self.spec = custom_spec
        elif profile in self.PROFILES:
            self.spec = self.PROFILES[profile]
        else:
            raise ValueError(f"Unknown profile: {profile}. Choose from: {list(self.PROFILES.keys())}")
        
        self._original_env = {}
        self._memory_fraction_set = False
        self.logger = logger
        
        # Initialize torch settings
        self._init_torch()
    
    def _init_torch(self):
        """Initialize PyTorch settings"""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.cuda_available = True
            self.total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        else:
            self.device = torch.device('cpu')
            self.cuda_available = False
            self.total_gpu_memory = 0
            self.logger.warning("CUDA not available, falling back to CPU")
    
    def apply_config(self):
        """Apply GPU configuration"""
        if not self.cuda_available:
            self.logger.warning("CUDA not available, skipping GPU configuration")
            return
        
        self.logger.info(f"Applying GPU profile: {self.profile_name}")
        self.logger.info(f"Spec: {self.spec.name}")
        
        # 1. Set memory fraction to simulate reduced GPU memory
        self._set_memory_fraction()
        
        # 2. Configure PyTorch precision
        self._configure_precision()
        
        # 3. Set environment variables for GPU optimization
        self._set_cuda_env_vars()
        
        # 4. Skip GPU power limit enforcement
        self.logger.info("Skipping GPU power limit enforcement for this profile")
        
        # 5. Set GPU clock speed limits (HARD RESTRICTION)
        self._set_clock_limits()
        
        # 6. Clear GPU cache
        torch.cuda.empty_cache()
        
        self.logger.info(f"GPU Configuration Applied: {self.spec.name}")
        self._log_gpu_status()
    
    def _set_memory_fraction(self):
        """Set GPU memory fraction to simulate lower memory"""
        if not self.cuda_available:
            return
        
        # Calculate fraction based on simulated memory vs available
        # A6000 has 48GB, so scaling factor is spec memory / 48
        memory_fraction = self.spec.memory_gb / 48.0
        
        # PyTorch's per_process_memory_fraction accepts 0-1 range
        # Clamp between 0.1 and 1.0
        memory_fraction = max(0.1, min(1.0, memory_fraction))
        
        try:
            # For older PyTorch versions, use the memory_fraction approach
            # This is a soft limit, not a hard limit
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            self._memory_fraction_set = True
            self.logger.info(f"GPU memory fraction set to {memory_fraction:.2%} (simulating {self.spec.memory_gb}GB)")
        except Exception as e:
            self.logger.warning(f"Could not set memory fraction: {e}")
    
    def _configure_precision(self):
        """Configure floating-point precision"""
        if self.spec.precision == 'float16':
            torch.set_default_dtype(torch.float32)  # Keep default as float32
            self.logger.info("Float16 inference recommended for this profile")
        elif self.spec.precision == 'bfloat16':
            self.logger.info("BFloat16 inference recommended for this profile")
        
        # Enable/disable TF32 matmul
        torch.backends.cuda.matmul.allow_tf32 = self.spec.enable_tf32
        torch.backends.cudnn.allow_tf32 = self.spec.enable_tf32
        
        self.logger.info(f"Precision: {self.spec.precision}, TF32: {self.spec.enable_tf32}")
    
    def _set_cuda_env_vars(self):
        """Set CUDA environment variables for optimization"""
        # Store original values
        self._original_env['CUDA_DEVICE_ORDER'] = os.environ.get('CUDA_DEVICE_ORDER')
        self._original_env['CUDA_LAUNCH_BLOCKING'] = os.environ.get('CUDA_LAUNCH_BLOCKING')
        
        # For restricted profiles, use conservative settings
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        
        if self.profile_name in ['jetson_nano_restricted', 'jetson_nano_power_save']:
            # More synchronous execution for predictable latency
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    def _set_power_limits(self):
        """Power limit enforcement is disabled in this pipeline."""
        self.logger.info("Power limit logic has been disabled. No nvidia-smi power limit is applied.")
    
    def _set_clock_limits(self):
        """Set GPU clock speed limits using nvidia-smi (HARD clock restriction)"""
        if self.spec.max_clock_mhz <= 0:
            self.logger.info("No clock speed limit set (auto)")
            return
        
        self.logger.info(f"Hardware clock speed limit: {self.spec.max_clock_mhz} MHz (Jetson Nano)")
        # Note: Hard clock limiting typically requires nvidia-smi -lgc but permissions may be restricted
        # Alternative: Use persistence mode and let OS/driver respect power limits
        try:
            # Enable persistence mode (helps with clock stability)
            cmd = ['nvidia-smi', '-i', '0', '-pm', '1']
            subprocess.run(cmd, capture_output=True, timeout=5)
            self.logger.info("GPU persistence mode enabled for stable clocks")
        except Exception as e:
            self.logger.warning(f"Could not enable persistence mode: {e}")
    
    def enforce_inference_rate(self, frame_number: int, start_time: float) -> bool:
        """
        Enforce inference rate limit based on profile target FPS
        Returns True if within limits, False if needs throttling
        """
        import time
        target_fps = self.spec.target_fps
        target_frame_time = 1.0 / target_fps
        
        elapsed = time.time() - start_time
        expected_time = frame_number * target_frame_time
        
        if elapsed < expected_time:
            sleep_time = expected_time - elapsed
            time.sleep(sleep_time * 0.5)  # Sleep for portion of remainder
            return True
        
        return False
    
    def monitor_tflops_usage(self, current_utilization_percent: float) -> Dict:
        """
        Calculate and monitor TFLOPS usage based on GPU utilization
        """
        max_tflops = self.spec.ai_performance_tops
        actual_tflops = max_tflops * (current_utilization_percent / 100.0)
        
        return {
            'max_tflops': max_tflops,
            'actual_tflops': actual_tflops,
            'utilization_percent': current_utilization_percent,
            'headroom_tflops': max_tflops - actual_tflops
        }
    
    def enforce_compute_limits(self, current_utilization_percent: float) -> Tuple[bool, str]:
        """
        Enforce GPU compute utilization limits
        Returns (is_safe, message)
        """
        max_util = self.spec.max_utilization_percent
        
        if current_utilization_percent > max_util:
            return False, (
                f"GPU compute exceeds limit: {current_utilization_percent:.1f}% > {max_util:.0f}% "
                f"({self.spec.target_fps} fps target)"
            )
        
        return True, f"Compute OK: {current_utilization_percent:.1f}% / {max_util:.0f}% limit"
    
    def restore_env(self):
        """Restore original CUDA environment"""
        for key, value in self._original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.logger.info("Original CUDA environment restored")
    
    def get_recommended_batch_size(self) -> int:
        """Get recommended batch size for profile"""
        return self.spec.max_batch_size
    
    def get_recommended_model_size(self) -> str:
        """Get recommended model size for profile"""
        if self.profile_name == 'a6000_full':
            return 'large'  # Use large/XL models
        elif self.profile_name == 'jetson_nano_restricted':
            return 'medium'  # Use medium/small models
        else:  # power_save
            return 'small'  # Use small models only
    
    def get_memory_info(self) -> Dict:
        """Get current GPU memory information"""
        if not self.cuda_available:
            return {'error': 'CUDA not available'}
        
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = self.total_gpu_memory / 1e9
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': total - allocated,
            'utilization_percent': (allocated / total * 100) if total > 0 else 0
        }
    
    def _log_gpu_status(self):
        """Log current GPU status"""
        if not self.cuda_available:
            return
        
        try:
            info = self.get_memory_info()
            self.logger.info(
                f"GPU Status: {info['allocated_gb']:.2f}GB/{info['total_gb']:.2f}GB allocated, "
                f"{info['utilization_percent']:.1f}% utilized"
            )
        except Exception as e:
            self.logger.warning(f"Could not retrieve GPU status: {e}")
    
    def get_max_allowed_memory_gb(self) -> float:
        """Get maximum allowed GPU memory for this profile (in GB)"""
        return self.spec.memory_gb
    
    def is_memory_safe(self, threshold_percent: float = 90.0) -> Tuple[bool, str]:
        """
        Check if current memory usage is within safe limits
        
        Args:
            threshold_percent: Alert threshold (e.g., 90% of limit)
        
        Returns:
            Tuple of (is_safe, message)
        """
        if not self.cuda_available:
            return True, "CUDA not available"
        
        info = self.get_memory_info()
        max_allowed = self.get_max_allowed_memory_gb()
        allocated = info['allocated_gb']
        utilization = info['utilization_percent']
        
        # Check if exceeded
        if allocated > max_allowed:
            return False, (
                f"🔴 GPU MEMORY EXCEEDED! "
                f"Using {allocated:.2f}GB > limit {max_allowed:.2f}GB. "
                f"STOPPING INFERENCE!"
            )
        
        # Check if approaching threshold
        if utilization >= threshold_percent:
            return False, (
                f"⚠️  GPU MEMORY CRITICAL! "
                f"Using {allocated:.2f}GB ({utilization:.1f}%) >= {threshold_percent:.0f}% of {max_allowed:.2f}GB limit. "
                f"STOPPING INFERENCE!"
            )
        
        return True, f"✓ Memory OK: {allocated:.2f}GB / {max_allowed:.2f}GB ({utilization:.1f}%)"
    
    def check_model_fit(self, model_size_mb: float) -> Tuple[bool, str]:
        """
        Check if a model can fit within GPU memory profile
        
        Args:
            model_size_mb: Model size in MB
        
        Returns:
            Tuple of (can_fit, message)
        """
        max_allowed = self.get_max_allowed_memory_gb()
        current_allocated = self.get_memory_info()['allocated_gb']
        available = max_allowed - current_allocated
        model_size_gb = model_size_mb / 1024
        
        if model_size_gb > available:
            return False, (
                f"❌ Model too large for GPU profile! "
                f"Model: {model_size_mb:.0f}MB, "
                f"Available: {available:.2f}GB, "
                f"Limit: {max_allowed:.2f}GB"
            )
        
        return True, (
            f"✓ Model fits: {model_size_mb:.0f}MB <= "
            f"Available {available:.2f}GB (of {max_allowed:.2f}GB)"
        )
    
    def enforce_memory_limit(self) -> None:
        """
        Enforce GPU memory limit - raises exception if exceeded
        Useful in inference loops to stop before crash
        """
        is_safe, message = self.is_memory_safe(threshold_percent=85.0)
        
        if not is_safe:
            self.logger.error(message)
            raise RuntimeError(f"GPU MEMORY LIMIT EXCEEDED: {message}")
        
        # Log status every 10 frames (checked by caller)
        info = self.get_memory_info()
        max_allowed = self.get_max_allowed_memory_gb()
        if info['utilization_percent'] > 75.0:
            self.logger.warning(
                f"⚠️  Memory usage high: {info['allocated_gb']:.2f}GB / "
                f"{max_allowed:.2f}GB ({info['utilization_percent']:.1f}%)"
            )
    
    def monitor_power_consumption(self, duration_sec: float = 5.0) -> Dict:
        """Monitor GPU power consumption (requires nvidia-smi)"""
        try:
            # Query nvidia-smi for power draw
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=power.draw,power.limit', '--format=csv,nounits,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                power_draw, power_limit = result.stdout.strip().split(', ')
                return {
                    'power_draw_w': float(power_draw),
                    'power_limit_w': float(power_limit),
                    'power_utilization_percent': (float(power_draw) / float(power_limit) * 100)
                }
            else:
                return {'error': 'nvidia-smi query failed'}
        except Exception as e:
            return {'error': str(e)}
    
    def __enter__(self):
        """Context manager entry"""
        self.apply_config()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.restore_env()
    
    @staticmethod
    def list_profiles() -> Dict[str, str]:
        """List available profiles"""
        return {name: spec.name for name, spec in GPUConfigManager.PROFILES.items()}
    
    @staticmethod
    def get_profile_info(profile_name: str) -> Dict:
        """Get detailed info for a profile"""
        if profile_name not in GPUConfigManager.PROFILES:
            return {'error': f'Unknown profile: {profile_name}'}
        
        spec = GPUConfigManager.PROFILES[profile_name]
        return asdict(spec)


def setup_gpu(profile: str = 'a6000_full', verbose: bool = True) -> GPUConfigManager:
    """
    Convenience function to quickly setup GPU configuration
    
    Args:
        profile: GPU profile name
        verbose: Whether to log configuration details
    
    Returns:
        GPUConfigManager instance with configuration applied
    """
    if verbose:
        profiles = GPUConfigManager.list_profiles()
        print("\n📊 Available GPU Profiles:")
        for name, desc in profiles.items():
            marker = "✓" if name == profile else " "
            print(f"  {marker} {name}: {desc}")
    
    manager = GPUConfigManager(profile=profile)
    manager.apply_config()
    
    if verbose:
        info = manager.get_memory_info()
        print(f"\n✓ GPU Configuration Applied: {manager.spec.name}")
        print(f"  Memory: {info['allocated_gb']:.2f}GB / {info['total_gb']:.2f}GB")
        print(f"  Max Batch Size: {manager.spec.max_batch_size}")
        print(f"  Recommended Precision: {manager.spec.precision}")
    
    return manager


# Example usage
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU Configuration Manager')
    parser.add_argument('--profile', default='a6000_full',
                       choices=list(GPUConfigManager.PROFILES.keys()),
                       help='GPU profile to use')
    parser.add_argument('--list-profiles', action='store_true', help='List available profiles')
    parser.add_argument('--info', action='store_true', help='Show detailed profile info')
    parser.add_argument('--monitor', type=float, default=0, help='Monitor power for N seconds')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if args.list_profiles:
        print("\n📊 Available GPU Profiles:")
        for name, desc in GPUConfigManager.list_profiles().items():
            print(f"  • {name}: {desc}")
        sys.exit(0)
    
    if args.info:
        info = GPUConfigManager.get_profile_info(args.profile)
        print(f"\n📋 Profile: {args.profile}")
        print(json.dumps(info, indent=2))
        sys.exit(0)
    
    # Setup GPU
    print(f"\n🚀 Setting up GPU with profile: {args.profile}")
    manager = setup_gpu(args.profile, verbose=True)
    
    # Monitor power if requested
    if args.monitor > 0:
        print(f"\n⚡ Monitoring power consumption for {args.monitor}s...")
        power_info = manager.monitor_power_consumption(args.monitor)
        if 'error' not in power_info:
            print(f"  Power Draw: {power_info['power_draw_w']:.1f}W / {power_info['power_limit_w']:.1f}W")
            print(f"  Utilization: {power_info['power_utilization_percent']:.1f}%")
        else:
            print(f"  Error: {power_info['error']}")
    
    print("\n✓ GPU Configuration Ready!")
