#!/usr/bin/env python3
"""
Model Optimizer for GPU Profiles
Provides utilities to load and optimize models based on GPU profile constraints.
Includes model quantization, pruning, and precision conversion.

Usage:
    from model_optimizer import ModelOptimizer
    
    optimizer = ModelOptimizer(profile='jetson_nano_restricted')
    model = optimizer.load_yolo_model()
    model = optimizer.optimize_model(model, target_latency_ms=100)
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import json

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """Optimizes models based on GPU profile constraints"""
    
    def __init__(self, profile: str = 'a6000_full', config_path: Optional[str] = None):
        """
        Initialize Model Optimizer
        
        Args:
            profile: GPU profile ('a6000_full', 'jetson_nano_restricted', 'jetson_nano_power_save')
            config_path: Path to gpu_profiles.json config file
        """
        self.profile = profile
        self.config_path = config_path or Path(__file__).parent / 'gpu_profiles.json'
        
        # Load configuration
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            self.profile_config = self.config['profiles'].get(profile)
            if not self.profile_config:
                raise ValueError(f"Unknown profile: {profile}")
        except Exception as e:
            logger.warning(f"Could not load config: {e}, using defaults")
            self.profile_config = {}
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def get_inference_settings(self) -> Dict:
        """Get inference settings for profile"""
        return self.profile_config.get('inference_settings', {})
    
    def get_optimization_settings(self) -> Dict:
        """Get optimization settings for profile"""
        return self.profile_config.get('optimization', {})
    
    def get_model_choice(self) -> Dict:
        """Get recommended model sizes for profile"""
        return self.profile_config.get('model_choice', {})
    
    def convert_to_fp16(self, model: nn.Module) -> nn.Module:
        """Convert model to float16 for memory efficiency"""
        if next(model.parameters()).dtype != torch.float16:
            model = model.half()
            logger.info(f"Model converted to Float16")
        return model
    
    def quantize_model_int8(self, model: nn.Module) -> nn.Module:
        """Prepare model for INT8 quantization"""
        try:
            # Post-training static quantization
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare_qat(model, inplace=True)
            logger.info("Model prepared for INT8 quantization")
        except Exception as e:
            logger.warning(f"Could not quantize model: {e}")
        return model
    
    def optimize_model(self, model: nn.Module, target_latency_ms: Optional[float] = None) -> nn.Module:
        """
        Apply optimization techniques based on profile
        
        Args:
            model: PyTorch model to optimize
            target_latency_ms: Optional target inference latency
        
        Returns:
            Optimized model
        """
        opt_settings = self.get_optimization_settings()
        
        # Precision conversion for restricted profiles
        if opt_settings.get('mixed_precision', False):
            logger.info("Applying mixed precision (FP16/FP32)")
            model = self.convert_to_fp16(model)
        
        # INT8 quantization for power-constrained profiles
        if opt_settings.get('quantization') == 'int8':
            logger.info("Preparing INT8 quantization")
            model = self.quantize_model_int8(model)
        
        # Fused operations (shouldn't hurt, helps with A6000 too)
        if opt_settings.get('fused_operations', True):
            try:
                torch.jit.fuse(model) if hasattr(model, '_apply') else None
                logger.info("Applied fused operations")
            except Exception as e:
                logger.debug(f"Fused operations not available: {e}")
        
        # Model compilation for newer PyTorch versions
        if opt_settings.get('compile_model', False):
            try:
                if hasattr(torch, 'compile'):
                    model = torch.compile(model, mode='reduce-overhead')
                    logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.debug(f"Model compilation not available: {e}")
        
        model = model.to(self.device)
        return model
    
    def get_batch_size(self) -> int:
        """Get recommended batch size"""
        return self.get_inference_settings().get('batch_size', 1)
    
    def get_num_workers(self) -> int:
        """Get recommended number of dataloader workers"""
        return self.get_inference_settings().get('num_workers', 0)
    
    def should_pin_memory(self) -> bool:
        """Get whether to pin memory in dataloader"""
        return self.get_inference_settings().get('pin_memory', False)
    
    def get_prefetch_factor(self) -> int:
        """Get dataloader prefetch factor"""
        return self.get_inference_settings().get('prefetch_factor', 0)
    
    def create_optimized_dataloader(self, dataset, **kwargs) -> torch.utils.data.DataLoader:
        """Create optimized DataLoader based on profile"""
        from torch.utils.data import DataLoader
        
        dataloader_config = {
            'batch_size': self.get_batch_size(),
            'num_workers': self.get_num_workers(),
            'pin_memory': self.should_pin_memory(),
            'prefetch_factor': self.get_prefetch_factor() if self.get_num_workers() > 0 else 0,
        }
        
        # Override with any provided kwargs
        dataloader_config.update(kwargs)
        
        return DataLoader(dataset, **dataloader_config)
    
    def log_profile_summary(self):
        """Log optimization profile summary"""
        logger.info(f"\n{'='*60}")
        logger.info(f"GPU Profile: {self.profile}")
        logger.info(f"{'='*60}")
        
        hardware = self.profile_config.get('hardware', {})
        logger.info(f"Hardware:")
        logger.info(f"  AI Performance: {hardware.get('ai_performance_tops', '?')} TOPS")
        logger.info(f"  Memory: {hardware.get('memory_gb', '?')}GB")
        logger.info(f"  Max Power: {hardware.get('power_max_w', '?')}W")
        
        inference = self.get_inference_settings()
        logger.info(f"Inference Settings:")
        logger.info(f"  Batch Size: {inference.get('batch_size', '?')}")
        logger.info(f"  Precision: {inference.get('model_precision', '?')}")
        logger.info(f"  Mixed Precision: {inference.get('mixed_precision', False)}")
        
        models = self.get_model_choice()
        logger.info(f"Recommended Models:")
        logger.info(f"  YOLO: {models.get('yolo_model', '?')}")
        logger.info(f"  Depth: {models.get('depth_model', '?')}")
        logger.info(f"{'='*60}\n")


class InferenceOptimizer:
    """Helper for inference-time optimizations"""
    
    @staticmethod
    def enable_cudnn_benchmark(enabled: bool = True):
        """Enable/disable cuDNN autotuning"""
        torch.backends.cudnn.benchmark = enabled
        logger.info(f"cuDNN benchmark: {enabled}")
    
    @staticmethod
    def enable_cudnn_deterministic(enabled: bool = False):
        """Enable/disable deterministic cuDNN behavior (slower but reproducible)"""
        torch.backends.cudnn.deterministic = enabled
        logger.info(f"cuDNN deterministic: {enabled}")
    
    @staticmethod
    def set_float32_matmul_precision(precision: str = 'highest'):
        """
        Set float32 matmul precision
        
        Args:
            precision: 'highest' (default), 'high', or 'medium'
        """
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision(precision)
            logger.info(f"Float32 matmul precision: {precision}")
    
    @staticmethod
    def use_tf32(enabled: bool = True):
        """Enable/disable TF32 operations"""
        torch.backends.cuda.matmul.allow_tf32 = enabled
        torch.backends.cudnn.allow_tf32 = enabled
        logger.info(f"TF32 operations: {enabled}")
    
    @staticmethod
    def set_memory_efficient_attention(enabled: bool = True):
        """Enable memory-efficient attention if available"""
        try:
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                # Already available in PyTorch 2.0+
                logger.info(f"Memory-efficient attention available")
        except Exception as e:
            logger.debug(f"Memory-efficient attention not available: {e}")


if __name__ == '__main__':
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', default='a6000_full', help='GPU profile')
    parser.add_argument('--config', help='Path to gpu_profiles.json')
    args = parser.parse_args()
    
    optimizer = ModelOptimizer(profile=args.profile, config_path=args.config)
    optimizer.log_profile_summary()
    
    print(f"Inference Settings: {optimizer.get_inference_settings()}")
    print(f"Optimization Settings: {optimizer.get_optimization_settings()}")
    print(f"Model Choice: {optimizer.get_model_choice()}")
