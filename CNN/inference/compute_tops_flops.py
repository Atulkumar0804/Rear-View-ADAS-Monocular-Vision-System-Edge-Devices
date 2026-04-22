#!/usr/bin/env python3
"""
TOPS and FLOPs Calculator for Camera Inference Pipeline
========================================================

Calculates computational requirements (FLOPs/TOPS) for all models in the
camera_inference.py pipeline and recommends suitable NVIDIA Jetson hardware.

Usage:
    python compute_tops_flops.py
    python compute_tops_flops.py --target-fps 30
    python compute_tops_flops.py --precision fp16
    python compute_tops_flops.py --save-report results.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# ============================================================================
# MODEL SPECIFICATIONS
# ============================================================================

# YOLO11 Models (Ultralytics)
YOLO_MODELS = {
    'yolo11n': {
        'name': 'YOLOv11 Nano',
        'parameters': 2.6e6,  # 2.6M parameters
        'input_size': (640, 640, 3),
        'gflops_fp32': 6.5,  # GFLOPs per inference
        'type': 'object_detection',
        'output': 'boxes',
    },
    'yolo11x-seg': {
        'name': 'YOLOv11 Extra-Large with Segmentation',
        'parameters': 58.0e6,  # 58M parameters
        'input_size': (640, 640, 3),
        'gflops_fp32': 344.1,  # GFLOPs per inference (seg adds ~10% overhead)
        'type': 'object_detection_segmentation',
        'output': 'boxes + masks',
    },
}

# YOLOv11 Classifier
CLASSIFIER_MODEL = {
    'yolo11m-cls': {
        'name': 'YOLOv11 Medium Classifier',
        'parameters': 10.0e6,  # 10M parameters (EfficientNet-based)
        'input_size': (224, 224, 3),
        'gflops_fp32': 3.9,  # GFLOPs per crop classification
        'type': 'classification',
        'output': '14 classes',
    }
}

# ZoeDepth Model (Intel ISL)
ZOEDEPTH_MODEL = {
    'zoedepth': {
        'name': 'ZoeDepth (ZoeD_K)',
        'parameters': 345.0e6,  # 345M parameters (ViT-B backbone + DPT)
        'input_size': (384, 768, 3),  # Typical automotive resolution
        'gflops_fp32': 150.0,  # Estimated GFLOPs (Vision Transformer heavy)
        'type': 'depth_estimation',
        'output': 'metric depth map',
        'note': 'Runs async every N frames (default: 30)',
    }
}

# Additional Processing (Classical CV)
CLASSICAL_CV = {
    'pinhole_calculation': {
        'name': 'Pinhole Camera Distance',
        'gflops_fp32': 0.001,  # Negligible (simple arithmetic)
        'type': 'classical_cv',
    },
    'kalman_filtering': {
        'name': 'Kalman Temporal Filtering',
        'gflops_fp32': 0.0005,  # Per track
        'type': 'classical_cv',
    },
    'tracking_iou': {
        'name': 'IoU-based Tracking',
        'gflops_fp32': 0.002,  # Per detection
        'type': 'classical_cv',
    }
}

# ============================================================================
# JETSON HARDWARE SPECIFICATIONS
# ============================================================================

JETSON_DEVICES = {
    'nano': {
        'name': 'NVIDIA Jetson Nano',
        'compute_capability': '5.3',
        'gpu_cores': 128,
        'gpu_tflops_fp32': 0.472,  # 472 GFLOPS
        'gpu_tflops_fp16': 0.472,  # No Tensor Cores
        'gpu_tops_int8': 0.472,  # No INT8 acceleration
        'cpu_cores': 4,
        'ram_gb': 4,
        'power_watts': 10,
        'price_usd': 99,
        'notes': 'Entry-level, limited for complex pipelines',
    },
    'xavier_nx': {
        'name': 'NVIDIA Jetson Xavier NX',
        'compute_capability': '7.2',
        'gpu_cores': 384,
        'gpu_tflops_fp32': 21.0,
        'gpu_tflops_fp16': 42.0,  # With Tensor Cores
        'gpu_tops_int8': 84.0,
        'cpu_cores': 6,
        'ram_gb': 8,
        'power_watts': 15,
        'price_usd': 399,
        'notes': 'Good balance of performance and power',
    },
    'orin_nano': {
        'name': 'NVIDIA Jetson Orin Nano',
        'compute_capability': '8.7',
        'gpu_cores': 1024,
        'gpu_tflops_fp32': 40.0,
        'gpu_tflops_fp16': 80.0,  # With Tensor Cores
        'gpu_tops_int8': 160.0,
        'cpu_cores': 6,
        'ram_gb': 8,
        'power_watts': 15,
        'price_usd': 499,
        'notes': 'Best performance for modern edge AI',
    },
    'orin_nx': {
        'name': 'NVIDIA Jetson Orin NX',
        'compute_capability': '8.7',
        'gpu_cores': 1024,
        'gpu_tflops_fp32': 70.0,
        'gpu_tflops_fp16': 140.0,
        'gpu_tops_int8': 280.0,
        'cpu_cores': 8,
        'ram_gb': 16,
        'power_watts': 25,
        'price_usd': 699,
        'notes': 'High-performance option',
    },
    'agx_orin': {
        'name': 'NVIDIA Jetson AGX Orin',
        'compute_capability': '8.7',
        'gpu_cores': 2048,
        'gpu_tflops_fp32': 275.0,
        'gpu_tflops_fp16': 550.0,
        'gpu_tops_int8': 1100.0,
        'cpu_cores': 12,
        'ram_gb': 64,
        'power_watts': 60,
        'price_usd': 1999,
        'notes': 'Maximum performance for robotics',
    },
}

# ============================================================================
# PRECISION MULTIPLIERS
# ============================================================================

PRECISION_SPEEDUP = {
    'fp32': 1.0,
    'fp16': 2.0,  # 2x faster with Tensor Cores
    'int8': 4.0,  # 4x faster with INT8 quantization
}

# ============================================================================
# FLOPS CALCULATOR
# ============================================================================

class FLOPsCalculator:
    """Calculate FLOPs and TOPS for the entire inference pipeline"""
    
    def __init__(self, target_fps: int = 30, precision: str = 'fp32',
                 zoedepth_interval: int = 30, avg_detections: int = 5):
        """
        Args:
            target_fps: Target frames per second
            precision: Precision format ('fp32', 'fp16', 'int8')
            zoedepth_interval: Run ZoeDepth every N frames
            avg_detections: Average number of detections per frame
        """
        self.target_fps = target_fps
        self.precision = precision
        self.zoedepth_interval = zoedepth_interval
        self.avg_detections = avg_detections
        self.speedup = PRECISION_SPEEDUP[precision]
        
    def calculate_model_flops(self, model_config: dict, runs_per_frame: float = 1.0) -> dict:
        """Calculate FLOPs for a single model"""
        gflops_fp32 = model_config['gflops_fp32']
        gflops_effective = gflops_fp32 / self.speedup
        
        # FLOPs per frame
        flops_per_frame = gflops_effective * runs_per_frame * 1e9
        
        # FLOPs per second at target FPS
        flops_per_second = flops_per_frame * self.target_fps
        
        # TOPS (Tera Operations Per Second)
        tops = flops_per_second / 1e12
        
        return {
            'gflops_fp32': gflops_fp32,
            'gflops_effective': gflops_effective,
            'runs_per_frame': runs_per_frame,
            'flops_per_frame': flops_per_frame,
            'flops_per_second': flops_per_second,
            'tops': tops,
        }
    
    def calculate_pipeline_fast_mode(self) -> dict:
        """Calculate for fast mode (CPU/Edge): YOLO Nano only"""
        results = {
            'mode': 'fast',
            'models': {},
            'classical_cv': {},
            'total_flops_per_frame': 0,
            'total_tops': 0,
        }
        
        # YOLO11 Nano (detection)
        yolo_stats = self.calculate_model_flops(YOLO_MODELS['yolo11n'], runs_per_frame=1.0)
        results['models']['yolo11n'] = yolo_stats
        results['total_flops_per_frame'] += yolo_stats['flops_per_frame']
        
        # Classical CV operations
        pinhole_gflops = CLASSICAL_CV['pinhole_calculation']['gflops_fp32'] * self.avg_detections
        pinhole_flops = pinhole_gflops * 1e9
        results['classical_cv']['pinhole_distance'] = {
            'gflops_fp32': pinhole_gflops,
            'gflops_effective': pinhole_gflops / self.speedup,
            'runs_per_frame': self.avg_detections,
            'flops_per_frame': pinhole_flops,
            'tops': pinhole_flops * self.target_fps / 1e12,
        }
        results['total_flops_per_frame'] += pinhole_flops
        
        results['total_tops'] = results['total_flops_per_frame'] * self.target_fps / 1e12
        
        return results
    
    def calculate_pipeline_heavy_mode(self) -> dict:
        """Calculate for heavy mode (GPU): YOLO11x-seg + Classifier + ZoeDepth"""
        results = {
            'mode': 'heavy',
            'models': {},
            'classical_cv': {},
            'total_flops_per_frame': 0,
            'total_tops': 0,
        }
        
        # YOLO11n (detection only - segmentation removed)
        yolo_stats = self.calculate_model_flops(YOLO_MODELS['yolo11n'], runs_per_frame=1.0)
        results['models']['yolo11n'] = yolo_stats
        results['total_flops_per_frame'] += yolo_stats['flops_per_frame']
        
        # Classifier (runs per vehicle detection, not person/bicycle/motorcycle)
        # Assume ~60% of detections are vehicles needing classification
        classifier_runs = self.avg_detections * 0.6
        classifier_stats = self.calculate_model_flops(
            CLASSIFIER_MODEL['yolo11m-cls'], 
            runs_per_frame=classifier_runs
        )
        results['models']['yolo11m-cls'] = classifier_stats
        results['total_flops_per_frame'] += classifier_stats['flops_per_frame']
        
        # ZoeDepth (runs intermittently - amortize cost)
        zoedepth_runs = 1.0 / self.zoedepth_interval
        zoedepth_stats = self.calculate_model_flops(
            ZOEDEPTH_MODEL['zoedepth'],
            runs_per_frame=zoedepth_runs
        )
        results['models']['zoedepth'] = zoedepth_stats
        results['total_flops_per_frame'] += zoedepth_stats['flops_per_frame']
        
        # Classical CV operations (tracking, Kalman, pinhole)
        # Pinhole distance calculation
        pinhole_gflops = CLASSICAL_CV['pinhole_calculation']['gflops_fp32'] * self.avg_detections
        pinhole_flops = pinhole_gflops * 1e9
        results['classical_cv']['pinhole_distance'] = {
            'gflops_fp32': pinhole_gflops,
            'gflops_effective': pinhole_gflops / self.speedup,
            'runs_per_frame': self.avg_detections,
            'flops_per_frame': pinhole_flops,
            'tops': pinhole_flops * self.target_fps / 1e12,
        }
        results['total_flops_per_frame'] += pinhole_flops
        
        # Kalman filtering (per tracked object)
        kalman_gflops = CLASSICAL_CV['kalman_filtering']['gflops_fp32'] * self.avg_detections
        kalman_flops = kalman_gflops * 1e9
        results['classical_cv']['kalman_filter'] = {
            'gflops_fp32': kalman_gflops,
            'gflops_effective': kalman_gflops / self.speedup,
            'runs_per_frame': self.avg_detections,
            'flops_per_frame': kalman_flops,
            'tops': kalman_flops * self.target_fps / 1e12,
        }
        results['total_flops_per_frame'] += kalman_flops
        
        # IoU tracking (per detection for matching)
        tracking_gflops = CLASSICAL_CV['tracking_iou']['gflops_fp32'] * self.avg_detections
        tracking_flops = tracking_gflops * 1e9
        results['classical_cv']['iou_tracking'] = {
            'gflops_fp32': tracking_gflops,
            'gflops_effective': tracking_gflops / self.speedup,
            'runs_per_frame': self.avg_detections,
            'flops_per_frame': tracking_flops,
            'tops': tracking_flops * self.target_fps / 1e12,
        }
        results['total_flops_per_frame'] += tracking_flops
        
        results['total_tops'] = results['total_flops_per_frame'] * self.target_fps / 1e12
        
        return results
        results['total_tops'] = results['total_flops_per_frame'] * self.target_fps / 1e12
        
        return results
    
    def recommend_jetson(self, required_tops: float, mode: str) -> List[dict]:
        """Recommend suitable Jetson devices based on required TOPS"""
        recommendations = []
        
        # Get appropriate compute metric based on precision
        if self.precision == 'fp32':
            metric_key = 'gpu_tflops_fp32'
        elif self.precision == 'fp16':
            metric_key = 'gpu_tflops_fp16'
        else:  # int8
            metric_key = 'gpu_tops_int8'
        
        for device_id, device in JETSON_DEVICES.items():
            # Get TOPS directly (already in TOPS/TFLOPS)
            available_tops = device[metric_key]  # All values already in TOPS/TFLOPS
            
            # Calculate utilization (leave ~30% headroom for OS and other tasks)
            effective_tops = available_tops * 0.7
            utilization = (required_tops / effective_tops) * 100 if effective_tops > 0 else float('inf')
            
            can_run = utilization <= 100
            is_recommended = 50 <= utilization <= 80  # Sweet spot
            
            recommendations.append({
                'device': device['name'],
                'device_id': device_id,
                'available_tops': available_tops,
                'effective_tops': effective_tops,
                'required_tops': required_tops,
                'utilization_percent': utilization,
                'can_run': can_run,
                'recommended': is_recommended,
                'expected_fps': min(self.target_fps, (effective_tops / required_tops) * self.target_fps),
                'price_usd': device['price_usd'],
                'power_watts': device['power_watts'],
                'notes': device['notes'],
            })
        
        # Sort by utilization (closer to 70% is better)
        recommendations.sort(key=lambda x: abs(x['utilization_percent'] - 70))
        
        return recommendations


# ============================================================================
# REPORT GENERATOR
# ============================================================================

def generate_report(calculator: FLOPsCalculator) -> dict:
    """Generate comprehensive TOPS/FLOPs report"""
    
    print("\n" + "="*80)
    print("TOPS/FLOPs CALCULATOR FOR CAMERA INFERENCE PIPELINE")
    print("="*80)
    print(f"Configuration:")
    print(f"  • Target FPS: {calculator.target_fps}")
    print(f"  • Precision: {calculator.precision.upper()}")
    print(f"  • ZoeDepth Interval: Every {calculator.zoedepth_interval} frames")
    print(f"  • Average Detections: {calculator.avg_detections} per frame")
    print(f"  • Speedup Factor: {calculator.speedup}x (vs FP32)")
    print("="*80 + "\n")
    
    # Calculate both modes
    fast_mode = calculator.calculate_pipeline_fast_mode()
    heavy_mode = calculator.calculate_pipeline_heavy_mode()
    
    report = {
        'config': {
            'target_fps': calculator.target_fps,
            'precision': calculator.precision,
            'zoedepth_interval': calculator.zoedepth_interval,
            'avg_detections': calculator.avg_detections,
        },
        'fast_mode': fast_mode,
        'heavy_mode': heavy_mode,
        'recommendations': {},
    }
    
    # Display Fast Mode
    print("🚀 FAST MODE (CPU/Edge Devices)")
    print("-" * 80)
    print(f"Models: YOLO11 Nano + Classical CV")
    print(f"Total FLOPs per frame: {fast_mode['total_flops_per_frame']/1e9:.4f} GFLOPs")
    print(f"Required TOPS @ {calculator.target_fps} FPS: {fast_mode['total_tops']:.4f} TOPS")
    print()
    
    print("  Deep Learning Models:")
    for model_name, stats in fast_mode['models'].items():
        print(f"  • {model_name}:")
        print(f"      GFLOPs (FP32): {stats['gflops_fp32']:.4f}")
        print(f"      GFLOPs (Effective): {stats['gflops_effective']:.4f}")
        print(f"      Runs per frame: {stats['runs_per_frame']:.2f}")
        print(f"      TOPS contribution: {stats['tops']:.6f}")
    
    print("\n  Classical CV Operations:")
    for cv_name, stats in fast_mode['classical_cv'].items():
        print(f"  • {cv_name}:")
        print(f"      GFLOPs (FP32): {stats['gflops_fp32']:.6f}")
        print(f"      GFLOPs (Effective): {stats['gflops_effective']:.6f}")
        print(f"      Runs per frame: {stats['runs_per_frame']:.2f}")
        print(f"      TOPS contribution: {stats['tops']:.8f}")
    print()
    
    # Display Heavy Mode
    print("⚡ HEAVY MODE (GPU High-Accuracy)")
    print("-" * 80)
    print(f"Models: {' + '.join(heavy_mode['models'].keys())} + Classical CV")
    print(f"Total FLOPs per frame: {heavy_mode['total_flops_per_frame']/1e9:.4f} GFLOPs")
    print(f"Required TOPS @ {calculator.target_fps} FPS: {heavy_mode['total_tops']:.4f} TOPS")
    print()
    
    print("  Deep Learning Models:")
    for model_name, stats in heavy_mode['models'].items():
        print(f"  • {model_name}:")
        print(f"      GFLOPs (FP32): {stats['gflops_fp32']:.4f}")
        print(f"      GFLOPs (Effective): {stats['gflops_effective']:.4f}")
        print(f"      Runs per frame: {stats['runs_per_frame']:.2f}")
        print(f"      TOPS contribution: {stats['tops']:.6f}")
    
    print("\n  Classical CV Operations:")
    for cv_name, stats in heavy_mode['classical_cv'].items():
        print(f"  • {cv_name}:")
        print(f"      GFLOPs (FP32): {stats['gflops_fp32']:.6f}")
        print(f"      GFLOPs (Effective): {stats['gflops_effective']:.6f}")
        print(f"      Runs per frame: {stats['runs_per_frame']:.2f}")
        print(f"      TOPS contribution: {stats['tops']:.8f}")
    print()
    
    # Jetson Recommendations
    print("="*80)
    print("JETSON DEVICE RECOMMENDATIONS")
    print("="*80 + "\n")
    
    # Display precision capabilities for all devices
    print("🔧 JETSON DEVICE PRECISION CAPABILITIES")
    print("-" * 80)
    print(f"{'Device':<30} {'FP32 TOPS':<15} {'FP16 TOPS':<15} {'INT8 TOPS':<15} {'Price':<10}")
    print("-" * 80)
    for device_id, device in JETSON_DEVICES.items():
        print(f"{device['name']:<30} {device['gpu_tflops_fp32']:<15.2f} {device['gpu_tflops_fp16']:<15.2f} {device['gpu_tops_int8']:<15.2f} ${device['price_usd']:<9}")
    print("\n")
    
    # Fast Mode Recommendations
    print("📱 FAST MODE - Suitable Devices:")
    print("-" * 80)
    fast_recommendations = calculator.recommend_jetson(fast_mode['total_tops'], 'fast')
    report['recommendations']['fast_mode'] = fast_recommendations
    
    for i, rec in enumerate(fast_recommendations[:3], 1):
        status = "✅ RECOMMENDED" if rec['recommended'] else ("✓ Can Run" if rec['can_run'] else "❌ Too Slow")
        print(f"\n{i}. {rec['device']} - {status}")
        print(f"   Available TOPS ({calculator.precision.upper()}): {rec['available_tops']:.2f}")
        print(f"   Effective TOPS (70% util): {rec['effective_tops']:.2f}")
        print(f"   Utilization: {rec['utilization_percent']:.1f}%")
        print(f"   Expected FPS: {rec['expected_fps']:.1f}")
        print(f"   Price: ${rec['price_usd']}")
        print(f"   Power: {rec['power_watts']}W")
        print(f"   Notes: {rec['notes']}")
    
    # Heavy Mode Recommendations
    print("\n\n🎯 HEAVY MODE - Suitable Devices:")
    print("-" * 80)
    heavy_recommendations = calculator.recommend_jetson(heavy_mode['total_tops'], 'heavy')
    report['recommendations']['heavy_mode'] = heavy_recommendations
    
    for i, rec in enumerate(heavy_recommendations[:3], 1):
        status = "✅ RECOMMENDED" if rec['recommended'] else ("✓ Can Run" if rec['can_run'] else "❌ Too Slow")
        print(f"\n{i}. {rec['device']} - {status}")
        print(f"   Available TOPS ({calculator.precision.upper()}): {rec['available_tops']:.2f}")
        print(f"   Effective TOPS (70% util): {rec['effective_tops']:.2f}")
        print(f"   Utilization: {rec['utilization_percent']:.1f}%")
        print(f"   Expected FPS: {rec['expected_fps']:.1f}")
        print(f"   Price: ${rec['price_usd']}")
        print(f"   Power: {rec['power_watts']}W")
        print(f"   Notes: {rec['notes']}")
    
    # Summary
    print("\n" + "="*80)
    print("⚙️ PRECISION OPTIMIZATION ANALYSIS (Heavy Mode)")
    print("="*80)
    print(f"\nRequired TOPS at different precisions for {calculator.target_fps} FPS:")
    print("-" * 80)
    
    # Calculate for all precisions
    for precision, speedup in PRECISION_SPEEDUP.items():
        calc_temp = FLOPsCalculator(calculator.target_fps, precision, calculator.zoedepth_interval, calculator.avg_detections)
        heavy_temp = calc_temp.calculate_pipeline_heavy_mode()
        tops_required = heavy_temp['total_tops']
        
        print(f"\n{precision.upper()} Precision:")
        print(f"  Required TOPS: {tops_required:.3f}")
        print(f"  Speedup vs FP32: {speedup}x")
        print(f"  Suitable Devices:")
        
        # Show which devices can handle this precision
        if precision == 'fp32':
            metric_key = 'gpu_tflops_fp32'
        elif precision == 'fp16':
            metric_key = 'gpu_tflops_fp16'
        else:
            metric_key = 'gpu_tops_int8'
        
        suitable_count = 0
        for device_id, device in JETSON_DEVICES.items():
            available = device[metric_key]
            effective = available * 0.7
            util = (tops_required / effective) * 100 if effective > 0 else float('inf')
            
            if util <= 100:
                suitable_count += 1
                status = "✅ Optimal" if 50 <= util <= 80 else "✓ Works"
                print(f"    {status} {device['name']:<30} {available:>7.2f} TOPS ({util:>5.1f}% util)")
        
        if suitable_count == 0:
            print(f"    ❌ No suitable devices found for {precision.upper()}")
    
    # Summary
    print("\n" + "="*80)
    print("💡 FINAL RECOMMENDATIONS")
    print("="*80)
    
    # Best for Fast Mode
    best_fast = fast_recommendations[0]
    if best_fast['recommended']:
        print(f"\n✅ FAST MODE (Edge Deployment):")
        print(f"   Recommended: {best_fast['device']}")
        print(f"   Expected Performance: {best_fast['expected_fps']:.1f} FPS @ ${best_fast['price_usd']}")
    
    # Best for Heavy Mode
    best_heavy = heavy_recommendations[0]
    if best_heavy['recommended']:
        print(f"\n✅ HEAVY MODE (High Accuracy):")
        print(f"   Recommended: {best_heavy['device']}")
        print(f"   Expected Performance: {best_heavy['expected_fps']:.1f} FPS @ ${best_heavy['price_usd']}")
    
    # Optimization Tips
    print("\n\n📈 OPTIMIZATION TIPS:")
    print("-" * 80)
    print("1. Use FP16 precision (2x speedup with Tensor Cores on Orin/Xavier)")
    print("2. Use INT8 quantization (4x speedup, minimal accuracy loss)")
    print("3. Increase ZoeDepth interval (e.g., 60 frames) for lower TOPS")
    print("4. Use TensorRT to optimize models (20-40% speedup)")
    print("5. Reduce input resolution for YOLO (e.g., 480x480 instead of 640x640)")
    print("6. Batch detections for classifier inference (if available)")
    print("="*80 + "\n")
    
    return report


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Calculate TOPS/FLOPs for Camera Inference Pipeline'
    )
    parser.add_argument('--target-fps', type=int, default=30,
                       help='Target frames per second (default: 30)')
    parser.add_argument('--precision', type=str, default='fp32',
                       choices=['fp32', 'fp16', 'int8'],
                       help='Precision format (default: fp32)')
    parser.add_argument('--zoedepth-interval', type=int, default=30,
                       help='Run ZoeDepth every N frames (default: 30)')
    parser.add_argument('--avg-detections', type=int, default=5,
                       help='Average detections per frame (default: 5)')
    parser.add_argument('--save-report', type=str, default=None,
                       help='Save report to JSON file')
    
    args = parser.parse_args()
    
    # Create calculator
    calculator = FLOPsCalculator(
        target_fps=args.target_fps,
        precision=args.precision,
        zoedepth_interval=args.zoedepth_interval,
        avg_detections=args.avg_detections,
    )
    
    # Generate report
    report = generate_report(calculator)
    
    # Save report if requested
    if args.save_report:
        output_path = Path(args.save_report)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Report saved to: {output_path}")


if __name__ == '__main__':
    main()
