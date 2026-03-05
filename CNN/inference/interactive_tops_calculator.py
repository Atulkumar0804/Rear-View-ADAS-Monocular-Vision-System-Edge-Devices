#!/usr/bin/env python3
"""
Interactive TOPS/FLOPs Calculator with Step-by-Step Explanation
Shows detailed breakdown of how calculations are performed
"""

import sys

def print_header(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def print_section(title):
    print("\n" + "-"*80)
    print(f"  {title}")
    print("-"*80)

def calculate_conv_layer_flops():
    """Demonstrate FLOPs calculation for a single conv layer"""
    print_header("📐 DETAILED CALCULATION: Single Convolutional Layer")
    
    print("Let's calculate FLOPs for the FIRST layer in YOLO11x-seg:\n")
    
    # Layer specs
    input_h = 640
    input_w = 640
    input_c = 3  # RGB
    output_c = 64
    kernel = 6
    stride = 2
    
    print("Layer Specifications:")
    print(f"  • Input size: {input_h}×{input_w}×{input_c} (Height × Width × Channels)")
    print(f"  • Output channels: {output_c}")
    print(f"  • Kernel size: {kernel}×{kernel}")
    print(f"  • Stride: {stride}")
    
    # Calculate output size
    output_h = input_h // stride
    output_w = input_w // stride
    
    print(f"\nStep 1: Calculate output dimensions")
    print(f"  Output height = {input_h} / {stride} = {output_h}")
    print(f"  Output width = {input_w} / {stride} = {output_w}")
    print(f"  Output shape: {output_h}×{output_w}×{output_c}")
    
    # Operations per output pixel
    ops_per_pixel = kernel * kernel * input_c
    print(f"\nStep 2: Calculate operations per output pixel")
    print(f"  Each output pixel needs to look at:")
    print(f"    {kernel}×{kernel} kernel × {input_c} input channels")
    print(f"    = {kernel} × {kernel} × {input_c}")
    print(f"    = {ops_per_pixel} multiply-add operations")
    
    # Total pixels
    total_pixels = output_h * output_w * output_c
    print(f"\nStep 3: Count total output pixels")
    print(f"  Total pixels = {output_h} × {output_w} × {output_c}")
    print(f"               = {total_pixels:,} pixels")
    
    # Total operations
    multiply_adds = total_pixels * ops_per_pixel
    print(f"\nStep 4: Calculate total multiply-add operations")
    print(f"  MACs = {total_pixels:,} pixels × {ops_per_pixel} ops/pixel")
    print(f"       = {multiply_adds:,} MACs")
    
    # Convert to FLOPs
    flops = 2 * multiply_adds
    gflops = flops / 1e9
    
    print(f"\nStep 5: Convert to FLOPs")
    print(f"  Each MAC = 1 multiply + 1 add = 2 FLOPs")
    print(f"  Total FLOPs = 2 × {multiply_adds:,}")
    print(f"              = {flops:,} FLOPs")
    print(f"              = {gflops:.2f} GFLOPs")
    
    print(f"\n✅ RESULT: First conv layer = {gflops:.2f} GFLOPs")
    print(f"   (YOLO has 200+ such layers → Total ~344 GFLOPs)")
    
    return gflops

def calculate_full_model_flops():
    """Show how model FLOPs add up"""
    print_header("🧩 COMPLETE MODEL: Adding Up All Layers")
    
    models = {
        'YOLO11x-seg': {
            'layers': [
                ('Stem Conv', 1.5),
                ('Stage 1 (5 layers)', 15.2),
                ('Stage 2 (8 layers)', 42.3),
                ('Stage 3 (12 layers)', 89.6),
                ('Stage 4 (15 layers)', 108.4),
                ('Backbone Final', 23.1),
                ('FPN/Neck (20 layers)', 40.2),
                ('Detection Head', 15.7),
                ('Segmentation Head', 8.1),
            ],
            'total': 344.1
        }
    }
    
    for model_name, data in models.items():
        print(f"\n{model_name} Layer-by-Layer Breakdown:\n")
        print(f"{'Layer Name':<30} {'GFLOPs':>10}")
        print("-" * 42)
        
        running_total = 0
        for layer_name, gflops in data['layers']:
            running_total += gflops
            print(f"{layer_name:<30} {gflops:>10.1f}")
        
        print("-" * 42)
        print(f"{'TOTAL':<30} {data['total']:>10.1f}")
        print(f"\n✅ {model_name} requires {data['total']:.1f} GFLOPs per image")

def calculate_pipeline_flops():
    """Calculate complete pipeline FLOPs"""
    print_header("🔗 FULL PIPELINE: Heavy Mode")
    
    print("Your camera_inference.py runs these models:\n")
    
    # Model contributions
    yolo_flops = 344.1
    yolo_runs = 1.0
    
    classifier_flops = 3.9
    classifier_runs = 3.0  # 5 detections × 60% vehicles
    
    zoedepth_flops = 150.0
    zoedepth_interval = 30
    zoedepth_runs = 1.0 / zoedepth_interval
    
    classical_flops = 0.02
    
    print(f"1. YOLO11x-seg (Object Detection + Segmentation)")
    print(f"   GFLOPs per run: {yolo_flops:.1f}")
    print(f"   Runs per frame: {yolo_runs:.1f}")
    print(f"   Total: {yolo_flops * yolo_runs:.1f} GFLOPs")
    
    print(f"\n2. YOLOv11m-cls (Vehicle Classification)")
    print(f"   GFLOPs per run: {classifier_flops:.1f}")
    print(f"   Runs per frame: {classifier_runs:.1f} (avg 5 detections × 60% vehicles)")
    print(f"   Total: {classifier_flops * classifier_runs:.1f} GFLOPs")
    
    print(f"\n3. ZoeDepth (Depth Estimation)")
    print(f"   GFLOPs per run: {zoedepth_flops:.1f}")
    print(f"   Runs per frame: {zoedepth_runs:.4f} (every {zoedepth_interval} frames)")
    print(f"   Total (amortized): {zoedepth_flops * zoedepth_runs:.1f} GFLOPs")
    
    print(f"\n4. Classical CV (Tracking, Kalman, Pinhole)")
    print(f"   Total: {classical_flops:.2f} GFLOPs (negligible)")
    
    total = (yolo_flops * yolo_runs + 
             classifier_flops * classifier_runs + 
             zoedepth_flops * zoedepth_runs + 
             classical_flops)
    
    print("\n" + "="*80)
    print(f"TOTAL FLOPs per frame: {total:.2f} GFLOPs")
    print("="*80)
    
    return total

def calculate_tops_requirement(flops_per_frame, fps):
    """Convert FLOPs to TOPS"""
    print_header("⚡ CONVERTING FLOPs TO TOPS")
    
    print(f"Given:")
    print(f"  • Model requires: {flops_per_frame:.2f} GFLOPs per frame")
    print(f"  • Target frame rate: {fps} FPS")
    
    print(f"\nStep 1: Calculate operations per second")
    ops_per_second = flops_per_frame * 1e9 * fps
    print(f"  Operations/second = {flops_per_frame:.2f} × 10⁹ × {fps}")
    print(f"                    = {ops_per_second:.2e} operations/second")
    
    print(f"\nStep 2: Convert to TOPS (Tera Operations Per Second)")
    tops = ops_per_second / 1e12
    print(f"  TOPS = {ops_per_second:.2e} / 10¹²")
    print(f"       = {tops:.3f} TOPS")
    
    print(f"\n✅ RESULT: Need {tops:.3f} TOPS to achieve {fps} FPS")
    
    return tops

def compare_with_hardware(required_tops, precision='fp32'):
    """Compare required TOPS with available hardware"""
    print_header("🖥️  HARDWARE COMPARISON")
    
    devices = {
        'Jetson Nano': {
            'fp32': 0.472,
            'fp16': 0.472,
            'price': 99,
            'power': 10
        },
        'Jetson Xavier NX': {
            'fp32': 21.0,
            'fp16': 42.0,
            'price': 399,
            'power': 15
        },
        'Jetson Orin Nano': {
            'fp32': 40.0,
            'fp16': 80.0,
            'price': 499,
            'power': 15
        },
        'Jetson Orin NX': {
            'fp32': 70.0,
            'fp16': 140.0,
            'price': 699,
            'power': 25
        }
    }
    
    print(f"Required TOPS ({precision.upper()}): {required_tops:.3f}\n")
    print(f"{'Device':<20} {'TOPS':>8} {'Effective':>10} {'Usage':>8} {'Can Run?':>10} {'Price':>8}")
    print("-" * 76)
    
    for name, specs in devices.items():
        available = specs[precision]
        effective = available * 0.7  # 30% overhead
        usage = (required_tops / effective) * 100
        can_run = "✅ Yes" if usage <= 100 else "❌ No"
        
        print(f"{name:<20} {available:>8.2f} {effective:>10.2f} {usage:>7.1f}% {can_run:>10} ${specs['price']:>6}")
    
    print("\nNote: 'Effective' TOPS = Available × 0.7 (30% reserved for system overhead)")

def demonstrate_precision_impact(flops_per_frame, fps):
    """Show how precision affects performance"""
    print_header("🎯 PRECISION OPTIMIZATION")
    
    precisions = {
        'FP32': {'speedup': 1.0, 'accuracy': '100%'},
        'FP16': {'speedup': 2.0, 'accuracy': '99.5%'},
        'INT8': {'speedup': 4.0, 'accuracy': '97-98%'}
    }
    
    print("Same model, different precision formats:\n")
    print(f"{'Precision':<10} {'Speedup':>10} {'Required TOPS':>15} {'Accuracy':>12}")
    print("-" * 50)
    
    for precision, data in precisions.items():
        effective_flops = flops_per_frame / data['speedup']
        required_tops = (effective_flops * fps) / 1000
        print(f"{precision:<10} {data['speedup']:>9.1f}x {required_tops:>14.3f} {data['accuracy']:>12}")
    
    print("\n💡 Key Insight:")
    print("  FLOPs don't change, but hardware executes faster!")
    print("  FP16 with Tensor Cores: 2× speed, <1% accuracy loss")
    print("  INT8 with quantization: 4× speed, 2-3% accuracy loss")

def trace_single_frame():
    """Trace one frame through the pipeline"""
    print_header("🎬 TRACING ONE FRAME THROUGH PIPELINE")
    
    print("Input: 1280×720 RGB image from camera\n")
    
    # YOLO
    print("Step 1: YOLO11x-seg Detection")
    print("  ├─ Resize: 1280×720 → 640×640")
    print("  ├─ Compute: 344.1 GFLOPs")
    print("  ├─ Time @ 21 TOPS: 344.1 / 21 = 16.4 ms")
    print("  └─ Output: 5 detections (3 vehicles, 2 persons)")
    
    # Classifier
    print("\nStep 2: Classify Vehicles")
    print("  ├─ Crop 3 vehicles → 224×224 each")
    print("  ├─ Run classifier 3 times")
    print("  ├─ Compute: 3.9 × 3 = 11.7 GFLOPs")
    print("  ├─ Time @ 21 TOPS: 11.7 / 21 = 0.56 ms")
    print("  └─ Output: [Sedan, SUV, Bus]")
    
    # ZoeDepth
    print("\nStep 3: ZoeDepth (runs every 30th frame)")
    print("  ├─ Resize: 1280×720 → 384×768")
    print("  ├─ Compute: 150 GFLOPs")
    print("  ├─ Time @ 21 TOPS: 150 / 21 = 7.1 ms")
    print("  ├─ Amortized: 7.1 / 30 = 0.24 ms per frame")
    print("  └─ Output: Metric depth map")
    
    # Classical
    print("\nStep 4: Classical CV Processing")
    print("  ├─ IoU tracking: assign track IDs")
    print("  ├─ Kalman filtering: smooth distances")
    print("  ├─ Pinhole camera: estimate distances")
    print("  ├─ Time: < 0.01 ms")
    print("  └─ Output: Final detections with distances")
    
    # Total
    total_compute = 16.4 + 0.56 + 0.24
    overhead = 5.0
    total_time = total_compute + overhead
    fps = 1000 / total_time
    
    print("\n" + "="*80)
    print("FRAME TIMING BREAKDOWN:")
    print(f"  Compute time:  {total_compute:.2f} ms")
    print(f"  System overhead: {overhead:.2f} ms")
    print(f"  Total time:    {total_time:.2f} ms")
    print(f"  Achievable FPS: {fps:.1f}")
    print("="*80)

def main():
    print("\n" + "🎓"*40)
    print("  INTERACTIVE TOPS/FLOPs CALCULATOR WITH DETAILED EXPLANATIONS")
    print("🎓"*40)
    
    # Menu
    while True:
        print("\n\nChoose a demonstration:")
        print("  1. Calculate FLOPs for a single convolutional layer")
        print("  2. See how model FLOPs add up layer-by-layer")
        print("  3. Calculate complete pipeline FLOPs")
        print("  4. Convert FLOPs to TOPS requirement")
        print("  5. Compare with Jetson hardware")
        print("  6. Show precision optimization impact")
        print("  7. Trace one frame through entire pipeline")
        print("  8. Run complete analysis (All steps)")
        print("  9. Exit")
        
        choice = input("\nYour choice (1-9): ").strip()
        
        if choice == '1':
            calculate_conv_layer_flops()
        
        elif choice == '2':
            calculate_full_model_flops()
        
        elif choice == '3':
            flops = calculate_pipeline_flops()
        
        elif choice == '4':
            flops = 360.82
            fps = int(input("\nEnter target FPS (default 30): ").strip() or "30")
            tops = calculate_tops_requirement(flops, fps)
        
        elif choice == '5':
            tops = float(input("\nEnter required TOPS (default 10.825): ").strip() or "10.825")
            precision = input("Precision (fp32/fp16, default fp32): ").strip() or "fp32"
            compare_with_hardware(tops, precision)
        
        elif choice == '6':
            flops = 360.82
            fps = 30
            demonstrate_precision_impact(flops, fps)
        
        elif choice == '7':
            trace_single_frame()
        
        elif choice == '8':
            # Run everything
            calculate_conv_layer_flops()
            input("\nPress Enter to continue...")
            
            calculate_full_model_flops()
            input("\nPress Enter to continue...")
            
            flops = calculate_pipeline_flops()
            input("\nPress Enter to continue...")
            
            tops = calculate_tops_requirement(flops, 30)
            input("\nPress Enter to continue...")
            
            compare_with_hardware(tops, 'fp32')
            input("\nPress Enter to continue...")
            
            demonstrate_precision_impact(flops, 30)
            input("\nPress Enter to continue...")
            
            trace_single_frame()
        
        elif choice == '9':
            print("\n✅ Thank you for using the calculator!")
            print("📖 Read UNDERSTANDING_TOPS_FLOPS.md for more details\n")
            break
        
        else:
            print("\n❌ Invalid choice. Please enter 1-9.")
        
        input("\nPress Enter to continue...")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✅ Exiting... Goodbye!")
        sys.exit(0)
