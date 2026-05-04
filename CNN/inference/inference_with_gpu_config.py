#!/usr/bin/env python3
"""
GPU-Aware Inference Wrapper
Supports camera inference and video inference with GPU profiles

This script:
1. Applies GPU configuration profile
2. Loads optimized models
3. Runs inference with selected constraints

Usage:
    # Camera inference with A6000 full performance
    python inference_with_gpu_config.py camera --profile a6000_full --camera 0
    
    # Camera inference with Jetson Nano restrictions
    python inference_with_gpu_config.py camera --profile jetson_nano_restricted --camera 0
    
    # Video inference with restrictions
    python inference_with_gpu_config.py video --profile jetson_nano_restricted --input video.mp4 --output result.mp4
    
    # List available profiles
    python inference_with_gpu_config.py list-profiles
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add parent directories to path
SCRIPT_DIR = Path(__file__).parent.resolve()
CNN_DIR = SCRIPT_DIR.parent
sys.path.append(str(CNN_DIR))

import cv2

from inference.gpu_config import GPUConfigManager, setup_gpu
from inference.model_optimizer import ModelOptimizer, InferenceOptimizer


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def print_profile_info(profile_name: str):
    """Print detailed information about a GPU profile"""
    info = GPUConfigManager.get_profile_info(profile_name)
    
    print(f"\n{'='*70}")
    print(f"GPU Profile: {profile_name}")
    print(f"{'='*70}")
    
    for key, value in info.items():
        if isinstance(value, dict):
            print(f"\n{key.upper()}:")
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue}")
        else:
            print(f"{key}: {value}")
    print(f"{'='*70}\n")


def run_camera_inference(
    profile: str = 'a6000_full',
    camera: int = 0,
    verbose: bool = False,
    monitor_power: bool = False,
    duration: Optional[int] = None,
):
    """
    Run camera inference with GPU configuration
    
    Args:
        profile: GPU profile to use
        camera: Camera index or video file path
        verbose: Verbose logging
        monitor_power: Monitor GPU power consumption
        duration: Run for N seconds (None = run until keyboard interrupt)
    """
    logger = setup_logging(verbose)
    
    logger.info(f"🚀 Starting Camera Inference with GPU Profile: {profile}")
    
    # Apply GPU configuration
    logger.info("Applying GPU configuration...")
    gpu_manager = setup_gpu(profile=profile, verbose=True)
    
    # Setup inference optimization
    opt_settings = ModelOptimizer(profile=profile).get_optimization_settings()
    InferenceOptimizer.enable_cudnn_benchmark(opt_settings.get('use_cudnn_autotuner', True))
    InferenceOptimizer.use_tf32(opt_settings.get('enable_tf32', False))
    
    logger.info(f"✓ GPU Configuration Applied")
    
    # Import camera inference
    try:
        from camera_inference import CameraVehicleDetector
        logger.info("✓ Camera inference module loaded")
    except ImportError as e:
        logger.error(f"Failed to import camera_inference: {e}")
        return False
    
    try:
        # Create detector with GPU profile
        logger.info("Initializing camera detector...")
        detector = CameraVehicleDetector(device='cuda')
        logger.info("✓ Camera detector initialized")
        
        # Get model recommendations for profile
        model_choice = ModelOptimizer(profile=profile).get_model_choice()
        logger.info(f"Using recommended models: {model_choice}")
        
        # Get batch size recommendation
        batch_size = ModelOptimizer(profile=profile).get_batch_size()
        logger.info(f"Batch size: {batch_size}")
        
        # Open camera/video source
        try:
            camera_id = int(camera)
            cap = cv2.VideoCapture(camera_id)
        except (ValueError, TypeError):
            cap = cv2.VideoCapture(camera)
        
        if not cap.isOpened():
            logger.error(f"Failed to open camera: {camera}")
            logger.warning("No physical camera found. Switching to demo video file...")
            
            # Try to find a test video file
            test_videos = [
                Path(CNN_DIR) / "testing_data" / "IISc _Road.mp4",
                Path(CNN_DIR) / "testing_data" / "Qualcomm_Rear.mp4",
                Path(CNN_DIR) / "testing_data" / "relative_speed_100.mp4",
            ]
            
            video_found = None
            for test_video in test_videos:
                if test_video.exists():
                    video_found = test_video
                    break
            
            if video_found:
                logger.info(f"\n✓ Found test video: {video_found.name}")
                logger.info("Running video inference instead...\n")
                # Fallback to video inference with same GPU profile
                return run_video_inference(
                    profile=profile,
                    input_video=str(video_found),
                    output_video=None,
                    verbose=verbose,
                    monitor_power=monitor_power
                )
            else:
                logger.error("❌ No camera connected and no test video files found")
                logger.info("Please provide a video file or connect a physical camera")
                return False
        
        logger.info(f"✓ Camera opened")
        
        # Main inference loop
        logger.info("\n▶ Starting inference (Press 'q' to exit)")
        logger.info(f"🔒 GPU LIMITS ENFORCED:")
        logger.info(f"   Memory: {gpu_manager.get_max_allowed_memory_gb():.0f}GB")
        logger.info(f"   Power: {gpu_manager.spec.max_power_limit_w:.0f}W")
        logger.info(f"   Clock: {gpu_manager.spec.max_clock_mhz} MHz")
        logger.info(f"   TFLOPS: {gpu_manager.spec.ai_performance_tops:.0f} TFLOPS")
        logger.info(f"   Compute: {gpu_manager.spec.max_utilization_percent:.0f}%")
        logger.info(f"   FPS Target: {gpu_manager.spec.target_fps} fps")
        
        start_time = None
        frame_count = 0
        import time as time_module
        inference_start = time_module.time()
        
        while True:
            try:
                # HARD MEMORY ENFORCEMENT - Check every 5 frames
                if frame_count % 5 == 0:
                    try:
                        gpu_manager.enforce_memory_limit()
                    except RuntimeError as mem_error:
                        logger.error(f"\n🛑 STOPPING INFERENCE: {mem_error}")
                        cap.release()
                        return False
                
                # HARD COMPUTE/UTILIZATION ENFORCEMENT - Check every 10 frames
                if frame_count % 10 == 0:
                    memory_info = gpu_manager.get_memory_info()
                    max_allowed = gpu_manager.get_max_allowed_memory_gb()
                    current_util = memory_info['utilization_percent']
                    
                    # Check compute limits
                    compute_ok, compute_msg = gpu_manager.enforce_compute_limits(current_util)
                    if not compute_ok:
                        logger.error(f"\n🛑 {compute_msg}")
                        logger.info("Stopping inference to maintain GPU constraints")
                        cap.release()
                        return False
                
                # Run inference
                ret, frame = cap.read()
                if not ret:
                    logger.info("No more frames")
                    break
                
                # Validate frame
                if frame is None or frame.size == 0:
                    logger.warning("Empty frame received, skipping")
                    continue
                
                if len(frame.shape) < 2:
                    logger.warning("Invalid frame shape, skipping")
                    continue
                
                detections = detector.detect_frame(frame)
                
                frame_count += 1
                
                # Draw detections on frame
                annotated = detector.draw_detections(frame, detections)
                if annotated is None:
                    annotated = frame.copy()
                    logger.warning("Annotation failed, using raw frame")
                
                # Ensure frame is displayable
                if len(annotated.shape) == 2:
                    annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
                
                # Overlay GPU stats on frame
                memory_info = gpu_manager.get_memory_info()
                max_allowed = gpu_manager.get_max_allowed_memory_gb()
                utilization = memory_info['utilization_percent']
                power_info = gpu_manager.monitor_power_consumption()
                
                # Add GPU stats to frame
                stats_text = [
                    f"GPU: {memory_info['allocated_gb']:.1f}GB/{max_allowed:.0f}GB ({utilization:.0f}%)",
                    f"TFLOPS: {gpu_manager.spec.ai_performance_tops * utilization/100:.0f}/{gpu_manager.spec.ai_performance_tops:.0f}",
                    f"Power: {power_info.get('power_draw_w', 0):.1f}W/{gpu_manager.spec.max_power_limit_w:.0f}W",
                    f"Detections: {len(detections)} | Profile: {profile}"
                ]
                
                # Draw text overlay on frame
                y_offset = 30
                for i, text in enumerate(stats_text):
                    color = (0, 255, 0) if utilization < 85 else (0, 165, 255) if utilization < 95 else (0, 0, 255)
                    cv2.putText(annotated, text, (10, y_offset + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Resize for display if too large
                display_frame = annotated
                if annotated.shape[0] > 1440:
                    scale = 1440 / annotated.shape[0]
                    display_frame = cv2.resize(annotated, (int(annotated.shape[1]*scale), 1440))
                
                # Display frame in fullscreen with GPU stats
                window_name = f'🚗 ADAS Detection - GPU: {profile} (q/ESC=quit, f=fullscreen)'
                try:
                    cv2.imshow(window_name, display_frame)
                    # Set fullscreen on first frame
                    if frame_count == 1:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                except Exception as display_error:
                    logger.error(f"Display error: {display_error}")
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q or ESC
                    logger.info("Exit requested by user")
                    break
                elif key == ord('f'):  # f for fullscreen toggle
                    current_state = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
                    new_state = cv2.WINDOW_NORMAL if current_state == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, new_state)
                
                # HARD FPS/INFERENCE RATE ENFORCEMENT
                gpu_manager.enforce_inference_rate(frame_count, inference_start)
                
                # Track duration if specified
                if duration:
                    if start_time is None:
                        import time
                        start_time = time.time()
                    
                    elapsed = time.time() - start_time
                    if elapsed > duration:
                        logger.info(f"Duration limit reached ({duration}s)")
                        break
                
                # Periodic status with all restrictions
                if frame_count % 30 == 0:
                    memory_info = gpu_manager.get_memory_info()
                    max_allowed = gpu_manager.get_max_allowed_memory_gb()
                    utilization = memory_info['utilization_percent']
                    
                    # Monitor TFLOPS
                    tflops_info = gpu_manager.monitor_tflops_usage(utilization)
                    
                    # Get power info
                    power_info = gpu_manager.monitor_power_consumption()
                    
                    if utilization > 85:
                        status = f"⚠️ Frame {frame_count} | Memory {memory_info['allocated_gb']:.2f}GB/{max_allowed:.0f}GB"
                    else:
                        status = f"✓ Frame {frame_count} | Memory {memory_info['allocated_gb']:.2f}GB/{max_allowed:.0f}GB"
                    
                    logger.info(status)
                    logger.info(f"   TFLOPS: {tflops_info['actual_tflops']:.0f}/{tflops_info['max_tflops']:.0f}")
                    logger.info(f"   Compute: {utilization:.1f}% / {gpu_manager.spec.max_utilization_percent:.0f}%")
                    logger.info(f"   Detections: {len(detections)}")
                    
                    if 'power_draw_w' in power_info:
                        logger.info(
                            f"   Power: {power_info['power_draw_w']:.1f}W / "
                            f"{gpu_manager.spec.max_power_limit_w:.0f}W limit"
                        )
            
            except KeyboardInterrupt:
                logger.info("\n⏹ Interrupted by user")
                break
            except RuntimeError as mem_error:
                if "GPU MEMORY" in str(mem_error):
                    logger.error(f"\n🛑 STOPPING INFERENCE: {mem_error}")
                    cap.release()
                    return False
                else:
                    logger.error(f"Error during inference: {mem_error}", exc_info=verbose)
                    cap.release()
                    return False
            except Exception as e:
                logger.error(f"Error during inference: {e}", exc_info=verbose)
                cap.release()
                return False
        
        cap.release()
        cv2.destroyAllWindows()
        logger.info(f"\n✓ Inference completed. Processed {frame_count} frames")
        return True
        
        logger.info(f"\n✓ Inference completed. Processed {frame_count} frames")
        return True
    
    except Exception as e:
        logger.error(f"Camera inference error: {e}", exc_info=verbose)
        return False


def run_video_inference(
    profile: str = 'a6000_full',
    input_video: Optional[str] = None,
    output_video: Optional[str] = None,
    verbose: bool = False,
    monitor_power: bool = False,
):
    """
    Run video inference with GPU configuration
    
    Args:
        profile: GPU profile to use
        input_video: Path to input video file
        output_video: Path to output video file (optional)
        verbose: Verbose logging
        monitor_power: Monitor GPU power consumption
    """
    logger = setup_logging(verbose)
    
    logger.info(f"🚀 Starting Video Inference with GPU Profile: {profile}")
    
    if not input_video:
        logger.error("Input video file required (--input VIDEO_PATH)")
        return False
    
    # Apply GPU configuration
    logger.info("Applying GPU configuration...")
    gpu_manager = setup_gpu(profile=profile, verbose=True)
    
    # Setup inference optimization
    opt_settings = ModelOptimizer(profile=profile).get_optimization_settings()
    InferenceOptimizer.enable_cudnn_benchmark(opt_settings.get('use_cudnn_autotuner', True))
    InferenceOptimizer.use_tf32(opt_settings.get('enable_tf32', False))
    
    logger.info(f"✓ GPU Configuration Applied")
    
    # Import video inference
    try:
        from video_inference import process_video
        logger.info("✓ Video inference module loaded")
    except ImportError as e:
        logger.error(f"Failed to import video_inference: {e}")
        return False
    
    try:
        # Get model recommendations for profile
        model_choice = ModelOptimizer(profile=profile).get_model_choice()
        logger.info(f"Using recommended models: {model_choice}")
        
        # Get batch size recommendation
        batch_size = ModelOptimizer(profile=profile).get_batch_size()
        logger.info(f"Batch size: {batch_size}")
        
        # Process video file with GPU configuration already applied
        output_file = output_video or f"{Path(input_video).stem}_detected.mp4"
        logger.info(f"\n▶ Processing video: {input_video}")
        logger.info(f"  Output: {output_file}")
        logger.info(f"  GPU Profile: {profile}")
        logger.info(f"\ud83d\udd12 GPU MEMORY LIMIT ENFORCED: {gpu_manager.get_max_allowed_memory_gb():.0f}GB")
        
        # Call process_video function
        success = process_video(
            input_path=input_video,
            output_path=output_file,
            device='cuda',
            zoedepth_interval=30,
            correction_alpha=0.3,
            alpha_lr=0.05,
            freeze_alpha=False
        )
        
        if success:
            logger.info(f"\n✓ Video processing completed successfully")
            logger.info(f"Output saved to: {output_file}")
        else:
            logger.error("Video processing failed")
        
        return success
    
    except Exception as e:
        logger.error(f"Video inference error: {e}", exc_info=verbose)
        return False





def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='GPU-Aware Inference Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Camera inference with full A6000 power
  %(prog)s camera --profile a6000_full --camera 0
  
  # Camera inference with Jetson Nano restrictions
  %(prog)s camera --profile jetson_nano_restricted --camera 0
  
  # Video inference with A6000 power
  %(prog)s video --profile a6000_full --input video.mp4 --output result.mp4
  
  # Video inference with Jetson Nano restrictions
  %(prog)s video --profile jetson_nano_restricted --input video.mp4 --output result.mp4
  
  
  # List all available profiles
  %(prog)s list-profiles
  
  # Show profile information
  %(prog)s profile-info --profile jetson_nano_restricted
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Camera inference command
    camera_parser = subparsers.add_parser('camera', help='Run camera inference')
    camera_parser.add_argument('--profile', default='a6000_full',
                              choices=list(GPUConfigManager.PROFILES.keys()),
                              help='GPU profile')
    camera_parser.add_argument('--camera', type=int, default=0, help='Camera index or video file')
    camera_parser.add_argument('--duration', type=int, help='Run for N seconds')
    camera_parser.add_argument('--monitor-power', action='store_true', help='Monitor power consumption')
    camera_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    
    # Video inference command
    video_parser = subparsers.add_parser('video', help='Run video file inference')
    video_parser.add_argument('--profile', default='a6000_full',
                             choices=list(GPUConfigManager.PROFILES.keys()),
                             help='GPU profile')
    video_parser.add_argument('--input', required=True, help='Input video file path')
    video_parser.add_argument('--output', help='Output video file path (auto-generated if not provided)')
    video_parser.add_argument('--monitor-power', action='store_true', help='Monitor power consumption')
    video_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    

    
    # List profiles command
    list_parser = subparsers.add_parser('list-profiles', help='List available profiles')
    
    # Profile info command
    info_parser = subparsers.add_parser('profile-info', help='Show profile information')
    info_parser.add_argument('--profile', default='a6000_full',
                            choices=list(GPUConfigManager.PROFILES.keys()),
                            help='Profile to display')
    
    args = parser.parse_args()
    
    # Handle commands
    if args.command == 'camera':
        run_camera_inference(
            profile=args.profile,
            camera=args.camera,
            duration=args.duration,
            monitor_power=args.monitor_power,
            verbose=args.verbose
        )
    
    elif args.command == 'video':
        run_video_inference(
            profile=args.profile,
            input_video=args.input,
            output_video=args.output,
            monitor_power=args.monitor_power,
            verbose=args.verbose
        )
    
    
    elif args.command == 'list-profiles':
        profiles = GPUConfigManager.list_profiles()
        print("\n📊 Available GPU Profiles:\n")
        for name, desc in profiles.items():
            print(f"  • {name}")
            print(f"    {desc}\n")
    
    elif args.command == 'profile-info':
        print_profile_info(args.profile)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
