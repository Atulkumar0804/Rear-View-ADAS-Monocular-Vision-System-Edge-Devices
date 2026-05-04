#!/bin/bash
#
# Main CNN Launcher - Easy access to all features with GPU Profile Selection
#

CNN_DIR="/home/atul/Desktop/atul/rear_view_adas_monocular/CNN"

# Prefer currently activated venv; fallback to project-local .venv.
if [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif [ -x "$CNN_DIR/.venv/bin/python" ]; then
    PYTHON="$CNN_DIR/.venv/bin/python"
else
    PYTHON="python3"
fi

echo "🐍 Using Python: $PYTHON"
export PYTHONPATH="$CNN_DIR:$PYTHONPATH"

clear
echo "================================================================"
echo "🚗 CNN VEHICLE DETECTION - MAIN LAUNCHER"
echo "================================================================"
echo ""

# GPU Profile Selection
echo "🎮 Select GPU Profile:"
echo ""
echo "  1. 🖥️  RTX A6000 (Full Performance)"
echo "  2. 📱 Jetson Nano (Restricted - 8GB memory)"
echo "  3. ⚡ Jetson Nano (Power Save Mode - 7W)"
echo ""
read -p "Enter GPU profile [1-3, default: 1]: " gpu_choice
gpu_choice=${gpu_choice:-1}

# Map choice to profile name
case $gpu_choice in
    1)
        GPU_PROFILE="a6000_full"
        echo "✓ Selected: RTX A6000 Full Performance"
        ;;
    2)
        GPU_PROFILE="jetson_nano_restricted"
        echo "✓ Selected: Jetson Nano Restricted (8GB)"
        ;;
    3)
        GPU_PROFILE="jetson_nano_power_save"
        echo "✓ Selected: Jetson Nano Power Save"
        ;;
    *)
        echo "❌ Invalid GPU profile choice"
        exit 1
        ;;
esac

echo ""
echo "================================================================"
echo "Select what you want to run:"
echo ""
echo "  1. 📹 Camera Detection (Real-time)"
echo "  2. 🎬 Video Processing"  
echo "  3. 🏋️  Train Models"
echo "  4. ❌ Exit"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "📹 Starting Camera Detection with GPU Profile: $GPU_PROFILE"
        echo ""
        echo "  1. 📷 RealSense D455 Camera (Recommended)"
        echo "  2. 🎥 USB Camera"
        echo "  3. 📹 Test Video (Fallback)"
        echo ""
        read -p "Select camera source [1-3, default: 1]: " cam_choice
        cam_choice=${cam_choice:-1}
        
        case $cam_choice in
            1)
                echo ""
                echo "🚀 Running with RealSense D455 camera..."
                cd "$CNN_DIR"
                $PYTHON inference/camera_inference.py \
                    --profile "$GPU_PROFILE" \
                    --realsense \
                    --save "detection_output_realsense.mp4" \
                    -v
                ;;
            2)
                read -p "Enter camera ID [default: 0]: " cam_id
                cam_id=${cam_id:-0}
                echo ""
                echo "🚀 Running with USB camera..."
                cd "$CNN_DIR"
                $PYTHON inference/camera_inference.py \
                    --profile "$GPU_PROFILE" \
                    --camera "$cam_id" \
                    --save "detection_output_usb.mp4" \
                    -v
                ;;
            3)
                echo ""
                echo "🚀 Running with test video..."
                cd "$CNN_DIR"
                $PYTHON inference/camera_inference.py \
                    --profile "$GPU_PROFILE" \
                    --camera "testing_data/IISc _Road.mp4" \
                    --save "detection_output_test.mp4" \
                    -v
                ;;
            *)
                echo "❌ Invalid camera choice"
                exit 1
                ;;
        esac
        ;;
    
    2)
        echo ""
        read -p "Enter video path: " video_path
        if [ ! -f "$video_path" ]; then
            echo "❌ Video not found: $video_path"
            exit 1
        fi
        
        output="${video_path%.*}_detected.mp4"
        echo ""
        echo "🎬 Processing video with GPU Profile: $GPU_PROFILE"
        echo "   Input: $video_path"
        echo "   Output: $output"
        echo ""
        echo "🚀 Running video inference..."
        echo ""
        
        cd "$CNN_DIR"
        $PYTHON inference/camera_inference.py \
            --profile "$GPU_PROFILE" \
            --camera "$video_path" \
            --save "$output" \
            -v
        ;;
    
    3)
        echo ""
        echo "🏋️  Starting Training..."
        echo ""
        cd "$CNN_DIR/training"
        $PYTHON train_classifier.py
        ;;
    
    4)
        echo "👋 Goodbye!"
        exit 0
        ;;
    
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac
