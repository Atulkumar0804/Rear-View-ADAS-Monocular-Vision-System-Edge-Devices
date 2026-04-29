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
        read -p "Enter camera ID [default: 0]: " cam_id
        cam_id=${cam_id:-0}
        echo ""
        echo "🚀 Running inference_with_gpu_config.py with profile: $GPU_PROFILE"
        echo ""
        
        cd "$CNN_DIR"
        $PYTHON inference/inference_with_gpu_config.py camera \
            --profile "$GPU_PROFILE" \
            --camera "$cam_id" \
            -v
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
        echo "🚀 Running inference_with_gpu_config.py with profile: $GPU_PROFILE"
        echo ""
        
        cd "$CNN_DIR"
        $PYTHON inference/inference_with_gpu_config.py video \
            --profile "$GPU_PROFILE" \
            --input "$video_path" \
            --output "$output" \
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
