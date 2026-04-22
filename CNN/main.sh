#!/bin/bash
#
# Main CNN Launcher - Easy access to all features
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
        echo "📹 Starting Camera Detection..."
        echo ""
        read -p "Enter camera ID [default: 3]: " cam_id
        cam_id=${cam_id:-3}
        cd "$CNN_DIR/inference"
        $PYTHON camera_inference.py --camera "$cam_id"
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
        echo "🎬 Processing video..."
        echo "   Input: $video_path"
        echo "   Output: $output"
        echo ""
        
        cd "$CNN_DIR/inference"
        $PYTHON video_inference.py --input "$video_path" --output "$output"
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
