#!/bin/bash
#
# Main CNN Launcher - Raspberry Pi focused
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CNN_DIR="$SCRIPT_DIR"

# Prefer currently activated venv; fallback to project-local .venv.
if [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif [ -x "$CNN_DIR/inference/venv/bin/python" ]; then
    PYTHON="$CNN_DIR/inference/venv/bin/python"
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

echo ""
echo "================================================================"
echo "TCP Streaming Server (RECOMMENDED - 20-30 FPS)"
echo "================================================================"
echo ""
echo "📡 Starting TCP Streaming Server..."
echo "Expected Performance: 20-30 FPS with safety metrics"
echo ""

cd "$CNN_DIR/inference"
$PYTHON adas_tcp_stream_wrapper.py \
    --width 480 \
    --height 360 \
    --detect-width 320 \
    --detect-height 240 \
    --jpeg-quality 80
