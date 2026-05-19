#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CNN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif [ -x "$CNN_DIR/.venv-pi/bin/python" ]; then
    PYTHON="$CNN_DIR/.venv-pi/bin/python"
elif [ -x "$CNN_DIR/.venv/bin/python" ]; then
    PYTHON="$CNN_DIR/.venv/bin/python"
else
    PYTHON="python3"
fi

export PYTHONPATH="$CNN_DIR:${PYTHONPATH:-}"

echo "============================================================"
echo " Raspberry Pi Live ADAS Camera"
echo "============================================================"
echo "Using Python: $PYTHON"
echo ""

cd "$CNN_DIR/inference"

exec "$PYTHON" camera_inference.py \
    --camera-backend picamera2 \
    --device cpu \
    --width 640 \
    --height 480 \
    "$@"
