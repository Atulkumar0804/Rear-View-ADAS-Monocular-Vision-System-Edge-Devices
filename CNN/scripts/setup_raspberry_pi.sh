
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CNN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$CNN_DIR/.venv-pi"

echo "============================================================"
echo " Raspberry Pi ADAS Setup"
echo "============================================================"

if ! command -v python3 >/dev/null 2>&1; then
    echo "❌ python3 is not installed"
    exit 1
fi

echo "📦 Installing Raspberry Pi OS dependencies..."
sudo apt update
sudo apt install -y \
    python3-venv \
    python3-pip \
    python3-picamera2 \
    libcamera-apps \
    libatlas-base-dev \
    libopenblas-dev \
    libjpeg-dev

if [ ! -d "$VENV_DIR" ]; then
    echo "🐍 Creating virtual environment: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "⬆️ Upgrading packaging tools..."
python -m pip install --upgrade pip setuptools wheel

echo "📚 Installing Python dependencies..."
python -m pip install --extra-index-url https://www.piwheels.org/simple -r "$CNN_DIR/requirements.txt"

echo "✅ Setup complete"
echo ""
echo "To run the live camera feed, use:"
echo "  bash $CNN_DIR/scripts/run_raspberry_pi_camera.sh"
