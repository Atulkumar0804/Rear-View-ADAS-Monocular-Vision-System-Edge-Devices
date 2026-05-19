#!/usr/bin/env bash
# Download KITTI Depth (Eigen split) into dataset/kitti_depth
# This uses public URLs; if they fail, you may need to supply a mirror or cookies.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT_DIR/dataset/kitti_depth"
mkdir -p "$DEST"
cd "$DEST"

echo "Downloading KITTI Depth (annotated + velodyne)..."

# Files (approx sizes):
#  - data_depth_annotated.zip (~1.4GB)
#  - data_depth_velodyne.zip (~12GB)
ANNOTATED_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip"
VELODYNE_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_velodyne.zip"

for URL in "$ANNOTATED_URL" "$VELODYNE_URL"; do
  FILE="${URL##*/}"
  echo "-> $FILE"
  wget -c "$URL" -O "$FILE"
  echo "Unzipping $FILE ..."
  unzip -o "$FILE"
  rm -f "$FILE"
done

echo "KITTI Depth download finished. Contents:"
find "$DEST" -maxdepth 2 -type d -print
