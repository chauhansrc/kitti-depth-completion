#!/usr/bin/env bash
# Downloads and extracts the KITTI depth completion "depth selection" subset (~2 GB).
#
# KITTI data is released under CC BY-NC-SA 3.0. Register at
# https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion
# and accept the terms before downloading. If the direct link below stops working,
# copy the "selected validation and test data sets" link from that page.
#
# Usage: bash scripts/download_data.sh /content/data
set -euo pipefail

DATA_DIR="${1:-./data}"
URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip"
ZIP="$DATA_DIR/data_depth_selection.zip"

mkdir -p "$DATA_DIR"

if [ -d "$DATA_DIR/depth_selection/val_selection_cropped" ]; then
    echo "Already extracted: $DATA_DIR/depth_selection"
    exit 0
fi

echo "Downloading $URL"
wget -c "$URL" -O "$ZIP"

echo "Extracting to $DATA_DIR"
unzip -q "$ZIP" -d "$DATA_DIR"
rm -f "$ZIP"

echo "Done. Frames in val_selection_cropped:"
ls "$DATA_DIR/depth_selection/val_selection_cropped/image" | wc -l
