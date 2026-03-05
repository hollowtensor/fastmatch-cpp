#!/bin/bash
# Download pretrained ONNX models for FastMatch
# Requires: gdown (pip install gdown) or manual download

set -e

MODELS_DIR="pretrained_models"
mkdir -p "$MODELS_DIR"

echo "Downloading pretrained models..."

# Check if gdown is available
if command -v gdown &> /dev/null; then
    gdown "https://drive.google.com/u/4/uc?id=1QAxtjwAe_lcRM75DeAlaUaQkGmcn2sNE" -O models.zip
    unzip -o models.zip -d "$MODELS_DIR"
    rm -f models.zip
    echo "Models downloaded to $MODELS_DIR/"
else
    echo "gdown not found. Install it with: pip install gdown"
    echo ""
    echo "Or manually download models and place them in $MODELS_DIR/:"
    echo "  - yolov4-tiny.onnx"
    echo "  - osnet_ain_x1_0_M.onnx"
    echo "  - coco.names"
    echo ""
    echo "Download link: https://drive.google.com/u/4/uc?id=1QAxtjwAe_lcRM75DeAlaUaQkGmcn2sNE"
    exit 1
fi

# Verify files exist
for f in yolov4-tiny.onnx osnet_ain_x1_0_M.onnx coco.names; do
    if [ -f "$MODELS_DIR/$f" ]; then
        echo "  ✓ $f"
    else
        echo "  ✗ $f missing!"
    fi
done
