#!/bin/bash
# Download face detection & recognition ONNX models for FastMatch
#
# Models from InsightFace:
#   SCRFD  — face detection (det_10g, det_2.5g, det_500m)
#   ArcFace — face recognition (w600k_mbf, w600k_r50)
#
# Usage: ./download_face_models.sh [--all]
#   Default: downloads det_10g + w600k_mbf (recommended combo)
#   --all:   downloads all available models

set -e

WEIGHTS_DIR="weights"
mkdir -p "$WEIGHTS_DIR"

BASE_URL="https://huggingface.co/mightyzau/InsightFace-REST/resolve/main/models"

# Model definitions: name → URL
declare -A MODELS
MODELS=(
    ["det_10g.onnx"]="https://huggingface.co/hollowtensor/insightface-onnx/resolve/main/det_10g.onnx"
    ["det_2.5g.onnx"]="https://huggingface.co/hollowtensor/insightface-onnx/resolve/main/det_2.5g.onnx"
    ["det_500m.onnx"]="https://huggingface.co/hollowtensor/insightface-onnx/resolve/main/det_500m.onnx"
    ["w600k_mbf.onnx"]="https://huggingface.co/hollowtensor/insightface-onnx/resolve/main/w600k_mbf.onnx"
    ["w600k_r50.onnx"]="https://huggingface.co/hollowtensor/insightface-onnx/resolve/main/w600k_r50.onnx"
)

# Default models (good balance of speed and accuracy)
DEFAULT_MODELS=("det_10g.onnx" "w600k_mbf.onnx")

download_model() {
    local name="$1"
    local url="${MODELS[$name]}"
    local dest="$WEIGHTS_DIR/$name"

    if [ -f "$dest" ]; then
        echo "  [skip] $name (already exists)"
        return 0
    fi

    echo "  Downloading $name ..."
    if command -v curl &> /dev/null; then
        curl -L -o "$dest" "$url" 2>/dev/null || {
            echo "  [FAIL] $name — download failed"
            rm -f "$dest"
            return 1
        }
    elif command -v wget &> /dev/null; then
        wget -q -O "$dest" "$url" || {
            echo "  [FAIL] $name — download failed"
            rm -f "$dest"
            return 1
        }
    else
        echo "Error: curl or wget required"
        exit 1
    fi
    echo "  [ok]   $name"
}

# Parse args
DOWNLOAD_ALL=false
if [ "$1" = "--all" ]; then
    DOWNLOAD_ALL=true
fi

echo "Face Detection & Recognition Models"
echo "===================================="
echo ""

if [ "$DOWNLOAD_ALL" = true ]; then
    echo "Downloading all models to $WEIGHTS_DIR/ ..."
    echo ""
    echo "Detection models (SCRFD):"
    download_model "det_10g.onnx"
    download_model "det_2.5g.onnx"
    download_model "det_500m.onnx"
    echo ""
    echo "Recognition models (ArcFace):"
    download_model "w600k_mbf.onnx"
    download_model "w600k_r50.onnx"
else
    echo "Downloading default models to $WEIGHTS_DIR/ ..."
    echo "(use --all for all available models)"
    echo ""
    for m in "${DEFAULT_MODELS[@]}"; do
        download_model "$m"
    done
fi

echo ""
echo "Available models in $WEIGHTS_DIR/:"
echo ""
echo "  Detection (--det-model):"
echo "    det_10g.onnx   — best accuracy, ~10 GFlops"
echo "    det_2.5g.onnx  — balanced, ~2.5 GFlops"
echo "    det_500m.onnx  — fastest, ~500 MFlops"
echo ""
echo "  Recognition (--rec-model):"
echo "    w600k_mbf.onnx — MobileFaceNet, fast"
echo "    w600k_r50.onnx — ResNet50, best accuracy"
echo ""

# Verify
echo "Status:"
for name in "${!MODELS[@]}"; do
    if [ -f "$WEIGHTS_DIR/$name" ]; then
        size=$(du -h "$WEIGHTS_DIR/$name" | cut -f1)
        echo "  [ok]   $name ($size)"
    else
        echo "  [--]   $name (not downloaded)"
    fi
done

echo ""
echo "Usage:"
echo "  ./face_demo --det-model ../weights/det_10g.onnx --rec-model ../weights/w600k_mbf.onnx"
