#!/bin/bash
# Download BGE-small-en-v1.5 ONNX model (quantized) from HuggingFace.
#
# Usage:
#   ./scripts/download_model.sh [target_dir]
#
# Default target: ~/.cache/cuba-memorys/models/

set -euo pipefail

TARGET_DIR="${1:-$HOME/.cache/cuba-memorys/models}"
REPO="Qdrant/bge-small-en-v1.5-onnx-Q"
BASE_URL="https://huggingface.co/${REPO}/resolve/main"

FILES=(
    "model_quantized.onnx"
    "tokenizer.json"
)

mkdir -p "$TARGET_DIR"

echo "📦 Downloading BGE-small-en-v1.5 (quantized) to ${TARGET_DIR}..."

for file in "${FILES[@]}"; do
    dest="${TARGET_DIR}/${file}"
    if [ -f "$dest" ]; then
        echo "  ✓ ${file} already exists, skipping"
    else
        echo "  ↓ ${file}..."
        curl -sSL -o "$dest" "${BASE_URL}/${file}"
        echo "  ✓ ${file} downloaded ($(du -h "$dest" | cut -f1))"
    fi
done

echo ""
echo "✅ Model ready. Set env var:"
echo "   export ONNX_MODEL_PATH=\"${TARGET_DIR}\""
echo ""
echo "Also install ONNX Runtime library:"
echo "   # Ubuntu/Debian:"
echo "   wget https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-linux-x64-1.21.0.tgz"
echo "   tar xzf onnxruntime-linux-x64-1.21.0.tgz"
echo "   export ORT_DYLIB_PATH=\$(pwd)/onnxruntime-linux-x64-1.21.0/lib/libonnxruntime.so"
