#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://www.jetson-ai-lab.com/code-samples/openpi_on_thor"
TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/openpi_on_thor"

mkdir -p "${TARGET_DIR}/patches"

files=(
    thor.Dockerfile
    pyproject.toml
    pi05_inference.py
    pytorch_to_onnx.py
    build_engine.sh
    trt_model_forward.py
    trt_torch.py
    calibration_data.py
)

echo "Downloading OpenPI Jetson Thor deployment scripts to ${TARGET_DIR}"

for file in "${files[@]}"; do
    echo "  ${file}"
    curl -fL "${BASE_URL}/${file}" -o "${TARGET_DIR}/${file}"
done

echo "  patches/apply_gemma_fixes.py"
curl -fL "${BASE_URL}/patches/apply_gemma_fixes.py" -o "${TARGET_DIR}/patches/apply_gemma_fixes.py"

chmod +x "${TARGET_DIR}/build_engine.sh"

echo "Done."
