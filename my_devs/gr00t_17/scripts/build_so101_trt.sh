#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_lerobot_flex_host
require_local_env
enable_offline_mode

CHECKPOINT_PATH="${CHECKPOINT_PATH:-${GR00T17_ROOT}/outputs/formal/so101_n17_b2_e10_20260711/train/checkpoint-63600}"
TRT_ROOT="${TRT_ROOT:-${GR00T17_ROOT}/artifacts/tensorrt/so101_n17_b1_bf16_full}"
TRT_STEPS="${TRT_STEPS:-export,build,verify}"
TRT_WORKSPACE_MB="${TRT_WORKSPACE_MB:-8192}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

require_path_within_root "${CHECKPOINT_PATH}"
require_path_within_root "${TRT_ROOT}"
if [[ ! -f "${CHECKPOINT_PATH}/model.safetensors.index.json" ]]; then
  echo "[ERROR] TensorRT source checkpoint is incomplete: ${CHECKPOINT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${GR00T17_DATASET_DIR}/meta/info.json" ]]; then
  echo "[ERROR] TensorRT calibration dataset is incomplete: ${GR00T17_DATASET_DIR}" >&2
  exit 1
fi
if [[ ! "${TRT_WORKSPACE_MB}" =~ ^[0-9]+$ ]] || ((TRT_WORKSPACE_MB < 1024)); then
  echo "[ERROR] TRT_WORKSPACE_MB must be an integer >= 1024." >&2
  exit 1
fi

mkdir -p "${TRT_ROOT}"
cd "${GR00T17_WORKSPACE}"
exec "${GR00T17_ENV}/bin/python" scripts/deployment/build_trt_pipeline.py \
  --model-path "${CHECKPOINT_PATH}" \
  --dataset-path "${GR00T17_DATASET_DIR}" \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir "${TRT_ROOT}" \
  --precision bf16 \
  --batch-size 1 \
  --export-mode full_pipeline \
  --video-backend torchcodec \
  --workspace "${TRT_WORKSPACE_MB}" \
  --skip-compile \
  --steps "${TRT_STEPS}"
