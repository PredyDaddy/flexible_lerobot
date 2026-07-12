#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_lerobot_flex_host
require_local_env
enable_offline_mode

CHECKPOINT_PATH="${CHECKPOINT_PATH:-${GR00T17_ROOT}/outputs/formal/so101_n17_b2_e10_20260711/train/checkpoint-63600}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-5556}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-tensorrt}"
TRT_MODE="${TRT_MODE:-n17_full_pipeline}"
TRT_ENGINE_PATH="${TRT_ENGINE_PATH:-${GR00T17_ROOT}/artifacts/tensorrt/so101_n17_b1_bf16_full/engines}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

require_path_within_root "${CHECKPOINT_PATH}"
if [[ ! -f "${CHECKPOINT_PATH}/model.safetensors.index.json" ]]; then
  echo "[ERROR] RTC checkpoint is incomplete: ${CHECKPOINT_PATH}" >&2
  exit 1
fi
if [[ "${SERVER_HOST}" != "127.0.0.1" && "${SERVER_HOST}" != "localhost" ]]; then
  echo "[ERROR] RTC server only binds to localhost." >&2
  exit 1
fi
if [[ "${INFERENCE_BACKEND}" != "pytorch" && "${INFERENCE_BACKEND}" != "tensorrt" ]]; then
  echo "[ERROR] INFERENCE_BACKEND must be pytorch or tensorrt." >&2
  exit 1
fi

BACKEND_ARGS=(--inference-backend "${INFERENCE_BACKEND}")
if [[ "${INFERENCE_BACKEND}" == "tensorrt" ]]; then
  require_path_within_root "${TRT_ENGINE_PATH}"
  REQUIRED_ENGINES=(state_encoder.engine action_encoder.engine dit_bf16.engine action_decoder.engine)
  if [[ "${TRT_MODE}" == "n17_full_pipeline" ]]; then
    REQUIRED_ENGINES+=(vit_bf16.engine llm_bf16.engine vl_self_attention.engine)
  elif [[ "${TRT_MODE}" != "action_head" ]]; then
    echo "[ERROR] TRT_MODE must be n17_full_pipeline or action_head." >&2
    exit 1
  fi
  for ENGINE in "${REQUIRED_ENGINES[@]}"; do
    if [[ ! -s "${TRT_ENGINE_PATH}/${ENGINE}" ]]; then
      echo "[ERROR] TensorRT engine is missing or empty: ${TRT_ENGINE_PATH}/${ENGINE}" >&2
      echo "[ERROR] Run scripts/build_so101_trt.sh first." >&2
      exit 1
    fi
  done
  BACKEND_ARGS+=(--trt-engine-path "${TRT_ENGINE_PATH}" --trt-mode "${TRT_MODE}")
fi

cd "${GR00T17_WORKSPACE}"
exec "${GR00T17_ENV}/bin/python" "${GR00T17_ROOT}/scripts/so101_rtc_policy_server.py" \
  --model-path "${CHECKPOINT_PATH}" \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 \
  --host "${SERVER_HOST}" \
  --port "${SERVER_PORT}" \
  "${BACKEND_ARGS[@]}"
