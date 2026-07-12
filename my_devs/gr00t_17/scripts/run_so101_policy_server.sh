#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_lerobot_flex_host
require_local_env
enable_offline_mode

CHECKPOINT_PATH="${CHECKPOINT_PATH:-${GR00T17_ROOT}/outputs/formal/so101_n17_b2_e10_20260711/train/checkpoint-63600}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-5555}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

require_path_within_root "${CHECKPOINT_PATH}"
if [[ ! -f "${CHECKPOINT_PATH}/model.safetensors.index.json" ]]; then
  echo "[ERROR] Inference checkpoint is incomplete: ${CHECKPOINT_PATH}" >&2
  exit 1
fi
if [[ "${SERVER_HOST}" != "127.0.0.1" && "${SERVER_HOST}" != "localhost" ]]; then
  echo "[ERROR] Guarded inference server only binds to localhost." >&2
  exit 1
fi
if [[ ! "${SERVER_PORT}" =~ ^[0-9]+$ ]] || ((SERVER_PORT < 1 || SERVER_PORT > 65535)); then
  echo "[ERROR] Invalid SERVER_PORT: ${SERVER_PORT}" >&2
  exit 1
fi

cd "${GR00T17_WORKSPACE}"
exec "${GR00T17_ENV}/bin/python" gr00t/eval/run_gr00t_server.py \
  --model-path "${CHECKPOINT_PATH}" \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 \
  --host "${SERVER_HOST}" \
  --port "${SERVER_PORT}"
