#!/usr/bin/env bash

set -Eeuo pipefail

# -----------------------------------------------------------------------------
# ACT training launcher for a fixed number of epochs.
# Reuses train_full.sh and only computes --steps from dataset total_frames.
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_full.sh"

if [[ ! -x "${TRAIN_SCRIPT}" ]]; then
  echo "ERROR: required script not found or not executable: ${TRAIN_SCRIPT}" >&2
  exit 2
fi

# User-tunable defaults
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lerobot_flex}"
DATASET_ROOT="${DATASET_ROOT:-/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1}"
DATASET_REPO_ID="${DATASET_REPO_ID:-admin123/grasp_block_in_bin1}"
EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-32}"
JOB_NAME="${JOB_NAME:-act_grasp_block_in_bin1_e${EPOCHS}}"
POLICY_TYPE="${POLICY_TYPE:-act}"
POLICY_DEVICE="${POLICY_DEVICE:-auto}"
DATASET_VIDEO_BACKEND="${DATASET_VIDEO_BACKEND:-pyav}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_FREQ="${LOG_FREQ:-100}"
EVAL_FREQ="${EVAL_FREQ:--1}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-5}"
SEED="${SEED:-1000}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "ERROR: dataset root not found: ${DATASET_ROOT}" >&2
  exit 2
fi

if [[ ! "${EPOCHS}" =~ ^[0-9]+$ ]] || [[ "${EPOCHS}" -le 0 ]]; then
  echo "ERROR: EPOCHS must be a positive integer, got: ${EPOCHS}" >&2
  exit 2
fi

if [[ ! "${BATCH_SIZE}" =~ ^[0-9]+$ ]] || [[ "${BATCH_SIZE}" -le 0 ]]; then
  echo "ERROR: BATCH_SIZE must be a positive integer, got: ${BATCH_SIZE}" >&2
  exit 2
fi

if [[ ! "${SAVE_EVERY_EPOCHS}" =~ ^[0-9]+$ ]] || [[ "${SAVE_EVERY_EPOCHS}" -le 0 ]]; then
  echo "ERROR: SAVE_EVERY_EPOCHS must be a positive integer, got: ${SAVE_EVERY_EPOCHS}" >&2
  exit 2
fi

INFO_JSON="${DATASET_ROOT}/meta/info.json"
if [[ ! -f "${INFO_JSON}" ]]; then
  echo "ERROR: dataset meta/info.json not found: ${INFO_JSON}" >&2
  exit 2
fi

TOTAL_FRAMES="$(python - <<'PY' "${INFO_JSON}"
import json
import sys
from pathlib import Path

info_path = Path(sys.argv[1])
info = json.loads(info_path.read_text())
total_frames = info.get("total_frames")
if not isinstance(total_frames, int) or total_frames <= 0:
    raise SystemExit(f"Invalid total_frames in {info_path}: {total_frames}")
print(total_frames)
PY
)"

STEPS_PER_EPOCH="$(( (TOTAL_FRAMES + BATCH_SIZE - 1) / BATCH_SIZE ))"
STEPS="$(( STEPS_PER_EPOCH * EPOCHS ))"
SAVE_FREQ="$(( STEPS_PER_EPOCH * SAVE_EVERY_EPOCHS ))"

echo "[calc] dataset_root=${DATASET_ROOT}"
echo "[calc] total_frames=${TOTAL_FRAMES}"
echo "[calc] batch_size=${BATCH_SIZE}"
echo "[calc] epochs=${EPOCHS}"
echo "[calc] steps_per_epoch=ceil(${TOTAL_FRAMES}/${BATCH_SIZE})=${STEPS_PER_EPOCH}"
echo "[calc] total_steps=${STEPS}"
echo "[calc] save_every_epochs=${SAVE_EVERY_EPOCHS} -> save_freq=${SAVE_FREQ}"
echo "[calc] dataset_video_backend=${DATASET_VIDEO_BACKEND}"

export CONDA_ENV_NAME
export DATASET_ROOT
export DATASET_REPO_ID
export JOB_NAME
export POLICY_TYPE
export POLICY_DEVICE
export BATCH_SIZE
export STEPS
export SAVE_FREQ
export EVAL_FREQ
export LOG_FREQ
export NUM_WORKERS
export DATASET_VIDEO_BACKEND
export SEED
export PUSH_TO_HUB
export WANDB_ENABLE
export DRY_RUN

exec "${TRAIN_SCRIPT}" \
  "--dataset.video_backend=${DATASET_VIDEO_BACKEND}" \
  "$@"
