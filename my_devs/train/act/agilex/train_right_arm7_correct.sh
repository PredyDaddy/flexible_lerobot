#!/usr/bin/env bash

set -Eeuo pipefail

# -----------------------------------------------------------------------------
# Correct ACT training entry for AgileX right-arm dataset.
# Goals:
# 1. default to the single-arm right-arm dataset,
# 2. default to the full dataset instead of a single episode,
# 3. compute steps from epochs to avoid miscounting,
# 4. support the correct resume path via --config_path + --resume=true.
# -----------------------------------------------------------------------------

CONDA_ENV_NAME="${CONDA_ENV_NAME:-lerobot_flex}"
export HF_HOME="${HF_HOME:-/tmp/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf_datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/train}"
RUN_LOG_ROOT="${RUN_LOG_ROOT:-${OUTPUT_ROOT}/.run_logs}"

DATASET_REPO_ID="${DATASET_REPO_ID:-test_pipeline_clean_12_19_removed_right_arm7}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT_DIR}/datasets/lerobot_datasets/test_pipeline_clean_12_19_removed_right_arm7}"
DATASET_EPISODES="${DATASET_EPISODES:-}"
ALLOW_SUBSET="${ALLOW_SUBSET:-0}"

JOB_NAME="${JOB_NAME:-act_right_arm7_full_correct}"
POLICY_TYPE="${POLICY_TYPE:-act}"
POLICY_DEVICE="${POLICY_DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TARGET_EPOCHS="${TARGET_EPOCHS:-5}"
STEPS="${STEPS:-}"
SAVE_FREQ="${SAVE_FREQ:-}"
EVAL_FREQ="${EVAL_FREQ:--1}"
LOG_FREQ="${LOG_FREQ:-100}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-1000}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"

RESUME_CONFIG_PATH="${RESUME_CONFIG_PATH:-}"
DRY_RUN="${DRY_RUN:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

mkdir -p "${OUTPUT_ROOT}" "${RUN_LOG_ROOT}" "${HF_HOME}" "${HF_DATASETS_CACHE}"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "ERROR: dataset root not found: ${DATASET_ROOT}" >&2
  exit 2
fi

if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
  echo "ERROR: missing dataset meta/info.json: ${DATASET_ROOT}/meta/info.json" >&2
  exit 2
fi

if [[ ! -f "${DATASET_ROOT}/meta/episodes/chunk-000/file-000.parquet" ]]; then
  echo "ERROR: missing dataset episodes parquet under: ${DATASET_ROOT}/meta/episodes" >&2
  exit 2
fi

if [[ -n "${DATASET_EPISODES}" && "${ALLOW_SUBSET}" != "1" ]]; then
  echo "ERROR: DATASET_EPISODES is set to ${DATASET_EPISODES}." >&2
  echo "ERROR: Correct training defaults to the full dataset." >&2
  echo "ERROR: This avoids repeating the previous mistake of training only episode [0]." >&2
  echo "ERROR: If you intentionally want a subset run, set ALLOW_SUBSET=1." >&2
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH." >&2
  exit 2
fi

RUNNER=()
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME}" ]]; then
  RUNNER=(conda run --no-capture-output -n "${CONDA_ENV_NAME}")
fi

if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  unset CUDA_VISIBLE_DEVICES
else
  export CUDA_VISIBLE_DEVICES
fi

if [[ "${POLICY_DEVICE}" == "auto" ]]; then
  detected_device="$("${RUNNER[@]}" python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" || true)"
  if [[ "${detected_device}" == "cuda" ]]; then
    POLICY_DEVICE="cuda"
  else
    POLICY_DEVICE="cpu"
  fi
fi

readarray -t DATASET_META < <("${RUNNER[@]}" python - "${DATASET_ROOT}" "${DATASET_EPISODES}" <<'PY'
import ast
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
episodes_raw = sys.argv[2].strip()
info = json.loads((root / "meta" / "info.json").read_text())
total_frames = int(info["total_frames"])
total_episodes = int(info["total_episodes"])
selected_frames = total_frames
selected_episodes = total_episodes

if episodes_raw:
    episodes = ast.literal_eval(episodes_raw)
    if not isinstance(episodes, list) or not episodes:
        raise SystemExit("DATASET_EPISODES must be a non-empty Python list literal, e.g. [0,1,2]")
    ep_df = pd.concat(
        pd.read_parquet(path)
        for path in sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    )
    ep_df = ep_df.set_index("episode_index")
    missing = [ep for ep in episodes if ep not in ep_df.index]
    if missing:
        raise SystemExit(f"Selected episodes not found in dataset: {missing}")
    selected_frames = int(ep_df.loc[episodes, "length"].sum())
    selected_episodes = len(episodes)

print(total_frames)
print(total_episodes)
print(selected_frames)
print(selected_episodes)
PY
)

TOTAL_FRAMES="${DATASET_META[0]}"
TOTAL_EPISODES="${DATASET_META[1]}"
SELECTED_FRAMES="${DATASET_META[2]}"
SELECTED_EPISODES="${DATASET_META[3]}"

if [[ -z "${STEPS}" ]]; then
  STEPS="$("${RUNNER[@]}" python - "${TARGET_EPOCHS}" "${SELECTED_FRAMES}" "${BATCH_SIZE}" <<'PY'
import math
import sys

target_epochs = float(sys.argv[1])
frames = int(sys.argv[2])
batch_size = int(sys.argv[3])
print(int(math.ceil(target_epochs * frames / batch_size)))
PY
)"
fi

STEPS_PER_EPOCH="$("${RUNNER[@]}" python - "${SELECTED_FRAMES}" "${BATCH_SIZE}" <<'PY'
import math
import sys

frames = int(sys.argv[1])
batch_size = int(sys.argv[2])
print(int(math.ceil(frames / batch_size)))
PY
)"

if [[ -z "${SAVE_FREQ}" ]]; then
  SAVE_FREQ="${STEPS_PER_EPOCH}"
fi

APPROX_EPOCHS="$("${RUNNER[@]}" python - "${STEPS}" "${BATCH_SIZE}" "${SELECTED_FRAMES}" <<'PY'
import sys

steps = int(sys.argv[1])
batch_size = int(sys.argv[2])
frames = int(sys.argv[3])
print(f"{steps * batch_size / frames:.4f}")
PY
)"

timestamp="$(date +'%Y%m%d_%H%M%S')"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${timestamp}_${JOB_NAME}}"
RUN_ID="$(basename "${OUTPUT_DIR}")"
STAGING_LOG_FILE="${RUN_LOG_ROOT}/${RUN_ID}.log"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/train.log}"

TRAIN_CMD=("${RUNNER[@]}" lerobot-train)

if [[ -n "${RESUME_CONFIG_PATH}" ]]; then
  if [[ ! -f "${RESUME_CONFIG_PATH}" ]]; then
    echo "ERROR: resume config not found: ${RESUME_CONFIG_PATH}" >&2
    exit 2
  fi
  TRAIN_CMD+=(
    "--config_path=${RESUME_CONFIG_PATH}"
    "--resume=true"
  )
else
  TRAIN_CMD+=(
    "--dataset.repo_id=${DATASET_REPO_ID}"
    "--dataset.root=${DATASET_ROOT}"
    "--policy.type=${POLICY_TYPE}"
  )
fi

if [[ -n "${DATASET_EPISODES}" ]]; then
  TRAIN_CMD+=("--dataset.episodes=${DATASET_EPISODES}")
fi

TRAIN_CMD+=(
  "--output_dir=${OUTPUT_DIR}"
  "--job_name=${JOB_NAME}"
  "--policy.device=${POLICY_DEVICE}"
  "--policy.push_to_hub=${PUSH_TO_HUB}"
  "--batch_size=${BATCH_SIZE}"
  "--steps=${STEPS}"
  "--save_freq=${SAVE_FREQ}"
  "--eval_freq=${EVAL_FREQ}"
  "--log_freq=${LOG_FREQ}"
  "--num_workers=${NUM_WORKERS}"
  "--seed=${SEED}"
  "--wandb.enable=${WANDB_ENABLE}"
)

echo "[start] conda_env=${CONDA_ENV_NAME}"
echo "[start] dataset_root=${DATASET_ROOT}"
echo "[start] dataset_repo_id=${DATASET_REPO_ID}"
echo "[start] selected_episodes=${SELECTED_EPISODES}/${TOTAL_EPISODES}"
echo "[start] selected_frames=${SELECTED_FRAMES}/${TOTAL_FRAMES}"
echo "[start] batch_size=${BATCH_SIZE} steps=${STEPS} approx_epochs=${APPROX_EPOCHS}"
echo "[start] save_freq=${SAVE_FREQ} eval_freq=${EVAL_FREQ} num_workers=${NUM_WORKERS}"
echo "[start] policy_device=${POLICY_DEVICE} output_dir=${OUTPUT_DIR}"
echo "[start] log_file=${LOG_FILE}"
echo "[start] staging_log_file=${STAGING_LOG_FILE}"

if [[ "${SELECTED_EPISODES}" == "1" ]]; then
  echo "[warn] only one episode selected; this is usually for smoke/debug, not correct training." >&2
fi

if [[ "${POLICY_DEVICE}" == "cpu" ]]; then
  echo "[warn] device=cpu. Full-dataset ACT training will be very slow on this machine." >&2
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY_RUN] Would execute:"
  printf '%q ' "${TRAIN_CMD[@]}" "$@"
  echo
  exit 0
fi

set +e
"${TRAIN_CMD[@]}" "$@" 2>&1 | tee "${STAGING_LOG_FILE}"
cmd_status=${PIPESTATUS[0]}
set -e

if [[ -d "${OUTPUT_DIR}" ]]; then
  cp "${STAGING_LOG_FILE}" "${LOG_FILE}"
  echo "[done] final_log_file=${LOG_FILE}"
else
  echo "[warn] output_dir was not created; staging log kept at ${STAGING_LOG_FILE}" >&2
fi

exit "${cmd_status}"
