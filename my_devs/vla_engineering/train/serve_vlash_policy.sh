#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/cqy_workspace/flexible_lerobot"
TRAIN_ROOT="${REPO_ROOT}/my_devs/vla_engineering/train"
VLASH_ROOT="${REPO_ROOT}/my_devs/vla_engineering/vlash-main"
CONDA_ROOT="${CONDA_ROOT:-/home/cqy/miniconda3}"
POLICY_ENV="${POLICY_ENV:-vlash_train}"

POLICY_PATH="${POLICY_PATH:-${TRAIN_ROOT}/outputs/pi05_so101_vlash_lora_10epochs_r192_bs1_accum8/checkpoints/033275/pretrained_model}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-Put the eraser into the small box}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8005}"
DEVICE="${DEVICE:-cuda}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${POLICY_ENV}"

cd "${REPO_ROOT}"

export PYTHONPATH="${VLASH_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "Serving VLASH policy"
echo "POLICY_ENV=${POLICY_ENV}"
echo "POLICY_PATH=${POLICY_PATH}"
echo "DEFAULT_PROMPT=${DEFAULT_PROMPT}"
echo "HOST=${HOST}"
echo "PORT=${PORT}"
echo "DEVICE=${DEVICE}"
echo "NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS}"

python "${TRAIN_ROOT}/serve_vlash_policy.py" \
  --policy-path "${POLICY_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --default-prompt "${DEFAULT_PROMPT}" \
  --device "${DEVICE}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  "$@"
