#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/cqy_workspace/flexible_lerobot"
TRAIN_ROOT="${REPO_ROOT}/my_devs/vla_engineering/train"
TRAIN_ALL_ROOT="${REPO_ROOT}/my_devs/vla_engineering/train_all"
VLASH_ROOT="${REPO_ROOT}/my_devs/vla_engineering/vlash-main"
CONDA_ROOT="${CONDA_ROOT:-/home/cqy/miniconda3}"
POLICY_ENV="${POLICY_ENV:-vlash_train}"

DEFAULT_POLICY_PATH="${TRAIN_ALL_ROOT}/outputs/pi05_so101_delay_lora_r192_delay8_bs8_accum1_5epochs/checkpoints/last/pretrained_model"
POLICY_PATH="${POLICY_PATH:-${DEFAULT_POLICY_PATH}}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-Put the eraser into the small box}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8005}"
DEVICE="${DEVICE:-cuda}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${POLICY_ENV}"

cd "${REPO_ROOT}"

export PYTHONPATH="${VLASH_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
export HF_HOME="${TRAIN_ALL_ROOT}/hf_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "Serving train_all VLASH policy"
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
