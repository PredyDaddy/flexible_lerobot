#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/cqy_workspace/flexible_lerobot"
TRAIN_ALL_ROOT="${REPO_ROOT}/my_devs/vla_engineering/train_all"
VLASH_ROOT="${REPO_ROOT}/my_devs/vla_engineering/vlash-main"
CONDA_ROOT="${CONDA_ROOT:-/home/cqy/miniconda3}"
POLICY_ENV="${POLICY_ENV:-vlash_train}"

DEFAULT_POLICY_PATH="${TRAIN_ALL_ROOT}/outputs/pi05_so101_delay_lora_r192_delay8_bs8_accum1_5epochs/checkpoints/last/pretrained_model"
POLICY_PATH="${POLICY_PATH:-${DEFAULT_POLICY_PATH}}"
DEVICE="${DEVICE:-cuda}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-2}"
TASK="${TASK:-Put the eraser into the small box}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${POLICY_ENV}"

cd "${REPO_ROOT}"

export PYTHONPATH="${VLASH_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
export HF_HOME="${TRAIN_ALL_ROOT}/hf_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "Smoke testing train_all VLASH checkpoint"
echo "POLICY_ENV=${POLICY_ENV}"
echo "POLICY_PATH=${POLICY_PATH}"
echo "DEVICE=${DEVICE}"
echo "NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS}"
echo "TASK=${TASK}"

python "${TRAIN_ALL_ROOT}/scripts/audit_checkpoint.py" \
  --policy-path "${POLICY_PATH}"

python "${TRAIN_ALL_ROOT}/scripts/smoke_checkpoint.py" \
  --policy-path "${POLICY_PATH}" \
  --device "${DEVICE}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  --task "${TASK}"

echo "train_all VLASH checkpoint smoke inference passed."
