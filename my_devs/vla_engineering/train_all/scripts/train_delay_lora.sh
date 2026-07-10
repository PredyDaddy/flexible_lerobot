#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/cqy_workspace/flexible_lerobot"
TRAIN_ALL_ROOT="${REPO_ROOT}/my_devs/vla_engineering/train_all"
VLASH_ROOT="${REPO_ROOT}/my_devs/vla_engineering/vlash-main"
ENV_NAME="${ENV_NAME:-vlash_train}"
CONFIG_PATH="${CONFIG_PATH:-${TRAIN_ALL_ROOT}/configs/train/pi05_so101_delay_lora_r192.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${TRAIN_ALL_ROOT}/outputs/pi05_so101_delay_lora_r192}"
FORCE_RESTART="${FORCE_RESTART:-false}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${TRAIN_ALL_ROOT}/logs/${RUN_ID}_pi05_delay_lora_train"

mkdir -p "${LOG_DIR}" "${TRAIN_ALL_ROOT}/outputs" "${TRAIN_ALL_ROOT}/tmp" "${TRAIN_ALL_ROOT}/hf_home"

exec > >(tee -a "${LOG_DIR}/train_terminal.log") 2>&1

source /home/cqy/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

cd "${REPO_ROOT}"

bash "${TRAIN_ALL_ROOT}/scripts/prepare_offline_assets.sh"

export PYTHONPATH="${VLASH_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${TRAIN_ALL_ROOT}/hf_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export MPLCONFIGDIR="${TRAIN_ALL_ROOT}/tmp/matplotlib"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ "${FORCE_RESTART}" == "true" ]]; then
  echo "FORCE_RESTART=true, removing OUTPUT_DIR=${OUTPUT_DIR}"
  rm -rf "${OUTPUT_DIR}"
fi

echo "CONFIG_PATH=${CONFIG_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LOG_DIR=${LOG_DIR}"
echo "ENV_NAME=${ENV_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "FORCE_RESTART=${FORCE_RESTART}"

if command -v nvidia-smi >/dev/null 2>&1; then
  while true; do
    date '+%F %T'
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
    sleep 10
  done > "${LOG_DIR}/gpu_usage.log" 2>&1 &
  GPU_MONITOR_PID=$!
  trap 'kill ${GPU_MONITOR_PID} 2>/dev/null || true' EXIT
fi

python "${TRAIN_ALL_ROOT}/scripts/probe_dataset.py" \
  --max-delay-steps 1 \
  --sample-index 0

python -m vlash.train --config_path="${CONFIG_PATH}" "$@"

POLICY_PATH="${OUTPUT_DIR}/checkpoints/last/pretrained_model"

python "${TRAIN_ALL_ROOT}/scripts/audit_checkpoint.py" \
  --policy-path "${POLICY_PATH}"

python "${TRAIN_ALL_ROOT}/scripts/smoke_checkpoint.py" \
  --policy-path "${POLICY_PATH}" \
  --device cuda \
  --num-inference-steps 2

echo "train_all VLASH PI0.5 delay LoRA training complete."
echo "POLICY_PATH=${POLICY_PATH}"
echo "LOG_DIR=${LOG_DIR}"
