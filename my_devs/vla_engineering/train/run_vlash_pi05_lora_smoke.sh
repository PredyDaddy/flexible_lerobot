#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/cqy_workspace/flexible_lerobot"
TRAIN_ROOT="${REPO_ROOT}/my_devs/vla_engineering/train"
VLASH_ROOT="${REPO_ROOT}/my_devs/vla_engineering/vlash-main"
ENV_NAME="${ENV_NAME:-vlash_train}"
CONFIG_PATH="${CONFIG_PATH:-${TRAIN_ROOT}/pi05_so101_vlash_lora_smoke.yaml}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${TRAIN_ROOT}/logs/${RUN_ID}_pi05_lora_smoke"
OUTPUT_DIR="${TRAIN_ROOT}/outputs/pi05_so101_vlash_lora_smoke"

mkdir -p "${LOG_DIR}" "${TRAIN_ROOT}/outputs" "${TRAIN_ROOT}/tmp" "${TRAIN_ROOT}/hf_home"

exec > >(tee -a "${LOG_DIR}/train_terminal.log") 2>&1

source /home/cqy/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

cd "${REPO_ROOT}"

bash "${TRAIN_ROOT}/prepare_offline_assets.sh"

export PYTHONPATH="${VLASH_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${TRAIN_ROOT}/hf_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export MPLCONFIGDIR="${TRAIN_ROOT}/tmp/matplotlib"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

rm -rf "${OUTPUT_DIR}"

echo "CONFIG_PATH=${CONFIG_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LOG_DIR=${LOG_DIR}"

if command -v nvidia-smi >/dev/null 2>&1; then
  while true; do
    date '+%F %T'
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
    sleep 10
  done > "${LOG_DIR}/gpu_usage.log" 2>&1 &
  GPU_MONITOR_PID=$!
  trap 'kill ${GPU_MONITOR_PID} 2>/dev/null || true' EXIT
fi

python -m vlash.train --config_path="${CONFIG_PATH}"

python "${TRAIN_ROOT}/smoke_vlash_pi05_checkpoint.py" \
  --policy-path "${OUTPUT_DIR}/checkpoints/last/pretrained_model" \
  --device cuda \
  --num-inference-steps 2

echo "VLASH PI0.5 LoRA smoke run complete."
