#!/usr/bin/env bash
set -e

cd /data/cqy_workspace/flexible_lerobot

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="/data/cqy_workspace/flexible_lerobot/outputs/pi05_eraser_cup_multi_task_runs/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}_logs"
exec > >(tee -a "${OUTPUT_DIR}_logs/train_terminal.log") 2>&1

echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TERMINAL_LOG=${OUTPUT_DIR}_logs/train_terminal.log"
echo "GPU_LOG=${OUTPUT_DIR}_logs/gpu_usage.log"

while true; do
  date '+%F %T'
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
  sleep 30
done > "${OUTPUT_DIR}_logs/gpu_usage.log" 2>&1 &
GPU_MONITOR_PID=$!
trap 'kill ${GPU_MONITOR_PID} 2>/dev/null || true' EXIT

source /home/cqy/miniconda3/etc/profile.d/conda.sh
conda activate lerobot_flex

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p /data/cqy_workspace/flexible_lerobot/google
ln -sfn \
  /data/cqy_workspace/flexible_lerobot/assets/modelscope/google/paligemma-3b-pt-224 \
  /data/cqy_workspace/flexible_lerobot/google/paligemma-3b-pt-224

lerobot-train \
  --dataset.repo_id=desk_cleanup_v1/eraser_cup_multi_task \
  --dataset.root=/data/cqy_workspace/flexible_lerobot/datasets/desk_cleanup_v1/eraser_cup_multi_task \
  --dataset.video_backend=pyav \
  --policy.type=pi05 \
  --policy.pretrained_path=/data/cqy_workspace/flexible_lerobot/assets/modelscope/lerobot/pi05_base \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=pi05_eraser_cup_multi_task \
  --batch_size=16 \
  --save_freq=1000 \
  --steps=33300
