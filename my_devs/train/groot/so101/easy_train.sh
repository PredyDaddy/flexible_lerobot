#!/usr/bin/env bash
set -e

cd /data/cqy_workspace/flexible_lerobot

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="/data/cqy_workspace/flexible_lerobot/outputs/groot_eraser_put_small_box_projector_runs/${RUN_ID}"
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

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

lerobot-train \
  --policy.type=groot \
  --policy.repo_id=robotech/groot \
  --policy.push_to_hub=false \
  --dataset.repo_id=desk_cleanup_v1/eraser_put_small_box \
  --dataset.root=/data/cqy_workspace/flexible_lerobot/datasets/desk_cleanup_v1/eraser_put_small_box \
  --dataset.video_backend=pyav \
  --batch_size=32 \
  --steps=5820 \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=groot_eraser_put_small_box_projector \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.base_model_path=/data/cqy_workspace/flexible_lerobot/assets/modelscope/GR00T-N1.5-3B \
  --policy.tune_llm=false \
  --policy.tune_visual=false \
  --policy.tune_projector=true \
  --policy.tune_diffusion_model=false \
  --save_freq=2000 \
  --eval_freq=20000 \
  --policy.use_bf16=true
