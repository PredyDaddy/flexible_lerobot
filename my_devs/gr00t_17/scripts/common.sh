#!/usr/bin/env bash
set -euo pipefail

GR00T17_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GR00T17_ROOT

export GR00T17_WORKSPACE="${GR00T17_ROOT}/workspace/Isaac-GR00T-n1.7"
export GR00T17_ENV="${GR00T17_ROOT}/env/gr00t_n17"
export GR00T17_MODEL_DIR="${GR00T17_ROOT}/models/GR00T-N1.7-3B"
export GR00T17_BACKBONE_ASSET_DIR="${GR00T17_ROOT}/models/Qwen3-VL-2B-Instruct-assets"
export GR00T17_DATASET_DIR="${GR00T17_ROOT}/data/converted_v21/desk_cleanup_v1/eraser_cup_multi_task"

if [[ -d "${GR00T17_ENV}/bin" ]]; then
  export PATH="${GR00T17_ENV}/bin:${PATH}"
fi

export XDG_CACHE_HOME="${GR00T17_ROOT}/cache/xdg"
export HF_HOME="${GR00T17_ROOT}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${GR00T17_ROOT}/cache/transformers"
export TORCH_HOME="${GR00T17_ROOT}/cache/torch"
export UV_CACHE_DIR="${GR00T17_ROOT}/cache/uv"
export UV_HTTP_TIMEOUT=600
export PIP_CACHE_DIR="${GR00T17_ROOT}/cache/pip"
export RUFF_CACHE_DIR="${GR00T17_ROOT}/cache/ruff"
export TRITON_CACHE_DIR="${GR00T17_ROOT}/cache/triton"
export CUDA_CACHE_PATH="${GR00T17_ROOT}/cache/cuda"
export PYTHONPYCACHEPREFIX="${GR00T17_ROOT}/cache/pycache"
export MPLCONFIGDIR="${GR00T17_ROOT}/cache/matplotlib"
export NUMBA_CACHE_DIR="${GR00T17_ROOT}/cache/numba"
export TMPDIR="${GR00T17_ROOT}/tmp"
export WANDB_DIR="${GR00T17_ROOT}/logs/wandb"
export WANDB_MODE="disabled"
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export NO_ALBUMENTATIONS_UPDATE=1

# The lerobot_flex base interpreter has an RPATH entry whose libffi.so.7
# compatibility symlink points at libffi.so.8. System FFmpeg 4 needs the real
# libffi 7 ABI when TorchCodec loads libgobject, so make that ABI explicit.
GR00T17_SYSTEM_LIBFFI="/lib/x86_64-linux-gnu/libffi.so.7"
if [[ -f "${GR00T17_SYSTEM_LIBFFI}" ]]; then
  case ":${LD_PRELOAD:-}:" in
    *":${GR00T17_SYSTEM_LIBFFI}:"*) ;;
    *) export LD_PRELOAD="${GR00T17_SYSTEM_LIBFFI}${LD_PRELOAD:+:${LD_PRELOAD}}" ;;
  esac
fi

mkdir -p \
  "${XDG_CACHE_HOME}" \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${TORCH_HOME}" \
  "${UV_CACHE_DIR}" \
  "${PIP_CACHE_DIR}" \
  "${RUFF_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${CUDA_CACHE_PATH}" \
  "${PYTHONPYCACHEPREFIX}" \
  "${MPLCONFIGDIR}" \
  "${NUMBA_CACHE_DIR}" \
  "${TMPDIR}" \
  "${WANDB_DIR}"

require_local_env() {
  if [[ ! -x "${GR00T17_ENV}/bin/python" ]]; then
    echo "[ERROR] Local GR00T environment is missing: ${GR00T17_ENV}" >&2
    echo "[ERROR] Run scripts/setup_env.sh first." >&2
    return 1
  fi
}

require_lerobot_flex_host() {
  if [[ "${CONDA_DEFAULT_ENV:-}" != "lerobot_flex" ]]; then
    echo "[ERROR] This workflow must run inside the lerobot_flex conda environment." >&2
    echo "[ERROR] Use: conda run -n lerobot_flex bash <script>" >&2
    return 1
  fi
}

require_path_within_root() {
  local candidate="$1"
  local resolved
  resolved="$(realpath -m "${candidate}")"
  case "${resolved}" in
    "${GR00T17_ROOT}"|"${GR00T17_ROOT}"/*) ;;
    *)
      echo "[ERROR] Path escapes GR00T17_ROOT: ${candidate} -> ${resolved}" >&2
      return 1
      ;;
  esac
}

enable_offline_mode() {
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
}

disable_offline_mode_for_download() {
  unset HF_HUB_OFFLINE || true
  unset TRANSFORMERS_OFFLINE || true
}

run_n17_finetune() {
  local output_dir="$1"
  local max_steps="$2"
  local save_steps="$3"
  local global_batch_size="$4"
  local gradient_accumulation_steps="$5"
  local dataloader_num_workers="$6"
  local shard_size="$7"
  local save_total_limit="$8"

  require_path_within_root "${output_dir}"
  (
    cd "${GR00T17_WORKSPACE}"
    "${GR00T17_ENV}/bin/python" gr00t/experiment/launch_finetune.py \
      --base-model-path "${GR00T17_MODEL_DIR}" \
      --backbone-model-path "${GR00T17_BACKBONE_ASSET_DIR}" \
      --dataset-path "${GR00T17_DATASET_DIR}" \
      --embodiment-tag NEW_EMBODIMENT \
      --modality-config-path "${GR00T17_ROOT}/configs/so101_modality.py" \
      --num-gpus 1 \
      --output-dir "${output_dir}" \
      --max-steps "${max_steps}" \
      --global-batch-size "${global_batch_size}" \
      --gradient-accumulation-steps "${gradient_accumulation_steps}" \
      --dataloader-num-workers "${dataloader_num_workers}" \
      --shard-size "${shard_size}" \
      --episode-sampling-rate 0.1 \
      --num-shards-per-epoch 100000 \
      --save-steps "${save_steps}" \
      --save-total-limit "${save_total_limit}" \
      --learning-rate 0.0001 \
      --weight-decay 0.00001 \
      --warmup-ratio 0.05 \
      --state-dropout-prob 0.2 \
      --no-tune-llm \
      --no-tune-visual \
      --tune-projector \
      --tune-diffusion-model \
      --no-use-wandb \
      --no-save-only-model \
      --no-skip-weight-loading
  )
}

start_gpu_monitor() {
  local log_path="$1"
  local interval_seconds="${2:-30}"
  require_path_within_root "${log_path}"
  (
    while true; do
      date --iso-8601=seconds
      nvidia-smi \
        --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,power.draw \
        --format=csv,noheader,nounits
      sleep "${interval_seconds}"
    done
  ) >>"${log_path}" 2>&1 &
  GR00T17_GPU_MONITOR_PID=$!
  export GR00T17_GPU_MONITOR_PID
}

stop_gpu_monitor() {
  if [[ -n "${GR00T17_GPU_MONITOR_PID:-}" ]]; then
    kill "${GR00T17_GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${GR00T17_GPU_MONITOR_PID}" 2>/dev/null || true
    unset GR00T17_GPU_MONITOR_PID
  fi
}
