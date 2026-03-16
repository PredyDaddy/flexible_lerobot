#!/usr/bin/env bash

set -Eeuo pipefail

# -----------------------------------------------------------------------------
# Agilex Diffusion Policy 双臂训练脚本
# 说明：
# - 风格对齐 my_devs/train/act/agilex/train_first_test.sh
# - 默认训练 datasets/lerobot_datasets/first_test
# - 支持通过环境变量覆盖常用参数
# -----------------------------------------------------------------------------

# 训练使用的 conda 环境名。
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lerobot_flex}"
# 是否离线读取 Hugging Face Hub；默认开启，避免训练时误走外网。
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# 是否离线读取 datasets；默认开启。
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
# 是否离线读取 transformers；默认开启。
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# Hugging Face datasets 缓存目录。
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf_datasets}"

# 当前脚本所在目录。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 仓库根目录。
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
# 训练日志输出目录。
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
# 训练产物总输出目录。
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/train}"

# 传给 lerobot-train 的数据集 repo_id；默认使用 first_test。
DATASET_REPO_ID="${DATASET_REPO_ID:-first_test}"
# 数据集根目录；必须直接指向具体数据集目录，而不是父目录。
DATASET_ROOT="${DATASET_ROOT:-${ROOT_DIR}/datasets/lerobot_datasets/first_test}"

# 本次训练任务名；会拼到输出目录和日志文件名中。
JOB_NAME="${JOB_NAME:-dp_agilex_first_test_full}"
# 策略类型固定为 Diffusion Policy，不允许通过环境变量覆盖。
POLICY_TYPE="diffusion"
# 训练设备；auto 会自动检测 cuda/cpu。
POLICY_DEVICE="${POLICY_DEVICE:-auto}"
# 单卡 batch size；Diffusion 比 ACT 更重，双臂默认更保守。
BATCH_SIZE="${BATCH_SIZE:-8}"
# 总训练步数。
STEPS="${STEPS:-100000}"
# 每隔多少步保存一次 checkpoint。
SAVE_FREQ="${SAVE_FREQ:-10000}"
# 每隔多少步做一次评估；默认 -1 表示关闭。
EVAL_FREQ="${EVAL_FREQ:--1}"
# 每隔多少步打印一次训练日志。
LOG_FREQ="${LOG_FREQ:-100}"
# DataLoader worker 数量。
NUM_WORKERS="${NUM_WORKERS:-4}"
# 随机种子。
SEED="${SEED:-1000}"
# 是否把训练产物推送到 Hub。
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
# 是否启用 wandb。
WANDB_ENABLE="${WANDB_ENABLE:-false}"

# 可选：仅训练某些 episode，例如 "[0,1,2]"。
DATASET_EPISODES="${DATASET_EPISODES:-}"
# 可选：手工指定 GPU，例如 "0" 或 "0,1"。
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
# 仅预演命令，不真正启动训练；1 表示开启。
DRY_RUN="${DRY_RUN:-0}"

# Diffusion 可选：输入多少帧观测。
DIFFUSION_N_OBS_STEPS="${DIFFUSION_N_OBS_STEPS:-}"
# Diffusion 可选：一次建模的动作 horizon。
DIFFUSION_HORIZON="${DIFFUSION_HORIZON:-}"
# Diffusion 可选：每次策略实际输出并执行多少步动作。
DIFFUSION_N_ACTION_STEPS="${DIFFUSION_N_ACTION_STEPS:-}"
# Diffusion 可选：扩散训练时间步数量。
DIFFUSION_NUM_TRAIN_TIMESTEPS="${DIFFUSION_NUM_TRAIN_TIMESTEPS:-}"
# Diffusion 可选：推理时反向去噪步数。
DIFFUSION_NUM_INFERENCE_STEPS="${DIFFUSION_NUM_INFERENCE_STEPS:-}"
mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}" "${HF_DATASETS_CACHE}"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "ERROR: dataset root not found: ${DATASET_ROOT}" >&2
  exit 2
fi

if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  unset CUDA_VISIBLE_DEVICES
else
  export CUDA_VISIBLE_DEVICES
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH." >&2
  exit 2
fi

RUNNER=()
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME}" ]]; then
  RUNNER=(conda run --no-capture-output -n "${CONDA_ENV_NAME}")
fi

if [[ "${POLICY_DEVICE}" == "auto" ]]; then
  detected_device="$("${RUNNER[@]}" python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" || true)"
  if [[ "${detected_device}" == "cuda" ]]; then
    POLICY_DEVICE="cuda"
  else
    POLICY_DEVICE="cpu"
  fi
fi

timestamp="$(date +'%Y%m%d_%H%M%S')"
# 当前这次训练的实际输出目录；默认自动按时间戳生成。
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${timestamp}_${JOB_NAME}}"
# 当前这次训练的日志文件。
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train_${JOB_NAME}_${timestamp}.log}"

TRAIN_CMD=(
  "${RUNNER[@]}"
  lerobot-train
  "--dataset.repo_id=${DATASET_REPO_ID}"
  "--dataset.root=${DATASET_ROOT}"
  "--policy.type=${POLICY_TYPE}"
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

if [[ -n "${DATASET_EPISODES}" ]]; then
  TRAIN_CMD+=("--dataset.episodes=${DATASET_EPISODES}")
fi

if [[ -n "${DIFFUSION_N_OBS_STEPS}" ]]; then
  TRAIN_CMD+=("--policy.n_obs_steps=${DIFFUSION_N_OBS_STEPS}")
fi

if [[ -n "${DIFFUSION_HORIZON}" ]]; then
  TRAIN_CMD+=("--policy.horizon=${DIFFUSION_HORIZON}")
fi

if [[ -n "${DIFFUSION_N_ACTION_STEPS}" ]]; then
  TRAIN_CMD+=("--policy.n_action_steps=${DIFFUSION_N_ACTION_STEPS}")
fi

if [[ -n "${DIFFUSION_NUM_TRAIN_TIMESTEPS}" ]]; then
  TRAIN_CMD+=("--policy.num_train_timesteps=${DIFFUSION_NUM_TRAIN_TIMESTEPS}")
fi

if [[ -n "${DIFFUSION_NUM_INFERENCE_STEPS}" ]]; then
  TRAIN_CMD+=("--policy.num_inference_steps=${DIFFUSION_NUM_INFERENCE_STEPS}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY_RUN] Would execute:"
  printf '%q ' "${TRAIN_CMD[@]}" "$@"
  echo
  echo "[DRY_RUN] Log file: ${LOG_FILE}"
  exit 0
fi

echo "[start] script=train_first_test.sh conda_env=${CONDA_ENV_NAME}"
echo "[start] dataset_repo_id=${DATASET_REPO_ID}"
echo "[start] dataset_root=${DATASET_ROOT}"
echo "[start] output_dir=${OUTPUT_DIR}"
echo "[start] policy_type=${POLICY_TYPE}"
echo "[start] policy_device=${POLICY_DEVICE}"
echo "[start] steps=${STEPS} batch_size=${BATCH_SIZE} num_workers=${NUM_WORKERS}"
echo "[start] save_freq=${SAVE_FREQ} eval_freq=${EVAL_FREQ} log_freq=${LOG_FREQ}"
echo "[start] log_file=${LOG_FILE}"

"${TRAIN_CMD[@]}" "$@" 2>&1 | tee "${LOG_FILE}"
