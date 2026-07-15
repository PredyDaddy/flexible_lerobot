#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-lerobot_flex}"
CONDA_ROOT="${CONDA_ROOT:-/home/cqy/miniconda3}"
CONDA_PYTHON="${CONDA_PYTHON:-${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python}"
LEROBOT_TRAIN_BIN="${LEROBOT_TRAIN_BIN:-${CONDA_ROOT}/envs/${CONDA_ENV}/bin/lerobot-train}"

DATASET_NAME="${DATASET_NAME:-jz_robot_pin_timed_curated_42eps_20260713}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/${DATASET_NAME}}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/${DATASET_NAME}}"
TRAINING_SCHEMA="${TRAINING_SCHEMA:-${DATASET_ROOT}/meta/jz_pin_training_schema.json}"

PI05_BASE_PATH="${PI05_BASE_PATH:-/data/cqy_workspace/flexible_lerobot/assets/modelscope/lerobot/pi05_base}"
PALIGEMMA_TOKENIZER_PATH="${PALIGEMMA_TOKENIZER_PATH:-/data/cqy_workspace/flexible_lerobot/assets/modelscope/google/paligemma-3b-pt-224}"

EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CHECKPOINT_EVERY_EPOCHS="${CHECKPOINT_EVERY_EPOCHS:-5}"
LOG_FREQ="${LOG_FREQ:-100}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
NORMALIZATION_MODE="${NORMALIZATION_MODE:-QUANTILES}"
DRY_RUN="${DRY_RUN:-false}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-pi05_${DATASET_NAME}_e${EPOCHS}_b${BATCH_SIZE}_${RUN_STAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/${RUN_NAME}}"
RUNTIME_DIR="${RUNTIME_DIR:-${SCRIPT_DIR}/runtime}"
MODEL16_STATS="${MODEL16_STATS:-${RUNTIME_DIR}/model16_stats.json}"

for value_name in EPOCHS BATCH_SIZE CHECKPOINT_EVERY_EPOCHS LOG_FREQ; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[jz/pi05/train] ${value_name} must be a positive integer, got ${value}" >&2
    exit 2
  fi
done
if ! [[ "${NUM_WORKERS}" =~ ^[0-9]+$ ]]; then
  echo "[jz/pi05/train] NUM_WORKERS must be a non-negative integer" >&2
  exit 2
fi
if [[ -n "${STEPS_OVERRIDE:-}" ]] && ! [[ "${STEPS_OVERRIDE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[jz/pi05/train] STEPS_OVERRIDE must be a positive integer" >&2
  exit 2
fi
if [[ "${SAVE_CHECKPOINT}" != "true" && "${SAVE_CHECKPOINT}" != "false" ]]; then
  echo "[jz/pi05/train] SAVE_CHECKPOINT must be true or false" >&2
  exit 2
fi
if [[ "${DRY_RUN}" != "true" && "${DRY_RUN}" != "false" ]]; then
  echo "[jz/pi05/train] DRY_RUN must be true or false" >&2
  exit 2
fi
if [[ "${NORMALIZATION_MODE}" != "QUANTILES" && "${NORMALIZATION_MODE}" != "MEAN_STD" ]]; then
  echo "[jz/pi05/train] NORMALIZATION_MODE must be QUANTILES or MEAN_STD" >&2
  exit 2
fi

for required_file in \
  "${CONDA_PYTHON}" \
  "${LEROBOT_TRAIN_BIN}" \
  "${DATASET_ROOT}/meta/info.json" \
  "${TRAINING_SCHEMA}" \
  "${PI05_BASE_PATH}/model.safetensors" \
  "${PALIGEMMA_TOKENIZER_PATH}/tokenizer.json"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "[jz/pi05/train] required file is missing: ${required_file}" >&2
    exit 2
  fi
done
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "[jz/pi05/train] output already exists: ${OUTPUT_DIR}" >&2
  exit 2
fi

read -r TOTAL_EPISODES TOTAL_FRAMES FPS RAW_ACTION_DIM RAW_STATE_DIM < <(
  "${CONDA_PYTHON}" - "${DATASET_ROOT}/meta/info.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    info = json.load(stream)
features = info["features"]
print(
    info["total_episodes"],
    info["total_frames"],
    info["fps"],
    features["action"]["shape"][0],
    features["observation.state"]["shape"][0],
)
PY
)
if [[ "${TOTAL_EPISODES}" != "42" || "${TOTAL_FRAMES}" != "8370" || "${FPS}" != "20" ]]; then
  echo "[jz/pi05/train] expected curated 42eps/8370frames/20fps, got ${TOTAL_EPISODES}/${TOTAL_FRAMES}/${FPS}" >&2
  exit 2
fi
if [[ "${RAW_ACTION_DIM}" != "18" || "${RAW_STATE_DIM}" != "18" ]]; then
  echo "[jz/pi05/train] expected raw18 action/state, got ${RAW_ACTION_DIM}/${RAW_STATE_DIM}" >&2
  exit 2
fi

STEPS_PER_EPOCH=$(( (TOTAL_FRAMES + BATCH_SIZE - 1) / BATCH_SIZE ))
if [[ -n "${STEPS_OVERRIDE:-}" ]]; then
  STEPS="${STEPS_OVERRIDE}"
else
  STEPS=$(( STEPS_PER_EPOCH * EPOCHS ))
fi
SAVE_FREQ=$(( STEPS_PER_EPOCH * CHECKPOINT_EVERY_EPOCHS ))
if (( SAVE_FREQ > STEPS )); then
  SAVE_FREQ="${STEPS}"
fi

mkdir -p "${LOG_DIR}" "${RUNTIME_DIR}/google" "${RUNTIME_DIR}/cache/huggingface" \
  "${RUNTIME_DIR}/cache/torch" "$(dirname "${OUTPUT_DIR}")"
ln -sfn "${PALIGEMMA_TOKENIZER_PATH}" "${RUNTIME_DIR}/google/paligemma-3b-pt-224"

export HF_HOME="${RUNTIME_DIR}/cache/huggingface"
export TORCH_HOME="${RUNTIME_DIR}/cache/torch"
export XDG_CACHE_HOME="${RUNTIME_DIR}/cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

BASE_PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
PYTHONPATH="${BASE_PYTHONPATH}" "${CONDA_PYTHON}" "${SCRIPT_DIR}/compute_model16_stats.py" \
  --dataset-root "${DATASET_ROOT}" \
  --schema "${TRAINING_SCHEMA}" \
  --output "${MODEL16_STATS}"

PYTHONPATH="${BASE_PYTHONPATH}" "${CONDA_PYTHON}" "${SCRIPT_DIR}/preflight.py" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-repo-id "${DATASET_REPO_ID}" \
  --schema "${TRAINING_SCHEMA}" \
  --stats "${MODEL16_STATS}" \
  --pi05-base "${PI05_BASE_PATH}" \
  --tokenizer "${PALIGEMMA_TOKENIZER_PATH}"

NORMALIZATION_MAPPING="{\"ACTION\":\"${NORMALIZATION_MODE}\",\"STATE\":\"${NORMALIZATION_MODE}\",\"VISUAL\":\"IDENTITY\"}"
TRAIN_COMMAND=(
  "${LEROBOT_TRAIN_BIN}"
  "--dataset.repo_id=${DATASET_REPO_ID}"
  "--dataset.root=${DATASET_ROOT}"
  "--dataset.video_backend=pyav"
  "--policy.type=pi05"
  "--policy.pretrained_path=${PI05_BASE_PATH}"
  "--policy.compile_model=false"
  "--policy.gradient_checkpointing=true"
  "--policy.dtype=bfloat16"
  "--policy.device=cuda"
  "--policy.normalization_mapping=${NORMALIZATION_MAPPING}"
  "--policy.push_to_hub=false"
  "--output_dir=${OUTPUT_DIR}"
  "--job_name=${RUN_NAME}"
  "--batch_size=${BATCH_SIZE}"
  "--steps=${STEPS}"
  "--save_checkpoint=${SAVE_CHECKPOINT}"
  "--save_freq=${SAVE_FREQ}"
  "--log_freq=${LOG_FREQ}"
  "--eval_freq=0"
  "--num_workers=${NUM_WORKERS}"
  "--wandb.enable=false"
)

{
  echo "[jz/pi05/train] conda_env=${CONDA_ENV} python=${CONDA_PYTHON}"
  echo "[jz/pi05/train] dataset=${DATASET_ROOT} episodes=${TOTAL_EPISODES} frames=${TOTAL_FRAMES} fps=${FPS}"
  echo "[jz/pi05/train] boundary=raw18->model16 cameras=3 normalization=${NORMALIZATION_MODE}"
  echo "[jz/pi05/train] pretrained=${PI05_BASE_PATH} tokenizer=${PALIGEMMA_TOKENIZER_PATH}"
  echo "[jz/pi05/train] batch=${BATCH_SIZE} steps_per_epoch=${STEPS_PER_EPOCH} epochs=${EPOCHS} steps=${STEPS}"
  echo "[jz/pi05/train] output=${OUTPUT_DIR} save_checkpoint=${SAVE_CHECKPOINT}"
  printf '[jz/pi05/train] command='
  printf '%q ' "${TRAIN_COMMAND[@]}"
  printf '\n'
} | tee "${LOG_DIR}/launch.log"

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "[jz/pi05/train] DRY_RUN PASS; training command was not started" | tee "${LOG_DIR}/DRY_RUN_SUCCESS"
  exit 0
fi

GPU_MONITOR_PID=""
if command -v nvidia-smi >/dev/null 2>&1; then
  (
    while true; do
      date '+%F %T'
      nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits
      sleep 30
    done
  ) > "${LOG_DIR}/gpu_usage.log" 2>&1 &
  GPU_MONITOR_PID=$!
fi
cleanup() {
  if [[ -n "${GPU_MONITOR_PID}" ]]; then
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

cd "${RUNTIME_DIR}"
PYTHONPATH="${SCRIPT_DIR}/cli_hook:${SCRIPT_DIR}:${BASE_PYTHONPATH}" \
JZ_PI05_ENABLE_TRAIN_HOOK=1 \
JZ_PI05_TRAINING_SCHEMA="${TRAINING_SCHEMA}" \
JZ_PI05_MODEL16_STATS="${MODEL16_STATS}" \
  "${TRAIN_COMMAND[@]}" 2>&1 | tee "${LOG_DIR}/train_terminal.log"

mkdir -p "${OUTPUT_DIR}"
printf 'status=PASS\nrun_name=%s\noutput_dir=%s\nsteps=%s\nbatch_size=%s\n' \
  "${RUN_NAME}" "${OUTPUT_DIR}" "${STEPS}" "${BATCH_SIZE}" > "${LOG_DIR}/SUCCESS"
cp "${LOG_DIR}/SUCCESS" "${OUTPUT_DIR}/TRAINING_SUCCESS"
echo "[jz/pi05/train] PASS output=${OUTPUT_DIR} log=${LOG_DIR}/train_terminal.log"
