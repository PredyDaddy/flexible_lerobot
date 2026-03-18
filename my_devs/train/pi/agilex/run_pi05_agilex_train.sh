#!/usr/bin/env bash
set -euo pipefail

find_repo_root() {
  local dir="$1"
  while [[ "${dir}" != "/" ]]; do
    if [[ -f "${dir}/pyproject.toml" && -d "${dir}/src/lerobot" ]]; then
      printf '%s\n' "${dir}"
      return 0
    fi
    dir="$(dirname "${dir}")"
  done
  return 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_DEFAULT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${PROJECT_ROOT_DEFAULT}" ]]; then
  PROJECT_ROOT_DEFAULT="$(find_repo_root "${SCRIPT_DIR}" || true)"
fi
if [[ -z "${PROJECT_ROOT_DEFAULT}" ]]; then
  echo "[ERROR] failed to locate repo root from script dir: ${SCRIPT_DIR}" >&2
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_ROOT_DEFAULT}}"
cd "${PROJECT_ROOT}"

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-lerobot_flex}"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[ERROR] conda init script not found: ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

DATASET_REPO_ID="${DATASET_REPO_ID:-local/first_test_right}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/datasets/first_test_right}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/assets/modelscope/lerobot/pi05_base}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${PROJECT_ROOT}/google/paligemma-3b-pt-224}"
RUN_MODE="${RUN_MODE:-smoke}"  # smoke | full

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "[ERROR] DATASET_ROOT not found: ${DATASET_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
  echo "[ERROR] dataset metadata not found: ${DATASET_ROOT}/meta/info.json" >&2
  exit 1
fi

DATASET_ROBOT_TYPE="$(
  python - <<PY
import json
from pathlib import Path
info = json.loads(Path("${DATASET_ROOT}/meta/info.json").read_text())
print(info.get("robot_type", ""))
PY
)"
if [[ "${DATASET_ROBOT_TYPE}" != "agilex" ]]; then
  echo "[ERROR] dataset robot_type must be agilex, got: ${DATASET_ROBOT_TYPE}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "[ERROR] MODEL_DIR not ready: ${MODEL_DIR}" >&2
  exit 1
fi

if [[ ! -d "${TOKENIZER_DIR}" ]]; then
  cat >&2 <<EOF
[ERROR] offline tokenizer path not found: ${TOKENIZER_DIR}
PI0.5 preprocessor needs "google/paligemma-3b-pt-224".
Please create this directory (or symlink) before training.
EOF
  exit 1
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/train/pi05_agilex_first_test_right_${RUN_ID}}"
TARGET_EPOCHS="${TARGET_EPOCHS:-}"

case "${RUN_MODE}" in
  smoke)
    STEPS="${STEPS:-20}"
    BATCH_SIZE="${BATCH_SIZE:-2}"
    SAVE_FREQ="${SAVE_FREQ:-20}"
    LOG_FREQ="${LOG_FREQ:-1}"
    JOB_NAME="${JOB_NAME:-pi05_agilex_first_test_right_smoke}"
    ;;
  full)
    STEPS="${STEPS:-15000}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    SAVE_FREQ="${SAVE_FREQ:-1000}"
    LOG_FREQ="${LOG_FREQ:-200}"
    JOB_NAME="${JOB_NAME:-pi05_agilex_first_test_right_full}"
    ;;
  *)
    echo "[ERROR] RUN_MODE must be smoke or full, got: ${RUN_MODE}" >&2
    exit 1
    ;;
esac

NUM_WORKERS="${NUM_WORKERS:-4}"
POLICY_COMPILE_MODEL="${POLICY_COMPILE_MODEL:-false}"
POLICY_GRADIENT_CHECKPOINTING="${POLICY_GRADIENT_CHECKPOINTING:-true}"
POLICY_DTYPE="${POLICY_DTYPE:-bfloat16}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
POLICY_NORMALIZATION_MAPPING="${POLICY_NORMALIZATION_MAPPING:-{\"ACTION\":\"MEAN_STD\",\"STATE\":\"MEAN_STD\",\"VISUAL\":\"IDENTITY\"}}"

TOTAL_FRAMES="$(
  python - <<PY
import json
from pathlib import Path
info = json.loads(Path("${DATASET_ROOT}/meta/info.json").read_text())
print(int(info["total_frames"]))
PY
)"

if [[ -n "${TARGET_EPOCHS}" ]]; then
  # When TARGET_EPOCHS is provided, compute steps automatically:
  # steps = ceil(target_epochs * total_frames / batch_size)
  STEPS="$(
    python - <<PY
import math
target_epochs = float("${TARGET_EPOCHS}")
batch_size = int("${BATCH_SIZE}")
total_frames = int("${TOTAL_FRAMES}")
if target_epochs <= 0:
    raise SystemExit("TARGET_EPOCHS must be > 0")
if batch_size <= 0:
    raise SystemExit("BATCH_SIZE must be > 0")
print(int(math.ceil(target_epochs * total_frames / batch_size)))
PY
  )"
fi

OUT_DIR="${OUTPUT_ROOT}/bs${BATCH_SIZE}_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${OUT_DIR}.log"
mkdir -p "${OUTPUT_ROOT}"

echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[INFO] RUN_MODE=${RUN_MODE}"
echo "[INFO] DATASET_REPO_ID=${DATASET_REPO_ID}"
echo "[INFO] DATASET_ROOT=${DATASET_ROOT}"
echo "[INFO] DATASET_ROBOT_TYPE=${DATASET_ROBOT_TYPE}"
echo "[INFO] DATASET_TOTAL_FRAMES=${TOTAL_FRAMES}"
echo "[INFO] MODEL_DIR=${MODEL_DIR}"
echo "[INFO] TOKENIZER_DIR=${TOKENIZER_DIR}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] LOG_FILE=${LOG_FILE}"
if [[ -n "${TARGET_EPOCHS}" ]]; then
  echo "[INFO] TARGET_EPOCHS=${TARGET_EPOCHS} (STEPS auto-computed)"
else
  echo "[INFO] TARGET_EPOCHS=<unset> (STEPS from RUN_MODE/default/env)"
fi
echo "[INFO] STEPS=${STEPS}"
echo "[INFO] BATCH_SIZE=${BATCH_SIZE}"
echo "[INFO] EST_EPOCHS=$(python - <<PY
steps = int("${STEPS}")
batch_size = int("${BATCH_SIZE}")
total_frames = int("${TOTAL_FRAMES}")
print(f"{steps * batch_size / total_frames:.2f}")
PY
)"
echo "[INFO] POLICY_COMPILE_MODEL=${POLICY_COMPILE_MODEL}"
echo "[INFO] POLICY_GRADIENT_CHECKPOINTING=${POLICY_GRADIENT_CHECKPOINTING}"

CMD=(
  python "${PROJECT_ROOT}/src/lerobot/scripts/lerobot_train.py"
  "--dataset.repo_id=${DATASET_REPO_ID}"
  "--dataset.root=${DATASET_ROOT}"
  "--dataset.video_backend=pyav"
  "--policy.type=pi05"
  "--policy.pretrained_path=${MODEL_DIR}"
  "--policy.push_to_hub=false"
  "--policy.compile_model=${POLICY_COMPILE_MODEL}"
  "--policy.gradient_checkpointing=${POLICY_GRADIENT_CHECKPOINTING}"
  "--policy.dtype=${POLICY_DTYPE}"
  "--policy.device=${POLICY_DEVICE}"
  "--policy.normalization_mapping=${POLICY_NORMALIZATION_MAPPING}"
  "--output_dir=${OUT_DIR}"
  "--job_name=${JOB_NAME}"
  "--batch_size=${BATCH_SIZE}"
  "--num_workers=${NUM_WORKERS}"
  "--steps=${STEPS}"
  "--save_freq=${SAVE_FREQ}"
  "--log_freq=${LOG_FREQ}"
  "--wandb.enable=false"
)

echo "[INFO] Train command:"
printf '  %q' "${CMD[@]}"
for arg in "$@"; do
  printf ' %q' "${arg}"
done
echo

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  echo "[INFO] DRY_RUN=true, exit without execution."
  exit 0
fi

{
  "${CMD[@]}" "$@"
} |& tee "${LOG_FILE}"

echo "[INFO] Training command finished."
if [[ -d "${OUT_DIR}/checkpoints" ]]; then
  echo "[INFO] Checkpoints generated under: ${OUT_DIR}/checkpoints"
  find "${OUT_DIR}/checkpoints" -maxdepth 1 -mindepth 1 -type d -printf '  - %f\n' | sort
fi
