#!/usr/bin/env bash
set -Eeuo pipefail

# Agilex GROOT training entrypoint.
# Defaults target the local first_test_right dataset and the repo-local GR00T base model.

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

is_local_path() {
  local value="$1"
  [[ "${value}" == /* || "${value}" == ./* || "${value}" == ../* ]]
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR_DEFAULT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_DIR_DEFAULT}" ]]; then
  REPO_DIR_DEFAULT="$(find_repo_root "${SCRIPT_DIR}" || true)"
fi
if [[ -z "${REPO_DIR_DEFAULT}" ]]; then
  echo "[ERROR] failed to locate repo root from script dir: ${SCRIPT_DIR}" >&2
  exit 1
fi

REPO_DIR="${REPO_DIR:-${REPO_DIR_DEFAULT}}"
cd "${REPO_DIR}"

CONDA_ENV="${CONDA_ENV:-lerobot_flex}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
DRY_RUN="${DRY_RUN:-false}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train_groot_agilex_${RUN_ID}.log}"

DATASET_REPO_ID="${DATASET_REPO_ID:-first_test_right}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_DIR}/datasets/first_test_right}"
DATASET_VIDEO_BACKEND="${DATASET_VIDEO_BACKEND:-pyav}"

DEFAULT_LOCAL_BASE_MODEL_PATH="${REPO_DIR}/assets/modelscope/GR00T-N1.5-3B"
DEFAULT_REMOTE_BASE_MODEL_PATH="nvidia/GR00T-N1.5-3B"
if [[ -e "${DEFAULT_LOCAL_BASE_MODEL_PATH}" || -L "${DEFAULT_LOCAL_BASE_MODEL_PATH}" ]]; then
  DEFAULT_BASE_MODEL_PATH="${DEFAULT_LOCAL_BASE_MODEL_PATH}"
else
  DEFAULT_BASE_MODEL_PATH="${DEFAULT_REMOTE_BASE_MODEL_PATH}"
fi

POLICY_TYPE="${POLICY_TYPE:-groot}"
POLICY_REPO_ID="${POLICY_REPO_ID:-}"
POLICY_PUSH_TO_HUB="${POLICY_PUSH_TO_HUB:-false}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
POLICY_BASE_MODEL_PATH="${POLICY_BASE_MODEL_PATH:-${DEFAULT_BASE_MODEL_PATH}}"
POLICY_TOKENIZER_ASSETS_REPO="${POLICY_TOKENIZER_ASSETS_REPO:-lerobot/eagle2hg-processor-groot-n1p5}"
POLICY_CHUNK_SIZE="${POLICY_CHUNK_SIZE:-16}"
POLICY_N_ACTION_STEPS="${POLICY_N_ACTION_STEPS:-16}"
POLICY_USE_BF16="${POLICY_USE_BF16:-true}"
POLICY_EMBODIMENT_TAG="${POLICY_EMBODIMENT_TAG:-new_embodiment}"
POLICY_TUNE_LLM="${POLICY_TUNE_LLM:-false}"
POLICY_TUNE_VISUAL="${POLICY_TUNE_VISUAL:-false}"
POLICY_TUNE_PROJECTOR="${POLICY_TUNE_PROJECTOR:-true}"
POLICY_TUNE_DIFFUSION_MODEL="${POLICY_TUNE_DIFFUSION_MODEL:-true}"

BATCH_SIZE="${BATCH_SIZE:-4}"
STEPS="${STEPS:-20000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
EVAL_FREQ="${EVAL_FREQ:--1}"
LOG_FREQ="${LOG_FREQ:-100}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-1000}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
JOB_NAME="${JOB_NAME:-groot_agilex_first_test_right}"

RESUME="${RESUME:-false}"
CONFIG_PATH="${CONFIG_PATH:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/train/${JOB_NAME}_${RUN_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/bs${BATCH_SIZE}_${RUN_ID}}"

mkdir -p "${LOG_DIR}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH." >&2
  exit 1
fi

RUNNER=()
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
  RUNNER=(conda run --no-capture-output -n "${CONDA_ENV}")
fi

if [[ "${POLICY_DEVICE}" == "auto" ]]; then
  DETECTED_DEVICE="$("${RUNNER[@]}" python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" || true)"
  POLICY_DEVICE="${DETECTED_DEVICE:-cpu}"
fi

if [[ "${POLICY_DEVICE}" != "cuda" ]]; then
  echo "[ERROR] GROOT training requires CUDA. Resolved policy device: ${POLICY_DEVICE}" >&2
  exit 1
fi

if [[ "${POLICY_PUSH_TO_HUB}" == "true" && -z "${POLICY_REPO_ID}" ]]; then
  echo "[ERROR] POLICY_REPO_ID is required when POLICY_PUSH_TO_HUB=true." >&2
  exit 1
fi

if [[ "${RESUME}" == "true" ]]; then
  if [[ -z "${CONFIG_PATH}" ]]; then
    echo "[ERROR] RESUME=true but CONFIG_PATH is empty." >&2
    exit 1
  fi
  if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[ERROR] resume config file not found: ${CONFIG_PATH}" >&2
    exit 1
  fi
else
  if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
    echo "[ERROR] dataset metadata not found: ${DATASET_ROOT}/meta/info.json" >&2
    exit 1
  fi
  if is_local_path "${POLICY_BASE_MODEL_PATH}" && [[ ! -e "${POLICY_BASE_MODEL_PATH}" && ! -L "${POLICY_BASE_MODEL_PATH}" ]]; then
    echo "[ERROR] local base model path not found: ${POLICY_BASE_MODEL_PATH}" >&2
    exit 1
  fi
  if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "[ERROR] output dir already exists: ${OUTPUT_DIR}" >&2
    echo "        Set a new RUN_ID/OUTPUT_DIR, or use RESUME=true with CONFIG_PATH." >&2
    exit 1
  fi
fi

export PYTHONUNBUFFERED=1
export HF_ENDPOINT
export TOKENIZERS_PARALLELISM

CMD=(
  "${RUNNER[@]}"
  python
  -m
  lerobot.scripts.lerobot_train
)

if [[ "${RESUME}" == "true" ]]; then
  CMD+=(
    "--config_path=${CONFIG_PATH}"
    "--resume=true"
  )
else
  CMD+=(
    "--policy.type=${POLICY_TYPE}"
    "--policy.push_to_hub=${POLICY_PUSH_TO_HUB}"
    "--policy.device=${POLICY_DEVICE}"
    "--policy.base_model_path=${POLICY_BASE_MODEL_PATH}"
    "--policy.tokenizer_assets_repo=${POLICY_TOKENIZER_ASSETS_REPO}"
    "--policy.chunk_size=${POLICY_CHUNK_SIZE}"
    "--policy.n_action_steps=${POLICY_N_ACTION_STEPS}"
    "--policy.use_bf16=${POLICY_USE_BF16}"
    "--policy.embodiment_tag=${POLICY_EMBODIMENT_TAG}"
    "--policy.tune_llm=${POLICY_TUNE_LLM}"
    "--policy.tune_visual=${POLICY_TUNE_VISUAL}"
    "--policy.tune_projector=${POLICY_TUNE_PROJECTOR}"
    "--policy.tune_diffusion_model=${POLICY_TUNE_DIFFUSION_MODEL}"
    "--dataset.repo_id=${DATASET_REPO_ID}"
    "--dataset.root=${DATASET_ROOT}"
    "--dataset.video_backend=${DATASET_VIDEO_BACKEND}"
    "--output_dir=${OUTPUT_DIR}"
    "--job_name=${JOB_NAME}"
    "--batch_size=${BATCH_SIZE}"
    "--steps=${STEPS}"
    "--save_freq=${SAVE_FREQ}"
    "--eval_freq=${EVAL_FREQ}"
    "--log_freq=${LOG_FREQ}"
    "--num_workers=${NUM_WORKERS}"
    "--seed=${SEED}"
    "--wandb.enable=${WANDB_ENABLE}"
  )
  if [[ -n "${POLICY_REPO_ID}" ]]; then
    CMD+=("--policy.repo_id=${POLICY_REPO_ID}")
  fi
fi

echo "[INFO] Repo: ${REPO_DIR}"
echo "[INFO] Conda env: ${CONDA_ENV}"
echo "[INFO] Log file: ${LOG_FILE}"
echo "[INFO] Resume: ${RESUME}"
if [[ "${RESUME}" == "true" ]]; then
  echo "[INFO] Resume config: ${CONFIG_PATH}"
else
  echo "[INFO] Dataset repo_id: ${DATASET_REPO_ID}"
  echo "[INFO] Dataset root: ${DATASET_ROOT}"
  echo "[INFO] Base model path: ${POLICY_BASE_MODEL_PATH}"
  echo "[INFO] Output dir: ${OUTPUT_DIR}"
  echo "[INFO] Steps: ${STEPS}, batch_size: ${BATCH_SIZE}, num_workers: ${NUM_WORKERS}"
fi
echo "[INFO] Command:"
printf '  %q' "${CMD[@]}" "$@"
echo

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "[INFO] DRY_RUN=true, exit without execution."
  exit 0
fi

"${CMD[@]}" "$@" 2>&1 | tee "${LOG_FILE}"
