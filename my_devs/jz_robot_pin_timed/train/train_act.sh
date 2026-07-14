#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"

CONDA_ENV="${CONDA_ENV:-lerobot_flex}"
CONDA_BIN="${CONDA_BIN:-$(command -v conda 2>/dev/null || true)}"
if [[ -z "${CONDA_BIN}" ]]; then
  for candidate in "${HOME}/miniconda3/bin/conda" "${HOME}/anaconda3/bin/conda" /opt/conda/bin/conda; do
    if [[ -x "${candidate}" ]]; then
      CONDA_BIN="${candidate}"
      break
    fi
  done
fi
DATASET_NAME="${DATASET_NAME:-jz_robot_pin_timed_curated_42eps_20260713}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/tests/outputs/${DATASET_NAME}}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/${DATASET_NAME}}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
RESIZE="${RESIZE:-224,224}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-25}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CHECKPOINT_EVERY_EPOCHS="${CHECKPOINT_EVERY_EPOCHS:-5}"
USE_AMP="${USE_AMP:-false}"
RESIZE_TAG="${RESIZE//,/x}"
RUN_NAME="${RUN_NAME:-act_${DATASET_NAME}_e${EPOCHS}_b${BATCH_SIZE}_r${RESIZE_TAG}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/tests/outputs/${RUN_NAME}}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/tests/outputs/.cache/jz_robot_pin_timed_act}"
TRAINING_SCHEMA="${TRAINING_SCHEMA:-${DATASET_ROOT}/meta/jz_pin_training_schema.json}"
LEROBOT_TRAIN_BIN="${LEROBOT_TRAIN_BIN:-}"
if [[ -z "${LEROBOT_TRAIN_BIN}" ]]; then
  for candidate in \
    "$(command -v lerobot-train 2>/dev/null || true)" \
    "${HOME}/miniconda3/envs/${CONDA_ENV}/bin/lerobot-train" \
    "${HOME}/anaconda3/envs/${CONDA_ENV}/bin/lerobot-train" \
    "${HOME}/.local/bin/lerobot-train"; do
    if [[ -n "${candidate}" && -f "${candidate}" ]]; then
      LEROBOT_TRAIN_BIN="${candidate}"
      break
    fi
  done
fi

for value_name in EPOCHS BATCH_SIZE CHUNK_SIZE N_ACTION_STEPS CHECKPOINT_EVERY_EPOCHS; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[jz_pin_timed/train] ${value_name} must be a positive integer, got ${value}" >&2
    exit 2
  fi
done
if [[ -n "${STEPS_OVERRIDE:-}" ]] && ! [[ "${STEPS_OVERRIDE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[jz_pin_timed/train] STEPS_OVERRIDE must be a positive integer" >&2
  exit 2
fi
if (( N_ACTION_STEPS > CHUNK_SIZE )); then
  echo "[jz_pin_timed/train] N_ACTION_STEPS cannot exceed CHUNK_SIZE" >&2
  exit 2
fi
if [[ "${USE_AMP}" != "true" && "${USE_AMP}" != "false" ]]; then
  echo "[jz_pin_timed/train] USE_AMP must be true or false" >&2
  exit 2
fi
if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
  echo "[jz_pin_timed/train] missing dataset metadata: ${DATASET_ROOT}/meta/info.json" >&2
  exit 2
fi
if [[ -z "${CONDA_BIN}" || ! -x "${CONDA_BIN}" ]]; then
  echo "[jz_pin_timed/train] conda executable was not found" >&2
  exit 2
fi
CONDA_BASE="$(cd "$(dirname "${CONDA_BIN}")/.." && pwd)"
CONDA_PYTHON="${CONDA_PYTHON:-${CONDA_BASE}/envs/${CONDA_ENV}/bin/python}"
if [[ ! -x "${CONDA_PYTHON}" ]]; then
  echo "[jz_pin_timed/train] conda environment python was not found: ${CONDA_PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${TRAINING_SCHEMA}" ]]; then
  echo "[jz_pin_timed/train] missing strict training schema: ${TRAINING_SCHEMA}" >&2
  exit 2
fi
if [[ -z "${LEROBOT_TRAIN_BIN}" || ! -f "${LEROBOT_TRAIN_BIN}" ]]; then
  echo "[jz_pin_timed/train] lerobot-train tool was not found" >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "[jz_pin_timed/train] output already exists: ${OUTPUT_DIR}" >&2
  exit 2
fi

read -r TOTAL_EPISODES TOTAL_FRAMES FPS ACTION_DIM STATE_DIM < <(
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python - "${DATASET_ROOT}/meta/info.json" <<'PY'
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

if [[ "${ACTION_DIM}" != "18" || "${STATE_DIM}" != "18" ]]; then
  echo "[jz_pin_timed/train] expected raw18 action/state, got ${ACTION_DIM}/${STATE_DIM}" >&2
  exit 2
fi
for pair in \
  "EXPECTED_TOTAL_EPISODES:${EXPECTED_TOTAL_EPISODES:-}:${TOTAL_EPISODES}" \
  "EXPECTED_TOTAL_FRAMES:${EXPECTED_TOTAL_FRAMES:-}:${TOTAL_FRAMES}" \
  "EXPECTED_FPS:${EXPECTED_FPS:-}:${FPS}"; do
  IFS=: read -r expected_name expected_value actual_value <<< "${pair}"
  if [[ -n "${expected_value}" && "${actual_value}" != "${expected_value}" ]]; then
    echo "[jz_pin_timed/train] ${expected_name}=${expected_value}, got ${actual_value}" >&2
    exit 2
  fi
done

STEPS_PER_EPOCH=$(( (TOTAL_FRAMES + BATCH_SIZE - 1) / BATCH_SIZE ))
if [[ -n "${STEPS_OVERRIDE:-}" ]]; then
  STEPS="${STEPS_OVERRIDE}"
  SAVE_FREQ="${SAVE_FREQ_OVERRIDE:-${STEPS}}"
  EXPECTED_EPOCH_ARGS=()
else
  STEPS=$(( STEPS_PER_EPOCH * EPOCHS ))
  SAVE_FREQ=$(( STEPS_PER_EPOCH * CHECKPOINT_EVERY_EPOCHS ))
  EXPECTED_EPOCH_ARGS=(--expected-epochs "${EPOCHS}")
fi

mkdir -p "${CACHE_ROOT}/huggingface" "${CACHE_ROOT}/torch" "$(dirname "${OUTPUT_DIR}")"
export HF_HOME="${CACHE_ROOT}/huggingface"
export TORCH_HOME="${CACHE_ROOT}/torch"
export WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1
BASE_PYTHONPATH="${SCRIPT_DIR}:${REPO_ROOT}/src:${PYTHONPATH:-}"

if [[ "$(head -n 1 "${LEROBOT_TRAIN_BIN}")" == "#!${CONDA_PYTHON}" ]]; then
  TRAIN_COMMAND=("${LEROBOT_TRAIN_BIN}")
else
  TRAIN_COMMAND=("${CONDA_PYTHON}" "${LEROBOT_TRAIN_BIN}")
fi

echo "[jz_pin_timed/train] tool=${LEROBOT_TRAIN_BIN} interpreter=${CONDA_PYTHON}"
echo "[jz_pin_timed/train] dataset=${DATASET_ROOT} episodes=${TOTAL_EPISODES} frames=${TOTAL_FRAMES} fps=${FPS}"
echo "[jz_pin_timed/train] raw18 boundary -> ACT model16; cameras=3 resize=${RESIZE}"
echo "[jz_pin_timed/train] batch=${BATCH_SIZE} steps_per_epoch=${STEPS_PER_EPOCH} steps=${STEPS} save_freq=${SAVE_FREQ}"
echo "[jz_pin_timed/train] output=${OUTPUT_DIR} amp=${USE_AMP}"

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" \
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python my_devs/jz_robot_pin_timed/data_check/check_training_projection.py \
  --dataset-root "${DATASET_ROOT}" \
  --manifest "${TRAINING_SCHEMA}"

# Run the installed official console entry script under the mandated conda
# interpreter.  The opt-in sitecustomize hook only supplies the JZ dataset view
# and serialized processor boundary; LeRobot owns configuration, optimization,
# checkpointing, logging and the complete training loop.
PYTHONPATH="${SCRIPT_DIR}/cli_hook:${BASE_PYTHONPATH}" \
JZ_PIN_ENABLE_LEROBOT_TRAIN_HOOK=1 \
JZ_PIN_TRAINING_SCHEMA="${TRAINING_SCHEMA}" \
LEROBOT_ACT_RESIZE="${RESIZE}" \
  "${TRAIN_COMMAND[@]}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --policy.use_amp="${USE_AMP}" \
  --policy.chunk_size="${CHUNK_SIZE}" \
  --policy.n_action_steps="${N_ACTION_STEPS}" \
  --policy.push_to_hub=false \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${RUN_NAME}" \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --save_checkpoint=true \
  --save_freq="${SAVE_FREQ}" \
  --log_freq="${LOG_FREQ_OVERRIDE:-${STEPS_PER_EPOCH}}" \
  --eval_freq=0 \
  --num_workers="${NUM_WORKERS}" \
  --wandb.enable=false

PYTHONPATH="${BASE_PYTHONPATH}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python "${SCRIPT_DIR}/verify_act_training.py" \
  --output-dir "${OUTPUT_DIR}" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-repo-id "${DATASET_REPO_ID}" \
  --expected-steps "${STEPS}" \
  --steps-per-epoch "${STEPS_PER_EPOCH}" \
  --expected-resize "${RESIZE}" \
  --expected-schema "${TRAINING_SCHEMA}" \
  --device cuda \
  "${EXPECTED_EPOCH_ARGS[@]}"
