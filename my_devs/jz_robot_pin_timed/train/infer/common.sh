#!/usr/bin/env bash

INFER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(cd "${INFER_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${INFER_DIR}/../../../.." && pwd)"

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
if [[ -z "${CONDA_BIN}" || ! -x "${CONDA_BIN}" ]]; then
  echo "[jz_pin_timed/infer] conda executable was not found" >&2
  exit 2
fi

CONDA_BASE="$(cd "$(dirname "${CONDA_BIN}")/.." && pwd)"
CONDA_PYTHON="${CONDA_PYTHON:-${CONDA_BASE}/envs/${CONDA_ENV}/bin/python}"
if [[ ! -x "${CONDA_PYTHON}" ]]; then
  echo "[jz_pin_timed/infer] conda environment python was not found: ${CONDA_PYTHON}" >&2
  exit 2
fi

DEFAULT_RUN_DIR="${REPO_ROOT}/tests/outputs/act_jz_robot_pin_timed_curated_42eps_20260713_e40_b16_r224x224"
DEFAULT_POLICY_PATH="${DEFAULT_RUN_DIR}/checkpoints/last/pretrained_model"
POLICY_PATH="${POLICY_PATH:-${DEFAULT_POLICY_PATH}}"
REFERENCE_DATASET_ROOT="${REFERENCE_DATASET_ROOT:-${REPO_ROOT}/tests/outputs/jz_robot_pin_timed_curated_42eps_20260713}"
REFERENCE_DATASET_REPO_ID="${REFERENCE_DATASET_REPO_ID:-local/jz_robot_pin_timed_curated_42eps_20260713}"

infer_require_policy() {
  local required=(
    config.json
    model.safetensors
    policy_preprocessor.json
    policy_postprocessor.json
    train_config.json
  )
  local name
  if [[ ! -d "${POLICY_PATH}" ]]; then
    echo "[jz_pin_timed/infer] policy directory does not exist: ${POLICY_PATH}" >&2
    echo "[jz_pin_timed/infer] set POLICY_PATH to a checkpoints/.../pretrained_model directory" >&2
    exit 2
  fi
  for name in "${required[@]}"; do
    if [[ ! -f "${POLICY_PATH}/${name}" ]]; then
      echo "[jz_pin_timed/infer] checkpoint is missing ${name}: ${POLICY_PATH}" >&2
      exit 2
    fi
  done

  local resolved_policy checkpoint_dir checkpoint_step configured_steps
  resolved_policy="$(realpath -e "${POLICY_PATH}")"
  checkpoint_dir="$(basename "$(dirname "${resolved_policy}")")"
  checkpoint_step="${checkpoint_dir}"
  configured_steps="$(
    "${CONDA_PYTHON}" - "${resolved_policy}/train_config.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    print(int(json.load(stream)["steps"]))
PY
  )"
  if ! [[ "${checkpoint_step}" =~ ^[0-9]+$ ]]; then
    echo "[jz_pin_timed/infer] cannot determine checkpoint step from ${resolved_policy}" >&2
    exit 2
  fi
  checkpoint_step="$((10#${checkpoint_step}))"
  if (( checkpoint_step != configured_steps )); then
    echo "[jz_pin_timed/infer] checkpoint is not the configured final step:" \
      "${checkpoint_step}/${configured_steps} (${resolved_policy})" >&2
    if [[ "${JZ_POLICY_INFERENCE_ARMED:-}" == "1" ]]; then
      echo "[jz_pin_timed/infer] armed inference never accepts an incomplete checkpoint" >&2
      exit 2
    fi
    if [[ "${ALLOW_INCOMPLETE_CHECKPOINT:-0}" != "1" ]]; then
      echo "[jz_pin_timed/infer] wait for training to finish, or set" \
        "ALLOW_INCOMPLETE_CHECKPOINT=1 for offline/dry-run inspection only" >&2
      exit 2
    fi
    echo "[jz_pin_timed/infer] WARNING: explicitly allowing an incomplete checkpoint for non-armed use"
  fi
}

infer_require_reference_dataset() {
  if [[ ! -f "${REFERENCE_DATASET_ROOT}/meta/info.json" ]]; then
    echo "[jz_pin_timed/infer] reference dataset is missing: ${REFERENCE_DATASET_ROOT}" >&2
    exit 2
  fi
}

infer_require_armed_confirmation() {
  if [[ "${JZ_ROBOT_PIN_ARMED:-}" != "1" ]]; then
    echo "[jz_pin_timed/infer] refusing armed inference: set JZ_ROBOT_PIN_ARMED=1" >&2
    exit 2
  fi
  if [[ "${I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT:-}" != "1" ]]; then
    echo "[jz_pin_timed/infer] refusing armed inference:" \
      "set I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1" >&2
    exit 2
  fi
  if [[ "${JZ_POLICY_INFERENCE_ARMED:-}" != "1" ]]; then
    echo "[jz_pin_timed/infer] refusing armed inference: set JZ_POLICY_INFERENCE_ARMED=1" >&2
    exit 2
  fi
}

infer_print_command() {
  printf '  %q' "$@"
  echo
}
