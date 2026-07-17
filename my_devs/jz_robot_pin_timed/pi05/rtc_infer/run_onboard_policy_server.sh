#!/usr/bin/env bash
set -euo pipefail

# Real-machine policy-server launcher for the audited PI0.5 run. The final
# checkpoint remains the default. The known 010470 checkpoint requires a
# separate, exact confirmation and cannot open the gate for any other step.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI05_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

RUN_NAME="pi05_jz_robot_pin_timed_curated_42eps_20260713_e15_b8_20260714_202432"
FINAL_STEP=15705
FINAL_CHECKPOINT_DIR=015705
INTERMEDIATE_STEP=10470
INTERMEDIATE_CHECKPOINT_DIR=010470
ROOT_CHECKPOINT="${REPO_ROOT}/${RUN_NAME}/checkpoints/${FINAL_CHECKPOINT_DIR}/pretrained_model"
OUTPUT_CHECKPOINT="${PI05_DIR}/outputs/${RUN_NAME}/checkpoints/${FINAL_CHECKPOINT_DIR}/pretrained_model"

if [[ -n "${POLICY_PATH:-}" ]]; then
  SELECTED_POLICY_PATH="${POLICY_PATH}"
elif [[ -d "${ROOT_CHECKPOINT}" ]]; then
  SELECTED_POLICY_PATH="${ROOT_CHECKPOINT}"
elif [[ -d "${OUTPUT_CHECKPOINT}" ]]; then
  SELECTED_POLICY_PATH="${OUTPUT_CHECKPOINT}"
else
  # Keep the expected path visible in CONFIG_ONLY/PRINT_COMMAND_ONLY output.
  SELECTED_POLICY_PATH="${ROOT_CHECKPOINT}"
fi

export CONDA_ROOT="${CONDA_ROOT:-/home/luzhuang/miniconda3}"
export CONDA_ENV="${CONDA_ENV:-lerobot_flex}"
export CONDA_PYTHON="${CONDA_PYTHON:-${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python}"
export PI05_15E_RUN_NAME="${RUN_NAME}"
export POLICY_PATH="$(readlink -m -- "${SELECTED_POLICY_PATH}")"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/assets/modelscope/google/paligemma-3b-pt-224}"
export SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
export SERVER_PORT="${SERVER_PORT:-8088}"
export POLICY_DEVICE="${POLICY_DEVICE:-cuda}"

source "${SCRIPT_DIR}/common.sh"

[[ "$#" -eq 0 ]] \
  || rtc_die "run_onboard_policy_server.sh accepts no CLI arguments; configure it with environment variables"

CONFIG_ONLY="$(rtc_normalize_bool CONFIG_ONLY "${CONFIG_ONLY:-false}")"
CHECK_POLICY_LOAD="$(rtc_normalize_bool CHECK_POLICY_LOAD "${CHECK_POLICY_LOAD:-false}")"
PRINT_COMMAND_ONLY="$(rtc_normalize_bool PRINT_COMMAND_ONLY "${PRINT_COMMAND_ONLY:-false}")"
[[ "${CONFIG_ONLY}" != "true" || "${CHECK_POLICY_LOAD}" != "true" ]] \
  || rtc_die "CONFIG_ONLY=true and CHECK_POLICY_LOAD=true are mutually exclusive"

case "${POLICY_PATH}" in
  */"${RUN_NAME}"/checkpoints/015705/pretrained_model)
    CHECKPOINT_MODE=final_015705
    EXPECTED_CHECKPOINT_STEP="${FINAL_STEP}"
    REQUIRE_COMPLETE_STEP="$(rtc_normalize_bool REQUIRE_COMPLETE_STEP "${REQUIRE_COMPLETE_STEP:-true}")"
    [[ "${REQUIRE_COMPLETE_STEP}" == "true" ]] \
      || rtc_die "the final onboard checkpoint requires REQUIRE_COMPLETE_STEP=true"
    ;;
  */"${RUN_NAME}"/checkpoints/010470/pretrained_model)
    [[ "${JZ_PI05_INTERMEDIATE_010470_CONFIRMED:-}" == "1" ]] \
      || rtc_die \
        "checkpoint 010470 requires JZ_PI05_INTERMEDIATE_010470_CONFIRMED=1"
    if [[ -n "${REQUIRE_COMPLETE_STEP:-}" ]]; then
      REQUIRE_COMPLETE_STEP="$(rtc_normalize_bool REQUIRE_COMPLETE_STEP "${REQUIRE_COMPLETE_STEP}")"
      [[ "${REQUIRE_COMPLETE_STEP}" == "false" ]] \
        || rtc_die "checkpoint 010470 requires REQUIRE_COMPLETE_STEP=false; omit it and let this wrapper set it"
    fi
    CHECKPOINT_MODE=intermediate_010470
    EXPECTED_CHECKPOINT_STEP="${INTERMEDIATE_STEP}"
    REQUIRE_COMPLETE_STEP=false
    ;;
  *)
    rtc_die \
      "POLICY_PATH must select ${RUN_NAME}/checkpoints/015705/pretrained_model or the explicitly confirmed 010470 checkpoint; got ${POLICY_PATH}"
    ;;
esac

export CONFIG_ONLY CHECK_POLICY_LOAD PRINT_COMMAND_ONLY REQUIRE_COMPLETE_STEP

validate_checkpoint_metadata() {
  local policy_path="$1"
  local expected_checkpoint_step="$2"
  local validation_error

  rtc_require_executable "${CONDA_PYTHON}"
  if ! validation_error="$("${CONDA_PYTHON}" - \
    "${policy_path}" "${FINAL_STEP}" "${expected_checkpoint_step}" 2>&1 <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path


policy_path = Path(sys.argv[1])
configured_step = int(sys.argv[2])
expected_checkpoint_step = int(sys.argv[3])
train_config_path = policy_path / "train_config.json"
training_step_path = policy_path.parent / "training_state" / "training_step.json"


def read_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise SystemExit(f"required final-checkpoint metadata is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read final-checkpoint metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"final-checkpoint metadata must be a JSON object: {path}")
    return value


def require_exact_step(value: object, *, label: str, expected: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise SystemExit(f"{label} must equal {expected}, got {value!r}")


require_exact_step(
    read_object(train_config_path).get("steps"),
    label="train_config.steps",
    expected=configured_step,
)
require_exact_step(
    read_object(training_step_path).get("step"),
    label="training_state.step",
    expected=expected_checkpoint_step,
)
PY
  )"; then
    rtc_die "${validation_error}"
  fi
}

if [[ -e "${POLICY_PATH}" && ! -d "${POLICY_PATH}" ]]; then
  rtc_die "POLICY_PATH is not a directory: ${POLICY_PATH}"
fi

if [[ -d "${POLICY_PATH}" ]]; then
  validate_checkpoint_metadata "${POLICY_PATH}" "${EXPECTED_CHECKPOINT_STEP}"
elif [[ "${CONFIG_ONLY}" == "true" || "${PRINT_COMMAND_ONLY}" == "true" ]]; then
  echo "[jz/pi05/onboard/server] selected checkpoint is not present; configuration output only"
else
  rtc_die \
    "selected checkpoint is missing: ${POLICY_PATH}"
fi

if [[ "${CONFIG_ONLY}" != "true" && "${PRINT_COMMAND_ONLY}" != "true" ]]; then
  rtc_require_policy
  rtc_require_tokenizer
fi

echo "[jz/pi05/onboard/server] run=${RUN_NAME} checkpoint_mode=${CHECKPOINT_MODE}"
echo "[jz/pi05/onboard/server] checkpoint_step=${EXPECTED_CHECKPOINT_STEP} configured_steps=${FINAL_STEP}"
echo "[jz/pi05/onboard/server] policy=${POLICY_PATH}"
echo "[jz/pi05/onboard/server] tokenizer=${TOKENIZER_PATH}"
echo "[jz/pi05/onboard/server] endpoint=http://${SERVER_HOST}:${SERVER_PORT}"

exec bash "${SCRIPT_DIR}/run_server.sh"
