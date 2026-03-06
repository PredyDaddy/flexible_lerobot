#!/usr/bin/env bash
set -euo pipefail

# Run policy-driven ACT evaluation recording for SO101.
# This is the shell/CLI counterpart of `run_act_eval_record.py`.
#
# Example:
#   DRY_RUN=true bash my_devs/train/act/so101/run_act_eval_record.sh
#   ROBOT_PORT=/dev/ttyACM0 TOP_CAM_INDEX=4 WRIST_CAM_INDEX=6 \
#   POLICY_N_ACTION_STEPS=16 \
#   bash my_devs/train/act/so101/run_act_eval_record.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR_DEFAULT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_DIR_DEFAULT}" ]]; then
  REPO_DIR_DEFAULT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
fi
REPO_DIR="${REPO_DIR:-${REPO_DIR_DEFAULT}}"
cd "${REPO_DIR}"

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-lerobot_flex}"

ROBOT_ID="${ROBOT_ID:-my_so101}"
ROBOT_TYPE="${ROBOT_TYPE:-so101_follower}"
CALIB_DIR="${CALIB_DIR:-/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"

TOP_CAM_INDEX="${TOP_CAM_INDEX:-4}"
WRIST_CAM_INDEX="${WRIST_CAM_INDEX:-6}"
IMG_WIDTH="${IMG_WIDTH:-640}"
IMG_HEIGHT="${IMG_HEIGHT:-480}"
FPS="${FPS:-30}"

DEFAULT_POLICY_PATH="/data/cqy_workspace/flexible_lerobot/outputs/train/\
20260305_190147_act_grasp_block_in_bin1_e15/checkpoints/last/pretrained_model"
POLICY_PATH="${POLICY_PATH:-${DEFAULT_POLICY_PATH}}"
POLICY_DEVICE_OVERRIDE="${POLICY_DEVICE_OVERRIDE:-}"
POLICY_N_ACTION_STEPS="${POLICY_N_ACTION_STEPS:-}"
POLICY_TEMPORAL_ENSEMBLE_COEFF="${POLICY_TEMPORAL_ENSEMBLE_COEFF:-}"

DATASET_REPO_ID="${DATASET_REPO_ID:-admin123/eval_act_grasp_block_in_bin1_01}"
DATASET_TASK="${DATASET_TASK:-Put the block in the bin}"
DATASET_ROOT="${DATASET_ROOT:-}"
NUM_EPISODES="${NUM_EPISODES:-5}"
EPISODE_TIME_S="${EPISODE_TIME_S:-40}"
RESET_TIME_S="${RESET_TIME_S:-10}"
DISPLAY_DATA="${DISPLAY_DATA:-false}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
PLAY_SOUNDS="${PLAY_SOUNDS:-true}"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[ERROR] conda init script not found: ${CONDA_SH}" >&2
  exit 1
fi

if [[ ! -d "${POLICY_PATH}" ]]; then
  echo "[ERROR] policy path does not exist: ${POLICY_PATH}" >&2
  exit 1
fi

CAMERAS_JSON=$(
  cat <<JSON
{
  "top": {
    "type": "opencv",
    "index_or_path": ${TOP_CAM_INDEX},
    "width": ${IMG_WIDTH},
    "height": ${IMG_HEIGHT},
    "fps": ${FPS}
  },
  "wrist": {
    "type": "opencv",
    "index_or_path": ${WRIST_CAM_INDEX},
    "width": ${IMG_WIDTH},
    "height": ${IMG_HEIGHT},
    "fps": ${FPS}
  }
}
JSON
)

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

if ! command -v lerobot-record >/dev/null 2>&1; then
  echo "[ERROR] lerobot-record not found in env ${CONDA_ENV}" >&2
  exit 1
fi

CMD=(
  lerobot-record
  "--robot.id=${ROBOT_ID}"
  "--robot.type=${ROBOT_TYPE}"
  "--robot.calibration_dir=${CALIB_DIR}"
  "--robot.port=${ROBOT_PORT}"
  "--robot.cameras=${CAMERAS_JSON}"
  "--policy.path=${POLICY_PATH}"
  "--dataset.repo_id=${DATASET_REPO_ID}"
  "--dataset.single_task=${DATASET_TASK}"
  "--dataset.num_episodes=${NUM_EPISODES}"
  "--dataset.episode_time_s=${EPISODE_TIME_S}"
  "--dataset.reset_time_s=${RESET_TIME_S}"
  "--dataset.push_to_hub=${PUSH_TO_HUB}"
  "--display_data=${DISPLAY_DATA}"
  "--play_sounds=${PLAY_SOUNDS}"
)

if [[ -n "${DATASET_ROOT}" ]]; then
  CMD+=("--dataset.root=${DATASET_ROOT}")
fi

if [[ -n "${POLICY_DEVICE_OVERRIDE}" ]]; then
  CMD+=("--policy.device=${POLICY_DEVICE_OVERRIDE}")
fi

if [[ -n "${POLICY_N_ACTION_STEPS}" ]]; then
  CMD+=("--policy.n_action_steps=${POLICY_N_ACTION_STEPS}")
fi

if [[ -n "${POLICY_TEMPORAL_ENSEMBLE_COEFF}" ]]; then
  CMD+=("--policy.temporal_ensemble_coeff=${POLICY_TEMPORAL_ENSEMBLE_COEFF}")
fi

echo "[INFO] Repo: ${REPO_DIR}"
echo "[INFO] Conda env: ${CONDA_ENV}"
echo "[INFO] Policy path: ${POLICY_PATH}"
echo "[INFO] Policy device override: ${POLICY_DEVICE_OVERRIDE:-<checkpoint default>}"
echo "[INFO] ACT override n_action_steps: ${POLICY_N_ACTION_STEPS:-<checkpoint default>}"
echo "[INFO] ACT override temporal_ensemble_coeff: ${POLICY_TEMPORAL_ENSEMBLE_COEFF:-<checkpoint default>}"
echo "[INFO] Robot port: ${ROBOT_PORT}"
echo "[INFO] Cameras: ${CAMERAS_JSON}"
echo "[INFO] Dataset repo_id: ${DATASET_REPO_ID}"
echo "[INFO] Episodes: ${NUM_EPISODES}, episode_time_s: ${EPISODE_TIME_S}, reset_time_s: ${RESET_TIME_S}"
echo "[INFO] Note: when policy.path is provided,"
echo "[INFO]       dataset.repo_id should start with 'eval_' per LeRobot checks."

echo "[INFO] Command:"
printf '  %q' "${CMD[@]}"
echo

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  echo "[INFO] DRY_RUN=true, exit without execution."
  exit 0
fi

exec "${CMD[@]}"
