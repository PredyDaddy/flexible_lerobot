#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Usage:
#   bash replay.sh
#   bash replay.sh <dataset_name>
#   bash replay.sh <dataset_name> <episode_index>
#
# Examples:
#   bash replay.sh
#   bash replay.sh demo_pick
#   bash replay.sh demo_pick 0

# 运行环境名字。
CONDA_ENV="${CONDA_ENV:-lerobot_flex}"
# 数据集根目录。默认就是当前目录下的 outputs。
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${SCRIPT_DIR}/outputs}"
# 自动生成的配置文件路径，一般不用改。
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/outputs/replay_config.json}"

# 数据集名字。命令第 1 个参数。
DATASET_NAME="${1:-agilex_record_demo_video}"
# LeRobot 内部使用的数据集 repo_id，一般不用改。
DATASET_REPO_ID="${DATASET_REPO_ID:-dummy/${DATASET_NAME}}"
# 回放第几个 episode。命令第 2 个参数，从 0 开始。
EPISODE_INDEX="${2:-0}"
# 是否播放语音提示。
PLAY_SOUNDS="${PLAY_SOUNDS:-false}"

# 等待 ROS 数据的超时时间。
OBSERVATION_TIMEOUT_S="${OBSERVATION_TIMEOUT_S:-10.0}"
# ROS 订阅/发布队列大小。
QUEUE_SIZE="${QUEUE_SIZE:-1}"

# 左臂状态 topic。
STATE_LEFT_TOPIC="${STATE_LEFT_TOPIC:-/puppet/joint_left}"
# 右臂状态 topic。
STATE_RIGHT_TOPIC="${STATE_RIGHT_TOPIC:-/puppet/joint_right}"
# 左臂回放命令 topic。
COMMAND_LEFT_TOPIC="${COMMAND_LEFT_TOPIC:-/master/joint_left}"
# 右臂回放命令 topic。
COMMAND_RIGHT_TOPIC="${COMMAND_RIGHT_TOPIC:-/master/joint_right}"
# 前视相机 topic。
FRONT_CAMERA_TOPIC="${FRONT_CAMERA_TOPIC:-/camera_f/color/image_raw}"
# 左相机 topic。
LEFT_CAMERA_TOPIC="${LEFT_CAMERA_TOPIC:-/camera_l/color/image_raw}"
# 右相机 topic。
RIGHT_CAMERA_TOPIC="${RIGHT_CAMERA_TOPIC:-/camera_r/color/image_raw}"
# 前视相机在数据集里的名字。
FRONT_CAMERA_KEY="${FRONT_CAMERA_KEY:-cam_high}"
# 左相机在数据集里的名字。
LEFT_CAMERA_KEY="${LEFT_CAMERA_KEY:-cam_left_wrist}"
# 右相机在数据集里的名字。
RIGHT_CAMERA_KEY="${RIGHT_CAMERA_KEY:-cam_right_wrist}"
# 图像高。
IMAGE_HEIGHT="${IMAGE_HEIGHT:-480}"
# 图像宽。
IMAGE_WIDTH="${IMAGE_WIDTH:-640}"

mkdir -p "${SCRIPT_DIR}/outputs"
mkdir -p "${HF_LEROBOT_HOME}"
mkdir -p "$(dirname "${CONFIG_PATH}")"

cat > "${CONFIG_PATH}" <<EOF
{
  "robot": {
    "type": "agilex",
    "control_mode": "command_master",
    "state_left_topic": "${STATE_LEFT_TOPIC}",
    "state_right_topic": "${STATE_RIGHT_TOPIC}",
    "command_left_topic": "${COMMAND_LEFT_TOPIC}",
    "command_right_topic": "${COMMAND_RIGHT_TOPIC}",
    "front_camera_topic": "${FRONT_CAMERA_TOPIC}",
    "left_camera_topic": "${LEFT_CAMERA_TOPIC}",
    "right_camera_topic": "${RIGHT_CAMERA_TOPIC}",
    "front_camera_key": "${FRONT_CAMERA_KEY}",
    "left_camera_key": "${LEFT_CAMERA_KEY}",
    "right_camera_key": "${RIGHT_CAMERA_KEY}",
    "image_height": ${IMAGE_HEIGHT},
    "image_width": ${IMAGE_WIDTH},
    "observation_timeout_s": ${OBSERVATION_TIMEOUT_S},
    "queue_size": ${QUEUE_SIZE}
  },
  "dataset": {
    "repo_id": "${DATASET_REPO_ID}",
    "episode": ${EPISODE_INDEX}
  },
  "play_sounds": ${PLAY_SOUNDS}
}
EOF

export DISPLAY="${DISPLAY:-:0}"
export HF_LEROBOT_HOME

cd "${REPO_ROOT}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
  cmd=(
    lerobot-replay
    --config_path "${CONFIG_PATH}"
  )
else
  cmd=(
    conda run --no-capture-output -n "${CONDA_ENV}" lerobot-replay
    --config_path "${CONFIG_PATH}"
  )
fi

echo "HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
echo "Dataset input=${HF_LEROBOT_HOME}/${DATASET_REPO_ID}"
echo "Config file=${CONFIG_PATH}"
echo "Running command:"
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
