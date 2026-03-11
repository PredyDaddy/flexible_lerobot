#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Usage:
#   bash record.sh
#   bash record.sh <dataset_name>
#   bash record.sh <dataset_name> <episode_time_s> <num_episodes> <fps> <reset_time_s> <resume>
#
# Examples:
#   bash record.sh
#   bash record.sh demo_pick
#   bash record.sh demo_pick 8 2 30 5 false
#   bash record.sh demo_pick 8 1 30 0 true

# 运行环境名字。
CONDA_ENV="${CONDA_ENV:-lerobot_flex}"
# 数据集根目录。最终数据会放到 outputs/dummy/<dataset_name>。
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${SCRIPT_DIR}/outputs}"
# 自动生成的配置文件路径，一般不用改。
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/outputs/record_config.json}"

# 数据集名字。命令第 1 个参数。
DATASET_NAME="${1:-agilex_record_demo_video}"
# LeRobot 内部使用的数据集 repo_id，一般不用改。
DATASET_REPO_ID="${DATASET_REPO_ID:-dummy/${DATASET_NAME}}"
# 任务描述，写进数据集元信息。
SINGLE_TASK="${SINGLE_TASK:-agilex static record test}"
# 每个 episode 录几秒。命令第 2 个参数。
EPISODE_TIME_S="${2:-8.0}"
# 一共录几个 episode。命令第 3 个参数。
NUM_EPISODES="${3:-1}"
# 录制帧率。命令第 4 个参数。
FPS="${4:-10}"
# 两个 episode 之间留多少秒做 reset。命令第 5 个参数。
RESET_TIME_S="${5:-0.0}"
# 是否在已有数据集上继续录。命令第 6 个参数，支持 true/false。
RESUME_RAW="${6:-false}"
# 是否保存相机视频。要保留相机画面就用 true。
VIDEO="${VIDEO:-true}"
# 视频编码格式。
VCODEC="${VCODEC:-h264}"
# 是否上传到 Hugging Face Hub。
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
# 是否边录边显示数据窗口。
DISPLAY_DATA="${DISPLAY_DATA:-true}"
# 是否播放语音提示。
PLAY_SOUNDS="${PLAY_SOUNDS:-true}"

# 等待 ROS 数据的超时时间。
OBSERVATION_TIMEOUT_S="${OBSERVATION_TIMEOUT_S:-10.0}"
# ROS 订阅/发布队列大小。
QUEUE_SIZE="${QUEUE_SIZE:-1}"

# 从臂关节状态 topic。
STATE_LEFT_TOPIC="${STATE_LEFT_TOPIC:-/puppet/joint_left}"
# 右臂关节状态 topic。
STATE_RIGHT_TOPIC="${STATE_RIGHT_TOPIC:-/puppet/joint_right}"
# 左臂主手动作 topic。
ACTION_LEFT_TOPIC="${ACTION_LEFT_TOPIC:-/master/joint_left}"
# 右臂主手动作 topic。
ACTION_RIGHT_TOPIC="${ACTION_RIGHT_TOPIC:-/master/joint_right}"
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

case "${RESUME_RAW,,}" in
  true|1|yes|y)
    RESUME=true
    ;;
  false|0|no|n)
    RESUME=false
    ;;
  *)
    echo "Invalid resume value: ${RESUME_RAW}. Use true or false."
    exit 1
    ;;
esac

DATASET_DIR="${HF_LEROBOT_HOME}/${DATASET_REPO_ID}"

if [[ -d "${DATASET_DIR}" && "${RESUME}" == "false" ]]; then
  echo "Dataset already exists: ${DATASET_DIR}"
  echo "Use a new dataset name, or pass true as the 6th argument to resume recording."
  exit 1
fi

cat > "${CONFIG_PATH}" <<EOF
{
  "robot": {
    "type": "agilex",
    "control_mode": "passive_follow",
    "state_left_topic": "${STATE_LEFT_TOPIC}",
    "state_right_topic": "${STATE_RIGHT_TOPIC}",
    "command_left_topic": "${ACTION_LEFT_TOPIC}",
    "command_right_topic": "${ACTION_RIGHT_TOPIC}",
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
  "teleop": {
    "type": "agilex_teleoperator",
    "action_left_topic": "${ACTION_LEFT_TOPIC}",
    "action_right_topic": "${ACTION_RIGHT_TOPIC}",
    "observation_timeout_s": ${OBSERVATION_TIMEOUT_S},
    "queue_size": ${QUEUE_SIZE}
  },
  "dataset": {
    "repo_id": "${DATASET_REPO_ID}",
    "single_task": "${SINGLE_TASK}",
    "num_episodes": ${NUM_EPISODES},
    "episode_time_s": ${EPISODE_TIME_S},
    "reset_time_s": ${RESET_TIME_S},
    "fps": ${FPS},
    "video": ${VIDEO},
    "vcodec": "${VCODEC}",
    "push_to_hub": ${PUSH_TO_HUB}
  },
  "display_data": ${DISPLAY_DATA},
  "play_sounds": ${PLAY_SOUNDS},
  "resume": ${RESUME}
}
EOF

export DISPLAY="${DISPLAY:-:0}"
export HF_LEROBOT_HOME

cd "${REPO_ROOT}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
  cmd=(
    lerobot-record
    --config_path "${CONFIG_PATH}"
  )
else
  cmd=(
    conda run --no-capture-output -n "${CONDA_ENV}" lerobot-record
    --config_path "${CONFIG_PATH}"
  )
fi

echo "HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
echo "Dataset output=${DATASET_DIR}"
echo "Resume=${RESUME}"
echo "Config file=${CONFIG_PATH}"
echo "Running command:"
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
