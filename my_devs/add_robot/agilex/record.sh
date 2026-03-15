#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Usage:
#   bash record.sh
#   bash record.sh <dataset_name>
#   bash record.sh <dataset_name> <episode_time_s> <num_episodes> <fps> <reset_time_s> <resume> <single_task_text>
#
# Examples:
#   bash record.sh
#   bash record.sh demo_pick
#   bash record.sh demo_pick 8 2 30 5 false
#   bash record.sh demo_pick 8 1 30 0 true
#   bash record.sh demo_pick 8 1 30 0 false 'Pick up the black cups and place them in the orange box.'
#   ACTION_SOURCE=master bash record.sh demo_pick 8 1 30 0 false 'Use master as action source'

# 运行环境名字。
CONDA_ENV="${CONDA_ENV:-lerobot_flex}"
# 数据集根目录。最终数据会放到 outputs/dummy/<dataset_name>。
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${SCRIPT_DIR}/outputs}"
# 自动生成的配置文件路径。默认会按数据集名字区分，避免互相覆盖。
CONFIG_PATH="${CONFIG_PATH:-}"

# 数据集名字。命令第 1 个参数。
DATASET_NAME="${1:-agilex_record_demo_video}"
# LeRobot 内部使用的数据集 repo_id，一般不用改。
DATASET_REPO_ID="${DATASET_REPO_ID:-dummy/${DATASET_NAME}}"
if [[ -z "${CONFIG_PATH}" ]]; then
  CONFIG_PATH="${SCRIPT_DIR}/outputs/record_config_${DATASET_NAME}.json"
fi
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
# 任务语义文本。从第 7 个参数开始直到命令结尾，都会拼成一句话写进数据集。
if [[ $# -ge 7 ]]; then
  SINGLE_TASK="${*:7}"
else
  SINGLE_TASK="${SINGLE_TASK:-agilex static record test}"
fi
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

# 录制 action 的来源。默认 follower，只记录从臂 /puppet 作为 action。
# 训练数据建议保持 follower，不要把 master 默认写进 action。
# 可选：
# - follower: action 来自 /puppet/joint_left 和 /puppet/joint_right
# - master:   action 来自 /master/joint_left 和 /master/joint_right
ACTION_SOURCE_RAW="${ACTION_SOURCE:-follower}"

case "${ACTION_SOURCE_RAW,,}" in
  follower|puppet)
    ACTION_SOURCE="follower"
    DEFAULT_ACTION_LEFT_TOPIC="/puppet/joint_left"
    DEFAULT_ACTION_RIGHT_TOPIC="/puppet/joint_right"
    ;;
  master|leader)
    ACTION_SOURCE="master"
    DEFAULT_ACTION_LEFT_TOPIC="/master/joint_left"
    DEFAULT_ACTION_RIGHT_TOPIC="/master/joint_right"
    ;;
  *)
    echo "Invalid ACTION_SOURCE: ${ACTION_SOURCE_RAW}. Use follower or master."
    exit 1
    ;;
esac

# 从臂关节状态 topic。
STATE_LEFT_TOPIC="${STATE_LEFT_TOPIC:-/puppet/joint_left}"
# 右臂关节状态 topic。
STATE_RIGHT_TOPIC="${STATE_RIGHT_TOPIC:-/puppet/joint_right}"
# 左臂录制 action topic。默认跟随 ACTION_SOURCE 自动选择，也可以手动覆盖。
ACTION_LEFT_TOPIC="${ACTION_LEFT_TOPIC:-${DEFAULT_ACTION_LEFT_TOPIC}}"
# 右臂录制 action topic。默认跟随 ACTION_SOURCE 自动选择，也可以手动覆盖。
ACTION_RIGHT_TOPIC="${ACTION_RIGHT_TOPIC:-${DEFAULT_ACTION_RIGHT_TOPIC}}"
# 前视相机 topic。
FRONT_CAMERA_TOPIC="${FRONT_CAMERA_TOPIC:-/camera_f/color/image_raw}"
# 左相机 topic。
LEFT_CAMERA_TOPIC="${LEFT_CAMERA_TOPIC:-/camera_l/color/image_raw}"
# 右相机 topic。
RIGHT_CAMERA_TOPIC="${RIGHT_CAMERA_TOPIC:-/camera_r/color/image_raw}"
# 前视相机在数据集里的名字。默认对齐参考数据集命名。
FRONT_CAMERA_KEY="${FRONT_CAMERA_KEY:-camera_front}"
# 左相机在数据集里的名字。默认对齐参考数据集命名。
LEFT_CAMERA_KEY="${LEFT_CAMERA_KEY:-camera_left}"
# 右相机在数据集里的名字。默认对齐参考数据集命名。
RIGHT_CAMERA_KEY="${RIGHT_CAMERA_KEY:-camera_right}"
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

if [[ ! -d "${DATASET_DIR}" && "${RESUME}" == "true" ]]; then
  echo "Dataset does not exist yet: ${DATASET_DIR}"
  echo "Use false as the 6th argument for the first recording."
  exit 1
fi

export DISPLAY="${DISPLAY:-:0}"
export HF_LEROBOT_HOME

cd "${REPO_ROOT}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
  py_cmd=(
    python
    -
  )
  cmd=(
    lerobot-record
    --config_path "${CONFIG_PATH}"
  )
else
  py_cmd=(
    conda run --no-capture-output -n "${CONDA_ENV}" python
    -
  )
  cmd=(
    conda run --no-capture-output -n "${CONDA_ENV}" lerobot-record
    --config_path "${CONFIG_PATH}"
  )
fi

export CONFIG_PATH
export DATASET_REPO_ID
export SINGLE_TASK
export NUM_EPISODES
export EPISODE_TIME_S
export RESET_TIME_S
export FPS
export VIDEO
export VCODEC
export PUSH_TO_HUB
export DISPLAY_DATA
export PLAY_SOUNDS
export RESUME
export ACTION_SOURCE
export STATE_LEFT_TOPIC
export STATE_RIGHT_TOPIC
export ACTION_LEFT_TOPIC
export ACTION_RIGHT_TOPIC
export FRONT_CAMERA_TOPIC
export LEFT_CAMERA_TOPIC
export RIGHT_CAMERA_TOPIC
export FRONT_CAMERA_KEY
export LEFT_CAMERA_KEY
export RIGHT_CAMERA_KEY
export IMAGE_HEIGHT
export IMAGE_WIDTH
export OBSERVATION_TIMEOUT_S
export QUEUE_SIZE

"${py_cmd[@]}" <<'PY'
import json
import os


def parse_bool(name: str) -> bool:
    value = os.environ[name].strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {os.environ[name]}")


config = {
    "robot": {
        "type": "agilex",
        "control_mode": "passive_follow",
        "state_left_topic": os.environ["STATE_LEFT_TOPIC"],
        "state_right_topic": os.environ["STATE_RIGHT_TOPIC"],
        "command_left_topic": os.environ["ACTION_LEFT_TOPIC"],
        "command_right_topic": os.environ["ACTION_RIGHT_TOPIC"],
        "front_camera_topic": os.environ["FRONT_CAMERA_TOPIC"],
        "left_camera_topic": os.environ["LEFT_CAMERA_TOPIC"],
        "right_camera_topic": os.environ["RIGHT_CAMERA_TOPIC"],
        "front_camera_key": os.environ["FRONT_CAMERA_KEY"],
        "left_camera_key": os.environ["LEFT_CAMERA_KEY"],
        "right_camera_key": os.environ["RIGHT_CAMERA_KEY"],
        "image_height": int(os.environ["IMAGE_HEIGHT"]),
        "image_width": int(os.environ["IMAGE_WIDTH"]),
        "observation_timeout_s": float(os.environ["OBSERVATION_TIMEOUT_S"]),
        "queue_size": int(os.environ["QUEUE_SIZE"]),
    },
    "teleop": {
        "type": "agilex_teleoperator",
        "action_left_topic": os.environ["ACTION_LEFT_TOPIC"],
        "action_right_topic": os.environ["ACTION_RIGHT_TOPIC"],
        "observation_timeout_s": float(os.environ["OBSERVATION_TIMEOUT_S"]),
        "queue_size": int(os.environ["QUEUE_SIZE"]),
    },
    "dataset": {
        "repo_id": os.environ["DATASET_REPO_ID"],
        "single_task": os.environ["SINGLE_TASK"],
        "num_episodes": int(os.environ["NUM_EPISODES"]),
        "episode_time_s": float(os.environ["EPISODE_TIME_S"]),
        "reset_time_s": float(os.environ["RESET_TIME_S"]),
        "fps": int(os.environ["FPS"]),
        "video": parse_bool("VIDEO"),
        "vcodec": os.environ["VCODEC"],
        "push_to_hub": parse_bool("PUSH_TO_HUB"),
    },
    "display_data": parse_bool("DISPLAY_DATA"),
    "play_sounds": parse_bool("PLAY_SOUNDS"),
    "resume": parse_bool("RESUME"),
}

with open(os.environ["CONFIG_PATH"], "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY

echo "HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
echo "Dataset output=${DATASET_DIR}"
echo "Resume=${RESUME}"
echo "Action source=${ACTION_SOURCE}"
echo "Action left topic=${ACTION_LEFT_TOPIC}"
echo "Action right topic=${ACTION_RIGHT_TOPIC}"
echo "Task=${SINGLE_TASK}"
echo "Config file=${CONFIG_PATH}"
echo "Running command:"
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
