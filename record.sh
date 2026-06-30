#!/usr/bin/env bash
set -euo pipefail

# X86-side LeRobot recording entrypoint for the UDP plug-flow setup.
#
# Required Orin-side read-only services before recording:
#   1. UDP feedback state bridge:
#      X86_IP=<x86_ip> bash udp_test/server_bash/orin_arm/start.sh
#   2. UDP target-action bridge:
#      PYTHONPATH=src conda run --no-capture-output -n lerobot python \
#        udp_test/test_scripts/arm_side/orin_ros_target_action_udp_bridge.py \
#        --target-ip <x86_ip> --target-port 39030 --bind-ip 192.168.1.81
#
# The default teleop records VR/upstream target commands into dataset action.
# Set TELEOP_MODE=hold only for legacy debugging where action intentionally
# copies observation.state.

CONDA_ENV="${CONDA_ENV:-lerobot_flex}"
ORIN_IP="${ORIN_IP:-192.168.1.81}"
STATE_PORT="${STATE_PORT:-39010}"
COMMAND_PORT="${COMMAND_PORT:-39020}"
TARGET_ACTION_PORT="${TARGET_ACTION_PORT:-39030}"
TELEOP_MODE="${TELEOP_MODE:-target_action}"

DATASET_NAME="${DATASET_NAME:-jz_robot_udp_vr_target_action_001}"
DATASET_ROOT="${DATASET_ROOT:-tests/outputs/${DATASET_NAME}}"
NUM_EPISODES="${NUM_EPISODES:-1}"
EPISODE_TIME_S="${EPISODE_TIME_S:-10}"
RESET_TIME_S="${RESET_TIME_S:-0}"
RECORD_FPS="${RECORD_FPS:-30}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"
PLAY_SOUNDS="${PLAY_SOUNDS:-true}"

ROBOT_SEND_ACTION_TRANSPORT="${ROBOT_SEND_ACTION_TRANSPORT:-local}"
ROBOT_SEND_ACTION_EXECUTION="${ROBOT_SEND_ACTION_EXECUTION:-dry_run}"

COMMON_ARGS=(
  --robot.type=jz_robot_udp
  --robot.bind_ip=0.0.0.0
  --robot.state_port="$STATE_PORT"
  --robot.allowed_sender_ip="$ORIN_IP"
  --robot.send_action_transport="$ROBOT_SEND_ACTION_TRANSPORT"
  --robot.send_action_execution="$ROBOT_SEND_ACTION_EXECUTION"
  --robot.command_target_ip="$ORIN_IP"
  --robot.command_target_port="$COMMAND_PORT"
  --robot.rtsp_cameras='{
    "camera_head": {
      "url": "rtsp://'"$ORIN_IP"':8554/robot_camera/camera_head",
      "fps": 30,
      "width": 1280,
      "height": 720,
      "timeout_ms": 5000,
      "warmup_frames": 1,
      "color_mode": "rgb",
      "transport": "tcp"
    },
    "camera_left": {
      "url": "rtsp://'"$ORIN_IP"':8554/robot_camera/camera_left",
      "fps": 30,
      "width": 640,
      "height": 480,
      "timeout_ms": 5000,
      "warmup_frames": 1,
      "color_mode": "rgb",
      "transport": "tcp"
    },
    "camera_right": {
      "url": "rtsp://'"$ORIN_IP"':8554/robot_camera/camera_right",
      "fps": 30,
      "width": 640,
      "height": 480,
      "timeout_ms": 5000,
      "warmup_frames": 1,
      "color_mode": "rgb",
      "transport": "tcp"
    }
  }'
)

case "$TELEOP_MODE" in
  target_action)
    TELEOP_ARGS=(
      --teleop.type=jz_robot_udp_target_action
      --teleop.bind_ip=0.0.0.0
      --teleop.target_action_port="$TARGET_ACTION_PORT"
      --teleop.allowed_sender_ip="$ORIN_IP"
    )
    ;;
  hold)
    TELEOP_ARGS=(
      --teleop.type=jz_robot_udp_hold
    )
    ;;
  *)
    echo "[record.sh] unsupported TELEOP_MODE=$TELEOP_MODE; use target_action or hold" >&2
    exit 2
    ;;
esac

echo "[record.sh] dataset=${DATASET_ROOT}"
echo "[record.sh] teleop_mode=${TELEOP_MODE} orin=${ORIN_IP} state_port=${STATE_PORT}"
echo "[record.sh] target_action_port=${TARGET_ACTION_PORT}"
echo "[record.sh] robot_send_action_transport=${ROBOT_SEND_ACTION_TRANSPORT}"

PYTHONPATH=src conda run --no-capture-output -n "$CONDA_ENV" \
  python -m lerobot.scripts.lerobot_record \
  "${COMMON_ARGS[@]}" \
  "${TELEOP_ARGS[@]}" \
  --dataset.repo_id="local/${DATASET_NAME}" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.episode_time_s="$EPISODE_TIME_S" \
  --dataset.reset_time_s="$RESET_TIME_S" \
  --dataset.fps="$RECORD_FPS" \
  --dataset.single_task="vr target-action and feedback-state record" \
  --dataset.push_to_hub=false \
  --dataset.video=true \
  --dataset.vcodec=h264 \
  --display_data="$DISPLAY_DATA" \
  --display_compressed_images=false \
  --play_sounds="$PLAY_SOUNDS"

echo "[record.sh] recording finished. Analyze with:"
echo "  DATASET_ROOT=$DATASET_ROOT bash x86_test_datasets.sh analyze"
