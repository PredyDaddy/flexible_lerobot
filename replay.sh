#!/usr/bin/env bash
set -euo pipefail

# X86-side LeRobot replay entrypoint for the UDP plug-flow setup.
#
# Safety default:
#   EXECUTION=dry_run only sends dry-run UDP command packets to Orin.
#   For physical replay you must explicitly set EXECUTION=armed and start the
#   Orin phase3 executor in armed mode.
#
# Orin physical replay service example:
#   CONFIG=udp_test/test_scripts/arm_side/orin_phase3_executor_config_armed_hold.yaml \
#   EXECUTION=armed \
#   JZ_UDP_EXECUTOR_ARMED=1 \
#   AUTO_TAIL=1 \
#   bash udp_test/server_bash/orin_arm/start_phase3_executor.sh

CONDA_ENV="${CONDA_ENV:-lerobot_flex}"
ORIN_IP="${ORIN_IP:-192.168.1.81}"
COMMAND_PORT="${COMMAND_PORT:-39020}"
EXECUTION="${EXECUTION:-dry_run}"

DATASET_NAME="${DATASET_NAME:-jz_robot_udp_vr_target_action_001}"
DATASET_ROOT="${DATASET_ROOT:-tests/outputs/${DATASET_NAME}}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/${DATASET_NAME}}"
EPISODE="${EPISODE:-0}"
REPLAY_FPS="${REPLAY_FPS:-15}"
PLAY_SOUNDS="${PLAY_SOUNDS:-true}"

case "$EXECUTION" in
  dry_run|armed)
    ;;
  *)
    echo "[replay.sh] unsupported EXECUTION=$EXECUTION; use dry_run or armed" >&2
    exit 2
    ;;
esac

if [[ "$EXECUTION" == "armed" && "${I_UNDERSTAND_REPLAY_MOVES_ROBOT:-}" != "1" ]]; then
  echo "[replay.sh] refusing armed replay: set I_UNDERSTAND_REPLAY_MOVES_ROBOT=1" >&2
  exit 2
fi

echo "[replay.sh] dataset=${DATASET_ROOT} episode=${EPISODE}"
echo "[replay.sh] execution=${EXECUTION} replay_fps=${REPLAY_FPS}"
echo "[replay.sh] target=${ORIN_IP}:${COMMAND_PORT}"

PYTHONPATH=src conda run --no-capture-output -n "$CONDA_ENV" \
  python -m lerobot.scripts.lerobot_replay \
  --robot.type=jz_robot_udp \
  --robot.send_action_transport=udp \
  --robot.send_action_execution="$EXECUTION" \
  --robot.command_target_ip="$ORIN_IP" \
  --robot.command_target_port="$COMMAND_PORT" \
  --robot.allowed_sender_ip="$ORIN_IP" \
  --robot.rtsp_cameras='{}' \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.episode="$EPISODE" \
  --dataset.fps="$REPLAY_FPS" \
  --play_sounds="$PLAY_SOUNDS"
