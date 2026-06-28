#!/usr/bin/env bash
set -euo pipefail

# Temporary x86-side Phase 2 test helper.
# This file is intentionally short-lived and can be deleted after Phase 2 dry-run validation.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_CMD="${PYTHON_CMD:-python}"
read -r -a PYTHON_ARGS <<< "$PYTHON_CMD"
ORIN_IP="${ORIN_IP:-192.168.1.81}"
COMMAND_PORT="${COMMAND_PORT:-39020}"
COUNT="${COUNT:-20}"
HZ="${HZ:-5}"

run_constant_teleop_import() {
  echo "[x86_test] constant teleop import/factory check"
  PYTHONPATH=src "${PYTHON_ARGS[@]}" - <<'PY'
from lerobot.teleoperators.jz_robot_udp_constant import JZRobotUDPConstantTeleopConfig
from lerobot.teleoperators.utils import make_teleoperator_from_config

cfg = JZRobotUDPConstantTeleopConfig()
teleop = make_teleoperator_from_config(cfg)
print("teleop_type", cfg.type)
print("action_keys", len(teleop.action_features))
print("sample_keys", sorted(teleop.action_features)[:3])
PY
}

run_command_dry_run() {
  echo "[x86_test] x86 -> Orin command dry-run"
  PYTHONPATH=src "${PYTHON_ARGS[@]}" \
    udp_test/test_scripts/x86_side/x86_jz_robot_udp_send_action_check.py \
    --robot-config src/lerobot/configs/robot/jz_robot_udp_three_rtsp.yaml \
    --transport udp \
    --execution dry_run \
    --command-only \
    --command-target-ip "$ORIN_IP" \
    --command-target-port "$COMMAND_PORT" \
    --count "$COUNT" \
    --hz "$HZ" \
    --print-every 1
}

run_record_dry_run() {
  echo "[x86_test] lerobot-record Phase 2 dry-run"
  echo "[x86_test] Orin command receiver and Orin state bridge must already be running."
  rm -rf tests/outputs/jz_robot_udp_phase2_dry_run
  PYTHONPATH=src "${PYTHON_ARGS[@]}" -m lerobot.scripts.lerobot_record \
    --robot.type=jz_robot_udp \
    --robot.id=jz_robot_udp_phase2 \
    --robot.send_action_transport=udp \
    --robot.send_action_execution=dry_run \
    --robot.command_target_ip="$ORIN_IP" \
    --robot.command_target_port="$COMMAND_PORT" \
    --dataset.repo_id=local/jz_robot_udp_phase2_dry_run \
    --dataset.root=tests/outputs/jz_robot_udp_phase2_dry_run \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=10 \
    --dataset.reset_time_s=0 \
    --dataset.fps=5 \
    --dataset.single_task="phase2 dry-run record chain test" \
    --dataset.push_to_hub=false \
    --dataset.video=true \
    --dataset.vcodec=h264 \
    --teleop.type=jz_robot_udp_constant \
    --display_data=false \
    --play_sounds=false
}

case "${1:-constant_teleop_import}" in
  constant_teleop_import)
    run_constant_teleop_import
    ;;
  command_dry_run)
    run_command_dry_run
    ;;
  record_dry_run)
    run_record_dry_run
    ;;
  *)
    echo "Usage: $0 [constant_teleop_import|command_dry_run|record_dry_run]" >&2
    exit 2
    ;;
esac
