#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_CMD="${PYTHON_CMD:-conda run --no-capture-output -n lerobot_flex python}"
PYTHONPATH_VALUE="${PYTHONPATH_VALUE:-src}"
ROBOT_CONFIG="${ROBOT_CONFIG:-src/lerobot/configs/robot/jz_robot_udp_three_rtsp.yaml}"
ORIN_IP="${ORIN_IP:-192.168.1.81}"
COMMAND_PORT="${COMMAND_PORT:-39020}"

run_python() {
  PYTHONPATH="$PYTHONPATH_VALUE" $PYTHON_CMD "$@"
}

constant_teleop_import() {
  echo "[x86_test] constant teleop import/factory check"
  run_python - <<'PY'
from lerobot.teleoperators.jz_robot_udp_constant import JZRobotUDPConstantTeleopConfig
from lerobot.teleoperators.utils import make_teleoperator_from_config

cfg = JZRobotUDPConstantTeleopConfig()
teleop = make_teleoperator_from_config(cfg)
print("teleop_type", cfg.type)
print("action_keys", len(teleop.action_features))
print("sample_keys", sorted(teleop.action_features)[:3])
PY
}

hold_teleop_import() {
  echo "[x86_test] hold teleop import/factory check"
  run_python - <<'PY'
from lerobot.teleoperators.jz_robot_udp_hold import JZRobotUDPHoldTeleopConfig
from lerobot.teleoperators.utils import make_teleoperator_from_config

cfg = JZRobotUDPHoldTeleopConfig()
teleop = make_teleoperator_from_config(cfg)
print("teleop_type", cfg.type)
print("action_keys", len(teleop.action_features))
print("sample_keys", sorted(teleop.action_features)[:3])
PY
}

observation_check() {
  echo "[x86_test] READONLY observation check"
  echo "[x86_test] This only calls get_observation(); it does not call send_action."
  run_python udp_test/test_scripts/x86_side/x86_jz_robot_udp_observation_check.py \
    --robot-config "$ROBOT_CONFIG" \
    --count "${OBS_COUNT:-20}" \
    --hz "${OBS_HZ:-5}" \
    --print-every "${PRINT_EVERY:-1}"
}

observation_values() {
  echo "[x86_test] READONLY observation numeric values"
  echo "[x86_test] This only calls get_observation(); it does not call send_action."
  run_python - <<PY
import draccus
from pathlib import Path

from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots.jz_robot_udp import JZRobotUDPConfig  # noqa: F401

cfg = draccus.parse(config_class=RobotConfig, config_path=Path("$ROBOT_CONFIG"), args=[])
robot = make_robot_from_config(cfg)
try:
    robot.connect()
    obs = robot.get_observation()
    for key in sorted(robot.action_features):
        value = obs[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{key} is not numeric: {type(value).__name__}")
        print(f"{key}: {float(value):.9f}")
finally:
    if robot.is_connected:
        robot.disconnect()
PY
}

send_action_dry_run() {
  echo "[x86_test] UDP send_action dry-run packet check"
  echo "[x86_test] Orin Phase 3 executor or command receiver must already be running in dry-run."
  run_python udp_test/test_scripts/x86_side/x86_jz_robot_udp_send_action_check.py \
    --robot-config "$ROBOT_CONFIG" \
    --transport udp \
    --execution dry_run \
    --command-target-ip "$ORIN_IP" \
    --command-target-port "$COMMAND_PORT" \
    --count "${COMMAND_COUNT:-50}" \
    --hz "${COMMAND_HZ:-5}" \
    --command-only \
    --skip-cameras \
    --print-every "${PRINT_EVERY:-10}"
}

record_dry_run() {
  echo "[x86_test] lerobot-record Phase 2 dry-run"
  echo "[x86_test] Orin command receiver and Orin state bridge must already be running."
  PYTHONPATH="$PYTHONPATH_VALUE" $PYTHON_CMD -m lerobot.scripts.lerobot_record \
    --robot.type=jz_robot_udp \
    --robot.send_action_transport=udp \
    --robot.send_action_execution=dry_run \
    --robot.command_target_ip="$ORIN_IP" \
    --robot.command_target_port="$COMMAND_PORT" \
    --robot.allowed_sender_ip="$ORIN_IP" \
    --robot.rtsp_cameras='{}' \
    --teleop.type=jz_robot_udp_constant \
    --dataset.repo_id=local/jz_robot_udp_phase2_dry_run \
    --dataset.root=tests/outputs/jz_robot_udp_phase2_dry_run \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s="${RECORD_SECONDS:-10}" \
    --dataset.reset_time_s=0 \
    --dataset.fps="${RECORD_FPS:-5}" \
    --dataset.single_task="phase2 dry-run record chain test" \
    --dataset.push_to_hub=false \
    --display_data=false \
    --play_sounds=false
}

record_hold_dry_run() {
  echo "[x86_test] lerobot-record hold-action dry-run"
  echo "[x86_test] Orin Phase 3 executor and Orin state bridge must already be running in dry-run."
  PYTHONPATH="$PYTHONPATH_VALUE" $PYTHON_CMD -m lerobot.scripts.lerobot_record \
    --robot.type=jz_robot_udp \
    --robot.send_action_transport=udp \
    --robot.send_action_execution=dry_run \
    --robot.command_target_ip="$ORIN_IP" \
    --robot.command_target_port="$COMMAND_PORT" \
    --robot.allowed_sender_ip="$ORIN_IP" \
    --robot.rtsp_cameras='{}' \
    --teleop.type=jz_robot_udp_hold \
    --dataset.repo_id=local/jz_robot_udp_hold_phase3_dry_run \
    --dataset.root=tests/outputs/jz_robot_udp_hold_phase3_dry_run \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s="${RECORD_SECONDS:-10}" \
    --dataset.reset_time_s=0 \
    --dataset.fps="${RECORD_FPS:-5}" \
    --dataset.single_task="phase3 hold-action dry-run record chain test" \
    --dataset.push_to_hub=false \
    --display_data=false \
    --play_sounds=false
}

usage() {
  cat <<'EOF'
Usage: bash x86_test.sh [command]

Commands:
  constant_teleop_import  Import/factory check for jz_robot_udp_constant. Default.
  hold_teleop_import      Import/factory check for jz_robot_udp_hold.
  observation_check       READONLY get_observation() shape check.
  observation_values      READONLY print one frame of numeric action-equivalent values.
  send_action_dry_run     Send UDP dry-run command packets only.
  record_dry_run          Run lerobot-record with constant teleop dry-run.
  record_hold_dry_run     Run lerobot-record with hold teleop dry-run.

Common env overrides:
  PYTHON_CMD, ROBOT_CONFIG, ORIN_IP, COMMAND_PORT, OBS_COUNT, COMMAND_COUNT, RECORD_SECONDS
EOF
}

case "${1:-constant_teleop_import}" in
  constant_teleop_import) constant_teleop_import ;;
  hold_teleop_import) hold_teleop_import ;;
  observation_check) observation_check ;;
  observation_values) observation_values ;;
  send_action_dry_run) send_action_dry_run ;;
  record_dry_run) record_dry_run ;;
  record_hold_dry_run) record_hold_dry_run ;;
  help|-h|--help) usage ;;
  *)
    echo "[x86_test] unknown command: $1" >&2
    usage >&2
    exit 2
    ;;
esac
