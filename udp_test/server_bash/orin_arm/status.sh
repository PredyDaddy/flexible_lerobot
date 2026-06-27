#!/usr/bin/env bash
set -euo pipefail

echo "[orin_arm/status] matching ROS state UDP bridge processes:"
pgrep -af "udp_test/test_scripts/arm_side/orin_ros_state_udp_bridge.py" || true
