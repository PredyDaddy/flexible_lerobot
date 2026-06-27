#!/usr/bin/env bash
set -euo pipefail

echo "[x86/status] matching observation monitor processes:"
pgrep -af "udp_test/test_scripts/x86_side/x86_jz_robot_udp_observation_check.py" || true
