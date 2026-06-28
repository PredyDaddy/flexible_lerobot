#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

echo "[orin_arm/stop_all] stopping Phase 3 command executor"
bash udp_test/server_bash/orin_arm/stop_phase3_executor.sh

echo "[orin_arm/stop_all] stopping Phase 2 command dry-run receiver"
bash udp_test/server_bash/orin_arm/stop_command_receiver.sh

echo "[orin_arm/stop_all] stopping ROS state UDP bridge"
bash udp_test/server_bash/orin_arm/stop.sh

echo "[orin_arm/stop_all] status after stop:"
bash udp_test/server_bash/orin_arm/status.sh
