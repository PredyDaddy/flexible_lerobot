#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "[all/stop_replay] stopping replay services through orin_arm/stop_all.sh"
bash "$ROOT_DIR/udp_test/server_bash/orin_arm/stop_all.sh"
