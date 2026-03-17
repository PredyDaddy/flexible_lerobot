#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="${1:-client}"
if [[ "${MODE}" != "client" && "${MODE}" != "server" ]]; then
  echo "Usage: $0 [client|server] [args...]" >&2
  exit 1
fi
shift || true

exec conda run -n lerobot_flex python "${SCRIPT_DIR}/sync_baseline_probe.py" "${MODE}" "$@"
