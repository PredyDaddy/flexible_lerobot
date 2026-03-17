#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec conda run -n lerobot_flex python "${SCRIPT_DIR}/run_groot_remote_async_policy_server.py" "$@"
