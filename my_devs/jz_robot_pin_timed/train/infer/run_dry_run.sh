#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXECUTION=dry_run
export SEND_ACTION_TRANSPORT=local
exec bash "${SCRIPT_DIR}/run_policy_record.sh"
