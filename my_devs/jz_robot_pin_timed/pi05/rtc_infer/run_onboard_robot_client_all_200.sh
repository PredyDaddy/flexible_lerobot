#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

[[ "$#" -eq 0 ]] \
  || rtc_die "run_onboard_robot_client_all_200.sh accepts no CLI arguments"
[[ "${JZ_PI05_ALL_200_007320_CONFIRMED:-}" == "1" ]] \
  || rtc_die "all_200 checkpoint 007320 requires JZ_PI05_ALL_200_007320_CONFIRMED=1"

exec env \
  ONBOARD_CHECKPOINT=all_200_007320 \
  JZ_PI05_INTERMEDIATE_010470_CONFIRMED=0 \
  JZ_PI05_ALL_170_047320_CONFIRMED=0 \
  JZ_PI05_ALL_200_007320_CONFIRMED=1 \
  bash "${SCRIPT_DIR}/run_onboard_robot_client.sh"
