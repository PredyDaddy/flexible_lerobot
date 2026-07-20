#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

[[ "$#" -eq 0 ]] \
  || rtc_die "run_onboard_robot_client_all_170.sh accepts no CLI arguments"
[[ "${JZ_PI05_ALL_170_047320_CONFIRMED:-}" == "1" ]] \
  || rtc_die "all_170 checkpoint 047320 requires JZ_PI05_ALL_170_047320_CONFIRMED=1"

exec env \
  ONBOARD_CHECKPOINT=all_170_047320 \
  JZ_PI05_INTERMEDIATE_010470_CONFIRMED=0 \
  JZ_PI05_ALL_170_047320_CONFIRMED=1 \
  JZ_PI05_ALL_200_007320_CONFIRMED=0 \
  bash "${SCRIPT_DIR}/run_onboard_robot_client.sh"
