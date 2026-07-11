#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BASE_SCRIPT="${REPO_ROOT}/my_devs/jz_robot_pin/edge/start_pin_state.sh"
STATE_HZ="${STATE_HZ:-30}"
JZ_TIMED_NON_30_STATE_HZ_CONFIRM="${JZ_TIMED_NON_30_STATE_HZ_CONFIRM:-}"
MAX_SOURCE_AGE_MS="${MAX_SOURCE_AGE_MS:-50}"
MAX_SOURCE_SKEW_MS="${MAX_SOURCE_SKEW_MS:-20}"
REQUIRE_ALL_SOURCES_ADVANCED="${REQUIRE_ALL_SOURCES_ADVANCED:-true}"
MIN_MEASURED_STATE_HZ_RATIO="${MIN_MEASURED_STATE_HZ_RATIO:-0.9}"
JZ_TIMED_SOURCE_AGE_MS_CONFIRM="${JZ_TIMED_SOURCE_AGE_MS_CONFIRM:-}"
JZ_TIMED_SOURCE_SKEW_MS_CONFIRM="${JZ_TIMED_SOURCE_SKEW_MS_CONFIRM:-}"
JZ_TIMED_REQUIRE_ALL_SOURCES_ADVANCED_CONFIRM="${JZ_TIMED_REQUIRE_ALL_SOURCES_ADVANCED_CONFIRM:-}"
JZ_TIMED_MIN_MEASURED_STATE_HZ_RATIO_CONFIRM="${JZ_TIMED_MIN_MEASURED_STATE_HZ_RATIO_CONFIRM:-}"

if [[ ! "${STATE_HZ}" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk -v hz="${STATE_HZ}" 'BEGIN { exit !(hz > 0) }'; then
  echo "[timed/edge/state] invalid STATE_HZ=${STATE_HZ}; expected a positive number" >&2
  exit 2
fi

for value_name in MAX_SOURCE_AGE_MS MAX_SOURCE_SKEW_MS; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "[timed/edge/state] invalid ${value_name}=${value}; expected a nonnegative number" >&2
    exit 2
  fi
done

if [[ ! "$MIN_MEASURED_STATE_HZ_RATIO" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]] || \
   ! awk -v value="$MIN_MEASURED_STATE_HZ_RATIO" 'BEGIN { exit !(value > 0 && value <= 1) }'; then
  echo "[timed/edge/state] invalid MIN_MEASURED_STATE_HZ_RATIO=$MIN_MEASURED_STATE_HZ_RATIO" >&2
  exit 2
fi

AGE_OVERRIDE_CONFIRMED=false
if ! awk -v value="$MAX_SOURCE_AGE_MS" 'BEGIN { exit !(value == 50) }'; then
  if [[ "$JZ_TIMED_SOURCE_AGE_MS_CONFIRM" != "$MAX_SOURCE_AGE_MS" ]]; then
    echo "[timed/edge/state] refusing MAX_SOURCE_AGE_MS=$MAX_SOURCE_AGE_MS without exact JZ_TIMED_SOURCE_AGE_MS_CONFIRM" >&2
    exit 2
  fi
  AGE_OVERRIDE_CONFIRMED=true
fi

SKEW_OVERRIDE_CONFIRMED=false
if ! awk -v value="$MAX_SOURCE_SKEW_MS" 'BEGIN { exit !(value == 20) }'; then
  if [[ "$JZ_TIMED_SOURCE_SKEW_MS_CONFIRM" != "$MAX_SOURCE_SKEW_MS" ]]; then
    echo "[timed/edge/state] refusing MAX_SOURCE_SKEW_MS=$MAX_SOURCE_SKEW_MS without exact JZ_TIMED_SOURCE_SKEW_MS_CONFIRM" >&2
    exit 2
  fi
  SKEW_OVERRIDE_CONFIRMED=true
fi

ADVANCED_OVERRIDE_CONFIRMED=false
case "${REQUIRE_ALL_SOURCES_ADVANCED,,}" in
  1|true|yes|on) REQUIRE_ALL_SOURCES_ADVANCED=true ;;
  0|false|no|off)
    if [[ "$JZ_TIMED_REQUIRE_ALL_SOURCES_ADVANCED_CONFIRM" != "$REQUIRE_ALL_SOURCES_ADVANCED" ]]; then
      echo "[timed/edge/state] refusing REQUIRE_ALL_SOURCES_ADVANCED=$REQUIRE_ALL_SOURCES_ADVANCED without exact JZ_TIMED_REQUIRE_ALL_SOURCES_ADVANCED_CONFIRM" >&2
      exit 2
    fi
    ADVANCED_OVERRIDE_CONFIRMED=true
    REQUIRE_ALL_SOURCES_ADVANCED=false
    ;;
  *)
    echo "[timed/edge/state] invalid REQUIRE_ALL_SOURCES_ADVANCED=$REQUIRE_ALL_SOURCES_ADVANCED" >&2
    exit 2
    ;;
esac

RATE_RATIO_OVERRIDE_CONFIRMED=false
if ! awk -v value="$MIN_MEASURED_STATE_HZ_RATIO" 'BEGIN { exit !(value == 0.9) }'; then
  if [[ "$JZ_TIMED_MIN_MEASURED_STATE_HZ_RATIO_CONFIRM" != "$MIN_MEASURED_STATE_HZ_RATIO" ]]; then
    echo "[timed/edge/state] refusing MIN_MEASURED_STATE_HZ_RATIO=$MIN_MEASURED_STATE_HZ_RATIO without exact JZ_TIMED_MIN_MEASURED_STATE_HZ_RATIO_CONFIRM" >&2
    exit 2
  fi
  RATE_RATIO_OVERRIDE_CONFIRMED=true
fi

NON_30_OVERRIDE_CONFIRMED=false
if ! awk -v hz="${STATE_HZ}" 'BEGIN { exit !(hz == 30) }'; then
  if [[ "${JZ_TIMED_NON_30_STATE_HZ_CONFIRM}" != "${STATE_HZ}" ]]; then
    echo "[timed/edge/state] refusing non-30 timed state rate STATE_HZ=${STATE_HZ}" >&2
    echo "[timed/edge/state] set JZ_TIMED_NON_30_STATE_HZ_CONFIRM=${STATE_HZ} to confirm this exact override" >&2
    exit 2
  fi
  NON_30_OVERRIDE_CONFIRMED=true
fi

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "[timed/edge/state] missing shared Pin edge script: ${BASE_SCRIPT}" >&2
  exit 1
fi

echo "[timed/edge/state] reusing the shared read-only ROS state UDP bridge"
echo "[timed/edge/state] profile=timed requested_hz=${STATE_HZ} expected_hz=${STATE_HZ} non_30_override_confirmed=${NON_30_OVERRIDE_CONFIRMED}"
echo "[timed/edge/state] freshness age_ms=${MAX_SOURCE_AGE_MS} skew_ms=${MAX_SOURCE_SKEW_MS} advanced=${REQUIRE_ALL_SOURCES_ADVANCED} min_rate_ratio=${MIN_MEASURED_STATE_HZ_RATIO} overrides_confirmed=${AGE_OVERRIDE_CONFIRMED}/${SKEW_OVERRIDE_CONFIRMED}/${ADVANCED_OVERRIDE_CONFIRMED}/${RATE_RATIO_OVERRIDE_CONFIRMED}"
exec env \
  STATE_HZ="${STATE_HZ}" \
  JZ_STATE_HZ_PROFILE=timed \
  JZ_EXPECTED_STATE_HZ="${STATE_HZ}" \
  JZ_TIMED_NON_30_STATE_HZ_CONFIRM="${JZ_TIMED_NON_30_STATE_HZ_CONFIRM}" \
  MAX_SOURCE_AGE_MS="${MAX_SOURCE_AGE_MS}" \
  MAX_SOURCE_SKEW_MS="${MAX_SOURCE_SKEW_MS}" \
  REQUIRE_ALL_SOURCES_ADVANCED="${REQUIRE_ALL_SOURCES_ADVANCED}" \
  MIN_MEASURED_STATE_HZ_RATIO="${MIN_MEASURED_STATE_HZ_RATIO}" \
  JZ_TIMED_MIN_MEASURED_STATE_HZ_RATIO_CONFIRM="${JZ_TIMED_MIN_MEASURED_STATE_HZ_RATIO_CONFIRM}" \
  bash "${BASE_SCRIPT}" "$@"
