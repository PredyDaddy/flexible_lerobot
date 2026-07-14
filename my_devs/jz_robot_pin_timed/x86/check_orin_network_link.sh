#!/usr/bin/env bash
set -euo pipefail

ORIN_IP="${ORIN_IP:-192.168.1.81}"
PING_COUNT="${PING_COUNT:-300}"
PING_INTERVAL_S="${PING_INTERVAL_S:-0.1}"
PING_TIMEOUT_S="${PING_TIMEOUT_S:-1}"

if ! command -v ip >/dev/null || ! command -v ping >/dev/null || ! command -v ethtool >/dev/null; then
  echo "[timed/network] requires ip, ping, and ethtool" >&2
  exit 2
fi

route="$(ip route get "${ORIN_IP}")"
interface="$(awk '/ dev / { for (i = 1; i <= NF; i++) if ($i == "dev") { print $(i + 1); exit } }' <<<"${route}")"
if [[ -z "${interface}" ]]; then
  echo "[timed/network] unable to determine interface for ${ORIN_IP}" >&2
  exit 2
fi

read_counters() {
  awk -v device="${interface}" '$1 == device ":" { print $2, $3, $4, $5, $10, $11, $12, $13 }' /proc/net/dev
}

read -r rx_bytes_before rx_packets_before rx_errors_before rx_dropped_before \
  tx_bytes_before tx_packets_before tx_errors_before tx_dropped_before < <(read_counters)

echo "[timed/network] target=${ORIN_IP} interface=${interface}"
echo "[timed/network] route=${route}"
ethtool "${interface}" | awk '/Speed:|Duplex:|Auto-negotiation:|Link detected:/'
echo "[timed/network] counters_before rx_errors=${rx_errors_before} rx_dropped=${rx_dropped_before}" \
  "tx_errors=${tx_errors_before} tx_dropped=${tx_dropped_before}"

echo "[timed/network] MTU probe (1472 byte payload, DF set)"
ping -n -M do -c 3 -W "${PING_TIMEOUT_S}" -s 1472 "${ORIN_IP}"

echo "[timed/network] latency probe count=${PING_COUNT} interval_s=${PING_INTERVAL_S}"
ping -n -c "${PING_COUNT}" -i "${PING_INTERVAL_S}" -W "${PING_TIMEOUT_S}" "${ORIN_IP}"

read -r rx_bytes_after rx_packets_after rx_errors_after rx_dropped_after \
  tx_bytes_after tx_packets_after tx_errors_after tx_dropped_after < <(read_counters)

echo "[timed/network] counter_delta rx_bytes=$((rx_bytes_after - rx_bytes_before))" \
  "rx_packets=$((rx_packets_after - rx_packets_before))" \
  "rx_errors=$((rx_errors_after - rx_errors_before))" \
  "rx_dropped=$((rx_dropped_after - rx_dropped_before))" \
  "tx_bytes=$((tx_bytes_after - tx_bytes_before))" \
  "tx_packets=$((tx_packets_after - tx_packets_before))" \
  "tx_errors=$((tx_errors_after - tx_errors_before))" \
  "tx_dropped=$((tx_dropped_after - tx_dropped_before))"
echo "[timed/network] PASS means zero packet loss, stable latency, and zero error/drop deltas."
