#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_CMD="${PYTHON_CMD:-conda run --no-capture-output -n lerobot_flex python}"
PYTHONPATH_VALUE="${PYTHONPATH_VALUE:-src}"
DATASET_ROOT="${DATASET_ROOT:-tests/outputs/jz_robot_udp_hold_phase3_dry_run_003}"

run_python() {
  PYTHONPATH="$PYTHONPATH_VALUE" $PYTHON_CMD "$@"
}

inspect_actions() {
  echo "[x86_test_datasets] inspect dataset actions"
  echo "[x86_test_datasets] DATASET_ROOT=$DATASET_ROOT"
  run_python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

root = Path("${DATASET_ROOT}")
parquet = root / "data/chunk-000/file-000.parquet"
info_json = root / "meta/info.json"
stats_json = root / "meta/stats.json"

print("root", root)
print("parquet", parquet)
if not parquet.exists():
    raise FileNotFoundError(f"Missing parquet file: {parquet}")

if info_json.exists():
    with info_json.open("r", encoding="utf-8") as f:
        info = json.load(f)
    print("info.fps", info.get("fps"))
    print("info.total_episodes", info.get("total_episodes"))
    print("info.total_frames", info.get("total_frames"))
else:
    print("WARN missing", info_json)

if stats_json.exists():
    print("stats_json", stats_json)
else:
    print("WARN missing", stats_json)

df = pd.read_parquet(parquet)
print("rows", len(df))
print("cols", len(df.columns))

action_cols = [col for col in df.columns if col.startswith("action.")]
obs_state_cols = [col for col in df.columns if col.startswith("observation.state.")]
direct_action_like_cols = [
    col
    for col in df.columns
    if (
        col.endswith(".pos")
        or col.endswith(".width")
        or col.endswith(".force")
    )
    and not col.startswith("observation.")
    and not col.startswith("action.")
]

print("action_columns", len(action_cols))
for col in action_cols:
    print("  action_col", col)

print("observation_state_columns", len(obs_state_cols))
for col in obs_state_cols:
    print("  obs_state_col", col)

if not action_cols and direct_action_like_cols:
    print("WARN no action.* columns found; using direct action-like columns")
    action_cols = direct_action_like_cols
    for col in action_cols:
        print("  direct_action_col", col)

if not action_cols:
    print("ERROR no action columns found")
    print("all_columns")
    for col in df.columns:
        print("  ", col)
    raise SystemExit(2)

print("action_summary")
all_zero = True
for col in action_cols:
    series = pd.to_numeric(df[col], errors="raise")
    min_value = float(series.min())
    max_value = float(series.max())
    first_value = float(series.iloc[0])
    mean_value = float(series.mean())
    nonzero_count = int((series != 0).sum())
    if nonzero_count:
        all_zero = False
    print(
        f"  {col}: first={first_value:.9f} min={min_value:.9f} "
        f"max={max_value:.9f} mean={mean_value:.9f} nonzero={nonzero_count}/{len(series)}"
    )

print("all_action_values_zero", all_zero)
if all_zero:
    raise SystemExit("ERROR action values are all zero; do not replay this dataset on hardware")

print("hold_action_vs_observation")
matched = 0
max_abs_diff = 0.0
for action_col in action_cols:
    suffix = action_col.removeprefix("action.")
    candidates = [
        f"observation.state.{suffix}",
        f"observation.{suffix}",
        suffix,
    ]
    obs_col = next((candidate for candidate in candidates if candidate in df.columns), None)
    if obs_col is None:
        print(f"  WARN no observation match for {action_col}")
        continue
    action_series = pd.to_numeric(df[action_col], errors="raise")
    obs_series = pd.to_numeric(df[obs_col], errors="raise")
    diff = (action_series - obs_series).abs()
    col_max_abs_diff = float(diff.max())
    max_abs_diff = max(max_abs_diff, col_max_abs_diff)
    matched += 1
    print(f"  {action_col} ~= {obs_col}: max_abs_diff={col_max_abs_diff:.12f}")

print("matched_action_observation_columns", matched)
print("max_abs_diff", f"{max_abs_diff:.12f}")
if matched == 0:
    print("WARN no action columns could be matched to observation columns")
elif max_abs_diff > 1e-9:
    raise SystemExit("ERROR action values do not exactly match observation hold values")

print("SUMMARY: PASS dataset action inspection")
PY
}

list_files() {
  echo "[x86_test_datasets] list dataset files"
  echo "[x86_test_datasets] DATASET_ROOT=$DATASET_ROOT"
  find "$DATASET_ROOT" -maxdepth 3 -type f -print | sort
}

usage() {
  cat <<'EOF'
Usage: bash x86_test_datasets.sh [command]

Commands:
  inspect_actions  Inspect action values and compare action.* with observation.*. Default.
  list_files       List dataset files.

Env:
  PYTHON_CMD       Default: conda run --no-capture-output -n lerobot_flex python
  PYTHONPATH_VALUE Default: src
  DATASET_ROOT     Default: tests/outputs/jz_robot_udp_hold_phase3_dry_run_003
EOF
}

case "${1:-inspect_actions}" in
  inspect_actions) inspect_actions ;;
  list_files) list_files ;;
  help|-h|--help) usage ;;
  *)
    echo "[x86_test_datasets] unknown command: $1" >&2
    usage >&2
    exit 2
    ;;
esac
