#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_CMD="${PYTHON_CMD:-conda run --no-capture-output -n lerobot_flex python}"
PYTHONPATH_VALUE="${PYTHONPATH_VALUE:-src}"
DATASET_ROOT="${DATASET_ROOT:-tests/outputs/jz_robot_udp_hold_phase3_dry_run_003}"

run_python() {
  PYTHONPATH="$PYTHONPATH_VALUE" DATASET_ROOT="$DATASET_ROOT" $PYTHON_CMD "$@"
}

inspect_actions() {
  echo "[x86_test_datasets] inspect dataset action/state semantics"
  echo "[x86_test_datasets] DATASET_ROOT=$DATASET_ROOT"
  run_python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(os.environ["DATASET_ROOT"])
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

def print_vector_summary(label, values, names):
    values = np.asarray(values, dtype=np.float64)
    nonzero_count = int(np.count_nonzero(values))
    print(f"{label}_summary")
    print(
        f"  {label}: shape={values.shape} first={float(values.flat[0]):.9f} "
        f"min={float(values.min()):.9f} max={float(values.max()):.9f} "
        f"mean={float(values.mean()):.9f} nonzero={nonzero_count}/{values.size}"
    )
    print(f"{label}_per_dim")
    for idx, name in enumerate(names):
        series = values[:, idx]
        min_value = float(series.min())
        max_value = float(series.max())
        amplitude = max_value - min_value
        print(
            f"  {idx:02d} {name}: start={float(series[0]):.9f} min={min_value:.9f} "
            f"max={max_value:.9f} amp_rad={amplitude:.9f} amp_deg={np.degrees(amplitude):.3f}"
        )
    return nonzero_count

if "action" in df.columns:
    action_names = None
    obs_names = None
    if info_json.exists():
        features = info.get("features", {})
        action_names = features.get("action", {}).get("names")
        obs_names = features.get("observation.state", {}).get("names")

    action_values = np.stack(df["action"].to_numpy())
    if action_values.ndim == 1:
        action_values = action_values.reshape(-1, 1)
    action_names = action_names or [f"action[{idx}]" for idx in range(action_values.shape[1])]

    print("action_columns", 1)
    print("  action_col action")
    nonzero_count = print_vector_summary("action", action_values, action_names)
    all_zero = nonzero_count == 0
    print("all_action_values_zero", all_zero)
    if all_zero:
        raise SystemExit("ERROR action values are all zero; do not replay this dataset on hardware")

    if "observation.state" in df.columns:
        obs_values = np.stack(df["observation.state"].to_numpy())
        if obs_values.ndim == 1:
            obs_values = obs_values.reshape(-1, 1)
        obs_names = obs_names or [f"observation.state[{idx}]" for idx in range(obs_values.shape[1])]
        print("observation_state_columns", 1)
        print("  obs_state_col observation.state")
        print_vector_summary("observation_state", obs_values, obs_names)

        shared_dim = min(action_values.shape[1], obs_values.shape[1])
        diff = np.abs(action_values[:, :shared_dim] - obs_values[:, :shared_dim])
        max_abs_diff = float(diff.max()) if diff.size else 0.0
        mean_abs_diff = float(diff.mean()) if diff.size else 0.0
        equal_to_state = max_abs_diff <= 1e-9
        print("action_vs_observation_state")
        print("  compared_dims", shared_dim)
        print("  max_abs_diff", f"{max_abs_diff:.12f}")
        print("  mean_abs_diff", f"{mean_abs_diff:.12f}")
        print("  action_is_feedback_state_copy", equal_to_state)
        if equal_to_state:
            print("  SEMANTICS: old hold dataset; action is a copy of feedback observation.state")
        else:
            print("  SEMANTICS: action differs from feedback state; verify it matches VR target/action logs")
    else:
        print("WARN no observation.state column could be compared to action")
        print("observation_state_columns", 0)

    print("SUMMARY: PASS dataset action/state inspection")
    raise SystemExit(0)

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
elif max_abs_diff <= 1e-9:
    print("action_is_feedback_state_copy", True)
    print("SEMANTICS: old hold dataset; action is a copy of feedback observation.state")
else:
    print("action_is_feedback_state_copy", False)
    print("SEMANTICS: action differs from feedback state; verify it matches VR target/action logs")

print("SUMMARY: PASS dataset action/state inspection")
PY
}

list_files() {
  echo "[x86_test_datasets] list dataset files"
  echo "[x86_test_datasets] DATASET_ROOT=$DATASET_ROOT"
  find "$DATASET_ROOT" -maxdepth 3 -type f -print | sort
}

analyze_action_state() {
  echo "[x86_test_datasets] analyze action/state amplitudes"
  echo "[x86_test_datasets] DATASET_ROOT=$DATASET_ROOT"
  run_python my_devs/jz_robot/analyze_lerobot_action_state.py --dataset-root "$DATASET_ROOT" "$@"
}

usage() {
  cat <<'EOF'
Usage: bash x86_test_datasets.sh [command]

Commands:
  inspect_actions  Inspect action values, amplitudes, and action-vs-state semantics. Default.
  analyze          Run reusable action/state amplitude analyzer. Extra args are passed through.
  list_files       List dataset files.

Env:
  PYTHON_CMD       Default: conda run --no-capture-output -n lerobot_flex python
  PYTHONPATH_VALUE Default: src
  DATASET_ROOT     Default: tests/outputs/jz_robot_udp_hold_phase3_dry_run_003
EOF
}

case "${1:-inspect_actions}" in
  inspect_actions) inspect_actions ;;
  analyze) shift; analyze_action_state "$@" ;;
  list_files) list_files ;;
  help|-h|--help) usage ;;
  *)
    echo "[x86_test_datasets] unknown command: $1" >&2
    usage >&2
    exit 2
    ;;
esac
