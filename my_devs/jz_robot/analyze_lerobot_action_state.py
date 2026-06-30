#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze LeRobot action vs observation.state amplitudes.")
    parser.add_argument("--dataset-root", required=True, help="Path to one LeRobot dataset root.")
    parser.add_argument("--episode", type=int, default=None, help="Optional episode_index filter.")
    parser.add_argument("--parquet", default=None, help="Optional explicit parquet file path.")
    parser.add_argument("--csv-out", default=None, help="Optional CSV output path for per-dimension stats.")
    return parser.parse_args()


def load_info(root: Path) -> dict:
    info_json = root / "meta/info.json"
    if not info_json.exists():
        return {}
    with info_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_parquet(root: Path, explicit: str | None) -> Path:
    if explicit:
        parquet = Path(explicit).expanduser()
        if not parquet.exists():
            raise FileNotFoundError(f"Missing parquet file: {parquet}")
        return parquet

    candidates = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")
    return candidates[0]


def stack_column(df: pd.DataFrame, column: str) -> np.ndarray | None:
    if column not in df.columns:
        return None
    values = np.stack(df[column].to_numpy()).astype(np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return values


def feature_names(info: dict, feature: str, width: int) -> list[str]:
    names = info.get("features", {}).get(feature, {}).get("names")
    if names and len(names) == width:
        return list(names)
    return [f"{feature}[{idx}]" for idx in range(width)]


def per_dim_rows(label: str, values: np.ndarray, names: list[str]) -> list[dict[str, float | int | str]]:
    rows = []
    for idx, name in enumerate(names):
        series = values[:, idx]
        min_value = float(series.min())
        max_value = float(series.max())
        amplitude_rad = max_value - min_value
        rows.append(
            {
                "source": label,
                "dim": idx,
                "name": name,
                "start": float(series[0]),
                "end": float(series[-1]),
                "min": min_value,
                "max": max_value,
                "amplitude_rad": amplitude_rad,
                "amplitude_deg": float(np.degrees(amplitude_rad)),
                "mean": float(series.mean()),
                "std": float(series.std()),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).expanduser()
    info = load_info(root)
    parquet = find_parquet(root, args.parquet)
    df = pd.read_parquet(parquet)
    if args.episode is not None and "episode_index" in df.columns:
        df = df[df["episode_index"] == args.episode]
    if df.empty:
        raise RuntimeError("No rows left to analyze after filtering")

    action = stack_column(df, "action")
    observation_state = stack_column(df, "observation.state")
    if action is None:
        raise RuntimeError("Dataset has no vector column named 'action'")

    print(f"dataset_root={root}")
    print(f"parquet={parquet}")
    print(f"rows={len(df)}")
    if info:
        print(f"fps={info.get('fps')}")
        print(f"total_episodes={info.get('total_episodes')}")
        print(f"total_frames={info.get('total_frames')}")

    action_names = feature_names(info, "action", action.shape[1])
    rows = per_dim_rows("action", action, action_names)
    print("action_amplitudes")
    for row in rows:
        print(
            f"  {row['dim']:02d} {row['name']}: "
            f"start={row['start']:.9f} min={row['min']:.9f} max={row['max']:.9f} "
            f"amp_rad={row['amplitude_rad']:.9f} amp_deg={row['amplitude_deg']:.3f}"
        )

    if observation_state is not None:
        obs_names = feature_names(info, "observation.state", observation_state.shape[1])
        obs_rows = per_dim_rows("observation.state", observation_state, obs_names)
        rows.extend(obs_rows)
        print("observation_state_amplitudes")
        for row in obs_rows:
            print(
                f"  {row['dim']:02d} {row['name']}: "
                f"start={row['start']:.9f} min={row['min']:.9f} max={row['max']:.9f} "
                f"amp_rad={row['amplitude_rad']:.9f} amp_deg={row['amplitude_deg']:.3f}"
            )

        shared_dim = min(action.shape[1], observation_state.shape[1])
        diff = np.abs(action[:, :shared_dim] - observation_state[:, :shared_dim])
        max_abs_diff = float(diff.max()) if diff.size else 0.0
        mean_abs_diff = float(diff.mean()) if diff.size else 0.0
        print("action_vs_observation_state")
        print(f"  compared_dims={shared_dim}")
        print(f"  max_abs_diff={max_abs_diff:.12f}")
        print(f"  mean_abs_diff={mean_abs_diff:.12f}")
        print(f"  action_is_feedback_state_copy={max_abs_diff <= 1e-9}")
    else:
        print("WARN: Dataset has no vector column named 'observation.state'")

    if args.csv_out:
        out = Path(args.csv_out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"csv_out={out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

