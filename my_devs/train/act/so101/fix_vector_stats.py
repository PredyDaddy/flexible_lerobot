#!/usr/bin/env python3
"""
Recompute vector feature statistics from dataset parquet files and apply a std floor.

Why this exists:
- Some datasets have near-constant or fully constant action/state dimensions (for example
  dual-arm data where one arm stays still).
- If `mean` has tiny numerical drift while `std` is zero, `(x - mean) / (std + eps)` can
  explode and dominate training loss.

This tool recomputes exact stats directly from parquet and clamps small std values.
It only updates vector stats (e.g. action / observation.state) in meta/stats.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="LeRobot dataset root, e.g. datasets/lerobot_datasets/test_pipeline_clean",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="action,observation.state",
        help="Comma-separated vector features to recompute.",
    )
    parser.add_argument(
        "--std-floor",
        type=float,
        default=1e-3,
        help="Minimum std applied per dimension for selected features.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print report without writing stats.json.",
    )
    return parser.parse_args()


def _to_numpy_column(column) -> np.ndarray:
    return np.asarray(column.to_pylist(), dtype=np.float64)


def _load_feature_matrix(data_files: list[Path], feature: str) -> np.ndarray:
    parts: list[np.ndarray] = []
    for file in data_files:
        table = pq.read_table(file, columns=[feature])
        parts.append(_to_numpy_column(table[feature]))
    if not parts:
        raise ValueError(f"No data found for feature '{feature}'.")
    return np.concatenate(parts, axis=0)


def _summarize_normalized_range(values: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8) -> str:
    norm = (values - mean[None, :]) / (std[None, :] + eps)
    abs_max = np.max(np.abs(norm), axis=0)
    return (
        f"abs(normalized) max: min={abs_max.min():.6f}, "
        f"median={np.median(abs_max):.6f}, max={abs_max.max():.6f}"
    )


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    stats_path = dataset_root / "meta" / "stats.json"
    data_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))

    if not stats_path.exists():
        raise FileNotFoundError(f"stats.json not found: {stats_path}")
    if not data_files:
        raise FileNotFoundError(f"No parquet data files under: {dataset_root / 'data'}")

    with stats_path.open() as f:
        stats = json.load(f)

    features = [x.strip() for x in args.features.split(",") if x.strip()]
    print(f"[info] dataset_root={dataset_root}")
    print(f"[info] data_files={len(data_files)}")
    print(f"[info] features={features}")
    print(f"[info] std_floor={args.std_floor:g}")

    for feature in features:
        values = _load_feature_matrix(data_files, feature)
        if values.ndim != 2:
            raise ValueError(f"Feature '{feature}' expected 2D matrix, got shape={values.shape}")

        old_mean = np.asarray(stats.get(feature, {}).get("mean", []), dtype=np.float64)
        old_std = np.asarray(stats.get(feature, {}).get("std", []), dtype=np.float64)
        old_min = np.asarray(stats.get(feature, {}).get("min", []), dtype=np.float64)
        old_max = np.asarray(stats.get(feature, {}).get("max", []), dtype=np.float64)

        new_mean = values.mean(axis=0, dtype=np.float64)
        new_std_raw = values.std(axis=0, dtype=np.float64)
        new_std = np.maximum(new_std_raw, args.std_floor)
        new_min = values.min(axis=0)
        new_max = values.max(axis=0)

        tiny_dims = np.where(new_std_raw < args.std_floor)[0].tolist()
        const_dims = np.where((new_max - new_min) == 0.0)[0].tolist()

        print("")
        print(f"[feature] {feature} shape={values.shape}")
        print(f"[feature] const_dims={const_dims}")
        print(f"[feature] std<{args.std_floor:g} dims={tiny_dims}")

        if old_mean.size == new_mean.size and old_std.size == new_std.size:
            mean_abs_delta = np.abs(old_mean - new_mean)
            std_abs_delta = np.abs(old_std - new_std)
            print(
                "[feature] old->new |mean| delta: "
                f"min={mean_abs_delta.min():.6e}, median={np.median(mean_abs_delta):.6e}, max={mean_abs_delta.max():.6e}"
            )
            print(
                "[feature] old->new |std|  delta: "
                f"min={std_abs_delta.min():.6e}, median={np.median(std_abs_delta):.6e}, max={std_abs_delta.max():.6e}"
            )

        print(f"[feature] old { _summarize_normalized_range(values, old_mean, old_std) if old_mean.size else 'N/A' }")
        print(f"[feature] new { _summarize_normalized_range(values, new_mean, new_std) }")

        if feature not in stats:
            stats[feature] = {}
        stats[feature]["mean"] = new_mean.tolist()
        stats[feature]["std"] = new_std.tolist()
        stats[feature]["min"] = new_min.tolist()
        stats[feature]["max"] = new_max.tolist()

    if args.dry_run:
        print("\n[done] dry-run only, no file written.")
        return

    backup_path = stats_path.with_name(f"stats.json.bak_std_fix")
    backup_path.write_text(stats_path.read_text())
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=4)
    print(f"\n[done] backup written: {backup_path}")
    print(f"[done] updated stats: {stats_path}")


if __name__ == "__main__":
    main()
