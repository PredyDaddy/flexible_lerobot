#!/usr/bin/env python3
"""Single-arm dataset building pipeline for LeRobot datasets.

This module builds a new dataset for either left-arm or right-arm training:
1) remove the opposite wrist camera,
2) slice `action` and `observation.state` from 14D to 7D,
3) update meta/info.json shapes,
4) recompute vector stats in meta/stats.json for sliced features,
5) run sanity checks.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.dataset_tools import remove_feature
from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass(frozen=True)
class ArmPlan:
    name: str
    start_idx: int
    end_idx: int
    camera_to_remove: str
    expected_cameras: tuple[str, str]


ARM_PLANS: dict[str, ArmPlan] = {
    "left": ArmPlan(
        name="left",
        start_idx=0,
        end_idx=7,
        camera_to_remove="observation.images.cam_right_wrist",
        expected_cameras=("observation.images.cam_high", "observation.images.cam_left_wrist"),
    ),
    "right": ArmPlan(
        name="right",
        start_idx=7,
        end_idx=14,
        camera_to_remove="observation.images.cam_left_wrist",
        expected_cameras=("observation.images.cam_high", "observation.images.cam_right_wrist"),
    ),
}

VECTOR_FEATURES = ("action", "observation.state")


class PipelineError(RuntimeError):
    """Raised when pipeline preconditions or checks fail."""


def _ensure_exists(path: Path, desc: str) -> None:
    if not path.exists():
        raise PipelineError(f"{desc} not found: {path}")


def _data_files(dataset_dir: Path) -> list[Path]:
    files = sorted((dataset_dir / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise PipelineError(f"No parquet files under {dataset_dir / 'data'}")
    return files


def _replace_vector_column(table: pa.Table, column_name: str, start: int, end: int) -> pa.Table:
    col_idx = table.schema.get_field_index(column_name)
    if col_idx < 0:
        raise PipelineError(f"Column '{column_name}' not found in parquet schema")

    array_2d = np.asarray(table[column_name].to_pylist(), dtype=np.float32)
    if array_2d.ndim != 2:
        raise PipelineError(f"Column '{column_name}' is not 2D list-like, got shape={array_2d.shape}")
    if array_2d.shape[1] < end:
        raise PipelineError(
            f"Column '{column_name}' width={array_2d.shape[1]} < required end index {end} for slicing"
        )

    sliced = np.ascontiguousarray(array_2d[:, start:end], dtype=np.float32)
    flat = pa.array(sliced.reshape(-1), type=pa.float32())
    replacement = pa.FixedSizeListArray.from_arrays(flat, end - start)
    return table.set_column(col_idx, column_name, replacement)


def slice_vector_features(dataset_dir: Path, plan: ArmPlan) -> None:
    """Slice action/state from 14D to 7D for all data parquet files."""
    files = _data_files(dataset_dir)
    for file in files:
        table = pq.read_table(file)
        table = _replace_vector_column(table, "action", plan.start_idx, plan.end_idx)
        table = _replace_vector_column(table, "observation.state", plan.start_idx, plan.end_idx)
        pq.write_table(table, file)
    logging.info("Sliced vector features in %d parquet files to %dD", len(files), plan.end_idx - plan.start_idx)


def update_info_shapes(info_path: Path, plan: ArmPlan) -> dict:
    _ensure_exists(info_path, "info.json")
    with info_path.open() as f:
        info = json.load(f)

    width = plan.end_idx - plan.start_idx
    for key in VECTOR_FEATURES:
        feature = info.get("features", {}).get(key)
        if feature is None:
            raise PipelineError(f"Feature '{key}' missing in info.json")
        feature["shape"] = [width]
        names = feature.get("names")
        if isinstance(names, list) and len(names) >= plan.end_idx:
            feature["names"] = names[plan.start_idx : plan.end_idx]

    with info_path.open("w") as f:
        json.dump(info, f, indent=4)
    logging.info("Updated info shapes to %dD for %s", width, ", ".join(VECTOR_FEATURES))
    return info


def _load_feature_matrix(files: Iterable[Path], feature: str) -> np.ndarray:
    parts: list[np.ndarray] = []
    for file in files:
        table = pq.read_table(file, columns=[feature])
        parts.append(np.asarray(table[feature].to_pylist(), dtype=np.float64))
    matrix = np.concatenate(parts, axis=0)
    if matrix.ndim != 2:
        raise PipelineError(f"Feature '{feature}' expected 2D matrix, got shape={matrix.shape}")
    return matrix


def _quantile_keys(existing_stats: dict) -> list[tuple[str, float]]:
    quantiles: list[tuple[str, float]] = []
    for key in existing_stats:
        m = re.fullmatch(r"q(\d{2})", key)
        if m:
            q = int(m.group(1)) / 100.0
            quantiles.append((key, q))
    return sorted(quantiles, key=lambda x: x[1])


def recompute_vector_stats(stats_path: Path, files: list[Path], std_floor: float, features: Iterable[str]) -> dict:
    _ensure_exists(stats_path, "stats.json")
    with stats_path.open() as f:
        stats = json.load(f)

    for feature in features:
        matrix = _load_feature_matrix(files, feature)
        f_stats = dict(stats.get(feature, {}))

        mean = matrix.mean(axis=0, dtype=np.float64)
        std_raw = matrix.std(axis=0, dtype=np.float64)
        std = np.maximum(std_raw, std_floor)
        min_val = matrix.min(axis=0)
        max_val = matrix.max(axis=0)

        f_stats["mean"] = mean.tolist()
        f_stats["std"] = std.tolist()
        f_stats["min"] = min_val.tolist()
        f_stats["max"] = max_val.tolist()
        f_stats["count"] = [int(matrix.shape[0])]

        for q_key, q_val in _quantile_keys(f_stats):
            f_stats[q_key] = np.quantile(matrix, q_val, axis=0).tolist()

        stats[feature] = f_stats

        tiny_count = int((std_raw < std_floor).sum())
        logging.info(
            "Recomputed stats for %s: rows=%d dim=%d tiny_std_dims=%d (floor=%g)",
            feature,
            matrix.shape[0],
            matrix.shape[1],
            tiny_count,
            std_floor,
        )

    with stats_path.open("w") as f:
        json.dump(stats, f, indent=4)
    return stats


def _camera_keys_from_info(info: dict) -> list[str]:
    features = info.get("features", {})
    return sorted([k for k in features if k.startswith("observation.images.")])


def validate_dataset(dataset_dir: Path, plan: ArmPlan) -> dict[str, int]:
    info_path = dataset_dir / "meta" / "info.json"
    stats_path = dataset_dir / "meta" / "stats.json"
    _ensure_exists(info_path, "info.json")
    _ensure_exists(stats_path, "stats.json")

    with info_path.open() as f:
        info = json.load(f)
    with stats_path.open() as f:
        stats = json.load(f)

    expected_cameras = sorted(plan.expected_cameras)
    cameras = _camera_keys_from_info(info)
    if cameras != expected_cameras:
        raise PipelineError(f"Unexpected camera keys: got={cameras}, expected={expected_cameras}")

    for key in VECTOR_FEATURES:
        shape = info.get("features", {}).get(key, {}).get("shape")
        if shape != [7]:
            raise PipelineError(f"{key} shape is {shape}, expected [7]")
        stat_shape = len(stats.get(key, {}).get("mean", []))
        if stat_shape != 7:
            raise PipelineError(f"{key} stats mean dim is {stat_shape}, expected 7")

    files = _data_files(dataset_dir)
    nan_count = 0
    inf_count = 0
    for file in files:
        table = pq.read_table(file, columns=list(VECTOR_FEATURES))
        for key in VECTOR_FEATURES:
            matrix = np.asarray(table[key].to_pylist(), dtype=np.float64)
            if matrix.shape[1] != 7:
                raise PipelineError(f"{key} width is {matrix.shape[1]} in {file}, expected 7")
            nan_count += int(np.isnan(matrix).sum())
            inf_count += int(np.isinf(matrix).sum())

    if nan_count > 0 or inf_count > 0:
        raise PipelineError(f"Found invalid values in vectors: nan={nan_count}, inf={inf_count}")

    result = {
        "num_files": len(files),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "num_cameras": len(cameras),
    }
    logging.info("Validation passed: %s", result)
    return result


def build_single_arm_dataset(
    source_root: Path,
    source_repo_id: str,
    target_repo_id: str,
    arm: str,
    work_root: Path | None = None,
    std_floor: float = 1e-3,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict:
    if arm not in ARM_PLANS:
        raise PipelineError(f"arm must be one of {list(ARM_PLANS)}, got '{arm}'")
    plan = ARM_PLANS[arm]

    source_root = source_root.resolve()
    work_root = (work_root or source_root).resolve()
    source_dir = source_root / source_repo_id
    target_dir = work_root / target_repo_id

    _ensure_exists(source_dir, "source dataset directory")
    if target_dir.exists():
        if not overwrite:
            raise PipelineError(f"Target directory already exists: {target_dir}")
        if dry_run:
            logging.info("Dry-run: target exists and would be removed because overwrite=true: %s", target_dir)
        else:
            shutil.rmtree(target_dir)

    logging.info("Plan: source=%s", source_dir)
    logging.info("Plan: target=%s", target_dir)
    logging.info("Plan: arm=%s slice=[%d:%d]", arm, plan.start_idx, plan.end_idx)
    logging.info("Plan: remove camera=%s", plan.camera_to_remove)
    logging.info("Plan: std_floor=%g", std_floor)

    if dry_run:
        return {
            "source_dir": str(source_dir),
            "target_dir": str(target_dir),
            "arm": arm,
            "camera_to_remove": plan.camera_to_remove,
            "slice": [plan.start_idx, plan.end_idx],
            "dry_run": True,
        }

    # NOTE: LeRobotDataset expects `root` to point to the concrete dataset directory
    # for local datasets in this repository layout.
    dataset = LeRobotDataset(repo_id=source_repo_id, root=source_dir)
    remove_feature(
        dataset=dataset,
        feature_names=[plan.camera_to_remove],
        output_dir=target_dir,
        repo_id=target_repo_id,
    )
    logging.info("Step A complete: removed camera and materialized dataset at %s", target_dir)

    slice_vector_features(target_dir, plan)
    info = update_info_shapes(target_dir / "meta" / "info.json", plan)
    stats = recompute_vector_stats(
        stats_path=target_dir / "meta" / "stats.json",
        files=_data_files(target_dir),
        std_floor=std_floor,
        features=VECTOR_FEATURES,
    )
    validation = validate_dataset(target_dir, plan)

    result = {
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
        "arm": arm,
        "camera_keys": _camera_keys_from_info(info),
        "action_dim": info["features"]["action"]["shape"][0],
        "state_dim": info["features"]["observation.state"]["shape"][0],
        "validation": validation,
        "stats_keys": list(stats.keys()),
    }
    logging.info("Pipeline finished successfully: %s", result)
    return result
