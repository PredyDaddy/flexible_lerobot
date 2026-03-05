from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from my_devs.remove_feature_pipeline.core import (
    ARM_PLANS,
    PipelineError,
    _data_files,
    recompute_vector_stats,
    slice_vector_features,
    update_info_shapes,
    validate_dataset,
)


def _fixed_list_column(array_2d: np.ndarray) -> pa.FixedSizeListArray:
    flat = pa.array(array_2d.astype(np.float32).reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, array_2d.shape[1])


def _write_fake_dataset(dataset_dir: Path, include_three_cameras: bool = True) -> tuple[np.ndarray, np.ndarray]:
    (dataset_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "meta").mkdir(parents=True, exist_ok=True)

    n = 5
    action = np.arange(n * 14, dtype=np.float32).reshape(n, 14)
    state = (np.arange(n * 14, dtype=np.float32).reshape(n, 14) + 1000.0).astype(np.float32)

    table = pa.table(
        {
            "action": _fixed_list_column(action),
            "observation.state": _fixed_list_column(state),
            "episode_index": pa.array([0] * n, type=pa.int64()),
            "frame_index": pa.array(list(range(n)), type=pa.int64()),
            "index": pa.array(list(range(n)), type=pa.int64()),
            "task_index": pa.array([0] * n, type=pa.int64()),
            "timestamp": pa.array(np.linspace(0.0, 0.4, n).astype(np.float32), type=pa.float32()),
        }
    )
    pq.write_table(table, dataset_dir / "data" / "chunk-000" / "file-000.parquet")

    features = {
        "action": {"dtype": "float32", "shape": [14], "names": None},
        "observation.state": {"dtype": "float32", "shape": [14], "names": None},
        "observation.images.cam_high": {"dtype": "video", "shape": [480, 640, 3], "names": None},
        "observation.images.cam_right_wrist": {"dtype": "video", "shape": [480, 640, 3], "names": None},
    }
    if include_three_cameras:
        features["observation.images.cam_left_wrist"] = {"dtype": "video", "shape": [480, 640, 3], "names": None}

    with (dataset_dir / "meta" / "info.json").open("w") as f:
        json.dump({"features": features}, f, indent=2)

    stats = {
        "action": {
            "mean": action.mean(axis=0).tolist(),
            "std": action.std(axis=0).tolist(),
            "min": action.min(axis=0).tolist(),
            "max": action.max(axis=0).tolist(),
            "count": [n],
        },
        "observation.state": {
            "mean": state.mean(axis=0).tolist(),
            "std": state.std(axis=0).tolist(),
            "min": state.min(axis=0).tolist(),
            "max": state.max(axis=0).tolist(),
            "count": [n],
        },
    }
    with (dataset_dir / "meta" / "stats.json").open("w") as f:
        json.dump(stats, f, indent=2)

    return action, state


def test_slice_and_info_update_right_arm(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    action, state = _write_fake_dataset(dataset_dir, include_three_cameras=False)

    plan = ARM_PLANS["right"]
    slice_vector_features(dataset_dir, plan)
    update_info_shapes(dataset_dir / "meta" / "info.json", plan)

    table = pq.read_table(dataset_dir / "data" / "chunk-000" / "file-000.parquet")
    action_new = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    state_new = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    assert action_new.shape == (action.shape[0], 7)
    assert state_new.shape == (state.shape[0], 7)
    np.testing.assert_allclose(action_new, action[:, 7:14])
    np.testing.assert_allclose(state_new, state[:, 7:14])

    with (dataset_dir / "meta" / "info.json").open() as f:
        info = json.load(f)
    assert info["features"]["action"]["shape"] == [7]
    assert info["features"]["observation.state"]["shape"] == [7]


def test_recompute_stats_with_floor(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    _write_fake_dataset(dataset_dir, include_three_cameras=False)

    plan = ARM_PLANS["right"]
    slice_vector_features(dataset_dir, plan)
    update_info_shapes(dataset_dir / "meta" / "info.json", plan)

    files = _data_files(dataset_dir)
    stats = recompute_vector_stats(
        stats_path=dataset_dir / "meta" / "stats.json",
        files=files,
        std_floor=0.5,
        features=("action", "observation.state"),
    )
    assert len(stats["action"]["mean"]) == 7
    assert len(stats["observation.state"]["mean"]) == 7
    assert min(stats["action"]["std"]) >= 0.5
    assert min(stats["observation.state"]["std"]) >= 0.5


def test_validate_dataset_checks_camera_keys(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    _write_fake_dataset(dataset_dir, include_three_cameras=False)

    plan = ARM_PLANS["right"]
    slice_vector_features(dataset_dir, plan)
    update_info_shapes(dataset_dir / "meta" / "info.json", plan)
    recompute_vector_stats(
        stats_path=dataset_dir / "meta" / "stats.json",
        files=_data_files(dataset_dir),
        std_floor=1e-3,
        features=("action", "observation.state"),
    )

    validate_dataset(dataset_dir, plan)

    with (dataset_dir / "meta" / "info.json").open() as f:
        info = json.load(f)
    info["features"]["observation.images.cam_left_wrist"] = {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": None,
    }
    with (dataset_dir / "meta" / "info.json").open("w") as f:
        json.dump(info, f, indent=2)

    with pytest.raises(PipelineError):
        validate_dataset(dataset_dir, plan)
