from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from my_devs.remove_feature_pipeline.core import (
    ARM_PLANS,
    PipelineError,
    recompute_vector_stats,
    slice_vector_features,
    update_info_shapes,
    validate_dataset,
)


def _to_fixed_size_list(matrix: np.ndarray) -> pa.Array:
    flat = pa.array(matrix.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, matrix.shape[1])


def _make_tiny_dataset(root: Path) -> Path:
    ds = root / "tiny_ds"
    (ds / "meta").mkdir(parents=True, exist_ok=True)
    (ds / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    info = {
        "features": {
            "action": {"dtype": "float32", "shape": [14], "names": [f"a{i}" for i in range(14)]},
            "observation.state": {"dtype": "float32", "shape": [14], "names": [f"s{i}" for i in range(14)]},
            "observation.images.cam_high": {"dtype": "video", "shape": [480, 640, 3]},
            "observation.images.cam_left_wrist": {"dtype": "video", "shape": [480, 640, 3]},
            "observation.images.cam_right_wrist": {"dtype": "video", "shape": [480, 640, 3]},
        }
    }
    (ds / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    # include q keys so recompute can preserve this style
    stats = {
        "action": {"q01": [], "q10": [], "q50": [], "q90": [], "q99": []},
        "observation.state": {"q01": [], "q10": [], "q50": [], "q90": [], "q99": []},
    }
    (ds / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))

    # action dim0 constant (for std_floor test), dim7~13 varying
    action = np.stack([np.arange(14, dtype=np.float32), np.arange(14, dtype=np.float32) + 10], axis=0)
    action[:, 0] = 1.0
    state = np.stack([np.arange(14, dtype=np.float32) * 2, np.arange(14, dtype=np.float32) * 3], axis=0)

    table = pa.table(
        {
            "action": _to_fixed_size_list(action),
            "observation.state": _to_fixed_size_list(state),
            "episode_index": pa.array([0, 0], type=pa.int64()),
            "frame_index": pa.array([0, 1], type=pa.int64()),
            "index": pa.array([0, 1], type=pa.int64()),
            "task_index": pa.array([0, 0], type=pa.int64()),
        }
    )
    pq.write_table(table, ds / "data" / "chunk-000" / "file-000.parquet")
    return ds


def test_arm_plan_indices():
    left = ARM_PLANS["left"]
    right = ARM_PLANS["right"]
    assert (left.start_idx, left.end_idx) == (0, 7)
    assert (right.start_idx, right.end_idx) == (7, 14)
    assert left.camera_to_remove.endswith("cam_right_wrist")
    assert right.camera_to_remove.endswith("cam_left_wrist")


def test_slice_vector_features_right_arm(tmp_path: Path):
    ds = _make_tiny_dataset(tmp_path)
    plan = ARM_PLANS["right"]
    slice_vector_features(ds, plan)

    table = pq.read_table(ds / "data" / "chunk-000" / "file-000.parquet", columns=["action", "observation.state"])
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    assert action.shape == (2, 7)
    assert state.shape == (2, 7)
    np.testing.assert_array_equal(action[0], np.arange(7, 14, dtype=np.float32))
    np.testing.assert_array_equal(action[1], np.arange(17, 24, dtype=np.float32))


def test_update_info_shapes_and_names(tmp_path: Path):
    ds = _make_tiny_dataset(tmp_path)
    info = update_info_shapes(ds / "meta" / "info.json", ARM_PLANS["right"])
    assert info["features"]["action"]["shape"] == [7]
    assert info["features"]["observation.state"]["shape"] == [7]
    assert info["features"]["action"]["names"] == [f"a{i}" for i in range(7, 14)]
    assert info["features"]["observation.state"]["names"] == [f"s{i}" for i in range(7, 14)]


def test_recompute_vector_stats_with_std_floor(tmp_path: Path):
    ds = _make_tiny_dataset(tmp_path)
    plan = ARM_PLANS["left"]
    slice_vector_features(ds, plan)
    update_info_shapes(ds / "meta" / "info.json", plan)

    files = [ds / "data" / "chunk-000" / "file-000.parquet"]
    stats = recompute_vector_stats(
        stats_path=ds / "meta" / "stats.json",
        files=files,
        std_floor=0.5,
        features=("action", "observation.state"),
    )
    action_std = np.array(stats["action"]["std"], dtype=np.float64)
    assert action_std.shape == (7,)
    # action dim0 was constant and retained by left-arm slice, so std should be floored
    assert action_std[0] == 0.5
    assert stats["action"]["count"] == [2]


def test_validate_dataset_camera_rules_and_shapes(tmp_path: Path):
    ds = _make_tiny_dataset(tmp_path)
    plan = ARM_PLANS["right"]

    # mimic camera-pruned dataset (only high + right wrist)
    info_path = ds / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"].pop("observation.images.cam_left_wrist")
    info_path.write_text(json.dumps(info, indent=2))

    slice_vector_features(ds, plan)
    update_info_shapes(info_path, plan)
    recompute_vector_stats(
        stats_path=ds / "meta" / "stats.json",
        files=[ds / "data" / "chunk-000" / "file-000.parquet"],
        std_floor=1e-3,
        features=("action", "observation.state"),
    )
    result = validate_dataset(ds, plan)
    assert result["num_files"] == 1
    assert result["num_cameras"] == 2
    assert result["nan_count"] == 0
    assert result["inf_count"] == 0


def test_validate_dataset_raises_on_wrong_cameras(tmp_path: Path):
    ds = _make_tiny_dataset(tmp_path)
    plan = ARM_PLANS["left"]

    # keep wrong pair for left plan (high + right wrist)
    info_path = ds / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"].pop("observation.images.cam_left_wrist")
    info_path.write_text(json.dumps(info, indent=2))

    slice_vector_features(ds, plan)
    update_info_shapes(info_path, plan)
    recompute_vector_stats(
        stats_path=ds / "meta" / "stats.json",
        files=[ds / "data" / "chunk-000" / "file-000.parquet"],
        std_floor=1e-3,
        features=("action", "observation.state"),
    )
    try:
        validate_dataset(ds, plan)
    except PipelineError:
        return
    raise AssertionError("Expected PipelineError on wrong camera pairing")
