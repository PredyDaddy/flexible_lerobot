from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from my_devs.split_datasets.split_bimanual_lerobot_dataset import (
    ACTION_KEY,
    COMMON_CAMERA_KEY,
    DEFAULT_LEFT_REPO_ID,
    DEFAULT_RIGHT_REPO_ID,
    DEFAULT_SOURCE_REPO_ID,
    DEFAULT_SOURCE_ROOT,
    DEFAULT_TARGET_ROOT,
    LEFT_CAMERA_KEY,
    RIGHT_CAMERA_KEY,
    STATE_KEY,
    resolve_existing_dataset_root,
)


@dataclass(frozen=True)
class DatasetSpec:
    side: str
    repo_id: str
    vector_slice: slice
    camera_keys: tuple[str, str]


@dataclass(frozen=True)
class ValidationConfig:
    source_root: Path
    source_repo_id: str
    target_root: Path
    left_repo_id: str
    right_repo_id: str
    video_backend: str
    expect_task_copy: bool


@dataclass(frozen=True)
class ValidationResult:
    repo_id: str
    arm: str
    total_episodes: int
    total_frames: int
    episode_lengths: list[int]
    lag0_mean: float
    lag1_mean: float
    lag1_max: float


def parse_args() -> ValidationConfig:
    parser = argparse.ArgumentParser(description="Validate left/right split datasets against the source dataset.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--source-repo-id", default=DEFAULT_SOURCE_REPO_ID)
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--left-repo-id", default=DEFAULT_LEFT_REPO_ID)
    parser.add_argument("--right-repo-id", default=DEFAULT_RIGHT_REPO_ID)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--skip-task-copy-check", action="store_true")
    args = parser.parse_args()
    return ValidationConfig(
        source_root=args.source_root,
        source_repo_id=args.source_repo_id,
        target_root=args.target_root,
        left_repo_id=args.left_repo_id,
        right_repo_id=args.right_repo_id,
        video_backend=args.video_backend,
        expect_task_copy=not args.skip_task_copy_check,
    )


def load_info(dataset_dir: Path) -> dict[str, Any]:
    return json.loads((dataset_dir / "meta" / "info.json").read_text())


def load_episode_rows(dataset_dir: Path) -> pd.DataFrame:
    tables = [pq.read_table(path).to_pandas() for path in sorted((dataset_dir / "meta" / "episodes").rglob("*.parquet"))]
    if not tables:
        raise FileNotFoundError(f"No episode metadata found under {dataset_dir / 'meta' / 'episodes'}")
    return pd.concat(tables, ignore_index=True).sort_values("episode_index").reset_index(drop=True)


def load_task_map(dataset_dir: Path) -> dict[int, str]:
    tasks = pq.read_table(dataset_dir / "meta" / "tasks.parquet").to_pandas()
    return {int(row["task_index"]): str(index) for index, row in tasks.iterrows()}


def read_data_arrays(dataset_dir: Path) -> dict[str, np.ndarray]:
    tables = [
        pq.read_table(path, columns=[ACTION_KEY, STATE_KEY, "episode_index", "frame_index", "timestamp", "task_index"])
        for path in sorted((dataset_dir / "data").rglob("*.parquet"))
    ]
    if not tables:
        raise FileNotFoundError(f"No frame parquet files found under {dataset_dir / 'data'}")
    return {
        ACTION_KEY: np.concatenate([np.stack(table[ACTION_KEY].to_pylist()) for table in tables], axis=0),
        STATE_KEY: np.concatenate([np.stack(table[STATE_KEY].to_pylist()) for table in tables], axis=0),
        "episode_index": np.concatenate(
            [np.asarray(table["episode_index"].to_pylist(), dtype=np.int64) for table in tables], axis=0
        ),
        "frame_index": np.concatenate(
            [np.asarray(table["frame_index"].to_pylist(), dtype=np.int64) for table in tables], axis=0
        ),
        "timestamp": np.concatenate(
            [np.asarray(table["timestamp"].to_pylist(), dtype=np.float32) for table in tables], axis=0
        ),
        "task_index": np.concatenate(
            [np.asarray(table["task_index"].to_pylist(), dtype=np.int64) for table in tables], axis=0
        ),
    }


def _frame_task_texts(dataset_dir: Path, task_indices: np.ndarray) -> list[str]:
    task_map = load_task_map(dataset_dir)
    return [task_map[int(task_index)] for task_index in task_indices.tolist()]


def _compute_lag_metrics(
    action: np.ndarray,
    state: np.ndarray,
    episode_index: np.ndarray,
    frame_index: np.ndarray,
) -> dict[str, float]:
    if len(action) != len(state):
        raise ValueError("Action/state lengths must match.")
    if len(action) < 2:
        raise ValueError("At least two frames are required to evaluate lag alignment.")

    same_next = (episode_index[:-1] == episode_index[1:]) & (frame_index[1:] == frame_index[:-1] + 1)
    if not np.any(same_next):
        raise ValueError("No valid consecutive frames found for lag+1 alignment checks.")

    lag0 = np.abs(action - state).reshape(-1)
    lag1 = np.abs(action[:-1][same_next] - state[1:][same_next]).reshape(-1)
    lagm1 = np.abs(action[1:][same_next] - state[:-1][same_next]).reshape(-1)
    return {
        "lag0_mean": float(lag0.mean()),
        "lag0_max": float(lag0.max()),
        "lag1_mean": float(lag1.mean()),
        "lag1_max": float(lag1.max()),
        "lagm1_mean": float(lagm1.mean()),
        "lagm1_max": float(lagm1.max()),
    }


def assert_dataset_loads(repo_id: str, root: Path, video_backend: str) -> None:
    dataset = LeRobotDataset(repo_id, root=root, episodes=[0], video_backend=video_backend)
    first_item = dataset[0]
    if ACTION_KEY not in first_item or STATE_KEY not in first_item:
        raise ValueError(f"{repo_id} failed LeRobotDataset load check")


def validate_dataset(
    *,
    source_dir: Path,
    target_dir: Path,
    spec: DatasetSpec | None = None,
    arm: str | None = None,
    video_backend: str = "pyav",
    expect_task_copy: bool = True,
) -> ValidationResult:
    if spec is None:
        if arm == "left":
            spec = DatasetSpec(
                side="left",
                repo_id=target_dir.name,
                vector_slice=slice(0, 7),
                camera_keys=(COMMON_CAMERA_KEY, LEFT_CAMERA_KEY),
            )
        elif arm == "right":
            spec = DatasetSpec(
                side="right",
                repo_id=target_dir.name,
                vector_slice=slice(7, 14),
                camera_keys=(COMMON_CAMERA_KEY, RIGHT_CAMERA_KEY),
            )
        else:
            raise ValueError("Either spec or arm must be provided.")

    source_info = load_info(source_dir)
    target_info = load_info(target_dir)
    source_episode_rows = load_episode_rows(source_dir)
    target_episode_rows = load_episode_rows(target_dir)
    source_arrays = read_data_arrays(source_dir)
    target_arrays = read_data_arrays(target_dir)

    actual_image_keys = {key for key in target_info["features"] if key.startswith("observation.images.")}
    expected_image_keys = set(spec.camera_keys)

    if target_info["features"][ACTION_KEY]["shape"] != [7]:
        raise ValueError(f"{target_dir} action shape is not [7]")
    if target_info["features"][STATE_KEY]["shape"] != [7]:
        raise ValueError(f"{target_dir} observation.state shape is not [7]")
    if actual_image_keys != expected_image_keys:
        raise ValueError(f"{target_dir} image keys {sorted(actual_image_keys)} != {sorted(expected_image_keys)}")
    if int(target_info["total_episodes"]) != int(source_info["total_episodes"]):
        raise ValueError(f"{target_dir} total_episodes mismatch")
    if int(target_info["total_frames"]) != int(source_info["total_frames"]):
        raise ValueError(f"{target_dir} total_frames mismatch")

    target_episode_lengths = target_episode_rows["length"].to_numpy(dtype=np.int64)
    source_episode_lengths = source_episode_rows["length"].to_numpy(dtype=np.int64)
    if not np.array_equal(target_episode_lengths, source_episode_lengths):
        raise ValueError(f"{target_dir} episode lengths differ from source")

    expected_action = source_arrays[ACTION_KEY][:, spec.vector_slice]
    expected_state = source_arrays[STATE_KEY][:, spec.vector_slice]
    if not np.array_equal(target_arrays[ACTION_KEY], expected_action):
        raise ValueError(f"{target_dir} action values do not match source {spec.side} slice")
    if not np.array_equal(target_arrays[STATE_KEY], expected_state):
        raise ValueError(f"{target_dir} state values do not match source {spec.side} slice")
    if not np.allclose(target_arrays["timestamp"], source_arrays["timestamp"], atol=1e-6, rtol=0.0):
        raise ValueError(f"{target_dir} timestamps differ from source")

    if expect_task_copy:
        source_tasks = _frame_task_texts(source_dir, source_arrays["task_index"])
        target_tasks = _frame_task_texts(target_dir, target_arrays["task_index"])
        if target_tasks != source_tasks:
            raise ValueError(f"{target_dir} task text differs from source")

    lag_metrics = _compute_lag_metrics(
        target_arrays[ACTION_KEY],
        target_arrays[STATE_KEY],
        target_arrays["episode_index"],
        target_arrays["frame_index"],
    )
    if lag_metrics["lag1_max"] != 0.0:
        raise ValueError(f"{target_dir} does not satisfy action_t = state_(t+1)")

    assert_dataset_loads(repo_id=spec.repo_id, root=target_dir, video_backend=video_backend)
    dataset = LeRobotDataset(spec.repo_id, root=target_dir, episodes=[0], video_backend=video_backend)
    sample_image_keys = {key for key in dataset[0] if key.startswith("observation.images.")}
    if sample_image_keys != expected_image_keys:
        raise ValueError(f"{target_dir} sample image keys {sorted(sample_image_keys)} != {sorted(expected_image_keys)}")

    return ValidationResult(
        repo_id=spec.repo_id,
        arm=spec.side,
        total_episodes=int(target_info["total_episodes"]),
        total_frames=int(target_info["total_frames"]),
        episode_lengths=target_episode_lengths.tolist(),
        lag0_mean=float(lag_metrics["lag0_mean"]),
        lag1_mean=float(lag_metrics["lag1_mean"]),
        lag1_max=float(lag_metrics["lag1_max"]),
    )


def validate_split_dataset(
    source_root: Path,
    source_repo_id: str,
    target_root: Path,
    spec: DatasetSpec,
    *,
    video_backend: str = "pyav",
    expect_task_copy: bool = True,
) -> dict[str, Any]:
    source_dir = resolve_existing_dataset_root(source_root, source_repo_id)
    target_dir = resolve_existing_dataset_root(target_root, spec.repo_id)
    result = validate_dataset(
        source_dir=source_dir,
        target_dir=target_dir,
        spec=spec,
        video_backend=video_backend,
        expect_task_copy=expect_task_copy,
    )
    return {
        "repo_id": result.repo_id,
        "arm": result.arm,
        "num_episodes": result.total_episodes,
        "num_frames": result.total_frames,
        "total_episodes": result.total_episodes,
        "total_frames": result.total_frames,
        "episode_lengths": result.episode_lengths,
        "lag0_mean": result.lag0_mean,
        "lag1_mean": result.lag1_mean,
        "lag1_max": result.lag1_max,
    }


def validate_pair(config: ValidationConfig) -> tuple[ValidationResult, ValidationResult]:
    source_dir = resolve_existing_dataset_root(config.source_root, config.source_repo_id)
    left_dir = resolve_existing_dataset_root(config.target_root, config.left_repo_id)
    right_dir = resolve_existing_dataset_root(config.target_root, config.right_repo_id)
    left_result = validate_dataset(
        source_dir=source_dir,
        target_dir=left_dir,
        spec=DatasetSpec(
            side="left",
            repo_id=config.left_repo_id,
            vector_slice=slice(0, 7),
            camera_keys=(COMMON_CAMERA_KEY, LEFT_CAMERA_KEY),
        ),
        video_backend=config.video_backend,
        expect_task_copy=config.expect_task_copy,
    )
    right_result = validate_dataset(
        source_dir=source_dir,
        target_dir=right_dir,
        spec=DatasetSpec(
            side="right",
            repo_id=config.right_repo_id,
            vector_slice=slice(7, 14),
            camera_keys=(COMMON_CAMERA_KEY, RIGHT_CAMERA_KEY),
        ),
        video_backend=config.video_backend,
        expect_task_copy=config.expect_task_copy,
    )
    return left_result, right_result


def validate_split_outputs(
    *,
    source_root: Path,
    source_repo_id: str,
    target_root: Path,
    left_repo_id: str = DEFAULT_LEFT_REPO_ID,
    right_repo_id: str = DEFAULT_RIGHT_REPO_ID,
    video_backend: str = "pyav",
    expect_task_copy: bool = True,
) -> dict[str, dict[str, Any]]:
    config = ValidationConfig(
        source_root=source_root,
        source_repo_id=source_repo_id,
        target_root=target_root,
        left_repo_id=left_repo_id,
        right_repo_id=right_repo_id,
        video_backend=video_backend,
        expect_task_copy=expect_task_copy,
    )
    left_result, right_result = validate_pair(config)
    return {
        "left": {
            "repo_id": left_result.repo_id,
            "arm": left_result.arm,
            "num_episodes": left_result.total_episodes,
            "num_frames": left_result.total_frames,
            "total_episodes": left_result.total_episodes,
            "total_frames": left_result.total_frames,
            "episode_lengths": left_result.episode_lengths,
            "lag0_mean": left_result.lag0_mean,
            "lag1_mean": left_result.lag1_mean,
            "lag1_max": left_result.lag1_max,
            "camera_keys": [COMMON_CAMERA_KEY, LEFT_CAMERA_KEY],
        },
        "right": {
            "repo_id": right_result.repo_id,
            "arm": right_result.arm,
            "num_episodes": right_result.total_episodes,
            "num_frames": right_result.total_frames,
            "total_episodes": right_result.total_episodes,
            "total_frames": right_result.total_frames,
            "episode_lengths": right_result.episode_lengths,
            "lag0_mean": right_result.lag0_mean,
            "lag1_mean": right_result.lag1_mean,
            "lag1_max": right_result.lag1_max,
            "camera_keys": [COMMON_CAMERA_KEY, RIGHT_CAMERA_KEY],
        },
    }


def main() -> None:
    config = parse_args()
    left_result, right_result = validate_pair(config)
    for result in (left_result, right_result):
        print(
            f"[ok] {result.repo_id}: arm={result.arm} episodes={result.total_episodes} "
            f"frames={result.total_frames} lag0_mean={result.lag0_mean:.6f} "
            f"lag1_mean={result.lag1_mean:.6f} lag1_max={result.lag1_max:.6f}"
        )


if __name__ == "__main__":
    main()
