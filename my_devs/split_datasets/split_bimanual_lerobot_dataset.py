from __future__ import annotations

import argparse
import copy
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lerobot.datasets.dataset_tools import _copy_data_with_feature_changes, _copy_videos
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import write_stats

DEFAULT_SOURCE_ROOT = Path("/home/agilex/cqy/flexible_lerobot/my_devs/add_robot/agilex/outputs")
DEFAULT_TARGET_ROOT = Path("/home/agilex/cqy/flexible_lerobot/my_devs/split_datasets/outputs")
DEFAULT_SOURCE_REPO_ID = "dummy/replay_test4"
DEFAULT_LEFT_REPO_ID = "dummy/replay_test4_left"
DEFAULT_RIGHT_REPO_ID = "dummy/replay_test4_right"

ACTION_KEY = "action"
STATE_KEY = "observation.state"
FRONT_CAMERA_KEY = "observation.images.camera_front"
LEFT_CAMERA_KEY = "observation.images.camera_left"
RIGHT_CAMERA_KEY = "observation.images.camera_right"
COMMON_CAMERA_KEY = FRONT_CAMERA_KEY

TASK_MODE_COPY = "copy"
TASK_MODE_OVERRIDE = "override"
EXPECTED_ARM_DIM = 7
EXPECTED_BIMANUAL_DIM = 14


@dataclass(frozen=True)
class ArmSpec:
    arm_name: str
    action_slice: slice
    state_slice: slice
    image_keys: tuple[str, str]


@dataclass(frozen=True)
class SplitSpec:
    side: str
    start: int
    end: int
    camera_key: str
    repo_id: str


ARM_SPECS = {
    "left": ArmSpec(
        arm_name="left",
        action_slice=slice(0, 7),
        state_slice=slice(0, 7),
        image_keys=(FRONT_CAMERA_KEY, LEFT_CAMERA_KEY),
    ),
    "right": ArmSpec(
        arm_name="right",
        action_slice=slice(7, 14),
        state_slice=slice(7, 14),
        image_keys=(FRONT_CAMERA_KEY, RIGHT_CAMERA_KEY),
    ),
}


@dataclass(frozen=True)
class SplitConfig:
    source_root: Path
    source_repo_id: str
    target_root: Path
    left_repo_id: str
    right_repo_id: str
    overwrite: bool
    task_mode: str
    task_text: str | None
    left_task_text: str | None
    right_task_text: str | None
    episode_indices: list[int] | None
    vcodec: str


def parse_args() -> SplitConfig:
    parser = argparse.ArgumentParser(
        description="Split a bimanual LeRobot dataset into left/right single-arm datasets."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--source-repo-id", default=DEFAULT_SOURCE_REPO_ID)
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--left-repo-id", default=DEFAULT_LEFT_REPO_ID)
    parser.add_argument("--right-repo-id", default=DEFAULT_RIGHT_REPO_ID)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--task-mode", choices=(TASK_MODE_COPY, TASK_MODE_OVERRIDE), default=TASK_MODE_COPY)
    parser.add_argument("--task-text", default=None)
    parser.add_argument("--left-task-text", default=None)
    parser.add_argument("--right-task-text", default=None)
    parser.add_argument(
        "--episode-indices",
        default=None,
        help="Comma-separated episode indices to export. Default exports all episodes.",
    )
    parser.add_argument("--vcodec", default="h264")
    args = parser.parse_args()
    if args.task_mode == TASK_MODE_OVERRIDE and not any(
        value for value in [args.task_text, args.left_task_text, args.right_task_text]
    ):
        parser.error("Override mode requires --task-text and/or --left-task-text / --right-task-text.")
    return SplitConfig(
        source_root=args.source_root,
        source_repo_id=args.source_repo_id,
        target_root=args.target_root,
        left_repo_id=args.left_repo_id,
        right_repo_id=args.right_repo_id,
        overwrite=args.overwrite,
        task_mode=args.task_mode,
        task_text=args.task_text,
        left_task_text=args.left_task_text,
        right_task_text=args.right_task_text,
        episode_indices=parse_episode_indices(args.episode_indices),
        vcodec=args.vcodec,
    )


def parse_episode_indices(raw_value: str | None) -> list[int] | None:
    if raw_value is None or raw_value.strip() == "":
        return None
    return sorted({int(chunk.strip()) for chunk in raw_value.split(",") if chunk.strip()})


def repo_path(root: Path, repo_id: str) -> Path:
    return root / repo_id


def resolve_existing_dataset_root(root: Path, repo_id: str) -> Path:
    candidates = [root, repo_path(root, repo_id)]
    for candidate in candidates:
        if (candidate / "meta" / "info.json").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate dataset for repo_id={repo_id!r} under {root}. "
        f"Tried {candidates[0]} and {candidates[1]}."
    )


def remove_repo_if_needed(root: Path, repo_id: str, overwrite: bool) -> None:
    dataset_dir = repo_path(root, repo_id)
    if not dataset_dir.exists():
        return
    if not overwrite:
        raise FileExistsError(
            f"Target dataset already exists: {dataset_dir}. Use --overwrite to replace it."
        )
    shutil.rmtree(dataset_dir)


def _arm_spec_from_any(spec: SplitSpec | ArmSpec) -> ArmSpec:
    if isinstance(spec, ArmSpec):
        return spec
    return ArmSpec(
        arm_name=spec.side,
        action_slice=slice(spec.start, spec.end),
        state_slice=slice(spec.start, spec.end),
        image_keys=(COMMON_CAMERA_KEY, spec.camera_key),
    )


def _slice_feature_shape(shape: list[int] | tuple[int, ...], size: int) -> list[int] | tuple[int, ...]:
    dims = list(shape)
    if not dims:
        raise ValueError(f"Expected non-empty feature shape, got {shape!r}.")
    dims[0] = size
    return tuple(dims) if isinstance(shape, tuple) else dims


def _slice_feature_names(names: Any, value_slice: slice) -> Any:
    if isinstance(names, list):
        return names[value_slice]
    if isinstance(names, dict):
        sliced: dict[str, Any] = {}
        for key, value in names.items():
            sliced[key] = value[value_slice] if isinstance(value, list) else value
        return sliced
    return names


def build_target_features(source_info_or_features: dict[str, Any], spec: SplitSpec | ArmSpec) -> dict[str, dict[str, Any]]:
    source_features = source_info_or_features.get("features", source_info_or_features)
    arm_spec = _arm_spec_from_any(spec)
    target_features: dict[str, dict[str, Any]] = {}

    action_feature = copy.deepcopy(source_features[ACTION_KEY])
    action_feature["shape"] = _slice_feature_shape(action_feature["shape"], EXPECTED_ARM_DIM)
    action_feature["names"] = _slice_feature_names(action_feature.get("names"), arm_spec.action_slice)
    target_features[ACTION_KEY] = action_feature

    state_feature = copy.deepcopy(source_features[STATE_KEY])
    state_feature["shape"] = _slice_feature_shape(state_feature["shape"], EXPECTED_ARM_DIM)
    state_feature["names"] = _slice_feature_names(state_feature.get("names"), arm_spec.state_slice)
    target_features[STATE_KEY] = state_feature

    for image_key in arm_spec.image_keys:
        target_features[image_key] = copy.deepcopy(source_features[image_key])

    return target_features


def resolve_task_text(
    *,
    source_task_text: str,
    task_mode: str,
    global_task_text: str | None,
    arm_task_text: str | None,
) -> str:
    if task_mode == TASK_MODE_COPY:
        return source_task_text
    return arm_task_text or global_task_text or source_task_text


def _to_channel_last(image: Any) -> Any:
    if not hasattr(image, "shape") or len(image.shape) != 3:
        return image
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        if hasattr(image, "permute"):
            return image.permute(1, 2, 0)
        if hasattr(image, "transpose"):
            return image.transpose((1, 2, 0))
    return image


def build_output_frame(
    item: dict[str, Any],
    spec: SplitSpec | ArmSpec,
    *,
    task_text: str | None,
) -> dict[str, Any]:
    arm_spec = _arm_spec_from_any(spec)
    frame = {
        "task": item["task"] if task_text is None else task_text,
        ACTION_KEY: item[ACTION_KEY][arm_spec.action_slice],
        STATE_KEY: item[STATE_KEY][arm_spec.state_slice],
        arm_spec.image_keys[0]: _to_channel_last(item[arm_spec.image_keys[0]]),
        arm_spec.image_keys[1]: _to_channel_last(item[arm_spec.image_keys[1]]),
    }
    if "timestamp" in item:
        timestamp = item["timestamp"]
        frame["timestamp"] = float(timestamp.item() if hasattr(timestamp, "item") else timestamp)
    return frame


def _validate_source_dataset(source_dataset: LeRobotDataset) -> None:
    required_features = {
        ACTION_KEY,
        STATE_KEY,
        FRONT_CAMERA_KEY,
        LEFT_CAMERA_KEY,
        RIGHT_CAMERA_KEY,
    }
    missing = sorted(required_features - set(source_dataset.features))
    if missing:
        raise ValueError(f"Source dataset is missing required features: {missing}")

    for key in [ACTION_KEY, STATE_KEY]:
        shape = source_dataset.features[key]["shape"]
        if int(shape[0]) != EXPECTED_BIMANUAL_DIM:
            raise ValueError(f"Source {key} shape must start with {EXPECTED_BIMANUAL_DIM}, got {shape}")


def _save_episode_with_timestamps(dataset: LeRobotDataset, timestamps: list[float]) -> None:
    if len(timestamps) != dataset.episode_buffer["size"]:
        raise ValueError(
            f"Timestamp count {len(timestamps)} does not match buffered frames {dataset.episode_buffer['size']}."
        )
    dataset.episode_buffer["timestamp"] = list(timestamps)
    dataset.save_episode()


def _slice_stat_value(value: Any, value_slice: slice) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim > 0 and value.shape[0] == EXPECTED_BIMANUAL_DIM:
            return value[value_slice]
        return value.copy()
    if isinstance(value, list):
        if len(value) == EXPECTED_BIMANUAL_DIM:
            return value[value_slice]
        return list(value)
    if isinstance(value, tuple):
        if len(value) == EXPECTED_BIMANUAL_DIM:
            return list(value[value_slice])
        return list(value)
    return value


def _build_target_stats(source_stats: dict[str, dict[str, Any]] | None, arm_spec: ArmSpec) -> dict[str, dict[str, Any]] | None:
    if source_stats is None:
        return None

    vector_slices = {
        ACTION_KEY: arm_spec.action_slice,
        STATE_KEY: arm_spec.state_slice,
    }
    allowed_keys = set(vector_slices) | set(arm_spec.image_keys) | {
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }

    target_stats: dict[str, dict[str, Any]] = {}
    for key in allowed_keys:
        if key not in source_stats:
            continue
        target_stats[key] = {}
        for stat_name, stat_value in source_stats[key].items():
            if key in vector_slices:
                target_stats[key][stat_name] = _slice_stat_value(stat_value, vector_slices[key])
            elif isinstance(stat_value, np.ndarray):
                target_stats[key][stat_name] = stat_value.copy()
            else:
                target_stats[key][stat_name] = stat_value

    return target_stats


def _rewrite_target_stats_file(target_dir: Path, source_dataset: LeRobotDataset, arm_spec: ArmSpec) -> None:
    target_stats = _build_target_stats(source_dataset.meta.stats, arm_spec)
    if target_stats is not None:
        write_stats(target_stats, target_dir)


def _drop_prefix_columns(df: pd.DataFrame, prefixes: list[str]) -> pd.DataFrame:
    cols_to_drop = [column for column in df.columns if any(column.startswith(prefix) for prefix in prefixes)]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df


def _rewrite_episode_metadata_for_arm(target_dir: Path, arm_spec: ArmSpec) -> None:
    dropped_camera_key = RIGHT_CAMERA_KEY if arm_spec.arm_name == "left" else LEFT_CAMERA_KEY
    vector_prefixes = {
        f"stats/{ACTION_KEY}/": arm_spec.action_slice,
        f"stats/{STATE_KEY}/": arm_spec.state_slice,
    }
    drop_prefixes = [
        f"videos/{dropped_camera_key}/",
        f"stats/{dropped_camera_key}/",
    ]

    for path in sorted((target_dir / "meta" / "episodes").rglob("*.parquet")):
        df = pd.read_parquet(path)
        df = _drop_prefix_columns(df, drop_prefixes)

        for prefix, value_slice in vector_prefixes.items():
            for column in [col for col in df.columns if col.startswith(prefix)]:
                df[column] = df[column].apply(lambda value, value_slice=value_slice: _slice_stat_value(value, value_slice))

        df.to_parquet(path, index=False)


def _fast_copy_split_dataset(
    *,
    config: SplitConfig,
    source_dataset: LeRobotDataset,
) -> tuple[Path, Path]:
    dataset_specs = (
        ("left", config.left_repo_id, RIGHT_CAMERA_KEY),
        ("right", config.right_repo_id, LEFT_CAMERA_KEY),
    )

    output_paths: dict[str, Path] = {}
    for side, repo_id, dropped_camera_key in dataset_specs:
        arm_spec = ARM_SPECS[side]
        target_dir = repo_path(config.target_root, repo_id)
        target_features = build_target_features(source_dataset.meta.info, arm_spec)
        target_meta = LeRobotDatasetMetadata.create(
            repo_id=repo_id,
            fps=source_dataset.meta.fps,
            features=target_features,
            robot_type=source_dataset.meta.robot_type,
            root=target_dir,
            use_videos=True,
        )
        _copy_data_with_feature_changes(
            dataset=source_dataset,
            new_meta=target_meta,
            add_features={
                ACTION_KEY: (
                    lambda row, _episode_idx, _frame_in_ep, value_slice=arm_spec.action_slice: row[ACTION_KEY][
                        value_slice
                    ],
                    target_features[ACTION_KEY],
                ),
                STATE_KEY: (
                    lambda row, _episode_idx, _frame_in_ep, value_slice=arm_spec.state_slice: row[STATE_KEY][
                        value_slice
                    ],
                    target_features[STATE_KEY],
                ),
            },
            remove_features=[dropped_camera_key],
        )
        _copy_videos(source_dataset, target_meta, exclude_keys=[dropped_camera_key])
        _rewrite_target_stats_file(target_dir, source_dataset, arm_spec)
        _rewrite_episode_metadata_for_arm(target_dir, arm_spec)
        output_paths[side] = target_dir

    return output_paths["left"], output_paths["right"]


def split_dataset(config: SplitConfig) -> tuple[Path, Path]:
    source_dataset_dir = resolve_existing_dataset_root(config.source_root, config.source_repo_id)
    source_dataset = LeRobotDataset(
        config.source_repo_id,
        root=source_dataset_dir,
        episodes=config.episode_indices,
    )
    _validate_source_dataset(source_dataset)

    config.target_root.mkdir(parents=True, exist_ok=True)
    remove_repo_if_needed(config.target_root, config.left_repo_id, overwrite=config.overwrite)
    remove_repo_if_needed(config.target_root, config.right_repo_id, overwrite=config.overwrite)

    left_target_dir = repo_path(config.target_root, config.left_repo_id)
    right_target_dir = repo_path(config.target_root, config.right_repo_id)

    if config.task_mode == TASK_MODE_COPY and config.episode_indices is None:
        return _fast_copy_split_dataset(
            config=config,
            source_dataset=source_dataset,
        )

    use_videos = len(source_dataset.meta.video_keys) > 0
    source_fps = int(source_dataset.fps)
    source_info = source_dataset.meta.info
    robot_type = source_dataset.meta.robot_type

    left_dataset = LeRobotDataset.create(
        repo_id=config.left_repo_id,
        fps=source_fps,
        features=build_target_features(source_info, ARM_SPECS["left"]),
        root=left_target_dir,
        robot_type=robot_type,
        use_videos=use_videos,
        vcodec=config.vcodec,
    )
    right_dataset = LeRobotDataset.create(
        repo_id=config.right_repo_id,
        fps=source_fps,
        features=build_target_features(source_info, ARM_SPECS["right"]),
        root=right_target_dir,
        robot_type=robot_type,
        use_videos=use_videos,
        vcodec=config.vcodec,
    )

    previous_episode: int | None = None
    current_timestamps: list[float] = []
    wrote_frames = False
    try:
        for idx in range(len(source_dataset)):
            item = source_dataset[idx]
            episode_index = int(item["episode_index"])

            if previous_episode is None:
                previous_episode = episode_index
            elif episode_index != previous_episode:
                _save_episode_with_timestamps(left_dataset, current_timestamps)
                _save_episode_with_timestamps(right_dataset, current_timestamps)
                previous_episode = episode_index
                current_timestamps = []
                wrote_frames = False

            source_task_text = item["task"]
            left_task_text = resolve_task_text(
                source_task_text=source_task_text,
                task_mode=config.task_mode,
                global_task_text=config.task_text,
                arm_task_text=config.left_task_text,
            )
            right_task_text = resolve_task_text(
                source_task_text=source_task_text,
                task_mode=config.task_mode,
                global_task_text=config.task_text,
                arm_task_text=config.right_task_text,
            )

            left_frame = build_output_frame(item, ARM_SPECS["left"], task_text=left_task_text)
            right_frame = build_output_frame(item, ARM_SPECS["right"], task_text=right_task_text)

            timestamp = left_frame.pop("timestamp", None)
            right_frame.pop("timestamp", None)
            if timestamp is None:
                timestamp = len(current_timestamps) / float(source_fps)
            current_timestamps.append(float(timestamp))

            left_dataset.add_frame(left_frame)
            right_dataset.add_frame(right_frame)
            wrote_frames = True

        if wrote_frames:
            _save_episode_with_timestamps(left_dataset, current_timestamps)
            _save_episode_with_timestamps(right_dataset, current_timestamps)
    finally:
        left_dataset.finalize()
        right_dataset.finalize()

    return left_target_dir, right_target_dir


def split_bimanual_dataset(
    *,
    source_root: Path,
    source_repo_id: str,
    target_root: Path,
    left_repo_id: str = DEFAULT_LEFT_REPO_ID,
    right_repo_id: str = DEFAULT_RIGHT_REPO_ID,
    overwrite: bool = False,
    task_mode: str = TASK_MODE_COPY,
    task_text: str | None = None,
    left_task_text: str | None = None,
    right_task_text: str | None = None,
    episode_indices: list[int] | None = None,
    vcodec: str = "h264",
) -> tuple[Path, Path]:
    return split_dataset(
        SplitConfig(
            source_root=source_root,
            source_repo_id=source_repo_id,
            target_root=target_root,
            left_repo_id=left_repo_id,
            right_repo_id=right_repo_id,
            overwrite=overwrite,
            task_mode=task_mode,
            task_text=task_text,
            left_task_text=left_task_text,
            right_task_text=right_task_text,
            episode_indices=episode_indices,
            vcodec=vcodec,
        )
    )


def main() -> None:
    config = parse_args()
    left_path, right_path = split_dataset(config)
    print(f"[ok] wrote left dataset: {left_path}")
    print(f"[ok] wrote right dataset: {right_path}")


if __name__ == "__main__":
    main()
