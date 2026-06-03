#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import maybe_path

SUPPORTED_ROBOT_TYPES = {"so100_follower", "so101_follower"}
EXPECTED_SO_FOLLOWER_JOINTS = {
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
}


@dataclass
class RobotRuntimeConfig:
    robot_id: str
    robot_type: str
    calib_dir: str
    robot_port: str
    top_cam: int | Path
    wrist_cam: int | Path
    top_cam_fourcc: str
    wrist_cam_fourcc: str
    img_width: int
    img_height: int
    fps: int
    max_relative_target: float | dict[str, float] | None = None


def _validate_device_path(name: str, path_or_index: int | Path) -> None:
    if isinstance(path_or_index, int):
        if path_or_index < 0:
            raise ValueError(f"{name} camera index must be non-negative, got {path_or_index}")
        return

    path = Path(path_or_index).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{name} camera device does not exist: {path}")


def validate_runtime_config(cfg: RobotRuntimeConfig) -> None:
    if cfg.robot_type not in SUPPORTED_ROBOT_TYPES:
        raise ValueError(
            f"Unsupported robot_type={cfg.robot_type!r}. "
            f"Supported values: {sorted(SUPPORTED_ROBOT_TYPES)}."
        )

    calibration_file = Path(cfg.calib_dir).expanduser() / f"{cfg.robot_id}.json"
    if not calibration_file.is_file():
        raise FileNotFoundError(f"Robot calibration file does not exist: {calibration_file}")

    import json

    with calibration_file.open() as f:
        calibration = json.load(f)
    missing_joints = sorted(EXPECTED_SO_FOLLOWER_JOINTS - set(calibration))
    if missing_joints:
        raise ValueError(f"Robot calibration file is missing joints: {missing_joints}")

    robot_port = Path(cfg.robot_port).expanduser()
    if not robot_port.exists():
        raise FileNotFoundError(f"Robot serial port does not exist: {robot_port}")

    _validate_device_path("top", cfg.top_cam)
    _validate_device_path("wrist", cfg.wrist_cam)


def build_so_follower_config(cfg: RobotRuntimeConfig):
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    cameras = {
        "top": OpenCVCameraConfig(
            index_or_path=cfg.top_cam,
            width=cfg.img_width,
            height=cfg.img_height,
            fps=cfg.fps,
            fourcc=cfg.top_cam_fourcc,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=cfg.wrist_cam,
            width=cfg.img_width,
            height=cfg.img_height,
            fps=cfg.fps,
            fourcc=cfg.wrist_cam_fourcc,
        ),
    }
    return SOFollowerRobotConfig(
        id=cfg.robot_id,
        calibration_dir=maybe_path(cfg.calib_dir),
        port=cfg.robot_port,
        max_relative_target=cfg.max_relative_target,
        cameras=cameras,
    )


def make_dataset_features_and_processors(robot: Any):
    from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
    from lerobot.datasets.utils import combine_feature_dicts
    from lerobot.processor import make_default_processors

    _, robot_action_processor, robot_observation_processor = make_default_processors()
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=robot_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )
    return dataset_features, robot_action_processor, robot_observation_processor


def print_runtime_summary(*, repo_root: Path, robot_cfg: RobotRuntimeConfig, policy_path: Path, task: str) -> None:
    calibration_file = Path(robot_cfg.calib_dir).expanduser() / f"{robot_cfg.robot_id}.json"
    print(f"[INFO] Repo root: {repo_root}")
    print(f"[INFO] Robot id: {robot_cfg.robot_id}")
    print(f"[INFO] Robot type (requested): {robot_cfg.robot_type}")
    print(f"[INFO] Robot port: {robot_cfg.robot_port}")
    print(f"[INFO] Robot calibration file: {calibration_file}")
    print(f"[INFO] Top camera: {robot_cfg.top_cam} fourcc={robot_cfg.top_cam_fourcc}")
    print(f"[INFO] Wrist camera: {robot_cfg.wrist_cam} fourcc={robot_cfg.wrist_cam_fourcc}")
    print(f"[INFO] max_relative_target: {robot_cfg.max_relative_target}")
    print(f"[INFO] Policy path: {policy_path}")
    print(f"[INFO] Task: {task}")
    print(f"[INFO] FPS: {robot_cfg.fps}")
