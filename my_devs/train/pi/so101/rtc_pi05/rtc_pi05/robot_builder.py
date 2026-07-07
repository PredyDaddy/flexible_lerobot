from __future__ import annotations

from pathlib import Path
from typing import Any

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.processor import make_default_processors
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

SUPPORTED_ROBOT_TYPES = {"so100_follower", "so101_follower"}


def maybe_path(path_str: str | None) -> Path | None:
    return None if not path_str else Path(path_str).expanduser()


def build_robot_config(
    *,
    robot_id: str,
    robot_type: str,
    calib_dir: str | None,
    robot_port: str,
    max_relative_target: float | None,
    top_cam: int | Path,
    wrist_cam: int | Path,
    top_cam_fourcc: str,
    wrist_cam_fourcc: str,
    img_width: int,
    img_height: int,
    fps: int,
) -> SOFollowerRobotConfig:
    if robot_type not in SUPPORTED_ROBOT_TYPES:
        raise ValueError(f"Unsupported robot_type={robot_type!r}; expected one of {sorted(SUPPORTED_ROBOT_TYPES)}")
    cameras = {
        "top": OpenCVCameraConfig(
            index_or_path=top_cam,
            width=img_width,
            height=img_height,
            fps=fps,
            fourcc=top_cam_fourcc,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=wrist_cam,
            width=img_width,
            height=img_height,
            fps=fps,
            fourcc=wrist_cam_fourcc,
        ),
    }
    return SOFollowerRobotConfig(
        id=robot_id,
        calibration_dir=maybe_path(calib_dir),
        port=robot_port,
        max_relative_target=max_relative_target,
        cameras=cameras,
    )


def build_dataset_artifacts(robot: Any) -> tuple[dict[str, dict[str, Any]], Any, Any]:
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
