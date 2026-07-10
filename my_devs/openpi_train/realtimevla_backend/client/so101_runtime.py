from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch


ACTION_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


def optional_float(value: str | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float):
        return value
    if str(value).strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


def parse_camera(value: str | int | Path) -> int | Path:
    if isinstance(value, int):
        return value
    value_str = str(value)
    return int(value_str) if value_str.isdecimal() else Path(value_str).expanduser()


def maybe_path(path_str: str | None) -> Path | None:
    return None if not path_str else Path(path_str).expanduser()


def image_to_hwc_uint8(value: Any) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape={image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(image)) <= 1.5 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8, copy=False)
    return np.ascontiguousarray(image)


def state_to_float32(value: Any) -> np.ndarray:
    state = np.asarray(value, dtype=np.float32)
    if state.shape != (6,):
        raise ValueError(f"Expected SO101 state shape=(6,), got shape={state.shape}")
    return np.ascontiguousarray(state)


def build_robot(robot_cfg):
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots import make_robot_from_config, so_follower  # noqa: F401
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    cameras = {
        "top": OpenCVCameraConfig(
            index_or_path=parse_camera(robot_cfg.top_cam),
            width=int(robot_cfg.img_width),
            height=int(robot_cfg.img_height),
            fps=int(robot_cfg.fps),
            fourcc=robot_cfg.top_cam_fourcc,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=parse_camera(robot_cfg.wrist_cam),
            width=int(robot_cfg.img_width),
            height=int(robot_cfg.img_height),
            fps=int(robot_cfg.fps),
            fourcc=robot_cfg.wrist_cam_fourcc,
        ),
    }
    config = SOFollowerRobotConfig(
        id=robot_cfg.robot_id,
        calibration_dir=maybe_path(robot_cfg.calib_dir),
        port=robot_cfg.robot_port,
        max_relative_target=optional_float(robot_cfg.max_relative_target),
        cameras=cameras,
    )
    return make_robot_from_config(config), config


def build_dataset_features(robot) -> tuple[dict, Any, Any]:
    from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features
    from lerobot.datasets.pipeline_features import create_initial_features
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


def build_observation(robot, robot_observation_processor, dataset_features: dict, task: str) -> tuple[dict, dict]:
    from lerobot.datasets.utils import build_dataset_frame
    from lerobot.utils.constants import OBS_STR

    raw_observation = robot.get_observation()
    processed_observation = robot_observation_processor(raw_observation)
    frame = build_dataset_frame(dataset_features, processed_observation, prefix=OBS_STR)
    observation = {
        "observation.images.top": image_to_hwc_uint8(frame["observation.images.top"]),
        "observation.images.wrist": image_to_hwc_uint8(frame["observation.images.wrist"]),
        "observation.state": state_to_float32(frame["observation.state"]),
        "prompt": task,
    }
    return observation, raw_observation


def action_to_robot_action(action: np.ndarray, dataset_features: dict) -> dict:
    from lerobot.policies.utils import make_robot_action

    action = np.asarray(action, dtype=np.float32)
    if action.shape != (6,):
        raise ValueError(f"Expected one SO101 action shape=(6,), got shape={action.shape}")
    return make_robot_action(torch.as_tensor(action, dtype=torch.float32).unsqueeze(0), dataset_features)


def patch_motor_bus_retries(robot: Any, retries: int) -> None:
    retries = max(int(retries), 0)
    if retries <= 0:
        print("[DIAG] Motor bus I/O uses native LeRobot retry behavior.")
        return

    bus = getattr(robot, "bus", None)
    if bus is None:
        print("[DIAG] Robot has no motor bus; skip motor I/O retry patch.")
        return

    original_read = bus.read
    original_sync_read = bus.sync_read
    original_write = bus.write
    original_sync_write = bus.sync_write

    def read_with_min_retries(data_name: str, motor: str, *, normalize: bool = True, num_retry: int = 0) -> Any:
        return original_read(data_name, motor, normalize=normalize, num_retry=max(num_retry, retries))

    def sync_read_with_min_retries(
        data_name: str,
        motors: str | list[str] | None = None,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> Any:
        return original_sync_read(data_name, motors, normalize=normalize, num_retry=max(num_retry, retries))

    def write_with_min_retries(
        data_name: str,
        motor: str | None,
        value: int | float | list | tuple,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> Any:
        return original_write(data_name, motor, value, normalize=normalize, num_retry=max(num_retry, retries))

    def sync_write_with_min_retries(
        data_name: str,
        values: Any,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> Any:
        return original_sync_write(data_name, values, normalize=normalize, num_retry=max(num_retry, retries))

    bus.read = read_with_min_retries
    bus.sync_read = sync_read_with_min_retries
    bus.write = write_with_min_retries
    bus.sync_write = sync_write_with_min_retries
    print(f"[DIAG] Motor bus read/write calls will use at least {retries} retries.")


def validate_action_names(dataset_features: dict) -> None:
    action_names = dataset_features["action"]["names"]
    if action_names != list(ACTION_NAMES):
        raise ValueError(f"Unexpected action names: {action_names}. Expected {list(ACTION_NAMES)}")


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")
