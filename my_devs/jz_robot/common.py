#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import draccus

# Import for side effects so camera configs are registered before parsing YAML.
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.ros2_topic.configuration_ros2_topic import ROS2TopicCameraConfig  # noqa: F401
from lerobot.robots.config import RobotConfig
from lerobot.robots.jz_robot import JZRobotConfig  # noqa: F401 - imports for ChoiceRegistry registration
from lerobot.teleoperators.jz_command_teleop import JZCommandTeleopConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROBOT_CONFIG = REPO_ROOT / "src/lerobot/configs/robot/jz_robot_three_realsense_ros2_topics.yaml"


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def maybe_path(path_str: str | None) -> Path | None:
    return None if not path_str else Path(path_str).expanduser()


def load_robot_config(config_path: str | Path) -> RobotConfig:
    cfg_path = Path(config_path).expanduser().resolve()
    return draccus.parse(config_class=RobotConfig, config_path=cfg_path, args=[])


def _override_camera_topics(
    robot_cfg: JZRobotConfig,
    *,
    head_image_topic: str | None,
    left_image_topic: str | None,
    right_image_topic: str | None,
    camera_timeout_ms: int | None,
) -> None:
    topic_overrides = {
        "chest": head_image_topic,
        "left_arm": left_image_topic,
        "right_arm": right_image_topic,
    }

    for camera_key, topic in topic_overrides.items():
        if topic is None:
            continue
        camera_cfg = robot_cfg.cameras.get(camera_key)
        if camera_cfg is None or not hasattr(camera_cfg, "image_topic"):
            raise ValueError(
                f"Camera '{camera_key}' does not support ROS2 image topics in config {type(camera_cfg)}"
            )
        camera_cfg.image_topic = topic

    if camera_timeout_ms is not None:
        for camera_cfg in robot_cfg.cameras.values():
            if hasattr(camera_cfg, "timeout_ms"):
                camera_cfg.timeout_ms = camera_timeout_ms


def apply_common_robot_overrides(
    robot_cfg: RobotConfig,
    *,
    robot_id: str | None,
    left_joint_state_topic: str | None,
    right_joint_state_topic: str | None,
    left_command_topic: str | None,
    right_command_topic: str | None,
    use_gripper: bool | None,
    left_gripper_state_topic: str | None,
    right_gripper_state_topic: str | None,
    left_gripper_command_topic: str | None,
    right_gripper_command_topic: str | None,
    init_state_timeout_s: float | None,
    state_timeout_s: float | None,
    qos_depth: int | None,
    use_external_commands: bool | None,
    img_width: int | None,
    img_height: int | None,
    camera_fps: int | None,
    warmup_s: int | None,
    head_image_topic: str | None = None,
    left_image_topic: str | None = None,
    right_image_topic: str | None = None,
    camera_timeout_ms: int | None = None,
) -> RobotConfig:
    assert isinstance(robot_cfg, JZRobotConfig), f"Expected JZRobotConfig, got {type(robot_cfg)}"
    if robot_id:
        robot_cfg.id = robot_id
    if left_joint_state_topic:
        robot_cfg.left_joint_state_topic = left_joint_state_topic
    if right_joint_state_topic:
        robot_cfg.right_joint_state_topic = right_joint_state_topic
    if left_command_topic:
        robot_cfg.left_position_command_topic = left_command_topic
    if right_command_topic:
        robot_cfg.right_position_command_topic = right_command_topic
    if use_gripper is not None:
        robot_cfg.use_gripper = use_gripper
    if left_gripper_state_topic:
        robot_cfg.left_gripper_state_topic = left_gripper_state_topic
    if right_gripper_state_topic:
        robot_cfg.right_gripper_state_topic = right_gripper_state_topic
    if left_gripper_command_topic:
        robot_cfg.left_gripper_command_topic = left_gripper_command_topic
    if right_gripper_command_topic:
        robot_cfg.right_gripper_command_topic = right_gripper_command_topic
    if init_state_timeout_s is not None:
        robot_cfg.init_state_timeout_s = init_state_timeout_s
    if state_timeout_s is not None:
        robot_cfg.state_timeout_s = state_timeout_s
    if qos_depth is not None:
        robot_cfg.qos_depth = qos_depth
    if use_external_commands is not None:
        robot_cfg.use_external_commands = use_external_commands

    for camera_cfg in robot_cfg.cameras.values():
        if img_width is not None:
            camera_cfg.width = img_width
        if img_height is not None:
            camera_cfg.height = img_height
        if camera_fps is not None:
            camera_cfg.fps = camera_fps
        if warmup_s is not None and hasattr(camera_cfg, "warmup_s"):
            camera_cfg.warmup_s = warmup_s

    _override_camera_topics(
        robot_cfg,
        head_image_topic=head_image_topic,
        left_image_topic=left_image_topic,
        right_image_topic=right_image_topic,
        camera_timeout_ms=camera_timeout_ms,
    )

    return robot_cfg


def build_jz_command_teleop_config(
    robot_cfg: RobotConfig,
    *,
    teleop_id: str,
    connect_timeout_s: float,
    command_timeout_s: float | None,
) -> JZCommandTeleopConfig:
    assert isinstance(robot_cfg, JZRobotConfig), f"Expected JZRobotConfig, got {type(robot_cfg)}"
    return JZCommandTeleopConfig(
        id=teleop_id,
        left_joint_names=list(robot_cfg.left_joint_names),
        right_joint_names=list(robot_cfg.right_joint_names),
        left_command_topic=robot_cfg.left_position_command_topic,
        right_command_topic=robot_cfg.right_position_command_topic,
        use_gripper=robot_cfg.use_gripper,
        left_gripper_command_topic=robot_cfg.left_gripper_command_topic,
        right_gripper_command_topic=robot_cfg.right_gripper_command_topic,
        qos_depth=robot_cfg.qos_depth,
        connect_timeout_s=connect_timeout_s,
        command_timeout_s=command_timeout_s,
    )


def _camera_summary(key: str, cfg: Any) -> str:
    if hasattr(cfg, "image_topic"):
        return (
            f"  - {key}: topic={cfg.image_topic}, {cfg.width}x{cfg.height}@{cfg.fps}, "
            f"type={cfg.type}, timeout_ms={getattr(cfg, 'timeout_ms', 'n/a')}"
        )
    if hasattr(cfg, "serial_number_or_name"):
        return (
            f"  - {key}: serial={cfg.serial_number_or_name}, "
            f"{cfg.width}x{cfg.height}@{cfg.fps}, use_depth={getattr(cfg, 'use_depth', 'n/a')}"
        )
    return f"  - {key}: type={cfg.type}, {cfg.width}x{cfg.height}@{cfg.fps}"


def summarize_robot_config(robot_cfg: RobotConfig) -> str:
    assert isinstance(robot_cfg, JZRobotConfig), f"Expected JZRobotConfig, got {type(robot_cfg)}"
    lines = [
        f"robot_id={robot_cfg.id}",
        f"left_joint_state_topic={robot_cfg.left_joint_state_topic}",
        f"right_joint_state_topic={robot_cfg.right_joint_state_topic}",
        f"left_position_command_topic={robot_cfg.left_position_command_topic}",
        f"right_position_command_topic={robot_cfg.right_position_command_topic}",
        f"use_gripper={robot_cfg.use_gripper}",
        f"init_state_timeout_s={robot_cfg.init_state_timeout_s}",
        f"state_timeout_s={robot_cfg.state_timeout_s}",
        f"use_external_commands={robot_cfg.use_external_commands}",
        "cameras:",
    ]
    if robot_cfg.use_gripper:
        lines.extend(
            [
                f"left_gripper_state_topic={robot_cfg.left_gripper_state_topic}",
                f"right_gripper_state_topic={robot_cfg.right_gripper_state_topic}",
                f"left_gripper_command_topic={robot_cfg.left_gripper_command_topic}",
                f"right_gripper_command_topic={robot_cfg.right_gripper_command_topic}",
            ]
        )
    for key, cfg in robot_cfg.cameras.items():
        lines.append(_camera_summary(key, cfg))
    return chr(10).join(lines)
