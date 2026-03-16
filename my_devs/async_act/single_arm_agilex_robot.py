from __future__ import annotations

from functools import cached_property

import numpy as np

from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.robots.agilex.agilex_ros_bridge import (
    ARM_PREFIXES,
    BridgeTopics,
    ImageTopicConfig,
    AgileXRosBridge,
)
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .common import ARM_TO_ACTION_NAMES, ARM_TO_LIVE_CAMERA_KEY, ARM_TO_STATE_NAMES
from .config_single_arm_agilex_robot import SingleArmAgileXRobotConfig


class SingleArmAgileXRobot(Robot):
    config_class = SingleArmAgileXRobotConfig
    name = "single_arm_agilex"

    def __init__(self, config: SingleArmAgileXRobotConfig):
        super().__init__(config)
        self.config = config
        self._last_arm_command: dict[str, float] | None = None
        self._bridge = AgileXRosBridge(
            topics=BridgeTopics(
                state_left_topic=config.state_left_topic,
                state_right_topic=config.state_right_topic,
                command_left_topic=config.command_left_topic if config.control_mode == "command_master" else None,
                command_right_topic=config.command_right_topic if config.control_mode == "command_master" else None,
                image_topics=(
                    ImageTopicConfig(config.front_camera_topic, config.front_camera_key),
                    ImageTopicConfig(
                        config.left_camera_topic if config.arm == "left" else config.right_camera_topic,
                        self._side_camera_key,
                    ),
                ),
            ),
            joint_names=config.joint_names,
            queue_size=config.queue_size,
        )

    @property
    def _side_camera_key(self) -> str:
        return self.config.left_camera_key if self.config.arm == "left" else self.config.right_camera_key

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        features = {key: float for key in ARM_TO_STATE_NAMES[self.config.arm]}
        features[self.config.front_camera_key] = (self.config.image_height, self.config.image_width, 3)
        features[self._side_camera_key] = (
            self.config.image_height,
            self.config.image_width,
            3,
        )
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {key: float for key in ARM_TO_ACTION_NAMES[self.config.arm]}

    @property
    def is_connected(self) -> bool:
        return self._bridge.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        self._bridge.connect(
            node_name="lerobot_agilex_single_arm_async",
            needs_publishers=self.config.control_mode == "command_master",
        )
        self._bridge.wait_for_ready(timeout_s=self.config.observation_timeout_s, require_images=True)
        self._last_arm_command = None
        self.configure()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        state = self._bridge.get_state_features()
        images = self._bridge.get_images()
        observation = {key: float(state[key]) for key in ARM_TO_STATE_NAMES[self.config.arm]}
        observation[self.config.front_camera_key] = images[self.config.front_camera_key]
        observation[self._side_camera_key] = images[self._side_camera_key]
        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        current_pose = self._bridge.get_action_features()
        arm_action = self._shape_arm_action(action, current_pose)
        full_action = dict(current_pose)
        for key, value in arm_action.items():
            full_action[key] = value

        inactive_arm = next(arm for arm in ARM_PREFIXES if arm != self.config.arm)
        inactive_keys = ARM_TO_ACTION_NAMES[inactive_arm]
        if not all(np.isclose(full_action[key], current_pose[key]) for key in inactive_keys):
            raise ValueError(f"Inactive arm command deviates from current pose: {inactive_arm}")

        if self.config.control_mode == "command_master":
            self._bridge.publish_action(full_action)
        return arm_action

    @check_if_not_connected
    def disconnect(self) -> None:
        self._last_arm_command = None
        self._bridge.disconnect()

    def _shape_arm_action(self, action: RobotAction, current_pose: dict[str, float]) -> RobotAction:
        keys = ARM_TO_ACTION_NAMES[self.config.arm]
        raw_vector = np.asarray([float(action[key]) for key in keys], dtype=np.float64)
        if self._last_arm_command is None:
            previous_vector = np.asarray([float(current_pose[key]) for key in keys], dtype=np.float64)
        else:
            previous_vector = np.asarray([self._last_arm_command[key] for key in keys], dtype=np.float64)

        shaped_vector = raw_vector
        alpha = self.config.action_smoothing_alpha
        if alpha is not None and alpha < 1.0:
            shaped_vector = previous_vector + alpha * (shaped_vector - previous_vector)

        max_step = self.config.max_joint_step_rad
        if max_step is not None:
            shaped_vector = previous_vector + np.clip(shaped_vector - previous_vector, -max_step, max_step)

        arm_action = {key: float(value) for key, value in zip(keys, shaped_vector, strict=True)}
        self._last_arm_command = arm_action
        return arm_action
