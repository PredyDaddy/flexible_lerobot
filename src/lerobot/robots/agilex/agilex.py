#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import cached_property

from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .agilex_ros_bridge import (
    ACTION_FEATURE_NAMES,
    BridgeTopics,
    EFFORT_FEATURE_NAMES,
    ImageTopicConfig,
    POSITION_FEATURE_NAMES,
    VELOCITY_FEATURE_NAMES,
    AgileXRosBridge,
)
from .config_agilex import AgileXRobotConfig


class AgileXRobot(Robot):
    config_class = AgileXRobotConfig
    name = "agilex"

    def __init__(self, config: AgileXRobotConfig):
        super().__init__(config)
        self.config = config
        self.cameras = {
            config.front_camera_key: None,
            config.left_camera_key: None,
            config.right_camera_key: None,
        }
        self._bridge = AgileXRosBridge(
            topics=BridgeTopics(
                state_left_topic=config.state_left_topic,
                state_right_topic=config.state_right_topic,
                command_left_topic=config.command_left_topic if config.control_mode == "command_master" else None,
                command_right_topic=config.command_right_topic if config.control_mode == "command_master" else None,
                image_topics=(
                    ImageTopicConfig(config.front_camera_topic, config.front_camera_key),
                    ImageTopicConfig(config.left_camera_topic, config.left_camera_key),
                    ImageTopicConfig(config.right_camera_topic, config.right_camera_key),
                ),
            ),
            joint_names=config.joint_names,
            queue_size=config.queue_size,
        )

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        features: dict[str, type | tuple[int, int, int]] = {}
        for key in POSITION_FEATURE_NAMES + VELOCITY_FEATURE_NAMES + EFFORT_FEATURE_NAMES:
            features[key] = float
        for key in self.cameras:
            features[key] = (self.config.image_height, self.config.image_width, 3)
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {key: float for key in ACTION_FEATURE_NAMES}

    @property
    def is_connected(self) -> bool:
        return self._bridge.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self._bridge.connect(
            node_name="lerobot_agilex_robot",
            needs_publishers=self.config.control_mode == "command_master",
        )
        self._bridge.wait_for_ready(timeout_s=self.config.observation_timeout_s, require_images=True)
        if calibrate:
            self.calibrate()
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
        observation = self._bridge.get_state_features()
        observation.update(self._bridge.get_images())
        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        sent_action = {key: float(action[key]) for key in ACTION_FEATURE_NAMES}
        if self.config.control_mode == "command_master":
            self._bridge.publish_action(sent_action)
        return sent_action

    @check_if_not_connected
    def disconnect(self) -> None:
        self._bridge.disconnect()
