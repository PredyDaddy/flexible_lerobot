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
from typing import Any

from lerobot.processor import RobotAction
from lerobot.robots.agilex.agilex_ros_bridge import ACTION_FEATURE_NAMES, BridgeTopics, AgileXRosBridge
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_agilex_teleoperator import AgileXTeleoperatorConfig


class AgileXTeleoperator(Teleoperator):
    config_class = AgileXTeleoperatorConfig
    name = "agilex_teleoperator"

    def __init__(self, config: AgileXTeleoperatorConfig):
        super().__init__(config)
        self.config = config
        self._bridge = AgileXRosBridge(
            topics=BridgeTopics(
                state_left_topic=config.action_left_topic,
                state_right_topic=config.action_right_topic,
                command_left_topic=None,
                command_right_topic=None,
                image_topics=(),
            ),
            joint_names=config.joint_names,
            queue_size=config.queue_size,
        )

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {key: float for key in ACTION_FEATURE_NAMES}

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._bridge.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self._bridge.connect(node_name="lerobot_agilex_teleoperator", needs_publishers=False)
        self._bridge.wait_for_ready(timeout_s=self.config.observation_timeout_s, require_images=False)
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
    def get_action(self) -> RobotAction:
        return self._bridge.get_action_features()

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        return None

    @check_if_not_connected
    def disconnect(self) -> None:
        self._bridge.disconnect()
