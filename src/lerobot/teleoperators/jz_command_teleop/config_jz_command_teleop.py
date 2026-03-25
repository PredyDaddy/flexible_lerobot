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

from dataclasses import dataclass, field

from lerobot.robots.jz_robot.config_jz_robot import (
    DEFAULT_LEFT_GRIPPER_COMMAND_TOPIC,
    DEFAULT_LEFT_JOINT_NAMES,
    DEFAULT_RIGHT_GRIPPER_COMMAND_TOPIC,
    DEFAULT_RIGHT_JOINT_NAMES,
)

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("jz_command_teleop")
@dataclass
class JZCommandTeleopConfig(TeleoperatorConfig):
    """Teleoperator that mirrors externally published JZRobot joint command topics."""

    left_joint_names: list[str] = field(default_factory=lambda: DEFAULT_LEFT_JOINT_NAMES.copy())
    right_joint_names: list[str] = field(default_factory=lambda: DEFAULT_RIGHT_JOINT_NAMES.copy())

    left_command_topic: str = "telecon/arm_left/joint_commands_input"
    right_command_topic: str = "telecon/arm_right/joint_commands_input"
    use_gripper: bool = False
    left_gripper_command_topic: str = DEFAULT_LEFT_GRIPPER_COMMAND_TOPIC
    right_gripper_command_topic: str = DEFAULT_RIGHT_GRIPPER_COMMAND_TOPIC

    qos_depth: int = 10
    # Wait timeout for the first command on startup. Set to 0 or a negative value to wait forever.
    connect_timeout_s: float = 0.0
    command_timeout_s: float | None = None

    def __post_init__(self) -> None:
        if len(self.left_joint_names) == 0:
            raise ValueError("left_joint_names must not be empty")
        if len(self.right_joint_names) == 0:
            raise ValueError("right_joint_names must not be empty")
        if self.use_gripper:
            if not self.left_gripper_command_topic:
                raise ValueError("left_gripper_command_topic must not be empty when use_gripper is true")
            if not self.right_gripper_command_topic:
                raise ValueError("right_gripper_command_topic must not be empty when use_gripper is true")
