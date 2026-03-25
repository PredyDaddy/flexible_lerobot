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

from lerobot.cameras import CameraConfig

from ..config import RobotConfig

DEFAULT_LEFT_JOINT_NAMES = [
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    "left_joint7",
]

DEFAULT_RIGHT_JOINT_NAMES = [
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
    "right_joint7",
]

DEFAULT_LEFT_GRIPPER_STATE_TOPIC = "left_gripper/gripper_status"
DEFAULT_RIGHT_GRIPPER_STATE_TOPIC = "right_gripper/gripper_status"
DEFAULT_LEFT_GRIPPER_COMMAND_TOPIC = "left_gripper/gripper_commands"
DEFAULT_RIGHT_GRIPPER_COMMAND_TOPIC = "right_gripper/gripper_commands"


@RobotConfig.register_subclass("jz_robot")
@dataclass
class JZRobotConfig(RobotConfig):
    """Configuration for the ROS2-based bimanual JZRobot."""

    left_joint_names: list[str] = field(default_factory=lambda: DEFAULT_LEFT_JOINT_NAMES.copy())
    right_joint_names: list[str] = field(default_factory=lambda: DEFAULT_RIGHT_JOINT_NAMES.copy())

    left_joint_state_topic: str = "arm_left/joint_states"
    right_joint_state_topic: str = "arm_right/joint_states"

    left_position_command_topic: str = "telecon/arm_left/joint_commands_input"
    right_position_command_topic: str = "telecon/arm_right/joint_commands_input"

    use_gripper: bool = False
    left_gripper_state_topic: str = DEFAULT_LEFT_GRIPPER_STATE_TOPIC
    right_gripper_state_topic: str = DEFAULT_RIGHT_GRIPPER_STATE_TOPIC
    left_gripper_command_topic: str = DEFAULT_LEFT_GRIPPER_COMMAND_TOPIC
    right_gripper_command_topic: str = DEFAULT_RIGHT_GRIPPER_COMMAND_TOPIC

    # Wait timeout for initial state messages on startup. Set to 0 or a negative value to wait forever.
    init_state_timeout_s: float = 0.0
    # Max allowed age of the latest state message after startup before it is treated as stale.
    state_timeout_s: float = 0.2
    qos_depth: int = 10

    # Set to True when another system (for example a VR teleoperation stack) is already publishing
    # joint commands to the command topics. In that mode, `send_action()` only validates and echoes the
    # requested action so recording can proceed without fighting the external controller.
    use_external_commands: bool = False

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all joints, or a dictionary keyed by action
    # feature names, e.g. "left_shoulder_pan.pos" -> 5.0.
    max_relative_target: float | dict[str, float] | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()

        if len(self.left_joint_names) == 0:
            raise ValueError("left_joint_names must not be empty")
        if len(self.right_joint_names) == 0:
            raise ValueError("right_joint_names must not be empty")
        if self.use_gripper:
            if not self.left_gripper_state_topic:
                raise ValueError("left_gripper_state_topic must not be empty when use_gripper is true")
            if not self.right_gripper_state_topic:
                raise ValueError("right_gripper_state_topic must not be empty when use_gripper is true")
            if not self.left_gripper_command_topic:
                raise ValueError("left_gripper_command_topic must not be empty when use_gripper is true")
            if not self.right_gripper_command_topic:
                raise ValueError("right_gripper_command_topic must not be empty when use_gripper is true")
