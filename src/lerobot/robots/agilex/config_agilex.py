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

from ..config import RobotConfig


@RobotConfig.register_subclass("agilex")
@dataclass(kw_only=True)
class AgileXRobotConfig(RobotConfig):
    control_mode: str = "passive_follow"
    state_left_topic: str = "/puppet/joint_left"
    state_right_topic: str = "/puppet/joint_right"
    command_left_topic: str = "/master/joint_left"
    command_right_topic: str = "/master/joint_right"
    front_camera_topic: str = "/camera_f/color/image_raw"
    left_camera_topic: str = "/camera_l/color/image_raw"
    right_camera_topic: str = "/camera_r/color/image_raw"
    front_camera_key: str = "cam_high"
    left_camera_key: str = "cam_left_wrist"
    right_camera_key: str = "cam_right_wrist"
    image_height: int = 480
    image_width: int = 640
    observation_timeout_s: float = 2.0
    queue_size: int = 1
    joint_names: list[str] = field(default_factory=lambda: [f"joint{i}" for i in range(7)])

    def __post_init__(self):
        super().__post_init__()
        if self.control_mode not in {"passive_follow", "command_master"}:
            raise ValueError(f"Unsupported control_mode: {self.control_mode}")
        if len(self.joint_names) != 7:
            raise ValueError("AgileX expects exactly 7 joint names per arm")
