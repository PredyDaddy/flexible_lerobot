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

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("agilex_teleoperator")
@dataclass(kw_only=True)
class AgileXTeleoperatorConfig(TeleoperatorConfig):
    action_left_topic: str = "/master/joint_left"
    action_right_topic: str = "/master/joint_right"
    observation_timeout_s: float = 2.0
    queue_size: int = 1
    joint_names: list[str] = field(default_factory=lambda: [f"joint{i}" for i in range(7)])

    def __post_init__(self):
        if len(self.joint_names) != 7:
            raise ValueError("AgileX teleoperator expects exactly 7 joint names per arm")
