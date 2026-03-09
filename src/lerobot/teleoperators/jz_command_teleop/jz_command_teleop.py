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

import logging
import threading
import time
from functools import cached_property
from typing import Any

try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import JointState
except Exception:
    rclpy = None
    SingleThreadedExecutor = None
    JointState = Any

from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_jz_command_teleop import JZCommandTeleopConfig

logger = logging.getLogger(__name__)

LEFT = "left"
RIGHT = "right"


class JZCommandTeleop(Teleoperator):
    """Read externally published ROS2 JointState commands and expose them as teleop actions."""

    config_class = JZCommandTeleopConfig
    name = "jz_command_teleop"

    def __init__(self, config: JZCommandTeleopConfig):
        super().__init__(config)
        self.config = config

        self._node = None
        self._executor = None
        self._spin_thread = None
        self._left_cmd_sub = None
        self._right_cmd_sub = None

        self._lock = threading.Lock()
        self._latest_joint_pos: dict[str, dict[str, float]] = {
            LEFT: {},
            RIGHT: {},
        }
        self._last_command_time: dict[str, float | None] = {
            LEFT: None,
            RIGHT: None,
        }

    @property
    def _left_motors_ft(self) -> dict[str, type]:
        return {f"left_{joint}.pos": float for joint in self.config.left_joint_names}

    @property
    def _right_motors_ft(self) -> dict[str, type]:
        return {f"right_{joint}.pos": float for joint in self.config.right_joint_names}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {**self._left_motors_ft, **self._right_motors_ft}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._node is not None and all(
            self._last_command_time[side] is not None for side in (LEFT, RIGHT)
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info("%s calibration is handled externally; skipping calibration.", self)

    def configure(self) -> None:
        logger.info(
            "%s listening to external joint command topics: left=%s, right=%s",
            self,
            self.config.left_command_topic,
            self.config.right_command_topic,
        )

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback

    def _required_joint_names(self, side: str) -> list[str]:
        if side == LEFT:
            return self.config.left_joint_names
        return self.config.right_joint_names

    def _topic(self, side: str) -> str:
        if side == LEFT:
            return self.config.left_command_topic
        return self.config.right_command_topic

    def _update_command(self, side: str, msg: Any) -> None:
        name_to_pos = {}
        for idx, name in enumerate(msg.name):
            if idx < len(msg.position):
                name_to_pos[name] = float(msg.position[idx])

        with self._lock:
            self._latest_joint_pos[side].update(name_to_pos)
            self._last_command_time[side] = time.monotonic()

    def _left_command_cb(self, msg: Any) -> None:
        self._update_command(LEFT, msg)

    def _right_command_cb(self, msg: Any) -> None:
        self._update_command(RIGHT, msg)

    def _missing_required_joints(self, side: str) -> list[str]:
        required = self._required_joint_names(side)
        with self._lock:
            latest = self._latest_joint_pos[side].copy()
        return [joint for joint in required if joint not in latest]

    def _wait_for_initial_commands(self) -> None:
        deadline = time.monotonic() + self.config.connect_timeout_s
        while time.monotonic() < deadline:
            if all(
                self._last_command_time[side] is not None and len(self._missing_required_joints(side)) == 0
                for side in (LEFT, RIGHT)
            ):
                return
            time.sleep(0.01)

        missing = {
            side: self._missing_required_joints(side)
            for side in (LEFT, RIGHT)
        }
        raise TimeoutError(
            "Timed out waiting for JointState commands on both arms. "
            f"Topics: left={self.config.left_command_topic}, right={self.config.right_command_topic}. "
            f"Missing joints: {missing}"
        )

    def _assert_ros_ready(self) -> None:
        if rclpy is None or SingleThreadedExecutor is None or JointState is Any:
            raise ImportError(
                "JZCommandTeleop requires ROS2 Python dependencies (`rclpy`, `sensor_msgs`). "
                "Please run it in a ROS2-enabled environment."
            )

    def _shutdown_ros(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()

        if self._spin_thread is not None and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=1.0)

        if self._node is not None:
            self._node.destroy_node()

        self._node = None
        self._executor = None
        self._spin_thread = None
        self._left_cmd_sub = None
        self._right_cmd_sub = None

        with self._lock:
            self._latest_joint_pos = {
                LEFT: {},
                RIGHT: {},
            }
            self._last_command_time = {
                LEFT: None,
                RIGHT: None,
            }

    def _command_is_stale(self, side: str) -> bool:
        if self.config.command_timeout_s is None:
            return False
        last_command_time = self._last_command_time[side]
        return last_command_time is None or (
            time.monotonic() - last_command_time > self.config.command_timeout_s
        )

    def _get_goal_positions(self, side: str) -> dict[str, float]:
        if self._command_is_stale(side):
            raise TimeoutError(f"{side} arm command is stale on topic {self._topic(side)}")

        with self._lock:
            joint_pos = self._latest_joint_pos[side].copy()

        missing = [joint for joint in self._required_joint_names(side) if joint not in joint_pos]
        if missing:
            raise RuntimeError(
                f"Missing required joints in latest {side} JointState command from {self._topic(side)}: {missing}"
            )

        return {joint: joint_pos[joint] for joint in self._required_joint_names(side)}

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self._assert_ros_ready()

        try:
            if not rclpy.ok():
                rclpy.init()

            self._node = rclpy.create_node(f"{self.name}_{self.id or 'default'}")
            self._left_cmd_sub = self._node.create_subscription(
                JointState,
                self.config.left_command_topic,
                self._left_command_cb,
                self.config.qos_depth,
            )
            self._right_cmd_sub = self._node.create_subscription(
                JointState,
                self.config.right_command_topic,
                self._right_command_cb,
                self.config.qos_depth,
            )

            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
            self._spin_thread.start()

            self._wait_for_initial_commands()
            self.configure()
            logger.info("%s connected.", self)
        except Exception:
            self._shutdown_ros()
            raise

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        left_goal = self._get_goal_positions(LEFT)
        right_goal = self._get_goal_positions(RIGHT)
        return {
            **{f"left_{joint}.pos": left_goal[joint] for joint in self.config.left_joint_names},
            **{f"right_{joint}.pos": right_goal[joint] for joint in self.config.right_joint_names},
        }

    def disconnect(self) -> None:
        if self._node is None:
            return
        self._shutdown_ros()
        logger.info("%s disconnected.", self)
