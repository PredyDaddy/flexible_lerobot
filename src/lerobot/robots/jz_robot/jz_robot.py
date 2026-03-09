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

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_jz_robot import JZRobotConfig

logger = logging.getLogger(__name__)

LEFT = "left"
RIGHT = "right"


class JZRobot(Robot):
    """ROS2-based bimanual robot using JointState topics for state and command transport."""

    config_class = JZRobotConfig
    name = "jz_robot"

    def __init__(self, config: JZRobotConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)

        self._node = None
        self._executor = None
        self._spin_thread = None

        self._left_cmd_pub = None
        self._right_cmd_pub = None
        self._left_state_sub = None
        self._right_state_sub = None

        self._lock = threading.Lock()
        self._latest_joint_pos: dict[str, dict[str, float]] = {
            LEFT: {},
            RIGHT: {},
        }
        self._last_state_time: dict[str, float | None] = {
            LEFT: None,
            RIGHT: None,
        }

    @property
    def _left_motors_ft(self) -> dict[str, type]:
        return {f"left_{joint}.pos": float for joint in self.config.left_joint_names}

    @property
    def _right_motors_ft(self) -> dict[str, type]:
        return {f"right_{joint}.pos": float for joint in self.config.right_joint_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            key: (cfg.height, cfg.width, 3)
            for key, cfg in self.config.cameras.items()
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._left_motors_ft, **self._right_motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {**self._left_motors_ft, **self._right_motors_ft}

    @property
    def is_connected(self) -> bool:
        ros_ready = self._node is not None and all(
            self._last_state_time[side] is not None for side in (LEFT, RIGHT)
        )
        cameras_ready = all(cam.is_connected for cam in self.cameras.values())
        return ros_ready and cameras_ready

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info("%s calibration is handled externally; skipping calibration.", self)

    def configure(self) -> None:
        logger.info(
            "%s configured with ROS2 JointState control. command topics: left=%s, right=%s, external=%s",
            self,
            self.config.left_position_command_topic,
            self.config.right_position_command_topic,
            self.config.use_external_commands,
        )

    def _update_joint_state(self, side: str, msg: Any) -> None:
        name_to_pos = {}
        for idx, name in enumerate(msg.name):
            if idx < len(msg.position):
                name_to_pos[name] = float(msg.position[idx])

        with self._lock:
            self._latest_joint_pos[side].update(name_to_pos)
            self._last_state_time[side] = time.monotonic()

    def _left_state_cb(self, msg: Any) -> None:
        self._update_joint_state(LEFT, msg)

    def _right_state_cb(self, msg: Any) -> None:
        self._update_joint_state(RIGHT, msg)

    def _required_joint_names(self, side: str) -> list[str]:
        if side == LEFT:
            return self.config.left_joint_names
        return self.config.right_joint_names

    def _state_topic(self, side: str) -> str:
        if side == LEFT:
            return self.config.left_joint_state_topic
        return self.config.right_joint_state_topic

    def _missing_required_joints(self, side: str) -> list[str]:
        required = self._required_joint_names(side)
        with self._lock:
            latest = self._latest_joint_pos[side].copy()
        return [joint for joint in required if joint not in latest]

    def _wait_for_initial_state(self) -> None:
        deadline = time.monotonic() + self.config.state_timeout_s
        while time.monotonic() < deadline:
            if all(
                self._last_state_time[side] is not None and len(self._missing_required_joints(side)) == 0
                for side in (LEFT, RIGHT)
            ):
                return
            time.sleep(0.01)

        missing = {
            side: self._missing_required_joints(side)
            for side in (LEFT, RIGHT)
        }
        raise TimeoutError(
            "Timed out waiting for JointState messages on both arms. "
            f"Topics: left={self.config.left_joint_state_topic}, right={self.config.right_joint_state_topic}. "
            f"Missing joints: {missing}"
        )

    def _assert_ros_ready(self) -> None:
        if rclpy is None or SingleThreadedExecutor is None or JointState is Any:
            raise ImportError(
                "JZRobot requires ROS2 Python dependencies (`rclpy`, `sensor_msgs`). "
                "Please run it in a ROS2-enabled environment."
            )

    def _stale(self, side: str) -> bool:
        last_state_time = self._last_state_time[side]
        return last_state_time is None or (time.monotonic() - last_state_time > self.config.state_timeout_s)

    def _get_present_positions(self, side: str) -> dict[str, float]:
        if self._stale(side):
            raise TimeoutError(f"{side} arm JointState is stale on topic {self._state_topic(side)}")

        with self._lock:
            joint_pos = self._latest_joint_pos[side].copy()

        missing = [joint for joint in self._required_joint_names(side) if joint not in joint_pos]
        if missing:
            raise RuntimeError(
                f"Missing required joints in latest {side} JointState from {self._state_topic(side)}: {missing}"
            )

        return {joint: joint_pos[joint] for joint in self._required_joint_names(side)}

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
        self._left_cmd_pub = None
        self._right_cmd_pub = None
        self._left_state_sub = None
        self._right_state_sub = None

        with self._lock:
            self._latest_joint_pos = {
                LEFT: {},
                RIGHT: {},
            }
            self._last_state_time = {
                LEFT: None,
                RIGHT: None,
            }

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self._assert_ros_ready()

        try:
            if not rclpy.ok():
                rclpy.init()

            self._node = rclpy.create_node(f"{self.name}_{self.id or 'default'}")
            self._left_cmd_pub = self._node.create_publisher(
                JointState,
                self.config.left_position_command_topic,
                self.config.qos_depth,
            )
            self._right_cmd_pub = self._node.create_publisher(
                JointState,
                self.config.right_position_command_topic,
                self.config.qos_depth,
            )
            self._left_state_sub = self._node.create_subscription(
                JointState,
                self.config.left_joint_state_topic,
                self._left_state_cb,
                self.config.qos_depth,
            )
            self._right_state_sub = self._node.create_subscription(
                JointState,
                self.config.right_joint_state_topic,
                self._right_state_cb,
                self.config.qos_depth,
            )

            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
            self._spin_thread.start()

            self._wait_for_initial_state()

            for cam in self.cameras.values():
                cam.connect()

            if calibrate and not self.is_calibrated:
                self.calibrate()

            self.configure()
            logger.info("%s connected.", self)
        except Exception:
            for cam in self.cameras.values():
                if cam.is_connected:
                    cam.disconnect()
            self._shutdown_ros()
            raise

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        left_joint_pos = self._get_present_positions(LEFT)
        right_joint_pos = self._get_present_positions(RIGHT)

        obs: RobotObservation = {
            **{f"left_{joint}.pos": val for joint, val in left_joint_pos.items()},
            **{f"right_{joint}.pos": val for joint, val in right_joint_pos.items()},
        }

        for key, cam in self.cameras.items():
            obs[key] = cam.async_read()

        return obs

    def _parse_goal_positions(self, action: RobotAction) -> tuple[dict[str, float], dict[str, float]]:
        invalid_keys = set(action) - set(self.action_features)
        if invalid_keys:
            raise KeyError(f"Invalid JZRobot action keys: {sorted(invalid_keys)}")

        left_present = self._get_present_positions(LEFT)
        right_present = self._get_present_positions(RIGHT)

        left_goal = left_present.copy()
        right_goal = right_present.copy()

        for key, value in action.items():
            if key.startswith("left_") and key.endswith(".pos"):
                joint = key.removeprefix("left_").removesuffix(".pos")
                left_goal[joint] = float(value)
            elif key.startswith("right_") and key.endswith(".pos"):
                joint = key.removeprefix("right_").removesuffix(".pos")
                right_goal[joint] = float(value)

        if self.config.max_relative_target is not None:
            goal_present_pos = {
                **{
                    f"left_{joint}.pos": (left_goal[joint], left_present[joint])
                    for joint in self.config.left_joint_names
                },
                **{
                    f"right_{joint}.pos": (right_goal[joint], right_present[joint])
                    for joint in self.config.right_joint_names
                },
            }
            safe_goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)
            left_goal = {
                key.removeprefix("left_").removesuffix(".pos"): val
                for key, val in safe_goal_pos.items()
                if key.startswith("left_")
            }
            right_goal = {
                key.removeprefix("right_").removesuffix(".pos"): val
                for key, val in safe_goal_pos.items()
                if key.startswith("right_")
            }

        return left_goal, right_goal

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        left_goal, right_goal = self._parse_goal_positions(action)

        if self.config.use_external_commands:
            return {
                **{f"left_{joint}.pos": left_goal[joint] for joint in self.config.left_joint_names},
                **{f"right_{joint}.pos": right_goal[joint] for joint in self.config.right_joint_names},
            }

        left_msg = JointState()
        left_msg.name = list(self.config.left_joint_names)
        left_msg.position = [left_goal[joint] for joint in self.config.left_joint_names]

        right_msg = JointState()
        right_msg.name = list(self.config.right_joint_names)
        right_msg.position = [right_goal[joint] for joint in self.config.right_joint_names]

        self._left_cmd_pub.publish(left_msg)
        self._right_cmd_pub.publish(right_msg)

        return {
            **{f"left_{joint}.pos": left_goal[joint] for joint in self.config.left_joint_names},
            **{f"right_{joint}.pos": right_goal[joint] for joint in self.config.right_joint_names},
        }

    @check_if_not_connected
    def disconnect(self) -> None:
        for cam in self.cameras.values():
            cam.disconnect()

        self._shutdown_ros()
        logger.info("%s disconnected.", self)
