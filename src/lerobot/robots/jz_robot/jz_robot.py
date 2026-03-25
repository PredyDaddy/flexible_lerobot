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
    from std_msgs.msg import Float64MultiArray
except Exception:
    rclpy = None
    SingleThreadedExecutor = None
    JointState = Any
    Float64MultiArray = Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_jz_robot import JZRobotConfig

logger = logging.getLogger(__name__)

LEFT = "left"
RIGHT = "right"
GRIPPER_WIDTH = "width"
GRIPPER_FORCE = "force"
GRIPPER_FIELDS = (GRIPPER_WIDTH, GRIPPER_FORCE)


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
        self._left_gripper_cmd_pub = None
        self._right_gripper_cmd_pub = None
        self._left_state_sub = None
        self._right_state_sub = None
        self._left_gripper_state_sub = None
        self._right_gripper_state_sub = None

        self._lock = threading.Lock()
        self._latest_joint_pos: dict[str, dict[str, float]] = {
            LEFT: {},
            RIGHT: {},
        }
        self._last_state_time: dict[str, float | None] = {
            LEFT: None,
            RIGHT: None,
        }
        self._latest_gripper_state: dict[str, dict[str, float]] = {
            LEFT: {},
            RIGHT: {},
        }
        self._last_gripper_state_time: dict[str, float | None] = {
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
    def _left_gripper_ft(self) -> dict[str, type]:
        if not self.config.use_gripper:
            return {}
        return {
            f"{LEFT}_gripper.{GRIPPER_WIDTH}": float,
            f"{LEFT}_gripper.{GRIPPER_FORCE}": float,
        }

    @property
    def _right_gripper_ft(self) -> dict[str, type]:
        if not self.config.use_gripper:
            return {}
        return {
            f"{RIGHT}_gripper.{GRIPPER_WIDTH}": float,
            f"{RIGHT}_gripper.{GRIPPER_FORCE}": float,
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {key: (cfg.height, cfg.width, 3) for key, cfg in self.config.cameras.items()}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {
            **self._left_motors_ft,
            **self._right_motors_ft,
            **self._left_gripper_ft,
            **self._right_gripper_ft,
            **self._cameras_ft,
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            **self._left_motors_ft,
            **self._right_motors_ft,
            **self._left_gripper_ft,
            **self._right_gripper_ft,
        }

    @property
    def is_connected(self) -> bool:
        ros_ready = self._node is not None and all(
            self._last_state_time[side] is not None for side in (LEFT, RIGHT)
        )
        if self.config.use_gripper:
            ros_ready = ros_ready and all(
                self._last_gripper_state_time[side] is not None for side in (LEFT, RIGHT)
            )
        cameras_ready = all(cam.is_connected for cam in self.cameras.values())
        return ros_ready and cameras_ready

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info("%s calibration is handled externally; skipping calibration.", self)

    def configure(self) -> None:
        if self.config.use_gripper:
            logger.info(
                "%s configured with ROS2 arm+gripper control. arm command topics: left=%s, right=%s; "
                "gripper command topics: left=%s, right=%s; gripper state topics: left=%s, right=%s; "
                "external=%s",
                self,
                self.config.left_position_command_topic,
                self.config.right_position_command_topic,
                self.config.left_gripper_command_topic,
                self.config.right_gripper_command_topic,
                self.config.left_gripper_state_topic,
                self.config.right_gripper_state_topic,
                self.config.use_external_commands,
            )
            return

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

    def _update_gripper_state(self, side: str, msg: Any) -> None:
        gripper_state = {}
        if len(msg.data) > 0:
            gripper_state[GRIPPER_WIDTH] = float(msg.data[0])
        if len(msg.data) > 1:
            gripper_state[GRIPPER_FORCE] = float(msg.data[1])
        if len(gripper_state) == 0:
            return

        with self._lock:
            self._latest_gripper_state[side].update(gripper_state)
            self._last_gripper_state_time[side] = time.monotonic()

    def _left_gripper_state_cb(self, msg: Any) -> None:
        self._update_gripper_state(LEFT, msg)

    def _right_gripper_state_cb(self, msg: Any) -> None:
        self._update_gripper_state(RIGHT, msg)

    def _required_joint_names(self, side: str) -> list[str]:
        if side == LEFT:
            return self.config.left_joint_names
        return self.config.right_joint_names

    def _state_topic(self, side: str) -> str:
        if side == LEFT:
            return self.config.left_joint_state_topic
        return self.config.right_joint_state_topic

    def _gripper_state_topic(self, side: str) -> str:
        if side == LEFT:
            return self.config.left_gripper_state_topic
        return self.config.right_gripper_state_topic

    def _missing_required_joints(self, side: str) -> list[str]:
        required = self._required_joint_names(side)
        with self._lock:
            latest = self._latest_joint_pos[side].copy()
        return [joint for joint in required if joint not in latest]

    def _missing_required_gripper_fields(self, side: str) -> list[str]:
        if not self.config.use_gripper:
            return []

        with self._lock:
            latest = self._latest_gripper_state[side].copy()
        return [field for field in GRIPPER_FIELDS if field not in latest]

    def _arm_state_ready(self, side: str) -> bool:
        return self._last_state_time[side] is not None and len(self._missing_required_joints(side)) == 0

    def _gripper_state_ready(self, side: str) -> bool:
        if not self.config.use_gripper:
            return True
        return self._last_gripper_state_time[side] is not None and len(self._missing_required_gripper_fields(side)) == 0

    def _initial_state_deadline(self) -> float | None:
        if self.config.init_state_timeout_s <= 0:
            return None
        return time.monotonic() + self.config.init_state_timeout_s

    def _format_initial_state_status(self) -> str:
        statuses: list[str] = []
        for side in (LEFT, RIGHT):
            missing_joints = self._missing_required_joints(side)
            if self._last_state_time[side] is None:
                arm_status = 'waiting_first_message'
            elif missing_joints:
                arm_status = f'missing_joints={missing_joints}'
            else:
                arm_status = 'ready'
            statuses.append(f'{side}_arm[{self._state_topic(side)}]={arm_status}')

        if self.config.use_gripper:
            for side in (LEFT, RIGHT):
                missing_fields = self._missing_required_gripper_fields(side)
                if self._last_gripper_state_time[side] is None:
                    gripper_status = 'waiting_first_message'
                elif missing_fields:
                    gripper_status = f'missing_fields={missing_fields}'
                else:
                    gripper_status = 'ready'
                statuses.append(
                    f'{side}_gripper[{self._gripper_state_topic(side)}]={gripper_status}'
                )

        return '; '.join(statuses)

    def _wait_for_initial_state(self) -> None:
        deadline = self._initial_state_deadline()

        while True:
            arms_ready = all(self._arm_state_ready(side) for side in (LEFT, RIGHT))
            gripper_ready = all(self._gripper_state_ready(side) for side in (LEFT, RIGHT))
            if arms_ready and gripper_ready:
                logger.info(
                    '%s initialization step 4/5 (robot state) complete: ROS2 state topics are ready. %s',
                    self,
                    self._format_initial_state_status(),
                )
                return

            now = time.monotonic()
            if deadline is not None and now >= deadline:
                missing: dict[str, list[str]] = {
                    side: self._missing_required_joints(side)
                    for side in (LEFT, RIGHT)
                }
                if self.config.use_gripper:
                    missing.update(
                        {
                            f'{side}_gripper': self._missing_required_gripper_fields(side)
                            for side in (LEFT, RIGHT)
                        }
                    )
                raise TimeoutError(
                    'Timed out waiting for initial JZRobot state messages. '
                    f'Arm topics: left={self.config.left_joint_state_topic}, '
                    f'right={self.config.right_joint_state_topic}. '
                    f'Missing fields: {missing}. Status: {self._format_initial_state_status()}'
                )

            time.sleep(0.01)

    def _assert_ros_ready(self) -> None:
        if rclpy is None or SingleThreadedExecutor is None or JointState is Any:
            raise ImportError(
                "JZRobot requires ROS2 Python dependencies (`rclpy`, `sensor_msgs`). "
                "Please run it in a ROS2-enabled environment."
            )
        if self.config.use_gripper and Float64MultiArray is Any:
            raise ImportError(
                "JZRobot gripper support requires ROS2 Python dependencies for `std_msgs`. "
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

    def _gripper_stale(self, side: str) -> bool:
        last_state_time = self._last_gripper_state_time[side]
        return last_state_time is None or (time.monotonic() - last_state_time > self.config.state_timeout_s)

    def _get_present_gripper_state(self, side: str) -> dict[str, float]:
        if not self.config.use_gripper:
            return {}
        if self._gripper_stale(side):
            raise TimeoutError(f"{side} gripper state is stale on topic {self._gripper_state_topic(side)}")

        with self._lock:
            gripper_state = self._latest_gripper_state[side].copy()

        missing = [field for field in GRIPPER_FIELDS if field not in gripper_state]
        if missing:
            raise RuntimeError(
                f"Missing required fields in latest {side} gripper state from "
                f"{self._gripper_state_topic(side)}: {missing}"
            )

        return {field: gripper_state[field] for field in GRIPPER_FIELDS}

    def _build_action_dict(
        self,
        left_goal: dict[str, float],
        right_goal: dict[str, float],
        left_gripper_goal: dict[str, float],
        right_gripper_goal: dict[str, float],
    ) -> RobotAction:
        action: RobotAction = {
            **{f"left_{joint}.pos": left_goal[joint] for joint in self.config.left_joint_names},
            **{f"right_{joint}.pos": right_goal[joint] for joint in self.config.right_joint_names},
        }
        if self.config.use_gripper:
            action.update(
                {
                    f"{LEFT}_gripper.{GRIPPER_WIDTH}": left_gripper_goal[GRIPPER_WIDTH],
                    f"{LEFT}_gripper.{GRIPPER_FORCE}": left_gripper_goal[GRIPPER_FORCE],
                    f"{RIGHT}_gripper.{GRIPPER_WIDTH}": right_gripper_goal[GRIPPER_WIDTH],
                    f"{RIGHT}_gripper.{GRIPPER_FORCE}": right_gripper_goal[GRIPPER_FORCE],
                }
            )
        return action

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
        self._left_gripper_cmd_pub = None
        self._right_gripper_cmd_pub = None
        self._left_state_sub = None
        self._right_state_sub = None
        self._left_gripper_state_sub = None
        self._right_gripper_state_sub = None

        with self._lock:
            self._latest_joint_pos = {
                LEFT: {},
                RIGHT: {},
            }
            self._last_state_time = {
                LEFT: None,
                RIGHT: None,
            }
            self._latest_gripper_state = {
                LEFT: {},
                RIGHT: {},
            }
            self._last_gripper_state_time = {
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
            if self.config.use_gripper:
                self._left_gripper_cmd_pub = self._node.create_publisher(
                    Float64MultiArray,
                    self.config.left_gripper_command_topic,
                    self.config.qos_depth,
                )
                self._right_gripper_cmd_pub = self._node.create_publisher(
                    Float64MultiArray,
                    self.config.right_gripper_command_topic,
                    self.config.qos_depth,
                )
                self._left_gripper_state_sub = self._node.create_subscription(
                    Float64MultiArray,
                    self.config.left_gripper_state_topic,
                    self._left_gripper_state_cb,
                    self.config.qos_depth,
                )
                self._right_gripper_state_sub = self._node.create_subscription(
                    Float64MultiArray,
                    self.config.right_gripper_state_topic,
                    self._right_gripper_state_cb,
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
        gripper_obs: RobotObservation = {}
        if self.config.use_gripper:
            left_gripper_state = self._get_present_gripper_state(LEFT)
            right_gripper_state = self._get_present_gripper_state(RIGHT)
            gripper_obs = {
                f"{LEFT}_gripper.{GRIPPER_WIDTH}": left_gripper_state[GRIPPER_WIDTH],
                f"{LEFT}_gripper.{GRIPPER_FORCE}": left_gripper_state[GRIPPER_FORCE],
                f"{RIGHT}_gripper.{GRIPPER_WIDTH}": right_gripper_state[GRIPPER_WIDTH],
                f"{RIGHT}_gripper.{GRIPPER_FORCE}": right_gripper_state[GRIPPER_FORCE],
            }

        obs: RobotObservation = {
            **{f"left_{joint}.pos": val for joint, val in left_joint_pos.items()},
            **{f"right_{joint}.pos": val for joint, val in right_joint_pos.items()},
            **gripper_obs,
        }

        for key, cam in self.cameras.items():
            obs[key] = cam.async_read()

        return obs

    def _parse_goal_positions(
        self, action: RobotAction
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
        invalid_keys = set(action) - set(self.action_features)
        if invalid_keys:
            raise KeyError(f"Invalid JZRobot action keys: {sorted(invalid_keys)}")

        left_present = self._get_present_positions(LEFT)
        right_present = self._get_present_positions(RIGHT)
        left_gripper_present = self._get_present_gripper_state(LEFT) if self.config.use_gripper else {}
        right_gripper_present = self._get_present_gripper_state(RIGHT) if self.config.use_gripper else {}

        left_goal = left_present.copy()
        right_goal = right_present.copy()
        left_gripper_goal = left_gripper_present.copy()
        right_gripper_goal = right_gripper_present.copy()

        for key, value in action.items():
            if key.startswith("left_") and key.endswith(".pos"):
                joint = key.removeprefix("left_").removesuffix(".pos")
                left_goal[joint] = float(value)
            elif key.startswith("right_") and key.endswith(".pos"):
                joint = key.removeprefix("right_").removesuffix(".pos")
                right_goal[joint] = float(value)
            elif key == f"{LEFT}_gripper.{GRIPPER_WIDTH}":
                left_gripper_goal[GRIPPER_WIDTH] = float(value)
            elif key == f"{LEFT}_gripper.{GRIPPER_FORCE}":
                left_gripper_goal[GRIPPER_FORCE] = float(value)
            elif key == f"{RIGHT}_gripper.{GRIPPER_WIDTH}":
                right_gripper_goal[GRIPPER_WIDTH] = float(value)
            elif key == f"{RIGHT}_gripper.{GRIPPER_FORCE}":
                right_gripper_goal[GRIPPER_FORCE] = float(value)

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

        return left_goal, right_goal, left_gripper_goal, right_gripper_goal

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        left_goal, right_goal, left_gripper_goal, right_gripper_goal = self._parse_goal_positions(action)
        action_to_return = self._build_action_dict(
            left_goal,
            right_goal,
            left_gripper_goal,
            right_gripper_goal,
        )

        if self.config.use_external_commands:
            return action_to_return

        left_msg = JointState()
        left_msg.name = list(self.config.left_joint_names)
        left_msg.position = [left_goal[joint] for joint in self.config.left_joint_names]

        right_msg = JointState()
        right_msg.name = list(self.config.right_joint_names)
        right_msg.position = [right_goal[joint] for joint in self.config.right_joint_names]

        self._left_cmd_pub.publish(left_msg)
        self._right_cmd_pub.publish(right_msg)
        if self.config.use_gripper:
            left_gripper_msg = Float64MultiArray()
            left_gripper_msg.data = [
                left_gripper_goal[GRIPPER_WIDTH],
                left_gripper_goal[GRIPPER_FORCE],
            ]
            right_gripper_msg = Float64MultiArray()
            right_gripper_msg.data = [
                right_gripper_goal[GRIPPER_WIDTH],
                right_gripper_goal[GRIPPER_FORCE],
            ]
            self._left_gripper_cmd_pub.publish(left_gripper_msg)
            self._right_gripper_cmd_pub.publish(right_gripper_msg)

        return action_to_return

    @check_if_not_connected
    def disconnect(self) -> None:
        for cam in self.cameras.values():
            cam.disconnect()

        self._shutdown_ros()
        logger.info("%s disconnected.", self)
