#!/usr/bin/env python

from __future__ import annotations

import time
from functools import cached_property
from typing import Any

from lerobot.processor.core import RobotAction
from lerobot.robots.jz_robot_udp.state_cache import StateCache
from lerobot.robots.jz_robot_udp.udp_client import UDPTargetActionReceiver
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_jz_robot_udp_target_action import JZRobotUDPTargetActionTeleopConfig

LEFT = "left"
RIGHT = "right"
GRIPPER_WIDTH = "width"
GRIPPER_FORCE = "force"
GRIPPER_FIELDS = (GRIPPER_WIDTH, GRIPPER_FORCE)


class JZRobotUDPTargetActionTeleop(Teleoperator):
    """Builds actions from Orin-observed target command packets, not feedback state."""

    config_class = JZRobotUDPTargetActionTeleopConfig
    name = "jz_robot_udp_target_action"

    def __init__(self, config: JZRobotUDPTargetActionTeleopConfig):
        super().__init__(config)
        self.config = config
        self._target_action_cache = StateCache()
        self._receiver = UDPTargetActionReceiver(
            bind_ip=config.bind_ip,
            port=config.target_action_port,
            cache=self._target_action_cache,
            buffer_size=config.receive_buffer_size,
        )
        self._is_connected = False

    @cached_property
    def action_features(self) -> dict[str, type]:
        features = {
            **{f"{LEFT}_{joint}.pos": float for joint in self.config.left_joint_names},
            **{f"{RIGHT}_{joint}.pos": float for joint in self.config.right_joint_names},
        }
        if self.config.use_gripper:
            features.update(
                {
                    f"{LEFT}_gripper.{GRIPPER_WIDTH}": float,
                    f"{LEFT}_gripper.{GRIPPER_FORCE}": float,
                    f"{RIGHT}_gripper.{GRIPPER_WIDTH}": float,
                    f"{RIGHT}_gripper.{GRIPPER_FORCE}": float,
                }
            )
        return features

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self._receiver.start()
        if self.config.connect_timeout_s > 0:
            started_after_s = time.monotonic()
            state = self._target_action_cache.wait_after(
                timeout_s=self.config.connect_timeout_s,
                after_monotonic_s=started_after_s,
            )
            if state is None:
                self._receiver.stop()
                raise TimeoutError(
                    "Timed out waiting for the first JZRobot UDP target action packet on "
                    f"{self.config.bind_ip}:{self.config.target_action_port}"
                )
        self._is_connected = True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        state = self._target_action_cache.latest()
        if state is None:
            raise TimeoutError("No JZRobot UDP target action packet has been received")

        age_s = self._target_action_cache.age_s(state)
        if age_s is None or age_s > self.config.target_action_timeout_s:
            raise TimeoutError(
                f"Latest JZRobot UDP target action is stale: "
                f"age_s={age_s}, timeout_s={self.config.target_action_timeout_s}"
            )

        self._assert_allowed_sender(state.sender)
        packet = state.packet
        return {
            **self._joint_action(packet, LEFT, self.config.left_joint_names),
            **self._joint_action(packet, RIGHT, self.config.right_joint_names),
            **self._gripper_action(packet, LEFT),
            **self._gripper_action(packet, RIGHT),
        }

    def _assert_allowed_sender(self, sender: tuple[str, int]) -> None:
        if self.config.allowed_sender_ip is None:
            return
        if sender[0] != self.config.allowed_sender_ip:
            raise RuntimeError(
                f"JZRobot UDP target action packet came from unexpected sender {sender[0]}:{sender[1]}, "
                f"expected {self.config.allowed_sender_ip}"
            )

    def _joint_action(self, packet: dict[str, Any], side: str, joint_names: list[str]) -> RobotAction:
        joint_values = packet["actions"][side]
        missing = [joint for joint in joint_names if joint not in joint_values]
        if missing:
            raise RuntimeError(f"Missing {side} joints in JZRobot UDP target action packet: {missing}")
        return {f"{side}_{joint}.pos": float(joint_values[joint]) for joint in joint_names}

    def _gripper_action(self, packet: dict[str, Any], side: str) -> RobotAction:
        if not self.config.use_gripper:
            return {}
        grippers = packet["actions"].get("grippers", {})
        gripper_values = grippers.get(side, {})
        missing = [field for field in GRIPPER_FIELDS if field not in gripper_values]
        if missing:
            raise RuntimeError(
                f"Missing {side} gripper fields in JZRobot UDP target action packet: {missing}"
            )
        return {
            f"{side}_gripper.{GRIPPER_WIDTH}": float(gripper_values[GRIPPER_WIDTH]),
            f"{side}_gripper.{GRIPPER_FORCE}": float(gripper_values[GRIPPER_FORCE]),
        }

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        if feedback:
            raise ValueError("JZRobotUDPTargetActionTeleop does not accept feedback")

    @check_if_not_connected
    def disconnect(self) -> None:
        self._receiver.stop()
        self._is_connected = False
