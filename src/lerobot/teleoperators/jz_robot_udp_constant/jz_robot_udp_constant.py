#!/usr/bin/env python

from __future__ import annotations

from functools import cached_property
from typing import Any

from lerobot.processor.core import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_jz_robot_udp_constant import JZRobotUDPConstantTeleopConfig

LEFT = "left"
RIGHT = "right"
GRIPPER_WIDTH = "width"
GRIPPER_FORCE = "force"


class JZRobotUDPConstantTeleop(Teleoperator):
    """Constant zero-action teleoperator for JZRobotUDP Phase 2 dry-run record tests."""

    config_class = JZRobotUDPConstantTeleopConfig
    name = "jz_robot_udp_constant"

    def __init__(self, config: JZRobotUDPConstantTeleopConfig):
        super().__init__(config)
        self.config = config
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
        self._is_connected = True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        action: RobotAction = {}
        for joint in self.config.left_joint_names:
            action[f"{LEFT}_{joint}.pos"] = float(self.config.joint_value)
        for joint in self.config.right_joint_names:
            action[f"{RIGHT}_{joint}.pos"] = float(self.config.joint_value)
        if self.config.use_gripper:
            action[f"{LEFT}_gripper.{GRIPPER_WIDTH}"] = float(self.config.gripper_width)
            action[f"{LEFT}_gripper.{GRIPPER_FORCE}"] = float(self.config.gripper_force)
            action[f"{RIGHT}_gripper.{GRIPPER_WIDTH}"] = float(self.config.gripper_width)
            action[f"{RIGHT}_gripper.{GRIPPER_FORCE}"] = float(self.config.gripper_force)
        return action

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        if feedback:
            raise ValueError("JZRobotUDPConstantTeleop does not accept feedback")

    @check_if_not_connected
    def disconnect(self) -> None:
        self._is_connected = False
