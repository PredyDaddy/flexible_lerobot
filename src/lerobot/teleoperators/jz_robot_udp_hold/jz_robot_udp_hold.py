#!/usr/bin/env python

from __future__ import annotations

from functools import cached_property
from typing import Any

from lerobot.processor.core import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_jz_robot_udp_hold import JZRobotUDPHoldTeleopConfig

LEFT = "left"
RIGHT = "right"
GRIPPER_WIDTH = "width"
GRIPPER_FORCE = "force"


class JZRobotUDPHoldTeleop(Teleoperator):
    """Builds a hold action by copying the current JZRobotUDP numeric observation."""

    config_class = JZRobotUDPHoldTeleopConfig
    name = "jz_robot_udp_hold"

    def __init__(self, config: JZRobotUDPHoldTeleopConfig):
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
        raise RuntimeError("JZRobotUDPHoldTeleop requires get_action_from_observation(observation)")

    def get_action_from_observation(self, observation: RobotObservation) -> RobotAction:
        action: RobotAction = {}
        missing = [key for key in self.action_features if key not in observation]
        if missing:
            raise RuntimeError(f"JZRobotUDPHoldTeleop observation is missing action keys: {missing}")

        for key in self.action_features:
            value = observation[key]
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise RuntimeError(f"JZRobotUDPHoldTeleop observation key {key!r} must be numeric")
            action[key] = float(value)
        return action

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        if feedback:
            raise ValueError("JZRobotUDPHoldTeleop does not accept feedback")

    @check_if_not_connected
    def disconnect(self) -> None:
        self._is_connected = False
