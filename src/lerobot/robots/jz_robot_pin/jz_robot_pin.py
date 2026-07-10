#!/usr/bin/env python

from __future__ import annotations

import logging
import os

from lerobot.processor import RobotAction
from lerobot.robots.jz_robot_udp.jz_robot_udp import GRIPPER_FORCE, GRIPPER_WIDTH, LEFT, RIGHT, JZRobotUDP
from lerobot.robots.jz_robot_udp.protocol import COMMAND_MODE_ARMED
from lerobot.robots.jz_robot_udp.state_cache import CachedState
from lerobot.utils.decorators import check_if_not_connected

from .config_jz_robot_pin import JZRobotPinConfig

logger = logging.getLogger(__name__)


class JZRobotPin(JZRobotUDP):
    """JZ robot client for the pin replay stack.

    The transport is intentionally close to JZRobotUDP, but this class has its
    own LeRobot type, defaults, and armed/safety gates so current pin teleop and
    recording work can evolve without changing existing JZRobotUDP users.
    """

    config_class = JZRobotPinConfig
    name = "jz_robot_pin"

    def __init__(self, config: JZRobotPinConfig):
        super().__init__(config)
        self.config = config
        self._last_sent_action: RobotAction | None = None
        self._last_accepted_state_seq: int | None = None
        self._last_accepted_state_received_monotonic_s: float | None = None

    def calibrate(self) -> None:
        logger.info("%s calibration is handled on the robot/edge side.", self)

    def configure(self) -> None:
        logger.info(
            "%s configured pin UDP state=%s:%s command=%s:%s transport=%s execution=%s.",
            self,
            self.config.bind_ip,
            self.config.state_port,
            self.config.command_target_ip,
            self.config.command_target_port,
            self.config.send_action_transport,
            self.config.send_action_execution,
        )

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        self._assert_armed_gate()
        if self.config.send_action_execution == COMMAND_MODE_ARMED:
            self._fresh_allowed_state()
        float_action = self._validate_and_float_action(action)
        safe_action = self._apply_pin_safety(float_action)
        sent_action = super().send_action(safe_action)
        self._last_sent_action = dict(sent_action)
        return sent_action

    def _assert_armed_gate(self) -> None:
        if self.config.send_action_execution != COMMAND_MODE_ARMED:
            return
        if not self.config.require_armed_env:
            return
        if os.getenv(self.config.armed_env_var) == "1":
            return
        raise RuntimeError(
            f"Refusing armed JZRobotPin command: set {self.config.armed_env_var}=1 "
            "after checking the robot, workspace, and physical emergency stop."
        )

    def _apply_pin_safety(self, action: RobotAction) -> RobotAction:
        safe_action = dict(action)
        self._check_initial_delta(safe_action)
        self._check_step_delta(safe_action)
        self._apply_gripper_limits(safe_action)
        return safe_action

    def _joint_action_keys(self) -> list[str]:
        return [
            *[f"{LEFT}_{joint}.pos" for joint in self.config.left_joint_names],
            *[f"{RIGHT}_{joint}.pos" for joint in self.config.right_joint_names],
        ]

    def _fresh_allowed_state(self) -> CachedState:
        state = self._state_cache.latest()
        if state is None:
            raise TimeoutError("Cannot send JZRobotPin action without a robot state packet")
        age_s = self._state_cache.age_s(state)
        if age_s is None or age_s > self.config.state_timeout_s:
            raise TimeoutError(
                "Cannot send JZRobotPin action with stale robot state: "
                f"age_s={age_s}, timeout_s={self.config.state_timeout_s}"
            )

        self._assert_allowed_sender(state.sender)
        packet_robot = state.packet.get("robot")
        if packet_robot != self.config.command_robot:
            raise RuntimeError(
                f"JZRobotPin state packet robot={packet_robot!r} does not match "
                f"command_robot={self.config.command_robot!r}"
            )

        if state.received_monotonic_s != self._last_accepted_state_received_monotonic_s:
            seq = state.packet["seq"]
            if self._last_accepted_state_seq is not None and seq <= self._last_accepted_state_seq:
                elapsed_s = state.received_monotonic_s - (
                    self._last_accepted_state_received_monotonic_s or state.received_monotonic_s
                )
                if elapsed_s <= self.config.state_seq_reset_timeout_s:
                    raise TimeoutError(
                        "JZRobotPin state packet sequence did not advance: "
                        f"seq={seq}, last_seq={self._last_accepted_state_seq}"
                    )
                logger.warning(
                    "Accepting JZRobotPin state sequence reset after %.3fs: last_seq=%s seq=%s",
                    elapsed_s,
                    self._last_accepted_state_seq,
                    seq,
                )
            self._last_accepted_state_seq = seq
            self._last_accepted_state_received_monotonic_s = state.received_monotonic_s
        return state

    def _latest_state_action_baseline(self) -> RobotAction:
        state = self._fresh_allowed_state()
        packet = state.packet
        baseline: RobotAction = {
            **self._joint_observation(packet, LEFT, self.config.left_joint_names),
            **self._joint_observation(packet, RIGHT, self.config.right_joint_names),
        }
        if self.config.use_gripper:
            baseline.update(self._gripper_observation(packet, LEFT))
            baseline.update(self._gripper_observation(packet, RIGHT))
        return baseline

    def _check_initial_delta(self, action: RobotAction) -> None:
        if self._last_sent_action is not None or self.config.max_initial_joint_delta_rad <= 0:
            return
        baseline = self._latest_state_action_baseline()
        self._raise_if_joint_delta_exceeds(
            action=action,
            baseline=baseline,
            limit=self.config.max_initial_joint_delta_rad,
            label="initial",
        )

    def _check_step_delta(self, action: RobotAction) -> None:
        if self.config.max_joint_step_rad <= 0 or self._last_sent_action is None:
            return
        self._raise_if_joint_delta_exceeds(
            action=action,
            baseline=self._last_sent_action,
            limit=self.config.max_joint_step_rad,
            label="step",
        )

    def _raise_if_joint_delta_exceeds(
        self,
        *,
        action: RobotAction,
        baseline: RobotAction,
        limit: float,
        label: str,
    ) -> None:
        violations: dict[str, float] = {}
        missing = [key for key in self._joint_action_keys() if key not in baseline]
        if missing:
            raise RuntimeError(f"JZRobotPin {label} safety baseline is missing joint keys: {missing}")
        for key in self._joint_action_keys():
            delta = abs(float(action[key]) - float(baseline[key]))
            if delta > limit + 1e-12:
                violations[key] = delta
        if violations:
            raise ValueError(
                f"JZRobotPin {label} joint delta exceeds limit {limit}: {violations}"
            )

    def _apply_gripper_limits(self, action: RobotAction) -> None:
        if not self.config.use_gripper:
            return
        limit_specs = {
            GRIPPER_WIDTH: (self.config.gripper_width_min, self.config.gripper_width_max),
            GRIPPER_FORCE: (self.config.gripper_force_min, self.config.gripper_force_max),
        }
        for side in (LEFT, RIGHT):
            for field, (minimum, maximum) in limit_specs.items():
                key = f"{side}_gripper.{field}"
                if key not in action:
                    continue
                action[key] = self._clamp_optional(float(action[key]), minimum, maximum)

    @staticmethod
    def _clamp_optional(value: float, minimum: float | None, maximum: float | None) -> float:
        if minimum is not None:
            value = max(value, float(minimum))
        if maximum is not None:
            value = min(value, float(maximum))
        return value

    @check_if_not_connected
    def disconnect(self) -> None:
        super().disconnect()
        self._last_sent_action = None
        self._last_accepted_state_seq = None
        self._last_accepted_state_received_monotonic_s = None
