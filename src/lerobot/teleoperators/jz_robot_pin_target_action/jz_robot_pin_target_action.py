#!/usr/bin/env python

from __future__ import annotations

import copy
import logging
import time
from typing import Any

from lerobot.processor.core import RobotAction, RobotObservation
from lerobot.robots.jz_robot_udp.state_cache import CachedState
from lerobot.teleoperators.jz_robot_udp_target_action.jz_robot_udp_target_action import (
    LEFT,
    RIGHT,
    JZRobotUDPTargetActionTeleop,
)
from lerobot.utils.decorators import check_if_not_connected

from .config_jz_robot_pin_target_action import JZRobotPinTargetActionTeleopConfig

logger = logging.getLogger(__name__)


class JZRobotPinTargetActionTeleop(JZRobotUDPTargetActionTeleop):
    """Target-action teleop for pin VR/joystick packets.

    It keeps the strict target-action parser from JZRobotUDP, and adds an
    observation-aware hold-current fallback for LeRobot record/teleoperate loops.
    """

    config_class = JZRobotPinTargetActionTeleopConfig
    name = "jz_robot_pin_target_action"

    def __init__(self, config: JZRobotPinTargetActionTeleopConfig):
        super().__init__(config)
        self.config = config
        self._last_accepted_seq: int | None = None
        self._last_accepted_stamp_ns: int | None = None
        self._last_accepted_received_monotonic_s: float | None = None
        self._last_action_timing: dict[str, Any] | None = None

    @property
    def last_action_timing(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._last_action_timing)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        state = self._target_action_cache.latest()
        if state is None:
            raise TimeoutError("No JZRobot pin target action packet has been received")

        age_s = self._target_action_cache.age_s(state)
        if age_s is None or age_s > self.config.target_action_timeout_s:
            raise TimeoutError(
                "Latest JZRobot pin target action is stale: "
                f"age_s={age_s}, timeout_s={self.config.target_action_timeout_s}"
            )

        self._assert_allowed_sender(state.sender)
        self._validate_packet_freshness_and_order(state)
        packet = state.packet
        self._last_action_timing = {
            "source": "target_action_packet",
            "packet_seq": packet["seq"],
            "packet_stamp_ns": packet["stamp_ns"],
            "receive_wall_ns": state.received_wall_ns,
            "receive_monotonic_ns": int(state.received_monotonic_s * 1_000_000_000),
            "age_ms": max(0.0, float(age_s) * 1000),
        }
        return {
            **self._joint_action(packet, LEFT, self.config.left_joint_names),
            **self._joint_action(packet, RIGHT, self.config.right_joint_names),
            **self._gripper_action(packet, LEFT),
            **self._gripper_action(packet, RIGHT),
        }

    def _validate_packet_freshness_and_order(self, state: CachedState) -> None:
        packet = state.packet
        now_ns = time.time_ns()
        stamp_ns = packet["stamp_ns"]
        stamp_age_s = (now_ns - stamp_ns) / 1_000_000_000
        if stamp_age_s > self.config.packet_max_age_s:
            raise TimeoutError(
                "JZRobot pin target action packet stamp is stale: "
                f"age_s={stamp_age_s}, max_age_s={self.config.packet_max_age_s}"
            )
        if stamp_age_s < -self.config.packet_max_future_skew_s:
            raise TimeoutError(
                "JZRobot pin target action packet stamp is too far in the future: "
                f"age_s={stamp_age_s}, max_future_skew_s={self.config.packet_max_future_skew_s}"
            )

        if state.received_monotonic_s == self._last_accepted_received_monotonic_s:
            return

        seq = packet["seq"]
        if self._last_accepted_seq is not None and seq <= self._last_accepted_seq:
            elapsed_s = state.received_monotonic_s - (
                self._last_accepted_received_monotonic_s or state.received_monotonic_s
            )
            if elapsed_s <= self.config.seq_reset_timeout_s:
                raise TimeoutError(
                    "JZRobot pin target action packet sequence did not advance: "
                    f"seq={seq}, last_seq={self._last_accepted_seq}"
                )
            logger.warning(
                "Accepting JZRobot pin target action sequence reset after %.3fs: last_seq=%s seq=%s",
                elapsed_s,
                self._last_accepted_seq,
                seq,
            )
        elif self._last_accepted_stamp_ns is not None and stamp_ns <= self._last_accepted_stamp_ns:
            raise TimeoutError(
                "JZRobot pin target action packet timestamp did not advance: "
                f"stamp_ns={stamp_ns}, last_stamp_ns={self._last_accepted_stamp_ns}"
            )

        self._last_accepted_seq = seq
        self._last_accepted_stamp_ns = stamp_ns
        self._last_accepted_received_monotonic_s = state.received_monotonic_s

    def get_action_from_observation(self, observation: RobotObservation) -> RobotAction:
        try:
            return self.get_action()
        except TimeoutError:
            if self.config.stale_policy != "hold_current":
                raise
            self._last_action_timing = {
                "source": "hold_current",
                "packet_seq": None,
                "packet_stamp_ns": None,
                "receive_wall_ns": time.time_ns(),
                "receive_monotonic_ns": time.monotonic_ns(),
                "age_ms": None,
            }
            return self._hold_current_action(observation)

    def _hold_current_action(self, observation: RobotObservation) -> RobotAction:
        missing = [key for key in self.action_features if key not in observation]
        if missing:
            raise RuntimeError(f"JZRobotPinTargetActionTeleop observation is missing action keys: {missing}")

        action: RobotAction = {}
        for key in self.action_features:
            value = observation[key]
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise RuntimeError(f"JZRobotPinTargetActionTeleop observation key {key!r} must be numeric")
            action[key] = float(value)
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        return super().send_feedback(feedback)

    @check_if_not_connected
    def disconnect(self) -> None:
        super().disconnect()
        self._last_accepted_seq = None
        self._last_accepted_stamp_ns = None
        self._last_accepted_received_monotonic_s = None
        self._last_action_timing = None
