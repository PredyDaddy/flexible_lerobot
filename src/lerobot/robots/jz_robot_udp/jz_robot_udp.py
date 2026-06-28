#!/usr/bin/env python

from __future__ import annotations

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_jz_robot_udp import JZRobotUDPConfig
from .protocol import (
    encode_jz_robot_udp_command_packet,
    make_jz_robot_udp_command_packet,
)
from .rtsp_camera import RTSPCamera
from .state_cache import StateCache
from .udp_client import UDPCommandSender, UDPStateReceiver

logger = logging.getLogger(__name__)

LEFT = "left"
RIGHT = "right"
GRIPPER_WIDTH = "width"
GRIPPER_FORCE = "force"
GRIPPER_FIELDS = (GRIPPER_WIDTH, GRIPPER_FORCE)


class JZRobotUDP(Robot):
    """Readonly x86-side JZRobot interface backed by UDP state packets and RTSP/OpenCV cameras."""

    config_class = JZRobotUDPConfig
    name = "jz_robot_udp"

    def __init__(self, config: JZRobotUDPConfig):
        super().__init__(config)
        self.config = config
        self.cameras = {key: RTSPCamera(cfg) for key, cfg in config.rtsp_cameras.items()}
        self._state_cache = StateCache()
        self._receiver = UDPStateReceiver(
            bind_ip=config.bind_ip,
            port=config.state_port,
            cache=self._state_cache,
            buffer_size=config.receive_buffer_size,
        )
        self._command_sender: UDPCommandSender | None = None
        self._command_seq = 0
        self._is_connected = False

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
        return {key: (cfg.height, cfg.width, 3) for key, cfg in self.config.rtsp_cameras.items()}

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
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info("%s is readonly; calibration is handled on the Orin side.", self)

    def configure(self) -> None:
        logger.info(
            "%s configured readonly UDP state receiver on %s:%s.",
            self,
            self.config.bind_ip,
            self.config.state_port,
        )

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        try:
            self._receiver.start()
            start_monotonic_s = time.monotonic()
            state = self._state_cache.wait_after(
                timeout_s=self.config.connect_timeout_s,
                after_monotonic_s=start_monotonic_s,
            )
            if state is None:
                raise TimeoutError(
                    "Timed out waiting for the first JZRobot UDP state packet on "
                    f"{self.config.bind_ip}:{self.config.state_port}"
                )
            for camera in self.cameras.values():
                camera.connect()
            self._is_connected = True
            if calibrate and not self.is_calibrated:
                self.calibrate()
            self.configure()
        except Exception:
            for camera in self.cameras.values():
                if camera.is_connected:
                    camera.disconnect()
            self._receiver.stop()
            self._is_connected = False
            raise

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        state = self._state_cache.latest()
        if state is None:
            raise TimeoutError("No JZRobot UDP state packet has been received")

        age_s = self._state_cache.age_s(state)
        if age_s is None or age_s > self.config.state_timeout_s:
            raise TimeoutError(
                f"Latest JZRobot UDP state is stale: age_s={age_s}, timeout_s={self.config.state_timeout_s}"
            )

        packet = state.packet
        self._assert_allowed_sender(state.sender)
        obs: RobotObservation = {
            **self._joint_observation(packet, LEFT, self.config.left_joint_names),
            **self._joint_observation(packet, RIGHT, self.config.right_joint_names),
        }
        if self.config.use_gripper:
            obs.update(self._gripper_observation(packet, LEFT))
            obs.update(self._gripper_observation(packet, RIGHT))

        for key, camera in self.cameras.items():
            obs[key] = camera.async_read()

        return obs

    def _assert_allowed_sender(self, sender: tuple[str, int]) -> None:
        if self.config.allowed_sender_ip is None:
            return
        if sender[0] != self.config.allowed_sender_ip:
            raise RuntimeError(
                f"JZRobot UDP state packet came from unexpected sender {sender[0]}:{sender[1]}, "
                f"expected {self.config.allowed_sender_ip}"
            )

    def _joint_observation(self, packet: dict[str, Any], side: str, joint_names: list[str]) -> RobotObservation:
        joint_values = packet["joints"][side]
        missing = [joint for joint in joint_names if joint not in joint_values]
        if missing:
            raise RuntimeError(f"Missing {side} joints in JZRobot UDP state packet: {missing}")
        return {f"{side}_{joint}.pos": float(joint_values[joint]) for joint in joint_names}

    def _gripper_observation(self, packet: dict[str, Any], side: str) -> RobotObservation:
        grippers = packet.get("grippers", {})
        gripper_values = grippers.get(side, {})
        missing = [field for field in GRIPPER_FIELDS if field not in gripper_values]
        if missing:
            raise RuntimeError(f"Missing {side} gripper fields in JZRobot UDP state packet: {missing}")
        return {
            f"{side}_gripper.{GRIPPER_WIDTH}": float(gripper_values[GRIPPER_WIDTH]),
            f"{side}_gripper.{GRIPPER_FORCE}": float(gripper_values[GRIPPER_FORCE]),
        }

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        float_action = self._validate_and_float_action(action)
        self._command_seq += 1
        packet = make_jz_robot_udp_command_packet(
            robot=self.config.command_robot,
            seq=self._command_seq,
            stamp_ns=time.time_ns(),
            mode=self.config.send_action_execution,
            actions=self._command_actions(float_action),
        )
        encoded = encode_jz_robot_udp_command_packet(packet)

        if self.config.send_action_transport == "local":
            logger.info(
                "JZRobotUDP command seq=%s mode=%s transport=local robot=%s target=local "
                "action_key_count=%s action_keys=%s",
                packet["seq"],
                packet["mode"],
                packet["robot"],
                len(float_action),
                sorted(float_action),
            )
        elif self.config.send_action_transport == "udp":
            if self._command_sender is None:
                self._command_sender = UDPCommandSender(
                    target_ip=self.config.command_target_ip,
                    target_port=self.config.command_target_port,
                    timeout_s=self.config.command_timeout_s,
                )
            sent_bytes = self._command_sender.send(encoded)
            logger.info(
                "JZRobotUDP command seq=%s mode=%s transport=udp robot=%s action_key_count=%s "
                "target=%s:%s bytes=%s",
                packet["seq"],
                packet["mode"],
                packet["robot"],
                len(float_action),
                self.config.command_target_ip,
                self.config.command_target_port,
                sent_bytes,
            )
        else:
            raise RuntimeError(f"unsupported send_action_transport: {self.config.send_action_transport}")

        return float_action

    def _validate_and_float_action(self, action: RobotAction) -> RobotAction:
        expected_keys = set(self.action_features)
        action_keys = set(action)
        missing = sorted(expected_keys - action_keys)
        unexpected = sorted(action_keys - expected_keys)
        if missing:
            raise ValueError(f"JZRobotUDP action is missing keys: {missing}")
        if unexpected:
            raise ValueError(f"JZRobotUDP action has unexpected keys: {unexpected}")

        float_action: RobotAction = {}
        for key in self.action_features:
            value = action[key]
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"JZRobotUDP action {key} must be numeric")
            float_action[key] = float(value)
        return float_action

    def _command_actions(self, action: RobotAction) -> dict[str, Any]:
        command_actions: dict[str, Any] = {
            LEFT: {},
            RIGHT: {},
            "grippers": {
                LEFT: {},
                RIGHT: {},
            },
        }
        for joint in self.config.left_joint_names:
            command_actions[LEFT][joint] = action[f"{LEFT}_{joint}.pos"]
        for joint in self.config.right_joint_names:
            command_actions[RIGHT][joint] = action[f"{RIGHT}_{joint}.pos"]
        if self.config.use_gripper:
            for side in (LEFT, RIGHT):
                command_actions["grippers"][side][GRIPPER_WIDTH] = action[f"{side}_gripper.{GRIPPER_WIDTH}"]
                command_actions["grippers"][side][GRIPPER_FORCE] = action[f"{side}_gripper.{GRIPPER_FORCE}"]
        else:
            command_actions["grippers"][LEFT] = {GRIPPER_WIDTH: 0.0, GRIPPER_FORCE: 0.0}
            command_actions["grippers"][RIGHT] = {GRIPPER_WIDTH: 0.0, GRIPPER_FORCE: 0.0}
        return command_actions

    @check_if_not_connected
    def disconnect(self) -> None:
        for camera in self.cameras.values():
            camera.disconnect()
        self._receiver.stop()
        if self._command_sender is not None:
            self._command_sender.close()
            self._command_sender = None
        self._is_connected = False
