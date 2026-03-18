from __future__ import annotations

import threading
from typing import Any


class SerializedRobotIO:
    """Serialize robot I/O so producer and actor never hit the same device concurrently."""

    def __init__(self, robot: Any) -> None:
        self.robot = robot
        self._lock = threading.RLock()

    @property
    def io_lock(self) -> threading.RLock:
        return self._lock

    @property
    def robot_type(self) -> Any:
        return getattr(self.robot, "robot_type", None)

    @property
    def action_features(self) -> Any:
        return getattr(self.robot, "action_features")

    @property
    def observation_features(self) -> Any:
        return getattr(self.robot, "observation_features")

    @property
    def is_connected(self) -> Any:
        return getattr(self.robot, "is_connected", None)

    def connect(self) -> Any:
        with self._lock:
            return self.robot.connect()

    def disconnect(self) -> Any:
        with self._lock:
            return self.robot.disconnect()

    def get_observation(self) -> Any:
        with self._lock:
            return self.robot.get_observation()

    def send_action(self, action: Any) -> Any:
        with self._lock:
            return self.robot.send_action(action)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.robot, name)
