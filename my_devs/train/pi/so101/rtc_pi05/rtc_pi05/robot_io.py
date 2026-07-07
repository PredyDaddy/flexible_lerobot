from __future__ import annotations

import threading
from typing import Any


class SerializedRobotIO:
    """Serialize robot observation/action calls through one lock."""

    def __init__(self, robot: Any) -> None:
        self.robot = robot
        self._lock = threading.RLock()

    def get_observation(self) -> dict[str, Any]:
        with self._lock:
            return self.robot.get_observation()

    def send_action(self, action: Any) -> Any:
        with self._lock:
            return self.robot.send_action(action)
