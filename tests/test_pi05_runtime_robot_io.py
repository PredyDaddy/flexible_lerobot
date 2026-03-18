from __future__ import annotations

import threading

from my_devs.pi05_engineering.runtime.robot_io import SerializedRobotIO


class BlockingRobot:
    def __init__(self) -> None:
        self.robot_type = "so101_follower"
        self.action_features = {"action": {"shape": [6]}}
        self.observation_features = {"observation": {"shape": [6]}}
        self.is_connected = False
        self.custom_attr = "delegated"
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.action_calls = 0
        self.observation_calls = 0
        self.observation_entered = threading.Event()
        self.release_observation = threading.Event()
        self.action_started = threading.Event()

    def connect(self) -> None:
        self.connect_calls += 1
        self.is_connected = True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.is_connected = False

    def get_observation(self) -> dict[str, float]:
        self.observation_calls += 1
        self.observation_entered.set()
        self.release_observation.wait(timeout=1.0)
        return {"joint": 1.0}

    def send_action(self, action) -> None:  # noqa: ANN001
        self.action_calls += 1
        self.action_started.set()


def test_serialized_robot_io_serializes_observation_and_action_calls() -> None:
    robot = BlockingRobot()
    wrapped = SerializedRobotIO(robot)

    obs_thread = threading.Thread(target=wrapped.get_observation, daemon=True)
    action_thread = threading.Thread(target=wrapped.send_action, args=({"joint": 2.0},), daemon=True)

    obs_thread.start()
    assert robot.observation_entered.wait(timeout=1.0) is True

    action_thread.start()
    assert robot.action_started.wait(timeout=0.05) is False

    robot.release_observation.set()
    obs_thread.join(timeout=1.0)
    action_thread.join(timeout=1.0)

    assert robot.observation_calls == 1
    assert robot.action_calls == 1
    assert robot.action_started.is_set() is True


def test_serialized_robot_io_delegates_properties_and_attributes() -> None:
    robot = BlockingRobot()
    wrapped = SerializedRobotIO(robot)

    wrapped.connect()

    assert wrapped.robot is robot
    assert wrapped.robot_type == "so101_follower"
    assert wrapped.action_features is robot.action_features
    assert wrapped.observation_features is robot.observation_features
    assert wrapped.is_connected is True
    assert wrapped.custom_attr == "delegated"

    wrapped.disconnect()

    assert robot.connect_calls == 1
    assert robot.disconnect_calls == 1
    assert wrapped.is_connected is False
