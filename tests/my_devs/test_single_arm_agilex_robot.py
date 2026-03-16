from __future__ import annotations

import numpy as np
import pytest

from lerobot.robots.utils import make_robot_from_config
from my_devs.async_act import SingleArmAgileXRobot, SingleArmAgileXRobotConfig


class FakeBridge:
    def __init__(self, *, arm: str):
        self.arm = arm
        self.is_connected = True
        self.published_actions: list[dict[str, float]] = []

    def get_state_features(self) -> dict[str, float]:
        state: dict[str, float] = {}
        for name, offset in (("left", 0.0), ("right", 100.0)):
            for idx in range(7):
                state[f"{name}_joint{idx}.pos"] = offset + idx
        return state

    def get_action_features(self) -> dict[str, float]:
        action: dict[str, float] = {}
        for name, offset in (("left", 0.0), ("right", 100.0)):
            for idx in range(7):
                action[f"{name}_joint{idx}.pos"] = offset + idx
        return action

    def get_images(self) -> dict[str, np.ndarray]:
        images = {
            "camera_front": np.full((480, 640, 3), 11, dtype=np.uint8),
            "camera_left": np.full((480, 640, 3), 22, dtype=np.uint8),
            "camera_right": np.full((480, 640, 3), 33, dtype=np.uint8),
        }
        return {
            "camera_front": images["camera_front"],
            f"camera_{self.arm}": images[f"camera_{self.arm}"],
        }

    def publish_action(self, action: dict[str, float]) -> None:
        self.published_actions.append(action)

    def disconnect(self) -> None:
        self.is_connected = False


def make_robot(
    *,
    arm: str = "right",
    control_mode: str = "passive_follow",
    action_smoothing_alpha: float | None = None,
    max_joint_step_rad: float | None = None,
) -> SingleArmAgileXRobot:
    robot = SingleArmAgileXRobot(
        SingleArmAgileXRobotConfig(
            id="robot-under-test",
            arm=arm,
            control_mode=control_mode,
            action_smoothing_alpha=action_smoothing_alpha,
            max_joint_step_rad=max_joint_step_rad,
        )
    )
    robot._bridge = FakeBridge(arm=arm)
    return robot


def test_single_arm_robot_factory_instantiation():
    config = SingleArmAgileXRobotConfig(id="factory-test", arm="right")

    robot = make_robot_from_config(config)

    assert isinstance(robot, SingleArmAgileXRobot)


def test_single_arm_observation_contract_right():
    robot = make_robot(arm="right")

    observation = robot.get_observation()

    assert set(observation) == {
        "right_joint0.pos",
        "right_joint1.pos",
        "right_joint2.pos",
        "right_joint3.pos",
        "right_joint4.pos",
        "right_joint5.pos",
        "right_joint6.pos",
        "camera_front",
        "camera_right",
    }
    np.testing.assert_allclose(
        [observation[f"right_joint{i}.pos"] for i in range(7)],
        np.arange(100, 107, dtype=np.float32),
    )
    assert observation["camera_front"].shape == (480, 640, 3)
    assert observation["camera_right"].shape == (480, 640, 3)


def test_single_arm_action_merge_keeps_inactive_arm_and_publishes_full_action():
    robot = make_robot(arm="right", control_mode="command_master")
    action = {f"right_joint{i}.pos": float(i) + 0.5 for i in range(7)}

    sent_action = robot.send_action(action)

    assert sent_action == action
    assert len(robot._bridge.published_actions) == 1
    published = robot._bridge.published_actions[0]
    np.testing.assert_allclose(
        [published[f"left_joint{i}.pos"] for i in range(7)],
        np.arange(7, dtype=np.float32),
    )
    np.testing.assert_allclose(
        [published[f"right_joint{i}.pos"] for i in range(7)],
        np.asarray([action[f"right_joint{i}.pos"] for i in range(7)], dtype=np.float32),
    )


def test_passive_follow_does_not_publish():
    robot = make_robot(arm="left", control_mode="passive_follow")
    action = {f"left_joint{i}.pos": float(i) + 1.0 for i in range(7)}

    sent_action = robot.send_action(action)

    assert sent_action == action
    assert robot._bridge.published_actions == []


def test_single_arm_action_smoothing_alpha_moves_command_towards_target():
    robot = make_robot(arm="right", control_mode="passive_follow", action_smoothing_alpha=0.5)
    target = {f"right_joint{i}.pos": float(110 + i) for i in range(7)}

    first_sent = robot.send_action(target)
    second_sent = robot.send_action(target)

    np.testing.assert_allclose(
        [first_sent[f"right_joint{i}.pos"] for i in range(7)],
        np.asarray([105 + i for i in range(7)], dtype=np.float32),
    )
    np.testing.assert_allclose(
        [second_sent[f"right_joint{i}.pos"] for i in range(7)],
        np.asarray([107.5 + i for i in range(7)], dtype=np.float32),
    )


def test_single_arm_action_smoothing_max_step_limits_single_update():
    robot = make_robot(arm="right", control_mode="passive_follow", max_joint_step_rad=0.2)
    target = {f"right_joint{i}.pos": float(101 + i) for i in range(7)}

    sent_action = robot.send_action(target)

    np.testing.assert_allclose(
        [sent_action[f"right_joint{i}.pos"] for i in range(7)],
        np.asarray([100.2 + i for i in range(7)], dtype=np.float32),
    )


def test_single_arm_robot_config_rejects_invalid_smoothing_parameters():
    with pytest.raises(ValueError, match="action_smoothing_alpha"):
        SingleArmAgileXRobotConfig(id="robot-under-test", arm="right", action_smoothing_alpha=0.0)

    with pytest.raises(ValueError, match="max_joint_step_rad"):
        SingleArmAgileXRobotConfig(id="robot-under-test", arm="right", max_joint_step_rad=0.0)
