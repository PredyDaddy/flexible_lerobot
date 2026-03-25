#!/usr/bin/env python

from unittest.mock import patch

from lerobot.robots.jz_robot import JZRobot, JZRobotConfig
from lerobot.robots.jz_robot.jz_robot import LEFT, RIGHT


def make_robot(**overrides) -> JZRobot:
    cfg = JZRobotConfig(
        id="test_jz_robot",
        cameras={},
        use_gripper=False,
        **overrides,
    )
    return JZRobot(cfg)


def test_initial_state_deadline_disabled_for_non_positive_timeout():
    robot = make_robot(init_state_timeout_s=0.0)
    assert robot._initial_state_deadline() is None

    robot.config.init_state_timeout_s = -1.0
    assert robot._initial_state_deadline() is None


def test_initial_state_deadline_uses_monotonic_time_for_positive_timeout():
    robot = make_robot(init_state_timeout_s=5.0)

    with patch("lerobot.robots.jz_robot.jz_robot.time.monotonic", return_value=10.0):
        assert robot._initial_state_deadline() == 15.0


def test_arm_state_ready_requires_first_message_and_all_joints():
    robot = make_robot()

    assert not robot._arm_state_ready(LEFT)

    robot._last_state_time[LEFT] = 1.0
    robot._latest_joint_pos[LEFT] = {robot.config.left_joint_names[0]: 0.0}
    assert not robot._arm_state_ready(LEFT)

    robot._latest_joint_pos[LEFT] = {joint: float(idx) for idx, joint in enumerate(robot.config.left_joint_names)}
    assert robot._arm_state_ready(LEFT)


def test_format_initial_state_status_reports_ready_and_waiting_topics():
    robot = make_robot()
    robot._last_state_time[LEFT] = 1.0
    robot._latest_joint_pos[LEFT] = {joint: float(idx) for idx, joint in enumerate(robot.config.left_joint_names)}

    status = robot._format_initial_state_status()

    assert f"{LEFT}_arm[{robot.config.left_joint_state_topic}]=ready" in status
    assert f"{RIGHT}_arm[{robot.config.right_joint_state_topic}]=waiting_first_message" in status
