#!/usr/bin/env python3

from __future__ import annotations

import sys

try:
    import rospy
    from sensor_msgs.msg import JointState
except ImportError as exc:
    raise SystemExit(
        "Failed to import rospy/sensor_msgs. Source the ROS workspace and run again."
    ) from exc


STATE_LEFT_TOPIC = "/puppet/joint_left"
STATE_RIGHT_TOPIC = "/puppet/joint_right"
COMMAND_LEFT_TOPIC = "/master/joint_left"
COMMAND_RIGHT_TOPIC = "/master/joint_right"
JOINT_NAMES = [f"joint{i}" for i in range(7)]
TIMEOUT_S = 5.0
INTERPOLATION_STEPS = 200
HOLD_STEPS = 100
PUBLISH_HZ = 100


def wait_joint_state(topic: str) -> JointState:
    try:
        return rospy.wait_for_message(topic, JointState, timeout=TIMEOUT_S)
    except rospy.ROSException as exc:
        raise SystemExit(f"Timed out waiting for {topic}: {exc}") from exc


def extract_positions(msg: JointState, topic: str) -> list[float]:
    positions = [float(value) for value in msg.position]
    if len(positions) != 7:
        raise SystemExit(f"Expected 7 joints on {topic}, got {len(positions)}")
    return positions


def interpolate(start: list[float], end: list[float], ratio: float) -> list[float]:
    return [start_value + (end_value - start_value) * ratio for start_value, end_value in zip(start, end)]


def make_joint_state(positions: list[float]) -> JointState:
    msg = JointState()
    msg.header.stamp = rospy.Time.now()
    msg.name = JOINT_NAMES
    msg.position = positions
    return msg


def main() -> int:
    rospy.init_node("agilex_move_robot_to_zero", anonymous=True, disable_signals=True)

    left_start = extract_positions(wait_joint_state(STATE_LEFT_TOPIC), STATE_LEFT_TOPIC)
    right_start = extract_positions(wait_joint_state(STATE_RIGHT_TOPIC), STATE_RIGHT_TOPIC)
    zero = [0.0] * 7

    left_publisher = rospy.Publisher(COMMAND_LEFT_TOPIC, JointState, queue_size=10)
    right_publisher = rospy.Publisher(COMMAND_RIGHT_TOPIC, JointState, queue_size=10)

    rospy.sleep(0.2)

    rate = rospy.Rate(PUBLISH_HZ)
    for step in range(1, INTERPOLATION_STEPS + 1):
        ratio = step / INTERPOLATION_STEPS
        left_publisher.publish(make_joint_state(interpolate(left_start, zero, ratio)))
        right_publisher.publish(make_joint_state(interpolate(right_start, zero, ratio)))
        rate.sleep()

    zero_left_msg = make_joint_state(zero)
    zero_right_msg = make_joint_state(zero)
    for _ in range(HOLD_STEPS):
        left_publisher.publish(zero_left_msg)
        right_publisher.publish(zero_right_msg)
        rate.sleep()

    print(f"left start : {left_start}")
    print(f"right start: {right_start}")
    print("Zero-position command has been published to both arms.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
