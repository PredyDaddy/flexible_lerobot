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


LEFT_TOPIC = "/puppet/joint_left"
RIGHT_TOPIC = "/puppet/joint_right"
TIMEOUT_S = 5.0


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


def main() -> int:
    rospy.init_node("agilex_read_current_joint_positions", anonymous=True, disable_signals=True)

    left_msg = wait_joint_state(LEFT_TOPIC)
    right_msg = wait_joint_state(RIGHT_TOPIC)

    left_positions = extract_positions(left_msg, LEFT_TOPIC)
    right_positions = extract_positions(right_msg, RIGHT_TOPIC)

    print(f"left  ({LEFT_TOPIC}): {left_positions}")
    print(f"right ({RIGHT_TOPIC}): {right_positions}")
    print(f"all joints: {left_positions + right_positions}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
