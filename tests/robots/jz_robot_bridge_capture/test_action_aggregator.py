import pytest

from lerobot.robots.jz_robot.bridge_capture.action_aggregator import ActionAggregator
from lerobot.robots.jz_robot.bridge_capture.config import ActionSourceConfig, VectorTopicConfig


def test_action_aggregator_emits_fixed_order_vector_from_partial_source_updates():
    config = VectorTopicConfig(
        topic="/robot1/lerobot/action",
        names=["left_j1", "left_j2", "right_j1", "left_gripper_width"],
        sources=[
            ActionSourceConfig(
                id="left_arm",
                topic="/robot1/telecon/arm_left/joint_commands_input",
                type="sensor_msgs/msg/JointState",
                names=["left_j1", "left_j2"],
            ),
            ActionSourceConfig(
                id="right_arm",
                topic="/robot1/telecon/arm_right/joint_commands_input",
                type="sensor_msgs/msg/JointState",
                names=["right_j1"],
            ),
            ActionSourceConfig(
                id="left_gripper",
                topic="/robot1/left_gripper/gripper_commands",
                type="std_msgs/msg/Float64MultiArray",
                names=["left_gripper_width", "left_gripper_force"],
                include_names=["left_gripper_width"],
            ),
        ],
    )
    aggregator = ActionAggregator(config)

    aggregator.update(
        source_id="left_arm",
        names=["left_j1", "left_j2"],
        values=[0.1, 0.2],
        source_time_ns=100,
        receive_time_ns=110,
    )
    assert aggregator.snapshot() is None

    aggregator.update(
        source_id="right_arm",
        names=["right_j1"],
        values=[1.1],
        source_time_ns=120,
        receive_time_ns=130,
    )
    aggregator.update(
        source_id="left_gripper",
        names=["left_gripper_width", "left_gripper_force"],
        values=[42.0, 80.0],
        source_time_ns=140,
        receive_time_ns=150,
    )

    snapshot = aggregator.snapshot()

    assert snapshot is not None
    assert snapshot.names == ["left_j1", "left_j2", "right_j1", "left_gripper_width"]
    assert snapshot.values == [0.1, 0.2, 1.1, 42.0]
    assert snapshot.source_time_ns == 140
    assert snapshot.receive_time_ns == 150


def test_action_aggregator_rejects_unknown_source_id():
    aggregator = ActionAggregator(
        VectorTopicConfig(
            topic="/robot1/lerobot/action",
            names=["left_j1"],
            sources=[
                ActionSourceConfig(
                    id="left_arm",
                    topic="/left",
                    type="sensor_msgs/msg/JointState",
                    names=["left_j1"],
                )
            ],
        )
    )

    with pytest.raises(KeyError, match="unknown action source"):
        aggregator.update(
            source_id="missing",
            names=["left_j1"],
            values=[1.0],
            source_time_ns=1,
            receive_time_ns=2,
        )
