from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .action_aggregator import ActionAggregator
from .config import ActionSourceConfig, BridgeCaptureConfig
from .ros_messages import float64_array_names_values, joint_state_names_values, vector_to_float64_multi_array
from .time_utils import message_stamp_or_receive_time_ns, now_wall_time_ns


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate JZRobot ROS topics into LeRobot action/state vectors.")
    parser.add_argument("--config", required=True, help="Path to bridge_capture YAML.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = BridgeCaptureConfig.from_yaml(Path(args.config))
    _run_vector_bridge(config)


def _run_vector_bridge(config: BridgeCaptureConfig) -> None:
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray
    except Exception as exc:
        raise ImportError("ros_action_bridge must run in a ROS 2 Python environment with rclpy installed") from exc

    class LerobotVectorBridgeNode(Node):
        def __init__(self) -> None:
            super().__init__("lerobot_vector_bridge")
            self._action_aggregator = ActionAggregator(config.action)
            self._state_aggregator = ActionAggregator(config.state) if config.state.sources else None
            self._action_publisher = self.create_publisher(Float64MultiArray, config.action.topic, 10)
            self._state_publisher = self.create_publisher(Float64MultiArray, config.state.topic, 10)
            self._subscriptions = []
            for source in config.action.sources:
                msg_type = _message_type_for_source(source, JointState, Float64MultiArray)
                sub = self.create_subscription(
                    msg_type,
                    source.topic,
                    self._make_callback(source, self._action_aggregator, self._action_publisher),
                    10,
                )
                self._subscriptions.append(sub)
                self.get_logger().info(f"Subscribed action source {source.id}: {source.topic} ({source.type})")
            for source in config.state.sources:
                msg_type = _message_type_for_source(source, JointState, Float64MultiArray)
                sub = self.create_subscription(
                    msg_type,
                    source.topic,
                    self._make_callback(source, self._state_aggregator, self._state_publisher),
                    10,
                )
                self._subscriptions.append(sub)
                self.get_logger().info(f"Subscribed state source {source.id}: {source.topic} ({source.type})")
            self.get_logger().info(
                f"Publishing LeRobot action vector {config.action.topic} with names={config.action.names}"
            )
            if config.state.sources:
                self.get_logger().info(
                    f"Publishing LeRobot state vector {config.state.topic} with names={config.state.names}"
                )
            else:
                self.get_logger().warning(
                    "No state.sources configured; state vector topic will not publish. "
                    "Recorder still expects state.topic unless you provide another publisher."
                )

        def _make_callback(self, source: ActionSourceConfig, aggregator: ActionAggregator | None, publisher: Any):
            def callback(message: Any) -> None:
                if aggregator is None:
                    return
                receive_time_ns = now_wall_time_ns()
                names, values = _extract_names_values(source, message)
                aggregator.update(
                    source_id=source.id,
                    names=names,
                    values=values,
                    source_time_ns=message_stamp_or_receive_time_ns(message, receive_time_ns),
                    receive_time_ns=receive_time_ns,
                )
                snapshot = aggregator.snapshot()
                if snapshot is None:
                    return
                publisher.publish(vector_to_float64_multi_array(snapshot.values))

            return callback

    rclpy.init()
    node = LerobotVectorBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _message_type_for_source(source: ActionSourceConfig, joint_state_type: Any, float64_array_type: Any) -> Any:
    if source.type == "sensor_msgs/msg/JointState":
        return joint_state_type
    if source.type == "std_msgs/msg/Float64MultiArray":
        return float64_array_type
    raise ValueError(f"unsupported source message type: {source.type}")


def _extract_names_values(source: ActionSourceConfig, message: Any) -> tuple[list[str], list[float]]:
    if source.type == "sensor_msgs/msg/JointState":
        return joint_state_names_values(message)
    if source.type == "std_msgs/msg/Float64MultiArray":
        return float64_array_names_values(message, source.names)
    raise ValueError(f"unsupported source message type: {source.type}")


if __name__ == "__main__":
    main()
