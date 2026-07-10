from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_PATH = REPO_ROOT / "udp_test" / "test_scripts" / "arm_side" / "orin_ros_state_udp_bridge.py"


def _load_bridge_module(monkeypatch):
    rclpy = types.ModuleType("rclpy")
    executors = types.ModuleType("rclpy.executors")
    executors.ExternalShutdownException = type("ExternalShutdownException", (Exception,), {})
    executors.SingleThreadedExecutor = object
    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.JointState = object
    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.Float64MultiArray = object
    protocol = types.ModuleType("lerobot.robots.jz_robot_udp.protocol")
    protocol.PROTOCOL_VERSION = 1
    protocol.STATE_MESSAGE_TYPE = "state"
    protocol.encode_state_packet = lambda packet: packet
    common = types.ModuleType("my_devs.jz_robot.common")
    common.DEFAULT_ROBOT_CONFIG = Path("robot.yaml")
    common.load_robot_config = lambda _path: None

    for name, module in {
        "rclpy": rclpy,
        "rclpy.executors": executors,
        "sensor_msgs": sensor_msgs,
        "sensor_msgs.msg": sensor_msgs_msg,
        "std_msgs": std_msgs,
        "std_msgs.msg": std_msgs_msg,
        "lerobot.robots.jz_robot_udp.protocol": protocol,
        "my_devs.jz_robot.common": common,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("test_orin_ros_state_udp_bridge_module", BRIDGE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_state_collector_reports_exact_missing_inputs(monkeypatch) -> None:
    bridge = _load_bridge_module(monkeypatch)
    cfg = SimpleNamespace(
        left_joint_names=["left_joint1", "left_joint2"],
        right_joint_names=["right_joint1", "right_joint2"],
        use_gripper=True,
    )
    collector = bridge.ReadonlyStateCollector(cfg)

    assert collector.missing_inputs() == {
        "left_joints": ["left_joint1", "left_joint2"],
        "right_joints": ["right_joint1", "right_joint2"],
        "left_gripper_fields": ["width", "force"],
        "right_gripper_fields": ["width", "force"],
    }

    collector.update_joints("left", SimpleNamespace(name=["left_joint1"], position=[0.1]))
    collector.update_joints(
        "right",
        SimpleNamespace(name=["right_joint1", "right_joint2"], position=[0.2, 0.3]),
    )
    collector.update_gripper("left", SimpleNamespace(data=[50.0]))
    collector.update_gripper("right", SimpleNamespace(data=[60.0, 70.0]))

    assert collector.missing_inputs() == {
        "left_joints": ["left_joint2"],
        "right_joints": [],
        "left_gripper_fields": ["force"],
        "right_gripper_fields": [],
    }
    details = collector.readiness_details()
    assert "counts=" in details
    assert "'left_joints': [\'left_joint2\']" in details

    collector.update_joints("left", SimpleNamespace(name=["left_joint2"], position=[0.4]))
    collector.update_gripper("left", SimpleNamespace(data=[50.0, 80.0]))

    assert all(not missing for missing in collector.missing_inputs().values())
    assert collector.ready()
