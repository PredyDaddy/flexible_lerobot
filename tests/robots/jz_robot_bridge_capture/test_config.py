from pathlib import Path

from lerobot.robots.jz_robot.bridge_capture.config import BridgeCaptureConfig


def test_bridge_capture_config_loads_project_defaults(tmp_path: Path):
    config_path = tmp_path / "bridge_capture.yaml"
    config_path.write_text(
        """
robot:
  name: jz_dual_arm
  namespace: /robot1
  orin_ip: 192.168.50.10
  collector_ip: 192.168.50.20
  ros_domain_id: 50

recording:
  sample_rate_hz: 20
  output_root: /tmp/jz_raw

cameras:
  camera_head:
    rtsp_url: rtsp://192.168.50.10:8554/robot_camera/camera_head
    width: 1280
    height: 720
    fps: 30

state:
  topic: /robot1/lerobot/state
  names: [left_j1, left_j2]

action:
  topic: /robot1/lerobot/action
  names: [left_j1, left_j2, left_gripper_width]
  sources:
    - id: left_arm
      topic: /robot1/telecon/arm_left/joint_commands_input
      type: sensor_msgs/msg/JointState
      names: [left_j1, left_j2]
    - id: left_gripper
      topic: /robot1/left_gripper/gripper_commands
      type: std_msgs/msg/Float64MultiArray
      names: [left_gripper_width, left_gripper_force]
      include_names: [left_gripper_width]
""",
        encoding="utf-8",
    )

    config = BridgeCaptureConfig.from_yaml(config_path)

    assert config.robot.namespace == "/robot1"
    assert config.recording.sample_rate_hz == 20
    assert config.cameras["camera_head"].rtsp_url == "rtsp://192.168.50.10:8554/robot_camera/camera_head"
    assert config.state.names == ["left_j1", "left_j2"]
    assert config.action.sources[1].include_names == ["left_gripper_width"]


def test_bridge_capture_config_validates_action_names_are_supplied_by_sources(tmp_path: Path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
robot:
  name: jz_dual_arm
  namespace: /robot1
recording:
  output_root: /tmp/jz_raw
cameras: {}
state:
  topic: /robot1/lerobot/state
  names: [left_j1]
action:
  topic: /robot1/lerobot/action
  names: [left_j1, missing_action]
  sources:
    - id: left_arm
      topic: /left
      type: sensor_msgs/msg/JointState
      names: [left_j1]
""",
        encoding="utf-8",
    )

    try:
        BridgeCaptureConfig.from_yaml(config_path)
    except ValueError as exc:
        assert "missing_action" in str(exc)
    else:
        raise AssertionError("expected config validation to fail")
