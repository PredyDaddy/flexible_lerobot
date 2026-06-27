#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
MY_DEVS_ROOT = REPO_ROOT / "my_devs"
for path in (str(SRC_ROOT), str(MY_DEVS_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray
except Exception as exc:  # pragma: no cover - depends on ROS env
    rclpy = None
    SingleThreadedExecutor = None
    JointState = Any
    Float64MultiArray = Any
    ROS_IMPORT_ERROR = exc
else:
    ROS_IMPORT_ERROR = None

LEFT = "left"
RIGHT = "right"
GRIPPER_FIELDS = ("width", "force")
DEFAULT_ROBOT_CONFIG = REPO_ROOT / "src/lerobot/configs/robot/jz_robot_three_realsense_ros2_topics.yaml"
DEFAULT_RTSP_CAMERAS = {
    "camera_head": "rtsp://192.168.1.81:8554/robot_camera/camera_head",
    "camera_left": "rtsp://192.168.1.81:8554/robot_camera/camera_left",
    "camera_right": "rtsp://192.168.1.81:8554/robot_camera/camera_right",
}


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    warn: bool = False

    @property
    def status(self) -> str:
        if self.ok and self.warn:
            return "WARN"
        if self.ok:
            return "PASS"
        return "FAIL"


@dataclass
class StateCollector:
    left_joint_names: list[str]
    right_joint_names: list[str]
    use_gripper: bool
    joint_positions: dict[str, dict[str, float]] = field(default_factory=lambda: {LEFT: {}, RIGHT: {}})
    joint_counts: dict[str, int] = field(default_factory=lambda: {LEFT: 0, RIGHT: 0})
    gripper_state: dict[str, dict[str, float]] = field(default_factory=lambda: {LEFT: {}, RIGHT: {}})
    gripper_counts: dict[str, int] = field(default_factory=lambda: {LEFT: 0, RIGHT: 0})

    def update_joint_state(self, side: str, msg: Any) -> None:
        for idx, name in enumerate(msg.name):
            if idx < len(msg.position):
                self.joint_positions[side][name] = float(msg.position[idx])
        self.joint_counts[side] += 1

    def update_gripper_state(self, side: str, msg: Any) -> None:
        if len(msg.data) > 0:
            self.gripper_state[side]["width"] = float(msg.data[0])
        if len(msg.data) > 1:
            self.gripper_state[side]["force"] = float(msg.data[1])
        self.gripper_counts[side] += 1

    def missing_joints(self, side: str) -> list[str]:
        required = self.left_joint_names if side == LEFT else self.right_joint_names
        return [name for name in required if name not in self.joint_positions[side]]

    def missing_gripper_fields(self, side: str) -> list[str]:
        if not self.use_gripper:
            return []
        return [name for name in GRIPPER_FIELDS if name not in self.gripper_state[side]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Readonly local ARM readiness check for JZRobot record inputs.")
    parser.add_argument("--robot-config", default=str(DEFAULT_ROBOT_CONFIG), help="Path to JZRobot YAML config.")
    parser.add_argument("--timeout-s", type=float, default=5.0, help="ROS state wait timeout.")
    parser.add_argument("--topic-discovery-timeout-s", type=float, default=3.0, help="ROS topic discovery wait timeout.")
    parser.add_argument(
        "--camera-source",
        choices=("config", "rtsp", "both"),
        default="rtsp",
        help="Camera check source: config uses the robot YAML cameras; rtsp uses default RTSP URLs.",
    )
    parser.add_argument("--skip-cameras", action="store_true", help="Skip camera read checks.")
    parser.add_argument("--camera-frames", type=int, default=2, help="Frames to read from each camera.")
    parser.add_argument("--camera-timeout-ms", type=int, default=None, help="Override camera read timeout.")
    parser.add_argument("--rtsp-head-url", default=DEFAULT_RTSP_CAMERAS["camera_head"])
    parser.add_argument("--rtsp-left-url", default=DEFAULT_RTSP_CAMERAS["camera_left"])
    parser.add_argument("--rtsp-right-url", default=DEFAULT_RTSP_CAMERAS["camera_right"])
    parser.add_argument("--print-config", action="store_true", help="Print parsed robot config summary.")
    return parser.parse_args()


def result(name: str, ok: bool, detail: str, warn: bool = False) -> CheckResult:
    item = CheckResult(name=name, ok=ok, detail=detail, warn=warn)
    print(f"[{item.status}] {item.name}: {item.detail}", flush=True)
    return item


def require_ros() -> CheckResult:
    if ROS_IMPORT_ERROR is not None or rclpy is None or SingleThreadedExecutor is None:
        return result("ros_import", False, f"ROS2 Python import failed: {ROS_IMPORT_ERROR}")
    return result("ros_import", True, "rclpy / sensor_msgs / std_msgs import ok")


def load_config(config_path: str, print_config: bool) -> tuple[Any | None, CheckResult]:
    try:
        from lerobot.robots.jz_robot import JZRobotConfig
        from my_devs.jz_robot.common import load_robot_config, summarize_robot_config

        robot_cfg = load_robot_config(config_path)
        if not isinstance(robot_cfg, JZRobotConfig):
            return None, result("config", False, f"expected JZRobotConfig, got {type(robot_cfg).__name__}")
        if print_config:
            print("[INFO] Parsed robot config:", flush=True)
            print(summarize_robot_config(robot_cfg), flush=True)
        return robot_cfg, result("config", True, f"loaded {Path(config_path).expanduser().resolve()}")
    except Exception as exc:
        return None, result("config", False, f"failed to load config: {exc}")


def check_ros_topics(robot_cfg: Any, timeout_s: float) -> CheckResult:
    needed = [
        robot_cfg.left_joint_state_topic,
        robot_cfg.right_joint_state_topic,
    ]
    if robot_cfg.use_gripper:
        needed.extend([robot_cfg.left_gripper_state_topic, robot_cfg.right_gripper_state_topic])
    for camera_cfg in robot_cfg.cameras.values():
        image_topic = getattr(camera_cfg, "image_topic", None)
        if image_topic:
            needed.append(image_topic)

    node = rclpy.create_node("jz_local_topic_list_check")
    try:
        deadline = time.monotonic() + timeout_s
        topic_names: set[str] = set()
        missing = list(needed)
        while time.monotonic() < deadline:
            topic_names = {name for name, _types in node.get_topic_names_and_types()}
            missing = [topic for topic in needed if topic not in topic_names]
            if not missing:
                break
            time.sleep(0.1)
        if missing:
            return result(
                "topic_list",
                True,
                f"ROS2 discovery did not list all topics within {timeout_s:.1f}s; missing={missing}. "
                "Later subscription checks are authoritative.",
                warn=True,
            )
        return result("topic_list", True, f"all {len(needed)} required topics are visible")
    finally:
        node.destroy_node()


def check_robot_state(robot_cfg: Any, timeout_s: float) -> list[CheckResult]:
    collector = StateCollector(
        left_joint_names=list(robot_cfg.left_joint_names),
        right_joint_names=list(robot_cfg.right_joint_names),
        use_gripper=robot_cfg.use_gripper,
    )
    node = rclpy.create_node("jz_local_state_readonly_check")
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    node.create_subscription(
        JointState,
        robot_cfg.left_joint_state_topic,
        lambda msg: collector.update_joint_state(LEFT, msg),
        robot_cfg.qos_depth,
    )
    node.create_subscription(
        JointState,
        robot_cfg.right_joint_state_topic,
        lambda msg: collector.update_joint_state(RIGHT, msg),
        robot_cfg.qos_depth,
    )
    if robot_cfg.use_gripper:
        node.create_subscription(
            Float64MultiArray,
            robot_cfg.left_gripper_state_topic,
            lambda msg: collector.update_gripper_state(LEFT, msg),
            robot_cfg.qos_depth,
        )
        node.create_subscription(
            Float64MultiArray,
            robot_cfg.right_gripper_state_topic,
            lambda msg: collector.update_gripper_state(RIGHT, msg),
            robot_cfg.qos_depth,
        )

    deadline = time.monotonic() + timeout_s
    try:
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
            arms_ready = all(collector.joint_counts[side] > 0 and not collector.missing_joints(side) for side in (LEFT, RIGHT))
            grippers_ready = all(
                collector.gripper_counts[side] > 0 and not collector.missing_gripper_fields(side)
                for side in (LEFT, RIGHT)
            )
            if arms_ready and (not robot_cfg.use_gripper or grippers_ready):
                break
    finally:
        executor.shutdown()
        node.destroy_node()

    checks: list[CheckResult] = []
    for side in (LEFT, RIGHT):
        missing = collector.missing_joints(side)
        count = collector.joint_counts[side]
        checks.append(
            result(
                f"{side}_arm_joint_state",
                count > 0 and not missing,
                f"messages={count}, joints={len(collector.joint_positions[side])}, missing={missing}",
            )
        )

    if robot_cfg.use_gripper:
        for side in (LEFT, RIGHT):
            missing = collector.missing_gripper_fields(side)
            count = collector.gripper_counts[side]
            checks.append(
                result(
                    f"{side}_gripper_state",
                    count > 0 and not missing,
                    f"messages={count}, fields={sorted(collector.gripper_state[side])}, missing={missing}",
                )
            )

    return checks


def check_cameras(robot_cfg: Any, frames: int, timeout_ms: int | None) -> list[CheckResult]:
    if not robot_cfg.cameras:
        return [result("cameras", True, "no cameras configured", warn=True)]

    from lerobot.cameras.utils import make_cameras_from_configs

    cameras = make_cameras_from_configs(robot_cfg.cameras)
    checks: list[CheckResult] = []
    try:
        for name, camera in cameras.items():
            cfg = robot_cfg.cameras[name]
            if timeout_ms is not None and hasattr(camera, "timeout_ms"):
                camera.timeout_ms = timeout_ms
            try:
                camera.connect()
                shapes = []
                for _ in range(max(1, frames)):
                    frame = camera.read(timeout_ms=timeout_ms)
                    shapes.append(tuple(frame.shape))
                checks.append(
                    result(
                        f"camera_{name}",
                        True,
                        f"type={cfg.type}, topic={getattr(cfg, 'image_topic', 'n/a')}, frames={len(shapes)}, shapes={shapes}",
                    )
                )
            except Exception as exc:
                checks.append(
                    result(
                        f"camera_{name}",
                        False,
                        f"type={cfg.type}, topic={getattr(cfg, 'image_topic', 'n/a')}, error={exc}",
                    )
                )
            finally:
                try:
                    if camera.is_connected:
                        camera.disconnect()
                except Exception:
                    pass
    finally:
        for camera in cameras.values():
            try:
                if camera.is_connected:
                    camera.disconnect()
            except Exception:
                pass
    return checks


def check_rtsp_cameras(rtsp_urls: dict[str, str], frames: int, timeout_ms: int | None) -> list[CheckResult]:
    import cv2

    checks: list[CheckResult] = []
    for name, url in rtsp_urls.items():
        cap = None
        start = time.monotonic()
        try:
            cap = cv2.VideoCapture(url)
            if timeout_ms is not None:
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
            if not cap.isOpened():
                checks.append(result(f"rtsp_{name}", False, f"url={url}, error=failed to open stream"))
                continue

            shapes = []
            for _ in range(max(1, frames)):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                shapes.append(tuple(frame.shape))

            elapsed_s = time.monotonic() - start
            checks.append(
                result(
                    f"rtsp_{name}",
                    len(shapes) == max(1, frames),
                    f"url={url}, frames={len(shapes)}/{max(1, frames)}, shapes={shapes}, elapsed_s={elapsed_s:.3f}",
                )
            )
        except Exception as exc:
            checks.append(result(f"rtsp_{name}", False, f"url={url}, error={exc}"))
        finally:
            if cap is not None:
                cap.release()
    return checks


def main() -> int:
    args = parse_args()
    print("[INFO] JZRobot local readiness check: READONLY ONLY", flush=True)
    print("[INFO] This script does not call send_action and does not publish command topics.", flush=True)

    checks: list[CheckResult] = [require_ros()]
    if not checks[-1].ok:
        print("SUMMARY: FAIL", flush=True)
        return 1

    if not rclpy.ok():
        rclpy.init()

    try:
        robot_cfg, config_check = load_config(args.robot_config, args.print_config)
        checks.append(config_check)
        if robot_cfg is None:
            print("SUMMARY: FAIL", flush=True)
            return 1

        checks.append(check_ros_topics(robot_cfg, args.topic_discovery_timeout_s))
        checks.extend(check_robot_state(robot_cfg, args.timeout_s))
        if args.skip_cameras:
            checks.append(result("cameras", True, "skipped by --skip-cameras", warn=True))
        else:
            if args.camera_source in ("config", "both"):
                checks.extend(check_cameras(robot_cfg, args.camera_frames, args.camera_timeout_ms))
            if args.camera_source in ("rtsp", "both"):
                checks.extend(
                    check_rtsp_cameras(
                        {
                            "camera_head": args.rtsp_head_url,
                            "camera_left": args.rtsp_left_url,
                            "camera_right": args.rtsp_right_url,
                        },
                        args.camera_frames,
                        args.camera_timeout_ms,
                    )
                )
    finally:
        if rclpy.ok():
            rclpy.shutdown()

    failed = [check for check in checks if not check.ok]
    warned = [check for check in checks if check.ok and check.warn]
    print(
        f"SUMMARY: {'FAIL' if failed else 'PASS'} "
        f"passed={len(checks) - len(failed)} failed={len(failed)} warnings={len(warned)}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
