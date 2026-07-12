#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
GR00T17_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_CALIBRATION_DIR = GR00T17_ROOT / "configs" / "calibration" / "so_follower"
DEFAULT_ROBOT_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00"


def ensure_within(path: Path, root: Path, *, must_exist: bool = False) -> Path:
    resolved = path.expanduser().resolve(strict=must_exist)
    allowed = root.expanduser().resolve(strict=True)
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"Path escapes allowed root: {resolved}") from exc
    return resolved


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path = ensure_within(path, GR00T17_ROOT)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite bus check report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only SO101 motor bus preflight; no torque enable or goal."
    )
    parser.add_argument("--robot-id", default="hfy_follower")
    parser.add_argument("--robot-port", default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    report_path = ensure_within(args.report, GR00T17_ROOT)
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite bus check report: {report_path}")
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "timestamp": datetime.now().astimezone().isoformat(),
        "robot_port": str(Path(args.robot_port).resolve(strict=True)),
        "writes_goal_position": False,
        "enables_torque": False,
    }

    bus = None
    primary_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    try:
        from lerobot.robots import make_robot_from_config
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

        calibration_dir = ensure_within(args.calibration_dir, GR00T17_ROOT, must_exist=True)
        calibration_file = calibration_dir / f"{args.robot_id}.json"
        if not calibration_file.is_file():
            raise FileNotFoundError(f"Calibration file is missing: {calibration_file}")
        robot = make_robot_from_config(
            SOFollowerRobotConfig(
                id=args.robot_id,
                calibration_dir=calibration_dir,
                port=args.robot_port,
                disable_torque_on_disconnect=True,
                cameras={},
            )
        )
        bus = robot.bus
        bus.connect()
        motors = {}
        for motor, definition in bus.motors.items():
            motors[motor] = {
                "id": definition.id,
                "present_position": float(bus.read("Present_Position", motor, num_retry=3)),
                "torque_enable_raw": int(bus.read("Torque_Enable", motor, normalize=False, num_retry=3)),
            }
        report["calibration_file"] = str(calibration_file)
        report["motors"] = motors
        report["status"] = "passed"
    except BaseException as exc:
        primary_error = exc
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    finally:
        if bus is not None and bus.is_connected:
            try:
                bus.disconnect(disable_torque=True)
                report["cleanup"] = "bus disconnected with torque disable"
            except BaseException as exc:
                cleanup_error = exc
                report["cleanup"] = f"failed: {type(exc).__name__}: {exc}"

    write_report(report_path, report)
    if cleanup_error is not None:
        raise RuntimeError(f"Motor bus cleanup failed: {cleanup_error}") from cleanup_error
    if primary_error is not None:
        raise primary_error


if __name__ == "__main__":
    main()
