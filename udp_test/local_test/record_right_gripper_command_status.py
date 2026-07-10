#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "logs"
DEFAULT_COMMAND_TOPIC = "/robot1/right_gripper/gripper_commands"
DEFAULT_STATUS_TOPIC = "/robot1/right_gripper/gripper_status"
CSV_FIELDS = (
    "event_index",
    "receive_time_ns",
    "elapsed_s",
    "kind",
    "topic",
    "message_index",
    "width",
    "force",
    "latest_command_index",
    "latest_command_age_ms",
    "latest_command_width",
    "latest_command_force",
    "previous_command_index",
    "previous_command_age_ms",
    "previous_command_width",
    "previous_command_force",
    "status_minus_latest_command_width",
    "status_minus_latest_command_force",
    "status_minus_previous_command_width",
    "status_minus_previous_command_force",
    "abs_width_delta_latest",
    "abs_width_delta_previous",
    "closer_to_previous_command",
    "status_matches_previous_command",
)


@dataclass(frozen=True)
class GripperSample:
    index: int
    receive_time_ns: int
    width: float | None
    force: float | None


class GripperCommandStatusRecorder:
    def __init__(
        self,
        *,
        csv_writer: csv.DictWriter,
        command_topic: str,
        status_topic: str,
        tolerance: float,
        no_screen: bool,
    ) -> None:
        self._csv_writer = csv_writer
        self._command_topic = command_topic
        self._status_topic = status_topic
        self._tolerance = tolerance
        self._no_screen = no_screen
        self._started_ns = time.time_ns()
        self._event_index = 0
        self._command_index = 0
        self._status_index = 0
        self._command_history: deque[GripperSample] = deque(maxlen=2)

    def record_command(self, msg: Any) -> None:
        self._command_index += 1
        sample = self._sample_from_message(self._command_index, msg)
        self._command_history.append(sample)
        self._write_row(
            kind="command",
            topic=self._command_topic,
            sample=sample,
            latest_command=sample,
            previous_command=self._previous_command(),
        )
        if not self._no_screen:
            print(
                "[command] "
                f"#{sample.index} width={format_float(sample.width)} force={format_float(sample.force)}",
                flush=True,
            )

    def record_status(self, msg: Any) -> None:
        self._status_index += 1
        sample = self._sample_from_message(self._status_index, msg)
        latest_command = self._latest_command()
        previous_command = self._previous_command()
        row = self._write_row(
            kind="status",
            topic=self._status_topic,
            sample=sample,
            latest_command=latest_command,
            previous_command=previous_command,
        )
        if not self._no_screen:
            self._print_status(sample, row)

    def _sample_from_message(self, index: int, msg: Any) -> GripperSample:
        data = list(getattr(msg, "data", []))
        return GripperSample(
            index=index,
            receive_time_ns=time.time_ns(),
            width=float(data[0]) if len(data) > 0 else None,
            force=float(data[1]) if len(data) > 1 else None,
        )

    def _latest_command(self) -> GripperSample | None:
        if not self._command_history:
            return None
        return self._command_history[-1]

    def _previous_command(self) -> GripperSample | None:
        if len(self._command_history) < 2:
            return None
        return self._command_history[-2]

    def _write_row(
        self,
        *,
        kind: str,
        topic: str,
        sample: GripperSample,
        latest_command: GripperSample | None,
        previous_command: GripperSample | None,
    ) -> dict[str, str]:
        self._event_index += 1

        latest_width_delta = subtract_optional(sample.width, get_width(latest_command))
        latest_force_delta = subtract_optional(sample.force, get_force(latest_command))
        previous_width_delta = subtract_optional(sample.width, get_width(previous_command))
        previous_force_delta = subtract_optional(sample.force, get_force(previous_command))
        abs_width_delta_latest = abs_optional(latest_width_delta)
        abs_width_delta_previous = abs_optional(previous_width_delta)
        closer_to_previous = bool_optional_less(abs_width_delta_previous, abs_width_delta_latest)
        status_matches_previous = bool_optional_le(abs_width_delta_previous, self._tolerance)

        row = {
            "event_index": str(self._event_index),
            "receive_time_ns": str(sample.receive_time_ns),
            "elapsed_s": format_float((sample.receive_time_ns - self._started_ns) / 1_000_000_000.0),
            "kind": kind,
            "topic": topic,
            "message_index": str(sample.index),
            "width": format_float(sample.width),
            "force": format_float(sample.force),
            "latest_command_index": format_index(latest_command),
            "latest_command_age_ms": format_age_ms(sample, latest_command),
            "latest_command_width": format_float(get_width(latest_command)),
            "latest_command_force": format_float(get_force(latest_command)),
            "previous_command_index": format_index(previous_command),
            "previous_command_age_ms": format_age_ms(sample, previous_command),
            "previous_command_width": format_float(get_width(previous_command)),
            "previous_command_force": format_float(get_force(previous_command)),
            "status_minus_latest_command_width": format_float(latest_width_delta),
            "status_minus_latest_command_force": format_float(latest_force_delta),
            "status_minus_previous_command_width": format_float(previous_width_delta),
            "status_minus_previous_command_force": format_float(previous_force_delta),
            "abs_width_delta_latest": format_float(abs_width_delta_latest),
            "abs_width_delta_previous": format_float(abs_width_delta_previous),
            "closer_to_previous_command": format_bool(closer_to_previous),
            "status_matches_previous_command": format_bool(status_matches_previous),
        }
        self._csv_writer.writerow(row)
        return row

    def _print_status(self, sample: GripperSample, row: dict[str, str]) -> None:
        latest_delta = row["status_minus_latest_command_width"] or "n/a"
        previous_delta = row["status_minus_previous_command_width"] or "n/a"
        previous_match = row["status_matches_previous_command"] or "n/a"
        closer_previous = row["closer_to_previous_command"] or "n/a"
        print(
            "[status] "
            f"#{sample.index} width={format_float(sample.width)} force={format_float(sample.force)} "
            f"delta_latest_width={latest_delta} delta_previous_width={previous_delta} "
            f"closer_previous={closer_previous} matches_previous={previous_match}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Readonly recorder comparing right gripper status against command messages. "
            "It only subscribes to ROS2 topics and writes a CSV log."
        )
    )
    parser.add_argument("--command-topic", default=DEFAULT_COMMAND_TOPIC, help="Right gripper command topic.")
    parser.add_argument("--status-topic", default=DEFAULT_STATUS_TOPIC, help="Right gripper status topic.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for timestamped CSV logs.",
    )
    parser.add_argument("--csv-file", default=None, help="Exact CSV output path. Overrides --output-dir.")
    parser.add_argument(
        "--duration-s",
        type=float,
        default=0.0,
        help="Stop after this many seconds. 0 means run until Ctrl-C.",
    )
    parser.add_argument("--qos-depth", type=int, default=10, help="ROS2 subscription QoS queue depth.")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Absolute width tolerance used for status_matches_previous_command.",
    )
    parser.add_argument("--no-screen", action="store_true", help="Only write CSV; do not print each sample.")
    return parser.parse_args()


def import_ros_dependencies() -> tuple[Any, Any, Any]:
    try:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from std_msgs.msg import Float64MultiArray
    except Exception as exc:
        raise ImportError(
            "This script must run in a ROS2 Python environment with rclpy and std_msgs available. "
            "Activate conda env `lerobot` and source the ROS setup used on this machine."
        ) from exc
    return rclpy, SingleThreadedExecutor, Float64MultiArray


def make_csv_path(output_dir: Path, csv_file: str | None) -> Path:
    if csv_file:
        return Path(csv_file).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (output_dir / f"right_gripper_command_status_{timestamp}.csv").expanduser().resolve()


def subtract_optional(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def abs_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return abs(value)


def bool_optional_less(left: float | None, right: float | None) -> bool | None:
    if left is None or right is None:
        return None
    return left < right


def bool_optional_le(left: float | None, right: float) -> bool | None:
    if left is None:
        return None
    return left <= right


def get_width(sample: GripperSample | None) -> float | None:
    return sample.width if sample is not None else None


def get_force(sample: GripperSample | None) -> float | None:
    return sample.force if sample is not None else None


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    if not math.isfinite(value):
        return str(value)
    return f"{value:.9g}"


def format_bool(value: bool | None) -> str:
    if value is None:
        return ""
    return "true" if value else "false"


def format_index(sample: GripperSample | None) -> str:
    if sample is None:
        return ""
    return str(sample.index)


def format_age_ms(sample: GripperSample, command: GripperSample | None) -> str:
    if command is None:
        return ""
    return format_float((sample.receive_time_ns - command.receive_time_ns) / 1_000_000.0)


def validate_args(args: argparse.Namespace) -> int:
    if args.duration_s < 0:
        print("[FAIL] --duration-s must be >= 0.", file=sys.stderr, flush=True)
        return 2
    if args.qos_depth <= 0:
        print("[FAIL] --qos-depth must be positive.", file=sys.stderr, flush=True)
        return 2
    if args.tolerance < 0:
        print("[FAIL] --tolerance must be >= 0.", file=sys.stderr, flush=True)
        return 2
    return 0


def main() -> int:
    args = parse_args()
    validation_code = validate_args(args)
    if validation_code != 0:
        return validation_code

    try:
        rclpy, executor_cls, message_cls = import_ros_dependencies()
    except ImportError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr, flush=True)
        return 127

    csv_path = make_csv_path(Path(args.output_dir), args.csv_file)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not rclpy.ok():
        rclpy.init()

    node = rclpy.create_node("jz_right_gripper_command_status_recorder")
    executor = executor_cls()
    executor.add_node(node)
    stop = {"requested": False}

    def request_stop(_signum: int, _frame: Any) -> None:
        stop["requested"] = True

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)

    try:
        with csv_path.open("w", encoding="utf-8", newline="", buffering=1) as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            recorder = GripperCommandStatusRecorder(
                csv_writer=writer,
                command_topic=args.command_topic,
                status_topic=args.status_topic,
                tolerance=args.tolerance,
                no_screen=args.no_screen,
            )

            node.create_subscription(
                message_cls,
                args.command_topic,
                recorder.record_command,
                args.qos_depth,
            )
            node.create_subscription(
                message_cls,
                args.status_topic,
                recorder.record_status,
                args.qos_depth,
            )

            print("[INFO] READONLY ONLY: this script only subscribes to ROS2 topics.", flush=True)
            print(f"[INFO] command_topic={args.command_topic}", flush=True)
            print(f"[INFO] status_topic={args.status_topic}", flush=True)
            print(f"[INFO] csv_file={csv_path}", flush=True)
            print("[INFO] press Ctrl-C to stop recording.", flush=True)

            deadline = time.monotonic() + args.duration_s if args.duration_s > 0 else None
            while not stop["requested"]:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                executor.spin_once(timeout_sec=0.05)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    print(f"[INFO] saved csv_file={csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
