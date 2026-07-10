#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TOPICS = (
    "/robot1/right_gripper/gripper_status",
    "/robot1/arm_right/joint_states",
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously record `ros2 topic echo` output for right-side robot status topics."
    )
    parser.add_argument(
        "--topic",
        action="append",
        default=None,
        help="ROS2 topic to echo and record. Can be passed multiple times. Defaults to right gripper + right arm.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the timestamped log file will be created.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Exact log file path. If omitted, a timestamped file is created under --output-dir.",
    )
    parser.add_argument(
        "--no-screen",
        action="store_true",
        help="Only write to the log file; do not mirror topic output to this terminal.",
    )
    return parser.parse_args()


def safe_topic_name(topic: str) -> str:
    return topic.strip("/").replace("/", "_")


def make_log_path(output_dir: Path, topic: str, timestamp: str, log_file: str | None) -> Path:
    if log_file:
        return Path(log_file).expanduser().resolve()

    return (output_dir / f"{safe_topic_name(topic)}_{timestamp}.log").expanduser().resolve()


def terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def record_topic(topic: str, log_path: Path, no_screen: bool) -> int:
    command = ["ros2", "topic", "echo", topic]
    prefix = safe_topic_name(topic)

    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        started_at = datetime.now().isoformat(timespec="seconds")
        log_file.write(f"# started_at={started_at}\n")
        log_file.write(f"# command={' '.join(command)}\n")
        log_file.write("# readonly=true\n")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        RUNNING_PROCESSES.append(process)

        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            if not no_screen:
                print(f"[{prefix}] {line}", end="", flush=True)

        return process.wait()


RUNNING_PROCESSES: list[subprocess.Popen[str]] = []


def main() -> int:
    args = parse_args()
    if shutil.which("ros2") is None:
        print("[FAIL] ros2 command not found. Activate/source the ROS2 environment first.", file=sys.stderr, flush=True)
        return 127

    topics = args.topic or list(DEFAULT_TOPICS)
    if args.log_file and len(topics) != 1:
        print("[FAIL] --log-file can only be used when recording exactly one --topic.", file=sys.stderr, flush=True)
        return 2

    output_dir = Path(args.output_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_paths = {topic: make_log_path(output_dir, topic, timestamp, args.log_file) for topic in topics}
    for log_path in log_paths.values():
        log_path.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] READONLY ONLY: this script only runs ros2 topic echo commands.", flush=True)
    for topic, log_path in log_paths.items():
        print(f"[INFO] command=ros2 topic echo {topic}", flush=True)
        print(f"[INFO] log_file={log_path}", flush=True)
    print("[INFO] press Ctrl-C to stop recording.", flush=True)

    try:
        results: dict[str, int] = {}

        def run_one(topic: str, log_path: Path) -> None:
            results[topic] = record_topic(topic, log_path, args.no_screen)

        threads = [
            threading.Thread(target=run_one, args=(topic, log_path), daemon=True)
            for topic, log_path in log_paths.items()
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        failed = {topic: code for topic, code in results.items() if code != 0}
        if not failed:
            print("[INFO] all ros2 topic echo commands exited normally.", flush=True)
            return 0

        for topic, return_code in failed.items():
            print(
                f"[FAIL] ros2 topic echo exited with code {return_code}: {topic} log_file={log_paths[topic]}",
                file=sys.stderr,
                flush=True,
            )
        return next(iter(failed.values()))
    except KeyboardInterrupt:
        print("\n[INFO] stopping recorder...", flush=True)
        for process in RUNNING_PROCESSES:
            terminate_process(process)
        for log_path in log_paths.values():
            print(f"[INFO] saved log_file={log_path}", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
