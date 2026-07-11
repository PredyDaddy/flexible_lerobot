#!/usr/bin/env python3
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
MY_DEVS_ROOT = REPO_ROOT / "my_devs"
for path in (str(SRC_ROOT), str(MY_DEVS_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from lerobot.robots.jz_robot_udp.protocol import PROTOCOL_VERSION, STATE_MESSAGE_TYPE, encode_state_packet
from my_devs.jz_robot.common import DEFAULT_ROBOT_CONFIG, load_robot_config

LEFT = "left"
RIGHT = "right"
SOURCE_NAMES = ("left_joints", "right_joints", "left_gripper", "right_gripper")
NSEC_PER_SEC = 1_000_000_000
MEASURED_RATE_WINDOW_PACKETS = 30
COMMON_IPV4_UDP_PAYLOAD_BYTES = 1472
STOP_REQUESTED = False


def request_stop(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than or equal to zero")
    return parsed


def _header_stamp_ns(msg: Any) -> int:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None or not hasattr(stamp, "sec") or not hasattr(stamp, "nanosec"):
        return 0
    return int(stamp.sec) * NSEC_PER_SEC + int(stamp.nanosec)


@dataclass
class SourceState:
    generation: int = 0
    receive_monotonic_ns: int | None = None
    receive_wall_ns: int | None = None
    header_stamp_ns: int | None = None


@dataclass(frozen=True)
class StateSnapshot:
    packet: dict[str, Any] | None
    source_timing: dict[str, Any] | None
    generations: dict[str, int]
    source_age_ms: dict[str, float | None]
    source_skew_ms: float | None
    progress_modes: dict[str, str]
    joint_header_stamps: dict[str, int | None]
    skip_reasons: tuple[str, ...]


class ReadonlyStateCollector:
    def __init__(
        self,
        robot_cfg: Any,
        *,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        wall_time_ns: Callable[[], int] = time.time_ns,
    ):
        self.robot_cfg = robot_cfg
        self._monotonic_ns = monotonic_ns
        self._wall_time_ns = wall_time_ns
        self._lock = threading.Lock()
        self.joints = {LEFT: {}, RIGHT: {}}
        self.grippers = {LEFT: {}, RIGHT: {}}
        self._sources = {name: SourceState() for name in SOURCE_NAMES}

    @property
    def counts(self) -> dict[str, int]:
        with self._lock:
            return {name: source.generation for name, source in self._sources.items()}

    def _record_source_update(self, source_name: str, *, header_stamp_ns: int | None) -> None:
        source = self._sources[source_name]
        source.generation += 1
        source.receive_monotonic_ns = self._monotonic_ns()
        source.receive_wall_ns = self._wall_time_ns()
        source.header_stamp_ns = header_stamp_ns

    def update_joints(self, side: str, msg: JointState) -> None:
        state = {
            name: float(msg.position[idx]) for idx, name in enumerate(msg.name) if idx < len(msg.position)
        }
        with self._lock:
            self.joints[side] = state
            self._record_source_update(f"{side}_joints", header_stamp_ns=_header_stamp_ns(msg))

    def update_gripper(self, side: str, msg: Float64MultiArray) -> None:
        state = {}
        if len(msg.data) > 0:
            state["width"] = float(msg.data[0])
        if len(msg.data) > 1:
            state["force"] = float(msg.data[1])
        with self._lock:
            self.grippers[side] = state
            self._record_source_update(f"{side}_gripper", header_stamp_ns=None)

    def _missing_inputs_unlocked(self) -> dict[str, list[str]]:
        missing = {
            "left_joints": [
                name for name in self.robot_cfg.left_joint_names if name not in self.joints[LEFT]
            ],
            "right_joints": [
                name for name in self.robot_cfg.right_joint_names if name not in self.joints[RIGHT]
            ],
            "left_gripper_fields": [],
            "right_gripper_fields": [],
        }
        if self.robot_cfg.use_gripper:
            for side in (LEFT, RIGHT):
                missing[f"{side}_gripper_fields"] = [
                    field for field in ("width", "force") if field not in self.grippers[side]
                ]
        return missing

    def ready(self) -> bool:
        with self._lock:
            return all(not missing for missing in self._missing_inputs_unlocked().values())

    def missing_inputs(self) -> dict[str, list[str]]:
        with self._lock:
            return self._missing_inputs_unlocked()

    def readiness_details(self) -> str:
        with self._lock:
            counts = {name: source.generation for name, source in self._sources.items()}
            return f"counts={counts} missing={self._missing_inputs_unlocked()}"

    def snapshot(
        self,
        *,
        seq: int,
        robot_name: str,
        max_source_age_ms: float,
        max_source_skew_ms: float,
        require_all_sources_advanced: bool,
        last_sent_generations: dict[str, int],
        last_sent_joint_header_stamps: dict[str, int] | None = None,
    ) -> StateSnapshot:
        with self._lock:
            snapshot_monotonic_ns = self._monotonic_ns()
            snapshot_wall_ns = self._wall_time_ns()
            generations = {name: source.generation for name, source in self._sources.items()}
            missing = self._missing_inputs_unlocked()
            source_age_ms: dict[str, float | None] = {}
            receive_monotonic_values: list[int] = []
            timing_sources: dict[str, dict[str, Any]] = {}
            progress_modes: dict[str, str] = {}
            joint_header_stamps: dict[str, int | None] = {}

            for name, source in self._sources.items():
                if source.receive_monotonic_ns is None or source.receive_wall_ns is None:
                    source_age_ms[name] = None
                    continue
                age_ms = (snapshot_monotonic_ns - source.receive_monotonic_ns) / 1_000_000
                source_age_ms[name] = age_ms
                receive_monotonic_values.append(source.receive_monotonic_ns)
                timing_sources[name] = {
                    "generation": source.generation,
                    "recv_wall_ns": source.receive_wall_ns,
                    "recv_monotonic_ns": source.receive_monotonic_ns,
                    "header_stamp_ns": source.header_stamp_ns,
                    "age_ms": age_ms,
                }
                if name.endswith("_joints"):
                    joint_header_stamps[name] = source.header_stamp_ns
                    if source.header_stamp_ns is None:
                        progress_modes[name] = "generation_missing_stamp_fallback"
                    elif source.header_stamp_ns == 0:
                        progress_modes[name] = "generation_zero_stamp_fallback"
                    else:
                        progress_modes[name] = "header_stamp"
                else:
                    progress_modes[name] = "generation"

            source_skew_ms = None
            if len(receive_monotonic_values) == len(SOURCE_NAMES):
                source_skew_ms = (max(receive_monotonic_values) - min(receive_monotonic_values)) / 1_000_000

            skip_reasons = []
            if any(missing_fields for missing_fields in missing.values()) or len(timing_sources) != len(
                SOURCE_NAMES
            ):
                skip_reasons.append("not_ready")
            if any(age_ms is not None and age_ms > max_source_age_ms for age_ms in source_age_ms.values()):
                skip_reasons.append("stale")
            if source_skew_ms is not None and source_skew_ms > max_source_skew_ms:
                skip_reasons.append("skew")
            if require_all_sources_advanced:
                last_headers = last_sent_joint_header_stamps or {}
                sources_not_advanced = []
                for name in SOURCE_NAMES:
                    if progress_modes.get(name) == "header_stamp":
                        current_header = joint_header_stamps[name]
                        last_header = last_headers.get(name)
                        if (
                            last_header is not None
                            and current_header is not None
                            and current_header <= last_header
                        ):
                            sources_not_advanced.append(name)
                    elif generations[name] <= last_sent_generations.get(name, 0):
                        sources_not_advanced.append(name)
                if sources_not_advanced:
                    skip_reasons.append("not_advanced")

            if skip_reasons:
                return StateSnapshot(
                    packet=None,
                    source_timing=None,
                    generations=generations,
                    source_age_ms=source_age_ms,
                    source_skew_ms=source_skew_ms,
                    progress_modes=progress_modes,
                    joint_header_stamps=joint_header_stamps,
                    skip_reasons=tuple(skip_reasons),
                )

            source_timing = {
                "schema_version": 1,
                "source_skew_ms": source_skew_ms,
                "sources": timing_sources,
            }
            packet = {
                "version": PROTOCOL_VERSION,
                "type": STATE_MESSAGE_TYPE,
                "robot": robot_name,
                "seq": seq,
                "stamp_ns": snapshot_wall_ns,
                "joints": {
                    LEFT: {name: self.joints[LEFT][name] for name in self.robot_cfg.left_joint_names},
                    RIGHT: {name: self.joints[RIGHT][name] for name in self.robot_cfg.right_joint_names},
                },
                "grippers": {
                    LEFT: self.grippers[LEFT].copy(),
                    RIGHT: self.grippers[RIGHT].copy(),
                },
                "source_timing": source_timing,
            }
            return StateSnapshot(
                packet=packet,
                source_timing=source_timing,
                generations=generations,
                source_age_ms=source_age_ms,
                source_skew_ms=source_skew_ms,
                progress_modes=progress_modes,
                joint_header_stamps=joint_header_stamps,
                skip_reasons=(),
            )


@dataclass
class SenderCounters:
    attempted: int = 0
    sent: int = 0
    skipped_total: int = 0
    skipped_not_ready: int = 0
    skipped_stale: int = 0
    skipped_skew: int = 0
    skipped_not_advanced: int = 0

    def skipped_packets(self) -> dict[str, int]:
        return {
            "total": self.skipped_total,
            "not_ready": self.skipped_not_ready,
            "stale": self.skipped_stale,
            "skew": self.skipped_skew,
            "not_advanced": self.skipped_not_advanced,
        }


class StateUdpSender:
    def __init__(
        self,
        *,
        collector: ReadonlyStateCollector,
        sock: Any,
        target: tuple[str, int],
        robot_name: str,
        hz: float,
        max_source_age_ms: float,
        max_source_skew_ms: float,
        require_all_sources_advanced: bool,
        print_every: int,
        printer: Callable[..., None] = print,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        process_id: int | None = None,
    ):
        self.collector = collector
        self.sock = sock
        self.target = target
        self.robot_name = robot_name
        self.hz = hz
        self.max_source_age_ms = max_source_age_ms
        self.max_source_skew_ms = max_source_skew_ms
        self.require_all_sources_advanced = require_all_sources_advanced
        self.print_every = print_every
        self.printer = printer
        self._monotonic_ns = monotonic_ns
        self.process_id = os.getpid() if process_id is None else process_id
        self.counters = SenderCounters()
        self.last_sent_generations = dict.fromkeys(SOURCE_NAMES, 0)
        self.last_sent_joint_header_stamps: dict[str, int] = {}
        self._successful_send_times_ns: deque[int] = deque(maxlen=MEASURED_RATE_WINDOW_PACKETS)

    @property
    def formatted_hz(self) -> str:
        return f"{self.hz:g}"

    @property
    def measured_send_hz(self) -> float | None:
        if len(self._successful_send_times_ns) < MEASURED_RATE_WINDOW_PACKETS:
            return None
        elapsed_ns = self._successful_send_times_ns[-1] - self._successful_send_times_ns[0]
        if elapsed_ns <= 0:
            return None
        return (MEASURED_RATE_WINDOW_PACKETS - 1) * NSEC_PER_SEC / elapsed_ns

    def _metrics_text(self, snapshot: StateSnapshot) -> str:
        ages = {
            name: None if age_ms is None else round(age_ms, 3)
            for name, age_ms in snapshot.source_age_ms.items()
        }
        skew = "null" if snapshot.source_skew_ms is None else f"{snapshot.source_skew_ms:.3f}"
        measured_hz = self.measured_send_hz
        measured_text = "unavailable" if measured_hz is None else f"{measured_hz:.6f}"
        return (
            f"configured_hz={self.formatted_hz} measured_send_hz={measured_text} "
            f"rate_window_packets={len(self._successful_send_times_ns)} "
            f"update_counts={json.dumps(snapshot.generations, sort_keys=True, separators=(',', ':'))} "
            f"progress_modes={json.dumps(snapshot.progress_modes, sort_keys=True, separators=(',', ':'))} "
            f"source_age_ms={json.dumps(ages, sort_keys=True, separators=(',', ':'))} "
            f"source_skew_ms={skew} "
            f"skipped={json.dumps(self.counters.skipped_packets(), sort_keys=True, separators=(',', ':'))}"
        )

    def _record_skip(self, reasons: tuple[str, ...]) -> None:
        self.counters.skipped_total += 1
        for reason in reasons:
            setattr(self.counters, f"skipped_{reason}", getattr(self.counters, f"skipped_{reason}") + 1)

    def attempt_send(self) -> bool:
        self.counters.attempted += 1
        next_seq = self.counters.sent + 1
        snapshot = self.collector.snapshot(
            seq=next_seq,
            robot_name=self.robot_name,
            max_source_age_ms=self.max_source_age_ms,
            max_source_skew_ms=self.max_source_skew_ms,
            require_all_sources_advanced=self.require_all_sources_advanced,
            last_sent_generations=self.last_sent_generations,
            last_sent_joint_header_stamps=self.last_sent_joint_header_stamps,
        )
        if snapshot.skip_reasons:
            self._record_skip(snapshot.skip_reasons)
            if self.counters.skipped_total == 1 or (
                self.print_every > 0 and self.counters.attempted % self.print_every == 0
            ):
                self.printer(
                    "[orin ros state udp bridge] "
                    f"skip reasons={','.join(snapshot.skip_reasons)} {self._metrics_text(snapshot)}",
                    flush=True,
                )
            return False

        assert snapshot.packet is not None and snapshot.source_timing is not None
        payload = encode_state_packet(snapshot.packet)
        if len(payload) > COMMON_IPV4_UDP_PAYLOAD_BYTES:
            self.printer(
                "[orin ros state udp bridge] WARNING "
                f"payload_bytes={len(payload)} exceeds_common_ipv4_udp_payload="
                f"{COMMON_IPV4_UDP_PAYLOAD_BYTES} seq={next_seq}",
                flush=True,
            )
        self.sock.sendto(payload, self.target)
        self._successful_send_times_ns.append(self._monotonic_ns())
        self.counters.sent += 1
        self.last_sent_generations = snapshot.generations.copy()
        for name, stamp in snapshot.joint_header_stamps.items():
            if stamp is not None and stamp > 0:
                self.last_sent_joint_header_stamps[name] = stamp
        if self.counters.sent == 1 or (self.print_every > 0 and self.counters.sent % self.print_every == 0):
            self.printer(
                "[orin ros state udp bridge] "
                f"sent seq={self.counters.sent} bytes={len(payload)} {self._metrics_text(snapshot)}",
                flush=True,
            )
        if self.measured_send_hz is not None and (
            self.counters.sent == MEASURED_RATE_WINDOW_PACKETS
            or self.counters.sent % MEASURED_RATE_WINDOW_PACKETS == 0
        ):
            self.printer(
                "[orin ros state udp bridge] "
                f"rate pid={self.process_id} seq={self.counters.sent} "
                f"configured_hz={self.formatted_hz} measured_send_hz={self.measured_send_hz:.6f} "
                f"window_packets={MEASURED_RATE_WINDOW_PACKETS}",
                flush=True,
            )
        return True

    def run(
        self,
        *,
        should_stop: Callable[[], bool],
        count: int,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
        health_check: Callable[[], None] | None = None,
    ) -> None:
        start_ns = monotonic_ns()
        tick_index = 0
        while not should_stop() and (count <= 0 or self.counters.sent < count):
            if health_check is not None:
                health_check()
            next_tick_ns = start_ns + round(tick_index * NSEC_PER_SEC / self.hz)
            now_ns = monotonic_ns()
            if now_ns < next_tick_ns:
                sleep(min((next_tick_ns - now_ns) / NSEC_PER_SEC, 0.05))
                continue

            self.attempt_send()
            tick_index += 1
            next_tick_ns = start_ns + round(tick_index * NSEC_PER_SEC / self.hz)
            if next_tick_ns <= now_ns:
                tick_index = math.floor((now_ns - start_ns) * self.hz / NSEC_PER_SEC) + 1


class RosExecutorThread:
    def __init__(self, executor: Any):
        self.executor = executor
        self._exception: BaseException | None = None
        self._thread = threading.Thread(
            target=self._spin,
            name="jz-ros-state-executor",
            daemon=False,
        )

    @property
    def thread(self) -> threading.Thread:
        return self._thread

    def _spin(self) -> None:
        try:
            self.executor.spin()
        except ExternalShutdownException:
            pass
        except BaseException as exc:
            self._exception = exc

    def start(self) -> None:
        self._thread.start()

    def raise_if_failed(self) -> None:
        if self._exception is not None:
            raise RuntimeError("ROS executor thread failed") from self._exception
        if self._thread.ident is not None and not self._thread.is_alive():
            raise RuntimeError("ROS executor thread stopped unexpectedly")

    def stop(self, timeout_s: float = 5.0) -> None:
        if self._thread.ident is None:
            return
        shutdown_error: BaseException | None = None
        try:
            self.executor.shutdown(timeout_sec=timeout_s)
        except BaseException as exc:
            shutdown_error = exc
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            raise RuntimeError("ROS executor thread did not stop within timeout")
        if shutdown_error is not None:
            raise RuntimeError("ROS executor shutdown failed") from shutdown_error
        if self._exception is not None:
            raise RuntimeError("ROS executor thread failed") from self._exception


def state_subscription_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Readonly ROS2 state to UDP bridge for JZRobot.")
    parser.add_argument("--robot-config", default=str(DEFAULT_ROBOT_CONFIG))
    parser.add_argument("--robot-name", default="robot1")
    parser.add_argument("--target-ip", required=True)
    parser.add_argument("--target-port", type=int, default=39010)
    parser.add_argument("--bind-ip", default="192.168.1.81")
    parser.add_argument("--hz", type=_positive_float, default=20.0)
    parser.add_argument("--count", type=int, default=0, help="0 means run forever.")
    parser.add_argument("--print-every", type=int, default=30)
    parser.add_argument("--wait-timeout-s", type=_positive_float, default=10.0)
    parser.add_argument("--max-source-age-ms", type=_nonnegative_float, default=50.0)
    parser.add_argument("--max-source-skew-ms", type=_nonnegative_float, default=20.0)
    parser.add_argument(
        "--require-all-sources-advanced",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> int:
    global STOP_REQUESTED
    STOP_REQUESTED = False
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    args = parse_args()
    robot_cfg = load_robot_config(args.robot_config)
    if not robot_cfg.use_gripper:
        raise ValueError("The 18D state bridge requires both arm and gripper state sources")
    collector = ReadonlyStateCollector(robot_cfg)

    rclpy.init()
    node = rclpy.create_node("jz_readonly_ros_state_udp_bridge")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor_thread = RosExecutorThread(executor)
    sock: socket.socket | None = None

    qos = state_subscription_qos()
    node.create_subscription(
        JointState,
        robot_cfg.left_joint_state_topic,
        lambda msg: collector.update_joints(LEFT, msg),
        qos,
    )
    node.create_subscription(
        JointState,
        robot_cfg.right_joint_state_topic,
        lambda msg: collector.update_joints(RIGHT, msg),
        qos,
    )
    node.create_subscription(
        Float64MultiArray,
        robot_cfg.left_gripper_state_topic,
        lambda msg: collector.update_gripper(LEFT, msg),
        qos,
    )
    node.create_subscription(
        Float64MultiArray,
        robot_cfg.right_gripper_state_topic,
        lambda msg: collector.update_gripper(RIGHT, msg),
        qos,
    )

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((args.bind_ip, 0))
        target = (args.target_ip, args.target_port)
        print(
            "[orin ros state udp bridge] READONLY subscribe-only bridge. "
            "It does not publish command topics and does not call send_action.",
            flush=True,
        )
        print(
            f"[orin ros state udp bridge] pid={os.getpid()} "
            f"local={sock.getsockname()[0]}:{sock.getsockname()[1]} "
            f"target={target[0]}:{target[1]} configured_hz={args.hz:g} "
            f"wait_timeout_s={args.wait_timeout_s:g} max_source_age_ms={args.max_source_age_ms:g} "
            f"max_source_skew_ms={args.max_source_skew_ms:g} "
            f"require_all_sources_advanced={str(args.require_all_sources_advanced).lower()} "
            "qos=keep_last:1,reliable,volatile",
            flush=True,
        )

        executor_thread.start()
        deadline = time.monotonic() + args.wait_timeout_s
        while not STOP_REQUESTED and not collector.ready() and time.monotonic() < deadline:
            executor_thread.raise_if_failed()
            time.sleep(0.01)
        if STOP_REQUESTED:
            print("[orin ros state udp bridge] stop requested before initial state ready", flush=True)
            return 0
        if not collector.ready():
            print(
                f"[orin ros state udp bridge] initial state timeout {collector.readiness_details()}",
                flush=True,
            )
            return 1

        sender = StateUdpSender(
            collector=collector,
            sock=sock,
            target=target,
            robot_name=args.robot_name,
            hz=args.hz,
            max_source_age_ms=args.max_source_age_ms,
            max_source_skew_ms=args.max_source_skew_ms,
            require_all_sources_advanced=args.require_all_sources_advanced,
            print_every=args.print_every,
            monotonic_ns=time.monotonic_ns,
        )
        sender.run(
            should_stop=lambda: STOP_REQUESTED,
            count=args.count,
            health_check=executor_thread.raise_if_failed,
        )
        executor_thread.raise_if_failed()
        return 0
    finally:
        if sock is not None:
            sock.close()
        try:
            executor_thread.stop()
        finally:
            try:
                node.destroy_node()
            finally:
                if rclpy.ok():
                    rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
