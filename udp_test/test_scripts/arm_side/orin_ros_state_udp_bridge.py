#!/usr/bin/env python3
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import signal
import socket
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from multiprocessing.connection import wait as wait_for_connections
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
MY_DEVS_ROOT = REPO_ROOT / "my_devs"
for path in (str(SRC_ROOT), str(MY_DEVS_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.context import Context
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from lerobot.robots.jz_robot_udp.protocol import PROTOCOL_VERSION, STATE_MESSAGE_TYPE, encode_state_packet
from my_devs.jz_robot.common import DEFAULT_ROBOT_CONFIG, load_robot_config
from my_devs.orin_session.orin.event_transport import UnixEventClient, runtime_event_gate
from my_devs.orin_session.orin.readiness import publish_ready, remove_ready

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


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be an integer greater than zero")
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
    worker_pid: int | None = None
    ipc_delay_ms: float | None = None


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
    stale_sources: tuple[str, ...]
    not_advanced_sources: tuple[str, ...]
    skew_sources: tuple[str, ...]
    source_worker_pids: dict[str, int | None]
    source_ipc_delay_ms: dict[str, float | None]


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

    def _record_source_update(
        self,
        source_name: str,
        *,
        header_stamp_ns: int | None,
        receive_monotonic_ns: int | None = None,
        receive_wall_ns: int | None = None,
        worker_pid: int | None = None,
        ipc_apply_monotonic_ns: int | None = None,
    ) -> None:
        source = self._sources[source_name]
        callback_monotonic_ns = (
            self._monotonic_ns() if receive_monotonic_ns is None else receive_monotonic_ns
        )
        apply_monotonic_ns = (
            self._monotonic_ns() if ipc_apply_monotonic_ns is None else ipc_apply_monotonic_ns
        )
        source.generation += 1
        source.receive_monotonic_ns = callback_monotonic_ns
        source.receive_wall_ns = self._wall_time_ns() if receive_wall_ns is None else receive_wall_ns
        source.header_stamp_ns = header_stamp_ns
        source.worker_pid = worker_pid
        source.ipc_delay_ms = max(0.0, (apply_monotonic_ns - callback_monotonic_ns) / 1_000_000)

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

    def apply_worker_update(self, update: tuple[Any, ...]) -> None:
        if len(update) == 5:
            source_name, values, receive_monotonic_ns, receive_wall_ns, header_stamp_ns = update
            worker_pid = None
        else:
            source_name, values, receive_monotonic_ns, receive_wall_ns, header_stamp_ns, worker_pid = update
        side, source_kind = source_name.split("_", maxsplit=1)
        with self._lock:
            if source_kind == "joints":
                names, positions = values
                self.joints[side] = {
                    name: float(positions[index])
                    for index, name in enumerate(names)
                    if index < len(positions)
                }
            elif source_kind == "gripper":
                state = {}
                if len(values) > 0:
                    state["width"] = float(values[0])
                if len(values) > 1:
                    state["force"] = float(values[1])
                self.grippers[side] = state
            else:
                raise ValueError(f"unknown source kind: {source_kind}")
            self._record_source_update(
                source_name,
                header_stamp_ns=header_stamp_ns,
                receive_monotonic_ns=receive_monotonic_ns,
                receive_wall_ns=receive_wall_ns,
                worker_pid=worker_pid,
                ipc_apply_monotonic_ns=self._monotonic_ns(),
            )

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
            source_worker_pids: dict[str, int | None] = {}
            source_ipc_delay_ms: dict[str, float | None] = {}
            receive_monotonic_by_source: dict[str, int] = {}

            for name, source in self._sources.items():
                if source.receive_monotonic_ns is None or source.receive_wall_ns is None:
                    source_age_ms[name] = None
                    continue
                age_ms = (snapshot_monotonic_ns - source.receive_monotonic_ns) / 1_000_000
                source_age_ms[name] = age_ms
                receive_monotonic_values.append(source.receive_monotonic_ns)
                receive_monotonic_by_source[name] = source.receive_monotonic_ns
                source_worker_pids[name] = source.worker_pid
                source_ipc_delay_ms[name] = source.ipc_delay_ms
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
            skew_sources: tuple[str, ...] = ()
            if len(receive_monotonic_values) == len(SOURCE_NAMES):
                source_skew_ms = (max(receive_monotonic_values) - min(receive_monotonic_values)) / 1_000_000
                oldest_source = min(receive_monotonic_by_source, key=receive_monotonic_by_source.get)
                newest_source = max(receive_monotonic_by_source, key=receive_monotonic_by_source.get)
                skew_sources = (oldest_source, newest_source)

            skip_reasons = []
            stale_sources = tuple(
                name
                for name in SOURCE_NAMES
                if source_age_ms.get(name) is not None and source_age_ms[name] > max_source_age_ms
            )
            sources_not_advanced: list[str] = []
            if any(missing_fields for missing_fields in missing.values()) or len(timing_sources) != len(
                SOURCE_NAMES
            ):
                skip_reasons.append("not_ready")
            if stale_sources:
                skip_reasons.append("stale")
            if source_skew_ms is not None and source_skew_ms > max_source_skew_ms:
                skip_reasons.append("skew")
            if require_all_sources_advanced:
                last_headers = last_sent_joint_header_stamps or {}
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
                    stale_sources=stale_sources,
                    not_advanced_sources=tuple(sources_not_advanced),
                    skew_sources=skew_sources,
                    source_worker_pids=source_worker_pids,
                    source_ipc_delay_ms=source_ipc_delay_ms,
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
                stale_sources=(),
                not_advanced_sources=(),
                skew_sources=skew_sources,
                source_worker_pids=source_worker_pids,
                source_ipc_delay_ms=source_ipc_delay_ms,
            )


@dataclass
class SenderCounters:
    attempted: int = 0
    recorded: int = 0
    sent: int = 0
    network_errors: int = 0
    skipped_total: int = 0
    skipped_not_ready: int = 0
    skipped_stale: int = 0
    skipped_skew: int = 0
    skipped_not_advanced: int = 0
    consecutive_skips: int = 0
    max_consecutive_skips: int = 0

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
        wall_time_ns: Callable[[], int] = time.time_ns,
        process_id: int | None = None,
        audit_sink: Callable[[str, dict[str, Any]], bool] | None = None,
        event_gate_path: str | Path | None = None,
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
        self._wall_time_ns = wall_time_ns
        self.process_id = os.getpid() if process_id is None else process_id
        self.audit_sink = audit_sink
        self.event_gate_path = Path(event_gate_path) if event_gate_path is not None else None
        self.counters = SenderCounters()
        self.last_sent_generations = dict.fromkeys(SOURCE_NAMES, 0)
        self.last_sent_joint_header_stamps: dict[str, int] = {}
        self._successful_send_times_ns: deque[int] = deque(maxlen=MEASURED_RATE_WINDOW_PACKETS)
        self._last_skip_signature: tuple[Any, ...] | None = None

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
        worker_pids = json.dumps(snapshot.source_worker_pids, sort_keys=True, separators=(",", ":"))
        ipc_delays = json.dumps(snapshot.source_ipc_delay_ms, sort_keys=True, separators=(",", ":"))
        return (
            f"configured_hz={self.formatted_hz} measured_send_hz={measured_text} "
            f"rate_window_packets={len(self._successful_send_times_ns)} "
            f"local_recorded={self.counters.recorded} udp_sent={self.counters.sent} "
            f"udp_network_errors={self.counters.network_errors} "
            f"update_counts={json.dumps(snapshot.generations, sort_keys=True, separators=(',', ':'))} "
            f"progress_modes={json.dumps(snapshot.progress_modes, sort_keys=True, separators=(',', ':'))} "
            f"source_age_ms={json.dumps(ages, sort_keys=True, separators=(',', ':'))} "
            f"source_skew_ms={skew} "
            f"stale_sources={json.dumps(snapshot.stale_sources, separators=(',', ':'))} "
            f"not_advanced_sources={json.dumps(snapshot.not_advanced_sources, separators=(',', ':'))} "
            f"skew_sources={json.dumps(snapshot.skew_sources, separators=(',', ':'))} "
            f"source_worker_pids={worker_pids} "
            f"source_ipc_delay_ms={ipc_delays} "
            f"event_wall_ns={self._wall_time_ns()} "
            f"consecutive_skips={self.counters.consecutive_skips} "
            f"max_consecutive_skips={self.counters.max_consecutive_skips} "
            f"skipped={json.dumps(self.counters.skipped_packets(), sort_keys=True, separators=(',', ':'))}"
        )

    def _record_skip(self, reasons: tuple[str, ...]) -> None:
        self.counters.skipped_total += 1
        self.counters.consecutive_skips += 1
        self.counters.max_consecutive_skips = max(
            self.counters.max_consecutive_skips, self.counters.consecutive_skips
        )
        for reason in reasons:
            setattr(self.counters, f"skipped_{reason}", getattr(self.counters, f"skipped_{reason}") + 1)

    def attempt_send(self) -> bool:
        self.counters.attempted += 1
        next_seq = self.counters.recorded + 1
        gate = (
            runtime_event_gate(self.event_gate_path)
            if self.event_gate_path is not None
            else nullcontext()
        )
        with gate:
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
                skip_signature = (
                    snapshot.skip_reasons,
                    snapshot.stale_sources,
                    snapshot.not_advanced_sources,
                    snapshot.skew_sources,
                )
                if self.counters.consecutive_skips == 1 or skip_signature != self._last_skip_signature or (
                    self.print_every > 0 and self.counters.attempted % self.print_every == 0
                ):
                    self.printer(
                        "[orin ros state udp bridge] "
                        f"skip reasons={','.join(snapshot.skip_reasons)} {self._metrics_text(snapshot)}",
                        flush=True,
                    )
                self._last_skip_signature = skip_signature
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
            if self.audit_sink is not None:
                local_record_wall_ns = self._wall_time_ns()
                local_record_monotonic_ns = self._monotonic_ns()
                self.audit_sink(
                    "state",
                    {
                        "packet": snapshot.packet,
                        "udp_target_ip": self.target[0],
                        "udp_target_port": self.target[1],
                        "encoded_bytes": len(payload),
                        "local_record_wall_ns": local_record_wall_ns,
                        "local_record_monotonic_ns": local_record_monotonic_ns,
                    },
                )
            self.counters.recorded += 1
            self.last_sent_generations = snapshot.generations.copy()
            for name, stamp in snapshot.joint_header_stamps.items():
                if stamp is not None and stamp > 0:
                    self.last_sent_joint_header_stamps[name] = stamp
        try:
            self.sock.sendto(payload, self.target)
        except OSError as exc:
            self.counters.network_errors += 1
            self.printer(
                "[orin ros state udp bridge] WARNING "
                f"local_recorded seq={next_seq} udp_delivery_failed={type(exc).__name__}: {exc}",
                flush=True,
            )
            if self.audit_sink is None:
                raise
            return True
        send_completed_monotonic_ns = self._monotonic_ns()
        self._successful_send_times_ns.append(send_completed_monotonic_ns)
        self.counters.sent += 1
        recovered_after_skips = self.counters.consecutive_skips
        if (
            self.counters.sent == 1
            or recovered_after_skips > 0
            or (self.print_every > 0 and self.counters.sent % self.print_every == 0)
        ):
            self.printer(
                "[orin ros state udp bridge] "
                f"sent seq={next_seq} bytes={len(payload)} "
                f"recovered_after_skips={recovered_after_skips} {self._metrics_text(snapshot)}",
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
        self.counters.consecutive_skips = 0
        self._last_skip_signature = None
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
        while not should_stop() and (count <= 0 or self.counters.recorded < count):
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


def state_subscription_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


@dataclass(frozen=True)
class SourceWorkerSpec:
    name: str
    topic: str
    message_type: Any


@dataclass(frozen=True)
class SourceSubscriptionHandle:
    callback_group: Any
    subscription: Any


def create_source_worker_specs(robot_cfg: Any) -> tuple[SourceWorkerSpec, ...]:
    return (
        SourceWorkerSpec("left_joints", robot_cfg.left_joint_state_topic, JointState),
        SourceWorkerSpec("right_joints", robot_cfg.right_joint_state_topic, JointState),
        SourceWorkerSpec("left_gripper", robot_cfg.left_gripper_state_topic, Float64MultiArray),
        SourceWorkerSpec("right_gripper", robot_cfg.right_gripper_state_topic, Float64MultiArray),
    )


def create_source_subscription(
    node: Any,
    spec: SourceWorkerSpec,
    send_connection: Any,
    *,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    wall_time_ns: Callable[[], int] = time.time_ns,
) -> SourceSubscriptionHandle:
    callback_group = MutuallyExclusiveCallbackGroup()
    worker_pid = os.getpid()

    def callback(msg: Any) -> None:
        receive_monotonic_ns = monotonic_ns()
        receive_wall_ns = wall_time_ns()
        if spec.name.endswith("_joints"):
            values: Any = (tuple(msg.name), tuple(msg.position))
            header_stamp_ns: int | None = _header_stamp_ns(msg)
        else:
            values = tuple(msg.data)
            header_stamp_ns = None
        send_connection.send(
            (spec.name, values, receive_monotonic_ns, receive_wall_ns, header_stamp_ns, worker_pid)
        )

    subscription = node.create_subscription(
        spec.message_type,
        spec.topic,
        callback,
        state_subscription_qos(),
        callback_group=callback_group,
    )
    return SourceSubscriptionHandle(callback_group=callback_group, subscription=subscription)


def run_source_worker(
    spec: SourceWorkerSpec,
    send_connection: Any,
    stop_event: Any,
) -> None:
    signal.signal(signal.SIGINT, lambda _signum, _frame: stop_event.set())
    signal.signal(signal.SIGTERM, lambda _signum, _frame: stop_event.set())
    context = Context()
    context.init(args=None)
    node = rclpy.create_node(
        f"jz_readonly_ros_state_udp_bridge_{spec.name}",
        context=context,
    )
    executor = SingleThreadedExecutor(context=context)
    subscription_handle = create_source_subscription(node, spec, send_connection)
    executor.add_node(node)
    try:
        while not stop_event.is_set():
            executor.spin_once(timeout_sec=0.1)
    except ExternalShutdownException:
        pass
    finally:
        # Keep both handles alive until spinning has completely stopped.
        _ = subscription_handle
        executor.shutdown(timeout_sec=5.0)
        node.destroy_node()
        if context.ok():
            context.shutdown()
        send_connection.close()


class SourceProcessManager:
    def __init__(
        self,
        collector: ReadonlyStateCollector,
        specs: tuple[SourceWorkerSpec, ...],
        *,
        process_context: Any | None = None,
    ):
        if len(specs) != len(SOURCE_NAMES):
            raise ValueError(f"expected {len(SOURCE_NAMES)} source workers, got {len(specs)}")
        self.collector = collector
        self.specs = specs
        self.process_context = process_context or multiprocessing.get_context("spawn")
        self.stop_event = self.process_context.Event()
        connection_pairs = tuple(self.process_context.Pipe(duplex=False) for _ in specs)
        self.receive_connections = tuple(pair[0] for pair in connection_pairs)
        self._worker_connections = tuple(pair[1] for pair in connection_pairs)
        self.processes = tuple(
            self.process_context.Process(
                target=run_source_worker,
                args=(spec, worker_connection, self.stop_event),
                name=f"jz-ros-state-source-{spec.name}",
                daemon=False,
            )
            for spec, worker_connection in zip(specs, self._worker_connections, strict=True)
        )
        self._collector_stop = threading.Event()
        self._collector_exception: BaseException | None = None
        self._collector_thread = threading.Thread(
            target=self._collect_updates,
            name="jz-ros-state-ipc-collector",
            daemon=False,
        )

    @property
    def collector_thread(self) -> threading.Thread:
        return self._collector_thread

    def _collect_updates(self) -> None:
        active_connections = list(self.receive_connections)
        try:
            while active_connections and not self._collector_stop.is_set():
                for connection in wait_for_connections(active_connections, timeout=0.1):
                    try:
                        update = connection.recv()
                    except (EOFError, OSError):
                        active_connections.remove(connection)
                        continue
                    self.collector.apply_worker_update(update)
        except BaseException as exc:
            self._collector_exception = exc

    def start(self) -> None:
        self._collector_thread.start()
        try:
            for process, worker_connection in zip(
                self.processes, self._worker_connections, strict=True
            ):
                process.start()
                worker_connection.close()
        except BaseException:
            self.stop()
            raise

    def raise_if_failed(self) -> None:
        if self._collector_exception is not None:
            raise RuntimeError("ROS source IPC collector failed") from self._collector_exception
        if self._collector_thread.ident is not None and not self._collector_thread.is_alive():
            raise RuntimeError("ROS source IPC collector stopped unexpectedly")
        for process in self.processes:
            if process.exitcode is not None:
                raise RuntimeError(
                    f"ROS source process {process.name} stopped unexpectedly exitcode={process.exitcode}"
                )

    def stop(self, timeout_s: float = 5.0) -> None:
        self.stop_event.set()
        forced_processes = []
        for process in self.processes:
            if process.pid is None:
                continue
            process.join(timeout=timeout_s)
            if process.is_alive():
                forced_processes.append(process.name)
                process.terminate()
                process.join(timeout=timeout_s)
            if process.is_alive():
                process.kill()
                process.join(timeout=timeout_s)
        self._collector_stop.set()
        if self._collector_thread.ident is not None:
            self._collector_thread.join(timeout=timeout_s)
        for connection in (*self.receive_connections, *self._worker_connections):
            connection.close()
        if self._collector_thread.is_alive():
            raise RuntimeError("ROS source IPC collector thread did not stop within timeout")
        if self._collector_exception is not None:
            raise RuntimeError("ROS source IPC collector failed") from self._collector_exception
        if forced_processes:
            raise RuntimeError(f"ROS source processes required forced shutdown: {forced_processes}")


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
    parser.add_argument("--executor-threads", type=_positive_int, default=4)
    parser.add_argument("--wait-timeout-s", type=_positive_float, default=10.0)
    parser.add_argument("--max-source-age-ms", type=_nonnegative_float, default=50.0)
    parser.add_argument("--max-source-skew-ms", type=_nonnegative_float, default=20.0)
    parser.add_argument(
        "--audit-socket",
        default=None,
        help="Optional Orin Session Unix event socket. Delivery is required when configured.",
    )
    parser.add_argument(
        "--ready-file",
        default=None,
        help="Optional atomic marker written only after all four State sources are ready.",
    )
    parser.add_argument("--event-gate-file")
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
    if args.executor_threads != len(SOURCE_NAMES):
        raise ValueError(
            f"The state bridge requires exactly {len(SOURCE_NAMES)} independent source executors"
        )
    robot_cfg = load_robot_config(args.robot_config)
    if not robot_cfg.use_gripper:
        raise ValueError("The 18D state bridge requires both arm and gripper state sources")
    collector = ReadonlyStateCollector(robot_cfg)

    sock: socket.socket | None = None

    source_processes = SourceProcessManager(
        collector,
        create_source_worker_specs(robot_cfg),
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
            f"executor=per_source_process_single_threaded:{len(source_processes.processes)} "
            f"contexts={len(source_processes.processes)} "
            f"callback_groups=mutually_exclusive:{len(source_processes.processes)} "
            "qos=keep_last:1,best_effort,volatile",
            flush=True,
        )

        source_processes.start()
        deadline = time.monotonic() + args.wait_timeout_s
        while not STOP_REQUESTED and not collector.ready() and time.monotonic() < deadline:
            source_processes.raise_if_failed()
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
            audit_sink=(
                UnixEventClient(args.audit_socket, required=True).emit if args.audit_socket else None
            ),
            event_gate_path=args.event_gate_file,
        )
        publish_ready(
            args.ready_file,
            role="state_bridge",
            details={
                "target_ip": args.target_ip,
                "target_port": args.target_port,
                "hz": args.hz,
                "source_count": len(SOURCE_NAMES),
                "read_only": True,
            },
        )
        sender.run(
            should_stop=lambda: STOP_REQUESTED,
            count=args.count,
            health_check=source_processes.raise_if_failed,
        )
        if not STOP_REQUESTED:
            source_processes.raise_if_failed()
        return 0
    finally:
        remove_ready(args.ready_file)
        if sock is not None:
            sock.close()
        source_processes.stop()


if __name__ == "__main__":
    raise SystemExit(main())
