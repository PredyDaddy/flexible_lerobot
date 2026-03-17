#!/usr/bin/env python

"""Bounded-async remote GR00T robot client for AgileX.

Phase 4 keeps the client in dry-consume mode by default:
1. read live AgileX observations
2. maintain one local action queue plus one inflight request
3. receive remote action chunks asynchronously
4. build the merged robot action that would be sent
5. do not actuate the robot unless a later phase explicitly enables it
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import socket
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if SCRIPT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, SCRIPT_DIR.as_posix())
PARENT_DIR = SCRIPT_DIR.parent
if PARENT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, PARENT_DIR.as_posix())

from lerobot.robots import make_robot_from_config
from lerobot.utils.robot_utils import precise_sleep

from remote_sync_common import (
    AUTO_ARM,
    DEFAULT_RECONNECT_RETRIES,
    DEFAULT_RECONNECT_RETRY_DELAY_S,
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    DEFAULT_SOCKET_TIMEOUT_S,
    AgileXPolicyContract,
    RemoteGrootPolicyConfig,
    RemoteMessage,
    build_agilex_hold_action,
    build_agilex_policy_observation_frame,
    build_agilex_policy_runtime_helpers,
    build_agilex_robot_config,
    configure_logging,
    decode_agilex_policy_action,
    env_bool,
    merge_agilex_action,
    open_client_socket,
    parse_bool,
    receive_message,
    send_message,
    summarize_observation,
    summarize_policy_contract,
    validate_live_observation,
)


DEFAULT_ASYNC_LOW_WATERMARK = 1
DEFAULT_ASYNC_CHUNK_SIZE = 2
DEFAULT_RESPONSE_TIMEOUT_S = 0.75
DEFAULT_RESET_ACK_TIMEOUT_S = 1.5
DEFAULT_CLOSE_ACK_TIMEOUT_S = 1.0
DEFAULT_MAX_CONSECUTIVE_TIMEOUTS = 2
DEFAULT_METRICS_OUTPUT_PATH = SCRIPT_DIR / "reports" / "async_client_metrics.json"


def normalize_optional_path(path_str: str | None) -> Path | None:
    if path_str in {None, "", "none", "null"}:
        return None
    return Path(path_str).expanduser()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(max(int(round((len(ordered) - 1) * q)), 0), len(ordered) - 1)
    return float(ordered[index])


def now_wall() -> float:
    return time.time()


def now_mono() -> float:
    return time.perf_counter()


def build_session_id() -> str:
    return f"agilex-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"


class AsyncProtocolAdapter:
    """Runtime adapter for `remote_async_common.py`.

    The common module is intentionally imported lazily so this file can still
    be syntax-checked before the protocol module lands.
    """

    def __init__(self) -> None:
        self._module: Any | None = None

    @property
    def module(self) -> Any:
        if self._module is None:
            try:
                self._module = importlib.import_module("remote_async_common")
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "remote_async_common.py is required for async client execution. "
                    "Finish Phase 1 before running the async client."
                ) from exc
        return self._module

    def _payload_class(self, name: str) -> type[Any]:
        payload_cls = getattr(self.module, name, None)
        if payload_cls is None:
            raise AttributeError(f"remote_async_common is missing payload class {name}")
        return payload_cls

    @staticmethod
    def _instantiate(payload_cls: type[Any], candidate_fields: dict[str, Any]) -> Any:
        signature = inspect.signature(payload_cls)
        filtered = {
            name: value
            for name, value in candidate_fields.items()
            if name in signature.parameters
        }
        missing = [
            name
            for name, parameter in signature.parameters.items()
            if parameter.default is inspect.Signature.empty and name not in filtered
        ]
        if missing:
            raise TypeError(
                f"Cannot instantiate {payload_cls.__name__}; missing required fields {missing}. "
                f"Available candidate fields: {sorted(candidate_fields)}"
            )
        return payload_cls(**filtered)

    def make_hello(
        self,
        *,
        session_id: str,
        setup: RemoteGrootPolicyConfig,
        control_fps: int,
        low_watermark: int,
        chunk_size: int,
        response_timeout_s: float,
        dry_consume: bool,
    ) -> Any:
        payload_cls = self._payload_class("AsyncHello")
        return self._instantiate(
            payload_cls,
            {
                "session_id": session_id,
                "setup": setup,
                "policy_setup": setup,
                "control_fps": control_fps,
                "fps": control_fps,
                "low_watermark": low_watermark,
                "chunk_size": chunk_size,
                "response_timeout_s": response_timeout_s,
                "dry_consume": dry_consume,
                "created_at": now_wall(),
                "client_name": "agilex_async_robot_client",
            },
        )

    def make_infer_request(
        self,
        *,
        request_id: int,
        session_id: str,
        capture_ts: float,
        must_go: bool,
        queue_size_at_capture: int,
        latest_action_id_at_capture: int,
        observation: dict[str, Any],
    ) -> Any:
        payload_cls = self._payload_class("AsyncInferRequest")
        return self._instantiate(
            payload_cls,
            {
                "request_id": request_id,
                "session_id": session_id,
                "capture_ts": capture_ts,
                "must_go": must_go,
                "queue_size_at_capture": queue_size_at_capture,
                "latest_action_id_at_capture": latest_action_id_at_capture,
                "observation": observation,
            },
        )

    def make_reset(self, *, session_id: str, reason: str) -> Any:
        payload_cls = self._payload_class("AsyncReset")
        return self._instantiate(
            payload_cls,
            {
                "session_id": session_id,
                "reason": reason,
                "requested_at": now_wall(),
            },
        )

    def make_close(self, *, session_id: str, reason: str) -> Any:
        payload_cls = self._payload_class("AsyncClose")
        return self._instantiate(
            payload_cls,
            {
                "session_id": session_id,
                "reason": reason,
                "requested_at": now_wall(),
            },
        )


@dataclass
class AsyncRobotClientConfig:
    robot_id: str
    control_mode: str
    state_left_topic: str
    state_right_topic: str
    command_left_topic: str
    command_right_topic: str
    front_camera_topic: str
    left_camera_topic: str
    right_camera_topic: str
    observation_timeout_s: float
    queue_size: int
    image_height: int
    image_width: int
    fps: int
    task: str
    robot_type: str
    policy_path: str | None
    backend: str
    trt_engine_path: str | None
    vit_dtype: str
    llm_dtype: str
    dit_dtype: str
    trt_action_head_only: bool
    policy_device: str | None
    control_arm: str
    server_host: str
    server_port: int
    socket_timeout_s: float
    reconnect_retries: int
    reconnect_retry_delay_s: float
    run_time_s: float
    log_interval: int
    log_level: str
    low_watermark: int
    chunk_size: int
    response_timeout_s: float
    reset_ack_timeout_s: float
    close_ack_timeout_s: float
    max_consecutive_timeouts: int
    dry_consume: bool
    reset_on_connect: bool
    metrics_output_path: Path | None
    dry_run: bool = False


@dataclass
class PendingAsyncRequest:
    request_id: int
    session_id: str
    capture_ts: float
    sent_at_wall: float
    sent_at_mono: float
    deadline_mono: float
    queue_size_at_capture: int
    latest_action_id_at_capture: int
    must_go: bool
    local_timeout_count: int = 0


@dataclass
class BufferedAction:
    action_id: int
    request_id: int
    capture_ts: float
    enqueued_at: float
    predicted_action: dict[str, float]


@dataclass
class LatestObservationCache:
    captured_at: float
    raw_observation: dict[str, Any]
    policy_observation_frame: dict[str, Any]


@dataclass
class AsyncClientMetrics:
    ack_count: int = 0
    retry_count: int = 0
    abort_count: int = 0
    reset_count: int = 0
    close_ack_count: int = 0
    error_count: int = 0
    hold_cycles: int = 0
    control_cycles: int = 0
    send_attempts: int = 0
    send_failures: int = 0
    pending_request_timeout_count: int = 0
    late_response_drop_count: int = 0
    foreign_session_drop_count: int = 0
    duplicate_response_drop_count: int = 0
    must_go_count: int = 0
    executed_action_count: int = 0
    request_send_latency_ms: list[float] = field(default_factory=list)
    round_trip_latency_ms: list[float] = field(default_factory=list)
    action_age_at_execution_ms: list[float] = field(default_factory=list)
    queue_size_before_send: list[int] = field(default_factory=list)
    queue_size_after_ack: list[int] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        stale_or_drop = (
            self.late_response_drop_count
            + self.foreign_session_drop_count
            + self.duplicate_response_drop_count
        )
        control_cycles = max(self.control_cycles, 1)
        return {
            "ack_count": self.ack_count,
            "retry_count": self.retry_count,
            "abort_count": self.abort_count,
            "reset_count": self.reset_count,
            "close_ack_count": self.close_ack_count,
            "error_count": self.error_count,
            "hold_cycles": self.hold_cycles,
            "control_cycles": self.control_cycles,
            "send_attempts": self.send_attempts,
            "send_failures": self.send_failures,
            "pending_request_timeout_count": self.pending_request_timeout_count,
            "late_response_drop_count": self.late_response_drop_count,
            "foreign_session_drop_count": self.foreign_session_drop_count,
            "duplicate_response_drop_count": self.duplicate_response_drop_count,
            "must_go_count": self.must_go_count,
            "executed_action_count": self.executed_action_count,
            "queue_underflow_rate": self.hold_cycles / float(control_cycles),
            "response_stale_or_drop_rate": stale_or_drop / float(max(self.ack_count + stale_or_drop, 1)),
            "request_send_latency_ms": _summarize_distribution(self.request_send_latency_ms),
            "round_trip_latency_ms": _summarize_distribution(self.round_trip_latency_ms),
            "action_age_at_execution_ms": _summarize_distribution(self.action_age_at_execution_ms),
            "queue_size_before_send": _summarize_distribution(
                [float(value) for value in self.queue_size_before_send]
            ),
            "queue_size_after_ack": _summarize_distribution(
                [float(value) for value in self.queue_size_after_ack]
            ),
        }


def _summarize_distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": float(len(values)),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "mean": sum(values) / float(len(values)),
    }


class RequestIdSequence:
    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            self._value += 1
            return self._value


class ActionIdSequence:
    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            self._value += 1
            return self._value

    @property
    def latest(self) -> int:
        with self._lock:
            return self._value


class LoopMetrics:
    def __init__(self) -> None:
        self._started_at = 0.0
        self._steps = 0

    def tick(self) -> float:
        if self._started_at == 0.0:
            self._started_at = now_mono()
        self._steps += 1
        elapsed = max(now_mono() - self._started_at, 1e-6)
        return self._steps / elapsed


class BoundedAsyncRemotePolicyClient:
    def __init__(self, config: AsyncRobotClientConfig, logger) -> None:
        self.config = config
        self.logger = logger
        self.protocol = AsyncProtocolAdapter()
        self.sock: socket.socket | None = None
        self.ready_payload: Any = None
        self.policy_contract: AgileXPolicyContract | None = None
        self.session_id = build_session_id()
        self.pending_request: PendingAsyncRequest | None = None
        self.latest_observation_cache: LatestObservationCache | None = None
        self.action_queue: deque[BufferedAction] = deque()
        self.metrics = AsyncClientMetrics()
        self.control_fps = LoopMetrics()
        self.request_ids = RequestIdSequence()
        self.action_ids = ActionIdSequence()
        self.last_executed_action_id = -1
        self.latest_send_debug_payload: dict[str, Any] | None = None
        self._receiver_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._send_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._close_ack_event = threading.Event()
        self._reset_ack_event = threading.Event()
        self._connection_error: Exception | None = None
        self._terminal_request_ids: deque[int] = deque(maxlen=256)
        self._terminal_request_set: set[int] = set()
        self._consecutive_timeouts = 0
        self.setup = RemoteGrootPolicyConfig(
            policy_path=config.policy_path,
            task=config.task,
            robot_type=config.robot_type,
            backend=config.backend,
            trt_engine_path=config.trt_engine_path,
            vit_dtype=config.vit_dtype,
            llm_dtype=config.llm_dtype,
            dit_dtype=config.dit_dtype,
            trt_action_head_only=config.trt_action_head_only,
            policy_device=config.policy_device,
            control_arm=config.control_arm,
            reset_policy_state=True,
        )

    def start(self) -> None:
        self._stop_event.clear()
        self.connect()
        if self.config.reset_on_connect:
            self.reset_runtime("startup_reset", reconnect_on_failure=False)

    def connect(self) -> None:
        if self.sock is not None:
            return

        attempts = max(1, self.config.reconnect_retries)
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            sock: socket.socket | None = None
            try:
                self.session_id = build_session_id()
                self._close_ack_event.clear()
                self._reset_ack_event.clear()
                sock = open_client_socket(
                    host=self.config.server_host,
                    port=self.config.server_port,
                    timeout_s=self.config.socket_timeout_s,
                )
                hello_payload = self.protocol.make_hello(
                    session_id=self.session_id,
                    setup=self.setup,
                    control_fps=self.config.fps,
                    low_watermark=self.config.low_watermark,
                    chunk_size=self.config.chunk_size,
                    response_timeout_s=self.config.response_timeout_s,
                    dry_consume=self.config.dry_consume,
                )
                send_message(sock, RemoteMessage(kind="hello", payload=hello_payload))
                reply = receive_message(sock)
                if not isinstance(reply, RemoteMessage):
                    raise TypeError(f"Unexpected handshake response type: {type(reply)}")
                if reply.kind == "error":
                    raise RuntimeError(self._format_error_payload(reply.payload))
                if reply.kind != "ready":
                    raise RuntimeError(f"Unexpected handshake response kind: {reply.kind}")

                policy_contract = getattr(reply.payload, "policy_contract", None)
                if not isinstance(policy_contract, AgileXPolicyContract):
                    raise TypeError(
                        "Async ready payload is missing AgileXPolicyContract, got "
                        f"{type(policy_contract)}"
                    )

                self.sock = sock
                self.ready_payload = reply.payload
                self.policy_contract = policy_contract
                self._connection_error = None
                self._receiver_thread = threading.Thread(
                    target=self._receiver_loop,
                    name="agilex-async-response-receiver",
                    daemon=True,
                )
                self._receiver_thread.start()
                self.logger.info(
                    "Connected to async server | session_id=%s | server=%s:%s | %s",
                    self.session_id,
                    self.config.server_host,
                    self.config.server_port,
                    summarize_policy_contract(policy_contract),
                )
                return
            except Exception as exc:
                last_error = exc
                self.logger.warning(
                    "Async connect attempt %s/%s failed: %s",
                    attempt,
                    attempts,
                    exc,
                )
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass
                if attempt < attempts:
                    time.sleep(max(0.0, self.config.reconnect_retry_delay_s))

        assert last_error is not None
        raise last_error

    def stop(self) -> None:
        self._stop_event.set()
        self.close(send_close=True, reason="client_shutdown")

    def close(self, *, send_close: bool, reason: str) -> None:
        self._clear_request_and_queue_state()
        self._consecutive_timeouts = 0
        sock = self.sock
        self.sock = None
        if sock is None:
            return

        if send_close:
            try:
                close_payload = self.protocol.make_close(session_id=self.session_id, reason=reason)
                self._close_ack_event.clear()
                with self._send_lock:
                    send_message(sock, RemoteMessage(kind="close", payload=close_payload))
                _ = self._close_ack_event.wait(timeout=self.config.close_ack_timeout_s)
            except Exception as exc:
                self.logger.debug("Ignoring close handshake failure: %s", exc)

        try:
            sock.close()
        finally:
            if self._receiver_thread is not None and self._receiver_thread.is_alive():
                self._receiver_thread.join(timeout=0.2)
            self._receiver_thread = None
            self.ready_payload = None
            self.policy_contract = None
            self.latest_send_debug_payload = None

    def reset_runtime(self, reason: str, *, reconnect_on_failure: bool) -> None:
        self.metrics.reset_count += 1
        self._clear_request_and_queue_state()
        if self.sock is None:
            self.connect()
            return

        reset_ack_received = False
        try:
            payload = self.protocol.make_reset(session_id=self.session_id, reason=reason)
            self._reset_ack_event.clear()
            with self._send_lock:
                send_message(self.sock, RemoteMessage(kind="reset", payload=payload))
            reset_ack_received = self._reset_ack_event.wait(timeout=self.config.reset_ack_timeout_s)
            if reset_ack_received:
                self.logger.info("Reset acknowledged by server | session_id=%s | reason=%s", self.session_id, reason)
                return
            self.logger.warning("Reset ack timed out | session_id=%s | reason=%s", self.session_id, reason)
        except Exception as exc:
            self.logger.warning("Reset failed, will reconnect | reason=%s | error=%s", reason, exc)

        if reconnect_on_failure or not reset_ack_received:
            self.close(send_close=False, reason=f"reset_reconnect:{reason}")
            self.connect()

    def send_infer_request(self, observation_frame: dict[str, Any]) -> None:
        if self.sock is None:
            self.connect()

        with self._state_lock:
            queue_size = len(self.action_queue)
            if self.pending_request is not None or queue_size > self.config.low_watermark:
                return

            request_id = self.request_ids.next()
            capture_ts = now_wall()
            must_go = queue_size == 0
            pending_request = PendingAsyncRequest(
                request_id=request_id,
                session_id=self.session_id,
                capture_ts=capture_ts,
                sent_at_wall=capture_ts,
                sent_at_mono=now_mono(),
                deadline_mono=now_mono() + self.config.response_timeout_s,
                queue_size_at_capture=queue_size,
                latest_action_id_at_capture=self.action_ids.latest,
                must_go=must_go,
            )
            self.pending_request = pending_request
            self.metrics.send_attempts += 1
            self.metrics.queue_size_before_send.append(queue_size)
            if must_go:
                self.metrics.must_go_count += 1

        request_payload = self.protocol.make_infer_request(
            request_id=pending_request.request_id,
            session_id=pending_request.session_id,
            capture_ts=pending_request.capture_ts,
            must_go=pending_request.must_go,
            queue_size_at_capture=pending_request.queue_size_at_capture,
            latest_action_id_at_capture=pending_request.latest_action_id_at_capture,
            observation=observation_frame,
        )
        self.latest_send_debug_payload = {
            "request_id": pending_request.request_id,
            "session_id": pending_request.session_id,
            "must_go": pending_request.must_go,
            "queue_size_at_capture": pending_request.queue_size_at_capture,
            "latest_action_id_at_capture": pending_request.latest_action_id_at_capture,
        }

        try:
            send_started = now_mono()
            assert self.sock is not None
            with self._send_lock:
                send_message(self.sock, RemoteMessage(kind="infer_request", payload=request_payload))
            self.metrics.request_send_latency_ms.append((now_mono() - send_started) * 1000.0)
        except Exception:
            with self._state_lock:
                self.pending_request = None
                self.metrics.send_failures += 1
            raise

    def handle_watchdog(self) -> None:
        pending: PendingAsyncRequest | None
        with self._state_lock:
            pending = self.pending_request
        if pending is None:
            return

        if now_mono() <= pending.deadline_mono:
            return

        with self._state_lock:
            current_pending = self.pending_request
            if current_pending is None or current_pending.request_id != pending.request_id:
                return
            self.pending_request = None
            self.metrics.pending_request_timeout_count += 1
            self._mark_request_terminal(current_pending.request_id)
            self._consecutive_timeouts += 1

        if self._consecutive_timeouts < self.config.max_consecutive_timeouts and self.sock is not None:
            self.metrics.retry_count += 1
            self.logger.warning(
                "Pending request timed out, allowing bounded retry | request_id=%s | session_id=%s",
                pending.request_id,
                pending.session_id,
            )
            return

        self.metrics.abort_count += 1
        self.logger.error(
            "Pending request timed out beyond budget, aborting session | request_id=%s | session_id=%s",
            pending.request_id,
            pending.session_id,
        )
        self._consecutive_timeouts = 0
        self.reset_runtime("client_watchdog_abort", reconnect_on_failure=True)

    def pop_action(self) -> BufferedAction | None:
        with self._state_lock:
            if not self.action_queue:
                return None
            return self.action_queue.popleft()

    def queue_size(self) -> int:
        with self._state_lock:
            return len(self.action_queue)

    def update_latest_observation(self, cache: LatestObservationCache) -> None:
        with self._state_lock:
            self.latest_observation_cache = cache

    def _receiver_loop(self) -> None:
        try:
            assert self.sock is not None
            while not self._stop_event.is_set() and self.sock is not None:
                message = receive_message(self.sock)
                if not isinstance(message, RemoteMessage):
                    raise TypeError(f"Unexpected async message type: {type(message)}")
                self._handle_remote_message(message)
        except Exception as exc:
            if not self._stop_event.is_set():
                self._connection_error = exc
                self.logger.warning("Async receiver loop terminated: %s", exc)
        finally:
            self.logger.debug("Async receiver loop exited")

    def _handle_remote_message(self, message: RemoteMessage) -> None:
        if message.kind == "infer_response":
            self._handle_infer_response(message.payload)
            return
        if message.kind == "reset_ack":
            self._reset_ack_event.set()
            return
        if message.kind == "close_ack":
            self.metrics.close_ack_count += 1
            self._close_ack_event.set()
            return
        if message.kind == "error":
            self.metrics.error_count += 1
            self.logger.error("Async server returned error: %s", self._format_error_payload(message.payload))
            return

        self.logger.warning("Ignoring unexpected async message kind=%s", message.kind)

    def _handle_infer_response(self, payload: Any) -> None:
        response_session_id = getattr(payload, "session_id", None)
        response_request_id = getattr(payload, "request_id", None)
        status = str(getattr(payload, "status", "ack")).lower()

        with self._state_lock:
            if response_session_id != self.session_id:
                self.metrics.foreign_session_drop_count += 1
                return

            pending = self.pending_request
            if pending is None:
                if response_request_id in self._terminal_request_set:
                    self.metrics.duplicate_response_drop_count += 1
                else:
                    self.metrics.late_response_drop_count += 1
                return

            if response_request_id != pending.request_id:
                if response_request_id in self._terminal_request_set:
                    self.metrics.duplicate_response_drop_count += 1
                else:
                    self.metrics.late_response_drop_count += 1
                return

            self.pending_request = None
            self._mark_request_terminal(pending.request_id)

        round_trip_ms = max(now_wall() - pending.sent_at_wall, 0.0) * 1000.0
        self.metrics.round_trip_latency_ms.append(round_trip_ms)

        if status == "ack":
            self.metrics.ack_count += 1
            self._consecutive_timeouts = 0
            self._enqueue_actions_from_response(payload, pending)
            return

        if status == "retry":
            self.metrics.retry_count += 1
            self.logger.warning(
                "Server requested retry | request_id=%s | session_id=%s | reason=%s",
                response_request_id,
                response_session_id,
                getattr(payload, "reason", None),
            )
            return

        self.metrics.abort_count += 1
        self.logger.error(
            "Server aborted request | request_id=%s | session_id=%s | reason=%s",
            response_request_id,
            response_session_id,
            getattr(payload, "reason", None),
        )
        self.reset_runtime("server_abort", reconnect_on_failure=True)

    def _enqueue_actions_from_response(self, payload: Any, pending: PendingAsyncRequest) -> None:
        if self.policy_contract is None:
            raise RuntimeError("Policy contract is not available while decoding async response")

        actions = getattr(payload, "actions", None)
        if actions is None:
            raise ValueError("Async infer_response is missing actions")

        decoded_actions = list(self._iter_decoded_actions(actions))
        with self._state_lock:
            for predicted_action in decoded_actions:
                self.action_queue.append(
                    BufferedAction(
                        action_id=self.action_ids.next(),
                        request_id=pending.request_id,
                        capture_ts=pending.capture_ts,
                        enqueued_at=now_wall(),
                        predicted_action=predicted_action,
                    )
                )
            self.metrics.queue_size_after_ack.append(len(self.action_queue))

    def _iter_decoded_actions(self, actions: Any) -> Iterable[dict[str, float]]:
        import torch

        if isinstance(actions, list):
            for action in actions:
                yield decode_agilex_policy_action(action, self.policy_contract)
            return

        action_tensor = torch.as_tensor(actions, dtype=torch.float32).detach().cpu()
        if action_tensor.ndim == 3:
            action_tensor = action_tensor[0]
        if action_tensor.ndim != 2:
            raise ValueError(f"Expected actions to have ndim=2 or ndim=3, got shape={tuple(action_tensor.shape)}")
        for index in range(action_tensor.shape[0]):
            yield decode_agilex_policy_action(action_tensor[index], self.policy_contract)

    def _mark_request_terminal(self, request_id: int) -> None:
        self._terminal_request_ids.append(request_id)
        self._terminal_request_set = set(self._terminal_request_ids)

    def _clear_request_and_queue_state(self) -> None:
        with self._state_lock:
            self.action_queue.clear()
            self.pending_request = None
            self.latest_observation_cache = None
            self._terminal_request_ids.clear()
            self._terminal_request_set.clear()

    @staticmethod
    def _format_error_payload(payload: Any) -> str:
        message = getattr(payload, "message", repr(payload))
        traceback = getattr(payload, "traceback", None)
        if traceback:
            return f"{message}\n{traceback}"
        return str(message)

    @property
    def connection_error(self) -> Exception | None:
        return self._connection_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded-async remote GR00T robot client for AgileX.")
    parser.add_argument("--robot-id", default=os.getenv("ROBOT_ID", "my_agilex"))
    parser.add_argument(
        "--control-mode",
        choices=("passive_follow", "command_master"),
        default=os.getenv("CONTROL_MODE", "passive_follow"),
    )
    parser.add_argument("--state-left-topic", default=os.getenv("STATE_LEFT_TOPIC", "/puppet/joint_left"))
    parser.add_argument("--state-right-topic", default=os.getenv("STATE_RIGHT_TOPIC", "/puppet/joint_right"))
    parser.add_argument("--command-left-topic", default=os.getenv("COMMAND_LEFT_TOPIC", "/master/joint_left"))
    parser.add_argument("--command-right-topic", default=os.getenv("COMMAND_RIGHT_TOPIC", "/master/joint_right"))
    parser.add_argument(
        "--front-camera-topic",
        default=os.getenv("FRONT_CAMERA_TOPIC", "/camera_f/color/image_raw"),
    )
    parser.add_argument("--left-camera-topic", default=os.getenv("LEFT_CAMERA_TOPIC", "/camera_l/color/image_raw"))
    parser.add_argument("--right-camera-topic", default=os.getenv("RIGHT_CAMERA_TOPIC", "/camera_r/color/image_raw"))
    parser.add_argument(
        "--observation-timeout-s",
        type=float,
        default=float(os.getenv("OBSERVATION_TIMEOUT_S", "2.0")),
    )
    parser.add_argument("--queue-size", type=int, default=int(os.getenv("QUEUE_SIZE", "1")))
    parser.add_argument("--image-height", type=int, default=int(os.getenv("IMAGE_HEIGHT", "480")))
    parser.add_argument("--image-width", type=int, default=int(os.getenv("IMAGE_WIDTH", "640")))
    parser.add_argument("--fps", type=int, default=int(os.getenv("FPS", "30")))

    parser.add_argument(
        "--task",
        default=os.getenv("DATASET_TASK", "Execute the trained AgileX GR00T task"),
    )
    parser.add_argument("--robot-type", default=os.getenv("ROBOT_TYPE", "agilex"))
    parser.add_argument(
        "--policy-path",
        default=os.getenv("REMOTE_POLICY_PATH", os.getenv("POLICY_PATH")),
        help="Checkpoint path on the server machine. Leave empty to use the server default.",
    )
    parser.add_argument("--backend", default=os.getenv("INFER_BACKEND", "pytorch"), choices=["pytorch", "tensorrt"])
    parser.add_argument("--trt-engine-path", default=os.getenv("TRT_ENGINE_PATH"))
    parser.add_argument("--vit-dtype", default=os.getenv("TRT_VIT_DTYPE", "fp16"), choices=["fp16", "fp8"])
    parser.add_argument(
        "--llm-dtype",
        default=os.getenv("TRT_LLM_DTYPE", "fp16"),
        choices=["fp16", "nvfp4", "fp8", "nvfp4_full"],
    )
    parser.add_argument("--dit-dtype", default=os.getenv("TRT_DIT_DTYPE", "fp16"), choices=["fp16", "fp8"])
    parser.add_argument(
        "--trt-action-head-only",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("TRT_ACTION_HEAD_ONLY", False),
    )
    parser.add_argument(
        "--policy-device",
        default=os.getenv("POLICY_DEVICE_OVERRIDE", os.getenv("POLICY_DEVICE")),
        help="Optional server-side device override forwarded during async hello.",
    )
    parser.add_argument(
        "--control-arm",
        default=os.getenv("CONTROL_ARM", AUTO_ARM),
        help="Requested AgileX control scope for this client: auto/left/right/both.",
    )

    parser.add_argument("--server-host", default=os.getenv("REMOTE_GROOT_SERVER_HOST", DEFAULT_SERVER_HOST))
    parser.add_argument(
        "--server-port",
        type=int,
        default=int(os.getenv("REMOTE_GROOT_SERVER_PORT", str(DEFAULT_SERVER_PORT))),
    )
    parser.add_argument(
        "--socket-timeout-s",
        type=float,
        default=float(os.getenv("REMOTE_GROOT_SOCKET_TIMEOUT_S", str(DEFAULT_SOCKET_TIMEOUT_S))),
    )
    parser.add_argument(
        "--reconnect-retries",
        type=int,
        default=int(os.getenv("REMOTE_GROOT_RECONNECT_RETRIES", str(DEFAULT_RECONNECT_RETRIES))),
    )
    parser.add_argument(
        "--reconnect-retry-delay-s",
        type=float,
        default=float(
            os.getenv("REMOTE_GROOT_RECONNECT_RETRY_DELAY_S", str(DEFAULT_RECONNECT_RETRY_DELAY_S))
        ),
    )

    parser.add_argument("--low-watermark", type=int, default=int(os.getenv("ASYNC_LOW_WATERMARK", "1")))
    parser.add_argument("--chunk-size", type=int, default=int(os.getenv("ASYNC_CHUNK_SIZE", "2")))
    parser.add_argument(
        "--response-timeout-s",
        type=float,
        default=float(os.getenv("ASYNC_RESPONSE_TIMEOUT_S", str(DEFAULT_RESPONSE_TIMEOUT_S))),
    )
    parser.add_argument(
        "--reset-ack-timeout-s",
        type=float,
        default=float(os.getenv("ASYNC_RESET_ACK_TIMEOUT_S", str(DEFAULT_RESET_ACK_TIMEOUT_S))),
    )
    parser.add_argument(
        "--close-ack-timeout-s",
        type=float,
        default=float(os.getenv("ASYNC_CLOSE_ACK_TIMEOUT_S", str(DEFAULT_CLOSE_ACK_TIMEOUT_S))),
    )
    parser.add_argument(
        "--max-consecutive-timeouts",
        type=int,
        default=int(os.getenv("ASYNC_MAX_CONSECUTIVE_TIMEOUTS", str(DEFAULT_MAX_CONSECUTIVE_TIMEOUTS))),
    )
    parser.add_argument(
        "--dry-consume",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("ASYNC_DRY_CONSUME", True),
        help="Phase 4 default. Consume the queue and build the merged action, but do not actuate the robot.",
    )
    parser.add_argument(
        "--reset-on-connect",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("ASYNC_RESET_ON_CONNECT", True),
    )
    parser.add_argument(
        "--metrics-output-path",
        default=os.getenv("ASYNC_METRICS_OUTPUT_PATH", str(DEFAULT_METRICS_OUTPUT_PATH)),
    )

    parser.add_argument("--run-time-s", type=float, default=float(os.getenv("RUN_TIME_S", "0")))
    parser.add_argument("--log-interval", type=int, default=int(os.getenv("LOG_INTERVAL", "30")))
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    parser.add_argument(
        "--dry-run",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("DRY_RUN", False),
    )
    return parser


def validate_config(config: AsyncRobotClientConfig) -> None:
    if config.low_watermark < 0:
        raise ValueError(f"low_watermark must be >= 0, got {config.low_watermark}")
    if config.chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {config.chunk_size}")
    if config.chunk_size < config.low_watermark + 1:
        raise ValueError(
            "chunk_size must be at least low_watermark + 1 per the frozen bounded-async sizing rule. "
            f"Got chunk_size={config.chunk_size}, low_watermark={config.low_watermark}."
        )
    if config.max_consecutive_timeouts < 1:
        raise ValueError("max_consecutive_timeouts must be >= 1")


def print_runtime_summary(config: AsyncRobotClientConfig) -> None:
    print(f"[INFO] Robot id: {config.robot_id}")
    print(f"[INFO] Control mode: {config.control_mode}")
    print(f"[INFO] Server: {config.server_host}:{config.server_port}")
    print(f"[INFO] control_arm: {config.control_arm}")
    print(f"[INFO] Backend: {config.backend}")
    print(f"[INFO] FPS: {config.fps}")
    print(
        "[INFO] Async policy: "
        f"low_watermark={config.low_watermark} chunk_size={config.chunk_size} "
        f"response_timeout_s={config.response_timeout_s:.3f} "
        f"max_consecutive_timeouts={config.max_consecutive_timeouts}"
    )
    print(
        f"[INFO] dry_consume={config.dry_consume} "
        "(Phase 4 default; merged action is produced but not actuated)"
    )
    print(f"[INFO] metrics_output_path: {config.metrics_output_path}")
    print(f"[INFO] run_time_s: {config.run_time_s} (<=0 means until Ctrl+C)")


def build_metrics_report(
    config: AsyncRobotClientConfig,
    client: BoundedAsyncRemotePolicyClient,
    avg_control_fps: float,
) -> dict[str, Any]:
    return {
        "config": {
            "server_host": config.server_host,
            "server_port": config.server_port,
            "fps": config.fps,
            "low_watermark": config.low_watermark,
            "chunk_size": config.chunk_size,
            "response_timeout_s": config.response_timeout_s,
            "dry_consume": config.dry_consume,
            "reset_on_connect": config.reset_on_connect,
        },
        "session_id": client.session_id,
        "avg_control_fps": avg_control_fps,
        "latest_send_debug_payload": client.latest_send_debug_payload,
        "metrics": client.metrics.summary(),
    }


def write_metrics_report(report: dict[str, Any], output_path: Path | None, logger) -> None:
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if output_path is None:
        logger.info("Async client metrics:\n%s", rendered)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered + "\n", encoding="utf-8")
    logger.info("Wrote async client metrics report to %s", output_path)


def run(config: AsyncRobotClientConfig) -> None:
    validate_config(config)
    logger = configure_logging("groot_agilex_async_robot_client", config.log_level)
    print_runtime_summary(config)
    if config.dry_run:
        print("[INFO] DRY_RUN=true, exit without execution.")
        return

    policy_client = BoundedAsyncRemotePolicyClient(config, logger)
    robot_cfg = build_agilex_robot_config(argparse.Namespace(**asdict(config)))
    robot = make_robot_from_config(robot_cfg)

    start_t = now_mono()
    end_t = start_t + config.run_time_s if config.run_time_s > 0 else None
    avg_control_fps = 0.0
    first_observation_logged = False
    robot_observation_processor = None
    observation_dataset_features = None

    try:
        policy_client.start()
        if policy_client.policy_contract is None:
            raise RuntimeError("Async handshake completed without a policy contract")
        robot_observation_processor, observation_dataset_features = build_agilex_policy_runtime_helpers(
            policy_client.policy_contract,
            image_height=config.image_height,
            image_width=config.image_width,
        )
        robot.connect()
        logger.info("AgileX connected for async dry-consume loop")

        step = 0
        while True:
            if end_t is not None and now_mono() >= end_t:
                logger.info("Reached requested run_time_s. Exiting async loop.")
                break

            if policy_client.connection_error is not None:
                logger.warning("Receiver reported connection loss, reconnecting: %s", policy_client.connection_error)
                policy_client.close(send_close=False, reason="receiver_connection_error")
                policy_client.connect()

            loop_started = now_mono()
            raw_observation = robot.get_observation()
            validate_live_observation(
                raw_observation,
                image_height=config.image_height,
                image_width=config.image_width,
            )
            if not first_observation_logged:
                logger.info("First observation summary | %s", summarize_observation(raw_observation))
                first_observation_logged = True

            assert robot_observation_processor is not None
            assert observation_dataset_features is not None
            policy_observation = build_agilex_policy_observation_frame(
                raw_observation,
                policy_client.policy_contract,
                observation_dataset_features,
                robot_observation_processor,
            )
            policy_client.update_latest_observation(
                LatestObservationCache(
                    captured_at=now_wall(),
                    raw_observation=raw_observation,
                    policy_observation_frame=policy_observation,
                )
            )

            policy_client.handle_watchdog()

            buffered_action = policy_client.pop_action()
            if buffered_action is None:
                policy_client.metrics.hold_cycles += 1
            else:
                hold_action = build_agilex_hold_action(raw_observation)
                merged_action = merge_agilex_action(hold_action, buffered_action.predicted_action)
                _ = merged_action
                policy_client.last_executed_action_id = buffered_action.action_id
                policy_client.metrics.executed_action_count += 1
                policy_client.metrics.action_age_at_execution_ms.append(
                    max(now_wall() - buffered_action.capture_ts, 0.0) * 1000.0
                )

            try:
                policy_client.send_infer_request(policy_observation)
            except Exception as exc:
                logger.warning("Failed to send infer_request, reconnecting: %s", exc)
                policy_client.close(send_close=False, reason="send_failure")
                policy_client.connect()

            policy_client.metrics.control_cycles += 1
            avg_control_fps = policy_client.control_fps.tick()
            if config.log_interval > 0 and step % config.log_interval == 0:
                logger.info(
                    "Async step=%s | queue=%s | pending=%s | avg_control_fps=%.2f | hold_cycles=%s | "
                    "ack=%s retry=%s abort=%s timeout=%s drops=%s",
                    step,
                    policy_client.queue_size(),
                    getattr(policy_client.pending_request, "request_id", None),
                    avg_control_fps,
                    policy_client.metrics.hold_cycles,
                    policy_client.metrics.ack_count,
                    policy_client.metrics.retry_count,
                    policy_client.metrics.abort_count,
                    policy_client.metrics.pending_request_timeout_count,
                    policy_client.metrics.late_response_drop_count
                    + policy_client.metrics.foreign_session_drop_count
                    + policy_client.metrics.duplicate_response_drop_count,
                )

            step += 1
            precise_sleep(max(1.0 / float(config.fps) - (now_mono() - loop_started), 0.0))
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Stopping async client.")
    finally:
        try:
            report = build_metrics_report(config, policy_client, avg_control_fps)
            write_metrics_report(report, config.metrics_output_path, logger)
        finally:
            policy_client.stop()
            if getattr(robot, "is_connected", False):
                robot.disconnect()


def main() -> None:
    args = build_parser().parse_args()
    config = AsyncRobotClientConfig(
        robot_id=args.robot_id,
        control_mode=args.control_mode,
        state_left_topic=args.state_left_topic,
        state_right_topic=args.state_right_topic,
        command_left_topic=args.command_left_topic,
        command_right_topic=args.command_right_topic,
        front_camera_topic=args.front_camera_topic,
        left_camera_topic=args.left_camera_topic,
        right_camera_topic=args.right_camera_topic,
        observation_timeout_s=args.observation_timeout_s,
        queue_size=args.queue_size,
        image_height=args.image_height,
        image_width=args.image_width,
        fps=args.fps,
        task=args.task,
        robot_type=args.robot_type,
        policy_path=args.policy_path,
        backend=args.backend,
        trt_engine_path=args.trt_engine_path,
        vit_dtype=args.vit_dtype,
        llm_dtype=args.llm_dtype,
        dit_dtype=args.dit_dtype,
        trt_action_head_only=args.trt_action_head_only,
        policy_device=args.policy_device,
        control_arm=args.control_arm,
        server_host=args.server_host,
        server_port=args.server_port,
        socket_timeout_s=args.socket_timeout_s,
        reconnect_retries=args.reconnect_retries,
        reconnect_retry_delay_s=args.reconnect_retry_delay_s,
        run_time_s=args.run_time_s,
        log_interval=args.log_interval,
        log_level=args.log_level,
        low_watermark=args.low_watermark,
        chunk_size=args.chunk_size,
        response_timeout_s=args.response_timeout_s,
        reset_ack_timeout_s=args.reset_ack_timeout_s,
        close_ack_timeout_s=args.close_ack_timeout_s,
        max_consecutive_timeouts=args.max_consecutive_timeouts,
        dry_consume=args.dry_consume,
        reset_on_connect=args.reset_on_connect,
        metrics_output_path=normalize_optional_path(args.metrics_output_path),
        dry_run=args.dry_run,
    )
    run(config)


if __name__ == "__main__":
    main()
