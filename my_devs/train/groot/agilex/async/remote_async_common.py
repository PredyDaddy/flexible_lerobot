#!/usr/bin/env python

from __future__ import annotations

import math
import sys
import time
import uuid
from collections.abc import Collection
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


def resolve_agilex_dir(script_path: Path) -> Path:
    resolved_path = script_path.resolve()
    for candidate in resolved_path.parents:
        if (candidate / "remote_sync_common.py").is_file():
            return candidate
    raise RuntimeError(f"Could not locate AgileX directory from script path: {script_path}")


AGILEX_DIR = resolve_agilex_dir(Path(__file__))
if AGILEX_DIR.as_posix() not in sys.path:
    sys.path.insert(0, AGILEX_DIR.as_posix())


from remote_sync_common import AgileXPolicyContract, RemoteGrootPolicyConfig, RemoteMessage  # noqa: E402

DEFAULT_PROTOCOL_VERSION = "async-v1"


class AsyncMessageKind(str, Enum):
    HELLO = "hello"
    READY = "ready"
    INFER_REQUEST = "infer_request"
    INFER_RESPONSE = "infer_response"
    RESET = "reset"
    RESET_ACK = "reset_ack"
    CLOSE = "close"
    CLOSE_ACK = "close_ack"
    ERROR = "error"


class AsyncInferStatus(str, Enum):
    ACK = "ack"
    RETRY = "retry"
    ABORT = "abort"


class AsyncTerminalSource(str, Enum):
    SERVER = "server"
    WATCHDOG = "watchdog"
    CLIENT = "client"


class AsyncDropReason(str, Enum):
    FOREIGN_SESSION = "foreign_session"
    NO_PENDING_REQUEST = "no_pending_request"
    REQUEST_ID_MISMATCH = "request_id_mismatch"
    DUPLICATE_TERMINAL = "duplicate_terminal"
    INVALID_STATUS = "invalid_status"


@dataclass(frozen=True)
class AsyncHello:
    session_id: str
    setup: RemoteGrootPolicyConfig | None = None
    policy_setup: RemoteGrootPolicyConfig | None = None
    control_fps: float | None = None
    fps: float | None = None
    low_watermark: int | None = None
    chunk_size: int | None = None
    response_timeout_s: float | None = None
    dry_consume: bool | None = None
    created_at: float = field(default_factory=time.time)
    client_name: str | None = None
    protocol_version: str = DEFAULT_PROTOCOL_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_setup(self) -> RemoteGrootPolicyConfig | None:
        return self.setup or self.policy_setup

    @property
    def requested_control_fps(self) -> float | None:
        return self.control_fps if self.control_fps is not None else self.fps

    @property
    def requested_low_watermark(self) -> int | None:
        return self.low_watermark

    @property
    def requested_chunk_size(self) -> int | None:
        return self.chunk_size

    @property
    def sent_at(self) -> float:
        return self.created_at


@dataclass(frozen=True)
class AsyncReady:
    session_id: str
    policy_contract: AgileXPolicyContract | None = None
    policy_path: str = ""
    task: str = ""
    robot_type: str = ""
    backend: str = ""
    policy_device: str = ""
    protocol_version: str = DEFAULT_PROTOCOL_VERSION
    accepted: bool = True
    message: str = "ready"
    negotiated_low_watermark: int | None = None
    negotiated_chunk_size: int | None = None
    max_returned_actions: int | None = None
    server_time: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AsyncInferRequest:
    request_id: int
    session_id: str
    capture_ts: float
    must_go: bool
    queue_size_at_capture: int
    latest_action_id_at_capture: int = -1
    observation: Any = None


@dataclass(frozen=True)
class AsyncInferResponse:
    request_id: int
    session_id: str
    status: str
    actions: list[Any]
    reason: str = ""
    server_received_at: float = 0.0
    server_sent_at: float = 0.0
    predict_latency_ms: float = 0.0
    queue_wait_ms: float = 0.0
    chunk_size: int | None = None
    preprocess_ms: float = 0.0
    infer_ms: float = 0.0
    postprocess_ms: float = 0.0
    serialize_ms: float = 0.0
    total_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_chunk_size(self) -> int:
        if self.chunk_size is not None:
            return self.chunk_size
        return len(self.actions)


@dataclass(frozen=True)
class AsyncReset:
    session_id: str
    reason: str
    requested_at: float = field(default_factory=time.time)
    new_session_id: str | None = None
    clear_runtime_state: bool = True

    @property
    def reset_started_at(self) -> float:
        return self.requested_at


@dataclass(frozen=True)
class AsyncResetAck:
    session_id: str
    reason: str
    accepted: bool = True
    reset_at: float = field(default_factory=time.time)
    new_session_id: str | None = None
    status: str = "ok"


@dataclass(frozen=True)
class AsyncClose:
    session_id: str
    reason: str = "client_close"
    requested_at: float = field(default_factory=time.time)

    @property
    def closed_at(self) -> float:
        return self.requested_at


@dataclass(frozen=True)
class AsyncCloseAck:
    session_id: str
    reason: str = "server_close_ack"
    accepted: bool = True
    closed_at: float = field(default_factory=time.time)
    status: str = "ok"


@dataclass(frozen=True)
class AsyncError:
    message: str
    session_id: str | None = None
    request_id: int | None = None
    traceback: str | None = None
    recoverable: bool = False
    emitted_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class QueueMetrics:
    queue_size: int
    low_watermark: int
    underflow: bool
    below_low_watermark: bool


@dataclass(frozen=True)
class RequestTerminalState:
    status: str
    source: str
    reason: str
    transitioned_at: float


@dataclass
class PendingRequest:
    request_id: int
    session_id: str
    capture_ts: float
    sent_at: float
    deadline: float
    timeout_budget_s: float
    queue_size_at_capture: int
    latest_action_id_at_capture: int
    must_go: bool
    attempt_count: int = 1
    terminal_state: RequestTerminalState | None = None

    def key(self) -> tuple[str, int]:
        return (self.session_id, self.request_id)

    @property
    def age_ms(self) -> float:
        return max((time.time() - self.sent_at) * 1000.0, 0.0)


@dataclass(frozen=True)
class ResponseValidation:
    accepted: bool
    drop_reason: str | None = None
    terminal_source: str = AsyncTerminalSource.SERVER.value


@dataclass(frozen=True)
class WatchdogDecision:
    expired: bool
    status: str | None = None
    reason: str | None = None
    source: str = AsyncTerminalSource.WATCHDOG.value


@dataclass(frozen=True)
class ResetRuntimeState:
    session_id: str
    cleared_pending_request_id: int | None
    cleared_queue_size: int
    preserved_latest_live_observation: bool
    latest_live_observation: Any | None


@dataclass
class RequestIdSequence:
    next_request_id: int = 1

    def next(self) -> int:
        request_id = self.next_request_id
        self.next_request_id += 1
        return request_id


def payload_value(payload: Any, key: str, default: Any | None = None) -> Any:
    if isinstance(payload, dict):
        return payload.get(key, default)
    return getattr(payload, key, default)


def new_session_id(prefix: str = "async") -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def build_message(kind: AsyncMessageKind | str, payload: Any = None) -> RemoteMessage:
    message_kind = kind.value if isinstance(kind, AsyncMessageKind) else kind
    return RemoteMessage(kind=message_kind, payload=payload)


def build_queue_metrics(queue_size: int, low_watermark: int) -> QueueMetrics:
    return QueueMetrics(
        queue_size=queue_size,
        low_watermark=low_watermark,
        underflow=queue_size <= 0,
        below_low_watermark=queue_size <= low_watermark,
    )


def compute_response_deadline(sent_at: float, response_timeout_s: float) -> float:
    return sent_at + max(response_timeout_s, 0.0)


def compute_required_buffer_steps(target_p95_ack_latency_s: float, control_fps: float) -> int:
    if control_fps <= 0:
        raise ValueError(f"control_fps must be positive, got {control_fps}")
    control_period_s = 1.0 / control_fps
    return max(0, math.ceil(target_p95_ack_latency_s / control_period_s))


def suggest_low_watermark(required_buffer_steps: int) -> int:
    return max(0, required_buffer_steps)


def suggest_chunk_size(low_watermark: int) -> int:
    return max(1, low_watermark + 1)


def should_force_must_go(queue_size: int) -> bool:
    return queue_size <= 0


def can_send_request(
    *,
    queue_size: int,
    low_watermark: int,
    pending_request: PendingRequest | None,
) -> bool:
    return pending_request is None and queue_size <= low_watermark


def build_pending_request(
    *,
    request_id: int,
    session_id: str,
    capture_ts: float,
    sent_at: float,
    response_timeout_s: float,
    queue_size_at_capture: int,
    latest_action_id_at_capture: int,
    must_go: bool,
    attempt_count: int = 1,
) -> PendingRequest:
    return PendingRequest(
        request_id=request_id,
        session_id=session_id,
        capture_ts=capture_ts,
        sent_at=sent_at,
        deadline=compute_response_deadline(sent_at, response_timeout_s),
        timeout_budget_s=response_timeout_s,
        queue_size_at_capture=queue_size_at_capture,
        latest_action_id_at_capture=latest_action_id_at_capture,
        must_go=must_go,
        attempt_count=attempt_count,
    )


def mark_request_terminal(
    pending_request: PendingRequest,
    *,
    status: AsyncInferStatus | str,
    source: AsyncTerminalSource | str,
    reason: str,
    now: float | None = None,
) -> RequestTerminalState:
    if pending_request.terminal_state is not None:
        raise ValueError(
            f"Request {pending_request.request_id} in session {pending_request.session_id} "
            f"already reached terminal state {pending_request.terminal_state.status}"
        )
    state = RequestTerminalState(
        status=status.value if isinstance(status, AsyncInferStatus) else status,
        source=source.value if isinstance(source, AsyncTerminalSource) else source,
        reason=reason,
        transitioned_at=time.time() if now is None else now,
    )
    pending_request.terminal_state = state
    return state


def coerce_async_infer_response(payload: AsyncInferResponse | Any) -> AsyncInferResponse:
    if isinstance(payload, AsyncInferResponse):
        return payload
    return AsyncInferResponse(
        request_id=int(payload_value(payload, "request_id")),
        session_id=str(payload_value(payload, "session_id")),
        status=str(payload_value(payload, "status")),
        actions=list(payload_value(payload, "actions", [])),
        reason=str(payload_value(payload, "reason", "")),
        server_received_at=float(payload_value(payload, "server_received_at", 0.0)),
        server_sent_at=float(payload_value(payload, "server_sent_at", 0.0)),
        predict_latency_ms=float(payload_value(payload, "predict_latency_ms", 0.0)),
        queue_wait_ms=float(payload_value(payload, "queue_wait_ms", 0.0)),
        chunk_size=payload_value(payload, "chunk_size", None),
        preprocess_ms=float(payload_value(payload, "preprocess_ms", 0.0)),
        infer_ms=float(payload_value(payload, "infer_ms", 0.0)),
        postprocess_ms=float(payload_value(payload, "postprocess_ms", 0.0)),
        serialize_ms=float(payload_value(payload, "serialize_ms", 0.0)),
        total_ms=float(payload_value(payload, "total_ms", 0.0)),
        metadata=dict(payload_value(payload, "metadata", {})),
    )


def validate_response(
    *,
    response: AsyncInferResponse | Any,
    pending_request: PendingRequest | None,
    expected_session_id: str | None = None,
    terminal_requests: Collection[tuple[str, int]] | None = None,
) -> ResponseValidation:
    typed_response = coerce_async_infer_response(response)
    valid_statuses = {status.value for status in AsyncInferStatus}
    if typed_response.status not in valid_statuses:
        return ResponseValidation(accepted=False, drop_reason=AsyncDropReason.INVALID_STATUS.value)
    response_key = (typed_response.session_id, typed_response.request_id)
    if terminal_requests is not None and response_key in terminal_requests:
        return ResponseValidation(accepted=False, drop_reason=AsyncDropReason.DUPLICATE_TERMINAL.value)
    if expected_session_id is not None and typed_response.session_id != expected_session_id:
        return ResponseValidation(accepted=False, drop_reason=AsyncDropReason.FOREIGN_SESSION.value)
    if pending_request is None:
        return ResponseValidation(accepted=False, drop_reason=AsyncDropReason.NO_PENDING_REQUEST.value)
    if typed_response.session_id != pending_request.session_id:
        return ResponseValidation(accepted=False, drop_reason=AsyncDropReason.FOREIGN_SESSION.value)
    if typed_response.request_id != pending_request.request_id:
        return ResponseValidation(accepted=False, drop_reason=AsyncDropReason.REQUEST_ID_MISMATCH.value)
    if pending_request.terminal_state is not None:
        return ResponseValidation(accepted=False, drop_reason=AsyncDropReason.DUPLICATE_TERMINAL.value)
    return ResponseValidation(accepted=True)


def decide_watchdog_timeout(
    *,
    pending_request: PendingRequest | None,
    now: float,
    consecutive_timeouts: int,
    max_consecutive_timeouts: int,
    connection_recoverable: bool,
) -> WatchdogDecision:
    if pending_request is None or now < pending_request.deadline:
        return WatchdogDecision(expired=False)
    if connection_recoverable and consecutive_timeouts < max_consecutive_timeouts:
        return WatchdogDecision(
            expired=True,
            status=AsyncInferStatus.RETRY.value,
            reason="watchdog_timeout_retry",
        )
    return WatchdogDecision(
        expired=True,
        status=AsyncInferStatus.ABORT.value,
        reason="watchdog_timeout_abort",
    )


def build_retry_pending_request(
    pending_request: PendingRequest,
    *,
    new_request_id: int,
    sent_at: float,
    capture_ts: float,
) -> PendingRequest:
    return build_pending_request(
        request_id=new_request_id,
        session_id=pending_request.session_id,
        capture_ts=capture_ts,
        sent_at=sent_at,
        response_timeout_s=pending_request.timeout_budget_s,
        queue_size_at_capture=pending_request.queue_size_at_capture,
        latest_action_id_at_capture=pending_request.latest_action_id_at_capture,
        must_go=pending_request.must_go,
        attempt_count=pending_request.attempt_count + 1,
    )


def cleanup_reset_state(
    *,
    session_id: str,
    pending_request: PendingRequest | None,
    queue_size: int,
    latest_live_observation: Any | None,
    preserve_latest_live_observation: bool = True,
) -> ResetRuntimeState:
    return ResetRuntimeState(
        session_id=session_id,
        cleared_pending_request_id=None if pending_request is None else pending_request.request_id,
        cleared_queue_size=max(queue_size, 0),
        preserved_latest_live_observation=preserve_latest_live_observation,
        latest_live_observation=latest_live_observation if preserve_latest_live_observation else None,
    )
