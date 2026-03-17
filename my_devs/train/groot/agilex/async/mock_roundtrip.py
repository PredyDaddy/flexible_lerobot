#!/usr/bin/env python

"""Mock roundtrip validation for AgileX bounded-async protocol semantics."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from remote_async_common import (
    AsyncClose,
    AsyncCloseAck,
    AsyncDropReason,
    AsyncError,
    AsyncHello,
    AsyncInferResponse,
    AsyncInferStatus,
    AsyncMessageKind,
    AsyncReady,
    AsyncReset,
    AsyncResetAck,
    PendingRequest,
    RequestIdSequence,
    build_message,
    build_pending_request,
    can_send_request,
    cleanup_reset_state,
    decide_watchdog_timeout,
    mark_request_terminal,
    new_session_id,
    should_force_must_go,
    suggest_chunk_size,
    suggest_low_watermark,
    validate_response,
)


DEFAULT_CONTROL_FPS = 30.0
DEFAULT_RESPONSE_TIMEOUT_S = 0.08
DEFAULT_LOW_WATERMARK = 1
DEFAULT_MAX_CONSECUTIVE_TIMEOUTS = 2
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "reports" / "mock_roundtrip_report.json"


@dataclass
class MockConfig:
    control_fps: float
    response_timeout_s: float
    low_watermark: int
    max_consecutive_timeouts: int
    output_path: Path | None


@dataclass
class ScenarioResult:
    name: str
    ok: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockClientState:
    control_fps: float
    response_timeout_s: float
    low_watermark: int
    max_consecutive_timeouts: int
    session_id: str = field(default_factory=lambda: new_session_id("mock"))
    pending_request: PendingRequest | None = None
    queue: list[dict[str, Any]] = field(default_factory=list)
    latest_live_observation: dict[str, Any] | None = None
    request_ids: RequestIdSequence = field(default_factory=RequestIdSequence)
    terminal_requests: deque[tuple[str, int]] = field(default_factory=lambda: deque(maxlen=64))
    consecutive_timeouts: int = 0

    def queue_size(self) -> int:
        return len(self.queue)

    def record_terminal(self, request: PendingRequest) -> None:
        key = request.key()
        if key not in self.terminal_requests:
            self.terminal_requests.append(key)

    def send_request(self, *, now: float, observation: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        self.latest_live_observation = observation
        allowed = can_send_request(
            queue_size=self.queue_size(),
            low_watermark=self.low_watermark,
            pending_request=self.pending_request,
        )
        if not allowed:
            return False, {
                "can_send": False,
                "queue_size": self.queue_size(),
                "has_pending": self.pending_request is not None,
            }

        request_id = self.request_ids.next()
        must_go = should_force_must_go(self.queue_size())
        self.pending_request = build_pending_request(
            request_id=request_id,
            session_id=self.session_id,
            capture_ts=now,
            sent_at=now,
            response_timeout_s=self.response_timeout_s,
            queue_size_at_capture=self.queue_size(),
            latest_action_id_at_capture=self.queue[-1]["action_id"] if self.queue else -1,
            must_go=must_go,
        )
        return True, {
            "can_send": True,
            "request_id": request_id,
            "must_go": must_go,
            "queue_size_at_capture": self.pending_request.queue_size_at_capture,
        }

    def apply_response(self, response: AsyncInferResponse) -> ScenarioResult:
        validation = validate_response(
            response=response,
            pending_request=self.pending_request,
            expected_session_id=self.session_id,
            terminal_requests=set(self.terminal_requests),
        )
        if not validation.accepted:
            return ScenarioResult(
                name="response_validation",
                ok=True,
                details={"accepted": False, "drop_reason": validation.drop_reason},
            )

        assert self.pending_request is not None
        request = self.pending_request
        mark_request_terminal(
            request,
            status=response.status,
            source=validation.terminal_source,
            reason=response.reason or response.status,
            now=response.server_sent_at or time.time(),
        )
        self.record_terminal(request)
        self.pending_request = None

        if response.status == AsyncInferStatus.ACK.value:
            for action in response.actions:
                self.queue.append(
                    {
                        "action_id": len(self.queue) + 1,
                        "request_id": response.request_id,
                        "action": action,
                    }
                )
            self.consecutive_timeouts = 0
        elif response.status == AsyncInferStatus.RETRY.value:
            self.consecutive_timeouts += 1
        else:
            self.queue.clear()
            self.consecutive_timeouts = 0

        return ScenarioResult(
            name="response_validation",
            ok=True,
            details={
                "accepted": True,
                "status": response.status,
                "queue_size_after": self.queue_size(),
                "terminal_requests": list(self.terminal_requests),
            },
        )

    def watchdog(self, *, now: float, connection_recoverable: bool) -> ScenarioResult:
        decision = decide_watchdog_timeout(
            pending_request=self.pending_request,
            now=now,
            consecutive_timeouts=self.consecutive_timeouts,
            max_consecutive_timeouts=self.max_consecutive_timeouts,
            connection_recoverable=connection_recoverable,
        )
        if not decision.expired:
            return ScenarioResult(name="watchdog", ok=True, details={"expired": False})

        assert self.pending_request is not None
        expired_request = self.pending_request
        mark_request_terminal(
            expired_request,
            status=decision.status or AsyncInferStatus.ABORT.value,
            source=decision.source,
            reason=decision.reason or "watchdog_timeout",
            now=now,
        )
        self.record_terminal(expired_request)
        self.pending_request = None

        if decision.status == AsyncInferStatus.RETRY.value:
            self.consecutive_timeouts += 1
        else:
            self.queue.clear()
            self.consecutive_timeouts = 0

        return ScenarioResult(
            name="watchdog",
            ok=True,
            details={
                "expired": True,
                "status": decision.status,
                "reason": decision.reason,
                "terminal_requests": list(self.terminal_requests),
            },
        )

    def reset(self, *, reason: str, new_session_id_value: str | None = None) -> ScenarioResult:
        reset_state = cleanup_reset_state(
            session_id=self.session_id,
            pending_request=self.pending_request,
            queue_size=self.queue_size(),
            latest_live_observation=self.latest_live_observation,
            preserve_latest_live_observation=True,
        )
        self.queue.clear()
        self.pending_request = None
        self.terminal_requests.clear()
        self.consecutive_timeouts = 0
        if new_session_id_value is not None:
            self.session_id = new_session_id_value
        return ScenarioResult(
            name="reset",
            ok=True,
            details={
                "reason": reason,
                "new_session_id": self.session_id,
                "reset_state": asdict(reset_state),
            },
        )


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def scenario_normal_ack(config: MockConfig) -> ScenarioResult:
    state = MockClientState(
        control_fps=config.control_fps,
        response_timeout_s=config.response_timeout_s,
        low_watermark=config.low_watermark,
        max_consecutive_timeouts=config.max_consecutive_timeouts,
    )
    sent, details = state.send_request(now=1.0, observation={"step": 0})
    _assert(sent, "normal_ack should allow the first request")
    _assert(details["must_go"] is True, "empty queue must set must_go")
    response = AsyncInferResponse(
        request_id=details["request_id"],
        session_id=state.session_id,
        status=AsyncInferStatus.ACK.value,
        actions=[{"joint": 1.0}, {"joint": 2.0}],
        reason="ok",
        server_received_at=1.01,
        server_sent_at=1.02,
        predict_latency_ms=12.0,
        chunk_size=2,
    )
    applied = state.apply_response(response)
    _assert(applied.details["accepted"] is True, "normal_ack should accept the matching response")
    _assert(state.queue_size() == 2, "ack should enqueue the returned chunk")
    return ScenarioResult(name="normal_ack", ok=True, details={"send": details, "apply": applied.details})


def scenario_pending_guard(config: MockConfig) -> ScenarioResult:
    state = MockClientState(
        control_fps=config.control_fps,
        response_timeout_s=config.response_timeout_s,
        low_watermark=config.low_watermark,
        max_consecutive_timeouts=config.max_consecutive_timeouts,
    )
    first_sent, first_details = state.send_request(now=2.0, observation={"step": 0})
    second_sent, second_details = state.send_request(now=2.01, observation={"step": 1})
    _assert(first_sent, "pending_guard first send should succeed")
    _assert(not second_sent, "pending_guard second send should be blocked while inflight exists")
    return ScenarioResult(
        name="pending_request_guard",
        ok=True,
        details={"first": first_details, "second": second_details},
    )


def scenario_server_retry_and_abort(config: MockConfig) -> ScenarioResult:
    state = MockClientState(
        control_fps=config.control_fps,
        response_timeout_s=config.response_timeout_s,
        low_watermark=config.low_watermark,
        max_consecutive_timeouts=config.max_consecutive_timeouts,
    )
    sent, details = state.send_request(now=3.0, observation={"step": 0})
    _assert(sent, "server_retry_and_abort should send initial request")
    original_session_id = state.session_id
    retry_response = AsyncInferResponse(
        request_id=details["request_id"],
        session_id=state.session_id,
        status=AsyncInferStatus.RETRY.value,
        actions=[],
        reason="slow_runtime",
        server_received_at=3.01,
        server_sent_at=3.02,
    )
    retry_result = state.apply_response(retry_response)
    _assert(retry_result.details["status"] == AsyncInferStatus.RETRY.value, "retry must be terminal for request")
    _assert(state.pending_request is None, "retry response must clear pending request")

    sent_again, next_details = state.send_request(now=3.1, observation={"step": 1})
    _assert(sent_again, "new request should be allowed after retry terminal state")
    abort_response = AsyncInferResponse(
        request_id=next_details["request_id"],
        session_id=state.session_id,
        status=AsyncInferStatus.ABORT.value,
        actions=[],
        reason="runtime_untrusted",
        server_received_at=3.11,
        server_sent_at=3.12,
    )
    abort_result = state.apply_response(abort_response)
    _assert(abort_result.details["status"] == AsyncInferStatus.ABORT.value, "abort must be terminal for request")
    _assert(state.queue_size() == 0, "abort should clear local queue")
    reset_result = state.reset(reason="server_abort", new_session_id_value=new_session_id("mock"))
    _assert(reset_result.details["new_session_id"] != original_session_id, "abort should roll to a new session")
    return ScenarioResult(
        name="server_retry_and_abort",
        ok=True,
        details={"retry": retry_result.details, "abort": abort_result.details, "reset": reset_result.details},
    )


def scenario_watchdog_retry_then_abort(config: MockConfig) -> ScenarioResult:
    state = MockClientState(
        control_fps=config.control_fps,
        response_timeout_s=config.response_timeout_s,
        low_watermark=config.low_watermark,
        max_consecutive_timeouts=config.max_consecutive_timeouts,
    )
    sent, _ = state.send_request(now=4.0, observation={"step": 0})
    _assert(sent, "watchdog_retry_then_abort should send initial request")
    retry_decision = state.watchdog(now=4.0 + config.response_timeout_s + 0.01, connection_recoverable=True)
    _assert(retry_decision.details["status"] == AsyncInferStatus.RETRY.value, "first timeout should retry")
    retry_results = [retry_decision.details]

    sent_again, _ = state.send_request(now=4.2, observation={"step": 1})
    _assert(sent_again, "second request should be allowed after retry timeout")
    second_retry = state.watchdog(now=4.2 + config.response_timeout_s + 0.01, connection_recoverable=True)
    _assert(second_retry.details["status"] == AsyncInferStatus.RETRY.value, "second timeout should still retry")
    retry_results.append(second_retry.details)

    sent_third, _ = state.send_request(now=4.4, observation={"step": 2})
    _assert(sent_third, "third request should be allowed after second retry timeout")
    abort_decision = state.watchdog(now=4.4 + config.response_timeout_s + 0.01, connection_recoverable=True)
    _assert(abort_decision.details["status"] == AsyncInferStatus.ABORT.value, "timeout budget exhaustion should abort")
    return ScenarioResult(
        name="watchdog_retry_then_abort",
        ok=True,
        details={"retries": retry_results, "abort": abort_decision.details},
    )


def scenario_reset_and_foreign_session_drop(config: MockConfig) -> ScenarioResult:
    state = MockClientState(
        control_fps=config.control_fps,
        response_timeout_s=config.response_timeout_s,
        low_watermark=config.low_watermark,
        max_consecutive_timeouts=config.max_consecutive_timeouts,
    )
    sent, details = state.send_request(now=5.0, observation={"step": 0})
    _assert(sent, "reset_and_foreign_session_drop should send initial request")
    old_session_id = state.session_id
    reset_result = state.reset(reason="episode_boundary", new_session_id_value=new_session_id("mock"))
    stale_response = AsyncInferResponse(
        request_id=details["request_id"],
        session_id=old_session_id,
        status=AsyncInferStatus.ACK.value,
        actions=[{"joint": 9.0}],
        reason="late_old_session",
        server_received_at=5.05,
        server_sent_at=5.06,
    )
    stale_validation = validate_response(
        response=stale_response,
        pending_request=state.pending_request,
        expected_session_id=state.session_id,
        terminal_requests=set(state.terminal_requests),
    )
    _assert(
        stale_validation.drop_reason == AsyncDropReason.FOREIGN_SESSION.value,
        "old-session response must be dropped as foreign session",
    )
    return ScenarioResult(
        name="reset_and_foreign_session_drop",
        ok=True,
        details={"reset": reset_result.details, "stale_drop_reason": stale_validation.drop_reason},
    )


def scenario_close_and_error(config: MockConfig) -> ScenarioResult:
    state = MockClientState(
        control_fps=config.control_fps,
        response_timeout_s=config.response_timeout_s,
        low_watermark=config.low_watermark,
        max_consecutive_timeouts=config.max_consecutive_timeouts,
    )
    original_session_id = state.session_id
    hello = AsyncHello(
        session_id=state.session_id,
        control_fps=config.control_fps,
        low_watermark=config.low_watermark,
        chunk_size=suggest_chunk_size(suggest_low_watermark(0)),
        response_timeout_s=config.response_timeout_s,
        dry_consume=True,
        client_name="mock_roundtrip",
    )
    ready = AsyncReady(session_id=hello.session_id, message="ready", server_time=6.0)
    close = AsyncClose(session_id=hello.session_id, reason="test_close")
    close_ack = AsyncCloseAck(session_id=hello.session_id, reason="ack_close")
    error = AsyncError(message="mock_error", session_id=hello.session_id, recoverable=True)
    close_message = build_message(AsyncMessageKind.CLOSE, close)
    error_message = build_message(AsyncMessageKind.ERROR, error)
    error_reset = state.reset(reason="remote_error", new_session_id_value=new_session_id("mock"))
    _assert(close_message.kind == AsyncMessageKind.CLOSE.value, "close message kind mismatch")
    _assert(error_message.kind == AsyncMessageKind.ERROR.value, "error message kind mismatch")
    _assert(close_ack.session_id == hello.session_id, "close_ack session mismatch")
    _assert(ready.session_id == hello.session_id, "ready should preserve session_id")
    _assert(error_reset.details["new_session_id"] != original_session_id, "error path should trigger reset/new session")
    return ScenarioResult(
        name="close_and_error",
        ok=True,
        details={
            "hello": asdict(hello),
            "ready": asdict(ready),
            "close": asdict(close),
            "close_ack": asdict(close_ack),
            "error": asdict(error),
            "error_reset": error_reset.details,
        },
    )


def scenario_reset_ack_fields(config: MockConfig) -> ScenarioResult:
    reset = AsyncReset(session_id=new_session_id("mock"), reason="manual_reset")
    ack = AsyncResetAck(session_id=reset.session_id, reason=reset.reason, accepted=True)
    _assert(reset.requested_at > 0.0, "reset should carry requested_at")
    _assert(ack.reset_at > 0.0, "reset_ack should carry reset_at")
    return ScenarioResult(
        name="reset_ack_fields",
        ok=True,
        details={"reset": asdict(reset), "reset_ack": asdict(ack)},
    )


def scenario_duplicate_terminal_drop(config: MockConfig) -> ScenarioResult:
    state = MockClientState(
        control_fps=config.control_fps,
        response_timeout_s=config.response_timeout_s,
        low_watermark=config.low_watermark,
        max_consecutive_timeouts=config.max_consecutive_timeouts,
    )
    sent, details = state.send_request(now=7.0, observation={"step": 0})
    _assert(sent, "duplicate_terminal_drop should send initial request")
    response = AsyncInferResponse(
        request_id=details["request_id"],
        session_id=state.session_id,
        status=AsyncInferStatus.ACK.value,
        actions=[{"joint": 1.0}],
        reason="ok",
        server_received_at=7.01,
        server_sent_at=7.02,
    )
    applied = state.apply_response(response)
    _assert(applied.details["accepted"] is True, "first response should be accepted")
    duplicate_validation = validate_response(
        response=response,
        pending_request=state.pending_request,
        expected_session_id=state.session_id,
        terminal_requests=set(state.terminal_requests),
    )
    _assert(
        duplicate_validation.drop_reason == AsyncDropReason.DUPLICATE_TERMINAL.value,
        "duplicate same-session response must be dropped as duplicate_terminal",
    )
    return ScenarioResult(
        name="duplicate_terminal_drop",
        ok=True,
        details={"first_apply": applied.details, "duplicate_drop_reason": duplicate_validation.drop_reason},
    )


def run_scenarios(config: MockConfig) -> list[ScenarioResult]:
    return [
        scenario_normal_ack(config),
        scenario_pending_guard(config),
        scenario_server_retry_and_abort(config),
        scenario_watchdog_retry_then_abort(config),
        scenario_reset_and_foreign_session_drop(config),
        scenario_duplicate_terminal_drop(config),
        scenario_close_and_error(config),
        scenario_reset_ack_fields(config),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mock roundtrip validation for bounded-async AgileX protocol.")
    parser.add_argument("--control-fps", type=float, default=float(os.getenv("FPS", str(DEFAULT_CONTROL_FPS))))
    parser.add_argument(
        "--response-timeout-s",
        type=float,
        default=float(os.getenv("ASYNC_RESPONSE_TIMEOUT_S", str(DEFAULT_RESPONSE_TIMEOUT_S))),
    )
    parser.add_argument(
        "--low-watermark",
        type=int,
        default=int(os.getenv("ASYNC_LOW_WATERMARK", str(DEFAULT_LOW_WATERMARK))),
    )
    parser.add_argument(
        "--max-consecutive-timeouts",
        type=int,
        default=int(os.getenv("ASYNC_MAX_CONSECUTIVE_TIMEOUTS", str(DEFAULT_MAX_CONSECUTIVE_TIMEOUTS))),
    )
    parser.add_argument("--output-path", default=os.getenv("ASYNC_MOCK_OUTPUT_PATH", str(DEFAULT_OUTPUT_PATH)))
    return parser


def normalize_optional_path(path_str: str | None) -> Path | None:
    if path_str in {None, "", "none", "null"}:
        return None
    return Path(path_str).expanduser()


def main() -> None:
    args = build_parser().parse_args()
    config = MockConfig(
        control_fps=args.control_fps,
        response_timeout_s=args.response_timeout_s,
        low_watermark=args.low_watermark,
        max_consecutive_timeouts=args.max_consecutive_timeouts,
        output_path=normalize_optional_path(args.output_path),
    )
    results = run_scenarios(config)
    report = {
        "config": asdict(config),
        "all_ok": all(result.ok for result in results),
        "results": [asdict(result) for result in results],
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if config.output_path is None:
        print(rendered)
        return
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(rendered + "\n", encoding="utf-8")
    print(f"[INFO] Wrote mock roundtrip report to {config.output_path}")


if __name__ == "__main__":
    main()
