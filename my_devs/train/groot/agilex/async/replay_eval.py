#!/usr/bin/env python

"""Offline replay gate for AgileX bounded-async scheduling semantics."""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import os
import pickle  # nosec
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_CONTROL_FPS = 30
DEFAULT_RESPONSE_TIMEOUT_S = 0.75
DEFAULT_LOW_WATERMARK = 1
DEFAULT_CHUNK_SIZE = 2
DEFAULT_MAX_CONSECUTIVE_TIMEOUTS = 2
DEFAULT_SYNTHETIC_STEPS = 120
DEFAULT_ACK_LATENCY_MS = 80.0
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "reports" / "replay_eval_report.json"


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


def summarize_distribution(values: list[float]) -> dict[str, float]:
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


@dataclass
class ReplayConfig:
    control_fps: int
    response_timeout_s: float
    low_watermark: int
    chunk_size: int
    max_consecutive_timeouts: int
    observation_file: Path | None
    synthetic_steps: int
    ack_latency_ms: float
    latencies_ms: list[float] | None
    status_sequence: list[str]
    output_path: Path | None


@dataclass
class ObservationRecord:
    index: int
    timestamp_s: float


@dataclass
class PendingRequest:
    request_id: int
    session_id: int
    capture_time_s: float
    deadline_s: float
    must_go: bool
    queue_size_at_capture: int


@dataclass(order=True)
class ResponseEvent:
    arrival_time_s: float
    request_id: int
    session_id: int
    status: str = field(compare=False)
    capture_time_s: float = field(compare=False)
    latency_ms: float = field(compare=False)


@dataclass
class QueuedAction:
    action_id: int
    request_id: int
    capture_time_s: float


@dataclass
class ReplayMetrics:
    label: str
    control_cycles: int = 0
    send_attempts: int = 0
    ack_count: int = 0
    retry_count: int = 0
    abort_count: int = 0
    hold_cycles: int = 0
    must_go_count: int = 0
    executed_action_count: int = 0
    pending_request_timeout_count: int = 0
    late_response_drop_count: int = 0
    foreign_session_drop_count: int = 0
    duplicate_response_drop_count: int = 0
    queue_size_before_send: list[float] = field(default_factory=list)
    queue_size_after_ack: list[float] = field(default_factory=list)
    round_trip_latency_ms: list[float] = field(default_factory=list)
    ack_round_trip_latency_ms: list[float] = field(default_factory=list)
    action_age_at_execution_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        control_cycles = max(self.control_cycles, 1)
        stale_or_drop = (
            self.late_response_drop_count
            + self.foreign_session_drop_count
            + self.duplicate_response_drop_count
        )
        return {
            "label": self.label,
            "control_cycles": self.control_cycles,
            "send_attempts": self.send_attempts,
            "ack_count": self.ack_count,
            "retry_count": self.retry_count,
            "abort_count": self.abort_count,
            "hold_cycles": self.hold_cycles,
            "must_go_count": self.must_go_count,
            "executed_action_count": self.executed_action_count,
            "pending_request_timeout_count": self.pending_request_timeout_count,
            "late_response_drop_count": self.late_response_drop_count,
            "foreign_session_drop_count": self.foreign_session_drop_count,
            "duplicate_response_drop_count": self.duplicate_response_drop_count,
            "queue_underflow_rate": self.hold_cycles / float(control_cycles),
            "response_stale_or_drop_rate": stale_or_drop / float(max(self.ack_count + stale_or_drop, 1)),
            "queue_size_before_send": summarize_distribution(self.queue_size_before_send),
            "queue_size_after_ack": summarize_distribution(self.queue_size_after_ack),
            "round_trip_latency_ms": summarize_distribution(self.round_trip_latency_ms),
            "ack_round_trip_latency_ms": summarize_distribution(self.ack_round_trip_latency_ms),
            "action_age_at_execution_ms": summarize_distribution(self.action_age_at_execution_ms),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline replay gate for AgileX bounded-async scheduling.")
    parser.add_argument("--control-fps", type=int, default=int(os.getenv("FPS", str(DEFAULT_CONTROL_FPS))))
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
        "--chunk-size",
        type=int,
        default=int(os.getenv("ASYNC_CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE))),
    )
    parser.add_argument(
        "--max-consecutive-timeouts",
        type=int,
        default=int(os.getenv("ASYNC_MAX_CONSECUTIVE_TIMEOUTS", str(DEFAULT_MAX_CONSECUTIVE_TIMEOUTS))),
    )
    parser.add_argument("--observation-file", default=os.getenv("ASYNC_REPLAY_OBSERVATION_FILE"))
    parser.add_argument(
        "--synthetic-steps",
        type=int,
        default=int(os.getenv("ASYNC_REPLAY_SYNTHETIC_STEPS", str(DEFAULT_SYNTHETIC_STEPS))),
    )
    parser.add_argument(
        "--ack-latency-ms",
        type=float,
        default=float(os.getenv("ASYNC_REPLAY_ACK_LATENCY_MS", str(DEFAULT_ACK_LATENCY_MS))),
    )
    parser.add_argument(
        "--latencies-ms",
        default=os.getenv("ASYNC_REPLAY_LATENCIES_MS"),
        help="Comma-separated latency sequence in milliseconds. Repeats if shorter than the replay horizon.",
    )
    parser.add_argument(
        "--status-sequence",
        default=os.getenv("ASYNC_REPLAY_STATUS_SEQUENCE", "ack"),
        help="Comma-separated response statuses: ack/retry/abort/drop. Repeats across requests.",
    )
    parser.add_argument(
        "--output-path",
        default=os.getenv("ASYNC_REPLAY_OUTPUT_PATH", str(DEFAULT_OUTPUT_PATH)),
    )
    return parser


def parse_csv_floats(value: str | None) -> list[float] | None:
    if value in {None, "", "none", "null"}:
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_statuses(value: str) -> list[str]:
    statuses = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not statuses:
        raise ValueError("status_sequence must contain at least one status")
    unsupported = [status for status in statuses if status not in {"ack", "retry", "abort", "drop"}]
    if unsupported:
        raise ValueError(f"Unsupported replay statuses: {unsupported}")
    return statuses


def load_observation_records(path: Path | None, *, synthetic_steps: int, control_period_s: float) -> list[ObservationRecord]:
    if path is None:
        return [ObservationRecord(index=i, timestamp_s=i * control_period_s) for i in range(synthetic_steps)]

    if not path.is_file():
        raise FileNotFoundError(f"Observation replay file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        entries = loaded if isinstance(loaded, list) else loaded.get("observations", [])
    elif suffix in {".pkl", ".pickle"}:
        entries = pickle.loads(path.read_bytes())  # nosec
    else:
        raise ValueError(f"Unsupported observation replay file extension: {suffix}")

    if not isinstance(entries, list):
        raise TypeError(f"Replay observations must deserialize to a list, got {type(entries)}")

    records: list[ObservationRecord] = []
    for index, entry in enumerate(entries):
        if isinstance(entry, dict) and "timestamp" in entry:
            timestamp_s = float(entry["timestamp"])
        elif isinstance(entry, dict) and "timestamp_s" in entry:
            timestamp_s = float(entry["timestamp_s"])
        else:
            timestamp_s = index * control_period_s
        records.append(ObservationRecord(index=index, timestamp_s=timestamp_s))
    return records


def simulate_bounded_async(
    *,
    label: str,
    records: list[ObservationRecord],
    control_period_s: float,
    chunk_size: int,
    low_watermark: int,
    response_timeout_s: float,
    max_consecutive_timeouts: int,
    latencies_ms: list[float],
    statuses: list[str],
) -> ReplayMetrics:
    metrics = ReplayMetrics(label=label)
    queue: deque[QueuedAction] = deque()
    pending: PendingRequest | None = None
    response_events: list[ResponseEvent] = []
    terminal_request_ids: set[tuple[int, int]] = set()
    session_id = 1
    next_request_id = 0
    next_action_id = 0
    consecutive_timeouts = 0
    latency_cycle = itertools.cycle(latencies_ms)
    status_cycle = itertools.cycle(statuses)

    def mark_terminal(request_id: int, request_session: int) -> None:
        terminal_request_ids.add((request_session, request_id))

    def process_arrivals(current_time_s: float) -> None:
        nonlocal pending
        nonlocal session_id
        nonlocal consecutive_timeouts
        nonlocal next_action_id
        while response_events and response_events[0].arrival_time_s <= current_time_s:
            event = heapq.heappop(response_events)
            if event.session_id != session_id:
                metrics.foreign_session_drop_count += 1
                continue
            if pending is None or pending.request_id != event.request_id:
                if (event.session_id, event.request_id) in terminal_request_ids:
                    metrics.duplicate_response_drop_count += 1
                else:
                    metrics.late_response_drop_count += 1
                continue

            pending = None
            mark_terminal(event.request_id, event.session_id)
            metrics.round_trip_latency_ms.append(event.latency_ms)

            if event.status == "ack":
                metrics.ack_count += 1
                metrics.ack_round_trip_latency_ms.append(event.latency_ms)
                consecutive_timeouts = 0
                remaining_actions = max(len(records) - metrics.executed_action_count, 0)
                for _ in range(min(chunk_size, remaining_actions)):
                    next_action_id += 1
                    queue.append(
                        QueuedAction(
                            action_id=next_action_id,
                            request_id=event.request_id,
                            capture_time_s=event.capture_time_s,
                        )
                    )
                metrics.queue_size_after_ack.append(float(len(queue)))
                continue

            if event.status == "retry":
                metrics.retry_count += 1
                continue

            metrics.abort_count += 1
            queue.clear()
            session_id += 1
            consecutive_timeouts = 0

    def handle_timeout(current_time_s: float) -> None:
        nonlocal pending
        nonlocal session_id
        nonlocal consecutive_timeouts
        if pending is None or current_time_s < pending.deadline_s:
            return

        expired = pending
        pending = None
        mark_terminal(expired.request_id, expired.session_id)
        metrics.pending_request_timeout_count += 1
        consecutive_timeouts += 1

        if consecutive_timeouts < max_consecutive_timeouts:
            metrics.retry_count += 1
            return

        metrics.abort_count += 1
        queue.clear()
        session_id += 1
        consecutive_timeouts = 0

    for record in records:
        current_time_s = record.timestamp_s
        process_arrivals(current_time_s)
        handle_timeout(current_time_s)

        if queue:
            action = queue.popleft()
            metrics.executed_action_count += 1
            metrics.action_age_at_execution_ms.append((current_time_s - action.capture_time_s) * 1000.0)
        else:
            metrics.hold_cycles += 1

        metrics.control_cycles += 1

        if pending is None and len(queue) <= low_watermark:
            next_request_id += 1
            latency_ms = float(next(latency_cycle))
            status = str(next(status_cycle)).lower()
            pending = PendingRequest(
                request_id=next_request_id,
                session_id=session_id,
                capture_time_s=current_time_s,
                deadline_s=current_time_s + response_timeout_s,
                must_go=len(queue) == 0,
                queue_size_at_capture=len(queue),
            )
            metrics.send_attempts += 1
            metrics.queue_size_before_send.append(float(len(queue)))
            if pending.must_go:
                metrics.must_go_count += 1

            if status != "drop":
                heapq.heappush(
                    response_events,
                    ResponseEvent(
                        arrival_time_s=current_time_s + max(latency_ms, 0.0) / 1000.0,
                        request_id=next_request_id,
                        session_id=session_id,
                        status=status,
                        capture_time_s=current_time_s,
                        latency_ms=max(latency_ms, 0.0),
                    ),
                )

    flush_time_s = records[-1].timestamp_s if records else 0.0
    while pending is not None or response_events:
        next_deadline = pending.deadline_s if pending is not None else math.inf
        next_arrival = response_events[0].arrival_time_s if response_events else math.inf
        flush_time_s = min(next_deadline, next_arrival)
        if not math.isfinite(flush_time_s):
            break
        process_arrivals(flush_time_s)
        handle_timeout(flush_time_s)

    return metrics


def simulate_sync_boundary(
    *,
    records: list[ObservationRecord],
    response_timeout_s: float,
    max_consecutive_timeouts: int,
    latencies_ms: list[float],
    statuses: list[str],
) -> ReplayMetrics:
    metrics = ReplayMetrics(label="sync_boundary")
    consecutive_timeouts = 0
    latency_cycle = itertools.cycle(latencies_ms)
    status_cycle = itertools.cycle(statuses)

    for _record in records:
        metrics.control_cycles += 1
        metrics.send_attempts += 1
        metrics.queue_size_before_send.append(0.0)

        latency_ms = max(float(next(latency_cycle)), 0.0)
        status = str(next(status_cycle)).lower()
        timed_out = status == "drop" or (latency_ms / 1000.0) > response_timeout_s

        if timed_out:
            metrics.pending_request_timeout_count += 1
            metrics.hold_cycles += 1
            consecutive_timeouts += 1
            if consecutive_timeouts < max_consecutive_timeouts:
                metrics.retry_count += 1
            else:
                metrics.abort_count += 1
                consecutive_timeouts = 0
            continue

        metrics.round_trip_latency_ms.append(latency_ms)

        if status == "ack":
            metrics.ack_count += 1
            metrics.ack_round_trip_latency_ms.append(latency_ms)
            metrics.executed_action_count += 1
            metrics.queue_size_after_ack.append(1.0)
            metrics.action_age_at_execution_ms.append(latency_ms)
            consecutive_timeouts = 0
            continue

        metrics.hold_cycles += 1
        if status == "retry":
            metrics.retry_count += 1
            continue

        metrics.abort_count += 1
        consecutive_timeouts = 0

    return metrics


def build_parameter_recommendation(
    *,
    control_period_s: float,
    async_metrics: ReplayMetrics,
    current_low_watermark: int,
    current_chunk_size: int,
) -> dict[str, Any]:
    measured_ack_latencies_ms = async_metrics.ack_round_trip_latency_ms
    p95_latency_s = percentile(measured_ack_latencies_ms, 0.95) / 1000.0 if measured_ack_latencies_ms else 0.0
    required_buffer_steps = int(math.ceil(p95_latency_s / max(control_period_s, 1e-9)))
    recommended_low_watermark = max(required_buffer_steps, current_low_watermark)
    recommended_chunk_size = max(recommended_low_watermark + 1, current_chunk_size)
    return {
        "control_period_s": control_period_s,
        "required_buffer_steps": required_buffer_steps,
        "recommended_low_watermark": recommended_low_watermark,
        "recommended_chunk_size": recommended_chunk_size,
        "recommended_action_age_upper_bound_ms": percentile(
            async_metrics.action_age_at_execution_ms,
            0.95,
        ),
        "measured_ack_latency_ms": summarize_distribution(measured_ack_latencies_ms),
        "note": (
            "Sizing uses async ack-latency samples only. Retry/abort/drop events stay in the replay report "
            "but do not participate in the target_p95_ack_latency buffer-sizing formula."
        ),
    }


def render_report(
    *,
    config: ReplayConfig,
    record_count: int,
    sync_metrics: ReplayMetrics,
    async_metrics: ReplayMetrics,
    parameter_recommendation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "config": asdict(config),
        "record_count": record_count,
        "sync_boundary": sync_metrics.summary(),
        "async_boundary": async_metrics.summary(),
        "parameter_recommendation": parameter_recommendation,
    }


def write_report(report: dict[str, Any], output_path: Path | None) -> None:
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if output_path is None:
        print(rendered)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered + "\n", encoding="utf-8")
    print(f"[INFO] Wrote replay report to {output_path}")


def main() -> None:
    args = build_parser().parse_args()
    config = ReplayConfig(
        control_fps=args.control_fps,
        response_timeout_s=args.response_timeout_s,
        low_watermark=args.low_watermark,
        chunk_size=args.chunk_size,
        max_consecutive_timeouts=args.max_consecutive_timeouts,
        observation_file=normalize_optional_path(args.observation_file),
        synthetic_steps=args.synthetic_steps,
        ack_latency_ms=args.ack_latency_ms,
        latencies_ms=parse_csv_floats(args.latencies_ms),
        status_sequence=parse_csv_statuses(args.status_sequence),
        output_path=normalize_optional_path(args.output_path),
    )

    if config.chunk_size < config.low_watermark + 1:
        raise ValueError(
            "chunk_size must be at least low_watermark + 1. "
            f"Got chunk_size={config.chunk_size}, low_watermark={config.low_watermark}."
        )

    control_period_s = 1.0 / float(config.control_fps)
    records = load_observation_records(
        config.observation_file,
        synthetic_steps=config.synthetic_steps,
        control_period_s=control_period_s,
    )
    if not records:
        raise ValueError("Replay requires at least one observation record")

    latencies_ms = config.latencies_ms or [config.ack_latency_ms]
    sync_metrics = simulate_sync_boundary(
        records=records,
        response_timeout_s=config.response_timeout_s,
        max_consecutive_timeouts=config.max_consecutive_timeouts,
        latencies_ms=latencies_ms,
        statuses=config.status_sequence,
    )
    async_metrics = simulate_bounded_async(
        label="async_boundary",
        records=records,
        control_period_s=control_period_s,
        chunk_size=config.chunk_size,
        low_watermark=config.low_watermark,
        response_timeout_s=config.response_timeout_s,
        max_consecutive_timeouts=config.max_consecutive_timeouts,
        latencies_ms=latencies_ms,
        statuses=config.status_sequence,
    )

    recommendation = build_parameter_recommendation(
        control_period_s=control_period_s,
        async_metrics=async_metrics,
        current_low_watermark=config.low_watermark,
        current_chunk_size=config.chunk_size,
    )
    report = render_report(
        config=config,
        record_count=len(records),
        sync_metrics=sync_metrics,
        async_metrics=async_metrics,
        parameter_recommendation=recommendation,
    )
    write_report(report, config.output_path)


if __name__ == "__main__":
    main()
