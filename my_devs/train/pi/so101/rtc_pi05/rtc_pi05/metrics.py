from __future__ import annotations

import statistics
import threading
from collections import deque
from dataclasses import dataclass


class LatencyWindow:
    def __init__(self, maxlen: int = 100) -> None:
        self._values: deque[float] = deque(maxlen=maxlen)

    def add(self, value_s: float) -> None:
        self._values.append(max(0.0, float(value_s)))

    def latest(self) -> float:
        return self._values[-1] if self._values else 0.0

    def p95(self) -> float:
        if not self._values:
            return 0.0
        values = sorted(self._values)
        index = min(len(values) - 1, int(round(0.95 * (len(values) - 1))))
        return values[index]

    def mean(self) -> float:
        return statistics.fmean(self._values) if self._values else 0.0


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    actor_ticks: int
    inference_count: int
    queue_depth: int
    latest_inference_s: float
    p95_inference_s: float
    latest_drop_steps: int
    latest_predicted_delay_steps: int
    latest_cursor_delta_steps: int | None
    empty_queue_events: int
    dropped_all_chunks: int


class RuntimeMetrics:
    def __init__(self, *, window_size: int = 100) -> None:
        self._lock = threading.RLock()
        self.inference_latency = LatencyWindow(maxlen=window_size)
        self.actor_ticks = 0
        self.inference_count = 0
        self.queue_depth = 0
        self.latest_drop_steps = 0
        self.latest_predicted_delay_steps = 0
        self.latest_cursor_delta_steps: int | None = None
        self.empty_queue_events = 0
        self.dropped_all_chunks = 0

    def record_actor_tick(self, *, queue_depth: int, empty: bool) -> None:
        with self._lock:
            self.actor_ticks += 1
            self.queue_depth = queue_depth
            if empty:
                self.empty_queue_events += 1

    def record_inference(
        self,
        *,
        total_s: float,
        queue_depth: int,
        drop_steps: int,
        predicted_delay_steps: int,
        cursor_delta_steps: int | None,
        dropped_all: bool,
    ) -> None:
        with self._lock:
            self.inference_count += 1
            self.inference_latency.add(total_s)
            self.queue_depth = queue_depth
            self.latest_drop_steps = drop_steps
            self.latest_predicted_delay_steps = predicted_delay_steps
            self.latest_cursor_delta_steps = cursor_delta_steps
            if dropped_all:
                self.dropped_all_chunks += 1

    def predicted_delay_steps(self, *, control_dt_s: float) -> int:
        with self._lock:
            if control_dt_s <= 0:
                return 0
            latest = self.inference_latency.p95()
        if latest <= 0:
            return 0
        import math

        return int(math.ceil(latest / control_dt_s))

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            return MetricsSnapshot(
                actor_ticks=self.actor_ticks,
                inference_count=self.inference_count,
                queue_depth=self.queue_depth,
                latest_inference_s=self.inference_latency.latest(),
                p95_inference_s=self.inference_latency.p95(),
                latest_drop_steps=self.latest_drop_steps,
                latest_predicted_delay_steps=self.latest_predicted_delay_steps,
                latest_cursor_delta_steps=self.latest_cursor_delta_steps,
                empty_queue_events=self.empty_queue_events,
                dropped_all_chunks=self.dropped_all_chunks,
            )
