from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeErrorRecord:
    source: str
    message: str
    traceback_text: str


class RuntimeState:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self.first_chunk_ready = threading.Event()
        self.start_time_s = time.perf_counter()
        self.stop_reason: str | None = None
        self.last_error: RuntimeErrorRecord | None = None
        self.actor_iterations = 0
        self.inference_iterations = 0
        self.sent_actions = 0
        self.drop_all_chunks_in_a_row = 0

    @property
    def running(self) -> bool:
        return not self._stop_event.is_set()

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    def request_stop(self, reason: str) -> bool:
        with self._lock:
            if self._stop_event.is_set():
                return False
            self.stop_reason = reason
            self._stop_event.set()
            self.first_chunk_ready.set()
            return True

    def mark_first_chunk_ready(self) -> None:
        self.first_chunk_ready.set()

    def record_exception(self, source: str, exc: BaseException) -> RuntimeErrorRecord:
        record = RuntimeErrorRecord(
            source=source,
            message=str(exc),
            traceback_text="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
        with self._lock:
            self.last_error = record
        self.request_stop(f"{source}: {exc}")
        return record

    def note_actor_iteration(self, *, sent: bool) -> None:
        with self._lock:
            self.actor_iterations += 1
            if sent:
                self.sent_actions += 1

    def note_inference_iteration(self, *, dropped_all: bool, drop_all_limit: int) -> None:
        with self._lock:
            self.inference_iterations += 1
            if dropped_all:
                self.drop_all_chunks_in_a_row += 1
            else:
                self.drop_all_chunks_in_a_row = 0
            if self.drop_all_chunks_in_a_row >= drop_all_limit:
                self.request_stop(
                    f"too many fully stale chunks in a row: {self.drop_all_chunks_in_a_row}"
                )
