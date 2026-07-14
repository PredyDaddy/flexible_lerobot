#!/usr/bin/env python

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CachedState:
    packet: dict[str, Any]
    sender: tuple[str, int]
    received_monotonic_s: float
    received_wall_ns: int | None = None
    revision: int = 0


class StateCache:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: CachedState | None = None
        self._revision = 0

    def update(self, packet: dict[str, Any], sender: tuple[str, int]) -> None:
        with self._condition:
            self._revision += 1
            self._latest = CachedState(
                packet=packet,
                sender=sender,
                received_monotonic_s=time.monotonic(),
                received_wall_ns=time.time_ns(),
                revision=self._revision,
            )
            self._condition.notify_all()

    def clear(self) -> None:
        with self._condition:
            self._latest = None

    def latest(self) -> CachedState | None:
        with self._condition:
            return self._latest

    def wait(self, timeout_s: float) -> CachedState | None:
        return self.wait_after(timeout_s=timeout_s, after_monotonic_s=None)

    def wait_after(self, timeout_s: float, after_monotonic_s: float | None) -> CachedState | None:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._latest is None or (
                after_monotonic_s is not None and self._latest.received_monotonic_s < after_monotonic_s
            ):
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0:
                    return None
                self._condition.wait(timeout=remaining_s)
            return self._latest

    def wait_after_revision(self, timeout_s: float, after_revision: int) -> CachedState | None:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._latest is None or self._latest.revision <= after_revision:
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0:
                    return None
                self._condition.wait(timeout=remaining_s)
            return self._latest

    def age_s(self, state: CachedState | None = None) -> float | None:
        item = self.latest() if state is None else state
        if item is None:
            return None
        return time.monotonic() - item.received_monotonic_s
