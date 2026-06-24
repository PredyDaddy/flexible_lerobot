from __future__ import annotations

from collections import deque

from .types import MatchedPayload, TimestampedPayload


class SampleBuffer:
    def __init__(self, maxlen: int = 512):
        self._items: deque[TimestampedPayload] = deque(maxlen=maxlen)

    def append(self, item: TimestampedPayload) -> None:
        self._items.append(item)

    def latest_not_after(self, target_time_ns: int, max_delta_ns: int) -> MatchedPayload | None:
        for item in reversed(self._items):
            if item.source_time_ns <= target_time_ns:
                delta_ns = target_time_ns - item.source_time_ns
                if delta_ns <= max_delta_ns:
                    return MatchedPayload(item=item, delta_ns=delta_ns)
                return None
        return None

    def __len__(self) -> int:
        return len(self._items)
