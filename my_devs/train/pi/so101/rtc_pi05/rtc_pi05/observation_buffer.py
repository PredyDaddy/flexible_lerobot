from __future__ import annotations

import copy
import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ObservationSnapshot:
    observation: dict[str, Any]
    timestamp_s: float
    sequence_id: int


class ObservationBuffer:
    """Thread-safe latest-observation store with timestamps."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._snapshot: ObservationSnapshot | None = None
        self._sequence_id = 0

    def update(self, observation: dict[str, Any], *, timestamp_s: float) -> ObservationSnapshot:
        with self._condition:
            self._sequence_id += 1
            snapshot = ObservationSnapshot(
                observation=copy.deepcopy(observation),
                timestamp_s=float(timestamp_s),
                sequence_id=self._sequence_id,
            )
            self._snapshot = snapshot
            self._condition.notify_all()
            return self._copy_snapshot(snapshot)

    def latest(self, *, timeout_s: float | None = None, min_sequence_id: int | None = None) -> ObservationSnapshot | None:
        with self._condition:
            if not self._has_usable_snapshot(min_sequence_id):
                self._condition.wait_for(lambda: self._has_usable_snapshot(min_sequence_id), timeout=timeout_s)
            if not self._has_usable_snapshot(min_sequence_id):
                return None
            return self._copy_snapshot(self._snapshot)

    def _has_usable_snapshot(self, min_sequence_id: int | None) -> bool:
        if self._snapshot is None:
            return False
        return min_sequence_id is None or self._snapshot.sequence_id >= min_sequence_id

    @staticmethod
    def _copy_snapshot(snapshot: ObservationSnapshot | None) -> ObservationSnapshot | None:
        if snapshot is None:
            return None
        return ObservationSnapshot(
            observation=copy.deepcopy(snapshot.observation),
            timestamp_s=snapshot.timestamp_s,
            sequence_id=snapshot.sequence_id,
        )
