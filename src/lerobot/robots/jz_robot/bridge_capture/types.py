from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TimestampedPayload:
    source_time_ns: int
    receive_time_ns: int
    payload: Any


@dataclass(frozen=True)
class MatchedPayload:
    item: TimestampedPayload
    delta_ns: int

    @property
    def payload(self) -> Any:
        return self.item.payload


@dataclass(frozen=True)
class VectorSnapshot:
    names: list[str]
    values: list[float]
    source_time_ns: int
    receive_time_ns: int
