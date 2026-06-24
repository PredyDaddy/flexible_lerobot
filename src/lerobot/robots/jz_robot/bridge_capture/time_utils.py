from __future__ import annotations

import time
from typing import Any


def now_wall_time_ns() -> int:
    return time.time_ns()


def now_monotonic_ns() -> int:
    return time.monotonic_ns()


def ros_stamp_to_ns(stamp: Any) -> int:
    if stamp is None:
        return 0
    sec = int(getattr(stamp, "sec", 0))
    nanosec = int(getattr(stamp, "nanosec", 0))
    return sec * 1_000_000_000 + nanosec


def message_stamp_or_receive_time_ns(message: Any, receive_time_ns: int) -> int:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    stamp_ns = ros_stamp_to_ns(stamp)
    return stamp_ns if stamp_ns > 0 else receive_time_ns
