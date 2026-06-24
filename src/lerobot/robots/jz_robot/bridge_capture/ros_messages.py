from __future__ import annotations

from typing import Any


def joint_state_names_values(message: Any) -> tuple[list[str], list[float]]:
    return list(message.name), [float(value) for value in message.position]


def float64_array_names_values(message: Any, names: list[str]) -> tuple[list[str], list[float]]:
    values = [float(value) for value in message.data]
    return list(names[: len(values)]), values


def vector_to_float64_multi_array(values: list[float]) -> Any:
    from std_msgs.msg import Float64MultiArray

    msg = Float64MultiArray()
    msg.data = list(values)
    return msg
