from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np

PROTOCOL_VERSION = 1
CONTENT_TYPE = "application/x-python-pickle"


@dataclass(slots=True)
class InferenceRequest:
    request_id: int
    observation_frame: dict[str, Any]
    task: str
    robot_type: str
    obs_timestamp_s: float
    obs_sequence_id: int
    enable_rtc: bool
    predicted_delay_steps: int
    prev_chunk_left_over: np.ndarray | None
    execution_horizon: int


@dataclass(slots=True)
class InferenceResponse:
    request_id: int
    raw_actions: np.ndarray
    processed_actions: np.ndarray
    server_latency_s: float
    model_latency_s: float
    action_shape: tuple[int, ...]
    error: str | None = None


def dumps_payload(payload: Any) -> bytes:
    return pickle.dumps({"version": PROTOCOL_VERSION, "payload": payload}, protocol=pickle.HIGHEST_PROTOCOL)


def loads_payload(data: bytes) -> Any:
    envelope = pickle.loads(data)
    if not isinstance(envelope, dict):
        raise ValueError(f"Invalid protocol envelope type: {type(envelope)}")
    version = envelope.get("version")
    if version != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported protocol version: {version}")
    return envelope.get("payload")
