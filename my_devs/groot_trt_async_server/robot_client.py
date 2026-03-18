#!/usr/bin/env python

"""Thin async client wrappers for GR00T TensorRT remote inference.

This module deliberately keeps all TensorRT execution on the server side.
The client only:

- connects to the robot and cameras
- streams raw observations to the remote policy server
- receives action chunks
- maintains a local action queue
- optionally suppresses real action sends for round-trip validation
"""

from __future__ import annotations

import pickle  # nosec
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from typing import Any

import torch

from lerobot.async_inference.configs import RobotClientConfig as BaseRobotClientConfig
from lerobot.async_inference.constants import SUPPORTED_ROBOTS
from lerobot.async_inference.helpers import (
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    map_robot_keys_to_lerobot_features,
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from my_devs.groot_trt_async_server.configs import (
    DEFAULT_SERVER_ENGINE_DIR,
    DEFAULT_SERVER_POLICY_PATH,
    DEFAULT_SERVER_RESOURCE_PROFILE,
)

_ROBOT_CLIENT_IMPORT_ERROR: Exception | None = None
try:
    import grpc

    from lerobot.async_inference.robot_client import RobotClient as BaseRobotClient
    from lerobot.async_inference.robot_client import visualize_action_queue_size
    from lerobot.transport import services_pb2  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional async deps.
    BaseRobotClient = object  # type: ignore[assignment]
    grpc = None  # type: ignore[assignment]
    services_pb2 = None  # type: ignore[assignment]
    _ROBOT_CLIENT_IMPORT_ERROR = exc

    def visualize_action_queue_size(_: list[int]) -> None:
        raise RuntimeError(
            "Async client visualization requires optional async dependencies. "
            "Install the repository with async extras first."
        ) from _ROBOT_CLIENT_IMPORT_ERROR


DEFAULT_POLICY_PATH = DEFAULT_SERVER_POLICY_PATH
DEFAULT_ENGINE_DIR = DEFAULT_SERVER_ENGINE_DIR

DEFAULT_CALIBRATION_ROOT = Path("/home/cqy/.cache/huggingface/lerobot/calibration/robots")
SUPPORTED_SO_FOLLOWER_TYPES = {"so100_follower", "so101_follower"}
OBSERVATION_PAYLOAD_KEY = "timed_observation"
ACTION_PAYLOAD_KEY = "timed_actions"
REQUEST_ID_PAYLOAD_KEY = "request_id"
ACKED_REQUEST_ID_PAYLOAD_KEY = "acked_request_id"
OBSERVATION_TIMESTEP_PAYLOAD_KEY = "observation_timestep"
SESSION_ID_PAYLOAD_KEY = "session_id"
REQUEST_STATE_PAYLOAD_KEY = "request_state"
REQUEST_STATE_REASON_PAYLOAD_KEY = "request_state_reason"
SESSION_ID_METADATA_KEY = "x-groot-session-id"
SESSION_MODE_METADATA_KEY = "x-groot-session-mode"
SESSION_MODE_CLAIM = "claim"
SESSION_MODE_RELEASE = "release"
SESSION_MODE_TAKEOVER = "takeover"
REQUEST_STATE_ACK = "ack"
REQUEST_STATE_RETRY = "retry"
REQUEST_STATE_ABORT = "abort"
CLIENT_START_SESSION_MODES = {
    SESSION_MODE_CLAIM,
    SESSION_MODE_TAKEOVER,
}
ALLOWED_REQUEST_STATES = {
    REQUEST_STATE_ACK,
    REQUEST_STATE_RETRY,
    REQUEST_STATE_ABORT,
}
FAILED_REQUEST_STATES = {
    REQUEST_STATE_RETRY,
    REQUEST_STATE_ABORT,
}
MAX_INFLIGHT_OBSERVATION_REQUESTS = 1


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


def maybe_path(path_str: str | None) -> Path | None:
    return None if not path_str else Path(path_str).expanduser()


def _normalize_required_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {type(value)}")
    return value


def _normalize_required_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a str, got {type(value)}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    return normalized


def _normalize_optional_str(value: object | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _normalize_required_str(value, field_name=field_name)


def _normalize_session_mode(value: object, *, field_name: str, allowed: set[str]) -> str:
    normalized = _normalize_required_str(value, field_name=field_name).lower()
    if normalized not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}, got {normalized!r}")
    return normalized


def _normalize_request_state(value: object, *, field_name: str) -> str:
    return _normalize_session_mode(value, field_name=field_name, allowed=ALLOWED_REQUEST_STATES)


def build_observation_payload(
    observation: TimedObservation,
    *,
    request_id: int,
    session_id: str,
) -> dict[str, Any]:
    normalized_request_id = _normalize_required_int(request_id, field_name=REQUEST_ID_PAYLOAD_KEY)
    normalized_session_id = _normalize_required_str(session_id, field_name=SESSION_ID_PAYLOAD_KEY)
    return {
        OBSERVATION_PAYLOAD_KEY: observation,
        REQUEST_ID_PAYLOAD_KEY: normalized_request_id,
        SESSION_ID_PAYLOAD_KEY: normalized_session_id,
    }


def unpack_observation_payload(payload: Any) -> tuple[TimedObservation, int, str]:
    if isinstance(payload, dict):
        observation = payload.get(OBSERVATION_PAYLOAD_KEY)
        request_id = payload.get(REQUEST_ID_PAYLOAD_KEY)
        session_id = payload.get(SESSION_ID_PAYLOAD_KEY)
        if not isinstance(observation, TimedObservation):
            raise TypeError(f"Expected TimedObservation payload, got {type(observation)}")
        if request_id is None:
            raise ValueError(
                f"Observation payload must include {REQUEST_ID_PAYLOAD_KEY!r}; legacy payloads are not accepted."
            )
        if session_id is None:
            raise ValueError(
                f"Observation payload must include {SESSION_ID_PAYLOAD_KEY!r}; legacy payloads are not accepted."
            )
        return (
            observation,
            _normalize_required_int(request_id, field_name=REQUEST_ID_PAYLOAD_KEY),
            _normalize_required_str(session_id, field_name=SESSION_ID_PAYLOAD_KEY),
        )

    raise TypeError(f"Unsupported observation payload type: {type(payload)}")


def build_action_payload(
    timed_actions: list[TimedAction],
    *,
    request_id: int,
    observation_timestep: int,
    session_id: str,
    request_state: str = REQUEST_STATE_ACK,
    request_state_reason: str | None = None,
) -> dict[str, Any]:
    normalized_request_id = _normalize_required_int(request_id, field_name=ACKED_REQUEST_ID_PAYLOAD_KEY)
    normalized_observation_timestep = _normalize_required_int(
        observation_timestep,
        field_name=OBSERVATION_TIMESTEP_PAYLOAD_KEY,
    )
    normalized_session_id = _normalize_required_str(session_id, field_name=SESSION_ID_PAYLOAD_KEY)
    normalized_request_state = _normalize_request_state(request_state, field_name=REQUEST_STATE_PAYLOAD_KEY)
    normalized_request_state_reason = _normalize_optional_str(
        request_state_reason,
        field_name=REQUEST_STATE_REASON_PAYLOAD_KEY,
    )
    payload = {
        ACTION_PAYLOAD_KEY: timed_actions,
        REQUEST_ID_PAYLOAD_KEY: normalized_request_id,
        ACKED_REQUEST_ID_PAYLOAD_KEY: normalized_request_id,
        OBSERVATION_TIMESTEP_PAYLOAD_KEY: normalized_observation_timestep,
        SESSION_ID_PAYLOAD_KEY: normalized_session_id,
        REQUEST_STATE_PAYLOAD_KEY: normalized_request_state,
    }
    if normalized_request_state_reason is not None:
        payload[REQUEST_STATE_REASON_PAYLOAD_KEY] = normalized_request_state_reason
    return payload


@dataclass(frozen=True)
class ActionResponseEnvelope:
    timed_actions: list[TimedAction]
    request_id: int
    observation_timestep: int | None
    session_id: str
    request_state: str = REQUEST_STATE_ACK
    request_state_reason: str | None = None

    @property
    def is_failure_terminal(self) -> bool:
        return self.request_state in FAILED_REQUEST_STATES

    @property
    def is_retryable(self) -> bool:
        return self.request_state == REQUEST_STATE_RETRY


def unpack_action_payload(payload: Any) -> ActionResponseEnvelope:
    if isinstance(payload, dict):
        timed_actions = payload.get(ACTION_PAYLOAD_KEY)
        request_id = payload.get(ACKED_REQUEST_ID_PAYLOAD_KEY)
        observation_timestep = payload.get(OBSERVATION_TIMESTEP_PAYLOAD_KEY)
        session_id = payload.get(SESSION_ID_PAYLOAD_KEY)
        request_state = payload.get(REQUEST_STATE_PAYLOAD_KEY, REQUEST_STATE_ACK)
        request_state_reason = payload.get(REQUEST_STATE_REASON_PAYLOAD_KEY)
        if not isinstance(timed_actions, list):
            raise TypeError(f"Expected action payload list, got {type(timed_actions)}")
        if request_id is None:
            raise ValueError(
                f"Action payload must include {ACKED_REQUEST_ID_PAYLOAD_KEY!r}; legacy payloads are not accepted."
            )
        if session_id is None:
            raise ValueError(
                f"Action payload must include {SESSION_ID_PAYLOAD_KEY!r}; legacy payloads are not accepted."
            )
        if observation_timestep is not None:
            observation_timestep = _normalize_required_int(
                observation_timestep,
                field_name=OBSERVATION_TIMESTEP_PAYLOAD_KEY,
            )
        return ActionResponseEnvelope(
            timed_actions,
            request_id=_normalize_required_int(request_id, field_name=ACKED_REQUEST_ID_PAYLOAD_KEY),
            observation_timestep=observation_timestep,
            session_id=_normalize_required_str(session_id, field_name=SESSION_ID_PAYLOAD_KEY),
            request_state=_normalize_request_state(request_state, field_name=REQUEST_STATE_PAYLOAD_KEY),
            request_state_reason=_normalize_optional_str(
                request_state_reason,
                field_name=REQUEST_STATE_REASON_PAYLOAD_KEY,
            ),
        )

    raise TypeError(f"Unsupported action payload type: {type(payload)}")


@dataclass
class GrootTrtRemotePolicyConfig:
    """Protocol-compatible TRT payload used when the canonical server config is unavailable.

    Intentionally does not inherit from `RemotePolicyConfig`: the server-side compatibility
    loader special-cases `BaseRemotePolicyConfig` and would otherwise silently rewrite
    `backend='tensorrt'` back to `backend='pytorch'`.
    """

    policy_type: str
    pretrained_name_or_path: str
    lerobot_features: dict[str, Any]
    actions_per_chunk: int
    device: str = "cpu"
    rename_map: dict[str, str] = field(default_factory=dict)
    resource_profile: str = DEFAULT_SERVER_RESOURCE_PROFILE

    backend: str = "tensorrt"
    engine_dir: str = DEFAULT_ENGINE_DIR
    tensorrt_py_dir: str | None = None
    vit_dtype: str = "fp16"
    llm_dtype: str = "fp16"
    dit_dtype: str = "fp16"
    num_denoising_steps: int | None = None

    def __post_init__(self) -> None:
        if not self.resource_profile:
            raise ValueError("resource_profile cannot be empty")
        if self.backend not in {"tensorrt", "pytorch"}:
            raise ValueError(f"Unsupported backend={self.backend!r}. Expected 'tensorrt' or 'pytorch'.")


@dataclass(kw_only=True)
class ExplicitSOFollowerRobotConfig(SOFollowerRobotConfig):
    """SO100/SO101 wrapper that preserves the robot type selected on the CLI."""

    robot_type: str = field(default="so101_follower", metadata={"help": "Actual SO follower variant to create."})

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_so_follower_robot_type(self.robot_type)

    @property
    def type(self) -> str:
        return self.robot_type


@dataclass(frozen=True)
class PendingActionRequest:
    request_id: int
    observation_timestep: int
    sent_at: float
    queue_empty: bool
    must_go: bool


@dataclass(frozen=True)
class RequestTerminalState:
    request_id: int
    observation_timestep: int | None
    session_id: str
    request_state: str
    request_state_reason: str | None


def validate_so_follower_robot_type(robot_type: str) -> str:
    if robot_type not in SUPPORTED_SO_FOLLOWER_TYPES:
        supported = ", ".join(sorted(SUPPORTED_SO_FOLLOWER_TYPES))
        raise ValueError(f"Unsupported robot_type={robot_type!r}. Expected one of: {supported}.")
    return robot_type


def default_calibration_dir(robot_type: str) -> str:
    return (DEFAULT_CALIBRATION_ROOT / validate_so_follower_robot_type(robot_type)).as_posix()


@dataclass
class GrootTrtRobotClientConfig(BaseRobotClientConfig):
    """Runtime config for the GR00T TensorRT async robot client."""

    resource_profile: str = field(
        default=DEFAULT_SERVER_RESOURCE_PROFILE,
        metadata={"help": "Logical server-managed resource profile to request from the remote server."},
    )
    backend: str = field(default="tensorrt", metadata={"help": "Remote inference backend. MVP expects tensorrt."})
    session_id: str | None = field(
        default=None,
        metadata={"help": "Optional sticky session_id to reuse across client restarts."},
    )
    session_mode: str = field(
        default=SESSION_MODE_CLAIM,
        metadata={"help": "Sticky session claim mode for Ready(); use takeover to replace the active owner."},
    )
    engine_dir: str = field(
        default=DEFAULT_ENGINE_DIR,
        metadata={"help": "Deprecated client-side path. Remote loading now resolves server-managed resources."},
    )
    tensorrt_py_dir: str | None = field(
        default=None,
        metadata={"help": "Deprecated client-side path. Remote loading now resolves server-managed resources."},
    )
    vit_dtype: str = field(default="fp16", metadata={"help": "Remote ViT engine precision suffix."})
    llm_dtype: str = field(default="fp16", metadata={"help": "Remote LLM engine precision suffix."})
    dit_dtype: str = field(default="fp16", metadata={"help": "Remote DiT engine precision suffix."})
    num_denoising_steps: int | None = field(
        default=None,
        metadata={"help": "Optional override for GR00T denoising steps on the server."},
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.resource_profile:
            raise ValueError("resource_profile cannot be empty")
        if self.backend not in {"tensorrt", "pytorch"}:
            raise ValueError(f"Unsupported backend={self.backend!r}. Expected 'tensorrt' or 'pytorch'.")
        if self.session_id is not None:
            self.session_id = _normalize_required_str(self.session_id, field_name="session_id")
        self.session_mode = _normalize_session_mode(
            self.session_mode,
            field_name="session_mode",
            allowed=CLIENT_START_SESSION_MODES,
        )


def _resolve_remote_policy_config_class() -> type[Any] | None:
    """Prefer the module-local config class when it is importable in this checkout."""

    try:
        from my_devs.groot_trt_async_server.configs import GrootTrtRemotePolicyConfig as resolved_cls

        return resolved_cls
    except Exception:
        return None


def build_remote_policy_config(
    config: GrootTrtRobotClientConfig, lerobot_features: dict[str, Any]
) -> Any:
    config_cls = _resolve_remote_policy_config_class()
    placeholder_pretrained_path = f"server-resource://{config.resource_profile}"
    kwargs = {
        "policy_type": config.policy_type,
        "pretrained_name_or_path": placeholder_pretrained_path,
        "lerobot_features": lerobot_features,
        "actions_per_chunk": config.actions_per_chunk,
        "device": config.policy_device,
        "resource_profile": config.resource_profile,
        "backend": config.backend,
        "engine_dir": None,
        "tensorrt_py_dir": None,
        "vit_dtype": config.vit_dtype,
        "llm_dtype": config.llm_dtype,
        "dit_dtype": config.dit_dtype,
        "num_denoising_steps": config.num_denoising_steps,
    }

    if config_cls is not None:
        try:
            return config_cls(**kwargs)
        except TypeError:
            pass

    return GrootTrtRemotePolicyConfig(**kwargs)


class GrootTrtRobotClient(BaseRobotClient):
    """Async client that sends GR00T TensorRT policy setup to the remote server."""

    def __init__(self, config: GrootTrtRobotClientConfig):
        if _ROBOT_CLIENT_IMPORT_ERROR is not None:
            raise RuntimeError(
                "Failed to import lerobot.async_inference.robot_client. "
                "The current environment is missing optional async dependencies, likely `grpcio`. "
                "Install the repository with async extras before running this client."
            ) from _ROBOT_CLIENT_IMPORT_ERROR
        super().__init__(config)
        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)
        self.policy_config = build_remote_policy_config(config, lerobot_features)
        self._request_lock = threading.Lock()
        self._next_request_id = 0
        self._last_sent_observation_timestep = -1
        self._last_acknowledged_request_id = -1
        self._last_acknowledged_observation_timestep = -1
        self._pending_request: PendingActionRequest | None = None
        self._last_request_terminal: RequestTerminalState | None = None
        self._accepted_action_keys: set[tuple[int, int]] = set()
        self._session_id = config.session_id or uuid.uuid4().hex
        self._startup_session_mode = config.session_mode
        self._session_claimed = False
        self._stop_lock = threading.Lock()
        self._stop_started = False
        self._robot_io_lock = threading.Lock()
        action_horizon_seconds = config.actions_per_chunk / float(config.fps)
        refresh_lead_seconds = config.chunk_size_threshold * action_horizon_seconds
        self.logger.info(
            "Configured remote policy backend=%s resource_profile=%s session_id=%s start_mode=%s "
            "inflight_window=%s action_horizon=%.3fs refresh_lead=%.3fs aggregate_fn=%s",
            config.backend,
            config.resource_profile,
            self._session_id,
            self._startup_session_mode,
            MAX_INFLIGHT_OBSERVATION_REQUESTS,
            action_horizon_seconds,
            refresh_lead_seconds,
            config.aggregate_fn_name,
        )
        if refresh_lead_seconds < 0.5:
            self.logger.warning(
                "Current async timing budget is very tight: actions_per_chunk=%s fps=%s "
                "chunk_size_threshold=%.3f leaves only %.3fs of queued action coverage before refresh. "
                "Expect stale refreshes or repeated micro-corrections unless you lower fps or increase chunk size.",
                config.actions_per_chunk,
                config.fps,
                config.chunk_size_threshold,
                refresh_lead_seconds,
            )
        if config.pretrained_name_or_path != DEFAULT_POLICY_PATH or config.engine_dir != DEFAULT_ENGINE_DIR:
            self.logger.warning(
                "Client-supplied policy_path/engine_dir are no longer used for remote loading. "
                "The server will resolve resource_profile=%s locally.",
                config.resource_profile,
            )
        if config.backend == "tensorrt" and type(self.policy_config) is GrootTrtRemotePolicyConfig:
            self.logger.warning(
                "Using protocol-compatible client fallback payload for backend=tensorrt. "
                "This preserves the TensorRT contract while delegating resource resolution to the server."
            )

    def _session_metadata(self, *, mode: str = SESSION_MODE_CLAIM) -> tuple[tuple[str, str], ...]:
        return (
            (SESSION_ID_METADATA_KEY, self._session_id),
            (SESSION_MODE_METADATA_KEY, mode),
        )

    def _queued_actions_snapshot(self) -> list[TimedAction]:
        with self.action_queue_lock:
            return list(self.action_queue.queue)

    def _latest_action_timestep(self) -> int:
        with self.latest_action_lock:
            return self.latest_action

    def _reserve_observation_request(self, *, latest_action: int, queued_timesteps: list[int]) -> PendingActionRequest | None:
        queue_empty = not queued_timesteps
        requested_timestep = max(latest_action + 1, 0)
        must_go = True

        with self._request_lock:
            if self._pending_request is not None:
                self.logger.debug(
                    "Holding observation reservation because inflight request_id=%s obs_step=%s is still pending",
                    self._pending_request.request_id,
                    self._pending_request.observation_timestep,
                )
                return None

            if queue_empty:
                requested_timestep = max(requested_timestep, self._last_sent_observation_timestep + 1)

            request = PendingActionRequest(
                request_id=self._next_request_id,
                observation_timestep=requested_timestep,
                sent_at=time.time(),
                queue_empty=queue_empty,
                must_go=must_go,
            )
            self._next_request_id += 1
            self._last_sent_observation_timestep = requested_timestep
            self._pending_request = request
            return request

    def _clear_pending_request(self, *, request_id: int | None = None) -> None:
        with self._request_lock:
            if self._pending_request is None:
                return
            if request_id is None or self._pending_request.request_id == request_id:
                self._pending_request = None

    def _finalize_request_terminal(
        self,
        *,
        request_id: int,
        observation_timestep: int | None,
        session_id: str,
        request_state: str,
        request_state_reason: str | None,
    ) -> None:
        normalized_request_id = _normalize_required_int(request_id, field_name=REQUEST_ID_PAYLOAD_KEY)
        normalized_session_id = _normalize_required_str(session_id, field_name=SESSION_ID_PAYLOAD_KEY)
        normalized_request_state = _normalize_request_state(request_state, field_name=REQUEST_STATE_PAYLOAD_KEY)
        normalized_request_state_reason = _normalize_optional_str(
            request_state_reason,
            field_name=REQUEST_STATE_REASON_PAYLOAD_KEY,
        )

        with self._request_lock:
            if normalized_request_id > self._last_acknowledged_request_id:
                self._last_acknowledged_request_id = normalized_request_id
            if observation_timestep is not None:
                self._last_acknowledged_observation_timestep = max(
                    self._last_acknowledged_observation_timestep,
                    observation_timestep,
                )
            if (
                self._pending_request is not None
                and self._pending_request.request_id == normalized_request_id
            ):
                self._pending_request = None
            self._last_request_terminal = RequestTerminalState(
                request_id=normalized_request_id,
                observation_timestep=observation_timestep,
                session_id=normalized_session_id,
                request_state=normalized_request_state,
                request_state_reason=normalized_request_state_reason,
            )

        self.must_go.set()
        self.logger.warning(
            "Request terminal state=%s request_id=%s obs_step=%s session_id=%s reason=%s",
            normalized_request_state,
            normalized_request_id,
            observation_timestep,
            normalized_session_id,
            normalized_request_state_reason or "<none>",
        )

    def _finalize_pending_request_failure(
        self,
        *,
        request_state: str,
        request_state_reason: str,
    ) -> None:
        with self._request_lock:
            pending_request = self._pending_request

        if pending_request is None:
            return

        self._finalize_request_terminal(
            request_id=pending_request.request_id,
            observation_timestep=pending_request.observation_timestep,
            session_id=self._session_id,
            request_state=request_state,
            request_state_reason=request_state_reason,
        )

    def _release_claimed_session(self, *, reason: str) -> None:
        if not self._session_claimed:
            return

        try:
            self.stub.Ready(services_pb2.Empty(), metadata=self._session_metadata(mode=SESSION_MODE_RELEASE))
        except Exception as exc:
            self.logger.warning(
                "Failed to release sticky session_id=%s during %s: %s",
                self._session_id,
                reason,
                exc,
            )
        finally:
            self._session_claimed = False

    def _stop_in_progress(self) -> bool:
        return self._stop_started or self.shutdown_event.is_set()

    def _clear_pending_request_due_to_shutdown(
        self,
        *,
        request_id: int | None,
        observation_timestep: int | None,
        context: str,
    ) -> None:
        if request_id is not None:
            self._clear_pending_request(request_id=request_id)
        self.logger.debug(
            "Skipping request_id=%s obs_step=%s because shutdown is in progress during %s.",
            request_id,
            observation_timestep,
            context,
        )

    def _is_expected_shutdown_cancellation(self, error: Exception) -> bool:
        return (
            grpc is not None
            and isinstance(error, grpc.RpcError)
            and error.code() == grpc.StatusCode.CANCELLED
            and self._stop_in_progress()
        )

    def _should_drop_action_chunk(
        self,
        *,
        request_id: int | None,
        observation_timestep: int | None,
        session_id: str,
    ) -> bool:
        if request_id is None and observation_timestep is None:
            return False

        with self._request_lock:
            pending_request = self._pending_request
            last_acknowledged_request_id = self._last_acknowledged_request_id
            last_acknowledged_observation_timestep = self._last_acknowledged_observation_timestep

        if session_id != self._session_id:
            self.logger.debug(
                "Dropping action chunk for request_id=%s because session_id=%s does not match client session_id=%s",
                request_id,
                session_id,
                self._session_id,
            )
            return True

        if request_id is not None and request_id < last_acknowledged_request_id:
            self.logger.debug(
                "Dropping stale action chunk for request_id=%s because it is older than last_ack=%s",
                request_id,
                last_acknowledged_request_id,
            )
            return True

        pending_request_id = pending_request.request_id if pending_request is not None else None
        if request_id is not None and request_id == last_acknowledged_request_id and pending_request_id != request_id:
            self.logger.debug("Dropping duplicate action chunk for request_id=%s", request_id)
            return True

        if (
            observation_timestep is not None
            and observation_timestep < last_acknowledged_observation_timestep
        ):
            self.logger.debug(
                "Dropping stale action chunk for obs_step=%s because it is older than last_ack_obs=%s",
                observation_timestep,
                last_acknowledged_observation_timestep,
            )
            return True

        if pending_request is None:
            self.logger.debug(
                "Dropping unexpected action chunk for request_id=%s because there is no inflight request "
                "(last_ack=%s)",
                request_id,
                last_acknowledged_request_id,
            )
            return True

        pending_request_id = pending_request.request_id
        if request_id is not None and request_id != pending_request_id:
            self.logger.debug(
                "Dropping action chunk for request_id=%s because inflight request_id=%s",
                request_id,
                pending_request_id,
            )
            return True

        pending_observation_timestep = pending_request.observation_timestep
        if observation_timestep is not None and observation_timestep != pending_observation_timestep:
            self.logger.debug(
                "Dropping action chunk for request_id=%s because obs_step=%s does not match inflight obs_step=%s",
                request_id,
                observation_timestep,
                pending_observation_timestep,
            )
            return True

        return False

    def _acknowledge_action_chunk(self, *, request_id: int | None, observation_timestep: int | None) -> None:
        with self._request_lock:
            if (
                request_id is not None
                and self._pending_request is not None
                and self._pending_request.request_id == request_id
            ):
                self._last_acknowledged_request_id = max(self._last_acknowledged_request_id, request_id)
                self._pending_request = None

            if (
                observation_timestep is not None
                and request_id is not None
                and request_id <= self._last_acknowledged_request_id
            ):
                self._last_acknowledged_observation_timestep = max(
                    self._last_acknowledged_observation_timestep,
                    observation_timestep,
                )

    def _filter_action_chunk(self, timed_actions: list[TimedAction], *, request_id: int | None) -> list[TimedAction]:
        latest_action = self._latest_action_timestep()
        filtered_actions: list[TimedAction] = []

        for timed_action in timed_actions:
            timestep = timed_action.get_timestep()
            if timestep <= latest_action:
                self.logger.debug(
                    "Dropping late action for request_id=%s timestep=%s latest_action=%s",
                    request_id,
                    timestep,
                    latest_action,
                )
                continue

            if request_id is not None:
                action_key = (request_id, timestep)
                if action_key in self._accepted_action_keys:
                    self.logger.debug("Dropping duplicate action for request_id=%s timestep=%s", request_id, timestep)
                    continue
                self._accepted_action_keys.add(action_key)

            filtered_actions.append(timed_action)

        return filtered_actions

    def start(self) -> bool:
        """Start the robot client and connect to the policy server."""
        try:
            start_time = time.perf_counter()
            self.stub.Ready(
                services_pb2.Empty(),
                metadata=self._session_metadata(mode=self._startup_session_mode),
            )
            end_time = time.perf_counter()
            self._session_claimed = True
            self.logger.debug(
                "Connected to policy server in %.4fs with session_id=%s mode=%s",
                end_time - start_time,
                self._session_id,
                self._startup_session_mode,
            )

            policy_config_bytes = pickle.dumps(self.policy_config)  # nosec
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup, metadata=self._session_metadata())
            self.shutdown_event.clear()
            return True
        except Exception as e:
            self._release_claimed_session(reason="startup_failure")
            self.logger.error("Failed to connect to policy server with session_id=%s: %s", self._session_id, e)
            return False

    def stop(self) -> None:
        """Stop the robot client and release the sticky session when possible."""
        with self._stop_lock:
            if self._stop_started:
                self.logger.debug("stop() already in progress for session_id=%s; skipping duplicate shutdown.", self._session_id)
                return
            self._stop_started = True

        with self._robot_io_lock:
            self._release_claimed_session(reason="client_stop")
            super().stop()

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        """Perform queued actions while preventing shutdown/disconnect races."""
        if self._stop_in_progress():
            return {}

        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start

        with self._robot_io_lock:
            if self._stop_in_progress():
                with self.action_queue_lock:
                    self.action_queue.put(timed_action)
                return {}

            performed_action = self.robot.send_action(
                self._action_tensor_to_action_dict(timed_action.get_action())
            )

        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()

            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )

            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return performed_action

    def send_observation(self, obs: TimedObservation, *, request_id: int | None = None) -> bool:
        try:
            if self._stop_in_progress():
                self._clear_pending_request_due_to_shutdown(
                    request_id=request_id,
                    observation_timestep=obs.get_timestep() if isinstance(obs, TimedObservation) else None,
                    context="send_observation_preflight",
                )
                return False

            if not self.running:
                raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

            if not isinstance(obs, TimedObservation):
                raise ValueError("Input observation needs to be a TimedObservation!")

            if request_id is None:
                raise ValueError("request_id is required when sending observations to the remote server.")

            payload = build_observation_payload(obs, request_id=request_id, session_id=self._session_id)

            start_time = time.perf_counter()
            observation_bytes = pickle.dumps(payload)  # nosec
            serialize_time = time.perf_counter() - start_time
            self.logger.debug("Observation serialization time: %.6fs", serialize_time)

            observation_iterator = self._send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
            )
            _ = self.stub.SendObservations(observation_iterator, metadata=self._session_metadata())
            self.logger.debug("Sent observation #%s request_id=%s", obs.get_timestep(), request_id)
            return True
        except Exception as e:
            if self._is_expected_shutdown_cancellation(e):
                self._clear_pending_request_due_to_shutdown(
                    request_id=request_id,
                    observation_timestep=obs.get_timestep() if isinstance(obs, TimedObservation) else None,
                    context="send_observation_shutdown_cancelled",
                )
                return False

            if request_id is not None:
                failure_kind = "rpc_error" if grpc is not None and isinstance(e, grpc.RpcError) else "error"
                self._finalize_request_terminal(
                    request_id=request_id,
                    observation_timestep=obs.get_timestep() if isinstance(obs, TimedObservation) else None,
                    session_id=self._session_id,
                    request_state=REQUEST_STATE_RETRY,
                    request_state_reason=f"send_observation_{failure_kind}:{type(e).__name__}",
                )
            self.logger.error(
                "Error sending observation #%s request_id=%s: %s",
                obs.get_timestep() if isinstance(obs, TimedObservation) else "<invalid>",
                request_id,
                e,
            )
            return False

    @staticmethod
    def _send_bytes_in_chunks(observation_bytes: bytes, proto_cls: Any, *, log_prefix: str):
        from lerobot.transport.utils import send_bytes_in_chunks

        return send_bytes_in_chunks(
            observation_bytes,
            proto_cls,
            log_prefix=log_prefix,
            silent=True,
        )

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if aggregate_fn is None:
            def aggregate_fn(_current: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
                return new

        latest_action = self._latest_action_timestep()
        queued_actions = self._queued_actions_snapshot()
        merged_actions = {
            action.get_timestep(): action for action in queued_actions if action.get_timestep() > latest_action
        }

        for new_action in incoming_actions:
            timestep = new_action.get_timestep()
            if timestep <= latest_action:
                continue

            if timestep in merged_actions:
                existing = merged_actions[timestep]
                merged_actions[timestep] = TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=timestep,
                    action=aggregate_fn(existing.get_action(), new_action.get_action()),
                )
            else:
                merged_actions[timestep] = new_action

        future_action_queue = Queue()
        for timestep in sorted(merged_actions):
            future_action_queue.put(merged_actions[timestep])

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self, verbose: bool = False) -> None:
        """Receive actions from the policy server."""
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                actions_chunk = self.stub.GetActions(services_pb2.Empty(), metadata=self._session_metadata())
                if len(actions_chunk.data) == 0:
                    continue

                receive_time = time.time()
                deserialize_start = time.perf_counter()
                payload = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start
                response = unpack_action_payload(payload)

                if response.session_id != self._session_id:
                    self._finalize_request_terminal(
                        request_id=response.request_id,
                        observation_timestep=response.observation_timestep,
                        session_id=self._session_id,
                        request_state=REQUEST_STATE_ABORT,
                        request_state_reason=f"foreign_session_chunk:{response.session_id}",
                    )
                    continue

                if self._should_drop_action_chunk(
                    request_id=response.request_id,
                    observation_timestep=response.observation_timestep,
                    session_id=response.session_id,
                ):
                    continue

                if response.is_failure_terminal:
                    self._finalize_request_terminal(
                        request_id=response.request_id,
                        observation_timestep=response.observation_timestep,
                        session_id=response.session_id,
                        request_state=response.request_state,
                        request_state_reason=response.request_state_reason,
                    )
                    continue

                self._acknowledge_action_chunk(
                    request_id=response.request_id,
                    observation_timestep=response.observation_timestep,
                )
                timed_actions = self._filter_action_chunk(response.timed_actions, request_id=response.request_id)
                if not timed_actions:
                    continue

                if len(timed_actions) > 0:
                    received_device = timed_actions[0].get_action().device.type
                    self.logger.debug(f"Received actions on device: {received_device}")

                client_device = self.config.client_device
                if client_device != "cpu":
                    for timed_action in timed_actions:
                        if timed_action.get_action().device.type != client_device:
                            timed_action.action = timed_action.get_action().to(client_device)
                    self.logger.debug(f"Converted actions to device: {client_device}")
                else:
                    self.logger.debug(f"Actions kept on device: {client_device}")

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                if len(timed_actions) > 0 and verbose:
                    latest_action = self._latest_action_timestep()
                    self.logger.debug(f"Current latest action: {latest_action}")

                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]

                    incoming_timesteps = [a.get_timestep() for a in timed_actions]
                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Request #{response.request_id} | "
                        f"Observation step #{response.observation_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time

                self.must_go.set()

                if verbose and timed_actions:
                    new_size, new_timesteps = self._inspect_action_queue()
                    latest_action = self._latest_action_timestep()

                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                        f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )

            except Exception as e:
                if self._is_expected_shutdown_cancellation(e):
                    self._clear_pending_request()
                    self.logger.debug("GetActions cancelled during shutdown; exiting receive_actions thread.")
                    break

                request_state = REQUEST_STATE_RETRY
                reason_prefix = "get_actions_rpc_error" if grpc is not None and isinstance(e, grpc.RpcError) else "receive_actions_error"
                self._finalize_pending_request_failure(
                    request_state=request_state,
                    request_state_reason=f"{reason_prefix}:{type(e).__name__}",
                )
                self.logger.error("Error receiving actions: %s", e)

    def _ready_to_send_observation(self) -> bool:
        with self._request_lock:
            if self._pending_request is not None:
                return False

        queued_actions = self._queued_actions_snapshot()
        queue_size = len(queued_actions)

        if queue_size == 0:
            return True

        if self.action_chunk_size <= 0:
            return True

        if queue_size / self.action_chunk_size > self._chunk_size_threshold:
            return False

        return True

    def control_loop_observation(self, task: str, verbose: bool = False) -> dict[str, Any] | None:
        request: PendingActionRequest | None = None
        try:
            start_time = time.perf_counter()
            if self._stop_in_progress():
                return None

            with self._robot_io_lock:
                if self._stop_in_progress():
                    return None
                raw_observation = self.robot.get_observation()
            raw_observation["task"] = task

            queued_actions = self._queued_actions_snapshot()
            queued_timesteps = sorted(action.get_timestep() for action in queued_actions)
            latest_action = self._latest_action_timestep()
            request = self._reserve_observation_request(
                latest_action=latest_action,
                queued_timesteps=queued_timesteps,
            )
            if request is None:
                return None

            if self._stop_in_progress():
                self._clear_pending_request_due_to_shutdown(
                    request_id=request.request_id,
                    observation_timestep=request.observation_timestep,
                    context="control_loop_observation_post_reserve",
                )
                return None

            observation = TimedObservation(
                timestamp=time.time(),
                observation=raw_observation,
                timestep=request.observation_timestep,
                must_go=request.must_go,
            )

            obs_capture_time = time.perf_counter() - start_time
            current_queue_size = len(queued_actions)
            send_ok = self.send_observation(observation, request_id=request.request_id)
            if not send_ok:
                self._clear_pending_request(request_id=request.request_id)
                return None

            self.logger.debug(
                "Sent observation request_id=%s obs_step=%s latest_action=%s queue_size=%s must_go=%s",
                request.request_id,
                request.observation_timestep,
                latest_action,
                current_queue_size,
                observation.must_go,
            )
            self.must_go.clear()

            if verbose:
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())
                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Request #{request.request_id} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )
                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            if request is not None:
                self._finalize_request_terminal(
                    request_id=request.request_id,
                    observation_timestep=request.observation_timestep,
                    session_id=self._session_id,
                    request_state=REQUEST_STATE_RETRY,
                    request_state_reason=f"control_loop_observation_error:{type(e).__name__}",
                )
            self.logger.error(f"Error in observation sender: {e}")
            return None


class MockGrootTrtRobotClient(GrootTrtRobotClient):
    """Client variant that consumes returned actions without sending them to the robot."""

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            timed_action: TimedAction = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start

        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        action_dict = self._action_tensor_to_action_dict(timed_action.get_action())

        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()
            self.logger.info(
                "[MOCK] Skipped send_action for step=%s queue_size=%s",
                timed_action.get_timestep(),
                current_queue_size,
            )
            self.logger.debug(
                "[MOCK] Popping queued action took %.6fs | step=%s",
                get_end,
                timed_action.get_timestep(),
            )

        return action_dict


@dataclass(frozen=True)
class MockSessionContext:
    peer_id: str
    metadata: tuple[tuple[str, str], ...]

    def peer(self) -> str:
        return self.peer_id

    def invocation_metadata(self) -> tuple[tuple[str, str], ...]:
        return self.metadata

    def abort(self, code: object, details: str) -> None:
        error_cls = RuntimeError
        if grpc is not None:
            error_cls = getattr(grpc, "RpcError", RuntimeError)
        raise error_cls(f"{code}: {details}")


class _SilentLogger:
    def debug(self, *args, **kwargs) -> None:
        return None

    info = debug
    warning = debug
    error = debug


def _set_pure_mock_client_running(client: GrootTrtRobotClient, *, running: bool) -> None:
    if running:
        client.shutdown_event.clear()
    else:
        client.shutdown_event.set()


def _build_pure_mock_client(*, session_id: str) -> GrootTrtRobotClient:
    client = object.__new__(GrootTrtRobotClient)
    client._request_lock = threading.Lock()
    client._next_request_id = 0
    client._last_sent_observation_timestep = -1
    client._last_acknowledged_request_id = -1
    client._last_acknowledged_observation_timestep = -1
    client._pending_request = None
    client._last_request_terminal = None
    client._accepted_action_keys = set()
    client._session_id = session_id
    client._startup_session_mode = SESSION_MODE_CLAIM
    client._session_claimed = False
    client._stop_lock = threading.Lock()
    client._stop_started = False
    client._robot_io_lock = threading.Lock()
    client.action_queue = Queue()
    client.action_queue_lock = threading.Lock()
    client.latest_action_lock = threading.Lock()
    client.latest_action = -1
    client.action_chunk_size = 4
    client._chunk_size_threshold = 0.5
    client.must_go = threading.Event()
    client.must_go.set()
    client.shutdown_event = threading.Event()
    _set_pure_mock_client_running(client, running=True)
    client.start_barrier = SimpleNamespace(wait=lambda: None)
    client.action_queue_size = []
    client.config = SimpleNamespace(client_device="cpu", aggregate_fn=None)
    client.logger = _SilentLogger()
    return client


def run_pure_protocol_roundtrip_mock() -> None:
    from my_devs.groot_trt_async_server.configs import GrootTrtPolicyServerConfig
    from my_devs.groot_trt_async_server.policy_server import GrootTrtPolicyServer

    def expect_failure(label: str, func: Callable[[], Any], expected_substring: str) -> None:
        try:
            func()
        except Exception as exc:
            if expected_substring not in str(exc):
                raise RuntimeError(
                    f"{label} raised the wrong error: expected substring {expected_substring!r}, got {exc!r}"
                ) from exc
        else:
            raise RuntimeError(f"{label} unexpectedly succeeded")

    def make_action(step: int) -> TimedAction:
        return TimedAction(
            timestamp=time.time(),
            timestep=step,
            action=torch.full((4,), float(step)),
        )

    class PureMockRpcError(RuntimeError):
        pass

    class _PureMockEmpty:
        def __init__(self, data: bytes = b"") -> None:
            self.data = data

    class _PureMockObservation:
        def __init__(self, data: bytes = b"") -> None:
            self.data = data

    class _PureMockPolicySetup:
        def __init__(self, data: bytes = b"") -> None:
            self.data = data

    class _PureMockActions:
        def __init__(self, data: bytes = b"") -> None:
            self.data = data

    class _PureMockStub:
        def __init__(self, server: Any, *, peer_id: str) -> None:
            self.server = server
            self.peer_id = peer_id
            self.client: GrootTrtRobotClient | None = None
            self.scripted_action_responses: list[Any] = []
            self.policy_setup_error: Exception | None = None

        def _context(self, metadata: tuple[tuple[str, str], ...]) -> MockSessionContext:
            return MockSessionContext(peer_id=self.peer_id, metadata=tuple(metadata))

        def Ready(self, request: Any, metadata: tuple[tuple[str, str], ...] = ()) -> Any:  # noqa: N802
            return self.server.Ready(request, self._context(metadata))

        def SendPolicyInstructions(self, request: Any, metadata: tuple[tuple[str, str], ...] = ()) -> Any:  # noqa: N802
            if self.policy_setup_error is not None:
                error = self.policy_setup_error
                self.policy_setup_error = None
                raise error
            return _PureMockEmpty()

        def SendObservations(self, request_iterator: Any, metadata: tuple[tuple[str, str], ...] = ()) -> Any:  # noqa: N802
            return self.server.SendObservations(request_iterator, self._context(metadata))

        def GetActions(self, request: Any, metadata: tuple[tuple[str, str], ...] = ()) -> Any:  # noqa: N802
            if self.scripted_action_responses:
                response = self.scripted_action_responses.pop(0)
            else:
                response = self.server.GetActions(request, self._context(metadata))

            if self.client is not None and not self.scripted_action_responses:
                _set_pure_mock_client_running(self.client, running=False)
            return response

    protocol_session_id = "session-alpha"
    observation = TimedObservation(
        timestamp=time.time(),
        observation={"task": "pure-mock", "observation.mock": torch.zeros(1)},
        timestep=7,
        must_go=True,
    )
    request_id = 42
    observation_payload = build_observation_payload(
        observation,
        request_id=request_id,
        session_id=protocol_session_id,
    )
    decoded_observation, decoded_request_id, decoded_session_id = unpack_observation_payload(observation_payload)
    if decoded_request_id != request_id:
        raise RuntimeError(f"Observation request_id mismatch: expected {request_id}, got {decoded_request_id}")
    if decoded_session_id != protocol_session_id:
        raise RuntimeError(
            f"Observation session_id mismatch: expected {protocol_session_id}, got {decoded_session_id}"
        )

    timed_actions = [
        TimedAction(
            timestamp=time.time(),
            timestep=decoded_observation.get_timestep(),
            action=torch.zeros(4),
        )
    ]
    action_payload = build_action_payload(
        timed_actions,
        request_id=decoded_request_id,
        observation_timestep=decoded_observation.get_timestep(),
        session_id=decoded_session_id,
    )
    decoded_response = unpack_action_payload(action_payload)

    if action_payload.get(ACKED_REQUEST_ID_PAYLOAD_KEY) != request_id:
        raise RuntimeError(
            "Action payload did not expose an explicit acked_request_id field: "
            f"expected {request_id}, got {action_payload.get(ACKED_REQUEST_ID_PAYLOAD_KEY)}"
        )
    if action_payload.get(SESSION_ID_PAYLOAD_KEY) != protocol_session_id:
        raise RuntimeError(
            "Action payload did not expose an explicit session_id field: "
            f"expected {protocol_session_id}, got {action_payload.get(SESSION_ID_PAYLOAD_KEY)}"
        )
    if decoded_response.request_id != request_id:
        raise RuntimeError(f"Ack request_id mismatch: expected {request_id}, got {decoded_response.request_id}")
    if decoded_response.observation_timestep != decoded_observation.get_timestep():
        raise RuntimeError(
            f"Ack observation_timestep mismatch: expected {decoded_observation.get_timestep()}, "
            f"got {decoded_response.observation_timestep}"
        )
    if decoded_response.session_id != protocol_session_id:
        raise RuntimeError(f"Ack session_id mismatch: expected {protocol_session_id}, got {decoded_response.session_id}")
    if decoded_response.request_state != REQUEST_STATE_ACK:
        raise RuntimeError(f"Action payload should default to request_state=ack, got {decoded_response.request_state}")
    if (
        len(decoded_response.timed_actions) != 1
        or decoded_response.timed_actions[0].get_timestep() != decoded_observation.get_timestep()
    ):
        raise RuntimeError("Decoded action payload did not round-trip correctly")

    expect_failure(
        "missing observation request_id",
        lambda: GrootTrtPolicyServer._decode_observation_payload({OBSERVATION_PAYLOAD_KEY: observation}),
        REQUEST_ID_PAYLOAD_KEY,
    )
    expect_failure(
        "missing observation session_id",
        lambda: GrootTrtPolicyServer._decode_observation_payload(
            {
                OBSERVATION_PAYLOAD_KEY: observation,
                REQUEST_ID_PAYLOAD_KEY: request_id,
            }
        ),
        SESSION_ID_PAYLOAD_KEY,
    )
    expect_failure(
        "missing action acked_request_id",
        lambda: unpack_action_payload(
            {
                ACTION_PAYLOAD_KEY: timed_actions,
                REQUEST_ID_PAYLOAD_KEY: request_id,
                OBSERVATION_TIMESTEP_PAYLOAD_KEY: decoded_observation.get_timestep(),
                SESSION_ID_PAYLOAD_KEY: protocol_session_id,
            }
        ),
        ACKED_REQUEST_ID_PAYLOAD_KEY,
    )
    expect_failure(
        "missing action session_id",
        lambda: unpack_action_payload(
            {
                ACTION_PAYLOAD_KEY: timed_actions,
                REQUEST_ID_PAYLOAD_KEY: request_id,
                ACKED_REQUEST_ID_PAYLOAD_KEY: request_id,
                OBSERVATION_TIMESTEP_PAYLOAD_KEY: decoded_observation.get_timestep(),
            }
        ),
        SESSION_ID_PAYLOAD_KEY,
    )
    expect_failure(
        "legacy bare action payload",
        lambda: unpack_action_payload(timed_actions),
        "Unsupported action payload type",
    )

    protocol_client = _build_pure_mock_client(session_id=protocol_session_id)
    with protocol_client.action_queue_lock:
        protocol_client.action_queue.put(make_action(0))
        protocol_client.action_queue.put(make_action(1))

    initial_request = protocol_client._reserve_observation_request(latest_action=-1, queued_timesteps=[0, 1])
    if initial_request is None:
        raise RuntimeError("Failed to reserve the initial inflight observation request in pure mock mode")
    if protocol_client._ready_to_send_observation():
        raise RuntimeError("Single inflight window allowed a second request while the first was still pending")
    if protocol_client._reserve_observation_request(latest_action=-1, queued_timesteps=[0, 1, 2]) is not None:
        raise RuntimeError("Pending request was overwritten instead of respecting the single inflight window")

    with protocol_client.action_queue_lock:
        protocol_client.action_queue = Queue()
    with protocol_client.latest_action_lock:
        protocol_client.latest_action = 1

    if protocol_client._ready_to_send_observation():
        raise RuntimeError("An empty action queue reopened observation sends before the slow inflight response landed")

    late_actions = [make_action(2), make_action(3)]
    if protocol_client._should_drop_action_chunk(
        request_id=initial_request.request_id,
        observation_timestep=initial_request.observation_timestep,
        session_id=protocol_session_id,
    ):
        raise RuntimeError("Slow response for the active inflight request was dropped unexpectedly")

    protocol_client._acknowledge_action_chunk(
        request_id=initial_request.request_id,
        observation_timestep=initial_request.observation_timestep,
    )
    filtered_actions = protocol_client._filter_action_chunk(late_actions, request_id=initial_request.request_id)
    protocol_client._aggregate_action_queues(filtered_actions)

    with protocol_client.action_queue_lock:
        refilled_timesteps = [action.get_timestep() for action in protocol_client.action_queue.queue]
    if refilled_timesteps != [2, 3]:
        raise RuntimeError(f"Slow response did not refill the action queue correctly: {refilled_timesteps}")

    if not protocol_client._should_drop_action_chunk(
        request_id=initial_request.request_id,
        observation_timestep=initial_request.observation_timestep,
        session_id=protocol_session_id,
    ):
        raise RuntimeError("Duplicate action chunk was accepted after its request had already been acknowledged")

    followup_request = protocol_client._reserve_observation_request(latest_action=1, queued_timesteps=refilled_timesteps)
    if followup_request is None:
        raise RuntimeError("Client did not reopen the request window after the inflight request was acknowledged")
    if not protocol_client._should_drop_action_chunk(
        request_id=followup_request.request_id,
        observation_timestep=followup_request.observation_timestep,
        session_id="session-foreign",
    ):
        raise RuntimeError("Mismatched session_id action chunk was not rejected in pure mock mode")
    protocol_client._clear_pending_request(request_id=followup_request.request_id)

    server = GrootTrtPolicyServer(GrootTrtPolicyServerConfig())
    initial_session_id = "session-alpha"
    reconnect_session_id = initial_session_id
    replacement_session_id = "session-beta"
    reclaimed_session_id = replacement_session_id
    final_session_id = "session-gamma"

    server.Ready(
        None,
        MockSessionContext(
            peer_id="ipv4:127.0.0.1:5000",
            metadata=((SESSION_ID_METADATA_KEY, initial_session_id), (SESSION_MODE_METADATA_KEY, SESSION_MODE_CLAIM)),
        ),
    )
    if server._active_session_id != initial_session_id or server._active_client_id != "ipv4:127.0.0.1:5000":
        raise RuntimeError("Sticky session claim did not bind the initial peer correctly")

    server.Ready(
        None,
        MockSessionContext(
            peer_id="ipv4:127.0.0.1:5001",
            metadata=((SESSION_ID_METADATA_KEY, reconnect_session_id), (SESSION_MODE_METADATA_KEY, SESSION_MODE_CLAIM)),
        ),
    )
    if server._active_session_id != reconnect_session_id or server._active_client_id != "ipv4:127.0.0.1:5001":
        raise RuntimeError("Sticky session reconnect did not refresh the peer for the active session")

    expect_failure(
        "conflicting session claim",
        lambda: server.Ready(
            None,
            MockSessionContext(
                peer_id="ipv4:127.0.0.1:5999",
                metadata=((SESSION_ID_METADATA_KEY, replacement_session_id), (SESSION_MODE_METADATA_KEY, SESSION_MODE_CLAIM)),
            ),
        ),
        "RESOURCE_EXHAUSTED",
    )

    server.Ready(
        None,
        MockSessionContext(
            peer_id="ipv4:127.0.0.1:6000",
            metadata=(
                (SESSION_ID_METADATA_KEY, replacement_session_id),
                (SESSION_MODE_METADATA_KEY, SESSION_MODE_TAKEOVER),
            ),
        ),
    )
    if server._active_session_id != replacement_session_id or server._active_client_id != "ipv4:127.0.0.1:6000":
        raise RuntimeError("Sticky session takeover did not replace the previous session owner")

    expect_failure(
        "stale release after takeover",
        lambda: server.Ready(
            None,
            MockSessionContext(
                peer_id="ipv4:127.0.0.1:5001",
                metadata=((SESSION_ID_METADATA_KEY, reconnect_session_id), (SESSION_MODE_METADATA_KEY, SESSION_MODE_RELEASE)),
            ),
        ),
        "RESOURCE_EXHAUSTED",
    )

    server.Ready(
        None,
        MockSessionContext(
            peer_id="ipv4:127.0.0.1:6001",
            metadata=((SESSION_ID_METADATA_KEY, reclaimed_session_id), (SESSION_MODE_METADATA_KEY, SESSION_MODE_CLAIM)),
        ),
    )
    if server._active_session_id != reclaimed_session_id or server._active_client_id != "ipv4:127.0.0.1:6001":
        raise RuntimeError("Taken-over session did not allow recovery by reusing the replacement session_id")

    expect_failure(
        "stale peer release after recovery",
        lambda: server.Ready(
            None,
            MockSessionContext(
                peer_id="ipv4:127.0.0.1:6000",
                metadata=((SESSION_ID_METADATA_KEY, reclaimed_session_id), (SESSION_MODE_METADATA_KEY, SESSION_MODE_RELEASE)),
            ),
        ),
        "RESOURCE_EXHAUSTED",
    )

    server.Ready(
        None,
        MockSessionContext(
            peer_id="ipv4:127.0.0.1:6001",
            metadata=((SESSION_ID_METADATA_KEY, reclaimed_session_id), (SESSION_MODE_METADATA_KEY, SESSION_MODE_RELEASE)),
        ),
    )
    if server._active_session_id is not None or server._active_client_id is not None:
        raise RuntimeError("Sticky session release did not clear the active session")

    server.Ready(
        None,
        MockSessionContext(
            peer_id="ipv4:127.0.0.1:7000",
            metadata=((SESSION_ID_METADATA_KEY, final_session_id), (SESSION_MODE_METADATA_KEY, SESSION_MODE_CLAIM)),
        ),
    )
    if server._active_session_id != final_session_id or server._active_client_id != "ipv4:127.0.0.1:7000":
        raise RuntimeError("Sticky session did not allow reclaim after release")

    import my_devs.groot_trt_async_server.policy_server as policy_server_module

    original_client_grpc = grpc
    original_client_services_pb2 = services_pb2
    original_server_services_pb2 = policy_server_module.services_pb2
    original_receive_bytes_in_chunks = policy_server_module.receive_bytes_in_chunks

    try:
        globals()["grpc"] = SimpleNamespace(RpcError=PureMockRpcError)
        globals()["services_pb2"] = SimpleNamespace(
            Empty=_PureMockEmpty,
            Observation=_PureMockObservation,
            PolicySetup=_PureMockPolicySetup,
            Actions=_PureMockActions,
        )
        policy_server_module.services_pb2 = SimpleNamespace(Empty=_PureMockEmpty, Actions=_PureMockActions)
        policy_server_module.receive_bytes_in_chunks = (
            lambda request_iterator, *_args: request_iterator.data if hasattr(request_iterator, "data") else request_iterator
        )

        startup_server = GrootTrtPolicyServer(GrootTrtPolicyServerConfig())
        startup_client = _build_pure_mock_client(session_id="startup-failure-session")
        startup_stub = _PureMockStub(startup_server, peer_id="ipv4:127.0.0.1:7050")
        startup_stub.client = startup_client
        startup_stub.policy_setup_error = PureMockRpcError("pure-mock setup failure")
        startup_client.stub = startup_stub
        startup_client.policy_config = SimpleNamespace(
            policy_type="groot",
            pretrained_name_or_path="server-resource://pure-mock-startup",
            device="cpu",
        )
        if startup_client.start():
            raise RuntimeError("Pure mock startup failure scenario unexpectedly succeeded")
        if startup_server._active_session_id is not None or startup_server._active_client_id is not None:
            raise RuntimeError("Ready()+SendPolicyInstructions failure leaked the sticky session on the server")
        if startup_client._session_claimed:
            raise RuntimeError("Client kept _session_claimed=True after startup failure cleanup")

        send_failure_server = GrootTrtPolicyServer(GrootTrtPolicyServerConfig())
        send_failure_client = _build_pure_mock_client(session_id="send-failure-session")
        send_failure_stub = _PureMockStub(send_failure_server, peer_id="ipv4:127.0.0.1:7060")
        send_failure_stub.client = send_failure_client
        send_failure_client.stub = send_failure_stub
        send_failure_client.policy_config = SimpleNamespace(
            policy_type="groot",
            pretrained_name_or_path="server-resource://pure-mock-send",
            device="cpu",
        )
        send_failure_client.robot = SimpleNamespace(
            get_observation=lambda: {"observation.mock": torch.ones(1)},
        )
        send_failure_client._send_bytes_in_chunks = (
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("pure-mock send serialization failure"))
        )
        if not send_failure_client.start():
            raise RuntimeError("Pure mock send-failure client failed to start")
        if send_failure_client.control_loop_observation(task="pure-mock-send-failure") is not None:
            raise RuntimeError("Pure mock send-failure scenario unexpectedly produced an observation payload")
        if send_failure_client._pending_request is not None:
            raise RuntimeError("Send-side ordinary exception left _pending_request hanging")
        if send_failure_client._last_request_terminal is None:
            raise RuntimeError("Send-side ordinary exception did not record a terminal state")
        if send_failure_client._last_request_terminal.request_state != REQUEST_STATE_RETRY:
            raise RuntimeError(
                "Send-side ordinary exception should end in retry terminal, "
                f"got {send_failure_client._last_request_terminal.request_state!r}"
            )
        if not (send_failure_client._last_request_terminal.request_state_reason or "").startswith("send_observation_error:"):
            raise RuntimeError(
                "Send-side ordinary exception did not use the expected terminal reason prefix: "
                f"{send_failure_client._last_request_terminal.request_state_reason!r}"
            )

        receive_failure_server = GrootTrtPolicyServer(GrootTrtPolicyServerConfig())

        def receive_failure_enqueue_observation(observation: TimedObservation) -> bool:
            receive_failure_server.observation_queue.put(observation)
            return True

        receive_failure_server._enqueue_observation = receive_failure_enqueue_observation  # type: ignore[method-assign]
        receive_failure_client = _build_pure_mock_client(session_id="receive-failure-session")
        receive_failure_stub = _PureMockStub(receive_failure_server, peer_id="ipv4:127.0.0.1:7070")
        receive_failure_stub.client = receive_failure_client
        receive_failure_client.stub = receive_failure_stub
        receive_failure_client.policy_config = SimpleNamespace(
            policy_type="groot",
            pretrained_name_or_path="server-resource://pure-mock-receive",
            device="cpu",
        )
        receive_failure_client.robot = SimpleNamespace(
            get_observation=lambda: {"observation.mock": torch.ones(1)},
        )
        receive_failure_client._send_bytes_in_chunks = (
            lambda observation_bytes, _proto_cls, *, log_prefix: _PureMockObservation(data=observation_bytes)
        )
        if not receive_failure_client.start():
            raise RuntimeError("Pure mock receive-failure client failed to start")
        if receive_failure_client.control_loop_observation(task="pure-mock-receive-failure") is None:
            raise RuntimeError("Pure mock receive-failure scenario did not send the inflight request")
        receive_failure_stub.scripted_action_responses = [_PureMockActions(data=b"not-a-pickle-payload")]
        receive_failure_thread = threading.Thread(target=receive_failure_client.receive_actions, daemon=True)
        _set_pure_mock_client_running(receive_failure_client, running=True)
        receive_failure_thread.start()
        receive_failure_thread.join(timeout=2.0)
        if receive_failure_thread.is_alive():
            raise RuntimeError("Pure mock receive-failure thread did not terminate")
        if receive_failure_client._pending_request is not None:
            raise RuntimeError("Receive-side ordinary exception left _pending_request hanging")
        if receive_failure_client._last_request_terminal is None:
            raise RuntimeError("Receive-side ordinary exception did not record a terminal state")
        if receive_failure_client._last_request_terminal.request_state != REQUEST_STATE_RETRY:
            raise RuntimeError(
                "Receive-side ordinary exception should end in retry terminal, "
                f"got {receive_failure_client._last_request_terminal.request_state!r}"
            )
        if not (receive_failure_client._last_request_terminal.request_state_reason or "").startswith(
            "receive_actions_error:"
        ):
            raise RuntimeError(
                "Receive-side ordinary exception did not use the expected terminal reason prefix: "
                f"{receive_failure_client._last_request_terminal.request_state_reason!r}"
            )

        entrypoint_server = GrootTrtPolicyServer(GrootTrtPolicyServerConfig())

        def enqueue_observation(observation: TimedObservation) -> bool:
            entrypoint_server.observation_queue.put(observation)
            return True

        def successful_predict_action_chunk(observation: TimedObservation) -> list[TimedAction]:
            base_timestep = observation.get_timestep() + 1
            return [make_action(base_timestep), make_action(base_timestep + 1)]

        def failing_predict_action_chunk(_observation: TimedObservation) -> list[TimedAction]:
            raise RuntimeError("pure-mock server exception")

        entrypoint_server._enqueue_observation = enqueue_observation  # type: ignore[method-assign]
        entrypoint_server._predict_action_chunk = successful_predict_action_chunk  # type: ignore[method-assign]

        entrypoint_client = _build_pure_mock_client(session_id="entrypoint-session")
        entrypoint_stub = _PureMockStub(entrypoint_server, peer_id="ipv4:127.0.0.1:7100")
        entrypoint_stub.client = entrypoint_client
        entrypoint_client.stub = entrypoint_stub
        entrypoint_client.policy_config = SimpleNamespace(
            policy_type="groot",
            pretrained_name_or_path="server-resource://pure-mock",
            device="cpu",
        )
        entrypoint_client.robot = SimpleNamespace(
            get_observation=lambda: {"observation.mock": torch.ones(1)},
        )
        entrypoint_client._send_bytes_in_chunks = (
            lambda observation_bytes, _proto_cls, *, log_prefix: _PureMockObservation(data=observation_bytes)
        )

        if not entrypoint_client.start():
            raise RuntimeError("Pure mock client start() failed")

        stale_payload = build_observation_payload(
            TimedObservation(
                timestamp=time.time(),
                observation={"task": "pure-mock-stale", "observation.mock": torch.ones(1)},
                timestep=91,
                must_go=True,
            ),
            request_id=91,
            session_id="session-mismatch",
        )
        expect_failure(
            "stale observation payload mismatch",
            lambda: entrypoint_server.SendObservations(
                _PureMockObservation(data=pickle.dumps(stale_payload)),  # nosec
                MockSessionContext(
                    peer_id="ipv4:127.0.0.1:7100",
                    metadata=((SESSION_ID_METADATA_KEY, "entrypoint-session"),),
                ),
            ),
            "request_id=91",
        )

        first_raw_observation = entrypoint_client.control_loop_observation(task="pure-mock-entrypoint")
        if first_raw_observation is None:
            raise RuntimeError("control_loop_observation() did not drive the real send path in pure mock mode")

        entrypoint_server._predict_action_chunk = failing_predict_action_chunk  # type: ignore[method-assign]
        failure_receiver = threading.Thread(target=entrypoint_client.receive_actions, daemon=True)
        _set_pure_mock_client_running(entrypoint_client, running=True)
        failure_receiver.start()
        failure_receiver.join(timeout=2.0)
        if failure_receiver.is_alive():
            raise RuntimeError("Pure mock failure receiver thread did not terminate")
        if entrypoint_client._pending_request is not None:
            raise RuntimeError("Failure terminal did not release the inflight request")
        if entrypoint_client._last_request_terminal is None:
            raise RuntimeError("Failure terminal was not recorded on the client")
        if entrypoint_client._last_request_terminal.request_state != REQUEST_STATE_RETRY:
            raise RuntimeError(
                "Server exception should terminate the request with retry semantics, "
                f"got {entrypoint_client._last_request_terminal.request_state!r}"
            )

        if entrypoint_client._reserve_observation_request(latest_action=-1, queued_timesteps=[]) is None:
            raise RuntimeError("Client did not reopen the inflight window after a retryable terminal")
        entrypoint_client._clear_pending_request()

        session_switch_request = entrypoint_client._reserve_observation_request(latest_action=-1, queued_timesteps=[])
        if session_switch_request is None:
            raise RuntimeError("Failed to reserve request for foreign-session drop check")

        foreign_session_payload = build_action_payload(
            [make_action(session_switch_request.observation_timestep + 1)],
            request_id=session_switch_request.request_id,
            observation_timestep=session_switch_request.observation_timestep,
            session_id="entrypoint-session-foreign",
        )
        entrypoint_stub.scripted_action_responses = [_PureMockActions(data=pickle.dumps(foreign_session_payload))]  # nosec
        session_switch_receiver = threading.Thread(target=entrypoint_client.receive_actions, daemon=True)
        _set_pure_mock_client_running(entrypoint_client, running=True)
        session_switch_receiver.start()
        session_switch_receiver.join(timeout=2.0)
        if session_switch_receiver.is_alive():
            raise RuntimeError("Pure mock session-switch receiver thread did not terminate")
        if entrypoint_client._pending_request is not None:
            raise RuntimeError("Foreign-session chunk did not release the inflight request directly")
        if entrypoint_client._last_request_terminal is None:
            raise RuntimeError("Foreign-session chunk did not record a terminal state")
        if entrypoint_client._last_request_terminal.request_state != REQUEST_STATE_ABORT:
            raise RuntimeError(
                "Foreign-session chunk should end in an abort terminal, "
                f"got {entrypoint_client._last_request_terminal.request_state!r}"
            )
        if not (entrypoint_client._last_request_terminal.request_state_reason or "").startswith("foreign_session_chunk:"):
            raise RuntimeError(
                "Foreign-session chunk did not use the expected terminal reason prefix: "
                f"{entrypoint_client._last_request_terminal.request_state_reason!r}"
            )
        if entrypoint_client._reserve_observation_request(latest_action=-1, queued_timesteps=[]) is None:
            raise RuntimeError("Foreign-session terminal did not reopen the inflight window")
        entrypoint_client._clear_pending_request()

        entrypoint_server._predict_action_chunk = successful_predict_action_chunk  # type: ignore[method-assign]
        _set_pure_mock_client_running(entrypoint_client, running=True)
        successful_observation = entrypoint_client.control_loop_observation(task="pure-mock-success")
        if successful_observation is None:
            raise RuntimeError("Pure mock client did not recover after terminal failure")
        with entrypoint_client._request_lock:
            success_request = entrypoint_client._pending_request
        if success_request is None:
            raise RuntimeError("Pure mock client did not keep the recovered request inflight before ack")

        success_receiver = threading.Thread(target=entrypoint_client.receive_actions, daemon=True)
        _set_pure_mock_client_running(entrypoint_client, running=True)
        success_receiver.start()
        success_receiver.join(timeout=2.0)
        if success_receiver.is_alive():
            raise RuntimeError("Pure mock success receiver thread did not terminate")
        with entrypoint_client.action_queue_lock:
            recovered_action_timesteps = [action.get_timestep() for action in entrypoint_client.action_queue.queue]
        expected_recovered_timesteps = [
            success_request.observation_timestep + 1,
            success_request.observation_timestep + 2,
        ]
        if recovered_action_timesteps[:2] != expected_recovered_timesteps:
            raise RuntimeError(
                "Pure mock recovery path did not enqueue the expected action chunk after terminal cleanup: "
                f"{recovered_action_timesteps}"
            )
    finally:
        globals()["grpc"] = original_client_grpc
        globals()["services_pb2"] = original_client_services_pb2
        policy_server_module.services_pb2 = original_server_services_pb2
        policy_server_module.receive_bytes_in_chunks = original_receive_bytes_in_chunks

    print(
        "Pure protocol mock passed: "
        f"request_id={decoded_response.request_id} observation_timestep={decoded_response.observation_timestep} "
        f"actions={len(decoded_response.timed_actions)} "
        "negative_paths=missing_observation_request_id,missing_observation_session_id,"
        "missing_action_ack,missing_action_session_id,legacy_action_payload,claim_conflict,"
        "stale_release_after_takeover,stale_peer_release_after_recovery,stale_observation_terminal,startup_release "
        "slow_response_inflight=window1_blocks_overwrite_and_refills_queue "
        "failure_terminals=server_exception_retry_release,send_exception_retry_release,"
        "receive_exception_retry_release,foreign_session_chunk_abort_release "
        "real_entrypoints=start,control_loop_observation,send_observation,receive_actions "
        "restart_recover_takeover=recover_same_session,takeover_success,recover_taken_over_session,"
        "release_and_reclaim"
    )


def build_so_follower_client_config(
    *,
    robot_id: str,
    robot_type: str,
    calib_dir: str | None,
    robot_port: str,
    top_cam_index: int,
    wrist_cam_index: int,
    img_width: int,
    img_height: int,
    fps: int,
    task: str,
    server_address: str,
    policy_path: str,
    actions_per_chunk: int,
    policy_device: str,
    client_device: str,
    chunk_size_threshold: float,
    aggregate_fn_name: str,
    debug_visualize_queue_size: bool,
    backend: str,
    session_id: str | None,
    session_mode: str,
    engine_dir: str,
    tensorrt_py_dir: str | None,
    vit_dtype: str,
    llm_dtype: str,
    dit_dtype: str,
    num_denoising_steps: int | None,
    resource_profile: str = DEFAULT_SERVER_RESOURCE_PROFILE,
) -> GrootTrtRobotClientConfig:
    cameras = {
        "top": OpenCVCameraConfig(
            index_or_path=top_cam_index,
            width=img_width,
            height=img_height,
            fps=fps,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=wrist_cam_index,
            width=img_width,
            height=img_height,
            fps=fps,
        ),
    }

    resolved_robot_type = validate_so_follower_robot_type(robot_type)
    resolved_calib_dir = calib_dir if calib_dir else default_calibration_dir(resolved_robot_type)

    robot_cfg = ExplicitSOFollowerRobotConfig(
        id=robot_id,
        calibration_dir=maybe_path(resolved_calib_dir),
        port=robot_port,
        cameras=cameras,
        robot_type=resolved_robot_type,
    )

    return GrootTrtRobotClientConfig(
        policy_type="groot",
        pretrained_name_or_path=policy_path,
        robot=robot_cfg,
        actions_per_chunk=actions_per_chunk,
        task=task,
        server_address=server_address,
        policy_device=policy_device,
        client_device=client_device,
        chunk_size_threshold=chunk_size_threshold,
        fps=fps,
        aggregate_fn_name=aggregate_fn_name,
        debug_visualize_queue_size=debug_visualize_queue_size,
        resource_profile=resource_profile,
        backend=backend,
        session_id=session_id,
        session_mode=session_mode,
        engine_dir=engine_dir,
        tensorrt_py_dir=tensorrt_py_dir,
        vit_dtype=vit_dtype,
        llm_dtype=llm_dtype,
        dit_dtype=dit_dtype,
        num_denoising_steps=num_denoising_steps,
    )


def run_async_robot_client(
    cfg: GrootTrtRobotClientConfig,
    *,
    mock_actions: bool = False,
    run_time_s: float = 0.0,
    verbose: bool = False,
) -> None:
    if cfg.robot.type not in SUPPORTED_ROBOTS:
        raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client_cls = MockGrootTrtRobotClient if mock_actions else GrootTrtRobotClient
    client = client_cls(cfg)
    if mock_actions:
        client.logger.warning("mock_actions=True: actions will be received and queued, but not sent to the robot.")

    timer_thread = None
    if run_time_s > 0:
        def _stop_after_timeout() -> None:
            time.sleep(run_time_s)
            if client.running:
                client.logger.info("Reached requested run_time_s=%.2fs, stopping client.", run_time_s)
                client.stop()

        timer_thread = threading.Thread(target=_stop_after_timeout, daemon=True)

    if client.start():
        client.logger.info("Starting action receiver thread...")
        action_receiver_thread = threading.Thread(target=client.receive_actions, kwargs={"verbose": verbose}, daemon=True)
        action_receiver_thread.start()
        if timer_thread is not None:
            timer_thread.start()

        try:
            client.control_loop(task=cfg.task, verbose=verbose)
        except KeyboardInterrupt:
            client.logger.info("KeyboardInterrupt received, stopping client.")
        finally:
            client.stop()
            action_receiver_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")
