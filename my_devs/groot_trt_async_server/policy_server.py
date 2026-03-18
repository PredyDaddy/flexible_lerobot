from __future__ import annotations

import logging
import pickle  # nosec
import threading
import time
from concurrent import futures
from dataclasses import asdict, dataclass
from pprint import pformat
from queue import Empty
from types import SimpleNamespace

import draccus

from lerobot import policies  # noqa: F401  # Ensure policy config classes are registered.
from lerobot.async_inference.constants import SUPPORTED_POLICIES
from lerobot.async_inference.helpers import FPSTracker, TimedObservation, get_logger
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from my_devs.groot_trt_async_server.configs import GrootTrtPolicyServerConfig, GrootTrtRemotePolicyConfig

_POLICY_SERVER_IMPORT_ERROR: Exception | None = None
try:
    import grpc

    from lerobot.async_inference.policy_server import PolicyServer as BasePolicyServer
    from lerobot.transport import services_pb2, services_pb2_grpc  # type: ignore
    from lerobot.transport.utils import receive_bytes_in_chunks
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional async deps.
    _POLICY_SERVER_IMPORT_ERROR = exc

    class _CompatStatusCode:
        INVALID_ARGUMENT = "INVALID_ARGUMENT"
        FAILED_PRECONDITION = "FAILED_PRECONDITION"
        RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"

    class _CompatGrpcModule:
        StatusCode = _CompatStatusCode

        class ServicerContext:  # pragma: no cover - typing shim only.
            pass

        @staticmethod
        def server(*args, **kwargs):
            raise RuntimeError(
                "gRPC runtime is unavailable. Install the repository with async extras to run the policy server."
            ) from _POLICY_SERVER_IMPORT_ERROR

    class _CompatEmpty:
        def __init__(self, data: bytes = b"") -> None:
            self.data = data

    class _CompatActions:
        def __init__(self, data: bytes = b"") -> None:
            self.data = data

    class _CompatServicesGrpc:
        @staticmethod
        def add_AsyncInferenceServicer_to_server(*args, **kwargs):
            raise RuntimeError(
                "gRPC runtime is unavailable. Install the repository with async extras to run the policy server."
            ) from _POLICY_SERVER_IMPORT_ERROR

    class _CompatPolicyServer:
        prefix = "policy_server"
        logger = get_logger(prefix)

        def __init__(self, config):
            from queue import Queue

            self.config = config
            self.shutdown_event = threading.Event()
            self.fps_tracker = FPSTracker(target_fps=config.fps)
            self.observation_queue = Queue(maxsize=1)
            self._predicted_timesteps_lock = threading.Lock()
            self._predicted_timesteps = set()
            self.last_processed_obs = None
            self.device = None
            self.policy_type = None
            self.lerobot_features = None
            self.actions_per_chunk = None
            self.policy = None
            self.preprocessor = None
            self.postprocessor = None

        @property
        def running(self):
            return not self.shutdown_event.is_set()

        def _reset_server(self) -> None:
            from queue import Queue

            self.shutdown_event.set()
            self.observation_queue = Queue(maxsize=1)
            with self._predicted_timesteps_lock:
                self._predicted_timesteps = set()

    grpc = _CompatGrpcModule()  # type: ignore[assignment]
    BasePolicyServer = _CompatPolicyServer  # type: ignore[assignment]
    services_pb2 = SimpleNamespace(Empty=_CompatEmpty, Actions=_CompatActions)  # type: ignore[assignment]
    services_pb2_grpc = _CompatServicesGrpc()  # type: ignore[assignment]

    def receive_bytes_in_chunks(*args, **kwargs):
        raise RuntimeError(
            "gRPC runtime is unavailable. Install the repository with async extras to receive observations."
        ) from _POLICY_SERVER_IMPORT_ERROR

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
ALLOWED_SESSION_MODES = {
    SESSION_MODE_CLAIM,
    SESSION_MODE_RELEASE,
    SESSION_MODE_TAKEOVER,
}
ALLOWED_REQUEST_STATES = {
    REQUEST_STATE_ACK,
    REQUEST_STATE_RETRY,
    REQUEST_STATE_ABORT,
}


@dataclass(frozen=True)
class StickySessionBinding:
    session_id: str
    peer: str
    mode: str


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


def _normalize_request_state(value: object, *, field_name: str) -> str:
    normalized = _normalize_required_str(value, field_name=field_name).lower()
    if normalized not in ALLOWED_REQUEST_STATES:
        raise ValueError(f"{field_name} must be one of {sorted(ALLOWED_REQUEST_STATES)}, got {normalized!r}")
    return normalized


class GrootTrtPolicyServer(BasePolicyServer):
    """Async policy server that can swap GR00T into a TensorRT backend."""

    prefix = "groot_trt_policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: GrootTrtPolicyServerConfig):
        super().__init__(config)
        self.config = config
        self.backend = "pytorch"
        self._session_lock = threading.RLock()
        self._active_client_id: str | None = None
        self._active_session_id: str | None = None

    def Ready(self, request, context):  # noqa: N802
        session = self._read_session_binding(context, rpc_name="Ready")
        with self._session_lock:
            if session.mode == SESSION_MODE_RELEASE:
                self._release_session_locked(session, context)
                return services_pb2.Empty()

            self._claim_or_refresh_session_locked(session, context)
            self.logger.info(
                "Client peer=%s session_id=%s connected and ready (mode=%s)",
                session.peer,
                session.session_id,
                session.mode,
            )
            self._reset_server()
            self.shutdown_event.clear()

        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy instructions from the robot client."""
        session = self._require_session_client(context, rpc_name="SendPolicyInstructions")
        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        raw_specs = pickle.loads(request.data)  # nosec
        policy_specs = GrootTrtRemotePolicyConfig.from_payload(raw_specs)
        resolved_policy_specs = self.config.resolve_policy_specs(policy_specs)

        if resolved_policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {resolved_policy_specs.policy_type!r} not supported. Supported policies: {SUPPORTED_POLICIES}"
            )

        self.logger.info(
            f"Receiving policy instructions from peer={session.peer} session_id={session.session_id} | "
            f"Policy type: {resolved_policy_specs.policy_type} | "
            f"Backend: {resolved_policy_specs.backend} | "
            f"Resource profile: {resolved_policy_specs.resource_profile} | "
            f"Resolved pretrained path: {resolved_policy_specs.pretrained_name_or_path} | "
            f"Resolved engine dir: {resolved_policy_specs.engine_dir} | "
            f"Actions per chunk: {resolved_policy_specs.actions_per_chunk} | "
            f"Device: {resolved_policy_specs.device}"
        )

        with self._session_lock:
            self.device = resolved_policy_specs.device
            self.policy_type = resolved_policy_specs.policy_type
            self.backend = resolved_policy_specs.backend
            self.lerobot_features = resolved_policy_specs.lerobot_features
            self.actions_per_chunk = resolved_policy_specs.actions_per_chunk

            start = time.perf_counter()
            self.policy = self._build_policy(resolved_policy_specs)
            self.preprocessor, self.postprocessor = self._build_processors(resolved_policy_specs)
            if hasattr(self.policy, "reset"):
                self.policy.reset()
            end = time.perf_counter()

        self.logger.info(
            f"Prepared backend={self.backend} for policy_type={self.policy_type} on {self.device} "
            f"in {end - start:.4f}s"
        )
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        session = self._require_session_client(context, rpc_name="SendObservations")
        self.logger.debug("Receiving observations from peer=%s session_id=%s", session.peer, session.session_id)

        receive_time = time.time()
        start_deserialize = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(request_iterator, None, self.shutdown_event, self.logger)
        payload = pickle.loads(received_bytes)  # nosec
        timed_observation, request_id, payload_session_id = self._decode_observation_payload(payload)
        deserialize_time = time.perf_counter() - start_deserialize

        if payload_session_id != session.session_id:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Observation payload mismatch for request_id={request_id} session_id={payload_session_id!r}: "
                f"sticky session_id={session.session_id!r}.",
            )

        setattr(timed_observation, REQUEST_ID_PAYLOAD_KEY, request_id)
        setattr(timed_observation, SESSION_ID_PAYLOAD_KEY, payload_session_id)

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        self.logger.debug(
            "Received observation #%s request_id=%s | Avg FPS: %.2f | Target: %.2f | One-way latency: %.2fms",
            obs_timestep,
            request_id,
            fps_metrics["avg_fps"],
            fps_metrics["target_fps"],
            (receive_time - obs_timestamp) * 1000,
        )
        self.logger.debug(
            "Server timestamp: %.6f | Client timestamp: %.6f | Deserialization time: %.6fs",
            receive_time,
            obs_timestamp,
            deserialize_time,
        )

        if not self._session_binding_is_current(session):
            context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                "Stale observation rejected for "
                f"request_id={request_id} session_id={session.session_id!r} peer={session.peer!r} "
                f"obs_step={obs_timestep}.",
            )

        if not self._enqueue_observation(timed_observation):
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Observation was filtered before inference for "
                f"request_id={request_id} session_id={session.session_id!r} obs_step={obs_timestep}.",
            )

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        session = self._require_session_client(context, rpc_name="GetActions")
        self.logger.debug("Client peer=%s session_id=%s connected for action streaming", session.peer, session.session_id)

        try:
            getactions_starts = time.perf_counter()
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            request_id = getattr(obs, REQUEST_ID_PAYLOAD_KEY, None)
            observation_session_id = getattr(obs, SESSION_ID_PAYLOAD_KEY, session.session_id)
            if observation_session_id != session.session_id:
                self.logger.warning(
                    "Returning abort terminal for observation #%s request_id=%s because payload session_id=%s "
                    "does not match action stream session_id=%s",
                    obs.get_timestep(),
                    request_id,
                    observation_session_id,
                    session.session_id,
                )
                return self._terminal_actions_response(
                    request_id=request_id,
                    observation_timestep=obs.get_timestep(),
                    session_id=observation_session_id,
                    request_state=REQUEST_STATE_ABORT,
                    request_state_reason="stale_observation_session",
                )
            if not self._session_binding_is_current(session):
                self.logger.warning(
                    "Returning abort terminal for request_id=%s because peer=%s session_id=%s is no longer active",
                    request_id,
                    session.peer,
                    session.session_id,
                )
                return self._terminal_actions_response(
                    request_id=request_id,
                    observation_timestep=obs.get_timestep(),
                    session_id=session.session_id,
                    request_state=REQUEST_STATE_ABORT,
                    request_state_reason="stale_peer_before_inference",
                )
            self.logger.info(
                "Running inference for observation #%s request_id=%s (must_go: %s)",
                obs.get_timestep(),
                request_id,
                obs.must_go,
            )

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            start_time = time.perf_counter()
            action_chunk = self._predict_action_chunk(obs)
            inference_time = time.perf_counter() - start_time

            if not self._session_binding_is_current(session):
                self.logger.warning(
                    "Returning abort terminal for request_id=%s because peer=%s session_id=%s changed during inference",
                    request_id,
                    session.peer,
                    session.session_id,
                )
                return self._terminal_actions_response(
                    request_id=request_id,
                    observation_timestep=obs.get_timestep(),
                    session_id=session.session_id,
                    request_state=REQUEST_STATE_ABORT,
                    request_state_reason="session_changed_during_inference",
                )

            action_payload = self._encode_action_payload(
                action_chunk,
                request_id=request_id,
                observation_timestep=obs.get_timestep(),
                session_id=session.session_id,
                request_state=REQUEST_STATE_ACK,
            )
            start_time = time.perf_counter()
            actions_bytes = pickle.dumps(action_payload)  # nosec
            serialize_time = time.perf_counter() - start_time

            actions = services_pb2.Actions(data=actions_bytes)
            self.logger.info(
                "Action chunk #%s generated for request_id=%s | Total time: %.2fms",
                obs.get_timestep(),
                request_id,
                (inference_time + serialize_time) * 1000,
            )
            self.logger.debug(
                "Action chunk #%s generated | Inference time: %.2fs | Serialize time: %.2fs | Total time: %.2fs",
                obs.get_timestep(),
                inference_time,
                serialize_time,
                inference_time + serialize_time,
            )

            time.sleep(max(0, self.config.inference_latency - max(0, time.perf_counter() - getactions_starts)))
            return actions

        except Empty:
            return self._empty_actions_response()
        except Exception as e:
            self.logger.error(f"Error in StreamActions: {e}")
            if "obs" in locals():
                request_id = getattr(obs, REQUEST_ID_PAYLOAD_KEY, None)
                observation_timestep = obs.get_timestep()
                failure_session_id = getattr(obs, SESSION_ID_PAYLOAD_KEY, session.session_id)
                if request_id is not None:
                    return self._terminal_actions_response(
                        request_id=request_id,
                        observation_timestep=observation_timestep,
                        session_id=failure_session_id,
                        request_state=REQUEST_STATE_RETRY,
                        request_state_reason=f"server_error:{type(e).__name__}",
                    )
            return self._empty_actions_response()

    def _build_policy(self, policy_specs: GrootTrtRemotePolicyConfig):
        if policy_specs.backend == "tensorrt":
            if policy_specs.policy_type != "groot":
                raise ValueError("TensorRT backend is currently only supported for GR00T.")

            backend_cls = self._resolve_groot_trt_backend_class()
            return backend_cls(
                pretrained_name_or_path=policy_specs.pretrained_name_or_path,
                device=policy_specs.device,
                engine_dir=str(policy_specs.engine_path),
                tensorrt_py_dir=policy_specs.tensorrt_py_dir,
                vit_dtype=policy_specs.vit_dtype,
                llm_dtype=policy_specs.llm_dtype,
                dit_dtype=policy_specs.dit_dtype,
                num_denoising_steps=policy_specs.num_denoising_steps,
            )

        policy_class = get_policy_class(policy_specs.policy_type)
        policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        policy.to(self.device)
        return policy

    def _build_processors(self, policy_specs: GrootTrtRemotePolicyConfig):
        policy_config = getattr(self.policy, "config", None)
        if policy_config is None:
            raise AttributeError("Loaded policy backend must expose a `.config` attribute.")

        device_override = {"device": self.device}
        return make_pre_post_processors(
            policy_config,
            pretrained_path=policy_specs.pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": policy_specs.rename_map},
            },
            postprocessor_overrides={"device_processor": device_override},
        )

    def _require_session_client(
        self,
        context: grpc.ServicerContext,
        *,
        rpc_name: str,
    ) -> StickySessionBinding:
        session = self._read_session_binding(context, rpc_name=rpc_name)
        with self._session_lock:
            if self._active_session_id is None:
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    f"{rpc_name} requires an active session. Call Ready() before invoking {rpc_name}.",
                )

            if self._active_session_id != session.session_id:
                context.abort(
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                    "This server only supports one active sticky session at a time. "
                    f"Active session_id={self._active_session_id!r} peer={self._active_client_id!r}. "
                    f"Rejecting session_id={session.session_id!r} peer={session.peer!r} for {rpc_name}.",
                )

            if self._active_client_id != session.peer:
                self.logger.warning(
                    "Accepting reconnected sticky session session_id=%s from peer=%s (previous peer=%s) for %s",
                    session.session_id,
                    session.peer,
                    self._active_client_id,
                    rpc_name,
                )
                self._active_client_id = session.peer

        return session

    @staticmethod
    def _normalize_required_int(value: object, *, field_name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field_name} must be an int, got {type(value)}")
        return value

    @classmethod
    def _decode_observation_payload(cls, payload: object) -> tuple[TimedObservation, int, str]:
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
                cls._normalize_required_int(request_id, field_name=REQUEST_ID_PAYLOAD_KEY),
                _normalize_required_str(session_id, field_name=SESSION_ID_PAYLOAD_KEY),
            )

        raise TypeError(f"Unsupported observation payload type: {type(payload)}")

    @classmethod
    def _encode_action_payload(
        cls,
        action_chunk: list[object],
        *,
        request_id: int,
        observation_timestep: int,
        session_id: str,
        request_state: str = REQUEST_STATE_ACK,
        request_state_reason: str | None = None,
    ) -> dict[str, object]:
        normalized_request_id = cls._normalize_required_int(request_id, field_name=ACKED_REQUEST_ID_PAYLOAD_KEY)
        normalized_observation_timestep = cls._normalize_required_int(
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
            ACTION_PAYLOAD_KEY: action_chunk,
            REQUEST_ID_PAYLOAD_KEY: normalized_request_id,
            ACKED_REQUEST_ID_PAYLOAD_KEY: normalized_request_id,
            OBSERVATION_TIMESTEP_PAYLOAD_KEY: normalized_observation_timestep,
            SESSION_ID_PAYLOAD_KEY: normalized_session_id,
            REQUEST_STATE_PAYLOAD_KEY: normalized_request_state,
        }
        if normalized_request_state_reason is not None:
            payload[REQUEST_STATE_REASON_PAYLOAD_KEY] = normalized_request_state_reason
        return payload

    @staticmethod
    def _empty_actions_response():
        return services_pb2.Actions(data=b"")

    @classmethod
    def _terminal_actions_response(
        cls,
        *,
        request_id: int,
        observation_timestep: int,
        session_id: str,
        request_state: str,
        request_state_reason: str,
    ):
        payload = cls._encode_action_payload(
            [],
            request_id=request_id,
            observation_timestep=observation_timestep,
            session_id=session_id,
            request_state=request_state,
            request_state_reason=request_state_reason,
        )
        return services_pb2.Actions(data=pickle.dumps(payload))  # nosec

    def _session_binding_is_current(self, session: StickySessionBinding) -> bool:
        with self._session_lock:
            return self._session_binding_is_current_locked(session)

    def _session_binding_is_current_locked(self, session: StickySessionBinding) -> bool:
        return self._active_session_id == session.session_id and self._active_client_id == session.peer

    @staticmethod
    def _metadata_items(context: grpc.ServicerContext) -> dict[str, str]:
        metadata: dict[str, str] = {}
        for item in context.invocation_metadata():
            if isinstance(item, tuple) and len(item) == 2:
                key, value = item
            else:
                key = getattr(item, "key", None)
                value = getattr(item, "value", None)
            if key is None:
                continue
            metadata[str(key).lower()] = "" if value is None else str(value)
        return metadata

    @classmethod
    def _read_session_binding(
        cls,
        context: grpc.ServicerContext,
        *,
        rpc_name: str,
    ) -> StickySessionBinding:
        metadata = cls._metadata_items(context)
        session_id = metadata.get(SESSION_ID_METADATA_KEY)
        if not session_id:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"{rpc_name} requires metadata {SESSION_ID_METADATA_KEY!r}.",
            )
        session_id = _normalize_required_str(session_id, field_name=SESSION_ID_METADATA_KEY)

        session_mode = metadata.get(SESSION_MODE_METADATA_KEY, SESSION_MODE_CLAIM).strip().lower()
        if session_mode not in ALLOWED_SESSION_MODES:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"{rpc_name} received unsupported {SESSION_MODE_METADATA_KEY}={session_mode!r}. "
                f"Expected one of {sorted(ALLOWED_SESSION_MODES)}.",
            )

        return StickySessionBinding(
            session_id=session_id,
            peer=context.peer(),
            mode=session_mode,
        )

    def _claim_or_refresh_session_locked(
        self,
        session: StickySessionBinding,
        context: grpc.ServicerContext,
    ) -> None:
        if self._active_session_id is None:
            self._active_session_id = session.session_id
            self._active_client_id = session.peer
            self.logger.info("Claimed sticky session_id=%s peer=%s", session.session_id, session.peer)
            return

        if self._active_session_id == session.session_id:
            if self._active_client_id != session.peer:
                self.logger.warning(
                    "Recovered sticky session_id=%s onto new peer=%s (previous peer=%s)",
                    session.session_id,
                    session.peer,
                    self._active_client_id,
                )
            self._active_client_id = session.peer
            return

        if session.mode == SESSION_MODE_TAKEOVER:
            self.logger.warning(
                "Taking over sticky session from session_id=%s peer=%s to session_id=%s peer=%s",
                self._active_session_id,
                self._active_client_id,
                session.session_id,
                session.peer,
            )
            self._active_session_id = session.session_id
            self._active_client_id = session.peer
            return

        context.abort(
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            "This server only supports one active sticky session at a time. "
            f"Active session_id={self._active_session_id!r} peer={self._active_client_id!r}. "
            f"Rejecting session_id={session.session_id!r} peer={session.peer!r}.",
        )

    def _release_session_locked(
        self,
        session: StickySessionBinding,
        context: grpc.ServicerContext,
    ) -> None:
        if self._active_session_id is None:
            self.logger.info("Ignoring release for inactive session_id=%s peer=%s", session.session_id, session.peer)
            return

        if self._active_session_id != session.session_id:
            context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                "Only the active sticky session can be released. "
                f"Active session_id={self._active_session_id!r}, got {session.session_id!r}.",
            )
        if self._active_client_id != session.peer:
            context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                "Only the active sticky peer can release the current session. "
                f"Active peer={self._active_client_id!r}, got {session.peer!r}.",
            )

        self.logger.info("Released sticky session_id=%s peer=%s", session.session_id, session.peer)
        self._active_session_id = None
        self._active_client_id = None
        self._reset_server()

    @staticmethod
    def _resolve_groot_trt_backend_class():
        try:
            from my_devs.groot_trt_async_server.groot_trt_policy import GrootTrtPolicyBackend
        except Exception as exc:  # pragma: no cover - integration path depends on worker A.
            raise ImportError(
                "Failed to import `GrootTrtPolicyBackend` from "
                "`my_devs.groot_trt_async_server.groot_trt_policy`.\n"
                "Expected contract:\n"
                "  GrootTrtPolicyBackend(\n"
                "    pretrained_name_or_path: str,\n"
                "    device: str,\n"
                "    engine_dir: str,\n"
                "    tensorrt_py_dir: str | None,\n"
                "    vit_dtype: str,\n"
                "    llm_dtype: str,\n"
                "    dit_dtype: str,\n"
                "    num_denoising_steps: int | None,\n"
                "  )\n"
                "and expose `.config`, `predict_action_chunk(batch)`, and `reset()`."
            ) from exc

        return GrootTrtPolicyBackend


@draccus.wrap()
def serve(cfg: GrootTrtPolicyServerConfig):
    if _POLICY_SERVER_IMPORT_ERROR is not None:
        raise RuntimeError(
            "gRPC runtime is unavailable. Install the repository with async extras to run the policy server."
        ) from _POLICY_SERVER_IMPORT_ERROR

    logging.info(pformat(asdict(cfg)))

    policy_server = GrootTrtPolicyServer(cfg)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"GrootTrtPolicyServer started on {cfg.host}:{cfg.port}")
    server.start()
    server.wait_for_termination()
    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
