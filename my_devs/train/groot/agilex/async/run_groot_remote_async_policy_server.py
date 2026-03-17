#!/usr/bin/env python

"""AgileX remote bounded-async GR00T policy server."""

from __future__ import annotations

import argparse
import os
import pickle  # nosec
import socket
import sys
import time
import traceback
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
for candidate in (SCRIPT_DIR, PARENT_DIR):
    if candidate.as_posix() not in sys.path:
        sys.path.insert(0, candidate.as_posix())

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.policies.utils import prepare_observation_for_inference

from remote_sync_common import (
    AUTO_ARM,
    DEFAULT_MAX_MESSAGE_BYTES,
    DEFAULT_SERVER_PORT,
    DEFAULT_SOCKET_TIMEOUT_S,
    HEADER_STRUCT,
    AgileXPolicyContract,
    RemoteGrootPolicyConfig,
    RemoteMessage,
    configure_logging,
    ensure_groot_checkpoint_assets,
    env_bool,
    load_pre_post_processors,
    parse_bool,
    receive_message,
    register_plugins_once,
    resolve_agilex_policy_contract,
    resolve_policy_device,
    resolve_remote_policy_config,
    summarize_policy_contract,
    sync_processor_device,
)
from remote_async_common import (
    AsyncCloseAck,
    AsyncError,
    AsyncInferResponse,
    AsyncReady,
    AsyncResetAck,
)


def _payload_value(payload: Any, key: str, default: Any | None = None) -> Any:
    if isinstance(payload, dict):
        return payload.get(key, default)
    return getattr(payload, key, default)


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, ratio)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _serialize_message(message: Any) -> tuple[bytes, float]:
    started = time.perf_counter()
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)  # nosec
    return payload, (time.perf_counter() - started) * 1000.0


def _send_serialized_message(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(HEADER_STRUCT.pack(len(payload)))
    sock.sendall(payload)


def _coerce_policy_setup(payload: Any) -> RemoteGrootPolicyConfig:
    if payload is None:
        return RemoteGrootPolicyConfig()
    if isinstance(payload, RemoteGrootPolicyConfig):
        return payload
    if isinstance(payload, dict):
        return RemoteGrootPolicyConfig(**payload)
    raise TypeError(f"Unsupported setup payload type: {type(payload)}")


def _parse_async_hello(payload: Any) -> tuple[str, RemoteGrootPolicyConfig]:
    session_id = _payload_value(payload, "session_id")
    setup_payload = _payload_value(payload, "setup", None)
    if setup_payload is None:
        setup_payload = _payload_value(payload, "policy_setup", None)
    if not session_id:
        raise ValueError("hello.session_id is required")
    return str(session_id), _coerce_policy_setup(setup_payload)


def _parse_async_infer_request(payload: Any) -> dict[str, Any]:
    request_id = _payload_value(payload, "request_id")
    session_id = _payload_value(payload, "session_id")
    capture_ts = float(_payload_value(payload, "capture_ts", 0.0))
    observation = _payload_value(payload, "observation")
    if request_id is None:
        raise ValueError("infer_request.request_id is required")
    if not session_id:
        raise ValueError("infer_request.session_id is required")
    if observation is None:
        raise ValueError("infer_request.observation is required")
    return {
        "request_id": int(request_id),
        "session_id": str(session_id),
        "capture_ts": capture_ts,
        "must_go": bool(_payload_value(payload, "must_go", False)),
        "queue_size_at_capture": int(_payload_value(payload, "queue_size_at_capture", 0)),
        "latest_action_id_at_capture": int(_payload_value(payload, "latest_action_id_at_capture", -1)),
        "observation": observation,
    }


def _parse_async_reset(payload: Any) -> dict[str, Any]:
    return {
        "session_id": str(_payload_value(payload, "session_id", "") or ""),
        "reason": str(_payload_value(payload, "reason", "") or ""),
        "new_session_id": _payload_value(payload, "new_session_id"),
    }


def _parse_async_close(payload: Any) -> dict[str, Any]:
    return {
        "session_id": str(_payload_value(payload, "session_id", "") or ""),
        "reason": str(_payload_value(payload, "reason", "") or ""),
    }


@dataclass
class AsyncPolicyServerConfig:
    host: str
    port: int
    policy_path: str | None
    task: str
    robot_type: str
    backend: str
    trt_engine_path: str | None
    vit_dtype: str
    llm_dtype: str
    dit_dtype: str
    trt_action_head_only: bool
    policy_device: str | None
    control_arm: str | None
    socket_timeout_s: float
    max_message_bytes: int
    max_returned_actions: int
    metrics_log_interval: int
    log_level: str
    dry_run: bool = False


class ServerMetricWindow:
    def __init__(self, logger, log_interval: int) -> None:
        self.logger = logger
        self.log_interval = log_interval
        self.total_requests = 0
        self.status_counts = {"ack": 0, "retry": 0, "abort": 0}
        self.total_ms: list[float] = []
        self.preprocess_ms: list[float] = []
        self.infer_ms: list[float] = []
        self.postprocess_ms: list[float] = []
        self.serialize_ms: list[float] = []
        self.queue_wait_ms: list[float] = []
        self.chunk_size: list[float] = []

    def record(self, response_payload: dict[str, Any]) -> None:
        status = str(response_payload.get("status", "abort"))
        self.total_requests += 1
        if status in self.status_counts:
            self.status_counts[status] += 1
        if status == "ack":
            self.total_ms.append(float(response_payload.get("total_ms", 0.0)))
            self.preprocess_ms.append(float(response_payload.get("preprocess_ms", 0.0)))
            self.infer_ms.append(float(response_payload.get("infer_ms", 0.0)))
            self.postprocess_ms.append(float(response_payload.get("postprocess_ms", 0.0)))
            self.serialize_ms.append(float(response_payload.get("serialize_ms", 0.0)))
            self.queue_wait_ms.append(float(response_payload.get("queue_wait_ms", 0.0)))
            self.chunk_size.append(float(response_payload.get("chunk_size", 0)))
        if self.log_interval > 0 and self.total_requests % self.log_interval == 0:
            self.logger.info(
                "Async server metrics | requests=%s | ack=%s retry=%s abort=%s | total_ms[p50=%.2f p95=%.2f p99=%.2f] | infer_ms[p95=%.2f] | queue_wait_ms[p95=%.2f] | chunk_size[p50=%.2f p95=%.2f]",
                self.total_requests,
                self.status_counts["ack"],
                self.status_counts["retry"],
                self.status_counts["abort"],
                _percentile(self.total_ms, 0.50),
                _percentile(self.total_ms, 0.95),
                _percentile(self.total_ms, 0.99),
                _percentile(self.infer_ms, 0.95),
                _percentile(self.queue_wait_ms, 0.95),
                _percentile(self.chunk_size, 0.50),
                _percentile(self.chunk_size, 0.95),
            )


class AsyncGrootPolicyRuntime:
    def __init__(self, config: AsyncPolicyServerConfig, logger) -> None:
        self.config = config
        self.logger = logger
        self.active_session_id: str | None = None
        self.active_setup: RemoteGrootPolicyConfig | None = None
        self.policy_cfg: PreTrainedConfig | None = None
        self.policy = None
        self.policy_device: torch.device | None = None
        self.preprocessor = None
        self.postprocessor = None
        self.policy_contract: AgileXPolicyContract | None = None
        self.runtime_untrusted = False

    def configure(self, session_id: str, setup: RemoteGrootPolicyConfig) -> Any:
        resolved = resolve_remote_policy_config(
            setup,
            default_policy_path=self.config.policy_path,
            default_task=self.config.task,
            default_robot_type=self.config.robot_type,
            default_backend=self.config.backend,
            default_trt_engine_path=self.config.trt_engine_path,
            default_vit_dtype=self.config.vit_dtype,
            default_llm_dtype=self.config.llm_dtype,
            default_dit_dtype=self.config.dit_dtype,
            default_trt_action_head_only=self.config.trt_action_head_only,
            default_policy_device=self.config.policy_device,
            default_control_arm=self.config.control_arm,
        )

        if self.active_setup is None or self.active_setup != resolved:
            self._load_policy(resolved)
        elif resolved.reset_policy_state:
            self._reset_policy_state()

        self.active_setup = resolved
        self.active_session_id = session_id
        self.runtime_untrusted = False
        assert self.policy_cfg is not None
        self.policy_contract = resolve_agilex_policy_contract(
            self.policy_cfg,
            requested_control_arm=resolved.control_arm,
        )
        self.logger.info(
            "Async session bound | session_id=%s | %s",
            session_id,
            summarize_policy_contract(self.policy_contract),
        )
        return AsyncReady(
            session_id=session_id,
            policy_path=resolved.policy_path or "",
            task=resolved.task or "",
            robot_type=resolved.robot_type or "",
            backend=resolved.backend or "",
            policy_device=str(self.policy_cfg.device),
            policy_contract=self.policy_contract,
            protocol_version="async-v1",
            accepted=True,
            message="ready",
            negotiated_low_watermark=None,
            negotiated_chunk_size=self.config.max_returned_actions or None,
            max_returned_actions=self.config.max_returned_actions,
            server_time=time.time(),
        )

    def _load_policy(self, setup: RemoteGrootPolicyConfig) -> None:
        register_plugins_once()
        assert setup.policy_path is not None

        policy_path = Path(setup.policy_path).expanduser()
        ensure_groot_checkpoint_assets(policy_path)

        policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path))
        policy_cfg.pretrained_path = policy_path
        if policy_cfg.type != "groot":
            raise ValueError(f"Expected a groot checkpoint, got {policy_cfg.type!r}")
        policy_cfg.device = resolve_policy_device(setup.policy_device, policy_cfg.device)

        policy_class = get_policy_class(policy_cfg.type)
        policy = policy_class.from_pretrained(str(policy_path), config=policy_cfg, strict=False)
        policy.config.device = policy_cfg.device
        policy.to(policy_cfg.device)

        if setup.backend == "tensorrt":
            if not setup.trt_engine_path:
                raise ValueError("--trt-engine-path is required when backend=tensorrt")
            from lerobot.policies.groot.trt_runtime.patch import setup_tensorrt_engines

            setup_tensorrt_engines(
                policy._groot_model,
                setup.trt_engine_path,
                vit_dtype=setup.vit_dtype or self.config.vit_dtype,
                llm_dtype=setup.llm_dtype or self.config.llm_dtype,
                dit_dtype=setup.dit_dtype or self.config.dit_dtype,
                action_head_only=bool(setup.trt_action_head_only),
            )

        preprocessor, postprocessor = load_pre_post_processors(policy_path)
        sync_processor_device(preprocessor, policy_cfg.device)
        sync_processor_device(postprocessor, "cpu")

        self.policy_cfg = policy_cfg
        self.policy = policy
        self.policy_device = torch.device(str(policy_cfg.device))
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self._reset_policy_state()

        self.logger.info(
            "Async policy loaded | path=%s | device=%s | backend=%s",
            policy_path,
            policy_cfg.device,
            setup.backend,
        )

    def _reset_policy_state(self) -> None:
        if self.policy is not None:
            self.policy.reset()
        if self.preprocessor is not None:
            self.preprocessor.reset()
        if self.postprocessor is not None:
            self.postprocessor.reset()

    def reset_runtime(self, session_id: str, reason: str, new_session_id: str | None = None) -> Any:
        if self.active_session_id and session_id and session_id != self.active_session_id:
            return AsyncResetAck(
                session_id=self.active_session_id,
                reason=f"foreign_session:{session_id}",
                accepted=False,
                reset_at=time.time(),
                new_session_id=None,
                status="ignored",
            )

        self._reset_policy_state()
        self.runtime_untrusted = False
        if new_session_id:
            self.active_session_id = str(new_session_id)
        return AsyncResetAck(
            session_id=self.active_session_id,
            reason=reason or "reset",
            accepted=True,
            reset_at=time.time(),
            new_session_id=self.active_session_id if new_session_id else None,
            status="ok",
        )

    def close_session(self, session_id: str, reason: str) -> Any:
        if self.active_session_id and session_id and session_id != self.active_session_id:
            return AsyncCloseAck(
                session_id=self.active_session_id,
                reason=f"foreign_session:{session_id}",
                accepted=False,
                closed_at=time.time(),
                status="ignored",
            )

        closed_session_id = self.active_session_id
        self.active_session_id = None
        return AsyncCloseAck(
            session_id=closed_session_id,
            reason=reason or "close",
            accepted=True,
            closed_at=time.time(),
            status="ok",
        )

    def predict_request(
        self,
        request: dict[str, Any],
        *,
        server_received_at: float,
        received_perf: float,
    ) -> dict[str, Any]:
        base_response = {
            "request_id": request["request_id"],
            "session_id": request["session_id"],
            "status": "abort",
            "reason": "unknown",
            "server_received_at": server_received_at,
            "server_sent_at": 0.0,
            "predict_latency_ms": 0.0,
            "queue_wait_ms": 0.0,
            "preprocess_ms": 0.0,
            "infer_ms": 0.0,
            "postprocess_ms": 0.0,
            "serialize_ms": 0.0,
            "total_ms": 0.0,
            "chunk_size": 0,
            "actions": [],
        }

        total_started = time.perf_counter()
        base_response["queue_wait_ms"] = max((total_started - received_perf) * 1000.0, 0.0)

        if self.active_session_id is None or self.active_setup is None:
            base_response["status"] = "retry"
            base_response["reason"] = "runtime_not_ready"
            return base_response

        if request["session_id"] != self.active_session_id:
            base_response["reason"] = f"foreign_session:{request['session_id']}"
            return base_response

        if self.runtime_untrusted:
            base_response["reason"] = "runtime_untrusted"
            return base_response

        if self.policy is None or self.preprocessor is None or self.postprocessor is None or self.policy_device is None:
            base_response["status"] = "retry"
            base_response["reason"] = "runtime_not_ready"
            return base_response

        try:
            observation = deepcopy(request["observation"])
            use_amp = bool(getattr(self.policy.config, "use_amp", False))

            with (
                torch.inference_mode(),
                torch.autocast(device_type=self.policy_device.type)
                if self.policy_device.type == "cuda" and use_amp
                else nullcontext(),
            ):
                preprocess_started = time.perf_counter()
                observation = prepare_observation_for_inference(
                    observation,
                    self.policy_device,
                    task=self.active_setup.task,
                    robot_type=self.active_setup.robot_type,
                )
                observation = self.preprocessor(observation)
                preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0

                infer_started = time.perf_counter()
                action_chunk = self.policy.predict_action_chunk(observation)
                if action_chunk.ndim != 3:
                    action_chunk = action_chunk.unsqueeze(0)
                if self.config.max_returned_actions > 0:
                    action_chunk = action_chunk[:, : self.config.max_returned_actions, :]
                infer_ms = (time.perf_counter() - infer_started) * 1000.0

                postprocess_started = time.perf_counter()
                processed_actions: list[Any] = []
                for index in range(action_chunk.shape[1]):
                    single_action = action_chunk[:, index, :]
                    processed_action = self.postprocessor(single_action)
                    if hasattr(processed_action, "detach"):
                        processed_action = processed_action.detach().to("cpu")
                    else:
                        processed_action = torch.as_tensor(processed_action).to("cpu")
                    processed_actions.append(processed_action.squeeze(0))
                postprocess_ms = (time.perf_counter() - postprocess_started) * 1000.0

            base_response.update(
                {
                    "status": "ack",
                    "reason": "ok",
                    "predict_latency_ms": preprocess_ms + infer_ms + postprocess_ms,
                    "preprocess_ms": preprocess_ms,
                    "infer_ms": infer_ms,
                    "postprocess_ms": postprocess_ms,
                    "total_ms": (time.perf_counter() - total_started) * 1000.0,
                    "chunk_size": len(processed_actions),
                    "actions": processed_actions,
                }
            )
            return base_response
        except Exception as exc:
            self.runtime_untrusted = True
            base_response["reason"] = f"runtime_error:{exc}"
            self.logger.exception(
                "Async inference failed | session_id=%s | request_id=%s",
                request["session_id"],
                request["request_id"],
            )
            return base_response


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remote bounded-async GR00T policy server for AgileX.")
    parser.add_argument("--host", default=os.getenv("REMOTE_GROOT_SERVER_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("REMOTE_GROOT_SERVER_PORT", str(DEFAULT_SERVER_PORT))),
    )
    parser.add_argument("--policy-path", default=os.getenv("POLICY_PATH"))
    parser.add_argument("--task", default=os.getenv("DATASET_TASK", "Execute the trained AgileX GR00T task"))
    parser.add_argument("--robot-type", default=os.getenv("ROBOT_TYPE", "agilex"))
    parser.add_argument(
        "--backend",
        default=os.getenv("INFER_BACKEND", "pytorch"),
        choices=["pytorch", "tensorrt"],
    )
    parser.add_argument("--trt-engine-path", default=os.getenv("TRT_ENGINE_PATH"))
    parser.add_argument("--vit-dtype", default=os.getenv("TRT_VIT_DTYPE", "fp16"), choices=["fp16", "fp8"])
    parser.add_argument(
        "--llm-dtype",
        default=os.getenv("TRT_LLM_DTYPE", "fp16"),
        choices=["fp16", "nvfp4", "fp8", "nvfp4_full"],
    )
    parser.add_argument("--dit-dtype", default=os.getenv("TRT_DIT_DTYPE", "fp16"), choices=["fp16", "fp8"])
    parser.add_argument(
        "--trt-action-head-only",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("TRT_ACTION_HEAD_ONLY", False),
    )
    parser.add_argument(
        "--policy-device",
        default=os.getenv("POLICY_DEVICE_OVERRIDE", os.getenv("POLICY_DEVICE")),
        help="Override checkpoint device with one of cpu/cuda/mps/xpu/auto.",
    )
    parser.add_argument(
        "--control-arm",
        default=os.getenv("CONTROL_ARM", AUTO_ARM),
        help="Expected AgileX control scope for this checkpoint: auto/left/right/both.",
    )
    parser.add_argument(
        "--socket-timeout-s",
        type=float,
        default=float(os.getenv("REMOTE_GROOT_SOCKET_TIMEOUT_S", str(DEFAULT_SOCKET_TIMEOUT_S))),
    )
    parser.add_argument(
        "--max-message-bytes",
        type=int,
        default=int(os.getenv("REMOTE_GROOT_MAX_MESSAGE_BYTES", str(DEFAULT_MAX_MESSAGE_BYTES))),
    )
    parser.add_argument(
        "--max-returned-actions",
        type=int,
        default=int(os.getenv("ASYNC_MAX_RETURNED_ACTIONS", "0")),
        help="0 means use the full model chunk.",
    )
    parser.add_argument(
        "--metrics-log-interval",
        type=int,
        default=int(os.getenv("ASYNC_SERVER_METRICS_LOG_INTERVAL", "20")),
    )
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    parser.add_argument(
        "--dry-run",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("DRY_RUN", False),
    )
    return parser


def handle_client(
    conn: socket.socket,
    addr: tuple[str, int],
    runtime: AsyncGrootPolicyRuntime,
    metrics: ServerMetricWindow,
    config: AsyncPolicyServerConfig,
    logger,
) -> None:
    logger.info("Async client connected | address=%s:%s", addr[0], addr[1])
    with conn:
        conn.settimeout(config.socket_timeout_s)
        while True:
            try:
                message = receive_message(conn, max_message_bytes=config.max_message_bytes)
            except socket.timeout:
                logger.warning("Async client timed out | address=%s:%s", addr[0], addr[1])
                break
            except EOFError:
                logger.info("Async client disconnected | address=%s:%s", addr[0], addr[1])
                break

            if not isinstance(message, RemoteMessage):
                payload = AsyncError(
                    message=f"Unexpected message type: {type(message)}",
                    traceback=None,
                    session_id=None,
                    request_id=None,
                    recoverable=False,
                )
                serialized, _ = _serialize_message(RemoteMessage(kind="error", payload=payload))
                _send_serialized_message(conn, serialized)
                continue

            try:
                if message.kind == "hello":
                    session_id, setup = _parse_async_hello(message.payload)
                    ready_payload = runtime.configure(session_id, setup)
                    serialized, _ = _serialize_message(RemoteMessage(kind="ready", payload=ready_payload))
                    _send_serialized_message(conn, serialized)
                elif message.kind == "infer_request":
                    server_received_at = time.time()
                    received_perf = time.perf_counter()
                    request = _parse_async_infer_request(message.payload)
                    response_payload = runtime.predict_request(
                        request,
                        server_received_at=server_received_at,
                        received_perf=received_perf,
                    )
                    preview = RemoteMessage(kind="infer_response", payload=AsyncInferResponse(**response_payload))
                    _, preview_serialize_ms = _serialize_message(preview)
                    response_payload["serialize_ms"] = preview_serialize_ms
                    response_payload["server_sent_at"] = time.time()
                    response_payload["total_ms"] += preview_serialize_ms
                    outbound_payload = AsyncInferResponse(**response_payload)
                    outbound = RemoteMessage(kind="infer_response", payload=outbound_payload)
                    serialized, outbound_serialize_ms = _serialize_message(outbound)
                    response_payload["serialize_ms"] += outbound_serialize_ms
                    response_payload["total_ms"] += outbound_serialize_ms
                    _send_serialized_message(conn, serialized)
                    metrics.record(response_payload)
                    logger.info(
                        "Async response | session_id=%s | request_id=%s | status=%s | reason=%s | total_ms=%.2f | chunk_size=%s | must_go=%s | queue_at_capture=%s",
                        response_payload["session_id"],
                        response_payload["request_id"],
                        response_payload["status"],
                        response_payload["reason"],
                        response_payload["total_ms"],
                        response_payload["chunk_size"],
                        request["must_go"],
                        request["queue_size_at_capture"],
                    )
                elif message.kind == "reset":
                    reset_payload = _parse_async_reset(message.payload)
                    ack_payload = runtime.reset_runtime(
                        session_id=reset_payload["session_id"],
                        reason=reset_payload["reason"],
                        new_session_id=reset_payload["new_session_id"],
                    )
                    serialized, _ = _serialize_message(RemoteMessage(kind="reset_ack", payload=ack_payload))
                    _send_serialized_message(conn, serialized)
                elif message.kind == "close":
                    close_payload = _parse_async_close(message.payload)
                    ack_payload = runtime.close_session(
                        session_id=close_payload["session_id"],
                        reason=close_payload["reason"],
                    )
                    serialized, _ = _serialize_message(RemoteMessage(kind="close_ack", payload=ack_payload))
                    _send_serialized_message(conn, serialized)
                    break
                else:
                    raise ValueError(f"Unsupported async message kind: {message.kind}")
            except Exception as exc:
                logger.exception("Async request failed | address=%s:%s | kind=%s", addr[0], addr[1], message.kind)
                error_payload = AsyncError(
                    message=str(exc),
                    traceback=traceback.format_exc(),
                    session_id=_payload_value(message.payload, "session_id", None),
                    request_id=_payload_value(message.payload, "request_id", None),
                    recoverable=False,
                )
                serialized, _ = _serialize_message(RemoteMessage(kind="error", payload=error_payload))
                _send_serialized_message(conn, serialized)


def serve(config: AsyncPolicyServerConfig) -> None:
    logger = configure_logging("groot_agilex_remote_async_policy_server", config.log_level)
    if config.dry_run:
        print(config)
        return

    runtime = AsyncGrootPolicyRuntime(config, logger)
    metrics = ServerMetricWindow(logger, log_interval=config.metrics_log_interval)

    with socket.create_server((config.host, config.port), backlog=1, reuse_port=False) as server_sock:
        server_sock.settimeout(1.0)
        logger.info(
            "Async policy server listening | host=%s | port=%s | policy_path=%s | max_returned_actions=%s",
            config.host,
            config.port,
            config.policy_path,
            config.max_returned_actions,
        )
        while True:
            try:
                conn, addr = server_sock.accept()
            except socket.timeout:
                continue
            handle_client(conn, addr, runtime, metrics, config, logger)


def main() -> None:
    args = build_parser().parse_args()
    serve(
        AsyncPolicyServerConfig(
            host=args.host,
            port=args.port,
            policy_path=args.policy_path,
            task=args.task,
            robot_type=args.robot_type,
            backend=args.backend,
            trt_engine_path=args.trt_engine_path,
            vit_dtype=args.vit_dtype,
            llm_dtype=args.llm_dtype,
            dit_dtype=args.dit_dtype,
            trt_action_head_only=args.trt_action_head_only,
            policy_device=args.policy_device,
            control_arm=args.control_arm,
            socket_timeout_s=args.socket_timeout_s,
            max_message_bytes=args.max_message_bytes,
            max_returned_actions=args.max_returned_actions,
            metrics_log_interval=args.metrics_log_interval,
            log_level=args.log_level,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
