#!/usr/bin/env python

"""Measure AgileX remote synchronous inference latency as the async baseline."""

from __future__ import annotations

import argparse
import json
import math
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
from typing import Any, Iterable

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
for candidate in (SCRIPT_DIR, PARENT_DIR):
    if candidate.as_posix() not in sys.path:
        sys.path.insert(0, candidate.as_posix())

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.robots import make_robot_from_config

from remote_sync_common import (
    AUTO_ARM,
    CAMERA_KEYS,
    DEFAULT_MAX_MESSAGE_BYTES,
    DEFAULT_RECONNECT_RETRIES,
    DEFAULT_RECONNECT_RETRY_DELAY_S,
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    DEFAULT_SOCKET_TIMEOUT_S,
    HEADER_STRUCT,
    POSITION_FEATURE_NAMES,
    AgileXPolicyContract,
    LoopMetrics,
    ObservationPacket,
    RemoteError,
    RemoteGrootPolicyConfig,
    RemoteMessage,
    build_agilex_policy_observation_frame,
    build_agilex_policy_runtime_helpers,
    build_agilex_robot_config,
    configure_logging,
    ensure_groot_checkpoint_assets,
    env_bool,
    load_pre_post_processors,
    open_client_socket,
    parse_bool,
    receive_message,
    register_plugins_once,
    resolve_agilex_policy_contract,
    resolve_policy_device,
    resolve_remote_policy_config,
    send_message,
    summarize_action_tensor,
    summarize_observation,
    summarize_policy_contract,
    sync_processor_device,
    validate_live_observation,
)


PROBE_ACTION_KIND = "probe_action"


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = max(0.0, min(1.0, ratio)) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summarize_latency(values_ms: list[float]) -> dict[str, float]:
    if not values_ms:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
        }
    return {
        "count": len(values_ms),
        "mean_ms": float(sum(values_ms) / len(values_ms)),
        "min_ms": float(min(values_ms)),
        "max_ms": float(max(values_ms)),
        "p50_ms": float(_percentile(values_ms, 0.50)),
        "p95_ms": float(_percentile(values_ms, 0.95)),
        "p99_ms": float(_percentile(values_ms, 0.99)),
    }


def _serialize_message(message: Any) -> tuple[bytes, float]:
    started = time.perf_counter()
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)  # nosec
    return payload, (time.perf_counter() - started) * 1000.0


def _send_serialized_message(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(HEADER_STRUCT.pack(len(payload)))
    sock.sendall(payload)


def _write_text(path: Path | None, content: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _load_pickled_observations(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        loaded = pickle.load(handle)  # nosec
    if isinstance(loaded, dict):
        return [loaded]
    if isinstance(loaded, list) and all(isinstance(item, dict) for item in loaded):
        return loaded
    raise TypeError(f"Unsupported sample payload type from {path}: {type(loaded)}")


def _build_synthetic_observation(image_height: int, image_width: int, seed: int, step: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed + step)
    observation: dict[str, Any] = {}
    for index, key in enumerate(POSITION_FEATURE_NAMES):
        observation[key] = np.float32(math.sin((step + index) / 20.0))
    for camera_offset, camera_key in enumerate(CAMERA_KEYS):
        observation[camera_key] = rng.integers(
            0,
            255,
            size=(image_height, image_width, 3),
            dtype=np.uint8,
        )
        if camera_offset == 0:
            observation[camera_key][0, 0, 0] = np.uint8(step % 255)
    return observation


@dataclass
class BaselineServerConfig:
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
    log_level: str
    dry_run: bool = False


@dataclass
class BaselineClientConfig:
    server_host: str
    server_port: int
    socket_timeout_s: float
    reconnect_retries: int
    reconnect_retry_delay_s: float
    fps: int
    steps: int
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
    control_arm: str
    observation_source: str
    sample_observation_path: str | None
    image_height: int
    image_width: int
    robot_id: str
    control_mode: str
    state_left_topic: str
    state_right_topic: str
    command_left_topic: str
    command_right_topic: str
    front_camera_topic: str
    left_camera_topic: str
    right_camera_topic: str
    observation_timeout_s: float
    queue_size: int
    seed: int
    report_json: str | None
    report_markdown: str | None
    log_interval: int
    log_level: str
    dry_run: bool = False


class SyncBaselineRuntime:
    def __init__(self, config: BaselineServerConfig, logger) -> None:
        self.config = config
        self.logger = logger
        self.active_setup: RemoteGrootPolicyConfig | None = None
        self.policy_cfg: PreTrainedConfig | None = None
        self.policy = None
        self.policy_device = None
        self.preprocessor = None
        self.postprocessor = None
        self.policy_contract: AgileXPolicyContract | None = None

    def configure(self, setup: RemoteGrootPolicyConfig) -> dict[str, Any]:
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
        assert self.policy_cfg is not None
        self.policy_contract = resolve_agilex_policy_contract(
            self.policy_cfg,
            requested_control_arm=resolved.control_arm,
        )
        self.logger.info("Probe server resolved contract | %s", summarize_policy_contract(self.policy_contract))
        return {
            "policy_path": resolved.policy_path or "",
            "task": resolved.task or "",
            "robot_type": resolved.robot_type or "",
            "backend": resolved.backend or "",
            "policy_device": str(self.policy_cfg.device),
            "policy_contract": self.policy_contract,
        }

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
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.policy_device = torch.device(str(policy_cfg.device))
        self._reset_policy_state()

        self.logger.info(
            "Probe policy loaded | path=%s | device=%s | backend=%s",
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

    def predict(self, packet: ObservationPacket) -> dict[str, Any]:
        if self.active_setup is None or self.policy is None or self.preprocessor is None or self.postprocessor is None:
            raise RuntimeError("Baseline probe runtime is not configured.")

        setup = self.active_setup
        assert self.policy_device is not None
        use_amp = bool(getattr(self.policy.config, "use_amp", False))

        server_received_at = time.time()
        total_started = time.perf_counter()

        # Mirror `lerobot.utils.control_utils.predict_action()` contexts so the
        # baseline timings reflect the real synchronous runtime boundary.
        with (
            torch.inference_mode(),
            torch.autocast(device_type=self.policy_device.type)
            if self.policy_device.type == "cuda" and use_amp
            else nullcontext(),
        ):
            preprocess_started = time.perf_counter()
            observation = deepcopy(packet.observation)
            observation = prepare_observation_for_inference(
                observation,
                self.policy_device,
                task=setup.task,
                robot_type=setup.robot_type,
            )
            observation = self.preprocessor(observation)
            preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0

            infer_started = time.perf_counter()
            action = self.policy.select_action(observation)
            infer_ms = (time.perf_counter() - infer_started) * 1000.0

            postprocess_started = time.perf_counter()
            action = self.postprocessor(action)
            if hasattr(action, "detach"):
                action = action.detach().to("cpu")
            postprocess_ms = (time.perf_counter() - postprocess_started) * 1000.0

        return {
            "step": packet.step,
            "timestamp": packet.timestamp,
            "action": action,
            "server_received_at": server_received_at,
            "server_sent_at": 0.0,
            "preprocess_ms": preprocess_ms,
            "infer_ms": infer_ms,
            "postprocess_ms": postprocess_ms,
            "serialize_ms": 0.0,
            "total_ms": (time.perf_counter() - total_started) * 1000.0,
            "predict_latency_ms": preprocess_ms + infer_ms + postprocess_ms,
        }


class ProbeClient:
    def __init__(self, config: BaselineClientConfig, logger) -> None:
        self.config = config
        self.logger = logger
        self.sock: socket.socket | None = None
        self.ready_payload: dict[str, Any] | None = None
        self.setup = RemoteGrootPolicyConfig(
            policy_path=config.policy_path,
            task=config.task,
            robot_type=config.robot_type,
            backend=config.backend,
            trt_engine_path=config.trt_engine_path,
            vit_dtype=config.vit_dtype,
            llm_dtype=config.llm_dtype,
            dit_dtype=config.dit_dtype,
            trt_action_head_only=config.trt_action_head_only,
            policy_device=config.policy_device,
            control_arm=config.control_arm,
            reset_policy_state=True,
        )

    def connect(self) -> dict[str, Any]:
        if self.sock is not None and self.ready_payload is not None:
            return self.ready_payload

        attempts = max(1, self.config.reconnect_retries)
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            sock: socket.socket | None = None
            try:
                sock = open_client_socket(
                    host=self.config.server_host,
                    port=self.config.server_port,
                    timeout_s=self.config.socket_timeout_s,
                )
                send_message(sock, RemoteMessage(kind="hello", payload=self.setup))
                reply = receive_message(sock, max_message_bytes=DEFAULT_MAX_MESSAGE_BYTES)
                if not isinstance(reply, RemoteMessage):
                    raise TypeError(f"Unexpected handshake response type: {type(reply)}")
                if reply.kind == "error":
                    self._raise_remote_error(reply.payload)
                if reply.kind != "ready":
                    raise RuntimeError(f"Unexpected handshake response kind: {reply.kind}")
                if not isinstance(reply.payload, dict):
                    raise TypeError(f"Unexpected ready payload type: {type(reply.payload)}")

                self.sock = sock
                self.ready_payload = reply.payload
                return reply.payload
            except Exception as exc:
                last_error = exc
                self.logger.warning("Probe connect attempt %s/%s failed: %s", attempt, attempts, exc)
                try:
                    if sock is not None:
                        sock.close()
                except Exception:
                    pass
                if attempt < attempts:
                    time.sleep(max(0.0, self.config.reconnect_retry_delay_s))

        assert last_error is not None
        raise last_error

    def close(self) -> None:
        if self.sock is None:
            return
        try:
            send_message(self.sock, RemoteMessage(kind="close"))
            _ = receive_message(self.sock, max_message_bytes=DEFAULT_MAX_MESSAGE_BYTES)
        except Exception:
            pass
        finally:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
            self.ready_payload = None

    def infer(self, packet: ObservationPacket) -> dict[str, Any]:
        self.connect()
        assert self.sock is not None

        send_started = time.perf_counter()
        send_message(self.sock, RemoteMessage(kind="infer", payload=packet))
        send_ms = (time.perf_counter() - send_started) * 1000.0

        receive_started = time.perf_counter()
        reply = receive_message(self.sock, max_message_bytes=DEFAULT_MAX_MESSAGE_BYTES)
        receive_ms = (time.perf_counter() - receive_started) * 1000.0

        if not isinstance(reply, RemoteMessage):
            raise TypeError(f"Unexpected inference response type: {type(reply)}")
        if reply.kind == "error":
            self._raise_remote_error(reply.payload)
        if reply.kind != PROBE_ACTION_KIND:
            raise RuntimeError(f"Unexpected probe response kind: {reply.kind}")
        if not isinstance(reply.payload, dict):
            raise TypeError(f"Unexpected probe payload type: {type(reply.payload)}")

        payload = dict(reply.payload)
        payload["send_ms"] = send_ms
        payload["receive_ms"] = receive_ms
        return payload

    @staticmethod
    def _raise_remote_error(payload: Any) -> None:
        if isinstance(payload, RemoteError):
            if payload.traceback:
                raise RuntimeError(f"{payload.message}\n{payload.traceback}")
            raise RuntimeError(payload.message)
        raise RuntimeError(f"Unexpected remote error payload: {payload}")


def _iter_raw_observations(
    config: BaselineClientConfig,
    *,
    policy_contract: AgileXPolicyContract,
) -> Iterable[dict[str, Any]]:
    if config.observation_source == "synthetic":
        for step in range(config.steps):
            yield _build_synthetic_observation(
                image_height=config.image_height,
                image_width=config.image_width,
                seed=config.seed,
                step=step,
            )
        return

    if config.observation_source == "pickle":
        if not config.sample_observation_path:
            raise ValueError("--sample-observation-path is required when observation-source=pickle")
        samples = _load_pickled_observations(Path(config.sample_observation_path).expanduser())
        for step in range(config.steps):
            yield deepcopy(samples[step % len(samples)])
        return

    if config.observation_source != "robot":
        raise ValueError(f"Unsupported observation-source={config.observation_source!r}")

    robot_cfg = build_agilex_robot_config(argparse.Namespace(**config.__dict__))
    robot = make_robot_from_config(robot_cfg)
    robot.connect()
    try:
        for _ in range(config.steps):
            observation = robot.get_observation()
            validate_live_observation(
                observation,
                image_height=config.image_height,
                image_width=config.image_width,
            )
            yield observation
    finally:
        if robot.is_connected:
            robot.disconnect()


def _initial_budget_recommendation(round_trip_p95_ms: float, round_trip_p99_ms: float, fps: int) -> dict[str, Any]:
    control_period_s = 1.0 / max(float(fps), 1.0)
    target_p95_ack_latency_s = round_trip_p95_ms / 1000.0
    target_p99_ack_latency_s = round_trip_p99_ms / 1000.0
    required_buffer_steps = max(0, math.ceil(target_p95_ack_latency_s / control_period_s))
    low_watermark = required_buffer_steps
    chunk_size = low_watermark + 1
    response_timeout_s = max(
        target_p99_ack_latency_s + control_period_s,
        target_p95_ack_latency_s + 2.0 * control_period_s,
    )
    return {
        "control_period_s": control_period_s,
        "target_p95_ack_latency_s": target_p95_ack_latency_s,
        "target_p99_ack_latency_s": target_p99_ack_latency_s,
        "required_buffer_steps": required_buffer_steps,
        "candidate_low_watermark": low_watermark,
        "candidate_chunk_size": chunk_size,
        "initial_response_timeout_s": response_timeout_s,
        "note": (
            "Initial upper bound from synchronous baseline only. "
            "Phase 4 async live latency and Phase 5 replay must recalibrate timeout and buffer sizing."
        ),
    }


def _render_markdown_report(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    budget = report["budget_recommendation"]
    return "\n".join(
        [
            "# AgileX Sync Baseline Probe Report",
            "",
            "## Summary",
            f"- samples: {metrics['round_trip_ms']['count']}",
            f"- control_fps_target: {report['config']['fps']}",
            f"- round_trip_p95_ms: {metrics['round_trip_ms']['p95_ms']:.2f}",
            f"- round_trip_p99_ms: {metrics['round_trip_ms']['p99_ms']:.2f}",
            f"- server_total_p95_ms: {metrics['server_total_ms']['p95_ms']:.2f}",
            "",
            "## Budget Recommendation",
            f"- initial_response_timeout_s: {budget['initial_response_timeout_s']:.4f}",
            f"- required_buffer_steps: {budget['required_buffer_steps']}",
            f"- candidate_low_watermark: {budget['candidate_low_watermark']}",
            f"- candidate_chunk_size: {budget['candidate_chunk_size']}",
            "",
            "## Caveat",
            f"- {budget['note']}",
            "",
        ]
    )


def run_server(config: BaselineServerConfig) -> None:
    logger = configure_logging("groot_agilex_sync_baseline_probe_server", config.log_level)
    if config.dry_run:
        print(config)
        return

    runtime = SyncBaselineRuntime(config, logger)
    if config.policy_path:
        runtime.configure(
            RemoteGrootPolicyConfig(
                policy_path=config.policy_path,
                control_arm=config.control_arm,
                reset_policy_state=True,
            )
        )

    with socket.create_server((config.host, config.port), backlog=1, reuse_port=False) as server_sock:
        server_sock.settimeout(1.0)
        logger.info("Sync baseline probe server listening | host=%s | port=%s", config.host, config.port)
        while True:
            try:
                conn, addr = server_sock.accept()
            except socket.timeout:
                continue

            logger.info("Probe client connected | address=%s:%s", addr[0], addr[1])
            with conn:
                conn.settimeout(config.socket_timeout_s)
                while True:
                    try:
                        message = receive_message(conn, max_message_bytes=config.max_message_bytes)
                    except socket.timeout:
                        logger.warning("Probe client timed out | address=%s:%s", addr[0], addr[1])
                        break
                    except EOFError:
                        logger.info("Probe client disconnected | address=%s:%s", addr[0], addr[1])
                        break

                    if not isinstance(message, RemoteMessage):
                        payload = RemoteError(message=f"Unexpected message type: {type(message)}")
                        send_message(conn, RemoteMessage(kind="error", payload=payload))
                        continue

                    try:
                        if message.kind == "hello":
                            if not isinstance(message.payload, RemoteGrootPolicyConfig):
                                raise TypeError(f"Expected RemoteGrootPolicyConfig, got {type(message.payload)}")
                            ready_payload = runtime.configure(message.payload)
                            send_message(conn, RemoteMessage(kind="ready", payload=ready_payload))
                        elif message.kind == "infer":
                            if not isinstance(message.payload, ObservationPacket):
                                raise TypeError(f"Expected ObservationPacket, got {type(message.payload)}")
                            response_payload = runtime.predict(message.payload)
                            preview_message = RemoteMessage(kind=PROBE_ACTION_KIND, payload=response_payload)
                            _, preview_serialize_ms = _serialize_message(preview_message)
                            response_payload["serialize_ms"] = preview_serialize_ms
                            response_payload["server_sent_at"] = time.time()
                            response_payload["total_ms"] += preview_serialize_ms
                            outbound = RemoteMessage(kind=PROBE_ACTION_KIND, payload=response_payload)
                            serialized, outbound_serialize_ms = _serialize_message(outbound)
                            response_payload["serialize_ms"] += outbound_serialize_ms
                            response_payload["total_ms"] += outbound_serialize_ms
                            _send_serialized_message(conn, serialized)
                        elif message.kind == "reset":
                            runtime._reset_policy_state()
                            send_message(conn, RemoteMessage(kind="reset_ack"))
                        elif message.kind == "close":
                            send_message(conn, RemoteMessage(kind="close_ack"))
                            break
                        else:
                            raise ValueError(f"Unsupported message kind: {message.kind}")
                    except Exception as exc:
                        logger.exception("Probe request failed | address=%s:%s | kind=%s", addr[0], addr[1], message.kind)
                        send_message(
                            conn,
                            RemoteMessage(
                                kind="error",
                                payload=RemoteError(message=str(exc), traceback=traceback.format_exc()),
                            ),
                        )


def run_client(config: BaselineClientConfig) -> dict[str, Any]:
    logger = configure_logging("groot_agilex_sync_baseline_probe_client", config.log_level)
    if config.dry_run:
        print(config)
        return {}

    client = ProbeClient(config, logger)
    loop_metrics = LoopMetrics()
    first_observation_logged = False

    try:
        ready_payload = client.connect()
        policy_contract = ready_payload.get("policy_contract")
        if not isinstance(policy_contract, AgileXPolicyContract):
            raise TypeError(f"Unexpected policy contract type: {type(policy_contract)}")

        logger.info("Probe ready | %s", summarize_policy_contract(policy_contract))
        robot_observation_processor, observation_dataset_features = build_agilex_policy_runtime_helpers(
            policy_contract,
            image_height=config.image_height,
            image_width=config.image_width,
        )

        raw_observations = _iter_raw_observations(config, policy_contract=policy_contract)
        metrics_raw: dict[str, list[float]] = {
            "observation_build_ms": [],
            "send_ms": [],
            "receive_ms": [],
            "round_trip_ms": [],
            "server_preprocess_ms": [],
            "server_infer_ms": [],
            "server_postprocess_ms": [],
            "server_serialize_ms": [],
            "server_total_ms": [],
        }

        for step, raw_observation in enumerate(raw_observations):
            if not first_observation_logged:
                logger.info("First baseline observation | %s", summarize_observation(raw_observation))
                first_observation_logged = True

            build_started = time.perf_counter()
            observation_frame = build_agilex_policy_observation_frame(
                raw_observation,
                policy_contract,
                observation_dataset_features,
                robot_observation_processor,
            )
            observation_build_ms = (time.perf_counter() - build_started) * 1000.0

            packet = ObservationPacket(step=step, timestamp=time.time(), observation=observation_frame)
            response = client.infer(packet)
            round_trip_ms = (time.time() - packet.timestamp) * 1000.0

            metrics_raw["observation_build_ms"].append(observation_build_ms)
            metrics_raw["send_ms"].append(float(response["send_ms"]))
            metrics_raw["receive_ms"].append(float(response["receive_ms"]))
            metrics_raw["round_trip_ms"].append(round_trip_ms)
            metrics_raw["server_preprocess_ms"].append(float(response["preprocess_ms"]))
            metrics_raw["server_infer_ms"].append(float(response["infer_ms"]))
            metrics_raw["server_postprocess_ms"].append(float(response["postprocess_ms"]))
            metrics_raw["server_serialize_ms"].append(float(response["serialize_ms"]))
            metrics_raw["server_total_ms"].append(float(response["total_ms"]))

            current_fps = loop_metrics.tick()
            if config.log_interval > 0 and step % config.log_interval == 0:
                logger.info(
                    "Baseline step %s | build=%.2fms | round_trip=%.2fms | server_infer=%.2fms | avg_fps=%.2f | %s",
                    step,
                    observation_build_ms,
                    round_trip_ms,
                    float(response["infer_ms"]),
                    current_fps,
                    summarize_action_tensor(response["action"]),
                )

        summarized = {name: _summarize_latency(values) for name, values in metrics_raw.items()}
        budget = _initial_budget_recommendation(
            round_trip_p95_ms=summarized["round_trip_ms"]["p95_ms"],
            round_trip_p99_ms=summarized["round_trip_ms"]["p99_ms"],
            fps=config.fps,
        )
        report = {
            "mode": "client",
            "config": {
                "server_host": config.server_host,
                "server_port": config.server_port,
                "fps": config.fps,
                "steps": config.steps,
                "observation_source": config.observation_source,
            },
            "policy_contract": summarize_policy_contract(policy_contract),
            "metrics": summarized,
            "budget_recommendation": budget,
        }

        report_json = json.dumps(report, indent=2, ensure_ascii=False)
        print(report_json)
        _write_text(
            Path(config.report_json).expanduser() if config.report_json else None,
            report_json + "\n",
        )
        _write_text(
            Path(config.report_markdown).expanduser() if config.report_markdown else None,
            _render_markdown_report(report),
        )
        return report
    finally:
        client.close()


def _add_policy_server_args(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    parser.add_argument(
        "--dry-run",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("DRY_RUN", False),
    )


def _add_robot_client_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--robot-id", default=os.getenv("ROBOT_ID", "my_agilex"))
    parser.add_argument(
        "--control-mode",
        choices=("passive_follow", "command_master"),
        default=os.getenv("CONTROL_MODE", "passive_follow"),
    )
    parser.add_argument("--state-left-topic", default=os.getenv("STATE_LEFT_TOPIC", "/puppet/joint_left"))
    parser.add_argument("--state-right-topic", default=os.getenv("STATE_RIGHT_TOPIC", "/puppet/joint_right"))
    parser.add_argument("--command-left-topic", default=os.getenv("COMMAND_LEFT_TOPIC", "/master/joint_left"))
    parser.add_argument("--command-right-topic", default=os.getenv("COMMAND_RIGHT_TOPIC", "/master/joint_right"))
    parser.add_argument("--front-camera-topic", default=os.getenv("FRONT_CAMERA_TOPIC", "/camera_f/color/image_raw"))
    parser.add_argument("--left-camera-topic", default=os.getenv("LEFT_CAMERA_TOPIC", "/camera_l/color/image_raw"))
    parser.add_argument("--right-camera-topic", default=os.getenv("RIGHT_CAMERA_TOPIC", "/camera_r/color/image_raw"))
    parser.add_argument(
        "--observation-timeout-s",
        type=float,
        default=float(os.getenv("OBSERVATION_TIMEOUT_S", "3.0")),
    )
    parser.add_argument("--queue-size", type=int, default=int(os.getenv("AGILEX_QUEUE_SIZE", "4")))
    parser.add_argument("--image-height", type=int, default=int(os.getenv("IMAGE_HEIGHT", "480")))
    parser.add_argument("--image-width", type=int, default=int(os.getenv("IMAGE_WIDTH", "640")))
    parser.add_argument("--fps", type=int, default=int(os.getenv("FPS", "10")))
    parser.add_argument("--steps", type=int, default=int(os.getenv("BASELINE_PROBE_STEPS", "50")))
    parser.add_argument("--server-host", default=os.getenv("REMOTE_GROOT_SERVER_HOST", DEFAULT_SERVER_HOST))
    parser.add_argument(
        "--server-port",
        type=int,
        default=int(os.getenv("REMOTE_GROOT_SERVER_PORT", str(DEFAULT_SERVER_PORT))),
    )
    parser.add_argument(
        "--reconnect-retries",
        type=int,
        default=int(os.getenv("REMOTE_GROOT_RECONNECT_RETRIES", str(DEFAULT_RECONNECT_RETRIES))),
    )
    parser.add_argument(
        "--reconnect-retry-delay-s",
        type=float,
        default=float(os.getenv("REMOTE_GROOT_RECONNECT_RETRY_DELAY_S", str(DEFAULT_RECONNECT_RETRY_DELAY_S))),
    )
    parser.add_argument(
        "--observation-source",
        choices=["synthetic", "pickle", "robot"],
        default=os.getenv("BASELINE_OBSERVATION_SOURCE", "synthetic"),
    )
    parser.add_argument("--sample-observation-path", default=os.getenv("BASELINE_SAMPLE_OBSERVATION_PATH"))
    parser.add_argument("--seed", type=int, default=int(os.getenv("BASELINE_SYNTHETIC_SEED", "7")))
    parser.add_argument("--report-json", default=os.getenv("BASELINE_REPORT_JSON"))
    parser.add_argument("--report-markdown", default=os.getenv("BASELINE_REPORT_MARKDOWN"))
    parser.add_argument("--log-interval", type=int, default=int(os.getenv("LOG_INTERVAL", "10")))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync baseline probe for AgileX remote GR00T inference.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    server_parser = subparsers.add_parser("server", help="Run the baseline probe server.")
    server_parser.add_argument("--host", default=os.getenv("REMOTE_GROOT_SERVER_HOST", "0.0.0.0"))
    server_parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("REMOTE_GROOT_SERVER_PORT", str(DEFAULT_SERVER_PORT))),
    )
    _add_policy_server_args(server_parser)

    client_parser = subparsers.add_parser("client", help="Run the baseline probe client.")
    _add_policy_server_args(client_parser)
    _add_robot_client_args(client_parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "server":
        run_server(
            BaselineServerConfig(
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
                log_level=args.log_level,
                dry_run=args.dry_run,
            )
        )
        return

    run_client(
        BaselineClientConfig(
            server_host=args.server_host,
            server_port=args.server_port,
            socket_timeout_s=args.socket_timeout_s,
            reconnect_retries=args.reconnect_retries,
            reconnect_retry_delay_s=args.reconnect_retry_delay_s,
            fps=args.fps,
            steps=args.steps,
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
            observation_source=args.observation_source,
            sample_observation_path=args.sample_observation_path,
            image_height=args.image_height,
            image_width=args.image_width,
            robot_id=args.robot_id,
            control_mode=args.control_mode,
            state_left_topic=args.state_left_topic,
            state_right_topic=args.state_right_topic,
            command_left_topic=args.command_left_topic,
            command_right_topic=args.command_right_topic,
            front_camera_topic=args.front_camera_topic,
            left_camera_topic=args.left_camera_topic,
            right_camera_topic=args.right_camera_topic,
            observation_timeout_s=args.observation_timeout_s,
            queue_size=args.queue_size,
            seed=args.seed,
            report_json=args.report_json,
            report_markdown=args.report_markdown,
            log_interval=args.log_interval,
            log_level=args.log_level,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
