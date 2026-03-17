#!/usr/bin/env python

"""Remote synchronous GR00T policy server for AgileX robot inference.

This server keeps the same inference boundary as `run_groot_infer.py`:
client sends one processed observation frame, server runs one `predict_action`
call, and returns one action tensor.

Example:
python my_devs/train/groot/agilex/run_groot_remote_policy_server.py \
    --host 0.0.0.0 \
    --port 5560 \
    --policy-path /path/to/pretrained_model \
    --task "Pick up the object"
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
if SCRIPT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, SCRIPT_DIR.as_posix())

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device

from remote_sync_common import (
    AUTO_ARM,
    DEFAULT_MAX_MESSAGE_BYTES,
    DEFAULT_SERVER_PORT,
    DEFAULT_SOCKET_TIMEOUT_S,
    AgileXPolicyContract,
    RemoteError,
    RemoteGrootPolicyConfig,
    RemoteMessage,
    RemoteReady,
    ActionPacket,
    ObservationPacket,
    configure_logging,
    ensure_groot_checkpoint_assets,
    env_bool,
    load_pre_post_processors,
    model_cache_key,
    parse_bool,
    receive_message,
    register_plugins_once,
    resolve_agilex_policy_contract,
    resolve_policy_device,
    resolve_remote_policy_config,
    send_message,
    summarize_policy_contract,
    sync_processor_device,
)


@dataclass
class PolicyServerConfig:
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


class GrootPolicyRuntime:
    def __init__(self, config: PolicyServerConfig, logger):
        self.config = config
        self.logger = logger
        self.lock = threading.Lock()
        self.active_setup: RemoteGrootPolicyConfig | None = None
        self.policy_cfg: PreTrainedConfig | None = None
        self.policy = None
        self.policy_device = None
        self.preprocessor = None
        self.postprocessor = None
        self.policy_contract: AgileXPolicyContract | None = None

    def configure(self, setup: RemoteGrootPolicyConfig) -> RemoteReady:
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

        with self.lock:
            if self.active_setup is None or model_cache_key(self.active_setup) != model_cache_key(resolved):
                self._load_policy(resolved)
            elif resolved.reset_policy_state:
                self._reset_policy_state()

            self.active_setup = resolved
            assert self.policy_cfg is not None
            policy_contract = resolve_agilex_policy_contract(
                self.policy_cfg,
                requested_control_arm=resolved.control_arm,
            )
            self.policy_contract = policy_contract
            self.logger.info("Resolved AgileX policy contract | %s", summarize_policy_contract(policy_contract))
            return RemoteReady(
                policy_path=resolved.policy_path or "",
                task=resolved.task or "",
                robot_type=resolved.robot_type or "",
                backend=resolved.backend or "",
                policy_device=str(self.policy_cfg.device),
                policy_contract=policy_contract,
            )

    def _load_policy(self, setup: RemoteGrootPolicyConfig) -> None:
        register_plugins_once()

        assert setup.policy_path is not None
        policy_path = Path(setup.policy_path).expanduser()
        if not policy_path.is_dir():
            raise FileNotFoundError(f"Policy path does not exist: {policy_path}")
        ensure_groot_checkpoint_assets(policy_path)

        policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path))
        policy_cfg.pretrained_path = policy_path
        if policy_cfg.type != "groot":
            raise ValueError(f"Expected a groot checkpoint, got policy type {policy_cfg.type!r}")
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
        self.policy_device = get_safe_torch_device(policy_cfg.device)
        self._reset_policy_state()

        self.logger.info(
            "Policy loaded | path=%s | device=%s | backend=%s",
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

    def predict(self, packet: ObservationPacket) -> ActionPacket:
        with self.lock:
            if self.active_setup is None:
                raise RuntimeError("Policy runtime is not configured. Client must send setup first.")
            if self.policy is None or self.preprocessor is None or self.postprocessor is None:
                raise RuntimeError("Policy runtime is not ready.")
            if self.policy_device is None:
                raise RuntimeError("Policy device has not been resolved.")

            setup = self.active_setup
            received_at = time.time()
            started = time.perf_counter()
            action = predict_action(
                observation=packet.observation,
                policy=self.policy,
                device=self.policy_device,
                preprocessor=self.preprocessor,
                postprocessor=self.postprocessor,
                use_amp=bool(getattr(self.policy.config, "use_amp", False)),
                task=setup.task,
                robot_type=setup.robot_type,
            )
            predict_latency_ms = (time.perf_counter() - started) * 1000.0
            if hasattr(action, "detach"):
                action = action.detach().to("cpu")
            sent_at = time.time()

            return ActionPacket(
                step=packet.step,
                timestamp=packet.timestamp,
                action=action,
                server_received_at=received_at,
                server_sent_at=sent_at,
                predict_latency_ms=predict_latency_ms,
                total_latency_ms=(sent_at - packet.timestamp) * 1000.0,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remote synchronous GR00T policy server for AgileX.")
    parser.add_argument("--host", default=os.getenv("REMOTE_GROOT_SERVER_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("REMOTE_GROOT_SERVER_PORT", str(DEFAULT_SERVER_PORT))),
    )
    parser.add_argument("--policy-path", default=os.getenv("POLICY_PATH"))
    parser.add_argument(
        "--task",
        default=os.getenv("DATASET_TASK", "Execute the trained AgileX GR00T task"),
    )
    parser.add_argument("--robot-type", default=os.getenv("ROBOT_TYPE", "agilex"))
    parser.add_argument(
        "--backend",
        default=os.getenv("INFER_BACKEND", "pytorch"),
        choices=["pytorch", "tensorrt"],
    )
    parser.add_argument("--trt-engine-path", default=os.getenv("TRT_ENGINE_PATH"))
    parser.add_argument(
        "--vit-dtype",
        default=os.getenv("TRT_VIT_DTYPE", "fp16"),
        choices=["fp16", "fp8"],
    )
    parser.add_argument(
        "--llm-dtype",
        default=os.getenv("TRT_LLM_DTYPE", "fp16"),
        choices=["fp16", "nvfp4", "fp8", "nvfp4_full"],
    )
    parser.add_argument(
        "--dit-dtype",
        default=os.getenv("TRT_DIT_DTYPE", "fp16"),
        choices=["fp16", "fp8"],
    )
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
    return parser


def handle_client(
    conn: socket.socket,
    addr: tuple[str, int],
    runtime: GrootPolicyRuntime,
    config: PolicyServerConfig,
    logger,
) -> None:
    logger.info("Client connected | address=%s:%s", addr[0], addr[1])
    with conn:
        conn.settimeout(config.socket_timeout_s)
        while True:
            try:
                message = receive_message(conn, max_message_bytes=config.max_message_bytes)
            except socket.timeout:
                logger.warning("Client timed out | address=%s:%s", addr[0], addr[1])
                break
            except EOFError:
                logger.info("Client disconnected | address=%s:%s", addr[0], addr[1])
                break

            if not isinstance(message, RemoteMessage):
                error = RemoteError(message=f"Unexpected message type: {type(message)}")
                send_message(conn, RemoteMessage(kind="error", payload=error))
                continue

            try:
                if message.kind == "hello":
                    if not isinstance(message.payload, RemoteGrootPolicyConfig):
                        raise TypeError(f"Expected RemoteGrootPolicyConfig, got {type(message.payload)}")
                    ready = runtime.configure(message.payload)
                    send_message(conn, RemoteMessage(kind="ready", payload=ready))
                elif message.kind == "infer":
                    if not isinstance(message.payload, ObservationPacket):
                        raise TypeError(f"Expected ObservationPacket, got {type(message.payload)}")
                    action_packet = runtime.predict(message.payload)
                    send_message(conn, RemoteMessage(kind="action", payload=action_packet))
                elif message.kind == "reset":
                    runtime._reset_policy_state()
                    send_message(conn, RemoteMessage(kind="reset_ack"))
                elif message.kind == "ping":
                    send_message(conn, RemoteMessage(kind="pong", payload=time.time()))
                elif message.kind == "close":
                    send_message(conn, RemoteMessage(kind="close_ack"))
                    break
                else:
                    raise ValueError(f"Unsupported message kind: {message.kind}")
            except Exception as exc:
                logger.exception("Client request failed | address=%s:%s | kind=%s", addr[0], addr[1], message.kind)
                send_message(
                    conn,
                    RemoteMessage(
                        kind="error",
                        payload=RemoteError(message=str(exc), traceback=traceback.format_exc()),
                    ),
                )


def serve(config: PolicyServerConfig) -> None:
    logger = configure_logging("groot_agilex_remote_policy_server", config.log_level)
    runtime = GrootPolicyRuntime(config, logger)

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
        logger.info(
            "Remote GR00T policy server listening | host=%s | port=%s | policy_path=%s",
            config.host,
            config.port,
            config.policy_path,
        )
        while True:
            try:
                conn, addr = server_sock.accept()
            except socket.timeout:
                continue
            handle_client(conn, addr, runtime, config, logger)


def main() -> None:
    args = build_parser().parse_args()
    config = PolicyServerConfig(
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
    if config.dry_run:
        print(config)
        return
    serve(config)


if __name__ == "__main__":
    main()
