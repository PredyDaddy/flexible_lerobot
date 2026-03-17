#!/usr/bin/env python

"""Remote synchronous GR00T robot client for AgileX.

This script runs on the robot machine:
1) read AgileX observation from ROS topics
2) send one observation frame to the remote policy server
3) receive one action tensor
4) publish that action back to AgileX

Example:
python my_devs/train/groot/agilex/run_groot_remote_robot_client.py \
    --server-host 10.1.26.37 \
    --server-port 5560 \
    --control-mode command_master \
    --run-time-s 60
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if SCRIPT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, SCRIPT_DIR.as_posix())

from lerobot.robots import make_robot_from_config
from lerobot.utils.robot_utils import precise_sleep

from remote_sync_common import (
    AUTO_ARM,
    DEFAULT_RECONNECT_RETRIES,
    DEFAULT_RECONNECT_RETRY_DELAY_S,
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    DEFAULT_SOCKET_TIMEOUT_S,
    AgileXPolicyContract,
    LoopMetrics,
    ObservationPacket,
    RemoteError,
    RemoteGrootPolicyConfig,
    RemoteMessage,
    RemoteReady,
    build_agilex_hold_action,
    build_agilex_policy_observation_frame,
    build_agilex_policy_runtime_helpers,
    build_agilex_robot_config,
    configure_logging,
    decode_agilex_policy_action,
    env_bool,
    merge_agilex_action,
    open_client_socket,
    parse_bool,
    receive_message,
    send_message,
    summarize_action_tensor,
    summarize_observation,
    summarize_policy_contract,
    validate_live_observation,
)


@dataclass
class RobotClientConfig:
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
    image_height: int
    image_width: int
    fps: int
    task: str
    robot_type: str
    policy_path: str | None
    backend: str
    trt_engine_path: str | None
    vit_dtype: str
    llm_dtype: str
    dit_dtype: str
    trt_action_head_only: bool
    policy_device: str | None
    control_arm: str
    server_host: str
    server_port: int
    socket_timeout_s: float
    reconnect_retries: int
    reconnect_retry_delay_s: float
    run_time_s: float
    log_interval: int
    log_level: str
    dry_run: bool = False


class RemotePolicyClient:
    def __init__(self, config: RobotClientConfig, logger):
        self.config = config
        self.logger = logger
        self.sock: socket.socket | None = None
        self.ready_payload: RemoteReady | None = None
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

    def connect(self) -> None:
        if self.sock is not None:
            return

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
                reply = receive_message(sock)
                if not isinstance(reply, RemoteMessage):
                    raise TypeError(f"Unexpected handshake response type: {type(reply)}")
                if reply.kind == "error":
                    self._raise_remote_error(reply.payload)
                if reply.kind != "ready":
                    raise RuntimeError(f"Unexpected handshake response kind: {reply.kind}")
                if not isinstance(reply.payload, RemoteReady):
                    raise TypeError(f"Unexpected ready payload type: {type(reply.payload)}")
                if not isinstance(reply.payload.policy_contract, AgileXPolicyContract):
                    raise TypeError(
                        "Unexpected ready policy contract type: "
                        f"{type(reply.payload.policy_contract)}"
                    )

                self.sock = sock
                self.ready_payload = reply.payload
                self.logger.info(
                    "Connected to remote server | server=%s:%s | policy_path=%s | backend=%s | device=%s",
                    self.config.server_host,
                    self.config.server_port,
                    reply.payload.policy_path,
                    reply.payload.backend,
                    reply.payload.policy_device,
                )
                return
            except Exception as exc:
                last_error = exc
                self.logger.warning(
                    "Remote policy connect attempt %s/%s failed: %s",
                    attempt,
                    attempts,
                    exc,
                )
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
            _ = receive_message(self.sock)
        except Exception:
            pass
        try:
            self.sock.close()
        finally:
            self.sock = None
            self.ready_payload = None

    def infer(self, packet: ObservationPacket):
        attempts = max(1, self.config.reconnect_retries)
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                self.connect()
                assert self.sock is not None
                send_message(self.sock, RemoteMessage(kind="infer", payload=packet))
                reply = receive_message(self.sock)
                if not isinstance(reply, RemoteMessage):
                    raise TypeError(f"Unexpected inference response type: {type(reply)}")
                if reply.kind == "error":
                    self._raise_remote_error(reply.payload)
                if reply.kind != "action":
                    raise RuntimeError(f"Unexpected inference response kind: {reply.kind}")
                return reply.payload
            except Exception as exc:
                last_error = exc
                self.logger.warning(
                    "Remote inference attempt %s/%s failed at step=%s: %s",
                    attempt,
                    attempts,
                    packet.step,
                    exc,
                )
                self.close()
                if attempt < attempts:
                    time.sleep(max(0.0, self.config.reconnect_retry_delay_s))
                    continue
                raise

        assert last_error is not None
        raise last_error

    @staticmethod
    def _raise_remote_error(payload: Any) -> None:
        if isinstance(payload, RemoteError):
            if payload.traceback:
                raise RuntimeError(f"{payload.message}\n{payload.traceback}")
            raise RuntimeError(payload.message)
        raise RuntimeError(f"Remote server returned an unexpected error payload: {payload}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remote synchronous GR00T robot client for AgileX.")
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
    parser.add_argument(
        "--front-camera-topic",
        default=os.getenv("FRONT_CAMERA_TOPIC", "/camera_f/color/image_raw"),
    )
    parser.add_argument(
        "--left-camera-topic",
        default=os.getenv("LEFT_CAMERA_TOPIC", "/camera_l/color/image_raw"),
    )
    parser.add_argument(
        "--right-camera-topic",
        default=os.getenv("RIGHT_CAMERA_TOPIC", "/camera_r/color/image_raw"),
    )
    parser.add_argument(
        "--observation-timeout-s",
        type=float,
        default=float(os.getenv("OBSERVATION_TIMEOUT_S", "2.0")),
    )
    parser.add_argument("--queue-size", type=int, default=int(os.getenv("QUEUE_SIZE", "1")))
    parser.add_argument("--image-height", type=int, default=int(os.getenv("IMAGE_HEIGHT", "480")))
    parser.add_argument("--image-width", type=int, default=int(os.getenv("IMAGE_WIDTH", "640")))
    parser.add_argument("--fps", type=int, default=int(os.getenv("FPS", "30")))

    parser.add_argument(
        "--task",
        default=os.getenv("DATASET_TASK", "Execute the trained AgileX GR00T task"),
    )
    parser.add_argument("--robot-type", default=os.getenv("ROBOT_TYPE", "agilex"))
    parser.add_argument(
        "--policy-path",
        default=os.getenv("REMOTE_POLICY_PATH", os.getenv("POLICY_PATH")),
        help="Checkpoint path on the server machine. Leave empty to use the server default.",
    )
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
        help="Optional server-side device override forwarded during handshake.",
    )
    parser.add_argument(
        "--control-arm",
        default=os.getenv("CONTROL_ARM", AUTO_ARM),
        help="Requested AgileX control scope for this client: auto/left/right/both.",
    )

    parser.add_argument("--server-host", default=os.getenv("REMOTE_GROOT_SERVER_HOST", DEFAULT_SERVER_HOST))
    parser.add_argument(
        "--server-port",
        type=int,
        default=int(os.getenv("REMOTE_GROOT_SERVER_PORT", str(DEFAULT_SERVER_PORT))),
    )
    parser.add_argument(
        "--socket-timeout-s",
        type=float,
        default=float(os.getenv("REMOTE_GROOT_SOCKET_TIMEOUT_S", str(DEFAULT_SOCKET_TIMEOUT_S))),
    )
    parser.add_argument(
        "--reconnect-retries",
        type=int,
        default=int(os.getenv("REMOTE_GROOT_RECONNECT_RETRIES", str(DEFAULT_RECONNECT_RETRIES))),
    )
    parser.add_argument(
        "--reconnect-retry-delay-s",
        type=float,
        default=float(
            os.getenv("REMOTE_GROOT_RECONNECT_RETRY_DELAY_S", str(DEFAULT_RECONNECT_RETRY_DELAY_S))
        ),
    )

    parser.add_argument("--run-time-s", type=float, default=float(os.getenv("RUN_TIME_S", "0")))
    parser.add_argument("--log-interval", type=int, default=int(os.getenv("LOG_INTERVAL", "30")))
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    parser.add_argument(
        "--dry-run",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("DRY_RUN", False),
    )
    return parser


def print_runtime_summary(config: RobotClientConfig) -> None:
    print(f"[INFO] Robot id: {config.robot_id}")
    print(f"[INFO] Control mode: {config.control_mode}")
    print(
        "[INFO] State topics: "
        f"left={config.state_left_topic} right={config.state_right_topic}"
    )
    print(
        "[INFO] Command topics: "
        f"left={config.command_left_topic} right={config.command_right_topic}"
    )
    print(
        "[INFO] Camera topics: "
        f"front={config.front_camera_topic} left={config.left_camera_topic} right={config.right_camera_topic}"
    )
    print(f"[INFO] Image shape (HWC): ({config.image_height}, {config.image_width}, 3)")
    print(f"[INFO] Server: {config.server_host}:{config.server_port}")
    print(f"[INFO] Task: {config.task}")
    print(f"[INFO] Policy path override: {config.policy_path}")
    print(f"[INFO] Backend: {config.backend}")
    print(f"[INFO] control_arm: {config.control_arm}")
    print(f"[INFO] FPS: {config.fps}")
    print(f"[INFO] run_time_s: {config.run_time_s} (<=0 means until Ctrl+C)")


def run(config: RobotClientConfig) -> None:
    logger = configure_logging("groot_agilex_remote_robot_client", config.log_level)
    print_runtime_summary(config)

    if config.dry_run:
        print("[INFO] DRY_RUN=true, exit without execution.")
        return

    robot_cfg = build_agilex_robot_config(argparse.Namespace(**config.__dict__))
    robot = make_robot_from_config(robot_cfg)
    policy_client = RemotePolicyClient(config, logger)
    loop_metrics = LoopMetrics()

    step = 0
    start_t = time.perf_counter()
    end_t = start_t + config.run_time_s if config.run_time_s > 0 else None
    first_observation_logged = False

    try:
        policy_client.connect()
        ready_payload = policy_client.ready_payload
        if ready_payload is None:
            raise RuntimeError("Remote server handshake completed without a ready payload")
        policy_contract = ready_payload.policy_contract
        logger.info("Remote AgileX policy contract | %s", summarize_policy_contract(policy_contract))
        robot_observation_processor, observation_dataset_features = build_agilex_policy_runtime_helpers(
            policy_contract,
            image_height=config.image_height,
            image_width=config.image_width,
        )
        robot.connect()
        logger.info("AgileX connected. publish_enabled=%s", config.control_mode == "command_master")

        while True:
            if end_t is not None and time.perf_counter() >= end_t:
                logger.info("Reached requested run_time_s. Exiting loop.")
                break

            loop_started = time.perf_counter()
            raw_observation = robot.get_observation()
            validate_live_observation(
                raw_observation,
                image_height=config.image_height,
                image_width=config.image_width,
            )
            if not first_observation_logged:
                logger.info("First observation summary | %s", summarize_observation(raw_observation))
                first_observation_logged = True

            observation_frame = build_agilex_policy_observation_frame(
                raw_observation,
                policy_contract,
                observation_dataset_features,
                robot_observation_processor,
            )
            packet = ObservationPacket(
                step=step,
                timestamp=time.time(),
                observation=observation_frame,
            )
            action_packet = policy_client.infer(packet)
            predicted_action = decode_agilex_policy_action(action_packet.action, policy_contract)
            hold_action = build_agilex_hold_action(raw_observation)
            robot_action_to_send = merge_agilex_action(hold_action, predicted_action)
            _ = robot.send_action(robot_action_to_send)

            current_fps = loop_metrics.tick()
            if config.log_interval > 0 and step % config.log_interval == 0:
                round_trip_ms = (time.time() - packet.timestamp) * 1000.0
                logger.info(
                    "Step %s | predict=%.2fms | round_trip=%.2fms | avg_fps=%.2f | %s",
                    step,
                    action_packet.predict_latency_ms,
                    round_trip_ms,
                    current_fps,
                    summarize_action_tensor(action_packet.action),
                )

            step += 1
            precise_sleep(max(1.0 / float(config.fps) - (time.perf_counter() - loop_started), 0.0))

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Stopping client.")
    finally:
        policy_client.close()
        if robot.is_connected:
            robot.disconnect()


def main() -> None:
    args = build_parser().parse_args()
    config = RobotClientConfig(
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
        image_height=args.image_height,
        image_width=args.image_width,
        fps=args.fps,
        task=args.task,
        robot_type=args.robot_type,
        policy_path=args.policy_path,
        backend=args.backend,
        trt_engine_path=args.trt_engine_path,
        vit_dtype=args.vit_dtype,
        llm_dtype=args.llm_dtype,
        dit_dtype=args.dit_dtype,
        trt_action_head_only=args.trt_action_head_only,
        policy_device=args.policy_device,
        control_arm=args.control_arm,
        server_host=args.server_host,
        server_port=args.server_port,
        socket_timeout_s=args.socket_timeout_s,
        reconnect_retries=args.reconnect_retries,
        reconnect_retry_delay_s=args.reconnect_retry_delay_s,
        run_time_s=args.run_time_s,
        log_interval=args.log_interval,
        log_level=args.log_level,
        dry_run=args.dry_run,
    )
    run(config)


if __name__ == "__main__":
    main()
