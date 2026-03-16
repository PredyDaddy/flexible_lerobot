#!/usr/bin/env python

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import argparse
import threading
import time
from pathlib import Path

from lerobot.utils.import_utils import register_third_party_plugins

from my_devs.async_act.config_single_arm_agilex_robot import SingleArmAgileXRobotConfig
from my_devs.train.groot.agilex.common import (
    ARM_CHOICES,
    DEFAULT_ARM,
    DEFAULT_TASK,
    MVP_ACTIONS_PER_CHUNK,
    MVP_AGGREGATE_FN_NAME,
    MVP_CHUNK_SIZE_THRESHOLD,
    env_bool,
    format_check_summary,
    format_client_runtime_summary,
    parse_bool,
    validate_groot_checkpoint_for_mvp,
)
from my_devs.train.groot.agilex.one_step_robot_client import GrootOneStepRobotClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Agilex single-arm GROOT async MVP.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    server = subparsers.add_parser("server", help="Start async policy server.")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8080)
    server.add_argument("--fps", type=int, default=30)
    server.add_argument("--inference-latency", type=float, default=None)
    server.add_argument("--obs-queue-timeout", type=float, default=2.0)

    check = subparsers.add_parser("check", help="Validate a local GROOT checkpoint for the Agilex MVP.")
    check.add_argument("--policy-path", required=True)
    check.add_argument("--arm", choices=ARM_CHOICES, default=DEFAULT_ARM)
    check.add_argument("--policy-device", default="cuda")

    client = subparsers.add_parser("client", help="Start the Agilex single-arm GROOT client.")
    client.add_argument("--robot-id", default="my_agilex")
    client.add_argument("--arm", choices=ARM_CHOICES, default=DEFAULT_ARM)
    client.add_argument("--control-mode", choices=("passive_follow", "command_master"), default="passive_follow")
    client.add_argument("--state-left-topic", default="/puppet/joint_left")
    client.add_argument("--state-right-topic", default="/puppet/joint_right")
    client.add_argument("--command-left-topic", default="/master/joint_left")
    client.add_argument("--command-right-topic", default="/master/joint_right")
    client.add_argument("--front-camera-topic", default="/camera_f/color/image_raw")
    client.add_argument("--left-camera-topic", default="/camera_l/color/image_raw")
    client.add_argument("--right-camera-topic", default="/camera_r/color/image_raw")
    client.add_argument("--observation-timeout-s", type=float, default=2.0)
    client.add_argument("--queue-size", type=int, default=1)
    client.add_argument("--fps", type=int, default=30)
    client.add_argument("--policy-path", required=True)
    client.add_argument("--policy-device", default="cuda")
    client.add_argument("--server-address", default="127.0.0.1:8080")
    client.add_argument("--actions-per-chunk", type=int, default=MVP_ACTIONS_PER_CHUNK)
    client.add_argument("--chunk-size-threshold", type=float, default=MVP_CHUNK_SIZE_THRESHOLD)
    client.add_argument("--aggregate-fn-name", default=MVP_AGGREGATE_FN_NAME)
    client.add_argument("--action-smoothing-alpha", type=float, default=None)
    client.add_argument("--max-joint-step-rad", type=float, default=None)
    client.add_argument("--task", default=DEFAULT_TASK)
    client.add_argument("--run-time-s", type=float, default=0.0)
    client.add_argument("--dry-run", type=parse_bool, nargs="?", const=True, default=env_bool("DRY_RUN", False))
    client.add_argument("--debug-visualize-queue-size", type=parse_bool, nargs="?", const=True, default=False)
    return parser


def build_robot_config(args: argparse.Namespace) -> SingleArmAgileXRobotConfig:
    return SingleArmAgileXRobotConfig(
        id=args.robot_id,
        arm=args.arm,
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
        action_smoothing_alpha=args.action_smoothing_alpha,
        max_joint_step_rad=args.max_joint_step_rad,
    )


def validate_runtime_args(args: argparse.Namespace) -> None:
    if args.arm != DEFAULT_ARM:
        raise ValueError(f"Agilex GROOT MVP only supports arm={DEFAULT_ARM!r}, got {args.arm!r}")
    if args.actions_per_chunk != MVP_ACTIONS_PER_CHUNK:
        raise ValueError(
            "Agilex GROOT MVP requires --actions-per-chunk=1, "
            f"got {args.actions_per_chunk!r}"
        )
    if args.chunk_size_threshold != MVP_CHUNK_SIZE_THRESHOLD:
        raise ValueError(
            "Agilex GROOT MVP requires --chunk-size-threshold=1.0, "
            f"got {args.chunk_size_threshold!r}"
        )
    if args.aggregate_fn_name != MVP_AGGREGATE_FN_NAME:
        raise ValueError(
            "Agilex GROOT MVP requires --aggregate-fn-name=latest_only, "
            f"got {args.aggregate_fn_name!r}"
        )
    if not args.task.strip():
        raise ValueError("--task must be a non-empty string for GROOT.")


def build_client_config(args: argparse.Namespace) -> "RobotClientConfig":
    from lerobot.async_inference.configs import RobotClientConfig

    validate_runtime_args(args)
    return RobotClientConfig(
        robot=build_robot_config(args),
        server_address=args.server_address,
        policy_type="groot",
        pretrained_name_or_path=args.policy_path,
        policy_device=args.policy_device,
        client_device="cpu",
        chunk_size_threshold=MVP_CHUNK_SIZE_THRESHOLD,
        fps=args.fps,
        actions_per_chunk=MVP_ACTIONS_PER_CHUNK,
        task=args.task,
        aggregate_fn_name=MVP_AGGREGATE_FN_NAME,
        debug_visualize_queue_size=args.debug_visualize_queue_size,
    )


def run_check(args: argparse.Namespace) -> int:
    policy_path = Path(args.policy_path).expanduser()
    report = validate_groot_checkpoint_for_mvp(policy_path, args.arm)
    print(
        format_check_summary(
            policy_path=policy_path,
            arm=args.arm,
            report=report,
            policy_device=args.policy_device,
        )
    )
    return 0


def run_client(args: argparse.Namespace) -> int:
    client_cfg = build_client_config(args)
    print(
        format_client_runtime_summary(
            arm=args.arm,
            policy_path=args.policy_path,
            server_address=args.server_address,
            policy_device=args.policy_device,
            control_mode=args.control_mode,
            task=args.task,
            run_time_s=args.run_time_s,
            front_camera_key="camera_front",
            side_camera_key="camera_right",
        )
    )
    if args.dry_run:
        print("[INFO] DRY_RUN=true, exit without starting async client.")
        return 0

    client: GrootOneStepRobotClient | None = None
    action_thread: threading.Thread | None = None
    control_thread: threading.Thread | None = None
    start_succeeded = False
    try:
        client = GrootOneStepRobotClient(client_cfg)
        start_succeeded = bool(client.start())
        if not start_succeeded:
            raise RuntimeError("Failed to connect to policy server")

        action_thread = threading.Thread(target=client.receive_actions, daemon=True)
        control_thread = threading.Thread(target=client.control_loop, kwargs={"task": args.task}, daemon=True)
        action_thread.start()
        control_thread.start()

        deadline = time.monotonic() + args.run_time_s if args.run_time_s > 0 else None
        while control_thread.is_alive():
            if deadline is not None and time.monotonic() >= deadline:
                print("[INFO] Reached requested run_time_s. Stopping client.")
                break
            control_thread.join(timeout=0.1)
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received. Stopping async client.")
    finally:
        if client is not None:
            try:
                client.stop()
            except Exception as exc:
                if start_succeeded:
                    raise
                print(f"[WARN] Failed to stop client cleanly after startup failure: {exc}")
        if action_thread is not None:
            action_thread.join(timeout=2.0)
        if control_thread is not None:
            control_thread.join(timeout=2.0)
        print("[INFO] Agilex GROOT async client finished.")
    return 0


def run_server(args: argparse.Namespace) -> int:
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.policy_server import serve

    cfg = PolicyServerConfig(
        host=args.host,
        port=args.port,
        fps=args.fps,
        inference_latency=(1 / args.fps) if args.inference_latency is None else args.inference_latency,
        obs_queue_timeout=args.obs_queue_timeout,
    )
    serve(cfg)
    return 0


def main() -> int:
    register_third_party_plugins()
    args = build_parser().parse_args()
    if args.command == "server":
        return run_server(args)
    if args.command == "check":
        return run_check(args)
    return run_client(args)


if __name__ == "__main__":
    raise SystemExit(main())
