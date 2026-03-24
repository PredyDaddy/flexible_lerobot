#!/usr/bin/env python

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import argparse
import threading
import time
from pathlib import Path

from lerobot.utils.import_utils import register_third_party_plugins

from my_devs.async_act.common import (
    ARM_CHOICES,
    DEFAULT_ARM,
    DEFAULT_TASK,
    env_bool,
    format_checkpoint_summary,
    load_policy_config,
    parse_bool,
    parse_optional_float,
    parse_optional_int,
    resolve_actions_per_chunk,
    resolve_temporal_ensemble_settings,
    resolve_policy_device,
    validate_single_arm_checkpoint_schema,
)
from my_devs.async_act.config_single_arm_agilex_robot import SingleArmAgileXRobotConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Agilex single-arm ACT async inference.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    server = subparsers.add_parser("server", help="Start async policy server.")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8080)
    server.add_argument("--fps", type=int, default=30)
    server.add_argument("--inference-latency", type=float, default=None)
    server.add_argument("--obs-queue-timeout", type=float, default=2.0)

    client = subparsers.add_parser("client", help="Start async single-arm client.")
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
    client.add_argument("--fps", type=int, default=60)
    client.add_argument("--policy-path", required=True)
    client.add_argument("--policy-device", default=None)
    client.add_argument("--server-address", default="127.0.0.1:8080")
    client.add_argument("--actions-per-chunk", type=int, default=None)
    client.add_argument("--policy-n-action-steps", type=int, default=None)
    client.add_argument("--policy-temporal-ensemble-coeff", type=parse_optional_float, default=None)
    client.add_argument("--chunk-size-threshold", type=float, default=0.5)
    client.add_argument("--aggregate-fn-name", default="weighted_average")
    client.add_argument("--action-smoothing-alpha", type=parse_optional_float, default=None)
    client.add_argument("--max-joint-step-rad", type=parse_optional_float, default=None)
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


def build_client_config(
    args: argparse.Namespace,
) -> tuple["RobotClientConfig", object, Path, dict[str, object]]:
    policy_path = Path(args.policy_path).expanduser()
    policy_cfg = load_policy_config(policy_path)
    validate_single_arm_checkpoint_schema(policy_cfg, args.arm)

    temporal_coeff = args.policy_temporal_ensemble_coeff
    policy_n_action_steps = parse_optional_int(
        str(args.policy_n_action_steps) if args.policy_n_action_steps is not None else None
    )
    if temporal_coeff is None:
        actions_per_chunk = resolve_actions_per_chunk(
            chunk_size=int(policy_cfg.chunk_size),
            explicit_actions_per_chunk=args.actions_per_chunk,
            policy_n_action_steps=policy_n_action_steps,
        )
        effective_chunk_size_threshold = args.chunk_size_threshold
        runtime_mode = "chunk_stream"
        runtime_notes: list[str] = []
    else:
        actions_per_chunk, effective_chunk_size_threshold, runtime_notes = resolve_temporal_ensemble_settings(
            chunk_size=int(policy_cfg.chunk_size),
            explicit_actions_per_chunk=args.actions_per_chunk,
            policy_n_action_steps=policy_n_action_steps,
            policy_temporal_ensemble_coeff=temporal_coeff,
            requested_chunk_size_threshold=args.chunk_size_threshold,
            aggregate_fn_name=args.aggregate_fn_name,
        )
        runtime_mode = "act_temporal_ensemble"

    policy_device = resolve_policy_device(args.policy_device, getattr(policy_cfg, "device", None))
    policy_cfg.device = policy_device

    from lerobot.async_inference.configs import RobotClientConfig

    client_cfg = RobotClientConfig(
        robot=build_robot_config(args),
        server_address=args.server_address,
        policy_type=policy_cfg.type,
        pretrained_name_or_path=str(policy_path),
        policy_device=policy_device,
        client_device="cpu",
        chunk_size_threshold=effective_chunk_size_threshold,
        fps=args.fps,
        actions_per_chunk=actions_per_chunk,
        task=args.task,
        aggregate_fn_name=args.aggregate_fn_name,
        debug_visualize_queue_size=args.debug_visualize_queue_size,
    )
    runtime_options: dict[str, object] = {
        "runtime_mode": runtime_mode,
        "temporal_ensemble_coeff": temporal_coeff,
        "runtime_notes": runtime_notes,
    }
    return client_cfg, policy_cfg, policy_path, runtime_options


def run_client(args: argparse.Namespace) -> int:
    client_cfg, policy_cfg, policy_path, runtime_options = build_client_config(args)
    runtime_mode = str(runtime_options["runtime_mode"])
    runtime_notes = list(runtime_options["runtime_notes"])

    print(
        format_checkpoint_summary(
            arm=args.arm,
            policy_path=policy_path,
            policy_cfg=policy_cfg,
            client_fps=client_cfg.fps,
            actions_per_chunk=client_cfg.actions_per_chunk,
            chunk_size_threshold=client_cfg.chunk_size_threshold,
            aggregate_fn_name=args.aggregate_fn_name,
            server_address=args.server_address,
            control_mode=args.control_mode,
            run_time_s=args.run_time_s,
            runtime_mode=runtime_mode,
            temporal_ensemble_coeff=args.policy_temporal_ensemble_coeff,
            action_smoothing_alpha=args.action_smoothing_alpha,
            max_joint_step_rad=args.max_joint_step_rad,
            runtime_notes=runtime_notes,
        )
    )
    if args.dry_run:
        print("[INFO] DRY_RUN=true, exit without starting async client.")
        return 0

    from lerobot.async_inference.robot_client import RobotClient

    if runtime_mode == "act_temporal_ensemble":
        from my_devs.async_act.temporal_ensemble_client import TemporalEnsembleRobotClient

        client = TemporalEnsembleRobotClient(
            client_cfg,
            temporal_ensemble_coeff=float(args.policy_temporal_ensemble_coeff),
            temporal_chunk_size=int(policy_cfg.chunk_size),
            force_send_every_step=True,
        )
    else:
        client = RobotClient(client_cfg)

    if not client.start():
        raise RuntimeError("Failed to connect to policy server")

    action_thread = threading.Thread(target=client.receive_actions, daemon=True)
    control_thread = threading.Thread(target=client.control_loop, kwargs={"task": args.task}, daemon=True)
    action_thread.start()
    control_thread.start()

    deadline = time.monotonic() + args.run_time_s if args.run_time_s > 0 else None
    try:
        while control_thread.is_alive():
            if deadline is not None and time.monotonic() >= deadline:
                print("[INFO] Reached requested run_time_s. Stopping client.")
                break
            control_thread.join(timeout=0.1)
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received. Stopping async client.")
    finally:
        client.stop()
        action_thread.join(timeout=2.0)
        control_thread.join(timeout=2.0)
        print("[INFO] Agilex single-arm async client finished.")
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
    return run_client(args)


if __name__ == "__main__":
    raise SystemExit(main())
