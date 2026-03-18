#!/usr/bin/env python

"""Run ACT TensorRT policy inference on SO follower robot.

This is the real robot entrypoint for the ACT TensorRT adapter. Unlike
`run_act_trt_mock.py`, this script can run against a live robot loop.

Safety:
- `--dry-run true`: resolve paths/config and exit before loading TRT or connecting robot
- `--mock-observation true`: run the full preprocess -> TRT policy -> postprocess path with synthetic
  observations and no robot connection
- `--mock-send true`: connect robot and run inference, but do not send actions
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lerobot import policies  # noqa: F401  # Register policy config classes.
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import auto_select_torch_device, get_safe_torch_device

from my_devs.act_trt.common import (
    DEFAULT_CHECKPOINT,
    apply_act_policy_runtime_overrides,
    apply_act_runtime_overrides,
    load_act_policy,
    load_model_spec,
    load_pre_post_processors,
    make_mock_observation,
    save_json,
)
from my_devs.act_trt.trt_policy import ActTrtPolicyAdapter

DEFAULT_POLICY_PATH = str(DEFAULT_CHECKPOINT)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else parse_bool(raw)


def maybe_path(path_str: str | None) -> Path | None:
    return None if not path_str else Path(path_str).expanduser()


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"", "none", "null"}:
        return None
    return float(value)


def parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"", "none", "null", "0"}:
        return None
    return int(value)


def resolve_policy_dir(policy_path: Path) -> Path:
    candidates = [
        policy_path,
        policy_path / "pretrained_model",
        policy_path / "checkpoints" / "last" / "pretrained_model",
    ]
    for candidate in candidates:
        if (candidate / "config.json").is_file():
            if candidate != policy_path:
                print(
                    "[WARN] --policy-path does not point to a `pretrained_model/` directory. "
                    f"Using detected path: {candidate}"
                )
            return candidate

    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Policy directory must contain `config.json` (the lerobot `pretrained_model/` folder).\n"
        f"You passed: {policy_path}\n"
        "Searched:\n"
        f"{searched}"
    )


def resolve_trt_device(requested: str | None, config_device: str | None) -> str:
    if requested is not None and requested.strip().lower() in {"", "none", "null"}:
        requested = None

    if requested is None:
        if config_device is not None and str(config_device).lower() not in {"", "none", "null"}:
            requested = str(config_device)
        else:
            auto_device = auto_select_torch_device()
            requested = "cuda:0" if auto_device.type == "cuda" else auto_device.type

    if requested == "auto":
        auto_device = auto_select_torch_device()
        requested = "cuda:0" if auto_device.type == "cuda" else auto_device.type

    return requested


def summarize_timings(samples: list[float]) -> tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    return float(np.mean(samples)), float(np.max(samples))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ACT TensorRT inference loop for SO101/SO100 follower robot."
    )
    parser.add_argument("--robot-id", default=os.getenv("ROBOT_ID", "my_so101"))
    parser.add_argument("--robot-type", default=os.getenv("ROBOT_TYPE", "so101_follower"))
    parser.add_argument(
        "--calib-dir",
        default=os.getenv(
            "CALIB_DIR", "/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower"
        ),
    )
    parser.add_argument("--robot-port", default=os.getenv("ROBOT_PORT", "/dev/ttyACM0"))

    parser.add_argument("--top-cam-index", type=int, default=int(os.getenv("TOP_CAM_INDEX", "4")))
    parser.add_argument("--wrist-cam-index", type=int, default=int(os.getenv("WRIST_CAM_INDEX", "6")))
    parser.add_argument("--img-width", type=int, default=int(os.getenv("IMG_WIDTH", "640")))
    parser.add_argument("--img-height", type=int, default=int(os.getenv("IMG_HEIGHT", "480")))
    parser.add_argument("--fps", type=int, default=int(os.getenv("FPS", "30")))

    parser.add_argument("--policy-path", default=os.getenv("POLICY_PATH", DEFAULT_POLICY_PATH))
    parser.add_argument("--engine", required=True, help="TensorRT engine path (.plan).")
    parser.add_argument("--export-metadata", default=os.getenv("EXPORT_METADATA"))
    parser.add_argument(
        "--policy-device",
        default=os.getenv("POLICY_DEVICE_OVERRIDE"),
        help="Override TensorRT execution device. Must resolve to CUDA for real execution.",
    )
    parser.add_argument(
        "--policy-n-action-steps",
        type=int,
        default=parse_optional_int(os.getenv("POLICY_N_ACTION_STEPS")),
        help="Optional ACT deployment override. Must be in [1, chunk_size].",
    )
    parser.add_argument(
        "--policy-temporal-ensemble-coeff",
        type=parse_optional_float,
        default=parse_optional_float(os.getenv("POLICY_TEMPORAL_ENSEMBLE_COEFF")),
        help="Optional ACT deployment override. If set, requires policy-n-action-steps=1.",
    )
    parser.add_argument(
        "--task",
        default=os.getenv("DATASET_TASK", "Put the block in the bin"),
        help="Language instruction passed to policy inference.",
    )
    parser.add_argument(
        "--run-time-s",
        type=float,
        default=float(os.getenv("RUN_TIME_S", "0")),
        help="Total inference duration in seconds. <=0 means run until Ctrl+C.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(os.getenv("MAX_STEPS", "0")),
        help="Optional hard stop after N steps. <=0 means no explicit step limit.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=int(os.getenv("LOG_INTERVAL", "30")),
        help="Print status every N steps.",
    )
    parser.add_argument(
        "--report",
        default=os.getenv("REPORT_PATH"),
        help="Optional JSON summary path. Defaults next to the TRT engine.",
    )
    parser.add_argument(
        "--dry-run",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("DRY_RUN", False),
        help="Print resolved config and exit without loading TRT or connecting robot.",
    )
    parser.add_argument(
        "--mock-observation",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("MOCK_OBSERVATION", False),
        help="Run with synthetic observations and no robot connection.",
    )
    parser.add_argument(
        "--mock-case",
        choices=["random", "zeros", "ones", "linspace"],
        default=os.getenv("MOCK_CASE", "random"),
        help="Synthetic observation pattern used when --mock-observation is enabled.",
    )
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "0")))
    parser.add_argument(
        "--mock-send",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("MOCK_SEND", False),
        help="Connect robot and run inference but do not send actions.",
    )
    parser.add_argument(
        "--compare-torch",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("COMPARE_TORCH", False),
        help="Also run Torch ACT on the same observations for step-by-step debugging.",
    )
    parser.add_argument(
        "--torch-policy-device",
        default=os.getenv("TORCH_POLICY_DEVICE", "cpu"),
        help="Device used when --compare-torch is enabled. Recommended: cpu for short debug runs.",
    )
    return parser


def main() -> None:
    register_third_party_plugins()
    args = build_parser().parse_args()

    if isinstance(args.policy_path, str) and not args.policy_path.strip():
        print("[WARN] --policy-path is empty. Falling back to DEFAULT_POLICY_PATH.")
        args.policy_path = DEFAULT_POLICY_PATH
    if isinstance(args.engine, str) and not args.engine.strip():
        raise ValueError("--engine is empty. Pass a valid TensorRT engine path.")

    policy_path = resolve_policy_dir(Path(args.policy_path).expanduser())
    engine_path = Path(args.engine).expanduser().resolve()
    if not engine_path.is_file():
        raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

    export_metadata = Path(args.export_metadata).expanduser().resolve() if args.export_metadata else None
    if export_metadata is not None and not export_metadata.is_file():
        raise FileNotFoundError(f"Export metadata not found: {export_metadata}")

    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else engine_path.parent / "run_act_trt_real_report.json"
    )

    if args.robot_type not in {"so100_follower", "so101_follower"}:
        raise ValueError(
            f"Unsupported robot_type={args.robot_type!r}. "
            "This script currently supports so100_follower/so101_follower."
        )

    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    if policy_cfg.type != "act":
        raise ValueError(f"Expected ACT policy, got {policy_cfg.type!r}")
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = resolve_trt_device(args.policy_device, getattr(policy_cfg, "device", None))
    apply_act_runtime_overrides(
        policy_cfg=policy_cfg,
        policy_n_action_steps=args.policy_n_action_steps,
        policy_temporal_ensemble_coeff=args.policy_temporal_ensemble_coeff,
    )

    print(f"[INFO] Robot id: {args.robot_id}")
    print(f"[INFO] Robot type (requested): {args.robot_type}")
    print(f"[INFO] Robot port: {args.robot_port}")
    print(f"[INFO] Policy path: {policy_path}")
    print(f"[INFO] Policy type: {policy_cfg.type}")
    print(f"[INFO] TRT engine: {engine_path}")
    print(f"[INFO] Export metadata: {export_metadata}")
    print(f"[INFO] Policy device: {policy_cfg.device}")
    print(f"[INFO] Task: {args.task}")
    print(f"[INFO] FPS: {args.fps}")
    print(f"[INFO] run_time_s: {args.run_time_s} (<=0 means until Ctrl+C)")
    print(f"[INFO] max_steps: {args.max_steps} (<=0 means no explicit step limit)")
    print(f"[INFO] mock_observation: {args.mock_observation}")
    print(f"[INFO] mock_send(no_send_action): {args.mock_send}")
    print(f"[INFO] compare_torch: {args.compare_torch}")
    print(
        "[INFO] ACT runtime config: "
        f"chunk_size={policy_cfg.chunk_size}, "
        f"n_action_steps={policy_cfg.n_action_steps}, "
        f"temporal_ensemble_coeff={policy_cfg.temporal_ensemble_coeff}"
    )

    if args.dry_run:
        print("[INFO] DRY_RUN=true, exit without loading TRT or connecting robot.")
        return

    policy_device = get_safe_torch_device(policy_cfg.device)
    if policy_device.type != "cuda":
        raise ValueError(
            "ACT TensorRT real inference requires a CUDA policy device. "
            f"Resolved device: {policy_device}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    trt_policy = ActTrtPolicyAdapter(
        checkpoint=policy_path,
        engine_path=engine_path,
        config=policy_cfg,
        export_metadata_path=export_metadata,
        device=str(policy_device),
    )
    preprocessor, postprocessor = load_pre_post_processors(policy_path)
    spec = load_model_spec(policy_path)
    torch_policy = None
    torch_policy_device = None
    torch_preprocessor = None
    torch_postprocessor = None
    compare_metrics_all: list[dict[str, float]] = []

    if args.compare_torch:
        torch_policy = load_act_policy(policy_path, device=args.torch_policy_device)
        apply_act_policy_runtime_overrides(
            policy=torch_policy,
            policy_n_action_steps=args.policy_n_action_steps,
            policy_temporal_ensemble_coeff=args.policy_temporal_ensemble_coeff,
        )
        torch_policy_device = get_safe_torch_device(args.torch_policy_device)
        torch_policy.to(torch_policy_device)
        torch_policy.eval()
        torch_preprocessor, torch_postprocessor = load_pre_post_processors(policy_path)

    robot = None
    dataset_features = None
    robot_action_processor = None
    robot_observation_processor = None
    robot_connected_once = False
    robot_connection_attempted = False

    if not args.mock_observation:
        cameras = {
            "top": OpenCVCameraConfig(
                index_or_path=args.top_cam_index,
                width=args.img_width,
                height=args.img_height,
                fps=args.fps,
            ),
            "wrist": OpenCVCameraConfig(
                index_or_path=args.wrist_cam_index,
                width=args.img_width,
                height=args.img_height,
                fps=args.fps,
            ),
        }
        robot_cfg = SOFollowerRobotConfig(
            id=args.robot_id,
            calibration_dir=maybe_path(args.calib_dir),
            port=args.robot_port,
            cameras=cameras,
        )

        from lerobot.robots import make_robot_from_config

        robot = make_robot_from_config(robot_cfg)
        _, robot_action_processor, robot_observation_processor = make_default_processors()
        dataset_features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=robot_action_processor,
                initial_features=create_initial_features(action=robot.action_features),
                use_videos=True,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=robot_observation_processor,
                initial_features=create_initial_features(observation=robot.observation_features),
                use_videos=True,
            ),
        )

    step = 0
    setup_start_t = time.perf_counter()
    start_t: float | None = None
    end_t: float | None = None
    action_min: float | None = None
    action_max: float | None = None
    observation_times: list[float] = []
    policy_times: list[float] = []
    send_times: list[float] = []
    total_times: list[float] = []
    step_debug_samples: list[dict[str, Any]] = []

    try:
        trt_policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        if torch_policy is not None:
            torch_policy.reset()
        if torch_preprocessor is not None:
            torch_preprocessor.reset()
        if torch_postprocessor is not None:
            torch_postprocessor.reset()

        if robot is not None:
            robot_connection_attempted = True
            robot.connect()
            robot_connected_once = True

        start_t = time.perf_counter()
        end_t = start_t + args.run_time_s if args.run_time_s > 0 else None

        while True:
            if end_t is not None and time.perf_counter() >= end_t:
                print("[INFO] Reached requested run_time_s. Exiting inference loop.")
                break
            if args.max_steps > 0 and step >= args.max_steps:
                print("[INFO] Reached requested max_steps. Exiting inference loop.")
                break

            loop_t0 = time.perf_counter()
            obs = None

            if args.mock_observation:
                seed = args.seed * 1000 + step if args.mock_case == "random" else args.seed
                observation = make_mock_observation(spec=spec, seed=seed, case=args.mock_case)
                observation_frame = observation
            else:
                assert robot is not None
                assert robot_observation_processor is not None
                assert dataset_features is not None
                obs = robot.get_observation()
                obs_processed = robot_observation_processor(obs)
                observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)

            loop_t1 = time.perf_counter()
            action_values = predict_action(
                observation=observation_frame,
                policy=trt_policy,
                device=policy_device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=False,
                task=args.task,
                robot_type=args.robot_type,
            )
            loop_t2 = time.perf_counter()

            action_dict = None
            robot_action_to_send = None
            if not args.mock_observation:
                assert robot_action_processor is not None
                assert dataset_features is not None
                assert obs is not None
                action_dict = make_robot_action(action_values, dataset_features)
                robot_action_to_send = robot_action_processor((action_dict, obs))

            if not args.mock_observation and not args.mock_send:
                assert robot is not None
                assert robot_action_to_send is not None
                robot.send_action(robot_action_to_send)

            loop_t3 = time.perf_counter()

            observation_times.append(loop_t1 - loop_t0)
            policy_times.append(loop_t2 - loop_t1)
            send_times.append(loop_t3 - loop_t2)
            total_times.append(loop_t3 - loop_t0)

            action_tensor = torch.as_tensor(action_values).detach().cpu().float().reshape(-1)
            current_min = float(action_tensor.min().item())
            current_max = float(action_tensor.max().item())
            action_min = current_min if action_min is None else min(action_min, current_min)
            action_max = current_max if action_max is None else max(action_max, current_max)

            torch_action_tensor = None
            compare_metrics = None
            if (
                torch_policy is not None
                and torch_policy_device is not None
                and torch_preprocessor is not None
                and torch_postprocessor is not None
            ):
                torch_action_values = predict_action(
                    observation=observation_frame,
                    policy=torch_policy,
                    device=torch_policy_device,
                    preprocessor=torch_preprocessor,
                    postprocessor=torch_postprocessor,
                    use_amp=False,
                    task=args.task,
                    robot_type=args.robot_type,
                )
                torch_action_tensor = torch.as_tensor(torch_action_values).detach().cpu().float().reshape(-1)
                diff = torch.abs(torch_action_tensor - action_tensor)
                compare_metrics = {
                    "max_abs_diff": float(diff.max().item()),
                    "mean_abs_diff": float(diff.mean().item()),
                }
                compare_metrics_all.append(compare_metrics)

            if len(step_debug_samples) < 3:
                debug_item: dict[str, Any] = {
                    "step": int(step),
                    "action_tensor": [float(value) for value in action_tensor.tolist()],
                }
                if action_dict is not None:
                    debug_item["action_dict"] = {key: float(value) for key, value in action_dict.items()}
                if robot_action_to_send is not None:
                    debug_item["robot_action_to_send"] = {
                        key: float(value) for key, value in robot_action_to_send.items()
                    }
                if obs is not None:
                    debug_item["observation_joint_pos"] = {
                        key: float(value)
                        for key, value in obs.items()
                        if isinstance(value, (int, float, np.floating)) and key.endswith(".pos")
                    }
                if torch_action_tensor is not None:
                    debug_item["torch_action_tensor"] = [float(value) for value in torch_action_tensor.tolist()]
                if compare_metrics is not None:
                    debug_item["torch_vs_trt"] = compare_metrics
                step_debug_samples.append(debug_item)

            step += 1
            if args.log_interval > 0 and step % args.log_interval == 0:
                assert start_t is not None
                elapsed = time.perf_counter() - start_t
                avg_total = float(np.mean(total_times)) if total_times else 0.0
                print(f"[INFO] Step {step} | elapsed={elapsed:.2f}s | avg_total_s={avg_total:.4f}")

            precise_sleep(max(1 / float(args.fps) - (time.perf_counter() - loop_t0), 0.0))
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received. Stopping inference.")
    finally:
        if robot is not None and robot.is_connected:
            robot.disconnect()
        print("[INFO] Inference finished.")

    avg_observation_s, max_observation_s = summarize_timings(observation_times)
    avg_policy_s, max_policy_s = summarize_timings(policy_times)
    avg_send_s, max_send_s = summarize_timings(send_times)
    avg_total_s, max_total_s = summarize_timings(total_times)
    setup_duration_s = float((start_t if start_t is not None else time.perf_counter()) - setup_start_t)
    loop_elapsed_s = float(time.perf_counter() - start_t) if start_t is not None else 0.0

    execution_mode = "mock_observation" if args.mock_observation else "robot_loop"
    torch_vs_trt_summary = None
    if compare_metrics_all:
        torch_vs_trt_summary = {
            "max_abs_diff": float(max(item["max_abs_diff"] for item in compare_metrics_all)),
            "mean_abs_diff": float(max(item["mean_abs_diff"] for item in compare_metrics_all)),
        }
    payload = {
        "entrypoint": Path(__file__).name,
        "execution_mode": execution_mode,
        "mock_observation": bool(args.mock_observation),
        "mock_send": bool(args.mock_send),
        "action_send_enabled": bool((not args.mock_observation) and (not args.mock_send)),
        "compare_torch": bool(args.compare_torch),
        "torch_policy_device": str(torch_policy_device) if torch_policy_device is not None else None,
        "robot_connection_attempted": bool(robot_connection_attempted),
        "robot_connected_once": bool(robot_connected_once),
        "checkpoint": str(policy_path),
        "engine": str(engine_path),
        "export_metadata": str(export_metadata) if export_metadata is not None else None,
        "policy_device": str(policy_device),
        "task": args.task,
        "fps": int(args.fps),
        "run_time_s": float(args.run_time_s),
        "max_steps": int(args.max_steps),
        "mock_case": args.mock_case if args.mock_observation else None,
        "seed": int(args.seed),
        "policy_config": {
            "chunk_size": int(policy_cfg.chunk_size),
            "n_action_steps": int(policy_cfg.n_action_steps),
            "temporal_ensemble_coeff": policy_cfg.temporal_ensemble_coeff,
            "image_features": list(getattr(policy_cfg, "image_features", []) or []),
        },
        "summary": {
            "steps_executed": int(step),
            "setup_duration_s": setup_duration_s,
            "loop_elapsed_s": loop_elapsed_s,
            "avg_observation_s": avg_observation_s,
            "max_observation_s": max_observation_s,
            "avg_policy_s": avg_policy_s,
            "max_policy_s": max_policy_s,
            "avg_send_s": avg_send_s,
            "max_send_s": max_send_s,
            "avg_total_s": avg_total_s,
            "max_total_s": max_total_s,
            "action_min": action_min,
            "action_max": action_max,
        },
        "torch_vs_trt_summary": torch_vs_trt_summary,
        "step_debug_samples": step_debug_samples,
    }
    save_json(report_path, payload)
    print(report_path)
    print(payload["summary"])


if __name__ == "__main__":
    main()
