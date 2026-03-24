#!/usr/bin/env python

"""Run Agilex ACT observation checks, shadow inference, and closed-loop inference.

This script keeps the implementation close to the repository's standard LeRobot
inference path:
1) read robot observation
2) optionally run policy inference
3) optionally send action to the robot

Compared with the SO101 ACT helper, the main differences are:
- replace the robot construction with `AgileXRobotConfig`
- validate the current Agilex ACT checkpoint schema before execution
- make `passive_follow` vs `command_master` explicit
- support an `observation_only` mode for stage-2 readiness checks

Examples:
python my_devs/train/act/agilex/run_act_infer.py --dry-run true

python my_devs/train/act/agilex/run_act_infer.py \
    --execution-mode observation_only \
    --run-time-s 10

# 模拟模式不下发真实指令
python my_devs/train/act/agilex/run_act_infer.py \
    --execution-mode policy_inference \
    --control-mode passive_follow \
    --policy-n-action-steps 100 \
    --run-time-s 12 \
    --policy-path 

python my_devs/train/act/agilex/run_act_infer.py \
      --execution-mode policy_inference \
      --control-mode command_master \
      --policy-n-action-steps 100 \
      --run-time-s 8
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from lerobot import policies  # noqa: F401  # Register policy config classes.
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.policies.factory import get_policy_class
from lerobot.policies.utils import make_robot_action
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, make_default_processors
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.processor.device_processor import DeviceProcessorStep
from lerobot.robots import make_robot_from_config
from lerobot.robots.agilex import AgileXRobotConfig
from lerobot.robots.agilex.agilex_ros_bridge import ACTION_FEATURE_NAMES, CAMERA_KEYS, POSITION_FEATURE_NAMES
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import auto_select_torch_device, get_safe_torch_device

DEFAULT_POLICY_PATH = (
    "/home/agilex/cqy/flexible_lerobot/outputs/train/"
    "20260313_194500_act_agilex_first_test_full/checkpoints/100000/pretrained_model"
)
EXPECTED_STATE_KEY = "observation.state"
EXPECTED_ACTION_KEY = "action"
EXPECTED_VISUAL_KEYS = tuple(f"observation.images.{camera_key}" for camera_key in CAMERA_KEYS)
DEFAULT_TASK = "Execute the trained Agilex ACT task"


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


def resolve_policy_device(requested: str | None, config_device: str | None) -> str:
    if requested is not None and requested.strip().lower() in {"", "none", "null"}:
        requested = None
    if config_device is None or str(config_device).lower() in {"", "none", "null"}:
        config_device = auto_select_torch_device().type
    if requested is None:
        return str(config_device)
    if requested == "auto":
        return auto_select_torch_device().type
    return requested


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agilex ACT observation check, shadow inference, and closed-loop inference."
    )
    parser.add_argument("--robot-id", default=os.getenv("ROBOT_ID", "my_agilex"))
    parser.add_argument(
        "--execution-mode",
        choices=("observation_only", "policy_inference"),
        default=os.getenv("EXECUTION_MODE", "policy_inference"),
        help=(
            "observation_only only checks live state/images; "
            "policy_inference runs ACT and optionally publishes actions."
        ),
    )
    parser.add_argument(
        "--control-mode",
        choices=("passive_follow", "command_master"),
        default=os.getenv("CONTROL_MODE", "passive_follow"),
        help="passive_follow keeps the loop read-only; command_master publishes real robot commands.",
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
        help="Timeout for waiting on state and image topics during connect().",
    )
    parser.add_argument("--queue-size", type=int, default=int(os.getenv("QUEUE_SIZE", "1")))
    parser.add_argument("--fps", type=int, default=int(os.getenv("FPS", "30")))

    parser.add_argument("--policy-path", default=os.getenv("POLICY_PATH", DEFAULT_POLICY_PATH))
    parser.add_argument(
        "--policy-device",
        default=os.getenv("POLICY_DEVICE_OVERRIDE", os.getenv("POLICY_DEVICE")),
        help="Override checkpoint device with one of cpu/cuda/mps/xpu/auto.",
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
        default=os.getenv("DATASET_TASK", DEFAULT_TASK),
        help="Language instruction passed to policy inference.",
    )
    parser.add_argument(
        "--run-time-s",
        type=float,
        default=float(os.getenv("RUN_TIME_S", "0")),
        help="Total runtime in seconds. <=0 means run until Ctrl+C.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=int(os.getenv("LOG_INTERVAL", "30")),
        help="Print status every N loop iterations. Set 0 to disable periodic logs.",
    )
    parser.add_argument(
        "--dry-run",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("DRY_RUN", False),
        help="Print resolved config and exit without connecting robot or loading model weights.",
    )
    return parser


def ensure_checkpoint_assets(policy_path: Path, *, require_weights: bool) -> None:
    required_files = [
        policy_path / "config.json",
        policy_path / "policy_preprocessor.json",
        policy_path / "policy_postprocessor.json",
    ]
    if require_weights:
        required_files.append(policy_path / "model.safetensors")

    missing_files = [path for path in required_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing checkpoint assets: {missing_files}")


def load_pre_post_processors(
    policy_path: Path,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(policy_path),
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(policy_path),
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


def sync_processor_device(pipeline: PolicyProcessorPipeline[Any, Any], device: str) -> None:
    for step in pipeline.steps:
        if isinstance(step, DeviceProcessorStep):
            step.device = device
            step.__post_init__()


def apply_act_runtime_overrides(
    policy_cfg: PreTrainedConfig,
    policy_n_action_steps: int | None,
    policy_temporal_ensemble_coeff: float | None,
) -> None:
    if policy_cfg.type != "act":
        raise ValueError(f"Expected ACT policy, got {policy_cfg.type!r}")

    chunk_size = int(policy_cfg.chunk_size)

    if policy_n_action_steps is not None:
        if not 1 <= policy_n_action_steps <= chunk_size:
            raise ValueError(
                f"--policy-n-action-steps must be within [1, {chunk_size}], got {policy_n_action_steps}"
            )
        policy_cfg.n_action_steps = policy_n_action_steps

    if policy_temporal_ensemble_coeff is not None:
        if policy_cfg.n_action_steps != 1:
            raise ValueError(
                "ACT temporal ensembling requires n_action_steps == 1. "
                f"Current value: {policy_cfg.n_action_steps}"
            )
        policy_cfg.temporal_ensemble_coeff = policy_temporal_ensemble_coeff


def feature_shape(feature: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in getattr(feature, "shape", ()))


def validate_checkpoint_schema(policy_cfg: PreTrainedConfig) -> tuple[int, int]:
    expected_input_shapes = {
        EXPECTED_STATE_KEY: (14,),
        "observation.images.camera_front": (3, 480, 640),
        "observation.images.camera_left": (3, 480, 640),
        "observation.images.camera_right": (3, 480, 640),
    }
    expected_output_shapes = {EXPECTED_ACTION_KEY: (14,)}

    actual_input_keys = set(policy_cfg.input_features)
    actual_output_keys = set(policy_cfg.output_features)
    missing_inputs = sorted(set(expected_input_shapes) - actual_input_keys)
    extra_inputs = sorted(actual_input_keys - set(expected_input_shapes))
    missing_outputs = sorted(set(expected_output_shapes) - actual_output_keys)
    extra_outputs = sorted(actual_output_keys - set(expected_output_shapes))
    if missing_inputs or extra_inputs or missing_outputs or extra_outputs:
        raise ValueError(
            "Checkpoint schema does not match the current Agilex deployment contract. "
            f"missing_inputs={missing_inputs}, extra_inputs={extra_inputs}, "
            f"missing_outputs={missing_outputs}, extra_outputs={extra_outputs}"
        )

    for key, expected_shape in expected_input_shapes.items():
        actual_shape = feature_shape(policy_cfg.input_features[key])
        if actual_shape != expected_shape:
            raise ValueError(f"Unexpected shape for {key}: expected {expected_shape}, got {actual_shape}")

    for key, expected_shape in expected_output_shapes.items():
        actual_shape = feature_shape(policy_cfg.output_features[key])
        if actual_shape != expected_shape:
            raise ValueError(f"Unexpected shape for {key}: expected {expected_shape}, got {actual_shape}")

    _, image_height, image_width = expected_input_shapes["observation.images.camera_front"]
    return image_height, image_width


def build_robot_config(args: argparse.Namespace, image_height: int, image_width: int) -> AgileXRobotConfig:
    return AgileXRobotConfig(
        id=args.robot_id,
        control_mode=args.control_mode,
        state_left_topic=args.state_left_topic,
        state_right_topic=args.state_right_topic,
        command_left_topic=args.command_left_topic,
        command_right_topic=args.command_right_topic,
        front_camera_topic=args.front_camera_topic,
        left_camera_topic=args.left_camera_topic,
        right_camera_topic=args.right_camera_topic,
        image_height=image_height,
        image_width=image_width,
        observation_timeout_s=args.observation_timeout_s,
        queue_size=args.queue_size,
    )


def validate_execution_mode(args: argparse.Namespace) -> None:
    if args.execution_mode == "observation_only" and args.control_mode == "command_master":
        raise ValueError(
            "--execution-mode=observation_only is read-only. Use --control-mode=passive_follow for that mode."
        )


def validate_live_observation(observation: dict[str, Any], image_height: int, image_width: int) -> None:
    required_keys = [*POSITION_FEATURE_NAMES, *CAMERA_KEYS]
    missing_keys = [key for key in required_keys if key not in observation]
    if missing_keys:
        raise KeyError(f"Missing Agilex observation keys: {missing_keys}")

    for key in POSITION_FEATURE_NAMES:
        value = observation[key]
        if not np.isscalar(value):
            raise TypeError(f"Expected scalar joint value for {key}, got {type(value)}")

    expected_image_shape = (image_height, image_width, 3)
    for key in CAMERA_KEYS:
        value = observation[key]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray image for {key}, got {type(value)}")
        if value.shape != expected_image_shape:
            raise ValueError(
                f"Unexpected live image shape for {key}: "
                f"expected {expected_image_shape}, got {value.shape}"
            )


def summarize_observation(observation: dict[str, Any]) -> str:
    state_vector = np.asarray([float(observation[key]) for key in POSITION_FEATURE_NAMES], dtype=np.float32)
    image_summary = ", ".join(
        f"{key}={observation[key].shape}/{observation[key].dtype}" for key in CAMERA_KEYS if key in observation
    )
    return (
        f"state[min={state_vector.min():.4f}, max={state_vector.max():.4f}, mean={state_vector.mean():.4f}] | "
        f"images[{image_summary}]"
    )


def summarize_robot_action(action: dict[str, Any]) -> str:
    action_vector = np.asarray([float(action[key]) for key in ACTION_FEATURE_NAMES], dtype=np.float32)
    return (
        f"action[min={action_vector.min():.4f}, max={action_vector.max():.4f}, "
        f"mean={action_vector.mean():.4f}, std={action_vector.std():.4f}]"
    )


def print_runtime_summary(
    args: argparse.Namespace,
    policy_path: Path,
    policy_cfg: PreTrainedConfig,
    image_height: int,
    image_width: int,
) -> None:
    print(f"[INFO] Robot id: {args.robot_id}")
    print(f"[INFO] Execution mode: {args.execution_mode}")
    print(f"[INFO] Control mode: {args.control_mode}")
    print(f"[INFO] Policy path: {policy_path}")
    print(f"[INFO] Policy type: {policy_cfg.type}")
    print(f"[INFO] Policy device: {policy_cfg.device}")
    print(f"[INFO] Task: {args.task}")
    print(f"[INFO] FPS: {args.fps}")
    print(f"[INFO] run_time_s: {args.run_time_s} (<=0 means until Ctrl+C)")
    print(
        "[INFO] ACT runtime config: "
        f"chunk_size={policy_cfg.chunk_size}, "
        f"n_action_steps={policy_cfg.n_action_steps}, "
        f"temporal_ensemble_coeff={policy_cfg.temporal_ensemble_coeff}"
    )
    print("[INFO] Input features:")
    print(f"[INFO]   {EXPECTED_STATE_KEY}: {feature_shape(policy_cfg.input_features[EXPECTED_STATE_KEY])}")
    for key in EXPECTED_VISUAL_KEYS:
        print(f"[INFO]   {key}: {feature_shape(policy_cfg.input_features[key])}")
    print(
        f"[INFO] Output features: {EXPECTED_ACTION_KEY}: "
        f"{feature_shape(policy_cfg.output_features[EXPECTED_ACTION_KEY])}"
    )
    print(f"[INFO] Expected live image shape (HWC): ({image_height}, {image_width}, 3)")
    print(f"[INFO] State topics: left={args.state_left_topic} right={args.state_right_topic}")
    print(f"[INFO] Command topics: left={args.command_left_topic} right={args.command_right_topic}")
    print(
        "[INFO] Camera topics: "
        f"front={args.front_camera_topic} left={args.left_camera_topic} right={args.right_camera_topic}"
    )
    print(f"[INFO] observation_timeout_s: {args.observation_timeout_s}")
    print(f"[INFO] queue_size: {args.queue_size}")


def main() -> None:
    register_third_party_plugins()
    args = build_parser().parse_args()
    validate_execution_mode(args)

    policy_path = Path(args.policy_path).expanduser()
    if not policy_path.is_dir():
        raise FileNotFoundError(f"Policy path does not exist: {policy_path}")

    ensure_checkpoint_assets(
        policy_path,
        require_weights=args.execution_mode == "policy_inference" and not args.dry_run,
    )

    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    image_height, image_width = validate_checkpoint_schema(policy_cfg)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = resolve_policy_device(args.policy_device, policy_cfg.device)
    apply_act_runtime_overrides(
        policy_cfg=policy_cfg,
        policy_n_action_steps=args.policy_n_action_steps,
        policy_temporal_ensemble_coeff=args.policy_temporal_ensemble_coeff,
    )

    print_runtime_summary(args, policy_path, policy_cfg, image_height, image_width)
    if args.dry_run:
        print("[INFO] DRY_RUN=true, exit without execution.")
        return

    robot_cfg = build_robot_config(args, image_height=image_height, image_width=image_width)
    robot = make_robot_from_config(robot_cfg)

    policy = None
    preprocessor = None
    postprocessor = None
    robot_action_processor = None
    robot_observation_processor = None
    dataset_features = None
    if args.execution_mode == "policy_inference":
        policy_class = get_policy_class(policy_cfg.type)
        policy = policy_class.from_pretrained(str(policy_path), config=policy_cfg, strict=False)
        policy.config.device = policy_cfg.device
        policy.to(policy_cfg.device)

        preprocessor, postprocessor = load_pre_post_processors(policy_path)
        sync_processor_device(preprocessor, policy_cfg.device)
        sync_processor_device(postprocessor, "cpu")

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
    start_t = time.perf_counter()
    end_t = start_t + args.run_time_s if args.run_time_s > 0 else None
    first_observation_logged = False
    publish_enabled = args.execution_mode == "policy_inference" and args.control_mode == "command_master"

    try:
        robot.connect()
        print(f"[INFO] Agilex connected. publish_enabled={publish_enabled}")

        if policy is not None and preprocessor is not None and postprocessor is not None:
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()

        while True:
            if end_t is not None and time.perf_counter() >= end_t:
                print("[INFO] Reached requested run_time_s. Exiting loop.")
                break

            loop_t = time.perf_counter()
            observation = robot.get_observation()
            validate_live_observation(observation, image_height=image_height, image_width=image_width)

            step += 1
            if not first_observation_logged:
                print(f"[INFO] First observation summary: {summarize_observation(observation)}")
                first_observation_logged = True

            if args.execution_mode == "observation_only":
                if args.log_interval > 0 and step % args.log_interval == 0:
                    elapsed = time.perf_counter() - start_t
                    print(
                        f"[INFO] Step {step} | elapsed={elapsed:.2f}s | "
                        f"{summarize_observation(observation)}"
                    )
            else:
                assert policy is not None
                assert preprocessor is not None
                assert postprocessor is not None
                assert robot_action_processor is not None
                assert robot_observation_processor is not None
                assert dataset_features is not None

                obs_processed = robot_observation_processor(observation)
                observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)

                action_values = predict_action(
                    observation=observation_frame,
                    policy=policy,
                    device=get_safe_torch_device(policy.config.device),
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=policy.config.use_amp,
                    task=args.task,
                    robot_type=robot.robot_type,
                )
                action_dict = make_robot_action(action_values, dataset_features)
                robot_action_to_send = robot_action_processor((action_dict, observation))
                sent_action = robot.send_action(robot_action_to_send)

                if args.log_interval > 0 and step % args.log_interval == 0:
                    elapsed = time.perf_counter() - start_t
                    mode_label = "closed_loop" if publish_enabled else "shadow_infer"
                    print(
                        f"[INFO] Step {step} | elapsed={elapsed:.2f}s | mode={mode_label} | "
                        f"{summarize_robot_action(sent_action)}"
                    )

            dt_s = time.perf_counter() - loop_t
            precise_sleep(max(1 / args.fps - dt_s, 0.0))
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received. Stopping inference.")
    finally:
        if robot.is_connected:
            robot.disconnect()
        print("[INFO] Agilex inference finished.")


if __name__ == "__main__":
    main()
