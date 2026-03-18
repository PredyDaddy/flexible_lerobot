#!/usr/bin/env python

"""Record with a LeRobot ACT checkpoint while replacing Torch policy forward with ONNXRuntime."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import onnxruntime as ort

from lerobot import policies  # noqa: F401  # Register policy config classes.
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.scripts.lerobot_record import DatasetRecordConfig, record_loop
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun

from my_devs.act_trt.ort_policy import ActOrtPolicyAdapter


def resolve_ort_providers(provider: str, policy_device: str) -> list[str]:
    available = ort.get_available_providers()
    if provider == "cpu":
        return ["CPUExecutionProvider"]
    if provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise ValueError(
                "Requested --onnx.provider=cuda but CUDAExecutionProvider is unavailable. "
                f"Available providers: {available}"
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if str(policy_device).startswith("cuda") and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


@dataclass
class OnnxRuntimeConfig:
    path: str | Path
    metadata_path: str | Path | None = None
    provider: str = "auto"


@dataclass
class OnnxRecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    onnx: OnnxRuntimeConfig
    policy: PreTrainedConfig | None = None
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    play_sounds: bool = True
    resume: bool = False
    dry_run: bool = False

    def __post_init__(self) -> None:
        policy_path = parser.get_path_arg("policy")
        if not policy_path:
            raise ValueError("ONNX record requires `--policy.path=<pretrained_model_dir>`.")

        resolved_policy_path = Path(policy_path).expanduser().resolve()
        cli_overrides = parser.get_cli_overrides("policy")
        self.policy = PreTrainedConfig.from_pretrained(str(resolved_policy_path), cli_overrides=cli_overrides)
        self.policy.pretrained_path = str(resolved_policy_path)

        if self.policy.type != "act":
            raise ValueError(f"This script only supports ACT checkpoints, got policy type `{self.policy.type}`.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


@parser.wrap()
def record_onnx(cfg: OnnxRecordConfig) -> LeRobotDataset | None:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.policy is None or cfg.policy.pretrained_path is None:
        raise ValueError("Missing resolved policy config. Pass `--policy.path=<pretrained_model_dir>`.")

    policy_path = Path(cfg.policy.pretrained_path).expanduser().resolve()
    onnx_path = Path(cfg.onnx.path).expanduser().resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    metadata_path = Path(cfg.onnx.metadata_path).expanduser().resolve() if cfg.onnx.metadata_path else None
    if metadata_path is not None and not metadata_path.is_file():
        raise FileNotFoundError(f"ONNX metadata file not found: {metadata_path}")

    ort_providers = resolve_ort_providers(cfg.onnx.provider, str(cfg.policy.device))
    onnx_policy = ActOrtPolicyAdapter(
        checkpoint=policy_path,
        onnx_path=onnx_path,
        config=cfg.policy,
        export_metadata_path=metadata_path,
        providers=ort_providers,
    )
    logging.info("Loaded ONNX policy from %s", onnx_path)
    logging.info("ONNX providers: %s", onnx_policy.session.get_providers())
    logging.info("ONNX metadata: %s", metadata_path)
    logging.info("Policy device for pre/post processing: %s", cfg.policy.device)

    if cfg.dry_run:
        logging.info("Dry run enabled. Exiting before robot, dataset, and record loop initialization.")
        return None

    if cfg.display_data:
        init_rerun(session_name="recording_act_onnx", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    dataset = None
    listener = None

    try:
        if cfg.resume:
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
            )
            if hasattr(robot, "cameras") and len(robot.cameras) > 0:
                dataset.start_image_writer(
                    num_processes=cfg.dataset.num_image_writer_processes,
                    num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
            )

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
                "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
            },
        )

        robot.connect()
        listener, events = init_keyboard_listener()

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=None,
                    policy=onnx_policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_compressed_images=display_compressed_images,
                )

                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", cfg.play_sounds)
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=None,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                        display_compressed_images=display_compressed_images,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1
    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)

        if dataset is not None:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()

        if not is_headless() and listener is not None:
            listener.stop()

        if cfg.dataset.push_to_hub:
            if dataset is not None:
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
            else:
                logging.warning(
                    "Skip push_to_hub because dataset is not initialized. Check earlier errors in the log."
                )

        log_say("Exiting", cfg.play_sounds)

    return dataset


def main() -> None:
    register_third_party_plugins()
    record_onnx()


if __name__ == "__main__":
    main()
