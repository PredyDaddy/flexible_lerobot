#!/usr/bin/env python

from __future__ import annotations

import argparse

from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig, record

from my_devs.jz_robot.common import (
    DEFAULT_ROBOT_CONFIG,
    apply_common_robot_overrides,
    build_jz_command_teleop_config,
    load_robot_config,
    maybe_path,
    parse_bool,
    summarize_robot_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record JZRobot dual-arm data with three ROS2 RealSense image topics via external teleoperation."
    )

    parser.add_argument("--robot-config", default=str(DEFAULT_ROBOT_CONFIG))
    parser.add_argument("--robot-id", default=None)
    parser.add_argument("--teleop-id", default="jz_command_teleop")

    parser.add_argument("--left-joint-state-topic", default=None)
    parser.add_argument("--right-joint-state-topic", default=None)
    parser.add_argument("--left-command-topic", default=None)
    parser.add_argument("--right-command-topic", default=None)
    parser.add_argument("--use-gripper", type=parse_bool, nargs="?", const=True, default=None)
    parser.add_argument("--left-gripper-state-topic", default=None)
    parser.add_argument("--right-gripper-state-topic", default=None)
    parser.add_argument("--left-gripper-command-topic", default=None)
    parser.add_argument("--right-gripper-command-topic", default=None)
    parser.add_argument(
        "--init-state-timeout-s",
        type=float,
        default=None,
        help="Set to 0 or a negative value to wait forever for the first robot state.",
    )
    parser.add_argument("--state-timeout-s", type=float, default=None)
    parser.add_argument("--qos-depth", type=int, default=None)
    parser.add_argument(
        "--use-external-commands",
        type=parse_bool,
        nargs="?",
        const=True,
        default=True,
        help="Default true. Keep true when another teleop stack is already publishing command topics.",
    )

    parser.add_argument("--img-width", type=int, default=None)
    parser.add_argument("--img-height", type=int, default=None)
    parser.add_argument("--camera-fps", type=int, default=None)
    parser.add_argument("--warmup-s", type=int, default=None)
    parser.add_argument("--head-image-topic", default=None)
    parser.add_argument("--left-image-topic", default=None)
    parser.add_argument("--right-image-topic", default=None)
    parser.add_argument("--camera-timeout-ms", type=int, default=None)

    parser.add_argument(
        "--teleop-connect-timeout-s",
        type=float,
        default=0.0,
        help="Set to 0 or a negative value to wait forever for the first external command.",
    )
    parser.add_argument("--teleop-command-timeout-s", type=float, default=None)

    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--dataset-task", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--episode-time-s", type=float, default=60.0)
    parser.add_argument("--reset-time-s", type=float, default=20.0)
    parser.add_argument("--video", type=parse_bool, nargs="?", const=True, default=True)
    parser.add_argument("--push-to-hub", type=parse_bool, nargs="?", const=True, default=False)
    parser.add_argument("--private", type=parse_bool, nargs="?", const=True, default=False)
    parser.add_argument("--display-data", type=parse_bool, nargs="?", const=True, default=False)
    parser.add_argument("--play-sounds", type=parse_bool, nargs="?", const=True, default=False)
    parser.add_argument("--resume", type=parse_bool, nargs="?", const=True, default=False)
    parser.add_argument("--num-image-writer-processes", type=int, default=0)
    parser.add_argument("--num-image-writer-threads-per-camera", type=int, default=4)
    parser.add_argument("--video-encoding-batch-size", type=int, default=1)
    parser.add_argument("--vcodec", default="libsvtav1")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    robot_cfg = load_robot_config(args.robot_config)
    robot_cfg = apply_common_robot_overrides(
        robot_cfg,
        robot_id=args.robot_id,
        left_joint_state_topic=args.left_joint_state_topic,
        right_joint_state_topic=args.right_joint_state_topic,
        left_command_topic=args.left_command_topic,
        right_command_topic=args.right_command_topic,
        use_gripper=args.use_gripper,
        left_gripper_state_topic=args.left_gripper_state_topic,
        right_gripper_state_topic=args.right_gripper_state_topic,
        left_gripper_command_topic=args.left_gripper_command_topic,
        right_gripper_command_topic=args.right_gripper_command_topic,
        init_state_timeout_s=args.init_state_timeout_s,
        state_timeout_s=args.state_timeout_s,
        qos_depth=args.qos_depth,
        use_external_commands=args.use_external_commands,
        img_width=args.img_width,
        img_height=args.img_height,
        camera_fps=args.camera_fps,
        warmup_s=args.warmup_s,
        head_image_topic=args.head_image_topic,
        left_image_topic=args.left_image_topic,
        right_image_topic=args.right_image_topic,
        camera_timeout_ms=args.camera_timeout_ms,
    )

    teleop_cfg = build_jz_command_teleop_config(
        robot_cfg,
        teleop_id=args.teleop_id,
        connect_timeout_s=args.teleop_connect_timeout_s,
        command_timeout_s=args.teleop_command_timeout_s,
    )

    dataset_cfg = DatasetRecordConfig(
        repo_id=args.dataset_repo_id,
        root=maybe_path(args.dataset_root),
        single_task=args.dataset_task,
        fps=args.fps,
        episode_time_s=args.episode_time_s,
        reset_time_s=args.reset_time_s,
        num_episodes=args.num_episodes,
        video=args.video,
        push_to_hub=args.push_to_hub,
        private=args.private,
        num_image_writer_processes=args.num_image_writer_processes,
        num_image_writer_threads_per_camera=args.num_image_writer_threads_per_camera,
        video_encoding_batch_size=args.video_encoding_batch_size,
        vcodec=args.vcodec,
    )

    cfg = RecordConfig(
        robot=robot_cfg,
        teleop=teleop_cfg,
        dataset=dataset_cfg,
        display_data=args.display_data,
        play_sounds=args.play_sounds,
        resume=args.resume,
    )

    print("[INFO] JZRobot three-ROS2-RealSense record config")
    print(summarize_robot_config(robot_cfg))
    print(
        "[INFO] dataset: "
        f"repo_id={args.dataset_repo_id}, root={maybe_path(args.dataset_root)}, "
        f"episodes={args.num_episodes}, episode_time_s={args.episode_time_s}, reset_time_s={args.reset_time_s}, "
        f"fps={args.fps}, video={args.video}, push_to_hub={args.push_to_hub}"
    )
    print(
        "[INFO] teleop: "
        f"left={teleop_cfg.left_command_topic}, right={teleop_cfg.right_command_topic}, "
        f"left_gripper={teleop_cfg.left_gripper_command_topic if teleop_cfg.use_gripper else 'disabled'}, "
        f"right_gripper={teleop_cfg.right_gripper_command_topic if teleop_cfg.use_gripper else 'disabled'}, "
        f"connect_timeout_s={teleop_cfg.connect_timeout_s}, command_timeout_s={teleop_cfg.command_timeout_s}"
    )

    record(cfg)


if __name__ == "__main__":
    main()
