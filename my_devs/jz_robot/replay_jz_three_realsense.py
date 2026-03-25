#!/usr/bin/env python

from __future__ import annotations

import argparse

from lerobot.scripts.lerobot_replay import DatasetReplayConfig, ReplayConfig, replay

from my_devs.jz_robot.common import (
    DEFAULT_ROBOT_CONFIG,
    apply_common_robot_overrides,
    load_robot_config,
    maybe_path,
    parse_bool,
    summarize_robot_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay a recorded JZRobot episode. Cameras are disabled by default during replay."
    )

    parser.add_argument("--robot-config", default=str(DEFAULT_ROBOT_CONFIG))
    parser.add_argument("--robot-id", default=None)

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
        default=False,
        help="Default false for replay so LeRobot actively publishes the recorded actions.",
    )

    parser.add_argument("--img-width", type=int, default=None)
    parser.add_argument("--img-height", type=int, default=None)
    parser.add_argument("--camera-fps", type=int, default=None)
    parser.add_argument("--warmup-s", type=int, default=None)
    parser.add_argument(
        "--connect-cameras",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
        help="Default false. Replay only needs robot state + command channels, not live cameras.",
    )

    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--play-sounds", type=parse_bool, nargs="?", const=True, default=False)

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
    )

    if not args.connect_cameras:
        robot_cfg.cameras = {}

    dataset_cfg = DatasetReplayConfig(
        repo_id=args.dataset_repo_id,
        root=maybe_path(args.dataset_root),
        episode=args.episode,
        fps=args.fps,
    )
    cfg = ReplayConfig(robot=robot_cfg, dataset=dataset_cfg, play_sounds=args.play_sounds)

    print("[INFO] JZRobot replay config")
    print(summarize_robot_config(robot_cfg))
    print(
        "[INFO] dataset: "
        f"repo_id={args.dataset_repo_id}, root={maybe_path(args.dataset_root)}, "
        f"episode={args.episode}, fps={args.fps}, connect_cameras={args.connect_cameras}"
    )

    replay(cfg)


if __name__ == "__main__":
    main()
