#!/usr/bin/env bash
set -euo pipefail

# Run this script directly on the robot workstation.
# LeRobot's live visualization is enabled with --display_data=true.
# When run locally with a desktop session, Rerun will open a viewer that shows
# camera observations plus action/state streams while recording.

DATASET_NAME="${DATASET_NAME:-jz_robot_udp_vr_record_with_cameras_003}"
DATASET_ROOT="${DATASET_ROOT:-tests/outputs/${DATASET_NAME}}"

PYTHONPATH=src conda run --no-capture-output -n lerobot_flex python -m lerobot.scripts.lerobot_record \
    --robot.type=jz_robot_udp \
    --robot.send_action_transport=udp \
    --robot.send_action_execution=dry_run \
    --robot.command_target_ip=192.168.1.81 \
    --robot.command_target_port=39020 \
    --robot.allowed_sender_ip=192.168.1.81 \
    --robot.rtsp_cameras='{
      "camera_head": {
        "url": "rtsp://192.168.1.81:8554/robot_camera/camera_head",
        "fps": 30,
        "width": 1280,
        "height": 720,
        "timeout_ms": 5000,
        "warmup_frames": 1,
        "color_mode": "rgb",
        "transport": "tcp"
      },
      "camera_left": {
        "url": "rtsp://192.168.1.81:8554/robot_camera/camera_left",
        "fps": 30,
        "width": 640,
        "height": 480,
        "timeout_ms": 5000,
        "warmup_frames": 1,
        "color_mode": "rgb",
        "transport": "tcp"
      },
      "camera_right": {
        "url": "rtsp://192.168.1.81:8554/robot_camera/camera_right",
        "fps": 30,
        "width": 640,
        "height": 480,
        "timeout_ms": 5000,
        "warmup_frames": 1,
        "color_mode": "rgb",
        "transport": "tcp"
      }
    }' \
    --teleop.type=jz_robot_udp_hold \
    --dataset.repo_id="local/${DATASET_NAME}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=10 \
    --dataset.reset_time_s=0 \
    --dataset.fps=30 \
    --dataset.single_task="vr teleop observation camera record 10s 30fps test" \
    --dataset.push_to_hub=false \
    --dataset.video=true \
    --dataset.vcodec=h264 \
    --display_data=true \
    --display_compressed_images=false \
    --play_sounds=true
