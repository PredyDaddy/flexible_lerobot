#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, TextIO

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig
from lerobot.robots.jz_robot_pin_timed import JZRobotPinTimed, JZRobotPinTimedConfig

CAMERA_SPECS = {
    "head": (5555, 1280, 720),
    "left": (5556, 640, 480),
    "right": (5557, 640, 480),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read JZ Pin UDP state and direct timestamped ZMQ observations without sending actions."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--orin-ip", default="192.168.1.81")
    parser.add_argument("--bind-ip", default="0.0.0.0")
    parser.add_argument("--state-port", type=int, default=39010)
    parser.add_argument("--connect-timeout-s", type=float, default=30.0)
    parser.add_argument("--state-timeout-s", type=float, default=1.0)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--camera", action="append", choices=sorted(CAMERA_SPECS))
    parser.add_argument("--state-only", action="store_true")
    parser.add_argument("--camera-buffer-size", type=int, default=8)
    parser.add_argument("--camera-timeout-ms", type=int, default=5000)
    parser.add_argument("--camera-stale-frame-timeout-ms", type=int, default=1000)
    parser.add_argument("--max-camera-state-receive-skew-ms", type=float, default=100.0)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> list[str]:
    if args.duration_s <= 0:
        parser.error("--duration-s must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.state_port <= 0 or args.state_port > 65535:
        parser.error("--state-port must be in 1..65535")
    if args.state_only and args.camera:
        parser.error("--state-only cannot be combined with --camera")
    return [] if args.state_only else (args.camera or list(CAMERA_SPECS))


def make_camera_configs(args: argparse.Namespace, camera_names: list[str]) -> dict[str, ZMQCameraConfig]:
    configs = {}
    for name in camera_names:
        port, width, height = CAMERA_SPECS[name]
        configs[f"camera_{name}"] = ZMQCameraConfig(
            server_address=args.orin_ip,
            port=port,
            camera_name=f"camera_{name}",
            fps=30,
            width=width,
            height=height,
            timeout_ms=args.camera_timeout_ms,
            color_mode=ColorMode.RGB,
        )
    return configs


def emit(payload: dict[str, Any], output: TextIO | None) -> None:
    line = json.dumps(payload, sort_keys=True)
    print(line, flush=True)
    if output is not None:
        output.write(line + "\n")
        output.flush()


def empty_camera_stats() -> dict[str, int | float]:
    return {
        "samples": 0,
        "reused_samples": 0,
        "skipped_decoder_sequences": 0,
        "missing_pts_samples": 0,
        "over_skew_limit_samples": 0,
        "age_ms_total": 0.0,
        "age_ms_max": 0.0,
        "state_receive_skew_ms_total": 0.0,
        "state_receive_skew_ms_max": 0.0,
    }


def update_camera_stats(
    stats: dict[str, int | float], timing: dict[str, Any], previous_sequence: int | None, skew_limit_ms: float
) -> int:
    is_zmq = timing.get("protocol") == "jz_realsense_zmq"
    sequence = int(timing["sequence"] if is_zmq else timing["decoder_sequence"])
    age_ms = float(timing["age_ms"])
    skew_ms = float(timing["state_receive_skew_ms"])
    stats["samples"] += 1
    stats["reused_samples"] += int(bool(timing["reused_by_observation_loop"]))
    stats["missing_pts_samples"] += int(not is_zmq and timing["decoder_pts_ns"] is None)
    stats["over_skew_limit_samples"] += int(skew_ms > skew_limit_ms)
    stats["age_ms_total"] += age_ms
    stats["age_ms_max"] = max(float(stats["age_ms_max"]), age_ms)
    stats["state_receive_skew_ms_total"] += skew_ms
    stats["state_receive_skew_ms_max"] = max(float(stats["state_receive_skew_ms_max"]), skew_ms)
    if previous_sequence is not None:
        stats["skipped_decoder_sequences"] += max(0, sequence - previous_sequence - 1)
    return sequence


def summarize_camera(stats: dict[str, int | float]) -> dict[str, int | float]:
    samples = int(stats["samples"])
    denominator = max(samples, 1)
    return {
        "samples": samples,
        "reused_samples": int(stats["reused_samples"]),
        "skipped_decoder_sequences": int(stats["skipped_decoder_sequences"]),
        "missing_pts_samples": int(stats["missing_pts_samples"]),
        "over_skew_limit_samples": int(stats["over_skew_limit_samples"]),
        "age_ms_mean": float(stats["age_ms_total"]) / denominator,
        "age_ms_max": float(stats["age_ms_max"]),
        "state_receive_skew_ms_mean": float(stats["state_receive_skew_ms_total"]) / denominator,
        "state_receive_skew_ms_max": float(stats["state_receive_skew_ms_max"]),
    }


def run(args: argparse.Namespace, camera_names: list[str]) -> int:
    output = None
    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        output = args.output_jsonl.open("w", encoding="utf-8")

    config = JZRobotPinTimedConfig(
        id="jz_robot_pin_timed_readonly_probe",
        bind_ip=args.bind_ip,
        state_port=args.state_port,
        allowed_state_sender_ip=args.orin_ip,
        connect_timeout_s=args.connect_timeout_s,
        state_timeout_s=args.state_timeout_s,
        send_action_transport="local",
        send_action_execution="dry_run",
        zmq_cameras=make_camera_configs(args, camera_names),
        rtsp_cameras={},
        camera_buffer_size=args.camera_buffer_size,
        camera_stale_frame_timeout_ms=args.camera_stale_frame_timeout_ms,
        max_camera_state_receive_skew_ms=args.max_camera_state_receive_skew_ms,
        enforce_camera_state_receive_skew=False,
        reject_reused_camera_frames=False,
        timing_log_every_n=0,
        timing_sidecar=False,
    )
    robot = JZRobotPinTimed(config)
    camera_stats = {f"camera_{name}": empty_camera_stats() for name in camera_names}
    previous_sequences: dict[str, int] = {}
    sample_count = 0
    started_ns = time.monotonic_ns()

    emit(
        {
            "type": "start",
            "read_only": True,
            "send_action_transport": "local",
            "send_action_execution": "dry_run",
            "camera_protocol": "jz_realsense_zmq",
            "camera_frame_selection": "buffered_frame_nearest_state_receive_time",
            "state": f"udp://{args.bind_ip}:{args.state_port}",
            "allowed_state_sender_ip": args.orin_ip,
            "cameras": camera_names,
        },
        output,
    )

    try:
        robot.connect()
        started_ns = time.monotonic_ns()
        deadline_ns = started_ns + int(args.duration_s * 1_000_000_000)
        period_ns = int(1_000_000_000 / args.fps)
        next_sample_ns = started_ns
        while time.monotonic_ns() < deadline_ns:
            observation = robot.get_observation()
            timing = robot.last_observation_timing
            if timing is None:
                raise RuntimeError("Timed robot did not expose observation timing")

            payload = {
                "type": "sample",
                "sample_index": sample_count,
                "sample_monotonic_ns": time.monotonic_ns(),
                "numeric_observation_count": len(observation) - len(camera_names),
                **timing,
            }
            emit(payload, output)
            for key, camera_timing in timing["cameras"].items():
                previous_sequences[key] = update_camera_stats(
                    camera_stats[key],
                    camera_timing,
                    previous_sequences.get(key),
                    args.max_camera_state_receive_skew_ms,
                )
            sample_count += 1
            next_sample_ns += period_ns
            remaining_ns = next_sample_ns - time.monotonic_ns()
            if remaining_ns > 0:
                time.sleep(remaining_ns / 1_000_000_000)
    except KeyboardInterrupt:
        logging.info("Probe interrupted by user")
    finally:
        if robot.is_connected:
            robot.disconnect()

    elapsed_s = (time.monotonic_ns() - started_ns) / 1_000_000_000
    emit(
        {
            "type": "summary",
            "read_only": True,
            "camera_frame_selection": "buffered_frame_nearest_state_receive_time",
            "samples": sample_count,
            "elapsed_s": elapsed_s,
            "observed_fps": sample_count / elapsed_s if elapsed_s > 0 else 0.0,
            "cameras": {key: summarize_camera(stats) for key, stats in camera_stats.items()},
        },
        output,
    )
    if output is not None:
        output.close()
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    camera_names = validate_args(parser, args)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    return run(args, camera_names)


if __name__ == "__main__":
    raise SystemExit(main())
