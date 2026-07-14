#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lerobot.robots.jz_robot_udp.protocol import (
    STATE_SOURCE_NAMES,
    ProtocolError,
    validate_source_timing,
)

EXPECTED_CAMERAS = ("camera_head", "camera_left", "camera_right")
TIMING_FILE_RE = re.compile(r"episode-(\d{6})\.jsonl")
SESSION_ID_RE = re.compile(r"[0-9a-f]{32}")
RTSP_CAMERA_TIMESTAMP_STAGE = "decoder_output_before_pixel_conversion"
ZMQ_CAMERA_PROTOCOL = "jz_realsense_zmq"
ZMQ_CAMERA_PROTOCOL_VERSION = 1
ZMQ_CAMERA_TIMESTAMP_STAGE = "x86_after_zmq_receive_before_json_decode"
SOURCE_SNAPSHOT_TOLERANCE_NS = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate jz_robot_pin_timed CRF metadata and frame timing JSONL sidecars. "
            "This command only reads an existing dataset and never connects to the robot."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--expected-robot-type", default="jz_robot_pin_timed")
    parser.add_argument("--expected-codec", default="h264")
    parser.add_argument("--expected-crf", type=int, default=18)
    parser.add_argument("--expected-cameras", nargs="+", default=list(EXPECTED_CAMERAS))
    parser.add_argument("--expected-camera-fps", type=float, default=30.0)
    parser.add_argument("--expected-camera-source-fps", type=float, default=30.0)
    parser.add_argument("--min-camera-source-fps-ratio", type=float, default=0.9)
    parser.add_argument(
        "--expected-camera-protocol",
        choices=("jz_realsense_zmq", "rtsp", "any"),
        default="jz_realsense_zmq",
    )
    parser.add_argument("--expected-command-mode", choices=("dry_run", "armed"), default="armed")
    parser.add_argument("--expected-command-transport", choices=("local", "udp"), default="udp")
    parser.add_argument("--expected-action-key-count", type=int, default=18)
    parser.add_argument("--max-camera-age-ms", type=float, default=1000.0)
    parser.add_argument("--max-camera-state-skew-ms", type=float, default=100.0)
    parser.add_argument(
        "--max-reuse-fraction",
        type=float,
        default=None,
        help="Optional hard limit per camera. By default reuse is reported but does not fail.",
    )
    parser.add_argument(
        "--allow-hold-current",
        action="store_true",
        help="Allow action timing records produced by stale_policy=hold_current.",
    )
    parser.add_argument(
        "--require-source-timing",
        action="store_true",
        help="Require every dataset frame to contain a valid state.source_timing v1 object.",
    )
    parser.add_argument(
        "--max-source-age-ms",
        type=float,
        default=50.0,
        help="Maximum allowed age_ms for each of the four Orin state sources.",
    )
    parser.add_argument(
        "--max-source-skew-ms",
        type=float,
        default=20.0,
        help="Maximum allowed receive-time skew across the four Orin state sources.",
    )
    parser.add_argument(
        "--max-state-reuse-fraction",
        type=float,
        default=None,
        help=(
            "Optional hard limit for repeated state packet transitions within X86 sessions. "
            "By default state reuse is reported but does not fail."
        ),
    )
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args()


def add_error(report: dict[str, Any], message: str) -> None:
    report["errors"].append(message)


def add_warning(report: dict[str, Any], message: str) -> None:
    report["warnings"].append(message)


def is_integer(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def is_finite_number(value: Any) -> bool:
    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def require_integer(mapping: dict[str, Any], key: str, location: str, report: dict[str, Any]) -> int | None:
    value = mapping.get(key)
    if not is_integer(value):
        add_error(report, f"{location}.{key} must be an integer, got {value!r}")
        return None
    return int(value)


def require_nonnegative_integer(
    mapping: dict[str, Any], key: str, location: str, report: dict[str, Any]
) -> int | None:
    value = require_integer(mapping, key, location, report)
    if value is not None and value < 0:
        add_error(report, f"{location}.{key} must be non-negative, got {value}")
        return None
    return value


def require_finite_number(
    mapping: dict[str, Any], key: str, location: str, report: dict[str, Any]
) -> float | None:
    value = mapping.get(key)
    if not is_finite_number(value):
        add_error(report, f"{location}.{key} must be a finite number, got {value!r}")
        return None
    return float(value)


def require_nonnegative_number(
    mapping: dict[str, Any], key: str, location: str, report: dict[str, Any]
) -> float | None:
    value = require_finite_number(mapping, key, location, report)
    if value is not None and value < 0:
        add_error(report, f"{location}.{key} must be a finite non-negative number, got {value!r}")
        return None
    return value


def number_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "p50": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def source_timing_is_valid(source_timing: Any) -> bool:
    try:
        validate_source_timing(source_timing)
    except ProtocolError:
        return False
    return True


def validate_source_timing_details(
    source_timing: Any,
    location: str,
    args: argparse.Namespace,
    report: dict[str, Any],
) -> bool:
    try:
        validate_source_timing(source_timing)
    except ProtocolError as exc:
        add_error(report, f"{location} is not valid source_timing v1: {exc}")
        return False

    sources = source_timing["sources"]
    receive_monotonic_ns = [int(sources[name]["recv_monotonic_ns"]) for name in STATE_SOURCE_NAMES]
    computed_skew_ms = (max(receive_monotonic_ns) - min(receive_monotonic_ns)) / 1_000_000
    recorded_skew_ms = float(source_timing["source_skew_ms"])
    if not math.isclose(recorded_skew_ms, computed_skew_ms, rel_tol=0.0, abs_tol=1e-6):
        add_error(
            report,
            f"{location}.source_skew_ms={recorded_skew_ms} does not match source receive "
            f"timestamps ({computed_skew_ms} ms; tolerance=1 ns)",
        )
    if recorded_skew_ms > args.max_source_skew_ms:
        add_error(
            report,
            f"{location}.source_skew_ms={recorded_skew_ms} exceeds {args.max_source_skew_ms} ms",
        )

    inferred_snapshot_ns: list[int] = []
    for source_name in STATE_SOURCE_NAMES:
        source = sources[source_name]
        age_ms = float(source["age_ms"])
        if age_ms > args.max_source_age_ms:
            add_error(
                report,
                f"{location}.sources.{source_name}.age_ms={age_ms} exceeds {args.max_source_age_ms} ms",
            )
        inferred_snapshot_ns.append(int(source["recv_monotonic_ns"]) + round(age_ms * 1_000_000))

    snapshot_spread_ns = max(inferred_snapshot_ns) - min(inferred_snapshot_ns)
    if snapshot_spread_ns > SOURCE_SNAPSHOT_TOLERANCE_NS:
        add_error(
            report,
            f"{location} sources imply different snapshot times: spread={snapshot_spread_ns} ns "
            f"exceeds {SOURCE_SNAPSHOT_TOLERANCE_NS} ns",
        )
    return True


def sequence_stats(values: list[int]) -> dict[str, int | None]:
    if not values:
        return {
            "count": 0,
            "first": None,
            "last": None,
            "repeated_transitions": 0,
            "reset_transitions": 0,
            "forward_gap_packets": 0,
        }
    differences = [current - previous for previous, current in zip(values[:-1], values[1:], strict=True)]
    return {
        "count": len(values),
        "first": values[0],
        "last": values[-1],
        "repeated_transitions": sum(difference == 0 for difference in differences),
        "reset_transitions": sum(difference < 0 for difference in differences),
        "forward_gap_packets": sum(max(difference - 1, 0) for difference in differences),
    }


def read_dataset_frame_keys(root: Path, report: dict[str, Any]) -> list[tuple[int, int]]:
    parquet_files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not parquet_files:
        add_error(report, f"no dataset parquet files found below {root / 'data'}")
        return []

    frames: list[pd.DataFrame] = []
    for path in parquet_files:
        try:
            frames.append(pd.read_parquet(path, columns=["episode_index", "frame_index"]))
        except Exception as exc:
            add_error(report, f"failed reading frame keys from {path}: {exc}")
    if not frames:
        return []

    data = pd.concat(frames, ignore_index=True)
    keys: list[tuple[int, int]] = []
    for row_index, row in data.iterrows():
        episode_index = row["episode_index"]
        frame_index = row["frame_index"]
        if not is_finite_number(episode_index) or not float(episode_index).is_integer():
            add_error(report, f"data row {row_index} has invalid episode_index={episode_index!r}")
            continue
        if not is_finite_number(frame_index) or not float(frame_index).is_integer():
            add_error(report, f"data row {row_index} has invalid frame_index={frame_index!r}")
            continue
        keys.append((int(episode_index), int(frame_index)))

    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        add_error(report, f"dataset parquet contains duplicate episode/frame keys: {duplicates[:10]}")
    for episode_index in sorted({key[0] for key in keys}):
        frame_indexes = sorted(key[1] for key in keys if key[0] == episode_index)
        if frame_indexes != list(range(len(frame_indexes))):
            add_error(
                report,
                f"dataset episode {episode_index} frame_index must be contiguous from zero, "
                f"got first values={frame_indexes[:10]}",
            )
    return sorted(keys)


def read_timing_records(root: Path, report: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    timing_dir = root / "meta" / "timing"
    if not timing_dir.is_dir():
        add_error(report, f"missing timing sidecar directory: {timing_dir}")
        return {}

    timing_files = sorted(timing_dir.glob("episode-*.jsonl"))
    if not timing_files:
        add_error(report, f"no timing sidecars found in {timing_dir}")
        return {}

    records: dict[tuple[int, int], dict[str, Any]] = {}
    for path in timing_files:
        match = TIMING_FILE_RE.fullmatch(path.name)
        if match is None:
            add_error(report, f"invalid timing sidecar filename: {path.name}")
            continue
        file_episode_index = int(match.group(1))
        file_frame_indexes: list[int] = []
        try:
            stream = path.open("r", encoding="utf-8")
        except OSError as exc:
            add_error(report, f"failed opening timing sidecar {path}: {exc}")
            continue
        with stream:
            for line_number, raw_line in enumerate(stream, start=1):
                location = f"{path}:{line_number}"
                if not raw_line.strip():
                    add_error(report, f"{location} contains a blank timing record")
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    add_error(report, f"{location} contains invalid JSON: {exc}")
                    continue
                if not isinstance(record, dict):
                    add_error(report, f"{location} timing record must be a JSON object")
                    continue
                episode_index = record.get("episode_index")
                frame_index = record.get("frame_index")
                if not is_integer(episode_index) or not is_integer(frame_index):
                    add_error(
                        report,
                        f"{location} episode_index/frame_index must both be integers",
                    )
                    continue
                key = (int(episode_index), int(frame_index))
                if key[0] < 0 or key[1] < 0:
                    add_error(report, f"{location} episode_index/frame_index must be non-negative")
                    continue
                file_frame_indexes.append(key[1])
                if key[0] != file_episode_index:
                    add_error(
                        report,
                        f"{location} contains episode {key[0]} but filename declares {file_episode_index}",
                    )
                if key in records:
                    add_error(report, f"duplicate timing record for episode/frame {key}")
                    continue
                records[key] = record
        if file_frame_indexes != list(range(len(file_frame_indexes))):
            add_error(
                report,
                f"{path} timing records must be ordered and contiguous from frame 0, "
                f"got first values={file_frame_indexes[:10]}",
            )
    return records


def validate_camera_timing(
    camera: dict[str, Any], location: str, state_receive_ns: int | None, report: dict[str, Any]
) -> None:
    is_zmq = camera.get("protocol") == ZMQ_CAMERA_PROTOCOL
    expected_stage = ZMQ_CAMERA_TIMESTAMP_STAGE if is_zmq else RTSP_CAMERA_TIMESTAMP_STAGE
    if camera.get("timestamp_stage") != expected_stage:
        add_error(
            report,
            f"{location}.timestamp_stage must be {expected_stage!r}, got {camera.get('timestamp_stage')!r}",
        )
    if is_zmq:
        if camera.get("protocol_version") != ZMQ_CAMERA_PROTOCOL_VERSION:
            add_error(report, f"{location}.protocol_version must be {ZMQ_CAMERA_PROTOCOL_VERSION}")
        require_nonnegative_integer(camera, "sequence", location, report)
        require_nonnegative_integer(camera, "sequence_gap", location, report)
        decode_completed_ns = require_nonnegative_integer(
            camera, "decode_completed_monotonic_ns", location, report
        )
        camera_timing = camera.get("camera_timing")
        if not isinstance(camera_timing, dict):
            add_error(report, f"{location}.camera_timing must be an object")
        else:
            sequence = require_nonnegative_integer(
                camera_timing, "sequence", f"{location}.camera_timing", report
            )
            capture_ns = require_nonnegative_integer(
                camera_timing, "capture_monotonic_ns", f"{location}.camera_timing", report
            )
            encode_ns = require_nonnegative_integer(
                camera_timing,
                "encode_completed_monotonic_ns",
                f"{location}.camera_timing",
                report,
            )
            require_nonnegative_integer(camera_timing, "capture_wall_ns", f"{location}.camera_timing", report)
            for field in ("width", "height", "channels", "jpeg_quality", "payload_bytes"):
                require_nonnegative_integer(camera_timing, field, f"{location}.camera_timing", report)
            if sequence is not None and is_integer(camera.get("sequence")) and sequence != camera["sequence"]:
                add_error(report, f"{location}.camera_timing.sequence does not match receiver sequence")
            if camera_timing.get("timestamp_stage") != "after_realsense_read_before_jpeg":
                add_error(report, f"{location}.camera_timing.timestamp_stage is invalid")
            if camera_timing.get("pixel_format") != "RGB8" or camera_timing.get("encoding") != "jpeg":
                add_error(report, f"{location}.camera_timing must describe RGB8/JPEG")
            if capture_ns is not None and encode_ns is not None and encode_ns < capture_ns:
                add_error(report, f"{location}.camera_timing encode time precedes capture time")
        if (
            decode_completed_ns is not None
            and is_integer(camera.get("receive_monotonic_ns"))
            and decode_completed_ns < camera["receive_monotonic_ns"]
        ):
            add_error(report, f"{location} decode completion precedes X86 receive time")
    else:
        if "decoder_pts_ns" not in camera:
            add_error(report, f"{location}.decoder_pts_ns is missing")
        decoder_pts_ns = camera.get("decoder_pts_ns")
        if "decoder_pts_ns" in camera and decoder_pts_ns is not None and not is_integer(decoder_pts_ns):
            add_error(report, f"{location}.decoder_pts_ns must be an integer or null")
        require_nonnegative_integer(camera, "decoder_sequence", location, report)
        require_nonnegative_integer(camera, "reconnect_generation", location, report)
    require_nonnegative_integer(camera, "receive_wall_ns", location, report)
    camera_receive_ns = require_nonnegative_integer(camera, "receive_monotonic_ns", location, report)
    require_nonnegative_number(camera, "age_ms", location, report)
    recorded_delta_ms = require_finite_number(camera, "state_receive_delta_ms", location, report)
    recorded_skew_ms = require_nonnegative_number(camera, "state_receive_skew_ms", location, report)
    if not isinstance(camera.get("reused_by_observation_loop"), bool):
        add_error(report, f"{location}.reused_by_observation_loop must be boolean")
    if camera_receive_ns is not None and state_receive_ns is not None:
        computed_delta_ms = (camera_receive_ns - state_receive_ns) / 1_000_000
        computed_skew_ms = abs(computed_delta_ms)
        if recorded_delta_ms is not None and not math.isclose(
            recorded_delta_ms, computed_delta_ms, abs_tol=1e-6
        ):
            add_error(
                report,
                f"{location}.state_receive_delta_ms={recorded_delta_ms} does not match "
                f"the receive timestamps ({computed_delta_ms})",
            )
        if recorded_skew_ms is None:
            return
        if not math.isclose(recorded_skew_ms, computed_skew_ms, abs_tol=1e-6):
            add_error(
                report,
                f"{location}.state_receive_skew_ms={recorded_skew_ms} does not match "
                f"the receive timestamps ({computed_skew_ms})",
            )


def validate_action_timing(
    action: Any, location: str, allow_hold_current: bool, report: dict[str, Any]
) -> None:
    if not isinstance(action, dict):
        add_error(report, f"{location}.action must be an object, got {action!r}")
        return
    source = action.get("source")
    if source not in {"target_action_packet", "hold_current", "policy_output"}:
        add_error(report, f"{location}.action.source is invalid: {source!r}")
        return
    require_nonnegative_integer(action, "receive_wall_ns", f"{location}.action", report)
    require_nonnegative_integer(action, "receive_monotonic_ns", f"{location}.action", report)
    if source in {"hold_current", "policy_output"}:
        if source == "hold_current" and not allow_hold_current:
            add_error(report, f"{location} used hold_current instead of a target-action packet")
        for key in ("packet_seq", "packet_stamp_ns", "age_ms"):
            if action.get(key) is not None:
                add_error(report, f"{location}.action.{key} must be null for {source}")
        return
    require_nonnegative_integer(action, "packet_seq", f"{location}.action", report)
    require_nonnegative_integer(action, "packet_stamp_ns", f"{location}.action", report)
    require_nonnegative_number(action, "age_ms", f"{location}.action", report)


def validate_command_timing(
    command: Any,
    record: dict[str, Any],
    location: str,
    args: argparse.Namespace,
    report: dict[str, Any],
) -> None:
    if not isinstance(command, dict):
        add_error(report, f"{location}.command must be an object, got {command!r}")
        return
    command_location = f"{location}.command"
    command_observation_sequence = require_nonnegative_integer(
        command, "observation_sequence", command_location, report
    )
    if command_observation_sequence is not None and command_observation_sequence != record.get(
        "observation_sequence"
    ):
        add_error(
            report,
            f"{command_location}.observation_sequence={command_observation_sequence} does not "
            f"match record observation_sequence={record.get('observation_sequence')!r}",
        )
    require_nonnegative_integer(command, "packet_seq", command_location, report)
    require_nonnegative_integer(command, "packet_stamp_ns", command_location, report)
    send_wall_ns = require_nonnegative_integer(command, "send_completed_wall_ns", command_location, report)
    send_monotonic_ns = require_nonnegative_integer(
        command, "send_completed_monotonic_ns", command_location, report
    )
    action_key_count = require_nonnegative_integer(command, "action_key_count", command_location, report)
    if action_key_count is not None and action_key_count != args.expected_action_key_count:
        add_error(
            report,
            f"{command_location}.action_key_count must be {args.expected_action_key_count}, "
            f"got {action_key_count}",
        )
    if command.get("mode") != args.expected_command_mode:
        add_error(
            report,
            f"{command_location}.mode must be {args.expected_command_mode!r}, got {command.get('mode')!r}",
        )
    if command.get("transport") != args.expected_command_transport:
        add_error(
            report,
            f"{command_location}.transport must be {args.expected_command_transport!r}, "
            f"got {command.get('transport')!r}",
        )

    tolerance_ns = 1_000
    packet_stamp_ns = command.get("packet_stamp_ns")
    if (
        send_wall_ns is not None
        and is_integer(packet_stamp_ns)
        and send_wall_ns - int(packet_stamp_ns) < -tolerance_ns
    ):
        add_error(report, f"{command_location} completed before its packet wall timestamp")
    action = record.get("action")
    action_receive_ns = action.get("receive_monotonic_ns") if isinstance(action, dict) else None
    if (
        send_monotonic_ns is not None
        and is_integer(action_receive_ns)
        and send_monotonic_ns - int(action_receive_ns) < -tolerance_ns
    ):
        add_error(report, f"{command_location} completed before target action receipt")
    state = record.get("state")
    state_receive_ns = state.get("receive_monotonic_ns") if isinstance(state, dict) else None
    if (
        send_monotonic_ns is not None
        and is_integer(state_receive_ns)
        and send_monotonic_ns - int(state_receive_ns) < -tolerance_ns
    ):
        add_error(report, f"{command_location} completed before state receipt")


def validate_record(
    record: dict[str, Any], expected_cameras: list[str], args: argparse.Namespace, report: dict[str, Any]
) -> None:
    key = (record.get("episode_index"), record.get("frame_index"))
    location = f"timing[{key[0]},{key[1]}]"
    session_id = record.get("session_id")
    if not isinstance(session_id, str) or SESSION_ID_RE.fullmatch(session_id) is None:
        add_error(report, f"{location}.session_id must be a 32-character lowercase hex string")
    observation_sequence = require_integer(record, "observation_sequence", location, report)
    if observation_sequence is not None and observation_sequence <= 0:
        add_error(report, f"{location}.observation_sequence must be positive")

    state = record.get("state")
    state_receive_ns = None
    if not isinstance(state, dict):
        add_error(report, f"{location}.state must be an object")
    else:
        require_nonnegative_integer(state, "packet_seq", f"{location}.state", report)
        require_nonnegative_integer(state, "packet_stamp_ns", f"{location}.state", report)
        require_nonnegative_integer(state, "receive_wall_ns", f"{location}.state", report)
        state_receive_ns = require_nonnegative_integer(
            state, "receive_monotonic_ns", f"{location}.state", report
        )
        if "source_timing" in state:
            validate_source_timing_details(
                state["source_timing"],
                f"{location}.state.source_timing",
                args,
                report,
            )

    cameras = record.get("cameras")
    if not isinstance(cameras, dict):
        add_error(report, f"{location}.cameras must be an object")
    else:
        actual_camera_keys = set(cameras)
        expected_camera_keys = set(expected_cameras)
        if actual_camera_keys != expected_camera_keys:
            add_error(
                report,
                f"{location}.cameras keys must be {sorted(expected_camera_keys)}, "
                f"got {sorted(actual_camera_keys)}",
            )
        for camera_key in sorted(expected_camera_keys & actual_camera_keys):
            camera = cameras[camera_key]
            if not isinstance(camera, dict):
                add_error(report, f"{location}.cameras.{camera_key} must be an object")
                continue
            expected_protocol = getattr(args, "expected_camera_protocol", "any")
            actual_protocol = ZMQ_CAMERA_PROTOCOL if camera.get("protocol") == ZMQ_CAMERA_PROTOCOL else "rtsp"
            if expected_protocol != "any" and actual_protocol != expected_protocol:
                add_error(
                    report,
                    f"{location}.cameras.{camera_key} protocol={actual_protocol!r}, "
                    f"expected {expected_protocol!r}",
                )
            validate_camera_timing(
                camera,
                f"{location}.cameras.{camera_key}",
                state_receive_ns,
                report,
            )
    validate_action_timing(record.get("action"), location, args.allow_hold_current, report)
    validate_command_timing(record.get("command"), record, location, args, report)


def summarize_state_timing(
    records: list[dict[str, Any]], args: argparse.Namespace, report: dict[str, Any]
) -> None:
    state_sequences: list[int] = []
    source_timing_present_frames = 0
    source_timing_object_frames = 0
    source_timing_valid_frames = 0
    source_skews_ms: list[float] = []
    snapshot_spreads_ns: list[float] = []
    source_ages_ms: dict[str, list[float]] = {name: [] for name in STATE_SOURCE_NAMES}
    joint_zero_header_counts: Counter[str] = Counter()

    session_state_sequences: dict[str, list[int]] = {}
    session_transition_counts: Counter[str] = Counter()
    session_reuse_counts: Counter[str] = Counter()
    previous_session_id: str | None = None
    previous_state: dict[str, Any] | None = None
    last_nonzero_joint_headers: dict[str, int] = {}

    for record in records:
        session_id = record.get("session_id")
        state = record.get("state")
        if not isinstance(state, dict):
            previous_session_id = session_id if isinstance(session_id, str) else None
            previous_state = None
            continue

        packet_seq = state.get("packet_seq")
        if is_integer(packet_seq):
            state_sequences.append(int(packet_seq))
            if isinstance(session_id, str):
                session_state_sequences.setdefault(session_id, []).append(int(packet_seq))

        source_timing = state.get("source_timing")
        if "source_timing" in state:
            source_timing_present_frames += 1
            source_timing_object_frames += isinstance(source_timing, dict)
            if source_timing_is_valid(source_timing):
                source_timing_valid_frames += 1
                sources = source_timing["sources"]
                source_skews_ms.append(float(source_timing["source_skew_ms"]))
                snapshot_values = []
                for source_name in STATE_SOURCE_NAMES:
                    source = sources[source_name]
                    source_ages_ms[source_name].append(float(source["age_ms"]))
                    snapshot_values.append(
                        int(source["recv_monotonic_ns"]) + round(float(source["age_ms"]) * 1_000_000)
                    )
                    if source_name.endswith("_joints") and source["header_stamp_ns"] == 0:
                        joint_zero_header_counts[source_name] += 1
                snapshot_spreads_ns.append(max(snapshot_values) - min(snapshot_values))

        same_session = isinstance(session_id, str) and session_id == previous_session_id
        if not same_session:
            last_nonzero_joint_headers.clear()
            if source_timing_is_valid(source_timing):
                for source_name in STATE_SOURCE_NAMES:
                    if not source_name.endswith("_joints"):
                        continue
                    header_stamp_ns = source_timing["sources"][source_name]["header_stamp_ns"]
                    if header_stamp_ns != 0:
                        last_nonzero_joint_headers[source_name] = header_stamp_ns
        if same_session and previous_state is not None:
            previous_seq = previous_state.get("packet_seq")
            if is_integer(previous_seq) and is_integer(packet_seq):
                session_transition_counts[session_id] += 1
                previous_seq_int = int(previous_seq)
                packet_seq_int = int(packet_seq)
                location = f"timing[{record.get('episode_index')},{record.get('frame_index')}].state"
                if packet_seq_int < previous_seq_int:
                    add_error(
                        report,
                        f"{location}.packet_seq regressed within session {session_id}: "
                        f"{previous_seq_int} -> {packet_seq_int}",
                    )
                elif packet_seq_int == previous_seq_int:
                    session_reuse_counts[session_id] += 1
                    if state.get("packet_stamp_ns") != previous_state.get("packet_stamp_ns"):
                        add_error(
                            report,
                            f"{location} reused packet_seq={packet_seq_int} but packet_stamp_ns changed",
                        )
                    if state.get("source_timing") != previous_state.get("source_timing"):
                        add_error(
                            report,
                            f"{location} reused packet_seq={packet_seq_int} but source_timing changed",
                        )
                else:
                    previous_source_timing = previous_state.get("source_timing")
                    if source_timing_is_valid(previous_source_timing) and source_timing_is_valid(
                        source_timing
                    ):
                        previous_sources = previous_source_timing["sources"]
                        current_sources = source_timing["sources"]
                        for source_name in STATE_SOURCE_NAMES:
                            previous_source = previous_sources[source_name]
                            current_source = current_sources[source_name]
                            if current_source["generation"] <= previous_source["generation"]:
                                add_error(
                                    report,
                                    f"{location}.source_timing.sources.{source_name}.generation "
                                    f"did not strictly advance for packet_seq "
                                    f"{previous_seq_int} -> {packet_seq_int}",
                                )
                            if current_source["recv_monotonic_ns"] <= previous_source["recv_monotonic_ns"]:
                                add_error(
                                    report,
                                    f"{location}.source_timing.sources.{source_name}.recv_monotonic_ns "
                                    f"did not strictly advance for packet_seq "
                                    f"{previous_seq_int} -> {packet_seq_int}",
                                )
                            if source_name.endswith("_joints"):
                                current_header = current_source["header_stamp_ns"]
                                previous_nonzero_header = last_nonzero_joint_headers.get(source_name)
                                if (
                                    current_header != 0
                                    and previous_nonzero_header is not None
                                    and current_header <= previous_nonzero_header
                                ):
                                    add_error(
                                        report,
                                        f"{location}.source_timing.sources.{source_name}.header_stamp_ns "
                                        f"did not strictly advance: "
                                        f"{previous_nonzero_header} -> {current_header}",
                                    )
                                if current_header != 0:
                                    last_nonzero_joint_headers[source_name] = current_header

        previous_session_id = session_id if isinstance(session_id, str) else None
        previous_state = state

    transition_count = sum(session_transition_counts.values())
    reuse_count = sum(session_reuse_counts.values())
    reuse_fraction = reuse_count / transition_count if transition_count else None
    state_packet_report = sequence_stats(state_sequences)
    state_packet_report.update(
        {
            "transition_count": transition_count,
            "reused_transitions": reuse_count,
            "reuse_fraction": reuse_fraction,
            "sessions": {
                session_id: {
                    **sequence_stats(sequences),
                    "transition_count": session_transition_counts[session_id],
                    "reused_transitions": session_reuse_counts[session_id],
                    "reuse_fraction": (
                        session_reuse_counts[session_id] / session_transition_counts[session_id]
                        if session_transition_counts[session_id]
                        else None
                    ),
                }
                for session_id, sequences in session_state_sequences.items()
            },
        }
    )
    report["state_packets"] = state_packet_report
    if (
        args.max_state_reuse_fraction is not None
        and reuse_fraction is not None
        and reuse_fraction > args.max_state_reuse_fraction
    ):
        add_error(
            report,
            f"state packet reuse fraction {reuse_fraction:.6f} exceeds {args.max_state_reuse_fraction:.6f}",
        )

    report["state_source_timing"] = {
        "present_frames": source_timing_present_frames,
        "missing_frames": len(records) - source_timing_present_frames,
        "presence_fraction": source_timing_present_frames / len(records) if records else None,
        "object_frames": source_timing_object_frames,
        "valid_frames": source_timing_valid_frames,
        "invalid_frames": source_timing_present_frames - source_timing_valid_frames,
        "source_skew_ms": number_stats(source_skews_ms),
        "inferred_snapshot_spread_ns": number_stats(snapshot_spreads_ns),
        "sources": {},
    }
    for source_name in STATE_SOURCE_NAMES:
        source_report: dict[str, Any] = {"age_ms": number_stats(source_ages_ms[source_name])}
        if source_name.endswith("_joints"):
            zero_count = joint_zero_header_counts[source_name]
            source_report.update(
                {
                    "zero_header_stamp_frames": zero_count,
                    "zero_header_stamp_fraction": (
                        zero_count / source_timing_valid_frames if source_timing_valid_frames else None
                    ),
                }
            )
            if zero_count:
                add_warning(
                    report,
                    f"state source {source_name} has header_stamp_ns=0 in {zero_count}/"
                    f"{source_timing_valid_frames} valid source_timing frames",
                )
        report["state_source_timing"]["sources"][source_name] = source_report


def summarize_records(
    records: list[dict[str, Any]],
    expected_cameras: list[str],
    args: argparse.Namespace,
    report: dict[str, Any],
) -> None:
    session_sequences: dict[str, list[int]] = {}
    session_episodes: dict[str, set[int]] = {}
    session_order: list[str] = []
    closed_sessions: set[str] = set()
    previous_session: str | None = None
    for record in records:
        session_id = record.get("session_id")
        observation_sequence = record.get("observation_sequence")
        if not isinstance(session_id, str) or not is_integer(observation_sequence):
            continue
        if session_id != previous_session:
            if session_id in closed_sessions:
                add_error(report, f"timing session {session_id} appears in multiple disjoint segments")
            if previous_session is not None:
                closed_sessions.add(previous_session)
            session_order.append(session_id)
            previous_session = session_id
        session_sequences.setdefault(session_id, []).append(int(observation_sequence))
        episode_index = record.get("episode_index")
        if is_integer(episode_index):
            session_episodes.setdefault(session_id, set()).add(int(episode_index))

    session_reports: dict[str, Any] = {}
    for session_id in session_order:
        stats = sequence_stats(session_sequences[session_id])
        session_reports[session_id] = {
            **stats,
            "episodes": sorted(session_episodes.get(session_id, set())),
        }
        if stats["repeated_transitions"]:
            add_error(report, f"session {session_id} observation_sequence contains repeated values")
        if stats["reset_transitions"]:
            add_error(report, f"session {session_id} observation_sequence is not strictly increasing")
    report["observation_sessions"] = {
        "count": len(session_reports),
        "order": session_order,
        "sessions": session_reports,
    }

    summarize_state_timing(records, args, report)

    camera_report: dict[str, Any] = {}
    pts_forward_jump_threshold_ms = (
        1.5 * 1000 / args.expected_camera_fps
        if math.isfinite(args.expected_camera_fps) and args.expected_camera_fps > 0
        else 0.0
    )
    for camera_key in expected_cameras:
        ages: list[float] = []
        receive_deltas: list[float] = []
        skews: list[float] = []
        sequences: list[int] = []
        protocols: Counter[str] = Counter()
        sequence_gaps: list[int] = []
        receive_samples: list[tuple[int, int, int]] = []
        capture_samples: list[tuple[int, int, int]] = []
        orin_encode_ms: list[float] = []
        x86_decode_ms: list[float] = []
        reconnect_generations: list[int] = []
        pts_samples: list[tuple[int, int, int | None]] = []
        camera_record_count = 0
        reused_count = 0
        missing_pts_count = 0
        for record in records:
            cameras = record.get("cameras")
            camera = cameras.get(camera_key) if isinstance(cameras, dict) else None
            if not isinstance(camera, dict):
                continue
            camera_record_count += 1
            protocol = ZMQ_CAMERA_PROTOCOL if camera.get("protocol") == ZMQ_CAMERA_PROTOCOL else "rtsp"
            protocols[protocol] += 1
            if is_finite_number(camera.get("age_ms")):
                ages.append(float(camera["age_ms"]))
            if is_finite_number(camera.get("state_receive_delta_ms")):
                receive_deltas.append(float(camera["state_receive_delta_ms"]))
            if is_finite_number(camera.get("state_receive_skew_ms")):
                skews.append(float(camera["state_receive_skew_ms"]))
            sequence_value = camera.get("sequence", camera.get("decoder_sequence"))
            if is_integer(sequence_value):
                sequence = int(sequence_value)
                sequences.append(sequence)
                episode_index = record.get("episode_index")
                receive_ns = camera.get("receive_monotonic_ns")
                if is_integer(episode_index) and is_integer(receive_ns):
                    receive_samples.append((int(episode_index), sequence, int(receive_ns)))
                camera_timing = camera.get("camera_timing")
                if isinstance(camera_timing, dict):
                    capture_ns = camera_timing.get("capture_monotonic_ns")
                    encode_ns = camera_timing.get("encode_completed_monotonic_ns")
                    if is_integer(episode_index) and is_integer(capture_ns):
                        capture_samples.append((int(episode_index), sequence, int(capture_ns)))
                    if is_integer(capture_ns) and is_integer(encode_ns) and encode_ns >= capture_ns:
                        orin_encode_ms.append((int(encode_ns) - int(capture_ns)) / 1_000_000)
                decode_ns = camera.get("decode_completed_monotonic_ns")
                if is_integer(receive_ns) and is_integer(decode_ns) and decode_ns >= receive_ns:
                    x86_decode_ms.append((int(decode_ns) - int(receive_ns)) / 1_000_000)
            if is_integer(camera.get("sequence_gap")):
                sequence_gaps.append(int(camera["sequence_gap"]))
            if is_integer(camera.get("reconnect_generation")):
                reconnect_generations.append(int(camera["reconnect_generation"]))
                decoder_pts_ns = camera.get("decoder_pts_ns")
                episode_index = record.get("episode_index")
                if is_integer(episode_index):
                    pts_samples.append(
                        (
                            int(episode_index),
                            int(camera["reconnect_generation"]),
                            int(decoder_pts_ns) if is_integer(decoder_pts_ns) else None,
                        )
                    )
            reused_count += camera.get("reused_by_observation_loop") is True
            if protocol == "rtsp":
                missing_pts_count += camera.get("decoder_pts_ns") is None

        sample_count = camera_record_count
        reuse_fraction = reused_count / sample_count if sample_count else None
        pts_by_generation: dict[str, Any] = {}
        for generation in sorted({item[1] for item in pts_samples}):
            intervals_ms = [
                (current_pts - previous_pts) / 1_000_000
                for (
                    previous_episode,
                    previous_generation,
                    previous_pts,
                ), (current_episode, current_generation, current_pts) in zip(
                    pts_samples[:-1], pts_samples[1:], strict=True
                )
                if previous_episode == current_episode
                and previous_generation == generation
                and current_generation == generation
                and previous_pts is not None
                and current_pts is not None
            ]
            pts_by_generation[str(generation)] = {
                "episode_segments": len(
                    {episode for episode, item_generation, _ in pts_samples if item_generation == generation}
                ),
                "interval_ms": number_stats(intervals_ms),
                "non_advancing_transitions": sum(interval <= 0 for interval in intervals_ms),
                "forward_jump_transitions": sum(
                    interval > pts_forward_jump_threshold_ms for interval in intervals_ms
                ),
            }

        def sequence_rate_stats(samples: list[tuple[int, int, int]]) -> dict[str, Any]:
            rates = [
                (current_sequence - previous_sequence) * 1_000_000_000 / (current_ns - previous_ns)
                for (previous_episode, previous_sequence, previous_ns), (
                    current_episode,
                    current_sequence,
                    current_ns,
                ) in zip(samples[:-1], samples[1:], strict=True)
                if current_episode == previous_episode
                and current_sequence > previous_sequence
                and current_ns > previous_ns
            ]
            return number_stats(rates)

        receive_intervals_hz = [
            1_000_000_000 / (current_ns - previous_ns)
            for (previous_episode, _, previous_ns), (current_episode, _, current_ns) in zip(
                receive_samples[:-1], receive_samples[1:], strict=True
            )
            if current_episode == previous_episode and current_ns > previous_ns
        ]
        source_capture_fps = sequence_rate_stats(capture_samples)
        camera_report[camera_key] = {
            "protocol_counts": dict(sorted(protocols.items())),
            "age_ms": number_stats(ages),
            "state_receive_delta_ms": number_stats(receive_deltas),
            "state_receive_skew_ms": number_stats(skews),
            "decoder_sequence": sequence_stats(sequences),
            "sequence": sequence_stats(sequences),
            "sequence_gap_total": sum(sequence_gaps),
            "sequence_gap_frames": sum(gap > 0 for gap in sequence_gaps),
            "x86_selected_receive_fps": number_stats(receive_intervals_hz),
            "source_fps_from_x86_receive": sequence_rate_stats(receive_samples),
            "source_fps_from_orin_capture": source_capture_fps,
            "orin_jpeg_encode_ms": number_stats(orin_encode_ms),
            "x86_jpeg_decode_ms": number_stats(x86_decode_ms),
            "reused_frames": reused_count,
            "reuse_fraction": reuse_fraction,
            "missing_decoder_pts_frames": missing_pts_count,
            "reconnect_generations": sorted(set(reconnect_generations)),
            "decoder_pts": {
                "stream_relative": True,
                "forward_jump_threshold_ms": pts_forward_jump_threshold_ms,
                "by_reconnect_generation": pts_by_generation,
            },
        }
        if ages and max(ages) > args.max_camera_age_ms:
            add_error(
                report,
                f"{camera_key} max age {max(ages):.3f}ms exceeds {args.max_camera_age_ms:.3f}ms",
            )
        if skews and max(skews) > args.max_camera_state_skew_ms:
            add_error(
                report,
                f"{camera_key} max state receive skew {max(skews):.3f}ms exceeds "
                f"{args.max_camera_state_skew_ms:.3f}ms",
            )
        expected_source_fps = getattr(args, "expected_camera_source_fps", 30.0)
        min_source_fps_ratio = getattr(args, "min_camera_source_fps_ratio", 0.9)
        if protocols[ZMQ_CAMERA_PROTOCOL] and source_capture_fps["mean"] is not None:
            minimum_source_fps = expected_source_fps * min_source_fps_ratio
            if source_capture_fps["mean"] < minimum_source_fps:
                add_error(
                    report,
                    f"{camera_key} Orin capture FPS {source_capture_fps['mean']:.3f} is below "
                    f"{minimum_source_fps:.3f}",
                )
        if reused_count:
            add_warning(
                report,
                f"{camera_key} reused {reused_count}/{sample_count} observation frames "
                f"({reuse_fraction:.2%})",
            )
        if (
            args.max_reuse_fraction is not None
            and reuse_fraction is not None
            and reuse_fraction > args.max_reuse_fraction
        ):
            add_error(
                report,
                f"{camera_key} reuse fraction {reuse_fraction:.2%} exceeds {args.max_reuse_fraction:.2%}",
            )
        if missing_pts_count:
            add_warning(
                report,
                f"{camera_key} has {missing_pts_count}/{sample_count} frames without decoder PTS",
            )
        non_advancing_pts = sum(
            generation_stats["non_advancing_transitions"] for generation_stats in pts_by_generation.values()
        )
        if non_advancing_pts:
            add_warning(
                report,
                f"{camera_key} has {non_advancing_pts} non-advancing decoder PTS transitions",
            )
        if len(set(reconnect_generations)) > 1:
            add_warning(
                report,
                f"{camera_key} spans reconnect generations {sorted(set(reconnect_generations))}",
            )
    report["cameras"] = camera_report

    source_counts: Counter[str] = Counter()
    packet_sequences: list[int] = []
    packet_ages: list[float] = []
    for record in records:
        action = record.get("action")
        if not isinstance(action, dict):
            continue
        source = action.get("source")
        if isinstance(source, str):
            source_counts[source] += 1
        if source == "target_action_packet":
            if is_integer(action.get("packet_seq")):
                packet_sequences.append(int(action["packet_seq"]))
            if is_finite_number(action.get("age_ms")):
                packet_ages.append(float(action["age_ms"]))
    report["action_packets"] = {
        "source_counts": dict(sorted(source_counts.items())),
        "sequence": sequence_stats(packet_sequences),
        "age_ms": number_stats(packet_ages),
    }

    command_sequences: list[int] = []
    command_modes: Counter[str] = Counter()
    command_transports: Counter[str] = Counter()
    target_action_to_send_ms: list[float] = []
    state_to_send_ms: list[float] = []
    packet_stamp_to_send_ms: list[float] = []
    for record in records:
        command = record.get("command")
        if not isinstance(command, dict):
            continue
        if is_integer(command.get("packet_seq")):
            command_sequences.append(int(command["packet_seq"]))
        if isinstance(command.get("mode"), str):
            command_modes[command["mode"]] += 1
        if isinstance(command.get("transport"), str):
            command_transports[command["transport"]] += 1
        send_monotonic_ns = command.get("send_completed_monotonic_ns")
        send_wall_ns = command.get("send_completed_wall_ns")
        packet_stamp_ns = command.get("packet_stamp_ns")
        action = record.get("action")
        if (
            isinstance(action, dict)
            and action.get("source") == "target_action_packet"
            and is_integer(action.get("receive_monotonic_ns"))
            and is_integer(send_monotonic_ns)
        ):
            target_action_to_send_ms.append(
                max(0.0, (int(send_monotonic_ns) - int(action["receive_monotonic_ns"])) / 1_000_000)
            )
        state = record.get("state")
        if (
            isinstance(state, dict)
            and is_integer(state.get("receive_monotonic_ns"))
            and is_integer(send_monotonic_ns)
        ):
            state_to_send_ms.append(
                max(0.0, (int(send_monotonic_ns) - int(state["receive_monotonic_ns"])) / 1_000_000)
            )
        if is_integer(packet_stamp_ns) and is_integer(send_wall_ns):
            packet_stamp_to_send_ms.append(max(0.0, (int(send_wall_ns) - int(packet_stamp_ns)) / 1_000_000))
    report["commands"] = {
        "sequence": sequence_stats(command_sequences),
        "mode_counts": dict(sorted(command_modes.items())),
        "transport_counts": dict(sorted(command_transports.items())),
        "target_action_receive_to_command_send_ms": number_stats(target_action_to_send_ms),
        "state_receive_to_command_send_ms": number_stats(state_to_send_ms),
        "packet_stamp_to_send_completed_ms": number_stats(packet_stamp_to_send_ms),
    }


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    root = args.dataset_root.expanduser().resolve()
    report: dict[str, Any] = {
        "dataset_root": str(root),
        "status": "FAIL",
        "errors": [],
        "warnings": [],
        "thresholds": {
            "expected_robot_type": args.expected_robot_type,
            "expected_codec": args.expected_codec,
            "expected_crf": args.expected_crf,
            "expected_cameras": args.expected_cameras,
            "expected_camera_fps": args.expected_camera_fps,
            "expected_camera_source_fps": getattr(args, "expected_camera_source_fps", 30.0),
            "min_camera_source_fps_ratio": getattr(args, "min_camera_source_fps_ratio", 0.9),
            "expected_camera_protocol": getattr(args, "expected_camera_protocol", "any"),
            "expected_command_mode": args.expected_command_mode,
            "expected_command_transport": args.expected_command_transport,
            "expected_action_key_count": args.expected_action_key_count,
            "max_camera_age_ms": args.max_camera_age_ms,
            "max_camera_state_skew_ms": args.max_camera_state_skew_ms,
            "max_reuse_fraction": args.max_reuse_fraction,
            "allow_hold_current": args.allow_hold_current,
            "require_source_timing": getattr(args, "require_source_timing", False),
            "max_source_age_ms": args.max_source_age_ms,
            "max_source_skew_ms": args.max_source_skew_ms,
            "max_state_reuse_fraction": args.max_state_reuse_fraction,
        },
    }
    if (
        not math.isfinite(args.max_camera_age_ms)
        or not math.isfinite(args.max_camera_state_skew_ms)
        or args.max_camera_age_ms < 0
        or args.max_camera_state_skew_ms < 0
    ):
        add_error(report, "camera age/skew thresholds must be non-negative")
    if not math.isfinite(args.expected_camera_fps) or args.expected_camera_fps <= 0:
        add_error(report, "expected_camera_fps must be a finite positive number")
    expected_camera_source_fps = getattr(args, "expected_camera_source_fps", 30.0)
    min_camera_source_fps_ratio = getattr(args, "min_camera_source_fps_ratio", 0.9)
    if not math.isfinite(expected_camera_source_fps) or expected_camera_source_fps <= 0:
        add_error(report, "expected_camera_source_fps must be a finite positive number")
    if not math.isfinite(min_camera_source_fps_ratio) or not 0 < min_camera_source_fps_ratio <= 1:
        add_error(report, "min_camera_source_fps_ratio must be in (0, 1]")
    if not 0 <= args.expected_crf <= 51:
        add_error(report, "expected_crf must be between 0 and 51")
    if args.expected_action_key_count <= 0:
        add_error(report, "expected_action_key_count must be positive")
    if args.max_reuse_fraction is not None and not 0 <= args.max_reuse_fraction <= 1:
        add_error(report, "max_reuse_fraction must be between 0 and 1")
    if (
        not math.isfinite(args.max_source_age_ms)
        or not math.isfinite(args.max_source_skew_ms)
        or args.max_source_age_ms < 0
        or args.max_source_skew_ms < 0
    ):
        add_error(report, "source age/skew thresholds must be non-negative")
    if args.max_state_reuse_fraction is not None and (
        not math.isfinite(args.max_state_reuse_fraction) or not 0 <= args.max_state_reuse_fraction <= 1
    ):
        add_error(report, "max_state_reuse_fraction must be finite and between 0 and 1")
    if len(set(args.expected_cameras)) != len(args.expected_cameras):
        add_error(report, "expected camera names must be unique")

    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        add_error(report, f"missing dataset info: {info_path}")
        return report
    try:
        with info_path.open("r", encoding="utf-8") as stream:
            info = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        add_error(report, f"failed reading {info_path}: {exc}")
        return report

    report["robot_type"] = info.get("robot_type")
    if info.get("robot_type") != args.expected_robot_type:
        add_error(
            report,
            f"robot_type must be {args.expected_robot_type!r}, got {info.get('robot_type')!r}",
        )
    video_encoding = info.get("video_encoding")
    if not isinstance(video_encoding, dict):
        add_error(report, "meta/info.json is missing the video_encoding object")
        report["video_encoding"] = video_encoding
    else:
        report["video_encoding"] = video_encoding
        persisted_codec = video_encoding.get("codec")
        if persisted_codec != args.expected_codec:
            add_error(
                report,
                f"video_encoding.codec must be {args.expected_codec!r}, got {persisted_codec!r}",
            )
        persisted_crf = video_encoding.get("crf")
        if not is_integer(persisted_crf) or int(persisted_crf) != args.expected_crf:
            add_error(
                report,
                f"video_encoding.crf must be {args.expected_crf}, got {persisted_crf!r}",
            )

    dataset_keys = read_dataset_frame_keys(root, report)
    timing_records = read_timing_records(root, report)
    dataset_key_set = set(dataset_keys)
    timing_key_set = set(timing_records)
    missing_keys = sorted(dataset_key_set - timing_key_set)
    extra_keys = sorted(timing_key_set - dataset_key_set)
    if missing_keys:
        add_error(
            report,
            f"missing timing for {len(missing_keys)} dataset frames; first keys={missing_keys[:10]}",
        )
    if extra_keys:
        add_error(
            report,
            f"timing contains {len(extra_keys)} frames absent from the dataset; first keys={extra_keys[:10]}",
        )
    report["dataset_frames"] = len(dataset_keys)
    report["timing_frames"] = len(timing_records)
    report["episodes"] = sorted({episode_index for episode_index, _ in dataset_keys})

    ordered_records = [timing_records[key] for key in dataset_keys if key in timing_records]
    for record in ordered_records:
        validate_record(record, args.expected_cameras, args, report)
    summarize_records(ordered_records, args.expected_cameras, args, report)
    source_timing = report.get("state_source_timing", {})
    if getattr(args, "require_source_timing", False) and source_timing.get("valid_frames") != len(
        ordered_records
    ):
        add_error(
            report,
            "a valid state.source_timing v1 object is required for every frame: "
            f"present={source_timing.get('present_frames', 0)} "
            f"valid={source_timing.get('valid_frames', 0)} "
            f"expected={len(ordered_records)}",
        )
    report["status"] = "PASS" if not report["errors"] else "FAIL"
    return report


def main() -> int:
    args = parse_args()
    report = run_check(args)
    report_path = args.report_json or args.dataset_root / "timing_check_report.json"
    report_path = report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    print(f"dataset_root={report['dataset_root']}")
    print(f"status={report['status']}")
    print(
        f"dataset_frames={report.get('dataset_frames', 0)} "
        f"timing_frames={report.get('timing_frames', 0)} "
        f"video_encoding={report.get('video_encoding')}"
    )
    state_source_timing = report.get("state_source_timing")
    if state_source_timing:
        print(
            f"state_source_timing_present={state_source_timing['present_frames']}/"
            f"{report.get('timing_frames', 0)} "
            f"presence_fraction={state_source_timing['presence_fraction']} "
            f"valid_frames={state_source_timing['valid_frames']} "
            f"invalid_frames={state_source_timing['invalid_frames']} "
            f"source_skew_max_ms={state_source_timing['source_skew_ms']['max']}"
        )
    state_packets = report.get("state_packets")
    if state_packets:
        print(
            f"state_sequence_first={state_packets['first']} state_sequence_last={state_packets['last']} "
            f"state_reused_transitions={state_packets['reused_transitions']}/"
            f"{state_packets['transition_count']} "
            f"state_reuse_fraction={state_packets['reuse_fraction']}"
        )
    for camera_key, camera in report.get("cameras", {}).items():
        age = camera["age_ms"]
        delta = camera["state_receive_delta_ms"]
        skew = camera["state_receive_skew_ms"]
        pts_generations = camera["decoder_pts"]["by_reconnect_generation"]
        pts_non_advancing = sum(item["non_advancing_transitions"] for item in pts_generations.values())
        pts_forward_jumps = sum(item["forward_jump_transitions"] for item in pts_generations.values())
        print(
            f"camera={camera_key} age_p95_ms={age['p95']} age_max_ms={age['max']} "
            f"delta_p50_ms={delta['p50']} delta_p95_ms={delta['p95']} "
            f"skew_p95_ms={skew['p95']} skew_max_ms={skew['max']} "
            f"reuse_fraction={camera['reuse_fraction']} pts_non_advancing={pts_non_advancing} "
            f"pts_forward_jumps={pts_forward_jumps} protocols={camera['protocol_counts']} "
            f"sequence_gaps={camera['sequence_gap_total']} "
            f"source_fps={camera['source_fps_from_orin_capture']['mean']} "
            f"x86_receive_fps={camera['x86_selected_receive_fps']['mean']}"
        )
    action_packets = report.get("action_packets", {})
    if action_packets:
        print(
            f"action_sources={action_packets['source_counts']} "
            f"action_age_p95_ms={action_packets['age_ms']['p95']} "
            f"action_sequence={action_packets['sequence']}"
        )
    commands = report.get("commands", {})
    if commands:
        print(
            f"command_sequence={commands['sequence']} "
            f"action_to_send_p95_ms="
            f"{commands['target_action_receive_to_command_send_ms']['p95']} "
            f"state_to_send_p95_ms={commands['state_receive_to_command_send_ms']['p95']}"
        )
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    for error in report["errors"]:
        print(f"ERROR: {error}")
    print(f"report_json={report_path}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
