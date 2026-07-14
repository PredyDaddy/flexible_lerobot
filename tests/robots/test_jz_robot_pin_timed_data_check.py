#!/usr/bin/env python

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from my_devs.jz_robot_pin_timed.data_check.check_timing import run_check

CAMERAS = ("camera_head", "camera_left", "camera_right")


def make_args(dataset_root: Path) -> Namespace:
    return Namespace(
        dataset_root=dataset_root,
        expected_robot_type="jz_robot_pin_timed",
        expected_codec="h264",
        expected_crf=18,
        expected_cameras=list(CAMERAS),
        expected_camera_fps=30.0,
        expected_command_mode="armed",
        expected_command_transport="udp",
        expected_action_key_count=18,
        max_camera_age_ms=1000.0,
        max_camera_state_skew_ms=100.0,
        max_reuse_fraction=None,
        allow_hold_current=False,
        require_source_timing=False,
        max_source_age_ms=50.0,
        max_source_skew_ms=20.0,
        max_state_reuse_fraction=None,
        report_json=None,
    )


def make_source_timing(generation: int) -> dict:
    receive_base_ns = 5_000_000_000_000_000 + generation * 100_000_000
    snapshot_ns = receive_base_ns + 10_000_000
    source_offsets_ns = {
        "left_joints": 0,
        "right_joints": 1_000_000,
        "left_gripper": 2_000_000,
        "right_gripper": 3_000_000,
    }
    sources = {}
    for source_name, offset_ns in source_offsets_ns.items():
        receive_ns = receive_base_ns + offset_ns
        sources[source_name] = {
            "generation": generation,
            "recv_wall_ns": 8_000_000_000_000_000 + generation * 100_000_000 + offset_ns,
            "recv_monotonic_ns": receive_ns,
            "header_stamp_ns": generation * 1_000 if source_name.endswith("_joints") else None,
            "age_ms": (snapshot_ns - receive_ns) / 1_000_000,
        }
    return {"schema_version": 1, "source_skew_ms": 3.0, "sources": sources}


def make_timing_record(episode_index: int, frame_index: int, sequence: int) -> dict:
    state_monotonic_ns = 1_000_000_000 + sequence * 100_000_000
    state_wall_ns = 2_000_000_000 + sequence * 100_000_000
    cameras = {}
    for camera_index, camera_name in enumerate(CAMERAS):
        delta_ms = float((camera_index - 1) * 5)
        camera_monotonic_ns = state_monotonic_ns + int(delta_ms * 1_000_000)
        cameras[camera_name] = {
            "timestamp_stage": "decoder_output_before_pixel_conversion",
            "decoder_pts_ns": episode_index * 1_000_000_000 + frame_index * 33_333_333,
            "receive_wall_ns": state_wall_ns + int(delta_ms * 1_000_000),
            "receive_monotonic_ns": camera_monotonic_ns,
            "decoder_sequence": sequence,
            "reconnect_generation": 1,
            "age_ms": 10.0,
            "reused_by_observation_loop": False,
            "state_receive_delta_ms": delta_ms,
            "state_receive_skew_ms": abs(delta_ms),
        }

    action_receive_monotonic_ns = state_monotonic_ns + 1_000_000
    command_send_monotonic_ns = state_monotonic_ns + 2_000_000
    command_stamp_ns = state_wall_ns + 1_000_000
    return {
        "session_id": "a" * 32,
        "episode_index": episode_index,
        "frame_index": frame_index,
        "observation_sequence": sequence,
        "state": {
            "packet_seq": sequence,
            "packet_stamp_ns": 10_000 + sequence,
            "receive_wall_ns": state_wall_ns,
            "receive_monotonic_ns": state_monotonic_ns,
        },
        "cameras": cameras,
        "action": {
            "source": "target_action_packet",
            "packet_seq": sequence,
            "packet_stamp_ns": state_wall_ns,
            "receive_wall_ns": state_wall_ns + 1_000_000,
            "receive_monotonic_ns": action_receive_monotonic_ns,
            "age_ms": 0.5,
        },
        "command": {
            "observation_sequence": sequence,
            "packet_seq": sequence,
            "packet_stamp_ns": command_stamp_ns,
            "mode": "armed",
            "transport": "udp",
            "send_completed_wall_ns": command_stamp_ns + 1_000_000,
            "send_completed_monotonic_ns": command_send_monotonic_ns,
            "action_key_count": 18,
        },
    }


def write_synthetic_timed_dataset(root: Path) -> None:
    (root / "meta/timing").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "robot_type": "jz_robot_pin_timed",
                "video_encoding": {"codec": "h264", "crf": 18},
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "episode_index": np.asarray([0, 0, 1, 1], dtype=np.int64),
            "frame_index": np.asarray([0, 1, 0, 1], dtype=np.int64),
        }
    ).to_parquet(root / "data/chunk-000/file-000.parquet")

    sequence = 0
    for episode_index in range(2):
        records = []
        for frame_index in range(2):
            sequence += 1
            records.append(make_timing_record(episode_index, frame_index, sequence))
        path = root / f"meta/timing/episode-{episode_index:06d}.jsonl"
        path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )


def read_episode_records(root: Path, episode_index: int) -> tuple[Path, list[dict]]:
    path = root / f"meta/timing/episode-{episode_index:06d}.jsonl"
    return path, [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_episode_records(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def add_valid_source_timing(root: Path) -> None:
    for episode_index in range(2):
        path, records = read_episode_records(root, episode_index)
        for record in records:
            generation = record["state"]["packet_seq"]
            record["state"]["source_timing"] = make_source_timing(generation)
        write_episode_records(path, records)


def replace_rtsp_with_zmq_timing(root: Path) -> None:
    for episode_index in range(2):
        path, records = read_episode_records(root, episode_index)
        for record in records:
            for camera_index, camera_name in enumerate(CAMERAS):
                old = record["cameras"][camera_name]
                sequence = old["decoder_sequence"]
                capture_ns = 9_000_000_000 + sequence * 33_333_333 + camera_index * 1_000
                record["cameras"][camera_name] = {
                    "protocol": "jz_realsense_zmq",
                    "protocol_version": 1,
                    "timestamp_stage": "x86_after_zmq_receive_before_json_decode",
                    "sequence": sequence,
                    "sequence_gap": 0,
                    "receive_wall_ns": old["receive_wall_ns"],
                    "receive_monotonic_ns": old["receive_monotonic_ns"],
                    "decode_completed_monotonic_ns": old["receive_monotonic_ns"] + 1_000_000,
                    "age_ms": old["age_ms"],
                    "reused_by_observation_loop": old["reused_by_observation_loop"],
                    "state_receive_delta_ms": old["state_receive_delta_ms"],
                    "state_receive_skew_ms": old["state_receive_skew_ms"],
                    "camera_timing": {
                        "sequence": sequence,
                        "timestamp_stage": "after_realsense_read_before_jpeg",
                        "capture_wall_ns": 10_000_000_000 + sequence * 33_333_333,
                        "capture_monotonic_ns": capture_ns,
                        "encode_completed_monotonic_ns": capture_ns + 2_000_000,
                        "width": 1280 if camera_name == "camera_head" else 640,
                        "height": 720 if camera_name == "camera_head" else 480,
                        "channels": 3,
                        "pixel_format": "RGB8",
                        "encoding": "jpeg",
                        "jpeg_quality": 95,
                        "payload_bytes": 100_000,
                    },
                }
        write_episode_records(path, records)


def test_timing_checker_accepts_current_schema_and_numpy_parquet_indices(tmp_path: Path) -> None:
    root = tmp_path / "timed_dataset"
    write_synthetic_timed_dataset(root)

    report = run_check(make_args(root))

    assert report["status"] == "PASS", report["errors"]
    assert report["dataset_frames"] == report["timing_frames"] == 4
    assert report["commands"]["sequence"]["count"] == 4
    assert report["commands"]["target_action_receive_to_command_send_ms"]["p95"] == 1.0
    assert report["state_source_timing"] == {
        "present_frames": 0,
        "missing_frames": 4,
        "presence_fraction": 0.0,
        "object_frames": 0,
        "valid_frames": 0,
        "invalid_frames": 0,
        "source_skew_ms": {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "max": None,
        },
        "inferred_snapshot_spread_ns": {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "max": None,
        },
        "sources": {
            source_name: {
                "age_ms": {
                    "count": 0,
                    "min": None,
                    "mean": None,
                    "p50": None,
                    "p95": None,
                    "max": None,
                },
                **(
                    {"zero_header_stamp_frames": 0, "zero_header_stamp_fraction": None}
                    if source_name.endswith("_joints")
                    else {}
                ),
            }
            for source_name in (
                "left_joints",
                "right_joints",
                "left_gripper",
                "right_gripper",
            )
        },
    }
    assert report["state_packets"]["reuse_fraction"] == 0.0
    for camera in report["cameras"].values():
        generation = camera["decoder_pts"]["by_reconnect_generation"]["1"]
        assert generation["episode_segments"] == 2
        assert generation["interval_ms"]["count"] == 2


def test_timing_checker_accepts_zmq_schema_and_reports_source_fps(tmp_path: Path) -> None:
    root = tmp_path / "zmq_timed_dataset"
    write_synthetic_timed_dataset(root)
    replace_rtsp_with_zmq_timing(root)
    args = make_args(root)
    args.expected_camera_protocol = "jz_realsense_zmq"
    args.expected_camera_source_fps = 30.0
    args.min_camera_source_fps_ratio = 0.9

    report = run_check(args)

    assert report["status"] == "PASS", report["errors"]
    for camera in report["cameras"].values():
        assert camera["protocol_counts"] == {"jz_realsense_zmq": 4}
        assert camera["sequence_gap_total"] == 0
        assert camera["source_fps_from_orin_capture"]["mean"] == pytest.approx(30.0, rel=1e-6)
        assert camera["orin_jpeg_encode_ms"]["mean"] == 2.0
        assert camera["x86_jpeg_decode_ms"]["mean"] == 1.0


def test_timing_checker_accepts_policy_output_action_timing(tmp_path: Path) -> None:
    root = tmp_path / "policy_timing_dataset"
    write_synthetic_timed_dataset(root)
    path, records = read_episode_records(root, 0)
    records[0]["action"].update(
        {
            "source": "policy_output",
            "packet_seq": None,
            "packet_stamp_ns": None,
            "age_ms": None,
        }
    )
    write_episode_records(path, records)

    report = run_check(make_args(root))

    assert report["status"] == "PASS", report["errors"]


def test_timing_checker_rejects_command_from_another_observation(tmp_path: Path) -> None:
    root = tmp_path / "bad_command_dataset"
    write_synthetic_timed_dataset(root)
    path = root / "meta/timing/episode-000000.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[0]["command"]["observation_sequence"] = 999
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    report = run_check(make_args(root))

    assert report["status"] == "FAIL"
    assert any("does not match record observation_sequence" in error for error in report["errors"])


def test_timing_checker_allows_observation_sequence_reset_in_new_session(tmp_path: Path) -> None:
    root = tmp_path / "resumed_dataset"
    write_synthetic_timed_dataset(root)
    path = root / "meta/timing/episode-000001.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for frame_index, record in enumerate(records):
        record["session_id"] = "b" * 32
        record["observation_sequence"] = frame_index + 1
        record["command"]["observation_sequence"] = frame_index + 1
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    report = run_check(make_args(root))

    assert report["status"] == "PASS", report["errors"]
    assert report["observation_sessions"]["count"] == 2


def test_timing_checker_accepts_source_timing_v1_across_episode_boundary(tmp_path: Path) -> None:
    root = tmp_path / "source_timing_dataset"
    write_synthetic_timed_dataset(root)
    add_valid_source_timing(root)
    args = make_args(root)
    args.require_source_timing = True

    report = run_check(args)

    assert report["status"] == "PASS", report["errors"]
    assert report["state_source_timing"]["present_frames"] == 4
    assert report["state_source_timing"]["valid_frames"] == 4
    assert report["state_source_timing"]["source_skew_ms"]["max"] == 3.0
    assert report["state_source_timing"]["inferred_snapshot_spread_ns"]["max"] == 0.0
    assert report["state_source_timing"]["sources"]["right_gripper"]["age_ms"]["max"] == 7.0


def test_timing_checker_warns_but_does_not_fail_on_zero_joint_header_stamp(tmp_path: Path) -> None:
    root = tmp_path / "zero_joint_header_dataset"
    write_synthetic_timed_dataset(root)
    add_valid_source_timing(root)
    path, records = read_episode_records(root, 0)
    records[0]["state"]["source_timing"]["sources"]["left_joints"]["header_stamp_ns"] = 0
    write_episode_records(path, records)

    report = run_check(make_args(root))

    assert report["status"] == "PASS", report["errors"]
    left_joints = report["state_source_timing"]["sources"]["left_joints"]
    assert left_joints["zero_header_stamp_frames"] == 1
    assert left_joints["zero_header_stamp_fraction"] == 0.25
    assert any("left_joints has header_stamp_ns=0 in 1/4" in warning for warning in report["warnings"])


def test_timing_checker_validates_source_timing_when_not_required(tmp_path: Path) -> None:
    root = tmp_path / "invalid_source_timing_dataset"
    write_synthetic_timed_dataset(root)
    path, records = read_episode_records(root, 0)
    records[0]["state"]["source_timing"] = {"future_schema": True}
    write_episode_records(path, records)

    report = run_check(make_args(root))

    assert report["status"] == "FAIL"
    assert report["state_source_timing"]["invalid_frames"] == 1
    assert any("not valid source_timing v1" in error for error in report["errors"])


def test_timing_checker_can_require_source_timing_on_every_frame(tmp_path: Path) -> None:
    root = tmp_path / "required_source_timing_dataset"
    write_synthetic_timed_dataset(root)
    args = make_args(root)
    args.require_source_timing = True

    report = run_check(args)

    assert report["status"] == "FAIL"
    assert any("valid state.source_timing v1 object is required" in error for error in report["errors"])


def test_timing_checker_rejects_source_age_skew_and_snapshot_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "bad_source_snapshot_dataset"
    write_synthetic_timed_dataset(root)
    add_valid_source_timing(root)
    path, records = read_episode_records(root, 0)
    source_timing = records[0]["state"]["source_timing"]
    source_timing["source_skew_ms"] = 21.0
    source_timing["sources"]["left_joints"]["age_ms"] = 51.0
    write_episode_records(path, records)

    report = run_check(make_args(root))

    assert report["status"] == "FAIL"
    assert any("does not match source receive timestamps" in error for error in report["errors"])
    assert any("source_skew_ms=21.0 exceeds 20.0 ms" in error for error in report["errors"])
    assert any("left_joints.age_ms=51.0 exceeds 50.0 ms" in error for error in report["errors"])
    assert any("sources imply different snapshot times" in error for error in report["errors"])


def test_timing_checker_snapshot_inference_allows_exactly_one_nanosecond(tmp_path: Path) -> None:
    root = tmp_path / "snapshot_tolerance_dataset"
    write_synthetic_timed_dataset(root)
    add_valid_source_timing(root)
    path, records = read_episode_records(root, 0)
    left_source = records[0]["state"]["source_timing"]["sources"]["left_joints"]
    left_source["age_ms"] += 0.000001
    write_episode_records(path, records)

    report = run_check(make_args(root))

    assert report["status"] == "PASS", report["errors"]
    assert report["state_source_timing"]["inferred_snapshot_spread_ns"]["max"] == 1.0

    left_source["age_ms"] += 0.000001
    write_episode_records(path, records)
    strict_report = run_check(make_args(root))
    assert strict_report["status"] == "FAIL"
    assert any("spread=2 ns exceeds 1.0 ns" in error for error in strict_report["errors"])


def test_timing_checker_allows_identical_state_reuse_and_reports_fraction(tmp_path: Path) -> None:
    root = tmp_path / "reused_state_dataset"
    write_synthetic_timed_dataset(root)
    add_valid_source_timing(root)
    path, records = read_episode_records(root, 0)
    records[1]["state"] = dict(records[0]["state"])
    state_receive_ns = records[1]["state"]["receive_monotonic_ns"]
    for camera in records[1]["cameras"].values():
        delta_ms = (camera["receive_monotonic_ns"] - state_receive_ns) / 1_000_000
        camera["state_receive_delta_ms"] = delta_ms
        camera["state_receive_skew_ms"] = abs(delta_ms)
    write_episode_records(path, records)

    args = make_args(root)
    args.max_camera_state_skew_ms = 110.0
    report = run_check(args)

    assert report["status"] == "PASS", report["errors"]
    assert report["state_packets"]["reused_transitions"] == 1
    assert report["state_packets"]["transition_count"] == 3
    assert report["state_packets"]["reuse_fraction"] == 1 / 3

    args = make_args(root)
    args.max_camera_state_skew_ms = 110.0
    args.max_state_reuse_fraction = 0.3
    strict_report = run_check(args)
    assert strict_report["status"] == "FAIL"
    assert any("state packet reuse fraction" in error for error in strict_report["errors"])


def test_timing_checker_rejects_changed_reused_state(tmp_path: Path) -> None:
    root = tmp_path / "changed_reused_state_dataset"
    write_synthetic_timed_dataset(root)
    add_valid_source_timing(root)
    path, records = read_episode_records(root, 0)
    records[1]["state"]["packet_seq"] = records[0]["state"]["packet_seq"]
    write_episode_records(path, records)

    report = run_check(make_args(root))

    assert report["status"] == "FAIL"
    assert any("reused packet_seq=1 but packet_stamp_ns changed" in error for error in report["errors"])
    assert any("reused packet_seq=1 but source_timing changed" in error for error in report["errors"])


def test_timing_checker_rejects_state_sequence_regression(tmp_path: Path) -> None:
    root = tmp_path / "regressed_state_dataset"
    write_synthetic_timed_dataset(root)
    add_valid_source_timing(root)
    path, records = read_episode_records(root, 1)
    records[0]["state"]["packet_seq"] = 1
    write_episode_records(path, records)

    report = run_check(make_args(root))

    assert report["status"] == "FAIL"
    assert any("packet_seq regressed within session" in error for error in report["errors"])


def test_timing_checker_rejects_nonadvancing_sources_for_new_state_packet(tmp_path: Path) -> None:
    root = tmp_path / "stale_sources_dataset"
    write_synthetic_timed_dataset(root)
    add_valid_source_timing(root)
    path, records = read_episode_records(root, 0)
    previous_sources = records[0]["state"]["source_timing"]["sources"]
    current_sources = records[1]["state"]["source_timing"]["sources"]
    current_sources["left_joints"]["generation"] = previous_sources["left_joints"]["generation"]
    current_sources["right_joints"]["recv_monotonic_ns"] = previous_sources["right_joints"][
        "recv_monotonic_ns"
    ]
    current_sources["right_joints"]["age_ms"] += 100.0
    current_sources["left_joints"]["header_stamp_ns"] = previous_sources["left_joints"]["header_stamp_ns"]
    write_episode_records(path, records)

    args = make_args(root)
    args.max_source_age_ms = 200.0
    args.max_source_skew_ms = 200.0
    report = run_check(args)

    assert report["status"] == "FAIL"
    assert any("left_joints.generation did not strictly advance" in error for error in report["errors"])
    assert any(
        "right_joints.recv_monotonic_ns did not strictly advance" in error for error in report["errors"]
    )
    assert any("left_joints.header_stamp_ns did not strictly advance" in error for error in report["errors"])


def test_timing_checker_segments_state_progression_by_x86_session(tmp_path: Path) -> None:
    root = tmp_path / "resumed_source_timing_dataset"
    write_synthetic_timed_dataset(root)
    add_valid_source_timing(root)
    path, records = read_episode_records(root, 1)
    for frame_index, record in enumerate(records):
        sequence = frame_index + 1
        record["session_id"] = "b" * 32
        record["observation_sequence"] = sequence
        record["command"]["observation_sequence"] = sequence
        record["state"]["packet_seq"] = sequence
        record["state"]["packet_stamp_ns"] = 10_000 + sequence
        record["state"]["source_timing"] = make_source_timing(sequence)
    write_episode_records(path, records)

    args = make_args(root)
    args.require_source_timing = True
    report = run_check(args)

    assert report["status"] == "PASS", report["errors"]
    assert report["observation_sessions"]["count"] == 2
    assert report["state_packets"]["transition_count"] == 2
    assert report["state_packets"]["reuse_fraction"] == 0.0
