#!/usr/bin/env python

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

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
        report_json=None,
    )


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


def test_timing_checker_accepts_current_schema_and_numpy_parquet_indices(tmp_path: Path) -> None:
    root = tmp_path / "timed_dataset"
    write_synthetic_timed_dataset(root)

    report = run_check(make_args(root))

    assert report["status"] == "PASS", report["errors"]
    assert report["dataset_frames"] == report["timing_frames"] == 4
    assert report["commands"]["sequence"]["count"] == 4
    assert report["commands"]["target_action_receive_to_command_send_ms"]["p95"] == 1.0
    for camera in report["cameras"].values():
        generation = camera["decoder_pts"]["by_reconnect_generation"]["1"]
        assert generation["episode_segments"] == 2
        assert generation["interval_ms"]["count"] == 2


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
