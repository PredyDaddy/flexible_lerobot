#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

DATA_PATH_V21 = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH_V21 = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_v3_episodes(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        records.extend(pq.read_table(path).to_pylist())
    return sorted(records, key=lambda row: int(row["episode_index"]))


def source_episode_table(
    source_root: Path,
    source_info: dict[str, Any],
    record: dict[str, Any],
    cache: dict[tuple[int, int], tuple[int, Any]],
):
    key = (int(record["data/chunk_index"]), int(record["data/file_index"]))
    if key not in cache:
        path = source_root / source_info["data_path"].format(chunk_index=key[0], file_index=key[1])
        same_file_records = [
            row
            for row in load_v3_episodes(source_root)
            if (int(row["data/chunk_index"]), int(row["data/file_index"])) == key
        ]
        offset = min(int(row["dataset_from_index"]) for row in same_file_records)
        cache[key] = (offset, pq.read_table(path))
    offset, table = cache[key]
    start = int(record["dataset_from_index"]) - offset
    return table.slice(start, int(record["length"]))


def probe_counted_frames(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_frames,nb_read_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=600)
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"Expected one stream in {path}, got {len(streams)}")
    return streams[0]


def recompute_relative_values(dataset_root: Path, records: list[dict[str, Any]]) -> np.ndarray:
    horizon = 16
    by_offset: list[list[np.ndarray]] = [[] for _ in range(horizon)]
    for record in records:
        episode_index = int(record["episode_index"])
        path = dataset_root / DATA_PATH_V21.format(
            episode_chunk=episode_index // 1000,
            episode_index=episode_index,
        )
        table = pq.read_table(path, columns=["action", "observation.state"])
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        valid = len(actions) - horizon + 1
        reference = states[:valid, :5]
        for offset in range(horizon):
            by_offset[offset].append(actions[offset : offset + valid, :5] - reference)
    return np.stack([np.concatenate(values, axis=0) for values in by_offset], axis=1)


def validate_relative_stats(dataset_root: Path, records: list[dict[str, Any]]) -> None:
    values = recompute_relative_values(dataset_root, records)
    expected_shape = (50880, 16, 5)
    if values.shape != expected_shape:
        raise RuntimeError(f"Relative action values have shape {values.shape}, expected {expected_shape}")
    stored = load_json(dataset_root / "meta" / "relative_stats.json")["single_arm"]
    calculated = {
        "max": np.max(values, axis=0),
        "min": np.min(values, axis=0),
        "q01": np.quantile(values, 0.01, axis=0),
        "q99": np.quantile(values, 0.99, axis=0),
        "mean": np.mean(values, axis=0),
        "std": np.std(values, axis=0),
    }
    for key, expected in calculated.items():
        actual = np.asarray(stored[key])
        if actual.shape != (16, 5):
            raise RuntimeError(f"relative_stats[{key}] shape={actual.shape}, expected=(16, 5)")
        if not np.allclose(actual, expected, rtol=1e-6, atol=1e-6):
            difference = float(np.max(np.abs(actual - expected)))
            raise RuntimeError(f"relative_stats[{key}] mismatch, max abs diff={difference}")


def validate_torchcodec_pixels(
    source_root: Path,
    output_root: Path,
    source_info: dict[str, Any],
    records: list[dict[str, Any]],
    video_keys: list[str],
) -> int:
    import torchcodec

    source_decoders: dict[Path, Any] = {}
    checked = 0
    for video_key in video_keys:
        for record in records:
            episode_index = int(record["episode_index"])
            expected_length = int(record["length"])
            source_path = source_root / source_info["video_path"].format(
                video_key=video_key,
                chunk_index=int(record[f"videos/{video_key}/chunk_index"]),
                file_index=int(record[f"videos/{video_key}/file_index"]),
            )
            output_path = output_root / VIDEO_PATH_V21.format(
                episode_chunk=episode_index // 1000,
                video_key=video_key,
                episode_index=episode_index,
            )
            if source_path not in source_decoders:
                source_decoders[source_path] = torchcodec.decoders.VideoDecoder(
                    str(source_path), device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
                )
            source_decoder = source_decoders[source_path]
            output_decoder = torchcodec.decoders.VideoDecoder(
                str(output_path), device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
            )
            if len(output_decoder) != expected_length:
                raise RuntimeError(
                    f"TorchCodec length {len(output_decoder)} != {expected_length}: {output_path}"
                )
            local_indices = np.asarray(sorted({0, expected_length // 2, expected_length - 1}))
            source_start = round(float(record[f"videos/{video_key}/from_timestamp"]) * 30)
            source_indices = local_indices + source_start
            source_frames = source_decoder.get_frames_at(indices=source_indices).data.numpy()
            output_frames = output_decoder.get_frames_at(indices=local_indices).data.numpy()
            if source_frames.shape != output_frames.shape:
                raise RuntimeError(
                    f"Frame shape mismatch {source_frames.shape} != {output_frames.shape}: {output_path}"
                )
            if source_frames.dtype != np.uint8 or output_frames.dtype != np.uint8:
                raise RuntimeError(f"Expected uint8 frames, got {source_frames.dtype}/{output_frames.dtype}")
            if not np.array_equal(source_frames, output_frames):
                max_difference = int(
                    np.max(np.abs(source_frames.astype(np.int16) - output_frames.astype(np.int16)))
                )
                raise RuntimeError(
                    f"Decoded pixels differ from v3 source (max diff={max_difference}): {output_path}"
                )
            checked += len(local_indices)
            if checked % 150 < len(local_indices):
                print(f"[TORCHCODEC] compared {checked} source/output frames", flush=True)
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent full validation of converted GR00T data.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--skip-torchcodec", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    report_path = args.report_path.expanduser().resolve()
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite validation report: {report_path}")

    source_info = load_json(source_root / "meta" / "info.json")
    output_info = load_json(dataset_root / "meta" / "info.json")
    records = load_v3_episodes(source_root)
    episodes = load_jsonl(dataset_root / "meta" / "episodes.jsonl")
    source_tasks = sorted(
        pq.read_table(source_root / "meta" / "tasks.parquet").to_pylist(),
        key=lambda row: int(row["task_index"]),
    )
    output_tasks = load_jsonl(dataset_root / "meta" / "tasks.jsonl")
    video_keys = [key for key, feature in source_info["features"].items() if feature.get("dtype") == "video"]

    if output_info["codebase_version"] != "v2.1":
        raise RuntimeError(f"Unexpected output version: {output_info['codebase_version']}")
    if output_tasks != source_tasks:
        raise RuntimeError("tasks.jsonl differs from source tasks.parquet")
    if not (len(episodes) == len(records) == int(source_info["total_episodes"])):
        raise RuntimeError("Episode count mismatch")

    source_cache: dict[tuple[int, int], tuple[int, Any]] = {}
    task_frames: dict[int, int] = defaultdict(int)
    parquet_rows = 0
    for index, (record, episode) in enumerate(zip(records, episodes, strict=True)):
        expected_episode = {
            "episode_index": int(record["episode_index"]),
            "tasks": record["tasks"],
            "length": int(record["length"]),
        }
        if episode != expected_episode:
            raise RuntimeError(f"Episode metadata mismatch at index {index}")
        episode_index = int(record["episode_index"])
        output_path = dataset_root / DATA_PATH_V21.format(
            episode_chunk=episode_index // int(output_info["chunks_size"]),
            episode_index=episode_index,
        )
        source_table = source_episode_table(source_root, source_info, record, source_cache)
        output_table = pq.read_table(output_path)
        if not source_table.equals(output_table, check_metadata=False):
            raise RuntimeError(f"Converted parquet differs from source: {output_path}")
        timestamps = np.asarray(output_table["timestamp"].to_pylist(), dtype=np.float64)
        if timestamps[0] != 0 or np.any(np.diff(timestamps) <= 0):
            raise RuntimeError(f"Invalid timestamps: {output_path}")
        frame_indices = output_table["frame_index"].to_pylist()
        if frame_indices != list(range(len(frame_indices))):
            raise RuntimeError(f"Invalid frame indices: {output_path}")
        task_index = int(output_table["task_index"][0].as_py())
        task_frames[task_index] += output_table.num_rows
        parquet_rows += output_table.num_rows
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            print(f"[PARQUET] exact match {index + 1}/{len(records)} episodes", flush=True)

    counted_videos = 0
    for video_key in video_keys:
        for record in records:
            episode_index = int(record["episode_index"])
            path = dataset_root / VIDEO_PATH_V21.format(
                episode_chunk=episode_index // int(output_info["chunks_size"]),
                video_key=video_key,
                episode_index=episode_index,
            )
            stream = probe_counted_frames(path)
            expected = int(record["length"])
            if int(stream["nb_frames"]) != expected or int(stream["nb_read_frames"]) != expected:
                raise RuntimeError(
                    f"Frame count mismatch in {path}: metadata={stream['nb_frames']} "
                    f"decoded={stream['nb_read_frames']} expected={expected}"
                )
            if stream["codec_name"] != "av1" or stream["avg_frame_rate"] != "30/1":
                raise RuntimeError(f"Unexpected video stream metadata: {path}: {stream}")
            counted_videos += 1
            if counted_videos % 50 == 0 or counted_videos == len(records) * len(video_keys):
                print(
                    f"[FFPROBE] fully counted {counted_videos}/{len(records) * len(video_keys)} videos",
                    flush=True,
                )

    validate_relative_stats(dataset_root, records)
    torchcodec_frames = 0
    if not args.skip_torchcodec:
        torchcodec_frames = validate_torchcodec_pixels(
            source_root, dataset_root, source_info, records, video_keys
        )

    report = {
        "schema_version": 1,
        "source_root": str(source_root),
        "dataset_root": str(dataset_root),
        "episodes": len(records),
        "parquet_rows_exactly_compared": parquet_rows,
        "fully_counted_videos": counted_videos,
        "torchcodec_source_output_frames_exactly_compared": torchcodec_frames,
        "task_frame_counts": dict(sorted(task_frames.items())),
        "relative_action_values": [50880, 16, 5],
        "status": "passed",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[OK] wrote validation report: {report_path}")


if __name__ == "__main__":
    main()
