#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

DATA_PATH_V21 = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH_V21 = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def ensure_within(path: Path, allowed_root: Path) -> Path:
    resolved = path.expanduser().resolve()
    allowed = allowed_root.expanduser().resolve(strict=True)
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"Write path escapes allowed root: {resolved} (allowed: {allowed})") from exc
    return resolved


def run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=600)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(serializable(row), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def load_episode_records(source_root: Path) -> list[dict[str, Any]]:
    paths = sorted((source_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No v3 episode metadata under {source_root / 'meta' / 'episodes'}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(pq.read_table(path).to_pylist())
    rows.sort(key=lambda row: int(row["episode_index"]))
    actual = [int(row["episode_index"]) for row in rows]
    expected = list(range(len(rows)))
    if actual != expected:
        raise ValueError("Episode indices are not contiguous from zero")
    return rows


def convert_info(
    source_info: dict[str, Any], episode_records: list[dict[str, Any]], video_keys: list[str]
) -> dict[str, Any]:
    info = json.loads(json.dumps(source_info))
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected v3.0 dataset, got {info.get('codebase_version')}")
    total_episodes = len(episode_records)
    chunk_size = int(info.get("chunks_size", 1000))
    info["codebase_version"] = "v2.1"
    info["total_episodes"] = total_episodes
    info["total_chunks"] = math.ceil(total_episodes / chunk_size)
    info["total_videos"] = total_episodes * len(video_keys)
    info["data_path"] = DATA_PATH_V21
    info["video_path"] = VIDEO_PATH_V21 if video_keys else None
    info.pop("data_files_size_in_mb", None)
    info.pop("video_files_size_in_mb", None)
    return info


def convert_tasks(source_root: Path, destination_root: Path) -> list[dict[str, Any]]:
    task_path = source_root / "meta" / "tasks.parquet"
    tasks = sorted(pq.read_table(task_path).to_pylist(), key=lambda row: int(row["task_index"]))
    indices = [int(row["task_index"]) for row in tasks]
    if indices != list(range(len(tasks))):
        raise ValueError(f"Task indices are not contiguous: {indices}")
    normalized = [{"task_index": int(row["task_index"]), "task": str(row["task"])} for row in tasks]
    write_jsonl(destination_root / "meta" / "tasks.jsonl", normalized)
    return normalized


def nested_episode_stats(record: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in record.items():
        if not key.startswith("stats/"):
            continue
        parts = key.split("/")[1:]
        cursor = output
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = serializable(value)
    return output


def convert_episode_metadata(destination_root: Path, records: list[dict[str, Any]]) -> None:
    episodes: list[dict[str, Any]] = []
    episode_stats: list[dict[str, Any]] = []
    for record in records:
        episodes.append(
            {
                "episode_index": int(record["episode_index"]),
                "tasks": [str(task) for task in record["tasks"]],
                "length": int(record["length"]),
            }
        )
        episode_stats.append(
            {
                "episode_index": int(record["episode_index"]),
                "stats": nested_episode_stats(record),
            }
        )
    write_jsonl(destination_root / "meta" / "episodes.jsonl", episodes)
    write_jsonl(destination_root / "meta" / "episodes_stats.jsonl", episode_stats)


def group_by_data_file(records: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(int(record["data/chunk_index"]), int(record["data/file_index"]))].append(record)
    return grouped


def convert_parquet(
    source_root: Path,
    destination_root: Path,
    source_info: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    chunk_size = int(source_info["chunks_size"])
    source_template = source_info["data_path"]
    converted = 0
    for (chunk_index, file_index), group in sorted(group_by_data_file(records).items()):
        source_path = source_root / source_template.format(chunk_index=chunk_index, file_index=file_index)
        source_table = pq.read_table(source_path)
        ordered = sorted(group, key=lambda row: int(row["dataset_from_index"]))
        file_offset = int(ordered[0]["dataset_from_index"])
        for record in ordered:
            episode_index = int(record["episode_index"])
            start = int(record["dataset_from_index"]) - file_offset
            stop = int(record["dataset_to_index"]) - file_offset
            expected_length = int(record["length"])
            if stop - start != expected_length:
                raise ValueError(
                    f"Episode {episode_index}: metadata length {expected_length} != slice {stop - start}"
                )
            episode_table = source_table.slice(start, expected_length)
            destination = destination_root / DATA_PATH_V21.format(
                episode_chunk=episode_index // chunk_size,
                episode_index=episode_index,
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(episode_table, destination)
            round_trip = pq.read_table(destination)
            if not episode_table.equals(round_trip, check_metadata=False):
                raise RuntimeError(f"Parquet round-trip mismatch: {destination}")
            converted += 1
            if converted % 25 == 0 or converted == len(records):
                print(f"[DATA] converted {converted}/{len(records)} episodes", flush=True)


def group_by_video_file(
    records: list[dict[str, Any]], video_key: str
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    chunk_column = f"videos/{video_key}/chunk_index"
    file_column = f"videos/{video_key}/file_index"
    for record in records:
        grouped[(int(record[chunk_column]), int(record[file_column]))].append(record)
    return grouped


def extract_video_segment(source: Path, destination: Path, start: float, end: float) -> None:
    if start < 0 or end <= start:
        raise ValueError(f"Invalid video interval: {start}..{end}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.9f}",
        "-i",
        str(source),
        "-t",
        f"{end - start:.9f}",
        "-map",
        "0:v:0",
        "-an",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "1",
        "-y",
        str(destination),
    ]
    try:
        run_checked(command)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"FFmpeg failed for {source} -> {destination}: {exc.stderr.strip()}") from exc


def convert_videos(
    source_root: Path,
    destination_root: Path,
    source_info: dict[str, Any],
    records: list[dict[str, Any]],
    video_keys: list[str],
) -> None:
    chunk_size = int(source_info["chunks_size"])
    source_template = source_info["video_path"]
    total = len(records) * len(video_keys)
    converted = 0
    for video_key in video_keys:
        for (chunk_index, file_index), group in sorted(group_by_video_file(records, video_key).items()):
            source_path = source_root / source_template.format(
                video_key=video_key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            ordered = sorted(group, key=lambda row: float(row[f"videos/{video_key}/from_timestamp"]))
            for record in ordered:
                episode_index = int(record["episode_index"])
                start = float(record[f"videos/{video_key}/from_timestamp"])
                end = float(record[f"videos/{video_key}/to_timestamp"])
                destination = destination_root / VIDEO_PATH_V21.format(
                    episode_chunk=episode_index // chunk_size,
                    video_key=video_key,
                    episode_index=episode_index,
                )
                extract_video_segment(source_path, destination, start, end)
                converted += 1
                if converted % 25 == 0 or converted == total:
                    print(f"[VIDEO] converted {converted}/{total} episode videos", flush=True)


def probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    result = run_checked(command)
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"Expected one video stream in {path}, got {len(streams)}")
    return streams[0]


def calculate_relative_stats(destination_root: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    horizon = 16
    dimensions = 5
    values_by_offset: list[list[np.ndarray]] = [[] for _ in range(horizon)]
    for record in records:
        episode_index = int(record["episode_index"])
        chunk_size = 1000
        path = destination_root / DATA_PATH_V21.format(
            episode_chunk=episode_index // chunk_size,
            episode_index=episode_index,
        )
        table = pq.read_table(path, columns=["action", "observation.state"])
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        valid = len(actions) - horizon + 1
        if valid <= 0:
            raise ValueError(f"Episode {episode_index} is shorter than horizon {horizon}")
        reference = states[:valid, :dimensions]
        for offset in range(horizon):
            values_by_offset[offset].append(actions[offset : offset + valid, :dimensions] - reference)

    values = np.stack([np.concatenate(offset_values, axis=0) for offset_values in values_by_offset], axis=1)
    if values.shape != (50880, horizon, dimensions):
        raise ValueError(f"Unexpected relative action shape: {values.shape}")
    stats = {
        "single_arm": {
            "max": np.max(values, axis=0),
            "min": np.min(values, axis=0),
            "q01": np.quantile(values, 0.01, axis=0),
            "q99": np.quantile(values, 0.99, axis=0),
            "mean": np.mean(values, axis=0),
            "std": np.std(values, axis=0),
        }
    }
    write_json(destination_root / "meta" / "relative_stats.json", stats)
    return stats


def validate_output(
    destination_root: Path,
    info: dict[str, Any],
    records: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    video_keys: list[str],
) -> dict[str, Any]:
    chunk_size = int(info["chunks_size"])
    total_rows = 0
    task_frame_counts: dict[int, int] = defaultdict(int)
    for record in records:
        episode_index = int(record["episode_index"])
        expected_length = int(record["length"])
        path = destination_root / DATA_PATH_V21.format(
            episode_chunk=episode_index // chunk_size,
            episode_index=episode_index,
        )
        table = pq.read_table(path)
        if table.num_rows != expected_length:
            raise RuntimeError(f"{path}: rows={table.num_rows}, expected={expected_length}")
        action_dims = {len(value) for value in table["action"].to_pylist()}
        state_dims = {len(value) for value in table["observation.state"].to_pylist()}
        if action_dims != {6} or state_dims != {6}:
            raise RuntimeError(f"{path}: action_dims={action_dims}, state_dims={state_dims}")
        task_indices = set(table["task_index"].to_pylist())
        if len(task_indices) != 1:
            raise RuntimeError(f"{path}: expected one task index, got {task_indices}")
        task_index = int(next(iter(task_indices)))
        expected_task = tasks[task_index]["task"]
        if record["tasks"] != [expected_task]:
            raise RuntimeError(
                f"Episode {episode_index}: metadata tasks={record['tasks']} expected={expected_task}"
            )
        total_rows += table.num_rows
        task_frame_counts[task_index] += table.num_rows

    expected_total = int(info["total_frames"])
    if total_rows != expected_total:
        raise RuntimeError(f"Total parquet rows {total_rows} != info total_frames {expected_total}")

    probed = 0
    for video_key in video_keys:
        feature = info["features"][video_key]
        expected_codec = feature["info"]["video.codec"]
        expected_width = int(feature["info"]["video.width"])
        expected_height = int(feature["info"]["video.height"])
        for record in records:
            episode_index = int(record["episode_index"])
            expected_length = int(record["length"])
            path = destination_root / VIDEO_PATH_V21.format(
                episode_chunk=episode_index // chunk_size,
                video_key=video_key,
                episode_index=episode_index,
            )
            stream = probe_video(path)
            if int(stream["nb_frames"]) != expected_length:
                raise RuntimeError(f"{path}: video frames={stream['nb_frames']} expected={expected_length}")
            if stream["codec_name"] != expected_codec:
                raise RuntimeError(f"{path}: codec={stream['codec_name']} expected={expected_codec}")
            if int(stream["width"]) != expected_width or int(stream["height"]) != expected_height:
                raise RuntimeError(f"{path}: unexpected dimensions {stream['width']}x{stream['height']}")
            if stream["avg_frame_rate"] != "30/1":
                raise RuntimeError(f"{path}: unexpected frame rate {stream['avg_frame_rate']}")
            probed += 1
            if probed % 50 == 0 or probed == len(records) * len(video_keys):
                print(f"[VERIFY] probed {probed}/{len(records) * len(video_keys)} videos", flush=True)

    return {
        "episodes": len(records),
        "tasks": len(tasks),
        "parquet_rows": total_rows,
        "videos": probed,
        "task_frame_counts": dict(sorted(task_frame_counts.items())),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-destructive LeRobot v3 to GR00T v2.1 converter.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allowed-write-root", type=Path, required=True)
    parser.add_argument("--modality-json", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve(strict=True)
    allowed_root = args.allowed_write_root.expanduser().resolve(strict=True)
    output_root = ensure_within(args.output_root, allowed_root)
    report_path = ensure_within(args.report_path, allowed_root)
    modality_path = args.modality_json.expanduser().resolve(strict=True)

    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite converted dataset: {output_root}")
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite conversion report: {report_path}")
    try:
        output_root.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ValueError("Output root must not be inside source root")

    partial_root = output_root.parent / f".{output_root.name}.partial-{os.getpid()}"
    ensure_within(partial_root, allowed_root)
    if partial_root.exists():
        raise FileExistsError(partial_root)
    partial_root.mkdir(parents=True)
    print(f"[SAFE] source is read-only input: {source_root}")
    print(f"[SAFE] writing partial output: {partial_root}")

    try:
        source_info = load_json(source_root / "meta" / "info.json")
        records = load_episode_records(source_root)
        video_keys = [
            key for key, feature in source_info["features"].items() if feature.get("dtype") == "video"
        ]
        output_info = convert_info(source_info, records, video_keys)
        write_json(partial_root / "meta" / "info.json", output_info)
        tasks = convert_tasks(source_root, partial_root)
        convert_episode_metadata(partial_root, records)
        shutil.copy2(source_root / "meta" / "stats.json", partial_root / "meta" / "stats.json")
        shutil.copy2(modality_path, partial_root / "meta" / "modality.json")
        convert_parquet(source_root, partial_root, source_info, records)
        convert_videos(source_root, partial_root, source_info, records, video_keys)
        relative_stats = calculate_relative_stats(partial_root, records)
        validation = validate_output(partial_root, output_info, records, tasks, video_keys)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        partial_root.rename(output_root)
        report = {
            "schema_version": 1,
            "status": "passed",
            "source_root": str(source_root),
            "output_root": str(output_root),
            "modality_sha256": file_sha256(modality_path),
            "action_horizon": 16,
            "relative_stats_shape": {
                key: list(np.asarray(value["mean"]).shape) for key, value in relative_stats.items()
            },
            "validation": validation,
        }
        write_json(report_path, report)
        print(f"[OK] published converted dataset: {output_root}")
        print(f"[OK] wrote conversion report: {report_path}")
    except Exception:
        print(f"[ERROR] Conversion failed; partial output retained for inspection: {partial_root}")
        raise


if __name__ == "__main__":
    main()
