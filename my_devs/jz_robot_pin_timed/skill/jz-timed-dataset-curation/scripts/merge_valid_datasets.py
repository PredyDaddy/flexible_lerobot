#!/usr/bin/env python

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lerobot.datasets.dataset_tools import (
    _copy_and_reindex_data,
    _copy_and_reindex_episodes_metadata,
    merge_datasets,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.robots.jz_robot_pin_timed.training_schema import JZPinTrainingSchema

CAMERA_PREFIX = "observation.images."
TIMING_FILENAME = "episode-{episode_index:06d}.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge complete, previously quality-approved JZ Pin timed datasets without modifying sources. "
            "The script restores JZ timing/schema metadata omitted by LeRobot's generic merge."
        )
    )
    parser.add_argument("--source-root", action="append", type=Path, required=True)
    parser.add_argument(
        "--source-episodes",
        action="append",
        default=[],
        metavar="ROOT=SPEC",
        help=(
            "Select saved episodes from one source without re-encoding shared MP4 files. "
            "SPEC accepts comma-separated indexes/ranges such as 0-1,4,7-9. "
            "Sources without this option use every saved episode."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-repo-id", required=True)
    parser.add_argument("--expected-codec", default="h264")
    parser.add_argument("--expected-crf", type=int, default=18)
    parser.add_argument("--sample-frames-per-episode", type=int, default=3)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args()


def parse_episode_spec(spec: str) -> list[int]:
    episodes: set[int] = set()
    for raw_token in spec.split(","):
        token = raw_token.strip()
        if not token:
            raise ValueError(f"Empty episode token in selection: {spec!r}")
        if "-" in token:
            start_text, end_text = token.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)
            if start < 0 or end < start:
                raise ValueError(f"Invalid episode range {token!r}")
            episodes.update(range(start, end + 1))
        else:
            episode_index = int(token)
            if episode_index < 0:
                raise ValueError(f"Episode index must be non-negative: {token!r}")
            episodes.add(episode_index)
    if not episodes:
        raise ValueError("Episode selection cannot be empty")
    return sorted(episodes)


def parse_source_episode_selections(values: list[str]) -> dict[Path, list[int]]:
    selections: dict[Path, list[int]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--source-episodes must use ROOT=SPEC, got {value!r}")
        root_text, spec = value.rsplit("=", maxsplit=1)
        root = Path(root_text).expanduser().resolve()
        if root in selections:
            raise ValueError(f"Duplicate --source-episodes entry for {root}")
        selections[root] = parse_episode_spec(spec)
    return selections


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_data(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No data parquet files found under {root}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def dataset_frame_keys(data: pd.DataFrame) -> set[tuple[int, int]]:
    if "episode_index" not in data or "frame_index" not in data:
        raise ValueError("Data parquet lacks episode_index/frame_index")
    keys = [
        (int(episode_index), int(frame_index))
        for episode_index, frame_index in zip(data["episode_index"], data["frame_index"], strict=True)
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Data parquet contains duplicate episode/frame keys")
    return set(keys)


def validate_data_grid(data: pd.DataFrame, total_episodes: int) -> dict[int, list[int]]:
    episode_frames: dict[int, list[int]] = {}
    actual_episodes = sorted(int(value) for value in data["episode_index"].unique())
    expected_episodes = list(range(total_episodes))
    if actual_episodes != expected_episodes:
        raise ValueError(f"Data episode ids must be {expected_episodes}, got {actual_episodes}")
    for episode_index, episode in data.groupby("episode_index"):
        frames = sorted(int(value) for value in episode["frame_index"])
        if frames != list(range(len(frames))):
            raise ValueError(f"Episode {episode_index} frame_index is not contiguous from zero")
        episode_frames[int(episode_index)] = frames
    return episode_frames


def read_timing(root: Path) -> tuple[dict[tuple[int, int], dict[str, Any]], list[Path]]:
    records: dict[tuple[int, int], dict[str, Any]] = {}
    paths = sorted((root / "meta" / "timing").glob("episode-*.jsonl"))
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                if not raw_line.strip():
                    raise ValueError(f"Blank timing line: {path}:{line_number}")
                record = json.loads(raw_line)
                key = (int(record["episode_index"]), int(record["frame_index"]))
                if key in records:
                    raise ValueError(f"Duplicate timing key {key} in {path}")
                records[key] = record
    return records, paths


def ffprobe_video(path: Path, expected_codec: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,avg_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected exactly one video stream in {path}")
    stream = streams[0]
    if stream.get("codec_name") != expected_codec:
        raise ValueError(
            f"Video codec mismatch for {path}: expected {expected_codec}, got {stream.get('codec_name')}"
        )
    duration = float(payload.get("format", {}).get("duration", 0.0))
    if duration <= 0:
        raise ValueError(f"Video has non-positive duration: {path}")
    return {**stream, "duration": duration, "bytes": path.stat().st_size}


def episodes_by_index(dataset: LeRobotDataset) -> dict[int, dict[str, Any]]:
    result = {int(episode["episode_index"]): episode for episode in dataset.meta.episodes}
    if len(result) != len(dataset.meta.episodes):
        raise ValueError(f"Duplicate standard episode metadata ids in {dataset.root}")
    return result


def referenced_video_paths(dataset: LeRobotDataset) -> dict[str, list[Path]]:
    episodes = episodes_by_index(dataset)
    result: dict[str, list[Path]] = {}
    for camera_key in dataset.meta.video_keys:
        pairs = {
            (
                int(episode[f"videos/{camera_key}/chunk_index"]),
                int(episode[f"videos/{camera_key}/file_index"]),
            )
            for episode in episodes.values()
        }
        result[camera_key] = [
            dataset.root / "videos" / camera_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
            for chunk_index, file_index in sorted(pairs)
        ]
    return result


def stage_selected_dataset(
    src_dataset: LeRobotDataset,
    source_info: dict[str, Any],
    selected_episodes: list[int],
    stage_root: Path,
) -> LeRobotDataset:
    """Create a temporary standard dataset while copying shared MP4 files byte-for-byte.

    Generic split/delete re-encodes a mixed shared MP4 to AV1.  Curated JZ timed data must retain
    H.264/CRF18, so this staging path filters parquet/meta rows but copies every referenced source
    video file whole.  Unselected tail frames may remain physically present, while only selected
    timestamp ranges are referenced by standard episode metadata.
    """

    if stage_root.exists():
        raise FileExistsError(f"Selection staging root already exists: {stage_root}")
    episode_mapping = {
        old_episode_index: new_episode_index
        for new_episode_index, old_episode_index in enumerate(selected_episodes)
    }
    stage_repo_id = f"local/{src_dataset.root.name}_selected_{len(selected_episodes)}eps"
    stage_meta = LeRobotDatasetMetadata.create(
        repo_id=stage_repo_id,
        fps=src_dataset.meta.fps,
        features=src_dataset.meta.features,
        robot_type=src_dataset.meta.robot_type,
        root=stage_root,
        use_videos=bool(src_dataset.meta.video_keys),
        chunks_size=src_dataset.meta.chunks_size,
        data_files_size_in_mb=src_dataset.meta.data_files_size_in_mb,
        video_files_size_in_mb=src_dataset.meta.video_files_size_in_mb,
    )

    video_metadata: dict[int, dict[str, Any]] = {
        new_episode_index: {} for new_episode_index in episode_mapping.values()
    }
    copied_video_paths: set[Path] = set()
    if src_dataset.meta.video_path is None and src_dataset.meta.video_keys:
        raise ValueError(f"Source has video keys but no video_path template: {src_dataset.root}")
    for old_episode_index, new_episode_index in episode_mapping.items():
        source_episode = src_dataset.meta.episodes[old_episode_index]
        for camera_key in src_dataset.meta.video_keys:
            prefix = f"videos/{camera_key}"
            chunk_index = source_episode.get(f"{prefix}/chunk_index")
            file_index = source_episode.get(f"{prefix}/file_index")
            from_timestamp = source_episode.get(f"{prefix}/from_timestamp")
            to_timestamp = source_episode.get(f"{prefix}/to_timestamp")
            if None in (chunk_index, file_index, from_timestamp, to_timestamp):
                raise ValueError(
                    f"Selected episode {old_episode_index} has incomplete {camera_key} video metadata"
                )
            metadata = video_metadata[new_episode_index]
            metadata[f"{prefix}/chunk_index"] = int(chunk_index)
            metadata[f"{prefix}/file_index"] = int(file_index)
            metadata[f"{prefix}/from_timestamp"] = float(from_timestamp)
            metadata[f"{prefix}/to_timestamp"] = float(to_timestamp)

            assert src_dataset.meta.video_path is not None
            relative_path = Path(
                src_dataset.meta.video_path.format(
                    video_key=camera_key,
                    chunk_index=int(chunk_index),
                    file_index=int(file_index),
                )
            )
            if relative_path in copied_video_paths:
                continue
            source_path = src_dataset.root / relative_path
            if not source_path.is_file():
                raise FileNotFoundError(f"Selected episode video is missing: {source_path}")
            destination_path = stage_root / relative_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)
            copied_video_paths.add(relative_path)

    data_metadata = _copy_and_reindex_data(src_dataset, stage_meta, episode_mapping)
    _copy_and_reindex_episodes_metadata(
        src_dataset,
        stage_meta,
        episode_mapping,
        data_metadata,
        video_metadata,
    )
    stage_info_path = stage_root / "meta" / "info.json"
    stage_info = read_json(stage_info_path)
    if "video_encoding" in source_info:
        stage_info["video_encoding"] = copy.deepcopy(source_info["video_encoding"])
    with stage_info_path.open("w", encoding="utf-8") as stream:
        json.dump(stage_info, stream, indent=4, ensure_ascii=False)
        stream.write("\n")
    return LeRobotDataset(stage_repo_id, root=stage_root)


def sample_video_frames(dataset: LeRobotDataset, samples_per_episode: int) -> dict[str, Any]:
    if samples_per_episode <= 0:
        return {"samples": 0, "camera_keys": list(dataset.meta.video_keys)}
    sampled = 0
    per_camera: dict[str, dict[str, Any]] = {
        key: {"samples": 0, "min": None, "max": None} for key in dataset.meta.video_keys
    }
    episodes = episodes_by_index(dataset)
    for episode_index in range(dataset.meta.total_episodes):
        episode = episodes[episode_index]
        start = int(episode["dataset_from_index"])
        stop = int(episode["dataset_to_index"])
        if stop <= start:
            raise ValueError(f"Episode {episode_index} has an empty metadata range")
        indexes = np.linspace(start, stop - 1, num=samples_per_episode, dtype=np.int64)
        for index in sorted({int(value) for value in indexes}):
            frame = dataset[index]
            for camera_key in dataset.meta.video_keys:
                image = frame[camera_key]
                if hasattr(image, "detach"):
                    image = image.detach().cpu().numpy()
                image = np.asarray(image)
                expected = dataset.meta.features[camera_key]["shape"]
                if image.shape != (expected[2], expected[0], expected[1]):
                    raise ValueError(
                        f"Decoded {camera_key} shape mismatch at dataset index {index}: "
                        f"expected {(expected[2], expected[0], expected[1])}, got {image.shape}"
                    )
                if not np.isfinite(image).all():
                    raise ValueError(f"Decoded {camera_key} contains non-finite pixels at index {index}")
                summary = per_camera[camera_key]
                minimum = float(image.min())
                maximum = float(image.max())
                summary["min"] = minimum if summary["min"] is None else min(summary["min"], minimum)
                summary["max"] = maximum if summary["max"] is None else max(summary["max"], maximum)
                summary["samples"] += 1
            sampled += 1
    return {"samples": sampled, "cameras": per_camera}


def inspect_source(
    root: Path,
    expected_codec: str,
    expected_crf: int,
    samples: int,
    selected_episodes: list[int] | None = None,
    stage_root: Path | None = None,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    info = read_json(root / "meta" / "info.json")
    total_episodes = int(info.get("total_episodes", 0))
    if total_episodes <= 0:
        raise ValueError(f"Source has no completed episodes: {root}")
    encoding = info.get("video_encoding")
    if encoding != {"codec": expected_codec, "crf": expected_crf}:
        raise ValueError(
            f"Source video_encoding mismatch for {root}: expected "
            f"{{'codec': {expected_codec!r}, 'crf': {expected_crf}}}, got {encoding!r}"
        )
    all_episode_indices = list(range(total_episodes))
    selected_episodes = all_episode_indices if selected_episodes is None else selected_episodes
    invalid_selected = sorted(set(selected_episodes) - set(all_episode_indices))
    if invalid_selected:
        raise ValueError(f"Selected episodes do not exist in {root}: {invalid_selected}")
    if not selected_episodes:
        raise ValueError(f"Episode selection is empty for {root}")
    if selected_episodes != sorted(set(selected_episodes)):
        raise ValueError(f"Selected episodes must be unique and sorted for {root}: {selected_episodes}")
    partial_selection = selected_episodes != all_episode_indices
    source_dataset = LeRobotDataset(
        f"local/{root.name}",
        root=root,
        episodes=selected_episodes if partial_selection else None,
    )
    expected_episode_ids = list(range(total_episodes))
    standard_episodes = episodes_by_index(source_dataset)
    if sorted(standard_episodes) != expected_episode_ids:
        raise ValueError(
            f"Standard episode metadata ids must be {expected_episode_ids}, "
            f"got {sorted(standard_episodes)} in {root}"
        )
    source_data = read_data(root)
    validate_data_grid(source_data, total_episodes)
    if len(source_data) != int(info.get("total_frames", -1)):
        raise ValueError(f"Data frame count differs from meta/info.json in {root}")
    source_keys = dataset_frame_keys(source_data)
    source_timing, timing_paths = read_timing(root)
    selected_source_keys = {key for key in source_keys if key[0] in set(selected_episodes)}
    missing_timing = sorted(selected_source_keys - set(source_timing))
    if missing_timing:
        raise ValueError(f"Source is missing timing records; first keys={missing_timing[:10]}")
    ignored_timing = sorted(set(source_timing) - selected_source_keys)

    if partial_selection:
        if stage_root is None:
            raise ValueError(f"A staging root is required for partial source {root}")
        dataset = stage_selected_dataset(source_dataset, info, selected_episodes, stage_root)
        data = read_data(stage_root)
        episode_frames = validate_data_grid(data, len(selected_episodes))
        keys = dataset_frame_keys(data)
        old_to_new = {
            old_episode_index: new_episode_index
            for new_episode_index, old_episode_index in enumerate(selected_episodes)
        }
        timing = {
            (old_to_new[old_episode_index], frame_index): record
            for (old_episode_index, frame_index), record in source_timing.items()
            if old_episode_index in old_to_new
        }
    else:
        dataset = source_dataset
        data = source_data
        episode_frames = validate_data_grid(data, total_episodes)
        keys = source_keys
        timing = {key: source_timing[key] for key in keys}
    if set(timing) != keys:
        raise ValueError(
            f"Selected timing/data keys differ for {root}: "
            f"missing={len(keys - set(timing))}, extra={len(set(timing) - keys)}"
        )

    schema_path = root / "meta" / "jz_pin_training_schema.json"
    schema = JZPinTrainingSchema.from_file(schema_path)
    schema.ensure_trainable()
    schema.validate_raw_features(info["features"])

    video_probe: dict[str, list[dict[str, Any]]] = {}
    for camera_key, paths in referenced_video_paths(dataset).items():
        video_probe[camera_key] = [
            {"path": str(path), **ffprobe_video(path, expected_codec)} for path in paths
        ]
    decoded = sample_video_frames(dataset, samples)
    return {
        "root": str(root),
        "name": root.name,
        "repo_id": dataset.repo_id,
        "episodes": len(selected_episodes),
        "frames": len(data),
        "source_total_episodes": total_episodes,
        "source_episode_indices": selected_episodes,
        "partial_selection": partial_selection,
        "episode_frames": {str(key): len(value) for key, value in episode_frames.items()},
        "data_keys": keys,
        "timing": timing,
        "timing_files": len(timing_paths),
        "extra_timing_records_ignored": len(ignored_timing),
        "extra_timing_first_keys": [list(key) for key in ignored_timing[:10]],
        "schema": schema,
        "schema_payload": read_json(schema_path),
        "video_probe": video_probe,
        "decoded_video_samples": decoded,
        "dataset": dataset,
    }


def public_source_report(source: dict[str, Any], episode_offset: int) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in source.items()
        if key
        not in {
            "data_keys",
            "timing",
            "schema",
            "schema_payload",
            "dataset",
        }
    } | {
        "merged_episode_from": episode_offset,
        "merged_episode_to_exclusive": episode_offset + source["episodes"],
    }


def write_timing(sources: list[dict[str, Any]], output_root: Path) -> list[dict[str, Any]]:
    timing_dir = output_root / "meta" / "timing"
    timing_dir.mkdir(parents=True, exist_ok=False)
    episode_map: list[dict[str, Any]] = []
    episode_offset = 0
    for source in sources:
        for old_episode_index in range(source["episodes"]):
            new_episode_index = episode_offset + old_episode_index
            source_episode_index = source["source_episode_indices"][old_episode_index]
            keys = sorted(key for key in source["data_keys"] if key[0] == old_episode_index)
            output_path = timing_dir / TIMING_FILENAME.format(episode_index=new_episode_index)
            with output_path.open("x", encoding="utf-8") as stream:
                for key in keys:
                    record = copy.deepcopy(source["timing"][key])
                    record["episode_index"] = new_episode_index
                    stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            episode_map.append(
                {
                    "merged_episode_index": new_episode_index,
                    "source_root": source["root"],
                    "source_episode_index": source_episode_index,
                    "frames": len(keys),
                }
            )
        episode_offset += source["episodes"]
    return episode_map


def write_jz_metadata(
    sources: list[dict[str, Any]],
    output_root: Path,
    output_repo_id: str,
    expected_codec: str,
    expected_crf: int,
) -> list[dict[str, Any]]:
    first_schema = sources[0]["schema"]
    for source in sources[1:]:
        if source["schema"].semantic_dict() != first_schema.semantic_dict():
            raise ValueError(f"Training schema semantics differ: {source['root']}")

    info_path = output_root / "meta" / "info.json"
    info = read_json(info_path)
    info["video_encoding"] = {"codec": expected_codec, "crf": expected_crf}
    with info_path.open("w", encoding="utf-8") as stream:
        json.dump(info, stream, indent=4, ensure_ascii=False)
        stream.write("\n")

    episode_map = write_timing(sources, output_root)
    schema_payload = first_schema.to_dict()
    schema_payload["provenance"] = {
        "kind": "curated_merge",
        "output_repo_id": output_repo_id,
        "sources": [
            {
                "root": source["root"],
                "episode_indices": source["source_episode_indices"],
            }
            for source in sources
        ],
        "original_raw18_preserved": True,
    }
    schema_path = output_root / "meta" / "jz_pin_training_schema.json"
    with schema_path.open("x", encoding="utf-8") as stream:
        json.dump(schema_payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return episode_map


def verify_output(
    output_root: Path,
    output_repo_id: str,
    expected_episodes: int,
    expected_frames: int,
    expected_codec: str,
    samples: int,
) -> dict[str, Any]:
    dataset = LeRobotDataset(output_repo_id, root=output_root)
    if dataset.meta.total_episodes != expected_episodes or len(dataset) != expected_frames:
        raise ValueError(
            f"Merged counts mismatch: episodes={dataset.meta.total_episodes}/{expected_episodes}, "
            f"frames={len(dataset)}/{expected_frames}"
        )
    data = read_data(output_root)
    keys = dataset_frame_keys(data)
    timing, _paths = read_timing(output_root)
    if set(timing) != keys:
        raise ValueError(
            f"Merged timing keys differ from data: missing={len(keys - set(timing))}, "
            f"extra={len(set(timing) - keys)}"
        )
    video_probe: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for camera_key, paths in referenced_video_paths(dataset).items():
        for path in paths:
            video_probe[camera_key].append({"path": str(path), **ffprobe_video(path, expected_codec)})
    decoded = sample_video_frames(dataset, samples)
    schema = JZPinTrainingSchema.from_file(output_root / "meta" / "jz_pin_training_schema.json")
    schema.ensure_trainable()
    schema.validate_raw_features(dataset.meta.features)
    return {
        "episodes": dataset.meta.total_episodes,
        "frames": len(dataset),
        "timing_records": len(timing),
        "video_probe": dict(video_probe),
        "decoded_video_samples": decoded,
        "training_observation_sources": schema.observation_sources,
    }


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    if args.sample_frames_per_episode < 0:
        raise ValueError("--sample-frames-per-episode must be non-negative")
    if not args.preflight_only and output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_root}")
    source_roots = [root.expanduser().resolve() for root in args.source_root]
    if len(source_roots) != len(set(source_roots)):
        raise ValueError("--source-root entries must be unique")
    selections = parse_source_episode_selections(args.source_episodes)
    unknown_selection_roots = sorted(set(selections) - set(source_roots))
    if unknown_selection_roots:
        raise ValueError(
            f"--source-episodes refers to roots not present in --source-root: {unknown_selection_roots}"
        )

    temporary_stages: list[tempfile.TemporaryDirectory[str]] = []
    try:
        sources: list[dict[str, Any]] = []
        for source_root in source_roots:
            selected_episodes = selections.get(source_root)
            stage_root = None
            if selected_episodes is not None:
                temporary_stage = tempfile.TemporaryDirectory(prefix=f"jz_timed_selected_{source_root.name}_")
                temporary_stages.append(temporary_stage)
                stage_root = Path(temporary_stage.name) / "dataset"
            sources.append(
                inspect_source(
                    source_root,
                    args.expected_codec,
                    args.expected_crf,
                    args.sample_frames_per_episode,
                    selected_episodes=selected_episodes,
                    stage_root=stage_root,
                )
            )

        first_semantics = sources[0]["schema"].semantic_dict()
        for source in sources[1:]:
            if source["schema"].semantic_dict() != first_semantics:
                raise ValueError(f"Training schema differs from the first source: {source['root']}")

        report: dict[str, Any] = {
            "format": "jz_pin_timed_curation",
            "version": 2,
            "status": "PREFLIGHT_PASS" if args.preflight_only else "IN_PROGRESS",
            "output_root": str(output_root),
            "output_repo_id": args.output_repo_id,
            "expected_video_encoding": {"codec": args.expected_codec, "crf": args.expected_crf},
            "sources": [],
        }
        episode_offset = 0
        for source in sources:
            report["sources"].append(public_source_report(source, episode_offset))
            episode_offset += source["episodes"]
        report["expected_episodes"] = sum(source["episodes"] for source in sources)
        report["expected_frames"] = sum(source["frames"] for source in sources)

        if not args.preflight_only:
            merge_datasets(
                [source["dataset"] for source in sources],
                output_repo_id=args.output_repo_id,
                output_dir=output_root,
            )
            episode_map = write_jz_metadata(
                sources,
                output_root,
                args.output_repo_id,
                args.expected_codec,
                args.expected_crf,
            )
            report["episode_map"] = episode_map
            report["verification"] = verify_output(
                output_root,
                args.output_repo_id,
                report["expected_episodes"],
                report["expected_frames"],
                args.expected_codec,
                args.sample_frames_per_episode,
            )
            report["status"] = "MERGE_PASS"

        report_path = args.report_json
        if report_path is None:
            report_path = (
                output_root / "meta" / "jz_pin_curation_report.json"
                if not args.preflight_only
                else Path.cwd() / "jz_pin_curation_preflight.json"
            )
        report_path = report_path.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        print(
            f"status={report['status']} sources={len(sources)} "
            f"episodes={report['expected_episodes']} frames={report['expected_frames']}"
        )
        print(f"report_json={report_path}")
        return 0
    finally:
        for temporary_stage in reversed(temporary_stages):
            temporary_stage.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
