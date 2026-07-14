#!/usr/bin/env python
"""Compare retained pre-encode PNG frames with their decoded dataset video frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image


def _metrics(raw_rgb: np.ndarray, video_rgb: np.ndarray) -> dict[str, object]:
    if raw_rgb.shape != video_rgb.shape:
        raise ValueError(f"frame shape mismatch: PNG {raw_rgb.shape}, video {video_rgb.shape}")
    raw = raw_rgb.astype(np.float32)
    video = video_rgb.astype(np.float32)
    raw_mean = raw.mean(axis=(0, 1))
    video_mean = video.mean(axis=(0, 1))
    channel_mae = np.abs(raw - video).mean(axis=(0, 1))
    raw_blue_excess = raw_mean[2] - (raw_mean[0] + raw_mean[1]) / 2
    video_blue_excess = video_mean[2] - (video_mean[0] + video_mean[1]) / 2
    return {
        "raw_rgb_mean": raw_mean.round(4).tolist(),
        "video_rgb_mean": video_mean.round(4).tolist(),
        "rgb_mean_shift": (video_mean - raw_mean).round(4).tolist(),
        "channel_mae": channel_mae.round(4).tolist(),
        "overall_mae": round(float(channel_mae.mean()), 4),
        "raw_blue_excess": round(float(raw_blue_excess), 4),
        "video_blue_excess": round(float(video_blue_excess), 4),
        "encoding_blue_excess_shift": round(float(video_blue_excess - raw_blue_excess), 4),
    }


def _decode_video(video_path: Path) -> list[np.ndarray]:
    with av.open(str(video_path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def _sample_indices(frame_count: int, sample_count: int) -> list[int]:
    if frame_count <= 0:
        return []
    count = min(frame_count, max(1, sample_count))
    return sorted({int(index) for index in np.linspace(0, frame_count - 1, count)})


def compare_dataset(dataset_root: Path, sample_count: int) -> dict[str, object]:
    image_root = dataset_root / "images"
    video_root = dataset_root / "videos"
    if not image_root.is_dir():
        raise FileNotFoundError(f"retained PNG directory not found: {image_root}")
    if not video_root.is_dir():
        raise FileNotFoundError(f"encoded video directory not found: {video_root}")

    camera_reports: dict[str, object] = {}
    for camera_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
        png_paths = sorted(camera_dir.glob("episode-*/frame-*.png"))
        video_paths = sorted((video_root / camera_dir.name).glob("**/*.mp4"))
        if not png_paths:
            continue
        if len(video_paths) != 1:
            raise ValueError(
                f"diagnostic expects one video file for {camera_dir.name}, found {len(video_paths)}"
            )

        video_frames = _decode_video(video_paths[0])
        comparable_count = min(len(png_paths), len(video_frames))
        if comparable_count == 0:
            raise ValueError(f"no comparable frames for {camera_dir.name}")
        samples = []
        for index in _sample_indices(comparable_count, sample_count):
            with Image.open(png_paths[index]) as image:
                raw_rgb = np.asarray(image.convert("RGB"))
            samples.append(
                {
                    "frame_index": index,
                    "png": str(png_paths[index]),
                    **_metrics(raw_rgb, video_frames[index]),
                }
            )

        mean_blue_shift = float(np.mean([sample["encoding_blue_excess_shift"] for sample in samples]))
        camera_reports[camera_dir.name] = {
            "png_frame_count": len(png_paths),
            "video_frame_count": len(video_frames),
            "compared_frame_count": len(samples),
            "video": str(video_paths[0]),
            "mean_encoding_blue_excess_shift": round(mean_blue_shift, 4),
            "samples": samples,
        }

    if not camera_reports:
        raise ValueError(f"no camera PNG frames found under {image_root}")
    return {
        "schema_version": 1,
        "dataset_root": str(dataset_root),
        "frame_semantics": "PNG is post-Orin-JPEG/X86-decode and pre-dataset-video-encode",
        "cameras": camera_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args()

    report = compare_dataset(args.dataset_root.resolve(), args.sample_count)
    report_json = args.report_json or args.dataset_root / "color_encoding_comparison.json"
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for camera, camera_report in report["cameras"].items():
        print(
            f"camera={camera} png={camera_report['png_frame_count']} "
            f"video={camera_report['video_frame_count']} "
            f"mean_encoding_blue_excess_shift={camera_report['mean_encoding_blue_excess_shift']:.4f}"
        )
    print(f"report_json={report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
