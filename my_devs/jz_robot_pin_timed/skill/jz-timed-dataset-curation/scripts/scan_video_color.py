#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan every encoded video frame for abrupt full-frame RGB chromaticity changes."
    )
    parser.add_argument("--dataset-root", action="append", type=Path, required=True)
    parser.add_argument("--max-chromaticity-step", type=float, default=0.008)
    parser.add_argument("--report-json", type=Path, required=True)
    return parser.parse_args()


def scan_video(path: Path, threshold: float) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed opening video: {path}")
    frame_index = 0
    previous: np.ndarray | None = None
    maximum_step = 0.0
    maximum_step_frame: int | None = None
    events: list[dict[str, Any]] = []
    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            small = cv2.resize(bgr, (64, 64), interpolation=cv2.INTER_AREA)
            rgb_mean = small[..., ::-1].mean(axis=(0, 1), dtype=np.float64)
            chromaticity = rgb_mean / max(float(rgb_mean.sum()), 1e-12)
            minimum = np.minimum(minimum, chromaticity)
            maximum = np.maximum(maximum, chromaticity)
            if previous is not None:
                step = float(np.max(np.abs(chromaticity - previous)))
                if step > maximum_step:
                    maximum_step = step
                    maximum_step_frame = frame_index
                if step > threshold:
                    events.append(
                        {
                            "frame_index": frame_index,
                            "max_channel_step": step,
                            "previous_rgb_chromaticity": previous.tolist(),
                            "rgb_chromaticity": chromaticity.tolist(),
                        }
                    )
            previous = chromaticity
            frame_index += 1
    finally:
        capture.release()
    if frame_index == 0:
        raise RuntimeError(f"Video decoded zero frames: {path}")
    return {
        "path": str(path),
        "frames": frame_index,
        "max_chromaticity_step": maximum_step,
        "max_step_frame": maximum_step_frame,
        "rgb_chromaticity_min": minimum.tolist(),
        "rgb_chromaticity_max": maximum.tolist(),
        "events": events,
    }


def main() -> int:
    args = parse_args()
    if not 0 < args.max_chromaticity_step < 1:
        raise ValueError("--max-chromaticity-step must be in (0, 1)")
    report: dict[str, Any] = {
        "format": "jz_video_color_scan",
        "version": 1,
        "threshold": args.max_chromaticity_step,
        "datasets": [],
        "events": 0,
        "status": "PASS",
    }
    for dataset_root in args.dataset_root:
        dataset_root = dataset_root.expanduser().resolve()
        videos = []
        for path in sorted((dataset_root / "videos").glob("*/chunk-*/*.mp4")):
            result = scan_video(path, args.max_chromaticity_step)
            report["events"] += len(result["events"])
            videos.append(result)
        if not videos:
            raise FileNotFoundError(f"No videos found under {dataset_root}")
        report["datasets"].append({"root": str(dataset_root), "videos": videos})
    if report["events"]:
        report["status"] = "REVIEW"
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with args.report_json.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"status={report['status']} events={report['events']}")
    for dataset in report["datasets"]:
        for video in dataset["videos"]:
            print(
                f"video={video['path']} frames={video['frames']} "
                f"max_chromaticity_step={video['max_chromaticity_step']:.6f} "
                f"events={len(video['events'])}"
            )
    print(f"report_json={args.report_json.resolve()}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
