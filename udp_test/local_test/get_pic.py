#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output_media"
DEFAULT_CAMERA_URLS = {
    "camera_head": "rtsp://192.168.1.81:8554/robot_camera/camera_head",
    "camera_left": "rtsp://192.168.1.81:8554/robot_camera/camera_left",
    "camera_right": "rtsp://192.168.1.81:8554/robot_camera/camera_right",
}


def output_path_for_camera(output_dir: Path, camera_name: str) -> Path:
    return output_dir / f"{camera_name}.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture and overwrite the three current RTSP camera preview images.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for camera_head/left/right.jpg.")
    parser.add_argument("--timeout-ms", type=int, default=5000, help="Open/read timeout for each RTSP stream.")
    parser.add_argument("--warmup-frames", type=int, default=1, help="Frames to discard before saving.")
    parser.add_argument("--head-url", default=DEFAULT_CAMERA_URLS["camera_head"])
    parser.add_argument("--left-url", default=DEFAULT_CAMERA_URLS["camera_left"])
    parser.add_argument("--right-url", default=DEFAULT_CAMERA_URLS["camera_right"])
    return parser.parse_args()


def capture_one(camera_name: str, url: str, output_path: Path, timeout_ms: int, warmup_frames: int) -> bool:
    import cv2

    cap = None
    start = time.monotonic()
    try:
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)

        if not cap.isOpened():
            print(f"[FAIL] {camera_name}: failed to open {url}", flush=True)
            return False

        frame = None
        for _ in range(max(0, warmup_frames) + 1):
            ok, current = cap.read()
            if not ok or current is None:
                print(f"[FAIL] {camera_name}: failed to read frame from {url}", flush=True)
                return False
            frame = current

        if frame is None:
            print(f"[FAIL] {camera_name}: empty frame from {url}", flush=True)
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), frame):
            print(f"[FAIL] {camera_name}: failed to write {output_path}", flush=True)
            return False

        elapsed_s = time.monotonic() - start
        print(f"[PASS] {camera_name}: {output_path} shape={tuple(frame.shape)} elapsed_s={elapsed_s:.3f}", flush=True)
        return True
    finally:
        if cap is not None:
            cap.release()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    camera_urls = {
        "camera_head": args.head_url,
        "camera_left": args.left_url,
        "camera_right": args.right_url,
    }

    print("[INFO] capture RTSP camera previews: READONLY ONLY", flush=True)
    print("[INFO] This script only reads RTSP frames and overwrites jpg files.", flush=True)
    print(f"[INFO] output_dir={output_dir}", flush=True)

    results = [
        capture_one(
            camera_name=name,
            url=url,
            output_path=output_path_for_camera(output_dir, name),
            timeout_ms=args.timeout_ms,
            warmup_frames=args.warmup_frames,
        )
        for name, url in camera_urls.items()
    ]
    print(f"SUMMARY: {'PASS' if all(results) else 'FAIL'} captured={sum(results)}/{len(results)}", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
