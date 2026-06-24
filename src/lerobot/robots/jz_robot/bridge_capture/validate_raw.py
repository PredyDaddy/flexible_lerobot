from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EpisodeValidationReport:
    episode_dir: Path
    sample_count: int
    valid_count: int
    invalid_count: int
    camera_frame_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a raw JZRobot LeRobot bridge episode.")
    parser.add_argument("episode_dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = validate_episode(Path(args.episode_dir))
    print(json.dumps(_report_to_dict(report), ensure_ascii=False, indent=2))
    if not report.ok:
        raise SystemExit(1)


def validate_episode(episode_dir: str | Path) -> EpisodeValidationReport:
    episode_path = Path(episode_dir)
    metadata = _read_json(episode_path / "metadata.json")
    samples = _read_samples(episode_path)
    errors: list[str] = []

    state_dim = len(metadata.get("state_names", []))
    action_dim = len(metadata.get("action_names", []))
    cameras = metadata.get("cameras", {})
    camera_frame_counts = {camera_name: 0 for camera_name in cameras}

    valid_count = 0
    for index, sample in enumerate(samples):
        if sample.get("valid", False):
            valid_count += 1
            state = sample.get("state", [])
            action = sample.get("action", [])
            if len(state) != state_dim:
                errors.append(f"sample {index} state dimension {len(state)} != expected {state_dim}")
            if len(action) != action_dim:
                errors.append(f"sample {index} action dimension {len(action)} != expected {action_dim}")
            for camera_name in cameras:
                frame_path = sample.get(f"{camera_name}_frame_path")
                if not frame_path:
                    errors.append(f"sample {index} missing frame path for camera: {camera_name}")
        for camera_name in cameras:
            frame_path = sample.get(f"{camera_name}_frame_path")
            if not frame_path:
                continue
            full_frame_path = episode_path / frame_path
            if not full_frame_path.exists():
                errors.append(f"sample {index} missing frame file: {frame_path}")
                continue
            camera_frame_counts[camera_name] += 1
            if not _can_decode_image(full_frame_path):
                errors.append(f"sample {index} cannot decode frame file: {frame_path}")

    if len(samples) == 0:
        errors.append("episode contains no samples")
    if valid_count == 0:
        errors.append("episode contains no valid samples")

    return EpisodeValidationReport(
        episode_dir=episode_path,
        sample_count=len(samples),
        valid_count=valid_count,
        invalid_count=len(samples) - valid_count,
        camera_frame_counts=camera_frame_counts,
        errors=errors,
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_samples(episode_dir: Path) -> list[dict[str, Any]]:
    parquet_path = episode_dir / "samples.parquet"
    if parquet_path.exists():
        import pandas as pd

        return pd.read_parquet(parquet_path).to_dict(orient="records")
    jsonl_path = episode_dir / "samples.jsonl"
    if jsonl_path.exists():
        return [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    raise FileNotFoundError(f"missing samples.parquet or samples.jsonl in {episode_dir}")


def _can_decode_image(path: Path) -> bool:
    try:
        import cv2

        return cv2.imread(str(path)) is not None
    except Exception:
        return False


def _report_to_dict(report: EpisodeValidationReport) -> dict[str, Any]:
    return {
        "episode_dir": str(report.episode_dir),
        "sample_count": report.sample_count,
        "valid_count": report.valid_count,
        "invalid_count": report.invalid_count,
        "camera_frame_counts": report.camera_frame_counts,
        "ok": report.ok,
        "errors": report.errors,
    }


if __name__ == "__main__":
    main()
