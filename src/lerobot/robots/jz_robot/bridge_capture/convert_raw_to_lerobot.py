from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RawToLeRobotResult:
    dataset: Any
    episodes_written: int
    frames_written: int
    invalid_samples_skipped: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert raw JZRobot bridge episodes to a LeRobot v3 dataset.")
    parser.add_argument("--repo-id", required=True, help="LeRobot dataset repo id, for example local/jz_bridge_capture.")
    parser.add_argument("--output-root", required=True, help="Output dataset root directory.")
    parser.add_argument(
        "--raw-root",
        help="Directory containing episode_* raw episode directories. Ignored when --raw-episode is supplied.",
    )
    parser.add_argument(
        "--raw-episode",
        action="append",
        default=[],
        help="Raw episode directory. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Convert samples marked invalid. By default invalid samples are skipped.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_episodes = [Path(path) for path in args.raw_episode]
    if not raw_episodes:
        if not args.raw_root:
            raise SystemExit("provide --raw-root or at least one --raw-episode")
        raw_episodes = discover_raw_episodes(Path(args.raw_root))

    result = RawToLeRobotConverter().convert(
        raw_episodes=raw_episodes,
        repo_id=args.repo_id,
        output_root=Path(args.output_root),
        skip_invalid=not args.include_invalid,
    )
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "output_root": str(args.output_root),
                "episodes_written": result.episodes_written,
                "frames_written": result.frames_written,
                "invalid_samples_skipped": result.invalid_samples_skipped,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


class RawToLeRobotConverter:
    def __init__(self, *, dataset_factory: Any | None = None):
        self._dataset_factory = dataset_factory

    def convert(
        self,
        *,
        raw_episodes: list[Path],
        repo_id: str,
        output_root: Path,
        skip_invalid: bool = True,
    ) -> RawToLeRobotResult:
        if not raw_episodes:
            raise ValueError("raw_episodes must not be empty")

        episode_paths = [Path(path) for path in raw_episodes]
        metadata_by_episode = [_read_json(path / "metadata.json") for path in episode_paths]
        first_metadata = metadata_by_episode[0]
        _validate_metadata_compatibility(metadata_by_episode)

        dataset_factory = self._dataset_factory
        if dataset_factory is None:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            dataset_factory = LeRobotDataset

        features = build_lerobot_features(first_metadata)
        dataset = dataset_factory.create(
            repo_id=repo_id,
            fps=int(first_metadata.get("sample_rate_hz", 20)),
            features=features,
            root=output_root,
            robot_type=(first_metadata.get("robot") or {}).get("name"),
            use_videos=False,
        )

        frames_written = 0
        invalid_samples_skipped = 0
        episodes_written = 0
        for episode_path, metadata in zip(episode_paths, metadata_by_episode, strict=True):
            episode_frames = 0
            for sample in _read_samples(episode_path):
                if skip_invalid and not bool(sample.get("valid", False)):
                    invalid_samples_skipped += 1
                    continue
                frame = _sample_to_lerobot_frame(episode_path, metadata, sample)
                dataset.add_frame(frame)
                frames_written += 1
                episode_frames += 1

            if episode_frames == 0:
                raise ValueError(f"raw episode has no convertible samples: {episode_path}")
            dataset.save_episode()
            episodes_written += 1

        dataset.finalize()
        return RawToLeRobotResult(
            dataset=dataset,
            episodes_written=episodes_written,
            frames_written=frames_written,
            invalid_samples_skipped=invalid_samples_skipped,
        )


def discover_raw_episodes(raw_root: Path) -> list[Path]:
    episodes = sorted(path for path in raw_root.glob("episode_*") if path.is_dir() and not path.name.endswith(".tmp"))
    if not episodes:
        raise FileNotFoundError(f"no raw episode directories found in {raw_root}")
    return episodes


def build_lerobot_features(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    state_names = list(metadata.get("state_names") or [])
    action_names = list(metadata.get("action_names") or [])
    if not state_names:
        raise ValueError("metadata.state_names must not be empty")
    if not action_names:
        raise ValueError("metadata.action_names must not be empty")

    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": action_names,
        },
    }
    for camera_name, camera in (metadata.get("cameras") or {}).items():
        height = int(camera["height"])
        width = int(camera["width"])
        features[f"observation.images.{camera_name}"] = {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        }
    return features


def _sample_to_lerobot_frame(episode_path: Path, metadata: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    state = np.asarray(sample.get("state") or [], dtype=np.float32)
    action = np.asarray(sample.get("action") or [], dtype=np.float32)
    expected_state_dim = len(metadata.get("state_names") or [])
    expected_action_dim = len(metadata.get("action_names") or [])
    if state.shape != (expected_state_dim,):
        raise ValueError(f"sample {sample.get('sample_index')} state shape {state.shape} != ({expected_state_dim},)")
    if action.shape != (expected_action_dim,):
        raise ValueError(f"sample {sample.get('sample_index')} action shape {action.shape} != ({expected_action_dim},)")

    frame: dict[str, Any] = {
        "observation.state": state,
        "action": action,
        "task": metadata.get("task") or "jz bridge capture",
    }
    for camera_name in (metadata.get("cameras") or {}):
        frame_key = f"observation.images.{camera_name}"
        sample_key = f"{camera_name}_frame_path"
        relpath = sample.get(sample_key)
        if not relpath:
            raise ValueError(f"sample {sample.get('sample_index')} missing {sample_key}")
        frame[frame_key] = _read_image_rgb(episode_path / relpath)
    return frame


def _read_image_rgb(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _validate_metadata_compatibility(metadata_items: list[dict[str, Any]]) -> None:
    first = metadata_items[0]
    keys = ("state_names", "action_names", "cameras", "sample_rate_hz")
    for index, metadata in enumerate(metadata_items[1:], start=1):
        for key in keys:
            if metadata.get(key) != first.get(key):
                raise ValueError(f"raw episode metadata mismatch at index {index}: {key}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_samples(episode_dir: Path) -> list[dict[str, Any]]:
    parquet_path = episode_dir / "samples.parquet"
    if parquet_path.exists():
        import pandas as pd

        return pd.read_parquet(parquet_path).to_dict(orient="records")
    jsonl_path = episode_dir / "samples.jsonl"
    if jsonl_path.exists():
        return [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    raise FileNotFoundError(f"missing samples.parquet or samples.jsonl in {episode_dir}")


if __name__ == "__main__":
    main()
