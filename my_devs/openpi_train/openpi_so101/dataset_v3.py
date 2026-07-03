from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
from typing import SupportsIndex

import av
import numpy as np
import pandas as pd


class LeRobotV3Metadata:
    def __init__(self, repo_id: str, root: str | Path):
        self.repo_id = repo_id
        self.root = Path(root)
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        self.fps = int(self.info["fps"])
        self.features = self.info["features"]
        self.video_keys = [key for key, value in self.features.items() if value.get("dtype") == "video"]
        self.tasks = self._load_tasks()
        self.episodes = self._load_episodes()

    def _load_tasks(self) -> dict[int, str]:
        tasks_path = self.root / "meta" / "tasks.parquet"
        tasks_df = pd.read_parquet(tasks_path)
        if "task" not in tasks_df.columns:
            tasks_df = tasks_df.reset_index()
        return {int(row.task_index): str(row.task) for row in tasks_df.itertuples(index=False)}

    def _load_episodes(self) -> list[dict]:
        paths = sorted((self.root / "meta" / "episodes").glob("chunk-*/*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No episode metadata parquet found under {self.root / 'meta' / 'episodes'}")
        frames = [pd.read_parquet(path) for path in paths]
        return pd.concat(frames, ignore_index=True).to_dict("records")


class _VideoReader:
    def __init__(self, path: Path):
        self.path = path
        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        self._iter = self.container.decode(self.stream)
        self._next_index = 0

    def close(self) -> None:
        self.container.close()

    def read(self, frame_index: int) -> np.ndarray:
        if frame_index < self._next_index:
            self.close()
            self.container = av.open(str(self.path))
            self.stream = self.container.streams.video[0]
            self._iter = self.container.decode(self.stream)
            self._next_index = 0

        for frame in self._iter:
            current = self._next_index
            self._next_index += 1
            if current == frame_index:
                return frame.to_ndarray(format="rgb24")
        raise IndexError(f"Frame {frame_index} is out of range for {self.path}")


class SO101LeRobotV3Dataset:
    """Small read-only LeRobot v3 dataset adapter for OpenPI smoke training.

    The adapter keeps processed data local to my_devs/openpi_train by reading the
    original LeRobot v3 dataset in place and exposing the fields OpenPI expects.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path,
        *,
        action_horizon: int,
        action_sequence_keys: Sequence[str] = ("action",),
        max_frames: int | None = None,
        decode_images: bool = True,
    ):
        del action_sequence_keys
        self.repo_id = repo_id
        self.root = Path(root)
        self.meta = LeRobotV3Metadata(repo_id, self.root)
        self.action_horizon = action_horizon
        self.decode_images = decode_images

        data_paths = sorted((self.root / "data").glob("chunk-*/*.parquet"))
        if not data_paths:
            raise FileNotFoundError(f"No data parquet found under {self.root / 'data'}")
        self._data = pd.concat([pd.read_parquet(path) for path in data_paths], ignore_index=True)
        self._length = min(len(self._data), max_frames) if max_frames is not None else len(self._data)
        self._episode_by_index = {
            int(ep["episode_index"]): ep
            for ep in self.meta.episodes
        }
        self._video_readers: dict[Path, _VideoReader] = {}

    def __len__(self) -> int:
        return self._length

    def _pad_indices(self, row: pd.Series, key: str) -> list[int]:
        del key
        ep_index = int(row["episode_index"])
        frame_index = int(row["index"])
        episode = self._episode_by_index[ep_index]
        end = int(episode["dataset_to_index"]) - 1
        return [min(end, frame_index + offset) for offset in range(self.action_horizon)]

    def _video_frame(self, row: pd.Series, video_key: str) -> np.ndarray:
        if not self.decode_images:
            shape = self.meta.features[video_key]["shape"]
            return np.zeros(shape, dtype=np.uint8)

        ep_index = int(row["episode_index"])
        episode = self._episode_by_index[ep_index]
        chunk_idx = int(episode[f"videos/{video_key}/chunk_index"])
        file_idx = int(episode[f"videos/{video_key}/file_index"])
        from_ts = float(episode[f"videos/{video_key}/from_timestamp"])
        timestamp = float(row["timestamp"])
        frame_index = int(round((from_ts + timestamp) * self.meta.fps))
        video_rel = self.meta.info["video_path"].format(
            video_key=video_key,
            chunk_index=chunk_idx,
            file_index=file_idx,
        )
        video_path = self.root / video_rel
        reader = self._video_readers.get(video_path)
        if reader is None:
            reader = _VideoReader(video_path)
            self._video_readers[video_path] = reader
        return reader.read(frame_index)

    def __getitem__(self, index: SupportsIndex) -> dict:
        idx = index.__index__()
        row = self._data.iloc[idx]
        action_indices = self._pad_indices(row, "action")
        action_values = np.stack(
            [np.asarray(self._data.iloc[action_idx]["action"], dtype=np.float32) for action_idx in action_indices],
            axis=0,
        )
        task_index = int(row["task_index"])

        return {
            "action": action_values,
            "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
            "observation.images.top": self._video_frame(row, "observation.images.top"),
            "observation.images.wrist": self._video_frame(row, "observation.images.wrist"),
            "prompt": self.meta.tasks[task_index],
            "task_index": np.asarray(task_index, dtype=np.int64),
        }

    def close(self) -> None:
        for reader in self._video_readers.values():
            reader.close()
        self._video_readers.clear()

    def __del__(self):
        self.close()
