from __future__ import annotations

import logging
import types

import datasets
import openpi.training.data_loader as data_loader
from openpi import transforms
import torch

from openpi_so101 import config as so101_config
from openpi_so101 import dataset_v3
from openpi_so101 import runtime


class LeRobotDatasetV21Compat:
    """Thin wrapper around the official v2.1 LeRobotDataset for datasets>=5 Column handling."""

    def __init__(
        self,
        repo_id: str,
        root,
        *,
        action_horizon: int,
        action_sequence_keys=("action",),
        max_frames=None,
        decode_images: bool = True,
    ):
        from lerobot.common.datasets import lerobot_dataset
        from lerobot.common.datasets import utils as lerobot_utils

        self._inner = lerobot_dataset.LeRobotDataset.__new__(lerobot_dataset.LeRobotDataset)
        inner = self._inner
        inner.repo_id = repo_id
        inner.root = root
        inner.image_transforms = None
        inner.delta_timestamps = None
        inner.episodes = None
        inner.tolerance_s = 1e-4
        inner.revision = lerobot_dataset.CODEBASE_VERSION
        inner.video_backend = "pyav"
        inner.delta_indices = None
        inner.image_writer = None
        inner.episode_buffer = None
        inner.root.mkdir(exist_ok=True, parents=True)
        inner.meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, inner.root, inner.revision)
        inner.delta_timestamps = {
            key: [t / inner.meta.fps for t in range(action_horizon)] for key in action_sequence_keys
        }
        inner.hf_dataset = datasets.load_dataset("parquet", data_dir=str(inner.root / "data"), split="train")
        inner.hf_dataset.set_transform(lerobot_utils.hf_transform_to_torch)
        inner.episode_data_index = lerobot_utils.get_episode_data_index(inner.meta.episodes, inner.episodes)

        timestamps = torch.stack(list(inner.hf_dataset["timestamp"])).numpy()
        episode_indices = torch.stack(list(inner.hf_dataset["episode_index"])).numpy()
        ep_data_index_np = {key: value.numpy() for key, value in inner.episode_data_index.items()}
        lerobot_utils.check_timestamps_sync(
            timestamps,
            episode_indices,
            ep_data_index_np,
            inner.fps,
            inner.tolerance_s,
        )
        lerobot_utils.check_delta_timestamps(inner.delta_timestamps, inner.fps, inner.tolerance_s)
        inner.delta_indices = lerobot_utils.get_delta_indices(inner.delta_timestamps, inner.fps)
        inner._query_hf_dataset = types.MethodType(_query_hf_dataset_compat, inner)
        self._length = min(len(inner), max_frames) if max_frames is not None else len(inner)
        self._decode_images = decode_images

    @property
    def meta(self):
        return self._inner.meta

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index):
        if not self._decode_images:
            return _getitem_without_video_decode(self._inner, index)
        return self._inner[index]


def _query_hf_dataset_compat(self, query_indices):
    return {
        key: torch.stack(list(self.hf_dataset.select(q_idx)[key]))
        for key, q_idx in query_indices.items()
        if key not in self.meta.video_keys
    }


def _getitem_without_video_decode(self, index):
    idx = index.__index__() if hasattr(index, "__index__") else int(index)
    item = self.hf_dataset[idx]
    ep_idx = item["episode_index"].item()

    if self.delta_indices is not None:
        query_indices, padding = self._get_query_indices(idx, ep_idx)
        query_result = self._query_hf_dataset(query_indices)
        item = {**item, **padding, **query_result}

    for key in self.meta.video_keys:
        item[key] = torch.zeros(self.meta.features[key]["shape"], dtype=torch.float32)

    task_idx = item["task_index"].item()
    item["task"] = self.meta.tasks[task_idx]
    return item


def patch_openpi_data_loader(
    *,
    max_frames: int | None = None,
    decode_images: bool = True,
    dataset_format: str = "v3",
) -> None:
    original_create_torch_dataset = data_loader.create_torch_dataset

    def create_torch_dataset_patched(data_config, action_horizon, model_config):
        if data_config.repo_id == so101_config.make_config().data.repo_id:
            if dataset_format == "v21":
                logging.info(
                    "Using SO101 converted LeRobot v2.1 dataset: root=%s repo_id=%s max_frames=%s decode_images=%s",
                    runtime.converted_dataset_root(),
                    data_config.repo_id,
                    max_frames,
                    decode_images,
                )
                dataset = LeRobotDatasetV21Compat(
                    data_config.repo_id,
                    runtime.converted_dataset_root(),
                    action_horizon=action_horizon,
                    action_sequence_keys=data_config.action_sequence_keys,
                    max_frames=max_frames,
                    decode_images=decode_images,
                )
                if data_config.prompt_from_task:
                    dataset = data_loader.TransformedDataset(
                        dataset,
                        [transforms.PromptFromLeRobotTask(dataset.meta.tasks)],
                    )
                return dataset

            logging.info(
                "Using SO101 LeRobot v3 adapter: root=%s repo_id=%s max_frames=%s decode_images=%s",
                runtime.dataset_root(),
                data_config.repo_id,
                max_frames,
                decode_images,
            )
            return dataset_v3.SO101LeRobotV3Dataset(
                data_config.repo_id,
                runtime.dataset_root(),
                action_horizon=action_horizon,
                action_sequence_keys=data_config.action_sequence_keys,
                max_frames=max_frames,
                decode_images=decode_images,
            )
        return original_create_torch_dataset(data_config, action_horizon, model_config)

    data_loader.create_torch_dataset = create_torch_dataset_patched
