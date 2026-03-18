from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from my_devs.pi05_engineering.runtime.chunk_inference import (
    build_chunk_observation_frame,
    prepare_policy_batch_from_frame,
    run_chunk_inference,
)


def _make_dataset_features() -> dict[str, dict]:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": [2],
            "names": ["joint_1", "joint_2"],
        },
        "observation.images.top": {
            "dtype": "image",
            "shape": [4, 5, 3],
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "image",
            "shape": [4, 5, 3],
            "names": ["height", "width", "channels"],
        },
    }


def _make_raw_observation() -> dict[str, object]:
    return {
        "joint_1": 1.0,
        "joint_2": -1.5,
        "top": np.full((4, 5, 3), 128, dtype=np.uint8),
        "wrist": np.full((4, 5, 3), 64, dtype=np.uint8),
    }


class RecordingPreprocessor:
    def __init__(self, call_order: list[str]):
        self.call_order = call_order
        self.last_batch: dict[str, object] | None = None

    def __call__(self, batch: dict[str, object]) -> dict[str, object]:
        self.call_order.append("pre")
        self.last_batch = batch
        out = dict(batch)
        out["preprocessed"] = True
        return out


class RecordingPolicy:
    def __init__(self, call_order: list[str], output: torch.Tensor):
        self.call_order = call_order
        self.output = output
        self.last_batch: dict[str, object] | None = None
        self.last_kwargs: dict[str, object] | None = None
        self.config = SimpleNamespace(device="cpu", use_amp=False)

    def predict_action_chunk(self, batch: dict[str, object], **kwargs: object) -> torch.Tensor:
        self.call_order.append("policy")
        self.last_batch = batch
        self.last_kwargs = dict(kwargs)
        return self.output.clone()


class RecordingPostprocessor:
    def __init__(self, call_order: list[str]):
        self.call_order = call_order
        self.last_input: torch.Tensor | None = None

    def __call__(self, actions: torch.Tensor) -> torch.Tensor:
        self.call_order.append("post")
        self.last_input = actions.clone()
        return actions + 10.0


def test_build_chunk_observation_frame_uses_dataset_features_and_observation_processor():
    dataset_features = _make_dataset_features()
    raw_observation = _make_raw_observation()

    def robot_observation_processor(obs: dict[str, object]) -> dict[str, object]:
        processed = dict(obs)
        processed["joint_1"] = 2.5
        return processed

    frame = build_chunk_observation_frame(
        raw_observation,
        dataset_features,
        robot_observation_processor=robot_observation_processor,
    )

    assert set(frame) == {"observation.state", "observation.images.top", "observation.images.wrist"}
    np.testing.assert_allclose(frame["observation.state"], np.array([2.5, -1.5], dtype=np.float32))
    assert frame["observation.images.top"].shape == (4, 5, 3)


def test_prepare_policy_batch_adds_batch_dim_normalizes_images_and_injects_metadata():
    dataset_features = _make_dataset_features()
    frame = build_chunk_observation_frame(_make_raw_observation(), dataset_features)

    prepared = prepare_policy_batch_from_frame(
        frame,
        "cpu",
        task="Put the block in the bin",
        robot_type="so101_follower",
    )

    assert prepared["observation.state"].shape == (1, 2)
    assert prepared["observation.images.top"].shape == (1, 3, 4, 5)
    assert prepared["observation.images.wrist"].shape == (1, 3, 4, 5)
    assert prepared["observation.images.top"].dtype == torch.float32
    assert torch.isclose(prepared["observation.images.top"][0, 0, 0, 0], torch.tensor(128.0 / 255.0))
    assert prepared["task"] == ["Put the block in the bin"]
    assert prepared["robot_type"] == "so101_follower"


def test_run_chunk_inference_calls_pre_policy_post_in_order_and_keeps_original_separate():
    dataset_features = _make_dataset_features()
    call_order: list[str] = []
    preprocessor = RecordingPreprocessor(call_order)
    policy = RecordingPolicy(call_order, output=torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32))
    postprocessor = RecordingPostprocessor(call_order)

    result = run_chunk_inference(
        observation=_make_raw_observation(),
        dataset_features=dataset_features,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        task="stack blocks",
        robot_type="so101_follower",
    )

    assert call_order == ["pre", "policy", "post"]
    assert preprocessor.last_batch is not None
    assert preprocessor.last_batch["task"] == ["stack blocks"]
    assert preprocessor.last_batch["robot_type"] == "so101_follower"

    assert torch.equal(result.raw_action_chunk, torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32))
    assert torch.equal(result.original_actions, torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32))
    assert torch.equal(result.processed_actions, torch.tensor([[11.0, 12.0], [13.0, 14.0]], dtype=torch.float32))

    result.processed_actions[0, 0] = -999.0
    assert result.original_actions[0, 0].item() == 1.0


def test_run_chunk_inference_forwards_rtc_kwargs_to_policy():
    dataset_features = _make_dataset_features()
    policy = RecordingPolicy([], output=torch.tensor([[[0.1, 0.2]]], dtype=torch.float32))
    preprocessor = RecordingPreprocessor([])
    postprocessor = RecordingPostprocessor([])

    rtc_kwargs = {
        "inference_delay": 2,
        "prev_chunk_left_over": torch.tensor([[0.5, 0.6]], dtype=torch.float32),
        "execution_horizon": 8,
    }
    _ = run_chunk_inference(
        observation=_make_raw_observation(),
        dataset_features=dataset_features,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        rtc_kwargs=rtc_kwargs,
    )

    assert policy.last_kwargs == rtc_kwargs


def test_run_chunk_inference_exposes_preprocessed_batch_and_processed_chunk():
    dataset_features = _make_dataset_features()
    preprocessor = RecordingPreprocessor([])
    policy = RecordingPolicy([], output=torch.tensor([[[5.0, 6.0]]], dtype=torch.float32))
    postprocessor = RecordingPostprocessor([])

    result = run_chunk_inference(
        observation=_make_raw_observation(),
        dataset_features=dataset_features,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        task="inspect",
    )

    assert result.preprocessed_batch["preprocessed"] is True
    assert torch.equal(result.processed_action_chunk, torch.tensor([[[15.0, 16.0]]], dtype=torch.float32))
    assert result.prepared_batch["task"] == ["inspect"]
    assert result.prepared_batch["robot_type"] == ""
