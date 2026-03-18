from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor

from lerobot.datasets.utils import build_dataset_frame
from lerobot.utils.constants import OBS_STR


class ChunkPolicy(Protocol):
    config: Any

    def predict_action_chunk(self, batch: dict[str, Any], **kwargs: Any) -> Tensor: ...


@dataclass(slots=True)
class ChunkInferenceResult:
    observation_frame: dict[str, np.ndarray]
    prepared_batch: dict[str, Any]
    preprocessed_batch: dict[str, Any]
    raw_action_chunk: Tensor
    original_actions: Tensor
    processed_action_chunk: Tensor
    processed_actions: Tensor


def build_chunk_observation_frame(
    observation: Mapping[str, Any],
    dataset_features: Mapping[str, dict[str, Any]],
    *,
    robot_observation_processor: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    prefix: str = OBS_STR,
) -> dict[str, np.ndarray]:
    """Build a single-timestep observation frame using the baseline dataset-feature path."""

    processed_observation = (
        dict(robot_observation_processor(observation)) if robot_observation_processor else dict(observation)
    )
    return build_dataset_frame(dict(dataset_features), processed_observation, prefix=prefix)


def prepare_policy_batch_from_frame(
    observation_frame: Mapping[str, np.ndarray | Tensor],
    device: torch.device | str,
    *,
    task: str | None = None,
    robot_type: str | None = None,
) -> dict[str, Any]:
    """Convert a dataset frame to a batched policy input dictionary.

    This follows the same broad pattern as the current PI05 baseline:
    image keys are converted from HWC uint8 arrays to BCHW float tensors in [0, 1],
    non-image values get a batch dimension added, and task / robot_type metadata are
    injected for downstream processors.
    """

    target_device = torch.device(device)
    prepared: dict[str, Any] = {}
    for name, value in observation_frame.items():
        tensor = _to_tensor(value)
        if _is_image_key(name):
            tensor = _prepare_image_tensor(tensor)
        tensor = tensor.unsqueeze(0).to(target_device)
        prepared[name] = tensor

    prepared["task"] = [task if task is not None else ""]
    prepared["robot_type"] = robot_type if robot_type is not None else ""
    return prepared


def run_chunk_inference(
    *,
    observation: Mapping[str, Any],
    dataset_features: Mapping[str, dict[str, Any]],
    policy: ChunkPolicy,
    preprocessor: Callable[[dict[str, Any]], dict[str, Any]],
    postprocessor: Callable[[Tensor], Tensor],
    device: torch.device | str | None = None,
    task: str | None = None,
    robot_type: str | None = None,
    robot_observation_processor: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    rtc_kwargs: Mapping[str, Any] | None = None,
) -> ChunkInferenceResult:
    """Run the offline PI05 chunk inference path from raw observation to processed action chunk.

    Returns both:
    - `original_actions`: model-output action chunk in model action space
    - `processed_actions`: postprocessed action chunk intended for robot rollout

    The caller must never derive `original_actions` back from `processed_actions`.
    """

    resolved_device = _resolve_device(policy, device)
    use_amp = bool(getattr(getattr(policy, "config", None), "use_amp", False))

    observation_frame = build_chunk_observation_frame(
        observation,
        dataset_features,
        robot_observation_processor=robot_observation_processor,
    )
    prepared_batch = prepare_policy_batch_from_frame(
        observation_frame,
        resolved_device,
        task=task,
        robot_type=robot_type,
    )

    predict_kwargs = dict(rtc_kwargs or {})
    with (
        torch.inference_mode(),
        torch.autocast(device_type=resolved_device.type) if resolved_device.type == "cuda" and use_amp else nullcontext(),
    ):
        preprocessed_batch = preprocessor(dict(prepared_batch))
        raw_action_chunk_inference = _ensure_chunk_batch(policy.predict_action_chunk(preprocessed_batch, **predict_kwargs))
        if raw_action_chunk_inference.shape[0] != 1:
            raise ValueError(
                f"Phase 1 chunk runtime expects batch_size=1, got {tuple(raw_action_chunk_inference.shape)}"
            )

        processed_action_chunk_inference = _ensure_chunk_batch(postprocessor(raw_action_chunk_inference))
        if processed_action_chunk_inference.shape[0] != 1:
            raise ValueError(
                "Postprocessor must preserve batch_size=1 for Phase 1 runtime, "
                f"got {tuple(processed_action_chunk_inference.shape)}"
            )
    raw_action_chunk = raw_action_chunk_inference.detach().clone()
    original_actions = raw_action_chunk.squeeze(0).clone()
    processed_action_chunk = processed_action_chunk_inference.detach().clone()
    processed_actions = processed_action_chunk.squeeze(0).clone()

    return ChunkInferenceResult(
        observation_frame=observation_frame,
        prepared_batch=prepared_batch,
        preprocessed_batch=preprocessed_batch,
        raw_action_chunk=raw_action_chunk,
        original_actions=original_actions,
        processed_action_chunk=processed_action_chunk,
        processed_actions=processed_actions,
    )


def _resolve_device(policy: ChunkPolicy, device: torch.device | str | None) -> torch.device:
    if device is not None:
        return torch.device(device)

    policy_device = getattr(getattr(policy, "config", None), "device", "cpu")
    return torch.device(policy_device)


def _to_tensor(value: np.ndarray | Tensor) -> Tensor:
    if isinstance(value, Tensor):
        return value.detach().clone()
    return torch.from_numpy(np.asarray(value))


def _is_image_key(name: str) -> bool:
    return "image" in name


def _prepare_image_tensor(tensor: Tensor) -> Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"Expected unbatched image tensor with 3 dims, got shape {tuple(tensor.shape)}")

    if tensor.shape[-1] in (1, 3, 4):
        tensor = tensor.permute(2, 0, 1).contiguous()

    tensor = tensor.to(dtype=torch.float32)
    if tensor.numel() > 0 and float(tensor.max().item()) > 1.0:
        tensor = tensor / 255.0
    return tensor


def _ensure_chunk_batch(actions: Tensor) -> Tensor:
    if actions.ndim == 2:
        return actions.unsqueeze(0)
    if actions.ndim == 3:
        return actions
    raise ValueError(f"Expected action chunk with shape (T, D) or (B, T, D), got {tuple(actions.shape)}")
