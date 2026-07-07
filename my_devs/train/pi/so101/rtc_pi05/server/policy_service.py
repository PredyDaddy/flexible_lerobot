from __future__ import annotations

import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor

from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.utils import get_safe_torch_device

from my_devs.train.pi.so101.rtc_pi05.rtc_pi05.checkpoint_loader import (
    disable_policy_rtc,
    enable_policy_rtc,
    load_policy_bundle,
)

from .protocol import InferenceRequest, InferenceResponse


@dataclass(slots=True)
class PolicyServiceConfig:
    policy_path: Path
    enable_rtc: bool = True
    device: str | None = None
    strict_so101_features: bool = True

    def __post_init__(self) -> None:
        self.policy_path = Path(self.policy_path).expanduser()


class PolicyService:
    """PI0.5 policy inference service with a serialized model lock."""

    def __init__(self, *, config: PolicyServiceConfig, bundle: Any) -> None:
        self.config = config
        self.bundle = bundle
        self.policy = bundle.policy
        self.preprocessor = bundle.preprocessor
        self.postprocessor = bundle.postprocessor
        self.device = get_safe_torch_device(self.policy.config.device)
        self._lock = threading.Lock()
        self.inference_count = 0
        self.last_latency_s = 0.0

    @classmethod
    def from_config(cls, config: PolicyServiceConfig) -> "PolicyService":
        return cls.from_config_with_rtc(config, rtc_config=None)

    @classmethod
    def from_config_with_rtc(cls, config: PolicyServiceConfig, *, rtc_config: Any | None = None) -> "PolicyService":
        bundle = load_policy_bundle(
            config.policy_path,
            device_override=config.device,
            strict_so101_features=config.strict_so101_features,
        )
        if config.enable_rtc:
            if rtc_config is None:
                from lerobot.policies.rtc.configuration_rtc import RTCConfig

                rtc_config = RTCConfig(enabled=True)
            enable_policy_rtc(bundle.policy, rtc_config)
        else:
            disable_policy_rtc(bundle.policy)
        return cls(config=config, bundle=bundle)

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        started_s = time.perf_counter()
        with self._lock:
            model_started_s = time.perf_counter()
            raw_chunk, processed_chunk = run_policy_chunk_inference(
                policy=self.policy,
                preprocessor=self.preprocessor,
                postprocessor=self.postprocessor,
                observation_frame=request.observation_frame,
                device=self.device,
                task=request.task,
                robot_type=request.robot_type,
                enable_rtc=self.config.enable_rtc and request.enable_rtc,
                predicted_delay_steps=request.predicted_delay_steps,
                prev_chunk_left_over=request.prev_chunk_left_over,
                execution_horizon=request.execution_horizon,
            )
            model_latency_s = time.perf_counter() - model_started_s
            self.inference_count += 1
            self.last_latency_s = time.perf_counter() - started_s

        raw_np = raw_chunk.detach().cpu().numpy()
        processed_np = processed_chunk.detach().cpu().numpy()
        return InferenceResponse(
            request_id=request.request_id,
            raw_actions=raw_np,
            processed_actions=processed_np,
            server_latency_s=time.perf_counter() - started_s,
            model_latency_s=model_latency_s,
            action_shape=tuple(processed_np.shape),
        )


def run_policy_chunk_inference(
    *,
    policy: Any,
    preprocessor: Callable[[dict[str, Any]], dict[str, Any]],
    postprocessor: Callable[[Tensor], Tensor],
    observation_frame: dict[str, Any],
    device: torch.device,
    task: str,
    robot_type: str,
    enable_rtc: bool,
    predicted_delay_steps: int,
    prev_chunk_left_over: np.ndarray | Tensor | None,
    execution_horizon: int,
) -> tuple[Tensor, Tensor]:
    batch = prepare_observation_for_inference(
        dict(observation_frame),
        device=device,
        task=task,
        robot_type=robot_type,
    )
    preprocessed_batch = preprocessor(batch)

    predict_kwargs: dict[str, Any] = {}
    if enable_rtc:
        left_over_tensor = None
        if prev_chunk_left_over is not None:
            left_over_tensor = torch.as_tensor(prev_chunk_left_over, dtype=torch.float32, device=device)
        predict_kwargs = {
            "inference_delay": int(predicted_delay_steps),
            "prev_chunk_left_over": left_over_tensor,
            "execution_horizon": int(execution_horizon),
        }

    use_amp = bool(getattr(getattr(policy, "config", None), "use_amp", False))
    with torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext():
        raw_chunk = ensure_chunk_batch(policy.predict_action_chunk(preprocessed_batch, **predict_kwargs))
        processed_chunk = postprocess_action_chunk(postprocessor, raw_chunk)
    return raw_chunk.squeeze(0), processed_chunk.squeeze(0)


def postprocess_action_chunk(postprocessor: Callable[[Tensor], Tensor], raw_chunk: Tensor) -> Tensor:
    raw_chunk = ensure_chunk_batch(raw_chunk)
    try:
        return ensure_chunk_batch(postprocessor(raw_chunk))
    except Exception as direct_error:
        batch_size, chunk_len, action_dim = raw_chunk.shape
        flattened = raw_chunk.reshape(batch_size * chunk_len, action_dim)
        try:
            processed = postprocessor(flattened)
        except Exception:
            raise direct_error
        return processed.reshape(batch_size, chunk_len, action_dim)


def ensure_chunk_batch(actions: Tensor) -> Tensor:
    if actions.ndim == 2:
        return actions.unsqueeze(0)
    if actions.ndim == 3:
        if actions.shape[0] != 1:
            raise ValueError(f"Only batch_size=1 is supported, got {tuple(actions.shape)}")
        return actions
    raise ValueError(f"Expected action chunk with shape (T,D) or (1,T,D), got {tuple(actions.shape)}")
