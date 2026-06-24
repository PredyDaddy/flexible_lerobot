#!/usr/bin/env python

"""PI0.5 runtime that uses TensorRT engines without loading PyTorch model weights."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from runtime.protocol import (
    DENOISE_OUTPUT_NAMES,
    PREFIX_CACHE_INPUT_NAMES,
    denoise_step_input_names,
    prefix_cache_tensor_names,
    validate_names,
)
from runtime.simple_pi05_split import (
    DEFAULT_DENOISE_FP16_CONSTRAINED_ENGINE,
    DEFAULT_DENOISE_FP32_ENGINE,
    DEFAULT_PREFIX_ENGINE,
)
from runtime.trt_engine import TorchTensorRTEngine


@dataclass(frozen=True)
class PurePI05TRTProfile:
    name: str
    prefix_engine_path: Path
    denoise_engine_path: Path
    num_layers: int = 18

    @classmethod
    def fp32(cls) -> "PurePI05TRTProfile":
        return cls(
            name="fp32",
            prefix_engine_path=DEFAULT_PREFIX_ENGINE,
            denoise_engine_path=DEFAULT_DENOISE_FP32_ENGINE,
        )

    @classmethod
    def fp16_constrained(cls) -> "PurePI05TRTProfile":
        return cls(
            name="fp16_constrained",
            prefix_engine_path=DEFAULT_PREFIX_ENGINE,
            denoise_engine_path=DEFAULT_DENOISE_FP16_CONSTRAINED_ENGINE,
        )

    def resolved(self) -> "PurePI05TRTProfile":
        return PurePI05TRTProfile(
            name=self.name,
            prefix_engine_path=self.prefix_engine_path.expanduser(),
            denoise_engine_path=self.denoise_engine_path.expanduser(),
            num_layers=self.num_layers,
        )

    def validate_files(self) -> None:
        if not self.prefix_engine_path.is_file():
            raise FileNotFoundError(f"Missing prefix_cache TensorRT engine: {self.prefix_engine_path}")
        if not self.denoise_engine_path.is_file():
            raise FileNotFoundError(f"Missing denoise_step TensorRT engine: {self.denoise_engine_path}")


class PurePI05TRTRuntime:
    """Run full PI0.5 sample_actions with prefix_cache TRT + denoise_step TRT."""

    def __init__(self, profile: PurePI05TRTProfile):
        self.profile = profile.resolved()
        self.profile.validate_files()
        self.prefix_engine = TorchTensorRTEngine(self.profile.prefix_engine_path)
        self.denoise_engine = TorchTensorRTEngine(self.profile.denoise_engine_path)
        self.cache_names = prefix_cache_tensor_names(self.profile.num_layers)
        self.denoise_input_names = denoise_step_input_names(self.profile.num_layers)
        self._validate_engine_io()

    def _validate_engine_io(self) -> None:
        validate_names(self.prefix_engine.input_names, list(PREFIX_CACHE_INPUT_NAMES), label="prefix_cache inputs")
        validate_names(self.prefix_engine.output_names, self.cache_names, label="prefix_cache outputs")
        validate_names(self.denoise_engine.input_names, self.denoise_input_names, label="denoise_step inputs")
        validate_names(self.denoise_engine.output_names, list(DENOISE_OUTPUT_NAMES), label="denoise_step outputs")

    @staticmethod
    def _cast_for_engine(engine: TorchTensorRTEngine, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        casted = {}
        for name in engine.input_names:
            tensor = inputs[name].contiguous()
            dtype = engine.tensor_dtypes[name]
            if tensor.dtype != dtype:
                tensor = tensor.to(dtype)
            casted[name] = tensor
        return casted

    @torch.no_grad()
    def sample_actions(
        self,
        *,
        images: list[Tensor],
        img_masks: list[Tensor],
        tokens: Tensor,
        masks: Tensor,
        chunk_size: int,
        max_action_dim: int,
        num_inference_steps: int,
        noise: Tensor | None = None,
    ) -> Tensor:
        if len(images) != 2 or len(img_masks) != 2:
            raise ValueError(f"Expected two SO101 images/masks, got {len(images)} images and {len(img_masks)} masks")

        if noise is None:
            noise = torch.normal(
                mean=0.0,
                std=1.0,
                size=(tokens.shape[0], chunk_size, max_action_dim),
                dtype=torch.float32,
                device=tokens.device,
            )

        prefix_inputs = {
            "image_0": images[0],
            "image_1": images[1],
            "img_mask_0": img_masks[0],
            "img_mask_1": img_masks[1],
            "tokens": tokens,
            "masks": masks,
        }
        prefix_outputs = self.prefix_engine(**self._cast_for_engine(self.prefix_engine, prefix_inputs))
        cache_inputs = {name: prefix_outputs[name].contiguous() for name in self.cache_names}

        dt = -1.0 / num_inference_steps
        x_t = noise.contiguous()
        batch_size = tokens.shape[0]
        for step in range(num_inference_steps):
            timestep = torch.full(
                (batch_size,),
                1.0 + step * dt,
                dtype=torch.float32,
                device=tokens.device,
            )
            denoise_inputs = dict(cache_inputs)
            denoise_inputs["x_t"] = x_t
            denoise_inputs["timestep"] = timestep
            denoise_outputs = self.denoise_engine(**self._cast_for_engine(self.denoise_engine, denoise_inputs))
            x_t = x_t + dt * denoise_outputs["v_t"]
        return x_t

    def describe(self) -> dict[str, Any]:
        return {
            "profile": self.profile.name,
            "prefix_backend": "tensorrt",
            "prefix_engine_path": str(self.profile.prefix_engine_path),
            "denoise_engine_path": str(self.profile.denoise_engine_path),
            "prefix_inputs": self.prefix_engine.input_names,
            "prefix_outputs": self.prefix_engine.output_names,
            "denoise_inputs": self.denoise_engine.input_names,
            "denoise_outputs": self.denoise_engine.output_names,
        }


class PurePI05TRTPolicyAdapter:
    """Small policy-compatible adapter with no PyTorch model weights.

    It implements the subset used by `lerobot.utils.control_utils.predict_action`:
    `config`, `reset`, `select_action`, and `predict_action_chunk`.
    """

    def __init__(self, runtime_assets_dir: Path, runtime: PurePI05TRTRuntime, *, device: str = "cuda") -> None:
        config = PreTrainedConfig.from_pretrained(str(runtime_assets_dir))
        config.device = device
        config.use_amp = False
        self.config = config
        self.runtime = runtime
        self._device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def eval(self) -> "PurePI05TRTPolicyAdapter":
        return self

    def parameters(self):
        yield torch.empty((), device=self._device)

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. batch={batch.keys()} "
                f"image_features={self.config.image_features}"
            )

        last_img = None
        last_mask = None
        for key in present_img_keys:
            img = batch[key].to(device=self._device, dtype=torch.float32)
            is_channels_first = img.shape[1] == 3
            if is_channels_first:
                img = img.permute(0, 2, 3, 1)
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)
            img = img * 2.0 - 1.0
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)
            images.append(img.contiguous())
            mask = torch.ones(img.shape[0], dtype=torch.bool, device=self._device)
            img_masks.append(mask)
            last_img = img
            last_mask = mask

        for _ in missing_img_keys:
            if last_img is None or last_mask is None:
                raise RuntimeError("Cannot create padded camera input before seeing a real image")
            images.append((torch.ones_like(last_img) * -1).contiguous())
            img_masks.append(torch.zeros_like(last_mask))

        return images, img_masks

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        if kwargs:
            raise NotImplementedError("RTC/action-select kwargs are not implemented in pure TensorRT adapter")
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[f"{OBS_LANGUAGE_TOKENS}"].to(self._device)
        masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"].to(self._device)
        actions = self.runtime.sample_actions(
            images=images,
            img_masks=img_masks,
            tokens=tokens,
            masks=masks,
            chunk_size=self.config.chunk_size,
            max_action_dim=self.config.max_action_dim,
            num_inference_steps=self.config.num_inference_steps,
        )
        original_action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :original_action_dim]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
