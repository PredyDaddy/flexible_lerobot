from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def resolve_repo_root(start: Path) -> Path:
    resolved = start.expanduser().resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {start}")


REPO_ROOT = resolve_repo_root(Path(__file__))
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402
from lerobot.policies.rtc.modeling_rtc import RTCProcessor  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: E402
from my_devs.train.pi.so101.pure_trt.runtime.pure_pi05_trt import (  # noqa: E402
    PurePI05TRTPolicyAdapter,
    PurePI05TRTRuntime,
)


class RTCPurePI05TRTPolicyAdapter(PurePI05TRTPolicyAdapter):
    """Pure TensorRT PI0.5 adapter with RTC-aware denoise-loop guidance.

    The TensorRT engines still implement the ordinary prefix_cache and denoise_step
    boundaries. RTC guidance stays in Python, mirroring PI05Pytorch.sample_actions.
    """

    _RTC_KWARGS = {"inference_delay", "prev_chunk_left_over", "execution_horizon", "noise"}

    def __init__(
        self,
        runtime_assets_dir: Path,
        runtime: PurePI05TRTRuntime,
        *,
        device: str = "cuda",
        rtc_config: RTCConfig | None = None,
    ) -> None:
        super().__init__(runtime_assets_dir, runtime, device=device)
        self.config.rtc_config = rtc_config
        self.rtc_processor = RTCProcessor(rtc_config) if rtc_config is not None else None

    def init_rtc_processor(self) -> None:
        self.rtc_processor = (
            RTCProcessor(self.config.rtc_config) if self.config.rtc_config is not None else None
        )

    def _rtc_enabled(self) -> bool:
        rtc_config = getattr(self.config, "rtc_config", None)
        return bool(rtc_config is not None and rtc_config.enabled and self.rtc_processor is not None)

    def reset(self) -> None:
        super().reset()
        if getattr(self, "rtc_processor", None) is not None:
            self.rtc_processor.reset_tracker()

    @staticmethod
    def _as_leftover_tensor(prev_chunk_left_over: Tensor | None, *, device: torch.device) -> Tensor | None:
        if prev_chunk_left_over is None:
            return None
        return torch.as_tensor(prev_chunk_left_over, dtype=torch.float32, device=device).contiguous()

    @torch.no_grad()
    def _sample_actions_rtc_aware(
        self,
        *,
        images: list[Tensor],
        img_masks: list[Tensor],
        tokens: Tensor,
        masks: Tensor,
        noise: Tensor | None = None,
        inference_delay: int | None = None,
        prev_chunk_left_over: Tensor | None = None,
        execution_horizon: int | None = None,
    ) -> Tensor:
        if len(images) != 2 or len(img_masks) != 2:
            raise ValueError(f"Expected two SO101 images/masks, got {len(images)} images and {len(img_masks)} masks")

        if noise is None:
            noise = torch.normal(
                mean=0.0,
                std=1.0,
                size=(tokens.shape[0], self.config.chunk_size, self.config.max_action_dim),
                dtype=torch.float32,
                device=tokens.device,
            )
        else:
            noise = noise.to(device=tokens.device, dtype=torch.float32).contiguous()

        prefix_inputs = {
            "image_0": images[0],
            "image_1": images[1],
            "img_mask_0": img_masks[0],
            "img_mask_1": img_masks[1],
            "tokens": tokens,
            "masks": masks,
        }
        runtime = self.runtime
        prefix_outputs = runtime.prefix_engine(**runtime._cast_for_engine(runtime.prefix_engine, prefix_inputs))
        cache_inputs = {name: prefix_outputs[name].contiguous() for name in runtime.cache_names}

        dt = -1.0 / self.config.num_inference_steps
        x_t = noise.contiguous()
        batch_size = tokens.shape[0]
        left_over = self._as_leftover_tensor(prev_chunk_left_over, device=x_t.device)
        delay_steps = 0 if inference_delay is None else int(inference_delay)

        for step in range(self.config.num_inference_steps):
            time_value = 1.0 + step * dt
            timestep = torch.full(
                (batch_size,),
                time_value,
                dtype=torch.float32,
                device=tokens.device,
            )

            def denoise_step_partial(input_x_t: Tensor, current_timestep: Tensor = timestep) -> Tensor:
                denoise_inputs = dict(cache_inputs)
                denoise_inputs["x_t"] = input_x_t.contiguous()
                denoise_inputs["timestep"] = current_timestep
                outputs = runtime.denoise_engine(
                    **runtime._cast_for_engine(runtime.denoise_engine, denoise_inputs)
                )
                return outputs["v_t"].to(dtype=torch.float32)

            if self._rtc_enabled():
                assert self.rtc_processor is not None
                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=left_over,
                    inference_delay=delay_steps,
                    time=time_value,
                    original_denoise_step_partial=denoise_step_partial,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time_value, x_t=x_t, v_t=v_t)

        return x_t

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        unsupported = set(kwargs) - self._RTC_KWARGS
        if unsupported:
            raise NotImplementedError(f"Unsupported pure TensorRT predict kwargs: {sorted(unsupported)}")

        images, img_masks = self._preprocess_images(batch)
        tokens = batch[f"{OBS_LANGUAGE_TOKENS}"].to(self._device)
        masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"].to(self._device)
        actions = self._sample_actions_rtc_aware(
            images=images,
            img_masks=img_masks,
            tokens=tokens,
            masks=masks,
            noise=kwargs.get("noise"),
            inference_delay=kwargs.get("inference_delay"),
            prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
            execution_horizon=kwargs.get("execution_horizon"),
        )
        original_action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :original_action_dim]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        if self._rtc_enabled():
            raise AssertionError("RTC is not supported for select_action, use predict_action_chunk")
        return super().select_action(batch)
