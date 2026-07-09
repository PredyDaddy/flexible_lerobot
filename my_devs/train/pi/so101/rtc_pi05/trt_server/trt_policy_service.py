from __future__ import annotations

import sys
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor


def resolve_repo_root(start: Path) -> Path:
    resolved = start.expanduser().resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {start}")


REPO_ROOT = resolve_repo_root(Path(__file__))
OPENPI_TRT_DIR = REPO_ROOT / "my_devs" / "openpi_trt"
for _path in (REPO_ROOT, OPENPI_TRT_DIR):
    if _path.as_posix() not in sys.path:
        sys.path.insert(0, _path.as_posix())

from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402
from lerobot.policies.utils import prepare_observation_for_inference  # noqa: E402
from lerobot.processor import PolicyAction, PolicyProcessorPipeline  # noqa: E402
from lerobot.processor.converters import (  # noqa: E402
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.utils import get_safe_torch_device  # noqa: E402
from runtime.pure_pi05_trt import PurePI05TRTProfile, PurePI05TRTRuntime  # noqa: E402

from my_devs.train.pi.so101.rtc_pi05.server.policy_service import (  # noqa: E402
    ensure_chunk_batch,
    postprocess_action_chunk,
)
from my_devs.train.pi.so101.rtc_pi05.server.protocol import InferenceRequest, InferenceResponse  # noqa: E402
from my_devs.train.pi.so101.rtc_pi05.trt_server.rtc_trt_policy_adapter import (  # noqa: E402
    RTCPurePI05TRTPolicyAdapter,
)


REQUIRED_RUNTIME_ASSETS = (
    "config.json",
    "policy_preprocessor.json",
    "policy_preprocessor_step_2_normalizer_processor.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)


@dataclass(slots=True)
class TRTPolicyServiceConfig:
    runtime_assets_dir: Path
    prefix_engine_path: Path
    denoise_engine_path: Path
    profile: str = "auto"
    enable_rtc: bool = True
    device: str = "cuda"
    strict_runtime_assets: bool = True

    def __post_init__(self) -> None:
        self.runtime_assets_dir = Path(self.runtime_assets_dir).expanduser()
        self.prefix_engine_path = Path(self.prefix_engine_path).expanduser()
        self.denoise_engine_path = Path(self.denoise_engine_path).expanduser()
        if self.profile not in {"auto", "fp32", "fp16_constrained"}:
            raise ValueError(f"Unsupported TensorRT profile: {self.profile}")

    def resolved_profile_name(self) -> str:
        if self.profile != "auto":
            return self.profile
        engine_names = f"{self.prefix_engine_path.name} {self.denoise_engine_path.name}"
        return "fp16_constrained" if "fp16" in engine_names else "fp32"


@dataclass(slots=True)
class TRTPolicyBundle:
    runtime: PurePI05TRTRuntime
    policy: RTCPurePI05TRTPolicyAdapter
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction]


class TRTPolicyService:
    """HTTP-service friendly wrapper for PI0.5 pure TensorRT chunk inference."""

    def __init__(self, *, config: TRTPolicyServiceConfig, bundle: TRTPolicyBundle) -> None:
        self.config = config
        self.bundle = bundle
        self.runtime = bundle.runtime
        self.policy = bundle.policy
        self.preprocessor = bundle.preprocessor
        self.postprocessor = bundle.postprocessor
        self.device = get_safe_torch_device(self.policy.config.device)
        self._lock = threading.Lock()
        self.inference_count = 0
        self.last_latency_s = 0.0

    @classmethod
    def from_config(cls, config: TRTPolicyServiceConfig, *, rtc_config: RTCConfig | None = None) -> "TRTPolicyService":
        bundle = load_trt_policy_bundle(config, rtc_config=rtc_config)
        return cls(config=config, bundle=bundle)

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        started_s = time.perf_counter()
        with self._lock:
            model_started_s = time.perf_counter()
            raw_chunk, processed_chunk = run_trt_chunk_inference(
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

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "pure_trt_rtc",
            "profile": self.config.resolved_profile_name(),
            "runtime_assets_dir": str(self.config.runtime_assets_dir),
            "enable_rtc": self.config.enable_rtc,
            "device": str(self.device),
            "runtime": self.runtime.describe(),
        }


def validate_runtime_assets(runtime_assets_dir: Path, *, strict: bool = True) -> None:
    if not runtime_assets_dir.is_dir():
        raise FileNotFoundError(f"Runtime assets directory does not exist: {runtime_assets_dir}")
    missing = [name for name in REQUIRED_RUNTIME_ASSETS if not (runtime_assets_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Runtime assets directory is missing required files: {missing}")
    if strict and (runtime_assets_dir / "model.safetensors").exists():
        raise RuntimeError(f"Pure TensorRT runtime assets must not contain model.safetensors: {runtime_assets_dir}")


def load_pre_post_processors(
    runtime_assets_dir: Path,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(runtime_assets_dir),
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(runtime_assets_dir),
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


def load_trt_policy_bundle(
    config: TRTPolicyServiceConfig,
    *,
    rtc_config: RTCConfig | None = None,
) -> TRTPolicyBundle:
    validate_runtime_assets(config.runtime_assets_dir, strict=config.strict_runtime_assets)
    if not config.prefix_engine_path.is_file():
        raise FileNotFoundError(f"Prefix TensorRT engine does not exist: {config.prefix_engine_path}")
    if not config.denoise_engine_path.is_file():
        raise FileNotFoundError(f"Denoise TensorRT engine does not exist: {config.denoise_engine_path}")

    resolved_rtc_config = rtc_config
    if config.enable_rtc and resolved_rtc_config is None:
        resolved_rtc_config = RTCConfig(enabled=True)
    if not config.enable_rtc:
        resolved_rtc_config = None

    runtime = PurePI05TRTRuntime(
        PurePI05TRTProfile(
            name=config.resolved_profile_name(),
            prefix_engine_path=config.prefix_engine_path,
            denoise_engine_path=config.denoise_engine_path,
        )
    )
    policy = RTCPurePI05TRTPolicyAdapter(
        config.runtime_assets_dir,
        runtime,
        device=config.device,
        rtc_config=resolved_rtc_config,
    )
    preprocessor, postprocessor = load_pre_post_processors(config.runtime_assets_dir)
    return TRTPolicyBundle(
        runtime=runtime,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )


def run_trt_chunk_inference(
    *,
    policy: RTCPurePI05TRTPolicyAdapter,
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
