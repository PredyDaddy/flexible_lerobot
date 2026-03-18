from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from lerobot.async_inference.configs import PolicyServerConfig as BasePolicyServerConfig
from lerobot.async_inference.helpers import RemotePolicyConfig as BaseRemotePolicyConfig

DEFAULT_SERVER_RESOURCE_PROFILE = "default"
DEFAULT_SERVER_POLICY_PATH = (
    "/data/cqy_workspace/flexible_lerobot/outputs/train/"
    "groot_grasp_block_in_bin1_repro_20260302_223413/bs32_20260302_223447/"
    "checkpoints/last/pretrained_model"
)
DEFAULT_SERVER_ENGINE_DIR = (
    "/data/cqy_workspace/flexible_lerobot/outputs/trt/"
    "consistency_rerun_20260305_102931/gr00t_engine_api_trt1013"
)


@dataclass(frozen=True)
class GrootTrtResolvedResourceSpec:
    resource_profile: str
    pretrained_name_or_path: str
    engine_dir: str | None
    tensorrt_py_dir: str | None


@dataclass
class GrootTrtPolicyServerConfig(BasePolicyServerConfig):
    """Config for the GR00T TensorRT-aware async policy server."""

    resource_profile: str = field(
        default=DEFAULT_SERVER_RESOURCE_PROFILE,
        metadata={"help": "Logical server-managed resource profile accepted from the client."},
    )
    resource_policy_path: str = field(
        default=DEFAULT_SERVER_POLICY_PATH,
        metadata={"help": "Server-local pretrained_model directory resolved for the active resource profile."},
    )
    resource_engine_dir: str | None = field(
        default=DEFAULT_SERVER_ENGINE_DIR,
        metadata={"help": "Server-local TensorRT engine directory resolved for the active resource profile."},
    )
    resource_tensorrt_py_dir: str | None = field(
        default=None,
        metadata={"help": "Optional server-local TensorRT Python package directory for the active profile."},
    )

    def __post_init__(self) -> None:
        if not self.resource_profile:
            raise ValueError("resource_profile cannot be empty")
        if not self.resource_policy_path:
            raise ValueError("resource_policy_path cannot be empty")
        if self.resource_engine_dir == "":
            raise ValueError("resource_engine_dir cannot be an empty string")

    def resolve_policy_resources(self, requested_profile: str, *, backend: str) -> GrootTrtResolvedResourceSpec:
        if requested_profile != self.resource_profile:
            raise ValueError(
                f"Unknown resource_profile={requested_profile!r}. "
                f"Only server-managed profile {self.resource_profile!r} is allowed."
            )

        engine_dir = self.resource_engine_dir if backend == "tensorrt" else None
        if backend == "tensorrt" and not engine_dir:
            raise ValueError(
                f"resource_profile={requested_profile!r} is missing a server-local engine directory "
                "required for backend='tensorrt'."
            )

        return GrootTrtResolvedResourceSpec(
            resource_profile=requested_profile,
            pretrained_name_or_path=self.resource_policy_path,
            engine_dir=engine_dir,
            tensorrt_py_dir=self.resource_tensorrt_py_dir,
        )

    def resolve_policy_specs(self, policy_specs: "GrootTrtRemotePolicyConfig") -> "GrootTrtRemotePolicyConfig":
        resolved = self.resolve_policy_resources(policy_specs.resource_profile, backend=policy_specs.backend)
        return replace(
            policy_specs,
            pretrained_name_or_path=resolved.pretrained_name_or_path,
            engine_dir=resolved.engine_dir,
            tensorrt_py_dir=resolved.tensorrt_py_dir,
        )


@dataclass
class GrootTrtRemotePolicyConfig(BaseRemotePolicyConfig):
    """Remote policy config extended with TensorRT backend settings."""

    resource_profile: str = field(
        default=DEFAULT_SERVER_RESOURCE_PROFILE,
        metadata={"help": "Logical server-managed resource profile to resolve locally on the server."},
    )

    backend: str = field(default="pytorch", metadata={"help": "Inference backend: 'pytorch' or 'tensorrt'."})
    engine_dir: str | None = field(
        default=None,
        metadata={"help": "Directory containing GR00T TensorRT `.engine` files when backend=tensorrt."},
    )
    tensorrt_py_dir: str | None = field(
        default=None,
        metadata={"help": "Optional TensorRT Python package target dir."},
    )
    vit_dtype: str = field(default="fp16", metadata={"help": "TensorRT ViT engine suffix."})
    llm_dtype: str = field(default="fp16", metadata={"help": "TensorRT LLM engine suffix."})
    dit_dtype: str = field(default="fp16", metadata={"help": "TensorRT DiT engine suffix."})
    num_denoising_steps: int | None = field(
        default=None,
        metadata={"help": "Optional override for the GR00T action-head denoising steps."},
    )

    def __post_init__(self) -> None:
        if not self.policy_type:
            raise ValueError("policy_type cannot be empty")

        if not self.pretrained_name_or_path:
            raise ValueError("pretrained_name_or_path cannot be empty")

        if not self.device:
            raise ValueError("device cannot be empty")

        if not self.resource_profile:
            raise ValueError("resource_profile cannot be empty")

        if self.actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {self.actions_per_chunk}")

        if self.backend not in {"pytorch", "tensorrt"}:
            raise ValueError(f"Unsupported backend {self.backend!r}. Expected 'pytorch' or 'tensorrt'.")

        if self.backend != "tensorrt":
            unexpected_trt_fields = []
            if self.engine_dir is not None:
                unexpected_trt_fields.append("engine_dir")
            if self.tensorrt_py_dir is not None:
                unexpected_trt_fields.append("tensorrt_py_dir")
            if self.vit_dtype != "fp16":
                unexpected_trt_fields.append("vit_dtype")
            if self.llm_dtype != "fp16":
                unexpected_trt_fields.append("llm_dtype")
            if self.dit_dtype != "fp16":
                unexpected_trt_fields.append("dit_dtype")
            if self.num_denoising_steps is not None:
                unexpected_trt_fields.append("num_denoising_steps")
            if unexpected_trt_fields:
                raise ValueError(
                    "TensorRT-only fields require backend='tensorrt'. "
                    f"Unexpected fields for backend={self.backend!r}: {unexpected_trt_fields}"
                )

        if self.backend == "tensorrt" and not self.engine_dir and not self.resource_profile:
            raise ValueError("backend='tensorrt' requires either engine_dir or a server-managed resource_profile.")

        if self.backend == "tensorrt" and self.policy_type != "groot":
            raise ValueError("backend='tensorrt' is currently only supported for policy_type='groot'.")

        if self.num_denoising_steps is not None and self.num_denoising_steps <= 0:
            raise ValueError("num_denoising_steps must be positive when provided.")

    @property
    def engine_path(self) -> Path | None:
        return None if not self.engine_dir else Path(self.engine_dir).expanduser()

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_type": self.policy_type,
            "pretrained_name_or_path": self.pretrained_name_or_path,
            "lerobot_features": self.lerobot_features,
            "actions_per_chunk": self.actions_per_chunk,
            "device": self.device,
            "rename_map": self.rename_map,
            "resource_profile": self.resource_profile,
            "backend": self.backend,
            "engine_dir": self.engine_dir,
            "tensorrt_py_dir": self.tensorrt_py_dir,
            "vit_dtype": self.vit_dtype,
            "llm_dtype": self.llm_dtype,
            "dit_dtype": self.dit_dtype,
            "num_denoising_steps": self.num_denoising_steps,
        }

    @classmethod
    def from_payload(cls, payload: Any) -> "GrootTrtRemotePolicyConfig":
        """Normalize a pickled client payload into the local TRT-aware config class.

        The async transport currently pickles dataclass instances over gRPC. We accept:
        - this local config class
        - the upstream `RemotePolicyConfig`
        - any object exposing the expected attributes
        """

        if isinstance(payload, cls):
            return payload

        required = [
            "policy_type",
            "pretrained_name_or_path",
            "lerobot_features",
            "actions_per_chunk",
            "device",
        ]
        missing = [name for name in required if not hasattr(payload, name)]
        if missing:
            raise TypeError(
                "Policy specs must match GrootTrtRemotePolicyConfig or expose the expected attributes. "
                f"Missing: {missing}. Got {type(payload)}"
            )

        data = {
            "policy_type": payload.policy_type,
            "pretrained_name_or_path": payload.pretrained_name_or_path,
            "lerobot_features": payload.lerobot_features,
            "actions_per_chunk": payload.actions_per_chunk,
            "device": getattr(payload, "device", "cpu"),
            "rename_map": getattr(payload, "rename_map", {}) or {},
            "resource_profile": getattr(payload, "resource_profile", DEFAULT_SERVER_RESOURCE_PROFILE),
            "backend": getattr(payload, "backend", "pytorch"),
            "engine_dir": getattr(payload, "engine_dir", None),
            "tensorrt_py_dir": getattr(payload, "tensorrt_py_dir", None),
            "vit_dtype": getattr(payload, "vit_dtype", "fp16"),
            "llm_dtype": getattr(payload, "llm_dtype", "fp16"),
            "dit_dtype": getattr(payload, "dit_dtype", "fp16"),
            "num_denoising_steps": getattr(payload, "num_denoising_steps", None),
        }
        return cls(**data)
