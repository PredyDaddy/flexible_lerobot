from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors import safe_open


def resolve_repo_root(start: Path) -> Path:
    resolved = start.expanduser().resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {start}")


REPO_ROOT = resolve_repo_root(Path(__file__))
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from lerobot import policies  # noqa: F401,E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import get_policy_class  # noqa: E402
from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402
from lerobot.processor import PolicyAction, PolicyProcessorPipeline  # noqa: E402
from lerobot.processor.converters import (  # noqa: E402
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402


@dataclass(slots=True)
class PolicyBundle:
    repo_root: Path
    policy_path: Path
    policy_config: PreTrainedConfig
    policy: Any
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction]


def ensure_policy_registry() -> None:
    register_third_party_plugins()


def ensure_local_tokenizer_dir(repo_root: Path) -> Path:
    local_tok = repo_root / "google" / "paligemma-3b-pt-224"
    if not local_tok.is_dir():
        raise FileNotFoundError(
            "Missing local tokenizer directory for offline PI0.5 inference.\n"
            f"Expected: {local_tok}"
        )
    return local_tok


def validate_policy_artifacts(policy_path: Path, *, strict_so101_features: bool = True) -> None:
    policy_path = Path(policy_path).expanduser()
    is_peft_checkpoint = (policy_path / "adapter_config.json").is_file()
    weight_file = "adapter_model.safetensors" if is_peft_checkpoint else "model.safetensors"
    required_files = [
        "config.json",
        weight_file,
        "policy_preprocessor.json",
        "policy_preprocessor_step_2_normalizer_processor.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "train_config.json",
    ]
    if is_peft_checkpoint:
        required_files.append("adapter_config.json")
    missing = [name for name in required_files if not (policy_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Policy checkpoint is missing required files: {missing}")

    with (policy_path / "config.json").open() as f:
        config = json.load(f)
    if strict_so101_features:
        expected_inputs = {
            "observation.state": [6],
            "observation.images.top": [3, 480, 640],
            "observation.images.wrist": [3, 480, 640],
        }
        for key, expected_shape in expected_inputs.items():
            feature = config.get("input_features", {}).get(key)
            if feature is None or feature.get("shape") != expected_shape:
                raise ValueError(f"Unexpected input feature for {key}: {feature}. Expected {expected_shape}.")
        action_feature = config.get("output_features", {}).get("action")
        if action_feature is None or action_feature.get("shape") != [6]:
            raise ValueError(f"Unexpected action feature: {action_feature}. Expected [6].")


def summarize_safetensors(path: Path, *, limit: int = 6) -> list[str]:
    lines: list[str] = []
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        lines.append(f"{path.name}: bytes={path.stat().st_size} tensors={len(keys)}")
        for key in keys[:limit]:
            tensor_slice = handle.get_slice(key)
            lines.append(f"  {key}: shape={tuple(tensor_slice.get_shape())} dtype={tensor_slice.get_dtype()}")
    return lines


def load_pre_post_processors(
    policy_path: Path,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(policy_path),
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(policy_path),
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


def load_policy(policy_path: Path, policy_cfg: PreTrainedConfig):
    policy_class = get_policy_class(policy_cfg.type)
    adapter_config_path = policy_path / "adapter_config.json"
    if adapter_config_path.is_file():
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(str(policy_path))
        base_model_path = peft_config.base_model_name_or_path
        if not base_model_path:
            raise ValueError(f"Missing base_model_name_or_path in {adapter_config_path}")
        base_policy = policy_class.from_pretrained(base_model_path, config=policy_cfg, strict=False)
        policy = PeftModel.from_pretrained(base_policy, str(policy_path), config=peft_config)
        policy.config = base_policy.config
        return policy
    return policy_class.from_pretrained(str(policy_path), strict=False)


def enable_policy_rtc(policy: Any, rtc_config: RTCConfig) -> Any:
    policy.config.rtc_config = rtc_config
    if hasattr(policy, "init_rtc_processor"):
        policy.init_rtc_processor()
        return policy

    base_model = getattr(policy, "base_model", None)
    candidates = [base_model, getattr(base_model, "model", None), getattr(policy, "model", None)]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "config") and hasattr(candidate, "init_rtc_processor"):
            candidate.config.rtc_config = rtc_config
            candidate.init_rtc_processor()
            return policy
    raise TypeError("Cannot enable RTC: underlying PI05Policy was not found.")


def disable_policy_rtc(policy: Any) -> Any:
    policy.config.rtc_config = None
    if hasattr(policy, "init_rtc_processor"):
        policy.init_rtc_processor()
        return policy
    base_model = getattr(policy, "base_model", None)
    candidates = [base_model, getattr(base_model, "model", None), getattr(policy, "model", None)]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "config") and hasattr(candidate, "init_rtc_processor"):
            candidate.config.rtc_config = None
            candidate.init_rtc_processor()
            return policy
    return policy


def load_policy_bundle(
    policy_path: Path,
    *,
    repo_root: Path | None = None,
    device_override: str | None = None,
    strict_so101_features: bool = True,
) -> PolicyBundle:
    ensure_policy_registry()
    resolved_repo_root = REPO_ROOT if repo_root is None else Path(repo_root).expanduser().resolve()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ensure_local_tokenizer_dir(resolved_repo_root)
    policy_path = Path(policy_path).expanduser()
    if not policy_path.is_dir():
        raise FileNotFoundError(f"Policy path does not exist: {policy_path}")
    validate_policy_artifacts(policy_path, strict_so101_features=strict_so101_features)
    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    policy_cfg.pretrained_path = policy_path
    if device_override is not None:
        policy_cfg.device = device_override
    policy = load_policy(policy_path, policy_cfg)
    policy.to(policy_cfg.device)
    preprocessor, postprocessor = load_pre_post_processors(policy_path)
    return PolicyBundle(
        repo_root=resolved_repo_root,
        policy_path=policy_path,
        policy_config=policy_cfg,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )
