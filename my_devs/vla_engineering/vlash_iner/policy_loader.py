#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .common import ensure_local_tokenizer_dir, validate_policy_artifacts


def log_info(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


@dataclass
class Pi05PolicyBundle:
    policy: Any
    policy_cfg: Any
    preprocessor: Any
    postprocessor: Any
    device: torch.device
    policy_path: Path

    def reset(self) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()


def load_pre_post_processors(policy_path: Path):
    from lerobot.processor import PolicyAction, PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition,
        policy_action_to_transition,
        transition_to_batch,
        transition_to_policy_action,
    )

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


def load_pi05_bundle(
    policy_path: Path,
    *,
    repo_root: Path,
    strict: bool = False,
    check_artifacts: bool = True,
    expected_state_dim: int | None = 6,
    expected_action_dim: int | None = 6,
    summarize_artifacts: bool = True,
) -> Pi05PolicyBundle:
    from lerobot import policies  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class
    from lerobot.utils.utils import get_safe_torch_device

    policy_path = policy_path.expanduser().resolve()
    log_info(f"Resolving policy path: {policy_path}")
    if not policy_path.is_dir():
        raise FileNotFoundError(f"Policy path does not exist: {policy_path}")

    if check_artifacts:
        log_info("Checking policy artifact files and feature shapes...")
        validate_policy_artifacts(
            policy_path,
            expected_state_dim=expected_state_dim,
            expected_action_dim=expected_action_dim,
            summarize=summarize_artifacts,
        )

    log_info("Checking local PI0.5 tokenizer directory...")
    ensure_local_tokenizer_dir(repo_root)

    log_info("Loading policy config...")
    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    policy_cfg.pretrained_path = policy_path
    log_info(f"Loading policy weights on configured device={policy_cfg.device}...")
    policy_class = get_policy_class(policy_cfg.type)
    policy = policy_class.from_pretrained(str(policy_path), strict=strict)
    device = get_safe_torch_device(policy.config.device)
    log_info(f"Moving policy to device={device}...")
    policy.to(device)
    policy.eval()

    log_info("Loading saved preprocessor and postprocessor...")
    preprocessor, postprocessor = load_pre_post_processors(policy_path)
    log_info("Policy bundle is ready.")
    return Pi05PolicyBundle(
        policy=policy,
        policy_cfg=policy_cfg,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=device,
        policy_path=policy_path,
    )


def warmup_select_action(bundle: Pi05PolicyBundle, observation: dict, *, task: str, robot_type: str, steps: int) -> None:
    from lerobot.utils.control_utils import predict_action

    if steps <= 0:
        return

    for _ in range(steps):
        _ = predict_action(
            observation=observation,
            policy=bundle.policy,
            device=bundle.device,
            preprocessor=bundle.preprocessor,
            postprocessor=bundle.postprocessor,
            use_amp=bundle.policy.config.use_amp,
            task=task,
            robot_type=robot_type,
        )
