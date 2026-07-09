#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


def resolve_repo_root(script_path: Path) -> Path:
    resolved = script_path.resolve()
    for candidate in resolved.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {script_path}")


REPO_ROOT = resolve_repo_root(Path(__file__))
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from my_devs.train.pi.so101.pure_trt.runtime.paths import default_paths
from my_devs.train.pi.so101.pure_trt.runtime.pure_pi05_trt import (
    PurePI05TRTPolicyAdapter,
    PurePI05TRTProfile,
    PurePI05TRTRuntime,
)


REQUIRED_RUNTIME_ASSETS = (
    "config.json",
    "policy_preprocessor.json",
    "policy_preprocessor_step_2_normalizer_processor.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)


def build_parser() -> argparse.ArgumentParser:
    paths = default_paths()
    parser = argparse.ArgumentParser(description="Load-check and smoke-test the clean SO101 PI0.5 pure TRT runtime.")
    parser.add_argument("--runtime-assets-dir", type=Path, default=paths.runtime_assets_dir)
    parser.add_argument("--profile", choices=["auto", "fp32", "fp16_constrained"], default="auto")
    parser.add_argument("--prefix-engine-path", type=Path, default=paths.prefix_fp16_constrained_engine)
    parser.add_argument("--denoise-engine-path", type=Path, default=paths.denoise_fp16_constrained_engine)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", default="Put the eraser into the small box")
    parser.add_argument("--robot-type", default="so101_follower")
    parser.add_argument("--dummy-infer", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_runtime_assets(args.runtime_assets_dir)
    profile_name = resolve_profile(args.profile, args.prefix_engine_path, args.denoise_engine_path)
    runtime = PurePI05TRTRuntime(
        PurePI05TRTProfile(
            name=profile_name,
            prefix_engine_path=args.prefix_engine_path,
            denoise_engine_path=args.denoise_engine_path,
        )
    )
    print("[PURE-TRT] Runtime loaded:")
    print(json.dumps(runtime.describe(), indent=2, ensure_ascii=True))
    if not args.dummy_infer:
        print("[PURE-TRT] CHECK_RUNTIME passed.")
        return 0

    policy = PurePI05TRTPolicyAdapter(args.runtime_assets_dir, runtime, device=args.device)
    preprocessor, postprocessor = load_pre_post_processors(args.runtime_assets_dir)
    batch = prepare_observation_for_inference(
        {
            "observation.state": np.zeros((6,), dtype=np.float32),
            "observation.images.top": np.zeros((480, 640, 3), dtype=np.uint8),
            "observation.images.wrist": np.zeros((480, 640, 3), dtype=np.uint8),
        },
        device=torch.device(args.device),
        task=args.task,
        robot_type=args.robot_type,
    )
    raw_chunk = ensure_chunk_batch(policy.predict_action_chunk(preprocessor(batch)))
    processed_chunk = postprocess_action_chunk(postprocessor, raw_chunk)
    print(f"[PURE-TRT] dummy raw_shape={tuple(raw_chunk.squeeze(0).shape)}")
    print(f"[PURE-TRT] dummy processed_shape={tuple(processed_chunk.squeeze(0).shape)}")
    print(f"[PURE-TRT] dummy first_action={processed_chunk.squeeze(0)[0].detach().cpu().tolist()}")
    return 0


def validate_runtime_assets(runtime_assets_dir: Path) -> None:
    if not runtime_assets_dir.is_dir():
        raise FileNotFoundError(f"Runtime assets directory does not exist: {runtime_assets_dir}")
    missing = [name for name in REQUIRED_RUNTIME_ASSETS if not (runtime_assets_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Runtime assets directory is missing required files: {missing}")
    if (runtime_assets_dir / "model.safetensors").exists():
        raise RuntimeError(f"Pure TRT runtime assets must not contain model.safetensors: {runtime_assets_dir}")


def resolve_profile(profile: str, prefix_engine_path: Path, denoise_engine_path: Path) -> str:
    if profile != "auto":
        return profile
    engine_names = f"{prefix_engine_path.name} {denoise_engine_path.name}"
    return "fp16_constrained" if "fp16" in engine_names else "fp32"


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


def ensure_chunk_batch(actions: Tensor) -> Tensor:
    if actions.ndim == 2:
        return actions.unsqueeze(0)
    if actions.ndim == 3 and actions.shape[0] == 1:
        return actions
    raise ValueError(f"Expected action chunk shape (T,D) or (1,T,D), got {tuple(actions.shape)}")


def postprocess_action_chunk(postprocessor: PolicyProcessorPipeline, raw_chunk: Tensor) -> Tensor:
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
        return ensure_chunk_batch(processed).reshape(batch_size, chunk_len, action_dim)


if __name__ == "__main__":
    raise SystemExit(main())
