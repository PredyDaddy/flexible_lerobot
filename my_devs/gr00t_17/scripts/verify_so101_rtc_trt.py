#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.functional import cosine_similarity

GR00T17_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = GR00T17_ROOT / "workspace" / "Isaac-GR00T-n1.7"
DEPLOYMENT = WORKSPACE / "scripts" / "deployment"
for path in (Path(__file__).resolve().parent, WORKSPACE, DEPLOYMENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from export_onnx_n1d7 import prepare_observation  # noqa: E402
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from so101_rtc_policy_server import RtcGr00tPolicy  # noqa: E402
from so101_rtc_trt import setup_rtc_tensorrt_engines  # noqa: E402

ENGINE_NAMES = (
    "vit_bf16.engine",
    "llm_bf16.engine",
    "vl_self_attention.engine",
    "state_encoder.engine",
    "action_encoder.engine",
    "dit_bf16.engine",
    "action_decoder.engine",
)
RTC_OPTIONS = {
    "rtc_enabled": True,
    "rtc_advance_steps": 8,
    "rtc_frozen_steps": 2,
    "rtc_ramp_rate": 2.0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flatten_action(action: dict[str, np.ndarray]) -> torch.Tensor:
    return torch.cat([torch.from_numpy(np.asarray(action[key])).float().flatten() for key in sorted(action)])


def compare(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> dict[str, float]:
    left = flatten_action(reference)
    right = flatten_action(candidate)
    return {
        "cosine": float(cosine_similarity(left.unsqueeze(0), right.unsqueeze(0)).item()),
        "l1_mean": float((left - right).abs().mean().item()),
        "linf": float((left - right).abs().max().item()),
    }


def run_sequence(policy: RtcGr00tPolicy, observation: dict[str, Any]) -> list[dict[str, np.ndarray]]:
    policy.reset({"reason": "trt_verification"})
    outputs = []
    for seed in (42, 43):
        torch.manual_seed(seed)
        action, _ = policy.get_action(observation, RTC_OPTIONS)
        outputs.append(action)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify RTC-aware SO101 TensorRT engines")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    model_path = args.model_path.resolve(strict=True)
    dataset_path = args.dataset_path.resolve(strict=True)
    engine_dir = args.engine_dir.resolve(strict=True)
    report_path = args.report.resolve()
    for path in (model_path, dataset_path, engine_dir, report_path.parent):
        if not path.is_relative_to(GR00T17_ROOT.resolve()):
            raise ValueError(f"Path escapes GR00T17_ROOT: {path}")
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite report: {report_path}")
    engine_files = [engine_dir / name for name in ENGINE_NAMES]
    for engine in engine_files:
        if not engine.is_file() or engine.stat().st_size == 0:
            raise FileNotFoundError(f"Missing TensorRT engine: {engine}")

    started = time.perf_counter()
    policy = RtcGr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=str(model_path),
        device="cuda:0",
        strict=True,
    )
    dataset = LeRobotEpisodeLoader(
        dataset_path=str(dataset_path),
        modality_configs=policy.get_modality_config(),
        video_backend="torchcodec",
        video_backend_kwargs=None,
    )
    observation = prepare_observation(policy, dataset, traj_idx=0)
    pytorch_outputs = run_sequence(policy, observation)

    setup_rtc_tensorrt_engines(policy, engine_dir, "n17_full_pipeline")
    policy.inference_backend = "tensorrt"
    policy.trt_mode = "n17_full_pipeline"
    tensorrt_outputs = run_sequence(policy, observation)
    comparisons = {
        "initial_chunk": compare(pytorch_outputs[0], tensorrt_outputs[0]),
        "rtc_chunk": compare(pytorch_outputs[1], tensorrt_outputs[1]),
    }
    if min(item["cosine"] for item in comparisons.values()) <= 0.999:
        raise RuntimeError(f"TensorRT RTC verification failed: {comparisons}")

    report = {
        "schema_version": 1,
        "status": "passed",
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "engine_dir": str(engine_dir),
        "rtc_options": RTC_OPTIONS,
        "comparisons": comparisons,
        "engines": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in engine_files
        ],
        "elapsed_s": time.perf_counter() - started,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
