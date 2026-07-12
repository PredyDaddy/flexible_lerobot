#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
from safetensors import safe_open

TRAINABLE_KEY = "action_head.action_decoder.layer1.W"
FROZEN_KEY = "backbone.model.model.language_model.layers.0.self_attn.q_proj.weight"


def ensure_within(path: Path, root: Path, *, must_exist: bool = False) -> Path:
    resolved = path.expanduser().resolve(strict=must_exist)
    allowed = root.expanduser().resolve(strict=True)
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"Path escapes allowed root: {resolved}") from exc
    return resolved


def load_tensor(model_dir: Path, key: str) -> torch.Tensor:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard = model_dir / index["weight_map"][key]
    else:
        shard = model_dir / "model.safetensors"
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def checkpoint_state(checkpoint: Path, expected_step: int) -> dict:
    state_path = checkpoint / "trainer_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["global_step"] != expected_step:
        raise RuntimeError(
            f"Unexpected global step in {state_path}: {state['global_step']} != {expected_step}"
        )
    for filename in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        path = checkpoint / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing resumable checkpoint state: {path}")
    return state


def scheduler_state(checkpoint: Path, expected_step: int) -> dict:
    state = torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=True)
    if state.get("last_epoch") != expected_step:
        raise RuntimeError(
            f"Unexpected scheduler epoch in {checkpoint}: {state.get('last_epoch')} != {expected_step}"
        )
    last_lrs = [float(value) for value in state.get("_last_lr", [])]
    if not last_lrs or not all(math.isfinite(value) and value >= 0 for value in last_lrs):
        raise RuntimeError(f"Invalid scheduler learning rates in {checkpoint}: {last_lrs}")
    return {"last_epoch": state["last_epoch"], "last_lrs": last_lrs}


def max_abs_difference(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    allow_dtype_change: bool = False,
) -> float:
    if left.shape != right.shape:
        raise RuntimeError(f"Tensor shape changed: {left.shape} != {right.shape}")
    if left.dtype != right.dtype and not allow_dtype_change:
        raise RuntimeError(f"Tensor dtype changed: {left.dtype} != {right.dtype}")
    return float((left.float() - right.float()).abs().max())


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate two-stage N1.7 smoke training.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--stage-one-log", type=Path, required=True)
    parser.add_argument("--stage-two-log", type=Path, required=True)
    parser.add_argument("--gpu-log", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.expanduser().resolve(strict=True)
    base_model = ensure_within(args.base_model, root, must_exist=True)
    train_dir = ensure_within(args.train_dir, root, must_exist=True)
    stage_one_log = ensure_within(args.stage_one_log, root, must_exist=True)
    stage_two_log = ensure_within(args.stage_two_log, root, must_exist=True)
    gpu_log = ensure_within(args.gpu_log, root, must_exist=True)
    report_path = ensure_within(args.report, root)
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite smoke report: {report_path}")

    checkpoint_one = train_dir / "checkpoint-1"
    checkpoint_two = train_dir / "checkpoint-2"
    state_one = checkpoint_state(checkpoint_one, 1)
    state_two = checkpoint_state(checkpoint_two, 2)
    scheduler_one = scheduler_state(checkpoint_one, 1)
    scheduler_two = scheduler_state(checkpoint_two, 2)

    stage_two_text = stage_two_log.read_text(encoding="utf-8", errors="replace")
    resume_pattern = re.compile(r"Resuming from checkpoint .*checkpoint-1")
    if resume_pattern.search(stage_two_text) is None:
        raise RuntimeError("Stage two log does not prove resume from checkpoint-1")

    base_trainable = load_tensor(base_model, TRAINABLE_KEY)
    step_one_trainable = load_tensor(checkpoint_one, TRAINABLE_KEY)
    step_two_trainable = load_tensor(checkpoint_two, TRAINABLE_KEY)
    base_frozen = load_tensor(base_model, FROZEN_KEY)
    step_two_frozen = load_tensor(checkpoint_two, FROZEN_KEY)

    trainable_base_to_step_one = max_abs_difference(
        base_trainable,
        step_one_trainable,
        allow_dtype_change=True,
    )
    trainable_step_one_to_step_two = max_abs_difference(step_one_trainable, step_two_trainable)
    frozen_base_to_step_two = max_abs_difference(base_frozen, step_two_frozen)
    if not math.isfinite(trainable_base_to_step_one):
        raise RuntimeError("Trainable action-head tensor has a non-finite step-1 delta")
    if not math.isfinite(trainable_step_one_to_step_two) or trainable_step_one_to_step_two <= 0:
        raise RuntimeError("Trainable action-head tensor did not update after resume at step 2")
    if frozen_base_to_step_two != 0:
        raise RuntimeError("Frozen backbone tensor changed during smoke training")

    metric_values = []
    for state in (state_one, state_two):
        for entry in state.get("log_history", []):
            for key in ("loss", "train_loss"):
                if key in entry:
                    value = float(entry[key])
                    if not math.isfinite(value):
                        raise RuntimeError(f"Non-finite {key} in trainer state: {value}")
                    metric_values.append({"step": entry.get("step"), "name": key, "value": value})
    if not metric_values:
        loss_pattern = re.compile(r"['\"](?:loss|train_loss)['\"]\s*:\s*([0-9.eE+-]+)")
        for log_path in (stage_one_log, stage_two_log):
            text = log_path.read_text(encoding="utf-8", errors="replace")
            for match in loss_pattern.finditer(text):
                value = float(match.group(1))
                if not math.isfinite(value):
                    raise RuntimeError(f"Non-finite loss in {log_path}: {value}")
                metric_values.append({"step": None, "name": "terminal_loss", "value": value})
    if not metric_values:
        raise RuntimeError("No finite training loss was recorded in trainer state or logs")

    used_memory_values = []
    for line in gpu_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line[0].isdigit() is False or "," not in line:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 4 and fields[0].isdigit():
            used_memory_values.append(int(float(fields[3])))

    report = {
        "schema_version": 1,
        "status": "passed",
        "train_dir": str(train_dir),
        "checkpoint_steps": [state_one["global_step"], state_two["global_step"]],
        "resume_from_checkpoint_one": True,
        "scheduler_states": {
            "checkpoint_1": scheduler_one,
            "checkpoint_2": scheduler_two,
        },
        "trainable_tensor": TRAINABLE_KEY,
        "trainable_tensor_dtypes": {
            "base": str(base_trainable.dtype),
            "checkpoint_1": str(step_one_trainable.dtype),
            "checkpoint_2": str(step_two_trainable.dtype),
        },
        "trainable_base_to_step_one_max_abs_diff": trainable_base_to_step_one,
        "trainable_step_one_to_step_two_max_abs_diff": trainable_step_one_to_step_two,
        "first_step_zero_delta_consistent_with_warmup": trainable_base_to_step_one == 0,
        "frozen_tensor": FROZEN_KEY,
        "frozen_tensor_dtypes": {
            "base": str(base_frozen.dtype),
            "checkpoint_2": str(step_two_frozen.dtype),
        },
        "frozen_base_to_step_two_max_abs_diff": frozen_base_to_step_two,
        "loss_metrics": metric_values,
        "peak_sampled_gpu_memory_mib": max(used_memory_values) if used_memory_values else None,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
