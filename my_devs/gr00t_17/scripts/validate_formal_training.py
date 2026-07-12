#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any


def ensure_within(path: Path, root: Path, *, must_exist: bool = False) -> Path:
    resolved = path.expanduser().resolve(strict=must_exist)
    allowed = root.expanduser().resolve(strict=True)
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"Path escapes allowed root: {resolved}") from exc
    return resolved


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def read_launch_parameters(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in result:
            raise ValueError(f"Invalid or duplicate launch parameter: {line!r}")
        result[key] = value
    return result


def final_train_metrics(log_text: str) -> dict[str, float]:
    for line in reversed(log_text.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("{'train_runtime':"):
            continue
        parsed = ast.literal_eval(stripped)
        required = ("train_runtime", "train_samples_per_second", "train_steps_per_second", "train_loss")
        metrics = {key: float(parsed[key]) for key in required}
        if not all(math.isfinite(value) for value in metrics.values()):
            raise RuntimeError(f"Final training metrics are non-finite: {metrics}")
        return metrics
    raise RuntimeError("Final training metrics were not found in the terminal log")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the completed SO101 N1.7 formal training run.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-report", type=Path, required=True)
    parser.add_argument("--model-load-report", type=Path, required=True)
    parser.add_argument("--model-server-log", type=Path, required=True)
    parser.add_argument("--expected-epochs", type=float, default=10.0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.expanduser().resolve(strict=True)
    run_dir = ensure_within(args.run_dir, root, must_exist=True)
    dataset_report_path = ensure_within(args.dataset_report, root, must_exist=True)
    model_load_report_path = ensure_within(args.model_load_report, root, must_exist=True)
    model_server_log_path = ensure_within(args.model_server_log, root, must_exist=True)
    report_path = ensure_within(args.report, root)
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite formal training report: {report_path}")

    train_dir = run_dir / "train"
    checkpoint_dir = train_dir / "checkpoint-63600"
    launch = read_launch_parameters(run_dir / "reports" / "launch_parameters.env")
    max_steps = int(launch["MAX_STEPS"])
    micro_batch = int(launch["GLOBAL_BATCH_SIZE"])
    gradient_accumulation = int(launch["GRADIENT_ACCUMULATION_STEPS"])
    effective_batch = micro_batch * gradient_accumulation

    dataset_report = read_json(dataset_report_path)
    if dataset_report.get("status") != "passed":
        raise RuntimeError("Dataset validation evidence is not passed")
    valid_windows = int(dataset_report["relative_action_values"][0])
    nominal_epochs = max_steps * effective_batch / valid_windows
    if not math.isclose(nominal_epochs, args.expected_epochs, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError(
            f"Training budget is not {args.expected_epochs} epochs: "
            f"{max_steps} * {effective_batch} / {valid_windows} = {nominal_epochs}"
        )

    trainer_state = read_json(checkpoint_dir / "trainer_state.json")
    if int(trainer_state["global_step"]) != max_steps:
        raise RuntimeError(
            f"Final checkpoint step differs from launch budget: {trainer_state['global_step']}"
        )
    log_history = trainer_state.get("log_history", [])
    if not log_history or int(log_history[-1].get("step", -1)) != max_steps:
        raise RuntimeError("Trainer history does not end at the final optimizer step")
    if not math.isfinite(float(log_history[-1]["loss"])):
        raise RuntimeError("Final logged loss is non-finite")

    index = read_json(checkpoint_dir / "model.safetensors.index.json")
    shards = sorted(set(index["weight_map"].values()))
    if len(shards) != 3:
        raise RuntimeError(f"Expected three model shards, got {shards}")
    artifact_sizes: dict[str, int] = {}
    for name in (
        *shards,
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "processor_config.json",
        "statistics.json",
        "training_args.bin",
    ):
        path = checkpoint_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Final checkpoint artifact is missing or empty: {path}")
        artifact_sizes[name] = path.stat().st_size

    model_config = read_json(checkpoint_dir / "experiment_cfg" / "final_model_config.json")
    tuning = {
        "llm": bool(model_config["tune_llm"]),
        "visual": bool(model_config["tune_visual"]),
        "projector": bool(model_config["tune_projector"]),
        "diffusion_model": bool(model_config["tune_diffusion_model"]),
        "vlln": bool(model_config["tune_vlln"]),
    }
    if tuning != {
        "llm": False,
        "visual": False,
        "projector": True,
        "diffusion_model": True,
        "vlln": True,
    }:
        raise RuntimeError(f"Unexpected trainable-module configuration: {tuning}")

    terminal_log = (run_dir / "logs" / "train_terminal.log").read_text(encoding="utf-8", errors="replace")
    fatal_markers = ("CUDA out of memory", "Traceback (most recent call last)", "nan loss", "inf loss")
    found_fatal_markers = [marker for marker in fatal_markers if marker in terminal_log]
    if found_fatal_markers:
        raise RuntimeError(f"Fatal marker found in training log: {found_fatal_markers}")
    metrics = final_train_metrics(terminal_log)

    load_evidence = read_json(model_load_report_path)
    loaded_checkpoint = Path(load_evidence["checkpoint"]["path"]).resolve(strict=True)
    if load_evidence.get("status") != "passed" or loaded_checkpoint != checkpoint_dir.resolve(strict=True):
        raise RuntimeError("Live model-load evidence does not match the final checkpoint")
    if int(load_evidence["checkpoint"]["global_step"]) != max_steps:
        raise RuntimeError("Loaded checkpoint evidence has the wrong global step")
    server_log = model_server_log_path.read_text(encoding="utf-8", errors="replace")
    if "Server ready" not in server_log or "Loading checkpoint shards: 100%" not in server_log:
        raise RuntimeError("Policy server log does not prove complete checkpoint loading")

    result = {
        "schema_version": 1,
        "status": "passed",
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_dir),
        "global_step": max_steps,
        "micro_batch_per_device": micro_batch,
        "gradient_accumulation_steps": gradient_accumulation,
        "effective_batch": effective_batch,
        "valid_windows": valid_windows,
        "nominal_sample_epochs": nominal_epochs,
        "tuning": tuning,
        "final_log_entry": log_history[-1],
        "train_metrics": metrics,
        "model_shards": shards,
        "model_bytes": sum(artifact_sizes[name] for name in shards),
        "checkpoint_artifact_sizes": artifact_sizes,
        "live_checkpoint_load": {
            "status": "passed",
            "evidence_report": str(model_load_report_path),
            "server_log": str(model_server_log_path),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
