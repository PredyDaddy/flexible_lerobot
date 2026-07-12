#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from create_manifest import build_manifest

REQUIRED_REPORTS = (
    "data_conversion.json",
    "dataset_validation.json",
    "model_download.json",
    "backbone_assets.json",
    "model_validation.json",
    "training_batch.json",
    "environment_validation.json",
)
SOURCE_MANIFESTS = (
    "original_dataset.json",
    "pi05_checkpoint.json",
    "gr00t_reference.json",
)
DERIVED_DATASET_MANIFEST = "converted_dataset.json"


def ensure_within(path: Path, root: Path, *, must_exist: bool = False) -> Path:
    resolved = path.expanduser().resolve(strict=must_exist)
    allowed = root.expanduser().resolve(strict=True)
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"Path escapes allowed root: {resolved}") from exc
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_reported_files(report: dict, root_key: str, entries_key: str) -> int:
    root = Path(report[root_key]).resolve(strict=True)
    checked = 0
    for item in report[entries_key]:
        path = root / item["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Reported file is missing: {path}")
        if path.stat().st_size != item["size"]:
            raise RuntimeError(f"Reported size changed: {path}")
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Reported checksum changed: {path}")
        checked += 1
    return checked


def verify_no_external_symlinks(path: Path, root: Path) -> int:
    checked = 0
    for item in path.rglob("*"):
        if not item.is_symlink():
            continue
        ensure_within(item.resolve(strict=True), root, must_exist=True)
        checked += 1
    return checked


def query_gpu() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    rows = subprocess.check_output(command, text=True).strip().splitlines()
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one GPU, found {len(rows)}")
    fields = [field.strip() for field in rows[0].split(",")]
    return {
        "index": int(fields[0]),
        "name": fields[1],
        "total_mib": int(fields[2]),
        "used_mib": int(fields[3]),
        "free_mib": int(fields[4]),
    }


def find_resume_checkpoint(output_dir: Path) -> Path | None:
    candidates = []
    if output_dir.is_dir():
        for path in output_dir.glob("checkpoint-*"):
            suffix = path.name.removeprefix("checkpoint-")
            if path.is_dir() and suffix.isdigit() and (path / "trainer_state.json").is_file():
                candidates.append((int(suffix), path))
    return max(candidates)[1] if candidates else None


def verify_smoke_evidence(report: dict, root: Path) -> dict:
    if report.get("checkpoint_steps") != [1, 2]:
        raise RuntimeError("Smoke report does not contain checkpoint steps [1, 2]")
    if report.get("resume_from_checkpoint_one") is not True:
        raise RuntimeError("Smoke report does not prove resume from checkpoint-1")

    trainable_delta = float(report.get("trainable_step_one_to_step_two_max_abs_diff", 0))
    frozen_delta = float(report.get("frozen_base_to_step_two_max_abs_diff", math.nan))
    if not math.isfinite(trainable_delta) or trainable_delta <= 0:
        raise RuntimeError(f"Smoke report has no finite trainable weight update: {trainable_delta}")
    if not math.isfinite(frozen_delta) or frozen_delta != 0:
        raise RuntimeError(f"Smoke report shows a changed frozen weight: {frozen_delta}")

    loss_metrics = report.get("loss_metrics", [])
    if not loss_metrics or not all(math.isfinite(float(item["value"])) for item in loss_metrics):
        raise RuntimeError("Smoke report does not contain finite loss metrics")

    train_dir = ensure_within(Path(report["train_dir"]), root, must_exist=True)
    for step in report["checkpoint_steps"]:
        checkpoint = train_dir / f"checkpoint-{step}"
        state_path = checkpoint / "trainer_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("global_step") != step:
            raise RuntimeError(f"Smoke checkpoint has an unexpected global step: {state_path}")
        for filename in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
            path = checkpoint / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Smoke checkpoint state is missing or empty: {path}")

    return {
        "train_dir": str(train_dir),
        "checkpoint_steps": report["checkpoint_steps"],
        "trainable_step_one_to_step_two_max_abs_diff": trainable_delta,
        "frozen_base_to_step_two_max_abs_diff": frozen_delta,
        "finite_loss_metrics": len(loss_metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed preflight for local N1.7 training.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-free-gpu-mib", type=int, default=40_000)
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--episode-sampling-rate", type=float, default=0.1)
    parser.add_argument("--allow-resume", action="store_true")
    parser.add_argument("--require-smoke", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser().resolve(strict=True)
    output_dir = ensure_within(args.output_dir, root)
    report_path = ensure_within(args.report, root)
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite preflight report: {report_path}")
    if os.environ.get("CONDA_DEFAULT_ENV") != "lerobot_flex":
        raise RuntimeError("CONDA_DEFAULT_ENV must be lerobot_flex")
    executable_path = Path(os.path.abspath(sys.executable))
    prefix_path = Path(sys.prefix).resolve(strict=True)
    if not executable_path.is_relative_to(root) or not prefix_path.is_relative_to(root):
        raise RuntimeError(
            f"Python virtual environment is outside the guarded root: {sys.executable}, {sys.prefix}"
        )
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("Hugging Face and Transformers offline mode must be enabled")

    reports = {}
    required_reports = list(REQUIRED_REPORTS)
    if args.require_smoke:
        required_reports.append("smoke_train.json")
    for filename in required_reports:
        path = root / "reports" / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "passed":
            raise RuntimeError(f"Required report did not pass: {path}")
        reports[filename] = payload

    smoke_evidence = None
    if args.require_smoke:
        smoke_evidence = verify_smoke_evidence(reports["smoke_train.json"], root)

    manifest_results = []
    for filename in SOURCE_MANIFESTS:
        path = root / "data" / "source_manifests" / filename
        expected = json.loads(path.read_text(encoding="utf-8"))
        current = build_manifest(Path(expected["root"]), expected["label"])
        if current != expected:
            raise RuntimeError(f"Immutable source manifest changed: {path}")
        manifest_results.append(
            {
                "manifest": str(path),
                "entries": current["entry_count"],
                "aggregate_sha256": current["aggregate_sha256"],
            }
        )

    derived_manifest_path = root / "data" / "derived_manifests" / DERIVED_DATASET_MANIFEST
    expected_derived = json.loads(derived_manifest_path.read_text(encoding="utf-8"))
    derived_root = ensure_within(Path(expected_derived["root"]), root, must_exist=True)
    current_derived = build_manifest(derived_root, expected_derived["label"])
    if current_derived != expected_derived:
        raise RuntimeError(f"Converted dataset manifest changed: {derived_manifest_path}")
    derived_manifest_result = {
        "manifest": str(derived_manifest_path),
        "entries": current_derived["entry_count"],
        "aggregate_sha256": current_derived["aggregate_sha256"],
    }

    model_files_checked = verify_reported_files(reports["model_download.json"], "output_dir", "files")
    backbone_files_checked = verify_reported_files(
        reports["backbone_assets.json"], "output_dir", "downloaded_files"
    )
    external_symlinks_checked = 0
    for relative in ("data/converted_v21", "models", "workspace"):
        external_symlinks_checked += verify_no_external_symlinks(root / relative, root)

    episodes = reports["dataset_validation.json"]["episodes"]
    effective_steps = reports["training_batch.json"]["effective_training_steps"]
    if not 0 < args.episode_sampling_rate <= 1:
        raise ValueError("episode_sampling_rate must be in (0, 1]")
    num_splits = int(1 / args.episode_sampling_rate)
    max_nonempty_shards = episodes * num_splits
    requested_shards = math.ceil(effective_steps / args.shard_size)
    if requested_shards > max_nonempty_shards:
        raise ValueError(
            "shard_size is too small for the upstream sharding algorithm: "
            f"requested_shards={requested_shards}, available_subsequences={max_nonempty_shards}"
        )

    resume_checkpoint = find_resume_checkpoint(output_dir)
    if args.allow_resume:
        if resume_checkpoint is None:
            raise RuntimeError(f"Resume requested but no valid checkpoint exists: {output_dir}")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing to start a fresh run in a non-empty directory: {output_dir}")

    import flash_attn
    import torch
    import torchcodec

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not flash_attn.__version__.startswith("2.7.4"):
        raise RuntimeError(f"Unexpected flash-attn version: {flash_attn.__version__}")
    gpu = query_gpu()
    if gpu["free_mib"] < args.min_free_gpu_mib:
        raise RuntimeError(
            f"Insufficient free GPU memory: {gpu['free_mib']} MiB < {args.min_free_gpu_mib} MiB"
        )

    result = {
        "schema_version": 1,
        "status": "passed",
        "root": str(root),
        "output_dir": str(output_dir),
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "gpu": gpu,
        "minimum_free_gpu_mib": args.min_free_gpu_mib,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "flash_attn": flash_attn.__version__,
        "torchcodec": torchcodec.__version__,
        "sharding": {
            "shard_size": args.shard_size,
            "episode_sampling_rate": args.episode_sampling_rate,
            "requested_shards": requested_shards,
            "available_subsequences": max_nonempty_shards,
        },
        "immutable_source_manifests": manifest_results,
        "derived_dataset_manifest": derived_manifest_result,
        "smoke_evidence": smoke_evidence,
        "model_files_checked": model_files_checked,
        "backbone_files_checked": backbone_files_checked,
        "internal_symlinks_checked": external_symlinks_checked,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
