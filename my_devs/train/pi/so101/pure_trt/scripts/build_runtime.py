#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


def resolve_repo_root(script_path: Path) -> Path:
    resolved = script_path.resolve()
    for candidate in resolved.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {script_path}")


BOOTSTRAP_REPO_ROOT = resolve_repo_root(Path(__file__))
if BOOTSTRAP_REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, BOOTSTRAP_REPO_ROOT.as_posix())

from my_devs.train.pi.so101.pure_trt.runtime.paths import LEGACY_OPENPI_TRT_ROOT, REPO_ROOT, default_paths


for _path in (REPO_ROOT, LEGACY_OPENPI_TRT_ROOT):
    if _path.as_posix() not in sys.path:
        sys.path.insert(0, _path.as_posix())

from scripts import simple_pi05_pipeline as legacy_pipeline  # noqa: E402
from scripts.pi05_onnx_common import DEFAULT_POLICY_PATH, DEFAULT_TASK  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    paths = default_paths()
    parser = argparse.ArgumentParser(
        description="Clean SO101 PI0.5 pure TensorRT export -> build -> validate pipeline."
    )
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--artifact-dir", type=Path, default=paths.artifact_dir)
    parser.add_argument("--profile", choices=["fp32", "fp16_constrained"], default="fp16_constrained")
    parser.add_argument("--runtime-backend", choices=["pure_trt", "hybrid"], default="pure_trt")
    parser.add_argument("--prefix-precision", choices=["fp32", "fp16_constrained"], default="fp16_constrained")
    parser.add_argument("--build-prefix-engine", action="store_true")
    parser.add_argument("--export-prefix-onnx", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--noise-seed", type=int, default=2026)
    parser.add_argument("--opset", type=int, default=19)
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--report", type=Path, default=paths.artifact_dir / "pure_trt_pipeline_report.json")
    parser.add_argument(
        "--prefix-engine-path",
        type=Path,
        default=None,
        help="Override prefix engine for pure_trt validation. Defaults to the selected staged/built prefix engine.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    legacy_pipeline.configure_runtime()
    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the PI0.5 TensorRT pipeline.")
    configure_legacy_artifact_paths(args.artifact_dir.expanduser())
    legacy_args = to_legacy_args(args)

    policy_path = legacy_args.policy_path.expanduser().resolve()
    legacy_pipeline.log("[PURE-TRT] Clean SO101 PI0.5 TensorRT pipeline")
    legacy_pipeline.log(f"[PURE-TRT] policy_path={policy_path}")
    legacy_pipeline.log(f"[PURE-TRT] artifact_dir={args.artifact_dir}")
    legacy_pipeline.log(f"[PURE-TRT] profile={legacy_args.profile} backend={legacy_args.runtime_backend}")

    t0 = time.perf_counter()
    policy = legacy_pipeline.load_policy(policy_path, device="cuda", model_dtype="float32")
    batch = legacy_pipeline.make_policy_batch(policy_path, task=legacy_args.task, seed=legacy_args.seed)
    legacy_pipeline.log(f"[PURE-TRT] Policy/batch ready in {time.perf_counter() - t0:.2f}s")

    report = {
        "policy_path": str(policy_path),
        "artifact_dir": str(args.artifact_dir),
        "profile": legacy_args.profile,
        "runtime_backend": legacy_args.runtime_backend,
    }
    if not legacy_args.convert_only and not legacy_args.validate_only:
        report["onnx"] = {k: str(v) for k, v in legacy_pipeline.export_onnx(policy, batch, legacy_args).items()}
    if not legacy_args.export_only and not legacy_args.validate_only:
        report["engines"] = {k: str(v) for k, v in legacy_pipeline.convert_engines(legacy_args).items()}
    if not legacy_args.export_only and not legacy_args.convert_only:
        report["inference"] = legacy_pipeline.validate_inference(policy, batch, legacy_args)

    legacy_pipeline.write_json(legacy_args.report.expanduser(), report)
    legacy_pipeline.log(f"[PURE-TRT] Report written: {legacy_args.report}")
    if report.get("inference", {}).get("passed_allclose") is False:
        return 2
    return 0


def configure_legacy_artifact_paths(artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    legacy_pipeline.ARTIFACT_DIR = artifact_dir
    legacy_pipeline.PREFIX_ONNX = artifact_dir / "pi05_so101_prefix_cache_b1_fp32.onnx"
    legacy_pipeline.PREFIX_FP32_ENGINE = artifact_dir / "pi05_so101_prefix_cache_b1_fp32.engine"
    legacy_pipeline.PREFIX_FP16_CONSTRAINED_ENGINE = artifact_dir / "pi05_so101_prefix_cache_b1_fp16_constrained.engine"
    legacy_pipeline.DENOISE_ONNX = artifact_dir / "pi05_so101_denoise_step_b1_fp32.onnx"
    legacy_pipeline.DENOISE_FP32_ENGINE = artifact_dir / "pi05_so101_denoise_step_b1_fp32.engine"
    legacy_pipeline.DENOISE_FP16_CONSTRAINED_ENGINE = artifact_dir / "pi05_so101_denoise_step_b1_fp16_constrained.engine"


def to_legacy_args(args: argparse.Namespace) -> argparse.Namespace:
    paths = default_paths()
    prefix_engine_path = args.prefix_engine_path
    if prefix_engine_path is None:
        prefix_engine_path = paths.prefix_engine(args.prefix_precision)
    return argparse.Namespace(
        policy_path=args.policy_path,
        task=args.task,
        profile=args.profile,
        runtime_backend=args.runtime_backend,
        prefix_engine_path=prefix_engine_path,
        build_prefix_engine=args.build_prefix_engine,
        export_prefix_onnx=args.export_prefix_onnx,
        prefix_precision=args.prefix_precision,
        seed=args.seed,
        noise_seed=args.noise_seed,
        opset=args.opset,
        workspace_gb=args.workspace_gb,
        force_export=args.force_export,
        force_convert=args.force_convert,
        export_only=args.export_only,
        convert_only=args.convert_only,
        validate_only=args.validate_only,
        report=args.report,
    )


if __name__ == "__main__":
    raise SystemExit(main())
