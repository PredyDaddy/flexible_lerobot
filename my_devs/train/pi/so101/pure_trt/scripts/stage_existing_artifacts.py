#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def resolve_repo_root(script_path: Path) -> Path:
    resolved = script_path.resolve()
    for candidate in resolved.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {script_path}")


REPO_ROOT = resolve_repo_root(Path(__file__))
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from my_devs.train.pi.so101.pure_trt.runtime.paths import (
    LEGACY_ARTIFACT_DIR,
    PI05PureTRTPaths,
    default_paths,
    legacy_artifact,
)


RUNTIME_ASSET_FILES = (
    "config.json",
    "policy_preprocessor.json",
    "policy_preprocessor_step_2_normalizer_processor.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)

LARGE_ARTIFACTS = {
    "pi05_so101_prefix_cache_b1_fp32.onnx": "prefix_onnx",
    "pi05_so101_denoise_step_b1_fp32.onnx": "denoise_onnx",
    "pi05_so101_prefix_cache_b1_fp32.engine": "prefix_fp32_engine",
    "pi05_so101_prefix_cache_b1_fp16_constrained.engine": "prefix_fp16_constrained_engine",
    "pi05_so101_denoise_step_b1_fp32.engine": "denoise_fp32_engine",
    "pi05_so101_denoise_step_b1_fp16_constrained.engine": "denoise_fp16_constrained_engine",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage validated openpi_trt artifacts into the clean SO101 pure_trt workspace."
    )
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--force", action="store_true", help="Replace existing staged files or symlinks.")
    parser.add_argument("--manifest", type=Path, default=default_paths().artifact_dir / "staged_artifacts_manifest.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = default_paths()
    paths.artifact_dir.mkdir(parents=True, exist_ok=True)
    staged = []

    for source_name, attr_name in LARGE_ARTIFACTS.items():
        source = legacy_artifact(source_name)
        target = getattr(paths, attr_name)
        stage_file(source, target, mode=args.mode, force=args.force)
        staged.append(describe(source, target, mode=args.mode, role=attr_name))

    source_assets_dir = LEGACY_ARTIFACT_DIR / "pi05_runtime_assets"
    stage_runtime_assets(source_assets_dir, paths.runtime_assets_dir, force=args.force)
    for file_name in RUNTIME_ASSET_FILES:
        staged.append(
            describe(
                source_assets_dir / file_name,
                paths.runtime_assets_dir / file_name,
                mode="copy",
                role=f"runtime_assets/{file_name}",
            )
        )

    manifest = {
        "artifact_dir": str(paths.artifact_dir),
        "legacy_artifact_dir": str(LEGACY_ARTIFACT_DIR),
        "mode": args.mode,
        "artifacts": staged,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"[PURE-TRT] Staged {len(staged)} artifacts. Manifest: {args.manifest}")
    return 0


def stage_file(source: Path, target: Path, *, mode: str, force: bool) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Missing source artifact: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not force:
            return
        if target.is_dir() and not target.is_symlink():
            raise RuntimeError(f"Refusing to replace directory target: {target}")
        target.unlink()
    if mode == "copy":
        shutil.copy2(source, target)
        return
    if mode != "symlink":
        raise ValueError(f"Unsupported staging mode: {mode}")
    target.symlink_to(relative_target(source, target.parent))


def stage_runtime_assets(source_dir: Path, target_dir: Path, *, force: bool) -> None:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing runtime assets directory: {source_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    for file_name in RUNTIME_ASSET_FILES:
        source = source_dir / file_name
        target = target_dir / file_name
        if not source.is_file():
            raise FileNotFoundError(f"Missing runtime asset: {source}")
        if target.exists() or target.is_symlink():
            if not force:
                continue
            target.unlink()
        shutil.copy2(source, target)


def relative_target(source: Path, from_dir: Path) -> Path:
    try:
        return source.resolve().relative_to(from_dir.resolve())
    except ValueError:
        return Path(os.path.relpath(source.resolve(), from_dir.resolve()))


def describe(source: Path, target: Path, *, mode: str, role: str) -> dict[str, object]:
    return {
        "role": role,
        "mode": mode,
        "source": str(source),
        "target": str(target),
        "bytes": source.stat().st_size,
    }


if __name__ == "__main__":
    raise SystemExit(main())
