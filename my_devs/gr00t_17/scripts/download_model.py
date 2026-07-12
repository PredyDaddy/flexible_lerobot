#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REQUIRED_FILES = [
    "config.json",
    "embodiment_id.json",
    "experiment_cfg/conf.yaml",
    "experiment_cfg/config.yaml",
    "experiment_cfg/dataset_statistics.json",
    "experiment_cfg/final_model_config.json",
    "experiment_cfg/final_processor_config.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "processor_config.json",
    "statistics.json",
]


def ensure_within(path: Path, root: Path) -> Path:
    resolved = path.expanduser().resolve()
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a pinned GR00T model snapshot locally.")
    parser.add_argument("--repo-id", default="nvidia/GR00T-N1.7-3B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allowed-write-root", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--required-file", action="append", default=[])
    args = parser.parse_args()

    allowed_root = args.allowed_write_root.expanduser().resolve(strict=True)
    output_dir = ensure_within(args.output_dir, allowed_root)
    report_path = ensure_within(args.report_path, allowed_root)
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    cache_dir = Path(os.environ["HF_HUB_CACHE"]).expanduser().resolve()
    ensure_within(cache_dir, allowed_root)

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite model directory: {output_dir}")
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite report: {report_path}")

    partial = output_dir.parent / f".{output_dir.name}.partial"
    ensure_within(partial, allowed_root)
    partial.mkdir(parents=True, exist_ok=True)

    api = HfApi(endpoint=endpoint)
    info = api.model_info(args.repo_id, revision=args.revision)
    if info.sha != args.revision:
        raise RuntimeError(f"Resolved revision {info.sha} != requested {args.revision}")
    print(f"[MODEL] repo={args.repo_id} revision={info.sha} endpoint={endpoint}", flush=True)

    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=partial,
        cache_dir=cache_dir,
        endpoint=endpoint,
        max_workers=4,
    )

    required_files = args.required_file or REQUIRED_FILES
    missing = [relative for relative in required_files if not (partial / relative).is_file()]
    if missing:
        raise RuntimeError(f"Downloaded model is missing required files: {missing}")

    entries = []
    total_bytes = 0
    for path in sorted(item for item in partial.rglob("*") if item.is_file()):
        relative = path.relative_to(partial).as_posix()
        size = path.stat().st_size
        total_bytes += size
        entries.append({"path": relative, "size": size, "sha256": sha256_file(path)})

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial.rename(output_dir)
    report = {
        "schema_version": 1,
        "repo_id": args.repo_id,
        "revision": info.sha,
        "endpoint": endpoint,
        "output_dir": str(output_dir),
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "required_files": required_files,
        "files": entries,
        "status": "passed",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] published model: {output_dir}")
    print(f"[OK] files={len(entries)} total_bytes={total_bytes}")
    print(f"[OK] report={report_path}")


if __name__ == "__main__":
    main()
