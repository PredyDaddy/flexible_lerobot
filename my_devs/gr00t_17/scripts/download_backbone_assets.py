#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

RUNTIME_FILES = (
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)
FORBIDDEN_SUFFIXES = (".bin", ".ckpt", ".pt", ".pth", ".safetensors")


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


def sibling_map(api: HfApi, repo_id: str, revision: str):
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError(f"Resolved {repo_id} revision {info.sha} != requested {revision}")
    return info, {item.rfilename: item for item in info.siblings}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download public Qwen3-VL runtime assets proven identical to Cosmos-Reason2 assets."
    )
    parser.add_argument("--source-repo-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--equivalent-repo-id", default="nvidia/Cosmos-Reason2-2B")
    parser.add_argument("--equivalent-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allowed-write-root", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    args = parser.parse_args()

    allowed_root = args.allowed_write_root.expanduser().resolve(strict=True)
    output_dir = ensure_within(args.output_dir, allowed_root)
    report_path = ensure_within(args.report_path, allowed_root)
    cache_dir = ensure_within(Path(os.environ["HF_HUB_CACHE"]), allowed_root)
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite asset directory: {output_dir}")
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite report: {report_path}")

    partial = ensure_within(output_dir.parent / f".{output_dir.name}.partial", allowed_root)
    if partial.exists():
        raise FileExistsError(f"Refusing to reuse partial directory: {partial}")
    partial.mkdir(parents=True)

    api = HfApi(endpoint=endpoint)
    source_info, source_files = sibling_map(api, args.source_repo_id, args.source_revision)
    equivalent_info, equivalent_files = sibling_map(api, args.equivalent_repo_id, args.equivalent_revision)

    comparisons = []
    for relative in RUNTIME_FILES:
        if relative not in source_files:
            raise RuntimeError(f"Source repository is missing runtime asset: {relative}")
        if relative not in equivalent_files:
            raise RuntimeError(f"Equivalent repository is missing runtime asset: {relative}")
        source = source_files[relative]
        equivalent = equivalent_files[relative]
        same_blob = source.blob_id == equivalent.blob_id and source.size == equivalent.size
        comparisons.append(
            {
                "path": relative,
                "source_blob_id": source.blob_id,
                "equivalent_blob_id": equivalent.blob_id,
                "source_size": source.size,
                "equivalent_size": equivalent.size,
                "same_blob_and_size": same_blob,
            }
        )
    unequal = [item["path"] for item in comparisons if not item["same_blob_and_size"]]
    if unequal:
        raise RuntimeError(f"Runtime assets are not byte-identical: {unequal}")

    snapshot_download(
        repo_id=args.source_repo_id,
        revision=source_info.sha,
        allow_patterns=list(RUNTIME_FILES),
        local_dir=partial,
        cache_dir=cache_dir,
        endpoint=endpoint,
        max_workers=4,
    )

    local_files = sorted(item for item in partial.rglob("*") if item.is_file())
    missing = [relative for relative in RUNTIME_FILES if not (partial / relative).is_file()]
    if missing:
        raise RuntimeError(f"Downloaded asset directory is incomplete: {missing}")
    forbidden = [
        str(path.relative_to(partial)) for path in local_files if path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]
    if forbidden:
        raise RuntimeError(f"Unexpected model weight files were downloaded: {forbidden}")

    entries = []
    for path in local_files:
        relative = path.relative_to(partial).as_posix()
        entries.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    partial.rename(output_dir)
    report = {
        "schema_version": 1,
        "source_repo_id": args.source_repo_id,
        "source_revision": source_info.sha,
        "equivalent_repo_id": args.equivalent_repo_id,
        "equivalent_revision": equivalent_info.sha,
        "endpoint": endpoint,
        "output_dir": str(output_dir),
        "method": (
            "Every downloaded runtime asset has the same Hugging Face git blob ID and size "
            "in the pinned public Qwen3-VL and pinned Cosmos-Reason2 repositories. Model "
            "weights are intentionally excluded; the GR00T N1.7 checkpoint supplies them."
        ),
        "runtime_asset_comparisons": comparisons,
        "downloaded_files": entries,
        "status": "passed",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] runtime assets published: {output_dir}")
    print(f"[OK] byte-identical runtime files: {len(comparisons)}")
    print(f"[OK] report: {report_path}")


if __name__ == "__main__":
    main()
