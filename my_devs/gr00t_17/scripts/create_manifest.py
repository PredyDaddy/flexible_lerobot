#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import stat
from pathlib import Path
from typing import Any

CHUNK_SIZE = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def iter_entries(root: Path) -> list[dict[str, Any]]:
    if root.is_file():
        paths = [root]
        base = root.parent
    elif root.is_dir():
        paths = sorted(path for path in root.rglob("*") if path.is_file() or path.is_symlink())
        base = root
    else:
        raise FileNotFoundError(root)

    entries: list[dict[str, Any]] = []
    for path in paths:
        relative_path = path.relative_to(base).as_posix()
        path_stat = path.lstat()
        if path.is_symlink():
            entries.append(
                {
                    "path": relative_path,
                    "type": "symlink",
                    "target": str(path.readlink()),
                    "mode": stat.S_IMODE(path_stat.st_mode),
                }
            )
            continue
        entries.append(
            {
                "path": relative_path,
                "type": "file",
                "size": path_stat.st_size,
                "mode": stat.S_IMODE(path_stat.st_mode),
                "sha256": sha256_file(path),
            }
        )
    return entries


def aggregate_digest(entries: list[dict[str, Any]]) -> str:
    payload = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_manifest(root: Path, label: str) -> dict[str, Any]:
    resolved = root.expanduser().resolve(strict=True)
    entries = iter_entries(resolved)
    return {
        "schema_version": 1,
        "label": label,
        "root": str(resolved),
        "entry_count": len(entries),
        "aggregate_sha256": aggregate_digest(entries),
        "entries": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or verify an immutable input manifest.")
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()

    if (args.output is None) == (args.verify is None):
        parser.error("Specify exactly one of --output or --verify")

    current = build_manifest(args.path, args.label)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite manifest: {output}")
        output.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] wrote manifest: {output}")
        print(f"[OK] entries={current['entry_count']} aggregate_sha256={current['aggregate_sha256']}")
        return

    expected = json.loads(args.verify.read_text(encoding="utf-8"))
    comparable_keys = ["label", "root", "entry_count", "aggregate_sha256", "entries"]
    mismatches = [key for key in comparable_keys if current.get(key) != expected.get(key)]
    if mismatches:
        raise RuntimeError(f"Manifest verification failed for keys: {mismatches}")
    print(f"[OK] manifest unchanged: {args.verify}")
    print(f"[OK] entries={current['entry_count']} aggregate_sha256={current['aggregate_sha256']}")


if __name__ == "__main__":
    main()
