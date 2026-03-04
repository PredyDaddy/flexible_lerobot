#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from modelscope.hub.snapshot_download import snapshot_download


def ensure_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        try:
            same = link.resolve() == target.resolve()
        except FileNotFoundError:
            same = False
        if not same:
            if link.is_symlink() or link.is_file():
                link.unlink()
            else:
                shutil.rmtree(link)
    if not link.exists():
        link.symlink_to(target.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a model from ModelScope and create stable symlink")
    parser.add_argument("--model-id", required=True, help="e.g. lerobot/pi05_base")
    parser.add_argument(
        "--cache-dir",
        default="/data/cqy_workspace/flexible_lerobot/assets/modelscope",
        help="modelscope cache root",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"START model_id={args.model_id}", flush=True)
    local_dir = Path(snapshot_download(args.model_id, cache_dir=str(cache_dir)))
    print(f"REAL_DIR={local_dir}", flush=True)

    link = cache_dir / args.model_id
    ensure_symlink(link, local_dir)
    print(f"LINK={link} -> {link.resolve()}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
