#!/usr/bin/env python3
"""CLI entry for single-arm dataset pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import sys

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from my_devs.remove_feature_pipeline.core import PipelineError, build_single_arm_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build left/right single-arm dataset from a dual-arm LeRobot dataset."
    )
    parser.add_argument("--source-root", type=Path, required=True, help="Parent root of source dataset repo.")
    parser.add_argument("--source-repo-id", type=str, required=True, help="Source dataset folder name/repo id.")
    parser.add_argument("--target-repo-id", type=str, required=True, help="Target dataset folder name/repo id.")
    parser.add_argument("--arm", type=str, choices=["left", "right"], required=True, help="Arm side to keep.")
    parser.add_argument(
        "--work-root",
        type=Path,
        default=None,
        help="Parent root of target dataset repo. Defaults to --source-root.",
    )
    parser.add_argument("--std-floor", type=float, default=1e-3, help="Std floor for recomputed vector stats.")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only, do not write files.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow deleting existing target directory before generation.",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG/INFO/WARNING/ERROR).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(asctime)s %(message)s",
    )
    try:
        result = build_single_arm_dataset(
            source_root=args.source_root,
            source_repo_id=args.source_repo_id,
            target_repo_id=args.target_repo_id,
            arm=args.arm,
            work_root=args.work_root,
            std_floor=args.std_floor,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except PipelineError as exc:
        logging.error("Pipeline failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
