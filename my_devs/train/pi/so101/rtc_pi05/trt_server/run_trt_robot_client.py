#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
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

from my_devs.train.pi.so101.rtc_pi05.server import run_robot_client as base_client  # noqa: E402


DEFAULT_TRT_SERVER_URL = "http://127.0.0.1:8090"
LOG_PREFIX = "[RTC-PI05-TRT-CLIENT]"


def build_parser() -> argparse.ArgumentParser:
    parser = base_client.build_parser()
    parser.description = "SO101 robot client for the PI0.5 RTC pure TensorRT policy server."
    parser.add_argument(
        "--confirm-control",
        action="store_true",
        help="Required for real robot action sending. Not required for --dry-run or --connect-smoke.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.dry_run and not args.connect_smoke and not args.confirm_control:
        print(f"{LOG_PREFIX} Missing --confirm-control. Exit before robot connection.")
        return 0

    base_argv = remove_flag(argv, "--confirm-control")
    if "--server-url" not in base_argv and os.getenv("PI05_SERVER_URL") is None:
        base_argv = ["--server-url", DEFAULT_TRT_SERVER_URL, *base_argv]
    return base_client.main(base_argv)


def remove_flag(argv: list[str], flag: str) -> list[str]:
    return [item for item in argv if item != flag]


if __name__ == "__main__":
    raise SystemExit(main())
