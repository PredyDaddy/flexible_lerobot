#!/usr/bin/env python

"""Environment and startup smoke checks for the GR00T TRT async server stack."""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Failed to locate repo root from: {start}")


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str

    @property
    def label(self) -> str:
        return "PASS" if self.ok else "FAIL"


def _check_conda_env() -> CheckResult:
    expected = "lerobot_flex"
    env_name = os.environ.get("CONDA_DEFAULT_ENV")
    executable = Path(sys.executable).as_posix()
    ok = env_name == expected or f"/envs/{expected}/" in executable
    detail = f"CONDA_DEFAULT_ENV={env_name!r}, sys.executable={executable}"
    return CheckResult("conda-env", ok, detail)


def _check_import(module_name: str) -> CheckResult:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return CheckResult(f"import:{module_name}", False, f"{type(exc).__name__}: {exc}")
    return CheckResult(f"import:{module_name}", True, f"loaded from {getattr(module, '__file__', '<builtin>')}")


def _check_robot_client_async_stack() -> CheckResult:
    try:
        module = importlib.import_module("my_devs.groot_trt_async_server.robot_client")
    except Exception as exc:
        return CheckResult("robot-client-async-stack", False, f"{type(exc).__name__}: {exc}")

    import_error = getattr(module, "_ROBOT_CLIENT_IMPORT_ERROR", None)
    if import_error is not None:
        return CheckResult(
            "robot-client-async-stack",
            False,
            f"{type(import_error).__name__}: {import_error}",
        )

    return CheckResult(
        "robot-client-async-stack",
        True,
        "robot_client optional async imports resolved successfully",
    )


def _run_server_dry_run() -> CheckResult:
    cmd = [
        sys.executable,
        "my_devs/groot_trt_async_server/run_server.py",
        "--dry-run",
    ]
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        detail = completed.stdout.strip() or "dry-run completed without output"
        return CheckResult("server-dry-run", True, detail)

    output = (completed.stderr or completed.stdout).strip()
    detail = output or f"dry-run exited with code {completed.returncode}"
    return CheckResult("server-dry-run", False, detail)


def _build_checks(mode: str, include_server_dry_run: bool) -> list[CheckResult]:
    checks: list[CheckResult] = [
        _check_conda_env(),
        _check_import("torch"),
        _check_import("my_devs.groot_trt_async_server.configs"),
    ]

    if mode in {"all", "server"}:
        checks.append(_check_import("grpc"))
        checks.append(_check_import("my_devs.groot_trt_async_server.policy_server"))
        if include_server_dry_run:
            checks.append(_run_server_dry_run())

    if mode in {"all", "client"}:
        checks.append(_check_robot_client_async_stack())
        checks.append(_check_import("my_devs.groot_trt_async_server.run_client"))

    return checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check GR00T TRT async server environment readiness.")
    parser.add_argument(
        "--mode",
        choices=["all", "server", "client"],
        default="all",
        help="Select which part of the stack to validate.",
    )
    parser.add_argument(
        "--skip-server-dry-run",
        action="store_true",
        help="Skip invoking run_server.py --dry-run.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    checks = _build_checks(args.mode, include_server_dry_run=not args.skip_server_dry_run)
    failed = [check for check in checks if not check.ok]

    print("GR00T TRT async environment check")
    print(f"repo_root={REPO_ROOT}")
    for check in checks:
        print(f"[{check.label}] {check.name}: {check.detail}")

    if failed:
        print(f"SUMMARY: {len(failed)} check(s) failed.")
        return 1

    print("SUMMARY: all checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
