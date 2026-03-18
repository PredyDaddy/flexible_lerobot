#!/usr/bin/env python

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    this_file = Path(__file__).resolve()
    for candidate in [this_file.parent, *this_file.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            repo_root = candidate
            if repo_root.as_posix() not in sys.path:
                sys.path.insert(0, repo_root.as_posix())
            return
    raise RuntimeError(f"Failed to locate repo root from {this_file}")


def main() -> int:
    _ensure_repo_root_on_path()

    from my_devs.groot_trt_async_server.groot_trt_policy import (
        backend_integration_validation_gaps,
        run_backend_self_checks,
    )

    results = run_backend_self_checks()
    failed = [result for result in results if not result.ok]
    behavior_results = [result for result in results if result.category == "behavior"]
    static_results = [result for result in results if result.category == "static"]
    pending_gaps = backend_integration_validation_gaps()

    print("GR00T TRT backend self-check")
    if behavior_results:
        print("Mock-Proven Behavior Checks:")
        for result in behavior_results:
            status = "PASS" if result.ok else "FAIL"
            print(f"[{status}] {result.name}: {result.detail}")

    if static_results:
        print("Static Tripwires:")
        for result in static_results:
            status = "PASS" if result.ok else "FAIL"
            print(f"[{status}] {result.name}: {result.detail}")

    print("Pending Integration Validation:")
    for item in pending_gaps:
        print(f"[PENDING] {item}")

    if failed:
        print(f"SUMMARY: {len(failed)} check(s) failed.")
        return 1

    print(
        "SUMMARY: mock-backed executable checks passed; pending integration validation remains before backend "
        "admission."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
