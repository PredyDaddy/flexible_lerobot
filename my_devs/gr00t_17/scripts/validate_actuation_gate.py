#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def ensure_within(path: Path, root: Path, *, must_exist: bool = False) -> Path:
    resolved = path.expanduser().resolve(strict=must_exist)
    allowed = root.expanduser().resolve(strict=True)
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"Path escapes allowed root: {resolved}") from exc
    return resolved


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def validate_gate(
    root: Path,
    smoke_report_path: Path,
    checkpoint: Path,
    task: str,
) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    smoke_report_path = ensure_within(smoke_report_path, root, must_exist=True)
    checkpoint = ensure_within(checkpoint, root, must_exist=True)
    report = read_json(smoke_report_path)

    if report.get("status") != "passed" or report.get("level") != "guarded_actuation":
        raise RuntimeError("Smoke report is not a passed guarded-actuation run")
    if report.get("actuation_performed") is not True or int(report.get("sent_action_count", 0)) <= 0:
        raise RuntimeError("Smoke report does not prove a guarded motor command was sent")
    if report.get("task") != task:
        raise RuntimeError(f"Smoke task differs from formal task: {report.get('task')!r} != {task!r}")

    smoke_checkpoint = Path(report["checkpoint"]["path"]).resolve(strict=True)
    if smoke_checkpoint != checkpoint or int(report["checkpoint"]["global_step"]) != 63600:
        raise RuntimeError("Smoke report used a different formal checkpoint")

    safety = report["actuation_safety"]
    maximums = {
        "run_time_s": 1.0,
        "execution_horizon": 1.0,
        "control_hz": 5.0,
        "max_command_delta": 0.25,
        "max_relative_target": 0.25,
    }
    for key, maximum in maximums.items():
        value = float(safety[key])
        if not math.isfinite(value) or value <= 0 or value > maximum:
            raise RuntimeError(f"Smoke safety limit is invalid for {key}: {value} > {maximum}")

    return {
        "status": "passed",
        "smoke_report": str(smoke_report_path),
        "checkpoint": str(checkpoint),
        "task": task,
        "sent_action_count": int(report["sent_action_count"]),
        "actuation_safety": safety,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate formal SO101 inference on a passed guarded smoke.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    result = validate_gate(args.root, args.smoke_report, args.checkpoint, args.task)
    if args.report is not None:
        root = args.root.expanduser().resolve(strict=True)
        report_path = ensure_within(args.report, root)
        if report_path.exists():
            raise FileExistsError(f"Refusing to overwrite actuation gate report: {report_path}")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = report_path.with_name(f"{report_path.name}.partial")
        if partial_path.exists():
            raise FileExistsError(f"Refusing to overwrite partial gate report: {partial_path}")
        partial_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        partial_path.replace(report_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
