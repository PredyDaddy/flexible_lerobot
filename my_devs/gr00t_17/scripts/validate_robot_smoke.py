#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def ensure_within(path: Path, root: Path, *, must_exist: bool = False) -> Path:
    resolved = path.expanduser().resolve(strict=must_exist)
    allowed = root.expanduser().resolve(strict=True)
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"Path escapes allowed root: {resolved}") from exc
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a staged SO101 robot smoke run.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expect-actuation", choices=("0", "1"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.expanduser().resolve(strict=True)
    run_dir = ensure_within(args.run_dir, root, must_exist=True)
    report_path = ensure_within(args.report, root)
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite smoke summary: {report_path}")

    predict = json.loads((run_dir / "reports" / "predict.json").read_text(encoding="utf-8"))
    if predict.get("status") != "passed" or predict.get("actuation_enabled") is not False:
        raise RuntimeError("Non-actuating prediction stage did not pass")
    predictions = predict.get("predictions", [])
    if len(predictions) != 2:
        raise RuntimeError(f"Expected two prediction chunks, got {len(predictions)}")
    latency = predict.get("inference_latency_s", {})
    if not math.isfinite(float(latency.get("max", math.nan))):
        raise RuntimeError("Prediction latency is missing or non-finite")
    for camera in ("top", "wrist"):
        summary = predict["cameras"][camera]
        if summary["shape"] != [480, 640, 3] or summary["dtype"] != "uint8":
            raise RuntimeError(f"Camera validation failed for {camera}: {summary}")

    bus_preflight = json.loads((run_dir / "reports" / "bus_preflight.json").read_text(encoding="utf-8"))
    if bus_preflight.get("status") != "passed" or len(bus_preflight.get("motors", {})) != 6:
        raise RuntimeError("Read-only motor bus preflight did not pass for all six motors")
    if bus_preflight.get("cleanup") != "bus disconnected with torque disable":
        raise RuntimeError("Motor bus preflight did not prove torque-disable cleanup")

    actuate = None
    if args.expect_actuation == "1":
        actuate = json.loads((run_dir / "reports" / "actuate.json").read_text(encoding="utf-8"))
        if actuate.get("status") != "passed" or actuate.get("actuation_enabled") is not True:
            raise RuntimeError("Actuated smoke stage did not pass")
        if int(actuate.get("sent_action_count", 0)) <= 0:
            raise RuntimeError("Actuated smoke did not send any guarded actions")
        if actuate.get("checkpoint") != predict.get("checkpoint") or actuate.get("task") != predict.get(
            "task"
        ):
            raise RuntimeError("Prediction and actuation stages used different model/task contracts")
        actuation_safety = actuate["safety"]
        expected_safety = {
            "run_time_s": 1.0,
            "execution_horizon": 1,
            "control_hz": 5.0,
            "max_command_delta": 0.25,
            "max_relative_target": 0.25,
        }
        for key, expected in expected_safety.items():
            if float(actuation_safety[key]) != expected:
                raise RuntimeError(f"Actuated smoke used an unexpected safety value: {key}")

    server_log = (run_dir / "logs" / "policy_server.log").read_text(encoding="utf-8", errors="replace")
    if "Server ready" not in server_log and "Server is ready" not in server_log:
        raise RuntimeError("Policy server log does not contain a ready marker")

    result = {
        "schema_version": 1,
        "status": "passed",
        "level": "guarded_actuation" if actuate else "live_observation_and_prediction",
        "run_dir": str(run_dir),
        "checkpoint": predict["checkpoint"],
        "task": predict["task"],
        "prediction_chunks": len(predictions),
        "motor_bus_preflight": "passed",
        "max_inference_latency_s": float(latency["max"]),
        "actuation_performed": actuate is not None,
        "sent_action_count": int(actuate.get("sent_action_count", 0)) if actuate else 0,
    }
    if actuate:
        result["actuation_safety"] = actuate["safety"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
