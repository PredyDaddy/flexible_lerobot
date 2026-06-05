#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if PACKAGE_PARENT.as_posix() not in sys.path:
    sys.path.insert(0, PACKAGE_PARENT.as_posix())

from vlash_iner.common import ensure_repo_on_path, resolve_repo_root


REPO_ROOT = resolve_repo_root(Path(__file__))
ensure_repo_on_path(REPO_ROOT)


DEFAULT_REPORT_DIR = REPO_ROOT / "my_devs/vla_engineering/vlash_iner/server/reports"


@dataclass
class GateResult:
    name: str
    passed: bool
    evidence: str
    details: dict[str, Any]
    missing: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit PI0.5 TensorRT async-chain acceptance evidence.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any gate is missing or failed.")
    return parser


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def parse_readonly_log(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    text = path.read_text(errors="replace")
    step_lines = [line for line in text.splitlines() if "[INFO] Step " in line]
    parsed_steps: list[dict[str, Any]] = []
    pattern = re.compile(
        r"Step (?P<step>\d+) \| elapsed=(?P<elapsed>[0-9.]+)s "
        r"request_latency=(?P<request>[0-9.]+)s "
        r"server_infer=(?P<infer>[0-9.]+)s "
        r"request_count=(?P<count>\d+) "
        r"pending=(?P<pending>\w+) "
        r"wait_count=(?P<wait>\d+)"
    )
    for line in step_lines:
        match = pattern.search(line)
        if not match:
            continue
        parsed_steps.append(
            {
                "step": int(match.group("step")),
                "elapsed": float(match.group("elapsed")),
                "request_latency": float(match.group("request")),
                "server_infer": float(match.group("infer")),
                "request_count": int(match.group("count")),
                "pending": match.group("pending") == "True",
                "wait_count": int(match.group("wait")),
            }
        )
    last_step = parsed_steps[-1] if parsed_steps else None
    request_latencies = [item["request_latency"] for item in parsed_steps]
    server_infers = [item["server_infer"] for item in parsed_steps]
    wait_counts = [item["wait_count"] for item in parsed_steps]
    return {
        "path": str(path),
        "line_count": len(text.splitlines()),
        "step_line_count": len(step_lines),
        "contains_error": "[ERROR]" in text,
        "contains_warn": "[WARN]" in text,
        "no_send_action": "no_send_action: True" in text,
        "confirm_control": "confirm_control: True" in text,
        "finished": "Remote async client finished." in text,
        "last_step": last_step,
        "observed_hz": (
            last_step["step"] / last_step["elapsed"] if last_step and last_step["elapsed"] > 0 else 0.0
        ),
        "request_latency_max": max(request_latencies) if request_latencies else 0.0,
        "server_infer_max": max(server_infers) if server_infers else 0.0,
        "max_wait_count": max(wait_counts) if wait_counts else None,
    }


def gate_missing(name: str, evidence: Path) -> GateResult:
    return GateResult(name=name, passed=False, evidence=str(evidence), details={}, missing=True)


def audit_l1_infer(report_dir: Path) -> GateResult:
    path = report_dir / "trt_infer_acceptance.json"
    report = load_json(path)
    if report is None:
        return gate_missing("L1 infer 100", path)
    details = {
        "backend": report.get("server_health", {}).get("backend"),
        "requests": report.get("requests"),
        "passed": report.get("passed"),
        "request_latency_p95": report.get("request_latency_s", {}).get("p95"),
        "server_infer_p95": report.get("server_infer_s", {}).get("p95"),
        "failures": report.get("failures"),
    }
    passed = (
        report.get("passed") is True
        and details["backend"] == "tensorrt_split"
        and int(details["requests"] or 0) >= 100
        and not details["failures"]
    )
    return GateResult("L1 infer 100", passed, str(path), details)


def audit_l1_async(report_dir: Path) -> GateResult:
    path = report_dir / "trt_async_acceptance.json"
    report = load_json(path)
    if report is None:
        return gate_missing("L1 async mock 120s", path)
    details = {
        "passed": report.get("passed"),
        "run_time_s": report.get("run_time_s"),
        "control_fps": report.get("control_fps"),
        "observed_hz": report.get("observed_hz"),
        "request_count": report.get("request_count"),
        "wait_count": report.get("wait_count"),
        "request_latency_p95": report.get("request_latency_s", {}).get("p95"),
        "failures": report.get("failures"),
    }
    passed = (
        report.get("passed") is True
        and float(details["run_time_s"] or 0.0) >= 120.0
        and int(details["wait_count"] or 0) == 0
        and not details["failures"]
    )
    return GateResult("L1 async mock 120s", passed, str(path), details)


def audit_l2_readonly(report_dir: Path) -> GateResult:
    json_path = report_dir / "trt_readonly_60s.json"
    json_report = load_json(json_path)
    if json_report is not None:
        details = {
            "finished": json_report.get("finished"),
            "error": json_report.get("error"),
            "elapsed_s": json_report.get("elapsed_s"),
            "steps": json_report.get("steps"),
            "observed_hz": json_report.get("observed_hz"),
            "wait_count": json_report.get("wait_count"),
            "no_send_action": json_report.get("no_send_action"),
            "confirm_control": json_report.get("confirm_control"),
            "request_latency_p95": json_report.get("request_latency_s", {}).get("p95"),
        }
        passed = (
            json_report.get("finished") is True
            and json_report.get("no_send_action") is True
            and json_report.get("confirm_control") is False
            and float(json_report.get("elapsed_s") or 0.0) >= 55.0
            and int(json_report.get("wait_count") or 0) == 0
        )
        return GateResult("L2 readonly 60s", passed, str(json_path), details)

    log_path = report_dir / "trt_readonly_60s.log"
    log_report = parse_readonly_log(log_path)
    if log_report is None:
        return gate_missing("L2 readonly 60s", log_path)
    last_step = log_report.get("last_step") or {}
    details = {
        "finished": log_report["finished"],
        "contains_error": log_report["contains_error"],
        "contains_warn": log_report["contains_warn"],
        "no_send_action": log_report["no_send_action"],
        "confirm_control": log_report["confirm_control"],
        "last_step": last_step,
        "observed_hz": log_report["observed_hz"],
        "max_wait_count": log_report["max_wait_count"],
        "request_latency_max": log_report["request_latency_max"],
    }
    passed = (
        log_report["finished"]
        and not log_report["contains_error"]
        and not log_report["contains_warn"]
        and log_report["no_send_action"]
        and not log_report["confirm_control"]
        and float(last_step.get("elapsed") or 0.0) >= 55.0
        and int(log_report["max_wait_count"] or 0) == 0
    )
    return GateResult("L2 readonly 60s", passed, str(log_path), details)


def audit_l2_lowspeed(report_dir: Path) -> GateResult:
    path = report_dir / "trt_lowspeed_robot_20s.json"
    report = load_json(path)
    if report is None:
        return gate_missing("L2 lowspeed robot 20s", path)
    details = {
        "finished": report.get("finished"),
        "error": report.get("error"),
        "elapsed_s": report.get("elapsed_s"),
        "steps": report.get("steps"),
        "observed_hz": report.get("observed_hz"),
        "wait_count": report.get("wait_count"),
        "no_send_action": report.get("no_send_action"),
        "confirm_control": report.get("confirm_control"),
        "backend": report.get("health", {}).get("backend"),
    }
    passed = (
        report.get("finished") is True
        and report.get("error") is None
        and report.get("no_send_action") is False
        and report.get("confirm_control") is True
        and details["backend"] == "tensorrt_split"
        and float(report.get("elapsed_s") or 0.0) >= 18.0
    )
    return GateResult("L2 lowspeed robot 20s", passed, str(path), details)


def audit_l3_main(report_dir: Path) -> GateResult:
    path = report_dir / "trt_main_robot_120s.json"
    report = load_json(path)
    if report is None:
        return gate_missing("L3 main robot 120s", path)
    control_fps = float(report.get("control_fps") or 0.0)
    overlap_steps = int(report.get("inference_overlap_steps") or 0)
    latency_budget = overlap_steps / control_fps if control_fps > 0 and overlap_steps > 0 else None
    request_latency_p95 = report.get("request_latency_s", {}).get("p95")
    details = {
        "finished": report.get("finished"),
        "error": report.get("error"),
        "elapsed_s": report.get("elapsed_s"),
        "steps": report.get("steps"),
        "observed_hz": report.get("observed_hz"),
        "wait_count": report.get("wait_count"),
        "no_send_action": report.get("no_send_action"),
        "confirm_control": report.get("confirm_control"),
        "backend": report.get("health", {}).get("backend"),
        "request_latency_p95": request_latency_p95,
        "latency_budget_s": latency_budget,
    }
    passed = (
        report.get("finished") is True
        and report.get("error") is None
        and report.get("no_send_action") is False
        and report.get("confirm_control") is True
        and details["backend"] == "tensorrt_split"
        and float(report.get("elapsed_s") or 0.0) >= 115.0
        and int(report.get("wait_count") or 0) == 0
        and (
            latency_budget is None
            or request_latency_p95 is not None
            and float(request_latency_p95) < latency_budget
        )
    )
    return GateResult("L3 main robot 120s", passed, str(path), details)


def main() -> None:
    args = build_parser().parse_args()
    report_dir = args.report_dir.expanduser().resolve()
    gates = [
        audit_l1_infer(report_dir),
        audit_l1_async(report_dir),
        audit_l2_readonly(report_dir),
        audit_l2_lowspeed(report_dir),
        audit_l3_main(report_dir),
    ]
    summary = {
        "report_dir": str(report_dir),
        "passed": all(gate.passed for gate in gates),
        "gates": [
            {
                "name": gate.name,
                "passed": gate.passed,
                "missing": gate.missing,
                "evidence": gate.evidence,
                "details": gate.details,
            }
            for gate in gates
        ],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print(f"[INFO] Audit written: {args.output_json}")
    if args.strict and not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
