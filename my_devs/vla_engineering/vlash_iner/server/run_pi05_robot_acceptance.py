#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if PACKAGE_PARENT.as_posix() not in sys.path:
    sys.path.insert(0, PACKAGE_PARENT.as_posix())

from vlash_iner.common import DEFAULT_ROBOT_PORT, ensure_repo_on_path, parse_bool, resolve_repo_root
from vlash_iner.server.run_pi05_acceptance_audit import DEFAULT_REPORT_DIR


REPO_ROOT = resolve_repo_root(Path(__file__))
ensure_repo_on_path(REPO_ROOT)
VLA_ENGINEERING_DIR = REPO_ROOT / "my_devs/vla_engineering"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run guarded PI0.5 TensorRT robot acceptance stages. "
            "This script never sends robot actions unless --confirm-control true is provided."
        )
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:8008")
    parser.add_argument("--endpoint", default="/infer")
    parser.add_argument("--task", default="Put the eraser into the small box")
    parser.add_argument("--robot-port", default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--top-cam", default="/dev/video4")
    parser.add_argument("--wrist-cam", default="/dev/video6")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--run-lowspeed", action="store_true", help="Run L2 lowspeed robot 20s.")
    parser.add_argument("--run-main", action="store_true", help="Run L3 main robot 120s.")
    parser.add_argument(
        "--confirm-control",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
        help="Required to allow any real robot.send_action call.",
    )
    parser.add_argument(
        "--skip-prereq-audit",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
        help="Skip checking that L1 and L2 readonly gates already passed.",
    )
    parser.add_argument(
        "--dry-run",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
        help="Print commands and preflight checks without running robot stages.",
    )
    return parser


def fetch_health(server_url: str) -> dict[str, Any]:
    with urlopen(f"{server_url.rstrip('/')}/health", timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def require_device(path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"Required device path does not exist: {path}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def require_prereq_gates(report_dir: Path) -> None:
    audit_path = report_dir / "trt_acceptance_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(
            f"Missing audit report: {audit_path}. Run run_pi05_acceptance_audit first."
        )
    audit = load_json(audit_path)
    gates = {gate["name"]: gate for gate in audit.get("gates", [])}
    required = ["L1 infer 100", "L1 async mock 120s", "L2 readonly 60s"]
    failed = [name for name in required if not gates.get(name, {}).get("passed")]
    if failed:
        raise RuntimeError(f"Prerequisite gates are not passed: {failed}")


def stage_command(args: argparse.Namespace, stage: str) -> list[str]:
    report_dir = args.report_dir.expanduser().resolve()
    common = [
        sys.executable,
        "-u",
        "-m",
        "vlash_iner.server.run_pi05_async_client",
        "--server-url",
        args.server_url,
        "--endpoint",
        args.endpoint,
        "--task",
        args.task,
        "--robot-port",
        args.robot_port,
        "--top-cam",
        args.top_cam,
        "--wrist-cam",
        args.wrist_cam,
        "--fps",
        "30",
        "--future-state-aware",
        "false",
        "--action-quant-ratio",
        "1",
        "--connect-retries",
        "3",
        "--connect-retry-s",
        "1.0",
        "--confirm-control",
        "true",
    ]
    if stage == "lowspeed":
        return [
            *common,
            "--run-time-s",
            "20",
            "--control-fps",
            "30",
            "--reuse-observation-within-chunk",
            "false",
            "--inference-overlap-steps",
            "0",
            "--background-inference",
            "false",
            "--chunk-blend-steps",
            "0",
            "--log-interval",
            "5",
            "--output-json",
            str(report_dir / "trt_lowspeed_robot_20s.json"),
        ]
    if stage == "main":
        return [
            *common,
            "--run-time-s",
            "120",
            "--control-fps",
            "45",
            "--reuse-observation-within-chunk",
            "true",
            "--inference-overlap-steps",
            "8",
            "--background-inference",
            "true",
            "--chunk-blend-steps",
            "2",
            "--log-interval",
            "10",
            "--output-json",
            str(report_dir / "trt_main_robot_120s.json"),
        ]
    raise ValueError(f"Unknown stage: {stage}")


def audit_command(args: argparse.Namespace, *, strict: bool) -> list[str]:
    report_dir = args.report_dir.expanduser().resolve()
    command = [
        sys.executable,
        "-m",
        "vlash_iner.server.run_pi05_acceptance_audit",
        "--report-dir",
        str(report_dir),
        "--output-json",
        str(report_dir / "trt_acceptance_audit.json"),
    ]
    if strict:
        command.append("--strict")
    return command


def run_command(command: list[str]) -> None:
    print("[INFO] Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=VLA_ENGINEERING_DIR, check=True)


def main() -> None:
    args = build_parser().parse_args()
    report_dir = args.report_dir.expanduser().resolve()

    if not args.run_lowspeed and not args.run_main:
        raise SystemExit("Nothing to run. Use --run-lowspeed and/or --run-main.")
    if not args.confirm_control and not args.dry_run:
        raise SystemExit(
            "Refusing to run robot action stages without --confirm-control true."
        )

    require_device(args.robot_port)
    require_device(args.top_cam)
    require_device(args.wrist_cam)

    try:
        health = fetch_health(args.server_url)
    except URLError as exc:
        if args.dry_run:
            print(
                f"[WARN] Could not reach PI0.5 server at {args.server_url}/health. "
                "Dry-run will print commands only.",
                flush=True,
            )
            health = {}
        else:
            raise SystemExit(
                f"Could not reach PI0.5 server at {args.server_url}/health. "
                "Start vlash_iner.server.run_pi05_async_server first."
            )
    backend = health.get("backend")
    if health and backend != "tensorrt_split":
        raise RuntimeError(
            f"Expected server backend tensorrt_split, got {backend!r}"
        )
    if health:
        print(
            f"[INFO] Server ready: backend={backend} "
            f"chunk_size={health.get('chunk_size')} n_action_steps={health.get('n_action_steps')}",
            flush=True,
        )

    if not args.skip_prereq_audit:
        require_prereq_gates(report_dir)
        print("[INFO] Prerequisite gates passed: L1 infer, L1 async, L2 readonly.", flush=True)

    stages: list[str] = []
    if args.run_lowspeed:
        stages.append("lowspeed")
    if args.run_main:
        lowspeed_report = report_dir / "trt_lowspeed_robot_20s.json"
        if not args.run_lowspeed and not lowspeed_report.is_file():
            raise RuntimeError(
                "L3 main requires L2 lowspeed evidence first. "
                "Run with --run-lowspeed or provide trt_lowspeed_robot_20s.json."
            )
        stages.append("main")

    for stage in stages:
        command = stage_command(args, stage)
        if args.dry_run:
            print("[DRY_RUN]", " ".join(command), flush=True)
            continue
        run_command(command)
        run_command(audit_command(args, strict=False))

    if not args.dry_run:
        run_command(audit_command(args, strict=args.run_main))


if __name__ == "__main__":
    main()
