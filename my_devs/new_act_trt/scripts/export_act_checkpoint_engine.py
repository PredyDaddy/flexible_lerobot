#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    from act_model_utils import resolve_checkpoint, resolve_output_dir
    from act_trt_paths import REPO_ROOT
else:
    from .act_model_utils import resolve_checkpoint, resolve_output_dir
    from .act_trt_paths import REPO_ROOT


SCRIPT_DIR = Path(__file__).resolve().parent
ACT_EXPORT_SCRIPT = SCRIPT_DIR / "export_act_onnx.py"
ACT_BUILD_SCRIPT = SCRIPT_DIR / "build_act_trt_engine.py"
ACT_VERIFY_CORE_SCRIPT = SCRIPT_DIR / "verify_act_torch_onnx_trt.py"
ACT_VERIFY_POLICY_SCRIPT = SCRIPT_DIR / "verify_act_torch_trt_policy.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a LeRobot ACT checkpoint to ONNX and TensorRT, then optionally verify\n"
            "Torch vs ONNX vs TensorRT core consistency and Torch vs TensorRT policy consistency."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Path to pretrained_model checkpoint directory.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Artifact output directory. Defaults to outputs/deploy/act_trt/<run_name>/<checkpoint_step>.",
    )
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device for ONNX export.")
    parser.add_argument("--trt-device", default="cuda:0", help="CUDA device string used by TensorRT runtime.")
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    parser.add_argument("--opt-level", type=int, default=3)
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--timing-cache", default=None, help="Optional TensorRT timing cache path.")
    parser.add_argument("--skip-export", action="store_true", help="Reuse existing act_single.onnx and metadata.")
    parser.add_argument("--verify-core", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-policy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamo", action="store_true")
    parser.add_argument("--core-random-cases", type=int, default=3)
    parser.add_argument("--core-threshold-max-abs-diff", type=float, default=1e-4)
    parser.add_argument("--policy-random-cases", type=int, default=4)
    parser.add_argument("--policy-sequence-length", type=int, default=12)
    parser.add_argument("--policy-n-action-steps", type=int, default=None)
    parser.add_argument("--policy-temporal-ensemble-coeff", type=float, default=None)
    parser.add_argument("--policy-threshold-max-abs-diff-norm", type=float, default=1e-4)
    parser.add_argument("--policy-threshold-max-abs-diff-real", type=float, default=1e-3)
    parser.add_argument("--policy-threshold-max-rel-diff", type=float, default=1e-2)
    parser.add_argument("--policy-threshold-min-cosine-similarity", type=float, default=0.999)
    parser.add_argument("--task", default="")
    parser.add_argument("--robot-type", default="")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("[RUN]", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint)
    output_dir = resolve_output_dir(checkpoint, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / "act_single.onnx"
    metadata_path = output_dir / "export_metadata.json"
    engine_path = output_dir / f"act_single_{args.precision}.plan"
    build_report_path = output_dir / f"trt_build_summary_{args.precision}.json"
    core_report_path = output_dir / f"consistency_report_act_single_{args.precision}.json"
    policy_report_path = output_dir / f"consistency_report_act_single_{args.precision}_policy.json"
    suite_report_path = output_dir / f"consistency_suite_act_single_{args.precision}.json"

    python_exe = sys.executable

    if not args.skip_export:
        export_command = [
            python_exe,
            str(ACT_EXPORT_SCRIPT),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output_dir),
            "--opset",
            str(args.opset),
            "--device",
            args.device,
        ]
        if args.dynamo:
            export_command.append("--dynamo")
        export_command.append("--simplify" if args.simplify else "--no-simplify")
        run_command(export_command)
    else:
        if not onnx_path.is_file():
            raise FileNotFoundError(f"--skip-export was set but ONNX file is missing: {onnx_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"--skip-export was set but metadata file is missing: {metadata_path}")

    build_command = [
        python_exe,
        str(ACT_BUILD_SCRIPT),
        "--onnx",
        str(onnx_path),
        "--metadata",
        str(metadata_path),
        "--engine",
        str(engine_path),
        "--precision",
        args.precision,
        "--workspace-gb",
        str(args.workspace_gb),
        "--opt-level",
        str(args.opt_level),
        "--allow-tf32" if args.allow_tf32 else "--no-allow-tf32",
        "--report",
        str(build_report_path),
        "--device",
        args.trt_device,
    ]
    if args.timing_cache:
        build_command.extend(["--timing-cache", str(Path(args.timing_cache).expanduser().resolve())])
    run_command(build_command)

    if args.verify_core:
        verify_core_command = [
            python_exe,
            str(ACT_VERIFY_CORE_SCRIPT),
            "--checkpoint",
            str(checkpoint),
            "--onnx",
            str(onnx_path),
            "--engine",
            str(engine_path),
            "--device",
            args.trt_device,
            "--random-cases",
            str(args.core_random_cases),
            "--threshold-max-abs-diff",
            str(args.core_threshold_max_abs_diff),
            "--report",
            str(core_report_path),
        ]
        run_command(verify_core_command)

    if args.verify_policy:
        verify_policy_command = [
            python_exe,
            str(ACT_VERIFY_POLICY_SCRIPT),
            "--checkpoint",
            str(checkpoint),
            "--engine",
            str(engine_path),
            "--metadata",
            str(metadata_path),
            "--device",
            args.trt_device,
            "--random-cases",
            str(args.policy_random_cases),
            "--sequence-length",
            str(args.policy_sequence_length),
            "--threshold-max-abs-diff-norm",
            str(args.policy_threshold_max_abs_diff_norm),
            "--threshold-max-abs-diff-real",
            str(args.policy_threshold_max_abs_diff_real),
            "--threshold-max-rel-diff",
            str(args.policy_threshold_max_rel_diff),
            "--threshold-min-cosine-similarity",
            str(args.policy_threshold_min_cosine_similarity),
            "--task",
            args.task,
            "--robot-type",
            args.robot_type,
            "--report",
            str(policy_report_path),
        ]
        if args.policy_n_action_steps is not None:
            verify_policy_command.extend(["--policy-n-action-steps", str(args.policy_n_action_steps)])
        if args.policy_temporal_ensemble_coeff is not None:
            verify_policy_command.extend(
                ["--policy-temporal-ensemble-coeff", str(args.policy_temporal_ensemble_coeff)]
            )
        run_command(verify_policy_command)

    build_summary = read_json(build_report_path)
    suite_payload = {
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "artifacts": {
            "onnx": str(onnx_path),
            "metadata": str(metadata_path),
            "engine": str(engine_path),
            "build_report": str(build_report_path),
            "core_report": str(core_report_path) if core_report_path.is_file() else None,
            "policy_report": str(policy_report_path) if policy_report_path.is_file() else None,
        },
        "build_summary": {
            "precision": build_summary.get("precision"),
            "allow_tf32": build_summary.get("allow_tf32"),
            "device": build_summary.get("device"),
            "engine_tensors": build_summary.get("engine_tensors", []),
        },
        "core_verify": read_json(core_report_path) if core_report_path.is_file() else None,
        "policy_verify": read_json(policy_report_path) if policy_report_path.is_file() else None,
    }
    suite_payload["passed"] = bool(
        (suite_payload["core_verify"] is None or suite_payload["core_verify"]["passed"])
        and (suite_payload["policy_verify"] is None or suite_payload["policy_verify"]["passed"])
    )
    write_json(suite_report_path, suite_payload)

    print(f"[ARTIFACT] onnx={onnx_path}")
    print(f"[ARTIFACT] metadata={metadata_path}")
    print(f"[ARTIFACT] engine={engine_path}")
    print(f"[ARTIFACT] build_report={build_report_path}")
    if core_report_path.is_file():
        print(f"[ARTIFACT] core_report={core_report_path}")
    if policy_report_path.is_file():
        print(f"[ARTIFACT] policy_report={policy_report_path}")
    print(f"[ARTIFACT] suite_report={suite_report_path}")

    tensors = build_summary.get("engine_tensors", [])
    if tensors:
        print("[ENGINE I/O]")
        for tensor in tensors:
            print(
                f"  - {tensor['name']}: mode={tensor['mode']} dtype={tensor['dtype']} "
                f"shape={tuple(tensor['shape'])}"
            )

    return 0 if suite_payload["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
