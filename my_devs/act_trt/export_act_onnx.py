from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from my_devs.act_trt.common import resolve_checkpoint, resolve_deploy_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ACT core model to ONNX via reference exporter.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamo", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint)
    output_dir = resolve_deploy_dir(checkpoint, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_script = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "act_trt"
        / "reference_docs"
        / "onnx_export_reference"
        / "export"
        / "export_single.py"
    )
    if not reference_script.is_file():
        raise FileNotFoundError(f"Reference export script not found: {reference_script}")

    command = [
        sys.executable,
        str(reference_script),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output_dir),
        "--opset",
        str(args.opset),
        "--device",
        args.device,
    ]
    if args.dynamo:
        command.append("--dynamo")
    command.append("--verify" if args.verify else "--no-verify")
    command.append("--simplify" if args.simplify else "--no-simplify")

    print("[INFO] Running:", " ".join(command))
    subprocess.run(command, check=True)
    print(output_dir / "act_single.onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
