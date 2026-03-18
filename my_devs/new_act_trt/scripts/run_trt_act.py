#!/usr/bin/env python3
"""
conda run -n lerobot_flex python my_devs/new_act_trt/scripts/run_trt_act.py \
    --policy-path outputs/train/20260305_190147_act_grasp_block_in_bin1_e15/checkpoints/last/pretrained_model \
    --task "Put the block in the bin"
"""


from __future__ import annotations

import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from my_devs.train.act.so101.run_act_infer import main as run_act_infer_main


def main(argv: list[str] | None = None) -> int:
    forwarded_args = list(sys.argv[1:] if argv is None else argv)

    if "--policy-backend" in forwarded_args:
        backend_index = max(index for index, value in enumerate(forwarded_args) if value == "--policy-backend")
        if backend_index + 1 < len(forwarded_args) and forwarded_args[backend_index + 1] != "trt":
            print("[INFO] run_trt_act.py overrides --policy-backend to `trt`.", flush=True)

    forwarded_args.extend(["--policy-backend", "trt"])
    run_act_infer_main(forwarded_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
