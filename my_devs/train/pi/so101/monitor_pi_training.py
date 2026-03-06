#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path


def parse_step_token(token: str) -> int:
    token = token.strip().replace(",", "")
    if token.endswith("K"):
        return int(float(token[:-1]) * 1000)
    return int(float(token))


def max_step_from_log(log_file: Path) -> int:
    if not log_file.exists():
        return 0
    text = log_file.read_text(errors="ignore")
    matches = re.findall(r"step:([0-9]+(?:\.[0-9]+)?K?)", text)
    if not matches:
        return 0
    return parse_step_token(matches[-1])


def max_step_from_ckpt(ckpt_dir: Path) -> int:
    if not ckpt_dir.exists():
        return 0
    vals = []
    for p in ckpt_dir.iterdir():
        if p.is_dir() and re.fullmatch(r"\d{6}", p.name):
            vals.append(int(p.name))
    return max(vals) if vals else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file")
    parser.add_argument("--target-step", type=int, default=15000)
    parser.add_argument("--milestone-step", type=int, default=10000)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    log_file = Path(args.log_file)
    run_dir = Path(str(log_file).removesuffix(".log"))
    ckpt_dir = run_dir / "checkpoints"
    status_file = run_dir / "monitor_status.txt"
    run_dir.mkdir(parents=True, exist_ok=True)

    milestone_reported = False

    while True:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            step_log = max_step_from_log(log_file)
            step_ckpt = max_step_from_ckpt(ckpt_dir)
            current_step = max(step_log, step_ckpt)
            last_line = ""
            if log_file.exists():
                with log_file.open("r", errors="ignore") as f:
                    lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()

            content = (
                f"timestamp={now}\n"
                f"log_file={log_file}\n"
                f"run_dir={run_dir}\n"
                f"target_step={args.target_step}\n"
                f"milestone_step={args.milestone_step}\n"
                f"current_step={current_step}\n"
                f"last_log_line={last_line}\n"
            )

            if current_step >= args.milestone_step and not milestone_reported:
                content += (
                    f"[MONITOR] {now} reached milestone step {args.milestone_step} "
                    f"(current={current_step})\n"
                )
                milestone_reported = True

            if current_step >= args.target_step:
                content += f"[MONITOR] {now} reached target step {args.target_step}.\n"
                status_file.write_text(content)
                return 0

            status_file.write_text(content)
        except Exception as e:  # pragma: no cover
            status_file.write_text(
                f"timestamp={now}\nlog_file={log_file}\nerror={repr(e)}\n",
            )

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
