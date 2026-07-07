#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def resolve_repo_root(script_path: Path) -> Path:
    resolved = script_path.resolve()
    for candidate in resolved.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {script_path}")


REPO_ROOT = resolve_repo_root(Path(__file__))
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402

from my_devs.train.pi.so101.rtc_pi05.server.http_policy_server import make_server  # noqa: E402
from my_devs.train.pi.so101.rtc_pi05.server.policy_service import PolicyService, PolicyServiceConfig  # noqa: E402

DEFAULT_POLICY_PATH = (
    "outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model"
)
LOG_PREFIX = "[RTC-PI05-SERVER]"


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PI0.5 RTC policy HTTP server.")
    parser.add_argument("--host", default=os.getenv("PI05_SERVER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PI05_SERVER_PORT", "8088")))
    parser.add_argument("--policy-path", default=os.getenv("POLICY_PATH", DEFAULT_POLICY_PATH))
    parser.add_argument("--device", default=os.getenv("DEVICE"))
    parser.add_argument("--enable-rtc", type=parse_bool, nargs="?", const=True, default=True)
    parser.add_argument("--rtc-execution-horizon", type=int, default=10)
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=10.0)
    parser.add_argument("--rtc-debug", type=parse_bool, nargs="?", const=True, default=False)
    parser.add_argument("--check-policy-load", type=parse_bool, nargs="?", const=True, default=False)
    parser.add_argument("--strict-so101-features", type=parse_bool, nargs="?", const=True, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.chdir(REPO_ROOT)
    service_config = PolicyServiceConfig(
        policy_path=Path(args.policy_path).expanduser(),
        enable_rtc=args.enable_rtc,
        device=args.device,
        strict_so101_features=args.strict_so101_features,
    )
    rtc_config = RTCConfig(
        enabled=args.enable_rtc,
        execution_horizon=args.rtc_execution_horizon,
        max_guidance_weight=args.rtc_max_guidance_weight,
        debug=args.rtc_debug,
    )
    print(f"{LOG_PREFIX} Loading policy: {service_config.policy_path}")
    service = PolicyService.from_config_with_rtc(service_config, rtc_config=rtc_config)
    print(
        f"{LOG_PREFIX} Policy ready: device={service.device} rtc={args.enable_rtc} "
        f"execution_horizon={args.rtc_execution_horizon}"
    )
    if args.check_policy_load:
        print(f"{LOG_PREFIX} CHECK_POLICY_LOAD passed.")
        return 0
    http_server = make_server(host=args.host, port=args.port, service=service)
    print(f"{LOG_PREFIX} Listening on http://{args.host}:{args.port}")
    try:
        http_server.serve_forever()
    except KeyboardInterrupt:
        print(f"{LOG_PREFIX} KeyboardInterrupt received, shutting down.")
    finally:
        http_server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
