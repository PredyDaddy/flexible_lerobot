#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
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

from lerobot.configs.types import RTCAttentionSchedule  # noqa: E402
from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402

from my_devs.train.pi.so101.rtc_pi05.server.http_policy_server import make_server  # noqa: E402
from my_devs.train.pi.so101.rtc_pi05.trt_server.trt_policy_service import (  # noqa: E402
    TRTPolicyService,
    TRTPolicyServiceConfig,
    validate_runtime_assets,
)


DEFAULT_RUNTIME_ASSETS_DIR = Path("my_devs/openpi_trt/artifacts/pi05_runtime_assets")
DEFAULT_PREFIX_ENGINE = Path("my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp16_constrained.engine")
DEFAULT_DENOISE_ENGINE = Path("my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp16_constrained.engine")
LOG_PREFIX = "[RTC-PI05-TRT-SERVER]"


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else parse_bool(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PI0.5 RTC pure TensorRT policy HTTP server.")
    parser.add_argument("--host", default=os.getenv("PI05_TRT_SERVER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PI05_TRT_SERVER_PORT", "8090")))
    parser.add_argument("--runtime-assets-dir", type=Path, default=DEFAULT_RUNTIME_ASSETS_DIR)
    parser.add_argument("--profile", choices=["auto", "fp32", "fp16_constrained"], default="auto")
    parser.add_argument("--prefix-engine-path", type=Path, default=DEFAULT_PREFIX_ENGINE)
    parser.add_argument("--denoise-engine-path", type=Path, default=DEFAULT_DENOISE_ENGINE)
    parser.add_argument("--device", default=os.getenv("DEVICE", "cuda"))
    parser.add_argument("--enable-rtc", type=parse_bool, nargs="?", const=True, default=env_bool("ENABLE_RTC", True))
    parser.add_argument("--rtc-execution-horizon", type=int, default=int(os.getenv("RTC_EXECUTION_HORIZON", "10")))
    parser.add_argument(
        "--rtc-max-guidance-weight",
        type=float,
        default=float(os.getenv("RTC_MAX_GUIDANCE_WEIGHT", "10.0")),
    )
    parser.add_argument("--rtc-prefix-attention-schedule", default=os.getenv("RTC_PREFIX_ATTENTION_SCHEDULE", "LINEAR"))
    parser.add_argument("--rtc-debug", type=parse_bool, nargs="?", const=True, default=env_bool("RTC_DEBUG", False))
    parser.add_argument("--rtc-debug-maxlen", type=int, default=int(os.getenv("RTC_DEBUG_MAXLEN", "100")))
    parser.add_argument(
        "--strict-runtime-assets",
        type=parse_bool,
        nargs="?",
        const=True,
        default=env_bool("STRICT_RUNTIME_ASSETS", True),
    )
    parser.add_argument("--dry-run", type=parse_bool, nargs="?", const=True, default=env_bool("DRY_RUN", False))
    parser.add_argument("--check-policy-load", type=parse_bool, nargs="?", const=True, default=False)
    return parser


def build_rtc_config(args: argparse.Namespace) -> RTCConfig:
    schedule_name = str(args.rtc_prefix_attention_schedule).upper()
    try:
        schedule = RTCAttentionSchedule[schedule_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported RTC prefix attention schedule: {args.rtc_prefix_attention_schedule}") from exc
    return RTCConfig(
        enabled=args.enable_rtc,
        prefix_attention_schedule=schedule,
        max_guidance_weight=args.rtc_max_guidance_weight,
        execution_horizon=args.rtc_execution_horizon,
        debug=args.rtc_debug,
        debug_maxlen=args.rtc_debug_maxlen,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.chdir(REPO_ROOT)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    service_config = TRTPolicyServiceConfig(
        runtime_assets_dir=args.runtime_assets_dir,
        prefix_engine_path=args.prefix_engine_path,
        denoise_engine_path=args.denoise_engine_path,
        profile=args.profile,
        enable_rtc=args.enable_rtc,
        device=args.device,
        strict_runtime_assets=args.strict_runtime_assets,
    )
    print_resolved_config(args, service_config)
    validate_runtime_assets(service_config.runtime_assets_dir, strict=service_config.strict_runtime_assets)
    if not service_config.prefix_engine_path.is_file():
        raise FileNotFoundError(f"Prefix TensorRT engine does not exist: {service_config.prefix_engine_path}")
    if not service_config.denoise_engine_path.is_file():
        raise FileNotFoundError(f"Denoise TensorRT engine does not exist: {service_config.denoise_engine_path}")
    if args.dry_run:
        print(f"{LOG_PREFIX} DRY_RUN passed: runtime assets and engine paths are present.")
        return 0

    rtc_config = build_rtc_config(args)
    print(f"{LOG_PREFIX} Loading pure TensorRT runtime...")
    service = TRTPolicyService.from_config(service_config, rtc_config=rtc_config)
    print(f"{LOG_PREFIX} Policy ready:")
    print(json.dumps(service.describe(), indent=2, ensure_ascii=True))
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


def print_resolved_config(args: argparse.Namespace, config: TRTPolicyServiceConfig) -> None:
    print(f"{LOG_PREFIX} Repo root: {REPO_ROOT}")
    print(f"{LOG_PREFIX} host={args.host} port={args.port}")
    print(f"{LOG_PREFIX} runtime_assets_dir={config.runtime_assets_dir}")
    print(f"{LOG_PREFIX} profile={config.resolved_profile_name()} requested={config.profile}")
    print(f"{LOG_PREFIX} prefix_engine={config.prefix_engine_path}")
    print(f"{LOG_PREFIX} denoise_engine={config.denoise_engine_path}")
    print(f"{LOG_PREFIX} device={config.device} enable_rtc={config.enable_rtc}")
    print(
        f"{LOG_PREFIX} rtc execution_horizon={args.rtc_execution_horizon} "
        f"max_guidance_weight={args.rtc_max_guidance_weight} "
        f"schedule={args.rtc_prefix_attention_schedule}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
