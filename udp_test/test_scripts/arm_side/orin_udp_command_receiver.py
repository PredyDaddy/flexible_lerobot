#!/usr/bin/env python3

from __future__ import annotations

import argparse
import signal
import socket
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(SRC_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from lerobot.robots.jz_robot_udp.protocol import ProtocolError, decode_jz_robot_udp_command_packet
except ImportError as exc:
    print(
        "[orin command receiver] ERROR: command decoder is not available yet. "
        "Expected lerobot.robots.jz_robot_udp.protocol.decode_jz_robot_udp_command_packet.",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(2) from exc

DEFAULT_BIND_IP = "192.168.1.81"
DEFAULT_ALLOWED_SENDER_IP = "192.168.1.106"
DEFAULT_COMMAND_PORT = 39020
DEFAULT_BUFFER_SIZE = 65535

_SHUTDOWN_REQUESTED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 Orin UDP command dry-run receiver.")
    parser.add_argument("--bind-ip", default=DEFAULT_BIND_IP)
    parser.add_argument("--port", type=int, default=DEFAULT_COMMAND_PORT)
    parser.add_argument("--allowed-sender-ip", default=DEFAULT_ALLOWED_SENDER_IP)
    parser.add_argument("--count", type=int, default=0, help="Stop after N valid commands. 0 means run forever.")
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--buffer-size", type=int, default=DEFAULT_BUFFER_SIZE)
    return parser.parse_args()


def request_shutdown(signum: int, _frame: object) -> None:
    global _SHUTDOWN_REQUESTED
    _SHUTDOWN_REQUESTED = True
    print(f"[orin command receiver] signal={signum} shutdown requested", flush=True)


def install_signal_handlers() -> None:
    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)


def action_counts(packet: dict) -> tuple[int, int, int]:
    actions = packet.get("actions", {})
    left = actions.get("left", {})
    right = actions.get("right", {})
    grippers = actions.get("grippers", {})
    gripper_fields = 0
    if isinstance(grippers, dict):
        for side in ("left", "right"):
            side_values = grippers.get(side, {})
            if isinstance(side_values, dict):
                gripper_fields += len(side_values)
    return len(left) if isinstance(left, dict) else 0, len(right) if isinstance(right, dict) else 0, gripper_fields


def print_startup(args: argparse.Namespace) -> None:
    print("PHASE2 COMMAND DRY-RUN ONLY", flush=True)
    print("NOT publishing ROS command topics", flush=True)
    print("robot will not move", flush=True)
    print(
        "[orin command receiver] "
        f"bind={args.bind_ip}:{args.port} allowed_sender_ip={args.allowed_sender_ip} "
        f"count={args.count} buffer_size={args.buffer_size}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    if args.port <= 0 or args.port > 65535:
        raise ValueError(f"--port must be in 1..65535, got {args.port}")
    if args.buffer_size <= 0:
        raise ValueError("--buffer-size must be positive")
    if args.count < 0:
        raise ValueError("--count must be non-negative")

    install_signal_handlers()
    print_startup(args)

    received = 0
    invalid = 0
    unexpected_sender = 0
    last_seq: int | None = None
    started = time.monotonic()

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((args.bind_ip, args.port))
        sock.settimeout(0.2)
        while not _SHUTDOWN_REQUESTED:
            if args.count and received >= args.count:
                break
            try:
                data, sender = sock.recvfrom(args.buffer_size)
            except socket.timeout:
                continue

            sender_ip, sender_port = sender
            if args.allowed_sender_ip and sender_ip != args.allowed_sender_ip:
                unexpected_sender += 1
                print(
                    "[orin command receiver] WARN unexpected sender "
                    f"{sender_ip}:{sender_port}, expected {args.allowed_sender_ip}",
                    flush=True,
                )
                continue

            try:
                packet = decode_jz_robot_udp_command_packet(data)
            except ProtocolError as exc:
                invalid += 1
                print(f"[orin command receiver] WARN invalid packet bytes={len(data)} reason={exc}", flush=True)
                continue

            seq = packet.get("seq")
            if isinstance(seq, int) and last_seq is not None and seq <= last_seq:
                invalid += 1
                print(
                    "[orin command receiver] WARN non-monotonic seq "
                    f"seq={seq} last_seq={last_seq} bytes={len(data)}",
                    flush=True,
                )
                continue
            if isinstance(seq, int):
                last_seq = seq

            received += 1
            left_count, right_count, gripper_count = action_counts(packet)
            if received == 1 or (args.print_every > 0 and received % args.print_every == 0):
                print(
                    "[orin command receiver] DRY_RUN received "
                    f"idx={received} seq={seq} bytes={len(data)} "
                    f"left_actions={left_count} right_actions={right_count} gripper_actions={gripper_count}",
                    flush=True,
                )

    elapsed_s = time.monotonic() - started
    print(
        "SUMMARY: "
        f"received={received} invalid={invalid} unexpected_sender={unexpected_sender} "
        f"elapsed_s={elapsed_s:.3f}",
        flush=True,
    )
    return 0 if invalid == 0 and unexpected_sender == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
