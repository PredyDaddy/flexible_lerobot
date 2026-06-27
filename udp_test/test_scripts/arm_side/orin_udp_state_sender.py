#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

from udp_common import DEFAULT_STATE_PORT, encode_packet, make_socket, monotonic_ns

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.robots.jz_robot_udp.protocol import PROTOCOL_VERSION, STATE_MESSAGE_TYPE, encode_state_packet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orin-side readonly fake state sender over UDP.")
    parser.add_argument("--target-ip", required=True, help="x86 receiver IP.")
    parser.add_argument("--target-port", type=int, default=DEFAULT_STATE_PORT, help="x86 receiver UDP port.")
    parser.add_argument("--bind-ip", default="192.168.1.81", help="Orin local IP to bind.")
    parser.add_argument("--bind-port", type=int, default=0, help="Orin local UDP port. 0 means auto.")
    parser.add_argument("--hz", type=float, default=20.0, help="State packet frequency.")
    parser.add_argument("--count", type=int, default=0, help="Number of packets to send. 0 means forever.")
    parser.add_argument(
        "--schema",
        choices=("legacy", "jz_robot_udp"),
        default="legacy",
        help="Packet schema. legacy keeps the original test receiver format.",
    )
    return parser.parse_args()


def fake_state(seq: int) -> dict[str, object]:
    phase = seq / 20.0
    return {
        "robot_id": "robot1",
        "mode": "readonly_fake_state",
        "arm_left": {
            "joint_names": ["j1", "j2", "j3", "j4", "j5", "j6"],
            "position": [round(math.sin(phase + i * 0.1), 6) for i in range(6)],
        },
        "arm_right": {
            "joint_names": ["j1", "j2", "j3", "j4", "j5", "j6"],
            "position": [round(math.cos(phase + i * 0.1), 6) for i in range(6)],
        },
    }


def fake_jz_robot_udp_state(seq: int) -> dict[str, object]:
    phase = seq / 20.0
    return {
        "version": PROTOCOL_VERSION,
        "type": STATE_MESSAGE_TYPE,
        "robot": "robot1",
        "seq": seq,
        "stamp_ns": time.time_ns(),
        "joints": {
            "left": {
                f"left_joint{i}": round(math.sin(phase + i * 0.1), 6)
                for i in range(1, 8)
            },
            "right": {
                f"right_joint{i}": round(math.cos(phase + i * 0.1), 6)
                for i in range(1, 8)
            },
        },
        "grippers": {
            "left": {"width": 0.01, "force": 1.0},
            "right": {"width": 0.02, "force": 2.0},
        },
    }


def main() -> None:
    args = parse_args()
    sock = make_socket(args.bind_ip, args.bind_port)
    local_ip, local_port = sock.getsockname()
    period_s = 1.0 / args.hz
    print(
        f"[orin state sender] local={local_ip}:{local_port} target={args.target_ip}:{args.target_port} "
        f"hz={args.hz}",
        flush=True,
    )

    seq = 0
    next_send = time.monotonic()
    while args.count == 0 or seq < args.count:
        seq += 1
        if args.schema == "jz_robot_udp":
            payload = encode_state_packet(fake_jz_robot_udp_state(seq))
        else:
            packet = {
                "type": "state",
                "seq": seq,
                "timestamp_ns": monotonic_ns(),
                "payload": fake_state(seq),
            }
            payload = encode_packet(packet)
        sock.sendto(payload, (args.target_ip, args.target_port))

        if seq == 1 or seq % max(1, int(args.hz)) == 0:
            print(f"[orin state sender] sent seq={seq}", flush=True)

        next_send += period_s
        sleep_s = next_send - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            next_send = time.monotonic()


if __name__ == "__main__":
    main()
