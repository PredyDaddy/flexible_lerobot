#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import time

from udp_common import DEFAULT_STATE_PORT, encode_packet, make_socket, monotonic_ns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orin-side readonly fake state sender over UDP.")
    parser.add_argument("--target-ip", required=True, help="x86 receiver IP.")
    parser.add_argument("--target-port", type=int, default=DEFAULT_STATE_PORT, help="x86 receiver UDP port.")
    parser.add_argument("--bind-ip", default="192.168.1.81", help="Orin local IP to bind.")
    parser.add_argument("--bind-port", type=int, default=0, help="Orin local UDP port. 0 means auto.")
    parser.add_argument("--hz", type=float, default=20.0, help="State packet frequency.")
    parser.add_argument("--count", type=int, default=0, help="Number of packets to send. 0 means forever.")
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
        packet = {
            "type": "state",
            "seq": seq,
            "timestamp_ns": monotonic_ns(),
            "payload": fake_state(seq),
        }
        sock.sendto(encode_packet(packet), (args.target_ip, args.target_port))

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
