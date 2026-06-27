#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time

from udp_common import DEFAULT_ORIN_IP, DEFAULT_PING_PORT, decode_packet, encode_packet, make_socket, monotonic_ns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="x86-side UDP ping client. Sends ping packets to Orin.")
    parser.add_argument("--orin-ip", default=DEFAULT_ORIN_IP, help="Orin UDP server IP.")
    parser.add_argument("--port", type=int, default=DEFAULT_PING_PORT, help="Orin UDP server port.")
    parser.add_argument("--bind-ip", default="0.0.0.0", help="x86 local IP to bind.")
    parser.add_argument("--bind-port", type=int, default=0, help="x86 local UDP port. 0 means auto.")
    parser.add_argument("--count", type=int, default=20, help="Number of ping packets to send.")
    parser.add_argument("--interval", type=float, default=0.2, help="Seconds between ping packets.")
    parser.add_argument("--timeout", type=float, default=1.0, help="Seconds to wait for each pong.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sock = make_socket(args.bind_ip, args.bind_port, timeout_s=args.timeout)
    local_ip, local_port = sock.getsockname()
    print(
        f"[x86 ping client] local={local_ip}:{local_port} target={args.orin_ip}:{args.port} "
        f"count={args.count}",
        flush=True,
    )

    sent = 0
    received = 0
    rtt_values_ms: list[float] = []

    for seq in range(1, args.count + 1):
        timestamp_ns = monotonic_ns()
        request = {
            "type": "ping",
            "seq": seq,
            "timestamp_ns": timestamp_ns,
            "payload": {"client": "x86"},
        }
        sock.sendto(encode_packet(request), (args.orin_ip, args.port))
        sent += 1

        try:
            data, addr = sock.recvfrom(4096)
            packet = decode_packet(data)
        except TimeoutError:
            print(f"[x86 ping client] timeout seq={seq}", flush=True)
            time.sleep(args.interval)
            continue

        now_ns = monotonic_ns()
        rtt_ms = (now_ns - timestamp_ns) / 1_000_000
        rtt_values_ms.append(rtt_ms)
        received += 1
        print(
            f"[x86 ping client] pong seq={packet.get('seq')} from={addr[0]}:{addr[1]} "
            f"rtt_ms={rtt_ms:.3f}",
            flush=True,
        )
        time.sleep(args.interval)

    loss = sent - received
    avg_rtt = sum(rtt_values_ms) / len(rtt_values_ms) if rtt_values_ms else 0.0
    max_rtt = max(rtt_values_ms) if rtt_values_ms else 0.0
    print(
        f"[x86 ping client] summary sent={sent} received={received} lost={loss} "
        f"avg_rtt_ms={avg_rtt:.3f} max_rtt_ms={max_rtt:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
