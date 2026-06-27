#!/usr/bin/env python3

from __future__ import annotations

import argparse

from udp_common import DEFAULT_PING_PORT, decode_packet, encode_packet, make_socket, monotonic_ns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orin-side UDP ping server. Replies pong to x86 ping packets.")
    parser.add_argument("--bind-ip", default="192.168.1.81", help="Orin local IP to bind.")
    parser.add_argument("--port", type=int, default=DEFAULT_PING_PORT, help="UDP port to listen on.")
    parser.add_argument("--count", type=int, default=0, help="Number of packets to reply. 0 means forever.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sock = make_socket(args.bind_ip, args.port)
    print(f"[orin ping server] listening on {args.bind_ip}:{args.port}", flush=True)

    replied = 0
    while args.count == 0 or replied < args.count:
        data, addr = sock.recvfrom(4096)
        try:
            packet = decode_packet(data)
        except Exception as exc:
            print(f"[orin ping server] invalid packet from {addr}: {exc}", flush=True)
            continue

        if packet.get("type") != "ping":
            print(f"[orin ping server] ignored type={packet.get('type')} from {addr}", flush=True)
            continue

        response = {
            "type": "pong",
            "seq": packet.get("seq"),
            "timestamp_ns": packet.get("timestamp_ns"),
            "server_timestamp_ns": monotonic_ns(),
            "payload": {"server": "orin", "bind_ip": args.bind_ip},
        }
        sock.sendto(encode_packet(response), addr)
        replied += 1
        print(f"[orin ping server] pong seq={packet.get('seq')} to {addr[0]}:{addr[1]}", flush=True)


if __name__ == "__main__":
    main()
