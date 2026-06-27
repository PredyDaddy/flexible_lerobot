#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time

from udp_common import DEFAULT_STATE_PORT, SequenceStats, RateCounter, decode_packet, make_socket, packet_age_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="x86-side readonly UDP state receiver.")
    parser.add_argument("--bind-ip", default="0.0.0.0", help="x86 local IP to bind.")
    parser.add_argument("--port", type=int, default=DEFAULT_STATE_PORT, help="x86 UDP port to listen on.")
    parser.add_argument("--count", type=int, default=0, help="Number of state packets to receive. 0 means forever.")
    parser.add_argument("--print-every", type=int, default=20, help="Print one line every N packets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sock = make_socket(args.bind_ip, args.port)
    print(f"[x86 state receiver] listening on {args.bind_ip}:{args.port}", flush=True)

    stats = SequenceStats()
    rate = RateCounter.start()
    first_time = time.monotonic()

    while args.count == 0 or stats.received < args.count:
        data, addr = sock.recvfrom(65535)
        try:
            packet = decode_packet(data)
        except Exception as exc:
            print(f"[x86 state receiver] invalid packet from {addr}: {exc}", flush=True)
            continue

        if packet.get("type") != "state":
            print(f"[x86 state receiver] ignored type={packet.get('type')} from {addr}", flush=True)
            continue

        seq = packet.get("seq")
        if not isinstance(seq, int):
            print(f"[x86 state receiver] ignored packet with invalid seq={seq} from {addr}", flush=True)
            continue

        stats.update(seq)
        rate.tick()
        age_ms = packet_age_ms(packet)
        payload = packet.get("payload") if isinstance(packet.get("payload"), dict) else {}
        robot_id = payload.get("robot_id") if isinstance(payload, dict) else None

        if stats.received == 1 or stats.received % args.print_every == 0:
            age_text = f"{age_ms:.3f}" if age_ms is not None else "unknown"
            print(
                f"[x86 state receiver] recv={stats.received} seq={seq} from={addr[0]}:{addr[1]} "
                f"robot={robot_id} age_ms={age_text} hz={rate.hz():.2f} "
                f"lost={stats.lost} reordered={stats.duplicates_or_reordered} "
                f"loss_percent={stats.loss_percent:.2f}",
                flush=True,
            )

    elapsed = time.monotonic() - first_time
    print(
        f"[x86 state receiver] summary received={stats.received} lost={stats.lost} "
        f"reordered={stats.duplicates_or_reordered} elapsed_s={elapsed:.3f} hz={rate.hz():.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
