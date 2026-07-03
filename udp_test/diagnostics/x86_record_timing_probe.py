#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import select
import socket
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def monotonic_ns() -> int:
    return time.monotonic_ns()


def decode_packet(data: bytes) -> dict[str, Any]:
    packet = json.loads(data.decode("utf-8"))
    if not isinstance(packet, dict):
        raise ValueError("packet must decode to a JSON object")
    return packet


def make_socket(bind_ip: str, bind_port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((bind_ip, bind_port))
    sock.setblocking(False)
    return sock


def pct(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile / 100.0)
    return ordered[index]


def fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


@dataclass
class StreamStats:
    name: str
    received: int = 0
    lost: int = 0
    duplicates_or_reordered: int = 0
    last_seq: int | None = None
    last_recv_ns: int | None = None
    last_sample_ns: int | None = None
    recv_intervals_ms: list[float] = field(default_factory=list)
    source_age_ms: list[float] = field(default_factory=list)
    source_skew_ms: list[float] = field(default_factory=list)
    network_age_ms: list[float] = field(default_factory=list)

    def update(self, packet: dict[str, Any], recv_ns: int) -> None:
        self.received += 1
        seq = packet.get("seq")
        if isinstance(seq, int):
            if self.last_seq is not None:
                expected = self.last_seq + 1
                if seq > expected:
                    self.lost += seq - expected
                elif seq <= self.last_seq:
                    self.duplicates_or_reordered += 1
            if self.last_seq is None or seq > self.last_seq:
                self.last_seq = seq

        if self.last_recv_ns is not None:
            self.recv_intervals_ms.append((recv_ns - self.last_recv_ns) / 1_000_000)
        self.last_recv_ns = recv_ns

        sample_ns = packet.get("sample_monotonic_ns")
        if isinstance(sample_ns, int):
            self.last_sample_ns = sample_ns
            self.network_age_ms.append((recv_ns - sample_ns) / 1_000_000)

        source_age = packet.get("source_age_ms")
        if isinstance(source_age, int | float):
            self.source_age_ms.append(float(source_age))
        source_skew = packet.get("source_skew_ms")
        if isinstance(source_skew, int | float):
            self.source_skew_ms.append(float(source_skew))

    @property
    def loss_percent(self) -> float:
        total = self.received + self.lost
        return 0.0 if total == 0 else self.lost / total * 100.0

    def summary(self) -> str:
        interval_p50 = pct(self.recv_intervals_ms, 50)
        interval_p99 = pct(self.recv_intervals_ms, 99)
        age_p50 = pct(self.source_age_ms, 50)
        age_p99 = pct(self.source_age_ms, 99)
        skew_p99 = pct(self.source_skew_ms, 99)
        net_p50 = pct(self.network_age_ms, 50)
        net_p99 = pct(self.network_age_ms, 99)
        hz = 1000.0 / statistics.mean(self.recv_intervals_ms) if self.recv_intervals_ms else 0.0
        return (
            f"{self.name}: recv={self.received} hz={hz:.2f} lost={self.lost} "
            f"loss={self.loss_percent:.3f}% dup/reorder={self.duplicates_or_reordered} "
            f"interval_ms_p50/p99={fmt_ms(interval_p50)}/{fmt_ms(interval_p99)} "
            f"source_age_ms_p50/p99={fmt_ms(age_p50)}/{fmt_ms(age_p99)} "
            f"source_skew_ms_p99={fmt_ms(skew_p99)} "
            f"network_age_ms_p50/p99={fmt_ms(net_p50)}/{fmt_ms(net_p99)}"
        )


@dataclass
class PairStats:
    state_action_sample_skew_ms: list[float] = field(default_factory=list)

    def update(self, state: StreamStats, action: StreamStats) -> None:
        if state.last_sample_ns is None or action.last_sample_ns is None:
            return
        self.state_action_sample_skew_ms.append(abs(state.last_sample_ns - action.last_sample_ns) / 1_000_000)

    def summary(self) -> str:
        return (
            "state_action_sample_skew_ms "
            f"p50/p90/p99/max={fmt_ms(pct(self.state_action_sample_skew_ms, 50))}/"
            f"{fmt_ms(pct(self.state_action_sample_skew_ms, 90))}/"
            f"{fmt_ms(pct(self.state_action_sample_skew_ms, 99))}/"
            f"{fmt_ms(max(self.state_action_sample_skew_ms) if self.state_action_sample_skew_ms else None)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="x86-side readonly receiver for Orin record timing probe diagnostic UDP packets."
    )
    parser.add_argument("--bind-ip", default="0.0.0.0")
    parser.add_argument("--state-port", type=int, default=39110)
    parser.add_argument("--action-port", type=int, default=39130)
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--print-every-s", type=float, default=5.0)
    parser.add_argument("--out", type=Path, default=None, help="Optional JSONL output path for raw packets.")
    parser.add_argument("--allowed-sender-ip", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state_sock = make_socket(args.bind_ip, args.state_port)
    action_sock = make_socket(args.bind_ip, args.action_port)
    sockets = {
        state_sock: "state",
        action_sock: "target_action",
    }
    stats = {
        "state": StreamStats("state"),
        "target_action": StreamStats("target_action"),
    }
    pair_stats = PairStats()

    out_fh = None
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_fh = args.out.open("w", encoding="utf-8")

    print(
        f"[x86_record_timing_probe] listening state={args.bind_ip}:{args.state_port} "
        f"target_action={args.bind_ip}:{args.action_port} duration_s={args.duration_s}",
        flush=True,
    )

    start = time.monotonic()
    next_print = start + args.print_every_s
    try:
        while time.monotonic() - start < args.duration_s:
            readable, _, _ = select.select(list(sockets), [], [], 0.2)
            for sock in readable:
                data, addr = sock.recvfrom(65535)
                if args.allowed_sender_ip is not None and addr[0] != args.allowed_sender_ip:
                    continue
                recv_ns = monotonic_ns()
                packet = decode_packet(data)
                stream = sockets[sock]
                if packet.get("stream") != stream:
                    print(
                        f"[x86_record_timing_probe] warning: packet stream={packet.get('stream')} "
                        f"arrived on {stream} port from={addr[0]}:{addr[1]}",
                        flush=True,
                    )
                stats[stream].update(packet, recv_ns)
                pair_stats.update(stats["state"], stats["target_action"])
                if out_fh is not None:
                    out_fh.write(
                        json.dumps(
                            {
                                "recv_monotonic_ns": recv_ns,
                                "sender": [addr[0], addr[1]],
                                "packet": packet,
                            },
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                        + "\n"
                    )

            now = time.monotonic()
            if now >= next_print:
                print(stats["state"].summary(), flush=True)
                print(stats["target_action"].summary(), flush=True)
                print(pair_stats.summary(), flush=True)
                next_print = now + args.print_every_s
    finally:
        state_sock.close()
        action_sock.close()
        if out_fh is not None:
            out_fh.close()

    print("[x86_record_timing_probe] final summary", flush=True)
    print(stats["state"].summary(), flush=True)
    print(stats["target_action"].summary(), flush=True)
    print(pair_stats.summary(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
