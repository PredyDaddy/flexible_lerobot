#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import socket
import time
from dataclasses import dataclass
from typing import Any


DEFAULT_ORIN_IP = "192.168.1.81"
DEFAULT_PING_PORT = 39001
DEFAULT_STATE_PORT = 39002


def monotonic_ns() -> int:
    return time.monotonic_ns()


def encode_packet(packet: dict[str, Any]) -> bytes:
    return json.dumps(packet, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_packet(data: bytes) -> dict[str, Any]:
    packet = json.loads(data.decode("utf-8"))
    if not isinstance(packet, dict):
        raise ValueError("UDP payload must decode to a JSON object")
    return packet


def make_socket(bind_ip: str, bind_port: int, timeout_s: float | None = None) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((bind_ip, bind_port))
    if timeout_s is not None:
        sock.settimeout(timeout_s)
    return sock


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bind-ip", default="0.0.0.0", help="Local IP to bind. Use 0.0.0.0 for all NICs.")
    parser.add_argument("--count", type=int, default=0, help="Number of packets to process. 0 means forever.")


def packet_age_ms(packet: dict[str, Any], now_ns: int | None = None) -> float | None:
    timestamp_ns = packet.get("timestamp_ns")
    if not isinstance(timestamp_ns, int):
        return None
    if now_ns is None:
        now_ns = monotonic_ns()
    return (now_ns - timestamp_ns) / 1_000_000


@dataclass
class SequenceStats:
    received: int = 0
    lost: int = 0
    duplicates_or_reordered: int = 0
    last_seq: int | None = None

    def update(self, seq: int) -> None:
        self.received += 1
        if self.last_seq is None:
            self.last_seq = seq
            return

        expected = self.last_seq + 1
        if seq == expected:
            self.last_seq = seq
        elif seq > expected:
            self.lost += seq - expected
            self.last_seq = seq
        else:
            self.duplicates_or_reordered += 1

    @property
    def loss_percent(self) -> float:
        total = self.received + self.lost
        if total == 0:
            return 0.0
        return self.lost / total * 100.0


@dataclass
class RateCounter:
    start_ns: int
    count: int = 0

    @classmethod
    def start(cls) -> "RateCounter":
        return cls(start_ns=monotonic_ns())

    def tick(self) -> None:
        self.count += 1

    def hz(self) -> float:
        elapsed_s = (monotonic_ns() - self.start_ns) / 1_000_000_000
        if elapsed_s <= 0:
            return 0.0
        return self.count / elapsed_s


def print_packet_summary(prefix: str, packet: dict[str, Any], addr: tuple[str, int] | None = None) -> None:
    source = f" from={addr[0]}:{addr[1]}" if addr else ""
    age_ms = packet_age_ms(packet)
    age_text = f" age_ms={age_ms:.3f}" if age_ms is not None else ""
    print(
        f"{prefix} type={packet.get('type')} seq={packet.get('seq')}{source}{age_text} "
        f"payload={packet.get('payload')}",
        flush=True,
    )
