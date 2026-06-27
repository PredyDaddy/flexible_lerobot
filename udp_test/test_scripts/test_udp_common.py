#!/usr/bin/env python3

from x86_side.udp_common import SequenceStats, decode_packet, encode_packet, packet_age_ms


def test_packet_round_trip() -> None:
    packet = {
        "type": "ping",
        "seq": 7,
        "timestamp_ns": 100,
        "payload": {"robot": "robot1"},
    }

    assert decode_packet(encode_packet(packet)) == packet


def test_sequence_stats_tracks_loss_and_reordered_packets() -> None:
    stats = SequenceStats()

    for seq in [1, 2, 5, 4, 6]:
        stats.update(seq)

    assert stats.received == 5
    assert stats.lost == 2
    assert stats.duplicates_or_reordered == 1


def test_packet_age_ms_uses_monotonic_timestamp() -> None:
    assert packet_age_ms({"timestamp_ns": 1_000_000_000}, now_ns=1_250_000_000) == 250.0
