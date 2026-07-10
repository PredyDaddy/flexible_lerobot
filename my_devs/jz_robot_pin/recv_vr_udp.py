#!/usr/bin/env python3
"""Simple UDP receiver for VR controller packet testing.

This script mirrors teleop_vr_recv's packet layout so the controller sender can
be tested without starting ROS 2.
"""

import argparse
import math
import socket
import struct
import time


FRAME_HEADER = b"\xAA\xBB"
MIN_PACKET_SIZE = 69
FULL_PACKET_SIZE = 185
VR_DEVICE_SIZE = 28
CONTROLLER_INPUT_SIZE = 16


def parse_float(data, offset):
    return struct.unpack_from("<f", data, offset)[0], offset + 4


def parse_pose(data, offset):
    x, offset = parse_float(data, offset)
    y, offset = parse_float(data, offset)
    z, offset = parse_float(data, offset)
    qx, offset = parse_float(data, offset)
    qy, offset = parse_float(data, offset)
    qz, offset = parse_float(data, offset)
    qw, offset = parse_float(data, offset)
    return (x, y, z, qx, qy, qz, qw), offset


def parse_input(data, offset):
    trigger, offset = parse_float(data, offset)
    grip = bool(data[offset])
    primary = bool(data[offset + 1])
    secondary = bool(data[offset + 2])
    menu = bool(data[offset + 3])
    offset += 4
    joystick_x, offset = parse_float(data, offset)
    joystick_y, offset = parse_float(data, offset)
    return {
        "trigger": trigger,
        "grip": grip,
        "primary": primary,
        "secondary": secondary,
        "menu": menu,
        "joystick_x": joystick_x,
        "joystick_y": joystick_y,
    }, offset


def is_finite(values):
    return all(math.isfinite(value) for value in values)


def parse_packet(data):
    if len(data) < MIN_PACKET_SIZE:
        raise ValueError(f"packet too short: {len(data)} bytes")
    if data[:2] != FRAME_HEADER:
        raise ValueError(f"bad frame header: {data[:2].hex(' ')}")

    length_units = data[2]
    declared_size = 3 + length_units * 4
    if declared_size > len(data):
        raise ValueError(
            f"declared size {declared_size} bytes is larger than received {len(data)} bytes"
        )

    torque_raw = struct.unpack_from("<h", data, 3)[0]
    offset = 5
    angles = list(struct.unpack_from("<16f", data, offset))
    offset += 16 * 4

    packet = {
        "length_units": length_units,
        "torque": torque_raw / 10.0,
        "left_arm_deg": angles[0:7],
        "left_gripper": angles[7],
        "right_arm_deg": angles[8:15],
        "right_gripper": angles[15],
        "has_extended": False,
    }

    remaining = len(data) - offset
    if remaining == 0:
        return packet

    needed = 3 * VR_DEVICE_SIZE + 2 * CONTROLLER_INPUT_SIZE
    if remaining < needed:
        raise ValueError(
            f"incomplete extended data: remaining {remaining} bytes, need {needed}"
        )

    left_pose, offset = parse_pose(data, offset)
    right_pose, offset = parse_pose(data, offset)
    headset_pose, offset = parse_pose(data, offset)
    left_input, offset = parse_input(data, offset)
    right_input, offset = parse_input(data, offset)

    if not is_finite(left_pose + right_pose + headset_pose):
        raise ValueError("pose contains NaN or Inf")

    packet.update(
        {
            "has_extended": True,
            "left_pose": left_pose,
            "right_pose": right_pose,
            "headset_pose": headset_pose,
            "left_input": left_input,
            "right_input": right_input,
        }
    )
    return packet


def fmt_pose(pose):
    x, y, z, qx, qy, qz, qw = pose
    return (
        f"pos=({x:+.4f}, {y:+.4f}, {z:+.4f}) "
        f"quat=({qx:+.4f}, {qy:+.4f}, {qz:+.4f}, {qw:+.4f})"
    )


def fmt_input(input_data):
    return (
        f"trigger={input_data['trigger']:.3f} "
        f"grip={int(input_data['grip'])} "
        f"primary={int(input_data['primary'])} "
        f"secondary={int(input_data['secondary'])} "
        f"menu={int(input_data['menu'])} "
        f"joy=({input_data['joystick_x']:+.3f}, {input_data['joystick_y']:+.3f})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Listen for VR controller UDP packets and print parsed data."
    )
    parser.add_argument("--host", default="10.1.42.3", help="local IP to bind")
    parser.add_argument("--port", type=int, default=8080, help="UDP port to bind")
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="print every N valid packets",
    )
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    sock.settimeout(1.0)

    print(f"listening on udp://{args.host}:{args.port}")
    print("waiting for controller packets... press Ctrl+C to stop")

    valid_count = 0
    bad_count = 0
    last_report = time.monotonic()
    last_count = 0

    try:
        while True:
            try:
                data, addr = sock.recvfrom(2048)
            except socket.timeout:
                continue

            try:
                packet = parse_packet(data)
            except ValueError as exc:
                bad_count += 1
                print(f"[bad #{bad_count}] from {addr[0]}:{addr[1]} {exc}")
                continue

            valid_count += 1
            now = time.monotonic()
            elapsed = now - last_report
            fps = 0.0
            if elapsed >= 1.0:
                fps = (valid_count - last_count) / elapsed
                last_report = now
                last_count = valid_count

            if valid_count % max(args.print_every, 1) != 0:
                continue

            print(
                f"\n[packet #{valid_count}] from {addr[0]}:{addr[1]} "
                f"bytes={len(data)} length={packet['length_units']} fps={fps:.1f}"
            )
            print(
                f"  torque={packet['torque']:.2f} "
                f"left_gripper={packet['left_gripper']:.2f} "
                f"right_gripper={packet['right_gripper']:.2f}"
            )

            if not packet["has_extended"]:
                print("  no extended controller pose data in this packet")
                continue

            print(f"  left_controller   {fmt_pose(packet['left_pose'])}")
            print(f"  right_controller  {fmt_pose(packet['right_pose'])}")
            print(f"  headset           {fmt_pose(packet['headset_pose'])}")
            print(f"  left_input        {fmt_input(packet['left_input'])}")
            print(f"  right_input       {fmt_input(packet['right_input'])}")

    except KeyboardInterrupt:
        print(f"\nstopped. valid_packets={valid_count}, bad_packets={bad_count}")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
