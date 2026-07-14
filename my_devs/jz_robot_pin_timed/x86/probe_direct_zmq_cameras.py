#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig
from lerobot.robots.jz_robot_pin_timed.timestamped_zmq_camera import TimestampedZMQCamera

CAMERA_SPECS = {
    "camera_head": (5555, 1280, 720),
    "camera_left": (5556, 640, 480),
    "camera_right": (5557, 640, 480),
}
MAX_ORIN_CAPTURE_INTERARRIVAL_MS = 100.0

ORIN_DURATION_FIELDS = {
    "read_ms": ("read_enter_monotonic_ns", "read_return_monotonic_ns"),
    "publisher_dequeue_delay_ms": ("capture_monotonic_ns", "publisher_dequeue_monotonic_ns"),
    "encode_ms": ("encode_started_monotonic_ns", "encode_completed_monotonic_ns"),
    "base64_ms": ("base64_started_monotonic_ns", "base64_completed_monotonic_ns"),
    "json_ms": ("json_started_monotonic_ns", "json_completed_monotonic_ns"),
}


def _summary(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": len(values),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def run_probe(host: str, duration_s: float, min_fps: float) -> dict:
    cameras = {
        name: TimestampedZMQCamera(
            ZMQCameraConfig(
                server_address=host,
                port=port,
                camera_name=name,
                fps=30,
                width=width,
                height=height,
                color_mode=ColorMode.RGB,
                timeout_ms=5000,
            ),
            buffer_size=8,
            stale_frame_timeout_ms=1000,
        )
        for name, (port, width, height) in CAMERA_SPECS.items()
    }
    start_diagnostics: dict[str, dict] = {}
    rgb_means: dict[str, list[list[float]]] = {name: [] for name in cameras}
    payload_bytes: dict[str, list[int]] = {name: [] for name in cameras}
    orin_encode_ms: dict[str, list[float]] = {name: [] for name in cameras}
    decode_stage_ms: dict[str, dict[str, list[float]]] = {name: {} for name in cameras}
    orin_stage_ms: dict[str, dict[str, list[float]]] = {
        name: {stage: [] for stage in ORIN_DURATION_FIELDS} for name in cameras
    }
    capture_thread_interval_ms: dict[str, list[float]] = {name: [] for name in cameras}
    realsense_device_interval_ms: dict[str, list[float]] = {name: [] for name in cameras}
    realsense_frame_gaps: dict[str, list[int]] = {name: [] for name in cameras}
    first_realsense_frame: dict[str, int] = {}
    last_realsense_frame: dict[str, int] = {}
    first_sequences: dict[str, int] = {}
    last_sequences: dict[str, int] = {}
    first_capture_ns: dict[str, int] = {}
    last_capture_ns: dict[str, int] = {}
    started_ns = 0
    stopped_ns = 0
    try:
        for camera in cameras.values():
            camera.connect()
        start_diagnostics = {name: camera.diagnostics for name, camera in cameras.items()}
        started_ns = time.monotonic_ns()
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            for name, camera in cameras.items():
                frame = camera.read_timed()
                if last_sequences.get(name) == frame.sequence:
                    continue
                first_sequences.setdefault(name, frame.sequence)
                last_sequences[name] = frame.sequence
                rgb_means[name].append(frame.image.mean(axis=(0, 1)).tolist())
                timing = frame.camera_timing
                first_capture_ns.setdefault(name, timing["capture_monotonic_ns"])
                last_capture_ns[name] = timing["capture_monotonic_ns"]
                payload_bytes[name].append(timing["payload_bytes"])
                orin_encode_ms[name].append(
                    (timing["encode_completed_monotonic_ns"] - timing["capture_monotonic_ns"]) / 1_000_000
                )
                for stage, value in frame.decode_timing.items():
                    decode_stage_ms[name].setdefault(stage, []).append(value)
                for stage, (started_key, completed_key) in ORIN_DURATION_FIELDS.items():
                    if started_key in timing and completed_key in timing:
                        orin_stage_ms[name][stage].append(
                            (timing[completed_key] - timing[started_key]) / 1_000_000
                        )
                if "capture_thread_interval_ns" in timing:
                    capture_thread_interval_ms[name].append(timing["capture_thread_interval_ns"] / 1_000_000)
                if "realsense_device_interval_ms" in timing:
                    realsense_device_interval_ms[name].append(timing["realsense_device_interval_ms"])
                if "realsense_frame_gap" in timing:
                    realsense_frame_gaps[name].append(int(timing["realsense_frame_gap"]))
                if "realsense_frame_number" in timing:
                    first_realsense_frame.setdefault(name, int(timing["realsense_frame_number"]))
                    last_realsense_frame[name] = int(timing["realsense_frame_number"])
            time.sleep(0.002)
        stopped_ns = time.monotonic_ns()
    finally:
        for camera in cameras.values():
            if camera.context is not None:
                camera.disconnect()

    elapsed_s = (stopped_ns - started_ns) / 1_000_000_000
    reports = {}
    errors = []
    for name, camera in cameras.items():
        before = start_diagnostics[name]
        after = camera.diagnostics
        accepted = after["accepted_frames"] - before["accepted_frames"]
        fps = accepted / elapsed_s
        means = np.asarray(rgb_means[name], dtype=np.float64)
        payload_mean = float(np.mean(payload_bytes[name])) if payload_bytes[name] else None
        sequence_span = last_sequences.get(name, 0) - first_sequences.get(name, 0)
        capture_span_ns = last_capture_ns.get(name, 0) - first_capture_ns.get(name, 0)
        decode_timing = {
            stage: {
                "mean_ms": float(np.mean(values)),
                "max_ms": float(np.max(values)),
            }
            for stage, values in decode_stage_ms[name].items()
        }
        orin_timing = {stage: _summary(values) for stage, values in orin_stage_ms[name].items()}
        frame_gaps = realsense_frame_gaps[name]
        reports[name] = {
            "accepted_frames": accepted,
            "fps": fps,
            "source_fps_from_orin_sequence": (
                None if capture_span_ns <= 0 else sequence_span * 1_000_000_000 / capture_span_ns
            ),
            "first_sequence": first_sequences.get(name),
            "last_sequence": after["latest_sequence"],
            "sequence_gaps": after["sequence_gaps"] - before["sequence_gaps"],
            "raw_queue_drops": after["raw_queue_drops"] - before["raw_queue_drops"],
            "last_socket_interarrival_ms": after["last_interarrival_ms"],
            "max_socket_interarrival_ms": after["max_interarrival_ms"],
            "last_orin_capture_interarrival_ms": after["last_capture_interarrival_ms"],
            "max_orin_capture_interarrival_ms": after["max_capture_interarrival_ms"],
            "connection_max_x86_queue_delay_ms": after["max_queue_delay_ms"],
            "connection_max_x86_decode_ms": after["max_decode_ms"],
            "invalid_messages": after["invalid_messages"] - before["invalid_messages"],
            "duplicate_or_out_of_order_messages": after["duplicate_or_out_of_order_messages"]
            - before["duplicate_or_out_of_order_messages"],
            "rgb_mean": None if means.size == 0 else means.mean(axis=0).round(4).tolist(),
            "jpeg_payload_bytes_mean": payload_mean,
            "estimated_base64_wire_mbps_at_30fps": (
                None if payload_mean is None else payload_mean * (4 / 3) * 30 * 8 / 1_000_000
            ),
            "orin_jpeg_encode_ms_mean": (
                None if not orin_encode_ms[name] else float(np.mean(orin_encode_ms[name]))
            ),
            "orin_stage_timing_ms": orin_timing,
            "capture_thread_interval_ms": _summary(capture_thread_interval_ms[name]),
            "realsense_first_frame_number": first_realsense_frame.get(name),
            "realsense_last_frame_number": last_realsense_frame.get(name),
            "realsense_frame_gap_total": sum(frame_gaps),
            "realsense_frame_gap_max": max(frame_gaps, default=0),
            "realsense_device_interval_ms": _summary(realsense_device_interval_ms[name]),
            "x86_decode_timing": decode_timing,
            "last_error": after["last_error"],
        }
        if fps < min_fps:
            errors.append(f"{name} fps={fps:.3f} below {min_fps:.3f}")
        if reports[name]["invalid_messages"]:
            errors.append(f"{name} rejected {reports[name]['invalid_messages']} messages")
        if reports[name]["sequence_gaps"]:
            errors.append(f"{name} sequence_gaps={reports[name]['sequence_gaps']}")
        if reports[name]["raw_queue_drops"]:
            errors.append(f"{name} raw_queue_drops={reports[name]['raw_queue_drops']}")
        max_capture_interval_ms = reports[name]["max_orin_capture_interarrival_ms"]
        if max_capture_interval_ms is None or max_capture_interval_ms >= MAX_ORIN_CAPTURE_INTERARRIVAL_MS:
            errors.append(
                f"{name} max_orin_capture_interarrival_ms={max_capture_interval_ms} "
                f"must be below {MAX_ORIN_CAPTURE_INTERARRIVAL_MS}"
            )
        if reports[name]["duplicate_or_out_of_order_messages"]:
            errors.append(f"{name} received duplicate/out-of-order messages")

    estimated_wire_mbps = sum(
        camera["estimated_base64_wire_mbps_at_30fps"] or 0 for camera in reports.values()
    )
    return {
        "status": "PASS" if not errors else "FAIL",
        "host": host,
        "duration_s": elapsed_s,
        "protocol": "jz_realsense_zmq",
        "protocol_version": 1,
        "output_pixel_format": "RGB8",
        "max_orin_capture_interarrival_limit_ms": MAX_ORIN_CAPTURE_INTERARRIVAL_MS,
        "estimated_total_base64_wire_mbps_at_30fps": estimated_wire_mbps,
        "cameras": reports,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only probe for direct Orin RealSense ZMQ cameras")
    parser.add_argument("--host", default="192.168.1.81")
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--min-fps", type=float, default=25.0)
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args()
    if args.duration_s <= 0 or args.min_fps <= 0:
        parser.error("duration-s and min-fps must be positive")

    report = run_probe(args.host, args.duration_s, args.min_fps)
    print(json.dumps(report, indent=2))
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
