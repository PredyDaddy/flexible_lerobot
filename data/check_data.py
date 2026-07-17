#!/usr/bin/env python3
"""Read-only integrity checker for local LeRobot v3 datasets.

The checker distinguishes three different states which are easy to conflate after
an interrupted recording:

* committed and usable episodes: metadata, parquet rows, videos, and timing sidecar
  all agree;
* committed but unusable episodes: metadata/data exist, but a required video or
  another training-critical artifact is missing/corrupt;
* uncommitted fragments: for example ``meta/timing/episode-XXXXXX.jsonl`` left by
  a recording that failed before ``dataset.save_episode()``.

The script never edits a dataset.  It only writes a JSON file when
``--json-report`` is explicitly provided.

Examples:

    /home/luzhuang/miniconda3/envs/lerobot_flex/bin/python data/check_data.py data/testupright3

    /home/luzhuang/miniconda3/envs/lerobot_flex/bin/python data/check_data.py \
        data/testdownright1 data/testrightright1 --verbose

    # Scan every child of data/ that contains meta/info.json.
    /home/luzhuang/miniconda3/envs/lerobot_flex/bin/python data/check_data.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - depends on the selected Python environment
    raise SystemExit("pyarrow is required. Run this script with the lerobot_flex conda environment.") from exc


TIMING_FILE_RE = re.compile(r"^episode-(\d+)\.jsonl$")
REQUIRED_DATA_COLUMNS = (
    "episode_index",
    "frame_index",
    "timestamp",
    "index",
    "task_index",
)
BASE_EPISODE_COLUMNS = (
    "episode_index",
    "length",
    "tasks",
    "data/chunk_index",
    "data/file_index",
    "dataset_from_index",
    "dataset_to_index",
)


@dataclass
class Issue:
    severity: str
    code: str
    message: str
    component: str | None = None


@dataclass
class CameraTimingSummary:
    frames: int = 0
    skew_min_ms: float | None = None
    skew_p50_ms: float | None = None
    skew_p95_ms: float | None = None
    skew_max_ms: float | None = None
    age_max_ms: float | None = None
    over_warning_skew: int = 0
    over_max_skew: int = 0
    reused_frames: int = 0
    backward_sequences: int = 0
    reported_sequence_gaps: int = 0


@dataclass
class EpisodeReport:
    episode_index: int
    length: int | None = None
    data_frames: int = 0
    timing_frames: int = 0
    status: str = "UNKNOWN"
    issues: list[Issue] = field(default_factory=list)
    cameras: dict[str, CameraTimingSummary] = field(default_factory=dict)
    video_paths: dict[str, str] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return not any(issue.severity == "ERROR" for issue in self.issues)

    def finish(self) -> None:
        if any(issue.severity == "ERROR" for issue in self.issues):
            self.status = "FAIL"
        elif any(issue.severity == "WARNING" for issue in self.issues):
            self.status = "WARN"
        else:
            self.status = "PASS"


@dataclass
class VideoProbe:
    path: str
    ok: bool
    codec: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    duration_s: float | None = None
    frame_count: int | None = None
    decode_ok: bool | None = None
    error: str | None = None


@dataclass
class DatasetReport:
    path: str
    status: str = "UNKNOWN"
    total_episodes_declared: int | None = None
    total_frames_declared: int | None = None
    fps: float | None = None
    issues: list[Issue] = field(default_factory=list)
    episodes: list[EpisodeReport] = field(default_factory=list)
    orphan_timing: list[dict[str, Any]] = field(default_factory=list)
    videos: dict[str, VideoProbe] = field(default_factory=dict)

    def finish(self) -> None:
        for episode in self.episodes:
            episode.finish()
        has_error = any(issue.severity == "ERROR" for issue in self.issues)
        has_error = has_error or any(not episode.usable for episode in self.episodes)
        has_warning = any(issue.severity == "WARNING" for issue in self.issues)
        has_warning = has_warning or any(episode.status == "WARN" for episode in self.episodes)
        has_warning = has_warning or bool(self.orphan_timing)
        self.status = "FAIL" if has_error else "WARN" if has_warning else "PASS"

    @property
    def usable_episode_count(self) -> int:
        return sum(episode.usable for episode in self.episodes)


@dataclass(frozen=True)
class CheckConfig:
    warn_camera_skew_ms: float
    max_camera_skew_ms: float
    warn_camera_age_ms: float
    max_camera_age_ms: float
    max_source_skew_ms: float
    max_source_age_ms: float
    require_timing: bool
    require_source_timing: bool
    deep_video: bool
    video_timeout_s: float


def add_dataset_issue(
    report: DatasetReport,
    severity: str,
    code: str,
    message: str,
    component: str | None = None,
) -> None:
    report.issues.append(Issue(severity, code, message, component))


def add_episode_issue(
    episode: EpisodeReport,
    severity: str,
    code: str,
    message: str,
    component: str | None = None,
) -> None:
    issue = Issue(severity, code, message, component)
    if issue not in episode.issues:
        episode.issues.append(issue)


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def as_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def read_json_file(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("top-level JSON value is not an object")
    return value


def read_parquet_rows(
    paths: list[Path], requested_columns: tuple[str, ...] | list[str]
) -> tuple[list[dict], set[str]]:
    rows: list[dict] = []
    all_columns: set[str] = set()
    for path in paths:
        schema_columns = set(pq.read_schema(path).names)
        all_columns.update(schema_columns)
        selected = [column for column in requested_columns if column in schema_columns]
        table = pq.read_table(path, columns=selected)
        for row in table.to_pylist():
            row["__parquet_file__"] = str(path)
            rows.append(row)
    return rows, all_columns


def parse_fraction(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None


def parse_optional_int(value: object) -> int | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def parse_optional_float(value: object) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        number = float(str(value))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def probe_video(path: Path, *, deep_decode: bool, timeout_s: float) -> VideoProbe:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return VideoProbe(str(path), ok=False, error="ffprobe was not found in PATH")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return VideoProbe(str(path), ok=False, error=f"ffprobe timed out after {timeout_s:g}s")
    if completed.returncode != 0:
        error = completed.stderr.strip() or f"ffprobe exited with {completed.returncode}"
        return VideoProbe(str(path), ok=False, error=error)
    try:
        payload = json.loads(completed.stdout)
        streams = payload.get("streams", [])
        if not streams:
            raise ValueError("no video stream")
        stream = streams[0]
        format_data = payload.get("format", {})
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return VideoProbe(str(path), ok=False, error=f"invalid ffprobe output: {exc}")

    frame_count = parse_optional_int(stream.get("nb_read_frames"))
    if frame_count is None:
        frame_count = parse_optional_int(stream.get("nb_frames"))
    duration = parse_optional_float(stream.get("duration"))
    if duration is None:
        duration = parse_optional_float(format_data.get("duration"))
    fps = parse_fraction(stream.get("avg_frame_rate"))
    if fps is None:
        fps = parse_fraction(stream.get("r_frame_rate"))
    probe = VideoProbe(
        path=str(path),
        ok=True,
        codec=stream.get("codec_name"),
        width=as_int(stream.get("width")),
        height=as_int(stream.get("height")),
        fps=fps,
        duration_s=duration,
        frame_count=frame_count,
    )

    if deep_decode:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            probe.ok = False
            probe.decode_ok = False
            probe.error = "ffmpeg was not found in PATH for --deep-video"
            return probe
        decode_command = [
            ffmpeg,
            "-v",
            "error",
            "-xerror",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ]
        try:
            decoded = subprocess.run(
                decode_command,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            probe.ok = False
            probe.decode_ok = False
            probe.error = f"ffmpeg decode timed out after {timeout_s:g}s"
        else:
            probe.decode_ok = decoded.returncode == 0
            if not probe.decode_ok:
                probe.ok = False
                probe.error = decoded.stderr.strip() or f"ffmpeg exited with {decoded.returncode}"
    return probe


def load_timing_files(root: Path) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[Issue]]]:
    timing_dir = root / "meta" / "timing"
    records: dict[int, list[dict[str, Any]]] = {}
    issues: dict[int, list[Issue]] = defaultdict(list)
    if not timing_dir.is_dir():
        return records, issues
    for path in sorted(timing_dir.glob("episode-*.jsonl")):
        match = TIMING_FILE_RE.match(path.name)
        if match is None:
            continue
        file_episode = int(match.group(1))
        if file_episode in records:
            issues[file_episode].append(
                Issue("ERROR", "TIMING_DUPLICATE_FILE", f"multiple timing files for episode {file_episode}")
            )
            continue
        file_records: list[dict[str, Any]] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            issues[file_episode].append(Issue("ERROR", "TIMING_READ", str(exc), str(path)))
            records[file_episode] = file_records
            continue
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                issues[file_episode].append(
                    Issue("ERROR", "TIMING_EMPTY_LINE", f"empty line {line_number}", str(path))
                )
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                issues[file_episode].append(
                    Issue(
                        "ERROR",
                        "TIMING_JSON",
                        f"line {line_number}: {exc}",
                        str(path),
                    )
                )
                continue
            if not isinstance(value, dict):
                issues[file_episode].append(
                    Issue(
                        "ERROR",
                        "TIMING_RECORD_TYPE",
                        f"line {line_number} is not a JSON object",
                        str(path),
                    )
                )
                continue
            record_episode = as_int(value.get("episode_index"))
            if record_episode != file_episode:
                issues[file_episode].append(
                    Issue(
                        "ERROR",
                        "TIMING_EPISODE_MISMATCH",
                        f"line {line_number}: record episode={record_episode}, filename episode={file_episode}",
                        str(path),
                    )
                )
            file_records.append(value)
        records[file_episode] = file_records
    return records, issues


def summarize_orphan_timing(episode_index: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame_indices = [as_int(row.get("frame_index")) for row in rows]
    valid_frames = [value for value in frame_indices if value is not None]
    camera_max_skew: dict[str, float] = {}
    for row in rows:
        cameras = row.get("cameras")
        if not isinstance(cameras, dict):
            continue
        for camera_name, camera in cameras.items():
            if not isinstance(camera, dict) or not is_number(camera.get("state_receive_skew_ms")):
                continue
            skew = abs(float(camera["state_receive_skew_ms"]))
            camera_max_skew[camera_name] = max(camera_max_skew.get(camera_name, 0.0), skew)
    return {
        "episode_index": episode_index,
        "rows": len(rows),
        "first_frame": min(valid_frames) if valid_frames else None,
        "last_frame": max(valid_frames) if valid_frames else None,
        "camera_max_skew_ms": camera_max_skew,
        "classification": "UNCOMMITTED_TIMING_FRAGMENT",
    }


def validate_source_timing(
    episode: EpisodeReport,
    rows: list[dict[str, Any]],
    config: CheckConfig,
) -> None:
    missing = 0
    source_skews: list[float] = []
    source_ages: list[float] = []
    inconsistent = 0
    for row in rows:
        state = row.get("state")
        source_timing = state.get("source_timing") if isinstance(state, dict) else None
        if not isinstance(source_timing, dict):
            missing += 1
            continue
        if is_number(source_timing.get("source_skew_ms")):
            recorded_skew = float(source_timing["source_skew_ms"])
            source_skews.append(recorded_skew)
        else:
            recorded_skew = None
        sources = source_timing.get("sources")
        recv_times: list[int] = []
        if isinstance(sources, dict):
            for source in sources.values():
                if not isinstance(source, dict):
                    continue
                if is_number(source.get("age_ms")):
                    source_ages.append(float(source["age_ms"]))
                recv_ns = as_int(source.get("recv_monotonic_ns"))
                if recv_ns is not None:
                    recv_times.append(recv_ns)
        if recorded_skew is not None and len(recv_times) >= 2:
            calculated = (max(recv_times) - min(recv_times)) / 1_000_000
            if abs(calculated - recorded_skew) > 0.01:
                inconsistent += 1
    if missing and config.require_source_timing:
        add_episode_issue(
            episode,
            "ERROR",
            "SOURCE_TIMING_MISSING",
            f"source_timing is missing from {missing}/{len(rows)} timing records",
            "state",
        )
    if inconsistent:
        add_episode_issue(
            episode,
            "ERROR",
            "SOURCE_SKEW_INCONSISTENT",
            f"recorded source skew disagrees with receive clocks in {inconsistent} frames",
            "state",
        )
    if source_skews and max(source_skews) > config.max_source_skew_ms:
        add_episode_issue(
            episode,
            "ERROR",
            "SOURCE_SKEW_TOO_LARGE",
            f"max source skew {max(source_skews):.3f}ms exceeds {config.max_source_skew_ms:.3f}ms",
            "state",
        )
    if source_ages and max(source_ages) > config.max_source_age_ms:
        add_episode_issue(
            episode,
            "ERROR",
            "SOURCE_AGE_TOO_LARGE",
            f"max source age {max(source_ages):.3f}ms exceeds {config.max_source_age_ms:.3f}ms",
            "state",
        )


def validate_timing_rows(
    episode: EpisodeReport,
    rows: list[dict[str, Any]],
    expected_cameras: list[str],
    config: CheckConfig,
) -> None:
    episode.timing_frames = len(rows)
    expected_frames = list(range(len(rows)))
    actual_frames = [as_int(row.get("frame_index")) for row in rows]
    if actual_frames != expected_frames:
        add_episode_issue(
            episode,
            "ERROR",
            "TIMING_FRAME_ORDER",
            "timing frame_index is not exactly contiguous from zero",
            "timing",
        )

    observation_sequences: list[int] = []
    state_sequences: list[int] = []
    command_mismatches = 0
    missing_base_fields = 0
    camera_values: dict[str, dict[str, list[float] | list[int]]] = {}
    for camera_name in expected_cameras:
        camera_values[camera_name] = {
            "skew": [],
            "age": [],
            "sequence": [],
            "gap": [],
            "reused_flag": [],
        }
    camera_field_errors: dict[str, int] = defaultdict(int)
    skew_recompute_errors: dict[str, int] = defaultdict(int)

    for row in rows:
        observation_sequence = as_int(row.get("observation_sequence"))
        if observation_sequence is not None:
            observation_sequences.append(observation_sequence)
        state = row.get("state")
        command = row.get("command")
        cameras = row.get("cameras")
        if not isinstance(state, dict) or not isinstance(command, dict) or not isinstance(cameras, dict):
            missing_base_fields += 1
            continue
        state_receive_ns = as_int(state.get("receive_monotonic_ns"))
        state_sequence = as_int(state.get("packet_seq"))
        if state_sequence is not None:
            state_sequences.append(state_sequence)
        if command.get("observation_sequence") != observation_sequence:
            command_mismatches += 1
        actual_camera_names = set(cameras)
        if actual_camera_names != set(expected_cameras):
            missing = sorted(set(expected_cameras) - actual_camera_names)
            extra = sorted(actual_camera_names - set(expected_cameras))
            add_episode_issue(
                episode,
                "ERROR",
                "TIMING_CAMERA_SET",
                f"camera keys differ; missing={missing}, extra={extra}",
                "timing",
            )
        for camera_name in expected_cameras:
            camera = cameras.get(camera_name)
            if not isinstance(camera, dict):
                camera_field_errors[camera_name] += 1
                continue
            skew = camera.get("state_receive_skew_ms")
            age = camera.get("age_ms")
            sequence = as_int(camera.get("sequence"))
            if not is_number(skew) or not is_number(age) or sequence is None:
                camera_field_errors[camera_name] += 1
                continue
            skew_value = abs(float(skew))
            camera_values[camera_name]["skew"].append(skew_value)
            camera_values[camera_name]["age"].append(float(age))
            camera_values[camera_name]["sequence"].append(sequence)
            gap = as_int(camera.get("sequence_gap"))
            camera_values[camera_name]["gap"].append(max(0, gap or 0))
            camera_values[camera_name]["reused_flag"].append(
                1 if camera.get("reused_by_observation_loop") is True else 0
            )
            receive_ns = as_int(camera.get("receive_monotonic_ns"))
            recorded_delta = camera.get("state_receive_delta_ms")
            if state_receive_ns is not None and receive_ns is not None:
                calculated_delta = (receive_ns - state_receive_ns) / 1_000_000
                calculated_skew = abs(calculated_delta)
                delta_ok = is_number(recorded_delta) and abs(float(recorded_delta) - calculated_delta) <= 0.01
                if abs(skew_value - calculated_skew) > 0.01 or not delta_ok:
                    skew_recompute_errors[camera_name] += 1

    if missing_base_fields:
        add_episode_issue(
            episode,
            "ERROR",
            "TIMING_BASE_FIELDS",
            f"state/command/cameras object missing in {missing_base_fields} frames",
            "timing",
        )
    if command_mismatches:
        add_episode_issue(
            episode,
            "ERROR",
            "COMMAND_OBSERVATION_MISMATCH",
            f"command observation_sequence mismatches in {command_mismatches} frames",
            "command",
        )
    if any(
        current <= previous
        for previous, current in zip(observation_sequences, observation_sequences[1:], strict=False)
    ):
        add_episode_issue(
            episode,
            "ERROR",
            "OBSERVATION_SEQUENCE_ORDER",
            "observation_sequence is not strictly increasing",
            "timing",
        )
    state_reuse = sum(
        current == previous for previous, current in zip(state_sequences, state_sequences[1:], strict=False)
    )
    state_backward = sum(
        current < previous for previous, current in zip(state_sequences, state_sequences[1:], strict=False)
    )
    if state_reuse:
        add_episode_issue(
            episode,
            "WARNING",
            "STATE_SEQUENCE_REUSED",
            f"robot state packet sequence was reused {state_reuse} times",
            "state",
        )
    if state_backward:
        add_episode_issue(
            episode,
            "ERROR",
            "STATE_SEQUENCE_BACKWARD",
            f"robot state packet sequence moved backward {state_backward} times",
            "state",
        )

    for camera_name in expected_cameras:
        skews = camera_values[camera_name]["skew"]
        ages = camera_values[camera_name]["age"]
        sequences = camera_values[camera_name]["sequence"]
        gaps = camera_values[camera_name]["gap"]
        reused_flags = camera_values[camera_name]["reused_flag"]
        assert isinstance(skews, list)
        assert isinstance(ages, list)
        assert isinstance(sequences, list)
        assert isinstance(gaps, list)
        assert isinstance(reused_flags, list)
        if camera_field_errors[camera_name]:
            add_episode_issue(
                episode,
                "ERROR",
                "CAMERA_TIMING_FIELDS",
                f"required timing fields missing in {camera_field_errors[camera_name]} frames",
                camera_name,
            )
        if skew_recompute_errors[camera_name]:
            add_episode_issue(
                episode,
                "ERROR",
                "CAMERA_SKEW_INCONSISTENT",
                f"recorded skew/delta disagrees with receive clocks in "
                f"{skew_recompute_errors[camera_name]} frames",
                camera_name,
            )
        equal_sequences = sum(
            current == previous for previous, current in zip(sequences, sequences[1:], strict=False)
        )
        backward_sequences = sum(
            current < previous for previous, current in zip(sequences, sequences[1:], strict=False)
        )
        reused = max(equal_sequences, sum(int(value) for value in reused_flags))
        summary = CameraTimingSummary(
            frames=len(skews),
            skew_min_ms=min(skews) if skews else None,
            skew_p50_ms=percentile(skews, 0.50),
            skew_p95_ms=percentile(skews, 0.95),
            skew_max_ms=max(skews) if skews else None,
            age_max_ms=max(ages) if ages else None,
            over_warning_skew=sum(value > config.warn_camera_skew_ms for value in skews),
            over_max_skew=sum(value > config.max_camera_skew_ms for value in skews),
            reused_frames=reused,
            backward_sequences=backward_sequences,
            reported_sequence_gaps=sum(int(value) for value in gaps),
        )
        episode.cameras[camera_name] = summary
        if summary.over_max_skew:
            add_episode_issue(
                episode,
                "ERROR",
                "CAMERA_SKEW_TOO_LARGE",
                f"{summary.over_max_skew} frames exceed {config.max_camera_skew_ms:.3f}ms; "
                f"max={summary.skew_max_ms:.3f}ms",
                camera_name,
            )
        elif summary.over_warning_skew:
            add_episode_issue(
                episode,
                "WARNING",
                "CAMERA_SKEW_ELEVATED",
                f"{summary.over_warning_skew} frames exceed warning threshold "
                f"{config.warn_camera_skew_ms:.3f}ms; max={summary.skew_max_ms:.3f}ms",
                camera_name,
            )
        if summary.age_max_ms is not None and summary.age_max_ms > config.max_camera_age_ms:
            add_episode_issue(
                episode,
                "ERROR",
                "CAMERA_AGE_TOO_LARGE",
                f"max camera age {summary.age_max_ms:.3f}ms exceeds {config.max_camera_age_ms:.3f}ms",
                camera_name,
            )
        elif summary.age_max_ms is not None and summary.age_max_ms > config.warn_camera_age_ms:
            add_episode_issue(
                episode,
                "WARNING",
                "CAMERA_AGE_ELEVATED",
                f"max camera age {summary.age_max_ms:.3f}ms exceeds warning threshold "
                f"{config.warn_camera_age_ms:.3f}ms",
                camera_name,
            )
        if reused:
            add_episode_issue(
                episode,
                "WARNING",
                "CAMERA_FRAME_REUSED",
                f"same camera sequence reused {reused} times",
                camera_name,
            )
        if backward_sequences:
            add_episode_issue(
                episode,
                "ERROR",
                "CAMERA_SEQUENCE_BACKWARD",
                f"camera sequence moved backward {backward_sequences} times",
                camera_name,
            )
        if summary.reported_sequence_gaps:
            add_episode_issue(
                episode,
                "WARNING",
                "CAMERA_SEQUENCE_GAPS",
                f"publisher reported {summary.reported_sequence_gaps} dropped sequence slots",
                camera_name,
            )
    validate_source_timing(episode, rows, config)


def check_dataset(root: Path, config: CheckConfig) -> DatasetReport:
    root = root.expanduser().resolve()
    report = DatasetReport(path=str(root))
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        add_dataset_issue(report, "ERROR", "INFO_MISSING", f"missing {info_path}")
        report.finish()
        return report
    try:
        info = read_json_file(info_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        add_dataset_issue(report, "ERROR", "INFO_INVALID", f"cannot read {info_path}: {exc}")
        report.finish()
        return report

    total_episodes = as_int(info.get("total_episodes"))
    total_frames = as_int(info.get("total_frames"))
    fps = float(info["fps"]) if is_number(info.get("fps")) else None
    report.total_episodes_declared = total_episodes
    report.total_frames_declared = total_frames
    report.fps = fps
    if total_episodes is None or total_episodes < 0:
        add_dataset_issue(report, "ERROR", "INFO_EPISODES", "total_episodes is invalid", str(info_path))
        total_episodes = 0
    if total_frames is None or total_frames < 0:
        add_dataset_issue(report, "ERROR", "INFO_FRAMES", "total_frames is invalid", str(info_path))
        total_frames = 0
    if fps is None or fps <= 0:
        add_dataset_issue(report, "ERROR", "INFO_FPS", "fps is not a positive finite number", str(info_path))

    for required_meta in ("tasks.parquet", "stats.json"):
        path = root / "meta" / required_meta
        if not path.is_file():
            add_dataset_issue(report, "ERROR", "META_FILE_MISSING", f"missing {path}", "meta")
    stats_path = root / "meta" / "stats.json"
    if stats_path.is_file():
        try:
            read_json_file(stats_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            add_dataset_issue(report, "ERROR", "STATS_INVALID", f"cannot read {stats_path}: {exc}")

    features = info.get("features") if isinstance(info.get("features"), dict) else {}
    video_keys = sorted(
        key for key, value in features.items() if isinstance(value, dict) and value.get("dtype") == "video"
    )
    camera_names = [key.rsplit(".", 1)[-1] for key in video_keys]
    video_columns: list[str] = []
    for video_key in video_keys:
        prefix = f"videos/{video_key}"
        video_columns.extend(
            [
                f"{prefix}/chunk_index",
                f"{prefix}/file_index",
                f"{prefix}/from_timestamp",
                f"{prefix}/to_timestamp",
            ]
        )

    episode_files = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    if not episode_files:
        add_dataset_issue(report, "ERROR", "EPISODE_META_MISSING", "no episode metadata parquet files")
        episode_rows: list[dict] = []
        episode_columns: set[str] = set()
    else:
        try:
            episode_rows, episode_columns = read_parquet_rows(
                episode_files, list(BASE_EPISODE_COLUMNS) + video_columns
            )
        except Exception as exc:
            add_dataset_issue(report, "ERROR", "EPISODE_META_READ", str(exc), "meta/episodes")
            episode_rows = []
            episode_columns = set()

    missing_episode_columns = set(BASE_EPISODE_COLUMNS) - episode_columns
    for column in sorted(missing_episode_columns):
        add_dataset_issue(
            report,
            "ERROR",
            "EPISODE_META_COLUMN",
            f"episode metadata is missing column {column}",
            "meta/episodes",
        )
    episode_row_map: dict[int, dict] = {}
    for row in episode_rows:
        episode_index = as_int(row.get("episode_index"))
        if episode_index is None:
            add_dataset_issue(report, "ERROR", "EPISODE_INDEX_INVALID", "invalid episode_index")
            continue
        if episode_index in episode_row_map:
            add_dataset_issue(
                report,
                "ERROR",
                "EPISODE_INDEX_DUPLICATE",
                f"duplicate episode metadata for episode {episode_index}",
            )
            continue
        episode_row_map[episode_index] = row

    expected_episode_ids = set(range(total_episodes))
    actual_episode_ids = set(episode_row_map)
    if actual_episode_ids != expected_episode_ids:
        add_dataset_issue(
            report,
            "ERROR",
            "EPISODE_SET_MISMATCH",
            f"info expects {sorted(expected_episode_ids)}, metadata has {sorted(actual_episode_ids)}",
            "meta/episodes",
        )
    if len(episode_rows) != total_episodes:
        add_dataset_issue(
            report,
            "ERROR",
            "EPISODE_COUNT_MISMATCH",
            f"info total_episodes={total_episodes}, episode metadata rows={len(episode_rows)}",
        )

    episode_reports: dict[int, EpisodeReport] = {}
    for episode_index in range(total_episodes):
        row = episode_row_map.get(episode_index)
        length = as_int(row.get("length")) if row is not None else None
        episode_reports[episode_index] = EpisodeReport(episode_index=episode_index, length=length)
    report.episodes = [episode_reports[index] for index in sorted(episode_reports)]

    expected_from = 0
    meta_length_sum = 0
    data_template = info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    for episode_index, row in sorted(episode_row_map.items()):
        episode = episode_reports.get(episode_index)
        if episode is None:
            continue
        length = as_int(row.get("length"))
        dataset_from = as_int(row.get("dataset_from_index"))
        dataset_to = as_int(row.get("dataset_to_index"))
        if length is None or length <= 0:
            add_episode_issue(episode, "ERROR", "EPISODE_LENGTH", f"invalid length={row.get('length')}")
        else:
            meta_length_sum += length
        if dataset_from is None or dataset_to is None or dataset_to < dataset_from:
            add_episode_issue(
                episode,
                "ERROR",
                "EPISODE_DATASET_RANGE",
                f"invalid dataset range [{dataset_from}, {dataset_to})",
            )
        else:
            if dataset_from != expected_from:
                add_episode_issue(
                    episode,
                    "ERROR",
                    "EPISODE_DATASET_CONTIGUITY",
                    f"dataset_from_index={dataset_from}, expected {expected_from}",
                )
            if length is not None and dataset_to - dataset_from != length:
                add_episode_issue(
                    episode,
                    "ERROR",
                    "EPISODE_RANGE_LENGTH",
                    f"range length={dataset_to - dataset_from}, metadata length={length}",
                )
            expected_from = dataset_to
        chunk_index = as_int(row.get("data/chunk_index"))
        file_index = as_int(row.get("data/file_index"))
        if chunk_index is None or file_index is None:
            add_episode_issue(episode, "ERROR", "DATA_REFERENCE_MISSING", "data parquet reference is null")
        else:
            try:
                referenced_data = root / str(data_template).format(
                    chunk_index=chunk_index, file_index=file_index
                )
            except (KeyError, ValueError) as exc:
                add_episode_issue(episode, "ERROR", "DATA_TEMPLATE", f"cannot format data_path: {exc}")
            else:
                if not referenced_data.is_file():
                    add_episode_issue(
                        episode,
                        "ERROR",
                        "DATA_FILE_MISSING",
                        f"referenced parquet does not exist: {referenced_data}",
                    )
    if meta_length_sum != total_frames:
        add_dataset_issue(
            report,
            "ERROR",
            "META_FRAME_SUM",
            f"sum of episode lengths={meta_length_sum}, info total_frames={total_frames}",
        )

    data_files = sorted((root / "data").glob("*/*.parquet"))
    data_rows: list[dict] = []
    data_columns: set[str] = set()
    if not data_files:
        add_dataset_issue(report, "ERROR", "DATA_PARQUET_MISSING", "no data parquet files")
    else:
        try:
            data_rows, data_columns = read_parquet_rows(data_files, REQUIRED_DATA_COLUMNS)
        except Exception as exc:
            add_dataset_issue(report, "ERROR", "DATA_PARQUET_READ", str(exc), "data")
    for column in sorted(set(REQUIRED_DATA_COLUMNS) - data_columns):
        add_dataset_issue(
            report,
            "ERROR",
            "DATA_COLUMN_MISSING",
            f"data parquet is missing column {column}",
            "data",
        )
    if len(data_rows) != total_frames:
        add_dataset_issue(
            report,
            "ERROR",
            "DATA_FRAME_COUNT",
            f"data parquet rows={len(data_rows)}, info total_frames={total_frames}",
        )

    task_ids: set[int] = set()
    tasks_path = root / "meta" / "tasks.parquet"
    if tasks_path.is_file():
        try:
            task_columns = set(pq.read_schema(tasks_path).names)
            if "task_index" not in task_columns:
                raise ValueError("tasks parquet is missing task_index")
            task_table = pq.read_table(tasks_path, columns=["task_index"])
            task_ids = {
                value
                for value in (as_int(row.get("task_index")) for row in task_table.to_pylist())
                if value is not None
            }
        except Exception as exc:
            add_dataset_issue(report, "ERROR", "TASKS_READ", str(exc), str(tasks_path))

    rows_by_episode: dict[int, list[dict]] = defaultdict(list)
    global_indices: list[int] = []
    for row in data_rows:
        episode_index = as_int(row.get("episode_index"))
        global_index = as_int(row.get("index"))
        if episode_index is not None:
            rows_by_episode[episode_index].append(row)
        if global_index is not None:
            global_indices.append(global_index)
        task_index = as_int(row.get("task_index"))
        if task_ids and task_index not in task_ids:
            add_dataset_issue(
                report,
                "ERROR",
                "TASK_INDEX_UNKNOWN",
                f"data contains unknown task_index={task_index}",
                "data",
            )
    if sorted(global_indices) != list(range(len(data_rows))):
        add_dataset_issue(
            report,
            "ERROR",
            "GLOBAL_INDEX_CONTIGUITY",
            "data index is not a unique contiguous range from zero",
            "data",
        )
    extra_data_episodes = sorted(set(rows_by_episode) - expected_episode_ids)
    if extra_data_episodes:
        add_dataset_issue(
            report,
            "ERROR",
            "ORPHAN_DATA_EPISODES",
            f"data parquet contains episodes absent from info/meta: {extra_data_episodes}",
            "data",
        )

    for episode_index, episode in episode_reports.items():
        rows = rows_by_episode.get(episode_index, [])
        rows.sort(key=lambda row: as_int(row.get("index")) if as_int(row.get("index")) is not None else -1)
        episode.data_frames = len(rows)
        if episode.length is not None and len(rows) != episode.length:
            add_episode_issue(
                episode,
                "ERROR",
                "EPISODE_DATA_COUNT",
                f"data rows={len(rows)}, metadata length={episode.length}",
                "data",
            )
        actual_frame_indices = [as_int(row.get("frame_index")) for row in rows]
        if actual_frame_indices != list(range(len(rows))):
            add_episode_issue(
                episode,
                "ERROR",
                "FRAME_INDEX_CONTIGUITY",
                "frame_index is not exactly contiguous from zero",
                "data",
            )
        meta_row = episode_row_map.get(episode_index)
        dataset_from = as_int(meta_row.get("dataset_from_index")) if meta_row else None
        if dataset_from is not None:
            actual_indices = [as_int(row.get("index")) for row in rows]
            if actual_indices != list(range(dataset_from, dataset_from + len(rows))):
                add_episode_issue(
                    episode,
                    "ERROR",
                    "EPISODE_GLOBAL_INDEX",
                    f"global indices do not match metadata start {dataset_from}",
                    "data",
                )
        timestamps = [row.get("timestamp") for row in rows]
        if any(not is_number(value) for value in timestamps):
            add_episode_issue(
                episode, "ERROR", "TIMESTAMP_INVALID", "timestamp contains non-finite values", "data"
            )
        elif fps is not None and rows:
            numeric_timestamps = [float(value) for value in timestamps]
            if any(
                current <= previous
                for previous, current in zip(numeric_timestamps, numeric_timestamps[1:], strict=False)
            ):
                add_episode_issue(
                    episode,
                    "ERROR",
                    "TIMESTAMP_ORDER",
                    "timestamps are not strictly increasing",
                    "data",
                )
            tolerance = max(0.02, 0.4 / fps)
            max_error = max(
                abs(timestamp - frame_index / fps) for frame_index, timestamp in enumerate(numeric_timestamps)
            )
            if max_error > tolerance:
                add_episode_issue(
                    episode,
                    "ERROR",
                    "TIMESTAMP_GRID",
                    f"timestamps differ from frame_index/fps by up to {max_error:.6f}s",
                    "data",
                )

    video_template = info.get(
        "video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    video_references: dict[Path, list[tuple[int, str, float, float, int]]] = defaultdict(list)
    video_probe_cache: dict[Path, VideoProbe] = {}
    duration_tolerance = max(0.05, 1.25 / fps) if fps else 0.1

    for episode_index, episode in episode_reports.items():
        row = episode_row_map.get(episode_index)
        if row is None:
            add_episode_issue(episode, "ERROR", "EPISODE_META_MISSING", "episode metadata row is missing")
            continue
        for video_key, camera_name in zip(video_keys, camera_names, strict=True):
            prefix = f"videos/{video_key}"
            chunk_index = as_int(row.get(f"{prefix}/chunk_index"))
            file_index = as_int(row.get(f"{prefix}/file_index"))
            from_timestamp = row.get(f"{prefix}/from_timestamp")
            to_timestamp = row.get(f"{prefix}/to_timestamp")
            if (
                chunk_index is None
                or file_index is None
                or not is_number(from_timestamp)
                or not is_number(to_timestamp)
            ):
                add_episode_issue(
                    episode,
                    "ERROR",
                    "VIDEO_REFERENCE_MISSING",
                    "video metadata reference is null or incomplete",
                    camera_name,
                )
                continue
            start_s = float(from_timestamp)
            end_s = float(to_timestamp)
            if start_s < 0 or end_s <= start_s:
                add_episode_issue(
                    episode,
                    "ERROR",
                    "VIDEO_TIMESTAMP_RANGE",
                    f"invalid video interval [{start_s}, {end_s})",
                    camera_name,
                )
                continue
            try:
                relative_path = str(video_template).format(
                    video_key=video_key,
                    chunk_index=chunk_index,
                    file_index=file_index,
                )
            except (KeyError, ValueError) as exc:
                add_episode_issue(
                    episode,
                    "ERROR",
                    "VIDEO_TEMPLATE",
                    f"cannot format video_path: {exc}",
                    camera_name,
                )
                continue
            video_path = root / relative_path
            episode.video_paths[camera_name] = str(video_path)
            if not video_path.is_file():
                add_episode_issue(
                    episode,
                    "ERROR",
                    "VIDEO_FILE_MISSING",
                    f"referenced video does not exist: {video_path}",
                    camera_name,
                )
                continue
            length = episode.length or 0
            video_references[video_path].append((episode_index, camera_name, start_s, end_s, length))
            if fps is not None and length:
                expected_span = length / fps
                if abs((end_s - start_s) - expected_span) > duration_tolerance:
                    add_episode_issue(
                        episode,
                        "ERROR",
                        "VIDEO_SEGMENT_LENGTH",
                        f"video interval={end_s - start_s:.6f}s, expected {expected_span:.6f}s",
                        camera_name,
                    )
            if video_path not in video_probe_cache:
                video_probe_cache[video_path] = probe_video(
                    video_path,
                    deep_decode=config.deep_video,
                    timeout_s=config.video_timeout_s,
                )
            probe = video_probe_cache[video_path]
            report.videos[str(video_path)] = probe
            if not probe.ok:
                add_episode_issue(
                    episode,
                    "ERROR",
                    "VIDEO_PROBE_FAILED",
                    probe.error or "video probe/decode failed",
                    camera_name,
                )
                continue
            feature = features.get(video_key, {})
            video_info = feature.get("info", {}) if isinstance(feature, dict) else {}
            expected_width = as_int(video_info.get("video.width"))
            expected_height = as_int(video_info.get("video.height"))
            expected_codec = video_info.get("video.codec")
            expected_fps = float(video_info["video.fps"]) if is_number(video_info.get("video.fps")) else fps
            if expected_width is not None and probe.width != expected_width:
                add_episode_issue(
                    episode,
                    "ERROR",
                    "VIDEO_WIDTH",
                    f"video width={probe.width}, expected {expected_width}",
                    camera_name,
                )
            if expected_height is not None and probe.height != expected_height:
                add_episode_issue(
                    episode,
                    "ERROR",
                    "VIDEO_HEIGHT",
                    f"video height={probe.height}, expected {expected_height}",
                    camera_name,
                )
            if expected_codec and probe.codec and probe.codec != expected_codec:
                add_episode_issue(
                    episode,
                    "ERROR",
                    "VIDEO_CODEC",
                    f"video codec={probe.codec}, expected {expected_codec}",
                    camera_name,
                )
            if expected_fps and probe.fps and abs(probe.fps - expected_fps) > 0.01:
                add_episode_issue(
                    episode,
                    "ERROR",
                    "VIDEO_FPS",
                    f"video fps={probe.fps:.6f}, expected {expected_fps:.6f}",
                    camera_name,
                )
            if probe.duration_s is not None and end_s > probe.duration_s + duration_tolerance:
                add_episode_issue(
                    episode,
                    "ERROR",
                    "VIDEO_SEGMENT_TRUNCATED",
                    f"episode needs video through {end_s:.3f}s, file duration={probe.duration_s:.3f}s",
                    camera_name,
                )
            if probe.frame_count is not None and expected_fps:
                required_frames = math.ceil(end_s * expected_fps - 1e-6)
                if probe.frame_count < required_frames:
                    add_episode_issue(
                        episode,
                        "ERROR",
                        "VIDEO_FRAMES_TRUNCATED",
                        f"episode needs at least {required_frames} frames in file, found {probe.frame_count}",
                        camera_name,
                    )

    for video_path, references in video_references.items():
        probe = video_probe_cache.get(video_path)
        if probe is None or not probe.ok:
            continue
        references.sort(key=lambda item: item[2])
        for previous, current in zip(references, references[1:], strict=False):
            if current[2] < previous[3] - duration_tolerance:
                add_dataset_issue(
                    report,
                    "ERROR",
                    "VIDEO_REFERENCE_OVERLAP",
                    f"overlapping episode intervals in {video_path}: {previous[0]} and {current[0]}",
                    current[1],
                )
            elif current[2] > previous[3] + duration_tolerance:
                add_dataset_issue(
                    report,
                    "WARNING",
                    "VIDEO_REFERENCE_GAP",
                    f"unreferenced time gap in {video_path}: {previous[3]:.3f}s to {current[2]:.3f}s",
                    current[1],
                )
        if probe.frame_count is not None and probe.fps:
            required_end = max(item[3] for item in references)
            required_frames = math.ceil(required_end * probe.fps - 1e-6)
            if probe.frame_count > required_frames + 1:
                add_dataset_issue(
                    report,
                    "WARNING",
                    "VIDEO_UNREFERENCED_TAIL",
                    f"{video_path} has {probe.frame_count} frames but metadata references only "
                    f"the first {required_frames}",
                    references[0][1],
                )

    referenced_video_paths = set(video_references)
    for video_path in sorted((root / "videos").glob("**/*.mp4")):
        if video_path not in referenced_video_paths:
            add_dataset_issue(
                report,
                "WARNING",
                "VIDEO_FILE_UNREFERENCED",
                f"video file is not referenced by any committed episode: {video_path}",
                "videos",
            )
    for video_path in sorted(root.glob("**/*.mp4")):
        try:
            video_path.relative_to(root / "videos")
        except ValueError:
            add_dataset_issue(
                report,
                "WARNING",
                "VIDEO_TEMP_FRAGMENT",
                f"mp4 outside the formal videos tree: {video_path}",
                "videos",
            )

    timing_records, timing_issues = load_timing_files(root)
    formal_episode_ids = set(episode_reports)
    orphan_episode_ids = sorted(set(timing_records) - formal_episode_ids)
    for episode_index in orphan_episode_ids:
        summary = summarize_orphan_timing(episode_index, timing_records[episode_index])
        summary["issues"] = [asdict(issue) for issue in timing_issues.get(episode_index, [])]
        report.orphan_timing.append(summary)
        add_dataset_issue(
            report,
            "WARNING",
            "UNCOMMITTED_TIMING_FRAGMENT",
            f"timing episode {episode_index} has {len(timing_records[episode_index])} rows but is "
            "absent from committed metadata/data",
            "meta/timing",
        )

    for episode_index, episode in episode_reports.items():
        rows = timing_records.get(episode_index)
        for issue in timing_issues.get(episode_index, []):
            episode.issues.append(issue)
        if rows is None:
            if config.require_timing:
                add_episode_issue(
                    episode,
                    "ERROR",
                    "TIMING_FILE_MISSING",
                    "committed episode has no timing sidecar",
                    "meta/timing",
                )
            continue
        if episode.length is not None and len(rows) != episode.length:
            add_episode_issue(
                episode,
                "ERROR",
                "TIMING_FRAME_COUNT",
                f"timing rows={len(rows)}, metadata length={episode.length}",
                "meta/timing",
            )
        validate_timing_rows(episode, rows, camera_names, config)

    report.finish()
    return report


def format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def print_report(report: DatasetReport, *, verbose: bool) -> None:
    usable = report.usable_episode_count
    total = len(report.episodes)
    warning_episodes = sum(episode.status == "WARN" for episode in report.episodes)
    failed_episodes = sum(episode.status == "FAIL" for episode in report.episodes)
    print(f"\n=== {report.path} ===")
    print(
        f"dataset={report.status} committed={total} usable={usable} "
        f"warn={warning_episodes} unusable={failed_episodes} "
        f"frames={report.total_frames_declared} fps={format_float(report.fps)} "
        f"orphan_timing={len(report.orphan_timing)}"
    )
    for episode in report.episodes:
        camera_bits = []
        for camera_name, summary in sorted(episode.cameras.items()):
            camera_bits.append(
                f"{camera_name}:max_skew={format_float(summary.skew_max_ms)}ms"
                f"/reuse={summary.reused_frames}/back={summary.backward_sequences}"
            )
        camera_text = " ".join(camera_bits)
        print(
            f"  ep {episode.episode_index:03d} {episode.status:<4} "
            f"data={episode.data_frames}/{episode.length} timing={episode.timing_frames}/{episode.length} "
            f"{camera_text}".rstrip()
        )
        should_print_issues = episode.status == "FAIL" or verbose
        if should_print_issues:
            for issue in episode.issues:
                location = f" [{issue.component}]" if issue.component else ""
                print(f"      {issue.severity:<7} {issue.code}{location}: {issue.message}")
    if report.orphan_timing:
        print("  uncommitted timing fragments:")
        for orphan in report.orphan_timing:
            print(
                f"      ep {orphan['episode_index']:03d}: rows={orphan['rows']} "
                f"frames={orphan['first_frame']}..{orphan['last_frame']} "
                f"max_skew_ms={orphan['camera_max_skew_ms']}"
            )
    if report.issues:
        print("  dataset issues:")
        for issue in report.issues:
            if (
                issue.severity == "ERROR"
                or verbose
                or issue.code
                in {
                    "UNCOMMITTED_TIMING_FRAGMENT",
                    "VIDEO_TEMP_FRAGMENT",
                    "VIDEO_UNREFERENCED_TAIL",
                }
            ):
                location = f" [{issue.component}]" if issue.component else ""
                print(f"      {issue.severity:<7} {issue.code}{location}: {issue.message}")


def report_to_dict(report: DatasetReport) -> dict[str, Any]:
    return asdict(report)


def discover_default_datasets() -> list[Path]:
    data_root = Path(__file__).resolve().parent
    return sorted(path.parent.parent for path in data_root.glob("*/meta/info.json"))


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative finite number")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only per-episode integrity check for LeRobot datasets."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        type=Path,
        help="dataset roots; when omitted, scan data/*/meta/info.json",
    )
    parser.add_argument("--warn-camera-skew-ms", type=nonnegative_float, default=50.0)
    parser.add_argument("--max-camera-skew-ms", type=positive_float, default=100.0)
    parser.add_argument("--warn-camera-age-ms", type=nonnegative_float, default=100.0)
    parser.add_argument("--max-camera-age-ms", type=positive_float, default=1000.0)
    parser.add_argument("--max-source-skew-ms", type=positive_float, default=20.0)
    parser.add_argument("--max-source-age-ms", type=positive_float, default=50.0)
    parser.add_argument(
        "--no-require-timing",
        action="store_true",
        help="do not fail a committed episode solely because its timing sidecar is absent",
    )
    parser.add_argument(
        "--no-require-source-timing",
        action="store_true",
        help="do not fail timing records that lack state.source_timing",
    )
    parser.add_argument(
        "--deep-video",
        action="store_true",
        help="fully decode every referenced video with ffmpeg in addition to ffprobe frame counting",
    )
    parser.add_argument(
        "--video-timeout-s",
        type=positive_float,
        default=600.0,
        help="timeout for each ffprobe/ffmpeg process",
    )
    parser.add_argument("--verbose", action="store_true", help="print warning details for usable episodes")
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="return exit status 1 when any dataset is WARN",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        help="optional path for a machine-readable JSON report; datasets remain untouched",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.warn_camera_skew_ms > args.max_camera_skew_ms:
        parser.error("--warn-camera-skew-ms cannot exceed --max-camera-skew-ms")
    if args.warn_camera_age_ms > args.max_camera_age_ms:
        parser.error("--warn-camera-age-ms cannot exceed --max-camera-age-ms")
    datasets = args.datasets or discover_default_datasets()
    if not datasets:
        parser.error("no datasets were supplied or discovered")
    config = CheckConfig(
        warn_camera_skew_ms=args.warn_camera_skew_ms,
        max_camera_skew_ms=args.max_camera_skew_ms,
        warn_camera_age_ms=args.warn_camera_age_ms,
        max_camera_age_ms=args.max_camera_age_ms,
        max_source_skew_ms=args.max_source_skew_ms,
        max_source_age_ms=args.max_source_age_ms,
        require_timing=not args.no_require_timing,
        require_source_timing=not args.no_require_source_timing,
        deep_video=args.deep_video,
        video_timeout_s=args.video_timeout_s,
    )
    reports = [check_dataset(path, config) for path in datasets]
    for report in reports:
        print_report(report, verbose=args.verbose)
    passed = sum(report.status == "PASS" for report in reports)
    warned = sum(report.status == "WARN" for report in reports)
    failed = sum(report.status == "FAIL" for report in reports)
    print(
        f"\nSUMMARY datasets={len(reports)} pass={passed} warn={warned} fail={failed} "
        f"usable_episodes={sum(report.usable_episode_count for report in reports)}/"
        f"{sum(len(report.episodes) for report in reports)}"
    )
    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(
            json.dumps(
                {
                    "summary": {
                        "datasets": len(reports),
                        "pass": passed,
                        "warn": warned,
                        "fail": failed,
                        "usable_episodes": sum(report.usable_episode_count for report in reports),
                        "committed_episodes": sum(len(report.episodes) for report in reports),
                    },
                    "reports": [report_to_dict(report) for report in reports],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    if failed or (args.fail_on_warning and warned):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
