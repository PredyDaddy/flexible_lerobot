#!/usr/bin/env python

"""Offline validator for the JZ Pin raw18-to-training16 projection.

This checker reads metadata, numeric statistics, and all Parquet numeric rows. It
does not decode video, open a robot connection, or modify the source dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from argparse import Namespace
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lerobot.robots.jz_robot_pin_timed.training_schema import (
    CANONICAL_CLOSED,
    CANONICAL_OPEN,
    RAW_DIM,
    RAW_SCHEMA_ID,
    TRAINING_DIM,
    TRAINING_SCHEMA_FILENAME,
    TRAINING_SCHEMA_ID,
    TRAINING_SCHEMA_VERSION,
    JZPinTrainingSchema,
    JZPinTrainingSchemaError,
)
from lerobot.robots.jz_robot_udp.protocol import validate_source_timing

FEATURE_KEYS = ("observation.state", "action")
FRAME_KEYS = ("episode_index", "frame_index")
REQUIRED_VECTOR_STATISTICS = ("min", "max", "mean", "std")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate every numeric row of the named JZ Pin raw18-to-training16 projection "
            "without decoding video "
            "or contacting the robot"
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(f"Projection manifest. Defaults to <dataset-root>/meta/{TRAINING_SCHEMA_FILENAME}."),
    )
    parser.add_argument(
        "--allow-unavailable",
        action="store_true",
        help=(
            "Audit a dataset whose gripper observation source is explicitly unavailable. "
            "The result is AUDIT, never PASS, and is not training approval."
        ),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional output path. The dataset is not written when this is omitted.",
    )
    return parser.parse_args()


def _add_error(report: dict[str, Any], message: str) -> None:
    report["errors"].append(message)
    report["status"] = "FAIL"


def _load_json(path: Path, description: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return value


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(np.asarray(value).shape)


def _array_digest(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _raw_to_canonical(value: float, specification: Mapping[str, Any]) -> float:
    raw_closed = float(specification["raw_closed"])
    raw_open = float(specification["raw_open"])
    return CANONICAL_CLOSED + (value - raw_closed) * (CANONICAL_OPEN - CANONICAL_CLOSED) / (
        raw_open - raw_closed
    )


def _build_schema_report(schema: JZPinTrainingSchema) -> dict[str, Any]:
    manifest = schema.to_dict()
    mapping: dict[str, list[dict[str, Any]]] = {}
    dropped_force_fields: dict[str, list[dict[str, Any]]] = {}
    for feature_key in FEATURE_KEYS:
        raw_names = schema.raw_feature_names[feature_key]
        training_names = schema.training_feature_names[feature_key]
        keep_indices = schema.keep_indices[feature_key]
        mapping[feature_key] = [
            {
                "training_index": training_index,
                "training_name": training_name,
                "raw_index": raw_index,
                "raw_name": raw_names[raw_index],
            }
            for training_index, (training_name, raw_index) in enumerate(
                zip(training_names, keep_indices, strict=True)
            )
        ]
        dropped_force_fields[feature_key] = [
            {"raw_index": raw_index, "raw_name": raw_names[raw_index]}
            for raw_index in schema.dropped_force_indices[feature_key]
        ]

    grippers = {}
    for side in ("left", "right"):
        gripper = manifest["grippers"][side]
        grippers[side] = {
            "observation_source": gripper["observation"]["source"],
            "observation_direction": {
                "raw_closed": gripper["observation"]["raw_closed"],
                "raw_open": gripper["observation"]["raw_open"],
                "raw_increases_toward": (
                    "open"
                    if gripper["observation"]["raw_open"] > gripper["observation"]["raw_closed"]
                    else "closed"
                ),
            },
            "action_source": gripper["action"]["source"],
            "action_direction": {
                "raw_closed": gripper["action"]["raw_closed"],
                "raw_open": gripper["action"]["raw_open"],
                "raw_increases_toward": (
                    "open" if gripper["action"]["raw_open"] > gripper["action"]["raw_closed"] else "closed"
                ),
            },
            "wire_force": dict(gripper["wire_force"]),
        }

    return {
        "format": manifest["format"],
        "schema_version": manifest["schema_version"],
        "raw": {"id": manifest["raw_schema"]["id"], "dimension": RAW_DIM},
        "training": {
            "id": manifest["training_schema"]["id"],
            "dimension": TRAINING_DIM,
            "canonical_opening": dict(manifest["training_schema"]["canonical_opening"]),
        },
        "feature_mapping": mapping,
        "dropped_force_fields": dropped_force_fields,
        "grippers": grippers,
        "provenance": manifest.get("provenance", {}),
    }


def _validate_schema_constants(schema_report: Mapping[str, Any]) -> None:
    if schema_report["schema_version"] != TRAINING_SCHEMA_VERSION:
        raise JZPinTrainingSchemaError(
            f"schema_version must be {TRAINING_SCHEMA_VERSION}, got {schema_report['schema_version']!r}"
        )
    if schema_report["raw"] != {"id": RAW_SCHEMA_ID, "dimension": RAW_DIM}:
        raise JZPinTrainingSchemaError("raw schema identity/dimension mismatch")
    training = schema_report["training"]
    if training["id"] != TRAINING_SCHEMA_ID or training["dimension"] != TRAINING_DIM:
        raise JZPinTrainingSchemaError("training schema identity/dimension mismatch")
    canonical = training["canonical_opening"]
    if canonical.get("closed") != CANONICAL_CLOSED or canonical.get("open") != CANONICAL_OPEN:
        raise JZPinTrainingSchemaError("canonical opening must be 0=closed,100=open")


def _validate_force_exclusion(schema: JZPinTrainingSchema, *, allow_unavailable: bool) -> dict[str, Any]:
    """Use unique sentinels to prove that neither force slot reaches model16."""

    manifest = schema.to_dict()
    result: dict[str, Any] = {}
    for feature_key in FEATURE_KEYS:
        keep_indices = schema.keep_indices[feature_key]
        dropped_indices = schema.dropped_force_indices[feature_key]
        if keep_indices != (*range(15), 16):
            raise JZPinTrainingSchemaError(
                f"{feature_key} keep indices must be [0..14,16], got {list(keep_indices)}"
            )
        if dropped_indices != (15, 17):
            raise JZPinTrainingSchemaError(
                f"{feature_key} force indices must be [15,17], got {list(dropped_indices)}"
            )
        if set(keep_indices) & set(dropped_indices):
            raise JZPinTrainingSchemaError(f"{feature_key} force fields overlap model inputs")

        probe = np.arange(RAW_DIM, dtype=np.float64)
        force_sentinels = (1_000_015.25, -1_000_017.75)
        probe[15], probe[17] = force_sentinels
        probe_before = probe.copy()
        if feature_key == "observation.state":
            projected = schema.project_observation(
                probe,
                require_available_source=not allow_unavailable,
            )
            modality = "observation"
        else:
            projected = schema.project_action(probe)
            modality = "action"
        projected = np.asarray(projected)
        if projected.shape != (TRAINING_DIM,):
            raise JZPinTrainingSchemaError(
                f"{feature_key} sentinel projection must be [{TRAINING_DIM}], got {list(projected.shape)}"
            )
        if not np.array_equal(probe, probe_before):
            raise JZPinTrainingSchemaError(f"{feature_key} sentinel projection mutated raw18 input")
        if any(bool(np.any(projected == sentinel)) for sentinel in force_sentinels):
            raise JZPinTrainingSchemaError(f"{feature_key} force sentinel entered model16")
        np.testing.assert_array_equal(projected[:14], probe[:14])
        for output_index, side in ((14, "left"), (15, "right")):
            raw_index = keep_indices[output_index]
            expected = _raw_to_canonical(float(probe[raw_index]), manifest["grippers"][side][modality])
            if not np.isclose(projected[output_index], expected, rtol=0.0, atol=1e-12):
                raise JZPinTrainingSchemaError(
                    f"{feature_key} {side} opening projection mismatch: "
                    f"expected {expected}, got {projected[output_index]}"
                )
        result[feature_key] = {
            "keep_indices": list(keep_indices),
            "dropped_force_indices": list(dropped_indices),
            "force_sentinels_excluded": True,
            "raw18_preserved": True,
            "projected_shape": list(projected.shape),
        }
    return result


def _validate_metadata(
    info: Mapping[str, Any], schema: JZPinTrainingSchema
) -> tuple[dict[str, Any], dict[str, Any]]:
    features = info.get("features")
    if not isinstance(features, Mapping):
        raise JZPinTrainingSchemaError("meta/info.json lacks a features object")
    schema.validate_raw_features(features)
    projected = schema.project_features(features)
    summary = {}
    for feature_key in FEATURE_KEYS:
        feature = projected[feature_key]
        if tuple(feature["shape"]) != (TRAINING_DIM,):
            raise JZPinTrainingSchemaError(
                f"projected {feature_key} shape must be [{TRAINING_DIM}], got {feature['shape']}"
            )
        if tuple(feature["names"]) != schema.training_feature_names[feature_key]:
            raise JZPinTrainingSchemaError(f"projected {feature_key} names do not match schema")
        summary[feature_key] = {
            "raw_shape": list(features[feature_key]["shape"]),
            "training_shape": list(feature["shape"]),
            "raw_names": list(features[feature_key]["names"]),
            "training_names": list(feature["names"]),
        }
    return projected, summary


def _validate_stats(stats: Mapping[str, Any], schema: JZPinTrainingSchema) -> dict[str, Any]:
    stats_before = json.dumps(stats, sort_keys=True, separators=(",", ":"))
    projected = schema.project_stats(stats)
    summary: dict[str, Any] = {}
    for feature_key in FEATURE_KEYS:
        raw_feature_stats = stats.get(feature_key)
        projected_feature_stats = projected.get(feature_key)
        if not isinstance(raw_feature_stats, Mapping) or not isinstance(projected_feature_stats, Mapping):
            raise JZPinTrainingSchemaError(f"statistics lack {feature_key!r}")
        vector_statistics = []
        for statistic_name in REQUIRED_VECTOR_STATISTICS:
            if statistic_name not in raw_feature_stats:
                raise JZPinTrainingSchemaError(
                    f"statistics for {feature_key!r} lack required {statistic_name!r}"
                )
            raw_shape = _shape(raw_feature_stats[statistic_name])
            projected_shape = _shape(projected_feature_stats[statistic_name])
            if not raw_shape or raw_shape[-1] != RAW_DIM:
                raise JZPinTrainingSchemaError(
                    f"raw {feature_key}.{statistic_name} must end in {RAW_DIM}, got {list(raw_shape)}"
                )
            if not projected_shape or projected_shape[-1] != TRAINING_DIM:
                raise JZPinTrainingSchemaError(
                    f"projected {feature_key}.{statistic_name} must end in {TRAINING_DIM}, "
                    f"got {list(projected_shape)}"
                )
            vector_statistics.append(statistic_name)

        for statistic_name, raw_value in raw_feature_stats.items():
            raw_shape = _shape(raw_value)
            if raw_shape and raw_shape[-1] == RAW_DIM:
                projected_shape = _shape(projected_feature_stats[statistic_name])
                if not projected_shape or projected_shape[-1] != TRAINING_DIM:
                    raise JZPinTrainingSchemaError(
                        f"projected {feature_key}.{statistic_name} must end in {TRAINING_DIM}"
                    )
                vector_statistics.append(statistic_name)
        summary[feature_key] = {
            "raw_dimension": RAW_DIM,
            "training_dimension": TRAINING_DIM,
            "projected_vector_statistics": sorted(set(vector_statistics)),
        }

    if json.dumps(stats, sort_keys=True, separators=(",", ":")) != stats_before:
        raise JZPinTrainingSchemaError("statistics projection mutated raw stats")
    summary["raw_stats_preserved"] = True
    return summary


def _validate_samples(
    dataset_root: Path,
    schema: JZPinTrainingSchema,
    *,
    allow_unavailable: bool,
) -> tuple[dict[str, Any], set[tuple[int, int]]]:
    data_paths = sorted((dataset_root / "data").glob("**/*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"no Parquet files found below {dataset_root / 'data'}")
    digest_before = {feature_key: hashlib.sha256() for feature_key in FEATURE_KEYS}
    digest_after = {feature_key: hashlib.sha256() for feature_key in FEATURE_KEYS}
    report: dict[str, Any] = {
        "parquet_path": None,
        "parquet_files_scanned": 0,
        "rows_scanned": 0,
        "columns_read": [*FRAME_KEYS, *FEATURE_KEYS],
        "video_decoded": False,
    }
    first_projected: dict[str, np.ndarray] = {}
    data_frame_keys: set[tuple[int, int]] = set()
    for parquet_path in data_paths:
        frame = pd.read_parquet(parquet_path, columns=[*FRAME_KEYS, *FEATURE_KEYS])
        if frame.empty:
            continue
        report["parquet_path"] = report["parquet_path"] or str(parquet_path)
        report["parquet_files_scanned"] += 1
        report["rows_scanned"] += len(frame)
        for local_row, (episode_index, frame_index) in enumerate(
            zip(frame["episode_index"], frame["frame_index"], strict=True)
        ):
            if not all(
                isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
                for value in (episode_index, frame_index)
            ):
                raise JZPinTrainingSchemaError(
                    f"{parquet_path} row {local_row} has invalid episode/frame index"
                )
            frame_key = (int(episode_index), int(frame_index))
            if frame_key in data_frame_keys:
                raise JZPinTrainingSchemaError(
                    f"{parquet_path} row {local_row} duplicates dataset frame {frame_key}"
                )
            data_frame_keys.add(frame_key)
        for feature_key in FEATURE_KEYS:
            vectors = []
            for local_row, raw_value in enumerate(frame[feature_key]):
                vector = np.asarray(raw_value)
                if vector.shape != (RAW_DIM,):
                    raise JZPinTrainingSchemaError(
                        f"{parquet_path} row {local_row} {feature_key} must be [{RAW_DIM}], "
                        f"got {list(vector.shape)}"
                    )
                vectors.append(vector)
            value = np.stack(vectors)
            value_before = value.copy()
            before_digest = _array_digest(value)
            digest_before[feature_key].update(before_digest.encode("ascii"))
            invalid_opening = np.argwhere(~np.isfinite(value[:, (14, 16)]))
            if invalid_opening.size:
                row, side_index = (int(item) for item in invalid_opening[0])
                raw_index = (14, 16)[side_index]
                raise JZPinTrainingSchemaError(
                    f"{parquet_path} row {row} {feature_key} gripper raw index {raw_index} "
                    "is missing or non-finite; cached values are not substituted"
                )
            if feature_key == "observation.state":
                projected = schema.project_observation(
                    value,
                    require_available_source=not allow_unavailable,
                )
            else:
                projected = schema.project_action(value)
            projected = np.asarray(projected)
            after_digest = _array_digest(value)
            digest_after[feature_key].update(after_digest.encode("ascii"))
            if not np.array_equal(value, value_before, equal_nan=True) or before_digest != after_digest:
                raise JZPinTrainingSchemaError(f"projecting {parquet_path} {feature_key} mutated raw18 input")
            if projected.shape != (len(frame), TRAINING_DIM):
                raise JZPinTrainingSchemaError(
                    f"{parquet_path} {feature_key} projection must be [N,{TRAINING_DIM}], "
                    f"got {list(projected.shape)}"
                )
            invalid = np.argwhere(~np.isfinite(projected))
            if invalid.size:
                row, training_index = (int(value) for value in invalid[0])
                raise JZPinTrainingSchemaError(
                    f"{parquet_path} row {row} {feature_key} training index {training_index} "
                    "is missing or non-finite"
                )
            first_projected.setdefault(feature_key, projected[0].copy())

    if report["rows_scanned"] == 0:
        raise ValueError("all dataset Parquet files are empty")
    for feature_key in FEATURE_KEYS:
        before_digest = digest_before[feature_key].hexdigest()
        after_digest = digest_after[feature_key].hexdigest()
        report[feature_key] = {
            "raw_shape": [RAW_DIM],
            "training_shape": [TRAINING_DIM],
            "raw_sha256_before": before_digest,
            "raw_sha256_after": after_digest,
            "raw18_preserved": before_digest == after_digest,
            "training_openings": [
                float(first_projected[feature_key][14]),
                float(first_projected[feature_key][15]),
            ],
        }
    return report, data_frame_keys


def _validate_gripper_source_freshness(
    dataset_root: Path,
    schema: JZPinTrainingSchema,
    *,
    expected_frame_keys: set[tuple[int, int]],
) -> dict[str, Any]:
    required_sides = [side for side, source in schema.observation_sources.items() if source != "unavailable"]
    if not required_sides:
        return {
            "required_sides": [],
            "timing_records": 0,
            "status": "not_required_for_unavailable_sources",
        }

    timing_paths = sorted((dataset_root / "meta" / "timing").glob("episode-*.jsonl"))
    if not timing_paths:
        raise JZPinTrainingSchemaError(
            "Strict gripper source semantics require timing sidecars with source_timing generations"
        )
    previous_generations: dict[tuple[str, str], int] = {}
    seen_frames: set[tuple[int, int]] = set()
    timing_records = 0
    side_counts = dict.fromkeys(required_sides, 0)
    stale_counts = dict.fromkeys(required_sides, 0)
    first_stale: str | None = None
    for timing_path in timing_paths:
        with timing_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                location = f"{timing_path}:{line_number}"
                frame_key = (record.get("episode_index"), record.get("frame_index"))
                if not all(isinstance(value, int) and not isinstance(value, bool) for value in frame_key):
                    raise JZPinTrainingSchemaError(f"{location} has invalid episode/frame index")
                if frame_key in seen_frames:
                    raise JZPinTrainingSchemaError(f"{location} duplicates timing frame {frame_key}")
                seen_frames.add(frame_key)
                session_id = record.get("session_id")
                if not isinstance(session_id, str) or not session_id:
                    raise JZPinTrainingSchemaError(f"{location} lacks a valid session_id")
                state = record.get("state")
                source_timing = state.get("source_timing") if isinstance(state, Mapping) else None
                try:
                    validate_source_timing(source_timing)
                except ValueError as exc:
                    raise JZPinTrainingSchemaError(
                        f"{location} lacks valid source_timing needed for gripper freshness"
                    ) from exc
                for side in required_sides:
                    source_name = f"{side}_gripper"
                    generation = int(source_timing["sources"][source_name]["generation"])
                    previous_key = (session_id, side)
                    previous = previous_generations.get(previous_key)
                    if previous is not None and generation <= previous:
                        stale_counts[side] += 1
                        first_stale = first_stale or (
                            f"{location} {source_name} generation did not strictly advance within session "
                            f"{session_id}: {previous}->{generation}"
                        )
                    previous_generations[previous_key] = generation
                    side_counts[side] += 1
                timing_records += 1

    missing_frames = sorted(expected_frame_keys - seen_frames)
    extra_frames = sorted(seen_frames - expected_frame_keys)
    if missing_frames or extra_frames:
        raise JZPinTrainingSchemaError(
            "Timing/source freshness frame keys must match Parquet exactly: "
            f"missing={missing_frames[:5]}, extra={extra_frames[:5]}, "
            f"timing={timing_records}, data={len(expected_frame_keys)}"
        )
    if any(stale_counts.values()):
        raise JZPinTrainingSchemaError(
            f"Gripper source generation reuse detected {stale_counts}; first={first_stale}; "
            "cached opening is not accepted"
        )
    return {
        "required_sides": required_sides,
        "timing_records": timing_records,
        "unique_frames": len(seen_frames),
        "side_generation_counts": side_counts,
        "stale_generation_counts": stale_counts,
        "status": "strictly_advanced_per_session",
    }


def run_check(args: Namespace) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    manifest_argument = getattr(args, "manifest", None)
    manifest_path = (
        Path(manifest_argument).expanduser().resolve()
        if manifest_argument is not None
        else dataset_root / "meta" / TRAINING_SCHEMA_FILENAME
    )
    allow_unavailable = bool(getattr(args, "allow_unavailable", False))
    report: dict[str, Any] = {
        "status": "FAIL",
        "dataset_root": str(dataset_root),
        "manifest_path": str(manifest_path),
        "allow_unavailable": allow_unavailable,
        "errors": [],
        "warnings": [],
    }

    info_path = dataset_root / "meta" / "info.json"
    stats_path = dataset_root / "meta" / "stats.json"
    for path, description in (
        (info_path, "dataset info"),
        (stats_path, "dataset statistics"),
        (manifest_path, "training projection manifest"),
    ):
        if not path.is_file():
            _add_error(report, f"missing {description}: {path}")
    if report["errors"]:
        return report

    try:
        info = _load_json(info_path, "meta/info.json")
        stats = _load_json(stats_path, "meta/stats.json")
        schema = JZPinTrainingSchema.from_file(manifest_path)
        schema_report = _build_schema_report(schema)
        _validate_schema_constants(schema_report)
        report["schema"] = schema_report

        sources = schema.observation_sources
        unavailable_sides = [side for side, source in sources.items() if source == "unavailable"]
        report["observation_sources"] = sources
        report["unavailable_sides"] = unavailable_sides
        if unavailable_sides and not allow_unavailable:
            schema.ensure_trainable()
        if unavailable_sides:
            report["warnings"].append(
                "gripper observation source is unavailable for "
                + ", ".join(unavailable_sides)
                + "; numeric projection is audit-only and must not be used for training"
            )

        report["force_exclusion"] = _validate_force_exclusion(schema, allow_unavailable=allow_unavailable)
        _, report["metadata_projection"] = _validate_metadata(info, schema)
        report["stats_projection"] = _validate_stats(stats, schema)
        sample_report, data_frame_keys = _validate_samples(
            dataset_root,
            schema,
            allow_unavailable=allow_unavailable,
        )
        report["sample"] = sample_report
        report["gripper_source_freshness"] = _validate_gripper_source_freshness(
            dataset_root,
            schema,
            expected_frame_keys=data_frame_keys,
        )
        report["status"] = "AUDIT" if unavailable_sides else "PASS"
    except (OSError, ValueError, TypeError, KeyError, AssertionError) as exc:
        _add_error(report, str(exc))
    return report


def main() -> int:
    args = parse_args()
    report = run_check(args)
    if args.report_json is not None:
        report_path = args.report_json.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        print(f"report_json={report_path}")

    print(f"status={report['status']}")
    print(f"dataset_root={report['dataset_root']}")
    print(f"manifest={report['manifest_path']}")
    if "schema" in report:
        print(
            f"mapping={RAW_SCHEMA_ID}:{RAW_DIM}D->{TRAINING_SCHEMA_ID}:{TRAINING_DIM}D "
            "keep=[0..14,16] drop_force=[15,17] canonical_opening=0=closed,100=open"
        )
        print(f"observation_sources={report['observation_sources']}")
    sample = report.get("sample")
    if sample:
        print(
            f"sample={sample['parquet_path']} raw18_preserved=true "
            f"rows_scanned={sample['rows_scanned']} observation_shape=16 "
            "action_shape=16 video_decoded=false"
        )
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    for error in report["errors"]:
        print(f"ERROR: {error}")
    return 0 if report["status"] in {"PASS", "AUDIT"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
