from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re

from my_devs.agilex_web_collection.app.api.schemas import NormalizedRecordRequest, RecordRequest


DATASET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class RecordRequestValidationError(ValueError):
    def __init__(self, message: str, code: str = "validation_error") -> None:
        super().__init__(message)
        self.code = code


def coerce_record_request(request: RecordRequest | Mapping[str, object]) -> RecordRequest:
    if isinstance(request, RecordRequest):
        return request
    if isinstance(request, Mapping):
        return RecordRequest.model_validate(dict(request))
    raise TypeError(f"Unsupported request type: {type(request)!r}")


def is_resumable_dataset_dir(dataset_dir: Path) -> bool:
    if not dataset_dir.is_dir():
        return False

    markers = [
        dataset_dir / "meta",
        dataset_dir / "data",
        dataset_dir / "videos",
        dataset_dir / "meta" / "info.json",
    ]
    return any(path.exists() for path in markers)


def validate_and_normalize_record_request(
    request: RecordRequest | Mapping[str, object],
    predicted_dataset_dir: Path,
) -> tuple[NormalizedRecordRequest, bool, list[str]]:
    raw_request = coerce_record_request(request)
    warnings: list[str] = []

    if not DATASET_NAME_PATTERN.fullmatch(raw_request.dataset_name):
        raise RecordRequestValidationError(
            "dataset_name must match ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ and cannot contain slashes,"
            " spaces, or path traversal segments.",
        )

    reset_time_s = raw_request.reset_time_s
    if raw_request.num_episodes == 1 and raw_request.reset_time_s != 0:
        warnings.append("reset_time_s was normalized to 0 because num_episodes=1.")
        reset_time_s = 0.0

    dataset_exists = predicted_dataset_dir.exists()
    if dataset_exists and not raw_request.resume:
        raise RecordRequestValidationError(
            f"Dataset already exists: {predicted_dataset_dir}. Set resume=true to append.",
            code="conflict_error",
        )

    if not dataset_exists and raw_request.resume:
        raise RecordRequestValidationError(
            f"Cannot resume missing dataset: {predicted_dataset_dir}.",
            code="conflict_error",
        )

    if dataset_exists and raw_request.resume and not is_resumable_dataset_dir(predicted_dataset_dir):
        raise RecordRequestValidationError(
            f"Existing directory is not a valid LeRobot dataset: {predicted_dataset_dir}.",
            code="conflict_error",
        )

    expected_total_time_s = (
        raw_request.num_episodes * raw_request.episode_time_s
        + max(raw_request.num_episodes - 1, 0) * reset_time_s
    )

    normalized = NormalizedRecordRequest(
        dataset_name=raw_request.dataset_name,
        episode_time_s=raw_request.episode_time_s,
        num_episodes=raw_request.num_episodes,
        fps=raw_request.fps,
        reset_time_s=reset_time_s,
        resume=raw_request.resume,
        repo_id=f"dummy/{raw_request.dataset_name}",
        expected_total_time_s=expected_total_time_s,
    )
    return normalized, dataset_exists, warnings
