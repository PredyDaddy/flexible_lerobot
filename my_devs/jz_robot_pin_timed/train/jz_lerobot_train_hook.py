"""JZ Pin dataset and processor extensions for the official LeRobot trainer.

The public training process is still the installed ``lerobot-train`` console
tool.  This module only replaces the two factory callables imported by that
tool so the immutable raw18 dataset is presented to ACT as model16 and so the
checkpoint serializes the raw18/model16 boundary plus a common image resize.
No training loop is implemented here.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import lerobot.scripts.lerobot_train as train_module
from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from lerobot.processor.hil_processor import ImageCropResizeProcessorStep
from lerobot.robots.jz_robot_pin_timed.training_schema import (
    JZPinRaw18ToTraining16ProcessorStep,
    JZPinTraining16ToRaw18ActionProcessorStep,
    JZPinTrainingDatasetView,
    JZPinTrainingSchema,
)

_INSTALLED = False


def parse_resize_size() -> tuple[int, int]:
    raw = os.environ.get("LEROBOT_ACT_RESIZE", "224,224").strip().lower()
    parts = raw.replace("x", ",").split(",")
    if len(parts) != 2:
        raise ValueError(f"LEROBOT_ACT_RESIZE must be '<height>,<width>', got {raw!r}")
    height, width = (int(part) for part in parts)
    if height <= 0 or width <= 0:
        raise ValueError(f"Resize values must be positive, got {(height, width)}")
    return height, width


def load_training_schema() -> tuple[Path, JZPinTrainingSchema]:
    raw_path = os.environ.get("JZ_PIN_TRAINING_SCHEMA", "").strip()
    if not raw_path:
        raise ValueError("JZ_PIN_TRAINING_SCHEMA must point to an explicit schema manifest")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"JZ Pin training schema does not exist: {path}")
    schema = JZPinTrainingSchema.from_file(path)
    schema.ensure_trainable()
    return path, schema


def insert_resize_step(preprocessor, resize_size: tuple[int, int]) -> None:
    resize_steps = [step for step in preprocessor.steps if isinstance(step, ImageCropResizeProcessorStep)]
    if len(resize_steps) > 1:
        raise ValueError("Preprocessor contains multiple image resize steps")
    if resize_steps:
        if tuple(resize_steps[0].resize_size or ()) != resize_size:
            raise ValueError(
                f"Checkpoint resize {resize_steps[0].resize_size} differs from requested {resize_size}"
            )
        return

    insert_index = 0
    for index, step in enumerate(preprocessor.steps):
        if step.__class__.__name__ == "AddBatchDimensionProcessorStep":
            insert_index = index + 1
            break
    preprocessor.steps.insert(insert_index, ImageCropResizeProcessorStep(resize_size=resize_size))


def insert_schema_steps(preprocessor, postprocessor, schema: JZPinTrainingSchema) -> None:
    normalizer_index = next(
        index for index, step in enumerate(preprocessor.steps) if isinstance(step, NormalizerProcessorStep)
    )
    projection_steps = [
        step for step in preprocessor.steps if isinstance(step, JZPinRaw18ToTraining16ProcessorStep)
    ]
    if len(projection_steps) > 1:
        raise ValueError("Preprocessor contains multiple JZ Pin raw18-to-model16 steps")
    if projection_steps:
        if projection_steps[0].schema.to_dict() != schema.to_dict():
            raise ValueError("Checkpoint preprocessor schema differs from requested training schema")
        if preprocessor.steps.index(projection_steps[0]) >= normalizer_index:
            raise ValueError("Checkpoint raw18-to-model16 step must run before normalization")
    else:
        preprocessor.steps.insert(
            normalizer_index,
            JZPinRaw18ToTraining16ProcessorStep(schema=schema.to_dict()),
        )

    unnormalizer_index = next(
        index for index, step in enumerate(postprocessor.steps) if isinstance(step, UnnormalizerProcessorStep)
    )
    expansion_steps = [
        step for step in postprocessor.steps if isinstance(step, JZPinTraining16ToRaw18ActionProcessorStep)
    ]
    if len(expansion_steps) > 1:
        raise ValueError("Postprocessor contains multiple JZ Pin model16-to-raw18 steps")
    if expansion_steps:
        if expansion_steps[0].schema.to_dict() != schema.to_dict():
            raise ValueError("Checkpoint postprocessor schema differs from requested training schema")
        if postprocessor.steps.index(expansion_steps[0]) <= unnormalizer_index:
            raise ValueError("Checkpoint model16-to-raw18 step must run after unnormalization")
    else:
        postprocessor.steps.insert(
            unnormalizer_index + 1,
            JZPinTraining16ToRaw18ActionProcessorStep(schema=schema.to_dict()),
        )


def install() -> None:
    """Install the idempotent factory patch before ``lerobot-train`` calls ``train()``."""

    global _INSTALLED
    if _INSTALLED:
        return

    resize_size = parse_resize_size()
    schema_path, schema = load_training_schema()
    original_dataset_factory: Callable = train_module.make_dataset
    original_processor_factory: Callable = train_module.make_pre_post_processors

    def make_projected_dataset(*args, **kwargs):
        raw_dataset = original_dataset_factory(*args, **kwargs)
        return JZPinTrainingDatasetView(raw_dataset, schema)

    def make_processors_with_jz_boundary(*args, **kwargs):
        preprocessor, postprocessor = original_processor_factory(*args, **kwargs)
        insert_resize_step(preprocessor, resize_size)
        insert_schema_steps(preprocessor, postprocessor, schema)
        return preprocessor, postprocessor

    train_module.make_dataset = make_projected_dataset
    train_module.make_pre_post_processors = make_processors_with_jz_boundary
    _INSTALLED = True
    print(
        "[jz_pin_timed/lerobot-train-hook] "
        f"schema={schema_path} raw18->model16 resize={resize_size} "
        f"sources={schema.observation_sources}"
    )
