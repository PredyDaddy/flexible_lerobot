#!/usr/bin/env python

"""Compatibility exports for the former Python training entry point.

Training now starts through ``train_act.sh`` and the official
``lerobot-train`` console tool.  Keep these imports because repository tests
and older offline utilities use the helper function names directly.
"""

from .jz_lerobot_train_hook import (
    insert_resize_step,
    insert_schema_steps,
    load_training_schema,
    parse_resize_size,
)

__all__ = [
    "insert_resize_step",
    "insert_schema_steps",
    "load_training_schema",
    "parse_resize_size",
]


if __name__ == "__main__":
    raise SystemExit(
        "This compatibility module is not a training entry point. "
        "Run train_act.sh or train_act_20_epochs.sh so training uses lerobot-train."
    )
