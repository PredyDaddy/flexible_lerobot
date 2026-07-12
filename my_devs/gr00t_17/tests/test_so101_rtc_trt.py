from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from transformers.feature_extraction_utils import BatchFeature

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "so101_rtc_trt.py"
SPEC = importlib.util.spec_from_file_location("so101_rtc_trt", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeConfig:
    action_horizon = 40


class FakeActionHead:
    config = FakeConfig()
    action_dim = 3

    def __init__(self) -> None:
        self.init_actions = torch.full((1, 40, 3), -1.0, dtype=torch.bfloat16)


def test_rtc_initialization_uses_previous_tail_and_freezes_prefix() -> None:
    previous = torch.arange(40 * 3, dtype=torch.float32).reshape(1, 40, 3)
    action_input = BatchFeature(data={"action": previous})
    actions, strength = MODULE._initialize_rtc_actions(
        FakeActionHead(),
        action_input,
        {
            "action_horizon": 16,
            "rtc_overlap_steps": 8,
            "rtc_frozen_steps": 2,
            "rtc_ramp_rate": 2.0,
        },
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    torch.testing.assert_close(actions[:, :8], previous[:, 8:16].to(torch.bfloat16))
    assert torch.all(actions[:, 8:] == -1)
    assert torch.all(strength[:, :2] == 0)
    assert torch.all(strength[:, 2:8] > 0)
    assert torch.all(strength[:, 2:8] < 1)
    assert torch.all(strength[:, 8:] == 1)


def test_non_rtc_initialization_keeps_engine_seed() -> None:
    actions, strength = MODULE._initialize_rtc_actions(
        FakeActionHead(),
        BatchFeature(data={}),
        None,
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    assert torch.all(actions == -1)
    assert torch.all(strength == 1)


def test_invalid_rtc_options_are_rejected() -> None:
    action_input = BatchFeature(data={"action": torch.zeros(1, 40, 3)})
    try:
        MODULE._initialize_rtc_actions(
            FakeActionHead(),
            action_input,
            {
                "action_horizon": 16,
                "rtc_overlap_steps": 8,
                "rtc_frozen_steps": 9,
                "rtc_ramp_rate": 2.0,
            },
            batch_size=1,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
    except ValueError as exc:
        assert "frozen" in str(exc)
    else:
        raise AssertionError("Invalid RTC frozen steps were accepted")
