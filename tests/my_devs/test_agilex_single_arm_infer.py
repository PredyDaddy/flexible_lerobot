from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "my_devs" / "train" / "act" / "agilex" / "run_act_single_arm_infer.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_act_single_arm_infer", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_policy_config(*, arm: str = "right", state_dim: int = 7, action_dim: int = 7):
    camera_key = f"observation.images.camera_{arm}"
    return SimpleNamespace(
        type="act",
        input_features={
            "observation.state": SimpleNamespace(shape=(state_dim,)),
            "observation.images.camera_front": SimpleNamespace(shape=(3, 480, 640)),
            camera_key: SimpleNamespace(shape=(3, 480, 640)),
        },
        output_features={
            "action": SimpleNamespace(shape=(action_dim,)),
        },
    )


def make_live_observation() -> dict[str, object]:
    observation: dict[str, object] = {}
    for arm in ("left", "right"):
        offset = 0 if arm == "left" else 100
        for idx in range(7):
            observation[f"{arm}_joint{idx}.pos"] = float(offset + idx)
    observation["camera_front"] = np.full((480, 640, 3), 11, dtype=np.uint8)
    observation["camera_left"] = np.full((480, 640, 3), 22, dtype=np.uint8)
    observation["camera_right"] = np.full((480, 640, 3), 33, dtype=np.uint8)
    return observation


def test_validate_single_arm_checkpoint_schema_accepts_matching_right_checkpoint():
    module = load_module()
    policy_cfg = make_policy_config(arm="right")

    image_height, image_width = module.validate_single_arm_checkpoint_schema(policy_cfg, arm="right")

    assert (image_height, image_width) == (480, 640)


def test_validate_single_arm_checkpoint_schema_rejects_camera_arm_mismatch():
    module = load_module()
    policy_cfg = make_policy_config(arm="right")

    with pytest.raises(ValueError, match="camera_left|camera_right|arm"):
        module.validate_single_arm_checkpoint_schema(policy_cfg, arm="left")


def test_validate_single_arm_checkpoint_schema_rejects_bad_action_shape():
    module = load_module()
    policy_cfg = make_policy_config(arm="right", action_dim=14)

    with pytest.raises(ValueError, match="action|shape|7"):
        module.validate_single_arm_checkpoint_schema(policy_cfg, arm="right")


def test_build_single_arm_observation_payload_right_slices_state_and_cameras():
    module = load_module()
    observation = make_live_observation()

    payload = module.build_single_arm_observation_payload(observation, arm="right")

    assert set(payload) == {
        "observation.state",
        "observation.images.camera_front",
        "observation.images.camera_right",
    }
    np.testing.assert_allclose(payload["observation.state"], np.arange(100, 107, dtype=np.float32))
    assert payload["observation.images.camera_front"].shape == (480, 640, 3)
    assert payload["observation.images.camera_right"].shape == (480, 640, 3)
    assert payload["observation.images.camera_front"].dtype == np.uint8
    assert payload["observation.images.camera_right"].dtype == np.uint8


def test_build_single_arm_observation_payload_left_uses_left_slice_and_camera():
    module = load_module()
    observation = make_live_observation()

    payload = module.build_single_arm_observation_payload(observation, arm="left")

    assert set(payload) == {
        "observation.state",
        "observation.images.camera_front",
        "observation.images.camera_left",
    }
    np.testing.assert_allclose(payload["observation.state"], np.arange(7, dtype=np.float32))


def test_merge_single_arm_action_with_hold_current_overwrites_selected_arm_only():
    module = load_module()
    observation = make_live_observation()
    right_action = np.asarray([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], dtype=np.float32)

    merged = module.merge_single_arm_action_with_hold_current(
        observation=observation,
        arm_action=right_action,
        arm="right",
    )

    assert len(merged) == 14
    left_values = [merged[f"left_joint{i}.pos"] for i in range(7)]
    right_values = [merged[f"right_joint{i}.pos"] for i in range(7)]
    np.testing.assert_allclose(left_values, np.arange(7, dtype=np.float32))
    np.testing.assert_allclose(right_values, right_action)


def test_merge_single_arm_action_with_hold_current_rejects_wrong_action_dim():
    module = load_module()
    observation = make_live_observation()

    with pytest.raises(ValueError, match="7"):
        module.merge_single_arm_action_with_hold_current(
            observation=observation,
            arm_action=[0.0] * 6,
            arm="left",
        )
