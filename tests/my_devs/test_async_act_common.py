from __future__ import annotations

from types import SimpleNamespace

import pytest

from my_devs.async_act.common import (
    resolve_temporal_ensemble_settings,
    resolve_actions_per_chunk,
    validate_single_arm_checkpoint_schema,
)


def make_policy_config(*, arm: str = "right", state_dim: int = 7, action_dim: int = 7):
    camera_key = f"observation.images.camera_{arm}"
    return SimpleNamespace(
        type="act",
        chunk_size=100,
        input_features={
            "observation.state": SimpleNamespace(shape=(state_dim,)),
            "observation.images.camera_front": SimpleNamespace(shape=(3, 480, 640)),
            camera_key: SimpleNamespace(shape=(3, 480, 640)),
        },
        output_features={"action": SimpleNamespace(shape=(action_dim,))},
    )


def test_validate_single_arm_checkpoint_schema_rejects_non_act_policy():
    policy_cfg = make_policy_config(arm="right")
    policy_cfg.type = "diffusion"

    with pytest.raises(ValueError, match="Expected ACT policy"):
        validate_single_arm_checkpoint_schema(policy_cfg, arm="right")


def test_validate_single_arm_checkpoint_schema_accepts_matching_right_checkpoint():
    policy_cfg = make_policy_config(arm="right")

    image_height, image_width = validate_single_arm_checkpoint_schema(policy_cfg, arm="right")

    assert (image_height, image_width) == (480, 640)


def test_validate_single_arm_checkpoint_schema_rejects_camera_arm_mismatch():
    policy_cfg = make_policy_config(arm="right")

    with pytest.raises(ValueError, match="camera_left|camera_right|arm"):
        validate_single_arm_checkpoint_schema(policy_cfg, arm="left")


def test_validate_single_arm_checkpoint_schema_rejects_bad_action_shape():
    policy_cfg = make_policy_config(arm="right", action_dim=14)

    with pytest.raises(ValueError, match="action|shape|7"):
        validate_single_arm_checkpoint_schema(policy_cfg, arm="right")


def test_resolve_actions_per_chunk_defaults_to_checkpoint_chunk_size():
    resolved = resolve_actions_per_chunk(
        chunk_size=100,
        explicit_actions_per_chunk=None,
        policy_n_action_steps=None,
    )

    assert resolved == 100


def test_resolve_actions_per_chunk_uses_alias_when_explicit_value_missing():
    resolved = resolve_actions_per_chunk(
        chunk_size=100,
        explicit_actions_per_chunk=None,
        policy_n_action_steps=20,
    )

    assert resolved == 20


def test_resolve_temporal_ensemble_settings_forces_full_chunk_and_every_step():
    actions_per_chunk, threshold, notes = resolve_temporal_ensemble_settings(
        chunk_size=100,
        explicit_actions_per_chunk=20,
        policy_n_action_steps=1,
        policy_temporal_ensemble_coeff=0.01,
        requested_chunk_size_threshold=0.2,
        aggregate_fn_name="weighted_average",
    )

    assert actions_per_chunk == 100
    assert threshold == 1.0
    assert any("actions_per_chunk" in note for note in notes)
    assert any("chunk_size_threshold" in note for note in notes)
    assert any("aggregate_fn_name" in note and "ACT-aware" in note for note in notes)


def test_resolve_temporal_ensemble_settings_accepts_none_n_action_steps():
    actions_per_chunk, threshold, _ = resolve_temporal_ensemble_settings(
        chunk_size=100,
        explicit_actions_per_chunk=None,
        policy_n_action_steps=None,
        policy_temporal_ensemble_coeff=0.01,
        requested_chunk_size_threshold=0.2,
        aggregate_fn_name="weighted_average",
    )

    assert actions_per_chunk == 100
    assert threshold == 1.0


def test_resolve_temporal_ensemble_settings_rejects_non_unit_n_action_steps():
    with pytest.raises(ValueError, match="n-action-steps|n_action_steps"):
        resolve_temporal_ensemble_settings(
            chunk_size=100,
            explicit_actions_per_chunk=None,
            policy_n_action_steps=2,
            policy_temporal_ensemble_coeff=0.01,
            requested_chunk_size_threshold=0.5,
            aggregate_fn_name="weighted_average",
        )
