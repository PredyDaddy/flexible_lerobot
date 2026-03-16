from __future__ import annotations

from types import SimpleNamespace

import pytest

from my_devs.async_act import run_async_act_single_arm as runner


def make_policy_config():
    return SimpleNamespace(
        type="act",
        chunk_size=100,
        device="cpu",
        input_features={
            "observation.state": SimpleNamespace(shape=(7,)),
            "observation.images.camera_front": SimpleNamespace(shape=(3, 480, 640)),
            "observation.images.camera_right": SimpleNamespace(shape=(3, 480, 640)),
        },
        output_features={"action": SimpleNamespace(shape=(7,))},
    )


def patch_checkpoint(monkeypatch):
    monkeypatch.setattr(runner, "load_policy_config", lambda path: make_policy_config())
    monkeypatch.setattr(runner, "validate_single_arm_checkpoint_schema", lambda policy_cfg, arm: (480, 640))


def test_build_client_config_uses_policy_n_action_steps_alias(tmp_path, monkeypatch):
    args = runner.build_parser().parse_args(
        [
            "client",
            "--arm",
            "right",
            "--policy-path",
            str(tmp_path),
            "--policy-n-action-steps",
            "20",
        ]
    )
    patch_checkpoint(monkeypatch)

    client_cfg, policy_cfg, policy_path, runtime_options = runner.build_client_config(args)

    assert client_cfg.actions_per_chunk == 20
    assert client_cfg.chunk_size_threshold == 0.5
    assert client_cfg.robot.type == "single_arm_agilex"
    assert policy_cfg.type == "act"
    assert policy_path == tmp_path
    assert runtime_options["runtime_mode"] == "chunk_stream"
    assert runtime_options["temporal_ensemble_coeff"] is None
    assert runtime_options["runtime_notes"] == []


def test_build_client_config_chunked_mode_preserves_v1_stream_settings(tmp_path, monkeypatch):
    args = runner.build_parser().parse_args(
        [
            "client",
            "--arm",
            "right",
            "--policy-path",
            str(tmp_path),
            "--chunk-size-threshold",
            "0.25",
            "--aggregate-fn-name",
            "latest_only",
            "--action-smoothing-alpha",
            "0.4",
            "--max-joint-step-rad",
            "0.1",
        ]
    )
    patch_checkpoint(monkeypatch)

    client_cfg, _, _, runtime_options = runner.build_client_config(args)

    assert client_cfg.actions_per_chunk == 100
    assert client_cfg.chunk_size_threshold == 0.25
    assert client_cfg.aggregate_fn_name == "latest_only"
    assert client_cfg.robot.action_smoothing_alpha == 0.4
    assert client_cfg.robot.max_joint_step_rad == 0.1
    assert runtime_options["runtime_mode"] == "chunk_stream"
    assert runtime_options["runtime_notes"] == []


def test_build_client_config_passes_action_smoothing_settings_to_robot(tmp_path, monkeypatch):
    args = runner.build_parser().parse_args(
        [
            "client",
            "--arm",
            "right",
            "--policy-path",
            str(tmp_path),
            "--action-smoothing-alpha",
            "0.4",
            "--max-joint-step-rad",
            "0.05",
        ]
    )
    patch_checkpoint(monkeypatch)

    client_cfg, _, _, runtime_options = runner.build_client_config(args)

    assert client_cfg.robot.action_smoothing_alpha == 0.4
    assert client_cfg.robot.max_joint_step_rad == 0.05
    assert runtime_options["runtime_mode"] == "chunk_stream"


def test_build_client_config_temporal_mode_uses_full_chunk_and_forces_refresh(tmp_path, monkeypatch):
    args = runner.build_parser().parse_args(
        [
            "client",
            "--arm",
            "right",
            "--policy-path",
            str(tmp_path),
            "--policy-n-action-steps",
            "1",
            "--policy-temporal-ensemble-coeff",
            "0.01",
        ]
    )
    patch_checkpoint(monkeypatch)

    client_cfg, policy_cfg, policy_path, runtime_options = runner.build_client_config(args)

    assert client_cfg.actions_per_chunk == 100
    assert client_cfg.chunk_size_threshold == 1.0
    assert policy_cfg.type == "act"
    assert policy_path == tmp_path
    assert runtime_options["runtime_mode"] == "act_temporal_ensemble"
    assert runtime_options["temporal_ensemble_coeff"] == 0.01
    assert runtime_options["runtime_notes"]


def test_build_client_config_temporal_mode_rejects_n_action_steps_other_than_one(tmp_path, monkeypatch):
    args = runner.build_parser().parse_args(
        [
            "client",
            "--arm",
            "right",
            "--policy-path",
            str(tmp_path),
            "--policy-n-action-steps",
            "2",
            "--policy-temporal-ensemble-coeff",
            "0.01",
        ]
    )
    patch_checkpoint(monkeypatch)

    with pytest.raises(ValueError, match="n-action-steps|n_action_steps"):
        runner.build_client_config(args)


def test_build_client_config_temporal_mode_overrides_partial_actions_per_chunk(tmp_path, monkeypatch):
    args = runner.build_parser().parse_args(
        [
            "client",
            "--arm",
            "right",
            "--policy-path",
            str(tmp_path),
            "--actions-per-chunk",
            "20",
            "--policy-n-action-steps",
            "1",
            "--policy-temporal-ensemble-coeff",
            "0.01",
            "--chunk-size-threshold",
            "0.2",
        ]
    )
    patch_checkpoint(monkeypatch)

    client_cfg, _, _, runtime_options = runner.build_client_config(args)

    assert client_cfg.actions_per_chunk == 100
    assert client_cfg.chunk_size_threshold == 1.0
    assert runtime_options["runtime_mode"] == "act_temporal_ensemble"
    assert any("actions_per_chunk" in note for note in runtime_options["runtime_notes"])
    assert any("chunk_size_threshold" in note for note in runtime_options["runtime_notes"])
