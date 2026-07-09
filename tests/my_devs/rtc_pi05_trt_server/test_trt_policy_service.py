from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from my_devs.train.pi.so101.rtc_pi05.trt_server.run_trt_policy_server import main as server_main
from my_devs.train.pi.so101.rtc_pi05.trt_server.run_trt_robot_client import main as robot_client_main
from my_devs.train.pi.so101.rtc_pi05.trt_server.trt_policy_service import (
    REQUIRED_RUNTIME_ASSETS,
    TRTPolicyServiceConfig,
    run_trt_chunk_inference,
    validate_runtime_assets,
)


def create_runtime_assets(tmp_path):  # noqa: ANN001
    assets = tmp_path / "assets"
    assets.mkdir()
    for name in REQUIRED_RUNTIME_ASSETS:
        (assets / name).write_bytes(b"{}")
    return assets


def test_validate_runtime_assets_rejects_model_safetensors_when_strict(tmp_path) -> None:
    assets = create_runtime_assets(tmp_path)
    validate_runtime_assets(assets, strict=True)

    (assets / "model.safetensors").write_bytes(b"not allowed here")
    try:
        validate_runtime_assets(assets, strict=True)
    except RuntimeError as exc:
        assert "model.safetensors" in str(exc)
    else:
        raise AssertionError("strict pure TRT runtime assets should reject model.safetensors")


def test_trt_policy_service_config_resolves_auto_profile_from_engine_names(tmp_path) -> None:
    config = TRTPolicyServiceConfig(
        runtime_assets_dir=tmp_path,
        prefix_engine_path=tmp_path / "prefix_fp16.engine",
        denoise_engine_path=tmp_path / "denoise.engine",
    )

    assert config.resolved_profile_name() == "fp16_constrained"


def test_run_trt_chunk_inference_forwards_rtc_kwargs(monkeypatch) -> None:
    from my_devs.train.pi.so101.rtc_pi05.trt_server import trt_policy_service

    monkeypatch.setattr(
        trt_policy_service,
        "prepare_observation_for_inference",
        lambda observation_frame, **_kwargs: observation_frame,
    )

    class FakePolicy:
        def __init__(self) -> None:
            self.config = SimpleNamespace(use_amp=False)
            self.last_kwargs = None

        def predict_action_chunk(self, batch, **kwargs):  # noqa: ANN001
            self.last_kwargs = kwargs
            assert batch == {"prepared": True}
            return torch.tensor([[[1.0], [2.0]]], dtype=torch.float32)

    policy = FakePolicy()
    raw_chunk, processed_chunk = run_trt_chunk_inference(
        policy=policy,
        preprocessor=lambda batch: batch,
        postprocessor=lambda actions: actions + 10.0,
        observation_frame={"prepared": True},
        device=torch.device("cpu"),
        task="task",
        robot_type="so101_follower",
        enable_rtc=True,
        predicted_delay_steps=3,
        prev_chunk_left_over=np.ones((2, 1), dtype=np.float32),
        execution_horizon=4,
    )

    assert torch.equal(raw_chunk, torch.tensor([[1.0], [2.0]]))
    assert torch.equal(processed_chunk, torch.tensor([[11.0], [12.0]]))
    assert policy.last_kwargs["inference_delay"] == 3
    assert policy.last_kwargs["execution_horizon"] == 4
    assert torch.equal(policy.last_kwargs["prev_chunk_left_over"], torch.ones(2, 1))


def test_run_trt_policy_server_dry_run_checks_assets_and_engine_paths(tmp_path, capsys) -> None:
    assets = create_runtime_assets(tmp_path)
    prefix_engine = tmp_path / "prefix.engine"
    denoise_engine = tmp_path / "denoise.engine"
    prefix_engine.write_bytes(b"engine")
    denoise_engine.write_bytes(b"engine")

    result = server_main(
        [
            "--runtime-assets-dir",
            str(assets),
            "--prefix-engine-path",
            str(prefix_engine),
            "--denoise-engine-path",
            str(denoise_engine),
            "--dry-run",
            "true",
        ]
    )

    assert result == 0
    assert "DRY_RUN passed" in capsys.readouterr().out


def test_run_trt_robot_client_requires_confirm_control_before_real_run(capsys) -> None:
    result = robot_client_main(["--run-time-s", "120"])

    assert result == 0
    assert "Missing --confirm-control" in capsys.readouterr().out


def test_run_trt_robot_client_injects_default_trt_server_url_for_safe_dry_run(monkeypatch) -> None:
    from my_devs.train.pi.so101.rtc_pi05.trt_server import run_trt_robot_client

    captured = {}

    def fake_base_main(argv):  # noqa: ANN001
        captured["argv"] = argv
        return 0

    monkeypatch.delenv("PI05_SERVER_URL", raising=False)
    monkeypatch.setattr(run_trt_robot_client.base_client, "main", fake_base_main)

    result = robot_client_main(["--dry-run", "true"])

    assert result == 0
    assert captured["argv"][:2] == ["--server-url", "http://127.0.0.1:8090"]
