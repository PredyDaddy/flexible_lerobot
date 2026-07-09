from __future__ import annotations

from pathlib import Path

from my_devs.train.pi.so101.pure_trt.runtime.paths import default_paths
from my_devs.train.pi.so101.pure_trt.scripts.stage_existing_artifacts import stage_file
from my_devs.train.pi.so101.rtc_pi05.trt_server import run_trt_policy_server


def test_default_paths_live_under_so101_pure_trt() -> None:
    paths = default_paths()

    assert "my_devs/train/pi/so101/pure_trt" in paths.artifact_dir.as_posix()
    assert paths.prefix_engine("fp16_constrained").name == "pi05_so101_prefix_cache_b1_fp16_constrained.engine"
    assert paths.denoise_engine("fp16_constrained").name == "pi05_so101_denoise_step_b1_fp16_constrained.engine"


def test_rtc_trt_server_defaults_point_to_pure_trt_artifacts() -> None:
    paths = default_paths()

    assert run_trt_policy_server.DEFAULT_RUNTIME_ASSETS_DIR == paths.runtime_assets_dir
    assert run_trt_policy_server.DEFAULT_PREFIX_ENGINE == paths.prefix_fp16_constrained_engine
    assert run_trt_policy_server.DEFAULT_DENOISE_ENGINE == paths.denoise_fp16_constrained_engine


def test_stage_file_can_symlink_large_artifacts(tmp_path) -> None:
    source = tmp_path / "source.engine"
    target = tmp_path / "nested" / "target.engine"
    source.write_bytes(b"engine")

    stage_file(source, target, mode="symlink", force=False)

    assert target.is_symlink()
    assert target.resolve() == source.resolve()


def test_stage_file_can_replace_with_copy(tmp_path) -> None:
    source = tmp_path / "source.engine"
    target = tmp_path / "target.engine"
    source.write_bytes(b"engine-v1")
    stage_file(source, target, mode="copy", force=False)
    source.write_bytes(b"engine-v2")

    stage_file(source, target, mode="copy", force=True)

    assert not target.is_symlink()
    assert target.read_bytes() == b"engine-v2"
