from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from my_devs.pi05_engineering.runtime import common


def test_resolve_repo_root_finds_repo_markers(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    nested = repo_root / "my_devs" / "pi05_engineering" / "runtime"
    nested.mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text("", encoding="utf-8")
    (repo_root / "src" / "lerobot").mkdir(parents=True)
    script_path = nested / "common.py"
    script_path.write_text("", encoding="utf-8")

    assert common.resolve_repo_root(script_path) == repo_root


def test_resolve_repo_root_raises_without_markers(tmp_path: Path) -> None:
    missing = tmp_path / "missing.py"
    missing.write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError):
        common.resolve_repo_root(missing)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("true", True),
        ("YES", True),
        ("1", True),
        ("off", False),
        ("No", False),
        (True, True),
        (False, False),
    ],
)
def test_parse_bool_supported_values(value: str | bool, expected: bool) -> None:
    assert common.parse_bool(value) is expected


def test_parse_bool_rejects_invalid_value() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        common.parse_bool("maybe")


def test_env_bool_uses_env_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PI05_BOOL", "yes")
    assert common.env_bool("PI05_BOOL", False) is True


def test_env_bool_uses_default_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PI05_BOOL", raising=False)
    assert common.env_bool("PI05_BOOL", True) is True


def test_maybe_path_handles_none_and_empty() -> None:
    assert common.maybe_path(None) is None
    assert common.maybe_path("") is None


def test_maybe_path_expands_path() -> None:
    result = common.maybe_path("~/pi05")
    assert isinstance(result, Path)
    assert result == Path("~/pi05").expanduser()


def test_ensure_local_tokenizer_dir_returns_existing_dir(tmp_path: Path) -> None:
    tokenizer_dir = tmp_path / "google" / "paligemma-3b-pt-224"
    tokenizer_dir.mkdir(parents=True)

    assert common.ensure_local_tokenizer_dir(tmp_path) == tokenizer_dir


def test_ensure_local_tokenizer_dir_raises_for_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        common.ensure_local_tokenizer_dir(tmp_path)


def test_load_pre_post_processors_uses_checkpoint_local_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    policy_path = tmp_path / "checkpoint"
    policy_path.mkdir()
    calls: list[dict[str, object]] = []

    def fake_from_pretrained(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(kind=kwargs["config_filename"])

    monkeypatch.setattr(common.PolicyProcessorPipeline, "from_pretrained", staticmethod(fake_from_pretrained))

    pre, post = common.load_pre_post_processors(policy_path)

    assert pre.kind == "policy_preprocessor.json"
    assert post.kind == "policy_postprocessor.json"
    assert [call["config_filename"] for call in calls] == [
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ]
    assert all(call["pretrained_model_name_or_path"] == str(policy_path) for call in calls)


def test_build_so101_robot_config_builds_opencv_cameras() -> None:
    robot_cfg = common.build_so101_robot_config(
        robot_id="robot-1",
        robot_type="so101_follower",
        calib_dir="~/calib",
        robot_port="/dev/null",
        top_cam_index=1,
        wrist_cam_index=2,
        img_width=640,
        img_height=480,
        camera_fps=30,
    )

    assert robot_cfg.id == "robot-1"
    assert robot_cfg.port == "/dev/null"
    assert robot_cfg.calibration_dir == Path("~/calib").expanduser()
    assert set(robot_cfg.cameras) == {"top", "wrist"}
    assert robot_cfg.cameras["top"].index_or_path == 1
    assert robot_cfg.cameras["wrist"].index_or_path == 2
    assert robot_cfg.cameras["top"].fps == 30
    assert robot_cfg.cameras["wrist"].fps == 30


def test_build_so101_robot_config_rejects_unsupported_robot_type() -> None:
    with pytest.raises(ValueError):
        common.build_so101_robot_config(
            robot_id="robot-1",
            robot_type="unsupported",
            calib_dir=None,
            robot_port="/dev/null",
            top_cam_index=1,
            wrist_cam_index=2,
            img_width=640,
            img_height=480,
            camera_fps=30,
        )


def test_build_dataset_features_orchestrates_processor_feature_build(monkeypatch: pytest.MonkeyPatch) -> None:
    teleop_processor = object()
    robot_action_processor = object()
    robot_observation_processor = object()
    calls: list[tuple[str, object]] = []

    def fake_make_default_processors():
        return teleop_processor, robot_action_processor, robot_observation_processor

    def fake_create_initial_features(*, action=None, observation=None):
        if action is not None:
            calls.append(("create_action", action))
            return {"action_features": action}
        calls.append(("create_observation", observation))
        return {"observation_features": observation}

    def fake_aggregate_pipeline_dataset_features(*, pipeline, initial_features, use_videos):
        calls.append(("aggregate", pipeline))
        return {"pipeline": pipeline, "initial_features": initial_features, "use_videos": use_videos}

    def fake_combine_feature_dicts(action_dict, observation_dict):
        return {"action": action_dict, "observation": observation_dict}

    monkeypatch.setattr(common, "make_default_processors", fake_make_default_processors)
    monkeypatch.setattr(common, "create_initial_features", fake_create_initial_features)
    monkeypatch.setattr(common, "aggregate_pipeline_dataset_features", fake_aggregate_pipeline_dataset_features)
    monkeypatch.setattr(common, "combine_feature_dicts", fake_combine_feature_dicts)

    artifacts = common.build_dataset_features(
        action_features={"joint": {"shape": [6]}},
        observation_features={"camera": {"shape": [3, 224, 224]}},
    )

    assert artifacts.robot_action_processor is robot_action_processor
    assert artifacts.robot_observation_processor is robot_observation_processor
    assert artifacts.dataset_features["action"]["pipeline"] is robot_action_processor
    assert artifacts.dataset_features["observation"]["pipeline"] is robot_observation_processor
    assert ("create_action", {"joint": {"shape": [6]}}) in calls
    assert ("create_observation", {"camera": {"shape": [3, 224, 224]}}) in calls


def test_load_policy_and_processors_preserves_strict_false(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    tokenizer_dir = repo_root / "google" / "paligemma-3b-pt-224"
    tokenizer_dir.mkdir(parents=True)
    preprocessor = object()
    postprocessor = object()
    observed: dict[str, object] = {}

    class FakePolicy:
        def __init__(self):
            self.to_device = None

        def to(self, device: str) -> None:
            self.to_device = device

    class FakePolicyClass:
        @classmethod
        def from_pretrained(cls, path: str, strict: bool = True):
            observed["path"] = path
            observed["strict"] = strict
            return FakePolicy()

    policy_cfg = SimpleNamespace(type="pi05", device="cpu", pretrained_path=None)

    monkeypatch.setattr(common, "ensure_policy_registry", lambda: None)
    monkeypatch.setattr(common, "ensure_local_tokenizer_dir", lambda path: tokenizer_dir)
    monkeypatch.setattr(common, "load_policy_config", lambda path, device_override=None: policy_cfg)
    monkeypatch.setattr(common, "get_policy_class", lambda policy_type: FakePolicyClass)
    monkeypatch.setattr(common, "load_pre_post_processors", lambda path: (preprocessor, postprocessor))

    artifacts = common.load_policy_and_processors(checkpoint_path, repo_root=repo_root)

    assert artifacts.repo_root == repo_root.resolve()
    assert artifacts.policy_path == checkpoint_path
    assert artifacts.tokenizer_dir == tokenizer_dir
    assert artifacts.policy_config is policy_cfg
    assert artifacts.policy_class is FakePolicyClass
    assert artifacts.preprocessor is preprocessor
    assert artifacts.postprocessor is postprocessor
    assert observed["path"] == str(checkpoint_path)
    assert observed["strict"] is False
    assert artifacts.policy.to_device == "cpu"


def test_run_offline_load_smoke_delegates_to_policy_load(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sentinel = SimpleNamespace(name="artifacts")
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()

    monkeypatch.setattr(common, "load_policy_and_processors", lambda *args, **kwargs: sentinel)

    result = common.run_offline_load_smoke(checkpoint_path, repo_root=tmp_path, strict=False)

    assert result is sentinel
