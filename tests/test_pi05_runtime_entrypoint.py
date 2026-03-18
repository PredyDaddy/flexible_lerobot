from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())


@pytest.fixture
def entrypoint_module():
    return importlib.import_module("my_devs.pi05_engineering.run_pi05_chunk_infer")


class FakeRobot:
    def __init__(self) -> None:
        self.action_features = {"action": {"shape": [6]}}
        self.observation_features = {"observation": {"shape": [6]}}
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.send_action_calls = []
        self.get_observation_calls = 0
        self.is_connected = False
        self.stop_state = None
        self.robot_type = "so101_follower"

    def connect(self) -> None:
        self.connect_calls += 1
        self.is_connected = True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.is_connected = False

    def get_observation(self) -> dict[str, float]:
        self.get_observation_calls += 1
        return {f"obs{i}": float(i) for i in range(6)}

    def send_action(self, action) -> None:  # noqa: ANN001
        self.send_action_calls.append(action)
        if self.stop_state is not None:
            self.stop_state.request_stop("fake-send-stop")


class FakeResettable:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class FakePolicy(FakeResettable):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(device="cpu", use_amp=False, rtc_config=None)
        self.init_rtc_processor_calls = 0

    def init_rtc_processor(self) -> None:
        self.init_rtc_processor_calls += 1


class ImmediateThread:
    def __init__(self, *, target, kwargs, name, daemon) -> None:
        self._target = target
        self._kwargs = kwargs
        self.name = name
        self.daemon = daemon
        self.started = False
        self.join_calls = 0
        self._alive = False

    def start(self) -> None:
        self.started = True
        self._alive = True
        if set(self._kwargs) == {"context"}:
            self._target(self._kwargs["context"])
            self._alive = False
            return
        self._target(**self._kwargs)
        self._alive = False

    def join(self, timeout=None) -> None:  # noqa: ANN001
        self.join_calls += 1

    def is_alive(self) -> bool:
        return self._alive


class StuckThread:
    def __init__(self, *, target, kwargs, name, daemon) -> None:
        self._target = target
        self._kwargs = kwargs
        self.name = name
        self.daemon = daemon
        self.started = False
        self.join_calls = 0

    def start(self) -> None:
        self.started = True

    def join(self, timeout=None) -> None:  # noqa: ANN001
        self.join_calls += 1

    def is_alive(self) -> bool:
        return True


def _build_wrapped_immediate_thread(entrypoint_module, *, target, kwargs, name, daemon):
    return ImmediateThread(
        target=entrypoint_module._thread_entrypoint,
        kwargs={"name": name, "target": target, "target_kwargs": kwargs},
        name=name,
        daemon=daemon,
    )


def _patch_common_dry_run(monkeypatch, entrypoint_module, tmp_path: Path):
    policy_path = tmp_path / "policy"
    policy_path.mkdir()

    monkeypatch.setattr(entrypoint_module, "resolve_repo_root_for_entry", lambda: tmp_path)
    monkeypatch.setattr(entrypoint_module.runtime_common, "load_policy_config", lambda *_args, **_kwargs: SimpleNamespace(type="pi05"))
    return policy_path


def test_dry_run_does_not_build_robot_or_load_runtime(monkeypatch, entrypoint_module, tmp_path, capsys):
    policy_path = _patch_common_dry_run(monkeypatch, entrypoint_module, tmp_path)
    make_robot_called = {"value": False}
    load_runtime_called = {"value": False}

    def _robot_factory():
        make_robot_called["value"] = True
        raise AssertionError("dry-run must not resolve robot factory")

    def _load_policy_and_processors(*_args, **_kwargs):
        load_runtime_called["value"] = True
        raise AssertionError("dry-run must not load policy weights/processors")

    monkeypatch.setattr(entrypoint_module, "get_robot_factory", _robot_factory)
    monkeypatch.setattr(entrypoint_module.runtime_common, "load_policy_and_processors", _load_policy_and_processors)

    context = entrypoint_module.run(["--dry-run", "true", "--policy-path", str(policy_path)])

    captured = capsys.readouterr().out
    assert context.runtime_config.dry_run is True
    assert context.robot is None
    assert context.state is None
    assert make_robot_called["value"] is False
    assert load_runtime_called["value"] is False
    assert "DRY_RUN only validates CLI/config/path wiring" in captured


def test_build_runtime_context_defaults_camera_fps_to_control_fps(monkeypatch, entrypoint_module, tmp_path):
    monkeypatch.setattr(entrypoint_module, "resolve_repo_root_for_entry", lambda: tmp_path)
    policy_path = tmp_path / "policy"
    policy_path.mkdir()

    context = entrypoint_module.build_runtime_context(
        entrypoint_module.parse_args(
            [
                "--policy-path",
                str(policy_path),
                "--fps",
                "15",
            ]
        )
    )

    assert context.runtime_config.fps == 15
    assert context.args.camera_fps == 15
    assert context.robot_cfg.cameras["top"].fps == 15
    assert context.robot_cfg.cameras["wrist"].fps == 15


def test_build_runtime_context_allows_camera_fps_override(monkeypatch, entrypoint_module, tmp_path):
    monkeypatch.setattr(entrypoint_module, "resolve_repo_root_for_entry", lambda: tmp_path)
    policy_path = tmp_path / "policy"
    policy_path.mkdir()

    context = entrypoint_module.build_runtime_context(
        entrypoint_module.parse_args(
            [
                "--policy-path",
                str(policy_path),
                "--fps",
                "10",
                "--camera-fps",
                "30",
            ]
        )
    )

    assert context.runtime_config.fps == 10
    assert context.args.camera_fps == 30
    assert context.robot_cfg.cameras["top"].fps == 30
    assert context.robot_cfg.cameras["wrist"].fps == 30


def test_non_dry_run_orchestrates_fake_runtime(monkeypatch, entrypoint_module, tmp_path):
    policy_path = tmp_path / "policy"
    policy_path.mkdir()
    fake_robot = FakeRobot()
    fake_policy = FakePolicy()
    fake_preprocessor = FakeResettable()
    fake_postprocessor = FakeResettable()
    built_threads = []

    monkeypatch.setattr(entrypoint_module, "resolve_repo_root_for_entry", lambda: tmp_path)
    monkeypatch.setattr(entrypoint_module, "get_robot_factory", lambda: (lambda _cfg: fake_robot))
    monkeypatch.setattr(
        entrypoint_module.runtime_common,
        "build_dataset_features",
        lambda **_kwargs: SimpleNamespace(
            dataset_features={"dummy": {"shape": [1]}},
            robot_action_processor=object(),
            robot_observation_processor=object(),
        ),
    )
    monkeypatch.setattr(
        entrypoint_module.runtime_common,
        "load_policy_and_processors",
        lambda *_args, **_kwargs: SimpleNamespace(
            policy_config=SimpleNamespace(type="pi05", device="cpu"),
            policy=fake_policy,
            preprocessor=fake_preprocessor,
            postprocessor=fake_postprocessor,
        ),
    )

    def _producer(**kwargs):
        state = kwargs["state"]
        metrics = kwargs["metrics"]
        state.mark_first_chunk_ready()
        metrics.mark_first_chunk_ready()
        state.note_producer_iteration(0)
        metrics.record_producer_iteration()
        state.request_stop("producer-finished")

    def _actor(**kwargs):
        state = kwargs["runtime_state"]
        metrics = kwargs["metrics"]
        queue_controller = kwargs["queue_controller"]
        state.note_actor_iteration()
        metrics.record_actor_step(queue_depth_after=queue_controller.qsize())

    monkeypatch.setattr(entrypoint_module, "resolve_loop_targets", lambda: (_producer, _actor))

    def _build_worker_thread(**kwargs):
        thread = ImmediateThread(**kwargs)
        built_threads.append(thread)
        return thread

    monkeypatch.setattr(entrypoint_module, "_build_worker_thread", _build_worker_thread)

    context = entrypoint_module.run(["--policy-path", str(policy_path), "--run-time-s", "0"])

    assert fake_robot.connect_calls == 1
    assert fake_robot.disconnect_calls == 1
    assert context.state is not None
    assert context.metrics is not None
    assert context.queue_controller is not None
    assert context.dataset_artifacts is not None
    assert context.policy_artifacts is not None
    assert context.robot is not None
    assert context.robot.robot is fake_robot
    assert context.state.stop_reason == "producer-finished"
    assert context.state.first_chunk_ready_event.is_set() is True
    assert context.metrics.snapshot().first_chunk_ready_at_s is not None
    assert fake_policy.reset_calls == 1
    assert fake_preprocessor.reset_calls == 1
    assert fake_postprocessor.reset_calls == 1
    assert [thread.name for thread in built_threads] == ["PI05Producer", "PI05Actor"]
    assert all(thread.started for thread in built_threads)
    assert all(thread.join_calls == 1 for thread in built_threads)


def test_run_live_rebases_runtime_clock_after_connect(monkeypatch, entrypoint_module, tmp_path):
    policy_path = tmp_path / "policy"
    policy_path.mkdir()
    fake_robot = FakeRobot()
    fake_policy = FakePolicy()
    state = entrypoint_module.PI05RuntimeState()
    metrics = entrypoint_module.PI05RuntimeMetrics(window_size=8)
    state.rebase_start_time(timestamp_s=1.0)
    metrics.rebase_start_time(timestamp_s=1.0)

    monkeypatch.setattr(entrypoint_module, "resolve_repo_root_for_entry", lambda: tmp_path)
    monkeypatch.setattr(entrypoint_module, "get_robot_factory", lambda: (lambda _cfg: fake_robot))
    monkeypatch.setattr(entrypoint_module, "build_state", lambda: state)
    monkeypatch.setattr(entrypoint_module, "build_metrics", lambda _cfg: metrics)
    monkeypatch.setattr(
        entrypoint_module.runtime_common,
        "build_dataset_features",
        lambda **_kwargs: SimpleNamespace(
            dataset_features={},
            robot_action_processor=object(),
            robot_observation_processor=object(),
        ),
    )
    monkeypatch.setattr(
        entrypoint_module.runtime_common,
        "load_policy_and_processors",
        lambda *_args, **_kwargs: SimpleNamespace(
            policy_config=SimpleNamespace(type="pi05", device="cpu"),
            policy=fake_policy,
            preprocessor=FakeResettable(),
            postprocessor=FakeResettable(),
        ),
    )
    monkeypatch.setattr(
        entrypoint_module,
        "resolve_loop_targets",
        lambda: (
            lambda **kwargs: kwargs["state"].request_stop("producer-finished"),
            lambda **_kwargs: None,
        ),
    )
    monkeypatch.setattr(entrypoint_module, "_build_worker_thread", lambda **kwargs: ImmediateThread(**kwargs))
    monkeypatch.setattr(entrypoint_module.time, "perf_counter", lambda: 99.0)

    context = entrypoint_module.run(["--policy-path", str(policy_path), "--run-time-s", "30"])

    assert context.state is state
    assert context.metrics is metrics
    assert context.state.start_time_s == pytest.approx(99.0)
    assert context.metrics.snapshot(now_s=100.0).uptime_s == pytest.approx(1.0)


def test_enable_rtc_configures_policy(monkeypatch, entrypoint_module, tmp_path):
    policy_path = tmp_path / "policy"
    policy_path.mkdir()
    fake_robot = FakeRobot()
    fake_policy = FakePolicy()

    monkeypatch.setattr(entrypoint_module, "resolve_repo_root_for_entry", lambda: tmp_path)
    monkeypatch.setattr(entrypoint_module, "get_robot_factory", lambda: (lambda _cfg: fake_robot))
    monkeypatch.setattr(
        entrypoint_module.runtime_common,
        "build_dataset_features",
        lambda **_kwargs: SimpleNamespace(
            dataset_features={},
            robot_action_processor=object(),
            robot_observation_processor=object(),
        ),
    )
    monkeypatch.setattr(
        entrypoint_module.runtime_common,
        "load_policy_and_processors",
        lambda *_args, **_kwargs: SimpleNamespace(
            policy_config=SimpleNamespace(type="pi05", device="cpu"),
            policy=fake_policy,
            preprocessor=FakeResettable(),
            postprocessor=FakeResettable(),
        ),
    )
    monkeypatch.setattr(
        entrypoint_module,
        "resolve_loop_targets",
        lambda: (
            lambda **kwargs: kwargs["state"].request_stop("producer-finished"),
            lambda **_kwargs: None,
        ),
    )
    monkeypatch.setattr(entrypoint_module, "_build_worker_thread", lambda **kwargs: ImmediateThread(**kwargs))

    context = entrypoint_module.run(["--policy-path", str(policy_path), "--enable-rtc", "true"])

    assert context.runtime_config.enable_rtc is True
    assert fake_policy.config.rtc_config is not None
    assert fake_policy.config.rtc_config.enabled is True
    assert fake_policy.init_rtc_processor_calls == 1


def test_disable_rtc_clears_checkpoint_rtc_config(monkeypatch, entrypoint_module, tmp_path):
    policy_path = tmp_path / "policy"
    policy_path.mkdir()
    fake_robot = FakeRobot()
    fake_policy = FakePolicy()
    fake_policy.config.rtc_config = SimpleNamespace(enabled=True, from_checkpoint=True)

    monkeypatch.setattr(entrypoint_module, "resolve_repo_root_for_entry", lambda: tmp_path)
    monkeypatch.setattr(entrypoint_module, "get_robot_factory", lambda: (lambda _cfg: fake_robot))
    monkeypatch.setattr(
        entrypoint_module.runtime_common,
        "build_dataset_features",
        lambda **_kwargs: SimpleNamespace(
            dataset_features={},
            robot_action_processor=object(),
            robot_observation_processor=object(),
        ),
    )
    monkeypatch.setattr(
        entrypoint_module.runtime_common,
        "load_policy_and_processors",
        lambda *_args, **_kwargs: SimpleNamespace(
            policy_config=SimpleNamespace(type="pi05", device="cpu"),
            policy=fake_policy,
            preprocessor=FakeResettable(),
            postprocessor=FakeResettable(),
        ),
    )
    monkeypatch.setattr(
        entrypoint_module,
        "resolve_loop_targets",
        lambda: (
            lambda **kwargs: kwargs["state"].request_stop("producer-finished"),
            lambda **_kwargs: None,
        ),
    )
    monkeypatch.setattr(entrypoint_module, "_build_worker_thread", lambda **kwargs: ImmediateThread(**kwargs))

    context = entrypoint_module.run(["--policy-path", str(policy_path), "--enable-rtc", "false"])

    assert context.runtime_config.enable_rtc is False
    assert fake_policy.config.rtc_config is None
    assert fake_policy.init_rtc_processor_calls == 0


def test_offline_load_smoke_loads_policy_without_building_or_connecting_robot(
    monkeypatch, entrypoint_module, tmp_path, capsys
):
    policy_path = tmp_path / "policy"
    policy_path.mkdir()
    fake_policy = FakePolicy()

    monkeypatch.setattr(entrypoint_module, "resolve_repo_root_for_entry", lambda: tmp_path)
    monkeypatch.setattr(
        entrypoint_module,
        "get_robot_factory",
        lambda: (_ for _ in ()).throw(AssertionError("offline load smoke must not build a robot")),
    )
    monkeypatch.setattr(
        entrypoint_module.runtime_common,
        "run_offline_load_smoke",
        lambda *_args, **_kwargs: SimpleNamespace(
            policy_config=SimpleNamespace(type="pi05", device="cpu"),
            policy=fake_policy,
            preprocessor=FakeResettable(),
            postprocessor=FakeResettable(),
        ),
    )
    monkeypatch.setattr(
        entrypoint_module.runtime_common,
        "load_policy_and_processors",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("offline load smoke must not call live policy loader")
        ),
    )

    context = entrypoint_module.run(["--policy-path", str(policy_path), "--offline-load-smoke", "true"])

    captured = capsys.readouterr().out
    assert context.robot is None
    assert context.policy_artifacts is not None
    assert context.policy_config is not None
    assert fake_policy.reset_calls == 1
    assert "OFFLINE_LOAD_SMOKE validates policy/processors loading only" in captured


def test_connect_smoke_builds_robot_without_loading_policy(monkeypatch, entrypoint_module, tmp_path, capsys):
    policy_path = tmp_path / "policy"
    policy_path.mkdir()
    fake_robot = FakeRobot()
    load_runtime_called = {"value": False}

    monkeypatch.setattr(entrypoint_module, "resolve_repo_root_for_entry", lambda: tmp_path)
    monkeypatch.setattr(entrypoint_module, "get_robot_factory", lambda: (lambda _cfg: fake_robot))

    def _load_policy_and_processors(*_args, **_kwargs):
        load_runtime_called["value"] = True
        raise AssertionError("connect smoke must not load policy weights/processors")

    monkeypatch.setattr(entrypoint_module.runtime_common, "load_policy_and_processors", _load_policy_and_processors)

    context = entrypoint_module.run(["--policy-path", str(policy_path), "--connect-smoke", "true"])

    captured = capsys.readouterr().out
    assert context.robot is not None
    assert context.robot.robot is fake_robot
    assert fake_robot.connect_calls == 1
    assert fake_robot.disconnect_calls == 1
    assert load_runtime_called["value"] is False
    assert "CONNECT_SMOKE validates robot.connect()/disconnect() only" in captured


def test_load_connect_smoke_loads_policy_and_connects_robot(monkeypatch, entrypoint_module, tmp_path, capsys):
    policy_path = tmp_path / "policy"
    policy_path.mkdir()
    fake_robot = FakeRobot()
    fake_policy = FakePolicy()
    fake_preprocessor = FakeResettable()
    fake_postprocessor = FakeResettable()

    monkeypatch.setattr(entrypoint_module, "resolve_repo_root_for_entry", lambda: tmp_path)
    monkeypatch.setattr(entrypoint_module, "get_robot_factory", lambda: (lambda _cfg: fake_robot))
    monkeypatch.setattr(
        entrypoint_module.runtime_common,
        "load_policy_and_processors",
        lambda *_args, **_kwargs: SimpleNamespace(
            policy_config=SimpleNamespace(type="pi05", device="cpu"),
            policy=fake_policy,
            preprocessor=fake_preprocessor,
            postprocessor=fake_postprocessor,
        ),
    )

    context = entrypoint_module.run(["--policy-path", str(policy_path), "--load-connect-smoke", "true"])

    captured = capsys.readouterr().out
    assert context.robot is not None
    assert context.robot.robot is fake_robot
    assert context.policy_artifacts is not None
    assert context.policy_config is not None
    assert fake_robot.connect_calls == 1
    assert fake_robot.disconnect_calls == 1
    assert fake_policy.reset_calls == 1
    assert fake_preprocessor.reset_calls == 1
    assert fake_postprocessor.reset_calls == 1
    assert "LOAD_CONNECT_SMOKE validates policy/processors loading plus robot.connect()/disconnect() only" in captured


def test_keyboard_interrupt_requests_shutdown_and_disconnect(monkeypatch, entrypoint_module, tmp_path):
    policy_path = tmp_path / "policy"
    policy_path.mkdir()
    fake_robot = FakeRobot()

    monkeypatch.setattr(entrypoint_module, "resolve_repo_root_for_entry", lambda: tmp_path)
    monkeypatch.setattr(entrypoint_module, "get_robot_factory", lambda: (lambda _cfg: fake_robot))
    monkeypatch.setattr(
        entrypoint_module.runtime_common,
        "build_dataset_features",
        lambda **_kwargs: SimpleNamespace(
            dataset_features={},
            robot_action_processor=object(),
            robot_observation_processor=object(),
        ),
    )
    monkeypatch.setattr(
        entrypoint_module.runtime_common,
        "load_policy_and_processors",
        lambda *_args, **_kwargs: SimpleNamespace(
            policy_config=SimpleNamespace(type="pi05", device="cpu"),
            policy=FakePolicy(),
            preprocessor=FakeResettable(),
            postprocessor=FakeResettable(),
        ),
    )
    monkeypatch.setattr(entrypoint_module, "resolve_loop_targets", lambda: (lambda **_kwargs: None, lambda **_kwargs: None))
    monkeypatch.setattr(entrypoint_module, "_build_worker_thread", lambda **kwargs: ImmediateThread(**kwargs))
    monkeypatch.setattr(entrypoint_module, "_run_until_stopped", lambda _context: (_ for _ in ()).throw(KeyboardInterrupt()))

    context = entrypoint_module.run(
        [
            "--policy-path",
            str(policy_path),
            "--startup-wait-for-first-chunk",
            "false",
        ]
    )

    assert context.state is not None
    assert context.state.stop_reason == "KeyboardInterrupt"
    assert fake_robot.connect_calls == 1
    assert fake_robot.disconnect_calls == 1


def test_startup_timeout_cleans_up_threads_and_disconnects_robot(monkeypatch, entrypoint_module, tmp_path):
    policy_path = tmp_path / "policy"
    policy_path.mkdir()
    fake_robot = FakeRobot()
    built_threads = []

    monkeypatch.setattr(entrypoint_module, "resolve_repo_root_for_entry", lambda: tmp_path)
    monkeypatch.setattr(entrypoint_module, "get_robot_factory", lambda: (lambda _cfg: fake_robot))
    monkeypatch.setattr(
        entrypoint_module.runtime_common,
        "build_dataset_features",
        lambda **_kwargs: SimpleNamespace(
            dataset_features={},
            robot_action_processor=object(),
            robot_observation_processor=object(),
        ),
    )
    monkeypatch.setattr(
        entrypoint_module.runtime_common,
        "load_policy_and_processors",
        lambda *_args, **_kwargs: SimpleNamespace(
            policy_config=SimpleNamespace(type="pi05", device="cpu"),
            policy=FakePolicy(),
            preprocessor=FakeResettable(),
            postprocessor=FakeResettable(),
        ),
    )
    monkeypatch.setattr(entrypoint_module, "resolve_loop_targets", lambda: (lambda **_kwargs: None, lambda **_kwargs: None))

    def _build_worker_thread(**kwargs):
        thread = ImmediateThread(**kwargs)
        built_threads.append(thread)
        return thread

    monkeypatch.setattr(entrypoint_module, "_build_worker_thread", _build_worker_thread)

    context = entrypoint_module.build_runtime_context(
        entrypoint_module.parse_args(
            [
                "--policy-path",
                str(policy_path),
                "--startup-timeout-s",
                "0.01",
            ]
        )
    )
    expected_message = (
        "Timed out waiting for the first chunk. "
        f"startup_timeout_s={context.runtime_config.startup_timeout_s}"
    )

    with pytest.raises(TimeoutError, match=re.escape(expected_message)):
        entrypoint_module.run_live(context)

    assert context.state is not None
    assert context.state.last_error is not None
    assert context.state.last_error.source == "entrypoint"
    assert context.state.last_error.message == expected_message
    assert context.state.stop_reason == expected_message
    assert fake_robot.connect_calls == 1
    assert fake_robot.disconnect_calls == 1
    assert all(thread.join_calls == 1 for thread in built_threads)


def test_worker_exception_cleans_up_and_returns_context(monkeypatch, entrypoint_module, tmp_path):
    policy_path = tmp_path / "policy"
    policy_path.mkdir()
    fake_robot = FakeRobot()
    built_threads = []

    monkeypatch.setattr(entrypoint_module, "resolve_repo_root_for_entry", lambda: tmp_path)
    monkeypatch.setattr(entrypoint_module, "get_robot_factory", lambda: (lambda _cfg: fake_robot))
    monkeypatch.setattr(
        entrypoint_module.runtime_common,
        "build_dataset_features",
        lambda **_kwargs: SimpleNamespace(
            dataset_features={},
            robot_action_processor=object(),
            robot_observation_processor=object(),
        ),
    )
    monkeypatch.setattr(
        entrypoint_module.runtime_common,
        "load_policy_and_processors",
        lambda *_args, **_kwargs: SimpleNamespace(
            policy_config=SimpleNamespace(type="pi05", device="cpu"),
            policy=FakePolicy(),
            preprocessor=FakeResettable(),
            postprocessor=FakeResettable(),
        ),
    )
    monkeypatch.setattr(
        entrypoint_module,
        "resolve_loop_targets",
        lambda: (
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("worker boom")),
            lambda **_kwargs: None,
        ),
    )

    def _build_worker_thread(**kwargs):
        thread = _build_wrapped_immediate_thread(entrypoint_module, **kwargs)
        built_threads.append(thread)
        return thread

    monkeypatch.setattr(entrypoint_module, "_build_worker_thread", _build_worker_thread)

    context = entrypoint_module.run(
        [
            "--policy-path",
            str(policy_path),
            "--startup-wait-for-first-chunk",
            "false",
        ]
    )

    assert context.state is not None
    assert context.state.stop_reason == "PI05Producer: worker boom"
    assert context.state.last_error is not None
    assert context.state.last_error.source == "PI05Producer"
    assert fake_robot.disconnect_calls == 1
    assert all(thread.join_calls == 1 for thread in built_threads)


def test_worker_shutdown_timeout_raises_and_skips_disconnect(monkeypatch, entrypoint_module, tmp_path):
    policy_path = tmp_path / "policy"
    policy_path.mkdir()
    fake_robot = FakeRobot()
    built_threads = []

    monkeypatch.setattr(entrypoint_module, "resolve_repo_root_for_entry", lambda: tmp_path)
    monkeypatch.setattr(entrypoint_module, "get_robot_factory", lambda: (lambda _cfg: fake_robot))
    monkeypatch.setattr(
        entrypoint_module.runtime_common,
        "build_dataset_features",
        lambda **_kwargs: SimpleNamespace(
            dataset_features={},
            robot_action_processor=object(),
            robot_observation_processor=object(),
        ),
    )
    monkeypatch.setattr(
        entrypoint_module.runtime_common,
        "load_policy_and_processors",
        lambda *_args, **_kwargs: SimpleNamespace(
            policy_config=SimpleNamespace(type="pi05", device="cpu"),
            policy=FakePolicy(),
            preprocessor=FakeResettable(),
            postprocessor=FakeResettable(),
        ),
    )
    monkeypatch.setattr(entrypoint_module, "resolve_loop_targets", lambda: (lambda **_kwargs: None, lambda **_kwargs: None))
    monkeypatch.setattr(
        entrypoint_module,
        "_run_until_stopped",
        lambda context: context.state.request_stop("synthetic-stop"),
    )

    def _build_worker_thread(**kwargs):
        thread = StuckThread(**kwargs)
        built_threads.append(thread)
        return thread

    monkeypatch.setattr(entrypoint_module, "_build_worker_thread", _build_worker_thread)

    context = entrypoint_module.build_runtime_context(
        entrypoint_module.parse_args(
            [
                "--policy-path",
                str(policy_path),
                "--startup-wait-for-first-chunk",
                "false",
            ]
        )
    )

    with pytest.raises(
        RuntimeError,
        match=re.escape("Timed out waiting for worker threads to exit before disconnect: PI05Producer, PI05Actor"),
    ):
        entrypoint_module.run_live(context)

    assert context.worker_shutdown_clean is False
    assert context.alive_worker_names == ["PI05Producer", "PI05Actor"]
    assert context.state is not None
    assert context.state.last_error is not None
    assert context.state.last_error.source == "entrypoint.shutdown"
    assert fake_robot.disconnect_calls == 0
    assert all(thread.join_calls == 1 for thread in built_threads)


def test_non_dry_run_wrapper_maps_context_into_real_loop_kwargs(monkeypatch, entrypoint_module, tmp_path):
    policy_path = tmp_path / "policy"
    policy_path.mkdir()
    fake_robot = FakeRobot()
    fake_policy = FakePolicy()
    fake_preprocessor = FakeResettable()
    fake_postprocessor = FakeResettable()
    sentinel_action_processor = object()
    sentinel_observation_processor = object()
    recorded = {}

    monkeypatch.setattr(entrypoint_module, "resolve_repo_root_for_entry", lambda: tmp_path)
    monkeypatch.setattr(entrypoint_module, "get_robot_factory", lambda: (lambda _cfg: fake_robot))
    monkeypatch.setattr(
        entrypoint_module.runtime_common,
        "build_dataset_features",
        lambda **_kwargs: SimpleNamespace(
            dataset_features={"dummy": {"shape": [1]}},
            robot_action_processor=sentinel_action_processor,
            robot_observation_processor=sentinel_observation_processor,
        ),
    )
    monkeypatch.setattr(
        entrypoint_module.runtime_common,
        "load_policy_and_processors",
        lambda *_args, **_kwargs: SimpleNamespace(
            policy_config=SimpleNamespace(type="pi05", device="cpu"),
            policy=fake_policy,
            preprocessor=fake_preprocessor,
            postprocessor=fake_postprocessor,
        ),
    )
    monkeypatch.setattr(entrypoint_module, "_build_worker_thread", lambda **kwargs: ImmediateThread(**kwargs))

    producer_module = importlib.import_module("my_devs.pi05_engineering.runtime.producer_loop")
    actor_module = importlib.import_module("my_devs.pi05_engineering.runtime.actor_loop")

    def _record_producer_loop(**kwargs):
        recorded["producer"] = kwargs
        kwargs["state"].mark_first_chunk_ready()
        kwargs["metrics"].mark_first_chunk_ready()
        kwargs["state"].request_stop("producer-recorded")
        return 1

    def _record_actor_loop(**kwargs):
        recorded["actor"] = kwargs
        return SimpleNamespace(iterations=0, startup_ready=True, stop_reason=kwargs["runtime_state"].stop_reason)

    monkeypatch.setattr(producer_module, "run_producer_loop", _record_producer_loop)
    monkeypatch.setattr(actor_module, "run_actor_loop", _record_actor_loop)

    context = entrypoint_module.run(
        [
            "--policy-path",
            str(policy_path),
        ]
    )

    assert context.state is not None
    assert context.state.stop_reason == "producer-recorded"
    assert context.state.first_chunk_ready_event.is_set() is True
    assert fake_robot.connect_calls == 1
    assert fake_robot.disconnect_calls == 1
    assert fake_preprocessor.reset_calls == 1
    assert fake_postprocessor.reset_calls == 1
    assert recorded["producer"]["config"] is context.runtime_config
    assert recorded["producer"]["state"] is context.state
    assert recorded["producer"]["metrics"] is context.metrics
    assert recorded["producer"]["queue_controller"] is context.queue_controller
    assert recorded["producer"]["dataset_features"] is context.dataset_artifacts.dataset_features
    assert recorded["producer"]["policy"] is fake_policy
    assert recorded["producer"]["preprocessor"] is fake_preprocessor
    assert recorded["producer"]["postprocessor"] is fake_postprocessor
    assert recorded["producer"]["task"] == context.args.task
    assert recorded["producer"]["robot_type"] == fake_robot.robot_type
    assert recorded["producer"]["robot_observation_processor"] is sentinel_observation_processor
    assert recorded["producer"]["observation_provider"].__self__ is context.robot
    assert recorded["producer"]["observation_provider"].__self__.robot is fake_robot
    assert recorded["actor"]["config"] is context.runtime_config
    assert recorded["actor"]["runtime_state"] is context.state
    assert recorded["actor"]["metrics"] is context.metrics
    assert recorded["actor"]["queue_controller"] is context.queue_controller
    assert recorded["actor"]["dataset_features"] is context.dataset_artifacts.dataset_features
    assert recorded["actor"]["robot_action_processor"] is sentinel_action_processor
    assert recorded["actor"]["observation_for_processor"] is None
    assert recorded["actor"]["send_action"].__self__ is context.robot
    assert recorded["actor"]["send_action"].__self__.robot is fake_robot


def test_observability_log_dict_contains_phase1_fields(entrypoint_module, tmp_path):
    context = entrypoint_module.RuntimeEntrypointContext(
        repo_root=tmp_path,
        args=SimpleNamespace(task="offline"),
        runtime_config=entrypoint_module.PI05RuntimeConfig(),
        robot_cfg=object(),
        policy_path=tmp_path / "policy",
    )
    context.metrics = entrypoint_module.build_metrics(context.runtime_config)
    context.state = entrypoint_module.build_state()
    context.queue_controller = entrypoint_module.build_queue_controller(context.runtime_config)

    context.metrics.record_producer_iteration(timestamp_s=1.0)
    context.metrics.record_producer_iteration(timestamp_s=1.5)
    context.metrics.record_actor_step(timestamp_s=2.0)
    context.metrics.record_inference(
        total_s=0.2,
        preprocess_s=0.05,
        model_s=0.1,
        postprocess_s=0.05,
        chunk_length=3,
        leftover_length=2,
        action_index_before_inference=7,
        action_index_delta=2,
        inference_delay=1,
        real_delay=2,
    )
    context.state.request_stop("producer-recorded")
    context.state.note_chunk_inference(
        prev_chunk_left_over=[1, 2],
        original_actions=[1, 2, 3],
        processed_actions=[1, 2],
        action_index_before_inference=7,
        action_index_delta=2,
        inference_delay=1,
        real_delay=2,
        merge_mode="plain",
        trimmed_prefix_steps=2,
        enqueued_steps=1,
    )
    context.queue_controller.pop_next_action()

    log_dict = entrypoint_module._build_observability_log_dict(context, now_s=3.0)

    assert "first_warmup_latency_s" in log_dict
    assert "producer_rate_hz" in log_dict
    assert "latency_total_max_s" in log_dict
    assert "latency_total_p95_s" in log_dict
    assert "state_stop_reason" in log_dict
    assert "state_last_action_index_delta" in log_dict
    assert "state_last_real_delay" in log_dict
    assert "state_last_original_actions_length" in log_dict
    assert "state_last_processed_actions_length" in log_dict
    assert "queue_skip_send_events" in log_dict
    assert "queue_actions_popped" in log_dict
    assert "worker_shutdown_clean" in log_dict
    assert "alive_worker_names" in log_dict
    assert log_dict["latency_total_latest_s"] == pytest.approx(0.2)
    assert log_dict["latency_total_max_s"] == pytest.approx(0.2)
    assert log_dict["latency_total_p95_s"] == pytest.approx(0.2)
    assert log_dict["state_stop_reason"] == "producer-recorded"
    assert log_dict["state_last_real_delay"] == 2
    assert log_dict["queue_skip_send_events"] == 1
    assert log_dict["queue_actions_popped"] == 0
    assert log_dict["worker_shutdown_clean"] is None
    assert log_dict["alive_worker_names"] == []
