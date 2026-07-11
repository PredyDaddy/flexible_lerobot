from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_PATH = REPO_ROOT / "udp_test" / "test_scripts" / "arm_side" / "orin_ros_state_udp_bridge.py"


def _load_bridge_module(monkeypatch):
    rclpy = types.ModuleType("rclpy")
    executors = types.ModuleType("rclpy.executors")
    executors.ExternalShutdownException = type("ExternalShutdownException", (Exception,), {})
    executors.SingleThreadedExecutor = object
    qos = types.ModuleType("rclpy.qos")

    class QoSProfile:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    qos.QoSProfile = QoSProfile
    qos.HistoryPolicy = SimpleNamespace(KEEP_LAST="keep_last")
    qos.ReliabilityPolicy = SimpleNamespace(RELIABLE="reliable")
    qos.DurabilityPolicy = SimpleNamespace(VOLATILE="volatile")
    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.JointState = object
    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.Float64MultiArray = object
    protocol = types.ModuleType("lerobot.robots.jz_robot_udp.protocol")
    protocol.PROTOCOL_VERSION = 1
    protocol.STATE_MESSAGE_TYPE = "state"
    protocol.encode_state_packet = lambda packet: json.dumps(
        packet, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    common = types.ModuleType("my_devs.jz_robot.common")
    common.DEFAULT_ROBOT_CONFIG = Path("robot.yaml")
    common.load_robot_config = lambda _path: None

    for name, module in {
        "rclpy": rclpy,
        "rclpy.executors": executors,
        "rclpy.qos": qos,
        "sensor_msgs": sensor_msgs,
        "sensor_msgs.msg": sensor_msgs_msg,
        "std_msgs": std_msgs,
        "std_msgs.msg": std_msgs_msg,
        "lerobot.robots.jz_robot_udp.protocol": protocol,
        "my_devs.jz_robot.common": common,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("test_orin_ros_state_udp_bridge_module", BRIDGE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class FakeClock:
    def __init__(self, monotonic_ns: int = 0, wall_base_ns: int = 1_700_000_000_000_000_000):
        self.now_ns = monotonic_ns
        self.wall_base_ns = wall_base_ns

    def monotonic_ns(self) -> int:
        return self.now_ns

    def wall_time_ns(self) -> int:
        return self.wall_base_ns + self.now_ns

    def set_ms(self, value_ms: float) -> None:
        self.now_ns = round(value_ms * 1_000_000)

    def advance(self, seconds: float) -> None:
        self.now_ns += round(seconds * 1_000_000_000)


class FakeSocket:
    def __init__(self, clock: FakeClock | None = None):
        self.clock = clock
        self.sent: list[tuple[int | None, bytes, tuple[str, int]]] = []

    def sendto(self, payload: bytes, target: tuple[str, int]) -> int:
        sent_at_ns = None if self.clock is None else self.clock.now_ns
        self.sent.append((sent_at_ns, payload, target))
        return len(payload)


def _robot_config() -> SimpleNamespace:
    return SimpleNamespace(
        left_joint_names=["left_joint1", "left_joint2"],
        right_joint_names=["right_joint1", "right_joint2"],
        use_gripper=True,
    )


def _joint_message(side: str, generation: int) -> SimpleNamespace:
    return SimpleNamespace(
        name=[f"{side}_joint1", f"{side}_joint2"],
        position=[float(generation), float(generation) + 0.25],
        header=SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=generation)),
    )


def _gripper_message(generation: int) -> SimpleNamespace:
    return SimpleNamespace(data=[float(generation), float(generation) + 0.5])


def _update_all_sources(collector, generation: int) -> None:
    collector.update_joints("left", _joint_message("left", generation))
    collector.update_joints("right", _joint_message("right", generation))
    collector.update_gripper("left", _gripper_message(generation))
    collector.update_gripper("right", _gripper_message(generation))


def _snapshot(collector, *, require_advanced: bool = False, last_sent=None):
    return collector.snapshot(
        seq=1,
        robot_name="robot1",
        max_source_age_ms=50.0,
        max_source_skew_ms=20.0,
        require_all_sources_advanced=require_advanced,
        last_sent_generations={} if last_sent is None else last_sent,
    )


def _assert_snapshot_matches_generations(snapshot, source_names, wall_base_ns: int) -> None:
    assert snapshot.packet is not None
    assert snapshot.source_timing is not None
    packet = snapshot.packet
    sources = snapshot.source_timing["sources"]
    for source_name in source_names:
        assert sources[source_name]["generation"] == snapshot.generations[source_name]
        assert sources[source_name]["recv_monotonic_ns"] == 0
        assert sources[source_name]["recv_wall_ns"] == wall_base_ns

    for side in ("left", "right"):
        joint_generation = snapshot.generations[f"{side}_joints"]
        assert packet["joints"][side] == {
            f"{side}_joint1": float(joint_generation),
            f"{side}_joint2": float(joint_generation) + 0.25,
        }
        assert sources[f"{side}_joints"]["header_stamp_ns"] == joint_generation

        gripper_generation = snapshot.generations[f"{side}_gripper"]
        assert packet["grippers"][side] == {
            "width": float(gripper_generation),
            "force": float(gripper_generation) + 0.5,
        }
        assert sources[f"{side}_gripper"]["header_stamp_ns"] is None


def _make_sender(bridge, collector, sock, **overrides):
    values = {
        "collector": collector,
        "sock": sock,
        "target": ("127.0.0.1", 39010),
        "robot_name": "robot1",
        "hz": 30.0,
        "max_source_age_ms": 50.0,
        "max_source_skew_ms": 20.0,
        "require_all_sources_advanced": True,
        "print_every": 0,
        "printer": lambda *_args, **_kwargs: None,
    }
    values.update(overrides)
    return bridge.StateUdpSender(**values)


def test_state_collector_reports_exact_missing_inputs(monkeypatch) -> None:
    bridge = _load_bridge_module(monkeypatch)
    cfg = _robot_config()
    collector = bridge.ReadonlyStateCollector(cfg)

    assert collector.missing_inputs() == {
        "left_joints": ["left_joint1", "left_joint2"],
        "right_joints": ["right_joint1", "right_joint2"],
        "left_gripper_fields": ["width", "force"],
        "right_gripper_fields": ["width", "force"],
    }

    collector.update_joints("left", SimpleNamespace(name=["left_joint1"], position=[0.1]))
    collector.update_joints(
        "right",
        SimpleNamespace(name=["right_joint1", "right_joint2"], position=[0.2, 0.3]),
    )
    collector.update_gripper("left", SimpleNamespace(data=[50.0]))
    collector.update_gripper("right", SimpleNamespace(data=[60.0, 70.0]))

    assert collector.missing_inputs() == {
        "left_joints": ["left_joint2"],
        "right_joints": [],
        "left_gripper_fields": ["force"],
        "right_gripper_fields": [],
    }
    details = collector.readiness_details()
    assert "counts=" in details
    assert "'left_joints': [\'left_joint2\']" in details

    collector.update_joints(
        "left",
        SimpleNamespace(name=["left_joint1", "left_joint2"], position=[0.1, 0.4]),
    )
    collector.update_gripper("left", SimpleNamespace(data=[50.0, 80.0]))

    assert all(not missing for missing in collector.missing_inputs().values())
    assert collector.ready()


def test_source_generations_advance_independently(monkeypatch) -> None:
    bridge = _load_bridge_module(monkeypatch)
    clock = FakeClock()
    collector = bridge.ReadonlyStateCollector(
        _robot_config(), monotonic_ns=clock.monotonic_ns, wall_time_ns=clock.wall_time_ns
    )

    assert collector.counts == {name: 0 for name in bridge.SOURCE_NAMES}

    collector.update_joints("left", _joint_message("left", 1))
    assert collector.counts == {
        "left_joints": 1,
        "right_joints": 0,
        "left_gripper": 0,
        "right_gripper": 0,
    }
    collector.update_gripper("right", _gripper_message(1))
    assert collector.counts == {
        "left_joints": 1,
        "right_joints": 0,
        "left_gripper": 0,
        "right_gripper": 1,
    }
    collector.update_joints("left", _joint_message("left", 2))
    assert collector.counts["left_joints"] == 2
    assert collector.counts["right_joints"] == 0


def test_concurrent_four_source_snapshot_keeps_data_and_metadata_atomic(monkeypatch) -> None:
    bridge = _load_bridge_module(monkeypatch)
    clock = FakeClock()
    collector = bridge.ReadonlyStateCollector(
        _robot_config(), monotonic_ns=clock.monotonic_ns, wall_time_ns=clock.wall_time_ns
    )
    start = threading.Barrier(5)
    first_updates = threading.Barrier(5)
    resume_updates = threading.Event()

    def update_joints(side: str) -> None:
        start.wait()
        for generation in range(1, 201):
            collector.update_joints(side, _joint_message(side, generation))
            if generation == 1:
                first_updates.wait()
                resume_updates.wait()
            time.sleep(0)

    def update_gripper(side: str) -> None:
        start.wait()
        for generation in range(1, 201):
            collector.update_gripper(side, _gripper_message(generation))
            if generation == 1:
                first_updates.wait()
                resume_updates.wait()
            time.sleep(0)

    snapshots_checked = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(update_joints, "left"),
            pool.submit(update_joints, "right"),
            pool.submit(update_gripper, "left"),
            pool.submit(update_gripper, "right"),
        ]
        start.wait()
        first_updates.wait()
        try:
            _assert_snapshot_matches_generations(
                _snapshot(collector), bridge.SOURCE_NAMES, clock.wall_base_ns
            )
            snapshots_checked += 1
        finally:
            resume_updates.set()

        while not all(future.done() for future in futures):
            snapshot = _snapshot(collector)
            if snapshot.packet is None:
                time.sleep(0)
                continue
            _assert_snapshot_matches_generations(snapshot, bridge.SOURCE_NAMES, clock.wall_base_ns)
            snapshots_checked += 1

        for future in futures:
            future.result()

    assert snapshots_checked > 0


def test_sender_skips_stale_skew_and_not_advanced_with_explicit_counters(monkeypatch) -> None:
    bridge = _load_bridge_module(monkeypatch)
    clock = FakeClock()
    collector = bridge.ReadonlyStateCollector(
        _robot_config(), monotonic_ns=clock.monotonic_ns, wall_time_ns=clock.wall_time_ns
    )
    sock = FakeSocket(clock)
    sender = _make_sender(bridge, collector, sock)

    _update_all_sources(collector, generation=1)
    assert sender.attempt_send()
    assert not sender.attempt_send()

    clock.set_ms(10)
    _update_all_sources(collector, generation=2)
    clock.set_ms(70)
    assert not sender.attempt_send()

    clock.set_ms(100)
    collector.update_joints("left", _joint_message("left", 3))
    clock.set_ms(105)
    collector.update_joints("right", _joint_message("right", 3))
    clock.set_ms(110)
    collector.update_gripper("left", _gripper_message(3))
    clock.set_ms(125)
    collector.update_gripper("right", _gripper_message(3))
    assert not sender.attempt_send()

    assert sender.counters.sent == 1
    assert sender.counters.skipped_packets() == {
        "total": 3,
        "not_ready": 0,
        "stale": 1,
        "skew": 1,
        "not_advanced": 1,
    }

    clock.set_ms(130)
    _update_all_sources(collector, generation=4)
    assert sender.attempt_send()
    assert sender.counters.sent == 2
    assert sender.last_sent_generations == {name: 4 for name in bridge.SOURCE_NAMES}
    assert len(sock.sent) == 2


def test_sender_runs_at_30_hz_with_fake_clock_and_fresh_sources(monkeypatch) -> None:
    bridge = _load_bridge_module(monkeypatch)
    clock = FakeClock()
    collector = bridge.ReadonlyStateCollector(
        _robot_config(), monotonic_ns=clock.monotonic_ns, wall_time_ns=clock.wall_time_ns
    )
    sock = FakeSocket(clock)
    sender = _make_sender(bridge, collector, sock)
    _update_all_sources(collector, generation=1)

    def sleep_and_update(seconds: float) -> None:
        clock.advance(seconds)
        next_generation = collector.counts["left_joints"] + 1
        _update_all_sources(collector, generation=next_generation)

    sender.run(
        should_stop=lambda: False,
        count=30,
        monotonic_ns=clock.monotonic_ns,
        sleep=sleep_and_update,
    )

    send_times = [sent_at_ns for sent_at_ns, _payload, _target in sock.sent]
    assert send_times == [round(index * 1_000_000_000 / 30) for index in range(30)]
    assert send_times[-1] < 1_000_000_000
    assert sender.counters.attempted == 30
    assert sender.counters.sent == 30
    assert sender.counters.skipped_total == 0
    generations = [
        json.loads(payload)["source_timing"]["sources"]["left_joints"]["generation"]
        for _sent_at_ns, payload, _target in sock.sent
    ]
    assert generations == list(range(1, 31))


def test_ros_executor_thread_shutdown_joins_non_daemon_thread(monkeypatch) -> None:
    bridge = _load_bridge_module(monkeypatch)

    class BlockingExecutor:
        def __init__(self):
            self.started = threading.Event()
            self.stopped = threading.Event()
            self.shutdown_timeout_s = None

        def spin(self) -> None:
            self.started.set()
            self.stopped.wait()

        def shutdown(self, *, timeout_sec: float) -> bool:
            self.shutdown_timeout_s = timeout_sec
            self.stopped.set()
            return True

    executor = BlockingExecutor()
    runner = bridge.RosExecutorThread(executor)
    runner.start()
    assert executor.started.wait(timeout=1.0)
    assert runner.thread.is_alive()
    assert not runner.thread.daemon
    thread_ident = runner.thread.ident

    runner.stop(timeout_s=0.5)

    assert executor.shutdown_timeout_s == 0.5
    assert not runner.thread.is_alive()
    assert all(thread.ident != thread_ident for thread in threading.enumerate())


def test_state_subscription_qos_is_explicit_latest_reliable_volatile(monkeypatch) -> None:
    bridge = _load_bridge_module(monkeypatch)

    qos = bridge.state_subscription_qos()

    assert qos.history == "keep_last"
    assert qos.depth == 1
    assert qos.reliability == "reliable"
    assert qos.durability == "volatile"
