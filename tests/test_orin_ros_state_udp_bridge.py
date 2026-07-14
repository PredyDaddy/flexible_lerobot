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

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_PATH = REPO_ROOT / "udp_test" / "test_scripts" / "arm_side" / "orin_ros_state_udp_bridge.py"


def _load_bridge_module(monkeypatch):
    rclpy = types.ModuleType("rclpy")
    callback_groups = types.ModuleType("rclpy.callback_groups")

    class MutuallyExclusiveCallbackGroup:
        pass

    callback_groups.MutuallyExclusiveCallbackGroup = MutuallyExclusiveCallbackGroup
    context_module = types.ModuleType("rclpy.context")

    class Context:
        def __init__(self):
            self.initialized = False
            self.shutdown_called = False

        def init(self, *, args=None) -> None:
            self.initialized = True
            self.args = args

        def ok(self) -> bool:
            return self.initialized and not self.shutdown_called

        def shutdown(self) -> None:
            self.shutdown_called = True

    context_module.Context = Context
    executors = types.ModuleType("rclpy.executors")
    executors.ExternalShutdownException = type("ExternalShutdownException", (Exception,), {})

    class SingleThreadedExecutor:
        def __init__(self, *, context=None):
            self.nodes = []
            self.context = context

        def add_node(self, node) -> None:
            self.nodes.append(node)

    executors.SingleThreadedExecutor = SingleThreadedExecutor
    qos = types.ModuleType("rclpy.qos")

    class QoSProfile:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    qos.QoSProfile = QoSProfile
    qos.HistoryPolicy = SimpleNamespace(KEEP_LAST="keep_last")
    qos.ReliabilityPolicy = SimpleNamespace(RELIABLE="reliable", BEST_EFFORT="best_effort")
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
        "rclpy.callback_groups": callback_groups,
        "rclpy.context": context_module,
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


class FailingSocket:
    def sendto(self, _payload: bytes, _target: tuple[str, int]) -> int:
        raise OSError("simulated route failure")


def _robot_config() -> SimpleNamespace:
    return SimpleNamespace(
        left_joint_names=["left_joint1", "left_joint2"],
        right_joint_names=["right_joint1", "right_joint2"],
        use_gripper=True,
        left_joint_state_topic="/left/joints",
        right_joint_state_topic="/right/joints",
        left_gripper_state_topic="/left/gripper",
        right_gripper_state_topic="/right/gripper",
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
    assert "'left_joints': ['left_joint2']" in details

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

    assert collector.counts == dict.fromkeys(bridge.SOURCE_NAMES, 0)

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


def test_worker_updates_preserve_callback_timing_and_atomic_metadata(monkeypatch) -> None:
    bridge = _load_bridge_module(monkeypatch)
    clock = FakeClock(monotonic_ns=100_000_000)
    collector = bridge.ReadonlyStateCollector(
        _robot_config(), monotonic_ns=clock.monotonic_ns, wall_time_ns=clock.wall_time_ns
    )
    updates = (
        ("left_joints", (("left_joint1", "left_joint2"), (1.0, 2.0)), 91_000_000, 191, 101),
        ("right_joints", (("right_joint1", "right_joint2"), (3.0, 4.0)), 92_000_000, 192, 102),
        ("left_gripper", (5.0, 6.0), 93_000_000, 193, None),
        ("right_gripper", (7.0, 8.0), 94_000_000, 194, None),
    )

    for update in updates:
        collector.apply_worker_update(update)

    snapshot = _snapshot(collector)

    assert snapshot.packet is not None
    assert snapshot.generations == dict.fromkeys(bridge.SOURCE_NAMES, 1)
    assert snapshot.packet["joints"]["left"] == {"left_joint1": 1.0, "left_joint2": 2.0}
    assert snapshot.packet["grippers"]["right"] == {"width": 7.0, "force": 8.0}
    assert snapshot.source_timing["sources"]["left_joints"] == {
        "generation": 1,
        "recv_wall_ns": 191,
        "recv_monotonic_ns": 91_000_000,
        "header_stamp_ns": 101,
        "age_ms": 9.0,
    }


def test_worker_update_reports_parent_ipc_delay_and_worker_pid(monkeypatch) -> None:
    bridge = _load_bridge_module(monkeypatch)
    clock = FakeClock(monotonic_ns=100_000_000)
    collector = bridge.ReadonlyStateCollector(
        _robot_config(), monotonic_ns=clock.monotonic_ns, wall_time_ns=clock.wall_time_ns
    )

    collector.apply_worker_update(
        ("left_gripper", (5.0, 6.0), 91_000_000, 191, None, 4321)
    )
    snapshot = _snapshot(collector)

    assert snapshot.source_worker_pids["left_gripper"] == 4321
    assert snapshot.source_ipc_delay_ms["left_gripper"] == 9.0


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
    assert sender.last_sent_generations == dict.fromkeys(bridge.SOURCE_NAMES, 4)
    assert len(sock.sent) == 2


def test_sender_logs_gripper_stall_sources_and_recovery_without_relaxing_gate(monkeypatch) -> None:
    bridge = _load_bridge_module(monkeypatch)
    clock = FakeClock()
    collector = bridge.ReadonlyStateCollector(
        _robot_config(), monotonic_ns=clock.monotonic_ns, wall_time_ns=clock.wall_time_ns
    )
    sock = FakeSocket(clock)
    log_lines = []
    sender = _make_sender(
        bridge,
        collector,
        sock,
        monotonic_ns=clock.monotonic_ns,
        wall_time_ns=clock.wall_time_ns,
        printer=lambda message, **_kwargs: log_lines.append(message),
    )

    _update_all_sources(collector, generation=1)
    assert sender.attempt_send()
    clock.set_ms(250)
    collector.update_joints("left", _joint_message("left", 2))
    collector.update_joints("right", _joint_message("right", 2))

    assert not sender.attempt_send()
    assert len(sock.sent) == 1
    assert sender.counters.consecutive_skips == 1
    assert any(
        'stale_sources=["left_gripper","right_gripper"]' in line
        and 'not_advanced_sources=["left_gripper","right_gripper"]' in line
        for line in log_lines
    )

    clock.set_ms(260)
    collector.update_gripper("left", _gripper_message(2))
    collector.update_gripper("right", _gripper_message(2))

    assert sender.attempt_send()
    assert len(sock.sent) == 2
    assert sender.counters.consecutive_skips == 0
    assert sender.counters.max_consecutive_skips == 1
    assert any("recovered_after_skips=1" in line for line in log_lines)


def test_sender_runs_at_30_hz_with_fake_clock_and_fresh_sources(monkeypatch) -> None:
    bridge = _load_bridge_module(monkeypatch)
    clock = FakeClock()
    collector = bridge.ReadonlyStateCollector(
        _robot_config(), monotonic_ns=clock.monotonic_ns, wall_time_ns=clock.wall_time_ns
    )
    sock = FakeSocket(clock)
    log_lines = []
    sender = _make_sender(
        bridge,
        collector,
        sock,
        monotonic_ns=clock.monotonic_ns,
        process_id=1234,
        printer=lambda message, **_kwargs: log_lines.append(message),
    )
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
    assert sender.measured_send_hz == pytest.approx(30.0)
    assert any(
        "rate pid=1234 seq=30 configured_hz=30 measured_send_hz=30.000000 window_packets=30" in line
        for line in log_lines
    )
    generations = [
        json.loads(payload)["source_timing"]["sources"]["left_joints"]["generation"]
        for _sent_at_ns, payload, _target in sock.sent
    ]
    assert generations == list(range(1, 31))


def test_measured_send_rate_uses_30_successful_packet_monotonic_window(monkeypatch) -> None:
    bridge = _load_bridge_module(monkeypatch)
    clock = FakeClock()
    collector = bridge.ReadonlyStateCollector(
        _robot_config(), monotonic_ns=clock.monotonic_ns, wall_time_ns=clock.wall_time_ns
    )
    sender = _make_sender(
        bridge,
        collector,
        FakeSocket(clock),
        monotonic_ns=clock.monotonic_ns,
    )

    for generation in range(1, 31):
        _update_all_sources(collector, generation)
        assert sender.attempt_send()
        if generation < 30:
            clock.advance(0.05)

    assert sender.measured_send_hz == pytest.approx(20.0)


def test_nonzero_joint_header_must_advance_but_zero_stamp_uses_generation(monkeypatch) -> None:
    bridge = _load_bridge_module(monkeypatch)
    clock = FakeClock()
    collector = bridge.ReadonlyStateCollector(
        _robot_config(), monotonic_ns=clock.monotonic_ns, wall_time_ns=clock.wall_time_ns
    )
    sender = _make_sender(bridge, collector, FakeSocket(clock))

    _update_all_sources(collector, 1)
    assert sender.attempt_send()

    repeated_stamp_left = _joint_message("left", 2)
    repeated_stamp_left.header.stamp.nanosec = 1
    repeated_stamp_right = _joint_message("right", 2)
    repeated_stamp_right.header.stamp.nanosec = 1
    collector.update_joints("left", repeated_stamp_left)
    collector.update_joints("right", repeated_stamp_right)
    collector.update_gripper("left", _gripper_message(2))
    collector.update_gripper("right", _gripper_message(2))
    assert not sender.attempt_send()
    assert sender.counters.skipped_not_advanced == 1

    zero_stamp_left = _joint_message("left", 3)
    zero_stamp_left.header.stamp.nanosec = 0
    zero_stamp_right = _joint_message("right", 3)
    zero_stamp_right.header.stamp.nanosec = 0
    collector.update_joints("left", zero_stamp_left)
    collector.update_joints("right", zero_stamp_right)
    collector.update_gripper("left", _gripper_message(3))
    collector.update_gripper("right", _gripper_message(3))
    snapshot = _snapshot(
        collector,
        require_advanced=True,
        last_sent=sender.last_sent_generations,
    )
    assert snapshot.progress_modes["left_joints"] == "generation_zero_stamp_fallback"
    assert snapshot.progress_modes["right_joints"] == "generation_zero_stamp_fallback"
    assert sender.attempt_send()


def test_missing_joint_header_is_encoded_as_zero_stamp_generation_fallback(monkeypatch) -> None:
    bridge = _load_bridge_module(monkeypatch)
    clock = FakeClock()
    collector = bridge.ReadonlyStateCollector(
        _robot_config(), monotonic_ns=clock.monotonic_ns, wall_time_ns=clock.wall_time_ns
    )
    collector.update_joints("left", SimpleNamespace(name=["left_joint1", "left_joint2"], position=[1.0, 2.0]))
    collector.update_joints(
        "right", SimpleNamespace(name=["right_joint1", "right_joint2"], position=[3.0, 4.0])
    )
    collector.update_gripper("left", _gripper_message(1))
    collector.update_gripper("right", _gripper_message(1))

    snapshot = _snapshot(collector)

    assert snapshot.source_timing is not None
    for source_name in ("left_joints", "right_joints"):
        assert snapshot.source_timing["sources"][source_name]["header_stamp_ns"] == 0
        assert snapshot.progress_modes[source_name] == "generation_zero_stamp_fallback"


def test_oversize_udp_payload_warns_but_is_still_sent(monkeypatch) -> None:
    bridge = _load_bridge_module(monkeypatch)
    clock = FakeClock()
    collector = bridge.ReadonlyStateCollector(
        _robot_config(), monotonic_ns=clock.monotonic_ns, wall_time_ns=clock.wall_time_ns
    )
    sock = FakeSocket(clock)
    log_lines = []
    monkeypatch.setattr(
        bridge,
        "encode_state_packet",
        lambda _packet: b"x" * (bridge.COMMON_IPV4_UDP_PAYLOAD_BYTES + 1),
    )
    sender = _make_sender(
        bridge,
        collector,
        sock,
        printer=lambda message, **_kwargs: log_lines.append(message),
    )
    _update_all_sources(collector, 1)

    assert sender.attempt_send()
    assert len(sock.sent) == 1
    assert any(
        "WARNING payload_bytes=1473 exceeds_common_ipv4_udp_payload=1472" in line for line in log_lines
    )


def test_successful_state_send_emits_canonical_audit_event(monkeypatch) -> None:
    bridge = _load_bridge_module(monkeypatch)
    clock = FakeClock()
    collector = bridge.ReadonlyStateCollector(
        _robot_config(), monotonic_ns=clock.monotonic_ns, wall_time_ns=clock.wall_time_ns
    )
    sock = FakeSocket(clock)
    events = []
    sender = _make_sender(
        bridge,
        collector,
        sock,
        monotonic_ns=clock.monotonic_ns,
        audit_sink=lambda event_type, payload: events.append((event_type, payload)) or True,
    )
    _update_all_sources(collector, 1)

    assert sender.attempt_send()

    assert len(events) == 1
    event_type, payload = events[0]
    assert event_type == "state"
    assert payload["packet"]["seq"] == 1
    assert payload["packet"]["source_timing"]["schema_version"] == 1
    assert payload["udp_target_ip"] == "127.0.0.1"
    assert payload["udp_target_port"] == 39010
    assert payload["encoded_bytes"] == len(sock.sent[0][1])
    assert payload["local_record_monotonic_ns"] == clock.monotonic_ns()
    assert "send_completed_monotonic_ns" not in payload


def test_local_state_audit_commits_before_optional_x86_udp_delivery(monkeypatch) -> None:
    bridge = _load_bridge_module(monkeypatch)
    clock = FakeClock()
    collector = bridge.ReadonlyStateCollector(
        _robot_config(), monotonic_ns=clock.monotonic_ns, wall_time_ns=clock.wall_time_ns
    )
    events = []
    logs = []
    sender = _make_sender(
        bridge,
        collector,
        FailingSocket(),
        monotonic_ns=clock.monotonic_ns,
        audit_sink=lambda event_type, payload: events.append((event_type, payload)) or True,
        printer=lambda message, **_kwargs: logs.append(message),
    )
    _update_all_sources(collector, 1)

    assert sender.attempt_send()

    assert len(events) == 1
    assert events[0][0] == "state"
    assert events[0][1]["packet"]["seq"] == 1
    assert sender.counters.recorded == 1
    assert sender.counters.sent == 0
    assert sender.counters.network_errors == 1
    assert sender.last_sent_generations == dict.fromkeys(bridge.SOURCE_NAMES, 1)
    assert any("local_recorded seq=1 udp_delivery_failed=OSError" in line for line in logs)


def test_source_process_manager_shutdown_joins_workers_and_collector_thread(monkeypatch) -> None:
    bridge = _load_bridge_module(monkeypatch)
    collector = bridge.ReadonlyStateCollector(_robot_config())

    class FakeEvent:
        def __init__(self):
            self._event = threading.Event()

        def set(self):
            self._event.set()

        def is_set(self):
            return self._event.is_set()

    class FakeConnection:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeProcess:
        next_pid = 1000

        def __init__(self, *, target, args, name, daemon):
            self.target = target
            self.args = args
            self.name = name
            self.daemon = daemon
            self.pid = None
            self.exitcode = None
            self._alive = False
            self.terminated = False
            self.killed = False

        def start(self):
            self.pid = FakeProcess.next_pid
            FakeProcess.next_pid += 1
            self._alive = True

        def join(self, timeout=None):
            if self.args[2].is_set():
                self._alive = False
                self.exitcode = 0

        def is_alive(self):
            return self._alive

        def terminate(self):
            self.terminated = True
            self._alive = False

        def kill(self):
            self.killed = True
            self._alive = False

    class FakeProcessContext:
        def __init__(self):
            self.processes = []

        def Event(self):
            return FakeEvent()

        def Pipe(self, *, duplex):
            assert not duplex
            return FakeConnection(), FakeConnection()

        def Process(self, **kwargs):
            process = FakeProcess(**kwargs)
            self.processes.append(process)
            return process

    process_context = FakeProcessContext()
    monkeypatch.setattr(
        bridge,
        "wait_for_connections",
        lambda _connections, timeout: time.sleep(timeout) or [],
    )
    specs = bridge.create_source_worker_specs(_robot_config())
    manager = bridge.SourceProcessManager(
        collector,
        specs,
        process_context=process_context,
    )

    manager.start()
    thread_ident = manager.collector_thread.ident

    assert len(manager.processes) == 4
    assert all(process.pid is not None for process in manager.processes)
    assert all(not process.daemon for process in manager.processes)
    assert manager.collector_thread.is_alive()
    assert not manager.collector_thread.daemon

    manager.stop(timeout_s=0.5)

    assert all(not process.is_alive() for process in manager.processes)
    assert all(not process.terminated and not process.killed for process in manager.processes)
    assert not manager.collector_thread.is_alive()
    assert all(thread.ident != thread_ident for thread in threading.enumerate())


def test_state_subscription_qos_is_explicit_latest_best_effort_volatile(monkeypatch) -> None:
    bridge = _load_bridge_module(monkeypatch)

    qos = bridge.state_subscription_qos()

    assert qos.history == "keep_last"
    assert qos.depth == 1
    assert qos.reliability == "best_effort"
    assert qos.durability == "volatile"


def test_four_state_subscriptions_use_distinct_mutually_exclusive_callback_groups(monkeypatch) -> None:
    bridge = _load_bridge_module(monkeypatch)
    calls = []

    def make_node(name: str):
        return SimpleNamespace(
            name=name,
            create_subscription=lambda message_type, topic, callback, qos, **kwargs: calls.append(
                (message_type, topic, callback, qos, kwargs)
            )
            or topic,
        )

    nodes = tuple(make_node(f"jz_readonly_ros_state_udp_bridge_{name}") for name in bridge.SOURCE_NAMES)
    specs = bridge.create_source_worker_specs(_robot_config())
    send_connections = tuple(SimpleNamespace(send=lambda _update: None) for _ in bridge.SOURCE_NAMES)

    handles = tuple(
        bridge.create_source_subscription(node, spec, connection)
        for node, spec, connection in zip(nodes, specs, send_connections, strict=True)
    )

    assert [node.name for node in nodes] == [
        "jz_readonly_ros_state_udp_bridge_left_joints",
        "jz_readonly_ros_state_udp_bridge_right_joints",
        "jz_readonly_ros_state_udp_bridge_left_gripper",
        "jz_readonly_ros_state_udp_bridge_right_gripper",
    ]
    assert tuple(handle.subscription for handle in handles) == (
        "/left/joints",
        "/right/joints",
        "/left/gripper",
        "/right/gripper",
    )
    callback_groups = tuple(handle.callback_group for handle in handles)
    assert len({id(group) for group in callback_groups}) == 4
    assert all(
        isinstance(group, bridge.MutuallyExclusiveCallbackGroup)
        for group in callback_groups
    )
    assert len(calls) == 4
    assert all(call[3].depth == 1 for call in calls)
    assert all(
        call[4] == {"callback_group": callback_groups[index]}
        for index, call in enumerate(calls)
    )


def test_four_source_workers_use_independent_processes(monkeypatch) -> None:
    bridge = _load_bridge_module(monkeypatch)
    robot_cfg = _robot_config()

    specs = bridge.create_source_worker_specs(robot_cfg)

    assert [spec.name for spec in specs] == list(bridge.SOURCE_NAMES)
    assert [spec.topic for spec in specs] == [
        robot_cfg.left_joint_state_topic,
        robot_cfg.right_joint_state_topic,
        robot_cfg.left_gripper_state_topic,
        robot_cfg.right_gripper_state_topic,
    ]
    assert [spec.message_type for spec in specs] == [
        bridge.JointState,
        bridge.JointState,
        bridge.Float64MultiArray,
        bridge.Float64MultiArray,
    ]


def test_generic_bridge_cli_keeps_legacy_20_hz_default(monkeypatch) -> None:
    bridge = _load_bridge_module(monkeypatch)
    monkeypatch.setattr(sys, "argv", [str(BRIDGE_PATH), "--target-ip", "127.0.0.1"])

    args = bridge.parse_args()

    assert args.hz == 20.0
    assert args.executor_threads == 4
