from lerobot.robots.jz_robot.bridge_capture.buffers import SampleBuffer
from lerobot.robots.jz_robot.bridge_capture.types import TimestampedPayload


def test_sample_buffer_selects_latest_sample_not_after_target_within_threshold():
    buffer = SampleBuffer(maxlen=4)
    buffer.append(TimestampedPayload(source_time_ns=1_000, receive_time_ns=1_010, payload="old"))
    buffer.append(TimestampedPayload(source_time_ns=2_000, receive_time_ns=2_010, payload="match"))
    buffer.append(TimestampedPayload(source_time_ns=3_000, receive_time_ns=3_010, payload="future"))

    match = buffer.latest_not_after(target_time_ns=2_020, max_delta_ns=50)

    assert match is not None
    assert match.payload == "match"
    assert match.delta_ns == 20


def test_sample_buffer_returns_none_when_delta_exceeds_threshold():
    buffer = SampleBuffer(maxlen=4)
    buffer.append(TimestampedPayload(source_time_ns=1_000, receive_time_ns=1_010, payload="old"))

    assert buffer.latest_not_after(target_time_ns=2_000, max_delta_ns=100) is None
