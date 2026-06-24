import json
from pathlib import Path

from lerobot.robots.jz_robot.bridge_capture.raw_episode import RawEpisodeWriter


def test_raw_episode_writer_writes_metadata_events_and_samples(tmp_path: Path):
    writer = RawEpisodeWriter(
        root=tmp_path,
        episode_id="000001",
        metadata={"task": "pick cube", "sample_rate_hz": 20},
    )
    writer.start()
    writer.write_event("start", {"operator": "test"})
    writer.write_sample(
        {
            "sample_index": 0,
            "capture_time_ns": 1_000,
            "state": [0.1, 0.2],
            "action": [0.3, 0.4],
            "valid": True,
            "invalid_reason": "",
        }
    )
    finished = writer.finish()

    assert finished.name == "episode_000001"
    assert (finished / "metadata.json").exists()
    assert (finished / "events.jsonl").exists()
    assert (finished / "samples.parquet").exists() or (finished / "samples.jsonl").exists()

    metadata = json.loads((finished / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["task"] == "pick cube"

    events = (finished / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(events[0])["event"] == "start"
