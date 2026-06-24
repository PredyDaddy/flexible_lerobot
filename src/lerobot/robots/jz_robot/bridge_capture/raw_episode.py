from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class RawEpisodeWriter:
    def __init__(self, *, root: str | Path, episode_id: str, metadata: dict[str, Any]):
        self.root = Path(root)
        self.episode_id = str(episode_id)
        self.metadata = dict(metadata)
        self.tmp_dir = self.root / f"episode_{self.episode_id}.tmp"
        self.final_dir = self.root / f"episode_{self.episode_id}"
        self._events_file = None
        self._samples: list[dict[str, Any]] = []

    def start(self) -> Path:
        if self.final_dir.exists():
            raise FileExistsError(f"episode already exists: {self.final_dir}")
        self.tmp_dir.mkdir(parents=True, exist_ok=False)
        (self.tmp_dir / "frames").mkdir()
        self._events_file = (self.tmp_dir / "events.jsonl").open("a", encoding="utf-8")
        self._write_json(self.tmp_dir / "metadata.json", self.metadata)
        return self.tmp_dir

    def write_event(self, event: str, payload: dict[str, Any] | None = None) -> None:
        if self._events_file is None:
            raise RuntimeError("episode writer is not started")
        row = {"time_ns": time.time_ns(), "event": event, "payload": payload or {}}
        self._events_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._events_file.flush()

    def write_sample(self, sample: dict[str, Any]) -> None:
        self._samples.append(dict(sample))

    def finish(self) -> Path:
        if self._events_file is not None:
            self._events_file.close()
            self._events_file = None
        self._write_samples()
        self.tmp_dir.rename(self.final_dir)
        return self.final_dir

    def abort(self) -> None:
        if self._events_file is not None:
            self._events_file.close()
            self._events_file = None

    def _write_samples(self) -> None:
        try:
            import pandas as pd

            pd.DataFrame(self._samples).to_parquet(self.tmp_dir / "samples.parquet", index=False)
        except Exception:
            with (self.tmp_dir / "samples.jsonl").open("w", encoding="utf-8") as stream:
                for sample in self._samples:
                    stream.write(json.dumps(sample, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
