from __future__ import annotations

from .config import VectorTopicConfig
from .types import VectorSnapshot


class ActionAggregator:
    """Aggregates multiple command topics into one fixed-order LeRobot action vector."""

    def __init__(self, config: VectorTopicConfig):
        self._config = config
        self._sources = {source.id: source for source in config.sources}
        self._values: dict[str, float] = {}
        self._source_time_ns = 0
        self._receive_time_ns = 0

    def update(
        self,
        *,
        source_id: str,
        names: list[str],
        values: list[float],
        source_time_ns: int,
        receive_time_ns: int,
    ) -> None:
        if source_id not in self._sources:
            raise KeyError(f"unknown action source: {source_id}")
        source = self._sources[source_id]
        incoming = {name: float(values[index]) for index, name in enumerate(names) if index < len(values)}
        for name in source.exported_names:
            if name in incoming:
                self._values[name] = incoming[name]
        self._source_time_ns = max(self._source_time_ns, int(source_time_ns))
        self._receive_time_ns = max(self._receive_time_ns, int(receive_time_ns))

    def snapshot(self) -> VectorSnapshot | None:
        if any(name not in self._values for name in self._config.names):
            return None
        return VectorSnapshot(
            names=list(self._config.names),
            values=[self._values[name] for name in self._config.names],
            source_time_ns=self._source_time_ns,
            receive_time_ns=self._receive_time_ns,
        )
