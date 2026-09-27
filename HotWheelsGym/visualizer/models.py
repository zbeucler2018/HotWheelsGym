"""Serializable semantic models for the RAM observation debugger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class VisualizerSnapshot:
    """One coherent policy-observation and geometry sample."""

    revision: int
    framebuffer: np.ndarray
    track: str
    controlled_slot: int
    raw_frame: int
    state_name: str
    observation_names: tuple[str, ...]
    observation_values: tuple[float, ...]
    observation_groups: Mapping[str, tuple[dict[str, Any], ...]]
    geometry: Mapping[str, Any]
    racers: tuple[Mapping[str, Any], ...]
    nearby_order: tuple[int, ...]
    history: Mapping[str, tuple[float, ...]]
    metadata: Mapping[str, Any]

    @property
    def named_observation(self) -> dict[str, float]:
        return dict(zip(self.observation_names, self.observation_values))

    def payload(self, *, frame_url: str = "/api/frame") -> dict[str, Any]:
        """Return the JSON-safe browser payload; framebuffer stays a binary route."""

        return {
            "revision": self.revision,
            "frame_url": f"{frame_url}?revision={self.revision}",
            "track": self.track,
            "controlled_slot": self.controlled_slot,
            "raw_frame": self.raw_frame,
            "state_name": self.state_name,
            "observation": {
                "names": list(self.observation_names),
                "values": list(self.observation_values),
                "named": self.named_observation,
                "groups": {
                    name: list(entries)
                    for name, entries in self.observation_groups.items()
                },
            },
            "geometry": dict(self.geometry),
            "racers": [dict(racer) for racer in self.racers],
            "nearby_order": list(self.nearby_order),
            "history": {name: list(values) for name, values in self.history.items()},
            "metadata": dict(self.metadata),
        }
