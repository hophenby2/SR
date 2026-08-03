"""Efficient delayed semantic vision for streaming live-engine policies."""

from __future__ import annotations

from collections import deque
import math
from typing import Any, Mapping

import numpy as np

from .adapters import adapt_engine_observation
from .vision import SemanticRasterizer, VisionConfig, VisionObservation


def _episode_frame(observation: Mapping[str, Any]) -> int | None:
    value = observation.get("episode_frame")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def controller_observation(
    delayed: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose delayed hazards while keeping the player's own pose current."""

    result = dict(delayed)
    result.pop("performance", None)
    player = current.get("player")
    if isinstance(player, Mapping):
        result["player"] = dict(player)
        result["own_player_observation_delay"] = 0
        result["own_player_observation_frame"] = _episode_frame(current)
    return result


def _visible_positions(snapshot: Mapping[str, Any]) -> dict[Any, tuple[float, float]]:
    result: dict[Any, tuple[float, float]] = {}
    for ordinal, threat in enumerate(snapshot.get("threats", ())):
        if not isinstance(threat, Mapping) or not bool(threat.get("visible", True)):
            continue
        key = threat.get("id", ("ordinal", ordinal))
        try:
            x = float(threat.get("x", 0.0))
            y = float(threat.get("y", 0.0))
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            result[key] = (x, y)
    return result


class EngineStreamVision:
    """Render one latest visible frame per decision from live observations.

    Raw observations are cheap to queue every engine frame. Rasterization is
    deferred until a policy decision, so a three-frame action hold performs
    one raster pass instead of three. Motion still comes from consecutive
    delayed engine frames, and the player's current pose is never delayed.
    """

    def __init__(
        self,
        config: VisionConfig = VisionConfig(history=1),
        rasterizer: SemanticRasterizer | None = None,
    ) -> None:
        self.config = config
        self.rasterizer = rasterizer or SemanticRasterizer(config)
        self._raw: deque[Mapping[str, Any]] = deque(
            maxlen=config.observation_delay + 2,
        )

    def reset(self, observation: Mapping[str, Any]) -> VisionObservation:
        self._raw.clear()
        capacity = self._raw.maxlen or 2
        self._raw.extend(dict(observation) for _ in range(capacity))
        return self.observe()

    def push(self, observation: Mapping[str, Any]) -> None:
        if not self._raw:
            raise RuntimeError("engine stream vision is not initialized")
        self._raw.append(dict(observation))

    def raw_observation(self) -> dict[str, Any]:
        if len(self._raw) < 2:
            raise RuntimeError("engine stream vision is not initialized")
        values = tuple(self._raw)
        delayed = values[-(self.config.observation_delay + 1)]
        current = values[-1]
        return controller_observation(delayed, current)

    def observe(self) -> VisionObservation:
        if len(self._raw) < 2:
            raise RuntimeError("engine stream vision is not initialized")
        values = tuple(self._raw)
        delayed_index = -(self.config.observation_delay + 1)
        delayed = values[delayed_index]
        previous_delayed = values[delayed_index - 1]
        current = values[-1]
        visible_raw = controller_observation(delayed, current)
        previous_raw = controller_observation(previous_delayed, current)
        visible = adapt_engine_observation(visible_raw)
        previous = adapt_engine_observation(previous_raw)
        visible_positions = _visible_positions(visible)
        previous_positions = _visible_positions(previous)
        current_frame = _episode_frame(delayed)
        previous_frame = _episode_frame(previous_delayed)
        elapsed = (
            current_frame - previous_frame
            if current_frame is not None
            and previous_frame is not None
            and current_frame > previous_frame
            else 1
        )
        motion = {
            key: (
                (position[0] - previous_positions[key][0]) / elapsed,
                (position[1] - previous_positions[key][1]) / elapsed,
            )
            if key in previous_positions else (0.0, 0.0)
            for key, position in visible_positions.items()
        }
        global_frame, local_frame = self.rasterizer.render(visible, motion=motion)
        source_frame = current_frame
        if source_frame is None:
            source_frame = max(0, len(values) - self.config.observation_delay - 1)
        return VisionObservation(
            global_frames=np.expand_dims(global_frame, axis=0),
            local_frames=np.expand_dims(local_frame, axis=0),
            source_frame=source_frame,
        )


__all__ = ["EngineStreamVision", "controller_observation"]
