"""Persistent visual phase cues kept outside policy weights.

The provider deliberately consumes only delayed semantic rasters.  It stores
the first visible horizontal Boss #3 motion so a window-mode policy does not
lose the attack's opening phase when that frame leaves its visual history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .training import Demonstrations
from .vision import VisionObservation


@dataclass(frozen=True, slots=True)
class CueMemoryConfig:
    """Raster coordinates and thresholds for the observable Boss #3 cue."""

    memory_size: int = 4
    world_bottom: float = -224.0
    world_top: float = 224.0
    boss_roi_bottom: float = 80.0
    boss_roi_top: float = 144.0
    minimum_motion: float = 1e-4
    minimum_occupancy: float = 1e-4

    def __post_init__(self) -> None:
        if self.memory_size < 4:
            raise ValueError("cue memory requires at least four slots")
        if self.world_top <= self.world_bottom:
            raise ValueError("world_top must be greater than world_bottom")
        if not self.world_bottom <= self.boss_roi_bottom < self.boss_roi_top <= self.world_top:
            raise ValueError("boss ROI must lie inside the configured world bounds")
        if self.minimum_motion < 0.0 or self.minimum_occupancy < 0.0:
            raise ValueError("cue thresholds cannot be negative")


def _scenario_identity(scenario_key: str, memory_size: int) -> np.ndarray:
    vector = np.zeros(memory_size, dtype=np.float32)
    normalized = scenario_key.lower()
    if "boss3" in normalized:
        vector[0] = 1.0
    elif "boss4" in normalized:
        vector[1] = 1.0
    return vector


class StatefulCueMemoryProvider:
    """Capture once, then persist an observable Boss #3 opening-phase cue.

    Slot 2 stores horizontal direction (-1 or +1), and slot 3 stores the
    absolute occupancy-weighted horizontal velocity.  Both remain zero until
    the first qualifying delayed raster is observed.  Boss #4 never captures
    this cue and therefore retains its ordinary scenario identity vector.
    """

    def __init__(self, config: CueMemoryConfig = CueMemoryConfig()) -> None:
        self.config = config
        self._scenario_key: str | None = None
        self._memory = np.zeros(config.memory_size, dtype=np.float32)
        self._captured = False

    @property
    def captured(self) -> bool:
        return self._captured

    @property
    def vector(self) -> np.ndarray:
        return self._memory.copy()

    def reset(self, scenario_key: str, visible: VisionObservation) -> None:
        """Start one episode without inspecting hidden environment state."""

        self._validate_global_frames(visible.global_frames)
        self._scenario_key = str(scenario_key)
        self._memory = _scenario_identity(self._scenario_key, self.config.memory_size)
        self._captured = False

    def __call__(self, scenario_key: str, visible: VisionObservation) -> np.ndarray:
        normalized = str(scenario_key)
        if self._scenario_key is None:
            raise RuntimeError("cue memory provider must be reset at the start of an episode")
        if normalized != self._scenario_key:
            raise RuntimeError("scenario changed without resetting cue memory provider")

        global_frames = self._validate_global_frames(visible.global_frames)
        if not self._captured and "boss3" in normalized.lower():
            motion = self._opening_horizontal_motion(global_frames[-1])
            if motion is not None:
                self._memory[2] = np.float32(np.copysign(1.0, motion))
                self._memory[3] = np.float32(min(1.0, abs(motion)))
                self._captured = True
        return self._memory.copy()

    @staticmethod
    def _validate_global_frames(value: np.ndarray) -> np.ndarray:
        frames = np.asarray(value)
        if frames.ndim != 4 or frames.shape[0] < 1 or frames.shape[1] < 2:
            raise ValueError("global_frames must have [time, channel, height, width] with two channels")
        if frames.shape[2] < 2 or frames.shape[3] < 1:
            raise ValueError("global raster must contain at least two rows and one column")
        return frames

    def _opening_horizontal_motion(self, frame: np.ndarray) -> float | None:
        height = frame.shape[-2]
        world_y = np.linspace(
            self.config.world_bottom,
            self.config.world_top,
            height,
            dtype=np.float32,
        )
        rows = (world_y >= self.config.boss_roi_bottom) & (world_y <= self.config.boss_roi_top)
        occupancy = np.asarray(frame[0, rows], dtype=np.float32)
        horizontal_velocity = np.asarray(frame[1, rows], dtype=np.float32)
        mass = float(occupancy.sum(dtype=np.float64))
        if mass <= self.config.minimum_occupancy:
            return None
        motion = float(
            np.sum(occupancy * horizontal_velocity, dtype=np.float64) / mass
        )
        if not np.isfinite(motion) or abs(motion) <= self.config.minimum_motion:
            return None
        return motion


ScenarioByEpisode = str | Mapping[int, str]


def _infer_episode_scenario(demonstrations: Demonstrations, sample: int) -> str:
    if demonstrations.memory is None or demonstrations.memory.shape[-1] < 2:
        raise ValueError("scenario_by_episode is required when scenario identity memory is unavailable")
    identity = demonstrations.memory[sample, -1, :2]
    if identity[0] > identity[1] and identity[0] > 0.0:
        return "stage5_boss3"
    if identity[1] > identity[0] and identity[1] > 0.0:
        return "stage5_boss4"
    raise ValueError("cannot infer scenario identity from demonstration memory")


def cue_condition_demonstrations(
    demonstrations: Demonstrations,
    *,
    scenario_by_episode: ScenarioByEpisode | None = None,
    config: CueMemoryConfig = CueMemoryConfig(),
) -> Demonstrations:
    """Rebuild online cue memory in archive order without pre-cue backfill.

    Episode IDs are copied unchanged.  Samples for each episode must be one
    contiguous block because that is the only ordering with online semantics.
    When ``scenario_by_episode`` is omitted, slots 0/1 of existing memory are
    used solely to recover the public scenario identity.
    """

    demonstrations.validate()
    if demonstrations.episode_ids is None:
        raise ValueError("episode_ids are required to reconstruct stateful cue memory")

    provider = StatefulCueMemoryProvider(config)
    samples, steps = demonstrations.actions.shape
    memory = np.empty((samples, steps, config.memory_size), dtype=np.float32)
    seen: set[int] = set()
    current_episode: int | None = None
    scenario_key = ""

    for sample in range(samples):
        episode_id = int(demonstrations.episode_ids[sample])
        visible = VisionObservation(
            global_frames=demonstrations.global_frames[sample],
            local_frames=demonstrations.local_frames[sample],
            # The provider intentionally ignores source_frame.  A constant
            # makes accidental timer coupling visible in parity tests.
            source_frame=0,
        )
        if episode_id != current_episode:
            if episode_id in seen:
                raise ValueError("samples for each episode_id must be contiguous")
            seen.add(episode_id)
            current_episode = episode_id
            if isinstance(scenario_by_episode, str):
                scenario_key = scenario_by_episode
            elif scenario_by_episode is not None:
                try:
                    scenario_key = scenario_by_episode[episode_id]
                except KeyError as error:
                    raise ValueError(f"missing scenario for episode_id {episode_id}") from error
            else:
                scenario_key = _infer_episode_scenario(demonstrations, sample)
            provider.reset(scenario_key, visible)
        vector = provider(scenario_key, visible)
        memory[sample] = np.broadcast_to(vector, (steps, config.memory_size))

    result = Demonstrations(
        global_frames=demonstrations.global_frames,
        local_frames=demonstrations.local_frames,
        actions=demonstrations.actions,
        risks=demonstrations.risks,
        memory=memory,
        episode_ids=demonstrations.episode_ids.copy(),
        supervision_mask=demonstrations.supervision_mask,
    )
    result.validate()
    return result


__all__ = [
    "CueMemoryConfig",
    "ScenarioByEpisode",
    "StatefulCueMemoryProvider",
    "cue_condition_demonstrations",
]
