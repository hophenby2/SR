"""Visible-cue-triggered routes stored outside neural policy weights."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .memory import EpisodeMemory
from .protocol import Action
from .rollout import shield_action_toward
from .sim import ActionLike, coerce_action
from .vision import VisionObservation


@dataclass(frozen=True, slots=True)
class RouteArtifact:
    route_id: str
    scenario: str
    cue: Mapping[str, Any]
    trigger_lead: int
    actions: tuple[Action, ...]
    decision_interval: int
    source: Mapping[str, Any]
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class RouteLibraryArtifact:
    library_id: str
    scenario: str
    memory_ids: tuple[int, ...]
    source: Mapping[str, Any]
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class RouteControllerConfig:
    shield: bool = False
    shield_horizon: int = 3
    exhaustion: str = "hold_last"
    route_origin: str = "episode"

    def __post_init__(self) -> None:
        if self.shield_horizon <= 0:
            raise ValueError("shield_horizon must be positive")
        if self.exhaustion not in {"hold_last", "neutral", "error"}:
            raise ValueError("exhaustion must be hold_last, neutral, or error")
        if self.route_origin not in {"episode", "trigger"}:
            raise ValueError("route_origin must be episode or trigger")


def _bounds(value: Any, names: tuple[str, str, str, str]) -> tuple[float, float, float, float]:
    if isinstance(value, Mapping):
        return tuple(float(value[name]) for name in names)  # type: ignore[return-value]
    if not isinstance(value, Sequence) or len(value) != 4:
        raise ValueError("cue bounds must contain four coordinates")
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def semantic_cue_matches(cue: Mapping[str, Any], visible: VisionObservation) -> bool:
    """Match a declarative cue using only the latest delayed semantic raster."""

    if cue.get("kind") == "semantic_signature":
        return semantic_signature(cue, visible) is not None
    if cue.get("kind") != "semantic_roi_mass":
        raise ValueError("unsupported route cue kind")
    frames = np.asarray(visible.global_frames)
    if frames.ndim != 4 or len(frames) == 0:
        raise ValueError("visible global frames must have [time, channel, height, width]")
    channel = int(cue.get("channel", 0))
    if not 0 <= channel < frames.shape[1]:
        raise ValueError("cue channel is outside the semantic raster")
    left, right, bottom, top = _bounds(
        cue.get("world_bounds", (-192.0, 192.0, -224.0, 224.0)),
        ("left", "right", "bottom", "top"),
    )
    roi_left, roi_right, roi_bottom, roi_top = _bounds(
        cue["roi"], ("left", "right", "bottom", "top"),
    )
    xs = np.linspace(left, right, frames.shape[-1], dtype=np.float32)
    ys = np.linspace(bottom, top, frames.shape[-2], dtype=np.float32)
    columns = (xs >= roi_left) & (xs <= roi_right)
    rows = (ys >= roi_bottom) & (ys <= roi_top)
    if not np.any(rows) or not np.any(columns):
        raise ValueError("cue ROI does not intersect the semantic raster")
    mass = float(np.asarray(frames[-1, channel][np.ix_(rows, columns)], dtype=np.float32).sum())
    minimum = float(cue.get("minimum_mass", 0.0))
    maximum = float(cue.get("maximum_mass", float("inf")))
    return np.isfinite(mass) and minimum <= mass <= maximum


def semantic_signature(
    cue: Mapping[str, Any],
    visible: VisionObservation,
) -> np.ndarray | None:
    """Pool a delayed semantic frame into a normalized observable cue vector."""

    frames = np.asarray(visible.global_frames, dtype=np.float32)
    if frames.ndim != 4 or len(frames) == 0:
        raise ValueError("visible global frames must have [time, channel, height, width]")
    latest = frames[-1]
    trigger_channel = int(cue.get("trigger_channel", 0))
    if not 0 <= trigger_channel < latest.shape[0]:
        raise ValueError("signature trigger channel is outside the semantic raster")
    mass = float(latest[trigger_channel].sum(dtype=np.float64))
    if not np.isfinite(mass) or mass < float(cue.get("minimum_mass", 0.0)):
        return None
    channels = tuple(int(value) for value in cue.get("channels", (0, 1, 2)))
    if not channels or any(value < 0 or value >= latest.shape[0] for value in channels):
        raise ValueError("signature channels are outside the semantic raster")
    pooled_height = int(cue.get("pooled_height", 14))
    pooled_width = int(cue.get("pooled_width", 12))
    height, width = latest.shape[-2:]
    if (
        pooled_height <= 0
        or pooled_width <= 0
        or height % pooled_height != 0
        or width % pooled_width != 0
    ):
        raise ValueError("signature pool shape must divide the semantic raster")
    selected = latest[np.asarray(channels)]
    pooled = selected.reshape(
        len(channels),
        pooled_height,
        height // pooled_height,
        pooled_width,
        width // pooled_width,
    ).mean(axis=(2, 4))
    vector = pooled.reshape(-1).astype(np.float32, copy=False)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else vector


def load_route_artifact(path: str | Path) -> RouteArtifact:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or int(value.get("schema_version", 0)) != 1:
        raise ValueError("unsupported external route artifact")
    actions = tuple(coerce_action(item) for item in value.get("actions", ()))
    if not actions:
        raise ValueError("external route must contain at least one action")
    interval = int(value.get("decision_interval", 0))
    if interval <= 0:
        raise ValueError("route decision_interval must be positive")
    cue = value.get("cue")
    source = value.get("source", {})
    if not isinstance(cue, Mapping) or not isinstance(source, Mapping):
        raise ValueError("route cue and source must be objects")
    return RouteArtifact(
        route_id=str(value["route_id"]),
        scenario=str(value["scenario"]),
        cue=dict(cue),
        trigger_lead=int(value.get("trigger_lead", 0)),
        actions=actions,
        decision_interval=interval,
        source=dict(source),
    )


def load_route_library_artifact(path: str | Path) -> RouteLibraryArtifact:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or int(value.get("schema_version", 0)) != 1:
        raise ValueError("unsupported external route-library artifact")
    raw_ids = value.get("memory_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError("external route library must contain memory_ids")
    memory_ids = tuple(int(value) for value in raw_ids)
    if any(value <= 0 for value in memory_ids) or len(set(memory_ids)) != len(memory_ids):
        raise ValueError("external route library memory_ids must be unique and positive")
    source = value.get("source", {})
    if not isinstance(source, Mapping):
        raise ValueError("external route library source must be an object")
    return RouteLibraryArtifact(
        library_id=str(value["library_id"]),
        scenario=str(value["scenario"]),
        memory_ids=memory_ids,
        source=dict(source),
    )


def validate_memory_route(artifact: RouteArtifact, memory: EpisodeMemory) -> None:
    if memory.scenario != artifact.scenario:
        raise ValueError("route artifact and SQLite memory scenario differ")
    if memory.cue != artifact.cue:
        raise ValueError("route artifact and SQLite memory cue differ")
    if memory.trigger_lead != artifact.trigger_lead:
        raise ValueError("route artifact and SQLite memory trigger lead differ")
    if tuple(coerce_action(item) for item in memory.route) != artifact.actions:
        raise ValueError("route artifact and SQLite memory actions differ")


class ExternalRouteController:
    """Replay one visible-cue-triggered route using only a local decision index."""

    def __init__(
        self,
        memory: EpisodeMemory,
        *,
        config: RouteControllerConfig = RouteControllerConfig(),
        fallback: ActionLike = Action(),
    ) -> None:
        self.memory = memory
        self.config = config
        self.fallback = coerce_action(fallback)
        self.actions = tuple(coerce_action(item) for item in memory.route)
        if not self.actions:
            raise ValueError("route memory must contain at least one action")
        if not isinstance(memory.cue, Mapping):
            raise ValueError("route memory cue must be a mapping")
        self.reset()

    def reset(self) -> None:
        self.triggered = False
        self.trigger_decision: int | None = None
        self.trigger_source_frame: int | None = None
        self.decision_index = 0
        self.decisions = 0
        self.overrides = 0

    def _next_route_action(self) -> Action:
        if self.decision_index < len(self.actions):
            action = self.actions[self.decision_index]
        elif self.config.exhaustion == "hold_last":
            action = self.actions[-1]
        elif self.config.exhaustion == "neutral":
            action = Action()
        else:
            raise RuntimeError("external route is exhausted")
        self.decision_index += 1
        return action

    def select(self, visible: VisionObservation, *, environment: Any | None = None) -> Action:
        if not self.triggered and semantic_cue_matches(self.memory.cue, visible):
            self.triggered = True
            self.trigger_decision = self.decisions
            self.trigger_source_frame = int(visible.source_frame)
            if self.config.route_origin == "episode":
                # Full-episode memories are indexed by this controller's own
                # decisions.  A delayed cue must not shift the remembered
                # timeline or require an environment/script frame number.
                self.decision_index = self.decisions
        preferred = self._next_route_action() if self.triggered else self.fallback
        selected = preferred
        if self.config.shield:
            if environment is None:
                raise ValueError("shielded external routes require an environment")
            selected = shield_action_toward(
                environment, preferred, horizon=self.config.shield_horizon,
            )
        self.decisions += 1
        self.overrides += int(selected != preferred)
        return selected

    def __call__(self, environment, visible, _teacher_plan, _memory) -> Action:
        return self.select(visible, environment=environment)


class ExternalRouteLibraryController:
    """Select a full-episode route by nearest delayed semantic signature."""

    def __init__(
        self,
        memories: Sequence[EpisodeMemory],
        *,
        config: RouteControllerConfig = RouteControllerConfig(shield=False),
        fallback: ActionLike = Action(),
    ) -> None:
        self.memories = tuple(memories)
        if not self.memories:
            raise ValueError("route library must contain at least one memory")
        scenarios = {memory.scenario for memory in self.memories}
        if len(scenarios) != 1:
            raise ValueError("route library memories must share one scenario")
        for memory in self.memories:
            if not isinstance(memory.cue, Mapping) or memory.cue.get("kind") != "semantic_signature":
                raise ValueError("route library memories require semantic_signature cues")
            expected = semantic_signature_size(memory.cue)
            vector = np.asarray(memory.cue.get("vector"), dtype=np.float32)
            if vector.shape != (expected,) or not np.all(np.isfinite(vector)):
                raise ValueError("route memory has an invalid semantic signature vector")
            if not memory.route:
                raise ValueError("route library memory has no actions")
        self.config = config
        self.fallback = coerce_action(fallback)
        self.reset()

    def reset(self) -> None:
        self.selected_memory: EpisodeMemory | None = None
        self.selection_decision: int | None = None
        self.selection_source_frame: int | None = None
        self.actions: tuple[Action, ...] = ()
        self.decision_index = 0
        self.decisions = 0
        self.overrides = 0

    def _try_select(self, visible: VisionObservation) -> None:
        if self.selected_memory is not None:
            return
        query = semantic_signature(self.memories[0].cue, visible)
        if query is None:
            return
        selected = min(
            self.memories,
            key=lambda memory: (
                float(np.linalg.norm(query - np.asarray(memory.cue["vector"], dtype=np.float32))),
                -memory.confidence,
                memory.id,
            ),
        )
        self.selected_memory = selected
        self.selection_decision = self.decisions
        self.selection_source_frame = int(visible.source_frame)
        self.actions = tuple(coerce_action(value) for value in selected.route)
        self.decision_index = self.decisions if self.config.route_origin == "episode" else 0

    def _next_action(self) -> Action:
        if self.decision_index < len(self.actions):
            action = self.actions[self.decision_index]
        elif self.config.exhaustion == "hold_last":
            action = self.actions[-1]
        elif self.config.exhaustion == "neutral":
            action = Action()
        else:
            raise RuntimeError("external route is exhausted")
        self.decision_index += 1
        return action

    def select(self, visible: VisionObservation, *, environment: Any | None = None) -> Action:
        self._try_select(visible)
        preferred = self._next_action() if self.selected_memory is not None else self.fallback
        selected = preferred
        if self.config.shield:
            if environment is None:
                raise ValueError("shielded external routes require an environment")
            selected = shield_action_toward(
                environment, preferred, horizon=self.config.shield_horizon,
            )
        self.decisions += 1
        self.overrides += int(selected != preferred)
        return selected

    def __call__(self, environment, visible, _teacher_plan, _memory) -> Action:
        return self.select(visible, environment=environment)


def semantic_signature_size(cue: Mapping[str, Any]) -> int:
    channels = tuple(cue.get("channels", (0, 1, 2)))
    return len(channels) * int(cue.get("pooled_height", 14)) * int(cue.get("pooled_width", 12))


__all__ = [
    "ExternalRouteController",
    "ExternalRouteLibraryController",
    "RouteArtifact",
    "RouteLibraryArtifact",
    "RouteControllerConfig",
    "load_route_artifact",
    "load_route_library_artifact",
    "semantic_cue_matches",
    "semantic_signature",
    "semantic_signature_size",
    "validate_memory_route",
]
