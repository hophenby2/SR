"""Deterministic space-time risk planning for bullet-hell movement.

The planner consumes either an observation or an environment exposing
``forecast_threats``.  It deliberately relies on duck typing so the same code
can be used with the standalone simulator and with snapshots exported by the
LuaSTG bridge.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, replace
import math
from typing import Any, Iterable

import numpy as np

from .protocol import Action


def _field(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _float(value: Any, *names: str, default: float = 0.0) -> float:
    result = _field(value, *names, default=default)
    return float(result if result is not None else default)


def _observation(source: Any, observation: Any | None) -> Any:
    if observation is not None:
        return observation
    observe = getattr(source, "observe", None)
    return observe(include_semantic=False) if callable(observe) else source


def _parse_bounds(raw: Any) -> tuple[float, float, float, float]:
    if isinstance(raw, Mapping) or hasattr(raw, "left"):
        return (
            _float(raw, "left", "xmin", "l", "pl", default=-192.0),
            _float(raw, "right", "xmax", "r", "pr", default=192.0),
            _float(raw, "bottom", "ymin", "b", "pb", default=-224.0),
            _float(raw, "top", "ymax", "t", "pt", default=224.0),
        )
    if len(raw) != 4:
        raise ValueError("bounds must contain left, right, bottom, and top")
    return tuple(float(item) for item in raw)  # type: ignore[return-value]


def _bounds(value: Any) -> tuple[float, float, float, float]:
    raw = _field(value, "bounds", default=None)
    if raw is None:
        raw = _field(value, "world", default=(-192.0, 192.0, -224.0, 224.0))
    return _parse_bounds(raw)


def _navigation_bounds(value: Any, source: Any) -> tuple[float, float, float, float]:
    explicit = _field(value, "player_bounds", default=_field(source, "player_bounds", default=None))
    if explicit is not None:
        return _parse_bounds(explicit)
    world = _field(value, "world", default=_field(source, "world", default=None))
    if world is not None:
        # THlib/player/player_system.lua uses these fixed center-position
        # margins, independently from the player's collision radius.
        return (
            _float(world, "pl", "l", default=-192.0) + 8.0,
            _float(world, "pr", "r", default=192.0) - 8.0,
            _float(world, "pb", "b", default=-224.0) + 16.0,
            _float(world, "pt", "t", default=224.0) - 32.0,
        )
    return _bounds(value)


@dataclass(frozen=True, slots=True)
class RiskConfig:
    horizon_frames: int = 120
    sample_every: int = 4
    cell_size: float = 8.0
    reaction_frames: int = 6
    proximity_margin: float = 20.0
    proximity_decay: float = 10.0
    uncertainty_per_frame: float = 0.025
    uncertainty_margin: float = 0.0
    collision_risk: float = 100.0
    proximity_weight: float = 4.0
    warning_weight: float = 0.35
    boundary_margin: float = 12.0
    boundary_weight: float = 0.2
    safety_thresholds: tuple[float, ...] = (0.2, 0.8, 2.5, 8.0)

    def __post_init__(self) -> None:
        if self.horizon_frames <= 0 or self.sample_every <= 0:
            raise ValueError("forecast horizon and sample interval must be positive")
        if self.cell_size <= 0.0 or self.proximity_decay <= 0.0:
            raise ValueError("grid size and proximity decay must be positive")
        if self.reaction_frames < 0:
            raise ValueError("reaction_frames cannot be negative")
        if (
            not math.isfinite(self.uncertainty_per_frame)
            or not math.isfinite(self.uncertainty_margin)
            or self.uncertainty_per_frame < 0.0
            or self.uncertainty_margin < 0.0
        ):
            raise ValueError("uncertainty margins must be finite and non-negative")
        if any(b <= a for a, b in zip(self.safety_thresholds, self.safety_thresholds[1:])):
            raise ValueError("safety thresholds must be strictly increasing")


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    risk: RiskConfig = RiskConfig()
    maximum_level: int | None = None
    allow_diagonal: bool = True
    cumulative_weight: float = 1.0
    distance_weight: float = 1.0
    cache_layers: bool = True
    max_cached_layers: int = 4096
    safe_region_level: int = 0

    def __post_init__(self) -> None:
        if self.max_cached_layers <= 0:
            raise ValueError("max_cached_layers must be positive")
        if self.safe_region_level < 0:
            raise ValueError("safe_region_level cannot be negative")


@dataclass(frozen=True, slots=True)
class ConnectedRegion:
    time_index: int
    region_id: int
    safety_level: int
    cell_count: int
    centroid: tuple[float, float]
    minimum_risk: float
    maximum_risk: float


@dataclass(frozen=True, slots=True)
class RiskField:
    """Risk and discrete danger levels in ``[time, y, x]`` order."""

    risk: np.ndarray
    levels: np.ndarray
    frames: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    bounds: tuple[float, float, float, float]
    player_radius: float
    player_speed: float
    focus_speed: float
    sample_every: int

    def __post_init__(self) -> None:
        if self.risk.ndim != 3 or self.levels.shape != self.risk.shape:
            raise ValueError("risk and levels must have matching [time, y, x] shapes")
        if self.risk.shape != (len(self.frames), len(self.ys), len(self.xs)):
            raise ValueError("grid axes do not match risk shape")

    def index_of(self, point: Sequence[float]) -> tuple[int, int]:
        x, y = float(point[0]), float(point[1])
        ix = int(round((x - float(self.xs[0])) * (len(self.xs) - 1) / float(self.xs[-1] - self.xs[0])))
        iy = int(round((y - float(self.ys[0])) * (len(self.ys) - 1) / float(self.ys[-1] - self.ys[0])))
        ix = min(max(ix, 0), len(self.xs) - 1)
        iy = min(max(iy, 0), len(self.ys) - 1)
        return iy, ix

    def point_of(self, index: Sequence[int]) -> tuple[float, float]:
        iy, ix = int(index[0]), int(index[1])
        return float(self.xs[ix]), float(self.ys[iy])

    def risk_at(self, point: Sequence[float], time_index: int = 0) -> float:
        iy, ix = self.index_of(point)
        return float(self.risk[time_index, iy, ix])

    def level_at(self, point: Sequence[float], time_index: int = 0) -> int:
        iy, ix = self.index_of(point)
        return int(self.levels[time_index, iy, ix])

    def regions(self, time_index: int, maximum_level: int = 1) -> tuple[ConnectedRegion, ...]:
        labels, count = connected_components(self.levels[time_index] <= maximum_level)
        result: list[ConnectedRegion] = []
        for region_id in range(count):
            cells = np.argwhere(labels == region_id)
            points_x = self.xs[cells[:, 1]]
            points_y = self.ys[cells[:, 0]]
            values = self.risk[time_index][labels == region_id]
            cell_levels = self.levels[time_index][labels == region_id]
            result.append(ConnectedRegion(
                time_index=time_index,
                region_id=region_id,
                safety_level=int(cell_levels.max(initial=0)),
                cell_count=len(cells),
                centroid=(float(points_x.mean()), float(points_y.mean())),
                minimum_risk=float(values.min()),
                maximum_risk=float(values.max()),
            ))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class PlanStep:
    time_index: int
    frame: int
    position: tuple[float, float]
    risk: float
    safety_level: int
    region_id: int | None


@dataclass(frozen=True, slots=True)
class PlanResult:
    field: RiskField
    steps: tuple[PlanStep, ...]
    actions: tuple[Action, ...]
    reached_goal: bool
    peak_level: int
    total_risk: float
    distance: float

    @property
    def first_action(self) -> Action:
        return self.actions[0] if self.actions else Action()

    @property
    def start_risk(self) -> float:
        return self.steps[0].risk if self.steps else math.inf

    @property
    def waypoints(self) -> tuple[tuple[float, float], ...]:
        return tuple(step.position for step in self.steps)


def classify_safety(risk: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    """Return danger levels where zero is safest and larger is worse."""

    values = np.asarray(risk, dtype=np.float64)
    limits = np.asarray(tuple(thresholds), dtype=np.float64)
    if limits.ndim != 1 or np.any(np.diff(limits) <= 0.0):
        raise ValueError("thresholds must be a strictly increasing sequence")
    return np.searchsorted(limits, values, side="right").astype(np.uint8)


def connected_components(mask: np.ndarray, *, diagonal: bool = False) -> tuple[np.ndarray, int]:
    """Label a 2D boolean mask deterministically in row-major order."""

    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2:
        raise ValueError("connected-components input must be two-dimensional")
    labels = np.full(source.shape, -1, dtype=np.int32)
    neighbors = ((-1, 0), (0, -1), (0, 1), (1, 0))
    if diagonal:
        neighbors += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    height, width = source.shape
    count = 0
    for sy in range(height):
        for sx in range(width):
            if not source[sy, sx] or labels[sy, sx] >= 0:
                continue
            labels[sy, sx] = count
            queue = [(sy, sx)]
            head = 0
            while head < len(queue):
                y, x = queue[head]
                head += 1
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and source[ny, nx] and labels[ny, nx] < 0:
                        labels[ny, nx] = count
                        queue.append((ny, nx))
            count += 1
    return labels, count


def _grid_axis(low: float, high: float, cell_size: float) -> np.ndarray:
    count = max(2, int(math.floor((high - low) / cell_size)) + 1)
    return np.linspace(low, high, count, dtype=np.float64)


def _current_threats(observation: Any) -> tuple[Any, ...]:
    threats = _field(observation, "threats", "entities", default=())
    return tuple(threats or ())


def _forecast_offsets(config: RiskConfig) -> tuple[int, ...]:
    offsets = list(range(config.sample_every, config.horizon_frames + 1, config.sample_every))
    if not offsets or offsets[-1] != config.horizon_frames:
        offsets.append(config.horizon_frames)
    return tuple(offsets)


def _forecast(
    source: Any,
    observation: Any,
    config: RiskConfig,
    future_override: tuple[tuple[int, tuple[Any, ...]], ...] | None = None,
) -> tuple[tuple[int, ...], tuple[tuple[Any, ...], ...]]:
    initial = _current_threats(observation)
    if future_override is not None:
        return (
            (0,) + tuple(item[0] for item in future_override),
            (initial,) + tuple(item[1] for item in future_override),
        )
    swept_method = getattr(source, "forecast_swept_threats", None)
    if callable(swept_method):
        future = tuple(swept_method(config.horizon_frames, config.sample_every))
        offsets = tuple(int(item[0]) for item in future)
        if any(offset <= 0 for offset in offsets) or any(
            second <= first for first, second in zip(offsets, offsets[1:])
        ):
            raise ValueError("forecast offsets must be positive and strictly increasing")
        return (0,) + offsets, (initial,) + tuple(tuple(item[1]) for item in future)

    method = getattr(source, "forecast_threats", None)
    if callable(method):
        future = tuple(tuple(frame) for frame in method(config.horizon_frames, config.sample_every))
        offsets = _forecast_offsets(config)[:len(future)]
        return (0,) + offsets, (initial,) + future

    result: list[tuple[Any, ...]] = [initial]
    for frame in _forecast_offsets(config):
        predicted: list[dict[str, Any]] = []
        for threat in initial:
            predicted.append({
                "x": _float(threat, "x") + _float(threat, "vx") * frame
                + 0.5 * _float(threat, "ax") * frame * frame,
                "y": _float(threat, "y") + _float(threat, "vy") * frame
                + 0.5 * _float(threat, "ay") * frame * frame,
                "vx": _float(threat, "vx") + _float(threat, "ax") * frame,
                "vy": _float(threat, "vy") + _float(threat, "ay") * frame,
                "radius_x": _float(threat, "radius_x", "a", "radius", default=2.0),
                "radius_y": _float(threat, "radius_y", "b", "radius", default=2.0),
                "angle": _float(threat, "angle"),
                "danger": _float(threat, "danger", "weight", default=1.0),
                "uncertainty": _float(threat, "uncertainty"),
                "lethal": bool(_field(threat, "lethal", default=True)),
                "warning": bool(_field(threat, "warning", default=False)),
            })
        result.append(tuple(predicted))
    return (0,) + _forecast_offsets(config), tuple(result)


def _add_threat_risk(
    target: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    threat: Any,
    player_radius: float,
    frame_offset: int,
    config: RiskConfig,
    previous: Any | None,
) -> None:
    if not bool(_field(threat, "lethal", default=True)) and not bool(_field(threat, "warning", default=False)):
        return
    x = _float(threat, "x")
    y = _float(threat, "y")
    angle = _float(threat, "angle")
    radius_x = max(0.1, _float(threat, "radius_x", "a", "radius", default=2.0))
    radius_y = max(0.1, _float(threat, "radius_y", "b", "radius", default=2.0))
    speed = math.hypot(_float(threat, "vx"), _float(threat, "vy"))
    uncertainty = max(0.0, _float(threat, "uncertainty"))
    uncertainty += config.uncertainty_margin + frame_offset * config.uncertainty_per_frame
    reaction = min(config.proximity_margin, speed * config.reaction_frames)
    inflate = player_radius + uncertainty
    weight = max(0.0, _float(threat, "danger", "weight", default=1.0))
    lethal = bool(_field(threat, "lethal", default=True))
    if bool(_field(threat, "warning", default=False)):
        weight *= config.warning_weight

    cos_a, sin_a = math.cos(angle), math.sin(angle)
    dx, dy = grid_x - x, grid_y - y
    local_x = cos_a * dx + sin_a * dy
    local_y = -sin_a * dx + cos_a * dy
    normalized = np.sqrt(
        (local_x / (radius_x + inflate)) ** 2
        + (local_y / (radius_y + inflate)) ** 2
    )
    clearance = np.maximum(0.0, (normalized - 1.0) * min(radius_x + inflate, radius_y + inflate))
    proximity = np.exp(-clearance / config.proximity_decay)
    proximity *= config.proximity_weight * weight
    proximity[clearance > config.proximity_margin + reaction] = 0.0
    target += proximity
    if lethal:
        target[normalized <= 1.0] += config.collision_risk * weight

    # Fill the gap between sampled positions so fast bullets cannot tunnel
    # through an apparently safe time slice.
    if previous is not None:
        px, py = _float(previous, "x"), _float(previous, "y")
        segment_x, segment_y = x - px, y - py
        length_sq = segment_x * segment_x + segment_y * segment_y
        if length_sq > 1e-9:
            t = np.clip(((grid_x - px) * segment_x + (grid_y - py) * segment_y) / length_sq, 0.0, 1.0)
            swept_distance = np.hypot(grid_x - (px + t * segment_x), grid_y - (py + t * segment_y))
            swept_radius = max(radius_x, radius_y) + inflate
            swept_clearance = np.maximum(0.0, swept_distance - swept_radius)
            swept = np.exp(-swept_clearance / config.proximity_decay) * config.proximity_weight * weight
            swept[swept_clearance > config.proximity_margin + reaction] = 0.0
            target[:] = np.maximum(target, swept)
            if lethal:
                target[swept_distance <= swept_radius] += config.collision_risk * weight


def build_risk_field(
    source: Any,
    *,
    observation: Any | None = None,
    config: RiskConfig = RiskConfig(),
    layer_cache: MutableMapping[Any, Any] | None = None,
    forecast_override: tuple[tuple[int, tuple[Any, ...]], ...] | None = None,
) -> RiskField:
    snapshot = _observation(source, observation)
    player = _field(snapshot, "player", default=_field(source, "player", default=None))
    if player is None:
        raise ValueError("planning source is missing player state")
    left, right, bottom, top = _navigation_bounds(snapshot, source)
    radius = max(0.0, _float(player, "radius", "hitbox", default=2.0))
    if right <= left or top <= bottom:
        raise ValueError("player movement bounds leave no navigable playfield")
    xs = _grid_axis(left, right, config.cell_size)
    ys = _grid_axis(bottom, top, config.cell_size)
    grid_x, grid_y = np.meshgrid(xs, ys)
    frame_offsets, forecast = _forecast(
        source,
        snapshot,
        config,
        future_override=forecast_override,
    )
    frames = np.asarray(frame_offsets, dtype=np.int32)
    risk = np.zeros((len(forecast), len(ys), len(xs)), dtype=np.float64)

    def render_layer(threats: tuple[Any, ...], layer_config: RiskConfig, frame_offset: int) -> np.ndarray:
        target = np.zeros((len(ys), len(xs)), dtype=np.float64)
        previous_by_id: dict[Any, Any] = {}
        for ordinal, threat in enumerate(threats):
            threat_id = _field(threat, "id", default=("ordinal", ordinal))
            _add_threat_risk(
                target,
                grid_x,
                grid_y,
                threat,
                radius,
                frame_offset,
                layer_config,
                previous_by_id.get(threat_id),
            )
            previous_by_id[threat_id] = threat
        return target

    scenario = _field(source, "scenario", default=None)
    scenario_key = _field(
        scenario,
        "scenario_key",
        default=_field(scenario, "name", default=type(scenario).__name__),
    )
    base_frame = int(_field(snapshot, "frame", default=0))
    low_uncertainty_config = replace(config, uncertainty_per_frame=0.0)
    high_uncertainty_config = replace(
        low_uncertainty_config,
        uncertainty_margin=(
            config.uncertainty_margin
            + config.horizon_frames * config.uncertainty_per_frame
        ),
    )

    for time_index, threats in enumerate(forecast):
        # The current layer is cheap and unique to this decision.  Future
        # interval geometry is cached at zero/max horizon uncertainty and
        # linearly interpolated, retaining the original horizon-dependent
        # safety margin without rerasterizing every overlapping window.
        if layer_cache is None or time_index == 0:
            risk[time_index] = render_layer(threats, config, int(frames[time_index]))
            continue

        absolute_frame = base_frame + int(frames[time_index])
        interval_start = base_frame + int(frames[time_index - 1])
        dynamic_uncertainty = config.uncertainty_per_frame > 0.0
        cache_key = (
            "uncertainty_pair" if dynamic_uncertainty else "exact",
            scenario_key,
            int(_field(source, "seed", default=0)),
            absolute_frame,
            interval_start,
            low_uncertainty_config,
            high_uncertainty_config.uncertainty_margin if dynamic_uncertainty else None,
            (left, right, bottom, top),
            radius,
        )
        cached = layer_cache.get(cache_key)
        if cached is None:
            low = render_layer(threats, low_uncertainty_config, 0)
            if dynamic_uncertainty:
                high = render_layer(threats, high_uncertainty_config, 0)
                cached = (low, high)
            else:
                cached = low
            layer_cache[cache_key] = cached

        if dynamic_uncertainty:
            low, high = cached
            alpha = min(1.0, int(frames[time_index]) / config.horizon_frames)
            risk[time_index] = low + alpha * (high - low)
        else:
            risk[time_index] = cached

    edge_distance = np.minimum.reduce((grid_x - left, right - grid_x, grid_y - bottom, top - grid_y))
    edge_risk = np.clip((config.boundary_margin - edge_distance) / max(config.boundary_margin, 1e-6), 0.0, 1.0)
    risk += config.boundary_weight * edge_risk[None, :, :]
    levels = classify_safety(risk, config.safety_thresholds)
    return RiskField(
        risk=risk.astype(np.float32),
        levels=levels,
        frames=frames,
        xs=xs.astype(np.float32),
        ys=ys.astype(np.float32),
        bounds=(left, right, bottom, top),
        player_radius=radius,
        player_speed=max(0.1, _float(player, "speed", "move_speed", default=4.0)),
        focus_speed=max(0.1, _float(player, "focus_speed", "slow_speed", default=2.0)),
        sample_every=config.sample_every,
    )


def _movement_transitions(
    field: RiskField,
    diagonal: bool,
    frame_count: int,
) -> tuple[tuple[float, float, Action], ...]:
    candidates: list[tuple[float, float, Action]] = [(0.0, 0.0, Action(slow=True))]
    for slow, speed in ((True, field.focus_speed), (False, field.player_speed)):
        reach = speed * frame_count
        for move_y in (-1, 0, 1):
            for move_x in (-1, 0, 1):
                if move_x == 0 and move_y == 0:
                    continue
                if not diagonal and move_x != 0 and move_y != 0:
                    continue
                normalization = math.sqrt(2.0) if move_x != 0 and move_y != 0 else 1.0
                intended_x = move_x * reach / normalization
                intended_y = move_y * reach / normalization
                action = Action(move_x=move_x, move_y=move_y, slow=slow)
                candidates.append((intended_x, intended_y, action))
    return tuple(sorted(candidates, key=lambda item: (
        math.hypot(item[0], item[1]), item[1], item[0], item[2].discrete,
    )))


def _line_cells(start: tuple[int, int], end: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    y0, x0 = start
    y1, x1 = end
    # Half-cell DDA is conservative enough for a grid risk field while keeping
    # the inner planner loop free of thousands of NumPy reductions.
    count = max(1, 2 * max(abs(y1 - y0), abs(x1 - x0)))
    return tuple(dict.fromkeys(
        (
            int(round(y0 + (y1 - y0) * index / count)),
            int(round(x0 + (x1 - x0) * index / count)),
        )
        for index in range(count + 1)
    ))


class SpatioTemporalPlanner:
    """Plan player-region motion using a lexicographic danger objective."""

    def __init__(self, config: PlannerConfig = PlannerConfig()) -> None:
        self.config = config
        self._layer_cache: dict[Any, Any] = {}
        self._timeline_key: tuple[Any, ...] | None = None
        self._timeline_environment: Any | None = None
        self._timeline_frames: dict[int, tuple[Any, ...]] = {}

    @staticmethod
    def _compress_timeline_interval(
        snapshots: Iterable[tuple[Any, ...]],
    ) -> tuple[Any, ...]:
        traces: dict[Any, list[Any]] = {}
        for threats in snapshots:
            for ordinal, threat in enumerate(threats):
                threat_id = _field(threat, "id", default=("ordinal", ordinal))
                traces.setdefault(threat_id, []).append(threat)
        result: list[Any] = []
        for threat_id in sorted(traces, key=str):
            trace = traces[threat_id]
            selected = {0, len(trace) - 1}
            selected.add(max(
                range(len(trace)),
                key=lambda index: (
                    _float(trace[index], "radius", "radius_x", "a", default=2.0),
                    _float(trace[index], "danger", "weight", default=1.0),
                    -index,
                ),
            ))
            result.extend(trace[index] for index in sorted(selected))
        return tuple(result)

    def _timeline_forecast(
        self,
        source: Any,
        snapshot: Any,
    ) -> tuple[tuple[int, tuple[Any, ...]], ...] | None:
        scenario = _field(source, "scenario", default=None)
        if not bool(_field(scenario, "forecast_independent_of_player", default=False)):
            return None
        clone_method = getattr(source, "clone", None)
        if not callable(clone_method):
            return None

        current = int(_field(snapshot, "frame", default=0))
        key = (
            _field(scenario, "scenario_key", default=_field(scenario, "name", default=None)),
            int(_field(source, "seed", default=0)),
            int(_field(scenario, "duration_frames", default=0)),
        )
        if (
            self._timeline_key != key
            or self._timeline_environment is None
            or current not in self._timeline_frames
            or (current == 0 and int(_field(self._timeline_environment, "frame", default=0)) > 0)
        ):
            self._timeline_key = key
            self._timeline_environment = clone_method()
            self._timeline_frames = {current: _current_threats(snapshot)}

        duration = int(_field(scenario, "duration_frames", default=current + self.config.risk.horizon_frames))
        target = min(current + self.config.risk.horizon_frames, duration)
        timeline = self._timeline_environment
        while int(_field(timeline, "frame", default=0)) < target:
            if bool(_field(timeline, "done", default=False)):
                break
            result = timeline._advance(Action(), build_semantic=False, detect_collision=False)
            self._timeline_frames[int(_field(timeline, "frame", default=0))] = tuple(result.observation.threats)

        available_target = min(target, int(_field(timeline, "frame", default=target)))
        remaining = available_target - current
        if remaining <= 0:
            return ()
        offsets = list(range(self.config.risk.sample_every, remaining + 1, self.config.risk.sample_every))
        if not offsets or offsets[-1] != remaining:
            offsets.append(remaining)
        result: list[tuple[int, tuple[Any, ...]]] = []
        previous = current
        for offset in offsets:
            absolute = current + offset
            snapshots = (
                self._timeline_frames[frame]
                for frame in range(previous, absolute + 1)
                if frame in self._timeline_frames
            )
            result.append((offset, self._compress_timeline_interval(snapshots)))
            previous = absolute

        self._timeline_frames = {
            frame: threats for frame, threats in self._timeline_frames.items() if frame >= current
        }
        return tuple(result)

    def build_field(self, source: Any, *, observation: Any | None = None) -> RiskField:
        scenario = _field(source, "scenario", default=None)
        cacheable = bool(_field(scenario, "forecast_independent_of_player", default=False))
        cache = self._layer_cache if self.config.cache_layers and cacheable else None
        snapshot = _observation(source, observation)
        forecast_override = self._timeline_forecast(source, snapshot) if cacheable else None
        field = build_risk_field(
            source,
            observation=snapshot,
            config=self.config.risk,
            layer_cache=cache,
            forecast_override=forecast_override,
        )
        if cache is not None:
            while len(cache) > self.config.max_cached_layers:
                cache.pop(next(iter(cache)))
        return field

    def clear_cache(self) -> None:
        self._layer_cache.clear()
        self._timeline_key = None
        self._timeline_environment = None
        self._timeline_frames.clear()

    def plan(
        self,
        source: Any,
        *,
        observation: Any | None = None,
        goal: Sequence[float] | None = None,
    ) -> PlanResult:
        snapshot = _observation(source, observation)
        field = self.build_field(source, observation=snapshot)
        player = _field(snapshot, "player", default=_field(source, "player", default=None))
        start = field.index_of((_float(player, "x"), _float(player, "y")))
        goal_index = field.index_of(goal) if goal is not None else None
        maximum_level = self.config.maximum_level

        # Each layer maps a cell to (peak danger, accumulated risk, distance).
        layers: list[dict[tuple[int, int], tuple[int, float, float]]] = []
        positions: list[dict[tuple[int, int], tuple[float, float]]] = []
        parents: list[dict[tuple[int, int], tuple[int, int]]] = [{}]
        parent_actions: list[dict[tuple[int, int], Action]] = [{}]
        start_cost = (int(field.levels[0, start[0], start[1]]), float(field.risk[0, start[0], start[1]]), 0.0)
        layers.append({start: start_cost})
        start_position = (_float(player, "x"), _float(player, "y"))
        positions.append({start: start_position})
        for time_index in range(1, field.risk.shape[0]):
            frame_count = int(field.frames[time_index] - field.frames[time_index - 1])
            transitions = _movement_transitions(field, self.config.allow_diagonal, frame_count)
            current: dict[tuple[int, int], tuple[int, float, float]] = {}
            current_positions: dict[tuple[int, int], tuple[float, float]] = {}
            parent_map: dict[tuple[int, int], tuple[int, int]] = {}
            action_map: dict[tuple[int, int], Action] = {}
            for cell, cost in sorted(layers[-1].items()):
                position = positions[-1][cell]
                for movement_x, movement_y, action in transitions:
                    target_position = (
                        min(max(position[0] + movement_x, field.bounds[0]), field.bounds[1]),
                        min(max(position[1] + movement_y, field.bounds[2]), field.bounds[3]),
                    )
                    target = field.index_of(target_position)
                    swept = _line_cells(cell, target)
                    swept_levels = [
                        max(int(field.levels[time_index - 1, y, x]), int(field.levels[time_index, y, x]))
                        for y, x in swept
                    ]
                    segment_peak = max(swept_levels)
                    if maximum_level is not None and segment_peak > maximum_level:
                        continue
                    segment_risk = sum(
                        max(float(field.risk[time_index - 1, y, x]), float(field.risk[time_index, y, x]))
                        for y, x in swept
                    ) / len(swept)
                    candidate = (
                        max(cost[0], segment_peak),
                        cost[1] + self.config.cumulative_weight * segment_risk * frame_count,
                        cost[2] + self.config.distance_weight * math.hypot(
                            target_position[0] - position[0],
                            target_position[1] - position[1],
                        ),
                    )
                    if target not in current or candidate < current[target]:
                        current[target] = candidate
                        current_positions[target] = target_position
                        parent_map[target] = cell
                        action_map[target] = action
            if not current:
                break
            layers.append(current)
            positions.append(current_positions)
            parents.append(parent_map)
            parent_actions.append(action_map)

        last_time = len(layers) - 1
        terminal_costs = layers[last_time]
        reached_goal = goal_index is not None and goal_index in terminal_costs
        if reached_goal:
            terminal = goal_index
        elif goal_index is not None:
            terminal = min(terminal_costs, key=lambda cell: (
                terminal_costs[cell][0],
                terminal_costs[cell][1],
                math.hypot(cell[0] - goal_index[0], cell[1] - goal_index[1]),
                terminal_costs[cell][2], cell,
            ))
        else:
            # With no explicit goal, survival dominates.  Prefer a low-risk
            # terminal region, then avoid needless displacement.
            terminal = min(terminal_costs, key=lambda cell: (terminal_costs[cell], cell))

        path = [terminal]
        reversed_actions: list[Action] = []
        for time_index in range(last_time, 0, -1):
            reversed_actions.append(parent_actions[time_index][path[-1]])
            path.append(parents[time_index][path[-1]])
        path.reverse()
        reversed_actions.reverse()

        region_labels: list[np.ndarray] = []
        for time_index in range(len(path)):
            labels, _ = connected_components(
                field.levels[time_index] <= self.config.safe_region_level,
            )
            region_labels.append(labels)
        base_frame = int(_field(snapshot, "frame", default=0))
        steps = tuple(
            PlanStep(
                time_index=time_index,
                frame=base_frame + int(field.frames[time_index]),
                # The grid cell is only the merge key.  Preserve the exact
                # endpoint reached by the stored LuaSTG action.
                position=positions[time_index][cell],
                risk=float(field.risk[time_index, cell[0], cell[1]]),
                safety_level=int(field.levels[time_index, cell[0], cell[1]]),
                region_id=(int(region_labels[time_index][cell]) if region_labels[time_index][cell] >= 0 else None),
            )
            for time_index, cell in enumerate(path)
        )
        cost = terminal_costs[terminal]
        return PlanResult(
            field=field,
            steps=steps,
            actions=tuple(reversed_actions),
            reached_goal=reached_goal,
            peak_level=cost[0],
            total_risk=cost[1],
            distance=cost[2],
        )


def plan_region_path(
    source: Any,
    *,
    observation: Any | None = None,
    goal: Sequence[float] | None = None,
    config: PlannerConfig = PlannerConfig(),
) -> PlanResult:
    return SpatioTemporalPlanner(config).plan(source, observation=observation, goal=goal)


__all__ = [
    "ConnectedRegion",
    "PlanResult",
    "PlanStep",
    "PlannerConfig",
    "RiskConfig",
    "RiskField",
    "SpatioTemporalPlanner",
    "build_risk_field",
    "classify_safety",
    "connected_components",
    "plan_region_path",
]
