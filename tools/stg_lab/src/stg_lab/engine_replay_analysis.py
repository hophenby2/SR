"""Deterministic observation analysis for native THlib replay playback."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .engine import EngineClient, EngineProtocolError
from .engine_mpc import _RegionPhaseMemory, _laser_circle_records
from .engine_play import _observation
from .engine_runtime import verify_runtime_source_fingerprints
from .protocol import Action


_OBJECT_SOURCES = (
    "enemy_bullets",
    "enemies",
    "nontjt_enemies",
    "indestructibles",
)
_LASER_KINDS = frozenset(("straight_laser", "bent_laser"))
_LOCAL_BULLET_RADII = (32.0, 64.0, 96.0)
_SPEED_BUCKETS = (
    ("stationary", 0.5),
    ("0.5_to_1", 1.0),
    ("1_to_2", 2.0),
    ("2_to_3", 3.0),
    ("3_to_4", 4.0),
    ("4_to_6", 6.0),
)
_CLEARANCE_BUCKETS = (
    ("collision", 0.0),
    ("0_to_4", 4.0),
    ("4_to_8", 8.0),
    ("8_to_16", 16.0),
    ("16_to_20", 20.0),
    ("20_to_48", 48.0),
)
_CRC32 = re.compile(r"[0-9a-f]{8}")


@dataclass(frozen=True, slots=True)
class EngineReplayAnalysisConfig:
    max_frames: int = 120_000
    render: bool = False
    render_every: int = 1
    timeline_every: int = 1
    region_grid_cell_size: float = 16.0

    def __post_init__(self) -> None:
        if isinstance(self.max_frames, bool) or not isinstance(self.max_frames, int):
            raise ValueError("max_frames must be an integer")
        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if (
            isinstance(self.render_every, bool)
            or not isinstance(self.render_every, int)
            or not 1 <= self.render_every <= 600
        ):
            raise ValueError("render_every must be an integer in [1, 600]")
        if (
            isinstance(self.timeline_every, bool)
            or not isinstance(self.timeline_every, int)
            or self.timeline_every <= 0
        ):
            raise ValueError("timeline_every must be a positive integer")
        if (
            isinstance(self.region_grid_cell_size, bool)
            or not isinstance(self.region_grid_cell_size, (int, float))
            or not math.isfinite(float(self.region_grid_cell_size))
            or self.region_grid_cell_size <= 0.0
        ):
            raise ValueError("region_grid_cell_size must be finite and positive")


@dataclass(frozen=True, slots=True)
class _CircleThreat:
    source: str
    x: float
    y: float
    radius: float
    region: bool


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _summary(values: Sequence[float]) -> dict[str, Any]:
    cleaned = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if not cleaned.size:
        return {
            "count": 0,
            "minimum": None,
            "p10": None,
            "median": None,
            "mean": None,
            "p90": None,
            "maximum": None,
        }
    percentiles = np.percentile(cleaned, (10.0, 50.0, 90.0))
    return {
        "count": int(cleaned.size),
        "minimum": float(np.min(cleaned)),
        "p10": float(percentiles[0]),
        "median": float(percentiles[1]),
        "mean": float(np.mean(cleaned)),
        "p90": float(percentiles[2]),
        "maximum": float(np.max(cleaned)),
    }


def _fraction_counts(counts: Mapping[Any, int], total: int) -> dict[str, Any]:
    return {
        str(key): {
            "frames": int(value),
            "fraction": (float(value) / total if total else 0.0),
        }
        for key, value in sorted(counts.items(), key=lambda item: str(item[0]))
    }


def _world_bounds(observation: Mapping[str, Any]) -> tuple[float, float, float, float]:
    world = observation.get("world")
    world = world if isinstance(world, Mapping) else {}
    left = _number(world.get("l"))
    right = _number(world.get("r"))
    bottom = _number(world.get("b"))
    top = _number(world.get("t"))
    return (
        -192.0 if left is None else left,
        192.0 if right is None else right,
        -224.0 if bottom is None else bottom,
        224.0 if top is None else top,
    )


def _player_bounds(observation: Mapping[str, Any]) -> tuple[float, float, float, float]:
    world = observation.get("world")
    world = world if isinstance(world, Mapping) else {}
    world_bounds = _world_bounds(observation)
    left = _number(world.get("pl"))
    right = _number(world.get("pr"))
    bottom = _number(world.get("pb"))
    top = _number(world.get("pt"))
    adjusted = (
        (world_bounds[0] if left is None else left) + 8.0,
        (world_bounds[1] if right is None else right) - 8.0,
        (world_bounds[2] if bottom is None else bottom) + 16.0,
        (world_bounds[3] if top is None else top) - 32.0,
    )
    if adjusted[0] >= adjusted[1] or adjusted[2] >= adjusted[3]:
        raise EngineProtocolError("replay observation has invalid player bounds")
    return adjusted


def _record_radius(record: Mapping[str, Any], default: float = 2.0) -> float:
    a = _number(record.get("a"))
    b = _number(record.get("b"))
    a = default if a is None else abs(a)
    b = a if b is None else abs(b)
    if record.get("rect") is True:
        return max(0.1, math.hypot(a, b))
    return max(0.1, a, b)


def _player_record(
    observation: Mapping[str, Any],
) -> tuple[Mapping[str, Any], float, float, float] | None:
    player = observation.get("player")
    if not isinstance(player, Mapping):
        return None
    x = _number(player.get("x"))
    y = _number(player.get("y"))
    if x is None or y is None:
        return None
    return player, x, y, _record_radius(player, 0.5)


def _object_records(observation: Mapping[str, Any], source: str) -> list[Mapping[str, Any]]:
    records = observation.get(source)
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return []
    return [record for record in records if isinstance(record, Mapping)]


def _circle_threats(observation: Mapping[str, Any]) -> tuple[_CircleThreat, ...]:
    result: list[_CircleThreat] = []
    for source in _OBJECT_SOURCES:
        for record in _object_records(observation, source):
            if record.get("kind") in _LASER_KINDS or record.get("collidable", True) is not True:
                continue
            x = _number(record.get("x"))
            y = _number(record.get("y"))
            if x is None or y is None:
                continue
            result.append(_CircleThreat(
                source=source,
                x=x,
                y=y,
                radius=_record_radius(record),
                region=source == "indestructibles",
            ))
    for ordinal, record in enumerate(_object_records(observation, "lasers")):
        if record.get("collidable", True) is not True:
            continue
        for circle in _laser_circle_records(record, ordinal):
            x = _number(circle.get("x"))
            y = _number(circle.get("y"))
            if x is None or y is None:
                continue
            result.append(_CircleThreat(
                source="lasers",
                x=x,
                y=y,
                radius=_record_radius(circle),
                region=False,
            ))
    return tuple(result)


def _minimum_clearance(
    x: float,
    y: float,
    player_radius: float,
    threats: Sequence[_CircleThreat],
) -> tuple[float, str | None]:
    minimum = math.inf
    source = None
    for threat in threats:
        margin = math.hypot(x - threat.x, y - threat.y) - player_radius - threat.radius
        if margin < minimum:
            minimum = margin
            source = threat.source
    return minimum, source


def _clearance_bucket(value: float) -> str:
    if not math.isfinite(value):
        return "no_threat"
    for name, upper in _CLEARANCE_BUCKETS:
        if value <= upper:
            return name
    return "over_48"


def _safety_level(overall: float, nonregion: float, region: float) -> int:
    if overall <= 0.0:
        return 3
    if nonregion < 16.0:
        return 2
    if nonregion < 20.0 or region < 8.0:
        return 1
    return 0


def _speed_bucket(value: float) -> str:
    for name, upper in _SPEED_BUCKETS:
        if value < upper:
            return name
    return "6_or_faster"


def _bullet_frame(
    observation: Mapping[str, Any],
    player: tuple[float, float] | None,
) -> dict[str, Any]:
    records = [
        record
        for record in _object_records(observation, "enemy_bullets")
        if record.get("kind") not in _LASER_KINDS
    ]
    collidable = [record for record in records if record.get("collidable", True) is True]
    speeds: list[float] = []
    speed_buckets: Counter[str] = Counter()
    local_counts = {radius: 0 for radius in _LOCAL_BULLET_RADII}
    for record in collidable:
        speed = _number(record.get("speed"))
        if speed is None:
            vx = _number(record.get("vx"))
            vy = _number(record.get("vy"))
            if vx is not None and vy is not None:
                speed = math.hypot(vx, vy)
        if speed is not None:
            speeds.append(speed)
            speed_buckets[_speed_bucket(speed)] += 1
        if player is not None:
            x = _number(record.get("x"))
            y = _number(record.get("y"))
            if x is not None and y is not None:
                distance = math.hypot(x - player[0], y - player[1])
                for radius in _LOCAL_BULLET_RADII:
                    local_counts[radius] += int(distance <= radius)
    left, right, bottom, top = _world_bounds(observation)
    area = max(1.0, (right - left) * (top - bottom))
    return {
        "visible_count": len(records),
        "collidable_count": len(collidable),
        "screen_density_per_10000": len(collidable) * 10_000.0 / area,
        "local_counts": local_counts,
        "speeds": speeds,
        "speed_buckets": speed_buckets,
    }


def _axis(lower: float, upper: float, cell_size: float) -> np.ndarray:
    values: list[float] = []
    start = lower
    while start < upper:
        stop = min(start + cell_size, upper)
        values.append(0.5 * (start + stop))
        start = stop
    return np.asarray(values, dtype=np.float64)


class _RegionTopologyTrace:
    def __init__(self, cell_size: float) -> None:
        self.cell_size = float(cell_size)
        self.bounds: tuple[float, float, float, float] | None = None
        self.x_axis = np.empty(0, dtype=np.float64)
        self.y_axis = np.empty(0, dtype=np.float64)
        self.grid_x = np.empty((0, 0), dtype=np.float64)
        self.grid_y = np.empty((0, 0), dtype=np.float64)
        self.last_component: frozenset[int] | None = None
        self.zone_id = 0
        self.transitions: list[dict[str, Any]] = []
        self.lateral_transitions: list[dict[str, Any]] = []
        self.last_lateral_side: str | None = None
        self.safe_fractions: list[float] = []
        self.component_sizes: list[float] = []
        self.crossing_frames = 0
        self.crossing_entries = 0
        self.crossing_run = 0
        self.crossing_runs: list[float] = []

    def _set_bounds(self, bounds: tuple[float, float, float, float]) -> None:
        if self.bounds == bounds:
            return
        self.bounds = bounds
        self.x_axis = _axis(bounds[0], bounds[1], self.cell_size)
        self.y_axis = _axis(bounds[2], bounds[3], self.cell_size)
        self.grid_x, self.grid_y = np.meshgrid(self.x_axis, self.y_axis)
        self.last_component = None

    def _component(self, passable: np.ndarray, start: int) -> frozenset[int]:
        height, width = passable.shape
        stack = [start]
        seen = {start}
        while stack:
            current = stack.pop()
            row, column = divmod(current, width)
            for next_row, next_column in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            ):
                if not (0 <= next_row < height and 0 <= next_column < width):
                    continue
                index = next_row * width + next_column
                if index not in seen and passable[next_row, next_column]:
                    seen.add(index)
                    stack.append(index)
        return frozenset(seen)

    def _describe_component(self, component: frozenset[int]) -> dict[str, Any]:
        width = len(self.x_axis)
        rows = np.asarray([index // width for index in component], dtype=np.int64)
        columns = np.asarray([index % width for index in component], dtype=np.int64)
        xs = self.x_axis[columns]
        ys = self.y_axis[rows]
        touches_left = bool(np.any(columns == 0))
        touches_right = bool(np.any(columns == width - 1))
        if touches_left and touches_right:
            side = "spanning"
        elif float(np.max(xs)) < 0.0:
            side = "left"
        elif float(np.min(xs)) > 0.0:
            side = "right"
        else:
            side = "center"
        return {
            "id": self.zone_id,
            "side": side,
            "cells": len(component),
            "centroid_x": float(np.mean(xs)),
            "centroid_y": float(np.mean(ys)),
            "touches_left": touches_left,
            "touches_right": touches_right,
            "touches_bottom": bool(np.any(rows == 0)),
            "touches_top": bool(np.any(rows == len(self.y_axis) - 1)),
        }

    def push(
        self,
        *,
        frame: int,
        bounds: tuple[float, float, float, float],
        player_x: float,
        player_y: float,
        player_radius: float,
        region_threats: Sequence[_CircleThreat],
        player_region_margin: float,
    ) -> dict[str, Any]:
        self._set_bounds(bounds)
        crossing = bool(region_threats) and player_region_margin < 8.0
        if crossing:
            self.crossing_frames += 1
            if self.crossing_run == 0:
                self.crossing_entries += 1
            self.crossing_run += 1
        elif self.crossing_run:
            self.crossing_runs.append(float(self.crossing_run))
            self.crossing_run = 0

        if not region_threats:
            self.safe_fractions.append(1.0)
            return {
                "crossing": False,
                "safe_cell_fraction": 1.0,
                "zone": None,
            }

        margins = np.full(self.grid_x.shape, math.inf, dtype=np.float64)
        for threat in region_threats:
            margins = np.minimum(
                margins,
                np.hypot(self.grid_x - threat.x, self.grid_y - threat.y)
                - player_radius
                - threat.radius,
            )
        passable = margins >= 8.0
        safe_fraction = float(np.mean(passable))
        self.safe_fractions.append(safe_fraction)
        zone = None
        if not crossing and np.any(passable):
            distances = (self.grid_x - player_x) ** 2 + (self.grid_y - player_y) ** 2
            distances = np.where(passable, distances, math.inf)
            start = int(np.argmin(distances))
            component = self._component(passable, start)
            if self.last_component is None:
                self.zone_id += 1
            elif not self.last_component.intersection(component):
                previous = self.zone_id
                self.zone_id += 1
                self.transitions.append({
                    "frame": frame,
                    "from_zone": previous,
                    "to_zone": self.zone_id,
                })
            self.last_component = component
            zone = self._describe_component(component)
            self.component_sizes.append(float(len(component)))
            side = zone["side"]
            if side in {"left", "right"}:
                if self.last_lateral_side is not None and side != self.last_lateral_side:
                    self.lateral_transitions.append({
                        "frame": frame,
                        "from_side": self.last_lateral_side,
                        "to_side": side,
                    })
                self.last_lateral_side = str(side)
        return {
            "crossing": crossing,
            "safe_cell_fraction": safe_fraction,
            "zone": zone,
        }

    def report(self, frames: int) -> dict[str, Any]:
        runs = list(self.crossing_runs)
        if self.crossing_run:
            runs.append(float(self.crossing_run))
        return {
            "definition": (
                "four-connected 16-unit-style grid using current collidable "
                "indestructible max-axis circle covers and 8-unit player clearance"
            ),
            "grid_cell_size": self.cell_size,
            "bounds": self.bounds,
            "grid_width": len(self.x_axis),
            "grid_height": len(self.y_axis),
            "safe_cell_fraction": _summary(self.safe_fractions),
            "player_component_cells": _summary(self.component_sizes),
            "component_switch_count": len(self.transitions),
            "component_switches": self.transitions,
            "lateral_switch_count": len(self.lateral_transitions),
            "lateral_switches": self.lateral_transitions,
            "region_crossing_frames": self.crossing_frames,
            "region_crossing_fraction": self.crossing_frames / frames if frames else 0.0,
            "region_crossing_entries": self.crossing_entries,
            "region_crossing_run_frames": _summary(runs),
        }


class _ReplayTelemetry:
    def __init__(self, config: EngineReplayAnalysisConfig) -> None:
        self.config = config
        self.frames: list[int] = []
        self.timeline: list[dict[str, Any]] = []
        self.x_values: list[float] = []
        self.y_values: list[float] = []
        self.step_distances: list[float] = []
        self.boundary_clearances: list[float] = []
        self.edge_frames: Counter[str] = Counter()
        self.boundary_bands: Counter[str] = Counter()
        self.previous_position: tuple[float, float] | None = None
        self.previous_direction: tuple[int, int] | None = None
        self.direction_changes = 0
        self.moving_to_moving_direction_changes = 0
        self.turns_at_least_90_degrees = 0
        self.turns_over_90_degrees = 0
        self.exact_reversals = 0
        self.movement_starts = 0
        self.movement_stops = 0
        self.direction_run = 0
        self.direction_runs: list[float] = []
        self.slow_frames = 0
        self.previous_slow: bool | None = None
        self.slow_mode_changes = 0
        self.slow_mode_changes_with_direction_change = 0
        self.player_frames = 0

        self.clearance_values: list[float] = []
        self.bullet_clearances: list[float] = []
        self.region_clearances: list[float] = []
        self.clearance_buckets: Counter[str] = Counter()
        self.nearest_sources: Counter[str] = Counter()
        self.safety_levels: Counter[int] = Counter()
        self.safety_transitions: list[dict[str, Any]] = []
        self.previous_safety_level: int | None = None
        self.safety_run = 0
        self.safety_runs: Counter[int] = Counter()

        self.visible_bullet_counts: list[float] = []
        self.collidable_bullet_counts: list[float] = []
        self.screen_densities: list[float] = []
        self.local_bullet_counts = {radius: [] for radius in _LOCAL_BULLET_RADII}
        self.bullet_speed_frame_means: list[float] = []
        self.bullet_speed_frame_p90: list[float] = []
        self.bullet_speed_samples = 0
        self.bullet_speed_sum = 0.0
        self.bullet_speed_maximum: float | None = None
        self.bullet_speed_buckets: Counter[str] = Counter()

        self.phase = _RegionPhaseMemory()
        self.phase_counts: Counter[str] = Counter()
        self.phase_transitions: list[dict[str, Any]] = []
        self.region_radii: list[float] = []
        self.topology = _RegionTopologyTrace(config.region_grid_cell_size)

        self.death_frames = 0
        self.death_entries = 0
        self.previous_death = 0.0
        self.maximum_death = 0.0
        self.protected_frames = 0
        self.life_values: list[float] = []
        self.enemy_counts: list[float] = []
        self.boss_hp_values: list[float] = []
        self.boss_hp_initial: float | None = None
        self.last_enemy_present_frame: int | None = None

    @staticmethod
    def _direction(dx: float, dy: float) -> tuple[int, int]:
        epsilon = 1e-6
        return (
            -1 if dx < -epsilon else (1 if dx > epsilon else 0),
            -1 if dy < -epsilon else (1 if dy > epsilon else 0),
        )

    def _push_direction(self, direction: tuple[int, int]) -> bool:
        previous = self.previous_direction
        changed = previous is not None and direction != previous
        if previous is None:
            self.direction_run = 1
        elif direction == previous:
            self.direction_run += 1
        else:
            self.direction_runs.append(float(self.direction_run))
            self.direction_run = 1
            self.direction_changes += 1
            if direction != (0, 0) and previous != (0, 0):
                self.moving_to_moving_direction_changes += 1
                dot = direction[0] * previous[0] + direction[1] * previous[1]
                self.turns_at_least_90_degrees += int(dot <= 0)
                self.turns_over_90_degrees += int(dot < 0)
                if direction == (-previous[0], -previous[1]):
                    self.exact_reversals += 1
            if direction == (0, 0) and previous != (0, 0):
                self.movement_stops += 1
            elif direction != (0, 0) and previous == (0, 0):
                self.movement_starts += 1
        self.previous_direction = direction
        return changed

    def _push_slow(self, slow: bool, *, direction_changed: bool) -> None:
        if self.previous_slow is not None and slow != self.previous_slow:
            self.slow_mode_changes += 1
            self.slow_mode_changes_with_direction_change += int(direction_changed)
        self.previous_slow = slow

    def _push_player_state(self, observation: Mapping[str, Any]) -> None:
        player = observation.get("player")
        if not isinstance(player, Mapping):
            return
        death = _number(player.get("death")) or 0.0
        protect = _number(player.get("protect")) or 0.0
        self.death_frames += int(death > 0.0)
        self.protected_frames += int(protect > 0.0)
        self.death_entries += int(death > 0.0 and self.previous_death <= 0.0)
        self.previous_death = death
        self.maximum_death = max(self.maximum_death, death)
        resources = observation.get("resources")
        if isinstance(resources, Mapping):
            life = _number(resources.get("lifeleft"))
            if life is not None:
                self.life_values.append(life)

    def _push_outcome(self, observation: Mapping[str, Any], frame: int) -> None:
        enemies = [
            *_object_records(observation, "enemies"),
            *_object_records(observation, "nontjt_enemies"),
        ]
        self.enemy_counts.append(float(len(enemies)))
        if enemies:
            self.last_enemy_present_frame = frame
        candidates: list[tuple[float, float]] = []
        for record in enemies:
            hp = _number(record.get("hp"))
            maximum = _number(record.get("maxhp"))
            # SR uses 999999999 HP for the ten immortal Boss #3 emitters.
            # They share GROUP_ENEMY with the real boss and must not win the
            # maximum-HP selection used for outcome telemetry.
            if hp is not None and maximum is not None and 0.0 < maximum < 1e8:
                candidates.append((maximum, hp))
        if candidates:
            hp = max(candidates)[1]
            self.boss_hp_values.append(hp)
            if self.boss_hp_initial is None:
                self.boss_hp_initial = hp

    def push(self, observation: Mapping[str, Any]) -> None:
        frame = _integer(observation.get("episode_frame"))
        if frame is None:
            raise EngineProtocolError("replay observation has no integer episode_frame")
        if self.frames and frame != self.frames[-1] + 1:
            raise EngineProtocolError("replay observation frames are not contiguous")
        self.frames.append(frame)
        self._push_player_state(observation)
        self._push_outcome(observation, frame)

        player_info = _player_record(observation)
        player_xy = None if player_info is None else (player_info[1], player_info[2])
        bullets = _bullet_frame(observation, player_xy)
        self.visible_bullet_counts.append(float(bullets["visible_count"]))
        self.collidable_bullet_counts.append(float(bullets["collidable_count"]))
        self.screen_densities.append(float(bullets["screen_density_per_10000"]))
        for radius, count in bullets["local_counts"].items():
            self.local_bullet_counts[radius].append(float(count))
        speeds = bullets["speeds"]
        if speeds:
            self.bullet_speed_frame_means.append(float(np.mean(speeds)))
            self.bullet_speed_frame_p90.append(float(np.percentile(speeds, 90.0)))
            self.bullet_speed_samples += len(speeds)
            self.bullet_speed_sum += float(sum(speeds))
            maximum = max(speeds)
            self.bullet_speed_maximum = (
                maximum if self.bullet_speed_maximum is None else max(self.bullet_speed_maximum, maximum)
            )
        self.bullet_speed_buckets.update(bullets["speed_buckets"])

        phase_before = self.phase.phase
        phase_records = [
            record
            for record in _object_records(observation, "indestructibles")
            if record.get("kind") not in _LASER_KINDS and record.get("collidable", True) is True
        ]
        phase_radii = [_record_radius(record) for record in phase_records]
        self.phase.update(frame, phase_radii)
        if phase_radii:
            self.region_radii.append(float(np.median(phase_radii)))
        if self.phase.phase != phase_before:
            self.phase_transitions.append({
                "observed_frame": frame,
                "inferred_start_frame": self.phase.phase_started_frame,
                "from": phase_before,
                "to": self.phase.phase,
                "observed_radius": self.phase.observed_radius,
                "learned_cycle_frames": self.phase.learned_cycle_frames,
            })
        self.phase_counts[self.phase.phase] += 1

        timeline_record: dict[str, Any] = {
            "episode_frame": frame,
            "visible_bullets": bullets["visible_count"],
            "collidable_bullets": bullets["collidable_count"],
            "screen_bullet_density_per_10000": bullets["screen_density_per_10000"],
            "local_bullets": {str(int(radius)): count for radius, count in bullets["local_counts"].items()},
            "bullet_speed_mean": float(np.mean(speeds)) if speeds else None,
            "bullet_speed_p90": float(np.percentile(speeds, 90.0)) if speeds else None,
            "region_phase": self.phase.phase,
            "region_observed_radius": self.phase.observed_radius,
            "region_learned_cycle_frames": self.phase.learned_cycle_frames,
        }
        if player_info is not None:
            player, x, y, player_radius = player_info
            self.player_frames += 1
            self.x_values.append(x)
            self.y_values.append(y)
            step_distance = 0.0
            direction = (0, 0)
            if self.previous_position is not None:
                dx, dy = x - self.previous_position[0], y - self.previous_position[1]
                step_distance = math.hypot(dx, dy)
                direction = self._direction(dx, dy)
            self.previous_position = (x, y)
            self.step_distances.append(step_distance)
            slow = player.get("slow") is True or (_number(player.get("slow")) or 0.0) > 0.0
            direction_changed = self._push_direction(direction)
            self._push_slow(slow, direction_changed=direction_changed)
            self.slow_frames += int(slow)

            bounds = _player_bounds(observation)
            edge_margins = {
                "left": x - bounds[0],
                "right": bounds[1] - x,
                "bottom": y - bounds[2],
                "top": bounds[3] - y,
            }
            edge = min(edge_margins, key=edge_margins.get)
            boundary_clearance = edge_margins[edge]
            self.edge_frames[edge] += 1
            self.boundary_clearances.append(boundary_clearance)
            for name, threshold in (("clamped", 0.25), ("within_8", 8.0), ("within_20", 20.0), ("within_48", 48.0)):
                self.boundary_bands[name] += int(boundary_clearance <= threshold)

            threats = _circle_threats(observation)
            region_threats = tuple(threat for threat in threats if threat.region)
            nonregion_threats = tuple(threat for threat in threats if not threat.region)
            bullet_threats = tuple(threat for threat in threats if threat.source == "enemy_bullets")
            overall_margin, nearest_source = _minimum_clearance(x, y, player_radius, threats)
            nonregion_margin, _ = _minimum_clearance(x, y, player_radius, nonregion_threats)
            region_margin, _ = _minimum_clearance(x, y, player_radius, region_threats)
            bullet_margin, _ = _minimum_clearance(x, y, player_radius, bullet_threats)
            if math.isfinite(overall_margin):
                self.clearance_values.append(overall_margin)
            if math.isfinite(bullet_margin):
                self.bullet_clearances.append(bullet_margin)
            if math.isfinite(region_margin):
                self.region_clearances.append(region_margin)
            bucket = _clearance_bucket(overall_margin)
            self.clearance_buckets[bucket] += 1
            if nearest_source is not None:
                self.nearest_sources[nearest_source] += 1
            level = _safety_level(overall_margin, nonregion_margin, region_margin)
            self.safety_levels[level] += 1
            if self.previous_safety_level is None:
                self.safety_run = 1
            elif level == self.previous_safety_level:
                self.safety_run += 1
            else:
                self.safety_runs[self.previous_safety_level] += self.safety_run
                self.safety_transitions.append({
                    "frame": frame,
                    "from": self.previous_safety_level,
                    "to": level,
                })
                self.safety_run = 1
            self.previous_safety_level = level
            topology = self.topology.push(
                frame=frame,
                bounds=bounds,
                player_x=x,
                player_y=y,
                player_radius=player_radius,
                region_threats=region_threats,
                player_region_margin=region_margin,
            )
            timeline_record.update({
                "player_x": x,
                "player_y": y,
                "step_distance": step_distance,
                "slow": slow,
                "boundary_clearance": boundary_clearance,
                "nearest_boundary": edge,
                "conservative_clearance": overall_margin if math.isfinite(overall_margin) else None,
                "bullet_clearance": bullet_margin if math.isfinite(bullet_margin) else None,
                "region_clearance": region_margin if math.isfinite(region_margin) else None,
                "clearance_bucket": bucket,
                "safety_level": level,
                "region_crossing": topology["crossing"],
                "region_safe_cell_fraction": topology["safe_cell_fraction"],
                "region_zone": topology["zone"],
            })
        else:
            self.previous_position = None

        timeline_due = (len(self.frames) - 1) % self.config.timeline_every == 0
        if timeline_due or observation.get("terminated") is True:
            if self.timeline and self.timeline[-1]["episode_frame"] == frame:
                self.timeline[-1] = timeline_record
            else:
                self.timeline.append(timeline_record)

    def report(self) -> dict[str, Any]:
        total = len(self.frames)
        if self.direction_run:
            self.direction_runs.append(float(self.direction_run))
            self.direction_run = 0
        if self.previous_safety_level is not None and self.safety_run:
            self.safety_runs[self.previous_safety_level] += self.safety_run
            self.safety_run = 0
        life_initial = self.life_values[0] if self.life_values else None
        life_final = self.life_values[-1] if self.life_values else None
        return {
            "trajectory": {
                "coordinate_samples": self.player_frames,
                "x": _summary(self.x_values),
                "y": _summary(self.y_values),
                "path_distance": float(sum(self.step_distances)),
                "step_distance": _summary(self.step_distances),
                "moving_frames": sum(value > 1e-6 for value in self.step_distances),
                "stationary_frames": sum(value <= 1e-6 for value in self.step_distances),
                "slow_frames": self.slow_frames,
                "slow_fraction": self.slow_frames / self.player_frames if self.player_frames else 0.0,
                "direction_changes": self.direction_changes,
                "moving_to_moving_direction_changes": self.moving_to_moving_direction_changes,
                "turns_at_least_90_degrees": self.turns_at_least_90_degrees,
                "turns_over_90_degrees": self.turns_over_90_degrees,
                "exact_reversals": self.exact_reversals,
                "movement_starts": self.movement_starts,
                "movement_stops": self.movement_stops,
                "slow_mode_changes": self.slow_mode_changes,
                "slow_mode_changes_with_direction_change": (
                    self.slow_mode_changes_with_direction_change
                ),
                "direction_run_frames": _summary(self.direction_runs),
            },
            "boundary": {
                "definition": "THlib player-center clamp: pl+8, pr-8, pb+16, pt-32",
                "clearance": _summary(self.boundary_clearances),
                "nearest_edge_frames": _fraction_counts(self.edge_frames, self.player_frames),
                "dwell_bands": _fraction_counts(self.boundary_bands, self.player_frames),
            },
            "conservative_clearance": {
                "definition": (
                    "current-frame center distance minus player radius and a max-axis "
                    "circle cover; rectangles and lasers use complete enclosing circles"
                ),
                "thresholds": {
                    "level_3_collision": 0.0,
                    "level_2_nonregion_danger": 16.0,
                    "level_1_nonregion_caution": 20.0,
                    "level_1_region_caution": 8.0,
                },
                "all_threats": _summary(self.clearance_values),
                "enemy_bullets": _summary(self.bullet_clearances),
                "indestructible_regions": _summary(self.region_clearances),
                "buckets": _fraction_counts(self.clearance_buckets, self.player_frames),
                "safety_levels": _fraction_counts(self.safety_levels, self.player_frames),
                "safety_level_transition_count": len(self.safety_transitions),
                "safety_level_transitions": self.safety_transitions,
                "safety_level_total_run_frames": {
                    str(level): frames for level, frames in sorted(self.safety_runs.items())
                },
                "nearest_threat_source_frames": _fraction_counts(self.nearest_sources, self.player_frames),
            },
            "bullets": {
                "visible_count_per_frame": _summary(self.visible_bullet_counts),
                "collidable_count_per_frame": _summary(self.collidable_bullet_counts),
                "screen_density_per_10000_world_units": _summary(self.screen_densities),
                "local_count_per_frame": {
                    str(int(radius)): _summary(values)
                    for radius, values in self.local_bullet_counts.items()
                },
                "speed": {
                    "object_frame_samples": self.bullet_speed_samples,
                    "object_frame_weighted_mean": (
                        self.bullet_speed_sum / self.bullet_speed_samples
                        if self.bullet_speed_samples else None
                    ),
                    "maximum": self.bullet_speed_maximum,
                    "frame_mean": _summary(self.bullet_speed_frame_means),
                    "frame_p90": _summary(self.bullet_speed_frame_p90),
                    "buckets": _fraction_counts(self.bullet_speed_buckets, self.bullet_speed_samples),
                },
            },
            "region_phase": {
                "analysis_model": "EngineMPC visible-radius phase tracker without hints or external memory",
                "frame_counts": _fraction_counts(self.phase_counts, total),
                "transitions": self.phase_transitions,
                "transition_count": len(self.phase_transitions),
                "observed_radius": _summary(self.region_radii),
                "minimum_radius": self.phase.minimum_radius,
                "maximum_radius": self.phase.maximum_radius,
                "minimum_plateau_radius": self.phase.minimum_plateau_radius,
                "maximum_plateau_radius": self.phase.maximum_plateau_radius,
                "growth_rate": self.phase.growth_rate,
                "contraction_rate": self.phase.contraction_rate,
                "expansion_starts": list(self.phase.expansion_starts),
                "learned_cycle_frames": self.phase.learned_cycle_frames,
                "final_phase": self.phase.phase,
            },
            "region_topology": self.topology.report(self.player_frames),
            "player_state": {
                "death_state_frames": self.death_frames,
                "death_state_entries": self.death_entries,
                "maximum_death_value": self.maximum_death,
                "protected_frames": self.protected_frames,
                "life_initial": life_initial,
                "life_final": life_final,
                "life_delta": (
                    None if life_initial is None or life_final is None else life_final - life_initial
                ),
            },
            "observed_outcome": {
                "enemy_count_per_frame": _summary(self.enemy_counts),
                "last_enemy_present_frame": self.last_enemy_present_frame,
                "boss_hp_initial": self.boss_hp_initial,
                "boss_hp_last_observed": self.boss_hp_values[-1] if self.boss_hp_values else None,
                "boss_hp_minimum_observed": min(self.boss_hp_values) if self.boss_hp_values else None,
                "reporting_only": True,
            },
            "timeline": self.timeline,
        }


def _validated_reset_metadata(
    response: Mapping[str, Any],
    *,
    requested_path: str,
) -> dict[str, Any]:
    reset = response.get("reset")
    if not isinstance(reset, Mapping):
        raise EngineProtocolError("engine replay reset has no metadata object")
    replay = reset.get("replay")
    if reset.get("episode_kind") != "replay" or not isinstance(replay, Mapping):
        raise EngineProtocolError("engine replay reset has no nested replay metadata")
    result = dict(replay)
    frame_count = _integer(result.get("frame_count"))
    frame_bytes = _integer(result.get("frame_bytes_verified"))
    file_size = _integer(result.get("file_size"))
    frame_position = _integer(result.get("frame_data_position"))
    crc32 = result.get("crc32")
    if (
        result.get("path") != requested_path
        or result.get("schema_version") != 1
        or result.get("file_version") != 1
        or result.get("stage_name") != "Spell Practice@Spell Practice"
        or not isinstance(result.get("game_name"), str)
        or not isinstance(result.get("stage_player"), str)
        or _integer(result.get("random_seed")) is None
        or frame_count is None
        or frame_count <= 0
        or frame_bytes != frame_count
        or file_size is None
        or file_size < frame_count
        or frame_position is None
        or frame_position < 0
        or not isinstance(crc32, str)
        or _CRC32.fullmatch(crc32) is None
    ):
        raise EngineProtocolError("engine returned invalid replay reset metadata")
    return result


def run_engine_replay_analysis(
    client: EngineClient,
    *,
    replay_path: str,
    config: EngineReplayAnalysisConfig = EngineReplayAnalysisConfig(),
) -> dict[str, Any]:
    """Replay every declared input frame and aggregate authority telemetry."""

    if not isinstance(replay_path, str) or not replay_path:
        raise ValueError("replay_path must be a nonempty string")
    ping = client.ping()
    commands = ping.get("commands")
    if not isinstance(commands, list) or "reset_replay" not in commands:
        raise EngineProtocolError("engine bridge does not advertise reset_replay")
    runtime_verification = verify_runtime_source_fingerprints(ping)
    response = client.reset_replay(replay_path)
    metadata = _validated_reset_metadata(response, requested_path=replay_path)
    client.set_rendering(config.render, every=config.render_every)
    observation = _observation(response)
    first_frame = _integer(observation.get("episode_frame"))
    if first_frame != 1:
        raise EngineProtocolError("replay reset must consume exactly the first input frame")

    telemetry = _ReplayTelemetry(config)
    termination_reason = None
    terminated = False
    while True:
        telemetry.push(observation)
        terminated = observation.get("terminated") is True
        termination_reason = observation.get("termination_reason") if terminated else None
        if terminated or len(telemetry.frames) >= config.max_frames:
            break
        if len(telemetry.frames) >= metadata["frame_count"]:
            raise EngineProtocolError(
                "engine consumed every declared replay frame without replay_exhausted"
            )
        response = client.step(Action(shoot=False), repeat=1)
        observation = _observation(response)

    declared_frames = int(metadata["frame_count"])
    analyzed_frames = len(telemetry.frames)
    accepted_terminal_reasons = frozenset((
        "replay_exhausted",
        "attack_complete",
        "player_hit",
    ))
    complete = terminated and termination_reason in accepted_terminal_reasons
    input_stream_fully_consumed = (
        analyzed_frames == declared_frames
        and telemetry.frames[-1] == declared_frames
    )
    return {
        "schema_version": 1,
        "run_kind": "native_replay_observation_analysis",
        "acceptance_claim": False,
        "replay_path": replay_path,
        "config": asdict(config),
        "engine": {
            "protocol": ping.get("protocol"),
            "session_id": ping.get("session_id"),
            "process_nonce": ping.get("process_nonce"),
            "runtime_identity": ping.get("runtime_identity"),
            "runtime_source_verification": runtime_verification,
        },
        "replay": metadata,
        "analysis_complete": complete,
        "input_stream_fully_consumed": input_stream_fully_consumed,
        "unconsumed_input_frames": max(0, declared_frames - analyzed_frames),
        "terminated": terminated,
        "termination_reason": termination_reason,
        "declared_frame_count": declared_frames,
        "frames_analyzed": analyzed_frames,
        "first_episode_frame": telemetry.frames[0],
        "last_episode_frame": telemetry.frames[-1],
        "frame_contract": {
            "contiguous": True,
            "reset_consumed_first_input": True,
            "neutral_step_action_is_overridden_by_thlib_replay_reader": True,
        },
        **telemetry.report(),
    }


__all__ = [
    "EngineReplayAnalysisConfig",
    "run_engine_replay_analysis",
]
