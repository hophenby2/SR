"""Offline space-time route teachers for protected native-engine captures.

This module deliberately sits outside the deployed controller.  It uses a
complete future threat timeline, so its output is suitable only as an
imitation-learning target.  A policy trained from the route must still make
decisions from delayed visual observations and recurrent state at runtime.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .protocol import Action


_SQRT_HALF = math.sqrt(0.5)


@dataclass(frozen=True, slots=True)
class TeacherRouteConfig:
    """Movement, geometry, and search settings for an offline teacher route."""

    decision_interval: int = 3
    fast_speed: float = 4.0
    focus_speed: float = 2.0
    player_radius: float = 0.5
    bounds: tuple[float, float, float, float] = (-184.0, 184.0, -208.0, 192.0)
    boundary_padding: float = 2.0
    beam_width: int = 4096
    position_bin_size: float = 2.0
    maximum_histories_per_cell: int = 3
    hard_clearance: float = 0.0
    desired_clearance: float = 12.0
    clearance_cap: float = 32.0
    spatial_bin_size: float = 32.0
    direction_switch_penalty: float = 18.0
    direction_reverse_penalty: float = 42.0
    direction_aba_penalty: float = 28.0
    speed_switch_penalty: float = 2.0
    movement_penalty: float = 0.008
    boundary_margin: float = 14.0
    boundary_penalty: float = 1.5
    anchor_weight: float = 0.0015
    anchor_x: float = 0.0
    anchor_y: float = -176.0
    anchor_y_scale: float = 0.15
    reference_action_penalty: float = 0.0
    reference_position_weight: float = 0.0
    start_x: float = 0.0
    start_y: float = -176.0
    warning_flag: float = 1.0

    def __post_init__(self) -> None:
        if self.decision_interval != 3:
            raise ValueError("teacher actions must be held for exactly three frames")
        positive = (
            self.fast_speed,
            self.focus_speed,
            self.player_radius,
            self.beam_width,
            self.position_bin_size,
            self.maximum_histories_per_cell,
            self.desired_clearance,
            self.clearance_cap,
            self.spatial_bin_size,
        )
        if not all(math.isfinite(float(value)) and value > 0 for value in positive):
            raise ValueError("route search sizes and movement values must be positive")
        if self.focus_speed > self.fast_speed:
            raise ValueError("focus speed cannot exceed fast speed")
        left, right, bottom, top = self.bounds
        if not all(math.isfinite(value) for value in self.bounds):
            raise ValueError("movement bounds must be finite")
        if left >= right or bottom >= top:
            raise ValueError("movement bounds must have positive area")
        if self.boundary_padding < 0.0:
            raise ValueError("boundary padding cannot be negative")
        if 2.0 * self.boundary_padding >= min(right - left, top - bottom):
            raise ValueError("boundary padding consumes the movement bounds")
        if self.hard_clearance < 0.0:
            raise ValueError("hard clearance cannot be negative")
        if self.clearance_cap < self.desired_clearance:
            raise ValueError("clearance cap must cover the desired clearance")
        nonnegative = (
            self.direction_switch_penalty,
            self.direction_reverse_penalty,
            self.direction_aba_penalty,
            self.speed_switch_penalty,
            self.movement_penalty,
            self.boundary_margin,
            self.boundary_penalty,
            self.anchor_weight,
            self.anchor_y_scale,
            self.reference_action_penalty,
            self.reference_position_weight,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in nonnegative):
            raise ValueError("route penalties must be finite and non-negative")
        if not math.isfinite(self.anchor_x) or not math.isfinite(self.anchor_y):
            raise ValueError("route anchor must be finite")
        if not math.isfinite(self.warning_flag):
            raise ValueError("warning flag must be finite")

    @property
    def navigation_bounds(self) -> tuple[float, float, float, float]:
        left, right, bottom, top = self.bounds
        padding = self.boundary_padding
        return left + padding, right - padding, bottom + padding, top - padding


@dataclass(frozen=True, slots=True)
class ProtectedThreatTimeline:
    """Compact per-frame threat geometry loaded from a protected capture.

    Threat columns are ``x, y, a, b, angle_degrees, warning``.  Warning rows
    describe non-player-collision geometry (for example the boss and the
    rotating emitter in the Boss #4 capture) and are excluded from lethal
    collision checks by default.
    """

    frames: np.ndarray
    offsets: np.ndarray
    threats: np.ndarray
    source_path: Path | None = None
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        frames = np.asarray(self.frames)
        offsets = np.asarray(self.offsets)
        threats = np.asarray(self.threats)
        if frames.ndim != 1 or not len(frames):
            raise ValueError("frames must be a non-empty one-dimensional array")
        if offsets.shape != (len(frames) + 1,):
            raise ValueError("offsets must contain one entry per frame plus a sentinel")
        if threats.ndim != 2 or threats.shape[1] != 6:
            raise ValueError("threats must have x, y, a, b, angle, warning columns")
        if not np.issubdtype(frames.dtype, np.integer):
            raise ValueError("frames must use an integer dtype")
        if not np.issubdtype(offsets.dtype, np.integer):
            raise ValueError("offsets must use an integer dtype")
        if np.any(np.diff(frames.astype(np.int64)) != 1):
            raise ValueError("protected threat frames must be consecutive")
        if offsets[0] != 0 or offsets[-1] != len(threats) or np.any(np.diff(offsets) < 0):
            raise ValueError("threat offsets are not a monotonic partition")
        if not np.all(np.isfinite(threats)):
            raise ValueError("threat geometry must be finite")
        if np.any(threats[:, 2:4] <= 0.0):
            raise ValueError("threat ellipse radii must be positive")

    @classmethod
    def from_npz(cls, path: str | Path) -> "ProtectedThreatTimeline":
        source = Path(path)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        with np.load(source, allow_pickle=False) as payload:
            missing = {"frames", "offsets", "threats"} - set(payload.files)
            if missing:
                raise ValueError(
                    "protected capture is missing arrays: " + ", ".join(sorted(missing))
                )
            return cls(
                frames=np.asarray(payload["frames"], dtype=np.int32),
                offsets=np.asarray(payload["offsets"], dtype=np.int64),
                threats=np.asarray(payload["threats"], dtype=np.float64),
                source_path=source,
                source_sha256=digest,
            )

    def at(self, frame_index: int, *, warning_flag: float = 1.0) -> np.ndarray:
        if not 0 <= frame_index < len(self.frames):
            raise IndexError("frame index is outside the threat timeline")
        start, end = int(self.offsets[frame_index]), int(self.offsets[frame_index + 1])
        rows = self.threats[start:end]
        return rows[~np.isclose(rows[:, 5], warning_flag)]


@dataclass(frozen=True, slots=True)
class RouteDecision:
    start_frame: int
    frame_count: int
    action: Action
    start_position: tuple[float, float]
    end_position: tuple[float, float]
    minimum_clearance: float


@dataclass(frozen=True, slots=True)
class RouteValidation:
    collision_free: bool
    minimum_clearance: float
    minimum_clearance_frame: int | None
    path_distance: float
    direction_changes: int
    direction_reversals: int
    aba_changes: int
    slow_frames: int
    total_frames: int
    positions: tuple[tuple[float, float], ...]
    frame_clearances: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TeacherRoute:
    decisions: tuple[RouteDecision, ...]
    validation: RouteValidation
    config: TeacherRouteConfig
    source_path: str | None
    source_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        validation = asdict(self.validation)
        validation.pop("positions")
        validation.pop("frame_clearances")
        return {
            "schema_version": 2,
            "kind": "offline_space_time_teacher_route",
            "purpose": "training_data_distillation_only",
            "not_for_deployment": True,
            "not_acceptance_evidence": True,
            "uses_complete_future_threat_timeline": True,
            "timeline_semantics": "initial_state_plus_post_step_frames",
            "source": {
                "path": self.source_path,
                "sha256": self.source_sha256,
            },
            "config": asdict(self.config),
            "summary": validation,
            "decisions": [
                {
                    "start_frame": item.start_frame,
                    "frame_count": item.frame_count,
                    "action": item.action.to_dict(),
                    "action_discrete": item.action.discrete,
                    "start_position": list(item.start_position),
                    "end_position": list(item.end_position),
                    "minimum_clearance": item.minimum_clearance,
                }
                for item in self.decisions
            ],
        }

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def teacher_actions() -> tuple[Action, ...]:
    """Return the 17 unique three-frame movement controls."""

    result: list[Action] = []
    for slow in (False, True):
        for move_y in (-1, 0, 1):
            for move_x in (-1, 0, 1):
                if slow and move_x == 0 and move_y == 0:
                    continue
                result.append(Action(move_x=move_x, move_y=move_y, slow=slow))
    return tuple(result)


def _velocity(action: Action, config: TeacherRouteConfig) -> tuple[float, float]:
    speed = config.focus_speed if action.slow else config.fast_speed
    if action.move_x and action.move_y:
        speed *= _SQRT_HALF
    return speed * action.move_x, speed * action.move_y


class _ThreatIndex:
    """Per-frame spatial bins with conservative point-to-ellipse margins."""

    def __init__(self, rows: np.ndarray, config: TeacherRouteConfig) -> None:
        self.rows = rows
        self.config = config
        self.bins: dict[tuple[int, int], np.ndarray] = {}
        if not len(rows):
            return
        size = config.spatial_bin_size
        bx = np.floor(rows[:, 0] / size).astype(np.int32)
        by = np.floor(rows[:, 1] / size).astype(np.int32)
        order = np.lexsort((by, bx))
        bx, by = bx[order], by[order]
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and bx[end] == bx[start] and by[end] == by[start]:
                end += 1
            self.bins[(int(bx[start]), int(by[start]))] = order[start:end]
            start = end

    @staticmethod
    def _ellipse_margins(points: np.ndarray, rows: np.ndarray, player_radius: float) -> np.ndarray:
        dx = points[:, None, 0] - rows[None, :, 0]
        dy = points[:, None, 1] - rows[None, :, 1]
        angle = np.deg2rad(rows[None, :, 4])
        cosine, sine = np.cos(angle), np.sin(angle)
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        normalized_distance = np.hypot(
            local_x / rows[None, :, 2],
            local_y / rows[None, :, 3],
        )
        # Mapping the unit circle by diag(a, b) can stretch every distance by
        # at least min(a, b).  This is therefore a lower bound on Euclidean
        # clearance, exact for the circular bullets in the protected capture.
        # A lower bound is intentional: hard safety must never depend on an
        # optimistic radial approximation for a rotated, elongated ellipse.
        ellipse_clearance_lower_bound = (
            np.minimum(rows[None, :, 2], rows[None, :, 3])
            * (normalized_distance - 1.0)
        )
        return ellipse_clearance_lower_bound - player_radius

    def margins(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("candidate points must have [x, y] columns")
        cap = self.config.clearance_cap
        result = np.full(len(points), cap, dtype=np.float64)
        if not len(self.rows) or not len(points):
            return result
        size = self.config.spatial_bin_size
        largest_radius = float(np.max(self.rows[:, 2:4]))
        reach = cap + self.config.player_radius + largest_radius
        neighborhood = int(math.floor(reach / size)) + 1
        bx = np.floor(points[:, 0] / size).astype(np.int32)
        by = np.floor(points[:, 1] / size).astype(np.int32)
        candidate_order = np.lexsort((by, bx))
        sorted_x, sorted_y = bx[candidate_order], by[candidate_order]
        starts = np.flatnonzero(np.concatenate((
            np.array([True]),
            (sorted_x[1:] != sorted_x[:-1]) | (sorted_y[1:] != sorted_y[:-1]),
        )))
        ends = np.concatenate((starts[1:], np.array([len(candidate_order)])))
        for start, end in zip(starts, ends, strict=True):
            candidate_indices = candidate_order[start:end]
            cell_x, cell_y = sorted_x[start], sorted_y[start]
            threat_parts = [
                self.bins[(int(cell_x + ox), int(cell_y + oy))]
                for ox in range(-neighborhood, neighborhood + 1)
                for oy in range(-neighborhood, neighborhood + 1)
                if (int(cell_x + ox), int(cell_y + oy)) in self.bins
            ]
            if not threat_parts:
                continue
            threat_indices = np.concatenate(threat_parts)
            margins = self._ellipse_margins(
                points[candidate_indices],
                self.rows[threat_indices],
                self.config.player_radius,
            )
            result[candidate_indices] = np.minimum(cap, margins.min(axis=1))
        return result


class _TimelineGeometry:
    def __init__(self, timeline: ProtectedThreatTimeline, config: TeacherRouteConfig) -> None:
        self.timeline = timeline
        self.config = config

    def margins(self, frame_index: int, points: np.ndarray) -> np.ndarray:
        # Solving and validation both consume frames monotonically exactly
        # once.  Retaining thousands of small spatial-bin dictionaries would
        # only increase peak memory without producing cache hits.
        index = _ThreatIndex(
            self.timeline.at(frame_index, warning_flag=self.config.warning_flag),
            self.config,
        )
        return index.margins(points)


def _motion_counts(actions: Sequence[Action]) -> tuple[int, int, int]:
    directions = [(item.move_x, item.move_y) for item in actions]
    changes = 0
    reversals = 0
    aba = 0
    for index in range(1, len(directions)):
        previous, current = directions[index - 1], directions[index]
        if previous != current:
            changes += 1
        if current == (-previous[0], -previous[1]) and current != (0, 0):
            reversals += 1
        if index >= 2 and current == directions[index - 2] and current != previous:
            aba += 1
    return changes, reversals, aba


def validate_teacher_route(
    timeline: ProtectedThreatTimeline,
    actions: Sequence[Action],
    config: TeacherRouteConfig,
    *,
    start: tuple[float, float] | None = None,
) -> RouteValidation:
    """Replay a route exactly, checking collision geometry on every frame.

    Native captures contain the reset observation first.  An action taken at
    that observation advances into the next captured frame, so a timeline of
    ``N`` observations has exactly ``N - 1`` controlled transitions.
    """

    transition_count = len(timeline.frames) - 1
    expected_actions = math.ceil(transition_count / config.decision_interval)
    if len(actions) != expected_actions:
        raise ValueError(f"route needs {expected_actions} decisions, got {len(actions)}")
    x, y = start or (config.start_x, config.start_y)
    left, right, bottom, top = config.navigation_bounds
    if not left <= x <= right or not bottom <= y <= top:
        raise ValueError("route start lies outside conservative navigation bounds")
    geometry = _TimelineGeometry(timeline, config)
    positions: list[tuple[float, float]] = [(x, y)]
    clearances: list[float] = [
        float(geometry.margins(0, np.array([[x, y]]))[0])
    ]
    path_distance = 0.0
    slow_frames = 0
    frame_index = 1
    for action in actions:
        vx, vy = _velocity(action, config)
        for _ in range(config.decision_interval):
            if frame_index >= len(timeline.frames):
                break
            next_x = min(max(x + vx, left), right)
            next_y = min(max(y + vy, bottom), top)
            path_distance += math.hypot(next_x - x, next_y - y)
            x, y = next_x, next_y
            margin = float(geometry.margins(frame_index, np.array([[x, y]]))[0])
            positions.append((x, y))
            clearances.append(margin)
            slow_frames += int(action.slow)
            frame_index += 1
    minimum_index = int(np.argmin(clearances)) if clearances else None
    changes, reversals, aba = _motion_counts(actions)
    minimum = float(clearances[minimum_index]) if minimum_index is not None else math.inf
    return RouteValidation(
        collision_free=minimum > config.hard_clearance,
        minimum_clearance=minimum,
        minimum_clearance_frame=(
            None if minimum_index is None else int(timeline.frames[minimum_index])
        ),
        path_distance=path_distance,
        direction_changes=changes,
        direction_reversals=reversals,
        aba_changes=aba,
        slow_frames=slow_frames,
        total_frames=len(positions),
        positions=tuple(positions),
        frame_clearances=tuple(clearances),
    )


def _select_beam(
    x: np.ndarray,
    y: np.ndarray,
    costs: np.ndarray,
    minimum_margin: np.ndarray,
    direction: np.ndarray,
    config: TeacherRouteConfig,
) -> np.ndarray:
    """Keep low-cost, spatially diverse histories deterministically."""

    left, _right, bottom, _top = config.navigation_bounds
    cell_x = np.rint((x - left) / config.position_bin_size).astype(np.int32)
    cell_y = np.rint((y - bottom) / config.position_bin_size).astype(np.int32)
    width = int(math.ceil((config.bounds[1] - config.bounds[0]) / config.position_bin_size)) + 3
    cell = cell_y.astype(np.int64) * width + cell_x
    # First retain distinct last directions in a cell.  Lexsort's final key is
    # primary, hence the reverse order below.
    direction = direction.astype(np.int64)
    order = np.lexsort((-minimum_margin, costs, direction, cell))
    keys = cell[order] * 9 + direction[order]
    distinct = np.ones(len(order), dtype=bool)
    distinct[1:] = keys[1:] != keys[:-1]
    candidates = order[distinct]
    ranking = np.lexsort((-minimum_margin[candidates], costs[candidates]))
    candidates = candidates[ranking]
    if config.maximum_histories_per_cell > 0:
        kept: list[int] = []
        counts: dict[int, int] = {}
        for raw in candidates:
            index = int(raw)
            key = int(cell[index])
            count = counts.get(key, 0)
            if count >= config.maximum_histories_per_cell:
                continue
            counts[key] = count + 1
            kept.append(index)
            if len(kept) >= config.beam_width:
                break
        return np.asarray(kept, dtype=np.int64)
    return candidates[: config.beam_width]


def solve_teacher_route(
    timeline: ProtectedThreatTimeline,
    config: TeacherRouteConfig = TeacherRouteConfig(),
    *,
    reference_actions: Sequence[Action] | None = None,
    reference_positions: Sequence[Sequence[float]] | None = None,
) -> TeacherRoute:
    """Find a collision-free, high-clearance offline teacher trajectory.

    Search nodes carry exact floating-point LuaSTG positions.  Quantization is
    used only to merge nearby histories after every three-frame action.  Every
    intermediate frame is checked before a node can enter the next beam.
    """

    actions = teacher_actions()
    action_count = len(actions)
    velocities = np.asarray([_velocity(item, config) for item in actions])
    direction_codes = np.asarray([
        (item.move_y + 1) * 3 + item.move_x + 1 for item in actions
    ], dtype=np.int8)
    left, right, bottom, top = config.navigation_bounds
    if not left <= config.start_x <= right or not bottom <= config.start_y <= top:
        raise ValueError("configured route start lies outside navigation bounds")
    transition_count = len(timeline.frames) - 1
    if reference_actions is not None and len(reference_actions) < transition_count:
        raise ValueError("reference route has fewer actions than timeline transitions")
    if reference_positions is not None and len(reference_positions) < len(timeline.frames):
        raise ValueError("reference route has fewer positions than timeline frames")

    x = np.array([config.start_x], dtype=np.float64)
    y = np.array([config.start_y], dtype=np.float64)
    costs = np.zeros(1, dtype=np.float64)
    geometry = _TimelineGeometry(timeline, config)
    initial_margin = float(geometry.margins(
        0,
        np.array([[config.start_x, config.start_y]], dtype=np.float64),
    )[0])
    if initial_margin <= config.hard_clearance:
        raise RuntimeError(
            f"teacher route start is unsafe at frame {int(timeline.frames[0])}"
        )
    minimum_margin = np.array([initial_margin], dtype=np.float64)
    last_action = np.full(1, -1, dtype=np.int16)
    two_actions_ago = np.full(1, -1, dtype=np.int16)
    parent_layers: list[np.ndarray] = []
    action_layers: list[np.ndarray] = []
    interval_margin_layers: list[np.ndarray] = []
    position_layers: list[tuple[np.ndarray, np.ndarray]] = [(x.copy(), y.copy())]
    # Frame zero is the reset observation before any teacher input. Search
    # actions produce frames one onward, matching EngineClient.step exactly.
    frame_index = 1
    while frame_index < len(timeline.frames):
        state_count = len(x)
        parents = np.repeat(np.arange(state_count, dtype=np.int32), action_count)
        chosen = np.tile(np.arange(action_count, dtype=np.int16), state_count)
        candidate_x = x[parents].copy()
        candidate_y = y[parents].copy()
        candidate_cost = costs[parents].copy()
        candidate_minimum = minimum_margin[parents].copy()
        interval_minimum = np.full(len(parents), config.clearance_cap, dtype=np.float64)
        candidate_last = chosen.copy()
        candidate_two_ago = last_action[parents].copy()

        if reference_actions is not None and config.reference_action_penalty > 0.0:
            reference = reference_actions[frame_index - 1]
            reference_mismatch = np.asarray([
                item.move_x != reference.move_x
                or item.move_y != reference.move_y
                or item.slow != reference.slow
                for item in actions
            ], dtype=np.float64)
            candidate_cost += (
                config.reference_action_penalty * reference_mismatch[chosen]
            )

        previous = last_action[parents]
        previous_two = two_actions_ago[parents]
        previous_valid = previous >= 0
        previous_safe = np.maximum(previous, 0)
        previous_two_safe = np.maximum(previous_two, 0)
        move_x = np.asarray([item.move_x for item in actions], dtype=np.int8)
        move_y = np.asarray([item.move_y for item in actions], dtype=np.int8)
        slow = np.asarray([item.slow for item in actions], dtype=bool)
        direction_changed = previous_valid & (
            (move_x[chosen] != move_x[previous_safe])
            | (move_y[chosen] != move_y[previous_safe])
        )
        candidate_cost += config.direction_switch_penalty * direction_changed
        reversed_direction = direction_changed & (
            (move_x[chosen] == -move_x[previous_safe])
            & (move_y[chosen] == -move_y[previous_safe])
            & ((move_x[chosen] != 0) | (move_y[chosen] != 0))
        )
        candidate_cost += config.direction_reverse_penalty * reversed_direction
        aba = direction_changed & (previous_two >= 0) & (
            (move_x[chosen] == move_x[previous_two_safe])
            & (move_y[chosen] == move_y[previous_two_safe])
        )
        candidate_cost += config.direction_aba_penalty * aba
        candidate_cost += config.speed_switch_penalty * (
            previous_valid & ~direction_changed & (slow[chosen] != slow[previous_safe])
        )

        step_count = min(config.decision_interval, len(timeline.frames) - frame_index)
        valid = np.ones(len(parents), dtype=bool)
        for step in range(step_count):
            candidate_x = np.clip(candidate_x + velocities[chosen, 0], left, right)
            candidate_y = np.clip(candidate_y + velocities[chosen, 1], bottom, top)
            margins = geometry.margins(
                frame_index + step,
                np.column_stack((candidate_x, candidate_y)),
            )
            interval_minimum = np.minimum(interval_minimum, margins)
            candidate_minimum = np.minimum(candidate_minimum, margins)
            valid &= margins > config.hard_clearance
            shortfall = np.maximum(0.0, config.desired_clearance - margins)
            candidate_cost += shortfall * shortfall
            candidate_cost -= 0.03 * np.minimum(margins, config.clearance_cap)
            boundary_clearance = np.minimum.reduce((
                candidate_x - left,
                right - candidate_x,
                candidate_y - bottom,
                top - candidate_y,
            ))
            candidate_cost += config.boundary_penalty * np.maximum(
                0.0, config.boundary_margin - boundary_clearance,
            ) ** 2
            candidate_cost += config.anchor_weight * (
                (candidate_x - config.anchor_x) ** 2
                + config.anchor_y_scale * (candidate_y - config.anchor_y) ** 2
            )
            if (
                reference_positions is not None
                and config.reference_position_weight > 0.0
            ):
                reference_position = reference_positions[frame_index + step]
                reference_x = float(reference_position[0])
                reference_y = float(reference_position[1])
                candidate_cost += config.reference_position_weight * (
                    (candidate_x - reference_x) ** 2
                    + (candidate_y - reference_y) ** 2
                )
        candidate_cost += config.movement_penalty * step_count * np.hypot(
            velocities[chosen, 0], velocities[chosen, 1],
        )
        if not np.any(valid):
            failed_frame = int(timeline.frames[frame_index + step_count - 1])
            raise RuntimeError(f"teacher route beam became empty at frame {failed_frame}")

        valid_indices = np.flatnonzero(valid)
        keep_relative = _select_beam(
            candidate_x[valid_indices],
            candidate_y[valid_indices],
            candidate_cost[valid_indices],
            candidate_minimum[valid_indices],
            direction_codes[candidate_last[valid_indices]],
            config,
        )
        keep = valid_indices[keep_relative]
        parent_layers.append(parents[keep])
        action_layers.append(chosen[keep])
        interval_margin_layers.append(interval_minimum[keep])
        x, y = candidate_x[keep], candidate_y[keep]
        costs = candidate_cost[keep]
        minimum_margin = candidate_minimum[keep]
        last_action = candidate_last[keep]
        two_actions_ago = candidate_two_ago[keep]
        position_layers.append((x.copy(), y.copy()))
        frame_index += step_count

    final_order = np.lexsort((-minimum_margin, costs))
    state = int(final_order[0])
    selected_actions: list[Action] = []
    selected_interval_margins: list[float] = []
    selected_positions: list[tuple[float, float]] = [
        (float(x[state]), float(y[state]))
    ]
    for layer_index in range(len(action_layers) - 1, -1, -1):
        selected_actions.append(actions[int(action_layers[layer_index][state])])
        selected_interval_margins.append(float(interval_margin_layers[layer_index][state]))
        state = int(parent_layers[layer_index][state])
        layer_x, layer_y = position_layers[layer_index]
        selected_positions.append((float(layer_x[state]), float(layer_y[state])))
    selected_actions.reverse()
    selected_interval_margins.reverse()
    selected_positions.reverse()

    validation = validate_teacher_route(timeline, selected_actions, config)
    if not validation.collision_free:
        raise RuntimeError("internal route replay found a collision")
    decisions: list[RouteDecision] = []
    used_frames = 0
    for index, action in enumerate(selected_actions):
        count = min(
            config.decision_interval,
            len(timeline.frames) - 1 - used_frames,
        )
        decisions.append(RouteDecision(
            start_frame=int(timeline.frames[used_frames]),
            frame_count=count,
            action=action,
            start_position=selected_positions[index],
            end_position=selected_positions[index + 1],
            minimum_clearance=selected_interval_margins[index],
        ))
        used_frames += count
    return TeacherRoute(
        decisions=tuple(decisions),
        validation=validation,
        config=config,
        source_path=str(timeline.source_path) if timeline.source_path is not None else None,
        source_sha256=timeline.source_sha256,
    )


def _parser() -> argparse.ArgumentParser:
    defaults = TeacherRouteConfig()
    parser = argparse.ArgumentParser(
        description="Solve a training-only route from a protected threat capture.",
    )
    parser.add_argument("capture", type=Path, help="NPZ with frames/offsets/threats arrays")
    parser.add_argument("--output", type=Path, required=True, help="teacher route JSON")
    parser.add_argument("--beam-width", type=int, default=defaults.beam_width)
    parser.add_argument("--hard-clearance", type=float, default=defaults.hard_clearance)
    parser.add_argument("--desired-clearance", type=float, default=defaults.desired_clearance)
    parser.add_argument("--position-bin-size", type=float, default=defaults.position_bin_size)
    parser.add_argument("--reference-route", type=Path)
    parser.add_argument(
        "--reference-action-penalty",
        type=float,
        default=defaults.reference_action_penalty,
    )
    parser.add_argument(
        "--reference-position-weight",
        type=float,
        default=defaults.reference_position_weight,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = TeacherRouteConfig(
        beam_width=arguments.beam_width,
        hard_clearance=arguments.hard_clearance,
        desired_clearance=arguments.desired_clearance,
        position_bin_size=arguments.position_bin_size,
        reference_action_penalty=arguments.reference_action_penalty,
        reference_position_weight=arguments.reference_position_weight,
    )
    timeline = ProtectedThreatTimeline.from_npz(arguments.capture)
    reference_actions = None
    reference_positions = None
    if arguments.reference_route is not None:
        from .teacher_route_engine import load_route_program

        reference = load_route_program(arguments.reference_route)
        reference_actions = reference.actions
        reference_positions = reference.expected_positions
    route = solve_teacher_route(
        timeline,
        config,
        reference_actions=reference_actions,
        reference_positions=reference_positions,
    )
    route.write_json(arguments.output)
    summary = route.validation
    print(
        f"route={arguments.output} frames={summary.total_frames} "
        f"min_clearance={summary.minimum_clearance:.3f} "
        f"direction_changes={summary.direction_changes} "
        f"reversals={summary.direction_reversals}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
