"""Visible-trajectory model-predictive teacher for live engine observations."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from .protocol import Action


_OBJECT_ARRAYS = (
    "enemy_bullets",
    "enemies",
    "nontjt_enemies",
    "indestructibles",
)
_SQRT_HALF = math.sqrt(0.5)


def _number(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    result = float(value)
    return result if math.isfinite(result) else default


def _frame(observation: Mapping[str, Any], fallback: int) -> int:
    value = observation.get("episode_frame", observation.get("frame"))
    if isinstance(value, bool) or not isinstance(value, int):
        return fallback
    return value


def _unwrap_observation(value: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = value.get("observation")
    return nested if isinstance(nested, Mapping) else value


def _radius(record: Mapping[str, Any], default: float = 2.0) -> float:
    values = [
        abs(value)
        for name in ("a", "b")
        if (value := _number(record.get(name))) is not None
    ]
    return max(0.1, *(values or [default]))


_REGION_PHASE_ORDER = (
    "expanding",
    "maximum_hold",
    "contracting",
    "minimum_hold",
)


@dataclass(frozen=True, slots=True)
class RegionDynamicsMemory:
    """Reusable safe-region dynamics learned without episode coordinates."""

    minimum_radius: float
    maximum_radius: float
    growth_rate: float
    contraction_rate: float
    expanding_frames: float
    maximum_hold_frames: float
    contracting_frames: float
    minimum_hold_frames: float
    cycle_frames: float
    lateral_flow_cycle_frames: float | None = None
    safe_side_rule: str | None = None

    def __post_init__(self) -> None:
        values = (
            self.minimum_radius,
            self.maximum_radius,
            self.growth_rate,
            self.contraction_rate,
            self.expanding_frames,
            self.maximum_hold_frames,
            self.contracting_frames,
            self.minimum_hold_frames,
            self.cycle_frames,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("region dynamics values must be finite and positive")
        if self.maximum_radius <= self.minimum_radius:
            raise ValueError("region dynamics maximum must exceed the minimum")
        duration_sum = sum(self.phase_durations.values())
        if not math.isclose(duration_sum, self.cycle_frames, abs_tol=3.0):
            raise ValueError("region dynamics phase durations must sum to the cycle")
        has_lateral_cycle = self.lateral_flow_cycle_frames is not None
        has_safe_side_rule = self.safe_side_rule is not None
        if has_lateral_cycle != has_safe_side_rule:
            raise ValueError(
                "region lateral flow cycle and safe-side rule must be paired"
            )
        if has_lateral_cycle and (
            not math.isfinite(self.lateral_flow_cycle_frames)
            or self.lateral_flow_cycle_frames <= 0.0
        ):
            raise ValueError("region lateral flow cycle must be finite and positive")
        if has_safe_side_rule and self.safe_side_rule != (
            "opposite_incoming_lateral_flow"
        ):
            raise ValueError("unsupported region safe-side rule")

    @property
    def phase_durations(self) -> dict[str, float]:
        return {
            "expanding": self.expanding_frames,
            "maximum_hold": self.maximum_hold_frames,
            "contracting": self.contracting_frames,
            "minimum_hold": self.minimum_hold_frames,
        }


def load_region_dynamics_memory(
    path: str | Path,
    *,
    scenario: str | None = None,
    attack: int | None = None,
) -> RegionDynamicsMemory:
    """Load a phase/topology memory artifact with no route-like fields."""

    artifact_path = Path(path)
    raw = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("region dynamics memory must be a JSON object")
    allowed_top_level = {"schema_version", "kind", "scenario", "attack", "model"}
    unexpected_top_level = set(raw) - allowed_top_level
    if unexpected_top_level:
        raise ValueError(
            "region dynamics memory contains unsupported top-level fields: "
            + ", ".join(sorted(str(value) for value in unexpected_top_level))
        )
    schema_version = raw.get("schema_version")
    if (
        schema_version not in {1, 2}
        or raw.get("kind") != "region_dynamics_memory"
    ):
        raise ValueError("unsupported region dynamics memory schema")
    if scenario is not None and raw.get("scenario") != scenario:
        raise ValueError("region dynamics memory scenario does not match")
    if attack is not None and raw.get("attack") != attack:
        raise ValueError("region dynamics memory attack does not match")
    model = raw.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("region dynamics memory has no model object")
    allowed = {
        "phase_order",
        "minimum_radius",
        "maximum_radius",
        "growth_rate",
        "contraction_rate",
        "phase_durations",
        "cycle_frames",
    }
    if schema_version == 2:
        allowed.add("lateral_flow")
    unexpected = set(model) - allowed
    if unexpected:
        raise ValueError(
            "region dynamics memory contains unsupported fields: "
            + ", ".join(sorted(str(value) for value in unexpected))
        )
    phase_order = model.get("phase_order")
    if phase_order != list(_REGION_PHASE_ORDER):
        raise ValueError("region dynamics memory has an invalid phase order")
    durations = model.get("phase_durations")
    if not isinstance(durations, Mapping) or set(durations) != set(_REGION_PHASE_ORDER):
        raise ValueError("region dynamics memory must define all four phase durations")

    def finite_number(value: Any, label: str) -> float:
        result = _number(value)
        if result is None:
            raise ValueError(f"region dynamics {label} must be a finite number")
        return result

    lateral_flow_cycle_frames: float | None = None
    safe_side_rule: str | None = None
    if schema_version == 2:
        lateral_flow = model.get("lateral_flow")
        if not isinstance(lateral_flow, Mapping) or set(lateral_flow) != {
            "cycle_frames",
            "safe_side_rule",
        }:
            raise ValueError(
                "region dynamics v2 must define only the lateral flow cycle "
                "and safe-side rule"
            )
        lateral_flow_cycle_frames = finite_number(
            lateral_flow.get("cycle_frames"),
            "lateral_flow.cycle_frames",
        )
        safe_side_rule_value = lateral_flow.get("safe_side_rule")
        if safe_side_rule_value != "opposite_incoming_lateral_flow":
            raise ValueError("unsupported region safe-side rule")
        safe_side_rule = safe_side_rule_value

    return RegionDynamicsMemory(
        minimum_radius=finite_number(model.get("minimum_radius"), "minimum_radius"),
        maximum_radius=finite_number(model.get("maximum_radius"), "maximum_radius"),
        growth_rate=finite_number(model.get("growth_rate"), "growth_rate"),
        contraction_rate=finite_number(
            model.get("contraction_rate"), "contraction_rate",
        ),
        expanding_frames=finite_number(durations.get("expanding"), "expanding"),
        maximum_hold_frames=finite_number(
            durations.get("maximum_hold"), "maximum_hold",
        ),
        contracting_frames=finite_number(
            durations.get("contracting"), "contracting",
        ),
        minimum_hold_frames=finite_number(
            durations.get("minimum_hold"), "minimum_hold",
        ),
        cycle_frames=finite_number(model.get("cycle_frames"), "cycle_frames"),
        lateral_flow_cycle_frames=lateral_flow_cycle_frames,
        safe_side_rule=safe_side_rule,
    )


@dataclass(frozen=True, slots=True)
class MPCConfig:
    horizon_frames: int = 36
    decision_interval: int = 3
    observation_delay: int = 5
    boundary_weight: float = 1.0
    boss_alignment_weight: float = 1.0
    stale_track_frames: int = 48
    beam_width: int = 128
    region_beam_width: int = 512
    radius_rate_horizon: int = 6
    safe_margin_target: float = 12.0
    nonbullet_motion_horizon: int = 9
    preferred_y_fraction: float = 2.0 / 25.0
    vertical_anchor_weight: float = 0.25
    beam_cell_size: float = 4.0
    region_anchor_weight: float = 2.0
    region_boundary_trigger_margin: float = 72.0
    region_safe_margin_target: float = 1.0
    portal_clearance: float = 6.0
    region_path_weight: float = 0.05
    region_learned_min_radius: float = 7.0
    region_learned_max_radius: float = 28.0
    region_radius_step: float = 0.7
    region_dynamics_memory: RegionDynamicsMemory | None = None
    track_displacement_tolerance: float = 1.0

    def __post_init__(self) -> None:
        if self.horizon_frames < 36:
            raise ValueError("live MPC horizon must be at least 36 frames")
        if self.decision_interval != 3:
            raise ValueError("live MPC decision interval must be exactly three frames")
        if self.observation_delay < 0:
            raise ValueError("observation_delay cannot be negative")
        if self.stale_track_frames <= 0:
            raise ValueError("stale_track_frames must be positive")
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive")
        if self.region_beam_width <= 0:
            raise ValueError("region_beam_width must be positive")
        if self.radius_rate_horizon < 0:
            raise ValueError("radius_rate_horizon cannot be negative")
        if not math.isfinite(self.safe_margin_target) or self.safe_margin_target < 0.0:
            raise ValueError("safe_margin_target must be finite and nonnegative")
        if self.nonbullet_motion_horizon < 0:
            raise ValueError("nonbullet_motion_horizon cannot be negative")
        if (
            not math.isfinite(self.preferred_y_fraction)
            or not 0.0 <= self.preferred_y_fraction <= 1.0
        ):
            raise ValueError("preferred_y_fraction must be in [0, 1]")
        if not math.isfinite(self.vertical_anchor_weight) or self.vertical_anchor_weight < 0.0:
            raise ValueError("vertical_anchor_weight must be finite and nonnegative")
        region_values = (
            self.beam_cell_size,
            self.region_anchor_weight,
            self.region_boundary_trigger_margin,
            self.region_safe_margin_target,
            self.portal_clearance,
            self.region_learned_min_radius,
            self.region_learned_max_radius,
            self.region_radius_step,
            self.track_displacement_tolerance,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in region_values):
            raise ValueError("region-planning values must be finite and positive")
        if self.region_learned_max_radius <= self.region_learned_min_radius:
            raise ValueError("learned maximum radius must exceed the minimum")
        if not math.isfinite(self.region_path_weight) or self.region_path_weight < 0.0:
            raise ValueError("region_path_weight must be finite and nonnegative")
        weights = (self.boundary_weight, self.boss_alignment_weight)
        if not all(math.isfinite(value) and value >= 0.0 for value in weights):
            raise ValueError("MPC weights must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class PredictedThreat:
    key: str
    source: str
    object_id: Any
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    radius_rate: float
    source_frame: int
    observation_delay: int
    radius_rate_horizon: int
    motion_horizon: int

    def at(self, future_frame: int) -> tuple[float, float, float]:
        if future_frame < 0:
            raise ValueError("future_frame cannot be negative")
        return (
            self.x + self.vx * min(future_frame, self.motion_horizon),
            self.y + self.vy * min(future_frame, self.motion_horizon),
            max(
                0.1,
                self.radius
                + self.radius_rate * min(future_frame, self.radius_rate_horizon),
            ),
        )


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    action: Action
    collided: bool
    collision_frames: int
    earliest_collision_frame: int | None
    minimum_margin: float
    boundary_penalty: float
    boss_alignment: float

    @property
    def selection_key(self) -> tuple[float, ...]:
        margin = self.minimum_margin
        earliest = (
            math.inf
            if self.earliest_collision_frame is None else
            float(self.earliest_collision_frame)
        )
        return (
            float(self.collided),
            -earliest,
            float(self.collision_frames),
            -margin,
            self.boundary_penalty,
            self.boss_alignment,
        )


@dataclass(frozen=True, slots=True)
class MPCDecision:
    action: Action
    source_frame: int
    recomputed: bool
    threats: tuple[PredictedThreat, ...]
    evaluations: tuple[CandidateEvaluation, ...]
    region_anchor: tuple[float, float] | None
    region_crossing: bool
    region_path_margin: float | None
    region_evacuating: bool
    region_target_rows_ahead: int
    region_navigation_mode: str
    region_current_component: str | None
    region_target_component: str | None
    region_portal: str | None
    region_deadline_slack: float | None
    planned_actions: tuple[Action, ...]
    using_committed_plan: bool
    committed_plan_immediate_margin: float | None
    committed_plan_current_horizon_margin: float | None
    region_phase: str
    region_phase_started_frame: int | None
    region_learned_cycle_frames: float | None
    region_frames_until_expansion: float | None
    region_observed_radius: float | None


@dataclass(slots=True)
class _Track:
    frame: int
    x: float
    y: float
    radius: float
    vx: float
    vy: float
    radius_rate: float


@dataclass(frozen=True, slots=True)
class _RegionAnchor:
    x: float
    y: float
    crossing: bool
    path_margin: float
    evacuating: bool
    target_rows_ahead: int
    navigation_mode: str
    current_component: str
    target_component: str
    portal: str | None
    deadline_slack: float

    @property
    def commitment_key(self) -> tuple[str, str, str, str | None]:
        return (
            self.navigation_mode,
            self.current_component,
            self.target_component,
            self.portal,
        )


@dataclass(frozen=True, slots=True)
class _RegionSideForecast:
    """Safe exterior inferred from the next relative expansion window."""

    side: str
    x: float
    frames_until_expansion: float
    open_samples: int
    total_samples: int


@dataclass(slots=True)
class _RegionRowTrack:
    identity: int
    keys: frozenset[str]
    center_x: float
    center_y: float
    vx: float
    vy: float
    frame: int


@dataclass(slots=True)
class _RegionTopologyMemory:
    """Persist safe-component intent while coordinates continue to move."""

    target_component: str | None = None
    portal: str | None = None
    navigation_mode: str = "hold"
    revision: int = 0
    next_row_identity: int = 1
    row_tracks: dict[int, _RegionRowTrack] = field(default_factory=dict)

    def update(
        self,
        *,
        target_component: str,
        portal: str | None,
        navigation_mode: str,
    ) -> None:
        state = (target_component, portal, navigation_mode)
        previous = (self.target_component, self.portal, self.navigation_mode)
        if state != previous:
            self.revision += 1
        self.target_component = target_component
        self.portal = portal
        self.navigation_mode = navigation_mode

    def match_rows(
        self,
        rows: Sequence[Sequence[PredictedThreat]],
        frame: int,
    ) -> tuple[str, ...]:
        """Match moving rows by overlap, then geometry, without frame coordinates."""

        features: list[tuple[frozenset[str], float, float, float, float]] = []
        for row in rows:
            count = len(row)
            features.append((
                frozenset(item.key for item in row),
                sum(item.x for item in row) / count,
                sum(item.y for item in row) / count,
                sum(item.vx for item in row) / count,
                sum(item.vy for item in row) / count,
            ))

        candidates: list[tuple[int, float, float, int, int]] = []
        for row_index, (keys, center_x, center_y, _vx, _vy) in enumerate(features):
            for identity, track in self.row_tracks.items():
                delta = max(0, frame - track.frame)
                predicted_x = track.center_x + track.vx * delta
                predicted_y = track.center_y + track.vy * delta
                distance = math.hypot(center_x - predicted_x, center_y - predicted_y)
                overlap = len(keys & track.keys)
                overlap_ratio = overlap / max(1, min(len(keys), len(track.keys)))
                geometry_limit = 18.0 + 0.5 * math.hypot(track.vx, track.vy) * delta
                if overlap == 0 and distance > geometry_limit:
                    continue
                candidates.append((
                    int(overlap > 0),
                    overlap_ratio,
                    -distance,
                    row_index,
                    identity,
                ))

        assigned_rows: dict[int, int] = {}
        assigned_tracks: set[int] = set()
        for _has_overlap, _ratio, _distance, row_index, identity in sorted(
            candidates,
            reverse=True,
        ):
            if row_index in assigned_rows or identity in assigned_tracks:
                continue
            assigned_rows[row_index] = identity
            assigned_tracks.add(identity)

        labels: list[str] = []
        for row_index, (keys, center_x, center_y, vx, vy) in enumerate(features):
            identity = assigned_rows.get(row_index)
            if identity is None:
                identity = self.next_row_identity
                self.next_row_identity += 1
            self.row_tracks[identity] = _RegionRowTrack(
                identity=identity,
                keys=keys,
                center_x=center_x,
                center_y=center_y,
                vx=vx,
                vy=vy,
                frame=frame,
            )
            labels.append(f"row:{identity}")

        oldest = frame - 12
        self.row_tracks = {
            identity: track
            for identity, track in self.row_tracks.items()
            if track.frame >= oldest or identity in assigned_tracks
        }
        return tuple(labels)


class _RegionPhaseMemory:
    """Learn the ordered wall-radius cycle without memorizing attack frames."""

    def __init__(
        self,
        *,
        trend_threshold: float = 0.1,
        minimum_hint: float | None = None,
        maximum_hint: float | None = None,
        rate_hint: float | None = None,
        dynamics_memory: RegionDynamicsMemory | None = None,
    ) -> None:
        self.trend_threshold = trend_threshold
        self.dynamics_memory = dynamics_memory
        self.minimum_hint = (
            dynamics_memory.minimum_radius
            if dynamics_memory is not None else minimum_hint
        )
        self.maximum_hint = (
            dynamics_memory.maximum_radius
            if dynamics_memory is not None else maximum_hint
        )
        self.rate_hint = (
            dynamics_memory.growth_rate
            if dynamics_memory is not None else rate_hint
        )
        self.duration_hints = (
            dynamics_memory.phase_durations
            if dynamics_memory is not None else {}
        )
        self.phase = "unknown"
        self.phase_started_frame: int | None = None
        self.phase_epoch = 0
        self.last_frame: int | None = None
        self.observed_radius: float | None = None
        self.observed_rate = 0.0
        self.minimum_radius: float | None = None
        self.maximum_radius: float | None = None
        self.minimum_plateau_radius: float | None = self.minimum_hint
        self.maximum_plateau_radius: float | None = None
        self.growth_rate: float | None = self.rate_hint
        self.contraction_rate: float | None = (
            dynamics_memory.contraction_rate
            if dynamics_memory is not None else rate_hint
        )
        self._growth_rates: list[float] = []
        self._contraction_rates: list[float] = []
        self.expansion_starts: list[int] = []
        self.phase_starts: dict[str, list[int]] = {}
        self.phase_durations: dict[str, list[int]] = {}
        self.history: list[tuple[int, float]] = []

    @property
    def learned_cycle_frames(self) -> float | None:
        intervals = [
            current - previous
            for previous, current in zip(
                self.expansion_starts,
                self.expansion_starts[1:],
            )
            if 60 <= current - previous <= 600
        ]
        if intervals:
            return float(statistics.median(intervals))
        if self.dynamics_memory is not None:
            return self.dynamics_memory.cycle_frames
        return None

    def _phase_duration(self, phase: str) -> float | None:
        observed = self.phase_durations.get(phase, ())
        hint = self.duration_hints.get(phase)
        if hint is None:
            return float(statistics.median(observed)) if observed else None
        # Two prior pseudo-observations prevent an exceptional opening hold
        # from replacing a repeated-cycle duration after only one transition.
        return float(statistics.median((hint, hint, *observed[-6:])))

    def _transition(
        self,
        phase: str,
        frame: int,
        *,
        started_frame: int | None = None,
    ) -> None:
        if phase == self.phase:
            return
        start = frame if started_frame is None else started_frame
        if self.phase_started_frame is not None:
            start = max(self.phase_started_frame, start)
        start = min(frame, start)
        if self.phase_started_frame is not None and self.phase != "unknown":
            duration = start - self.phase_started_frame
            if duration > 0:
                durations = self.phase_durations.setdefault(self.phase, [])
                durations.append(duration)
                self.phase_durations[self.phase] = durations[-8:]
        self.phase = phase
        self.phase_started_frame = start
        self.phase_epoch += 1
        starts = self.phase_starts.setdefault(phase, [])
        if not starts or start > starts[-1]:
            starts.append(start)
            self.phase_starts[phase] = starts[-8:]
        if phase == "expanding":
            if not self.expansion_starts or start > self.expansion_starts[-1]:
                self.expansion_starts.append(start)
                self.expansion_starts = self.expansion_starts[-8:]

    def _local_rate(self, direction: str) -> float | None:
        if len(self.history) < 2:
            return None
        (previous_frame, previous_radius), (frame, radius) = self.history[-2:]
        if frame <= previous_frame:
            return None
        slope = (radius - previous_radius) / (frame - previous_frame)
        if direction == "up" and slope > self.trend_threshold:
            return slope
        if direction == "down" and slope < -self.trend_threshold:
            return -slope
        return None

    def _moving_start(
        self,
        *,
        frame: int,
        radius: float,
        baseline: float,
        rate: float,
        direction: str,
    ) -> int:
        distance = radius - baseline if direction == "up" else baseline - radius
        inferred = frame - int(round(max(0.0, distance) / max(rate, 1e-6)))
        if self.phase_started_frame is not None:
            inferred = max(self.phase_started_frame, inferred)
        return min(frame, inferred)

    def _hold_start(
        self,
        *,
        frame: int,
        plateau: float,
        rate: float,
        direction: str,
    ) -> int:
        for sample_frame, sample_radius in reversed(self.history[:-1]):
            distance = (
                plateau - sample_radius
                if direction == "up" else
                sample_radius - plateau
            )
            if distance > 1e-6:
                inferred = sample_frame + int(round(distance / max(rate, 1e-6)))
                if self.phase_started_frame is not None:
                    inferred = max(self.phase_started_frame, inferred)
                return min(frame, inferred)
        return frame

    def _plateau_hysteresis(self, radius: float) -> bool:
        tolerance = max(1.05, 1.5 * (self.rate_hint or 0.0))
        if (
            self.phase == "minimum_hold"
            and self.minimum_plateau_radius is not None
            and abs(radius - self.minimum_plateau_radius) <= tolerance
        ):
            return True
        return (
            self.phase == "maximum_hold"
            and self.maximum_plateau_radius is not None
            and abs(radius - self.maximum_plateau_radius) <= tolerance
        )

    def _trend(self) -> tuple[str, float]:
        if len(self.history) < 4:
            return "stable", 0.0
        window = self.history[-4:]
        slopes = sorted(
            (radius_b - radius_a) / (frame_b - frame_a)
            for index, (frame_a, radius_a) in enumerate(window)
            for frame_b, radius_b in window[index + 1:]
            if frame_b > frame_a
        )
        slope = float(statistics.median(slopes)) if slopes else 0.0
        net = window[-1][1] - window[0][1]
        variation = sum(
            abs(current[1] - previous[1])
            for previous, current in zip(window, window[1:])
        )
        efficiency = abs(net) / variation if variation > 1e-9 else 0.0
        if (
            slope > self.trend_threshold
            and net > 1.0
            and efficiency >= 0.6
        ):
            return "up", slope
        if (
            slope < -self.trend_threshold
            and net < -1.0
            and efficiency >= 0.6
        ):
            return "down", slope
        return "stable", slope

    def update(self, frame: int, radii: Sequence[float]) -> None:
        if not radii or (self.last_frame is not None and frame <= self.last_frame):
            return
        radius = float(statistics.median(radii))
        previous_minimum = self.minimum_radius
        self.last_frame = frame
        self.observed_radius = radius
        self.history.append((frame, radius))
        self.history = self.history[-8:]
        self.minimum_radius = (
            radius if previous_minimum is None else min(previous_minimum, radius)
        )
        self.maximum_radius = (
            radius if self.maximum_radius is None else max(self.maximum_radius, radius)
        )
        trend, slope = self._trend()
        if self._plateau_hysteresis(radius):
            trend = "stable"
        self.observed_rate = slope
        new_phase = self.phase
        if self.phase == "unknown":
            if trend == "up":
                new_phase = "expanding"
            elif trend == "down":
                new_phase = "contracting"
            elif self.minimum_hint is not None and abs(
                radius - self.minimum_hint
            ) <= max(1.05, 1.5 * (self.rate_hint or 0.0)):
                new_phase = "minimum_hold"
            elif self.maximum_hint is not None and abs(
                radius - self.maximum_hint
            ) <= max(1.05, 1.5 * (self.rate_hint or 0.0)):
                new_phase = "maximum_hold"
        elif self.phase == "minimum_hold" and trend == "up":
            new_phase = "expanding"
        elif self.phase == "expanding" and trend == "stable":
            new_phase = "maximum_hold"
        elif self.phase == "maximum_hold" and trend == "down":
            new_phase = "contracting"
        elif self.phase == "contracting" and trend == "stable":
            new_phase = "minimum_hold"

        started_frame: int | None = None
        if new_phase == "expanding" and self.phase != "expanding":
            local_rate = self._local_rate("up") or self.growth_rate or abs(slope)
            baseline = (
                self.minimum_plateau_radius
                if self.minimum_plateau_radius is not None else
                min(value for _sample_frame, value in self.history)
            )
            started_frame = self._moving_start(
                frame=frame,
                radius=radius,
                baseline=baseline,
                rate=local_rate,
                direction="up",
            )
            self._growth_rates.append(local_rate)
            self._growth_rates = self._growth_rates[-16:]
            self.growth_rate = float(statistics.median(self._growth_rates))
        elif new_phase == "contracting" and self.phase != "contracting":
            local_rate = self._local_rate("down") or self.contraction_rate or abs(slope)
            ceiling = (
                self.maximum_plateau_radius
                if self.maximum_plateau_radius is not None else
                max(value for _sample_frame, value in self.history)
            )
            started_frame = self._moving_start(
                frame=frame,
                radius=radius,
                baseline=ceiling,
                rate=local_rate,
                direction="down",
            )
            self._contraction_rates.append(local_rate)
            self._contraction_rates = self._contraction_rates[-16:]
            self.contraction_rate = float(statistics.median(self._contraction_rates))
        elif new_phase == "maximum_hold" and self.phase == "expanding":
            plateau = max(value for _sample_frame, value in self.history[-4:])
            self.maximum_plateau_radius = max(
                plateau,
                self.maximum_plateau_radius or -math.inf,
            )
            rate = self.growth_rate or self.rate_hint or self.trend_threshold
            started_frame = self._hold_start(
                frame=frame,
                plateau=self.maximum_plateau_radius,
                rate=rate,
                direction="up",
            )
        elif new_phase == "minimum_hold" and self.phase == "contracting":
            recent = [value for _sample_frame, value in self.history[-4:]]
            plateau = float(statistics.median(recent))
            if self.minimum_hint is not None:
                plateau = self.minimum_hint
            self.minimum_plateau_radius = plateau
            rate = self.contraction_rate or self.rate_hint or self.trend_threshold
            started_frame = self._hold_start(
                frame=frame,
                plateau=plateau,
                rate=rate,
                direction="down",
            )
        elif new_phase == "expanding" and trend == "up":
            local_rate = self._local_rate("up")
            if local_rate is not None:
                self._growth_rates.append(local_rate)
                self._growth_rates = self._growth_rates[-16:]
                self.growth_rate = float(statistics.median(self._growth_rates))
        elif new_phase == "contracting" and trend == "down":
            local_rate = self._local_rate("down")
            if local_rate is not None:
                self._contraction_rates.append(local_rate)
                self._contraction_rates = self._contraction_rates[-16:]
                self.contraction_rate = float(statistics.median(self._contraction_rates))
        self._transition(new_phase, frame, started_frame=started_frame)

    def frames_until_expansion(self) -> float | None:
        if self.phase == "expanding":
            return 0.0
        cycle = self.learned_cycle_frames
        if self.last_frame is None:
            return None
        if cycle is not None and self.expansion_starts:
            next_start = self.expansion_starts[-1] + cycle
            return max(0.0, next_start - self.last_frame)
        if self.phase_started_frame is None or self.phase not in _REGION_PHASE_ORDER:
            return None
        phase_index = _REGION_PHASE_ORDER.index(self.phase)
        current_duration = self._phase_duration(self.phase)
        if current_duration is None:
            return None
        elapsed = max(0.0, self.last_frame - self.phase_started_frame)
        remaining = max(0.0, current_duration - elapsed)
        for phase in _REGION_PHASE_ORDER[phase_index + 1:]:
            if phase == "expanding":
                break
            duration = self._phase_duration(phase)
            if duration is None:
                return None
            remaining += duration
        return remaining

    def frames_until_radius(self, target_radius: float) -> float | None:
        if not math.isfinite(target_radius) or target_radius <= 0.0:
            raise ValueError("target_radius must be finite and positive")
        if self.observed_radius is None or self.growth_rate is None:
            return None
        envelope_radius = self.observed_radius
        if self.phase == "maximum_hold" and self.maximum_plateau_radius is not None:
            envelope_radius = max(envelope_radius, self.maximum_plateau_radius)
        if envelope_radius >= target_radius:
            return 0.0
        ceiling = self.maximum_plateau_radius or self.maximum_hint
        if ceiling is not None and target_radius > ceiling + 1e-6:
            return math.inf
        if self.phase == "expanding":
            return max(
                0.0,
                (target_radius - self.observed_radius) / self.growth_rate,
            )
        until = self.frames_until_expansion()
        if until is None:
            return None
        baseline = self.minimum_plateau_radius or self.observed_radius
        return until + max(0.0, target_radius - baseline) / self.growth_rate

    def radius_after(self, future_frame: int) -> float | None:
        if future_frame < 0:
            raise ValueError("future_frame cannot be negative")
        if self.observed_radius is None:
            return None
        rate = self.growth_rate
        ceiling = self.maximum_plateau_radius or self.maximum_hint
        if self.phase == "expanding" and rate is not None:
            value = self.observed_radius + rate * future_frame
            return value if ceiling is None else min(ceiling, value)
        if self.phase == "minimum_hold" and rate is not None:
            until = self.frames_until_expansion()
            if until is None or future_frame <= until:
                return max(
                    self.observed_radius,
                    self.minimum_plateau_radius or self.observed_radius,
                )
            baseline = self.minimum_plateau_radius or self.observed_radius
            value = baseline + rate * (future_frame - until)
            return value if ceiling is None else min(ceiling, value)
        if self.phase == "maximum_hold" and ceiling is not None:
            return ceiling
        # Holding the current envelope while contracting is conservative and
        # avoids inventing future safe space from an uncertain shrink rate.
        return self.observed_radius


def movement_actions() -> tuple[Action, ...]:
    """Return 17 unique movement choices: neutral plus 8 directions at 2 speeds."""

    values = [Action(move_x=0, move_y=0, slow=True, spell=False)]
    directions = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )
    for slow in (False, True):
        values.extend(
            Action(move_x=x, move_y=y, slow=slow, spell=False)
            for x, y in directions
        )
    return tuple(values)


class VisibleTrackEstimator:
    """Estimate visible motion without consuming raw vx/vy authority fields."""

    def __init__(self, config: MPCConfig = MPCConfig()) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._tracks: dict[str, _Track] = {}
        self._previous_visible: set[str] = set()
        self._fallback_frame = -1
        self.last_frame: int | None = None

    @staticmethod
    def _key(source: str, record: Mapping[str, Any], ordinal: int) -> str:
        return f"{source}:{record.get('id', 'ordinal-' + str(ordinal))}"

    def update(self, observation: Mapping[str, Any]) -> tuple[PredictedThreat, ...]:
        observation = _unwrap_observation(observation)
        self._fallback_frame += 1
        frame = _frame(observation, self._fallback_frame)
        if self.last_frame is not None and frame < self.last_frame:
            self.reset()
            self._fallback_frame = frame
        self.last_frame = frame
        visible: list[PredictedThreat] = []
        seen: set[str] = set()
        delay = self.config.observation_delay

        for source in _OBJECT_ARRAYS:
            records = observation.get(source)
            if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
                continue
            for ordinal, record in enumerate(records):
                if not isinstance(record, Mapping) or record.get("collidable", True) is not True:
                    continue
                x = _number(record.get("x"))
                y = _number(record.get("y"))
                if x is None or y is None:
                    continue
                radius = _radius(record)
                key = self._key(source, record, ordinal)
                previous = self._tracks.get(key)
                continuous = previous is not None and key in self._previous_visible
                elapsed = frame - previous.frame if continuous else 0
                estimated_vx = (
                    (x - previous.x) / elapsed
                    if continuous and elapsed > 0 else 0.0
                )
                estimated_vy = (
                    (y - previous.y) / elapsed
                    if continuous and elapsed > 0 else 0.0
                )
                dx = _number(record.get("dx"))
                dy = _number(record.get("dy"))
                visible_vx = dx or 0.0
                visible_vy = dy or 0.0
                displacement_available = dx is not None and dy is not None
                if (
                    continuous
                    and elapsed > 0
                    and displacement_available
                    and math.hypot(
                        estimated_vx - visible_vx,
                        estimated_vy - visible_vy,
                    ) > self.config.track_displacement_tolerance
                ):
                    continuous = False
                    elapsed = 0
                # Once two visible samples exist, derive motion from them.
                # Per-frame visible displacement also rejects immediately reused IDs.
                vx = estimated_vx if continuous and elapsed > 0 else visible_vx
                vy = estimated_vy if continuous and elapsed > 0 else visible_vy
                radius_rate = (
                    (radius - previous.radius) / elapsed
                    if continuous and elapsed > 0 else 0.0
                )
                current = _Track(frame, x, y, radius, vx, vy, radius_rate)
                self._tracks[key] = current
                seen.add(key)
                motion_horizon = (
                    self.config.nonbullet_motion_horizon
                    if source in {"enemies", "nontjt_enemies"} else
                    self.config.horizon_frames
                )
                predicted_radius_rate = radius_rate
                radius_rate_horizon = self.config.radius_rate_horizon
                predicted_radius = radius
                if source == "indestructibles":
                    minimum = self.config.region_learned_min_radius
                    maximum = self.config.region_learned_max_radius
                    step = self.config.region_radius_step
                    if radius <= minimum + step + 1e-6:
                        # The 0.1 target oscillates by one Lua scale step too.
                        # Treat it as a stable lower plateau, with the upper
                        # edge of that oscillation as the collision envelope.
                        predicted_radius = minimum + step
                        predicted_radius_rate = 0.0
                        radius_rate_horizon = 0
                    elif radius >= maximum - 1e-6:
                        predicted_radius_rate = 0.0
                        radius_rate_horizon = 0
                    elif (
                        radius >= maximum - step - 1e-6
                        and radius_rate < -0.1
                    ):
                        # Lua's 0.01 scale step alternates at the 0.4 target
                        # because of floating-point comparison. Keep the visible
                        # one-step dip inside the learned maximum envelope.
                        predicted_radius = maximum
                        predicted_radius_rate = 0.0
                        radius_rate_horizon = 0
                    elif radius_rate > 0.1:
                        predicted_radius_rate = max(radius_rate, step)
                visible.append(PredictedThreat(
                    key=key,
                    source=source,
                    object_id=record.get("id", ordinal),
                    x=x + vx * min(delay, motion_horizon),
                    y=y + vy * min(delay, motion_horizon),
                    vx=vx,
                    vy=vy,
                    radius=max(
                        0.1,
                        predicted_radius
                        + predicted_radius_rate * min(delay, radius_rate_horizon),
                    ),
                    radius_rate=predicted_radius_rate,
                    source_frame=frame,
                    observation_delay=delay,
                    radius_rate_horizon=radius_rate_horizon,
                    motion_horizon=motion_horizon,
                ))

        oldest = frame - self.config.stale_track_frames
        self._tracks = {
            key: track
            for key, track in self._tracks.items()
            if key in seen or track.frame >= oldest
        }
        self._previous_visible = seen
        return tuple(visible)


class EngineMPC:
    """Short-horizon teacher callable from a live runner's main thread."""

    def __init__(self, config: MPCConfig = MPCConfig()) -> None:
        self.config = config
        self.estimator = VisibleTrackEstimator(config)
        self.actions = movement_actions()
        self.reset()

    def reset(self) -> None:
        self.estimator.reset()
        self._last_source_frame: int | None = None
        self._last_decision_frame: int | None = None
        self._decision: MPCDecision | None = None
        self._region_phase = _RegionPhaseMemory(
            minimum_hint=self.config.region_learned_min_radius,
            maximum_hint=self.config.region_learned_max_radius,
            rate_hint=self.config.region_radius_step,
            dynamics_memory=self.config.region_dynamics_memory,
        )
        self._region_topology = _RegionTopologyMemory()
        self._committed_plan: tuple[Action, ...] = ()
        self._committed_plan_is_region = False
        self._committed_plan_evacuating = False
        self._committed_plan_key: tuple[Any, ...] | None = None

    @staticmethod
    def _player(observation: Mapping[str, Any], delay: int) -> tuple[float, float, float, float, float]:
        player = observation.get("player")
        if not isinstance(player, Mapping):
            raise ValueError("engine observation has no player object")
        x = _number(player.get("x"))
        y = _number(player.get("y"))
        if x is None or y is None:
            raise ValueError("engine player has no finite position")
        dx = _number(player.get("dx"), 0.0) or 0.0
        dy = _number(player.get("dy"), 0.0) or 0.0
        speed = max(0.1, _number(player.get("hspeed"), 4.0) or 4.0)
        focus_speed = max(0.1, _number(player.get("lspeed"), 2.0) or 2.0)
        player_delay = _number(observation.get("own_player_observation_delay"), delay)
        assert player_delay is not None
        player_delay = max(0.0, player_delay)
        return (
            x + dx * player_delay,
            y + dy * player_delay,
            _radius(player, 0.5),
            speed,
            focus_speed,
        )

    @staticmethod
    def _bounds(observation: Mapping[str, Any], player_radius: float) -> tuple[float, float, float, float]:
        world = observation.get("world")
        world = world if isinstance(world, Mapping) else {}
        raw_bounds = (
            _number(world.get("pl", world.get("l")), -192.0),
            _number(world.get("pr", world.get("r")), 192.0),
            _number(world.get("pb", world.get("b")), -224.0),
            _number(world.get("pt", world.get("t")), 224.0),
        )
        left, right, bottom, top = (float(value) for value in raw_bounds)
        # THlib clamps the player's center with fixed sprite margins.
        adjusted = left + 8.0, right - 8.0, bottom + 16.0, top - 32.0
        if adjusted[0] >= adjusted[1] or adjusted[2] >= adjusted[3]:
            inset = max(0.0, player_radius)
            adjusted = left + inset, right - inset, bottom + inset, top - inset
        if adjusted[0] >= adjusted[1] or adjusted[2] >= adjusted[3]:
            raise ValueError("engine observation has invalid player bounds")
        return adjusted

    def _preferred_y(self, bounds: tuple[float, float, float, float]) -> float:
        bottom, top = bounds[2], bounds[3]
        return bottom + (top - bottom) * self.config.preferred_y_fraction

    def _update_region_phase(
        self,
        observation: Mapping[str, Any],
        source_frame: int,
    ) -> None:
        records = observation.get("indestructibles")
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            return
        radii = [
            _radius(record)
            for record in records
            if isinstance(record, Mapping)
            and record.get("collidable", True) is True
        ]
        self._region_phase.update(source_frame, radii)

    def _threat_at(
        self,
        threat: PredictedThreat,
        future_frame: int,
    ) -> tuple[float, float, float]:
        x, y, radius = threat.at(future_frame)
        if threat.source == "indestructibles":
            phase_radius = self._region_phase.radius_after(
                self.config.observation_delay + future_frame,
            )
            if phase_radius is not None:
                radius = max(radius, phase_radius)
        return x, y, radius

    def _frames_until_region_expansion(self) -> float | None:
        value = self._region_phase.frames_until_expansion()
        if value is None:
            return None
        return max(0.0, value - self.config.observation_delay)

    def _frames_until_region_radius(self, radius: float) -> float | None:
        value = self._region_phase.frames_until_radius(radius)
        if value is None:
            return None
        return max(0.0, value - self.config.observation_delay)

    def _region_side_forecast(
        self,
        rows: Sequence[Sequence[PredictedThreat]],
        player: tuple[float, float, float, float, float],
        bounds: tuple[float, float, float, float],
    ) -> _RegionSideForecast | None:
        """Infer the safer exterior from flow during the next expansion.

        This is intentionally episode-translation invariant. It projects only
        currently visible row geometry from the learned relative phase; it does
        not retain a side sequence, coordinate, action, or episode-frame cue.
        """

        dynamics_memory = self._region_phase.dynamics_memory
        if (
            dynamics_memory is None
            or dynamics_memory.safe_side_rule
            != "opposite_incoming_lateral_flow"
        ):
            return None
        if self._region_phase.phase not in {
            "minimum_hold",
            "contracting",
            "expanding",
        }:
            return None
        frames_until_expansion = self._frames_until_region_expansion()
        expansion_frames = self._region_phase._phase_duration("expanding")
        if frames_until_expansion is None or expansion_frames is None:
            return None
        if not math.isfinite(frames_until_expansion + expansion_frames):
            return None

        minimum_radius = (
            self._region_phase.minimum_plateau_radius
            or self.config.region_learned_min_radius
        )
        maximum_radius = (
            self._region_phase.maximum_plateau_radius
            or self.config.region_learned_max_radius
        )
        growth_rate = (
            self._region_phase.growth_rate
            or self.config.region_radius_step
        )
        left, right, bottom, top = bounds
        player_radius = player[2]
        side_clearance = (
            player_radius
            + self.config.portal_clearance
            + self.config.region_safe_margin_target
        )
        # Start, midpoint, and end describe topology over the expansion rather
        # than at a single trigger instant.
        phase_offsets = (0.0, 0.5 * expansion_frames, expansion_frames)
        widths: dict[str, list[float]] = {"left": [], "right": []}
        targets: dict[str, list[float]] = {"left": [], "right": []}
        for phase_offset in phase_offsets:
            future_frame = frames_until_expansion + phase_offset
            radius = min(
                maximum_radius,
                minimum_radius + growth_rate * phase_offset,
            )
            for row in rows:
                center_y = sum(
                    item.y + item.vy * future_frame for item in row
                ) / len(row)
                if (
                    center_y + radius + player_radius < bottom
                    or center_y - radius - player_radius > top
                ):
                    continue
                projected_left = min(
                    item.x + item.vx * future_frame for item in row
                )
                projected_right = max(
                    item.x + item.vx * future_frame for item in row
                )
                left_target = projected_left - radius - side_clearance
                right_target = projected_right + radius + side_clearance
                widths["left"].append(left_target - left)
                widths["right"].append(right - right_target)
                targets["left"].append(left_target)
                targets["right"].append(right_target)

        if not widths["left"] or len(widths["left"]) != len(widths["right"]):
            return None

        def side_key(side: str) -> tuple[int, float, float]:
            values = widths[side]
            return (
                sum(value >= 0.0 for value in values),
                sum(values),
                min(values),
            )

        left_key = side_key("left")
        right_key = side_key("right")
        if left_key == right_key:
            return None
        side = "left" if left_key > right_key else "right"
        other = "right" if side == "left" else "left"
        # A one-sample numerical edge is not enough to reverse a persistent
        # component. Require either more open samples or a full-player-diameter
        # aggregate clearance advantage.
        if (
            side_key(side)[0] == side_key(other)[0]
            and side_key(side)[1] - side_key(other)[1]
            < 2.0 * player_radius
        ):
            return None
        target_x = (
            min(targets[side]) if side == "left" else max(targets[side])
        )
        return _RegionSideForecast(
            side=side,
            x=min(max(target_x, left), right),
            frames_until_expansion=frames_until_expansion,
            open_samples=side_key(side)[0],
            total_samples=len(widths[side]),
        )

    @staticmethod
    def _boss_x(observation: Mapping[str, Any], delay: int) -> float | None:
        candidates: list[tuple[float, float]] = []
        for source in ("enemies", "nontjt_enemies"):
            records = observation.get(source)
            if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
                continue
            for record in records:
                if not isinstance(record, Mapping):
                    continue
                x = _number(record.get("x"))
                if x is None:
                    continue
                dx = _number(record.get("dx"), 0.0) or 0.0
                max_hp = _number(record.get("maxhp"), _number(record.get("hp"), 0.0)) or 0.0
                if 0.0 < max_hp < 100_000_000.0:
                    candidates.append((max_hp, x + dx * delay))
        return max(candidates, default=(0.0, None), key=lambda item: item[0])[1]

    def _region_anchor(
        self,
        player: tuple[float, float, float, float, float],
        bounds: tuple[float, float, float, float],
        threats: Sequence[PredictedThreat],
        source_frame: int,
    ) -> _RegionAnchor | None:
        """Track time-varying safe components and route through their next portal."""

        objects = sorted(
            (value for value in threats if value.source == "indestructibles"),
            key=lambda value: (value.y, value.x, str(value.object_id)),
        )
        if len(objects) < 3:
            return None
        rows: list[list[PredictedThreat]] = []
        for value in objects:
            if not rows:
                rows.append([value])
                continue
            center_y = sum(item.y for item in rows[-1]) / len(rows[-1])
            if abs(value.y - center_y) <= 6.0:
                rows[-1].append(value)
            else:
                rows.append([value])
        rows = [sorted(row, key=lambda item: item.x) for row in rows if len(row) >= 3]
        if not rows:
            return None
        row_labels = self._region_topology.match_rows(rows, source_frame)
        row_label_by_keys = {
            tuple(item.key for item in row): label
            for row, label in zip(rows, row_labels, strict=True)
        }

        row_y = [sum(item.y for item in row) / len(row) for row in rows]
        px, py, player_radius, speed = player[:4]
        left, right, bottom, top = bounds
        side_forecast = self._region_side_forecast(rows, player, bounds)
        preferred_y = self._preferred_y(bounds)
        lower_index = next((
            index
            for index in range(len(rows) - 1)
            if row_y[index] <= py <= row_y[index + 1]
        ), None)
        virtual_lower = py < row_y[0]
        if lower_index is None and not virtual_lower:
            return None
        upper_index = 0 if virtual_lower else lower_index + 1
        upper = rows[upper_index]

        def row_signature(row: Sequence[PredictedThreat]) -> str:
            return row_label_by_keys[tuple(item.key for item in row)]

        def component_id(
            lower_row: Sequence[PredictedThreat] | None,
            upper_row: Sequence[PredictedThreat] | None,
        ) -> str:
            lower_id = "bottom" if lower_row is None else row_signature(lower_row)
            upper_id = "top" if upper_row is None else row_signature(upper_row)
            return f"band:{lower_id}>{upper_id}"

        current_component = component_id(
            None if virtual_lower else rows[lower_index],
            upper,
        )
        upper_radius = max(item.radius for item in upper)
        upper_y = row_y[upper_index]
        upper_edge = upper_y - upper_radius
        if virtual_lower:
            lower: list[PredictedThreat] = []
            lower_radius = 0.0
            lower_edge = bottom
            band_y = min(
                preferred_y,
                upper_edge - self.config.portal_clearance - player_radius,
            )
            mean_vy = 0.0
        else:
            assert lower_index is not None
            lower = rows[lower_index]
            lower_radius = max(item.radius for item in lower)
            lower_edge = row_y[lower_index] + lower_radius
            band_y = 0.5 * (
                row_y[lower_index] + upper_y
                if upper_edge <= lower_edge else
                lower_edge + upper_edge
            )
            mean_vy = (
                sum(item.vy for item in lower + upper) / (len(lower) + len(upper))
            )
        component_clearance = player_radius + self.config.portal_clearance
        adjacent_rows = [upper] if virtual_lower else [lower, upper]
        exterior_left = min(
            row[0].x - row[0].radius - component_clearance
            for row in adjacent_rows
        )
        exterior_right = max(
            row[-1].x + row[-1].radius + component_clearance
            for row in adjacent_rows
        )
        if px <= exterior_left:
            current_component = "exterior:left"
        elif px >= exterior_right:
            current_component = "exterior:right"
        band_y += mean_vy * self.config.decision_interval
        band_y = min(max(band_y, bottom), top)

        if virtual_lower:
            transition_edge = upper_edge
            transition_vy = sum(item.vy for item in upper) / len(upper)
        else:
            assert lower_index is not None
            transition_edge = row_y[lower_index] - lower_radius
            transition_vy = sum(item.vy for item in lower) / len(lower)
        trigger_y = bottom + self.config.region_boundary_trigger_margin
        flow_wait = math.inf
        if transition_edge <= trigger_y:
            flow_wait = 0.0
        elif transition_vy < -0.1:
            flow_wait = (transition_edge - trigger_y) / -transition_vy

        maximum_radius = (
            self._region_phase.maximum_plateau_radius
            or self.config.region_learned_max_radius
        )
        expansion_duration = (
            self._region_phase._phase_duration("expanding") or 30.0
        )
        if self._region_phase.phase == "expanding":
            observed_radius = self._region_phase.observed_radius or maximum_radius
            growth_rate = self._region_phase.growth_rate or self.config.region_radius_step
            maximum_lead = max(
                0.0,
                (maximum_radius - observed_radius) / max(0.1, growth_rate),
            )
        elif self._region_phase.phase == "maximum_hold":
            maximum_lead = 0.0
        else:
            until_expansion = self._frames_until_region_expansion()
            maximum_lead = (
                0.0
                if until_expansion is None else
                until_expansion + expansion_duration
            )
        projected_maximum_y = [
            center + maximum_lead * (
                sum(item.vy for item in row) / len(row)
            )
            for center, row in zip(row_y, rows, strict=True)
        ]
        target_index = upper_index
        target_component = component_id(
            rows[target_index],
            rows[target_index + 1] if target_index + 1 < len(rows) else None,
        )
        stable_margin = self.config.region_safe_margin_target
        for candidate_index in range(upper_index, len(rows)):
            candidate_lower = rows[candidate_index]
            candidate_upper = (
                rows[candidate_index + 1]
                if candidate_index + 1 < len(rows) else
                None
            )
            lower_center = projected_maximum_y[candidate_index]
            away_from_bottom = lower_center - maximum_radius > trigger_y
            if candidate_upper is None:
                band_survives_maximum = top - lower_center - maximum_radius > (
                    player_radius + stable_margin
                )
            else:
                band_survives_maximum = (
                    projected_maximum_y[candidate_index + 1] - lower_center
                    - 2.0 * maximum_radius - 2.0 * player_radius
                    >= 2.0 * stable_margin
                )
            candidate_component = component_id(candidate_lower, candidate_upper)
            if (
                self._region_topology.target_component == candidate_component
                and band_survives_maximum
            ):
                target_index = candidate_index
                target_component = candidate_component
                break
            if away_from_bottom and band_survives_maximum:
                target_index = candidate_index
                target_component = candidate_component
                break
            target_index = candidate_index
            target_component = candidate_component
        target_rows_ahead = target_index - upper_index + 1

        clearance = player_radius + self.config.portal_clearance
        guard_frames = float(self.config.decision_interval) + 2.0

        def travel_frames(x: float, y: float) -> float:
            dx = abs(x - px)
            dy = abs(y - py)
            diagonal = min(dx, dy)
            return (
                diagonal / max(0.1, speed * _SQRT_HALF)
                + (max(dx, dy) - diagonal) / max(0.1, speed)
            )

        def path_margin(
            waypoints: Sequence[tuple[float, float]],
        ) -> tuple[float, int]:
            current_x, current_y = px, py
            elapsed = 0
            margin = math.inf
            for waypoint_x, waypoint_y in waypoints:
                dx = abs(waypoint_x - current_x)
                dy = abs(waypoint_y - current_y)
                diagonal = min(dx, dy)
                segment_frames = max(1, int(math.ceil(
                    diagonal / max(0.1, speed * _SQRT_HALF)
                    + (max(dx, dy) - diagonal) / max(0.1, speed)
                )))
                for segment_frame in range(1, segment_frames + 1):
                    elapsed += 1
                    progress = segment_frame / segment_frames
                    path_x = current_x + (waypoint_x - current_x) * progress
                    path_y = current_y + (waypoint_y - current_y) * progress
                    for threat in threats:
                        tx, ty, threat_radius = self._threat_at(threat, elapsed)
                        margin = min(
                            margin,
                            math.hypot(path_x - tx, path_y - ty)
                            - player_radius - threat_radius,
                        )
                current_x, current_y = waypoint_x, waypoint_y
            return margin, elapsed

        initial_vertical_distance = max(
            0.0,
            upper_y + upper_radius + clearance - py,
        )
        upper_vy = sum(item.vy for item in upper) / len(upper)
        initial_portal_frame = min(
            self.config.horizon_frames,
            max(1, int(math.ceil(
                initial_vertical_distance / max(0.1, speed - upper_vy)
            ))),
        )
        projected = sorted(
            (
                (item, *self._threat_at(item, initial_portal_frame))
                for item in upper
            ),
            key=lambda value: value[1],
        )
        portal_candidates: list[dict[str, Any]] = []
        upper_signature = row_signature(upper)

        def add_portal(
            *,
            portal: str,
            portal_x: float,
            target_y: float,
            close_radius: float,
            approach_y: float,
            side: bool,
            currently_open: bool,
            boundary_close_frames: float = math.inf,
        ) -> None:
            if not (left <= portal_x <= right and bottom <= target_y <= top):
                return
            if not currently_open:
                return
            close_frames = (
                self._frames_until_region_radius(close_radius)
                if close_radius > 0.0 else
                0.0
            )
            if close_radius > maximum_radius + 1e-6:
                close_frames = math.inf
            if close_frames is not None:
                close_frames = min(close_frames, boundary_close_frames)
            route = (
                ((portal_x, approach_y), (portal_x, target_y))
                if side else
                ((portal_x, target_y),)
            )
            margin, immediate_travel = path_margin(route)
            target_y_final = (
                row_y[target_index] + maximum_radius + clearance
            )
            remaining_vertical = max(0.0, target_y_final - target_y)
            route_travel = immediate_travel + remaining_vertical / max(0.1, speed)
            deadline_slack = (
                -math.inf
                if close_frames is None else
                close_frames - route_travel - guard_frames
            )
            side_name = (
                "left" if ":side:left" in portal else
                "right" if ":side:right" in portal else
                None
            )
            follows_exterior = target_rows_ahead > 1 and side_name is not None
            portal_candidates.append({
                "portal": portal,
                "target_component": (
                    f"exterior:{side_name}" if follows_exterior else target_component
                ),
                "persistent": follows_exterior,
                "corridor": False,
                "x": portal_x,
                "target_y": target_y,
                "approach_y": approach_y,
                "close_frames": close_frames,
                "deadline_slack": deadline_slack,
                "path_margin": margin,
                "travel": float(immediate_travel),
                "lateral": abs(portal_x - px) / max(0.1, speed),
            })

        for gap_rank, (first, second) in enumerate(zip(projected, projected[1:])):
            first_item, first_x, first_y, first_radius = first
            second_item, second_x, second_y, second_radius = second
            center_spacing = second_x - first_x
            close_radius = 0.5 * (center_spacing - 2.0 * clearance)
            width = center_spacing - first_radius - second_radius
            portal_x = 0.5 * (
                first_x + first_radius + second_x - second_radius
            )
            target_y = (
                0.5 * (first_y + second_y)
                + max(first_radius, second_radius) + clearance
            )
            add_portal(
                portal=f"{upper_signature}:gap:{gap_rank}",
                portal_x=portal_x,
                target_y=target_y,
                close_radius=close_radius,
                approach_y=band_y,
                side=False,
                currently_open=width >= 2.0 * clearance,
            )

        first_item, first_x, first_y, first_radius = projected[0]
        last_item, last_x, last_y, last_radius = projected[-1]
        side_clearance = clearance + stable_margin

        def side_boundary_close_frames(
            item: PredictedThreat,
            side: str,
        ) -> float:
            cycle = self._region_phase.learned_cycle_frames
            forecast = max(
                self.config.horizon_frames,
                int(math.ceil(cycle or self.config.horizon_frames)),
            )
            for future_frame in range(forecast + 1):
                x = item.x + item.vx * future_frame
                _tx, _ty, radius = self._threat_at(item, future_frame)
                portal_x = (
                    x - radius - side_clearance
                    if side == "left" else
                    x + radius + side_clearance
                )
                if (
                    side == "left" and portal_x < left
                    or side == "right" and portal_x > right
                ):
                    return float(future_frame)
            return math.inf

        left_portal = first_x - first_radius - side_clearance
        right_portal = last_x + last_radius + side_clearance
        add_portal(
            portal=f"{upper_signature}:side:left",
            portal_x=left_portal,
            target_y=first_y + first_radius + clearance,
            close_radius=first_x - left - side_clearance,
            approach_y=band_y,
            side=True,
            currently_open=left_portal >= left,
            boundary_close_frames=side_boundary_close_frames(first_item, "left"),
        )
        add_portal(
            portal=f"{upper_signature}:side:right",
            portal_x=right_portal,
            target_y=last_y + last_radius + clearance,
            close_radius=right - last_x - side_clearance,
            approach_y=band_y,
            side=True,
            currently_open=right_portal <= right,
            boundary_close_frames=side_boundary_close_frames(last_item, "right"),
        )

        if target_rows_ahead > 1:
            route_rows = rows[upper_index:target_index + 1]
            left_corridor = min(
                min(
                    item.x + item.vx * maximum_lead
                    for item in row
                ) - maximum_radius - side_clearance
                for row in route_rows
            )
            right_corridor = max(
                max(
                    item.x + item.vx * maximum_lead
                    for item in row
                ) + maximum_radius + side_clearance
                for row in route_rows
            )
            internal_close_radii = [
                0.5 * (
                    second.x - first.x - 2.0 * clearance
                )
                for first, second in zip(upper, upper[1:])
            ]
            internal_close_values = [
                self._frames_until_region_radius(radius)
                for radius in internal_close_radii
                if radius > 0.0
            ]
            internal_close = (
                None
                if any(value is None for value in internal_close_values) else
                max(
                    (float(value) for value in internal_close_values),
                    default=math.inf,
                )
            )
            corridor_deadline = (
                flow_wait
                if internal_close is None and math.isfinite(flow_wait) else
                None
                if internal_close is None else
                min(flow_wait, internal_close)
            )
            target_y_final = min(
                top,
                row_y[target_index] + maximum_radius + clearance,
            )

            def add_corridor(side: str, corridor_x: float) -> None:
                if not left <= corridor_x <= right:
                    return
                aligned = abs(px - corridor_x) <= speed * self.config.decision_interval
                route = ((corridor_x, band_y),)
                margin, alignment_travel = path_margin(route)
                boundary_deadline = min(
                    side_boundary_close_frames(
                        row[0] if side == "left" else row[-1],
                        side,
                    )
                    for row in route_rows
                )
                effective_deadline = (
                    None
                    if corridor_deadline is None else
                    min(corridor_deadline, boundary_deadline)
                )
                portal_candidates.append({
                    "portal": f"corridor:{side}",
                    "target_component": f"exterior:{side}",
                    "persistent": True,
                    "corridor": True,
                    "aligned": aligned,
                    "x": corridor_x,
                    "target_y": target_y_final,
                    "approach_y": band_y,
                    "close_frames": effective_deadline,
                    "deadline_slack": (
                        -math.inf
                        if effective_deadline is None else
                        effective_deadline - alignment_travel - guard_frames
                    ),
                    "path_margin": margin,
                    "travel": float(alignment_travel),
                    "lateral": abs(corridor_x - px) / max(0.1, speed),
                })

            add_corridor("left", left_corridor)
            add_corridor("right", right_corridor)

        phase_candidate: dict[str, Any] | None = None
        if side_forecast is not None:
            forecast_component = f"exterior:{side_forecast.side}"
            existing_forecast_route = next((
                candidate
                for candidate in portal_candidates
                if candidate["target_component"] == forecast_component
                and candidate["path_margin"] >= 0.0
                and candidate["deadline_slack"] >= 0.0
            ), None)
            if existing_forecast_route is not None:
                phase_candidate = existing_forecast_route
            else:
                route = ((side_forecast.x, band_y),)
                margin, phase_travel = path_margin(route)
                phase_deadline = (
                    side_forecast.frames_until_expansion
                    + 0.5 * expansion_duration
                )
                phase_candidate = {
                    "portal": f"phase-flow:{side_forecast.side}",
                    "target_component": forecast_component,
                    "persistent": True,
                    "corridor": True,
                    "aligned": (
                        abs(px - side_forecast.x)
                        <= speed * self.config.decision_interval
                    ),
                    "x": side_forecast.x,
                    "target_y": band_y,
                    "approach_y": band_y,
                    "close_frames": phase_deadline,
                    "deadline_slack": phase_deadline - phase_travel - guard_frames,
                    "path_margin": margin,
                    "travel": float(phase_travel),
                    "lateral": abs(side_forecast.x - px) / max(0.1, speed),
                }
                portal_candidates.append(phase_candidate)

        if not portal_candidates:
            navigation_mode = "evacuate" if flow_wait <= 0.0 else "hold"
            self._region_topology.update(
                target_component=target_component,
                portal=None,
                navigation_mode=navigation_mode,
            )
            return _RegionAnchor(
                x=min(max(px, left), right),
                y=band_y,
                crossing=False,
                path_margin=-math.inf,
                evacuating=navigation_mode != "hold",
                target_rows_ahead=target_rows_ahead,
                navigation_mode=navigation_mode,
                current_component=current_component,
                target_component=target_component,
                portal=None,
                deadline_slack=-math.inf,
            )

        def candidate_key(candidate: Mapping[str, Any]) -> tuple[float, ...]:
            safe = candidate["path_margin"] >= 0.0
            permanently_open = (
                candidate["persistent"]
                and candidate["close_frames"] is not None
                and math.isinf(candidate["close_frames"])
            )
            viable = candidate["deadline_slack"] >= 0.0 and safe
            route_available = viable or (permanently_open and safe)
            retains_component = (
                route_available
                and candidate["target_component"] in {
                    self._region_topology.target_component,
                    current_component
                    if current_component in {"exterior:left", "exterior:right"}
                    else None,
                }
            )
            persistent = (
                target_rows_ahead > 1
                and candidate["persistent"]
                and route_available
            )
            return (
                float(safe),
                float(route_available),
                float(retains_component),
                float(persistent),
                -candidate["travel"],
                candidate["deadline_slack"],
                candidate["path_margin"],
                -candidate["lateral"],
            )

        selected = max(portal_candidates, key=candidate_key)
        retained_target = self._region_topology.target_component

        def retains_exterior_intent(candidate: Mapping[str, Any]) -> bool:
            return (
                retained_target in {"exterior:left", "exterior:right"}
                and candidate["target_component"] == retained_target
                and candidate["persistent"]
                and candidate["close_frames"] is not None
                and (
                    candidate["deadline_slack"] >= 0.0
                    or math.isinf(candidate["close_frames"])
                )
            )

        if retained_target is not None:
            retained = next((
                candidate
                for candidate in portal_candidates
                if candidate["portal"] == self._region_topology.portal
                and candidate["target_component"] == retained_target
                and (
                    candidate["deadline_slack"] >= 0.0
                    or candidate["persistent"]
                    and candidate["close_frames"] is not None
                    and math.isinf(candidate["close_frames"])
                )
                and (
                    candidate["path_margin"] >= 0.0
                    or retains_exterior_intent(candidate)
                )
            ), None)
            if retained is not None:
                selected = retained

        remembered_exterior = self._region_topology.target_component
        remembered_candidates = [
            candidate
            for candidate in portal_candidates
            if candidate["target_component"] == remembered_exterior
            and (
                candidate["deadline_slack"] >= 0.0
                or candidate["persistent"]
                and candidate["close_frames"] is not None
                and math.isinf(candidate["close_frames"])
            )
            and (
                candidate["path_margin"] >= 0.0
                or retains_exterior_intent(candidate)
            )
        ]
        if (
            remembered_exterior in {"exterior:left", "exterior:right"}
            and remembered_candidates
            and not selected["persistent"]
        ):
            # Keep steering toward the remembered component. Collision status
            # is handled by the beam, which may take a nonlinear local detour.
            selected = min(
                remembered_candidates,
                key=lambda candidate: (
                    candidate["travel"],
                    -candidate["path_margin"],
                ),
            )

        if (
            phase_candidate is not None
            and selected["target_component"]
            != phase_candidate["target_component"]
            and (
                phase_candidate["deadline_slack"] >= 0.0
                or self._region_phase.phase == "expanding"
            )
        ):
            # The phase snapshot describes which exterior remains connected
            # after the visible rows advect into the next expansion. A locally
            # blocked straight segment is left to the beam as a short detour.
            selected = phase_candidate

        raw_close_frames = selected["close_frames"]
        close_frames = (
            None if raw_close_frames is None else float(raw_close_frames)
        )
        deadline_slack = float(selected["deadline_slack"])
        selected_target_component = str(selected["target_component"])
        if (
            remembered_exterior in {"exterior:left", "exterior:right"}
            and (
                side_forecast is None
                or remembered_exterior == f"exterior:{side_forecast.side}"
            )
            and any(
                candidate["target_component"] == remembered_exterior
                and (
                    candidate["deadline_slack"] >= 0.0
                    or candidate["persistent"]
                    and candidate["close_frames"] is not None
                    and math.isinf(candidate["close_frames"])
                )
                and (
                    candidate["path_margin"] >= 0.0
                    or retains_exterior_intent(candidate)
                )
                for candidate in portal_candidates
            )
        ):
            # A locally blocked waypoint does not mean that the surrounding
            # time-varying component disappeared. Keep its identity while the
            # beam takes a short collision-avoidance detour.
            selected_target_component = remembered_exterior
        flow_requires_early_crossing = (
            close_frames is not None
            and math.isfinite(close_frames)
            and flow_wait >= close_frames
        )
        topology_urgent = (
            target_rows_ahead > 1
            and self._region_phase.phase in {"expanding", "maximum_hold"}
        )
        side_preposition = (
            selected["persistent"]
            and selected["lateral"] > self.config.decision_interval
        )
        if (
            selected["corridor"] and not selected["aligned"]
            or topology_urgent and side_preposition
            or selected["persistent"] and selected["path_margin"] < 0.0
        ):
            navigation_mode = "preposition"
        elif flow_wait <= 0.0 or deadline_slack <= 0.0:
            navigation_mode = "evacuate"
        elif flow_requires_early_crossing:
            latest_departure = max(0.0, deadline_slack)
            navigation_mode = (
                "preposition"
                if selected["lateral"] >= latest_departure else
                "hold"
            )
        elif current_component == self._region_topology.target_component:
            navigation_mode = "settle"
        else:
            navigation_mode = "hold"

        self._region_topology.update(
            target_component=selected_target_component,
            portal=str(selected["portal"]),
            navigation_mode=navigation_mode,
        )
        crossing_viable = selected["path_margin"] >= 0.0
        crossing = navigation_mode == "evacuate" and crossing_viable
        anchor_y = (
            selected["target_y"]
            if crossing else
            selected["approach_y"]
            if navigation_mode == "preposition" else
            band_y
        )
        return _RegionAnchor(
            x=min(max(float(selected["x"]), left), right),
            y=min(max(float(anchor_y), bottom), top),
            crossing=crossing,
            path_margin=float(selected["path_margin"]),
            evacuating=navigation_mode in {"preposition", "evacuate"},
            target_rows_ahead=target_rows_ahead,
            navigation_mode=navigation_mode,
            current_component=current_component,
            target_component=selected_target_component,
            portal=str(selected["portal"]),
            deadline_slack=deadline_slack,
        )

    def _diverse_keep(
        self,
        order: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        first_action: np.ndarray,
        region_anchor: _RegionAnchor | None,
    ) -> np.ndarray:
        configured_limit = (
            self.config.beam_width
            if region_anchor is None else
            max(self.config.beam_width, self.config.region_beam_width)
        )
        limit = min(configured_limit, len(order))
        if region_anchor is None:
            return np.asarray(order[:limit], dtype=np.int64)

        selected: list[int] = []
        seen: set[tuple[int, int, int]] = set()
        per_action = np.zeros(len(self.actions), dtype=np.int32)
        quota = max(1, limit // len(self.actions))
        scale = self.config.beam_cell_size
        for raw_index in order:
            index = int(raw_index)
            action_index = int(first_action[index])
            cell = (
                action_index,
                int(round(float(x[index]) / scale)),
                int(round(float(y[index]) / scale)),
            )
            if cell in seen or per_action[action_index] >= quota:
                continue
            seen.add(cell)
            per_action[action_index] += 1
            selected.append(index)
            if len(selected) >= limit:
                break
        if len(selected) < limit:
            chosen = set(selected)
            for raw_index in order:
                index = int(raw_index)
                action_index = int(first_action[index])
                cell = (
                    action_index,
                    int(round(float(x[index]) / scale)),
                    int(round(float(y[index]) / scale)),
                )
                if index in chosen or cell in seen:
                    continue
                seen.add(cell)
                selected.append(index)
                if len(selected) >= limit:
                    break
        if len(selected) < limit:
            chosen = set(selected)
            for raw_index in order:
                index = int(raw_index)
                if index in chosen:
                    continue
                selected.append(index)
                if len(selected) >= limit:
                    break
        return np.asarray(selected, dtype=np.int64)

    def _beam_evaluations(
        self,
        player: tuple[float, float, float, float, float],
        bounds: tuple[float, float, float, float],
        threats: Sequence[PredictedThreat],
        boss_x: float | None,
        region_anchor: _RegionAnchor | None = None,
    ) -> tuple[
        tuple[CandidateEvaluation, ...],
        tuple[tuple[Action, ...], ...],
    ]:
        """Search complete three-frame action sequences over the horizon."""

        px, py, player_radius, speed, focus_speed = player
        action_count = len(self.actions)
        move_x = np.asarray([action.move_x for action in self.actions], dtype=np.float64)
        move_y = np.asarray([action.move_y for action in self.actions], dtype=np.float64)
        action_speed = np.asarray([
            focus_speed if action.slow else speed for action in self.actions
        ], dtype=np.float64)
        diagonal = (move_x != 0.0) & (move_y != 0.0)
        action_speed[diagonal] *= _SQRT_HALF
        velocity_x = move_x * action_speed
        velocity_y = move_y * action_speed

        if threats:
            threat_x = np.asarray([value.x for value in threats], dtype=np.float64)
            threat_y = np.asarray([value.y for value in threats], dtype=np.float64)
            threat_vx = np.asarray([value.vx for value in threats], dtype=np.float64)
            threat_vy = np.asarray([value.vy for value in threats], dtype=np.float64)
            threat_radius = np.asarray([value.radius for value in threats], dtype=np.float64)
            threat_radius_rate = np.asarray(
                [value.radius_rate for value in threats], dtype=np.float64,
            )
            threat_rate_horizon = np.asarray(
                [value.radius_rate_horizon for value in threats], dtype=np.float64,
            )
            threat_motion_horizon = np.asarray(
                [value.motion_horizon for value in threats], dtype=np.float64,
            )
            threat_is_region = np.asarray(
                [value.source == "indestructibles" for value in threats],
                dtype=np.bool_,
            )
        else:
            threat_x = threat_y = threat_vx = threat_vy = np.empty(0, dtype=np.float64)
            threat_radius = threat_radius_rate = threat_rate_horizon = np.empty(
                0, dtype=np.float64,
            )
            threat_motion_horizon = np.empty(0, dtype=np.float64)
            threat_is_region = np.empty(0, dtype=np.bool_)

        x = np.asarray([px], dtype=np.float64)
        y = np.asarray([py], dtype=np.float64)
        first_action = np.asarray([-1], dtype=np.int16)
        collision_frames = np.zeros(1, dtype=np.int32)
        earliest_collision = np.full(1, self.config.horizon_frames + 1, dtype=np.int32)
        minimum_margin = np.full(1, math.inf, dtype=np.float64)
        boundary_penalty = np.zeros(1, dtype=np.float64)
        plans = np.empty((1, 0), dtype=np.int8)
        left, right, bottom, top = bounds
        preferred_y = self._preferred_y(bounds)

        for segment_start in range(0, self.config.horizon_frames, self.config.decision_interval):
            parents = np.repeat(np.arange(len(x)), action_count)
            action_indices = np.tile(np.arange(action_count), len(x))
            x = x[parents]
            y = y[parents]
            first_action = first_action[parents]
            first_action[first_action < 0] = action_indices[first_action < 0]
            collision_frames = collision_frames[parents]
            earliest_collision = earliest_collision[parents]
            minimum_margin = minimum_margin[parents]
            boundary_penalty = boundary_penalty[parents]
            plans = plans[parents]
            plans = np.concatenate(
                (plans, action_indices[:, None].astype(np.int8)),
                axis=1,
            )

            frames = min(
                self.config.decision_interval,
                self.config.horizon_frames - segment_start,
            )
            for step in range(1, frames + 1):
                raw_x = x + velocity_x[action_indices]
                raw_y = y + velocity_y[action_indices]
                x = np.clip(raw_x, left, right)
                y = np.clip(raw_y, bottom, top)
                absolute_frame = segment_start + step
                if threats:
                    motion_frame = np.minimum(absolute_frame, threat_motion_horizon)
                    tx = threat_x + threat_vx * motion_frame
                    ty = threat_y + threat_vy * motion_frame
                    radius = np.maximum(
                        0.1,
                        threat_radius
                        + threat_radius_rate
                        * np.minimum(absolute_frame, threat_rate_horizon),
                    )
                    phase_radius = self._region_phase.radius_after(
                        self.config.observation_delay + absolute_frame,
                    )
                    if phase_radius is not None and np.any(threat_is_region):
                        radius[threat_is_region] = np.maximum(
                            radius[threat_is_region],
                            phase_radius,
                        )
                    margins = np.hypot(
                        x[:, None] - tx[None, :],
                        y[:, None] - ty[None, :],
                    ) - player_radius - radius[None, :]
                    frame_margin = margins.min(axis=1)
                    minimum_margin = np.minimum(minimum_margin, frame_margin)
                    collided_now = frame_margin <= 0.0
                    collision_frames += collided_now.astype(np.int32)
                    first_hit = collided_now & (
                        earliest_collision > self.config.horizon_frames
                    )
                    earliest_collision[first_hit] = absolute_frame
                clearance = np.maximum(
                    0.0,
                    np.minimum.reduce((x - left, right - x, y - bottom, top - y)),
                )
                boundary_penalty += self.config.boundary_weight / (1.0 + clearance)
                boundary_penalty += self.config.boundary_weight * (
                    (x != raw_x) | (y != raw_y)
                )
                if region_anchor is not None:
                    boundary_penalty += self.config.region_path_weight * (
                        np.abs(x - region_anchor.x) + np.abs(y - region_anchor.y)
                    )

            collided = collision_frames > 0
            collision_order = -earliest_collision
            if region_anchor is None:
                alignment = (
                    np.zeros_like(x)
                    if boss_x is None else
                    self.config.boss_alignment_weight * np.abs(x - boss_x)
                )
                alignment += self.config.vertical_anchor_weight * np.abs(
                    y - preferred_y,
                )
            else:
                alignment = self.config.region_anchor_weight * (
                    np.abs(x - region_anchor.x) + np.abs(y - region_anchor.y)
                )
            margin_target = (
                self.config.safe_margin_target
                if region_anchor is None else
                self.config.region_safe_margin_target
            )
            margin_shortfall = np.maximum(
                0.0, margin_target - minimum_margin,
            )
            preference = boundary_penalty + alignment
            if region_anchor is None:
                order = np.lexsort((
                    first_action,
                    -minimum_margin,
                    preference,
                    margin_shortfall,
                    collision_frames,
                    collision_order,
                    collided.astype(np.int8),
                ))
            else:
                order = np.lexsort((
                    first_action,
                    -minimum_margin,
                    preference,
                    margin_shortfall,
                    collision_frames,
                    collision_order,
                    collided.astype(np.int8),
                ))
            keep = self._diverse_keep(
                order,
                x,
                y,
                first_action,
                region_anchor,
            )
            x, y = x[keep], y[keep]
            first_action = first_action[keep]
            collision_frames = collision_frames[keep]
            earliest_collision = earliest_collision[keep]
            minimum_margin = minimum_margin[keep]
            boundary_penalty = boundary_penalty[keep]
            plans = plans[keep]

        evaluations: list[CandidateEvaluation] = []
        action_plans: list[tuple[Action, ...]] = []
        for action_index, action in enumerate(self.actions):
            matches = np.flatnonzero(first_action == action_index)
            if not len(matches):
                evaluations.append(self._evaluate(
                    action,
                    player,
                    bounds,
                    threats,
                    boss_x,
                    region_anchor,
                ))
                action_plans.append((action,))
                continue
            if region_anchor is None:
                alignment = (
                    np.zeros(len(matches), dtype=np.float64)
                    if boss_x is None else
                    self.config.boss_alignment_weight * np.abs(x[matches] - boss_x)
                )
                alignment += self.config.vertical_anchor_weight * np.abs(
                    y[matches] - preferred_y,
                )
            else:
                alignment = self.config.region_anchor_weight * (
                    np.abs(x[matches] - region_anchor.x)
                    + np.abs(y[matches] - region_anchor.y)
                )
            collided = collision_frames[matches] > 0
            margin_target = (
                self.config.safe_margin_target
                if region_anchor is None else
                self.config.region_safe_margin_target
            )
            margin_shortfall = np.maximum(
                0.0, margin_target - minimum_margin[matches],
            )
            preference = boundary_penalty[matches] + alignment
            if region_anchor is None:
                order = np.lexsort((
                    -minimum_margin[matches],
                    preference,
                    margin_shortfall,
                    collision_frames[matches],
                    -earliest_collision[matches],
                    collided.astype(np.int8),
                ))
            else:
                order = np.lexsort((
                    -minimum_margin[matches],
                    margin_shortfall,
                    preference,
                    collision_frames[matches],
                    -earliest_collision[matches],
                    collided.astype(np.int8),
                ))
            selected = matches[int(order[0])]
            earliest = int(earliest_collision[selected])
            evaluations.append(CandidateEvaluation(
                action=replace(action, spell=False),
                collided=bool(collision_frames[selected] > 0),
                collision_frames=int(collision_frames[selected]),
                earliest_collision_frame=(
                    None if earliest > self.config.horizon_frames else earliest
                ),
                minimum_margin=float(minimum_margin[selected]),
                boundary_penalty=float(boundary_penalty[selected]),
                boss_alignment=float(alignment[int(order[0])]),
            ))
            action_plans.append(tuple(
                self.actions[int(value)] for value in plans[selected]
            ))
        return tuple(evaluations), tuple(action_plans)

    @staticmethod
    def _path(
        action: Action,
        start: tuple[float, float],
        speed: float,
        focus_speed: float,
        bounds: tuple[float, float, float, float],
        horizon: int,
        hold_frames: int,
    ) -> tuple[tuple[float, float, bool], ...]:
        x, y = start
        magnitude = focus_speed if action.slow else speed
        diagonal = action.move_x != 0 and action.move_y != 0
        scale = magnitude * (_SQRT_HALF if diagonal else 1.0)
        velocity_x = action.move_x * scale
        velocity_y = action.move_y * scale
        left, right, bottom, top = bounds
        result = [(x, y, False)]
        for future_frame in range(1, horizon + 1):
            active = future_frame <= hold_frames
            raw_x = x + (velocity_x if active else 0.0)
            raw_y = y + (velocity_y if active else 0.0)
            x = min(max(raw_x, left), right)
            y = min(max(raw_y, bottom), top)
            result.append((x, y, x != raw_x or y != raw_y))
        return tuple(result)

    def _evaluate(
        self,
        action: Action,
        player: tuple[float, float, float, float, float],
        bounds: tuple[float, float, float, float],
        threats: Sequence[PredictedThreat],
        boss_x: float | None,
        region_anchor: _RegionAnchor | None = None,
    ) -> CandidateEvaluation:
        px, py, player_radius, speed, focus_speed = player
        path = self._path(
            action,
            (px, py),
            speed,
            focus_speed,
            bounds,
            self.config.horizon_frames,
            self.config.decision_interval,
        )
        minimum_margin = math.inf
        collision_frames = 0
        earliest_collision = None
        boundary_penalty = 0.0
        left, right, bottom, top = bounds
        preferred_y = self._preferred_y(bounds)
        for future_frame, (x, y, clamped) in enumerate(path):
            frame_collision = False
            for threat in threats:
                tx, ty, threat_radius = self._threat_at(threat, future_frame)
                margin = math.hypot(x - tx, y - ty) - player_radius - threat_radius
                minimum_margin = min(minimum_margin, margin)
                if margin <= 0.0:
                    frame_collision = True
            if frame_collision:
                collision_frames += 1
                if earliest_collision is None:
                    earliest_collision = future_frame
            clearance = max(0.0, min(x - left, right - x, y - bottom, top - y))
            boundary_penalty += self.config.boundary_weight / (1.0 + clearance)
            if clamped:
                boundary_penalty += self.config.boundary_weight
            if region_anchor is not None:
                boundary_penalty += self.config.region_path_weight * (
                    abs(x - region_anchor.x) + abs(y - region_anchor.y)
                )
        final_x = path[-1][0]
        final_y = path[-1][1]
        if region_anchor is None:
            alignment = (
                0.0 if boss_x is None else
                self.config.boss_alignment_weight * abs(final_x - boss_x)
            )
            alignment += self.config.vertical_anchor_weight * abs(
                final_y - preferred_y,
            )
        else:
            alignment = self.config.region_anchor_weight * (
                abs(final_x - region_anchor.x) + abs(final_y - region_anchor.y)
            )
        return CandidateEvaluation(
            action=replace(action, spell=False),
            collided=collision_frames > 0,
            collision_frames=collision_frames,
            earliest_collision_frame=earliest_collision,
            minimum_margin=minimum_margin,
            boundary_penalty=boundary_penalty,
            boss_alignment=alignment,
        )

    def _compute(
        self,
        observation: Mapping[str, Any],
        threats: tuple[PredictedThreat, ...],
        source_frame: int,
    ) -> MPCDecision:
        player = self._player(observation, self.config.observation_delay)
        bounds = self._bounds(observation, player[2])
        boss_x = self._boss_x(observation, self.config.observation_delay)
        region_anchor = self._region_anchor(player, bounds, threats, source_frame)
        evaluations, action_plans = self._beam_evaluations(
            player,
            bounds,
            threats,
            boss_x,
            region_anchor,
        )
        def key(index: int) -> tuple[float, ...]:
            value = evaluations[index]
            earliest = (
                math.inf
                if value.earliest_collision_frame is None else
                float(value.earliest_collision_frame)
            )
            preference = value.boundary_penalty + value.boss_alignment
            if region_anchor is None:
                return (
                    float(value.collided),
                    -earliest,
                    float(value.collision_frames),
                    max(0.0, self.config.safe_margin_target - value.minimum_margin),
                    preference,
                    -value.minimum_margin,
                    float(index),
                )
            return (
                float(value.collided),
                -earliest,
                float(value.collision_frames),
                max(
                    0.0,
                    self.config.region_safe_margin_target - value.minimum_margin,
                ),
                preference,
                -value.minimum_margin,
                float(index),
            )
        selected_index = min(
            range(len(evaluations)),
            key=key,
        )
        selected_plan = action_plans[selected_index]
        return MPCDecision(
            action=replace(evaluations[selected_index].action, spell=False),
            source_frame=source_frame,
            recomputed=True,
            threats=threats,
            evaluations=evaluations,
            region_anchor=(
                None if region_anchor is None else
                (region_anchor.x, region_anchor.y)
            ),
            region_crossing=(
                region_anchor.crossing if region_anchor is not None else False
            ),
            region_path_margin=(
                region_anchor.path_margin if region_anchor is not None else None
            ),
            region_evacuating=(
                region_anchor.evacuating if region_anchor is not None else False
            ),
            region_target_rows_ahead=(
                region_anchor.target_rows_ahead if region_anchor is not None else 0
            ),
            region_navigation_mode=(
                region_anchor.navigation_mode if region_anchor is not None else "none"
            ),
            region_current_component=(
                region_anchor.current_component if region_anchor is not None else None
            ),
            region_target_component=(
                region_anchor.target_component if region_anchor is not None else None
            ),
            region_portal=(
                region_anchor.portal if region_anchor is not None else None
            ),
            region_deadline_slack=(
                region_anchor.deadline_slack if region_anchor is not None else None
            ),
            planned_actions=selected_plan,
            using_committed_plan=False,
            committed_plan_immediate_margin=None,
            committed_plan_current_horizon_margin=None,
            region_phase=self._region_phase.phase,
            region_phase_started_frame=self._region_phase.phase_started_frame,
            region_learned_cycle_frames=(
                self._region_phase.learned_cycle_frames
            ),
            region_frames_until_expansion=(
                self._frames_until_region_expansion()
            ),
            region_observed_radius=self._region_phase.observed_radius,
        )

    def _immediate_action_margin(
        self,
        action: Action,
        observation: Mapping[str, Any],
        threats: Sequence[PredictedThreat],
    ) -> float:
        player = self._player(observation, self.config.observation_delay)
        bounds = self._bounds(observation, player[2])
        path = self._path(
            action,
            player[:2],
            player[3],
            player[4],
            bounds,
            self.config.decision_interval,
            self.config.decision_interval,
        )
        minimum_margin = math.inf
        for future_frame, (x, y, _clamped) in enumerate(path[1:], start=1):
            for threat in threats:
                tx, ty, radius = self._threat_at(threat, future_frame)
                minimum_margin = min(
                    minimum_margin,
                    math.hypot(x - tx, y - ty) - player[2] - radius,
                )
        return minimum_margin

    def select(self, observation: Mapping[str, Any]) -> MPCDecision:
        observation = _unwrap_observation(observation)
        threats = self.estimator.update(observation)
        source_frame = self.estimator.last_frame
        assert source_frame is not None
        if self._last_source_frame is not None and source_frame < self._last_source_frame:
            self.reset()
            threats = self.estimator.update(observation)
            source_frame = self.estimator.last_frame
            assert source_frame is not None
        self._last_source_frame = source_frame
        self._update_region_phase(observation, source_frame)
        should_recompute = (
            self._decision is None
            or self._last_decision_frame is None
            or source_frame - self._last_decision_frame >= self.config.decision_interval
        )
        if should_recompute:
            proposed = self._compute(observation, threats, source_frame)
            proposed_plan_key = (
                proposed.region_phase,
                proposed.region_phase_started_frame,
                proposed.region_navigation_mode,
                proposed.region_current_component,
                proposed.region_target_component,
                proposed.region_portal,
            )
            committed_action = (
                self._committed_plan[0]
                if (
                    self._committed_plan_is_region
                    and self._committed_plan
                    and self._committed_plan_key == proposed_plan_key
                    and proposed.region_anchor is not None
                ) else
                None
            )
            committed_margin = (
                None
                if committed_action is None else
                self._immediate_action_margin(
                    committed_action,
                    observation,
                    threats,
                )
            )
            committed_evaluation = (
                None
                if committed_action is None else
                next(
                    value
                    for value in proposed.evaluations
                    if value.action.discrete == committed_action.discrete
                )
            )
            if (
                committed_action is not None
                and self._committed_plan_evacuating == proposed.region_evacuating
                and committed_margin is not None
                and committed_margin >= self.config.region_safe_margin_target
                and committed_evaluation is not None
                and (
                    (
                        not committed_evaluation.collided
                        and committed_evaluation.minimum_margin
                        >= self.config.region_safe_margin_target
                    )
                    or (
                        committed_evaluation.collided
                        and committed_evaluation.earliest_collision_frame is not None
                        and committed_evaluation.earliest_collision_frame
                        >= self.config.horizon_frames - self.config.decision_interval
                    )
                )
            ):
                self._decision = replace(
                    proposed,
                    action=replace(committed_action, spell=False),
                    planned_actions=self._committed_plan,
                    using_committed_plan=True,
                    committed_plan_immediate_margin=committed_margin,
                    committed_plan_current_horizon_margin=(
                        committed_evaluation.minimum_margin
                    ),
                )
                self._committed_plan = self._committed_plan[1:]
            else:
                self._decision = proposed
                self._committed_plan_is_region = proposed.region_anchor is not None
                self._committed_plan_evacuating = proposed.region_evacuating
                self._committed_plan_key = (
                    proposed_plan_key if self._committed_plan_is_region else None
                )
                self._committed_plan = (
                    proposed.planned_actions[1:]
                    if self._committed_plan_is_region else ()
                )
            self._last_decision_frame = source_frame
            return self._decision
        assert self._decision is not None
        return replace(
            self._decision,
            source_frame=source_frame,
            recomputed=False,
            threats=threats,
        )

    def observe(self, observation: Mapping[str, Any]) -> int:
        """Update visible tracks and external phase memory without beam search."""

        observation = _unwrap_observation(observation)
        threats = self.estimator.update(observation)
        source_frame = self.estimator.last_frame
        assert source_frame is not None
        if self._last_source_frame is not None and source_frame < self._last_source_frame:
            self.reset()
            threats = self.estimator.update(observation)
            source_frame = self.estimator.last_frame
            assert source_frame is not None
        self._last_source_frame = source_frame
        self._update_region_phase(observation, source_frame)
        player = self._player(observation, self.config.observation_delay)
        bounds = self._bounds(observation, player[2])
        self._region_anchor(player, bounds, threats, source_frame)
        return source_frame

    def select_action(self, observation: Mapping[str, Any]) -> Action:
        return self.select(observation).action


__all__ = [
    "CandidateEvaluation",
    "EngineMPC",
    "MPCConfig",
    "MPCDecision",
    "PredictedThreat",
    "RegionDynamicsMemory",
    "VisibleTrackEstimator",
    "load_region_dynamics_memory",
    "movement_actions",
]
