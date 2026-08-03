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
_LASER_KINDS = frozenset(("straight_laser", "bent_laser"))
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


def _laser_width(record: Mapping[str, Any]) -> float:
    value = _number(record.get("w"), _number(record.get("w0"), 2.0))
    return max(0.5, abs(value or 2.0))


def _segment_circle_records(
    *,
    key: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    start_half_width: float,
    end_half_width: float,
) -> list[dict[str, Any]]:
    """Conservatively cover a tapered segment with trackable circles."""

    length = math.hypot(x2 - x1, y2 - y1)
    if length <= 1e-6:
        return []
    maximum_half_width = max(0.25, start_half_width, end_half_width)
    # The circle radius below uses the exact segment half-step, so wider
    # spacing remains a complete cover. A 16-32 px target avoids turning a
    # screen-length laser into hundreds of planner threats. For an 8 px laser
    # it adds about 12.5 px of conservative side clearance, intentionally
    # keeping the controller away from grazing trajectories.
    spacing = max(16.0, min(32.0, maximum_half_width * 8.0))
    count = max(1, math.ceil(length / spacing))
    half_step = 0.5 * length / count
    result: list[dict[str, Any]] = []
    for index in range(count):
        start_t = index / count
        end_t = (index + 1) / count
        middle_t = 0.5 * (start_t + end_t)
        local_width = max(
            start_half_width + (end_half_width - start_half_width) * start_t,
            start_half_width + (end_half_width - start_half_width) * end_t,
        )
        # The diagonal covers both the strip half-width and the along-segment
        # half step. This never certifies a gap between adjacent samples.
        radius = math.hypot(max(0.0, local_width), half_step)
        sample: dict[str, Any] = {
            "id": f"{key}:{index}",
            "x": x1 + (x2 - x1) * middle_t,
            "y": y1 + (y2 - y1) * middle_t,
            "a": radius,
            "b": radius,
            "collidable": True,
        }
        result.append(sample)
    return result


def _laser_circle_records(
    record: Mapping[str, Any],
    ordinal: int,
) -> list[dict[str, Any]]:
    """Convert visible straight/bent laser geometry to a circle cover."""

    kind = record.get("kind")
    if kind not in _LASER_KINDS:
        return []
    laser_id = record.get("id", ordinal)
    width = _laser_width(record)
    half_width = 0.5 * width
    if kind == "bent_laser":
        points = record.get("points")
        if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
            return []
        valid_points = [
            (
                _number(point.get("x")),
                _number(point.get("y")),
            )
            for point in points
            if isinstance(point, Mapping)
        ]
        valid_points = [
            (float(x), float(y))
            for x, y in valid_points
            if x is not None and y is not None
        ]
        result: list[dict[str, Any]] = []
        for segment, (first, second) in enumerate(zip(valid_points, valid_points[1:])):
            result.extend(_segment_circle_records(
                key=f"{laser_id}:bent:{segment}",
                x1=first[0],
                y1=first[1],
                x2=second[0],
                y2=second[1],
                start_half_width=half_width,
                end_half_width=half_width,
            ))
        return result

    x = _number(record.get("x"))
    y = _number(record.get("y"))
    if x is None or y is None:
        return []
    rotation = math.radians(_number(record.get("rot"), 0.0) or 0.0)
    l1 = max(0.0, _number(record.get("l1"), 0.0) or 0.0)
    l2 = max(0.0, _number(record.get("l2"), 0.0) or 0.0)
    l3 = max(0.0, _number(record.get("l3"), 0.0) or 0.0)
    total = l1 + l2 + l3
    if total <= 1e-6:
        total = max(0.0, abs(_number(record.get("l"), 0.0) or 0.0))
        l1, l2, l3 = 0.0, total, 0.0
    if total <= 1e-6:
        return []

    direction_x = math.cos(rotation)
    direction_y = math.sin(rotation)
    sections = (
        (0.0, l1, 0.0 if l1 > 0.0 else half_width, half_width),
        (l1, l1 + l2, half_width, half_width),
        (
            l1 + l2,
            total,
            half_width,
            0.0 if l3 > 0.0 else half_width,
        ),
    )
    result = []
    for section, (start, stop, start_width, stop_width) in enumerate(sections):
        if stop - start <= 1e-6:
            continue
        result.extend(_segment_circle_records(
            key=f"{laser_id}:straight:{section}",
            x1=x + direction_x * start,
            y1=y + direction_y * start,
            x2=x + direction_x * stop,
            y2=y + direction_y * stop,
            start_half_width=start_width,
            end_half_width=stop_width,
        ))
    return result


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
    danger_margin_target: float = 16.0
    safe_margin_target: float = 20.0
    clearance_reward_cap: float = 48.0
    clearance_reward_weight: float = 0.35
    corner_reserve_target: float = 48.0
    corner_reserve_weight: float = 0.25
    minimum_direction_hold_frames: int = 12
    collision_priority_frames: int = 36
    emergency_collision_frames: int = 12
    emergency_margin: float = 4.0
    switch_margin_gain: float = 8.0
    direction_switch_penalty: float = 3.0
    direction_reverse_penalty: float = 9.0
    direction_aba_penalty: float = 6.0
    speed_switch_penalty: float = 0.75
    nonbullet_motion_horizon: int = 9
    preferred_y_fraction: float = 2.0 / 25.0
    vertical_anchor_weight: float = 0.25
    beam_cell_size: float = 4.0
    region_anchor_weight: float = 2.0
    region_boundary_trigger_margin: float = 72.0
    region_safe_margin_target: float = 8.0
    portal_clearance: float = 6.0
    region_path_weight: float = 0.05
    region_learned_min_radius: float = 7.0
    region_learned_max_radius: float = 28.0
    region_radius_step: float = 0.7
    region_dynamics_memory: RegionDynamicsMemory | None = None
    track_displacement_tolerance: float = 1.0
    gap_prediction_enabled: bool = True
    gap_direction_tolerance_degrees: float = 12.0
    gap_speed_relative_tolerance: float = 0.20
    gap_speed_absolute_tolerance: float = 0.35
    gap_minimum_speed: float = 0.75
    gap_minimum_group_size: int = 3
    gap_wavefront_depth: float = 24.0
    gap_maximum_lateral_spacing: float = 112.0
    gap_safety_margin: float = 10.0
    gap_minimum_usable_width: float = 4.0
    gap_sample_interval: int = 6
    gap_minimum_lifetime_frames: int = 12
    gap_entry_guard_frames: int = 6
    gap_hold_frames: int = 6
    gap_path_minimum_margin: float = 4.0
    gap_group_coverage_fraction: float = 0.45
    gap_anchor_weight: float = 1.5
    gap_entry_candidate_limit: int = 8
    gap_detour_beam_width: int = 48

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
        margin_targets = (
            self.danger_margin_target,
            self.safe_margin_target,
            self.corner_reserve_target,
        )
        if not all(
            math.isfinite(value) and value >= 0.0 for value in margin_targets
        ):
            raise ValueError("safety margin targets must be finite and nonnegative")
        if self.danger_margin_target > self.safe_margin_target:
            raise ValueError("danger_margin_target cannot exceed safe_margin_target")
        if self.minimum_direction_hold_frames < 0:
            raise ValueError("minimum_direction_hold_frames cannot be negative")
        if self.minimum_direction_hold_frames % self.decision_interval != 0:
            raise ValueError(
                "minimum_direction_hold_frames must align to the decision interval"
            )
        if not 1 <= self.collision_priority_frames <= self.horizon_frames:
            raise ValueError("collision_priority_frames must be within the horizon")
        if not 0 <= self.emergency_collision_frames <= self.horizon_frames:
            raise ValueError("emergency_collision_frames must be within the horizon")
        temporal_values = (
            self.clearance_reward_cap,
            self.switch_margin_gain,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in temporal_values):
            raise ValueError("clearance and hysteresis thresholds must be finite and positive")
        nonnegative_temporal_values = (
            self.clearance_reward_weight,
            self.corner_reserve_weight,
            self.emergency_margin,
            self.direction_switch_penalty,
            self.direction_reverse_penalty,
            self.direction_aba_penalty,
            self.speed_switch_penalty,
        )
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in nonnegative_temporal_values
        ):
            raise ValueError("movement preference values must be finite and nonnegative")
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
        if not isinstance(self.gap_prediction_enabled, bool):
            raise ValueError("gap_prediction_enabled must be boolean")
        gap_positive_values = (
            self.gap_direction_tolerance_degrees,
            self.gap_speed_absolute_tolerance,
            self.gap_minimum_speed,
            self.gap_wavefront_depth,
            self.gap_maximum_lateral_spacing,
            self.gap_minimum_usable_width,
        )
        if not all(
            math.isfinite(value) and value > 0.0 for value in gap_positive_values
        ):
            raise ValueError("gap geometry values must be finite and positive")
        if not 0.0 < self.gap_direction_tolerance_degrees < 90.0:
            raise ValueError("gap direction tolerance must be below 90 degrees")
        if (
            not math.isfinite(self.gap_speed_relative_tolerance)
            or not 0.0 <= self.gap_speed_relative_tolerance < 1.0
        ):
            raise ValueError("gap relative speed tolerance must be in [0, 1)")
        if self.gap_minimum_group_size < 2:
            raise ValueError("gap groups require at least two bullets")
        gap_frame_values = (
            self.gap_sample_interval,
            self.gap_minimum_lifetime_frames,
            self.gap_entry_guard_frames,
            self.gap_hold_frames,
            self.gap_entry_candidate_limit,
            self.gap_detour_beam_width,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in gap_frame_values
        ):
            raise ValueError("gap frame and beam values must be positive integers")
        gap_nonnegative_values = (
            self.gap_safety_margin,
            self.gap_path_minimum_margin,
            self.gap_anchor_weight,
        )
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in gap_nonnegative_values
        ):
            raise ValueError("gap safety and preference values must be nonnegative")
        if self.gap_path_minimum_margin < self.emergency_margin:
            raise ValueError("gap path margin cannot be below the emergency margin")
        if (
            not math.isfinite(self.gap_group_coverage_fraction)
            or not 0.0 <= self.gap_group_coverage_fraction <= 1.0
        ):
            raise ValueError("gap group coverage fraction must be in [0, 1]")
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
    motion_penalty: float = 0.0
    minimum_nonregion_margin: float = math.inf
    minimum_region_margin: float = math.inf
    immediate_corner_clearance: float = math.inf

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
            self.motion_penalty,
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
    gap_bullet_group_count: int = 0
    gap_corridor_count: int = 0
    gap_selected_center: tuple[float, float] | None = None
    gap_selected_width: float | None = None
    gap_selected_lifetime_frames: int | None = None
    gap_navigation_mode: str = "inactive"


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
class _GapBulletGroup:
    key: str
    members: tuple[PredictedThreat, ...]
    direction_x: float
    direction_y: float
    speed: float
    coverage_fraction: float


@dataclass(frozen=True, slots=True)
class _GapCorridor:
    key: str
    group_key: str
    center_x: float
    center_y: float
    usable_width: float
    lifetime_frames: int
    arrival_frames: float
    path_margin: float
    normal_x: float
    normal_y: float
    member_count: int
    entry_plan: tuple[Action, ...] = ()

    @property
    def center(self) -> tuple[float, float]:
        return self.center_x, self.center_y

    @property
    def entry_action(self) -> Action | None:
        return self.entry_plan[0] if self.entry_plan else None


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

        source_records: list[tuple[str, Sequence[Any]]] = []
        fallback_lasers: list[Mapping[str, Any]] = []
        for source in _OBJECT_ARRAYS:
            records = observation.get(source)
            if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
                continue
            ordinary: list[Any] = []
            for record in records:
                if isinstance(record, Mapping) and record.get("kind") in _LASER_KINDS:
                    fallback_lasers.append(record)
                else:
                    ordinary.append(record)
            source_records.append((source, ordinary))

        laser_records = observation.get("lasers")
        if not isinstance(laser_records, Sequence) or isinstance(laser_records, (str, bytes)):
            laser_records = fallback_lasers
        sampled_lasers: list[dict[str, Any]] = []
        laser_identities: set[tuple[Any, Any]] = set()
        for ordinal, record in enumerate(laser_records):
            if not isinstance(record, Mapping):
                continue
            identity = (record.get("kind"), record.get("id", ordinal))
            if identity in laser_identities:
                continue
            laser_identities.add(identity)
            sampled_lasers.extend(_laser_circle_records(record, ordinal))
        source_records.append(("lasers", sampled_lasers))

        for source, records in source_records:
            for ordinal, record in enumerate(records):
                if not isinstance(record, Mapping):
                    continue
                # Enemy bullets and laser warnings may become collidable inside
                # the observation-delay window. Humans can already see them,
                # and dropping them here makes the teacher less informed than
                # the semantic policy input.
                if (
                    record.get("collidable", True) is not True
                    and source not in {"enemy_bullets", "lasers"}
                ):
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
                    and source != "lasers"
                    and math.hypot(
                        estimated_vx - visible_vx,
                        estimated_vy - visible_vy,
                    ) > self.config.track_displacement_tolerance
                ):
                    continuous = False
                    elapsed = 0
                # Once two visible samples exist, derive motion from them. Laser
                # sample displacement includes rotation and length changes that
                # the parent object's dx/dy cannot represent. Other per-frame
                # displacement still rejects immediately reused IDs.
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
                    if source in {"enemies", "nontjt_enemies", "lasers"} else
                    self.config.horizon_frames
                )
                predicted_radius_rate = radius_rate
                radius_rate_horizon = self.config.radius_rate_horizon
                predicted_radius = radius
                if (
                    source == "indestructibles"
                    and self.config.region_dynamics_memory is not None
                ):
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
        dynamics_memory = self.config.region_dynamics_memory
        self._region_phase = _RegionPhaseMemory(
            minimum_hint=(
                self.config.region_learned_min_radius
                if dynamics_memory is not None else None
            ),
            maximum_hint=(
                self.config.region_learned_max_radius
                if dynamics_memory is not None else None
            ),
            rate_hint=(
                self.config.region_radius_step
                if dynamics_memory is not None else None
            ),
            dynamics_memory=dynamics_memory,
        )
        self._region_topology = _RegionTopologyMemory()
        self._committed_plan: tuple[Action, ...] = ()
        self._committed_plan_is_region = False
        self._committed_plan_evacuating = False
        self._committed_plan_key: tuple[Any, ...] | None = None
        self._last_action: Action | None = None
        self._previous_action: Action | None = None
        self._direction_started_frame: int | None = None
        self._active_gap_key: str | None = None

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

    @staticmethod
    def _corner_clearance(
        x: float,
        y: float,
        bounds: tuple[float, float, float, float],
    ) -> float:
        left, right, bottom, top = bounds
        horizontal = max(0.0, min(x - left, right - x))
        vertical = max(0.0, min(y - bottom, top - y))
        return math.hypot(horizontal, vertical)

    def _corner_shortfall(self, clearance: float) -> float:
        return self.config.corner_reserve_weight * max(
            0.0,
            self.config.corner_reserve_target - clearance,
        )

    @staticmethod
    def _direction(action: Action | None) -> tuple[int, int] | None:
        if action is None:
            return None
        return action.move_x, action.move_y

    def _transition_penalty(
        self,
        action: Action,
        previous: Action | None,
        two_ago: Action | None,
    ) -> float:
        if previous is None:
            return 0.0
        direction = self._direction(action)
        previous_direction = self._direction(previous)
        penalty = 0.0
        if direction != previous_direction:
            penalty += self.config.direction_switch_penalty
            if (
                direction != (0, 0)
                and previous_direction != (0, 0)
                and direction == (-previous.move_x, -previous.move_y)
            ):
                penalty += self.config.direction_reverse_penalty
            if (
                two_ago is not None
                and direction == self._direction(two_ago)
                and direction != previous_direction
            ):
                penalty += self.config.direction_aba_penalty
        elif action.slow != previous.slow:
            penalty += self.config.speed_switch_penalty
        return penalty

    def _clearance_reward(
        self,
        margin: float,
        *,
        region: bool = False,
        nonregion_margin: float = math.inf,
        region_margin: float = math.inf,
    ) -> float:
        if math.isnan(margin):
            return 0.0
        if not region:
            clearance = min(max(0.0, margin), self.config.clearance_reward_cap)
            return self.config.clearance_reward_weight * clearance

        # A narrow forced-movement portal must not cap the reward for staying
        # away from ordinary bullets. Preserve a separate reserve for the
        # indestructible region geometry and reward both independently.
        ordinary_clearance = min(
            max(0.0, nonregion_margin),
            self.config.clearance_reward_cap,
        )
        forced_region_clearance = min(
            max(0.0, region_margin),
            max(self.config.region_safe_margin_target, self.config.portal_clearance),
        )
        clearance = ordinary_clearance + forced_region_clearance
        return self.config.clearance_reward_weight * clearance

    def _remember_action(self, action: Action, source_frame: int) -> None:
        direction = self._direction(action)
        if self._last_action is None:
            self._direction_started_frame = source_frame
        elif direction != self._direction(self._last_action):
            self._direction_started_frame = source_frame
        self._previous_action = self._last_action
        self._last_action = replace(action, spell=False)

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
            and record.get("kind") not in _LASER_KINDS
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

    def _gap_bullet_groups(
        self,
        bounds: tuple[float, float, float, float],
        threats: Sequence[PredictedThreat],
    ) -> tuple[_GapBulletGroup, ...]:
        """Cluster visible bullets by velocity and moving wavefront."""

        if not self.config.gap_prediction_enabled:
            return ()
        eligible = [
            value
            for value in threats
            if value.source == "enemy_bullets"
            and math.hypot(value.vx, value.vy) >= self.config.gap_minimum_speed
        ]
        if len(eligible) < self.config.gap_minimum_group_size:
            return ()

        direction_cosine = math.cos(math.radians(
            self.config.gap_direction_tolerance_degrees,
        ))
        velocity_clusters: list[dict[str, Any]] = []
        for value in sorted(
            eligible,
            key=lambda item: (
                math.atan2(item.vy, item.vx),
                math.hypot(item.vx, item.vy),
                item.key,
            ),
        ):
            speed = math.hypot(value.vx, value.vy)
            unit_x, unit_y = value.vx / speed, value.vy / speed
            selected: dict[str, Any] | None = None
            for cluster in velocity_clusters:
                mean_speed = float(cluster["speed_sum"]) / len(cluster["members"])
                speed_tolerance = max(
                    self.config.gap_speed_absolute_tolerance,
                    self.config.gap_speed_relative_tolerance
                    * max(speed, mean_speed),
                )
                if (
                    unit_x * float(cluster["direction_x"])
                    + unit_y * float(cluster["direction_y"])
                    >= direction_cosine
                    and abs(speed - mean_speed) <= speed_tolerance
                ):
                    selected = cluster
                    break
            if selected is None:
                velocity_clusters.append({
                    "members": [value],
                    "velocity_x": value.vx,
                    "velocity_y": value.vy,
                    "speed_sum": speed,
                    "direction_x": unit_x,
                    "direction_y": unit_y,
                })
                continue
            selected["members"].append(value)
            selected["velocity_x"] += value.vx
            selected["velocity_y"] += value.vy
            selected["speed_sum"] += speed
            magnitude = math.hypot(
                float(selected["velocity_x"]),
                float(selected["velocity_y"]),
            )
            if magnitude > 1e-9:
                selected["direction_x"] = float(selected["velocity_x"]) / magnitude
                selected["direction_y"] = float(selected["velocity_y"]) / magnitude

        left, right, bottom, top = bounds
        groups: list[_GapBulletGroup] = []
        for cluster in velocity_clusters:
            members = list(cluster["members"])
            if len(members) < self.config.gap_minimum_group_size:
                continue
            direction_x = float(cluster["direction_x"])
            direction_y = float(cluster["direction_y"])
            normal_x, normal_y = -direction_y, direction_x
            projected = sorted(
                (
                    value.x * direction_x + value.y * direction_y,
                    value.key,
                    value,
                )
                for value in members
            )
            wavefronts: list[list[PredictedThreat]] = []
            wavefront_centers: list[float] = []
            for longitudinal, _, value in projected:
                if (
                    not wavefronts
                    or abs(longitudinal - wavefront_centers[-1])
                    > self.config.gap_wavefront_depth
                ):
                    wavefronts.append([value])
                    wavefront_centers.append(longitudinal)
                else:
                    wavefronts[-1].append(value)
                    wavefront_centers[-1] = sum(
                        item.x * direction_x + item.y * direction_y
                        for item in wavefronts[-1]
                    ) / len(wavefronts[-1])

            bound_projection = [
                x * normal_x + y * normal_y
                for x, y in (
                    (left, bottom),
                    (left, top),
                    (right, bottom),
                    (right, top),
                )
            ]
            cross_span = max(bound_projection) - min(bound_projection)
            for wavefront in wavefronts:
                if len(wavefront) < self.config.gap_minimum_group_size:
                    continue
                ordered = tuple(sorted(
                    wavefront,
                    key=lambda item: (
                        item.x * normal_x + item.y * normal_y,
                        item.key,
                    ),
                ))
                lower = min(
                    item.x * normal_x + item.y * normal_y - item.radius
                    for item in ordered
                )
                upper = max(
                    item.x * normal_x + item.y * normal_y + item.radius
                    for item in ordered
                )
                member_keys = ",".join(sorted(item.key for item in ordered))
                groups.append(_GapBulletGroup(
                    key=f"wavefront:{member_keys}",
                    members=ordered,
                    direction_x=direction_x,
                    direction_y=direction_y,
                    speed=sum(
                        item.vx * direction_x + item.vy * direction_y
                        for item in ordered
                    ) / len(ordered),
                    coverage_fraction=(
                        0.0 if cross_span <= 1e-9 else
                        min(1.0, max(0.0, (upper - lower) / cross_span))
                    ),
                ))
        return tuple(sorted(groups, key=lambda value: value.key))

    def _gap_entry_path(
        self,
        player: tuple[float, float, float, float, float],
        bounds: tuple[float, float, float, float],
        target: tuple[float, float],
        normal: tuple[float, float],
        usable_width: float,
        arrival_frames: float,
        end_frame: int,
    ) -> tuple[np.ndarray, np.ndarray, float, bool, tuple[Action, ...]]:
        """Build a legal three-frame-block route into a center-space corridor."""

        px, py, _player_radius, speed, focus_speed = player
        normal_x, normal_y = normal
        center_u = target[0] * normal_x + target[1] * normal_y
        half_width = 0.5 * max(0.0, usable_width)
        lower_u, upper_u = center_u - half_width, center_u + half_width
        entry_deadline = max(0.0, arrival_frames - self.config.gap_entry_guard_frames)
        deadline_frame = max(0, int(math.floor(entry_deadline + 1e-9)))
        left, right, bottom, top = bounds

        def outside_distance(x: float, y: float) -> float:
            value = x * normal_x + y * normal_y
            if value < lower_u:
                return lower_u - value
            if value > upper_u:
                return value - upper_u
            return 0.0

        path_x = np.empty(end_frame, dtype=np.float64)
        path_y = np.empty(end_frame, dtype=np.float64)
        current_x, current_y = px, py
        travel_frames = 0.0 if outside_distance(px, py) <= 1e-6 else math.inf
        plan: list[Action] = []
        for block_start in range(0, end_frame, self.config.decision_interval):
            block_length = min(
                self.config.decision_interval,
                end_frame - block_start,
            )
            best: tuple[
                tuple[float, float, float, int],
                tuple[tuple[float, float], ...],
                Action,
            ] | None = None
            for action_index, action in enumerate(self.actions):
                magnitude = focus_speed if action.slow else speed
                diagonal = action.move_x != 0 and action.move_y != 0
                scale = magnitude * (_SQRT_HALF if diagonal else 1.0)
                velocity_x = action.move_x * scale
                velocity_y = action.move_y * scale
                candidate: list[tuple[float, float]] = []
                x, y = current_x, current_y
                for _ in range(block_length):
                    x = min(max(x + velocity_x, left), right)
                    y = min(max(y + velocity_y, bottom), top)
                    candidate.append((x, y))
                endpoint_u = x * normal_x + y * normal_y
                movement = math.hypot(x - current_x, y - current_y)
                score = (
                    outside_distance(x, y),
                    movement,
                    abs(endpoint_u - center_u),
                    action_index,
                )
                if best is None or score < best[0]:
                    best = score, tuple(candidate), action
            assert best is not None
            plan.append(best[2])
            for offset, (current_x, current_y) in enumerate(best[1]):
                frame_index = block_start + offset
                path_x[frame_index] = current_x
                path_y[frame_index] = current_y
                if (
                    not math.isfinite(travel_frames)
                    and outside_distance(current_x, current_y) <= 1e-6
                ):
                    travel_frames = float(frame_index + 1)

        settled = outside_distance(px, py) <= 1e-6 if deadline_frame == 0 else True
        if deadline_frame > 0:
            settled = all(
                outside_distance(float(path_x[index]), float(path_y[index])) <= 1e-6
                for index in range(deadline_frame - 1, end_frame)
            )
        elif settled:
            settled = all(
                outside_distance(float(path_x[index]), float(path_y[index])) <= 1e-6
                for index in range(end_frame)
            )
        return path_x, path_y, travel_frames, settled, tuple(plan)

    def _gap_detour_entry_path(
        self,
        player: tuple[float, float, float, float, float],
        bounds: tuple[float, float, float, float],
        target: tuple[float, float],
        normal: tuple[float, float],
        usable_width: float,
        arrival_frames: float,
        end_frame: int,
        threat_forecast: tuple[np.ndarray, np.ndarray, np.ndarray],
        minimum_margin: float,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        float,
        float,
        bool,
        tuple[Action, ...],
    ]:
        """Search a small diverse beam when the direct gap route is obstructed."""

        px, py, player_radius, speed, focus_speed = player
        normal_x, normal_y = normal
        center_u = target[0] * normal_x + target[1] * normal_y
        half_width = 0.5 * max(0.0, usable_width)
        lower_u, upper_u = center_u - half_width, center_u + half_width
        entry_deadline = max(0.0, arrival_frames - self.config.gap_entry_guard_frames)
        deadline_frame = max(0, int(math.floor(entry_deadline + 1e-9)))
        initial_u = px * normal_x + py * normal_y
        initially_inside = lower_u - 1e-6 <= initial_u <= upper_u + 1e-6
        if deadline_frame == 0 and not initially_inside:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty, -math.inf, math.inf, False, ()

        action_count = len(self.actions)
        action_speed = np.asarray([
            focus_speed if action.slow else speed for action in self.actions
        ], dtype=np.float64)
        diagonal = np.asarray([
            action.move_x != 0 and action.move_y != 0
            for action in self.actions
        ], dtype=np.bool_)
        action_speed[diagonal] *= _SQRT_HALF
        velocity_x = action_speed * np.asarray([
            action.move_x for action in self.actions
        ], dtype=np.float64)
        velocity_y = action_speed * np.asarray([
            action.move_y for action in self.actions
        ], dtype=np.float64)
        left, right, bottom, top = bounds
        future_x, future_y, future_radius = threat_forecast

        state_x = np.asarray([px], dtype=np.float64)
        state_y = np.asarray([py], dtype=np.float64)
        state_margin = np.asarray([math.inf], dtype=np.float64)
        state_travel = np.asarray([
            0.0 if initially_inside else math.inf
        ], dtype=np.float64)
        state_plans = np.empty((1, 0), dtype=np.int16)
        state_path_x = np.empty((1, 0), dtype=np.float64)
        state_path_y = np.empty((1, 0), dtype=np.float64)

        for block_start in range(0, end_frame, self.config.decision_interval):
            block_length = min(
                self.config.decision_interval,
                end_frame - block_start,
            )
            parent = np.repeat(np.arange(len(state_x), dtype=np.int64), action_count)
            action_index = np.tile(
                np.arange(action_count, dtype=np.int64),
                len(state_x),
            )
            current_x = state_x[parent].copy()
            current_y = state_y[parent].copy()
            expanded_margin = state_margin[parent].copy()
            expanded_travel = state_travel[parent].copy()
            block_path_x = np.empty((len(parent), block_length), dtype=np.float64)
            block_path_y = np.empty((len(parent), block_length), dtype=np.float64)
            valid = np.ones(len(parent), dtype=np.bool_)
            outside = np.full(len(parent), math.inf, dtype=np.float64)
            for offset in range(block_length):
                current_x = np.clip(
                    current_x + velocity_x[action_index],
                    left,
                    right,
                )
                current_y = np.clip(
                    current_y + velocity_y[action_index],
                    bottom,
                    top,
                )
                block_path_x[:, offset] = current_x
                block_path_y[:, offset] = current_y
                future_index = block_start + offset
                if future_x.shape[1]:
                    frame_margin = np.min(
                        np.hypot(
                            current_x[:, None] - future_x[future_index][None, :],
                            current_y[:, None] - future_y[future_index][None, :],
                        )
                        - player_radius
                        - future_radius[future_index][None, :],
                        axis=1,
                    )
                    expanded_margin = np.minimum(expanded_margin, frame_margin)
                projected = current_x * normal_x + current_y * normal_y
                outside = np.maximum(
                    np.maximum(lower_u - projected, projected - upper_u),
                    0.0,
                )
                reached = np.isinf(expanded_travel) & (outside <= 1e-6)
                expanded_travel[reached] = float(future_index + 1)
                absolute_frame = future_index + 1
                if deadline_frame == 0 or absolute_frame >= deadline_frame:
                    valid &= outside <= 1e-6

            valid &= expanded_margin >= minimum_margin
            remaining_frames = max(
                0,
                deadline_frame - (block_start + block_length),
            )
            if remaining_frames:
                valid &= outside <= speed * remaining_frames + 1e-6
            keepable = np.flatnonzero(valid)
            if not len(keepable):
                empty = np.empty(0, dtype=np.float64)
                return empty, empty, -math.inf, math.inf, False, ()

            expanded_plans = np.concatenate((
                state_plans[parent],
                action_index[:, None].astype(np.int16),
            ), axis=1)
            expanded_path_x = np.concatenate((
                state_path_x[parent],
                block_path_x,
            ), axis=1)
            expanded_path_y = np.concatenate((
                state_path_y[parent],
                block_path_y,
            ), axis=1)
            endpoint_distance = np.hypot(
                current_x - target[0],
                current_y - target[1],
            )
            order = keepable[np.lexsort((
                action_index[keepable],
                endpoint_distance[keepable],
                -expanded_margin[keepable],
                outside[keepable],
            ))]

            selected: list[int] = []
            seen: set[tuple[int, int, int]] = set()
            per_first_action = np.zeros(action_count, dtype=np.int32)
            quota = max(1, self.config.gap_detour_beam_width // action_count)
            for raw_index in order:
                index = int(raw_index)
                first_action = int(expanded_plans[index, 0])
                cell = (
                    first_action,
                    int(round(float(current_x[index]) / self.config.beam_cell_size)),
                    int(round(float(current_y[index]) / self.config.beam_cell_size)),
                )
                if cell in seen or per_first_action[first_action] >= quota:
                    continue
                seen.add(cell)
                per_first_action[first_action] += 1
                selected.append(index)
                if len(selected) >= self.config.gap_detour_beam_width:
                    break
            if len(selected) < self.config.gap_detour_beam_width:
                chosen = set(selected)
                for raw_index in order:
                    index = int(raw_index)
                    first_action = int(expanded_plans[index, 0])
                    cell = (
                        first_action,
                        int(round(float(current_x[index]) / self.config.beam_cell_size)),
                        int(round(float(current_y[index]) / self.config.beam_cell_size)),
                    )
                    if index in chosen or cell in seen:
                        continue
                    seen.add(cell)
                    selected.append(index)
                    if len(selected) >= self.config.gap_detour_beam_width:
                        break
            selected_array = np.asarray(selected, dtype=np.int64)
            state_x = current_x[selected_array]
            state_y = current_y[selected_array]
            state_margin = expanded_margin[selected_array]
            state_travel = expanded_travel[selected_array]
            state_plans = expanded_plans[selected_array]
            state_path_x = expanded_path_x[selected_array]
            state_path_y = expanded_path_y[selected_array]

        final_distance = np.hypot(state_x - target[0], state_y - target[1])
        selected = int(np.lexsort((final_distance, -state_margin))[0])
        plan = tuple(
            self.actions[int(value)] for value in state_plans[selected]
        )
        return (
            state_path_x[selected],
            state_path_y[selected],
            float(state_margin[selected]),
            float(state_travel[selected]),
            True,
            plan,
        )

    def _gap_path_margin(
        self,
        player: tuple[float, float, float, float, float],
        bounds: tuple[float, float, float, float],
        target: tuple[float, float],
        normal: tuple[float, float],
        usable_width: float,
        arrival_frames: float,
        threats: Sequence[PredictedThreat],
        threat_forecast: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        *,
        minimum_margin: float | None = None,
    ) -> tuple[float, float, tuple[Action, ...]]:
        end_frame = min(
            self.config.horizon_frames,
            max(
                self.config.gap_minimum_lifetime_frames,
                int(math.ceil(max(0.0, arrival_frames)))
                + self.config.gap_hold_frames,
            ),
        )
        path_x, path_y, travel_frames, settled, plan = self._gap_entry_path(
            player,
            bounds,
            target,
            normal,
            usable_width,
            arrival_frames,
            end_frame,
        )
        if not settled:
            return -math.inf, travel_frames, ()
        if not threats:
            return math.inf, travel_frames, plan
        if threat_forecast is None:
            threat_forecast = self._gap_threat_forecast(threats)
        player_radius = player[2]
        future_threat_x, future_threat_y, future_threat_radius = threat_forecast
        margins = np.hypot(
            path_x[:, None] - future_threat_x[:end_frame],
            path_y[:, None] - future_threat_y[:end_frame],
        ) - player_radius - future_threat_radius[:end_frame]
        path_margin = float(np.min(margins))
        required_margin = (
            self.config.gap_path_minimum_margin
            if minimum_margin is None else
            minimum_margin
        )
        if path_margin >= required_margin:
            return path_margin, travel_frames, plan
        (
            _detour_x,
            _detour_y,
            detour_margin,
            detour_travel,
            detour_settled,
            detour_plan,
        ) = self._gap_detour_entry_path(
            player,
            bounds,
            target,
            normal,
            usable_width,
            arrival_frames,
            end_frame,
            threat_forecast,
            required_margin,
        )
        if detour_settled:
            return detour_margin, detour_travel, detour_plan
        return path_margin, travel_frames, plan

    def _gap_threat_forecast(
        self,
        threats: Sequence[PredictedThreat],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        horizon = self.config.horizon_frames
        if not threats:
            empty = np.empty((horizon, 0), dtype=np.float64)
            return empty, empty.copy(), empty.copy()
        frames = np.arange(1, horizon + 1, dtype=np.float64)[:, None]
        motion_horizon = np.asarray([
            value.motion_horizon for value in threats
        ], dtype=np.float64)[None, :]
        radius_horizon = np.asarray([
            value.radius_rate_horizon for value in threats
        ], dtype=np.float64)[None, :]
        motion_frames = np.minimum(frames, motion_horizon)
        radius_frames = np.minimum(frames, radius_horizon)
        future_x = (
            np.asarray([value.x for value in threats], dtype=np.float64)[None, :]
            + motion_frames
            * np.asarray([value.vx for value in threats], dtype=np.float64)[None, :]
        )
        future_y = (
            np.asarray([value.y for value in threats], dtype=np.float64)[None, :]
            + motion_frames
            * np.asarray([value.vy for value in threats], dtype=np.float64)[None, :]
        )
        future_radius = np.maximum(
            0.1,
            np.asarray([
                value.radius for value in threats
            ], dtype=np.float64)[None, :]
            + radius_frames
            * np.asarray([
                value.radius_rate for value in threats
            ], dtype=np.float64)[None, :],
        )
        region_indices = np.asarray([
            index for index, value in enumerate(threats)
            if value.source == "indestructibles"
        ], dtype=np.int64)
        if len(region_indices):
            for frame_index in range(horizon):
                phase_radius = self._region_phase.radius_after(
                    self.config.observation_delay + frame_index + 1,
                )
                if phase_radius is not None:
                    future_radius[frame_index, region_indices] = np.maximum(
                        future_radius[frame_index, region_indices],
                        phase_radius,
                    )
        return future_x, future_y, future_radius

    def _gap_corridors(
        self,
        player: tuple[float, float, float, float, float],
        bounds: tuple[float, float, float, float],
        threats: Sequence[PredictedThreat],
    ) -> tuple[tuple[_GapBulletGroup, ...], tuple[_GapCorridor, ...]]:
        """Forecast stable center-space corridors through parallel wavefronts."""

        groups = self._gap_bullet_groups(bounds, threats)
        if not groups:
            return (), ()
        px, py, player_radius = player[:3]
        left, right, bottom, top = bounds
        delay_uncertainty = (
            self.config.track_displacement_tolerance
            * min(
                float(self.config.observation_delay),
                float(self.config.gap_sample_interval),
            )
        )
        center_clearance = (
            player_radius + self.config.gap_safety_margin + delay_uncertainty
        )
        corridors: list[_GapCorridor] = []
        for group in groups:
            direction_x, direction_y = group.direction_x, group.direction_y
            normal_x, normal_y = -direction_y, direction_x
            player_longitudinal = px * direction_x + py * direction_y
            mean_longitudinal = sum(
                value.x * direction_x + value.y * direction_y
                for value in group.members
            ) / len(group.members)
            if group.speed <= 0.1:
                continue
            arrival_frames = (
                player_longitudinal - mean_longitudinal
            ) / group.speed
            if (
                arrival_frames < -self.config.gap_hold_frames
                or arrival_frames > self.config.horizon_frames
            ):
                continue

            initial = sorted(
                group.members,
                key=lambda value: (
                    value.x * normal_x + value.y * normal_y,
                    value.key,
                ),
            )
            required_end = min(
                self.config.horizon_frames,
                max(
                    self.config.gap_minimum_lifetime_frames,
                    int(math.ceil(max(0.0, arrival_frames)))
                    + self.config.gap_hold_frames,
                ),
            )
            arrival_samples = {
                min(
                    self.config.horizon_frames,
                    max(0, int(math.floor(arrival_frames))),
                ),
                min(
                    self.config.horizon_frames,
                    max(0, int(math.ceil(arrival_frames))),
                ),
            }
            sample_frames = set(range(
                0,
                self.config.horizon_frames + 1,
                self.config.gap_sample_interval,
            ))
            sample_frames.update(arrival_samples)
            sample_frames.add(required_end)
            snapshots: dict[
                int,
                tuple[
                    list[tuple[PredictedThreat, float, float]],
                    dict[str, int],
                ],
            ] = {}
            for future_frame in sorted(sample_frames):
                projected = sorted(
                    (
                        (
                            value,
                            (position := self._threat_at(value, future_frame))[0]
                            * normal_x + position[1] * normal_y,
                            position[2],
                        )
                        for value in group.members
                    ),
                    key=lambda item: (item[1], item[0].key),
                )
                snapshots[future_frame] = (
                    projected,
                    {
                        value.key: index
                        for index, (value, _, _) in enumerate(projected)
                    },
                )
            for lower, upper in zip(initial, initial[1:]):
                initial_spacing = (
                    (upper.x - lower.x) * normal_x
                    + (upper.y - lower.y) * normal_y
                )
                if initial_spacing > self.config.gap_maximum_lateral_spacing:
                    continue
                minimum_usable_width = math.inf
                lifetime_frames = self.config.horizon_frames
                valid_through_required_end = True
                center_at_arrival: float | None = None
                for future_frame in sorted(sample_frames):
                    projected, indices = snapshots[future_frame]
                    lower_index = indices[lower.key]
                    upper_index = indices[upper.key]
                    adjacent = upper_index == lower_index + 1
                    if adjacent:
                        _, lower_u, lower_radius = projected[lower_index]
                        _, upper_u, upper_radius = projected[upper_index]
                        raw_width = upper_u - upper_radius - lower_u - lower_radius
                        usable_width = raw_width - 2.0 * center_clearance
                    else:
                        usable_width = -math.inf
                        lower_u = upper_u = lower_radius = upper_radius = 0.0
                    if usable_width < self.config.gap_minimum_usable_width:
                        lifetime_frames = min(lifetime_frames, future_frame)
                        if future_frame <= required_end:
                            valid_through_required_end = False
                    else:
                        minimum_usable_width = min(
                            minimum_usable_width,
                            usable_width,
                        )
                    if future_frame in arrival_samples and adjacent:
                        center_at_arrival = 0.5 * (
                            lower_u + lower_radius + upper_u - upper_radius
                        )
                if (
                    not valid_through_required_end
                    or lifetime_frames < self.config.gap_minimum_lifetime_frames
                    or not math.isfinite(minimum_usable_width)
                    or center_at_arrival is None
                ):
                    continue

                anchor_x = (
                    direction_x * player_longitudinal
                    + normal_x * center_at_arrival
                )
                anchor_y = (
                    direction_y * player_longitudinal
                    + normal_y * center_at_arrival
                )
                clamped_x = min(max(anchor_x, left), right)
                clamped_y = min(max(anchor_y, bottom), top)
                clamped_u = clamped_x * normal_x + clamped_y * normal_y
                if abs(clamped_u - center_at_arrival) > 0.5 * minimum_usable_width:
                    continue
                corridors.append(_GapCorridor(
                    key=f"{group.key}:gap:{lower.key}>{upper.key}",
                    group_key=group.key,
                    center_x=clamped_x,
                    center_y=clamped_y,
                    usable_width=minimum_usable_width,
                    lifetime_frames=lifetime_frames,
                    arrival_frames=arrival_frames,
                    path_margin=math.nan,
                    normal_x=normal_x,
                    normal_y=normal_y,
                    member_count=len(group.members),
                ))
        return groups, tuple(sorted(
            corridors,
            key=lambda value: (
                value.arrival_frames,
                -value.usable_width,
                value.key,
            ),
        ))

    def _stationary_nonregion_margin(
        self,
        player: tuple[float, float, float, float, float],
        threats: Sequence[PredictedThreat],
        threat_forecast: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> float:
        px, py, player_radius = player[:3]
        bullet_indices = np.asarray([
            index for index, value in enumerate(threats)
            if value.source == "enemy_bullets"
        ], dtype=np.int64)
        if not len(bullet_indices):
            return math.inf
        if threat_forecast is None:
            threat_forecast = self._gap_threat_forecast(threats)
        future_x, future_y, future_radius = threat_forecast
        return float(np.min(
            np.hypot(
                px - future_x[:, bullet_indices],
                py - future_y[:, bullet_indices],
            )
            - player_radius
            - future_radius[:, bullet_indices]
        ))

    def _gap_navigation(
        self,
        player: tuple[float, float, float, float, float],
        bounds: tuple[float, float, float, float],
        threats: Sequence[PredictedThreat],
        region_anchor: _RegionAnchor | None,
    ) -> tuple[
        tuple[_GapBulletGroup, ...],
        tuple[_GapCorridor, ...],
        _GapCorridor | None,
        str,
    ]:
        if not self.config.gap_prediction_enabled:
            self._active_gap_key = None
            return (), (), None, "inactive"
        groups, corridors = self._gap_corridors(player, bounds, threats)
        px, py = player[:2]
        coverage = {
            group.key: group.coverage_fraction for group in groups
        }
        candidates = [
            value for value in corridors
            if coverage.get(value.group_key, 0.0)
            >= self.config.gap_group_coverage_fraction
        ]
        previous_active_key = self._active_gap_key

        def normal_distance(value: _GapCorridor) -> float:
            return abs(
                (px - value.center_x) * value.normal_x
                + (py - value.center_y) * value.normal_y
            )

        def entry_width(value: _GapCorridor) -> float:
            if (
                value.key == previous_active_key
                or normal_distance(value) <= 0.5 * value.usable_width
            ):
                return value.usable_width
            return min(
                value.usable_width,
                self.config.decision_interval * player[4],
            )

        def coarse_reachable(value: _GapCorridor) -> bool:
            outside = max(
                0.0,
                normal_distance(value) - 0.5 * entry_width(value),
            )
            deadline = max(
                0,
                int(math.floor(
                    max(
                        0.0,
                        value.arrival_frames - self.config.gap_entry_guard_frames,
                    ) + 1e-9
                )),
            )
            return outside <= player[3] * deadline + 1e-6

        def corridor_key(value: _GapCorridor) -> tuple[float | str, ...]:
            region_distance = (
                0.0
                if region_anchor is None else
                math.hypot(
                    value.center_x - region_anchor.x,
                    value.center_y - region_anchor.y,
                )
            )
            outside = max(
                0.0,
                normal_distance(value) - 0.5 * entry_width(value),
            )
            return (
                region_distance,
                outside,
                value.arrival_frames,
                -value.usable_width,
                value.key,
            )

        threat_forecast: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        certified: dict[str, _GapCorridor] = {}
        required_margin = max(
            self.config.gap_path_minimum_margin,
            (
                self.config.region_safe_margin_target
                if region_anchor is not None else
                self.config.gap_path_minimum_margin
            ),
        )

        def certify(value: _GapCorridor) -> _GapCorridor:
            nonlocal threat_forecast
            existing = certified.get(value.key)
            if existing is not None:
                return existing
            if threat_forecast is None:
                threat_forecast = self._gap_threat_forecast(threats)
            path_margin, _travel, entry_plan = self._gap_path_margin(
                player,
                bounds,
                value.center,
                (value.normal_x, value.normal_y),
                entry_width(value),
                max(0.0, value.arrival_frames),
                threats,
                threat_forecast,
                minimum_margin=required_margin,
            )
            result = replace(
                value,
                path_margin=path_margin,
                entry_plan=(entry_plan if path_margin >= required_margin else ()),
            )
            certified[value.key] = result
            return result

        active_geometry = next(
            (
                value for value in candidates
                if value.key == previous_active_key and coarse_reachable(value)
            ),
            None,
        )
        active = certify(active_geometry) if active_geometry is not None else None
        if active is not None and active.path_margin < required_margin:
            active = None

        needs_gap = active is not None
        if not needs_gap and candidates:
            if threat_forecast is None:
                threat_forecast = self._gap_threat_forecast(threats)
            needs_gap = (
                self._stationary_nonregion_margin(
                    player,
                    threats,
                    threat_forecast,
                )
                < self.config.safe_margin_target
            )
        if needs_gap and active is None:
            attempts = 0
            for value in sorted(candidates, key=corridor_key):
                if not coarse_reachable(value):
                    continue
                attempts += 1
                candidate = certify(value)
                if candidate.path_margin >= required_margin:
                    active = candidate
                    break
                if attempts >= self.config.gap_entry_candidate_limit:
                    break

        corridors = tuple(
            certified.get(value.key, value) for value in corridors
        )
        if not needs_gap or active is None:
            leaving = previous_active_key is not None
            self._active_gap_key = None
            return (
                groups,
                corridors,
                None,
                "exit" if leaving else "observe" if corridors else "inactive",
            )

        self._active_gap_key = active.key
        mode = (
            "hold"
            if normal_distance(active) <= max(1.0, 0.5 * active.usable_width) else
            "enter"
        )
        return groups, corridors, active, mode

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
            or (
                self._region_phase.dynamics_memory.maximum_radius
                if self._region_phase.dynamics_memory is not None else None
            )
            or self._region_phase.maximum_radius
            or max(item.radius for row in rows for item in row)
        )
        expansion_duration = (
            self._region_phase._phase_duration("expanding")
            or float(self.config.horizon_frames)
        )
        if self._region_phase.phase == "expanding":
            observed_radius = self._region_phase.observed_radius or maximum_radius
            growth_rate = (
                self._region_phase.growth_rate
                or max(self._region_phase.observed_rate, 0.1)
            )
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
        gap_anchor: _GapCorridor | None,
    ) -> np.ndarray:
        configured_limit = self.config.beam_width
        if region_anchor is not None:
            configured_limit = max(configured_limit, self.config.region_beam_width)
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

    @staticmethod
    def _gap_center_space_distance(
        x: float,
        y: float,
        gap: _GapCorridor,
    ) -> float:
        return max(
            0.0,
            abs(
                (x - gap.center_x) * gap.normal_x
                + (y - gap.center_y) * gap.normal_y
            ) - 0.5 * gap.usable_width,
        )

    def _beam_evaluations(
        self,
        player: tuple[float, float, float, float, float],
        bounds: tuple[float, float, float, float],
        threats: Sequence[PredictedThreat],
        boss_x: float | None,
        region_anchor: _RegionAnchor | None = None,
        gap_anchor: _GapCorridor | None = None,
        *,
        _prefilter_threats: bool = True,
    ) -> tuple[
        tuple[CandidateEvaluation, ...],
        tuple[tuple[Action, ...], ...],
    ]:
        """Search complete three-frame action sequences over the horizon."""

        px, py, player_radius, speed, focus_speed = player
        action_count = len(self.actions)
        move_x = np.asarray([action.move_x for action in self.actions], dtype=np.float64)
        move_y = np.asarray([action.move_y for action in self.actions], dtype=np.float64)
        action_slow = np.asarray(
            [action.slow for action in self.actions], dtype=np.bool_,
        )
        action_speed = np.asarray([
            focus_speed if action.slow else speed for action in self.actions
        ], dtype=np.float64)
        diagonal = (move_x != 0.0) & (move_y != 0.0)
        action_speed[diagonal] *= _SQRT_HALF
        velocity_x = move_x * action_speed
        velocity_y = move_y * action_speed
        action_lookup = {
            (action.move_x, action.move_y, action.slow): index
            for index, action in enumerate(self.actions)
        }
        previous_action_index = (
            -1
            if self._last_action is None else
            action_lookup[
                (
                    self._last_action.move_x,
                    self._last_action.move_y,
                    self._last_action.slow,
                )
            ]
        )
        two_ago_action_index = (
            -1
            if self._previous_action is None else
            action_lookup[
                (
                    self._previous_action.move_x,
                    self._previous_action.move_y,
                    self._previous_action.slow,
                )
            ]
        )

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

            # Threat motion is shared by every beam candidate. Precomputing it
            # once avoids rebuilding these vectors inside all 60 search steps.
            future_frames = np.arange(
                1,
                self.config.horizon_frames + 1,
                dtype=np.float64,
            )[:, None]
            motion_frames = np.minimum(
                future_frames,
                threat_motion_horizon[None, :],
            )
            future_threat_x = (
                threat_x[None, :] + threat_vx[None, :] * motion_frames
            )
            future_threat_y = (
                threat_y[None, :] + threat_vy[None, :] * motion_frames
            )
            future_radius = np.maximum(
                0.1,
                threat_radius[None, :]
                + threat_radius_rate[None, :]
                * np.minimum(future_frames, threat_rate_horizon[None, :]),
            )
            if np.any(threat_is_region):
                for frame_index in range(self.config.horizon_frames):
                    phase_radius = self._region_phase.radius_after(
                        self.config.observation_delay + frame_index + 1,
                    )
                    if phase_radius is not None:
                        future_radius[frame_index, threat_is_region] = np.maximum(
                            future_radius[frame_index, threat_is_region],
                            phase_radius,
                        )
        else:
            threat_x = threat_y = threat_vx = threat_vy = np.empty(0, dtype=np.float64)
            threat_radius = threat_radius_rate = threat_rate_horizon = np.empty(
                0, dtype=np.float64,
            )
            threat_motion_horizon = np.empty(0, dtype=np.float64)
            threat_is_region = np.empty(0, dtype=np.bool_)
            future_threat_x = future_threat_y = future_radius = np.empty(
                (self.config.horizon_frames, 0),
                dtype=np.float64,
            )

        x = np.asarray([px], dtype=np.float64)
        y = np.asarray([py], dtype=np.float64)
        first_action = np.asarray([-1], dtype=np.int16)
        collision_frames = np.zeros(1, dtype=np.int32)
        earliest_collision = np.full(1, self.config.horizon_frames + 1, dtype=np.int32)
        minimum_margin = np.full(1, math.inf, dtype=np.float64)
        minimum_nonregion_margin = np.full(1, math.inf, dtype=np.float64)
        minimum_region_margin = np.full(1, math.inf, dtype=np.float64)
        immediate_corner_clearance = np.full(1, math.inf, dtype=np.float64)
        boundary_penalty = np.zeros(1, dtype=np.float64)
        motion_penalty = np.zeros(1, dtype=np.float64)
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
            minimum_nonregion_margin = minimum_nonregion_margin[parents]
            minimum_region_margin = minimum_region_margin[parents]
            immediate_corner_clearance = immediate_corner_clearance[parents]
            boundary_penalty = boundary_penalty[parents]
            motion_penalty = motion_penalty[parents]
            plans = plans[parents]
            if plans.shape[1] >= 1:
                prior_indices = plans[:, -1].astype(np.int64)
            else:
                prior_indices = np.full(
                    len(action_indices), previous_action_index, dtype=np.int64,
                )
            if plans.shape[1] >= 2:
                two_prior_indices = plans[:, -2].astype(np.int64)
            elif plans.shape[1] == 1:
                two_prior_indices = np.full(
                    len(action_indices), previous_action_index, dtype=np.int64,
                )
            else:
                two_prior_indices = np.full(
                    len(action_indices), two_ago_action_index, dtype=np.int64,
                )

            has_prior = prior_indices >= 0
            safe_prior = np.maximum(prior_indices, 0)
            changed_direction = has_prior & (
                (move_x[action_indices] != move_x[safe_prior])
                | (move_y[action_indices] != move_y[safe_prior])
            )
            motion_penalty += (
                self.config.direction_switch_penalty
                * changed_direction.astype(np.float64)
            )
            reversed_direction = (
                changed_direction
                & ((move_x[action_indices] != 0.0) | (move_y[action_indices] != 0.0))
                & ((move_x[safe_prior] != 0.0) | (move_y[safe_prior] != 0.0))
                & (move_x[action_indices] == -move_x[safe_prior])
                & (move_y[action_indices] == -move_y[safe_prior])
            )
            motion_penalty += (
                self.config.direction_reverse_penalty
                * reversed_direction.astype(np.float64)
            )
            has_two_prior = two_prior_indices >= 0
            safe_two_prior = np.maximum(two_prior_indices, 0)
            aba = (
                changed_direction
                & has_two_prior
                & (move_x[action_indices] == move_x[safe_two_prior])
                & (move_y[action_indices] == move_y[safe_two_prior])
            )
            motion_penalty += (
                self.config.direction_aba_penalty * aba.astype(np.float64)
            )
            changed_speed = (
                has_prior
                & ~changed_direction
                & (action_slow[action_indices] != action_slow[safe_prior])
            )
            motion_penalty += (
                self.config.speed_switch_penalty
                * changed_speed.astype(np.float64)
            )
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
                    tx = future_threat_x[absolute_frame - 1]
                    ty = future_threat_y[absolute_frame - 1]
                    radius = future_radius[absolute_frame - 1]
                    active_threats: slice | np.ndarray = slice(None)
                    if _prefilter_threats:
                        # Once a candidate has an incumbent minimum margin, a
                        # threat outside this expanded candidate AABB cannot
                        # improve that margin or collide this frame. Bounds are
                        # rounded outward so filtering remains conservative.
                        ordinary_threshold = max(
                            self.config.safe_margin_target,
                            self.config.clearance_reward_cap,
                            float(np.max(minimum_nonregion_margin)),
                        )
                        region_threshold = max(
                            self.config.region_safe_margin_target,
                            self.config.portal_clearance,
                            float(np.max(minimum_region_margin)),
                        )
                        class_threshold = np.where(
                            threat_is_region,
                            region_threshold,
                            ordinary_threshold,
                        )
                        reach = radius + player_radius + class_threshold
                        candidate_left = np.nextafter(
                            float(np.min(x)) - reach,
                            -math.inf,
                        )
                        candidate_right = np.nextafter(
                            float(np.max(x)) + reach,
                            math.inf,
                        )
                        candidate_bottom = np.nextafter(
                            float(np.min(y)) - reach,
                            -math.inf,
                        )
                        candidate_top = np.nextafter(
                            float(np.max(y)) + reach,
                            math.inf,
                        )
                        active_threats = (
                            (tx >= candidate_left)
                            & (tx <= candidate_right)
                            & (ty >= candidate_bottom)
                            & (ty <= candidate_top)
                        )
                    active_tx = tx[active_threats]
                    active_ty = ty[active_threats]
                    active_radius = radius[active_threats]
                    active_is_region = threat_is_region[active_threats]
                    if not len(active_tx):
                        frame_margin = np.full(len(x), math.inf, dtype=np.float64)
                        frame_nonregion_margin = np.full(
                            len(x), math.inf, dtype=np.float64,
                        )
                        frame_region_margin = np.full(
                            len(x), math.inf, dtype=np.float64,
                        )
                    else:
                        margins = np.hypot(
                            x[:, None] - active_tx[None, :],
                            y[:, None] - active_ty[None, :],
                        ) - player_radius - active_radius[None, :]
                        frame_margin = margins.min(axis=1)
                        frame_nonregion_margin = (
                            margins[:, ~active_is_region].min(axis=1)
                            if np.any(~active_is_region) else
                            np.full(len(x), math.inf, dtype=np.float64)
                        )
                        frame_region_margin = (
                            margins[:, active_is_region].min(axis=1)
                            if np.any(active_is_region) else
                            np.full(len(x), math.inf, dtype=np.float64)
                        )
                    minimum_margin = np.minimum(minimum_margin, frame_margin)
                    minimum_nonregion_margin = np.minimum(
                        minimum_nonregion_margin,
                        frame_nonregion_margin,
                    )
                    minimum_region_margin = np.minimum(
                        minimum_region_margin,
                        frame_region_margin,
                    )
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
            if segment_start == 0:
                immediate_corner_clearance = np.hypot(
                    np.minimum(x - left, right - x),
                    np.minimum(y - bottom, top - y),
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
            if region_anchor is None:
                margin_shortfall = np.maximum(
                    0.0,
                    self.config.safe_margin_target - minimum_margin,
                )
                danger_margin_shortfall = np.maximum(
                    0.0,
                    self.config.danger_margin_target - minimum_margin,
                )
                corner_clearance = np.hypot(
                    np.minimum(x - left, right - x),
                    np.minimum(y - bottom, top - y),
                )
                margin_shortfall += self.config.corner_reserve_weight * np.maximum(
                    0.0,
                    self.config.corner_reserve_target - corner_clearance,
                )
                region_margin_shortfall = np.zeros_like(margin_shortfall)
                clearance_reward = self.config.clearance_reward_weight * np.clip(
                    minimum_margin,
                    0.0,
                    self.config.clearance_reward_cap,
                )
            else:
                margin_shortfall = np.maximum(
                    0.0,
                    self.config.safe_margin_target - minimum_nonregion_margin,
                )
                danger_margin_shortfall = np.maximum(
                    0.0,
                    self.config.danger_margin_target - minimum_nonregion_margin,
                )
                region_margin_shortfall = np.maximum(
                    0.0,
                    self.config.region_safe_margin_target - minimum_region_margin,
                )
                clearance_reward = self.config.clearance_reward_weight * (
                    np.clip(
                        minimum_nonregion_margin,
                        0.0,
                        self.config.clearance_reward_cap,
                    )
                    + np.clip(
                        minimum_region_margin,
                        0.0,
                        max(
                            self.config.region_safe_margin_target,
                            self.config.portal_clearance,
                        ),
                    )
                )
            preference = (
                boundary_penalty + alignment + motion_penalty - clearance_reward
            )
            order = self._beam_candidate_order(
                collided,
                earliest_collision,
                collision_frames,
                danger_margin_shortfall,
                margin_shortfall,
                region_margin_shortfall,
                preference,
                minimum_margin,
                collision_priority_frames=self.config.collision_priority_frames,
                tie_breaker=first_action,
            )
            keep = self._diverse_keep(
                order,
                x,
                y,
                first_action,
                region_anchor,
                gap_anchor,
            )
            x, y = x[keep], y[keep]
            first_action = first_action[keep]
            collision_frames = collision_frames[keep]
            earliest_collision = earliest_collision[keep]
            minimum_margin = minimum_margin[keep]
            minimum_nonregion_margin = minimum_nonregion_margin[keep]
            minimum_region_margin = minimum_region_margin[keep]
            immediate_corner_clearance = immediate_corner_clearance[keep]
            boundary_penalty = boundary_penalty[keep]
            motion_penalty = motion_penalty[keep]
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
                    gap_anchor,
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
            if region_anchor is None:
                margin_shortfall = np.maximum(
                    0.0,
                    self.config.safe_margin_target - minimum_margin[matches],
                )
                danger_margin_shortfall = np.maximum(
                    0.0,
                    self.config.danger_margin_target - minimum_margin[matches],
                )
                margin_shortfall += self.config.corner_reserve_weight * np.maximum(
                    0.0,
                    self.config.corner_reserve_target
                    - immediate_corner_clearance[matches],
                )
                region_margin_shortfall = np.zeros_like(margin_shortfall)
                clearance_reward = self.config.clearance_reward_weight * np.clip(
                    minimum_margin[matches],
                    0.0,
                    self.config.clearance_reward_cap,
                )
            else:
                margin_shortfall = np.maximum(
                    0.0,
                    self.config.safe_margin_target
                    - minimum_nonregion_margin[matches],
                )
                danger_margin_shortfall = np.maximum(
                    0.0,
                    self.config.danger_margin_target
                    - minimum_nonregion_margin[matches],
                )
                region_margin_shortfall = np.maximum(
                    0.0,
                    self.config.region_safe_margin_target
                    - minimum_region_margin[matches],
                )
                clearance_reward = self.config.clearance_reward_weight * (
                    np.clip(
                        minimum_nonregion_margin[matches],
                        0.0,
                        self.config.clearance_reward_cap,
                    )
                    + np.clip(
                        minimum_region_margin[matches],
                        0.0,
                        max(
                            self.config.region_safe_margin_target,
                            self.config.portal_clearance,
                        ),
                    )
                )
            preference = (
                boundary_penalty[matches]
                + alignment
                + motion_penalty[matches]
                - clearance_reward
            )
            order = self._beam_candidate_order(
                collided,
                earliest_collision[matches],
                collision_frames[matches],
                danger_margin_shortfall,
                margin_shortfall,
                region_margin_shortfall,
                preference,
                minimum_margin[matches],
                collision_priority_frames=self.config.collision_priority_frames,
            )
            selected = matches[int(order[0])]
            earliest = int(earliest_collision[selected])
            selected_alignment = float(alignment[int(order[0])])
            if gap_anchor is not None:
                immediate_path = self._path(
                    action,
                    player[:2],
                    player[3],
                    player[4],
                    bounds,
                    self.config.decision_interval,
                    self.config.decision_interval,
                )
                immediate_x, immediate_y, _ = immediate_path[-1]
                selected_alignment += self.config.gap_anchor_weight * (
                    self._gap_center_space_distance(
                        immediate_x,
                        immediate_y,
                        gap_anchor,
                    )
                )
            evaluations.append(CandidateEvaluation(
                action=replace(action, spell=False),
                collided=bool(collision_frames[selected] > 0),
                collision_frames=int(collision_frames[selected]),
                earliest_collision_frame=(
                    None if earliest > self.config.horizon_frames else earliest
                ),
                minimum_margin=float(minimum_margin[selected]),
                boundary_penalty=float(boundary_penalty[selected]),
                boss_alignment=selected_alignment,
                motion_penalty=float(motion_penalty[selected]),
                minimum_nonregion_margin=float(
                    minimum_nonregion_margin[selected]
                ),
                minimum_region_margin=float(minimum_region_margin[selected]),
                immediate_corner_clearance=float(
                    immediate_corner_clearance[selected]
                ),
            ))
            action_plans.append(tuple(
                self.actions[int(value)] for value in plans[selected]
            ))
        return tuple(evaluations), tuple(action_plans)

    @staticmethod
    def _beam_candidate_order(
        collided: np.ndarray,
        earliest_collision: np.ndarray,
        collision_frames: np.ndarray,
        danger_margin_shortfall: np.ndarray,
        margin_shortfall: np.ndarray,
        region_margin_shortfall: np.ndarray,
        preference: np.ndarray,
        minimum_margin: np.ndarray,
        *,
        collision_priority_frames: int = 36,
        tie_breaker: np.ndarray | None = None,
    ) -> np.ndarray:
        """Order beam candidates with safety ahead of motion preferences."""

        priority_earliest = np.minimum(
            earliest_collision,
            collision_priority_frames,
        )
        keys: tuple[np.ndarray, ...] = (
            -minimum_margin,
            -earliest_collision,
            preference,
            collision_frames,
            margin_shortfall,
            region_margin_shortfall,
            danger_margin_shortfall,
            -priority_earliest,
            collided.astype(np.int8),
        )
        if tie_breaker is not None:
            keys = (tie_breaker, *keys)
        return np.lexsort(keys)

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
        gap_anchor: _GapCorridor | None = None,
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
        minimum_nonregion_margin = math.inf
        minimum_region_margin = math.inf
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
                if threat.source == "indestructibles":
                    minimum_region_margin = min(minimum_region_margin, margin)
                else:
                    minimum_nonregion_margin = min(
                        minimum_nonregion_margin,
                        margin,
                    )
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
        if gap_anchor is not None:
            alignment += self.config.gap_anchor_weight * (
                self._gap_center_space_distance(
                    final_x,
                    final_y,
                    gap_anchor,
                )
            )
        return CandidateEvaluation(
            action=replace(action, spell=False),
            collided=collision_frames > 0,
            collision_frames=collision_frames,
            earliest_collision_frame=earliest_collision,
            minimum_margin=minimum_margin,
            boundary_penalty=boundary_penalty,
            boss_alignment=alignment,
            motion_penalty=self._transition_penalty(
                action,
                self._last_action,
                self._previous_action,
            ),
            minimum_nonregion_margin=minimum_nonregion_margin,
            minimum_region_margin=minimum_region_margin,
            immediate_corner_clearance=self._corner_clearance(
                path[self.config.decision_interval][0],
                path[self.config.decision_interval][1],
                bounds,
            ),
        )

    def _apply_direction_hold(
        self,
        selected_index: int,
        evaluations: Sequence[CandidateEvaluation],
        region_anchor: _RegionAnchor | None,
        source_frame: int,
        gap_anchor: _GapCorridor | None = None,
    ) -> int:
        """Keep a safe direction briefly, while never masking urgent avoidance."""

        if self._last_action is None or self._direction_started_frame is None:
            return selected_index
        selected = evaluations[selected_index]
        if self._direction(selected.action) == self._direction(self._last_action):
            return selected_index

        held_frames = max(0, source_frame - self._direction_started_frame)
        if held_frames >= self.config.minimum_direction_hold_frames:
            return selected_index
        incumbent_index = next(
            (
                index
                for index, value in enumerate(evaluations)
                if value.action.discrete == self._last_action.discrete
            ),
            None,
        )
        if incumbent_index is None:
            return selected_index
        incumbent = evaluations[incumbent_index]
        remaining_hold_frames = max(
            0,
            self.config.minimum_direction_hold_frames - held_frames,
        )
        evacuation_progress_release = (
            region_anchor is not None
            and region_anchor.navigation_mode == "evacuate"
            and region_anchor.deadline_slack <= remaining_hold_frames
            and math.isfinite(
                selected.boundary_penalty + selected.boss_alignment
            )
            and math.isfinite(
                incumbent.boundary_penalty + incumbent.boss_alignment
            )
            and (
                incumbent.boundary_penalty + incumbent.boss_alignment
                - selected.boundary_penalty - selected.boss_alignment
            )
            >= self.config.direction_switch_penalty
        )
        gap_entry_release = (
            gap_anchor is not None
            and (
                (
                    gap_anchor.entry_action is not None
                    and selected.action.discrete
                    == gap_anchor.entry_action.discrete
                    and incumbent.action.discrete
                    != gap_anchor.entry_action.discrete
                )
                or (
                    gap_anchor.arrival_frames
                    <= remaining_hold_frames + self.config.gap_entry_guard_frames
                    and math.isfinite(
                        selected.boundary_penalty + selected.boss_alignment
                    )
                    and math.isfinite(
                        incumbent.boundary_penalty + incumbent.boss_alignment
                    )
                    and (
                        incumbent.boundary_penalty + incumbent.boss_alignment
                        - selected.boundary_penalty - selected.boss_alignment
                    ) >= self.config.direction_switch_penalty
                )
            )
        )
        collision_release_horizon = min(
            self.config.horizon_frames,
            self.config.emergency_collision_frames + remaining_hold_frames,
        )
        time_critical_collision = (
            incumbent.collided
            and incumbent.earliest_collision_frame is not None
            and incumbent.earliest_collision_frame
            <= collision_release_horizon
        )
        dangerously_close = incumbent.minimum_margin <= self.config.emergency_margin
        avoids_predicted_collision = time_critical_collision and not selected.collided
        delays_collision = (
            time_critical_collision
            and selected.collided
            and incumbent.earliest_collision_frame is not None
            and selected.earliest_collision_frame is not None
            and selected.earliest_collision_frame
            >= incumbent.earliest_collision_frame + self.config.decision_interval
        )
        if region_anchor is None:
            margin_gain = selected.minimum_margin - incumbent.minimum_margin
            reaches_safe_reserve = (
                incumbent.minimum_margin < self.config.safe_margin_target
                <= selected.minimum_margin
            )
            material_margin_gain = margin_gain >= self.config.switch_margin_gain
            corner_escape = (
                incumbent.immediate_corner_clearance
                < self.config.corner_reserve_target
                and selected.immediate_corner_clearance
                - incumbent.immediate_corner_clearance
                >= self.config.switch_margin_gain
            )
        else:
            ordinary_margin_gain = (
                selected.minimum_nonregion_margin
                - incumbent.minimum_nonregion_margin
            )
            forced_region_margin_gain = (
                selected.minimum_region_margin - incumbent.minimum_region_margin
            )
            reaches_safe_reserve = (
                incumbent.minimum_nonregion_margin
                < self.config.safe_margin_target
                <= selected.minimum_nonregion_margin
            ) or (
                incumbent.minimum_region_margin
                < self.config.region_safe_margin_target
                <= selected.minimum_region_margin
            )
            material_margin_gain = (
                ordinary_margin_gain >= self.config.switch_margin_gain
                or forced_region_margin_gain >= self.config.switch_margin_gain
            )
            corner_escape = False
        reserve_release_allowed = held_frames >= max(
            0,
            self.config.minimum_direction_hold_frames
            - self.config.decision_interval,
        )
        if (
            time_critical_collision
            or avoids_predicted_collision
            or delays_collision
            or corner_escape
            or evacuation_progress_release
            or gap_entry_release
            or reserve_release_allowed
            and (
                dangerously_close
                or reaches_safe_reserve
                or material_margin_gain
            )
        ):
            return selected_index
        return incumbent_index

    def _committed_action_respects_direction_hold(
        self,
        committed_action: Action,
        proposed_action: Action,
        source_frame: int,
    ) -> bool:
        if self._last_action is None or self._direction_started_frame is None:
            return True
        committed_direction = self._direction(committed_action)
        proposed_direction = self._direction(proposed_action)
        if committed_direction == proposed_direction:
            return True
        held_frames = max(0, source_frame - self._direction_started_frame)
        if held_frames >= self.config.minimum_direction_hold_frames:
            return True
        # _compute already applied the hold. A different proposed direction
        # inside this window therefore passed an emergency/safety release and
        # must not be replaced by a stale committed action.
        return False

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
        gap_groups, gap_corridors, gap_anchor, gap_mode = self._gap_navigation(
            player,
            bounds,
            threats,
            region_anchor,
        )
        evaluations, action_plans = self._beam_evaluations(
            player,
            bounds,
            threats,
            boss_x,
            region_anchor,
            gap_anchor,
        )
        gap_entry_action = (
            gap_anchor.entry_action
            if gap_anchor is not None and gap_mode == "enter" else
            None
        )

        def key(index: int) -> tuple[float, ...]:
            value = evaluations[index]
            earliest = (
                math.inf
                if value.earliest_collision_frame is None else
                float(value.earliest_collision_frame)
            )
            priority_earliest = min(
                earliest,
                float(self.config.collision_priority_frames),
            )
            preference = (
                value.boundary_penalty
                + value.boss_alignment
                + value.motion_penalty
                - self._clearance_reward(
                    value.minimum_margin,
                    region=region_anchor is not None,
                    nonregion_margin=value.minimum_nonregion_margin,
                    region_margin=value.minimum_region_margin,
                )
            )
            gap_entry_mismatch = float(
                gap_entry_action is not None
                and value.action.discrete != gap_entry_action.discrete
            )
            if region_anchor is None:
                return (
                    float(value.collided),
                    -priority_earliest,
                    gap_entry_mismatch,
                    max(
                        0.0,
                        self.config.danger_margin_target - value.minimum_margin,
                    ),
                    max(
                        0.0,
                        self.config.safe_margin_target - value.minimum_margin,
                    ) + self._corner_shortfall(
                        value.immediate_corner_clearance,
                    ),
                    float(value.collision_frames),
                    preference,
                    -earliest,
                    -value.minimum_margin,
                    float(index),
                )
            return (
                float(value.collided),
                -priority_earliest,
                gap_entry_mismatch,
                max(
                    0.0,
                    self.config.danger_margin_target
                    - value.minimum_nonregion_margin,
                ),
                max(
                    0.0,
                    self.config.region_safe_margin_target
                    - value.minimum_region_margin,
                ),
                max(
                    0.0,
                    self.config.safe_margin_target
                    - value.minimum_nonregion_margin,
                ),
                float(value.collision_frames),
                preference,
                -earliest,
                -value.minimum_nonregion_margin,
                -value.minimum_region_margin,
                -value.minimum_margin,
                float(index),
            )
        selected_index = min(
            range(len(evaluations)),
            key=key,
        )
        selected_index = self._apply_direction_hold(
            selected_index,
            evaluations,
            region_anchor,
            source_frame,
            gap_anchor,
        )
        selected_plan = (
            gap_anchor.entry_plan
            if (
                gap_entry_action is not None
                and evaluations[selected_index].action.discrete
                == gap_entry_action.discrete
                and gap_anchor is not None
            ) else
            action_plans[selected_index]
        )
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
            gap_bullet_group_count=len(gap_groups),
            gap_corridor_count=len(gap_corridors),
            gap_selected_center=(
                None if gap_anchor is None else gap_anchor.center
            ),
            gap_selected_width=(
                None if gap_anchor is None else gap_anchor.usable_width
            ),
            gap_selected_lifetime_frames=(
                None if gap_anchor is None else gap_anchor.lifetime_frames
            ),
            gap_navigation_mode=gap_mode,
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
                proposed.gap_navigation_mode,
                proposed.gap_selected_center,
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
                and self._committed_action_respects_direction_hold(
                    committed_action,
                    proposed.action,
                    source_frame,
                )
                and committed_margin is not None
                and committed_margin >= self.config.region_safe_margin_target
                and committed_evaluation is not None
                and committed_evaluation.minimum_nonregion_margin
                >= self.config.safe_margin_target
                and (
                    (
                        not committed_evaluation.collided
                        and committed_evaluation.minimum_region_margin
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
            self._remember_action(self._decision.action, source_frame)
            self._last_decision_frame = source_frame
            return self._decision
        assert self._decision is not None
        return replace(
            self._decision,
            source_frame=source_frame,
            recomputed=False,
            threats=threats,
        )

    def controller_overlay_state(
        self,
        decision: MPCDecision,
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Expose the live planner geometry without duplicating its inference."""

        observation = _unwrap_observation(observation)
        player_radius = self._player(
            observation,
            self.config.observation_delay,
        )[2]
        left, right, bottom, top = self._bounds(observation, player_radius)
        region_phase_radii = [
            self._region_phase.radius_after(
                self.config.observation_delay + future_frame,
            )
            for future_frame in range(1, self.config.horizon_frames + 1)
        ]
        return {
            "schema_version": 1,
            "revision": decision.source_frame,
            "source_frame": decision.source_frame,
            "horizon_frames": self.config.horizon_frames,
            "future_start": 1,
            "danger_margin": self.config.danger_margin_target,
            "safe_margin": self.config.safe_margin_target,
            "region_safe_margin": self.config.region_safe_margin_target,
            "region_navigation_active": decision.region_anchor is not None,
            "player_radius": player_radius,
            "bounds": {
                "left": left,
                "right": right,
                "bottom": bottom,
                "top": top,
            },
            "threats": [
                {
                    "key": threat.key,
                    "source": threat.source,
                    "x": threat.x,
                    "y": threat.y,
                    "vx": threat.vx,
                    "vy": threat.vy,
                    "radius": threat.radius,
                    "radius_rate": threat.radius_rate,
                    "radius_rate_horizon": threat.radius_rate_horizon,
                    "motion_horizon": threat.motion_horizon,
                }
                for threat in decision.threats
            ],
            "region_phase_radii": region_phase_radii,
        }

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
