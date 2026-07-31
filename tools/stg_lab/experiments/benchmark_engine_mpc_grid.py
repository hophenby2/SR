"""Compare live MPC beam plans with time-layered safety-grid plans."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

from stg_lab.engine_mpc import (
    EngineMPC,
    MPCConfig,
    MPCDecision,
    PredictedThreat,
    RegionDynamicsMemory,
)
from stg_lab.planning import PlannerConfig, RiskConfig, SpatioTemporalPlanner
from stg_lab.protocol import Action


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = (
    ROOT / "artifacts" / "engine-mpc-boss3-heldout-v40-d5-region-dynamics-v2.json"
)
DEFAULT_SOURCE_FRAMES = (488, 995, 1292, 1532, 2102, 2672, 2801, 3242, 3695)
GRID_COLLISION_RISK = 1_000_000.0
GRID_SAFETY_THRESHOLDS = (0.2, 2.5, 20.0, 500_000.0)
CURRENT_CONTROLLER_OVERRIDES = {
    "safe_margin_target": 20.0,
    "region_safe_margin_target": 8.0,
}


def _controller_config(
    report: Mapping[str, Any],
    controller_profile: str,
) -> MPCConfig:
    values = dict(report["controller"]["config"])
    if controller_profile == "current":
        values.update(CURRENT_CONTROLLER_OVERRIDES)
    elif controller_profile != "recorded":
        raise ValueError(f"unknown controller profile: {controller_profile}")
    memory = values.get("region_dynamics_memory")
    if isinstance(memory, Mapping):
        values["region_dynamics_memory"] = RegionDynamicsMemory(**memory)
    return MPCConfig(**values)


def _report_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _threat_record(
    controller: EngineMPC,
    threat: PredictedThreat,
    future_frame: int,
) -> dict[str, Any]:
    x, y, radius = controller._threat_at(threat, future_frame)
    return {
        "id": threat.key,
        "x": x,
        "y": y,
        "vx": threat.vx,
        "vy": threat.vy,
        "radius_x": radius,
        "radius_y": radius,
        "danger": 1.0,
        "lethal": True,
    }


class _GridSource:
    def __init__(
        self,
        controller: EngineMPC,
        threats: Sequence[PredictedThreat],
    ) -> None:
        self.controller = controller
        self.threats = tuple(threats)

    def forecast_threats(
        self,
        horizon_frames: int,
        sample_every: int,
    ) -> tuple[tuple[dict[str, Any], ...], ...]:
        offsets = list(range(sample_every, horizon_frames + 1, sample_every))
        if not offsets or offsets[-1] != horizon_frames:
            offsets.append(horizon_frames)
        return tuple(
            tuple(
                _threat_record(self.controller, threat, future_frame)
                for threat in self.threats
            )
            for future_frame in offsets
        )


@dataclass(frozen=True, slots=True)
class _PlanMetrics:
    evaluated_frames: int
    collision_frames: int
    collision_events: int
    earliest_collision_frame: int | None
    closest_approach_frame: int | None
    minimum_margin: float
    sampled_layer_minimum_margin: float
    between_layer_minimum_margin: float
    sampled_layer_collision_frames: int
    between_layer_collision_frames: int
    margin_p10: float
    margin_median: float
    frames_at_or_below_margin_4: int
    frames_at_or_below_margin_8: int
    frames_at_or_below_margin_12: int
    frames_at_or_below_margin_20: int
    direction_changes: int
    exact_reversals: int
    aba_changes: int
    speed_mode_changes: int
    distance: float


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return math.inf
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _rate(numerator: float, denominator: int, scale: float = 1.0) -> float | None:
    if denominator <= 0:
        return None
    return scale * numerator / denominator


def _evaluate_plan(
    controller: EngineMPC,
    player: tuple[float, float, float, float, float],
    bounds: tuple[float, float, float, float],
    threats: Sequence[PredictedThreat],
    actions: Sequence[Action],
) -> _PlanMetrics:
    x, y, player_radius, speed, focus_speed = player
    left, right, bottom, top = bounds
    minimum_margin = math.inf
    closest_approach_frame = None
    collision_frames = 0
    collision_events = 0
    earliest_collision = None
    sampled_layer_minimum_margin = math.inf
    between_layer_minimum_margin = math.inf
    sampled_layer_collision_frames = 0
    between_layer_collision_frames = 0
    margins: list[float] = []
    frames_at_or_below = {4.0: 0, 8.0: 0, 12.0: 0, 20.0: 0}
    direction_changes = 0
    exact_reversals = 0
    aba_changes = 0
    speed_mode_changes = 0
    distance = 0.0
    previous: Action | None = None
    two_ago: Action | None = None
    previous_colliding = False
    future_frame = 0
    for action in actions:
        if previous is not None:
            direction_changed = (
                action.move_x != previous.move_x
                or action.move_y != previous.move_y
            )
            direction_changes += int(direction_changed)
            exact_reversals += int(
                (previous.move_x != 0 or previous.move_y != 0)
                and action.move_x == -previous.move_x
                and action.move_y == -previous.move_y
            )
            aba_changes += int(
                direction_changed
                and two_ago is not None
                and action.move_x == two_ago.move_x
                and action.move_y == two_ago.move_y
            )
            speed_mode_changes += int(
                not direction_changed and action.slow != previous.slow
            )
        two_ago = previous
        previous = action
        magnitude = focus_speed if action.slow else speed
        if action.move_x != 0 and action.move_y != 0:
            magnitude /= math.sqrt(2.0)
        velocity_x = action.move_x * magnitude
        velocity_y = action.move_y * magnitude
        for _ in range(controller.config.decision_interval):
            if future_frame >= controller.config.horizon_frames:
                break
            next_x = min(max(x + velocity_x, left), right)
            next_y = min(max(y + velocity_y, bottom), top)
            distance += math.hypot(next_x - x, next_y - y)
            x, y = next_x, next_y
            future_frame += 1
            frame_margin = math.inf
            for threat in threats:
                tx, ty, radius = controller._threat_at(threat, future_frame)
                frame_margin = min(
                    frame_margin,
                    math.hypot(x - tx, y - ty) - player_radius - radius,
                )
            margins.append(frame_margin)
            if frame_margin < minimum_margin:
                minimum_margin = frame_margin
                closest_approach_frame = future_frame
            sampled_layer = future_frame % controller.config.decision_interval == 0
            if sampled_layer:
                sampled_layer_minimum_margin = min(
                    sampled_layer_minimum_margin,
                    frame_margin,
                )
            else:
                between_layer_minimum_margin = min(
                    between_layer_minimum_margin,
                    frame_margin,
                )
            for threshold in frames_at_or_below:
                frames_at_or_below[threshold] += int(frame_margin <= threshold)
            colliding = frame_margin <= 0.0
            if colliding:
                collision_frames += 1
                if sampled_layer:
                    sampled_layer_collision_frames += 1
                else:
                    between_layer_collision_frames += 1
                if earliest_collision is None:
                    earliest_collision = future_frame
                if not previous_colliding:
                    collision_events += 1
            previous_colliding = colliding
        if future_frame >= controller.config.horizon_frames:
            break
    return _PlanMetrics(
        evaluated_frames=future_frame,
        collision_frames=collision_frames,
        collision_events=collision_events,
        earliest_collision_frame=earliest_collision,
        closest_approach_frame=closest_approach_frame,
        minimum_margin=minimum_margin,
        sampled_layer_minimum_margin=sampled_layer_minimum_margin,
        between_layer_minimum_margin=between_layer_minimum_margin,
        sampled_layer_collision_frames=sampled_layer_collision_frames,
        between_layer_collision_frames=between_layer_collision_frames,
        margin_p10=_percentile(margins, 0.10),
        margin_median=_percentile(margins, 0.50),
        frames_at_or_below_margin_4=frames_at_or_below[4.0],
        frames_at_or_below_margin_8=frames_at_or_below[8.0],
        frames_at_or_below_margin_12=frames_at_or_below[12.0],
        frames_at_or_below_margin_20=frames_at_or_below[20.0],
        direction_changes=direction_changes,
        exact_reversals=exact_reversals,
        aba_changes=aba_changes,
        speed_mode_changes=speed_mode_changes,
        distance=distance,
    )


def _grid_snapshot(
    controller: EngineMPC,
    observation: Mapping[str, Any],
    threats: Sequence[PredictedThreat],
) -> dict[str, Any]:
    player = controller._player(observation, controller.config.observation_delay)
    return {
        "frame": controller.estimator.last_frame,
        "world": dict(observation.get("world", {})),
        "player": {
            "x": player[0],
            "y": player[1],
            "radius": player[2],
            "speed": player[3],
            "focus_speed": player[4],
        },
        "threats": [
            _threat_record(controller, threat, 0) for threat in threats
        ],
    }


def _grid_plan(
    controller: EngineMPC,
    observation: Mapping[str, Any],
    decision: MPCDecision,
    *,
    cell_size: float,
    conservative: bool,
    goal_policy: str,
) -> tuple[Any, float, tuple[float, float]]:
    player = controller._player(observation, controller.config.observation_delay)
    bounds = controller._bounds(observation, player[2])
    boss_x = controller._boss_x(observation, controller.config.observation_delay)
    preferred_y = controller._preferred_y(bounds)
    intended_goal = decision.region_anchor or (
        player[0] if boss_x is None else min(max(boss_x, bounds[0]), bounds[1]),
        preferred_y,
    )
    if goal_policy == "survival":
        planner_goal = None
    elif goal_policy == "hard":
        planner_goal = intended_goal
    else:
        raise ValueError(f"unknown grid goal policy: {goal_policy}")
    cell_guard = cell_size * math.sqrt(2.0) * 0.5 if conservative else 0.0
    planner = SpatioTemporalPlanner(PlannerConfig(
        risk=RiskConfig(
            horizon_frames=controller.config.horizon_frames,
            sample_every=controller.config.decision_interval,
            cell_size=cell_size,
            reaction_frames=6,
            proximity_margin=max(32.0, controller.config.safe_margin_target),
            proximity_decay=12.0,
            uncertainty_per_frame=0.025,
            uncertainty_margin=cell_guard,
            collision_risk=GRID_COLLISION_RISK,
            boundary_margin=16.0,
            # Dense fields can accumulate hundreds of points of proximity
            # risk.  Keep collision in a distinct lexicographic level instead
            # of saturating it together with ordinary bullet density.
            safety_thresholds=GRID_SAFETY_THRESHOLDS,
        ),
        cumulative_weight=1.0,
        distance_weight=0.02,
        safe_region_level=0,
        cache_layers=False,
    ))
    source = _GridSource(controller, decision.threats)
    snapshot = _grid_snapshot(controller, observation, decision.threats)
    started = time.perf_counter()
    result = planner.plan(source, observation=snapshot, goal=planner_goal)
    return result, time.perf_counter() - started, intended_goal


def _sample(
    controller: EngineMPC,
    observation: Mapping[str, Any],
    decision: MPCDecision,
    record: Mapping[str, Any],
    cell_sizes: Sequence[float],
    beam_seconds: float,
    grid_goal_policy: str,
) -> dict[str, Any]:
    player = controller._player(observation, controller.config.observation_delay)
    bounds = controller._bounds(observation, player[2])
    beam_metrics = _evaluate_plan(
        controller,
        player,
        bounds,
        decision.threats,
        decision.planned_actions,
    )
    grids: list[dict[str, Any]] = []
    for cell_size in cell_sizes:
        for conservative in (False, True):
            result, duration, intended_goal = _grid_plan(
                controller,
                observation,
                decision,
                cell_size=cell_size,
                conservative=conservative,
                goal_policy=grid_goal_policy,
            )
            metrics = _evaluate_plan(
                controller,
                player,
                bounds,
                decision.threats,
                result.actions,
            )
            terminal_position = (
                result.steps[-1].position
                if result.steps else
                (player[0], player[1])
            )
            grids.append({
                "cell_size": cell_size,
                "conservative_whole_cell": conservative,
                "goal_policy": grid_goal_policy,
                "seconds": duration,
                "grid_shape": list(result.field.risk.shape),
                "peak_level": result.peak_level,
                "total_risk": result.total_risk,
                "reached_goal": result.reached_goal,
                "intended_goal": list(intended_goal),
                "terminal_position": list(terminal_position),
                "terminal_goal_distance": math.hypot(
                    terminal_position[0] - intended_goal[0],
                    terminal_position[1] - intended_goal[1],
                ),
                "first_action": result.first_action.to_dict(),
                "metrics": asdict(metrics),
            })
    return {
        "source_frame": int(record["source_frame"]),
        "threat_count": len(decision.threats),
        "region_navigation_mode": decision.region_navigation_mode,
        "beam": {
            "seconds": beam_seconds,
            "first_action": decision.action.to_dict(),
            "using_committed_plan": decision.using_committed_plan,
            "metrics": asdict(beam_metrics),
        },
        "grids": grids,
    }


def benchmark(
    report: Mapping[str, Any],
    source_frames: Sequence[int],
    cell_sizes: Sequence[float],
    *,
    report_path: Path = DEFAULT_REPORT,
    controller_profile: str = "current",
    grid_goal_policy: str = "survival",
) -> dict[str, Any]:
    requested = set(source_frames)
    controller = EngineMPC(_controller_config(report, controller_profile))
    samples: list[dict[str, Any]] = []
    previous_action: Action | None = None
    beam_first_action_changes = 0
    replayed_decisions = 0
    for record in report["decisions"]:
        observation = record.get("recorded_controller_input_observation")
        if not isinstance(observation, Mapping):
            continue
        started = time.perf_counter()
        decision = controller.select(observation)
        beam_seconds = time.perf_counter() - started
        replayed_decisions += 1
        if previous_action is not None and (
            decision.action.move_x != previous_action.move_x
            or decision.action.move_y != previous_action.move_y
        ):
            beam_first_action_changes += 1
        previous_action = decision.action
        if int(record["source_frame"]) in requested:
            samples.append(_sample(
                controller,
                observation,
                decision,
                record,
                cell_sizes,
                beam_seconds,
                grid_goal_policy,
            ))
        if len(samples) == len(requested):
            break
    found = {sample["source_frame"] for sample in samples}
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"source frames have no recorded observations: {missing}")

    def aggregate(values: Sequence[float]) -> dict[str, float]:
        return {
            "minimum": min(values),
            "median": statistics.median(values),
            "mean": statistics.fmean(values),
            "maximum": max(values),
        }

    beam_runtime = [sample["beam"]["seconds"] for sample in samples]
    beam_margins = [sample["beam"]["metrics"]["minimum_margin"] for sample in samples]
    beam_collision_frames = sum(
        sample["beam"]["metrics"]["collision_frames"] for sample in samples
    )
    beam_runtime_mean = statistics.fmean(beam_runtime)
    beam_margin_mean = statistics.fmean(beam_margins)
    variants: list[dict[str, Any]] = []
    for cell_size in cell_sizes:
        for conservative in (False, True):
            selected = [
                grid
                for sample in samples
                for grid in sample["grids"]
                if grid["cell_size"] == cell_size
                and grid["conservative_whole_cell"] is conservative
            ]
            runtime_values = [value["seconds"] for value in selected]
            margin_values = [
                value["metrics"]["minimum_margin"] for value in selected
            ]
            goal_distances = [
                value["terminal_goal_distance"] for value in selected
            ]
            collision_frame_total = sum(
                value["metrics"]["collision_frames"] for value in selected
            )
            evaluated_frame_total = sum(
                value["metrics"]["evaluated_frames"] for value in selected
            )
            direction_change_total = sum(
                value["metrics"]["direction_changes"] for value in selected
            )
            aba_change_total = sum(
                value["metrics"]["aba_changes"] for value in selected
            )
            variants.append({
                "cell_size": cell_size,
                "conservative_whole_cell": conservative,
                "sample_count": len(selected),
                "runtime_seconds": aggregate(runtime_values),
                "runtime_mean_ratio_vs_beam": (
                    statistics.fmean(runtime_values) / beam_runtime_mean
                    if beam_runtime_mean > 0.0 else math.inf
                ),
                "minimum_margin": aggregate(margin_values),
                "minimum_margin_mean_delta_vs_beam": (
                    statistics.fmean(margin_values) - beam_margin_mean
                ),
                "terminal_goal_distance": aggregate(goal_distances),
                "collision_plan_count": sum(
                    value["metrics"]["collision_frames"] > 0 for value in selected
                ),
                "complete_horizon_plan_count": sum(
                    value["metrics"]["evaluated_frames"]
                    == controller.config.horizon_frames
                    for value in selected
                ),
                "evaluated_frame_total": evaluated_frame_total,
                "collision_frame_total": collision_frame_total,
                "collision_frame_rate": _rate(
                    collision_frame_total,
                    evaluated_frame_total,
                ),
                "collision_frame_delta_vs_beam": (
                    collision_frame_total - beam_collision_frames
                ),
                "collision_event_total": sum(
                    value["metrics"]["collision_events"] for value in selected
                ),
                "between_layer_collision_frame_total": sum(
                    value["metrics"]["between_layer_collision_frames"]
                    for value in selected
                ),
                "sampled_layer_collision_frame_total": sum(
                    value["metrics"]["sampled_layer_collision_frames"]
                    for value in selected
                ),
                "frames_at_or_below_margin_4_total": sum(
                    value["metrics"]["frames_at_or_below_margin_4"]
                    for value in selected
                ),
                "frames_at_or_below_margin_8_total": sum(
                    value["metrics"]["frames_at_or_below_margin_8"]
                    for value in selected
                ),
                "frames_at_or_below_margin_12_total": sum(
                    value["metrics"]["frames_at_or_below_margin_12"]
                    for value in selected
                ),
                "frames_at_or_below_margin_20_total": sum(
                    value["metrics"]["frames_at_or_below_margin_20"]
                    for value in selected
                ),
                "frames_at_or_below_margin_4_rate": _rate(sum(
                    value["metrics"]["frames_at_or_below_margin_4"]
                    for value in selected
                ), evaluated_frame_total),
                "frames_at_or_below_margin_8_rate": _rate(sum(
                    value["metrics"]["frames_at_or_below_margin_8"]
                    for value in selected
                ), evaluated_frame_total),
                "frames_at_or_below_margin_12_rate": _rate(sum(
                    value["metrics"]["frames_at_or_below_margin_12"]
                    for value in selected
                ), evaluated_frame_total),
                "frames_at_or_below_margin_20_rate": _rate(sum(
                    value["metrics"]["frames_at_or_below_margin_20"]
                    for value in selected
                ), evaluated_frame_total),
                "direction_change_total": direction_change_total,
                "direction_changes_per_60_frames": _rate(
                    direction_change_total,
                    evaluated_frame_total,
                    60.0,
                ),
                "exact_reversal_total": sum(
                    value["metrics"]["exact_reversals"] for value in selected
                ),
                "aba_change_total": aba_change_total,
                "aba_changes_per_60_frames": _rate(
                    aba_change_total,
                    evaluated_frame_total,
                    60.0,
                ),
                "reached_goal_count": sum(value["reached_goal"] for value in selected),
            })
    beam_evaluated_frames = sum(
        sample["beam"]["metrics"]["evaluated_frames"] for sample in samples
    )
    beam_direction_changes = sum(
        sample["beam"]["metrics"]["direction_changes"] for sample in samples
    )
    beam_aba_changes = sum(
        sample["beam"]["metrics"]["aba_changes"] for sample in samples
    )
    return {
        "report": _report_label(report_path),
        "controller_profile": controller_profile,
        "grid_goal_policy": grid_goal_policy,
        "source_frames": sorted(requested),
        "replayed_decisions": replayed_decisions,
        "beam_replay_transition_count": max(0, replayed_decisions - 1),
        "beam_replay_first_action_changes": beam_first_action_changes,
        "method": {
            "comparison_scope": (
                "Open-loop plans recomputed from identical recorded delayed "
                "observations; this is not an episode survival result."
            ),
            "controller_profile": (
                "Current overrides the recorded clearance targets with 20/8; "
                "recorded retains the legacy report values 12/1."
            ),
            "continuous_validation": (
                "Every logical frame uses EngineMPC threat motion/radius and "
                "continuous Euclidean circle clearance."
            ),
            "grid_temporal_sampling": (
                "Threat centers are sampled every three frames. Candidate "
                "plans are therefore not treated as collision proofs."
            ),
            "grid_collision_level": (
                f"Collision adds {GRID_COLLISION_RISK:g} risk and is separated "
                f"from proximity by thresholds {GRID_SAFETY_THRESHOLDS}."
            ),
            "conservative_whole_cell": (
                "Inflates threat uncertainty by half the grid-cell diagonal."
            ),
            "grid_goal_policy": (
                "Survival mode passes goal=None so peak danger and accumulated "
                "risk precede movement cost. Intended region/boss goal distance "
                "is reported separately. Hard mode is diagnostic only."
            ),
            "grid_action_policy": (
                "The grid planner may change direction every three frames and "
                "has no cross-segment smoothing penalty."
            ),
            "runtime_scope": (
                "Beam time includes EngineMPC.select and may reuse a committed "
                "plan; grid time includes field rasterization and dynamic programming."
            ),
        },
        "recorded_controller_config": report["controller"]["config"],
        "effective_controller_config": asdict(controller.config),
        "beam": {
            "runtime_seconds": aggregate(beam_runtime),
            "using_committed_plan_count": sum(
                sample["beam"]["using_committed_plan"] for sample in samples
            ),
            "minimum_margin": aggregate(beam_margins),
            "collision_plan_count": sum(
                sample["beam"]["metrics"]["collision_frames"] > 0
                for sample in samples
            ),
            "complete_horizon_plan_count": sum(
                sample["beam"]["metrics"]["evaluated_frames"]
                == controller.config.horizon_frames
                for sample in samples
            ),
            "evaluated_frame_total": beam_evaluated_frames,
            "collision_frame_total": beam_collision_frames,
            "collision_frame_rate": _rate(
                beam_collision_frames,
                beam_evaluated_frames,
            ),
            "collision_event_total": sum(
                sample["beam"]["metrics"]["collision_events"] for sample in samples
            ),
            "between_layer_collision_frame_total": sum(
                sample["beam"]["metrics"]["between_layer_collision_frames"]
                for sample in samples
            ),
            "sampled_layer_collision_frame_total": sum(
                sample["beam"]["metrics"]["sampled_layer_collision_frames"]
                for sample in samples
            ),
            "frames_at_or_below_margin_4_total": sum(
                sample["beam"]["metrics"]["frames_at_or_below_margin_4"]
                for sample in samples
            ),
            "frames_at_or_below_margin_8_total": sum(
                sample["beam"]["metrics"]["frames_at_or_below_margin_8"]
                for sample in samples
            ),
            "frames_at_or_below_margin_12_total": sum(
                sample["beam"]["metrics"]["frames_at_or_below_margin_12"]
                for sample in samples
            ),
            "frames_at_or_below_margin_20_total": sum(
                sample["beam"]["metrics"]["frames_at_or_below_margin_20"]
                for sample in samples
            ),
            "frames_at_or_below_margin_4_rate": _rate(sum(
                sample["beam"]["metrics"]["frames_at_or_below_margin_4"]
                for sample in samples
            ), beam_evaluated_frames),
            "frames_at_or_below_margin_8_rate": _rate(sum(
                sample["beam"]["metrics"]["frames_at_or_below_margin_8"]
                for sample in samples
            ), beam_evaluated_frames),
            "frames_at_or_below_margin_12_rate": _rate(sum(
                sample["beam"]["metrics"]["frames_at_or_below_margin_12"]
                for sample in samples
            ), beam_evaluated_frames),
            "frames_at_or_below_margin_20_rate": _rate(sum(
                sample["beam"]["metrics"]["frames_at_or_below_margin_20"]
                for sample in samples
            ), beam_evaluated_frames),
            "direction_change_total": beam_direction_changes,
            "direction_changes_per_60_frames": _rate(
                beam_direction_changes,
                beam_evaluated_frames,
                60.0,
            ),
            "exact_reversal_total": sum(
                sample["beam"]["metrics"]["exact_reversals"] for sample in samples
            ),
            "aba_change_total": beam_aba_changes,
            "aba_changes_per_60_frames": _rate(
                beam_aba_changes,
                beam_evaluated_frames,
                60.0,
            ),
        },
        "grid_variants": variants,
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--source-frame", type=int, action="append")
    parser.add_argument("--cell-size", type=float, action="append")
    parser.add_argument(
        "--controller-profile",
        choices=("current", "recorded"),
        default="current",
        help="current applies the new 20/8 clearance targets",
    )
    parser.add_argument(
        "--grid-goal-policy",
        choices=("survival", "hard"),
        default="survival",
        help="survival keeps goal distance diagnostic instead of overriding safety",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source_frames = tuple(args.source_frame or DEFAULT_SOURCE_FRAMES)
    cell_sizes = tuple(args.cell_size or (8.0, 12.0, 16.0))
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = benchmark(
        report,
        source_frames,
        cell_sizes,
        report_path=args.report,
        controller_profile=args.controller_profile,
        grid_goal_policy=args.grid_goal_policy,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
