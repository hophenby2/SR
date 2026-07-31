"""Fit phase-relative safe-region dynamics from live engine MPC artifacts."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from .provenance import file_sha256


_PHASE_ORDER = (
    "expanding",
    "maximum_hold",
    "contracting",
    "minimum_hold",
)
_ENGINE_MPC_RUN_KIND = "live_luastg_delayed_visible_mpc_teacher"
_FLOW_ROW_Y_TOLERANCE = 6.0
_FLOW_DISPLACEMENT_CONSISTENCY = 0.75
_MINIMUM_FLOW_LAG_PAIRS = 12
_FLOW_CORRELATION_THRESHOLD = 0.9
_FLOW_NORMALIZED_RMSE_THRESHOLD = 0.2
_FLOW_SAFE_SIDE_RULE = "opposite_incoming_lateral_flow"


@dataclass(frozen=True, slots=True)
class RegionDynamicsTrainingResult:
    """Strict strategy memory and its separate provenance report."""

    memory: dict[str, Any]
    report: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _Sample:
    time: float
    radius: float


@dataclass(frozen=True, slots=True)
class _FlowSample:
    time: float
    vx: float


@dataclass(frozen=True, slots=True)
class _Ramp:
    direction: str
    first_edge: int
    last_edge: int
    rate: float

    @property
    def first_sample(self) -> int:
        return self.first_edge

    @property
    def last_sample(self) -> int:
        return self.last_edge + 1


@dataclass(frozen=True, slots=True)
class _Trace:
    path: Path
    sha256: str
    scenario: str
    attack: int
    samples: tuple[_Sample, ...]
    flow_samples: tuple[_FlowSample, ...]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _clean(value: float) -> float:
    return float(round(value, 9))


def _visible_rows(
    observation: Any,
) -> list[list[tuple[Any, float, float]]]:
    if not isinstance(observation, Mapping):
        return []
    records = observation.get("indestructibles")
    if not isinstance(records, list):
        return []
    visible: list[tuple[Any, float, float]] = []
    for record in records:
        if (
            not isinstance(record, Mapping)
            or record.get("collidable", True) is not True
        ):
            continue
        identity = record.get("id")
        if isinstance(identity, bool) or not isinstance(identity, (int, str)):
            continue
        x = _number(record.get("x"))
        y = _number(record.get("y"))
        if x is None or y is None:
            continue
        visible.append((identity, x, y))
    visible.sort(key=lambda item: (item[2], item[1], str(item[0])))

    rows: list[list[tuple[Any, float, float]]] = []
    for item in visible:
        if not rows:
            rows.append([item])
            continue
        center_y = float(statistics.median(value[2] for value in rows[-1]))
        if abs(item[2] - center_y) <= _FLOW_ROW_Y_TOLERANCE:
            rows[-1].append(item)
        else:
            rows.append([item])
    return [row for row in rows if len(row) >= 3]


def _visible_top_row_flow(
    previous_observation: Any,
    observation: Any,
    elapsed: float,
) -> float | None:
    """Estimate the highest row's flow from consecutive visible positions."""

    if elapsed <= 0.0:
        return None
    previous_rows = _visible_rows(previous_observation)
    current_rows = _visible_rows(observation)
    if not previous_rows or not current_rows:
        return None
    previous = max(
        previous_rows,
        key=lambda row: statistics.median(value[2] for value in row),
    )
    current = max(
        current_rows,
        key=lambda row: statistics.median(value[2] for value in row),
    )
    previous_by_id = {value[0]: value for value in previous}
    displacements = [
        (
            (x - previous_by_id[identity][1]) / elapsed,
            (y - previous_by_id[identity][2]) / elapsed,
        )
        for identity, x, y in current
        if identity in previous_by_id
    ]
    if len(displacements) < 3:
        return None
    median_vx = float(statistics.median(value[0] for value in displacements))
    median_vy = float(statistics.median(value[1] for value in displacements))
    consistent = [
        value
        for value in displacements
        if math.hypot(value[0] - median_vx, value[1] - median_vy)
        <= _FLOW_DISPLACEMENT_CONSISTENCY
    ]
    if len(consistent) < 3 or len(consistent) < math.ceil(0.75 * len(displacements)):
        return None
    return float(statistics.median(value[0] for value in consistent))


def _read_trace(path: str | Path) -> _Trace:
    artifact_path = Path(path)
    raw = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"engine MPC artifact must be a JSON object: {artifact_path}")
    if raw.get("run_kind") != _ENGINE_MPC_RUN_KIND:
        raise ValueError(f"artifact is not a live engine MPC report: {artifact_path}")
    recorded_prefix = raw.get("recorded_prefix")
    if isinstance(recorded_prefix, Mapping) and recorded_prefix.get("enabled") is not False:
        raise ValueError(f"action-assisted artifacts cannot train region memory: {artifact_path}")
    if raw.get("policy_validation_eligible") is False:
        raise ValueError(f"policy-ineligible artifacts cannot train region memory: {artifact_path}")
    config = raw.get("config")
    if not isinstance(config, Mapping):
        raise ValueError(f"engine MPC artifact has no runner config: {artifact_path}")
    if config.get("authority_state_shield") is not False:
        raise ValueError(f"region training source used an authority shield: {artifact_path}")
    if config.get("spell_forced_off") is not True:
        raise ValueError(f"region training source does not force spell off: {artifact_path}")
    if config.get("prefix_artifact") is not None:
        raise ValueError(f"action-assisted artifacts cannot train region memory: {artifact_path}")

    scenario = raw.get("scenario")
    attack = raw.get("attack")
    if not isinstance(scenario, str) or not scenario:
        raise ValueError(f"engine MPC artifact has no scenario identity: {artifact_path}")
    if isinstance(attack, bool) or not isinstance(attack, int) or attack <= 0:
        raise ValueError(f"engine MPC artifact has no attack identity: {artifact_path}")
    decisions = raw.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError(f"engine MPC artifact has no decisions array: {artifact_path}")

    by_time: dict[float, list[float]] = {}
    flow_by_time: dict[float, list[float]] = {}
    previous_observation: Mapping[str, Any] | None = None
    previous_observation_time: float | None = None
    for decision in decisions:
        if not isinstance(decision, Mapping):
            continue
        if decision.get("control_source") != "live_mpc":
            raise ValueError(
                f"region training source contains non-live actions: {artifact_path}"
            )
        time = _number(decision.get("source_frame"))
        radius = _number(decision.get("region_observed_radius"))
        if time is None:
            continue
        if radius is not None and radius > 0.0:
            by_time.setdefault(time, []).append(radius)
        observation = decision.get("recorded_controller_input_observation")
        if isinstance(observation, Mapping):
            if (
                previous_observation is not None
                and previous_observation_time is not None
            ):
                flow = _visible_top_row_flow(
                    previous_observation,
                    observation,
                    time - previous_observation_time,
                )
                if flow is not None:
                    flow_by_time.setdefault(time, []).append(flow)
            previous_observation = observation
            previous_observation_time = time
        else:
            previous_observation = None
            previous_observation_time = None
    samples = tuple(
        _Sample(time, float(statistics.median(values)))
        for time, values in sorted(by_time.items())
    )
    flow_samples = tuple(
        _FlowSample(time, float(statistics.median(values)))
        for time, values in sorted(flow_by_time.items())
    )
    if len(samples) < 8:
        raise ValueError(
            f"engine MPC artifact has too few region radius samples: {artifact_path}"
        )
    return _Trace(
        path=artifact_path,
        sha256=file_sha256(artifact_path),
        scenario=scenario,
        attack=attack,
        samples=samples,
        flow_samples=flow_samples,
    )


def _edges(samples: Sequence[_Sample]) -> tuple[tuple[int, float], ...]:
    intervals = [
        current.time - previous.time
        for previous, current in zip(samples, samples[1:])
        if current.time > previous.time
    ]
    if not intervals:
        return ()
    cadence = float(statistics.median(intervals))
    maximum_gap = max(cadence + 1.0, 4.0 * cadence)
    result: list[tuple[int, float]] = []
    for index, (previous, current) in enumerate(zip(samples, samples[1:])):
        elapsed = current.time - previous.time
        if elapsed <= 0.0 or elapsed > maximum_gap:
            continue
        result.append((index, (current.radius - previous.radius) / elapsed))
    return tuple(result)


def _clusters(values: Sequence[float]) -> list[list[float]]:
    clusters: list[list[float]] = []
    for value in sorted(values):
        if not clusters:
            clusters.append([value])
            continue
        center = float(statistics.median(clusters[-1]))
        tolerance = max(0.02, 0.12 * max(abs(center), abs(value)))
        if abs(value - center) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return clusters


def _dominant_value(values: Sequence[float], label: str) -> float:
    finite = [value for value in values if math.isfinite(value) and value > 0.0]
    if not finite:
        raise ValueError(f"no positive {label} samples were observed")
    clusters = _clusters(finite)
    dominant = max(
        clusters,
        key=lambda cluster: (len(cluster), statistics.median(cluster)),
    )
    return float(statistics.median(dominant))


def _ramps(
    samples: Sequence[_Sample],
    edges: Sequence[tuple[int, float]],
    *,
    growth_rate: float,
    contraction_rate: float,
) -> tuple[_Ramp, ...]:
    labels: dict[int, str] = {}
    rates: dict[int, float] = {}
    for index, slope in edges:
        if slope >= 0.5 * growth_rate:
            labels[index] = "up"
            rates[index] = slope
        elif slope <= -0.5 * contraction_rate:
            labels[index] = "down"
            rates[index] = -slope

    result: list[_Ramp] = []
    ordered = sorted(labels)
    position = 0
    while position < len(ordered):
        first = ordered[position]
        direction = labels[first]
        last = first
        position += 1
        while (
            position < len(ordered)
            and ordered[position] == last + 1
            and labels[ordered[position]] == direction
        ):
            last = ordered[position]
            position += 1
        if last - first + 1 < 3:
            continue
        run_rates = [rates[index] for index in range(first, last + 1)]
        result.append(_Ramp(
            direction=direction,
            first_edge=first,
            last_edge=last,
            rate=float(statistics.median(run_rates)),
        ))
    return tuple(result)


def _next_ramp(
    ramps: Sequence[_Ramp],
    current: _Ramp,
    direction: str,
) -> _Ramp | None:
    return next(
        (
            candidate
            for candidate in ramps
            if candidate.first_edge > current.last_edge
            and candidate.direction == direction
        ),
        None,
    )


def _upper_plateau(values: Sequence[float]) -> float:
    ordered = sorted(values)
    upper_half = ordered[len(ordered) // 2:]
    return float(statistics.median(upper_half))


def _plateau_samples(
    traces: Sequence[_Trace],
    trace_ramps: Sequence[Sequence[_Ramp]],
) -> tuple[list[float], list[float]]:
    minima: list[float] = []
    maxima: list[float] = []
    for trace, ramps in zip(traces, trace_ramps):
        for ramp in ramps:
            if ramp.direction == "up":
                following = _next_ramp(ramps, ramp, "down")
                if following is None:
                    continue
                values = [
                    sample.radius
                    for sample in trace.samples[
                        ramp.last_sample:following.first_sample + 1
                    ]
                ]
                if values:
                    maxima.append(_upper_plateau(values))
            else:
                following = _next_ramp(ramps, ramp, "up")
                if following is None:
                    continue
                values = [
                    sample.radius
                    for sample in trace.samples[
                        ramp.last_sample:following.first_sample + 1
                    ]
                ]
                if values:
                    minima.append(float(statistics.median(values)))
    if not minima or not maxima:
        raise ValueError("complete minimum and maximum hold transitions are required")
    return minima, maxima


def _moving_samples(
    trace: _Trace,
    ramp: _Ramp,
    minimum_radius: float,
    maximum_radius: float,
) -> list[_Sample]:
    tolerance = 1e-6
    return [
        sample
        for sample in trace.samples[ramp.first_sample + 1:ramp.last_sample + 1]
        if minimum_radius + tolerance < sample.radius < maximum_radius - tolerance
    ]


def _expansion_start(
    trace: _Trace,
    ramp: _Ramp,
    minimum_radius: float,
    maximum_radius: float,
    growth_rate: float,
) -> float:
    moving = _moving_samples(trace, ramp, minimum_radius, maximum_radius)
    if not moving:
        raise ValueError("an expansion ramp has no interior radius samples")
    return float(statistics.median(
        sample.time - (sample.radius - minimum_radius) / growth_rate
        for sample in moving
    ))


def _contraction_end(
    trace: _Trace,
    ramp: _Ramp,
    minimum_radius: float,
    maximum_radius: float,
    contraction_rate: float,
) -> float:
    moving = _moving_samples(trace, ramp, minimum_radius, maximum_radius)
    if not moving:
        raise ValueError("a contraction ramp has no interior radius samples")
    return float(statistics.median(
        sample.time + (sample.radius - minimum_radius) / contraction_rate
        for sample in moving
    ))


def _relative_intervals(
    traces: Sequence[_Trace],
    trace_ramps: Sequence[Sequence[_Ramp]],
    *,
    minimum_radius: float,
    maximum_radius: float,
    growth_rate: float,
    contraction_rate: float,
) -> tuple[list[float], list[float]]:
    cycles: list[float] = []
    minimum_holds: list[float] = []
    for trace, ramps in zip(traces, trace_ramps):
        expansion_starts = [
            _expansion_start(
                trace,
                ramp,
                minimum_radius,
                maximum_radius,
                growth_rate,
            )
            for ramp in ramps
            if ramp.direction == "up"
        ]
        contraction_ends = [
            _contraction_end(
                trace,
                ramp,
                minimum_radius,
                maximum_radius,
                contraction_rate,
            )
            for ramp in ramps
            if ramp.direction == "down"
        ]
        cycles.extend(
            current - previous
            for previous, current in zip(expansion_starts, expansion_starts[1:])
            if current > previous
        )
        for contraction_end in contraction_ends:
            following = next(
                (start for start in expansion_starts if start > contraction_end),
                None,
            )
            if following is not None:
                minimum_holds.append(following - contraction_end)
    if not cycles:
        raise ValueError("at least two observed expansion transitions are required")
    if not minimum_holds:
        raise ValueError("no contraction-to-expansion hold interval was observed")
    return cycles, minimum_holds


def _sample_summary(values: Sequence[float]) -> dict[str, Any]:
    cleaned = sorted(_clean(value) for value in values)
    return {
        "count": len(cleaned),
        "values": cleaned,
        "median": _clean(float(statistics.median(cleaned))),
    }


def _lag_pairs(
    samples: Sequence[_FlowSample],
    lag: float,
) -> list[tuple[float, float]]:
    if len(samples) < 2 or lag <= 0.0:
        return []
    intervals = [
        current.time - previous.time
        for previous, current in zip(samples, samples[1:])
        if current.time > previous.time
    ]
    if not intervals:
        return []
    tolerance = max(1e-6, 0.51 * float(statistics.median(intervals)))
    times = [sample.time for sample in samples]
    pairs: list[tuple[float, float]] = []
    for sample in samples:
        target = sample.time + lag
        position = bisect_left(times, target)
        candidates = [
            index
            for index in (position - 1, position)
            if 0 <= index < len(samples) and samples[index].time > sample.time
        ]
        if not candidates:
            continue
        match = min(candidates, key=lambda index: abs(times[index] - target))
        if abs(times[match] - target) <= tolerance:
            pairs.append((sample.vx, samples[match].vx))
    return pairs


def _correlation(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second) or len(first) < 2:
        return 0.0
    first_mean = float(statistics.fmean(first))
    second_mean = float(statistics.fmean(second))
    first_delta = [value - first_mean for value in first]
    second_delta = [value - second_mean for value in second]
    denominator = math.sqrt(
        sum(value * value for value in first_delta)
        * sum(value * value for value in second_delta)
    )
    if denominator <= 1e-12:
        return 0.0
    return sum(
        left * right for left, right in zip(first_delta, second_delta)
    ) / denominator


def _flow_relation_metrics(
    pairs: Sequence[tuple[float, float]],
    relation: float,
) -> dict[str, Any]:
    scale = math.sqrt(float(statistics.fmean(
        0.5 * (first * first + second * second)
        for first, second in pairs
    )))
    residual = math.sqrt(float(statistics.fmean(
        (second - relation * first) ** 2
        for first, second in pairs
    )))
    correlation = _correlation(
        [first for first, _second in pairs],
        [relation * second for _first, second in pairs],
    )
    return {
        "pair_count": len(pairs),
        "normalized_rmse": _clean(residual / max(1e-12, scale)),
        "correlation": _clean(correlation),
    }


def _fit_lateral_flow(
    traces: Sequence[_Trace],
    radius_cycle_frames: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    samples = [sample for trace in traces for sample in trace.flow_samples]
    if not samples:
        raise ValueError(
            "lateral-flow fitting requires recorded controller input observations"
        )
    rms = math.sqrt(float(statistics.fmean(
        sample.vx * sample.vx for sample in samples
    )))
    if rms <= 1e-9:
        raise ValueError("visible top-row lateral flow has no measurable motion")
    sign_threshold = 0.05 * rms
    positive_count = sum(sample.vx > sign_threshold for sample in samples)
    negative_count = sum(sample.vx < -sign_threshold for sample in samples)
    if min(positive_count, negative_count) < _MINIMUM_FLOW_LAG_PAIRS // 2:
        raise ValueError(
            "visible top-row lateral flow must contain repeated motion in both directions"
        )

    maximum_span = max(
        (
            trace.flow_samples[-1].time - trace.flow_samples[0].time
            for trace in traces
            if len(trace.flow_samples) >= 2
        ),
        default=0.0,
    )
    maximum_multiple = int(math.floor(maximum_span / radius_cycle_frames + 1e-9))
    candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for multiple in range(1, maximum_multiple + 1):
        cycle = radius_cycle_frames * multiple
        repeat_pairs = [
            pair
            for trace in traces
            for pair in _lag_pairs(trace.flow_samples, cycle)
        ]
        inverse_pairs = [
            pair
            for trace in traces
            for pair in _lag_pairs(trace.flow_samples, 0.5 * cycle)
        ]
        if (
            len(repeat_pairs) < _MINIMUM_FLOW_LAG_PAIRS
            or len(inverse_pairs) < _MINIMUM_FLOW_LAG_PAIRS
        ):
            continue
        repeat = _flow_relation_metrics(repeat_pairs, 1.0)
        inverse = _flow_relation_metrics(inverse_pairs, -1.0)
        if (
            repeat["normalized_rmse"] <= _FLOW_NORMALIZED_RMSE_THRESHOLD
            and inverse["normalized_rmse"] <= _FLOW_NORMALIZED_RMSE_THRESHOLD
            and repeat["correlation"] >= _FLOW_CORRELATION_THRESHOLD
            and inverse["correlation"] >= _FLOW_CORRELATION_THRESHOLD
        ):
            candidates.append((cycle, repeat, inverse))
    if not candidates:
        raise ValueError(
            "visible top-row lateral flow does not demonstrate a repeated cycle "
            "with half-cycle sign inversion"
        )

    cycle_frames, repeat, inverse = min(candidates, key=lambda item: item[0])
    model = {
        "cycle_frames": _clean(cycle_frames),
        "safe_side_rule": _FLOW_SAFE_SIDE_RULE,
    }
    fit = {
        "observation_contract": {
            "row_selection": "highest_visible_indestructible_collision_row",
            "velocity_estimator": "consecutive_visible_position_displacement",
            "raw_velocity_fields_used": False,
            "class_or_script_timer_fields_used": False,
            "minimum_matched_row_objects": 3,
        },
        "sample_count": len(samples),
        "positive_sample_count": positive_count,
        "negative_sample_count": negative_count,
        "repeat": {
            "lag_frames": _clean(cycle_frames),
            **repeat,
        },
        "half_cycle_sign_inversion": {
            "lag_frames": _clean(0.5 * cycle_frames),
            **inverse,
        },
    }
    return model, fit


def train_region_dynamics(
    artifact_paths: Sequence[str | Path],
) -> RegionDynamicsTrainingResult:
    """Fit a strict four-phase dynamics memory from live MPC reports."""

    if not artifact_paths:
        raise ValueError("at least one engine MPC artifact is required")
    traces = tuple(_read_trace(path) for path in artifact_paths)
    scenario = traces[0].scenario
    attack = traces[0].attack
    if any(trace.scenario != scenario or trace.attack != attack for trace in traces[1:]):
        raise ValueError("all engine MPC artifacts must have the same scenario and attack")

    trace_edges = tuple(_edges(trace.samples) for trace in traces)
    positive_slopes = [
        slope
        for edges in trace_edges
        for _index, slope in edges
        if slope > 1e-9
    ]
    negative_slopes = [
        -slope
        for edges in trace_edges
        for _index, slope in edges
        if slope < -1e-9
    ]
    preliminary_growth = _dominant_value(positive_slopes, "growth-rate")
    preliminary_contraction = _dominant_value(
        negative_slopes,
        "contraction-rate",
    )
    trace_ramps = tuple(
        _ramps(
            trace.samples,
            edges,
            growth_rate=preliminary_growth,
            contraction_rate=preliminary_contraction,
        )
        for trace, edges in zip(traces, trace_edges)
    )
    growth_samples = [
        ramp.rate
        for ramps in trace_ramps
        for ramp in ramps
        if ramp.direction == "up"
    ]
    contraction_samples = [
        ramp.rate
        for ramps in trace_ramps
        for ramp in ramps
        if ramp.direction == "down"
    ]
    growth_rate = _dominant_value(growth_samples, "expansion-ramp rate")
    contraction_rate = _dominant_value(
        contraction_samples,
        "contraction-ramp rate",
    )

    minimum_samples, maximum_samples = _plateau_samples(traces, trace_ramps)
    minimum_radius = _dominant_value(minimum_samples, "minimum-radius")
    maximum_radius = _dominant_value(maximum_samples, "maximum-radius")
    if maximum_radius <= minimum_radius:
        raise ValueError("fitted maximum radius must exceed the minimum radius")

    cycle_samples, minimum_hold_samples = _relative_intervals(
        traces,
        trace_ramps,
        minimum_radius=minimum_radius,
        maximum_radius=maximum_radius,
        growth_rate=growth_rate,
        contraction_rate=contraction_rate,
    )
    cycle_frames = _dominant_value(cycle_samples, "cycle interval")
    minimum_hold_frames = _dominant_value(
        minimum_hold_samples,
        "minimum-hold interval",
    )
    radius_span = maximum_radius - minimum_radius
    expanding_frames = radius_span / growth_rate
    contracting_frames = radius_span / contraction_rate
    maximum_hold_frames = (
        cycle_frames
        - expanding_frames
        - contracting_frames
        - minimum_hold_frames
    )
    if maximum_hold_frames <= 0.0:
        raise ValueError("relative phase intervals leave no positive maximum hold")

    lateral_flow, lateral_flow_fit = _fit_lateral_flow(traces, cycle_frames)

    model = {
        "phase_order": list(_PHASE_ORDER),
        "minimum_radius": _clean(minimum_radius),
        "maximum_radius": _clean(maximum_radius),
        "growth_rate": _clean(growth_rate),
        "contraction_rate": _clean(contraction_rate),
        "phase_durations": {
            "expanding": _clean(expanding_frames),
            "maximum_hold": _clean(maximum_hold_frames),
            "contracting": _clean(contracting_frames),
            "minimum_hold": _clean(minimum_hold_frames),
        },
        "cycle_frames": _clean(cycle_frames),
        "lateral_flow": lateral_flow,
    }
    memory = {
        "schema_version": 2,
        "kind": "region_dynamics_memory",
        "scenario": scenario,
        "attack": attack,
        "model": model,
    }
    report = {
        "schema_version": 2,
        "kind": "region_dynamics_training_report",
        "scenario": scenario,
        "attack": attack,
        "inputs": [
            {
                "path": str(trace.path),
                "sha256": trace.sha256,
                "radius_sample_count": len(trace.samples),
                "lateral_flow_sample_count": len(trace.flow_samples),
            }
            for trace in traces
        ],
        "aggregate_samples": {
            "minimum_radius": _sample_summary(minimum_samples),
            "maximum_radius": _sample_summary(maximum_samples),
            "growth_rate": _sample_summary(growth_samples),
            "contraction_rate": _sample_summary(contraction_samples),
            "cycle_interval": _sample_summary(cycle_samples),
            "minimum_hold_interval": _sample_summary(minimum_hold_samples),
            "lateral_flow_fit": lateral_flow_fit,
        },
        "fitted_model": model,
    }
    return RegionDynamicsTrainingResult(memory=memory, report=report)


def write_region_dynamics_training(
    result: RegionDynamicsTrainingResult,
    *,
    memory_output: str | Path,
    report_output: str | Path,
) -> None:
    """Write strategy memory separately from training provenance."""

    memory_path = Path(memory_output)
    report_path = Path(report_output)
    if memory_path.resolve() == report_path.resolve():
        raise ValueError("memory and training report outputs must be different files")
    for path, payload in (
        (memory_path, result.memory),
        (report_path, result.report),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


__all__ = [
    "RegionDynamicsTrainingResult",
    "train_region_dynamics",
    "write_region_dynamics_training",
]
