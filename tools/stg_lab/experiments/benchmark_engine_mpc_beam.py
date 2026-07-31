"""Benchmark exact MPC beam equivalence on recorded engine observations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

from stg_lab.engine_mpc import (
    CandidateEvaluation,
    EngineMPC,
    MPCConfig,
    RegionDynamicsMemory,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = (
    ROOT / "artifacts" / "engine-mpc-boss3-heldout-v40-d5-region-dynamics-v2.json"
)


class _UnfilteredEngineMPC(EngineMPC):
    def _beam_evaluations(self, *args, **kwargs):
        kwargs["_prefilter_threats"] = False
        return super()._beam_evaluations(*args, **kwargs)


def _controller_config(report: Mapping[str, Any]) -> MPCConfig:
    values = dict(report["controller"]["config"])
    memory = values.get("region_dynamics_memory")
    if isinstance(memory, Mapping):
        values["region_dynamics_memory"] = RegionDynamicsMemory(**memory)
    return MPCConfig(**values)


def _prepare(
    report: Mapping[str, Any],
    source_frame: int,
) -> tuple[EngineMPC, tuple[Any, ...], Mapping[str, Any]]:
    controller = EngineMPC(_controller_config(report))
    for record in report["decisions"]:
        observation = record.get("recorded_controller_input_observation")
        if not isinstance(observation, Mapping):
            continue
        frame = int(record["source_frame"])
        if frame < source_frame:
            controller.observe(observation)
            continue
        if frame != source_frame:
            break
        threats = controller.estimator.update(observation)
        observed_frame = controller.estimator.last_frame
        assert observed_frame is not None
        controller._last_source_frame = observed_frame
        controller._update_region_phase(observation, observed_frame)
        player = controller._player(observation, controller.config.observation_delay)
        bounds = controller._bounds(observation, player[2])
        boss_x = controller._boss_x(observation, controller.config.observation_delay)
        anchor = controller._region_anchor(
            player,
            bounds,
            threats,
            observed_frame,
        )
        return (
            controller,
            (player, bounds, threats, boss_x, anchor),
            record,
        )
    raise ValueError(f"source frame {source_frame} has no recorded observation")


def _selection_key(
    evaluation: CandidateEvaluation,
    index: int,
    controller: EngineMPC,
    *,
    region: bool,
) -> tuple[float, ...]:
    earliest = (
        math.inf
        if evaluation.earliest_collision_frame is None else
        float(evaluation.earliest_collision_frame)
    )
    margin_target = (
        controller.config.region_safe_margin_target
        if region else
        controller.config.safe_margin_target
    )
    return (
        float(evaluation.collided),
        -earliest,
        float(evaluation.collision_frames),
        max(0.0, margin_target - evaluation.minimum_margin),
        evaluation.boundary_penalty + evaluation.boss_alignment,
        -evaluation.minimum_margin,
        float(index),
    )


def _measure(
    controller: EngineMPC,
    arguments: tuple[Any, ...],
    *,
    prefilter: bool,
    repeat: int,
):
    durations: list[float] = []
    result = None
    for _ in range(repeat):
        started = time.perf_counter()
        result = controller._beam_evaluations(
            *arguments,
            _prefilter_threats=prefilter,
        )
        durations.append(time.perf_counter() - started)
    assert result is not None
    return result, durations


def _replay_decision_window(
    report: Mapping[str, Any],
    source_frame: int,
    *,
    warmup_frames: int,
) -> dict[str, Any]:
    config = _controller_config(report)
    optimized = EngineMPC(config)
    reference = _UnfilteredEngineMPC(config)
    start_frame = source_frame - warmup_frames
    optimized_decision = reference_decision = None
    target_record = None
    replayed = 0
    for record in report["decisions"]:
        observation = record.get("recorded_controller_input_observation")
        if not isinstance(observation, Mapping):
            continue
        frame = int(record["source_frame"])
        if frame < start_frame:
            optimized.observe(observation)
            reference.observe(observation)
            continue
        if frame > source_frame:
            break
        optimized_decision = optimized.select(observation)
        reference_decision = reference.select(observation)
        if optimized_decision != reference_decision:
            raise AssertionError(f"stateful decision changed at source frame {frame}")
        target_record = record
        replayed += 1
    if optimized_decision is None or target_record is None:
        raise ValueError(f"source frame {source_frame} was not replayed")
    if int(target_record["source_frame"]) != source_frame:
        raise ValueError(f"source frame {source_frame} has no recorded observation")
    selected = next(
        evaluation
        for evaluation in optimized_decision.evaluations
        if evaluation.action.discrete == optimized_decision.action.discrete
    )
    action_matches = (
        optimized_decision.action.to_dict() == target_record["action"]
    )
    plan_matches = (
        [action.to_dict() for action in optimized_decision.planned_actions]
        == target_record["planned_actions"]
    )
    collision_matches = (
        selected.collided == target_record["predicted_collision"]
        and selected.collision_frames == target_record["predicted_collision_frames"]
        and selected.earliest_collision_frame
        == target_record["predicted_earliest_collision_frame"]
        and selected.minimum_margin == target_record["predicted_minimum_margin"]
    )
    if not action_matches or not plan_matches or not collision_matches:
        raise AssertionError(
            f"stateful replay does not match report at source frame {source_frame}"
        )
    return {
        "warmup_frames": warmup_frames,
        "replayed_decisions": replayed,
        "optimized_equals_unfiltered": True,
        "action_matches_report": action_matches,
        "planned_actions_match_report": plan_matches,
        "collision_fields_match_report": collision_matches,
        "using_committed_plan": optimized_decision.using_committed_plan,
    }


def benchmark(
    report: Mapping[str, Any],
    source_frames: Sequence[int],
    *,
    repeat: int,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for source_frame in source_frames:
        controller, arguments, record = _prepare(report, source_frame)
        optimized, optimized_times = _measure(
            controller,
            arguments,
            prefilter=True,
            repeat=repeat,
        )
        reference, reference_times = _measure(
            controller,
            arguments,
            prefilter=False,
            repeat=repeat,
        )
        if optimized != reference:
            raise AssertionError(f"beam output changed at source frame {source_frame}")
        evaluations, plans = optimized
        region = arguments[-1] is not None
        selected = min(
            range(len(evaluations)),
            key=lambda index: _selection_key(
                evaluations[index],
                index,
                controller,
                region=region,
            ),
        )
        optimized_median = statistics.median(optimized_times)
        reference_median = statistics.median(reference_times)
        recorded_action = record["action"]
        fresh_beam_matches_record = (
            evaluations[selected].action.move_x == recorded_action["move_x"]
            and evaluations[selected].action.move_y == recorded_action["move_y"]
            and evaluations[selected].action.slow == recorded_action["slow"]
        )
        samples.append({
            "source_frame": source_frame,
            "threat_count": len(arguments[2]),
            "region_planning": region,
            "exact_beam_output_equal": True,
            "reference_seconds": reference_times,
            "optimized_seconds": optimized_times,
            "reference_median_seconds": reference_median,
            "optimized_median_seconds": optimized_median,
            "speedup": reference_median / optimized_median,
            "fresh_beam_selected_action": evaluations[selected].action.to_dict(),
            "fresh_beam_selected_action_matches_recorded_movement": (
                None
                if bool(record.get("using_committed_plan")) else
                fresh_beam_matches_record
            ),
            "recorded_action_uses_committed_plan": bool(
                record.get("using_committed_plan")
            ),
            "selected_collision": evaluations[selected].collided,
            "selected_collision_frames": evaluations[selected].collision_frames,
            "selected_earliest_collision_frame": (
                evaluations[selected].earliest_collision_frame
            ),
            "selected_minimum_margin": evaluations[selected].minimum_margin,
            "fresh_beam_selected_plan": [
                action.to_dict() for action in plans[selected]
            ],
            "stateful_report_replay": (
                _replay_decision_window(
                    report,
                    source_frame,
                    warmup_frames=60,
                )
                if bool(record.get("using_committed_plan")) else
                None
            ),
        })
    return {
        "report": str(DEFAULT_REPORT.relative_to(ROOT)),
        "repeat": repeat,
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--source-frame", type=int, action="append", dest="frames")
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    report_path = args.report.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = benchmark(report, args.frames or (995, 1292), repeat=args.repeat)
    result["report"] = str(report_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
