"""Command-line entry points for STG Lab."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Sequence

import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CHECKPOINT = Path("artifacts/policy.pt")


class CLIError(RuntimeError):
    """An expected command-line usage or dependency error."""


def _scenario_names(value: str) -> tuple[str, ...]:
    from .scenarios import available_scenarios

    return available_scenarios() if value == "all" else (value,)


def _seeds(args: argparse.Namespace) -> tuple[int, ...]:
    return tuple(range(args.seed, args.seed + args.episodes))


def _environment_factory(scenario: str, args: argparse.Namespace):
    from .benchmark import ScenarioEnvironmentFactory

    return ScenarioEnvironmentFactory(scenario, args.difficulty, _simulation_config(args))


def _simulation_config(args: argparse.Namespace):
    from .sim import SimulationConfig

    return SimulationConfig(
        reaction_frames=args.motor_delay_frames,
        action_hold_frames=args.action_hold_frames,
    )


def _planner_config(args: argparse.Namespace):
    from .planning import PlannerConfig, RiskConfig

    return PlannerConfig(risk=RiskConfig(
        horizon_frames=args.planner_horizon,
        sample_every=args.planner_sample_every,
        cell_size=args.planner_cell_size,
        reaction_frames=args.planner_reaction_frames,
    ))


def _planner(args: argparse.Namespace):
    from .planning import SpatioTemporalPlanner

    return SpatioTemporalPlanner(_planner_config(args))


def _vision_config(args: argparse.Namespace):
    from .vision import VisionConfig

    return VisionConfig(
        global_width=args.global_size[0],
        global_height=args.global_size[1],
        local_width=args.local_size[0],
        local_height=args.local_size[1],
        local_extent_x=args.local_extent[0],
        local_extent_y=args.local_extent[1],
        history=args.vision_history,
        observation_delay=args.observation_delay,
    )


def _rollout_config(args: argparse.Namespace):
    from .rollout import RolloutConfig

    return RolloutConfig(
        decision_interval=args.decision_interval,
        max_frames=args.duration_frames,
        risk_scale=args.risk_scale,
        shield_horizon=args.shield_horizon,
        shield_strategy=args.shield_strategy,
    )


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def _emit_json(payload: Any, output: Path | None = None) -> None:
    rendered = json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)


def _run_pytest(arguments: Sequence[str]) -> int:
    command = [sys.executable, "-m", "pytest", *arguments]
    return subprocess.run(command, cwd=_PROJECT_ROOT, check=False).returncode


def _run_description(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "run_kind": "smoke" if args.duration_frames is not None else "full_duration",
        "acceptance_claim": False,
        "difficulty": args.difficulty,
        "seeds": _seeds(args),
        "simulation_config": _simulation_config(args),
        "vision_config": _vision_config(args),
        "rollout_config": _rollout_config(args),
    }


def _dataset_training_description() -> dict[str, Any]:
    return {"run_kind": "dataset_training", "acceptance_claim": False}


def _command_test(args: argparse.Namespace) -> int:
    pytest_args = list(args.pytest_args)
    if pytest_args[:1] == ["--"]:
        pytest_args.pop(0)
    if not args.planner:
        return _run_pytest(pytest_args)
    if pytest_args:
        raise CLIError("pytest arguments cannot be combined with --planner")

    from .benchmark import run_planner_benchmark

    result = run_planner_benchmark(
        _scenario_names(args.scenario),
        args.difficulty,
        _seeds(args),
        planner_config=_planner_config(args),
        vision_config=_vision_config(args),
        rollout_config=_rollout_config(args),
        simulation_config=_simulation_config(args),
        shield=args.teacher_shield,
        workers=args.workers,
    )
    _emit_json(result, args.output)
    return 0


def _merge_demonstrations(parts: Iterable[Any]):
    """Concatenate scenario datasets while preserving episode boundaries."""

    from .training import Demonstrations

    values = tuple(parts)
    if not values:
        raise ValueError("at least one demonstration dataset is required")
    for demonstrations in values:
        demonstrations.validate()
        if demonstrations.episode_ids is None:
            raise ValueError("collected demonstrations must include episode_ids")
    memory_presence = {demonstrations.memory is not None for demonstrations in values}
    if len(memory_presence) != 1:
        raise ValueError("cannot merge demonstration datasets with mixed memory features")

    episode_ids: list[np.ndarray] = []
    next_id = 0
    for demonstrations in values:
        source = np.asarray(demonstrations.episode_ids, dtype=np.int64)
        ordered_ids = tuple(dict.fromkeys(int(value) for value in source))
        remap = {old: next_id + index for index, old in enumerate(ordered_ids)}
        episode_ids.append(np.asarray([remap[int(value)] for value in source], dtype=np.int64))
        next_id += len(ordered_ids)

    memory = None
    if values[0].memory is not None:
        memory = np.concatenate([demonstrations.memory for demonstrations in values], axis=0)
    merged = Demonstrations(
        global_frames=np.concatenate([item.global_frames for item in values], axis=0),
        local_frames=np.concatenate([item.local_frames for item in values], axis=0),
        actions=np.concatenate([item.actions for item in values], axis=0),
        risks=np.concatenate([item.risks for item in values], axis=0),
        memory=memory,
        episode_ids=np.concatenate(episode_ids),
    )
    merged.validate()
    return merged


def _collect_demonstrations(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    from .benchmark import summarize_episodes
    from .rollout import collect_demonstrations

    datasets = []
    scenario_reports = {}
    for scenario in _scenario_names(args.scenario):
        demonstrations, episodes = collect_demonstrations(
            _environment_factory(scenario, args),
            _seeds(args),
            planner=_planner(args),
            vision_config=_vision_config(args),
            config=_rollout_config(args),
            shield=args.teacher_shield,
        )
        datasets.append(demonstrations)
        scenario_reports[scenario] = summarize_episodes(episodes)
    return _merge_demonstrations(datasets), scenario_reports


def _command_train(args: argparse.Namespace) -> int:
    from .provenance import file_sha256

    try:
        from .policy import PolicyConfig
        from .training import (
            Demonstrations,
            TrainingConfig,
            train_behavior_cloning,
            write_metrics,
        )
    except ImportError as error:
        raise CLIError("training requires the 'train' optional dependency set") from error

    collection = None
    if args.demos is not None:
        demonstrations = Demonstrations.load(args.demos)
    else:
        demonstrations, collection = _collect_demonstrations(args)
    if args.save_demos is not None:
        demonstrations.save(args.save_demos)

    channels = int(demonstrations.global_frames.shape[2])
    memory_size = (
        int(demonstrations.memory.shape[-1])
        if demonstrations.memory is not None
        else args.memory_size
    )
    policy_config = PolicyConfig(
        channels=channels,
        feature_size=args.feature_size,
        recurrent_size=args.recurrent_size,
        memory_size=memory_size,
        inference_mode=args.inference_mode,
    )
    training_config = TrainingConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction,
        risk_loss_weight=args.risk_loss_weight,
        class_balance=args.class_balance,
        device=args.device,
    )
    _, history = train_behavior_cloning(
        demonstrations,
        policy_config=policy_config,
        training_config=training_config,
        output=args.checkpoint,
        training_data={
            "path": str(args.demos or args.save_demos or "<collected-in-process>"),
            "sha256": file_sha256(args.demos or args.save_demos)
            if (args.demos or args.save_demos) is not None else None,
            "episode_ids": (
                sorted(int(value) for value in np.unique(demonstrations.episode_ids))
                if demonstrations.episode_ids is not None else None
            ),
        },
    )
    if args.metrics is not None:
        write_metrics(args.metrics, history)

    episode_groups = (
        len(np.unique(demonstrations.episode_ids))
        if demonstrations.episode_ids is not None
        else None
    )
    summary = {
        "run": _run_description(args) if collection is not None else _dataset_training_description(),
        "checkpoint": args.checkpoint,
        "demonstrations": args.demos or args.save_demos,
        "samples": int(demonstrations.actions.shape[0]),
        "sequence_length": int(demonstrations.actions.shape[1]),
        "episode_groups": episode_groups,
        "collection": collection,
        "planner_config": _planner_config(args) if collection is not None else None,
        "epochs": len(history),
        "final_metrics": history[-1] if history else None,
    }
    _emit_json(summary)
    return 0


def _load_checkpoint(path: Path, device: str) -> tuple[Any, dict[str, Any]]:
    try:
        from .training import load_checkpoint
    except ImportError as error:
        raise CLIError("checkpoint evaluation requires the 'train' optional dependency set") from error
    return load_checkpoint(path, device=device)


def _checkpoint_metadata(path: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    history = checkpoint.get("history", [])
    return {
        "checkpoint": path,
        "version": checkpoint.get("version"),
        "policy_config": checkpoint.get("policy_config", {}),
        "training_config": checkpoint.get("training_config"),
        "training_data": checkpoint.get("training_data"),
        "epochs": len(history),
        "final_training_metrics": history[-1] if history else None,
    }


def _command_evaluate_parallel(args: argparse.Namespace) -> int:
    from .benchmark import summarize_episodes
    from .policy_benchmark import PolicyScenarioFactoryConfig, run_policy_benchmark

    scenario_reports = {}
    all_episodes = []
    checkpoint_metadata = None
    elapsed_seconds = 0.0
    effective_workers = []
    for scenario in _scenario_names(args.scenario):
        benchmark = run_policy_benchmark(
            args.checkpoint,
            PolicyScenarioFactoryConfig(scenario, difficulty=args.difficulty),
            _seeds(args),
            vision_config=_vision_config(args),
            rollout_config=_rollout_config(args),
            simulation_config=_simulation_config(args),
            shield=args.shield,
            workers=args.workers,
        )
        metadata = benchmark["checkpoint_metadata"]
        if checkpoint_metadata is None:
            checkpoint_metadata = metadata
        elif metadata != checkpoint_metadata:
            raise RuntimeError("parallel scenario runs loaded inconsistent checkpoint metadata")
        scenario_report = benchmark["scenarios"][scenario]
        scenario_reports[scenario] = scenario_report
        all_episodes.extend(scenario_report["episodes"])
        elapsed_seconds += float(benchmark["elapsed_seconds"])
        effective_workers.append(int(benchmark["workers"]))

    run = _run_description(args)
    run.update({
        "device": "cpu",
        "workers": min(effective_workers),
        "requested_workers": args.workers,
    })
    result = {
        "run": run,
        "checkpoint_metadata": checkpoint_metadata,
        "shield": args.shield,
        "teacher_metrics": False,
        "planner_config": None,
        "elapsed_seconds": elapsed_seconds,
        "overall": summarize_episodes(all_episodes),
        "scenarios": scenario_reports,
    }
    _emit_json(result, args.output)
    return 0


def _command_evaluate(args: argparse.Namespace) -> int:
    if not args.checkpoint.is_file():
        raise CLIError(f"checkpoint does not exist: {args.checkpoint}")
    if args.workers > 1:
        if args.device != "cpu":
            raise CLIError("parallel policy evaluation requires --device cpu")
        if args.teacher_metrics:
            raise CLIError("parallel policy evaluation cannot use --teacher-metrics")
        if not args.metadata_only:
            return _command_evaluate_parallel(args)
    model, checkpoint = _load_checkpoint(args.checkpoint, args.device)
    metadata = _checkpoint_metadata(args.checkpoint, checkpoint)
    if args.metadata_only:
        _emit_json(metadata, args.output)
        return 0

    from .benchmark import summarize_episodes
    from .rollout import evaluate_policy

    scenario_reports = {}
    all_episodes = []
    for scenario in _scenario_names(args.scenario):
        episodes = evaluate_policy(
            model,
            _environment_factory(scenario, args),
            _seeds(args),
            planner=_planner(args) if args.teacher_metrics else None,
            vision_config=_vision_config(args),
            config=_rollout_config(args),
            device=args.device,
            shield=args.shield,
        )
        scenario_reports[scenario] = summarize_episodes(episodes)
        all_episodes.extend(episodes)
    result = {
        "run": _run_description(args),
        "checkpoint_metadata": metadata,
        "shield": args.shield,
        "teacher_metrics": args.teacher_metrics,
        "planner_config": _planner_config(args) if args.teacher_metrics else None,
        "overall": summarize_episodes(all_episodes),
        "scenarios": scenario_reports,
    }
    _emit_json(result, args.output)
    return 0


def _command_evaluate_route(args: argparse.Namespace) -> int:
    if args.scenario == "all":
        raise CLIError("external route evaluation requires one explicit scenario")
    if not args.checkpoint.is_file():
        raise CLIError(f"checkpoint does not exist: {args.checkpoint}")
    if args.shield and args.shield_strategy != "toward":
        raise CLIError("external route evaluation supports only the toward shield strategy")

    from .memory import EpisodicMemory
    from .route_benchmark import run_route_benchmark
    from .route_memory import (
        RouteControllerConfig,
        load_route_artifact,
        validate_memory_route,
    )

    artifact = load_route_artifact(args.route_artifact)
    if artifact.scenario.split(":", 1)[0] != args.scenario:
        raise CLIError("route artifact scenario does not match --scenario")
    if artifact.decision_interval != args.decision_interval:
        raise CLIError("route artifact decision interval does not match evaluation")
    with EpisodicMemory(args.memory_database, readonly=True) as store:
        memory = store.get(args.memory_id)
    validate_memory_route(artifact, memory)
    _model, checkpoint = _load_checkpoint(args.checkpoint, "cpu")
    result = run_route_benchmark(
        args.scenario,
        args.difficulty,
        _seeds(args),
        memory=memory,
        route_artifact=args.route_artifact,
        memory_database=args.memory_database,
        checkpoint=args.checkpoint,
        checkpoint_metadata=_checkpoint_metadata(args.checkpoint, checkpoint),
        route_config=RouteControllerConfig(
            shield=args.shield,
            shield_horizon=args.shield_horizon,
            exhaustion=args.route_exhaustion,
        ),
        vision_config=_vision_config(args),
        rollout_config=_rollout_config(args),
        simulation_config=_simulation_config(args),
        workers=args.workers,
    )
    _emit_json(result, args.output)
    return 0


def _command_evaluate_route_library(args: argparse.Namespace) -> int:
    if args.scenario == "all":
        raise CLIError("external route-library evaluation requires one explicit scenario")
    if not args.checkpoint.is_file():
        raise CLIError(f"checkpoint does not exist: {args.checkpoint}")
    if args.shield and args.shield_strategy != "toward":
        raise CLIError("external route-library evaluation supports only toward shield strategy")

    from .memory import EpisodicMemory
    from .route_benchmark import run_route_library_benchmark
    from .route_memory import RouteControllerConfig, load_route_library_artifact

    artifact = load_route_library_artifact(args.library_artifact)
    if artifact.scenario.split(":", 1)[0] != args.scenario:
        raise CLIError("route-library artifact scenario does not match --scenario")
    with EpisodicMemory(args.memory_database, readonly=True) as store:
        memories = tuple(store.get(memory_id) for memory_id in artifact.memory_ids)
    if any(memory.scenario != artifact.scenario for memory in memories):
        raise CLIError("route-library memory scenario does not match its artifact")
    _model, checkpoint = _load_checkpoint(args.checkpoint, "cpu")
    result = run_route_library_benchmark(
        args.scenario,
        args.difficulty,
        _seeds(args),
        memories=memories,
        library_artifact=args.library_artifact,
        memory_database=args.memory_database,
        checkpoint=args.checkpoint,
        checkpoint_metadata=_checkpoint_metadata(args.checkpoint, checkpoint),
        route_config=RouteControllerConfig(
            shield=args.shield,
            shield_horizon=args.shield_horizon,
            exhaustion=args.route_exhaustion,
            route_origin="episode",
        ),
        vision_config=_vision_config(args),
        rollout_config=_rollout_config(args),
        simulation_config=_simulation_config(args),
        workers=args.workers,
    )
    _emit_json(result, args.output)
    return 0


def _command_accept(args: argparse.Namespace) -> int:
    from .acceptance import compile_acceptance_report

    report = compile_acceptance_report(
        planner_artifacts=args.planner_artifact,
        visual_artifacts=args.visual_artifact,
        agreement_artifact=args.agreement_artifact,
        memory_artifact=args.memory_artifact,
        determinism_artifact=args.determinism_artifact,
    )
    _emit_json(report, args.output)
    return 0 if report["passed"] else 1


def _command_engine_test(args: argparse.Namespace) -> int:
    from .engine import EngineClient
    from .engine_benchmark import run_engine_benchmark
    from .protocol import Action

    with EngineClient.connect(args.host, args.port, timeout=args.timeout) as client:
        report = run_engine_benchmark(
            client,
            seed=args.seed,
            player=args.player,
            frames_per_attack=args.frames_per_attack,
            step_batch=args.step_batch,
            expected_attacks=args.expected_attacks,
            action=Action(shoot=args.shoot, slow=args.slow),
        )
    _emit_json(report, args.output)
    return 0 if report["passed"] else 1


def _command_engine_play(args: argparse.Namespace) -> int:
    from .engine import EngineClient
    from .engine_play import (
        EnginePlayConfig,
        VisualPolicyController,
        load_route_controller,
        load_route_library_controller,
        run_engine_play,
    )

    if args.route_artifact is not None:
        if args.memory_database is None or args.memory_id is None:
            raise CLIError("--route-artifact requires --memory-database and --memory-id")
        controller, metadata = load_route_controller(
            args.route_artifact,
            args.memory_database,
            args.memory_id,
            exhaustion=args.route_exhaustion,
        )
    elif args.library_artifact is not None:
        if args.memory_database is None:
            raise CLIError("--library-artifact requires --memory-database")
        if args.memory_id is not None:
            raise CLIError("--memory-id cannot be combined with --library-artifact")
        controller, metadata = load_route_library_controller(
            args.library_artifact,
            args.memory_database,
            exhaustion=args.route_exhaustion,
        )
    else:
        assert args.checkpoint is not None
        if not args.checkpoint.is_file():
            raise CLIError(f"checkpoint does not exist: {args.checkpoint}")
        model, checkpoint = _load_checkpoint(args.checkpoint, args.device)
        difficulty = args.scenario.rsplit(":", 1)[-1].lower()
        scenario_key = args.policy_scenario_key or f"stage5_boss{args.attack}:{difficulty}"
        controller = VisualPolicyController(model, scenario_key, device=args.device)
        metadata = {
            "kind": "visual_policy",
            "checkpoint": args.checkpoint,
            "checkpoint_metadata": _checkpoint_metadata(args.checkpoint, checkpoint),
        }

    config = EnginePlayConfig(
        max_frames=args.max_frames,
        vision=_vision_config(args),
        shoot_gate_radius=args.shoot_gate_radius,
        shoot_risk_threshold=args.shoot_risk_threshold,
        shoot_motion_weight=args.shoot_motion_weight,
        render=args.render,
        render_every=args.render_every,
    )
    with EngineClient.connect(args.host, args.port, timeout=args.timeout) as client:
        report = run_engine_play(
            client,
            scenario=args.scenario,
            attack=args.attack,
            seed=args.seed,
            player=args.player,
            controller=controller,
            controller_metadata=metadata,
            config=config,
        )
    _emit_json(report, args.output)
    return 0 if report["success"] else 1


def _command_engine_mpc_play(args: argparse.Namespace) -> int:
    from .engine import EngineClient
    from .engine_mpc import (
        EngineMPC,
        MPCConfig,
        load_region_dynamics_memory,
    )
    from .engine_mpc_play import EngineMPCPlayConfig, run_engine_mpc_play
    from .provenance import file_sha256

    region_memory = (
        None
        if args.region_dynamics_memory is None else
        load_region_dynamics_memory(
            args.region_dynamics_memory,
            scenario=args.scenario,
            attack=args.attack,
        )
    )

    controller = EngineMPC(MPCConfig(
        horizon_frames=args.horizon_frames,
        observation_delay=args.observation_delay,
        boundary_weight=args.boundary_weight,
        boss_alignment_weight=args.boss_alignment_weight,
        stale_track_frames=args.stale_track_frames,
        region_dynamics_memory=region_memory,
    ))
    config = EngineMPCPlayConfig(
        max_frames=args.max_frames,
        observation_delay=args.observation_delay,
        shoot_minimum_margin=args.shoot_minimum_margin,
        render=args.render,
        render_every=args.render_every,
        record_observations_from_frame=args.record_observations_from_frame,
        region_dynamics_memory_path=(
            None
            if args.region_dynamics_memory is None else
            str(args.region_dynamics_memory)
        ),
        region_dynamics_memory_sha256=(
            None
            if args.region_dynamics_memory is None else
            file_sha256(args.region_dynamics_memory)
        ),
    )
    with EngineClient.connect(args.host, args.port, timeout=args.timeout) as client:
        report = run_engine_mpc_play(
            client,
            scenario=args.scenario,
            attack=args.attack,
            seed=args.seed,
            player=args.player,
            controller=controller,
            config=config,
        )
    _emit_json(report, args.output)
    return 0 if report["passed"] else 1


def _command_train_region_dynamics(args: argparse.Namespace) -> int:
    from .region_dynamics_training import (
        train_region_dynamics,
        write_region_dynamics_training,
    )

    result = train_region_dynamics(args.artifacts)
    write_region_dynamics_training(
        result,
        memory_output=args.memory_output,
        report_output=args.report_output,
    )
    _emit_json(result.report)
    return 0


def _command_engine_train(args: argparse.Namespace) -> int:
    from .engine import EngineClient
    from .engine_play import (
        EnginePlayConfig,
        VisualPolicyController,
        load_route_controller,
        load_route_library_controller,
    )
    from .engine_training import (
        EngineTrainingConfig,
        controller_factory_from_template,
        run_engine_training,
        write_strategy_artifact,
    )
    from .provenance import file_sha256

    if args.route_artifact is not None:
        if args.memory_database is None or args.memory_id is None:
            raise CLIError("--route-artifact requires --memory-database and --memory-id")
        template, metadata = load_route_controller(
            args.route_artifact,
            args.memory_database,
            args.memory_id,
            exhaustion=args.route_exhaustion,
        )
    elif args.library_artifact is not None:
        if args.memory_database is None:
            raise CLIError("--library-artifact requires --memory-database")
        if args.memory_id is not None:
            raise CLIError("--memory-id cannot be combined with --library-artifact")
        template, metadata = load_route_library_controller(
            args.library_artifact,
            args.memory_database,
            exhaustion=args.route_exhaustion,
        )
    else:
        assert args.checkpoint is not None
        if not args.checkpoint.is_file():
            raise CLIError(f"checkpoint does not exist: {args.checkpoint}")
        model, checkpoint = _load_checkpoint(args.checkpoint, args.device)
        difficulty = args.scenario.rsplit(":", 1)[-1].lower()
        scenario_key = args.policy_scenario_key or f"stage5_boss{args.attack}:{difficulty}"
        template = VisualPolicyController(model, scenario_key, device=args.device)
        metadata = {
            "kind": "visual_policy",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "checkpoint_metadata": _checkpoint_metadata(args.checkpoint, checkpoint),
        }

    play_config = EnginePlayConfig(
        max_frames=args.max_frames,
        vision=_vision_config(args),
        shoot_gate_radius=args.shoot_gate_radius,
        shoot_risk_threshold=args.shoot_risk_threshold,
        shoot_motion_weight=args.shoot_motion_weight,
        render=False,
    )
    training_config = EngineTrainingConfig(
        train_seeds=tuple(args.train_seeds),
        heldout_seeds=tuple(args.heldout_seeds),
        candidate_count=args.candidate_count,
        search_seed=args.search_seed,
    )
    trace_directory = args.trace_directory or (
        args.output.parent / f"{args.output.stem}_traces"
    )
    with EngineClient.connect(args.host, args.port, timeout=args.timeout) as client:
        report = run_engine_training(
            client,
            scenario=args.scenario,
            attack=args.attack,
            player=args.player,
            controller_factory=controller_factory_from_template(template),
            controller_metadata=metadata,
            play_config=play_config,
            training_config=training_config,
            trace_directory=trace_directory,
        )
    artifact_sha256 = write_strategy_artifact(args.strategy_output, report)
    report["strategy_artifact"] = {
        "path": args.strategy_output,
        "sha256": artifact_sha256,
    }
    _emit_json(report, args.output)
    return 0 if report["passed"] else 1


def _command_engine_accept(args: argparse.Namespace) -> int:
    from .engine_acceptance import compare_engine_reports

    first = json.loads(args.first.read_text(encoding="utf-8"))
    second = json.loads(args.second.read_text(encoding="utf-8"))
    if not isinstance(first, dict) or not isinstance(second, dict):
        raise CLIError("engine reports must contain JSON objects")
    report = compare_engine_reports(
        first,
        second,
        expected_attacks=args.expected_attacks,
    )
    _emit_json(report, args.output)
    return 0 if report["passed"] else 1


def _add_scenario_arguments(parser: argparse.ArgumentParser) -> None:
    from .scenarios import available_scenarios

    parser.add_argument(
        "--scenario",
        choices=("all", *available_scenarios()),
        default="all",
        help="scenario to run (default: all)",
    )
    parser.add_argument(
        "--difficulty",
        choices=("normal", "lunatic"),
        default="lunatic",
    )
    parser.add_argument("--episodes", type=int, default=1, help="episodes per scenario")
    parser.add_argument("--seed", type=int, default=20260729, help="first deterministic seed")
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument(
        "--duration-frames",
        type=int,
        default=600,
        help="maximum frames per episode (default: 600-frame smoke run)",
    )
    duration.add_argument(
        "--full-duration",
        action="store_const",
        const=None,
        dest="duration_frames",
        help="run each scenario for its complete scripted duration",
    )


def _add_rollout_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--decision-interval", type=int, default=3)
    parser.add_argument("--risk-scale", type=float, default=8.0)
    parser.add_argument("--shield-horizon", type=int, default=12)
    parser.add_argument(
        "--shield-strategy",
        choices=("logits", "toward"),
        default="logits",
        help="rank safe actions by policy logits or preserve the preferred endpoint",
    )
    parser.add_argument("--vision-history", type=int, default=4)
    parser.add_argument("--observation-delay", type=int, default=5)
    parser.add_argument("--global-size", type=int, nargs=2, default=(48, 56), metavar=("W", "H"))
    parser.add_argument("--local-size", type=int, nargs=2, default=(40, 40), metavar=("W", "H"))
    parser.add_argument(
        "--local-extent",
        type=float,
        nargs=2,
        default=(72.0, 72.0),
        metavar=("X", "Y"),
    )


def _add_simulation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--motor-delay-frames",
        type=int,
        default=0,
        help="simulator input delay in frames (default: 0)",
    )
    parser.add_argument(
        "--action-hold-frames",
        type=int,
        default=1,
        help="simulator action latch duration in frames (default: 1)",
    )


def _add_planner_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--planner-horizon", type=int, default=48)
    parser.add_argument("--planner-sample-every", type=int, default=6)
    parser.add_argument("--planner-cell-size", type=float, default=20.0)
    parser.add_argument("--planner-reaction-frames", type=int, default=6)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stg-lab",
        description="Deterministic STG tests, demonstrations, training, and evaluation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    test_parser = subparsers.add_parser("test", help="run pytest, or an explicit planner benchmark")
    test_parser.add_argument("--planner", action="store_true", help="benchmark the planner instead")
    _add_scenario_arguments(test_parser)
    _add_simulation_arguments(test_parser)
    _add_rollout_arguments(test_parser)
    _add_planner_arguments(test_parser)
    test_parser.add_argument(
        "--teacher-shield",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="filter unsafe held teacher actions",
    )
    test_parser.add_argument("--workers", type=int, default=1, help="planner worker processes")
    test_parser.add_argument("--output", type=Path, help="write planner JSON to this path")
    test_parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="pytest arguments after '--' (ignored with --planner)",
    )
    test_parser.set_defaults(handler=_command_test)

    train_parser = subparsers.add_parser("train", help="train from planner demonstrations")
    _add_scenario_arguments(train_parser)
    _add_simulation_arguments(train_parser)
    _add_rollout_arguments(train_parser)
    _add_planner_arguments(train_parser)
    train_parser.add_argument(
        "--teacher-shield",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="filter unsafe held teacher actions during collection",
    )
    train_parser.add_argument("--demos", type=Path, help="load demonstrations from an .npz file")
    train_parser.add_argument("--save-demos", type=Path, help="save the effective demonstrations")
    train_parser.add_argument("--checkpoint", type=Path, default=_DEFAULT_CHECKPOINT)
    train_parser.add_argument("--metrics", type=Path, default=Path("artifacts/training_metrics.json"))
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-5)
    train_parser.add_argument("--validation-fraction", type=float, default=0.2)
    train_parser.add_argument("--risk-loss-weight", type=float, default=0.2)
    train_parser.add_argument(
        "--class-balance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train_parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    train_parser.add_argument("--feature-size", type=int, default=96)
    train_parser.add_argument("--recurrent-size", type=int, default=128)
    train_parser.add_argument("--memory-size", type=int, default=4)
    train_parser.add_argument(
        "--inference-mode",
        choices=("window", "stream"),
        default="window",
        help="window for direct decision archives; stream for recurrent sequences",
    )
    train_parser.set_defaults(handler=_command_train)

    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate a policy checkpoint")
    _add_scenario_arguments(evaluate_parser)
    _add_simulation_arguments(evaluate_parser)
    _add_rollout_arguments(evaluate_parser)
    _add_planner_arguments(evaluate_parser)
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    evaluate_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="CPU policy worker processes; values above one require --device cpu",
    )
    evaluate_parser.add_argument(
        "--shield",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable the authority-state diagnostic shield (disabled by default)",
    )
    evaluate_parser.add_argument(
        "--teacher-metrics",
        action="store_true",
        help="run the exact-state planner to attach agreement and risk metrics",
    )
    evaluate_parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="validate and inspect the checkpoint without running episodes",
    )
    evaluate_parser.add_argument("--output", type=Path, help="write evaluation JSON to this path")
    evaluate_parser.set_defaults(handler=_command_evaluate)

    route_parser = subparsers.add_parser(
        "evaluate-route",
        help="evaluate a visible-cue-triggered external route memory",
    )
    _add_scenario_arguments(route_parser)
    _add_simulation_arguments(route_parser)
    _add_rollout_arguments(route_parser)
    route_parser.add_argument("--route-artifact", type=Path, required=True)
    route_parser.add_argument("--memory-database", type=Path, required=True)
    route_parser.add_argument("--memory-id", type=int, required=True)
    route_parser.add_argument("--checkpoint", type=Path, required=True)
    route_parser.add_argument("--workers", type=int, default=1)
    route_parser.add_argument("--shield", action=argparse.BooleanOptionalAction, default=False)
    route_parser.add_argument(
        "--route-exhaustion",
        choices=("hold_last", "neutral", "error"),
        default="hold_last",
    )
    route_parser.add_argument("--output", type=Path)
    route_parser.set_defaults(handler=_command_evaluate_route)

    library_parser = subparsers.add_parser(
        "evaluate-route-library",
        help="evaluate a visible-signature-selected external route library",
    )
    _add_scenario_arguments(library_parser)
    _add_simulation_arguments(library_parser)
    _add_rollout_arguments(library_parser)
    library_parser.add_argument("--library-artifact", type=Path, required=True)
    library_parser.add_argument("--memory-database", type=Path, required=True)
    library_parser.add_argument("--checkpoint", type=Path, required=True)
    library_parser.add_argument("--workers", type=int, default=1)
    library_parser.add_argument("--shield", action=argparse.BooleanOptionalAction, default=False)
    library_parser.add_argument(
        "--route-exhaustion",
        choices=("hold_last", "neutral", "error"),
        default="hold_last",
    )
    library_parser.add_argument("--output", type=Path)
    library_parser.set_defaults(handler=_command_evaluate_route_library)

    engine_parser = subparsers.add_parser(
        "engine-test",
        help="regress every attack exposed by a live LuaSTG test bridge",
    )
    engine_parser.add_argument("--host", default="127.0.0.1")
    engine_parser.add_argument("--port", type=int, default=24816)
    engine_parser.add_argument("--timeout", type=float, default=30.0)
    engine_parser.add_argument("--seed", type=int, default=20260729)
    engine_parser.add_argument("--player", default="reimu_player")
    engine_parser.add_argument("--frames-per-attack", type=int, default=300)
    engine_parser.add_argument("--step-batch", type=int, default=1)
    engine_parser.add_argument("--expected-attacks", type=int, default=53)
    engine_parser.add_argument(
        "--shoot",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    engine_parser.add_argument(
        "--slow",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    engine_parser.add_argument("--output", type=Path)
    engine_parser.set_defaults(handler=_command_engine_test)

    engine_play_parser = subparsers.add_parser(
        "engine-play",
        help="demonstrate a delayed-visible controller in one live LuaSTG attack",
    )
    engine_play_parser.add_argument("--host", default="127.0.0.1")
    engine_play_parser.add_argument("--port", type=int, default=24816)
    engine_play_parser.add_argument("--timeout", type=float, default=30.0)
    engine_play_parser.add_argument("--scenario", default="okuu:Lunatic")
    engine_play_parser.add_argument("--attack", type=int, default=3)
    engine_play_parser.add_argument("--seed", type=int, default=20260729)
    engine_play_parser.add_argument("--player", default="reimu_player")
    engine_play_parser.add_argument("--max-frames", type=int, default=7200)
    source = engine_play_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--route-artifact", type=Path)
    source.add_argument("--library-artifact", type=Path)
    source.add_argument("--checkpoint", type=Path)
    engine_play_parser.add_argument("--memory-database", type=Path)
    engine_play_parser.add_argument("--memory-id", type=int)
    engine_play_parser.add_argument(
        "--route-exhaustion",
        choices=("hold_last", "neutral", "error"),
        default="hold_last",
    )
    engine_play_parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    engine_play_parser.add_argument("--policy-scenario-key")
    engine_play_parser.add_argument("--vision-history", type=int, default=4)
    engine_play_parser.add_argument("--observation-delay", type=int, default=5)
    engine_play_parser.add_argument(
        "--global-size", type=int, nargs=2, default=(48, 56), metavar=("W", "H"),
    )
    engine_play_parser.add_argument(
        "--local-size", type=int, nargs=2, default=(40, 40), metavar=("W", "H"),
    )
    engine_play_parser.add_argument(
        "--local-extent",
        type=float,
        nargs=2,
        default=(72.0, 72.0),
        metavar=("X", "Y"),
    )
    engine_play_parser.add_argument("--shoot-gate-radius", type=float, default=20.0)
    engine_play_parser.add_argument("--shoot-risk-threshold", type=float, default=0.25)
    engine_play_parser.add_argument("--shoot-motion-weight", type=float, default=0.5)
    engine_play_parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable RenderFunc after the attack reset has completed",
    )
    engine_play_parser.add_argument(
        "--render-every",
        type=int,
        default=1,
        help=(
            "native-engine compatibility hint; visible LuaSTG redraws every "
            "present to avoid black-buffer flicker"
        ),
    )
    engine_play_parser.add_argument("--output", type=Path)
    engine_play_parser.set_defaults(handler=_command_engine_play)

    engine_mpc_parser = subparsers.add_parser(
        "engine-mpc-play",
        help="run the delayed visible-trajectory MPC teacher in live LuaSTG",
    )
    engine_mpc_parser.add_argument("--host", default="127.0.0.1")
    engine_mpc_parser.add_argument("--port", type=int, default=24816)
    engine_mpc_parser.add_argument("--timeout", type=float, default=30.0)
    engine_mpc_parser.add_argument("--scenario", default="okuu:Lunatic")
    engine_mpc_parser.add_argument("--attack", type=int, default=3)
    engine_mpc_parser.add_argument("--seed", type=int, default=20260729)
    engine_mpc_parser.add_argument("--player", default="reimu_player")
    engine_mpc_parser.add_argument("--max-frames", type=int, default=7200)
    engine_mpc_parser.add_argument("--horizon-frames", type=int, default=36)
    engine_mpc_parser.add_argument("--observation-delay", type=int, default=5)
    engine_mpc_parser.add_argument("--boundary-weight", type=float, default=1.0)
    engine_mpc_parser.add_argument("--boss-alignment-weight", type=float, default=1.0)
    engine_mpc_parser.add_argument("--stale-track-frames", type=int, default=48)
    engine_mpc_parser.add_argument("--shoot-minimum-margin", type=float, default=12.0)
    engine_mpc_parser.add_argument(
        "--region-dynamics-memory",
        type=Path,
        help="load phase durations and safe-region dynamics without route actions",
    )
    engine_mpc_parser.add_argument(
        "--record-observations-from-frame",
        type=int,
        help="embed decision-boundary controller inputs at or after this episode frame",
    )
    engine_mpc_parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    engine_mpc_parser.add_argument(
        "--render-every",
        type=int,
        default=1,
        help=(
            "native-engine compatibility hint; visible LuaSTG redraws every "
            "present to avoid black-buffer flicker"
        ),
    )
    engine_mpc_parser.add_argument("--output", type=Path)
    engine_mpc_parser.set_defaults(handler=_command_engine_mpc_play)

    region_training_parser = subparsers.add_parser(
        "train-region-dynamics",
        help="fit phase-relative region dynamics from live engine MPC artifacts",
    )
    region_training_parser.add_argument(
        "--input",
        "--artifact",
        dest="artifacts",
        action="append",
        type=Path,
        required=True,
        help="live engine MPC JSON artifact; repeat to aggregate runs",
    )
    region_training_parser.add_argument("--memory-output", type=Path, required=True)
    region_training_parser.add_argument("--report-output", type=Path, required=True)
    region_training_parser.set_defaults(handler=_command_train_region_dynamics)

    engine_train_parser = subparsers.add_parser(
        "engine-train",
        help="search visible-controller candidates on live LuaSTG and validate held out",
    )
    engine_train_parser.add_argument("--host", default="127.0.0.1")
    engine_train_parser.add_argument("--port", type=int, default=24816)
    engine_train_parser.add_argument("--timeout", type=float, default=30.0)
    engine_train_parser.add_argument("--scenario", default="okuu:Lunatic")
    engine_train_parser.add_argument("--attack", type=int, default=3)
    engine_train_parser.add_argument("--player", default="reimu_player")
    engine_train_parser.add_argument("--max-frames", type=int, default=7200)
    train_source = engine_train_parser.add_mutually_exclusive_group(required=True)
    train_source.add_argument("--route-artifact", type=Path)
    train_source.add_argument("--library-artifact", type=Path)
    train_source.add_argument("--checkpoint", type=Path)
    engine_train_parser.add_argument("--memory-database", type=Path)
    engine_train_parser.add_argument("--memory-id", type=int)
    engine_train_parser.add_argument(
        "--route-exhaustion",
        choices=("hold_last", "neutral", "error"),
        default="hold_last",
    )
    engine_train_parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    engine_train_parser.add_argument("--policy-scenario-key")
    engine_train_parser.add_argument(
        "--train-seeds",
        type=int,
        nargs="+",
        default=(20260729, 20260730),
    )
    engine_train_parser.add_argument(
        "--heldout-seeds",
        type=int,
        nargs="+",
        default=(20260731,),
    )
    engine_train_parser.add_argument("--candidate-count", type=int, default=8)
    engine_train_parser.add_argument("--search-seed", type=int, default=20260730)
    engine_train_parser.add_argument("--vision-history", type=int, default=4)
    engine_train_parser.add_argument("--observation-delay", type=int, default=5)
    engine_train_parser.add_argument(
        "--global-size", type=int, nargs=2, default=(48, 56), metavar=("W", "H"),
    )
    engine_train_parser.add_argument(
        "--local-size", type=int, nargs=2, default=(40, 40), metavar=("W", "H"),
    )
    engine_train_parser.add_argument(
        "--local-extent",
        type=float,
        nargs=2,
        default=(72.0, 72.0),
        metavar=("X", "Y"),
    )
    engine_train_parser.add_argument("--shoot-gate-radius", type=float, default=20.0)
    engine_train_parser.add_argument("--shoot-risk-threshold", type=float, default=0.25)
    engine_train_parser.add_argument("--shoot-motion-weight", type=float, default=0.5)
    engine_train_parser.add_argument("--trace-directory", type=Path)
    engine_train_parser.add_argument("--strategy-output", type=Path, required=True)
    engine_train_parser.add_argument("--output", type=Path, required=True)
    engine_train_parser.set_defaults(handler=_command_engine_train)

    engine_accept_parser = subparsers.add_parser(
        "engine-accept",
        help="compare attack hashes from two fresh live LuaSTG processes",
    )
    engine_accept_parser.add_argument("--first", type=Path, required=True)
    engine_accept_parser.add_argument("--second", type=Path, required=True)
    engine_accept_parser.add_argument("--expected-attacks", type=int, default=53)
    engine_accept_parser.add_argument("--output", type=Path, required=True)
    engine_accept_parser.set_defaults(handler=_command_engine_accept)

    accept_parser = subparsers.add_parser(
        "accept",
        help="compile strict standalone acceptance evidence",
    )
    accept_parser.add_argument(
        "--planner-artifact",
        action="append",
        type=Path,
        required=True,
        help="planner benchmark JSON (repeat for each scenario)",
    )
    accept_parser.add_argument(
        "--visual-artifact",
        action="append",
        type=Path,
        required=True,
        help="shielded visual-policy JSON (repeat for each scenario)",
    )
    accept_parser.add_argument("--agreement-artifact", type=Path, required=True)
    accept_parser.add_argument("--memory-artifact", type=Path, required=True)
    accept_parser.add_argument("--determinism-artifact", type=Path, required=True)
    accept_parser.add_argument("--output", type=Path, required=True)
    accept_parser.set_defaults(handler=_command_accept)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    positive = (
        "episodes",
        "decision_interval",
        "action_hold_frames",
        "shield_horizon",
        "vision_history",
        "planner_horizon",
        "planner_sample_every",
        "workers",
        "epochs",
        "batch_size",
        "feature_size",
        "recurrent_size",
        "memory_size",
        "port",
        "attack",
        "max_frames",
        "candidate_count",
        "frames_per_attack",
        "step_batch",
        "expected_attacks",
    )
    for name in positive:
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if (
        hasattr(args, "duration_frames")
        and args.duration_frames is not None
        and args.duration_frames <= 0
    ):
        parser.error("--duration-frames must be positive")
    if hasattr(args, "motor_delay_frames") and args.motor_delay_frames < 0:
        parser.error("--motor-delay-frames cannot be negative")
    if hasattr(args, "validation_fraction") and not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between zero and one")
    if hasattr(args, "learning_rate") and args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")
    if hasattr(args, "timeout") and args.timeout <= 0.0:
        parser.error("--timeout must be positive")
    if hasattr(args, "memory_id") and args.memory_id is not None and args.memory_id <= 0:
        parser.error("--memory-id must be positive")
    for name in ("weight_decay", "risk_loss_weight"):
        if hasattr(args, name) and getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")
    try:
        return int(args.handler(args))
    except CLIError as error:
        print(f"stg-lab: error: {error}", file=sys.stderr)
        return 2
    except (OSError, RuntimeError, ValueError) as error:
        print(f"stg-lab: error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
