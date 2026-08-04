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
        "proficiency": getattr(args, "proficiency", "expert"),
    }


def _dataset_training_description() -> dict[str, Any]:
    return {"run_kind": "dataset_training", "acceptance_claim": False}


def _live_policy_scenario_key(
    model: Any,
    explicit: str | None,
    *,
    episode_kind: str,
    scenario: str,
    attack: int | None,
    legacy_key: str,
) -> str:
    if explicit:
        return explicit
    if getattr(model, "scenario_vocabulary", None) is None:
        return legacy_key
    from .native_dataset import episode_context_key

    return episode_context_key(episode_kind, scenario, attack)


def _scenario_context_from_manifest(
    path: Path,
) -> tuple[tuple[str, ...], int, int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CLIError(f"cannot read scenario vocabulary manifest {path}: {error}") from error
    values = payload.get("scenario_vocabulary") if isinstance(payload, dict) else None
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(value, str) and value for value in values)
    ):
        raise CLIError("scenario vocabulary manifest has no nonempty string list")
    if len(set(values)) != len(values):
        raise CLIError("scenario vocabulary entries must be unique")
    previous_action_size = payload.get("previous_action_size", 0)
    previous_action_offset = payload.get("previous_action_offset", len(values))
    if previous_action_size not in {0, 18}:
        raise CLIError("previous_action_size must be 0 or 18")
    if previous_action_offset != len(values):
        raise CLIError("previous action features must follow the scenario vocabulary")
    return tuple(values), previous_action_size, previous_action_offset


def _validate_initial_policy_context(
    model: Any,
    scenario_vocabulary: tuple[str, ...] | None,
    previous_action_size: int,
    previous_action_offset: int,
) -> None:
    """Prevent full-checkpoint continuation with permuted context semantics."""

    inherited_vocabulary = getattr(model, "scenario_vocabulary", None)
    if inherited_vocabulary is not None:
        inherited_vocabulary = tuple(inherited_vocabulary)
    inherited_size = int(getattr(model, "previous_action_size", 0))
    inherited_offset = int(getattr(model, "previous_action_offset", 0))
    if (
        inherited_vocabulary != scenario_vocabulary
        or inherited_size != previous_action_size
        or inherited_offset != previous_action_offset
    ):
        raise CLIError(
            "initial checkpoint scenario/action context metadata does not "
            "match --scenario-vocabulary-manifest"
        )


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

    from .training import Demonstrations, previous_actions_from_targets

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
    proficiency_presence = {
        demonstrations.proficiency is not None for demonstrations in values
    }
    if len(proficiency_presence) != 1:
        raise ValueError(
            "cannot merge demonstration datasets with mixed proficiency features"
        )

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
    proficiency = None
    if values[0].proficiency is not None:
        proficiency = np.concatenate(
            [demonstrations.proficiency for demonstrations in values], axis=0,
        )
    previous_actions = None
    if any(item.previous_actions is not None for item in values):
        previous_actions = np.concatenate([
            item.previous_actions
            if item.previous_actions is not None else
            previous_actions_from_targets(item)
            for item in values
        ], axis=0)
    supervision_mask = None
    if any(item.supervision_mask is not None for item in values):
        supervision_mask = np.concatenate([
            item.supervision_mask
            if item.supervision_mask is not None else
            np.ones_like(item.actions, dtype=np.bool_)
            for item in values
        ], axis=0)
    merged = Demonstrations(
        global_frames=np.concatenate([item.global_frames for item in values], axis=0),
        local_frames=np.concatenate([item.local_frames for item in values], axis=0),
        actions=np.concatenate([item.actions for item in values], axis=0),
        risks=np.concatenate([item.risks for item in values], axis=0),
        previous_actions=previous_actions,
        memory=memory,
        proficiency=proficiency,
        episode_ids=np.concatenate(episode_ids),
        supervision_mask=supervision_mask,
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
            include_scenario_memory=args.scenario_memory_conditioning,
            proficiency=args.proficiency,
        )
        datasets.append(demonstrations)
        scenario_reports[scenario] = summarize_episodes(episodes)
    return _merge_demonstrations(datasets), scenario_reports


def _command_train(args: argparse.Namespace) -> int:
    from .provenance import file_sha256

    try:
        from .policy import HumanVisionPolicy, PolicyConfig
        from .training import (
            Demonstrations,
            TrainingConfig,
            train_behavior_cloning,
            write_metrics,
        )
    except ImportError as error:
        raise CLIError("training requires the 'train' optional dependency set") from error

    if (
        not args.scenario_memory_conditioning
        and args.memory_size not in (None, 0)
    ):
        raise CLIError(
            "--no-scenario-memory-conditioning requires --memory-size 0 "
            "or no explicit --memory-size"
        )
    collection = None
    if args.demos is not None:
        demonstrations = Demonstrations.load(args.demos)
    else:
        demonstrations, collection = _collect_demonstrations(args)
    if not args.scenario_memory_conditioning:
        demonstrations.memory = None
    if args.proficiency_conditioning and demonstrations.proficiency is None:
        from .policy import proficiency_vector

        vector = proficiency_vector(args.proficiency)
        demonstrations.proficiency = np.broadcast_to(
            vector,
            (*demonstrations.actions.shape, len(vector)),
        ).copy()
    elif not args.proficiency_conditioning:
        demonstrations.proficiency = None
    if args.save_demos is not None:
        demonstrations.save(args.save_demos)

    scenario_vocabulary = None
    previous_action_size = 0
    previous_action_offset = 0
    if args.scenario_vocabulary_manifest is not None:
        if demonstrations.memory is None:
            raise CLIError(
                "--scenario-vocabulary-manifest requires demonstration memory"
            )
        (
            scenario_vocabulary,
            previous_action_size,
            previous_action_offset,
        ) = _scenario_context_from_manifest(
            args.scenario_vocabulary_manifest,
        )
        if (
            previous_action_offset + previous_action_size
            != demonstrations.memory.shape[-1]
        ):
            raise CLIError(
                "scenario/action context width does not match demonstration memory"
            )

    channels = int(demonstrations.global_frames.shape[2])
    if demonstrations.memory is not None:
        memory_size = int(demonstrations.memory.shape[-1])
        if args.memory_size is not None and args.memory_size != memory_size:
            raise CLIError(
                "--memory-size does not match the demonstration memory width "
                f"({args.memory_size} requested, {memory_size} present)"
            )
    elif not args.scenario_memory_conditioning:
        memory_size = 0
    else:
        memory_size = 4 if args.memory_size is None else args.memory_size
    policy_config = PolicyConfig(
        channels=channels,
        feature_size=args.feature_size,
        recurrent_size=args.recurrent_size,
        memory_size=memory_size,
        proficiency_size=(
            0
            if demonstrations.proficiency is None else
            int(demonstrations.proficiency.shape[-1])
        ),
        inference_mode=("stream" if args.stateful_tbptt else args.inference_mode),
    )
    training_data = {
        "path": str(args.demos or args.save_demos or "<collected-in-process>"),
        "sha256": file_sha256(args.demos or args.save_demos)
        if (args.demos or args.save_demos) is not None else None,
        "episode_ids": (
            sorted(int(value) for value in np.unique(demonstrations.episode_ids))
            if demonstrations.episode_ids is not None else None
        ),
        "scenario_memory_input": demonstrations.memory is not None,
        "scenario_vocabulary": (
            None if scenario_vocabulary is None else list(scenario_vocabulary)
        ),
        "previous_action_size": previous_action_size,
        "previous_action_offset": previous_action_offset,
        "proficiency_conditioning_input": demonstrations.proficiency is not None,
    }
    initial_model = None
    if args.init_checkpoint is not None and args.init_visual_encoder_checkpoint is not None:
        raise CLIError(
            "--init-checkpoint and --init-visual-encoder-checkpoint are mutually exclusive"
        )
    if args.init_checkpoint is not None:
        if not args.stateful_tbptt:
            raise CLIError("--init-checkpoint requires --stateful-tbptt")
        if not args.init_checkpoint.is_file():
            raise CLIError(f"initial checkpoint does not exist: {args.init_checkpoint}")
        initial_model, source_checkpoint = _load_checkpoint(args.init_checkpoint, "cpu")
        if getattr(initial_model, "config", None) != policy_config:
            raise CLIError(
                "initial checkpoint policy config does not match the requested model"
            )
        _validate_initial_policy_context(
            initial_model,
            scenario_vocabulary,
            previous_action_size,
            previous_action_offset,
        )
        training_data.update({
            "parent_checkpoint": str(args.init_checkpoint),
            "parent_checkpoint_sha256": file_sha256(args.init_checkpoint),
            "parent_checkpoint_policy_config": source_checkpoint.get(
                "policy_config", {}
            ),
            "initialization": "complete_policy_state",
        })
    if args.init_visual_encoder_checkpoint is not None:
        if not args.stateful_tbptt:
            raise CLIError(
                "--init-visual-encoder-checkpoint requires --stateful-tbptt"
            )
        if not args.init_visual_encoder_checkpoint.is_file():
            raise CLIError(
                "visual encoder checkpoint does not exist: "
                f"{args.init_visual_encoder_checkpoint}"
            )
        from .stateful_training import initialize_visual_encoders

        source_model, source_checkpoint = _load_checkpoint(
            args.init_visual_encoder_checkpoint,
            "cpu",
        )
        initial_model = HumanVisionPolicy(policy_config)
        initialize_visual_encoders(initial_model, source_model)
        training_data.update({
            "visual_encoder_parent": str(args.init_visual_encoder_checkpoint),
            "visual_encoder_parent_sha256": file_sha256(
                args.init_visual_encoder_checkpoint,
            ),
            "visual_encoder_parent_policy_config": source_checkpoint.get(
                "policy_config", {}
            ),
            "visual_encoder_transfer": ["global_encoder", "local_encoder"],
        })
    if not args.stateful_tbptt and (
        args.episode_balanced
        or args.correction_only
        or args.movement_onset_weight != 1.0
        or args.direction_change_weight != 1.0
        or args.exact_action_loss_weight != 1.0
        or args.direction_loss_weight != 0.0
        or args.speed_loss_weight != 0.0
        or args.direction_consistency_weight != 0.0
        or args.previous_action_dropout_probability != 0.0
        or args.future_visual_loss_weight != 0.0
        or tuple(args.future_visual_horizons) != (20, 40, 80)
    ):
        raise CLIError(
            "stateful loss controls require --stateful-tbptt"
        )
    if args.stateful_tbptt:
        from .stateful_training import (
            StatefulTrainingConfig,
            train_stateful_behavior_cloning,
        )

        training_config = StatefulTrainingConfig(
            seed=args.seed,
            epochs=args.epochs,
            chunk_length=args.tbptt_chunk_length,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            validation_fraction=args.validation_fraction,
            risk_loss_weight=args.risk_loss_weight,
            class_balance=args.class_balance,
            class_balance_power=args.class_balance_power,
            gradient_clip=args.gradient_clip,
            device=args.device,
            validation_episode_ids=(
                None
                if args.validation_episode_ids is None else
                tuple(args.validation_episode_ids)
            ),
            horizontal_reflection_probability=(
                args.horizontal_reflection_probability
            ),
            restore_best_validation=args.restore_best_validation,
            movement_onset_weight=args.movement_onset_weight,
            direction_change_weight=args.direction_change_weight,
            episode_balanced=args.episode_balanced,
            exact_action_loss_weight=args.exact_action_loss_weight,
            direction_loss_weight=args.direction_loss_weight,
            speed_loss_weight=args.speed_loss_weight,
            direction_consistency_weight=args.direction_consistency_weight,
            correction_only=args.correction_only,
            previous_action_dropout_probability=(
                args.previous_action_dropout_probability
            ),
            future_visual_loss_weight=args.future_visual_loss_weight,
            future_visual_horizons=tuple(args.future_visual_horizons),
        )
        stateful_arguments = {
            "policy_config": policy_config,
            "training_config": training_config,
            "output": args.checkpoint,
            "training_data": training_data,
        }
        if initial_model is not None:
            stateful_arguments["model"] = initial_model
        _, history = train_stateful_behavior_cloning(
            demonstrations,
            **stateful_arguments,
        )
    else:
        training_config = TrainingConfig(
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            validation_fraction=args.validation_fraction,
            risk_loss_weight=args.risk_loss_weight,
            class_balance=args.class_balance,
            class_balance_power=args.class_balance_power,
            device=args.device,
        )
        _, history = train_behavior_cloning(
            demonstrations,
            policy_config=policy_config,
            training_config=training_config,
            output=args.checkpoint,
            training_data=training_data,
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
        "training_mode": (
            "episode_stateful_tbptt" if args.stateful_tbptt else "window_behavior_cloning"
        ),
        "scenario_memory_input": demonstrations.memory is not None,
        "proficiency_conditioning_input": demonstrations.proficiency is not None,
        "stateful_loss_controls": (
            {
                "movement_onset_weight": args.movement_onset_weight,
                "direction_change_weight": args.direction_change_weight,
                "episode_balanced": args.episode_balanced,
                "exact_action_loss_weight": args.exact_action_loss_weight,
                "direction_loss_weight": args.direction_loss_weight,
                "speed_loss_weight": args.speed_loss_weight,
                "direction_consistency_weight": (
                    args.direction_consistency_weight
                ),
                "previous_action_dropout_probability": (
                    args.previous_action_dropout_probability
                ),
                "future_visual_loss_weight": args.future_visual_loss_weight,
                "future_visual_horizons": list(args.future_visual_horizons),
                **(
                    {
                        "correction_only": True,
                        "action_supervision": "supervision_mask",
                        "risk_supervision": "all_decisions",
                    }
                    if args.correction_only else {}
                ),
            }
            if args.stateful_tbptt else None
        ),
        "final_metrics": history[-1] if history else None,
    }
    _emit_json(summary)
    return 0


def _command_merge_demos(args: argparse.Namespace) -> int:
    from .provenance import file_sha256
    from .training import Demonstrations

    demonstrations = _merge_demonstrations(
        Demonstrations.load(path) for path in args.inputs
    )
    demonstrations.save(args.output)
    report = {
        "schema_version": 1,
        "run_kind": "demonstration_archive_merge",
        "acceptance_claim": False,
        "inputs": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in args.inputs
        ],
        "output": str(args.output),
        "output_sha256": file_sha256(args.output),
        "samples": int(demonstrations.actions.shape[0]),
        "episode_groups": int(len(np.unique(demonstrations.episode_ids))),
        "memory_input": demonstrations.memory is not None,
        "proficiency_input": demonstrations.proficiency is not None,
        "action_supervision": (
            "supervision_mask"
            if demonstrations.supervision_mask is not None else
            "all_actions"
        ),
        "supervised_labels": (
            int(demonstrations.actions.size)
            if demonstrations.supervision_mask is None else
            int(np.count_nonzero(demonstrations.supervision_mask))
        ),
    }
    _emit_json(report, args.manifest)
    return 0


def _command_contextualize_demos(args: argparse.Namespace) -> int:
    from .native_dataset import contextualize_demonstration_archive

    report = contextualize_demonstration_archive(
        args.demos,
        args.source_manifest,
        args.output,
        include_previous_action=args.previous_action_conditioning,
    )
    _emit_json(report, args.manifest)
    return 0


def _command_relabel_dagger(args: argparse.Namespace) -> int:
    from .native_dataset import relabel_dagger_demonstration_archive

    report = relabel_dagger_demonstration_archive(
        args.demos,
        args.dagger_report,
        args.output,
        args.manifest,
        interventions_only=args.interventions_only,
    )
    _emit_json(report)
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
            proficiency=args.proficiency,
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
            proficiency=args.proficiency,
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


def _command_engine_replay_analyze(args: argparse.Namespace) -> int:
    from .engine import EngineClient
    from .engine_replay_analysis import (
        EngineReplayAnalysisConfig,
        run_engine_replay_analysis,
    )

    config = EngineReplayAnalysisConfig(
        max_frames=args.max_frames,
        render=args.render,
        render_every=args.render_every,
        timeline_every=args.timeline_every,
        region_grid_cell_size=args.region_grid_cell_size,
    )
    with EngineClient.connect(args.host, args.port, timeout=args.timeout) as client:
        report = run_engine_replay_analysis(
            client,
            replay_path=args.replay,
            config=config,
        )
    _emit_json(report, args.output)
    return 0 if report["analysis_complete"] else 1


def _command_engine_play(args: argparse.Namespace) -> int:
    from .engine import EngineClient
    from .engine_play import (
        EnginePlayConfig,
        VisualPolicyController,
        load_route_controller,
        load_route_library_controller,
        run_engine_play,
    )
    from .native_dataset import (
        NativeDemonstrationBuilder,
        NativeEpisodeIdentity,
    )

    if args.stage is not None and (
        args.route_artifact is not None or args.library_artifact is not None
    ):
        raise CLIError("--stage currently requires a visual-policy checkpoint")
    episode_scenario = args.stage or args.scenario
    episode_attack = None if args.stage is not None else args.attack
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
        difficulty = (
            args.scenario.rsplit(":", 1)[-1].lower()
            if args.stage is None else
            args.stage.rsplit("@", 1)[-1].lower()
        )
        scenario_key = _live_policy_scenario_key(
            model,
            args.policy_scenario_key,
            episode_kind="attack" if args.stage is None else "stage",
            scenario=episode_scenario,
            attack=episode_attack,
            legacy_key=(
                f"stage5_boss{args.attack}:{difficulty}"
                if args.stage is None else
                f"stage:{args.stage}"
            ),
        )
        controller = VisualPolicyController(
            model,
            scenario_key,
            device=args.device,
            proficiency=args.proficiency,
            seed=args.seed,
        )
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
        visible_safety_shield=args.visible_safety_shield,
        visible_safety_horizon=args.visible_safety_horizon,
        visible_safety_minimum_margin=args.visible_safety_minimum_margin,
        render=args.render,
        render_every=args.render_every,
    )
    builder = None
    episode = None
    if args.save_demos is not None:
        if not isinstance(controller, VisualPolicyController):
            raise CLIError("--save-demos requires a visual-policy checkpoint")
        if controller.inference_mode != "stream":
            raise CLIError("--save-demos requires a streaming policy checkpoint")
        builder = NativeDemonstrationBuilder()
        episode = builder.begin(NativeEpisodeIdentity(
            episode_kind="stage" if args.stage is not None else "attack",
            scenario=episode_scenario,
            attack=episode_attack,
            seed=args.seed,
            profile=(
                f"visual_policy_{args.proficiency}_visible_safety"
                if args.visible_safety_shield else
                f"visual_policy_{args.proficiency}"
            ),
        ))
    with EngineClient.connect(args.host, args.port, timeout=args.timeout) as client:
        report = run_engine_play(
            client,
            scenario=episode_scenario,
            attack=episode_attack,
            seed=args.seed,
            player=args.player,
            controller=controller,
            controller_metadata=metadata,
            config=config,
            decision_observer=None if episode is None else episode.record,
            stage=args.stage,
        )
    if builder is not None and episode is not None:
        builder.finish(
            episode,
            strict_success=report["success"] is True,
            termination_reason=str(report["termination_reason"]),
        )
        if builder.accepted_count:
            manifest_path = args.demos_manifest or args.save_demos.with_suffix(
                ".manifest.json",
            )
            report["demonstrations"] = builder.save(
                args.save_demos,
                manifest_path=manifest_path,
            )
        else:
            report["demonstrations"] = {
                "saved": False,
                "reason": "episode did not satisfy strict native success",
            }
    _emit_json(report, args.output)
    return 0 if report["success"] else 1


def _command_engine_mpc_play(args: argparse.Namespace) -> int:
    from .engine import EngineClient
    from .engine_matrix import apply_controller_profile
    from .engine_mpc import (
        EngineMPC,
        MPCConfig,
        load_region_dynamics_memory,
    )
    from .engine_mpc_play import EngineMPCPlayConfig, run_engine_mpc_play
    from .native_dataset import (
        NativeDemonstrationBuilder,
        NativeEpisodeIdentity,
    )
    from .provenance import file_sha256
    from .vision import VisionConfig

    if args.stage is not None and args.region_dynamics_memory is not None:
        raise CLIError("--region-dynamics-memory cannot be used with --stage")
    episode_scenario = args.stage or args.scenario
    episode_attack = None if args.stage is not None else args.attack
    region_memory = (
        None
        if args.region_dynamics_memory is None else
        load_region_dynamics_memory(
            args.region_dynamics_memory,
            scenario=args.scenario,
            attack=args.attack,
        )
    )

    controller = EngineMPC(apply_controller_profile(
        args.profile,
        MPCConfig(
            horizon_frames=args.horizon_frames,
            observation_delay=args.observation_delay,
            boundary_weight=args.boundary_weight,
            boss_alignment_weight=args.boss_alignment_weight,
            stale_track_frames=args.stale_track_frames,
            region_dynamics_memory=region_memory,
            gap_prediction_enabled=args.gap_prediction,
        ),
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
        replay_name=args.replay_name,
    )
    builder = NativeDemonstrationBuilder() if args.save_demos is not None else None
    episode = (
        None
        if builder is None else
        builder.begin(NativeEpisodeIdentity(
            episode_kind="stage" if args.stage is not None else "attack",
            scenario=episode_scenario,
            attack=episode_attack,
            seed=args.seed,
            profile=args.profile,
        ))
    )
    runner_options = {}
    if episode is not None:
        runner_options = {
            "decision_observer": episode.record,
            "vision_config": VisionConfig(
                history=1,
                observation_delay=args.observation_delay,
            ),
        }
    with EngineClient.connect(args.host, args.port, timeout=args.timeout) as client:
        report = run_engine_mpc_play(
            client,
            scenario=episode_scenario,
            attack=episode_attack,
            seed=args.seed,
            player=args.player,
            controller=controller,
            config=config,
            stage=args.stage,
            **runner_options,
        )
    report["profile"] = args.profile
    if builder is not None and episode is not None:
        builder.finish(
            episode,
            strict_success=report["success"] is True,
            termination_reason=str(report["termination_reason"]),
        )
        if builder.accepted_count:
            manifest_path = args.demos_manifest or args.save_demos.with_suffix(
                ".manifest.json",
            )
            report["demonstrations"] = builder.save(
                args.save_demos,
                manifest_path=manifest_path,
            )
        else:
            report["demonstrations"] = {
                "saved": False,
                "reason": "episode did not satisfy strict native success",
            }
    _emit_json(report, args.output)
    return 0 if report["passed"] else 1


def _command_engine_mpc_campaign(args: argparse.Namespace) -> int:
    from .engine import EngineClient
    from .engine_matrix import apply_controller_profile
    from .engine_mpc import EngineMPC, MPCConfig
    from .engine_mpc_campaign import (
        EngineMPCCampaignConfig,
        run_engine_mpc_campaign,
    )

    controller = EngineMPC(apply_controller_profile(
        args.profile,
        MPCConfig(
            horizon_frames=args.horizon_frames,
            observation_delay=args.observation_delay,
            boundary_weight=args.boundary_weight,
            boss_alignment_weight=args.boss_alignment_weight,
            stale_track_frames=args.stale_track_frames,
            gap_prediction_enabled=args.gap_prediction,
            region_dynamics_memory=None,
        ),
    ))
    config = EngineMPCCampaignConfig(
        max_frames=args.max_frames,
        observation_delay=args.observation_delay,
        render=args.render,
        render_every=args.render_every,
    )
    with EngineClient.connect(args.host, args.port, timeout=args.timeout) as client:
        report = run_engine_mpc_campaign(
            client,
            difficulty=args.difficulty,
            seed=args.seed,
            player=args.player,
            controller=controller,
            config=config,
        )
    report["profile"] = args.profile
    _emit_json(report, args.output)
    return 0 if report["passed"] else 1


def _command_engine_dagger_play(args: argparse.Namespace) -> int:
    from .engine import EngineClient
    from .engine_dagger import EngineDAggerConfig, run_engine_dagger_play
    from .engine_matrix import apply_controller_profile
    from .engine_mpc import EngineMPC, MPCConfig, load_region_dynamics_memory
    from .engine_play import VisualPolicyController
    from .native_dataset import NativeDemonstrationBuilder, NativeEpisodeIdentity
    from .provenance import file_sha256

    if not args.checkpoint.is_file():
        raise CLIError(f"checkpoint does not exist: {args.checkpoint}")
    if args.stage is not None and args.region_dynamics_memory is not None:
        raise CLIError("--region-dynamics-memory cannot be used with --stage")
    model, checkpoint = _load_checkpoint(args.checkpoint, args.device)
    if getattr(getattr(model, "config", None), "inference_mode", None) != "stream":
        raise CLIError("engine-dagger-play requires a streaming policy checkpoint")
    episode_scenario = args.stage or args.scenario
    episode_attack = None if args.stage is not None else args.attack
    difficulty = (
        args.scenario.rsplit(":", 1)[-1].lower()
        if args.stage is None else
        args.stage.rsplit("@", 1)[-1].lower()
    )
    scenario_key = _live_policy_scenario_key(
        model,
        args.policy_scenario_key,
        episode_kind="attack" if args.stage is None else "stage",
        scenario=episode_scenario,
        attack=episode_attack,
        legacy_key=(
            f"stage5_boss{args.attack}:{difficulty}"
            if args.stage is None else
            f"stage:{args.stage}"
        ),
    )
    student = VisualPolicyController(
        model,
        scenario_key,
        device=args.device,
        proficiency=args.proficiency,
        seed=args.seed,
    )
    region_memory = (
        None
        if args.region_dynamics_memory is None else
        load_region_dynamics_memory(
            args.region_dynamics_memory,
            scenario=args.scenario,
            attack=args.attack,
        )
    )
    teacher = EngineMPC(apply_controller_profile(
        args.profile,
        MPCConfig(
            horizon_frames=args.horizon_frames,
            observation_delay=args.observation_delay,
            boundary_weight=args.boundary_weight,
            boss_alignment_weight=args.boss_alignment_weight,
            stale_track_frames=args.stale_track_frames,
            region_dynamics_memory=region_memory,
        ),
    ))
    config = EngineDAggerConfig(
        max_frames=args.max_frames,
        observation_delay=args.observation_delay,
        teacher_probability=args.teacher_probability,
        intervention_margin=args.intervention_margin,
        intervention_regret=args.intervention_regret,
        intervene_on_disagreement=args.intervene_on_disagreement,
        shoot_minimum_margin=args.shoot_minimum_margin,
        supervision_mode=args.supervision_mode,
        render=args.render,
        render_every=args.render_every,
    )
    builder = NativeDemonstrationBuilder()
    episode = builder.begin(NativeEpisodeIdentity(
        episode_kind="stage" if args.stage is not None else "attack",
        scenario=episode_scenario,
        attack=episode_attack,
        seed=args.seed,
        profile=(
            f"dagger_{args.profile}_beta_{args.teacher_probability:g}_"
            f"margin_{args.intervention_margin:g}"
        ),
    ))
    with EngineClient.connect(args.host, args.port, timeout=args.timeout) as client:
        report = run_engine_dagger_play(
            client,
            scenario=episode_scenario,
            attack=episode_attack,
            stage=args.stage,
            seed=args.seed,
            player=args.player,
            student=student,
            teacher=teacher,
            episode=episode,
            config=config,
            vision_config=_vision_config(args),
        )
    report["profile"] = args.profile
    report["controller"]["student"].update({
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_metadata": _checkpoint_metadata(args.checkpoint, checkpoint),
    })
    if args.region_dynamics_memory is not None:
        report["controller"]["teacher"].update({
            "region_dynamics_memory": str(args.region_dynamics_memory),
            "region_dynamics_memory_sha256": file_sha256(args.region_dynamics_memory),
        })
    builder.finish(
        episode,
        strict_success=report["success"] is True,
        termination_reason=str(report["termination_reason"]),
    )
    if builder.accepted_count:
        manifest_path = args.demos_manifest or args.save_demos.with_suffix(
            ".manifest.json"
        )
        report["demonstrations"] = builder.save(
            args.save_demos,
            manifest_path=manifest_path,
        )
    else:
        report["demonstrations"] = {
            "saved": False,
            "reason": "episode did not satisfy strict native success",
        }
    _emit_json(report, args.output)
    return 0 if report["passed"] else 1


def _command_engine_mpc_matrix(args: argparse.Namespace) -> int:
    from .engine import EngineClient
    from .engine_matrix import (
        EngineMatrixConfig,
        run_engine_matrix,
        select_catalog_targets,
    )
    from .native_dataset import NativeDemonstrationBuilder

    config = EngineMatrixConfig(
        max_frames=args.max_frames,
        horizon_frames=args.horizon_frames,
        observation_delay=args.observation_delay,
        boundary_weight=args.boundary_weight,
        boss_alignment_weight=args.boss_alignment_weight,
        stale_track_frames=args.stale_track_frames,
        shoot_minimum_margin=args.shoot_minimum_margin,
        render=args.render,
        render_every=args.render_every,
    )
    seeds = tuple(args.seeds or (20260730,))
    profiles = tuple(args.profiles or ("current",))
    builder = NativeDemonstrationBuilder() if args.save_demos is not None else None
    with EngineClient.connect(args.host, args.port, timeout=args.timeout) as client:
        targets = select_catalog_targets(
            client.catalog(),
            scenarios=tuple(args.scenarios or ()),
            attacks=tuple(args.attacks or ()),
            stages=tuple(args.stages or ()),
            all_attacks=args.all_attacks,
            all_stages=args.all_stages,
        )
        report = run_engine_matrix(
            client,
            targets=targets,
            seeds=seeds,
            profiles=profiles,
            player=args.player,
            config=config,
            trace_directory=args.trace_directory,
            demonstration_builder=builder,
        )
    if builder is not None:
        if builder.accepted_count:
            manifest_path = args.demos_manifest or args.save_demos.with_suffix(
                ".manifest.json",
            )
            report["demonstrations"] = builder.save(
                args.save_demos,
                manifest_path=manifest_path,
            )
        else:
            report["demonstrations"] = {
                "saved": False,
                "reason": "matrix contained no strictly successful episodes",
            }
    _emit_json(report, args.output)
    return 0 if report["passed"] else 1


def _command_engine_policy_matrix(args: argparse.Namespace) -> int:
    from .engine import EngineClient
    from .engine_matrix import (
        run_engine_policy_matrix,
        select_catalog_targets,
    )
    from .engine_play import EnginePlayConfig, VisualPolicyController
    from .provenance import file_sha256

    if not args.checkpoint.is_file():
        raise CLIError(f"checkpoint does not exist: {args.checkpoint}")
    model, checkpoint = _load_checkpoint(args.checkpoint, args.device)
    if getattr(getattr(model, "config", None), "inference_mode", None) != "stream":
        raise CLIError("engine-policy-matrix requires a streaming policy checkpoint")

    def controller_factory(target, proficiency: str, seed: int):
        difficulty = target.scenario.rsplit(":", 1)[-1].lower()
        scenario_key = _live_policy_scenario_key(
            model,
            None,
            episode_kind=target.episode_kind,
            scenario=target.scenario,
            attack=target.attack,
            legacy_key=(
                f"stage5_boss{target.attack}:{difficulty}"
                if target.episode_kind == "attack" else
                f"stage:{target.scenario}"
            ),
        )
        return VisualPolicyController(
            model,
            scenario_key,
            device=args.device,
            proficiency=proficiency,
            seed=seed,
        )

    config = EnginePlayConfig(
        max_frames=args.max_frames,
        vision=_vision_config(args),
        shoot_gate_radius=args.shoot_gate_radius,
        shoot_risk_threshold=args.shoot_risk_threshold,
        shoot_motion_weight=args.shoot_motion_weight,
        visible_safety_shield=args.visible_safety_shield,
        visible_safety_horizon=args.visible_safety_horizon,
        visible_safety_minimum_margin=args.visible_safety_minimum_margin,
        render=args.render,
        render_every=args.render_every,
    )
    metadata = {
        "kind": "streaming_visual_policy",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_metadata": _jsonable(
            _checkpoint_metadata(args.checkpoint, checkpoint),
        ),
    }
    seeds = tuple(args.seeds or (20260730,))
    proficiencies = tuple(args.proficiencies or ("expert",))
    with EngineClient.connect(args.host, args.port, timeout=args.timeout) as client:
        targets = select_catalog_targets(
            client.catalog(),
            scenarios=tuple(args.scenarios or ()),
            attacks=tuple(args.attacks or ()),
            stages=tuple(args.stages or ()),
            all_attacks=args.all_attacks,
            all_stages=args.all_stages,
        )
        report = run_engine_policy_matrix(
            client,
            targets=targets,
            seeds=seeds,
            proficiencies=proficiencies,
            controller_factory=controller_factory,
            controller_metadata=metadata,
            player=args.player,
            config=config,
            trace_directory=args.trace_directory,
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
        scenario_key = _live_policy_scenario_key(
            model,
            args.policy_scenario_key,
            episode_kind="attack",
            scenario=args.scenario,
            attack=args.attack,
            legacy_key=f"stage5_boss{args.attack}:{difficulty}",
        )
        template = VisualPolicyController(
            model,
            scenario_key,
            device=args.device,
            proficiency=args.proficiency,
            seed=args.search_seed,
        )
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
    from .policy import available_proficiencies

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
    parser.add_argument(
        "--proficiency",
        choices=available_proficiencies(),
        default="expert",
        help="human execution profile used by the shared recurrent policy",
    )
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
    train_parser.add_argument(
        "--validation-episode-id",
        dest="validation_episode_ids",
        type=int,
        action="append",
        help=(
            "complete episode held out for validation; repeat to provide one "
            "held-out seed per represented attack"
        ),
    )
    train_parser.add_argument("--risk-loss-weight", type=float, default=0.2)
    train_parser.add_argument(
        "--class-balance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train_parser.add_argument("--class-balance-power", type=float, default=0.5)
    train_parser.add_argument(
        "--movement-onset-weight",
        type=float,
        default=1.0,
        help=(
            "action-loss multiplier when the teacher starts moving after a "
            "stationary decision; derived only from adjacent episode actions"
        ),
    )
    train_parser.add_argument(
        "--direction-change-weight",
        type=float,
        default=1.0,
        help=(
            "action-loss multiplier when the moving teacher changes move_xy; "
            "slow-mode-only changes are excluded"
        ),
    )
    train_parser.add_argument(
        "--exact-action-loss-weight",
        type=float,
        default=1.0,
        help="weight for the original joint direction-and-speed action loss",
    )
    train_parser.add_argument(
        "--direction-loss-weight",
        type=float,
        default=0.0,
        help="additional loss on the nine movement directions, ignoring speed",
    )
    train_parser.add_argument(
        "--speed-loss-weight",
        type=float,
        default=0.0,
        help="additional loss on focused versus unfocused movement speed",
    )
    train_parser.add_argument(
        "--direction-consistency-weight",
        type=float,
        default=0.0,
        help=(
            "penalize adjacent direction-distribution changes only while the "
            "teacher direction is unchanged"
        ),
    )
    train_parser.add_argument(
        "--episode-balanced",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "give every complete training episode one optimizer update and "
            "equal loss mass regardless of its decision count"
        ),
    )
    train_parser.add_argument(
        "--stateful-tbptt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="carry GRU state across complete episode blocks and detach at TBPTT boundaries",
    )
    train_parser.add_argument(
        "--correction-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "apply stateful action loss only at true supervision_mask entries "
            "while retaining complete episodes and risk targets from every decision"
        ),
    )
    train_parser.add_argument(
        "--previous-action-dropout-probability",
        type=float,
        default=0.0,
        help=(
            "training-only probability of hiding the recorded previous executed "
            "action so the streaming GRU cannot rely on teacher-forced motor context"
        ),
    )
    train_parser.add_argument(
        "--future-visual-loss-weight",
        type=float,
        default=0.0,
        help=(
            "training-only weight for predicting detached future global/local "
            "semantic visual encodings from each streaming GRU hidden state"
        ),
    )
    train_parser.add_argument(
        "--future-visual-horizons",
        type=int,
        nargs="+",
        default=(20, 40, 80),
        metavar="DECISIONS",
        help=(
            "positive increasing within-episode decision horizons for future "
            "visual latent prediction (default: 20 40 80)"
        ),
    )
    train_parser.add_argument("--tbptt-chunk-length", type=int, default=32)
    train_parser.add_argument("--gradient-clip", type=float, default=5.0)
    train_parser.add_argument(
        "--horizontal-reflection-probability",
        type=float,
        default=0.0,
        help=(
            "per-episode training probability for a consistent horizontal "
            "mirror with x-motion and action remapping"
        ),
    )
    train_parser.add_argument(
        "--restore-best-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "restore the minimum complete-episode validation-loss epoch; disable "
            "when retaining a requested epoch for strict native outcome selection"
        ),
    )
    train_parser.add_argument(
        "--init-visual-encoder-checkpoint",
        type=Path,
        help=(
            "initialize only the global/local visible-geometry encoders; the "
            "streaming GRU and heads remain newly initialized"
        ),
    )
    train_parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help=(
            "continue stateful training from the complete policy, GRU, and heads; "
            "the requested architecture must match"
        ),
    )
    train_parser.add_argument(
        "--proficiency-conditioning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "include a proficiency vector in the network input; disable it when "
            "runtime execution limits alone should model player skill"
        ),
    )
    train_parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    train_parser.add_argument("--feature-size", type=int, default=96)
    train_parser.add_argument("--recurrent-size", type=int, default=128)
    train_parser.add_argument(
        "--memory-size",
        type=int,
        default=None,
        help=(
            "explicit context width (default: 4 with scenario conditioning, "
            "0 when --no-scenario-memory-conditioning is selected)"
        ),
    )
    train_parser.add_argument(
        "--scenario-vocabulary-manifest",
        type=Path,
        help=(
            "identity-token vocabulary produced by contextualize-demos; the "
            "width must match demonstration memory"
        ),
    )
    train_parser.add_argument(
        "--scenario-memory-conditioning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "include explicit scenario memory features; disable it to make the "
            "streaming GRU infer phase and cycle only from visible history"
        ),
    )
    train_parser.add_argument(
        "--inference-mode",
        choices=("window", "stream"),
        default="window",
        help="window for direct decision archives; stream for recurrent sequences",
    )
    train_parser.set_defaults(handler=_command_train)

    merge_parser = subparsers.add_parser(
        "merge-demos",
        help="merge episode-grouped demonstration archives without identity leakage",
    )
    merge_parser.add_argument("inputs", type=Path, nargs="+")
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.add_argument("--manifest", type=Path)
    merge_parser.set_defaults(handler=_command_merge_demos)

    context_parser = subparsers.add_parser(
        "contextualize-demos",
        help="attach identity-only one-hot context to strict native episodes",
    )
    context_parser.add_argument("--demos", type=Path, required=True)
    context_parser.add_argument("--source-manifest", type=Path, required=True)
    context_parser.add_argument("--output", type=Path, required=True)
    context_parser.add_argument("--manifest", type=Path, required=True)
    context_parser.add_argument(
        "--previous-action-conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="append the previous executed 18-way motor action as one-hot context",
    )
    context_parser.set_defaults(handler=_command_contextualize_demos)

    relabel_dagger_parser = subparsers.add_parser(
        "relabel-dagger",
        help=(
            "replace strict native DAgger teacher labels with the actions that "
            "were actually executed"
        ),
    )
    relabel_dagger_parser.add_argument("--demos", type=Path, required=True)
    relabel_dagger_parser.add_argument(
        "--dagger-report", type=Path, required=True,
    )
    relabel_dagger_parser.add_argument("--output", type=Path, required=True)
    relabel_dagger_parser.add_argument("--manifest", type=Path, required=True)
    relabel_dagger_parser.add_argument(
        "--interventions-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "mark only teacher_intervened decisions in supervision_mask while "
            "retaining the complete successful episode"
        ),
    )
    relabel_dagger_parser.set_defaults(handler=_command_relabel_dagger)

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

    replay_analysis_parser = subparsers.add_parser(
        "engine-replay-analyze",
        help="replay a native Spell Practice file and summarize live telemetry",
    )
    replay_analysis_parser.add_argument("--host", default="127.0.0.1")
    replay_analysis_parser.add_argument("--port", type=int, default=24816)
    replay_analysis_parser.add_argument("--timeout", type=float, default=30.0)
    replay_analysis_parser.add_argument(
        "--replay",
        required=True,
        help="path visible to the LuaSTG process; Windows/Wine paths are accepted",
    )
    replay_analysis_parser.add_argument("--max-frames", type=int, default=120_000)
    replay_analysis_parser.add_argument(
        "--timeline-every",
        type=int,
        default=1,
        help="retain one compact telemetry row every N replay frames",
    )
    replay_analysis_parser.add_argument(
        "--region-grid-cell-size",
        type=float,
        default=16.0,
        help="cell size for indestructible-region connectivity analysis",
    )
    replay_analysis_parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    replay_analysis_parser.add_argument("--render-every", type=int, default=1)
    replay_analysis_parser.add_argument("--output", type=Path)
    replay_analysis_parser.set_defaults(handler=_command_engine_replay_analyze)

    engine_play_parser = subparsers.add_parser(
        "engine-play",
        help="demonstrate a delayed-visible controller in a live attack or stage",
    )
    engine_play_parser.add_argument("--host", default="127.0.0.1")
    engine_play_parser.add_argument("--port", type=int, default=24816)
    engine_play_parser.add_argument("--timeout", type=float, default=30.0)
    engine_play_parser.add_argument("--scenario", default="okuu:Lunatic")
    engine_play_parser.add_argument("--attack", type=int, default=3)
    engine_play_parser.add_argument(
        "--stage",
        help="full registered stage target; success requires stage_complete and death=0",
    )
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
    from .policy import available_proficiencies

    engine_play_parser.add_argument(
        "--proficiency",
        choices=available_proficiencies(),
        default="expert",
    )
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
    engine_play_parser.add_argument(
        "--shoot-gate-radius",
        type=float,
        default=20.0,
        help="reporting-only local threat radius; firing remains continuous",
    )
    engine_play_parser.add_argument(
        "--shoot-risk-threshold",
        type=float,
        default=0.25,
        help="reporting-only local threat threshold; firing remains continuous",
    )
    engine_play_parser.add_argument(
        "--shoot-motion-weight",
        type=float,
        default=0.5,
        help="reporting-only local threat motion weight; firing remains continuous",
    )
    engine_play_parser.add_argument(
        "--visible-safety-shield",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply local semantic-raster avoidance without engine authority state",
    )
    engine_play_parser.add_argument(
        "--visible-safety-horizon",
        type=int,
        help=(
            "optional prediction-horizon cap; checkpoint policies otherwise use "
            "the selected proficiency profile"
        ),
    )
    engine_play_parser.add_argument(
        "--visible-safety-minimum-margin", type=float, default=6.0,
    )
    engine_play_parser.add_argument(
        "--save-demos",
        type=Path,
        help="save executed streaming-policy actions only after strict native success",
    )
    engine_play_parser.add_argument("--demos-manifest", type=Path)
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

    from .engine_matrix import available_engine_profiles

    engine_mpc_parser = subparsers.add_parser(
        "engine-mpc-play",
        help="run the delayed visible-trajectory MPC teacher in live LuaSTG",
    )
    engine_mpc_parser.add_argument("--host", default="127.0.0.1")
    engine_mpc_parser.add_argument("--port", type=int, default=24816)
    engine_mpc_parser.add_argument("--timeout", type=float, default=30.0)
    engine_mpc_parser.add_argument("--scenario", default="okuu:Lunatic")
    engine_mpc_parser.add_argument("--attack", type=int, default=3)
    engine_mpc_parser.add_argument(
        "--stage",
        help="full registered stage target; success requires stage_complete and death=0",
    )
    engine_mpc_parser.add_argument("--seed", type=int, default=20260729)
    engine_mpc_parser.add_argument("--player", default="reimu_player")
    engine_mpc_parser.add_argument(
        "--profile",
        choices=available_engine_profiles(),
        default="current",
        help="named live MPC parameter profile",
    )
    engine_mpc_parser.add_argument("--max-frames", type=int, default=7200)
    engine_mpc_parser.add_argument("--horizon-frames", type=int, default=36)
    engine_mpc_parser.add_argument("--observation-delay", type=int, default=5)
    engine_mpc_parser.add_argument("--boundary-weight", type=float, default=1.0)
    engine_mpc_parser.add_argument("--boss-alignment-weight", type=float, default=1.0)
    engine_mpc_parser.add_argument("--stale-track-frames", type=int, default=48)
    engine_mpc_parser.add_argument(
        "--shoot-minimum-margin",
        type=float,
        default=12.0,
        help="legacy reporting-only threshold; firing remains continuous",
    )
    engine_mpc_parser.add_argument(
        "--gap-prediction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "group parallel visible bullets and predict persistent traversable "
            "gaps for MPC navigation"
        ),
    )
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
        "--save-demos",
        type=Path,
        help="save streaming visual labels only when strict native completion succeeds",
    )
    engine_mpc_parser.add_argument(
        "--demos-manifest",
        type=Path,
        help="write native demonstration provenance (defaults beside --save-demos)",
    )
    engine_mpc_parser.add_argument(
        "--replay-name",
        help=(
            "save a verified native THlib replay under the engine replay "
            "analysis directory"
        ),
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

    engine_campaign_parser = subparsers.add_parser(
        "engine-mpc-campaign",
        help=(
            "run one continuous memory-free Stage 1-5 campaign under live MPC"
        ),
    )
    engine_campaign_parser.add_argument("--host", default="127.0.0.1")
    engine_campaign_parser.add_argument("--port", type=int, default=24816)
    engine_campaign_parser.add_argument("--timeout", type=float, default=30.0)
    engine_campaign_parser.add_argument(
        "--difficulty", choices=("Normal", "Lunatic"), default="Lunatic",
    )
    engine_campaign_parser.add_argument("--seed", type=int, default=20260729)
    engine_campaign_parser.add_argument("--player", default="reimu_player")
    engine_campaign_parser.add_argument(
        "--profile",
        choices=available_engine_profiles(),
        default="bullet-group-expert",
        help="named in-source live MPC parameter profile",
    )
    engine_campaign_parser.add_argument("--max-frames", type=int, default=120000)
    engine_campaign_parser.add_argument("--horizon-frames", type=int, default=60)
    engine_campaign_parser.add_argument("--observation-delay", type=int, default=5)
    engine_campaign_parser.add_argument("--boundary-weight", type=float, default=1.0)
    engine_campaign_parser.add_argument(
        "--boss-alignment-weight", type=float, default=1.0,
    )
    engine_campaign_parser.add_argument("--stale-track-frames", type=int, default=48)
    engine_campaign_parser.add_argument(
        "--gap-prediction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="predict traversable gaps from parallel visible bullet groups",
    )
    engine_campaign_parser.add_argument(
        "--render", action=argparse.BooleanOptionalAction, default=False,
    )
    engine_campaign_parser.add_argument("--render-every", type=int, default=1)
    engine_campaign_parser.add_argument("--output", type=Path, required=True)
    engine_campaign_parser.set_defaults(handler=_command_engine_mpc_campaign)

    engine_dagger_parser = subparsers.add_parser(
        "engine-dagger-play",
        help="collect MPC labels on policy-visited native LuaSTG trajectories",
    )
    engine_dagger_parser.add_argument("--host", default="127.0.0.1")
    engine_dagger_parser.add_argument("--port", type=int, default=24816)
    engine_dagger_parser.add_argument("--timeout", type=float, default=30.0)
    engine_dagger_parser.add_argument("--scenario", default="okuu:Lunatic")
    engine_dagger_parser.add_argument("--attack", type=int, default=3)
    engine_dagger_parser.add_argument(
        "--stage",
        help="full registered stage target; success requires stage_complete and death=0",
    )
    engine_dagger_parser.add_argument("--seed", type=int, default=20260729)
    engine_dagger_parser.add_argument("--player", default="reimu_player")
    engine_dagger_parser.add_argument("--checkpoint", type=Path, required=True)
    engine_dagger_parser.add_argument(
        "--device", choices=("cpu", "mps", "cuda"), default="cpu",
    )
    engine_dagger_parser.add_argument("--policy-scenario-key")
    engine_dagger_parser.add_argument(
        "--proficiency",
        choices=available_proficiencies(),
        default="expert",
    )
    engine_dagger_parser.add_argument(
        "--profile",
        choices=available_engine_profiles(),
        default="general",
        help="training-only MPC teacher parameter profile",
    )
    engine_dagger_parser.add_argument("--max-frames", type=int, default=7200)
    engine_dagger_parser.add_argument("--horizon-frames", type=int, default=60)
    engine_dagger_parser.add_argument("--observation-delay", type=int, default=5)
    engine_dagger_parser.add_argument("--vision-history", type=int, default=1)
    engine_dagger_parser.add_argument(
        "--global-size", type=int, nargs=2, default=(48, 56), metavar=("W", "H"),
    )
    engine_dagger_parser.add_argument(
        "--local-size", type=int, nargs=2, default=(40, 40), metavar=("W", "H"),
    )
    engine_dagger_parser.add_argument(
        "--local-extent",
        type=float,
        nargs=2,
        default=(72.0, 72.0),
        metavar=("X", "Y"),
    )
    engine_dagger_parser.add_argument("--boundary-weight", type=float, default=1.0)
    engine_dagger_parser.add_argument(
        "--boss-alignment-weight", type=float, default=1.0,
    )
    engine_dagger_parser.add_argument("--stale-track-frames", type=int, default=48)
    engine_dagger_parser.add_argument(
        "--shoot-minimum-margin",
        type=float,
        default=12.0,
        help="legacy reporting-only threshold; firing remains continuous",
    )
    engine_dagger_parser.add_argument(
        "--teacher-probability",
        type=float,
        default=0.25,
        help="scheduled teacher execution probability before safety interventions",
    )
    engine_dagger_parser.add_argument("--intervention-margin", type=float, default=12.0)
    engine_dagger_parser.add_argument("--intervention-regret", type=float, default=8.0)
    engine_dagger_parser.add_argument(
        "--intervene-on-disagreement",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "training-only hard-negative collection: execute the MPC correction "
            "whenever the student's discrete movement differs"
        ),
    )
    engine_dagger_parser.add_argument(
        "--supervision-mode",
        choices=("teacher", "corrective"),
        default="teacher",
        help=(
            "teacher labels every visited state with MPC; corrective retains "
            "safe student executions and uses MPC only on interventions"
        ),
    )
    engine_dagger_parser.add_argument(
        "--region-dynamics-memory",
        type=Path,
        help="optional phase/topology teacher memory; never exposed to the student",
    )
    engine_dagger_parser.add_argument(
        "--render", action=argparse.BooleanOptionalAction, default=False,
    )
    engine_dagger_parser.add_argument("--render-every", type=int, default=1)
    engine_dagger_parser.add_argument("--save-demos", type=Path, required=True)
    engine_dagger_parser.add_argument("--demos-manifest", type=Path)
    engine_dagger_parser.add_argument("--output", type=Path)
    engine_dagger_parser.set_defaults(handler=_command_engine_dagger_play)

    engine_matrix_parser = subparsers.add_parser(
        "engine-mpc-matrix",
        help="strictly evaluate live MPC across attacks, full stages, seeds, and profiles",
    )
    engine_matrix_parser.add_argument("--host", default="127.0.0.1")
    engine_matrix_parser.add_argument("--port", type=int, default=24816)
    engine_matrix_parser.add_argument("--timeout", type=float, default=30.0)
    engine_matrix_parser.add_argument(
        "--scenario",
        dest="scenarios",
        action="append",
        help="spell-practice scenario; repeat to select multiple",
    )
    engine_matrix_parser.add_argument(
        "--attack",
        dest="attacks",
        type=int,
        action="append",
        help="attack ordinal applied to every selected scenario",
    )
    engine_matrix_parser.add_argument(
        "--stage",
        dest="stages",
        action="append",
        help="full registered stage name, for example 'Stage 5@Lunatic'",
    )
    engine_matrix_parser.add_argument("--all-attacks", action="store_true")
    engine_matrix_parser.add_argument("--all-stages", action="store_true")
    engine_matrix_parser.add_argument(
        "--seed",
        dest="seeds",
        type=int,
        action="append",
        help="engine seed; repeat to evaluate multiple seeds",
    )
    engine_matrix_parser.add_argument(
        "--profile",
        dest="profiles",
        choices=available_engine_profiles(),
        action="append",
        help="MPC profile; repeat to compare profiles",
    )
    engine_matrix_parser.add_argument("--player", default="reimu_player")
    engine_matrix_parser.add_argument("--max-frames", type=int, default=7200)
    engine_matrix_parser.add_argument("--horizon-frames", type=int, default=60)
    engine_matrix_parser.add_argument("--observation-delay", type=int, default=5)
    engine_matrix_parser.add_argument("--boundary-weight", type=float, default=1.0)
    engine_matrix_parser.add_argument(
        "--boss-alignment-weight", type=float, default=1.0,
    )
    engine_matrix_parser.add_argument("--stale-track-frames", type=int, default=48)
    engine_matrix_parser.add_argument(
        "--shoot-minimum-margin",
        type=float,
        default=12.0,
        help="legacy reporting-only threshold; firing remains continuous",
    )
    engine_matrix_parser.add_argument(
        "--render", action=argparse.BooleanOptionalAction, default=False,
    )
    engine_matrix_parser.add_argument("--render-every", type=int, default=1)
    engine_matrix_parser.add_argument("--trace-directory", type=Path)
    engine_matrix_parser.add_argument(
        "--save-demos",
        type=Path,
        help="merge only strictly successful matrix episodes into a stream dataset",
    )
    engine_matrix_parser.add_argument(
        "--demos-manifest",
        type=Path,
        help="write native demonstration provenance (defaults beside --save-demos)",
    )
    engine_matrix_parser.add_argument("--output", type=Path, required=True)
    engine_matrix_parser.set_defaults(handler=_command_engine_mpc_matrix)

    engine_policy_matrix_parser = subparsers.add_parser(
        "engine-policy-matrix",
        help=(
            "strictly evaluate one streaming policy across native attacks, "
            "stages, seeds, and proficiency profiles"
        ),
    )
    engine_policy_matrix_parser.add_argument("--host", default="127.0.0.1")
    engine_policy_matrix_parser.add_argument("--port", type=int, default=24816)
    engine_policy_matrix_parser.add_argument("--timeout", type=float, default=30.0)
    engine_policy_matrix_parser.add_argument("--checkpoint", type=Path, required=True)
    engine_policy_matrix_parser.add_argument(
        "--device", choices=("cpu", "mps", "cuda"), default="cpu",
    )
    engine_policy_matrix_parser.add_argument(
        "--scenario",
        dest="scenarios",
        action="append",
        help="spell-practice scenario; repeat to select multiple",
    )
    engine_policy_matrix_parser.add_argument(
        "--attack",
        dest="attacks",
        type=int,
        action="append",
        help="attack ordinal applied to every selected scenario",
    )
    engine_policy_matrix_parser.add_argument(
        "--stage",
        dest="stages",
        action="append",
        help="full registered stage name, for example 'Stage 1@Normal'",
    )
    engine_policy_matrix_parser.add_argument("--all-attacks", action="store_true")
    engine_policy_matrix_parser.add_argument("--all-stages", action="store_true")
    engine_policy_matrix_parser.add_argument(
        "--seed", dest="seeds", type=int, action="append",
    )
    engine_policy_matrix_parser.add_argument(
        "--proficiency",
        dest="proficiencies",
        choices=available_proficiencies(),
        action="append",
    )
    engine_policy_matrix_parser.add_argument("--player", default="reimu_player")
    engine_policy_matrix_parser.add_argument("--max-frames", type=int, default=7200)
    engine_policy_matrix_parser.add_argument("--vision-history", type=int, default=1)
    engine_policy_matrix_parser.add_argument("--observation-delay", type=int, default=5)
    engine_policy_matrix_parser.add_argument(
        "--global-size", type=int, nargs=2, default=(48, 56), metavar=("W", "H"),
    )
    engine_policy_matrix_parser.add_argument(
        "--local-size", type=int, nargs=2, default=(40, 40), metavar=("W", "H"),
    )
    engine_policy_matrix_parser.add_argument(
        "--local-extent",
        type=float,
        nargs=2,
        default=(72.0, 72.0),
        metavar=("X", "Y"),
    )
    engine_policy_matrix_parser.add_argument(
        "--shoot-gate-radius",
        type=float,
        default=20.0,
        help="reporting-only local threat radius; firing remains continuous",
    )
    engine_policy_matrix_parser.add_argument(
        "--shoot-risk-threshold",
        type=float,
        default=0.25,
        help="reporting-only local threat threshold; firing remains continuous",
    )
    engine_policy_matrix_parser.add_argument(
        "--shoot-motion-weight",
        type=float,
        default=0.5,
        help="reporting-only local threat motion weight; firing remains continuous",
    )
    engine_policy_matrix_parser.add_argument(
        "--visible-safety-shield",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    engine_policy_matrix_parser.add_argument("--visible-safety-horizon", type=int)
    engine_policy_matrix_parser.add_argument(
        "--visible-safety-minimum-margin", type=float, default=6.0,
    )
    engine_policy_matrix_parser.add_argument(
        "--render", action=argparse.BooleanOptionalAction, default=False,
    )
    engine_policy_matrix_parser.add_argument("--render-every", type=int, default=1)
    engine_policy_matrix_parser.add_argument("--trace-directory", type=Path)
    engine_policy_matrix_parser.add_argument("--output", type=Path, required=True)
    engine_policy_matrix_parser.set_defaults(handler=_command_engine_policy_matrix)

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
        "--proficiency",
        choices=available_proficiencies(),
        default="expert",
    )
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
    engine_train_parser.add_argument(
        "--shoot-gate-radius",
        type=float,
        default=20.0,
        help="reporting-only local threat radius; firing remains continuous",
    )
    engine_train_parser.add_argument(
        "--shoot-risk-threshold",
        type=float,
        default=0.25,
        help="reporting-only local threat threshold; firing remains continuous",
    )
    engine_train_parser.add_argument(
        "--shoot-motion-weight",
        type=float,
        default=0.5,
        help="reporting-only local threat motion weight; firing remains continuous",
    )
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
        "tbptt_chunk_length",
        "movement_onset_weight",
        "direction_change_weight",
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
        hasattr(args, "memory_size")
        and args.memory_size is not None
        and args.memory_size < 0
    ):
        parser.error("--memory-size cannot be negative")
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
    for name in (
        "weight_decay",
        "risk_loss_weight",
        "exact_action_loss_weight",
        "direction_loss_weight",
        "speed_loss_weight",
        "direction_consistency_weight",
        "future_visual_loss_weight",
    ):
        if hasattr(args, name) and getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")
    if hasattr(args, "future_visual_horizons"):
        horizons = tuple(args.future_visual_horizons)
        if (
            any(horizon <= 0 for horizon in horizons)
            or len(set(horizons)) != len(horizons)
            or tuple(sorted(horizons)) != horizons
        ):
            parser.error(
                "--future-visual-horizons must be positive and strictly increasing"
            )
    if hasattr(args, "gradient_clip") and args.gradient_clip <= 0.0:
        parser.error("--gradient-clip must be positive")
    if hasattr(args, "class_balance_power") and not 0.0 <= args.class_balance_power <= 1.0:
        parser.error("--class-balance-power must be between zero and one")
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
