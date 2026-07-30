"""Parallel evaluation for external episodic routes."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import time
from typing import Any, Iterable

from .benchmark import summarize_episodes
from .memory import EpisodeMemory
from .rollout import RolloutConfig, _run_episode
from .route_memory import (
    ExternalRouteController,
    ExternalRouteLibraryController,
    RouteControllerConfig,
)
from .scenarios import make_environment
from .sim import SimulationConfig
from .vision import VisionConfig
from .provenance import file_sha256, source_tree_sha256


@dataclass(frozen=True, slots=True)
class RouteBenchmarkJob:
    scenario: str
    difficulty: str
    seed: int
    memory: EpisodeMemory
    route_config: RouteControllerConfig
    vision_config: VisionConfig
    rollout_config: RolloutConfig
    simulation_config: SimulationConfig


@dataclass(frozen=True, slots=True)
class RouteLibraryBenchmarkJob:
    scenario: str
    difficulty: str
    seed: int
    memories: tuple[EpisodeMemory, ...]
    route_config: RouteControllerConfig
    vision_config: VisionConfig
    rollout_config: RolloutConfig
    simulation_config: SimulationConfig


def evaluate_route_job(job: RouteBenchmarkJob):
    controller = ExternalRouteController(job.memory, config=job.route_config)
    callback = controller if job.route_config.shield else (
        lambda _environment, visible, _plan, _memory: controller.select(visible)
    )
    trace = _run_episode(
        lambda seed: make_environment(
            job.scenario,
            difficulty=job.difficulty,
            seed=seed,
            config=job.simulation_config,
            duration_frames=job.rollout_config.max_frames,
        ),
        job.seed,
        planner=None,
        vision_config=job.vision_config,
        config=job.rollout_config,
        controller=callback,
    )
    return replace(trace.metrics, teacher_overrides=controller.overrides), {
        "seed": job.seed,
        "triggered": controller.triggered,
        "decision": controller.trigger_decision,
        "source_frame": controller.trigger_source_frame,
    }


def evaluate_route_library_job(job: RouteLibraryBenchmarkJob):
    controller = ExternalRouteLibraryController(job.memories, config=job.route_config)
    callback = controller if job.route_config.shield else (
        lambda _environment, visible, _plan, _memory: controller.select(visible)
    )
    trace = _run_episode(
        lambda seed: make_environment(
            job.scenario,
            difficulty=job.difficulty,
            seed=seed,
            config=job.simulation_config,
            duration_frames=job.rollout_config.max_frames,
        ),
        job.seed,
        planner=None,
        vision_config=job.vision_config,
        config=job.rollout_config,
        controller=callback,
    )
    metrics = replace(trace.metrics, teacher_overrides=controller.overrides)
    selected = controller.selected_memory.id if controller.selected_memory is not None else None
    return metrics, {
        "seed": job.seed,
        "memory_id": selected,
        "decision": controller.selection_decision,
        "source_frame": controller.selection_source_frame,
    }


def run_route_benchmark(
    scenario: str,
    difficulty: str,
    seeds: Iterable[int],
    *,
    memory: EpisodeMemory,
    route_artifact: str | Path,
    memory_database: str | Path,
    checkpoint: str | Path,
    checkpoint_metadata: dict[str, Any],
    route_config: RouteControllerConfig = RouteControllerConfig(),
    vision_config: VisionConfig = VisionConfig(),
    rollout_config: RolloutConfig = RolloutConfig(),
    simulation_config: SimulationConfig = SimulationConfig(),
    workers: int = 1,
) -> dict[str, Any]:
    seed_values = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if not seed_values:
        raise ValueError("at least one seed is required")
    if workers <= 0:
        raise ValueError("workers must be positive")
    jobs = tuple(RouteBenchmarkJob(
        scenario, difficulty, seed, memory, route_config,
        vision_config, rollout_config, simulation_config,
    ) for seed in seed_values)
    effective_workers = min(workers, len(jobs))
    started = time.perf_counter()
    if effective_workers == 1:
        outputs = tuple(evaluate_route_job(job) for job in jobs)
    else:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            outputs = tuple(executor.map(evaluate_route_job, jobs, chunksize=1))
    episodes = tuple(value[0] for value in outputs)
    triggers = tuple(value[1] for value in outputs)
    checkpoint_path = Path(checkpoint)
    report_metadata = {
        "checkpoint": str(checkpoint_path),
        "sha256": file_sha256(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
        **checkpoint_metadata,
        "role": "system_checkpoint_reference",
        "policy_actions_used": False,
    }
    summary = summarize_episodes(episodes)
    return {
        "implementation_sha256": source_tree_sha256(),
        "run": {
            "run_kind": "smoke" if rollout_config.max_frames is not None else "full_duration",
            "acceptance_claim": False,
            "device": "cpu",
            "difficulty": difficulty,
            "seeds": seed_values,
            "workers": effective_workers,
            "requested_workers": workers,
            "vision_config": asdict(vision_config),
            "rollout_config": asdict(rollout_config),
            "simulation_config": asdict(simulation_config),
        },
        "controller_kind": "external_route_memory",
        "controller_stack": (
            "delayed_semantic_cue",
            "sqlite_external_route",
        ) + (("toward_collision_shield",) if route_config.shield else ()),
        "checkpoint_metadata": report_metadata,
        "route_memory": {
            "artifact": str(route_artifact),
            "artifact_sha256": file_sha256(route_artifact),
            "database": str(memory_database),
            "database_sha256": file_sha256(memory_database),
            "database_read_only": True,
            "memory_id": memory.id,
            "route_actions": len(memory.route),
            "cue": memory.cue,
            "config": asdict(route_config),
            "triggers": triggers,
            "untriggered_episodes": sum(item["triggered"] is not True for item in triggers),
        },
        "shield": route_config.shield,
        "teacher_metrics": False,
        "planner_config": None,
        "authority_state_used": bool(route_config.shield),
        "online_visible_cue": True,
        "elapsed_seconds": time.perf_counter() - started,
        "overall": summary,
        "scenarios": {scenario: summary},
    }


def run_route_library_benchmark(
    scenario: str,
    difficulty: str,
    seeds: Iterable[int],
    *,
    memories: Iterable[EpisodeMemory],
    library_artifact: str | Path,
    memory_database: str | Path,
    checkpoint: str | Path,
    checkpoint_metadata: dict[str, Any],
    route_config: RouteControllerConfig = RouteControllerConfig(shield=False),
    vision_config: VisionConfig = VisionConfig(),
    rollout_config: RolloutConfig = RolloutConfig(),
    simulation_config: SimulationConfig = SimulationConfig(),
    workers: int = 1,
) -> dict[str, Any]:
    seed_values = tuple(dict.fromkeys(int(seed) for seed in seeds))
    memory_values = tuple(memories)
    if not seed_values or not memory_values:
        raise ValueError("route library benchmark requires seeds and memories")
    if workers <= 0:
        raise ValueError("workers must be positive")
    jobs = tuple(RouteLibraryBenchmarkJob(
        scenario, difficulty, seed, memory_values, route_config,
        vision_config, rollout_config, simulation_config,
    ) for seed in seed_values)
    effective_workers = min(workers, len(jobs))
    started = time.perf_counter()
    if effective_workers == 1:
        outputs = tuple(evaluate_route_library_job(job) for job in jobs)
    else:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            outputs = tuple(executor.map(evaluate_route_library_job, jobs, chunksize=1))
    episodes = tuple(value[0] for value in outputs)
    selections = tuple(value[1] for value in outputs)
    selected_ids = tuple(value["memory_id"] for value in selections)
    selection_counts = {
        str(memory_id): selected_ids.count(memory_id)
        for memory_id in sorted({value for value in selected_ids if value is not None})
    }
    checkpoint_path = Path(checkpoint)
    report_metadata = {
        "checkpoint": str(checkpoint_path),
        "sha256": file_sha256(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
        **checkpoint_metadata,
        "role": "system_checkpoint_reference",
        "policy_actions_used": False,
    }
    summary = summarize_episodes(episodes)
    return {
        "implementation_sha256": source_tree_sha256(),
        "run": {
            "run_kind": "smoke" if rollout_config.max_frames is not None else "full_duration",
            "acceptance_claim": False,
            "device": "cpu",
            "difficulty": difficulty,
            "seeds": seed_values,
            "workers": effective_workers,
            "requested_workers": workers,
            "vision_config": asdict(vision_config),
            "rollout_config": asdict(rollout_config),
            "simulation_config": asdict(simulation_config),
        },
        "controller_kind": "external_route_library_memory",
        "controller_stack": (
            "delayed_semantic_signature",
            "sqlite_external_route_library",
        ) + (("toward_collision_shield",) if route_config.shield else ()),
        "checkpoint_metadata": report_metadata,
        "route_memory": {
            "artifact": str(library_artifact),
            "artifact_sha256": file_sha256(library_artifact),
            "database": str(memory_database),
            "database_sha256": file_sha256(memory_database),
            "database_read_only": True,
            "memory_ids": [memory.id for memory in memory_values],
            "route_actions": [len(memory.route) for memory in memory_values],
            "config": asdict(route_config),
            "selection_counts": selection_counts,
            "unselected_episodes": selected_ids.count(None),
            "selections": selections,
        },
        "shield": route_config.shield,
        "teacher_metrics": False,
        "planner_config": None,
        "authority_state_used": bool(route_config.shield),
        "online_visible_cue": True,
        "elapsed_seconds": time.perf_counter() - started,
        "overall": summary,
        "scenarios": {scenario: summary},
    }


__all__ = [
    "RouteBenchmarkJob",
    "RouteLibraryBenchmarkJob",
    "evaluate_route_job",
    "evaluate_route_library_job",
    "file_sha256",
    "run_route_benchmark",
    "run_route_library_benchmark",
]
