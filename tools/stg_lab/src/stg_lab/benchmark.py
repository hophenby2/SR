"""Parallel standalone planner benchmarks used by the CLI."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import time
from typing import Iterable

from .metrics import EpisodeMetrics
from .planning import PlannerConfig, SpatioTemporalPlanner
from .rollout import RolloutConfig, evaluate_planner, survival_rate
from .scenarios import make_environment
from .sim import SimulationConfig
from .vision import VisionConfig
from .provenance import source_tree_sha256


@dataclass(frozen=True, slots=True)
class ScenarioEnvironmentFactory:
    """Pickle-safe factory for one standalone SR scenario."""

    scenario: str
    difficulty: str
    simulation_config: SimulationConfig = SimulationConfig()

    def __call__(self, seed: int):
        return make_environment(
            self.scenario,
            difficulty=self.difficulty,
            seed=int(seed),
            config=self.simulation_config,
        )


@dataclass(frozen=True, slots=True)
class PlannerBenchmarkJob:
    scenario: str
    difficulty: str
    seed: int
    planner_config: PlannerConfig
    vision_config: VisionConfig
    rollout_config: RolloutConfig
    shield: bool
    simulation_config: SimulationConfig = SimulationConfig()


def evaluate_planner_job(job: PlannerBenchmarkJob) -> EpisodeMetrics:
    """Run one seed; this top-level function is safe for process executors."""

    factory = ScenarioEnvironmentFactory(
        job.scenario,
        job.difficulty,
        job.simulation_config,
    )
    episodes = evaluate_planner(
        factory,
        (job.seed,),
        planner=SpatioTemporalPlanner(job.planner_config),
        vision_config=job.vision_config,
        config=job.rollout_config,
        shield=job.shield,
    )
    return episodes[0]


def summarize_episodes(episodes: Iterable[EpisodeMetrics]) -> dict[str, object]:
    values = tuple(episodes)
    hashes = {episode.state_hash for episode in values if episode.state_hash is not None}
    return {
        "episode_count": len(values),
        "survived": sum(episode.survived for episode in values),
        "survival_rate": survival_rate(values),
        "unique_state_hashes": len(hashes),
        "episodes": values,
    }


def run_planner_benchmark(
    scenarios: Iterable[str],
    difficulty: str,
    seeds: Iterable[int],
    *,
    planner_config: PlannerConfig,
    vision_config: VisionConfig,
    rollout_config: RolloutConfig,
    simulation_config: SimulationConfig = SimulationConfig(),
    shield: bool = True,
    workers: int = 1,
) -> dict[str, object]:
    """Evaluate the Cartesian product of scenarios and seeds."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    scenario_names = tuple(scenarios)
    seed_values = tuple(int(seed) for seed in seeds)
    if not scenario_names or not seed_values:
        raise ValueError("at least one scenario and seed are required")
    jobs = tuple(
        PlannerBenchmarkJob(
            scenario=scenario,
            difficulty=difficulty,
            seed=seed,
            planner_config=planner_config,
            vision_config=vision_config,
            rollout_config=rollout_config,
            shield=shield,
            simulation_config=simulation_config,
        )
        for scenario in scenario_names
        for seed in seed_values
    )

    started = time.perf_counter()
    if workers == 1:
        episodes = tuple(evaluate_planner_job(job) for job in jobs)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            episodes = tuple(executor.map(evaluate_planner_job, jobs, chunksize=1))
    elapsed = time.perf_counter() - started

    per_scenario = {
        scenario: summarize_episodes(
            episode for episode in episodes if episode.scenario.startswith(scenario + ":")
        )
        for scenario in scenario_names
    }
    return {
        "implementation_sha256": source_tree_sha256(),
        "run_kind": "smoke" if rollout_config.max_frames is not None else "full_duration",
        "acceptance_claim": False,
        "elapsed_seconds": elapsed,
        "workers": workers,
        "difficulty": difficulty,
        "seeds": seed_values,
        "teacher_shield": shield,
        "planner_config": asdict(planner_config),
        "simulation_config": asdict(simulation_config),
        "vision_config": asdict(vision_config),
        "rollout_config": asdict(rollout_config),
        "overall": summarize_episodes(episodes),
        "scenarios": per_scenario,
    }


__all__ = [
    "PlannerBenchmarkJob",
    "ScenarioEnvironmentFactory",
    "evaluate_planner_job",
    "run_planner_benchmark",
    "summarize_episodes",
]
