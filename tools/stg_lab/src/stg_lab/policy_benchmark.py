"""Spawn-safe parallel visual-policy evaluation.

Seeds are split into one stable chunk per worker.  A worker loads the policy
checkpoint once, then evaluates every scenario chunk assigned to it serially.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

from .benchmark import summarize_episodes
from .metrics import EpisodeMetrics
from .policy import PlayerProficiencyProfile, resolve_proficiency
from .rollout import RolloutConfig, evaluate_policy
from .scenarios import make_environment
from .sim import SimulationConfig
from .training import load_checkpoint
from .vision import VisionConfig
from .provenance import source_tree_sha256


@dataclass(frozen=True, slots=True)
class PolicyScenarioFactoryConfig:
    """Pickle-safe description of one standalone scenario factory."""

    scenario: str
    difficulty: str = "lunatic"
    duration_frames: int | None = None

    def __post_init__(self) -> None:
        if not self.scenario.strip():
            raise ValueError("scenario cannot be empty")
        if not self.difficulty.strip():
            raise ValueError("difficulty cannot be empty")
        if self.duration_frames is not None and self.duration_frames <= 0:
            raise ValueError("scenario duration_frames must be positive or None")


@dataclass(frozen=True, slots=True)
class PolicyEnvironmentFactory:
    """Concrete environment factory reconstructed inside a worker."""

    scenario: PolicyScenarioFactoryConfig
    simulation_config: SimulationConfig

    def __call__(self, seed: int):
        return make_environment(
            self.scenario.scenario,
            difficulty=self.scenario.difficulty,
            seed=int(seed),
            config=self.simulation_config,
            duration_frames=self.scenario.duration_frames,
        )


@dataclass(frozen=True, slots=True)
class PolicyScenarioSeedChunk:
    scenario_index: int
    factory: PolicyScenarioFactoryConfig
    seeds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PolicyBenchmarkWorkerJob:
    worker_index: int
    checkpoint: Path
    chunks: tuple[PolicyScenarioSeedChunk, ...]
    vision_config: VisionConfig
    rollout_config: RolloutConfig
    simulation_config: SimulationConfig
    shield: bool
    proficiency: PlayerProficiencyProfile = resolve_proficiency("expert")


@dataclass(frozen=True, slots=True)
class PolicyScenarioChunkResult:
    scenario_index: int
    seeds: tuple[int, ...]
    episodes: tuple[EpisodeMetrics, ...]


@dataclass(frozen=True, slots=True)
class PolicyBenchmarkWorkerResult:
    worker_index: int
    checkpoint_metadata: Mapping[str, Any]
    chunks: tuple[PolicyScenarioChunkResult, ...]


def _loaded_checkpoint_metadata(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    history = checkpoint.get("history", ())
    try:
        training_epochs = len(history)
    except TypeError:
        training_epochs = None
    policy_config = checkpoint.get("policy_config", {})
    if not isinstance(policy_config, Mapping):
        raise ValueError("checkpoint policy_config must be a mapping")
    return {
        "version": checkpoint.get("version"),
        "policy_config": dict(policy_config),
        "training_config": checkpoint.get("training_config"),
        "training_data": checkpoint.get("training_data"),
        "epochs": training_epochs,
        "final_training_metrics": history[-1] if training_epochs else None,
    }


def evaluate_policy_worker(job: PolicyBenchmarkWorkerJob) -> PolicyBenchmarkWorkerResult:
    """Load one checkpoint and serially evaluate all chunks for one worker."""

    model, checkpoint = load_checkpoint(job.checkpoint, device="cpu")
    chunks = []
    for chunk in job.chunks:
        factory = PolicyEnvironmentFactory(chunk.factory, job.simulation_config)
        episodes = evaluate_policy(
            model,
            factory,
            chunk.seeds,
            vision_config=job.vision_config,
            config=job.rollout_config,
            shield=job.shield,
            proficiency=job.proficiency,
            device="cpu",
        )
        chunks.append(PolicyScenarioChunkResult(
            scenario_index=chunk.scenario_index,
            seeds=chunk.seeds,
            episodes=tuple(episodes),
        ))
    return PolicyBenchmarkWorkerResult(
        worker_index=job.worker_index,
        checkpoint_metadata=_loaded_checkpoint_metadata(checkpoint),
        chunks=tuple(chunks),
    )


def _distinct_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    result = []
    seen = set()
    for seed in seeds:
        value = int(seed)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _scenario_configs(
    value: PolicyScenarioFactoryConfig | Iterable[PolicyScenarioFactoryConfig],
) -> tuple[PolicyScenarioFactoryConfig, ...]:
    if isinstance(value, PolicyScenarioFactoryConfig):
        result = (value,)
    else:
        result = tuple(value)
    if not result:
        raise ValueError("at least one policy scenario factory is required")
    if not all(isinstance(item, PolicyScenarioFactoryConfig) for item in result):
        raise TypeError("scenario factories must be PolicyScenarioFactoryConfig values")
    names = [item.scenario for item in result]
    if len(set(names)) != len(names):
        raise ValueError("policy scenario factory names must be unique")
    return result


def _split_seeds(seeds: tuple[int, ...], count: int) -> tuple[tuple[int, ...], ...]:
    quotient, remainder = divmod(len(seeds), count)
    result = []
    start = 0
    for index in range(count):
        size = quotient + int(index < remainder)
        result.append(seeds[start:start + size])
        start += size
    return tuple(result)


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordered_scenario_episodes(
    scenario_index: int,
    seeds: tuple[int, ...],
    results: tuple[PolicyBenchmarkWorkerResult, ...],
) -> tuple[EpisodeMetrics, ...]:
    by_seed: dict[int, EpisodeMetrics] = {}
    for result in results:
        for chunk in result.chunks:
            if chunk.scenario_index != scenario_index:
                continue
            if tuple(episode.seed for episode in chunk.episodes) != chunk.seeds:
                raise RuntimeError("policy worker returned episodes out of chunk seed order")
            for episode in chunk.episodes:
                if episode.seed in by_seed:
                    raise RuntimeError(f"policy worker returned duplicate seed {episode.seed}")
                by_seed[episode.seed] = episode
    missing = [seed for seed in seeds if seed not in by_seed]
    if missing:
        raise RuntimeError(f"policy workers omitted {len(missing)} seed(s): {missing[:8]}")
    return tuple(by_seed[seed] for seed in seeds)


def run_policy_benchmark(
    checkpoint: str | Path,
    scenarios: PolicyScenarioFactoryConfig | Iterable[PolicyScenarioFactoryConfig],
    seeds: Iterable[int],
    *,
    vision_config: VisionConfig = VisionConfig(),
    rollout_config: RolloutConfig = RolloutConfig(),
    simulation_config: SimulationConfig = SimulationConfig(),
    shield: bool = True,
    workers: int = 1,
    proficiency: str | PlayerProficiencyProfile = "expert",
) -> dict[str, Any]:
    """Evaluate one checkpoint on CPU while loading it once per worker."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    scenario_values = _scenario_configs(scenarios)
    seed_values = _distinct_seeds(seeds)
    if not seed_values:
        raise ValueError("at least one policy benchmark seed is required")

    effective_workers = min(int(workers), len(seed_values))
    profile = resolve_proficiency(proficiency)
    seed_chunks = _split_seeds(seed_values, effective_workers)
    jobs = tuple(
        PolicyBenchmarkWorkerJob(
            worker_index=worker_index,
            checkpoint=checkpoint_path,
            chunks=tuple(
                PolicyScenarioSeedChunk(scenario_index, factory, seed_chunks[worker_index])
                for scenario_index, factory in enumerate(scenario_values)
            ),
            vision_config=vision_config,
            rollout_config=rollout_config,
            simulation_config=simulation_config,
            shield=bool(shield),
            proficiency=profile,
        )
        for worker_index in range(effective_workers)
    )

    started = time.perf_counter()
    if effective_workers == 1:
        worker_results = (evaluate_policy_worker(jobs[0]),)
    else:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            worker_results = tuple(executor.map(evaluate_policy_worker, jobs, chunksize=1))
    elapsed = time.perf_counter() - started

    if tuple(result.worker_index for result in worker_results) != tuple(range(effective_workers)):
        raise RuntimeError("policy worker results are out of order")
    metadata_values = tuple(result.checkpoint_metadata for result in worker_results)
    if any(metadata != metadata_values[0] for metadata in metadata_values[1:]):
        raise RuntimeError("policy workers loaded inconsistent checkpoint metadata")

    per_scenario_episodes = tuple(
        _ordered_scenario_episodes(index, seed_values, worker_results)
        for index in range(len(scenario_values))
    )
    all_episodes = tuple(
        episode
        for episodes in per_scenario_episodes
        for episode in episodes
    )
    checkpoint_metadata = {
        "checkpoint": str(checkpoint_path),
        "sha256": _checkpoint_sha256(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
        **metadata_values[0],
    }
    return {
        "implementation_sha256": source_tree_sha256(),
        "run": {
            "run_kind": "smoke" if rollout_config.max_frames is not None else "full_duration",
            "acceptance_claim": False,
            "device": "cpu",
            "seeds": seed_values,
            "scenario_factories": tuple(asdict(value) for value in scenario_values),
            "vision_config": asdict(vision_config),
            "rollout_config": asdict(rollout_config),
            "simulation_config": asdict(simulation_config),
            "proficiency": asdict(profile),
        },
        "checkpoint_metadata": checkpoint_metadata,
        "shield": bool(shield),
        "elapsed_seconds": elapsed,
        "workers": effective_workers,
        "requested_workers": int(workers),
        "overall": summarize_episodes(all_episodes),
        "scenarios": {
            factory.scenario: summarize_episodes(episodes)
            for factory, episodes in zip(scenario_values, per_scenario_episodes, strict=True)
        },
    }


__all__ = [
    "PolicyBenchmarkWorkerJob",
    "PolicyBenchmarkWorkerResult",
    "PolicyEnvironmentFactory",
    "PolicyScenarioChunkResult",
    "PolicyScenarioFactoryConfig",
    "PolicyScenarioSeedChunk",
    "evaluate_policy_worker",
    "run_policy_benchmark",
]
