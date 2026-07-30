from __future__ import annotations

import pickle

from stg_lab import benchmark
from stg_lab.metrics import EpisodeMetrics
from stg_lab.planning import PlannerConfig, RiskConfig
from stg_lab.rollout import RolloutConfig
from stg_lab.sim import SimulationConfig
from stg_lab.vision import VisionConfig


def metric(scenario: str, seed: int) -> EpisodeMetrics:
    return EpisodeMetrics(
        scenario=f"{scenario}:normal",
        seed=seed,
        survived=seed % 2 == 0,
        frames=10,
        peak_risk=0.1,
        total_risk=0.2,
        state_hash=f"{scenario}-{seed}",
    )


def configs():
    return (
        PlannerConfig(risk=RiskConfig(horizon_frames=4, sample_every=2, cell_size=64.0)),
        VisionConfig(global_width=8, global_height=8, local_width=8, local_height=8),
        RolloutConfig(decision_interval=2, max_frames=10),
    )


def test_planner_job_is_pickle_safe() -> None:
    planner, vision, rollout = configs()
    job = benchmark.PlannerBenchmarkJob(
        "stage5_boss3",
        "normal",
        4,
        planner,
        vision,
        rollout,
        True,
    )
    loaded = pickle.loads(pickle.dumps(job))
    assert loaded == job
    assert loaded.scenario == "stage5_boss3"


def test_environment_factory_preserves_simulation_timing() -> None:
    config = SimulationConfig(reaction_frames=2, action_hold_frames=3)
    factory = benchmark.ScenarioEnvironmentFactory("stage5_boss3", "normal", config)
    loaded = pickle.loads(pickle.dumps(factory))
    environment = loaded(4)
    assert environment.config.reaction_frames == 2
    assert environment.config.action_hold_frames == 3


def test_benchmark_reports_per_scenario_and_unique_hashes(monkeypatch) -> None:
    planner, vision, rollout = configs()
    monkeypatch.setattr(
        benchmark,
        "evaluate_planner_job",
        lambda job: metric(job.scenario, job.seed),
    )
    report = benchmark.run_planner_benchmark(
        ("stage5_boss3", "stage5_boss4"),
        "normal",
        (3, 4),
        planner_config=planner,
        vision_config=vision,
        rollout_config=rollout,
        simulation_config=SimulationConfig(reaction_frames=2, action_hold_frames=3),
    )
    assert report["run_kind"] == "smoke"
    assert report["acceptance_claim"] is False
    assert report["overall"]["episode_count"] == 4
    assert report["overall"]["survival_rate"] == 0.5
    assert report["overall"]["unique_state_hashes"] == 4
    assert report["scenarios"]["stage5_boss3"]["episode_count"] == 2
    assert report["simulation_config"]["reaction_frames"] == 2
    assert report["simulation_config"]["action_hold_frames"] == 3


def test_workers_branch_uses_top_level_worker(monkeypatch) -> None:
    planner, vision, rollout = configs()
    observed = {}

    class FakeExecutor:
        def __init__(self, *, max_workers):
            observed["workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def map(self, function, jobs, *, chunksize):
            jobs = tuple(jobs)
            observed.update(function=function, jobs=jobs, chunksize=chunksize)
            return tuple(metric(job.scenario, job.seed) for job in jobs)

    monkeypatch.setattr(benchmark, "ProcessPoolExecutor", FakeExecutor)
    report = benchmark.run_planner_benchmark(
        ("stage5_boss3",),
        "normal",
        (7, 8),
        planner_config=planner,
        vision_config=vision,
        rollout_config=rollout,
        workers=2,
    )
    assert observed["workers"] == 2
    assert observed["function"] is benchmark.evaluate_planner_job
    assert observed["chunksize"] == 1
    assert all(isinstance(job, benchmark.PlannerBenchmarkJob) for job in observed["jobs"])
    assert report["workers"] == 2
