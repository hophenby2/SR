from __future__ import annotations

import hashlib
from pathlib import Path
import pickle
from types import SimpleNamespace

import pytest

from stg_lab.metrics import EpisodeMetrics
from stg_lab import policy_benchmark
from stg_lab.policy_benchmark import PolicyScenarioFactoryConfig
from stg_lab.rollout import RolloutConfig
from stg_lab.sim import SimulationConfig
from stg_lab.vision import VisionConfig


def metric(scenario: str, seed: int) -> EpisodeMetrics:
    return EpisodeMetrics(
        scenario=f"{scenario}:lunatic",
        seed=seed,
        survived=seed % 2 == 0,
        frames=12,
        peak_risk=0.1,
        total_risk=0.2,
        state_hash=f"{scenario}-{seed}",
    )


def configs():
    return (
        VisionConfig(global_width=8, global_height=8, local_width=8, local_height=8),
        RolloutConfig(decision_interval=3, max_frames=12),
        SimulationConfig(reaction_frames=0, action_hold_frames=3),
    )


def install_fake_policy(monkeypatch):
    loads = []
    evaluations = []

    def fake_load(path, *, device):
        model = SimpleNamespace(load_index=len(loads))
        loads.append((path, device, model))
        return model, {
            "version": 7,
            "policy_config": {"channels": 6, "inference_mode": "window"},
            "history": [{"epoch": 1}, {"epoch": 2}],
        }

    def fake_evaluate(model, factory, seeds, **kwargs):
        seeds = tuple(seeds)
        evaluations.append((model, factory, seeds, kwargs))
        return tuple(metric(factory.scenario.scenario, seed) for seed in seeds)

    monkeypatch.setattr(policy_benchmark, "load_checkpoint", fake_load)
    monkeypatch.setattr(policy_benchmark, "evaluate_policy", fake_evaluate)
    return loads, evaluations


def test_worker_job_and_environment_factory_are_pickle_safe() -> None:
    vision, rollout, simulation = configs()
    factory = PolicyScenarioFactoryConfig("stage5_boss3", duration_frames=600)
    chunk = policy_benchmark.PolicyScenarioSeedChunk(0, factory, (4, 5))
    job = policy_benchmark.PolicyBenchmarkWorkerJob(
        worker_index=0,
        checkpoint=Path("policy.pt"),
        chunks=(chunk,),
        vision_config=vision,
        rollout_config=rollout,
        simulation_config=simulation,
        shield=True,
    )
    assert pickle.loads(pickle.dumps(job)) == job
    assert pickle.loads(pickle.dumps(
        policy_benchmark.PolicyEnvironmentFactory(factory, simulation)
    )).scenario.duration_frames == 600


def test_parallel_chunks_load_once_per_worker_and_preserve_distinct_seed_order(
    monkeypatch,
    tmp_path,
) -> None:
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"checkpoint")
    loads, evaluations = install_fake_policy(monkeypatch)
    observed = {}

    class FakeExecutor:
        def __init__(self, *, max_workers):
            observed["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def map(self, function, jobs, *, chunksize):
            jobs = tuple(jobs)
            observed.update(function=function, jobs=jobs, chunksize=chunksize)
            return tuple(function(job) for job in jobs)

    monkeypatch.setattr(policy_benchmark, "ProcessPoolExecutor", FakeExecutor)
    vision, rollout, simulation = configs()
    report = policy_benchmark.run_policy_benchmark(
        checkpoint,
        (
            PolicyScenarioFactoryConfig("stage5_boss3", duration_frames=600),
            PolicyScenarioFactoryConfig("stage5_boss4", duration_frames=700),
        ),
        (9, 7, 9, 8, 6),
        vision_config=vision,
        rollout_config=rollout,
        simulation_config=simulation,
        shield=True,
        workers=2,
    )

    assert observed["max_workers"] == 2
    assert observed["function"] is policy_benchmark.evaluate_policy_worker
    assert observed["chunksize"] == 1
    assert len(observed["jobs"]) == 2
    assert [chunk.seeds for chunk in observed["jobs"][0].chunks] == [(9, 7), (9, 7)]
    assert [chunk.seeds for chunk in observed["jobs"][1].chunks] == [(8, 6), (8, 6)]
    assert len(loads) == 2
    assert all(device == "cpu" for _path, device, _model in loads)
    assert len(evaluations) == 4
    assert all(call[3]["device"] == "cpu" for call in evaluations)

    assert report["run"]["seeds"] == (9, 7, 8, 6)
    assert report["workers"] == 2
    assert report["overall"]["episode_count"] == 8
    assert [item.seed for item in report["scenarios"]["stage5_boss3"]["episodes"]] == [9, 7, 8, 6]
    assert [item.seed for item in report["scenarios"]["stage5_boss4"]["episodes"]] == [9, 7, 8, 6]
    assert report["checkpoint_metadata"] == {
        "checkpoint": str(checkpoint),
        "sha256": hashlib.sha256(b"checkpoint").hexdigest(),
        "size_bytes": len(b"checkpoint"),
            "version": 7,
            "policy_config": {"channels": 6, "inference_mode": "window"},
            "training_config": None,
            "training_data": None,
            "epochs": 2,
        "final_training_metrics": {"epoch": 2},
    }


def test_single_worker_loads_once_for_all_scenarios(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"one")
    loads, evaluations = install_fake_policy(monkeypatch)

    class UnexpectedExecutor:
        def __init__(self, **_kwargs):
            pytest.fail("workers=1 must not construct a process executor")

    monkeypatch.setattr(policy_benchmark, "ProcessPoolExecutor", UnexpectedExecutor)
    report = policy_benchmark.run_policy_benchmark(
        checkpoint,
        (
            PolicyScenarioFactoryConfig("stage5_boss3"),
            PolicyScenarioFactoryConfig("stage5_boss4"),
        ),
        (5, 4, 3),
        workers=1,
        shield=False,
    )

    assert len(loads) == 1
    assert len(evaluations) == 2
    assert all(call[0] is loads[0][2] for call in evaluations)
    assert report["workers"] == 1
    assert report["shield"] is False
    assert [item.seed for item in report["overall"]["episodes"]] == [5, 4, 3, 5, 4, 3]


def test_policy_benchmark_validates_empty_inputs_and_worker_count(tmp_path) -> None:
    checkpoint = tmp_path / "policy.pt"
    checkpoint.touch()
    factory = PolicyScenarioFactoryConfig("stage5_boss3")
    with pytest.raises(ValueError, match="workers"):
        policy_benchmark.run_policy_benchmark(checkpoint, factory, (1,), workers=0)
    with pytest.raises(ValueError, match="scenario factory"):
        policy_benchmark.run_policy_benchmark(checkpoint, (), (1,))
    with pytest.raises(ValueError, match="seed"):
        policy_benchmark.run_policy_benchmark(checkpoint, factory, ())
