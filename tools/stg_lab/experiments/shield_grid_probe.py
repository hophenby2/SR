"""Development-only grid for short-horizon shield settings."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from stg_lab.policy import safety_shield
from stg_lab.protocol import Action
from stg_lab.rollout import (
    RolloutConfig,
    _action_endpoint_if_safe,
    _policy_logits,
    _run_episode,
    imminent_safe_actions,
    scenario_memory_vector,
    shield_action_toward,
)
from stg_lab.scenarios import make_environment
from stg_lab.sim import SimulationConfig
from stg_lab.training import load_checkpoint
from stg_lab.vision import VisionConfig


CHECKPOINT = Path("artifacts/policy_canonical_best.pt")
VISION = VisionConfig(history=4, observation_delay=5)
ROLLOUT = RolloutConfig(decision_interval=3, max_frames=600)
SIMULATION = SimulationConfig(reaction_frames=0, action_hold_frames=3)
CANDIDATES = ("raw", *(f"toward_h{h}" for h in range(1, 7)), *(f"logits_h{h}" for h in range(1, 7)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ShieldController:
    def __init__(self, model: object, strategy: str) -> None:
        self.model = model
        self.strategy = strategy
        self.decisions = 0
        self.switches = 0
        self.overrides = 0
        self.previous: int | None = None

    def __call__(self, environment, visible, _plan, _memory) -> Action:
        if environment.frame == 0:
            self.decisions = self.switches = self.overrides = 0
            self.previous = None
        vector = scenario_memory_vector("stage5_boss3:lunatic", self.model.config.memory_size)
        logits, _hidden = _policy_logits(
            self.model, visible, device="cpu", memory=vector, hidden=None, latest_only=False,
        )
        preferred = Action.from_discrete(int(np.argmax(logits)))
        if self.strategy == "raw":
            selected = preferred
        else:
            kind, raw_horizon = self.strategy.split("_h")
            horizon = int(raw_horizon)
            if kind == "toward":
                selected = shield_action_toward(environment, preferred, horizon=horizon)
            elif _action_endpoint_if_safe(environment, preferred, horizon) is not None:
                selected = preferred
            else:
                selected = Action.from_discrete(
                    safety_shield(logits, imminent_safe_actions(environment, horizon))
                )
        self.decisions += 1
        self.switches += int(self.previous is not None and selected.discrete != self.previous)
        self.overrides += int(selected != preferred)
        self.previous = selected.discrete
        return selected


def _worker(seeds: tuple[int, ...]) -> list[dict[str, object]]:
    torch.set_num_threads(1)
    model, _checkpoint = load_checkpoint(CHECKPOINT, device="cpu")
    result = []
    for seed in seeds:
        for strategy in CANDIDATES:
            controller = ShieldController(model, strategy)
            trace = _run_episode(
                lambda value: make_environment(
                    "stage5_boss3", difficulty="lunatic", seed=value,
                    config=SIMULATION, duration_frames=600,
                ),
                seed,
                planner=None,
                vision_config=VISION,
                config=ROLLOUT,
                controller=controller,
            )
            result.append({
                "strategy": strategy,
                **asdict(trace.metrics),
                "decisions": controller.decisions,
                "switches": controller.switches,
                "shield_overrides": controller.overrides,
            })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = (2001, 2002, *range(1021, 1041))
    workers = min(max(1, args.workers), len(seeds))
    chunks = tuple(seeds[index::workers] for index in range(workers))
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        records = [record for chunk in executor.map(_worker, chunks) for record in chunk]
    order = {(seed, strategy): index for index, (seed, strategy) in enumerate(
        (seed, strategy) for seed in seeds for strategy in CANDIDATES
    )}
    records.sort(key=lambda value: order[(int(value["seed"]), str(value["strategy"]))])
    summaries = {}
    for strategy in CANDIDATES:
        values = [record for record in records if record["strategy"] == strategy]
        summaries[strategy] = {
            "episodes": len(values),
            "survived": sum(bool(value["survived"]) for value in values),
            "survival_rate": sum(bool(value["survived"]) for value in values) / len(values),
            "mean_switches": sum(int(value["switches"]) for value in values) / len(values),
            "mean_shield_overrides": sum(int(value["shield_overrides"]) for value in values) / len(values),
            "failed": [{"seed": value["seed"], "frames": value["frames"]} for value in values if not value["survived"]],
        }
    report = {
        "run_kind": "shield_grid_development_probe",
        "acceptance_claim": False,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": _sha256(CHECKPOINT),
        "seeds": list(seeds),
        "excluded_acceptance_seeds": list(range(3001, 3101)),
        "vision_config": asdict(VISION),
        "rollout_config": asdict(ROLLOUT),
        "simulation_config": asdict(SIMULATION),
        "workers": workers,
        "elapsed_seconds": time.perf_counter() - started,
        "summaries": summaries,
        "episodes": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"elapsed_seconds": report["elapsed_seconds"], "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
