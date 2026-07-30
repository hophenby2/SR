"""Development-only closed-loop probe for policy action smoothing."""

from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from stg_lab.protocol import Action
from stg_lab.rollout import (
    RolloutConfig,
    _policy_logits,
    _run_episode,
    scenario_memory_vector,
    shield_action_toward,
)
from stg_lab.scenarios import make_environment
from stg_lab.sim import SimulationConfig
from stg_lab.training import load_checkpoint
from stg_lab.vision import VisionConfig


CHECKPOINT = Path("artifacts/policy_canonical_best.pt")
VISION = VisionConfig(history=4, observation_delay=5)
ROLLOUT = RolloutConfig(
    decision_interval=3,
    max_frames=600,
    shield_horizon=3,
    shield_strategy="toward",
)
SIMULATION = SimulationConfig(reaction_frames=0, action_hold_frames=3)
CANDIDATES = (
    "raw",
    "ema_025",
    "ema_050",
    "ema_075",
    "confirm_2",
    "confirm_3",
    "majority_3",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SmoothedController:
    def __init__(self, model: object, strategy: str) -> None:
        self.model = model
        self.strategy = strategy
        self.ema: np.ndarray | None = None
        self.raw_history: deque[int] = deque(maxlen=3)
        self.selected: int | None = None
        self.pending: int | None = None
        self.pending_count = 0
        self.previous_raw: int | None = None
        self.previous_policy: int | None = None
        self.previous_final: int | None = None
        self.decisions = 0
        self.raw_switches = 0
        self.policy_switches = 0
        self.final_switches = 0
        self.shield_overrides = 0

    def _reset(self) -> None:
        self.ema = None
        self.raw_history.clear()
        self.selected = None
        self.pending = None
        self.pending_count = 0
        self.previous_raw = None
        self.previous_policy = None
        self.previous_final = None
        self.decisions = 0
        self.raw_switches = 0
        self.policy_switches = 0
        self.final_switches = 0
        self.shield_overrides = 0

    def _choose(self, logits: np.ndarray) -> int:
        raw = int(np.argmax(logits))
        self.raw_history.append(raw)
        if self.strategy == "raw":
            return raw
        if self.strategy.startswith("ema_"):
            decay = int(self.strategy.rsplit("_", 1)[1]) / 100.0
            self.ema = logits.copy() if self.ema is None else decay * self.ema + (1.0 - decay) * logits
            return int(np.argmax(self.ema))
        if self.strategy.startswith("confirm_"):
            required = int(self.strategy.rsplit("_", 1)[1])
            if self.selected is None:
                self.selected = raw
            elif raw == self.selected:
                self.pending = None
                self.pending_count = 0
            else:
                if raw == self.pending:
                    self.pending_count += 1
                else:
                    self.pending = raw
                    self.pending_count = 1
                if self.pending_count >= required:
                    self.selected = raw
                    self.pending = None
                    self.pending_count = 0
            return self.selected
        if self.strategy == "majority_3":
            counts = Counter(self.raw_history)
            best_count = max(counts.values())
            winners = {value for value, count in counts.items() if count == best_count}
            if self.previous_policy in winners:
                return int(self.previous_policy)
            return raw if raw in winners else min(winners)
        raise ValueError(f"unknown strategy: {self.strategy}")

    def __call__(self, environment, visible, _plan, _memory) -> Action:
        if environment.frame == 0:
            self._reset()
        vector = scenario_memory_vector("stage5_boss3:lunatic", self.model.config.memory_size)
        logits, _hidden = _policy_logits(
            self.model,
            visible,
            device="cpu",
            memory=vector,
            hidden=None,
            latest_only=False,
        )
        raw = int(np.argmax(logits))
        policy = self._choose(logits)
        preferred = Action.from_discrete(policy)
        final = shield_action_toward(environment, preferred, horizon=3)
        self.decisions += 1
        self.raw_switches += int(self.previous_raw is not None and raw != self.previous_raw)
        self.policy_switches += int(self.previous_policy is not None and policy != self.previous_policy)
        self.final_switches += int(self.previous_final is not None and final.discrete != self.previous_final)
        self.shield_overrides += int(final.discrete != policy)
        self.previous_raw = raw
        self.previous_policy = policy
        self.previous_final = final.discrete
        return final


def _worker(seeds: tuple[int, ...]) -> list[dict[str, object]]:
    torch.set_num_threads(1)
    model, _checkpoint = load_checkpoint(CHECKPOINT, device="cpu")
    result: list[dict[str, object]] = []
    for seed in seeds:
        for strategy in CANDIDATES:
            controller = SmoothedController(model, strategy)
            trace = _run_episode(
                lambda value: make_environment(
                    "stage5_boss3",
                    difficulty="lunatic",
                    seed=value,
                    config=SIMULATION,
                    duration_frames=600,
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
                "raw_switches": controller.raw_switches,
                "policy_switches": controller.policy_switches,
                "final_switches": controller.final_switches,
                "shield_overrides": controller.shield_overrides,
            })
    return result


def _split(values: tuple[int, ...], count: int) -> tuple[tuple[int, ...], ...]:
    return tuple(values[index::count] for index in range(count))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = (2001, 2002, *range(1021, 1041))
    workers = min(max(1, args.workers), len(seeds))
    started = time.perf_counter()
    chunks = _split(seeds, workers)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        nested = tuple(executor.map(_worker, chunks))
    records = [record for values in nested for record in values]
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
            "mean_raw_switches": sum(int(value["raw_switches"]) for value in values) / len(values),
            "mean_policy_switches": sum(int(value["policy_switches"]) for value in values) / len(values),
            "mean_final_switches": sum(int(value["final_switches"]) for value in values) / len(values),
            "mean_shield_overrides": sum(int(value["shield_overrides"]) for value in values) / len(values),
            "failed": [
                {"seed": value["seed"], "frames": value["frames"]}
                for value in values if not value["survived"]
            ],
        }
    report = {
        "run_kind": "controller_smoothing_development_probe",
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
