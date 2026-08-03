"""Closed-loop candidate search against one live LuaSTG engine connection."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence

from .engine import EngineClient
from .engine_play import (
    EnginePlayConfig,
    VisibleController,
    VisualPolicyController,
    run_engine_play,
)
from .protocol import Action
from .provenance import source_tree_sha256
from .route_memory import ExternalRouteController, ExternalRouteLibraryController
from .sim import coerce_action
from .vision import VisionObservation


ControllerFactory = Callable[[], VisibleController]


@dataclass(frozen=True, slots=True)
class CandidateStrategy:
    """Visible-controller transformations explored by the live search."""

    horizontal_sign: int = 1
    vertical_sign: int = 1
    slow_mode: str = "preserve"
    decision_offset: int = 0
    # Compatibility metadata for the reporting-only local threat diagnostic.
    shoot_gate_radius: float = 20.0
    shoot_risk_threshold: float = 0.25
    shoot_motion_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.horizontal_sign not in (-1, 1) or self.vertical_sign not in (-1, 1):
            raise ValueError("candidate axis signs must be -1 or 1")
        if self.slow_mode not in {"preserve", "focus", "unfocus"}:
            raise ValueError("candidate slow_mode must be preserve, focus, or unfocus")
        if not -60 <= self.decision_offset <= 60:
            raise ValueError("candidate decision_offset must be in [-60, 60]")
        numeric = (
            self.shoot_gate_radius,
            self.shoot_risk_threshold,
            self.shoot_motion_weight,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("candidate shoot values must be finite")
        if self.shoot_gate_radius <= 0.0:
            raise ValueError("candidate shoot_gate_radius must be positive")
        if self.shoot_risk_threshold < 0.0 or self.shoot_motion_weight < 0.0:
            raise ValueError("candidate shoot risk values cannot be negative")

    @property
    def candidate_id(self) -> str:
        encoded = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        return "candidate-" + hashlib.sha256(encoded).hexdigest()[:16]

    def report(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, **asdict(self)}


@dataclass(frozen=True, slots=True)
class EngineTrainingConfig:
    train_seeds: tuple[int, ...] = (20260729, 20260730)
    heldout_seeds: tuple[int, ...] = (20260731,)
    candidate_count: int = 8
    search_seed: int = 20260730

    def __post_init__(self) -> None:
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        if not self.train_seeds or not self.heldout_seeds:
            raise ValueError("training and held-out seeds must both be nonempty")
        if len(set(self.train_seeds)) != len(self.train_seeds):
            raise ValueError("training seeds must be unique")
        if len(set(self.heldout_seeds)) != len(self.heldout_seeds):
            raise ValueError("held-out seeds must be unique")
        overlap = set(self.train_seeds) & set(self.heldout_seeds)
        if overlap:
            raise ValueError(f"training and held-out seeds overlap: {sorted(overlap)}")


class CandidateController:
    """Apply one deterministic candidate without exposing authority telemetry."""

    def __init__(self, base: VisibleController, strategy: CandidateStrategy) -> None:
        self.base = base
        self.strategy = strategy
        self.reset()

    def reset(self) -> None:
        self.base.reset()
        self.decisions = 0
        self._negative_offset_applied = False

    def _base_action(self, visible: VisionObservation) -> Action:
        calls = 1
        if self.strategy.decision_offset < 0 and not self._negative_offset_applied:
            calls += -self.strategy.decision_offset
            self._negative_offset_applied = True
        action = Action()
        for _ in range(calls):
            action = coerce_action(self.base.select(visible))
        return action

    def select(self, visible: VisionObservation) -> Action:
        if self.decisions < self.strategy.decision_offset:
            preferred = Action(slow=True)
        else:
            preferred = self._base_action(visible)
        self.decisions += 1
        if self.strategy.slow_mode == "focus":
            slow = True
        elif self.strategy.slow_mode == "unfocus":
            slow = False
        else:
            slow = preferred.slow
        return Action(
            move_x=preferred.move_x * self.strategy.horizontal_sign,
            move_y=preferred.move_y * self.strategy.vertical_sign,
            slow=slow,
            shoot=preferred.shoot,
            spell=False,
        )


def controller_factory_from_template(controller: VisibleController) -> ControllerFactory:
    """Clone supported controller state while sharing only immutable model data."""

    if isinstance(controller, ExternalRouteController):
        return lambda: ExternalRouteController(
            controller.memory,
            config=controller.config,
            fallback=controller.fallback,
        )
    if isinstance(controller, ExternalRouteLibraryController):
        return lambda: ExternalRouteLibraryController(
            controller.memories,
            config=controller.config,
            fallback=controller.fallback,
        )
    if isinstance(controller, VisualPolicyController):
        return lambda: VisualPolicyController(
            controller.model,
            controller.scenario_key,
            device=controller.device,
            proficiency=controller.proficiency,
            seed=controller.seed,
            scenario_vocabulary=controller.scenario_vocabulary,
        )
    raise TypeError(f"cannot clone controller type {type(controller).__name__}")


def generate_candidate_strategies(
    base: EnginePlayConfig,
    *,
    count: int,
    search_seed: int,
) -> tuple[CandidateStrategy, ...]:
    """Generate a baseline-first, seed-reproducible subset of a finite grid."""

    if count <= 0:
        raise ValueError("candidate count must be positive")
    values = [
        CandidateStrategy(
            x,
            y,
            slow,
            offset,
            base.shoot_gate_radius,
            base.shoot_risk_threshold,
            base.shoot_motion_weight,
        )
        for x in (1, -1)
        for y in (1, -1)
        for slow in ("preserve", "focus", "unfocus")
        for offset in (0, -6, 6, -3, 3)
    ]
    baseline = CandidateStrategy(
        shoot_gate_radius=base.shoot_gate_radius,
        shoot_risk_threshold=base.shoot_risk_threshold,
        shoot_motion_weight=base.shoot_motion_weight,
    )
    by_id = {candidate.candidate_id: candidate for candidate in values}
    by_id[baseline.candidate_id] = baseline
    remaining = [value for key, value in sorted(by_id.items()) if key != baseline.candidate_id]
    if count > len(remaining) + 1:
        raise ValueError(f"candidate count exceeds finite search space of {len(remaining) + 1}")
    random.Random(int(search_seed)).shuffle(remaining)
    return (baseline, *remaining[:count - 1])


def _canonical_bytes(value: Any) -> bytes:
    def fallback(item: Any) -> Any:
        if isinstance(item, Path):
            return str(item)
        if hasattr(item, "tolist"):
            return item.tolist()
        if hasattr(item, "item"):
            return item.item()
        raise TypeError(f"value of type {type(item).__name__} is not JSON serializable")

    return (json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=fallback,
    ) + "\n").encode("utf-8")


def _write_json(path: Path, value: Any) -> str:
    raw = _canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _strict_episode_evidence(
    report: Mapping[str, Any],
    *,
    phase: str,
    candidate_index: int,
    candidate: CandidateStrategy,
    trace_path: Path | None,
) -> dict[str, Any]:
    terminated = report.get("terminated") is True
    reason = report.get("engine_termination_reason") if terminated else None
    if terminated and reason is None:
        reason = report.get("termination_reason")
    outcome_evidence = report.get("outcome_evidence")
    outcome_evidence = outcome_evidence if isinstance(outcome_evidence, Mapping) else {}
    final_player = outcome_evidence.get("final_player")
    final_player = final_player if isinstance(final_player, Mapping) else {}
    death_value = final_player.get("death")
    final_death = (
        float(death_value)
        if (
            not isinstance(death_value, bool)
            and isinstance(death_value, (int, float))
            and math.isfinite(float(death_value))
        ) else None
    )
    strict_success = (
        terminated
        and reason == "attack_complete"
        and final_death == 0.0
    )
    action_steps = report.get("action_steps")
    if not isinstance(action_steps, list):
        action_steps = []
    trace_sha256 = hashlib.sha256(_canonical_bytes(report)).hexdigest()
    if trace_path is not None:
        trace_sha256 = _write_json(trace_path, report)
    return {
        "phase": phase,
        "candidate_index": candidate_index,
        "candidate_id": candidate.candidate_id,
        "seed": report.get("seed"),
        "success": strict_success,
        "runner_success": report.get("success") is True,
        "runner_success_matches_strict_metric": (report.get("success") is True) == strict_success,
        "terminated": terminated,
        "termination_reason": reason if terminated else "max_frames",
        "final_death": final_death,
        "frames": report.get("frames"),
        "decision_count": report.get("decision_count"),
        "shoot_rate": report.get("shoot_rate"),
        "action_trace_sha256": hashlib.sha256(_canonical_bytes(action_steps)).hexdigest(),
        "full_trace_sha256": trace_sha256,
        "trace_path": None if trace_path is None else str(trace_path),
        "engine": report.get("engine"),
        "outcome_evidence": report.get("outcome_evidence"),
    }


def _episode_summary(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successes = sum(value.get("success") is True for value in episodes)
    reasons = Counter(str(value.get("termination_reason")) for value in episodes)
    return {
        "attempts": len(episodes),
        "strict_successes": successes,
        "strict_success_rate": successes / len(episodes) if episodes else 0.0,
        "termination_reasons": dict(sorted(reasons.items())),
    }


def _engine_identity(report: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    engine = report.get("engine")
    if not isinstance(engine, Mapping):
        return None, None, None
    runtime = engine.get("runtime_identity")
    process_id = runtime.get("process_id") if isinstance(runtime, Mapping) else None
    return engine.get("session_id"), engine.get("process_nonce"), process_id


def run_engine_training(
    client: EngineClient,
    *,
    scenario: str,
    attack: int,
    player: str,
    controller_factory: ControllerFactory,
    controller_metadata: Mapping[str, Any],
    play_config: EnginePlayConfig,
    training_config: EngineTrainingConfig = EngineTrainingConfig(),
    candidates: Sequence[CandidateStrategy] | None = None,
    trace_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Search on training seeds, then evaluate the selected candidate held out."""

    candidate_values = tuple(candidates or generate_candidate_strategies(
        play_config,
        count=training_config.candidate_count,
        search_seed=training_config.search_seed,
    ))
    if not candidate_values:
        raise ValueError("at least one candidate strategy is required")
    if len({value.candidate_id for value in candidate_values}) != len(candidate_values):
        raise ValueError("candidate strategies must be unique")
    traces = None if trace_directory is None else Path(trace_directory)
    expected_identity: tuple[Any, Any, Any] | None = None
    candidate_reports: list[dict[str, Any]] = []

    def run_one(
        controller: CandidateController,
        candidate: CandidateStrategy,
        candidate_index: int,
        phase: str,
        seed: int,
    ) -> dict[str, Any]:
        nonlocal expected_identity
        config = replace(
            play_config,
            shoot_gate_radius=candidate.shoot_gate_radius,
            shoot_risk_threshold=candidate.shoot_risk_threshold,
            shoot_motion_weight=candidate.shoot_motion_weight,
            render=False,
        )
        metadata = {
            **dict(controller_metadata),
            "training_candidate": candidate.report(),
            "training_phase": phase,
        }
        raw_report = run_engine_play(
            client,
            scenario=scenario,
            attack=attack,
            seed=int(seed),
            player=player,
            controller=controller,
            controller_metadata=metadata,
            config=config,
        )
        identity = _engine_identity(raw_report)
        if expected_identity is None:
            expected_identity = identity
        elif identity != expected_identity:
            raise RuntimeError("live training episodes did not use one engine process/session")
        trace_path = None
        if traces is not None:
            trace_path = traces / f"candidate-{candidate_index:03d}" / f"{phase}-seed-{seed}.json"
        return _strict_episode_evidence(
            raw_report,
            phase=phase,
            candidate_index=candidate_index,
            candidate=candidate,
            trace_path=trace_path,
        )

    for candidate_index, candidate in enumerate(candidate_values):
        controller = CandidateController(controller_factory(), candidate)
        episodes = [
            run_one(controller, candidate, candidate_index, "train", int(seed))
            for seed in training_config.train_seeds
        ]
        candidate_reports.append({
            "candidate_index": candidate_index,
            "strategy": candidate.report(),
            "training": _episode_summary(episodes),
            "episodes": episodes,
        })

    selected_index = min(
        range(len(candidate_reports)),
        key=lambda index: (
            -int(candidate_reports[index]["training"]["strict_successes"]),
            index,
        ),
    )
    selected = candidate_values[selected_index]
    heldout_controller = CandidateController(controller_factory(), selected)
    heldout_episodes = [
        run_one(heldout_controller, selected, selected_index, "heldout", int(seed))
        for seed in training_config.heldout_seeds
    ]
    heldout = _episode_summary(heldout_episodes)
    passed = heldout["strict_successes"] == heldout["attempts"]
    effective_play_config = replace(play_config, render=False)
    return {
        "schema_version": 1,
        "run_kind": "live_luastg_closed_loop_candidate_training",
        "implementation_sha256": source_tree_sha256(),
        "search_completed": True,
        "passed": passed,
        "success_criterion": (
            "terminated=true, engine_termination_reason=attack_complete, and "
            "outcome_evidence.final_player.death=0"
        ),
        "selection_metric": "maximum strict training successes; generation order breaks ties",
        "scenario": scenario,
        "attack": int(attack),
        "player": player,
        "train_seeds": list(training_config.train_seeds),
        "heldout_seeds": list(training_config.heldout_seeds),
        "train_heldout_disjoint": True,
        "search_seed": int(training_config.search_seed),
        "candidate_count": len(candidate_values),
        "engine_episode_count": len(candidate_values) * len(training_config.train_seeds)
        + len(training_config.heldout_seeds),
        "same_engine_connection": True,
        "engine_identity": {
            "session_id": None if expected_identity is None else expected_identity[0],
            "process_nonce": None if expected_identity is None else expected_identity[1],
            "process_id": None if expected_identity is None else expected_identity[2],
        },
        "base_controller": dict(controller_metadata),
        "play_config": asdict(effective_play_config),
        "candidate_results": candidate_reports,
        "selected_candidate_index": selected_index,
        "selected_strategy": selected.report(),
        "selected_training": candidate_reports[selected_index]["training"],
        "heldout": heldout,
        "heldout_episodes": heldout_episodes,
        "trace_directory": None if traces is None else str(traces),
    }


def strategy_artifact(training_report: Mapping[str, Any]) -> dict[str, Any]:
    if training_report.get("run_kind") != "live_luastg_closed_loop_candidate_training":
        raise ValueError("strategy source is not a live closed-loop training report")
    selected = training_report.get("selected_strategy")
    if not isinstance(selected, Mapping):
        raise ValueError("training report has no selected strategy")
    return {
        "schema_version": 1,
        "kind": "live_luastg_candidate_strategy",
        "implementation_sha256": training_report.get("implementation_sha256"),
        "success_criterion": training_report.get("success_criterion"),
        "scenario": training_report.get("scenario"),
        "attack": training_report.get("attack"),
        "player": training_report.get("player"),
        "base_controller": training_report.get("base_controller"),
        "search_seed": training_report.get("search_seed"),
        "train_seeds": training_report.get("train_seeds"),
        "heldout_seeds": training_report.get("heldout_seeds"),
        "selected_strategy": dict(selected),
        "selected_training": training_report.get("selected_training"),
        "heldout": training_report.get("heldout"),
        "engine_identity": training_report.get("engine_identity"),
    }


def write_strategy_artifact(path: str | Path, training_report: Mapping[str, Any]) -> str:
    return _write_json(Path(path), strategy_artifact(training_report))


def load_candidate_strategy(path: str | Path) -> CandidateStrategy:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("kind") != "live_luastg_candidate_strategy":
        raise ValueError("unsupported live candidate strategy artifact")
    raw = value.get("selected_strategy")
    if not isinstance(raw, Mapping):
        raise ValueError("strategy artifact has no selected strategy")
    fields = {
        name: raw[name]
        for name in CandidateStrategy.__dataclass_fields__
        if name in raw
    }
    return CandidateStrategy(**fields)


__all__ = [
    "CandidateController",
    "CandidateStrategy",
    "ControllerFactory",
    "EngineTrainingConfig",
    "controller_factory_from_template",
    "generate_candidate_strategies",
    "load_candidate_strategy",
    "run_engine_training",
    "strategy_artifact",
    "write_strategy_artifact",
]
