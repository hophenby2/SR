"""Demonstration collection and reproducible standalone evaluation.

The learned policy only receives frames produced by :class:`DelayedVision`.
The exact-state planner is used as a teacher, for metric labels, and by the
optional short-horizon safety shield; it is never added to policy inputs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import inspect
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - base installation intentionally omits torch
    torch = None

from .memory import EpisodeMemory, EpisodicMemory
from .metrics import EpisodeMetrics, state_hash
from .planning import PlanResult, SpatioTemporalPlanner
from .policy import (
    PlayerProficiencyProfile,
    ProficiencyRuntime,
    proficiency_vector,
    resolve_proficiency,
    safety_shield,
)
from .protocol import Action
from .sim import ActionLike, Observation, Outcome, STGEnvironment, coerce_action
from .training import Demonstrations
from .vision import DelayedVision, VisionConfig, VisionObservation


EnvironmentFactory = Callable[[int], STGEnvironment]
MemoryProvider = Callable[[str, VisionObservation], Sequence[float] | np.ndarray]
CueBuilder = Callable[[Observation], Any]


class RolloutController(Protocol):
    """Select an action from human-visible input and optional recalled memory."""

    def __call__(
        self,
        environment: STGEnvironment,
        vision: VisionObservation,
        teacher_plan: PlanResult | None,
        memory: EpisodeMemory | None,
    ) -> ActionLike: ...


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    """Timing and shield settings shared by data collection and evaluation."""

    decision_interval: int = 3
    max_frames: int | None = None
    risk_scale: float = 8.0
    shield_horizon: int = 12
    shield_strategy: str = "logits"

    def __post_init__(self) -> None:
        if self.decision_interval <= 0:
            raise ValueError("decision_interval must be positive")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive or None")
        if self.risk_scale <= 0.0:
            raise ValueError("risk_scale must be positive")
        if self.shield_horizon <= 0:
            raise ValueError("shield_horizon must be positive")
        if self.shield_strategy not in {"logits", "toward"}:
            raise ValueError("shield_strategy must be 'logits' or 'toward'")


@dataclass(frozen=True, slots=True)
class RolloutTrace:
    metrics: EpisodeMetrics
    actions: tuple[Action, ...]
    risks: tuple[float, ...]
    initial_observation: Observation
    final_observation: Observation


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    raw: tuple[EpisodeMetrics, ...]
    shielded: tuple[EpisodeMetrics, ...]

    @property
    def raw_survival(self) -> float:
        return survival_rate(self.raw)

    @property
    def shielded_survival(self) -> float:
        return survival_rate(self.shielded)


@dataclass(frozen=True, slots=True)
class MemoryBenchmarkResult:
    first: EpisodeMetrics
    second: EpisodeMetrics
    memory: EpisodeMemory
    minimum_risk_improvement: float = 0.30

    @property
    def risk_improvement(self) -> float:
        if self.first.peak_risk <= 0.0:
            return 1.0 if self.second.peak_risk <= 0.0 else 0.0
        return (self.first.peak_risk - self.second.peak_risk) / self.first.peak_risk

    @property
    def passed(self) -> bool:
        return (
            not self.first.survived
            and (self.second.survived or self.risk_improvement >= self.minimum_risk_improvement)
        )


def survival_rate(episodes: Iterable[EpisodeMetrics]) -> float:
    values = tuple(episodes)
    return sum(item.survived for item in values) / len(values) if values else 0.0


def save_demonstrations(demonstrations: Demonstrations, path: str | Path) -> None:
    demonstrations.save(path)


def load_demonstrations(path: str | Path) -> Demonstrations:
    return Demonstrations.load(path)


def scenario_memory_vector(
    scenario_key: str,
    memory_size: int = 4,
    vocabulary: Sequence[str] | None = None,
) -> np.ndarray:
    """Encode an identity token, with legacy Boss #3/#4 compatibility."""

    if memory_size < 0:
        raise ValueError("scenario memory size cannot be negative")
    vector = np.zeros(memory_size, dtype=np.float32)
    if vocabulary is not None:
        from .native_dataset import UNKNOWN_SCENARIO_CONTEXT

        values = tuple(str(value) for value in vocabulary)
        if len(values) > memory_size:
            raise ValueError(
                "scenario vocabulary width exceeds policy memory_size"
            )
        if len(set(values)) != len(values):
            raise ValueError("scenario vocabulary entries must be unique")
        try:
            index = values.index(str(scenario_key))
        except ValueError:
            try:
                index = values.index(UNKNOWN_SCENARIO_CONTEXT)
            except ValueError:
                return vector
        vector[index] = 1.0
        return vector
    # New stream policies deliberately use memory_size=0: their GRU must infer
    # attack phase from visible motion rather than a handwritten boss identity.
    if memory_size < 2:
        return vector
    normalized = scenario_key.lower()
    if "boss3" in normalized:
        vector[0] = 1.0
    elif "boss4" in normalized:
        vector[1] = 1.0
    return vector


def policy_context_vector(
    model: Any,
    scenario_key: str,
    previous_action: Action | None = None,
) -> np.ndarray:
    """Build the model's identity and observed prior-motor context."""

    config = getattr(model, "config", None)
    memory_size = int(getattr(config, "memory_size", 4))
    vector = scenario_memory_vector(
        scenario_key,
        memory_size,
        getattr(model, "scenario_vocabulary", None),
    )
    previous_action_size = int(getattr(model, "previous_action_size", 0))
    if previous_action_size:
        if previous_action_size != 18:
            raise ValueError("previous action context must contain 18 entries")
        offset = int(getattr(model, "previous_action_offset", 0))
        if offset + previous_action_size > memory_size:
            raise ValueError("previous action context does not fit policy memory")
        vector[offset:offset + previous_action_size] = 0.0
        if previous_action is not None:
            vector[offset + previous_action.discrete] = 1.0
    return vector


def _frame_limit(environment: STGEnvironment, config: RolloutConfig) -> int:
    duration = int(environment.scenario.duration_frames)
    return duration if config.max_frames is None else min(duration, config.max_frames)


def _normalized_risk(value: float, scale: float) -> float:
    value = max(0.0, float(value))
    return value / (value + scale)


def _run_episode(
    environment_factory: EnvironmentFactory,
    seed: int,
    *,
    planner: SpatioTemporalPlanner | None,
    vision_config: VisionConfig,
    config: RolloutConfig,
    controller: RolloutController,
    memory: EpisodeMemory | None = None,
    on_decision: Callable[[VisionObservation, Action, float], None] | None = None,
) -> RolloutTrace:
    environment = environment_factory(int(seed))
    observation = environment.reset(seed=int(seed))
    initial_observation = observation
    vision = DelayedVision(config=vision_config)
    visible = vision.reset(observation)
    limit = _frame_limit(environment, config)
    actions: list[Action] = []
    risks: list[float] = []
    agreements = 0
    teacher_decisions = 0

    while not environment.done and environment.frame < limit:
        plan = planner.plan(environment, observation=observation) if planner is not None else None
        risk = max(0.0, float(plan.start_risk)) if plan is not None else 0.0
        action = coerce_action(controller(environment, visible, plan, memory))
        actions.append(action)
        risks.append(risk)
        if plan is not None:
            agreements += int(action.discrete == plan.first_action.discrete)
            teacher_decisions += 1
        if on_decision is not None:
            on_decision(visible, action, risk)

        for _ in range(config.decision_interval):
            if environment.done or environment.frame >= limit:
                break
            # DelayedVision rasterizes authority geometry itself; building the
            # simulator's duplicate semantic grid would double frame cost.
            result = environment._advance(action, build_semantic=False, detect_collision=True)
            observation = result.observation
            visible = vision.push(observation)

    survived = environment.outcome is Outcome.CLEAR or (
        environment.outcome is not Outcome.HIT and environment.frame >= limit
    )
    metrics = EpisodeMetrics(
        scenario=str(getattr(environment.scenario, "scenario_key", environment.scenario.name)),
        seed=int(seed),
        survived=survived,
        frames=int(environment.frame),
        peak_risk=max(risks, default=0.0),
        total_risk=sum(risks) * config.decision_interval,
        deaths=int(environment.outcome is Outcome.HIT),
        action_agreement=agreements / teacher_decisions if teacher_decisions else None,
        state_hash=state_hash(observation),
    )
    return RolloutTrace(
        metrics=metrics,
        actions=tuple(actions),
        risks=tuple(risks),
        initial_observation=initial_observation,
        final_observation=observation,
    )


def collect_demonstrations(
    environment_factory: EnvironmentFactory,
    seeds: Iterable[int],
    *,
    planner: SpatioTemporalPlanner | None = None,
    vision_config: VisionConfig = VisionConfig(),
    config: RolloutConfig = RolloutConfig(),
    shield: bool = True,
    include_scenario_memory: bool = True,
    proficiency: str | PlayerProficiencyProfile = "expert",
    output: str | Path | None = None,
) -> tuple[Demonstrations, tuple[EpisodeMetrics, ...]]:
    """Collect delayed-vision samples labelled by the exact-state planner."""

    planner = planner or SpatioTemporalPlanner()
    global_frames: list[np.ndarray] = []
    local_frames: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    risks: list[np.ndarray] = []
    episode_ids: list[int] = []
    metrics: list[EpisodeMetrics] = []
    current_episode_id = 0
    profile = resolve_proficiency(proficiency)

    def record(visible: VisionObservation, action: Action, risk: float) -> None:
        steps = visible.global_frames.shape[0]
        global_frames.append(visible.global_frames.copy())
        local_frames.append(visible.local_frames.copy())
        actions.append(np.full((steps,), action.discrete, dtype=np.int64))
        risks.append(np.full(
            (steps,), _normalized_risk(risk, config.risk_scale), dtype=np.float32,
        ))
        episode_ids.append(current_episode_id)

    def teacher(
        _environment: STGEnvironment,
        _vision: VisionObservation,
        plan: PlanResult | None,
        _memory: EpisodeMemory | None,
    ) -> Action:
        if plan is None:  # The collector always supplies a planner.
            raise RuntimeError("teacher plan is unavailable")
        return planner_teacher_action(
            _environment,
            plan,
            hold_frames=config.decision_interval,
            shield=shield,
        )

    for episode_index, seed in enumerate(seeds):
        current_episode_id = episode_index
        trace = _run_episode(
            environment_factory,
            int(seed),
            planner=planner,
            vision_config=vision_config,
            config=config,
            controller=teacher,
            on_decision=record,
        )
        metrics.append(trace.metrics)

    if not global_frames:
        raise ValueError("at least one seed producing one decision is required")
    memory = None
    if include_scenario_memory:
        scenario_keys = {metric.scenario for metric in metrics}
        if len(scenario_keys) != 1:
            raise ValueError("one demonstration archive cannot mix scenario identities")
        vector = scenario_memory_vector(next(iter(scenario_keys)))
        memory = np.broadcast_to(
            vector,
            (len(global_frames), global_frames[0].shape[0], len(vector)),
        ).copy()
    demonstrations = Demonstrations(
        global_frames=np.stack(global_frames).astype(np.float16, copy=False),
        local_frames=np.stack(local_frames).astype(np.float16, copy=False),
        actions=np.stack(actions),
        risks=np.stack(risks),
        memory=memory,
        proficiency=np.broadcast_to(
            proficiency_vector(profile),
            (
                len(global_frames),
                global_frames[0].shape[0],
                len(proficiency_vector(profile)),
            ),
        ).copy(),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
    )
    demonstrations.validate()
    if output is not None:
        demonstrations.save(output)
    return demonstrations, tuple(metrics)


def evaluate_planner(
    environment_factory: EnvironmentFactory,
    seeds: Iterable[int],
    *,
    planner: SpatioTemporalPlanner | None = None,
    vision_config: VisionConfig = VisionConfig(),
    config: RolloutConfig = RolloutConfig(),
    shield: bool = True,
) -> tuple[EpisodeMetrics, ...]:
    """Evaluate one planner configuration over a deterministic seed set."""

    planner = planner or SpatioTemporalPlanner()

    def controller(
        _environment: STGEnvironment,
        _vision: VisionObservation,
        plan: PlanResult | None,
        _memory: EpisodeMemory | None,
    ) -> Action:
        if plan is None:  # The evaluator always supplies a planner.
            raise RuntimeError("teacher plan is unavailable")
        return planner_teacher_action(
            _environment,
            plan,
            hold_frames=config.decision_interval,
            shield=shield,
        )

    return tuple(
        _run_episode(
            environment_factory,
            int(seed),
            planner=planner,
            vision_config=vision_config,
            config=config,
            controller=controller,
        ).metrics
        for seed in seeds
    )


def _safe_action_endpoints(
    environment: STGEnvironment,
    horizon: int,
) -> dict[int, tuple[float, float]]:
    """Clone each candidate and retain its endpoint if the held move survives."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    scenario = getattr(environment, "scenario", None)
    if bool(getattr(scenario, "forecast_independent_of_player", False)):
        return _batched_safe_action_endpoints(environment, horizon)

    allowed: dict[int, tuple[float, float]] = {}
    for discrete in range(18):
        clone = environment.clone()
        action = Action.from_discrete(discrete)
        safe = True
        for _ in range(horizon):
            if clone.done:
                break
            result = clone._advance(action, build_semantic=False, detect_collision=True)
            if result.outcome is Outcome.HIT:
                safe = False
                break
        if safe:
            allowed[discrete] = (float(clone.player.x), float(clone.player.y))
    return allowed


def _action_endpoint_if_safe(
    environment: STGEnvironment,
    action: Action,
    horizon: int,
) -> tuple[float, float] | None:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    scenario = getattr(environment, "scenario", None)
    if bool(getattr(scenario, "forecast_independent_of_player", False)):
        path = _candidate_player_paths(environment, action, horizon)
        timeline = environment.clone()
        for frame_index in range(horizon):
            if timeline.done:
                break
            timeline._advance(Action(), build_semantic=False, detect_collision=False)
            for threat in timeline.frame_collision_threats:
                if threat.lethal and threat.collides_swept(
                    path[frame_index],
                    path[frame_index + 1],
                    environment.player.radius,
                ):
                    return None
        return path[-1]

    clone = environment.clone()
    for _ in range(horizon):
        if clone.done:
            break
        result = clone._advance(action, build_semantic=False, detect_collision=True)
        if result.outcome is Outcome.HIT:
            return None
    return float(clone.player.x), float(clone.player.y)


def _candidate_player_paths(
    environment: STGEnvironment,
    action: Action,
    horizon: int,
) -> tuple[tuple[float, float], ...]:
    x, y = float(environment.player.x), float(environment.player.y)
    queue = deque(environment._action_queue)
    held = environment._held_action
    remaining = int(environment._hold_remaining)
    path = [(x, y)]
    for _ in range(horizon):
        if remaining <= 0:
            held = action
            remaining = environment.config.action_hold_frames
        remaining -= 1
        queue.append(held)
        applied = queue.popleft()
        dx, dy = float(applied.move_x), float(applied.move_y)
        if dx != 0.0 and dy != 0.0:
            dx *= math.sqrt(0.5)
            dy *= math.sqrt(0.5)
        speed = environment.player.focus_speed if applied.slow else environment.player.speed
        x, y = environment.player_bounds.clamp(x + dx * speed, y + dy * speed)
        path.append((x, y))
    return tuple(path)


def _batched_safe_action_endpoints(
    environment: STGEnvironment,
    horizon: int,
) -> dict[int, tuple[float, float]]:
    paths = {
        discrete: _candidate_player_paths(environment, Action.from_discrete(discrete), horizon)
        for discrete in range(18)
    }
    safe = {discrete: True for discrete in paths}
    timeline = environment.clone()
    for frame_index in range(horizon):
        if timeline.done:
            break
        timeline._advance(Action(), build_semantic=False, detect_collision=False)
        for discrete, path in paths.items():
            if not safe[discrete]:
                continue
            for threat in timeline.frame_collision_threats:
                if threat.lethal and threat.collides_swept(
                    path[frame_index],
                    path[frame_index + 1],
                    environment.player.radius,
                ):
                    safe[discrete] = False
                    break
    return {
        discrete: path[-1]
        for discrete, path in paths.items()
        if safe[discrete]
    }


def imminent_safe_actions(environment: STGEnvironment, horizon: int) -> tuple[int, ...]:
    """Return actions with no swept collision during a short held-action clone."""

    return tuple(_safe_action_endpoints(environment, horizon))


def planner_teacher_action(
    environment: STGEnvironment,
    plan: PlanResult,
    *,
    hold_frames: int,
    shield: bool = True,
) -> Action:
    """Choose the safe held action ending nearest the planner's next waypoint."""

    if not shield:
        return plan.first_action
    if _action_endpoint_if_safe(environment, plan.first_action, hold_frames) is not None:
        return plan.first_action
    endpoints = _safe_action_endpoints(environment, hold_frames)
    if not endpoints:
        return plan.first_action
    steps = tuple(getattr(plan, "steps", ()))
    if len(steps) < 2:
        return Action.from_discrete(min(endpoints))
    frames = tuple(int(value) for value in getattr(getattr(plan, "field", None), "frames", ()))
    if len(frames) == len(steps):
        target_index = min(range(1, len(steps)), key=lambda index: abs(frames[index] - hold_frames))
    else:
        target_index = 1
    target_x, target_y = steps[target_index].position
    selected = min(
        endpoints,
        key=lambda discrete: (
            (endpoints[discrete][0] - target_x) ** 2
            + (endpoints[discrete][1] - target_y) ** 2,
            discrete != plan.first_action.discrete,
            discrete,
        ),
    )
    return Action.from_discrete(selected)


def shield_action_toward(
    environment: STGEnvironment,
    preferred: Action,
    *,
    horizon: int,
) -> Action:
    """Keep a preferred route action unless it collides, then stay nearest it."""

    if _action_endpoint_if_safe(environment, preferred, horizon) is not None:
        return preferred
    endpoints = _safe_action_endpoints(environment, horizon)
    if not endpoints:
        return preferred
    target = _candidate_player_paths(environment, preferred, horizon)[-1]
    selected = min(endpoints, key=lambda discrete: (
        (endpoints[discrete][0] - target[0]) ** 2
        + (endpoints[discrete][1] - target[1]) ** 2,
        discrete,
    ))
    return Action.from_discrete(selected)


def _model_device(model: Any, requested: str) -> str:
    if torch is None:
        raise RuntimeError("PyTorch is required for visual-policy evaluation")
    if requested != "auto":
        return requested
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"


def _policy_logits(
    model: Any,
    visible: VisionObservation,
    *,
    device: str,
    memory: Sequence[float] | np.ndarray | None,
    proficiency: Sequence[float] | np.ndarray | None = None,
    hidden: Any | None = None,
    latest_only: bool = False,
) -> tuple[np.ndarray, Any | None]:
    if torch is None:
        raise RuntimeError("PyTorch is required for visual-policy evaluation")
    global_array = visible.global_frames[-1:] if latest_only else visible.global_frames
    local_array = visible.local_frames[-1:] if latest_only else visible.local_frames
    global_frames = torch.from_numpy(global_array[None]).float().to(device)
    local_frames = torch.from_numpy(local_array[None]).float().to(device)
    memory_tensor = None
    if memory is not None:
        memory_array = np.asarray(memory, dtype=np.float32)
        if memory_array.ndim != 1:
            raise ValueError("policy memory provider must return a one-dimensional vector")
        expected = int(getattr(getattr(model, "config", None), "memory_size", len(memory_array)))
        if len(memory_array) != expected:
            raise ValueError(f"policy memory vector has size {len(memory_array)}; expected {expected}")
        memory_tensor = torch.from_numpy(memory_array[None]).to(device)
    proficiency_tensor = None
    expected_proficiency = int(
        getattr(getattr(model, "config", None), "proficiency_size", 0)
    )
    if expected_proficiency > 0:
        proficiency_array = np.asarray(
            proficiency_vector("expert") if proficiency is None else proficiency,
            dtype=np.float32,
        )
        if proficiency_array.ndim != 1 or len(proficiency_array) != expected_proficiency:
            raise ValueError(
                "policy proficiency vector has size "
                f"{len(proficiency_array)}; expected {expected_proficiency}"
            )
        proficiency_tensor = torch.from_numpy(proficiency_array[None]).to(device)
    with torch.no_grad():
        parameters = inspect.signature(model.forward).parameters
        supports_hidden = "hidden" in parameters
        kwargs = {"hidden": hidden} if supports_hidden else {}
        if proficiency_tensor is not None and "proficiency" in parameters:
            kwargs["proficiency"] = proficiency_tensor
        if supports_hidden:
            logits, _risk, next_hidden = model(
                global_frames, local_frames, memory_tensor, **kwargs,
            )
        else:
            logits, _risk, next_hidden = model(
                global_frames, local_frames, memory_tensor, **kwargs,
            )
            next_hidden = None
    return logits[0, -1].detach().cpu().numpy(), next_hidden


def _policy_inference_mode(model: Any) -> str:
    mode = str(getattr(getattr(model, "config", None), "inference_mode", "stream"))
    if mode not in {"window", "stream"}:
        raise ValueError(f"unsupported policy inference mode: {mode}")
    return mode


def _policy_behavior_action(
    model: Any,
    environment: STGEnvironment | None,
    visible: VisionObservation,
    *,
    device: str,
    memory: Sequence[float] | np.ndarray,
    hidden: Any | None,
    inference_mode: str,
    config: RolloutConfig,
    shield: bool,
    runtime: ProficiencyRuntime | None = None,
    commit_runtime: bool = True,
) -> tuple[Action, Any | None]:
    profile = resolve_proficiency(
        runtime.profile if runtime is not None else "expert"
    )
    if inference_mode == "window":
        logits, next_hidden = _policy_logits(
            model,
            visible,
            device=device,
            memory=memory,
            proficiency=proficiency_vector(profile),
            hidden=None,
            latest_only=False,
        )
        next_hidden = None
    else:
        logits, next_hidden = _policy_logits(
            model,
            visible,
            device=device,
            memory=memory,
            proficiency=proficiency_vector(profile),
            hidden=hidden,
            latest_only=True,
        )
    preferred = (
        runtime.preferred_action(logits, decision_interval=config.decision_interval)
        if runtime is not None else
        Action.from_discrete(int(np.argmax(logits)))
    )

    def finish(action: Action) -> tuple[Action, Any | None]:
        if runtime is not None and commit_runtime:
            runtime.commit(action, decision_interval=config.decision_interval)
        return action, next_hidden

    if not shield:
        return finish(preferred)
    if environment is None:
        raise ValueError("authority-state shields require an environment")
    if runtime is not None and not runtime.should_apply_shield():
        return finish(preferred)
    shield_horizon = min(
        config.shield_horizon,
        profile.prediction_horizon_frames,
    )
    if shield_horizon <= 0:
        return finish(preferred)
    if config.shield_strategy == "toward":
        return finish(shield_action_toward(
            environment,
            preferred,
            horizon=shield_horizon,
        ))
    if _action_endpoint_if_safe(environment, preferred, shield_horizon) is not None:
        return finish(preferred)
    allowed = imminent_safe_actions(environment, shield_horizon)
    return finish(Action.from_discrete(safety_shield(logits, allowed)))


def teacher_action_agreement(
    model: Any,
    demonstrations: Demonstrations,
    *,
    device: str = "auto",
    batch_size: int = 64,
) -> float:
    """Measure action agreement on every labelled timestep in a dataset."""

    if torch is None:
        raise RuntimeError("PyTorch is required for visual-policy evaluation")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    demonstrations.validate()
    resolved = _model_device(model, device)
    model.to(resolved)
    model.eval()
    correct = 0
    total = 0
    memory_size = int(getattr(getattr(model, "config", None), "memory_size", 4))
    with torch.no_grad():
        for start in range(0, len(demonstrations.actions), batch_size):
            end = min(start + batch_size, len(demonstrations.actions))
            global_frames = torch.from_numpy(demonstrations.global_frames[start:end]).float().to(resolved)
            local_frames = torch.from_numpy(demonstrations.local_frames[start:end]).float().to(resolved)
            if demonstrations.memory is None:
                shape = (*demonstrations.actions[start:end].shape, memory_size)
                memory = torch.zeros(shape, dtype=global_frames.dtype, device=resolved)
            else:
                memory = torch.from_numpy(demonstrations.memory[start:end]).float().to(resolved)
            proficiency_size = int(
                getattr(getattr(model, "config", None), "proficiency_size", 0)
            )
            if proficiency_size == 0:
                proficiency = None
            elif demonstrations.proficiency is None:
                values = np.broadcast_to(
                    proficiency_vector("expert"),
                    (*demonstrations.actions[start:end].shape, proficiency_size),
                ).copy()
                proficiency = torch.from_numpy(values).float().to(resolved)
            else:
                proficiency = torch.from_numpy(
                    demonstrations.proficiency[start:end]
                ).float().to(resolved)
            parameters = inspect.signature(model.forward).parameters
            kwargs = (
                {"proficiency": proficiency}
                if proficiency is not None and "proficiency" in parameters else
                {}
            )
            logits, _risk, _hidden = model(
                global_frames, local_frames, memory, **kwargs,
            )
            labels = torch.from_numpy(demonstrations.actions[start:end]).long().to(resolved)
            if demonstrations.supervision_mask is None:
                mask = torch.zeros_like(labels, dtype=torch.bool)
                mask[:, -1] = True
            else:
                mask = torch.from_numpy(
                    demonstrations.supervision_mask[start:end]
                ).bool().to(resolved)
            correct += int((logits.argmax(dim=-1)[mask] == labels[mask]).sum().item())
            total += int(mask.sum())
    return correct / total if total else 0.0


def evaluate_policy(
    model: Any,
    environment_factory: EnvironmentFactory,
    seeds: Iterable[int],
    *,
    planner: SpatioTemporalPlanner | None = None,
    vision_config: VisionConfig = VisionConfig(),
    config: RolloutConfig = RolloutConfig(),
    shield: bool = False,
    device: str = "auto",
    memory_provider: MemoryProvider | None = None,
    proficiency: str | PlayerProficiencyProfile = "expert",
) -> tuple[EpisodeMetrics, ...]:
    """Run a delayed visual policy, with an optional imminent-collision shield."""

    resolved = _model_device(model, device)
    model.to(resolved)
    model.eval()
    inference_mode = _policy_inference_mode(model)
    hidden: Any | None = None
    runtime: ProficiencyRuntime | None = None
    previous_action: Action | None = None
    episode_started = False
    profile = resolve_proficiency(proficiency)

    def controller(
        environment: STGEnvironment,
        visible: VisionObservation,
        _plan: PlanResult | None,
        _memory: EpisodeMemory | None,
    ) -> Action:
        nonlocal hidden, episode_started, runtime, previous_action
        first_decision = not episode_started
        if first_decision:
            hidden = None
            previous_action = None
            episode_started = True
        scenario_key = str(getattr(environment.scenario, "scenario_key", environment.scenario.name))
        if first_decision and memory_provider is not None:
            reset_provider = getattr(memory_provider, "reset", None)
            if reset_provider is not None:
                reset_provider(scenario_key, visible)
        vector = (
            memory_provider(scenario_key, visible)
            if memory_provider is not None
            else policy_context_vector(model, scenario_key, previous_action)
        )
        action, hidden = _policy_behavior_action(
            model,
            environment if shield else None,
            visible,
            device=resolved,
            memory=vector,
            hidden=hidden,
            inference_mode=inference_mode,
            config=config,
            shield=shield,
            runtime=runtime,
        )
        previous_action = action
        return action

    results = []
    for seed in seeds:
        hidden = None
        previous_action = None
        runtime = ProficiencyRuntime(profile, seed=int(seed))
        episode_started = False
        results.append(_run_episode(
            environment_factory,
            int(seed),
            planner=planner,
            vision_config=vision_config,
            config=config,
            controller=controller,
        ).metrics)
    return tuple(results)


def collect_dagger_demonstrations(
    model: Any,
    environment_factory: EnvironmentFactory,
    seeds: Iterable[int],
    *,
    planner: SpatioTemporalPlanner | None = None,
    vision_config: VisionConfig = VisionConfig(),
    config: RolloutConfig = RolloutConfig(),
    device: str = "auto",
    shield: bool = True,
    teacher_shield: bool = True,
    proficiency: str | PlayerProficiencyProfile = "expert",
    output: str | Path | None = None,
) -> tuple[Demonstrations, tuple[EpisodeMetrics, ...]]:
    """Label policy-visited states with exact-state planner actions."""

    planner = planner or SpatioTemporalPlanner()
    resolved = _model_device(model, device)
    model.to(resolved)
    model.eval()
    inference_mode = _policy_inference_mode(model)
    global_frames: list[np.ndarray] = []
    local_frames: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    risks: list[np.ndarray] = []
    memories: list[np.ndarray] = []
    proficiencies: list[np.ndarray] = []
    episode_ids: list[int] = []
    metrics: list[EpisodeMetrics] = []
    current_episode_id = 0
    pending_teacher: Action | None = None
    pending_memory: np.ndarray | None = None
    hidden: Any | None = None
    runtime: ProficiencyRuntime | None = None
    previous_action: Action | None = None
    decisions = 0
    overrides = 0

    def behavior(
        environment: STGEnvironment,
        visible: VisionObservation,
        plan: PlanResult | None,
        _memory: EpisodeMemory | None,
    ) -> Action:
        nonlocal pending_teacher, pending_memory, hidden, decisions, overrides, runtime
        nonlocal previous_action
        if plan is None:
            raise RuntimeError("DAgger teacher plan is unavailable")
        if environment.frame == 0:
            hidden = None
            previous_action = None
        pending_teacher = planner_teacher_action(
            environment,
            plan,
            hold_frames=config.decision_interval,
            shield=teacher_shield,
        )
        scenario_key = str(getattr(environment.scenario, "scenario_key", environment.scenario.name))
        pending_memory = policy_context_vector(
            model,
            scenario_key,
            previous_action,
        )
        selected, hidden = _policy_behavior_action(
            model,
            environment,
            visible,
            device=resolved,
            memory=pending_memory,
            hidden=hidden,
            inference_mode=inference_mode,
            config=config,
            shield=shield,
            runtime=runtime,
        )
        decisions += 1
        overrides += int(selected != pending_teacher)
        previous_action = selected
        return selected

    def record(visible: VisionObservation, _behavior: Action, risk: float) -> None:
        if pending_teacher is None or pending_memory is None:
            raise RuntimeError("DAgger decision was recorded before teacher labelling")
        steps = visible.global_frames.shape[0]
        global_frames.append(visible.global_frames.copy())
        local_frames.append(visible.local_frames.copy())
        actions.append(np.full((steps,), pending_teacher.discrete, dtype=np.int64))
        risks.append(np.full(
            (steps,), _normalized_risk(risk, config.risk_scale), dtype=np.float32,
        ))
        memories.append(np.broadcast_to(pending_memory, (steps, len(pending_memory))).copy())
        vector = proficiency_vector(proficiency)
        proficiencies.append(np.broadcast_to(vector, (steps, len(vector))).copy())
        episode_ids.append(current_episode_id)

    for episode_index, seed in enumerate(seeds):
        current_episode_id = episode_index
        pending_teacher = None
        pending_memory = None
        hidden = None
        runtime = ProficiencyRuntime(proficiency, seed=int(seed))
        decisions = 0
        overrides = 0
        trace = _run_episode(
            environment_factory,
            int(seed),
            planner=planner,
            vision_config=vision_config,
            config=config,
            controller=behavior,
            on_decision=record,
        )
        metrics.append(replace(
            trace.metrics,
            action_agreement=(decisions - overrides) / decisions if decisions else None,
            teacher_overrides=overrides,
        ))

    if not global_frames:
        raise ValueError("at least one seed producing one DAgger decision is required")
    demonstrations = Demonstrations(
        global_frames=np.stack(global_frames).astype(np.float16, copy=False),
        local_frames=np.stack(local_frames).astype(np.float16, copy=False),
        actions=np.stack(actions),
        risks=np.stack(risks),
        memory=np.stack(memories).astype(np.float32, copy=False),
        proficiency=np.stack(proficiencies).astype(np.float32, copy=False),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
    )
    demonstrations.validate()
    if output is not None:
        demonstrations.save(output)
    return demonstrations, tuple(metrics)


def evaluate_policy_variants(
    model: Any,
    environment_factory: EnvironmentFactory,
    seeds: Iterable[int],
    **kwargs: Any,
) -> PolicyEvaluation:
    """Evaluate raw and shielded policies on exactly the same seed tuple."""

    seed_tuple = tuple(int(seed) for seed in seeds)
    raw = evaluate_policy(model, environment_factory, seed_tuple, shield=False, **kwargs)
    shielded = evaluate_policy(model, environment_factory, seed_tuple, shield=True, **kwargs)
    return PolicyEvaluation(raw=raw, shielded=shielded)


def observable_cue(observation: Observation) -> dict[str, Any]:
    """Build a stable cue from visible references without hidden script timers."""

    references = []
    for threat in observation.threats:
        if not threat.visible or not (threat.warning or threat.radius >= 10.0):
            continue
        references.append({
            "x": round(threat.x, 1),
            "y": round(threat.y, 1),
            "radius": round(threat.radius, 1),
            "vx": round(threat.vx, 2),
            "vy": round(threat.vy, 2),
            "warning": threat.warning,
        })
    references.sort(key=lambda value: (value["x"], value["y"], value["radius"]))
    return {"references": references}


def benchmark_second_attempt(
    environment_factory: EnvironmentFactory,
    seed: int,
    *,
    first_controller: RolloutController,
    second_controller: RolloutController,
    route_builder: Callable[[RolloutTrace], Iterable[Any]],
    planner: SpatioTemporalPlanner | None = None,
    memory_store: EpisodicMemory | None = None,
    cue_builder: CueBuilder = observable_cue,
    vision_config: VisionConfig = VisionConfig(),
    config: RolloutConfig = RolloutConfig(),
    trigger_lead: int = 90,
    confidence: float = 0.6,
    minimum_similarity: float = 0.0,
    minimum_risk_improvement: float = 0.30,
) -> MemoryBenchmarkResult:
    """Record a failed route externally, recall it, and evaluate attempt two."""

    if trigger_lead < 0:
        raise ValueError("trigger_lead cannot be negative")
    if not 0.0 <= minimum_risk_improvement <= 1.0:
        raise ValueError("minimum_risk_improvement must be in [0, 1]")
    store = memory_store if memory_store is not None else EpisodicMemory()
    owns_store = memory_store is None
    try:
        first = _run_episode(
            environment_factory,
            int(seed),
            planner=planner,
            vision_config=vision_config,
            config=config,
            controller=first_controller,
        )
        cue = cue_builder(first.initial_observation)
        player = first.final_observation.player
        death_point = (
            (player.x, player.y, first.final_observation.frame)
            if first.metrics.deaths else None
        )
        saved = store.remember(
            first.metrics.scenario,
            cue,
            death_point=death_point,
            trigger_lead=trigger_lead,
            route=route_builder(first),
            confidence=confidence,
        )
        recalled = store.best(
            first.metrics.scenario,
            cue,
            minimum_similarity=minimum_similarity,
        )
        if recalled is None:  # Defensive: the just-written exact cue must match.
            raise RuntimeError("episodic memory could not recall the recorded attempt")
        second = _run_episode(
            environment_factory,
            int(seed),
            planner=planner,
            vision_config=vision_config,
            config=config,
            controller=second_controller,
            memory=recalled,
        )
        updated = store.update_outcome(saved.id, success=second.metrics.survived)
        return MemoryBenchmarkResult(
            first=first.metrics,
            second=second.metrics,
            memory=updated,
            minimum_risk_improvement=minimum_risk_improvement,
        )
    finally:
        if owns_store:
            store.close()


__all__ = [
    "CueBuilder",
    "EnvironmentFactory",
    "MemoryBenchmarkResult",
    "MemoryProvider",
    "PolicyEvaluation",
    "RolloutConfig",
    "RolloutController",
    "RolloutTrace",
    "benchmark_second_attempt",
    "collect_demonstrations",
    "collect_dagger_demonstrations",
    "evaluate_planner",
    "evaluate_policy",
    "evaluate_policy_variants",
    "imminent_safe_actions",
    "load_demonstrations",
    "observable_cue",
    "planner_teacher_action",
    "save_demonstrations",
    "scenario_memory_vector",
    "shield_action_toward",
    "survival_rate",
    "teacher_action_agreement",
]
