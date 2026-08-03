"""Visible-only controller runner for a live LuaSTG spell-practice attack."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from .adapters import adapt_engine_observation
from .engine import EngineClient, EngineProtocolError
from .engine_runtime import verify_runtime_source_fingerprints
from .engine_vision import EngineStreamVision
from .memory import EpisodicMemory
from .native_dataset import risk_from_clearance
from .policy import (
    PlayerProficiencyProfile,
    ProficiencyRuntime,
    resolve_proficiency,
)
from .protocol import Action
from .provenance import file_sha256, source_tree_sha256
from .route_memory import (
    ExternalRouteController,
    ExternalRouteLibraryController,
    RouteControllerConfig,
    load_route_artifact,
    load_route_library_artifact,
    validate_memory_route,
)
from .sim import coerce_action
from .vision import DelayedVision, VisionConfig, VisionObservation


class VisibleController(Protocol):
    def reset(self) -> None: ...

    def select(self, visible: VisionObservation) -> Action: ...


EnginePlayDecisionObserver = Callable[[VisionObservation, Action, float], None]


@dataclass(frozen=True, slots=True)
class EnginePlayConfig:
    """Configuration whose control-facing inputs remain delayed and visible."""

    max_frames: int = 7200
    decision_interval: int = 3
    vision: VisionConfig = VisionConfig()
    # Legacy reporting-only threat diagnostic. These values no longer gate
    # shooting because firing has no movement or collision side effect.
    shoot_gate_radius: float = 20.0
    shoot_risk_threshold: float = 0.25
    shoot_motion_weight: float = 0.5
    render: bool = False
    render_every: int = 1
    visible_safety_shield: bool = False
    visible_safety_horizon: int | None = None
    visible_safety_minimum_margin: float = 6.0

    def __post_init__(self) -> None:
        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if self.decision_interval != 3:
            raise ValueError("live demonstration actions must be held for exactly three frames")
        if not 0.0 < self.shoot_gate_radius <= min(
            self.vision.local_extent_x,
            self.vision.local_extent_y,
        ):
            raise ValueError("shoot_gate_radius must fit inside the local visible region")
        if self.shoot_risk_threshold < 0.0:
            raise ValueError("shoot_risk_threshold cannot be negative")
        if self.shoot_motion_weight < 0.0:
            raise ValueError("shoot_motion_weight cannot be negative")
        if (
            self.visible_safety_horizon is not None
            and (
                isinstance(self.visible_safety_horizon, bool)
                or not isinstance(self.visible_safety_horizon, int)
                or self.visible_safety_horizon <= 0
            )
        ):
            raise ValueError("visible_safety_horizon must be a positive integer or None")
        if (
            not math.isfinite(self.visible_safety_minimum_margin)
            or self.visible_safety_minimum_margin < 0.0
        ):
            raise ValueError("visible_safety_minimum_margin must be finite and nonnegative")
        if (
            isinstance(self.render_every, bool)
            or not isinstance(self.render_every, int)
            or not 1 <= self.render_every <= 600
        ):
            raise ValueError("render_every must be an integer in [1, 600]")


@dataclass(frozen=True, slots=True)
class ShootGateSample:
    safe: bool
    risk: float
    occupancy_peak: float
    occupancy_mass: float


@dataclass(frozen=True, slots=True)
class VisibleSafetySample:
    action: Action
    intervened: bool
    preferred_margin: float | None
    selected_margin: float | None
    threat_pixels: int


def _held_action_displacement(action: Action, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speed = 2.0 if action.slow else 4.0
    length = math.hypot(action.move_x, action.move_y)
    if length == 0.0:
        return np.zeros_like(frames), np.zeros_like(frames)
    scale = speed / length
    held = np.minimum(frames, 3.0)
    return held * action.move_x * scale, held * action.move_y * scale


def visible_safety_action(
    preferred: Action,
    visible: VisionObservation,
    vision_config: VisionConfig,
    *,
    horizon: int = 12,
    minimum_margin: float = 6.0,
) -> VisibleSafetySample:
    """Apply short-horizon local avoidance using only the semantic vision tensor."""

    local = np.asarray(visible.local_frames[-1], dtype=np.float32)
    if local.ndim != 3 or local.shape[0] < 4:
        raise ValueError("local semantic frame must have at least four channels")
    occupancy = np.maximum(local[0], local[3])
    mask = occupancy > 1e-3
    threat_pixels = int(np.count_nonzero(mask))
    if threat_pixels == 0:
        return VisibleSafetySample(preferred, False, None, None, 0)

    height, width = occupancy.shape
    xs = np.linspace(
        -vision_config.local_extent_x,
        vision_config.local_extent_x,
        width,
        dtype=np.float32,
    )
    ys = np.linspace(
        -vision_config.local_extent_y,
        vision_config.local_extent_y,
        height,
        dtype=np.float32,
    )
    rows, columns = np.nonzero(mask)
    threat_x = xs[columns]
    threat_y = ys[rows]
    threat_vx = local[1][mask] * vision_config.velocity_scale
    threat_vy = local[2][mask] * vision_config.velocity_scale
    cell_radius = 0.5 * math.hypot(
        2.0 * vision_config.local_extent_x / max(1, width - 1),
        2.0 * vision_config.local_extent_y / max(1, height - 1),
    )
    frames = np.arange(1, horizon + 1, dtype=np.float32)
    observed_ahead = frames + float(vision_config.observation_delay)
    future_x = threat_x[:, None] + threat_vx[:, None] * observed_ahead[None, :]
    future_y = threat_y[:, None] + threat_vy[:, None] * observed_ahead[None, :]

    margins: dict[int, float] = {}
    actions = tuple(Action.from_discrete(value, shoot=preferred.shoot) for value in range(18))
    for action in actions:
        player_x, player_y = _held_action_displacement(action, frames)
        distances = np.hypot(
            future_x - player_x[None, :],
            future_y - player_y[None, :],
        )
        margins[action.discrete] = float(np.min(distances) - cell_radius - 1.0)

    preferred_margin = margins[preferred.discrete]
    if preferred_margin >= minimum_margin:
        return VisibleSafetySample(
            preferred,
            False,
            preferred_margin,
            preferred_margin,
            threat_pixels,
        )

    safe = [action for action in actions if margins[action.discrete] >= minimum_margin]
    if safe:
        def change_cost(action: Action) -> tuple[float, float]:
            cost = (
                abs(action.move_x - preferred.move_x)
                + abs(action.move_y - preferred.move_y)
                + 0.25 * int(action.slow != preferred.slow)
            )
            return cost, -margins[action.discrete]

        selected = min(safe, key=change_cost)
    else:
        selected = max(actions, key=lambda action: margins[action.discrete])
    selected = Action(
        move_x=selected.move_x,
        move_y=selected.move_y,
        slow=selected.slow,
        shoot=preferred.shoot,
        spell=preferred.spell,
    )
    return VisibleSafetySample(
        selected,
        selected.discrete != preferred.discrete,
        preferred_margin,
        margins[selected.discrete],
        threat_pixels,
    )


class VisualPolicyController:
    """Two-phase live wrapper whose motor state follows the executed action."""

    def __init__(
        self,
        model: Any,
        scenario_key: str,
        *,
        device: str = "cpu",
        proficiency: str | PlayerProficiencyProfile = "expert",
        seed: int = 0,
        scenario_vocabulary: tuple[str, ...] | None = None,
    ) -> None:
        from .rollout import _model_device, _policy_inference_mode, scenario_memory_vector

        self.model = model
        self.scenario_key = str(scenario_key)
        self.device = _model_device(model, device)
        self.inference_mode = _policy_inference_mode(model)
        self.proficiency = resolve_proficiency(proficiency)
        self.seed = int(seed)
        self.runtime = ProficiencyRuntime(self.proficiency, seed=self.seed)
        memory_size = int(getattr(getattr(model, "config", None), "memory_size", 4))
        inherited_vocabulary = getattr(model, "scenario_vocabulary", None)
        self.scenario_vocabulary = (
            tuple(scenario_vocabulary)
            if scenario_vocabulary is not None else
            inherited_vocabulary
        )
        self.previous_action_size = int(
            getattr(model, "previous_action_size", 0)
        )
        self.previous_action_offset = int(
            getattr(
                model,
                "previous_action_offset",
                len(self.scenario_vocabulary or ()),
            )
        )
        if self.previous_action_size not in {0, 18}:
            raise ValueError("previous action context must have 0 or 18 entries")
        if (
            self.previous_action_offset + self.previous_action_size
            > memory_size
        ):
            raise ValueError("previous action context does not fit policy memory")
        identity_size = (
            self.previous_action_offset
            if self.previous_action_size else
            memory_size
        )
        if identity_size > 0 and self.scenario_vocabulary is None:
            raise ValueError(
                "live policy identity context requires a checkpoint-declared "
                "scenario vocabulary"
            )
        self.memory = scenario_memory_vector(
            self.scenario_key,
            memory_size,
            self.scenario_vocabulary,
        )
        self.model.to(self.device)
        self.model.eval()
        self.reset()

    def reset(self) -> None:
        self.hidden: Any | None = None
        self.decisions = 0
        self.runtime.reset(self.seed)
        if self.previous_action_size:
            start = self.previous_action_offset
            self.memory[start:start + self.previous_action_size] = 0.0

    def reset_for_seed(self, seed: int) -> None:
        self.seed = int(seed)
        self.reset()

    def select(self, visible: VisionObservation) -> Action:
        from .rollout import RolloutConfig, _policy_behavior_action

        action, self.hidden = _policy_behavior_action(
            self.model,
            None,
            visible,
            device=self.device,
            memory=self.memory,
            hidden=self.hidden,
            inference_mode=self.inference_mode,
            config=RolloutConfig(decision_interval=3),
            shield=False,
            runtime=self.runtime,
            commit_runtime=False,
        )
        self.decisions += 1
        return action

    def commit_executed_action(self, action: Action, *, frames: int) -> None:
        self.runtime.commit(action, decision_interval=frames)
        previous_action_size = int(getattr(self, "previous_action_size", 0))
        if previous_action_size:
            start = self.previous_action_offset
            self.memory[start:start + previous_action_size] = 0.0
            self.memory[start + action.discrete] = 1.0


def _commit_executed_action(
    controller: VisibleController,
    action: Action,
    *,
    frames: int,
) -> None:
    """Commit optional two-phase controller state after native movement."""

    commit = getattr(controller, "commit_executed_action", None)
    if callable(commit):
        commit(action, frames=frames)


def _stream_policy_control_inputs(controller: VisibleController) -> list[str]:
    """Describe every explicit input consumed by a streaming policy."""

    inputs = [
        "latest_delayed_global_semantic_frame",
        "latest_delayed_local_semantic_frame",
        "current_visible_player_pose",
        "recurrent_hidden_state",
    ]
    if getattr(controller, "scenario_vocabulary", None) is not None:
        inputs.append("registered_episode_identity_one_hot")
    if int(getattr(controller, "previous_action_size", 0)) > 0:
        inputs.append("previous_executed_motor_action_one_hot")
    return inputs


def load_route_controller(
    route_artifact: str | Path,
    memory_database: str | Path,
    memory_id: int,
    *,
    exhaustion: str = "hold_last",
) -> tuple[ExternalRouteController, dict[str, Any]]:
    artifact = load_route_artifact(route_artifact)
    if artifact.decision_interval != 3:
        raise ValueError("live route artifacts must use a three-frame decision interval")
    with EpisodicMemory(memory_database, readonly=True) as store:
        memory = store.get(memory_id)
    validate_memory_route(artifact, memory)
    controller = ExternalRouteController(
        memory,
        config=RouteControllerConfig(
            shield=False,
            exhaustion=exhaustion,
            route_origin="episode",
        ),
    )
    return controller, {
        "kind": "external_route",
        "route_artifact": str(Path(route_artifact)),
        "route_artifact_sha256": file_sha256(route_artifact),
        "route_id": artifact.route_id,
        "route_scenario": artifact.scenario,
        "memory_database": str(Path(memory_database)),
        "memory_database_sha256": file_sha256(memory_database),
        "memory_ids": [memory.id],
    }


def load_route_library_controller(
    library_artifact: str | Path,
    memory_database: str | Path,
    *,
    exhaustion: str = "hold_last",
) -> tuple[ExternalRouteLibraryController, dict[str, Any]]:
    artifact = load_route_library_artifact(library_artifact)
    with EpisodicMemory(memory_database, readonly=True) as store:
        memories = tuple(store.get(memory_id) for memory_id in artifact.memory_ids)
    if any(memory.scenario != artifact.scenario for memory in memories):
        raise ValueError("route-library memory scenario does not match its artifact")
    controller = ExternalRouteLibraryController(
        memories,
        config=RouteControllerConfig(
            shield=False,
            exhaustion=exhaustion,
            route_origin="episode",
        ),
    )
    return controller, {
        "kind": "external_route_library",
        "library_artifact": str(Path(library_artifact)),
        "library_artifact_sha256": file_sha256(library_artifact),
        "library_id": artifact.library_id,
        "route_scenario": artifact.scenario,
        "memory_database": str(Path(memory_database)),
        "memory_database_sha256": file_sha256(memory_database),
        "memory_ids": list(artifact.memory_ids),
    }


def _observation(response: Mapping[str, Any]) -> Mapping[str, Any]:
    observation = response.get("observation")
    if not isinstance(observation, Mapping):
        raise EngineProtocolError("engine response has no observation object")
    return observation


def _episode_frame(observation: Mapping[str, Any]) -> int | None:
    value = observation.get("episode_frame")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _catalog_entry(response: Mapping[str, Any], scenario: str, attack: int) -> dict[str, Any]:
    catalog = response.get("catalog")
    if not isinstance(catalog, Mapping) or not isinstance(catalog.get("attacks"), list):
        raise EngineProtocolError("engine response has no live attack catalog")
    matches = [
        item for item in catalog["attacks"]
        if isinstance(item, Mapping)
        and item.get("scenario") == scenario
        and item.get("attack") == attack
    ]
    if len(matches) != 1:
        raise EngineProtocolError(
            f"live catalog does not contain exactly one {scenario} attack {attack}",
        )
    return dict(matches[0])


def _catalog_stage(response: Mapping[str, Any], stage: str) -> dict[str, Any]:
    catalog = response.get("catalog")
    if not isinstance(catalog, Mapping) or not isinstance(catalog.get("stages"), list):
        raise EngineProtocolError("engine response has no live stage catalog")
    matches = [
        item for item in catalog["stages"]
        if isinstance(item, Mapping) and item.get("stage") == stage
    ]
    if len(matches) != 1:
        raise EngineProtocolError(
            f"live catalog does not contain exactly one stage {stage!r}",
        )
    return dict(matches[0])


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


class _OutcomeTrace:
    """Collect authority-only evidence that is never passed to the controller."""

    _COUNT_NAMES = (
        "enemy_bullets",
        "enemies",
        "nontjt_enemies",
        "indestructibles",
        "lasers",
    )

    def __init__(self) -> None:
        self.peak_counts = {name: 0 for name in self._COUNT_NAMES}
        self.player_path_distance = 0.0
        self.player_bounds: list[float] | None = None
        self._last_player: tuple[float, float] | None = None
        self.boss_hp_initial: float | None = None
        self.boss_hp_last: float | None = None
        self.boss_hp_minimum: float | None = None

    def push(self, observation: Mapping[str, Any]) -> None:
        counts = observation.get("counts")
        for name in self._COUNT_NAMES:
            value = counts.get(name) if isinstance(counts, Mapping) else None
            if isinstance(value, bool) or not isinstance(value, int):
                records = observation.get(name)
                value = len(records) if isinstance(records, list) else 0
            self.peak_counts[name] = max(self.peak_counts[name], value)

        player = observation.get("player")
        if isinstance(player, Mapping):
            x = _finite_number(player.get("x"))
            y = _finite_number(player.get("y"))
            if x is not None and y is not None:
                if self._last_player is not None:
                    self.player_path_distance += math.hypot(
                        x - self._last_player[0], y - self._last_player[1],
                    )
                self._last_player = (x, y)
                if self.player_bounds is None:
                    self.player_bounds = [x, x, y, y]
                else:
                    self.player_bounds[0] = min(self.player_bounds[0], x)
                    self.player_bounds[1] = max(self.player_bounds[1], x)
                    self.player_bounds[2] = min(self.player_bounds[2], y)
                    self.player_bounds[3] = max(self.player_bounds[3], y)

        hp_values = []
        for name in ("enemies", "nontjt_enemies"):
            records = observation.get(name)
            if not isinstance(records, list):
                continue
            for record in records:
                hp = _finite_number(record.get("hp")) if isinstance(record, Mapping) else None
                # SR uses very large sentinel HP values for indestructible
                # helpers that share the enemy group with the actual boss.
                if hp is not None and 0.0 <= hp < 100_000_000.0:
                    hp_values.append(hp)
        if hp_values:
            hp = max(hp_values)
            if self.boss_hp_initial is None:
                self.boss_hp_initial = hp
            self.boss_hp_last = hp
            self.boss_hp_minimum = hp if self.boss_hp_minimum is None else min(
                self.boss_hp_minimum, hp,
            )

    def report(self, final: Mapping[str, Any]) -> dict[str, Any]:
        bounds = None
        if self.player_bounds is not None:
            bounds = {
                "min_x": self.player_bounds[0],
                "max_x": self.player_bounds[1],
                "min_y": self.player_bounds[2],
                "max_y": self.player_bounds[3],
            }
        player = final.get("player")
        final_player = None
        if isinstance(player, Mapping):
            final_player = {
                name: player.get(name)
                for name in ("x", "y", "death", "protect", "status")
            }
        return {
            "reporting_only_not_controller_input": True,
            "peak_counts": self.peak_counts,
            "player_path_distance": self.player_path_distance,
            "player_bounds": bounds,
            "boss_hp_initial": self.boss_hp_initial,
            "boss_hp_last_observed": self.boss_hp_last,
            "boss_hp_minimum_observed": self.boss_hp_minimum,
            "final_player": final_player,
            "final_stage": dict(final.get("stage") or {}),
        }


def visible_shoot_gate(
    visible: VisionObservation,
    config: EnginePlayConfig,
) -> ShootGateSample:
    """Report local threat intensity without controlling the shoot button."""

    frames = np.asarray(visible.local_frames, dtype=np.float32)
    if frames.ndim != 4 or frames.shape[0] == 0 or frames.shape[1] < 3:
        raise ValueError("local semantic frames must have [time, channel, height, width]")
    latest = frames[-1]
    height, width = latest.shape[-2:]
    xs = np.linspace(
        -config.vision.local_extent_x,
        config.vision.local_extent_x,
        width,
        dtype=np.float32,
    )
    ys = np.linspace(
        -config.vision.local_extent_y,
        config.vision.local_extent_y,
        height,
        dtype=np.float32,
    )
    grid_x, grid_y = np.meshgrid(xs, ys)
    region = grid_x * grid_x + grid_y * grid_y <= config.shoot_gate_radius ** 2
    occupancy = np.asarray(latest[0], dtype=np.float32)[region]
    motion = np.hypot(latest[1], latest[2])[region]
    weighted = occupancy * (1.0 + config.shoot_motion_weight * motion)
    risk = float(np.max(weighted, initial=0.0))
    peak = float(np.max(occupancy, initial=0.0))
    mass = float(np.sum(occupancy, dtype=np.float64))
    return ShootGateSample(
        safe=risk <= config.shoot_risk_threshold,
        risk=risk,
        occupancy_peak=peak,
        occupancy_mass=mass,
    )


def _effective_action(preferred: Action) -> Action:
    return Action(
        move_x=preferred.move_x,
        move_y=preferred.move_y,
        slow=preferred.slow,
        shoot=True,
        # The live dodge demonstration does not use bombs as a survival aid.
        spell=False,
    )


def _controller_state(controller: VisibleController) -> dict[str, Any]:
    if isinstance(controller, ExternalRouteController):
        return {
            "triggered": controller.triggered,
            "trigger_decision": controller.trigger_decision,
            "trigger_source_frame": controller.trigger_source_frame,
            "decisions": controller.decisions,
            "memory_id": controller.memory.id,
        }
    if isinstance(controller, ExternalRouteLibraryController):
        selected = controller.selected_memory
        return {
            "selected_memory_id": None if selected is None else selected.id,
            "selection_decision": controller.selection_decision,
            "selection_source_frame": controller.selection_source_frame,
            "decisions": controller.decisions,
        }
    if isinstance(controller, VisualPolicyController):
        return {
            "scenario_key": controller.scenario_key,
            "device": controller.device,
            "inference_mode": controller.inference_mode,
            "proficiency": asdict(controller.proficiency),
            "proficiency_seed": controller.seed,
            "decisions": controller.decisions,
        }
    return {"type": type(controller).__name__}


def _visible_safety_proficiency(
    controller: VisibleController,
    config: EnginePlayConfig,
) -> tuple[int, float, ProficiencyRuntime | None, str | None]:
    """Resolve shield execution limits without consulting engine authority state."""

    profile = getattr(controller, "proficiency", None)
    runtime = getattr(controller, "runtime", None)
    if isinstance(profile, PlayerProficiencyProfile) and isinstance(
        runtime, ProficiencyRuntime,
    ):
        horizon = profile.prediction_horizon_frames
        if config.visible_safety_horizon is not None:
            horizon = min(horizon, config.visible_safety_horizon)
        return horizon, profile.shield_probability, runtime, profile.name
    return (
        12 if config.visible_safety_horizon is None else config.visible_safety_horizon,
        1.0,
        None,
        None,
    )


def run_engine_play(
    client: EngineClient,
    *,
    scenario: str,
    attack: int | None,
    seed: int,
    player: str,
    controller: VisibleController,
    controller_metadata: Mapping[str, Any] | None = None,
    config: EnginePlayConfig = EnginePlayConfig(),
    decision_observer: EnginePlayDecisionObserver | None = None,
    stage: str | None = None,
) -> dict[str, Any]:
    """Run one native episode; completion plus explicit zero death passes."""

    if stage is None:
        if attack is None or attack <= 0:
            raise ValueError("attack must be positive")
        episode_kind = "attack"
        completion_reason = "attack_complete"
    else:
        if not stage.strip():
            raise ValueError("stage must be a nonempty string")
        if attack is not None:
            raise ValueError("attack must be None for a stage episode")
        episode_kind = "stage"
        completion_reason = "stage_complete"
    ping = client.ping()
    runtime_source_verification = verify_runtime_source_fingerprints(ping)
    catalog_response = client.catalog()
    if stage is None:
        assert attack is not None
        catalog_entry = _catalog_entry(catalog_response, scenario, attack)
        reset_response = client.reset(
            scenario,
            attack,
            seed=int(seed),
            player=player,
            options={},
        )
    else:
        catalog_entry = _catalog_stage(catalog_response, stage)
        reset_response = client.reset_stage(
            stage,
            seed=int(seed),
            player=player,
            options={},
        )
    raw = _observation(reset_response)
    initial_episode_frame = _episode_frame(raw)
    stream_vision = (
        EngineStreamVision(config.vision)
        if getattr(controller, "inference_mode", None) == "stream" else
        None
    )
    vision = None if stream_vision is not None else DelayedVision(config=config.vision)
    visible = (
        stream_vision.reset(raw)
        if stream_vision is not None else
        vision.reset(adapt_engine_observation(reset_response))
    )
    reset_for_seed = getattr(controller, "reset_for_seed", None)
    if reset_for_seed is None:
        controller.reset()
    else:
        reset_for_seed(int(seed))
    display_response = client.set_rendering(config.render, every=config.render_every)
    if display_response.get("render") is not config.render:
        raise EngineProtocolError("engine did not apply the requested display state")
    if display_response.get("every") != config.render_every:
        raise EngineProtocolError("engine did not apply the requested display interval")
    outcome_trace = _OutcomeTrace()
    outcome_trace.push(raw)

    action_steps: list[dict[str, Any]] = []
    logical_frames = 0
    shot_frames = 0
    safe_decisions = 0
    visible_safety_interventions = 0
    visible_safety_checks = 0
    visible_safety_probability_skips = 0
    (
        visible_safety_horizon,
        visible_safety_probability,
        visible_safety_runtime,
        visible_safety_profile,
    ) = _visible_safety_proficiency(controller, config)
    while raw.get("terminated") is not True and logical_frames < config.max_frames:
        model_action = coerce_action(controller.select(visible))
        safety_gate_open = (
            config.visible_safety_shield
            and visible_safety_horizon > 0
            and (
                visible_safety_runtime is None
                or visible_safety_runtime.should_apply_shield()
            )
        )
        visible_safety_checks += int(safety_gate_open)
        visible_safety_probability_skips += int(
            config.visible_safety_shield
            and visible_safety_horizon > 0
            and not safety_gate_open
        )
        safety = (
            visible_safety_action(
                model_action,
                visible,
                config.vision,
                horizon=visible_safety_horizon,
                minimum_margin=config.visible_safety_minimum_margin,
            )
            if safety_gate_open else
            VisibleSafetySample(model_action, False, None, None, 0)
        )
        preferred = safety.action
        visible_safety_interventions += int(safety.intervened)
        gate = visible_shoot_gate(visible, config)
        action = _effective_action(preferred)
        if decision_observer is not None:
            decision_observer(
                visible,
                action,
                risk_from_clearance(
                    math.inf
                    if safety.selected_margin is None else
                    safety.selected_margin
                ),
            )
        safe_decisions += int(gate.safe)
        requested = min(config.decision_interval, config.max_frames - logical_frames)
        start_episode_frame = _episode_frame(raw)
        source_frame = int(visible.source_frame)
        advanced = 0
        for _ in range(requested):
            response = client.step(action, repeat=1)
            raw = _observation(response)
            outcome_trace.push(raw)
            logical_frames += 1
            advanced += 1
            shot_frames += int(action.shoot)
            if stream_vision is not None:
                stream_vision.push(raw)
            else:
                assert vision is not None
                visible = vision.push(adapt_engine_observation(response))
            if raw.get("terminated") is True:
                break
        if advanced:
            _commit_executed_action(controller, action, frames=advanced)
        if stream_vision is not None and raw.get("terminated") is not True:
            visible = stream_vision.observe()
        action_steps.append({
            "decision": len(action_steps),
            "source_frame": source_frame,
            "start_episode_frame": start_episode_frame,
            "end_episode_frame": _episode_frame(raw),
            "requested_frames": requested,
            "advanced_frames": advanced,
            "preferred_action": model_action.to_dict(),
            "visible_safety": {
                **asdict(safety),
                "action": safety.action.to_dict(),
                "profile_gate_open": safety_gate_open,
                "prediction_horizon_frames": visible_safety_horizon,
            },
            "action": action.to_dict(),
            "local_threat_diagnostic": {
                "low_risk": gate.safe,
                "risk": gate.risk,
                "occupancy_peak": gate.occupancy_peak,
                "occupancy_mass": gate.occupancy_mass,
                "reporting_only": True,
                "controls_fire": False,
            },
        })

    engine_terminated = raw.get("terminated") is True
    engine_reason = raw.get("termination_reason") if engine_terminated else None
    termination_reason = engine_reason if engine_terminated else "max_frames"
    outcome_evidence = outcome_trace.report(raw)
    final_player = outcome_evidence.get("final_player")
    final_death = (
        final_player.get("death")
        if isinstance(final_player, Mapping) else
        None
    )
    zero_death_evidence = (
        not isinstance(final_death, bool)
        and isinstance(final_death, (int, float))
        and math.isfinite(float(final_death))
        and float(final_death) == 0.0
    )
    success = (
        engine_terminated
        and engine_reason == completion_reason
        and zero_death_evidence
    )
    final_episode_frame = _episode_frame(raw)
    engine_advanced = None
    if initial_episode_frame is not None and final_episode_frame is not None:
        engine_advanced = max(0, final_episode_frame - initial_episode_frame)
    metadata = dict(controller_metadata or {})
    metadata.update(_controller_state(controller))
    runtime_identity = ping.get("runtime_identity")
    pure_policy = (
        isinstance(controller, VisualPolicyController)
        and not config.visible_safety_shield
        and visible_safety_checks == 0
        and visible_safety_interventions == 0
    )
    raw_model_action_execution = (
        pure_policy
        and controller.proficiency.name == "expert"
        and controller.proficiency.reaction_delay_frames == 0
        and controller.proficiency.direction_hold_frames <= config.decision_interval
        and controller.proficiency.suboptimal_action_probability == 0.0
    )
    return {
        "schema_version": 3,
        "run_kind": "live_luastg_ai_demonstration",
        "acceptance_claim": False,
        "implementation_sha256": source_tree_sha256(),
        "success": success,
        "passed": success,
        "episode_completed": success,
        "pure_policy": pure_policy,
        "unassisted_learned_policy": pure_policy,
        "raw_model_action_execution": raw_model_action_execution,
        "pure_policy_success": pure_policy and success,
        "pure_policy_validation_eligible": pure_policy,
        "success_criterion": (
            f"terminated with termination_reason={completion_reason} and explicit death=0"
        ),
        "episode_kind": episode_kind,
        "scenario": scenario,
        "attack": None if attack is None else int(attack),
        "stage": stage,
        "seed": int(seed),
        "player": player,
        "terminated": engine_terminated,
        "termination_reason": termination_reason,
        "engine_termination_reason": engine_reason,
        "frames": logical_frames,
        "engine_advanced_frames": engine_advanced,
        "initial_episode_frame": initial_episode_frame,
        "final_episode_frame": final_episode_frame,
        "decision_count": len(action_steps),
        "shoot_rate": shot_frames / logical_frames if logical_frames else 0.0,
        "shoot_frames": shot_frames,
        "shoot_command_rate": shot_frames / logical_frames if logical_frames else 0.0,
        "shoot_command_frames": shot_frames,
        "continuous_fire": True,
        "low_risk_decisions": safe_decisions,
        "visible_safety_interventions": visible_safety_interventions,
        "visible_safety_checks": visible_safety_checks,
        "visible_safety_probability_skips": visible_safety_probability_skips,
        "unsafe_shot_frames": None,
        "unsafe_shot_frames_deprecated": True,
        "unsafe_shot_frames_definition": (
            "retired: shooting does not affect movement or collision"
        ),
        "controller": metadata,
        "engine": {
            "protocol": ping.get("protocol"),
            "session_id": ping.get("session_id"),
            "process_nonce": ping.get("process_nonce"),
            "runtime_identity": dict(runtime_identity) if isinstance(runtime_identity, Mapping) else {},
            "runtime_source_verification": runtime_source_verification,
            "catalog_entry": catalog_entry,
        },
        "outcome_evidence": outcome_evidence,
        "config": {
            "max_frames": config.max_frames,
            "decision_interval": config.decision_interval,
            "vision": asdict(config.vision),
            "shoot_gate_radius": config.shoot_gate_radius,
            "shoot_risk_threshold": config.shoot_risk_threshold,
            "shoot_motion_weight": config.shoot_motion_weight,
            "continuous_fire": True,
            "shoot_gate_controls_fire": False,
            "reset_options": {},
            "authority_state_shield": False,
            "visible_safety_shield": config.visible_safety_shield,
            "visible_safety_horizon": visible_safety_horizon,
            "visible_safety_horizon_cap": config.visible_safety_horizon,
            "visible_safety_probability": visible_safety_probability,
            "visible_safety_proficiency": visible_safety_profile,
            "visible_safety_minimum_margin": config.visible_safety_minimum_margin,
            "spell_forced_off": True,
            "render": config.render,
            "render_every": config.render_every,
            "control_inputs": (
                _stream_policy_control_inputs(controller)
                if stream_vision is not None else
                ["delayed_global_semantic_frames", "delayed_local_semantic_frames"]
            ),
        },
        "action_steps": action_steps,
    }


__all__ = [
    "EnginePlayConfig",
    "ShootGateSample",
    "VisibleController",
    "VisualPolicyController",
    "load_route_controller",
    "load_route_library_controller",
    "run_engine_play",
    "visible_shoot_gate",
]
