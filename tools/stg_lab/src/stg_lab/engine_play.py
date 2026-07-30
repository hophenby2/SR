"""Visible-only controller runner for a live LuaSTG spell-practice attack."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np

from .adapters import adapt_engine_observation
from .engine import EngineClient, EngineProtocolError
from .memory import EpisodicMemory
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


@dataclass(frozen=True, slots=True)
class EnginePlayConfig:
    """Configuration whose control-facing inputs remain delayed and visible."""

    max_frames: int = 7200
    decision_interval: int = 3
    vision: VisionConfig = VisionConfig()
    shoot_gate_radius: float = 20.0
    shoot_risk_threshold: float = 0.25
    shoot_motion_weight: float = 0.5
    render: bool = False
    render_every: int = 1

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


class VisualPolicyController:
    """Small live-engine wrapper around the existing delayed visual policy."""

    def __init__(self, model: Any, scenario_key: str, *, device: str = "cpu") -> None:
        from .rollout import _model_device, _policy_inference_mode, scenario_memory_vector

        self.model = model
        self.scenario_key = str(scenario_key)
        self.device = _model_device(model, device)
        self.inference_mode = _policy_inference_mode(model)
        memory_size = int(getattr(getattr(model, "config", None), "memory_size", 4))
        self.memory = scenario_memory_vector(self.scenario_key, memory_size)
        self.model.to(self.device)
        self.model.eval()
        self.reset()

    def reset(self) -> None:
        self.hidden: Any | None = None
        self.decisions = 0

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
        )
        self.decisions += 1
        return action


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
    """Gate shooting from the latest delayed local semantic frame only."""

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


def _effective_action(preferred: Action, gate: ShootGateSample) -> Action:
    return Action(
        move_x=preferred.move_x,
        move_y=preferred.move_y,
        slow=preferred.slow,
        shoot=gate.safe,
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
            "decisions": controller.decisions,
        }
    return {"type": type(controller).__name__}


def run_engine_play(
    client: EngineClient,
    *,
    scenario: str,
    attack: int,
    seed: int,
    player: str,
    controller: VisibleController,
    controller_metadata: Mapping[str, Any] | None = None,
    config: EnginePlayConfig = EnginePlayConfig(),
) -> dict[str, Any]:
    """Run one real attack; only ``attack_complete`` is counted as success."""

    if attack <= 0:
        raise ValueError("attack must be positive")
    ping = client.ping()
    catalog_entry = _catalog_entry(client.catalog(), scenario, attack)
    reset_response = client.reset(
        scenario,
        attack,
        seed=int(seed),
        player=player,
        options={},
    )
    raw = _observation(reset_response)
    initial_episode_frame = _episode_frame(raw)
    vision = DelayedVision(config=config.vision)
    visible = vision.reset(adapt_engine_observation(reset_response))
    controller.reset()
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
    while raw.get("terminated") is not True and logical_frames < config.max_frames:
        preferred = coerce_action(controller.select(visible))
        gate = visible_shoot_gate(visible, config)
        action = _effective_action(preferred, gate)
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
            visible = vision.push(adapt_engine_observation(response))
            if raw.get("terminated") is True:
                break
        action_steps.append({
            "decision": len(action_steps),
            "source_frame": source_frame,
            "start_episode_frame": start_episode_frame,
            "end_episode_frame": _episode_frame(raw),
            "requested_frames": requested,
            "advanced_frames": advanced,
            "preferred_action": preferred.to_dict(),
            "action": action.to_dict(),
            "shoot_gate": asdict(gate),
        })

    engine_terminated = raw.get("terminated") is True
    engine_reason = raw.get("termination_reason") if engine_terminated else None
    termination_reason = engine_reason if engine_terminated else "max_frames"
    success = engine_terminated and engine_reason == "attack_complete"
    final_episode_frame = _episode_frame(raw)
    engine_advanced = None
    if initial_episode_frame is not None and final_episode_frame is not None:
        engine_advanced = max(0, final_episode_frame - initial_episode_frame)
    metadata = dict(controller_metadata or {})
    metadata.update(_controller_state(controller))
    runtime_identity = ping.get("runtime_identity")
    return {
        "schema_version": 1,
        "run_kind": "live_luastg_ai_demonstration",
        "acceptance_claim": False,
        "implementation_sha256": source_tree_sha256(),
        "success": success,
        "passed": success,
        "success_criterion": "terminated with termination_reason=attack_complete",
        "scenario": scenario,
        "attack": int(attack),
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
        "safe_shoot_decisions": safe_decisions,
        "unsafe_shot_frames": 0,
        "controller": metadata,
        "engine": {
            "protocol": ping.get("protocol"),
            "session_id": ping.get("session_id"),
            "process_nonce": ping.get("process_nonce"),
            "runtime_identity": dict(runtime_identity) if isinstance(runtime_identity, Mapping) else {},
            "catalog_entry": catalog_entry,
        },
        "outcome_evidence": outcome_trace.report(raw),
        "config": {
            "max_frames": config.max_frames,
            "decision_interval": config.decision_interval,
            "vision": asdict(config.vision),
            "shoot_gate_radius": config.shoot_gate_radius,
            "shoot_risk_threshold": config.shoot_risk_threshold,
            "shoot_motion_weight": config.shoot_motion_weight,
            "reset_options": {},
            "authority_state_shield": False,
            "spell_forced_off": True,
            "render": config.render,
            "render_every": config.render_every,
            "control_inputs": ["delayed_global_semantic_frames", "delayed_local_semantic_frames"],
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
