"""Run the visible-trajectory MPC teacher against a live LuaSTG attack."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Callable, Mapping

from .engine import EngineClient, EngineProtocolError
from .engine_runtime import verify_runtime_source_fingerprints
from .engine_mpc import EngineMPC, MPCDecision
from .engine_play import _OutcomeTrace, _catalog_entry, _observation
from .engine_vision import EngineStreamVision, controller_observation
from .native_dataset import risk_from_clearance
from .protocol import Action
from .provenance import source_tree_sha256
from .render_performance import RenderPerformanceTrace
from .vision import VisionConfig, VisionObservation


DecisionObserver = Callable[[VisionObservation, Action, float], None]
_REPLAY_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")
_WINDOWS_RESERVED_REPLAY_BASENAMES = frozenset({
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
})


@dataclass(frozen=True, slots=True)
class EngineMPCPlayConfig:
    max_frames: int = 7200
    decision_interval: int = 3
    observation_delay: int = 5
    # Retained for CLI/report compatibility. Shooting is continuous because it
    # does not alter player movement or collision; this threshold is diagnostic.
    shoot_minimum_margin: float = 12.0
    render: bool = False
    render_every: int = 1
    record_observations_from_frame: int | None = None
    region_dynamics_memory_path: str | None = None
    region_dynamics_memory_sha256: str | None = None
    replay_name: str | None = None

    def __post_init__(self) -> None:
        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if self.decision_interval != 3:
            raise ValueError("live actions must be held for exactly three frames")
        if self.observation_delay < 0:
            raise ValueError("observation_delay cannot be negative")
        if not math.isfinite(self.shoot_minimum_margin):
            raise ValueError("shoot_minimum_margin must be finite")
        if (
            isinstance(self.render_every, bool)
            or not isinstance(self.render_every, int)
            or not 1 <= self.render_every <= 600
        ):
            raise ValueError("render_every must be an integer in [1, 600]")
        if self.record_observations_from_frame is not None and (
            isinstance(self.record_observations_from_frame, bool)
            or not isinstance(self.record_observations_from_frame, int)
            or self.record_observations_from_frame < 0
        ):
            raise ValueError(
                "record_observations_from_frame must be a nonnegative integer"
            )
        if (self.region_dynamics_memory_path is None) != (
            self.region_dynamics_memory_sha256 is None
        ):
            raise ValueError("region dynamics memory path and SHA-256 must be paired")
        if self.region_dynamics_memory_sha256 is not None and (
            len(self.region_dynamics_memory_sha256) != 64
            or any(
                value not in "0123456789abcdef"
                for value in self.region_dynamics_memory_sha256
            )
        ):
            raise ValueError("region dynamics memory SHA-256 is invalid")
        if self.replay_name is not None:
            if not isinstance(self.replay_name, str):
                raise ValueError("replay_name must be a string")
            replay_name = (
                self.replay_name[:-4]
                if self.replay_name.lower().endswith(".rep") else self.replay_name
            )
            if (
                _REPLAY_NAME_PATTERN.fullmatch(replay_name) is None
                or replay_name.endswith(".")
            ):
                raise ValueError(
                    "replay_name must be 1-96 portable filename characters"
                )
            windows_basename = replay_name.partition(".")[0].upper()
            if windows_basename in _WINDOWS_RESERVED_REPLAY_BASENAMES:
                raise ValueError("replay_name uses a Windows reserved basename")
            object.__setattr__(self, "replay_name", replay_name)


def _replay_start_metadata(
    response: Mapping[str, Any],
    *,
    expected_name: str,
    expected_episode_kind: str,
    expected_stage_name: str,
    expected_seed: int,
) -> dict[str, Any]:
    reset = response.get("reset")
    replay = reset.get("replay") if isinstance(reset, Mapping) else None
    if not isinstance(replay, Mapping):
        raise EngineProtocolError("engine reset did not start the requested replay")
    result = dict(replay)
    random_seed = result.get("random_seed")
    if (
        result.get("schema_version") != 1
        or result.get("name") != expected_name
        or result.get("episode_kind") != expected_episode_kind
        or result.get("stage_name") != expected_stage_name
        or random_seed != expected_seed
        or result.get("saved") is not False
        or not isinstance(result.get("path"), str)
        or not result["path"]
        or isinstance(random_seed, bool)
        or not isinstance(random_seed, int)
        or not isinstance(result.get("player"), str)
        or not result["player"]
    ):
        raise EngineProtocolError("engine returned invalid replay start metadata")
    return result


def _saved_replay_metadata(
    response: Mapping[str, Any],
    *,
    expected_name: str,
    expected_episode_kind: str,
    expected_stage_name: str,
    expected_seed: int,
    expected_player: str,
    expected_path: str,
    expected_finish: bool,
    expected_reason: str,
) -> dict[str, Any]:
    replay = response.get("replay")
    if not isinstance(replay, Mapping):
        raise EngineProtocolError("engine did not return saved replay metadata")
    result = dict(replay)
    frame_count = result.get("frame_count")
    frame_bytes_verified = result.get("frame_bytes_verified")
    file_size = result.get("file_size")
    crc32 = result.get("crc32")
    expected_group_finish = 1 if expected_finish else 0
    if (
        result.get("schema_version") != 1
        or result.get("name") != expected_name
        or result.get("episode_kind") != expected_episode_kind
        or result.get("stage_name") != expected_stage_name
        or result.get("random_seed") != expected_seed
        or result.get("player") != expected_player
        or result.get("path") != expected_path
        or result.get("finish") is not expected_finish
        or result.get("group_finish") != expected_group_finish
        or result.get("reason") != expected_reason
        or result.get("saved") is not True
        or result.get("verified") is not True
        or not isinstance(result.get("path"), str)
        or not result["path"]
        or isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count <= 0
        or isinstance(frame_bytes_verified, bool)
        or not isinstance(frame_bytes_verified, int)
        or frame_bytes_verified != frame_count
        or isinstance(file_size, bool)
        or not isinstance(file_size, int)
        or file_size < frame_bytes_verified
        or not isinstance(crc32, str)
        or re.fullmatch(r"[0-9a-f]{8}", crc32) is None
    ):
        raise EngineProtocolError("engine returned invalid saved replay metadata")
    return result


def _episode_frame(observation: Mapping[str, Any]) -> int | None:
    value = observation.get("episode_frame")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


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
            f"live catalog does not contain exactly one stage {stage}",
        )
    return dict(matches[0])


def _player_position(observation: Mapping[str, Any]) -> dict[str, float] | None:
    player = observation.get("player")
    if not isinstance(player, Mapping):
        return None
    x, y = player.get("x"), player.get("y")
    if (
        isinstance(x, bool)
        or isinstance(y, bool)
        or not isinstance(x, (int, float))
        or not isinstance(y, (int, float))
        or not math.isfinite(float(x))
        or not math.isfinite(float(y))
    ):
        return None
    return {"x": float(x), "y": float(y)}


def _radius(record: Mapping[str, Any], default: float) -> float:
    values = []
    for name in ("a", "b"):
        value = record.get(name)
        if (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        ):
            values.append(abs(float(value)))
    return max([default, *values])


def _nearest_collidables(
    observation: Mapping[str, Any],
    *,
    limit: int = 24,
) -> list[dict[str, Any]]:
    player = observation.get("player")
    if not isinstance(player, Mapping):
        return []
    position = _player_position(observation)
    if position is None:
        return []
    player_radius = _radius(player, 0.5)
    values: list[tuple[float, dict[str, Any]]] = []
    for source in (
        "enemy_bullets",
        "enemies",
        "nontjt_enemies",
        "indestructibles",
        "lasers",
    ):
        records = observation.get(source)
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, Mapping) or record.get("collidable", True) is not True:
                continue
            x, y = record.get("x"), record.get("y")
            if (
                isinstance(x, bool)
                or isinstance(y, bool)
                or not isinstance(x, (int, float))
                or not isinstance(y, (int, float))
                or not math.isfinite(float(x))
                or not math.isfinite(float(y))
            ):
                continue
            radius = _radius(record, 2.0)
            margin = (
                math.hypot(position["x"] - float(x), position["y"] - float(y))
                - player_radius - radius
            )
            values.append((margin, {
                "source": source,
                "id": record.get("id"),
                "kind": record.get("kind"),
                "image": record.get("image"),
                "x": float(x),
                "y": float(y),
                "dx": record.get("dx"),
                "dy": record.get("dy"),
                "a": record.get("a"),
                "b": record.get("b"),
                "margin": margin,
            }))
    values.sort(key=lambda item: item[0])
    return [record for _, record in values[:limit]]


def _selected_evaluation(decision: MPCDecision):
    for evaluation in decision.evaluations:
        if evaluation.action.discrete == decision.action.discrete:
            return evaluation
    raise RuntimeError("MPC decision has no matching candidate evaluation")


def _controller_observation(
    delayed: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    """Backward-compatible alias for the shared live-vision contract."""

    return controller_observation(delayed, current)


def _effective_action(decision: MPCDecision) -> Action:
    return Action(
        move_x=decision.action.move_x,
        move_y=decision.action.move_y,
        slow=decision.action.slow,
        shoot=True,
        spell=False,
    )


def run_engine_mpc_play(
    client: EngineClient,
    *,
    scenario: str,
    attack: int | None,
    seed: int,
    player: str,
    controller: EngineMPC,
    config: EngineMPCPlayConfig = EngineMPCPlayConfig(),
    stage: str | None = None,
    decision_observer: DecisionObserver | None = None,
    vision_config: VisionConfig | None = None,
) -> dict[str, Any]:
    """Run one strict attack or full-stage episode using delayed geometry."""

    if stage is None:
        if attack is None or attack <= 0:
            raise ValueError("attack must be positive")
        episode_kind = "attack"
        completion_reason = "attack_complete"
    else:
        if not stage:
            raise ValueError("stage must be a nonempty string")
        if attack is not None:
            raise ValueError("attack must be None for a stage episode")
        episode_kind = "stage"
        completion_reason = "stage_complete"
    if controller.config.decision_interval != config.decision_interval:
        raise ValueError("controller and runner decision intervals differ")
    if controller.config.observation_delay != config.observation_delay:
        raise ValueError("controller and runner observation delays differ")

    ping = client.ping()
    runtime_source_verification = verify_runtime_source_fingerprints(ping)
    if config.replay_name is not None:
        commands = ping.get("commands")
        if not isinstance(commands, list) or "save_replay" not in commands:
            raise EngineProtocolError(
                "engine bridge does not advertise native replay capture"
            )
    catalog_response = client.catalog()
    replay_request = (
        {} if config.replay_name is None else {"replay_name": config.replay_name}
    )
    if stage is None:
        assert attack is not None
        catalog_entry = _catalog_entry(catalog_response, scenario, attack)
        response = client.reset(
            scenario,
            attack,
            seed=int(seed),
            player=player,
            options={},
            **replay_request,
        )
    else:
        catalog_entry = _catalog_stage(catalog_response, stage)
        response = client.reset_stage(
            stage,
            seed=int(seed),
            player=player,
            options={},
            **replay_request,
        )
    replay_start = (
        None
        if config.replay_name is None else
        _replay_start_metadata(
            response,
            expected_name=config.replay_name,
            expected_episode_kind=episode_kind,
            expected_stage_name=(
                "Spell Practice@Spell Practice" if stage is None else stage
            ),
            expected_seed=int(seed),
        )
    )
    raw = _observation(response)
    initial_episode_frame = _episode_frame(raw)
    controller.reset()
    stream_vision = None
    if decision_observer is not None:
        stream_vision = EngineStreamVision(
            vision_config or VisionConfig(
                history=1,
                observation_delay=config.observation_delay,
            ),
        )
        if stream_vision.config.observation_delay != config.observation_delay:
            raise ValueError("demonstration and MPC observation delays differ")
        stream_vision.reset(raw)
    delayed_observations: deque[Mapping[str, Any]] = deque(
        [raw] * (config.observation_delay + 1),
        maxlen=config.observation_delay + 1,
    )
    display = client.set_rendering(config.render, every=config.render_every)
    if display.get("render") is not config.render or display.get("every") != config.render_every:
        raise EngineProtocolError("engine did not apply the requested display state")

    trace = _OutcomeTrace()
    trace.push(raw)
    render_performance = RenderPerformanceTrace()
    render_performance.push(raw)
    decisions: list[dict[str, Any]] = []
    logical_frames = 0
    shot_frames = 0
    predicted_collision_plan_frames = 0
    terminal_before: Mapping[str, Any] | None = None
    terminal_action: Action | None = None

    while raw.get("terminated") is not True and logical_frames < config.max_frames:
        delayed = delayed_observations[0]
        controller_observation = _controller_observation(delayed, raw)
        start_frame = _episode_frame(raw)
        decision = controller.select(controller_observation)
        evaluation = _selected_evaluation(decision)
        action = _effective_action(decision)
        controller_overlay_state = (
            controller.controller_overlay_state(decision, controller_observation)
            if config.render else None
        )
        if decision_observer is not None:
            assert stream_vision is not None
            decision_observer(
                stream_vision.observe(),
                action,
                risk_from_clearance(evaluation.minimum_margin),
            )
        source_frame = decision.source_frame
        requested = min(config.decision_interval, config.max_frames - logical_frames)
        advanced = 0
        for action_frame in range(requested):
            before = raw
            before_frame = _episode_frame(before)
            response = client.step(
                action,
                repeat=1,
                controller_overlay_state=(
                    controller_overlay_state if action_frame == 0 else None
                ),
            )
            raw = _observation(response)
            if stream_vision is not None:
                stream_vision.push(raw)
            after_frame = _episode_frame(raw)
            if (
                before_frame is None
                or after_frame is None
                or after_frame != before_frame + 1
            ):
                raise EngineProtocolError(
                    "engine episode_frame did not advance by exactly one frame"
                )
            delayed_observations.append(raw)
            trace.push(raw)
            render_performance.push(raw)
            logical_frames += 1
            advanced += 1
            shot_frames += int(action.shoot)
            predicted_collision_plan_frames += int(evaluation.collided)
            if raw.get("terminated") is True:
                terminal_before = before
                terminal_action = action
                break
        decisions.append({
            "decision": len(decisions),
            "control_source": "live_mpc",
            "source_frame": source_frame,
            "start_episode_frame": start_frame,
            "end_episode_frame": _episode_frame(raw),
            "requested_frames": requested,
            "advanced_frames": advanced,
            "action": action.to_dict(),
            "predicted_threat_count": len(decision.threats),
            "predicted_collision": evaluation.collided,
            "predicted_collision_frames": evaluation.collision_frames,
            "predicted_earliest_collision_frame": evaluation.earliest_collision_frame,
            "predicted_minimum_margin": evaluation.minimum_margin,
            "predicted_minimum_nonregion_margin": (
                evaluation.minimum_nonregion_margin
                if math.isfinite(evaluation.minimum_nonregion_margin) else
                None
            ),
            "predicted_minimum_region_margin": (
                evaluation.minimum_region_margin
                if math.isfinite(evaluation.minimum_region_margin) else
                None
            ),
            "predicted_immediate_corner_clearance": (
                evaluation.immediate_corner_clearance
                if math.isfinite(evaluation.immediate_corner_clearance) else
                None
            ),
            "region_anchor": (
                None if decision.region_anchor is None else
                {"x": decision.region_anchor[0], "y": decision.region_anchor[1]}
            ),
            "region_crossing": decision.region_crossing,
            "region_path_margin": decision.region_path_margin,
            "region_evacuating": decision.region_evacuating,
            "region_target_rows_ahead": decision.region_target_rows_ahead,
            "region_navigation_mode": decision.region_navigation_mode,
            "region_current_component": decision.region_current_component,
            "region_target_component": decision.region_target_component,
            "region_portal": decision.region_portal,
            "region_deadline_slack": decision.region_deadline_slack,
            "region_phase": decision.region_phase,
            "region_phase_started_frame": decision.region_phase_started_frame,
            "region_learned_cycle_frames": decision.region_learned_cycle_frames,
            "region_frames_until_expansion": decision.region_frames_until_expansion,
            "region_observed_radius": decision.region_observed_radius,
            "gap_bullet_group_count": decision.gap_bullet_group_count,
            "gap_corridor_count": decision.gap_corridor_count,
            "gap_selected_center": (
                None if decision.gap_selected_center is None else
                {
                    "x": decision.gap_selected_center[0],
                    "y": decision.gap_selected_center[1],
                }
            ),
            "gap_selected_width": decision.gap_selected_width,
            "gap_selected_lifetime_frames": (
                decision.gap_selected_lifetime_frames
            ),
            "gap_navigation_mode": decision.gap_navigation_mode,
            "using_committed_plan": decision.using_committed_plan,
            "committed_plan_immediate_margin": decision.committed_plan_immediate_margin,
            "committed_plan_current_horizon_margin": (
                decision.committed_plan_current_horizon_margin
            ),
            "planned_actions": [
                value.to_dict() for value in decision.planned_actions
            ],
            "reporting_only_authority_player": _player_position(raw),
            "recorded_controller_input_observation": (
                dict(controller_observation)
                if config.record_observations_from_frame is not None
                and start_frame is not None
                and start_frame >= config.record_observations_from_frame
                else None
            ),
        })

    engine_terminated = raw.get("terminated") is True
    engine_reason = raw.get("termination_reason") if engine_terminated else None
    termination_reason = engine_reason if engine_terminated else "max_frames"
    outcome_evidence = trace.report(raw)
    final_player = outcome_evidence.get("final_player")
    death_value = (
        final_player.get("death") if isinstance(final_player, Mapping) else None
    )
    zero_death_evidence = (
        not isinstance(death_value, bool)
        and isinstance(death_value, (int, float))
        and math.isfinite(float(death_value))
        and float(death_value) == 0.0
    )
    success = (
        engine_terminated
        and engine_reason == completion_reason
        and zero_death_evidence
    )
    final_episode_frame = _episode_frame(raw)
    native_replay = None
    if config.replay_name is not None:
        assert replay_start is not None
        replay_reason = (
            termination_reason
            if isinstance(termination_reason, str) and termination_reason else
            "unknown"
        )
        replay_finish = episode_kind == "stage" and success
        replay_response = client.save_replay(
            finish=replay_finish,
            reason=replay_reason,
        )
        native_replay = _saved_replay_metadata(
            replay_response,
            expected_name=config.replay_name,
            expected_episode_kind=episode_kind,
            expected_stage_name=replay_start["stage_name"],
            expected_seed=replay_start["random_seed"],
            expected_player=replay_start["player"],
            expected_path=replay_start["path"],
            expected_finish=replay_finish,
            expected_reason=replay_reason,
        )
    engine_advanced = None
    if initial_episode_frame is not None and final_episode_frame is not None:
        engine_advanced = max(0, final_episode_frame - initial_episode_frame)
    runtime_identity = ping.get("runtime_identity")
    overlay_status = raw.get("safety_zone_overlay")
    gap_diagnostics = {
        "enabled": controller.config.gap_prediction_enabled,
        "detected_decision_count": sum(
            item["gap_corridor_count"] > 0 for item in decisions
        ),
        "selected_decision_count": sum(
            item["gap_selected_center"] is not None for item in decisions
        ),
        "observe_decision_count": sum(
            item["gap_navigation_mode"] == "observe" for item in decisions
        ),
        "enter_decision_count": sum(
            item["gap_navigation_mode"] == "enter" for item in decisions
        ),
        "hold_decision_count": sum(
            item["gap_navigation_mode"] == "hold" for item in decisions
        ),
        "exit_decision_count": sum(
            item["gap_navigation_mode"] == "exit" for item in decisions
        ),
        "maximum_bullet_group_count": max(
            (item["gap_bullet_group_count"] for item in decisions),
            default=0,
        ),
        "maximum_corridor_count": max(
            (item["gap_corridor_count"] for item in decisions),
            default=0,
        ),
    }
    return {
        "schema_version": 3,
        "run_kind": "live_luastg_delayed_visible_mpc_teacher",
        "acceptance_claim": False,
        "implementation_sha256": source_tree_sha256(),
        "success": success,
        "passed": success,
        "episode_completed": success,
        "teacher_success": success,
        "pure_policy": False,
        "pure_policy_success": False,
        "pure_policy_validation_eligible": False,
        "region_dynamics_training_eligible": True,
        "success_criterion": (
            f"terminated with termination_reason={completion_reason} and "
            "outcome_evidence.final_player.death=0"
        ),
        "policy_validation_criterion": (
            f"{completion_reason} with zero deaths from reset under live MPC"
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
        "decision_count": len(decisions),
        "shoot_frames": shot_frames,
        "shoot_rate": shot_frames / logical_frames if logical_frames else 0.0,
        "shoot_command_frames": shot_frames,
        "shoot_command_rate": shot_frames / logical_frames if logical_frames else 0.0,
        "continuous_fire": True,
        "unsafe_shot_frames": None,
        "unsafe_shot_frames_deprecated": True,
        "unsafe_shot_frames_definition": (
            "retired: shooting does not affect movement or collision"
        ),
        "predicted_collision_plan_frames": predicted_collision_plan_frames,
        "native_replay": native_replay,
        "engine": {
            "protocol": ping.get("protocol"),
            "session_id": ping.get("session_id"),
            "process_nonce": ping.get("process_nonce"),
            "runtime_identity": (
                dict(runtime_identity) if isinstance(runtime_identity, Mapping) else {}
            ),
            "runtime_source_verification": runtime_source_verification,
            "catalog_entry": catalog_entry,
            "safety_zone_overlay": (
                dict(overlay_status)
                if isinstance(overlay_status, Mapping) else None
            ),
        },
        "outcome_evidence": outcome_evidence,
        "render_performance": render_performance.report(),
        "gap_prediction": gap_diagnostics,
        "terminal_transition_evidence": (
            None if terminal_before is None else {
                "reporting_only_not_controller_input": True,
                "before_episode_frame": _episode_frame(terminal_before),
                "after_episode_frame": _episode_frame(raw),
                "before_player": _player_position(terminal_before),
                "after_player": _player_position(raw),
                "action": (
                    terminal_action.to_dict() if terminal_action is not None else None
                ),
                "nearest_collidables_before": _nearest_collidables(terminal_before),
            }
        ),
        "controller": {
            "kind": "visible_trajectory_mpc_teacher",
            "config": asdict(controller.config),
            "uses_raw_object_ids": True,
            "uses_raw_velocity_fields": True,
            "uses_class_names_or_script_timers_for_dodging": False,
        },
        "config": {
            **asdict(config),
            "reset_options": {},
            "authority_state_shield": False,
            "continuous_fire": True,
            "shoot_minimum_margin_controls_fire": False,
            "spell_forced_off": True,
            "controller_overlay_state_published": config.render,
            "control_inputs": [
                "delayed_visible_positions",
                "delayed_visible_collision_shapes",
                "visible_displacement_motion_estimates",
                "own_visible_player_position",
                "teacher_only_raw_object_ids_and_initial_velocity",
            ],
        },
        "decisions": decisions,
    }


__all__ = [
    "EngineMPCPlayConfig",
    "run_engine_mpc_play",
]
