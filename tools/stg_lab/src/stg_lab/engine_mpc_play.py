"""Run the visible-trajectory MPC teacher against a live LuaSTG attack."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .engine import EngineClient, EngineProtocolError
from .engine_mpc import EngineMPC, MPCDecision
from .engine_play import _OutcomeTrace, _catalog_entry, _observation
from .protocol import Action
from .provenance import file_sha256, source_tree_sha256


@dataclass(frozen=True, slots=True)
class EngineMPCPlayConfig:
    max_frames: int = 7200
    decision_interval: int = 3
    observation_delay: int = 5
    shoot_minimum_margin: float = 12.0
    render: bool = False
    render_every: int = 1
    record_observations_from_frame: int | None = None
    region_dynamics_memory_path: str | None = None
    region_dynamics_memory_sha256: str | None = None

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


@dataclass(frozen=True, slots=True)
class RecordedPrefixDecision:
    decision: int
    start_episode_frame: int
    end_episode_frame: int
    requested_frames: int
    advanced_frames: int
    action: Action


@dataclass(frozen=True, slots=True)
class RecordedActionPrefix:
    path: Path
    sha256: str
    scenario: str
    attack: int
    seed: int
    player: str
    initial_episode_frame: int
    decisions: tuple[RecordedPrefixDecision, ...]

    def decision_at(self, episode_frame: int) -> RecordedPrefixDecision | None:
        for decision in self.decisions:
            if decision.start_episode_frame == episode_frame:
                return decision
        return None


def _artifact_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"recorded prefix {label} must be an integer")
    return value


def _artifact_boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"recorded prefix {label} must be a boolean")
    return value


def load_recorded_action_prefix(
    path: str | Path,
    *,
    scenario: str,
    attack: int,
    seed: int,
    player: str,
) -> RecordedActionPrefix:
    """Load an identity-bound, spell-free action prefix from a live artifact."""

    artifact_path = Path(path)
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read recorded prefix artifact: {artifact_path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("recorded prefix artifact must contain a JSON object")
    schema_version = _artifact_integer(payload.get("schema_version"), "schema_version")
    if schema_version != 1:
        raise ValueError("recorded prefix artifact schema_version must be 1")
    if payload.get("run_kind") != "live_luastg_delayed_visible_mpc_teacher":
        raise ValueError("recorded prefix must come from a live LuaSTG MPC run")

    actual_attack = _artifact_integer(payload.get("attack"), "attack")
    actual_seed = _artifact_integer(payload.get("seed"), "seed")
    identity = (payload.get("scenario"), actual_attack, actual_seed, payload.get("player"))
    expected = (scenario, int(attack), int(seed), player)
    if identity != expected:
        raise ValueError(
            "recorded prefix identity does not match requested scenario/attack/seed/player"
        )

    artifact_config = payload.get("config")
    if not isinstance(artifact_config, Mapping):
        raise ValueError("recorded prefix artifact has no runner config")
    if artifact_config.get("reset_options") != {}:
        raise ValueError("recorded prefix must have been recorded with empty reset options")
    if artifact_config.get("authority_state_shield") is not False:
        raise ValueError("recorded prefix must not use an authority-state shield")
    if artifact_config.get("spell_forced_off") is not True:
        raise ValueError("recorded prefix does not prove that spell was forced off")

    raw_decisions = payload.get("decisions")
    if not isinstance(raw_decisions, list) or not raw_decisions:
        raise ValueError("recorded prefix artifact has no decisions")
    decision_count = _artifact_integer(payload.get("decision_count"), "decision_count")
    if decision_count != len(raw_decisions):
        raise ValueError("recorded prefix decision_count does not match decisions")
    initial_episode_frame = _artifact_integer(
        payload.get("initial_episode_frame"),
        "initial_episode_frame",
    )
    decisions: list[RecordedPrefixDecision] = []
    previous_end: int | None = None
    for index, raw_decision in enumerate(raw_decisions):
        if not isinstance(raw_decision, Mapping):
            raise ValueError(f"recorded prefix decision {index} must be an object")
        recorded_index = _artifact_integer(
            raw_decision.get("decision"),
            f"decision {index} decision",
        )
        if recorded_index != index:
            raise ValueError(f"recorded prefix decision {index} has an invalid index")
        start = _artifact_integer(
            raw_decision.get("start_episode_frame"),
            f"decision {index} start_episode_frame",
        )
        end = _artifact_integer(
            raw_decision.get("end_episode_frame"),
            f"decision {index} end_episode_frame",
        )
        advanced = _artifact_integer(
            raw_decision.get("advanced_frames"),
            f"decision {index} advanced_frames",
        )
        requested = _artifact_integer(
            raw_decision.get("requested_frames"),
            f"decision {index} requested_frames",
        )
        if requested <= 0 or advanced > requested:
            raise ValueError(f"recorded prefix decision {index} has invalid frame counts")
        if advanced <= 0 or end - start != advanced:
            raise ValueError(f"recorded prefix decision {index} has inconsistent frame span")
        if index == 0 and start != initial_episode_frame:
            raise ValueError(
                "recorded prefix first decision does not start at initial_episode_frame"
            )
        if previous_end is not None and start != previous_end:
            raise ValueError(f"recorded prefix decision {index} is not contiguous")
        raw_action = raw_decision.get("action")
        if not isinstance(raw_action, Mapping):
            raise ValueError(f"recorded prefix decision {index} has no action")
        expected_action_fields = {"move_x", "move_y", "slow", "shoot", "spell"}
        if set(raw_action) != expected_action_fields:
            raise ValueError(
                f"recorded prefix decision {index} action fields must be exactly "
                "move_x/move_y/slow/shoot/spell"
            )
        move_x = _artifact_integer(raw_action.get("move_x"), f"decision {index} move_x")
        move_y = _artifact_integer(raw_action.get("move_y"), f"decision {index} move_y")
        if move_x not in (-1, 0, 1) or move_y not in (-1, 0, 1):
            raise ValueError(f"recorded prefix decision {index} has invalid movement")
        slow = _artifact_boolean(raw_action.get("slow"), f"decision {index} slow")
        shoot = _artifact_boolean(raw_action.get("shoot"), f"decision {index} shoot")
        spell = _artifact_boolean(raw_action.get("spell"), f"decision {index} spell")
        if spell:
            raise ValueError(f"recorded prefix decision {index} uses a spell")
        decisions.append(RecordedPrefixDecision(
            decision=recorded_index,
            start_episode_frame=start,
            end_episode_frame=end,
            requested_frames=requested,
            advanced_frames=advanced,
            action=Action(
                move_x=move_x,
                move_y=move_y,
                slow=slow,
                shoot=shoot,
                spell=False,
            ),
        ))
        previous_end = end
    return RecordedActionPrefix(
        path=artifact_path.resolve(),
        sha256=file_sha256(artifact_path),
        scenario=scenario,
        attack=int(attack),
        seed=int(seed),
        player=player,
        initial_episode_frame=initial_episode_frame,
        decisions=tuple(decisions),
    )


def _episode_frame(observation: Mapping[str, Any]) -> int | None:
    value = observation.get("episode_frame")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


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
    """Keep hazards delayed while exposing the player's current visible state."""

    result = dict(delayed)
    player = current.get("player")
    if isinstance(player, Mapping):
        result["player"] = dict(player)
        result["own_player_observation_delay"] = 0
        result["own_player_observation_frame"] = _episode_frame(current)
    return result


def _effective_action(decision: MPCDecision, config: EngineMPCPlayConfig) -> Action:
    evaluation = _selected_evaluation(decision)
    safe_to_shoot = (
        not evaluation.collided
        and evaluation.minimum_margin >= config.shoot_minimum_margin
    )
    return Action(
        move_x=decision.action.move_x,
        move_y=decision.action.move_y,
        slow=decision.action.slow,
        shoot=safe_to_shoot,
        spell=False,
    )


def run_engine_mpc_play(
    client: EngineClient,
    *,
    scenario: str,
    attack: int,
    seed: int,
    player: str,
    controller: EngineMPC,
    config: EngineMPCPlayConfig = EngineMPCPlayConfig(),
    prefix_artifact: str | Path | None = None,
    prefix_until_frame: int | None = None,
) -> dict[str, Any]:
    """Run one strict real-engine episode using delayed visible geometry."""

    if attack <= 0:
        raise ValueError("attack must be positive")
    if controller.config.decision_interval != config.decision_interval:
        raise ValueError("controller and runner decision intervals differ")
    if controller.config.observation_delay != config.observation_delay:
        raise ValueError("controller and runner observation delays differ")
    if (prefix_artifact is None) != (prefix_until_frame is None):
        raise ValueError(
            "prefix_artifact and prefix_until_frame must be provided together"
        )
    if prefix_until_frame is not None and (
        isinstance(prefix_until_frame, bool)
        or not isinstance(prefix_until_frame, int)
        or prefix_until_frame <= 0
    ):
        raise ValueError("prefix_until_frame must be a positive integer")
    recorded_prefix = (
        None
        if prefix_artifact is None else
        load_recorded_action_prefix(
            prefix_artifact,
            scenario=scenario,
            attack=attack,
            seed=seed,
            player=player,
        )
    )
    selected_prefix_decisions: tuple[RecordedPrefixDecision, ...] = ()
    if recorded_prefix is not None:
        assert prefix_until_frame is not None
        selected_prefix_decisions = tuple(
            decision
            for decision in recorded_prefix.decisions
            if decision.start_episode_frame < prefix_until_frame
        )
        if not selected_prefix_decisions:
            raise ValueError(
                "recorded prefix has no decision before prefix_until_frame"
            )
        for decision in selected_prefix_decisions:
            if decision.requested_frames != 3 or decision.advanced_frames != 3:
                raise ValueError(
                    "recorded prefix decisions before prefix_until_frame must request "
                    "and advance exactly three frames"
                )
        if selected_prefix_decisions[-1].end_episode_frame < prefix_until_frame:
            raise ValueError(
                "recorded prefix does not cover prefix_until_frame"
            )

    ping = client.ping()
    catalog_entry = _catalog_entry(client.catalog(), scenario, attack)
    response = client.reset(
        scenario,
        attack,
        seed=int(seed),
        player=player,
        options={},
    )
    raw = _observation(response)
    initial_episode_frame = _episode_frame(raw)
    if (
        recorded_prefix is not None
        and initial_episode_frame != recorded_prefix.initial_episode_frame
    ):
        raise EngineProtocolError(
            "engine reset initial_episode_frame does not match recorded prefix"
        )
    controller.reset()
    delayed_observations: deque[Mapping[str, Any]] = deque(
        [raw] * (config.observation_delay + 1),
        maxlen=config.observation_delay + 1,
    )
    display = client.set_rendering(config.render, every=config.render_every)
    if display.get("render") is not config.render or display.get("every") != config.render_every:
        raise EngineProtocolError("engine did not apply the requested display state")

    trace = _OutcomeTrace()
    trace.push(raw)
    decisions: list[dict[str, Any]] = []
    logical_frames = 0
    shot_frames = 0
    unsafe_shot_frames = 0
    prefix_decisions_used = 0
    prefix_frames_used = 0
    prefix_last_end_frame: int | None = None
    live_first_start_frame: int | None = None
    terminal_before: Mapping[str, Any] | None = None
    terminal_action: Action | None = None

    while raw.get("terminated") is not True and logical_frames < config.max_frames:
        delayed = delayed_observations[0]
        controller_observation = _controller_observation(delayed, raw)
        start_frame = _episode_frame(raw)
        use_recorded_prefix = (
            recorded_prefix is not None
            and prefix_until_frame is not None
            and start_frame is not None
            and start_frame < prefix_until_frame
        )
        decision: MPCDecision | None
        evaluation = None
        recorded_decision: RecordedPrefixDecision | None = None
        if use_recorded_prefix:
            assert recorded_prefix is not None
            assert start_frame is not None
            recorded_decision = recorded_prefix.decision_at(start_frame)
            if recorded_decision is None:
                raise EngineProtocolError(
                    f"recorded prefix has no action starting at episode frame {start_frame}"
                )
            source_frame = controller.observe(controller_observation)
            decision = None
            action = recorded_decision.action
            prefix_decisions_used += 1
            prefix_last_end_frame = recorded_decision.end_episode_frame
        else:
            decision = controller.select(controller_observation)
            evaluation = _selected_evaluation(decision)
            action = _effective_action(decision, config)
            source_frame = decision.source_frame
            if recorded_prefix is not None and live_first_start_frame is None:
                live_first_start_frame = start_frame
        requested = min(config.decision_interval, config.max_frames - logical_frames)
        if (
            recorded_decision is not None
            and recorded_decision.end_episode_frame - recorded_decision.start_episode_frame
            < requested
        ):
            raise EngineProtocolError(
                f"recorded prefix action at episode frame {start_frame} is too short"
            )
        advanced = 0
        for _ in range(requested):
            before = raw
            before_frame = _episode_frame(before)
            response = client.step(action, repeat=1)
            raw = _observation(response)
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
            logical_frames += 1
            advanced += 1
            prefix_frames_used += int(use_recorded_prefix)
            shot_frames += int(action.shoot)
            unsafe_shot_frames += int(
                action.shoot
                and evaluation is not None
                and evaluation.collided
            )
            if raw.get("terminated") is True:
                terminal_before = before
                terminal_action = action
                break
        if (
            recorded_decision is not None
            and advanced == recorded_decision.advanced_frames
            and _episode_frame(raw) != recorded_decision.end_episode_frame
        ):
            raise EngineProtocolError(
                f"recorded prefix action at episode frame {start_frame} "
                "ended at an unexpected episode frame"
            )
        decisions.append({
            "decision": len(decisions),
            "control_source": (
                "recorded_prefix" if use_recorded_prefix else "live_mpc"
            ),
            "source_frame": source_frame,
            "start_episode_frame": start_frame,
            "end_episode_frame": _episode_frame(raw),
            "requested_frames": requested,
            "advanced_frames": advanced,
            "action": action.to_dict(),
            "predicted_threat_count": (
                None if decision is None else len(decision.threats)
            ),
            "predicted_collision": (
                None if evaluation is None else evaluation.collided
            ),
            "predicted_collision_frames": (
                None if evaluation is None else evaluation.collision_frames
            ),
            "predicted_earliest_collision_frame": (
                None if evaluation is None else evaluation.earliest_collision_frame
            ),
            "predicted_minimum_margin": (
                None if evaluation is None else evaluation.minimum_margin
            ),
            "region_anchor": (
                None if decision is None or decision.region_anchor is None else
                {"x": decision.region_anchor[0], "y": decision.region_anchor[1]}
            ),
            "region_crossing": None if decision is None else decision.region_crossing,
            "region_path_margin": None if decision is None else decision.region_path_margin,
            "region_evacuating": None if decision is None else decision.region_evacuating,
            "region_target_rows_ahead": (
                None if decision is None else decision.region_target_rows_ahead
            ),
            "region_navigation_mode": (
                None if decision is None else decision.region_navigation_mode
            ),
            "region_current_component": (
                None if decision is None else decision.region_current_component
            ),
            "region_target_component": (
                None if decision is None else decision.region_target_component
            ),
            "region_portal": (
                None if decision is None else decision.region_portal
            ),
            "region_deadline_slack": (
                None if decision is None else decision.region_deadline_slack
            ),
            "region_phase": None if decision is None else decision.region_phase,
            "region_phase_started_frame": (
                None if decision is None else decision.region_phase_started_frame
            ),
            "region_learned_cycle_frames": (
                None if decision is None else decision.region_learned_cycle_frames
            ),
            "region_frames_until_expansion": (
                None if decision is None else decision.region_frames_until_expansion
            ),
            "region_observed_radius": (
                None if decision is None else decision.region_observed_radius
            ),
            "using_committed_plan": (
                None if decision is None else decision.using_committed_plan
            ),
            "committed_plan_immediate_margin": (
                None
                if decision is None else
                decision.committed_plan_immediate_margin
            ),
            "committed_plan_current_horizon_margin": (
                None
                if decision is None else
                decision.committed_plan_current_horizon_margin
            ),
            "planned_actions": (
                None
                if decision is None else
                [value.to_dict() for value in decision.planned_actions]
            ),
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
    success = engine_terminated and engine_reason == "attack_complete"
    policy_validation_eligible = recorded_prefix is None
    final_episode_frame = _episode_frame(raw)
    engine_advanced = None
    if initial_episode_frame is not None and final_episode_frame is not None:
        engine_advanced = max(0, final_episode_frame - initial_episode_frame)
    runtime_identity = ping.get("runtime_identity")
    return {
        "schema_version": 1,
        "run_kind": "live_luastg_delayed_visible_mpc_teacher",
        "acceptance_claim": False,
        "implementation_sha256": source_tree_sha256(),
        "success": success,
        "passed": success and policy_validation_eligible,
        "episode_completed": success,
        "policy_validation_eligible": policy_validation_eligible,
        "success_criterion": "terminated with termination_reason=attack_complete",
        "policy_validation_criterion": (
            "attack_complete with no recorded action prefix"
        ),
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
        "decision_count": len(decisions),
        "shoot_frames": shot_frames,
        "shoot_rate": shot_frames / logical_frames if logical_frames else 0.0,
        "unsafe_shot_frames": unsafe_shot_frames,
        "unsafe_shot_frames_excludes_recorded_prefix": recorded_prefix is not None,
        "recorded_prefix": (
            {"enabled": False}
            if recorded_prefix is None else {
                "enabled": True,
                "artifact": str(recorded_prefix.path),
                "artifact_sha256": recorded_prefix.sha256,
                "scenario": recorded_prefix.scenario,
                "attack": recorded_prefix.attack,
                "seed": recorded_prefix.seed,
                "player": recorded_prefix.player,
                "initial_episode_frame": recorded_prefix.initial_episode_frame,
                "until_episode_frame": prefix_until_frame,
                "available_decisions": len(recorded_prefix.decisions),
                "selected_decisions": len(selected_prefix_decisions),
                "used_decisions": prefix_decisions_used,
                "used_frames": prefix_frames_used,
                "last_recorded_end_episode_frame": prefix_last_end_frame,
                "effective_live_switch_episode_frame": live_first_start_frame,
            }
        ),
        "engine": {
            "protocol": ping.get("protocol"),
            "session_id": ping.get("session_id"),
            "process_nonce": ping.get("process_nonce"),
            "runtime_identity": (
                dict(runtime_identity) if isinstance(runtime_identity, Mapping) else {}
            ),
            "catalog_entry": catalog_entry,
        },
        "outcome_evidence": trace.report(raw),
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
            "prefix_artifact": (
                None if recorded_prefix is None else str(recorded_prefix.path)
            ),
            "prefix_until_frame": prefix_until_frame,
            "reset_options": {},
            "authority_state_shield": False,
            "spell_forced_off": True,
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
    "RecordedActionPrefix",
    "RecordedPrefixDecision",
    "load_recorded_action_prefix",
    "run_engine_mpc_play",
]
