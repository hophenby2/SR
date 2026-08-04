"""Run one continuous Stage 1-5 campaign with the memory-free live MPC."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from .engine import EngineClient, EngineProtocolError
from .engine_mpc import EngineMPC
from .engine_mpc_play import (
    _MaturedControllerObservationFeed,
    _effective_action,
    _episode_frame,
    _finite_or_none,
    _nearest_collidables,
    _player_position,
    _selected_evaluation,
)
from .engine_play import _OutcomeTrace, _observation
from .engine_runtime import verify_runtime_source_fingerprints
from .provenance import source_tree_sha256
from .render_performance import RenderPerformanceTrace


_CAMPAIGN_STAGE_COUNT = 5
_TERMINAL_OBSERVATION_WINDOW_FRAMES = 24
_TERMINAL_CONTROLLER_WINDOW_DECISIONS = 8


@dataclass(frozen=True, slots=True)
class EngineMPCCampaignConfig:
    max_frames: int = 120_000
    decision_interval: int = 3
    observation_delay: int = 5
    render: bool = False
    render_every: int = 1

    def __post_init__(self) -> None:
        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if self.decision_interval != 3:
            raise ValueError("live actions must be held for exactly three frames")
        if self.observation_delay < 0:
            raise ValueError("observation_delay cannot be negative")
        if (
            isinstance(self.render_every, bool)
            or not isinstance(self.render_every, int)
            or not 1 <= self.render_every <= 600
        ):
            raise ValueError("render_every must be an integer in [1, 600]")


def _expected_stage_name(index: int, difficulty: str) -> str:
    return f"Stage {index}@{difficulty}"


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _campaign_snapshot(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    campaign = observation.get("campaign")
    if not isinstance(campaign, Mapping):
        raise EngineProtocolError("campaign observation has no campaign object")
    return campaign


def _campaign_snapshot_errors(
    campaign: Mapping[str, Any],
    *,
    difficulty: str,
) -> list[str]:
    errors: list[str] = []
    stage_index = _integer(campaign.get("stage_index"))
    stage_count = _integer(campaign.get("stage_count"))
    stages_completed = _integer(campaign.get("stages_completed"))
    transition_count = _integer(campaign.get("stage_transition_count"))
    campaign_complete = campaign.get("campaign_complete")
    completed = campaign.get("completed_stages")
    transitions = campaign.get("transitions")

    if campaign.get("schema_version") != 1:
        errors.append("campaign schema_version must be 1")
    if campaign.get("difficulty") != difficulty:
        errors.append("campaign difficulty differs from the reset request")
    if stage_count != _CAMPAIGN_STAGE_COUNT:
        errors.append("campaign stage_count must be exactly five")
    if stage_index is None or not 1 <= stage_index <= _CAMPAIGN_STAGE_COUNT:
        errors.append("campaign stage_index is invalid")
    elif campaign.get("stage_name") != _expected_stage_name(stage_index, difficulty):
        errors.append("campaign stage_name does not match stage_index")
    if stages_completed is None or not 0 <= stages_completed <= _CAMPAIGN_STAGE_COUNT:
        errors.append("campaign stages_completed is invalid")
    if transition_count is None or not 0 <= transition_count <= _CAMPAIGN_STAGE_COUNT:
        errors.append("campaign stage_transition_count is invalid")
    if not isinstance(completed, list):
        errors.append("campaign completed_stages must be an array")
    elif stages_completed is not None and len(completed) != stages_completed:
        errors.append("campaign completed_stages length differs from its counter")
    if not isinstance(transitions, list):
        errors.append("campaign transitions must be an array")
    elif transition_count is not None and len(transitions) != transition_count:
        errors.append("campaign transitions length differs from its counter")
    if not isinstance(campaign.get("active_content_seen"), bool):
        errors.append("campaign active_content_seen must be Boolean")
    if not isinstance(campaign.get("stage_active_content_seen"), bool):
        errors.append("campaign stage_active_content_seen must be Boolean")
    if not isinstance(campaign_complete, bool):
        errors.append("campaign campaign_complete must be Boolean")
    if not isinstance(campaign.get("initial_resources"), Mapping):
        errors.append("campaign initial_resources must be an object")
    if not isinstance(campaign.get("resources"), Mapping):
        errors.append("campaign resources must be an object")
    if not isinstance(campaign.get("initial_hidden_route"), bool):
        errors.append("campaign initial_hidden_route must be Boolean")
    if not isinstance(campaign.get("hidden_route"), bool):
        errors.append("campaign hidden_route must be Boolean")

    if stage_index is not None and transition_count is not None:
        expected_transitions = _CAMPAIGN_STAGE_COUNT if campaign_complete is True else stage_index - 1
        if transition_count != expected_transitions:
            errors.append("campaign transition count does not match its current stage")
    if stage_index is not None and stages_completed is not None:
        expected_completed = _CAMPAIGN_STAGE_COUNT if campaign_complete is True else stage_index - 1
        if stages_completed != expected_completed:
            errors.append("campaign completed count does not match its current stage")

    if isinstance(completed, list):
        for ordinal, value in enumerate(completed, start=1):
            prefix = f"completed stage {ordinal}"
            if not isinstance(value, Mapping):
                errors.append(f"{prefix} must be an object")
                continue
            if value.get("stage_index") != ordinal:
                errors.append(f"{prefix} has the wrong stage_index")
            if value.get("stage_name") != _expected_stage_name(ordinal, difficulty):
                errors.append(f"{prefix} has the wrong stage_name")
            if _integer(value.get("completion_episode_frame")) is None:
                errors.append(f"{prefix} has no completion frame")
            if not isinstance(value.get("active_content_seen"), bool):
                errors.append(f"{prefix} active-content evidence is not Boolean")
            if not isinstance(value.get("resources"), Mapping):
                errors.append(f"{prefix} resources must be an object")
            if not isinstance(value.get("hidden_route"), bool):
                errors.append(f"{prefix} hidden_route must be Boolean")

    if isinstance(transitions, list):
        for ordinal, value in enumerate(transitions, start=1):
            prefix = f"campaign transition {ordinal}"
            if not isinstance(value, Mapping):
                errors.append(f"{prefix} must be an object")
                continue
            final = ordinal == _CAMPAIGN_STAGE_COUNT
            expected_to_index = 0 if final else ordinal + 1
            expected_to_name = (
                "menu" if final else _expected_stage_name(ordinal + 1, difficulty)
            )
            expected = {
                "from_stage_index": ordinal,
                "from_stage_name": _expected_stage_name(ordinal, difficulty),
                "to_stage_index": expected_to_index,
                "to_stage_name": expected_to_name,
            }
            if any(value.get(name) != item for name, item in expected.items()):
                errors.append(f"{prefix} does not preserve the Stage 1-5 order")
            if _integer(value.get("episode_frame")) is None:
                errors.append(f"{prefix} has no episode frame")
            if not isinstance(value.get("active_content_seen"), bool):
                errors.append(f"{prefix} active-content evidence is not Boolean")
            if not isinstance(value.get("resources"), Mapping):
                errors.append(f"{prefix} resources must be an object")
            if not isinstance(value.get("hidden_route"), bool):
                errors.append(f"{prefix} hidden_route must be Boolean")
    return errors


def _validated_campaign_snapshot(
    observation: Mapping[str, Any],
    *,
    difficulty: str,
) -> Mapping[str, Any]:
    campaign = _campaign_snapshot(observation)
    errors = _campaign_snapshot_errors(campaign, difficulty=difficulty)
    if errors:
        raise EngineProtocolError("; ".join(errors))
    return campaign


def _validate_reset_metadata(
    response: Mapping[str, Any],
    *,
    difficulty: str,
    seed: int,
    player: str,
) -> dict[str, Any]:
    reset = response.get("reset")
    expected_stage = _expected_stage_name(1, difficulty)
    if not isinstance(reset, Mapping) or (
        reset.get("episode_kind") != "campaign"
        or reset.get("difficulty") != difficulty
        or reset.get("stage_index") != 1
        or reset.get("stage_name") != expected_stage
        or reset.get("stage_count") != _CAMPAIGN_STAGE_COUNT
        or reset.get("seed") != seed
        or reset.get("player") != player
        or "replay" in reset
    ):
        raise EngineProtocolError("engine returned invalid campaign reset metadata")
    return dict(reset)


def _catalog_campaign_stages(
    response: Mapping[str, Any],
    *,
    difficulty: str,
) -> list[dict[str, Any]]:
    catalog = response.get("catalog")
    stages = catalog.get("stages") if isinstance(catalog, Mapping) else None
    if not isinstance(stages, list):
        raise EngineProtocolError("engine catalog has no stages array")
    selected = [
        dict(value)
        for value in stages
        if isinstance(value, Mapping) and value.get("difficulty") == difficulty
    ]
    expected_names = [
        _expected_stage_name(index, difficulty)
        for index in range(1, _CAMPAIGN_STAGE_COUNT + 1)
    ]
    if len(selected) != _CAMPAIGN_STAGE_COUNT or [
        value.get("stage") for value in selected
    ] != expected_names or [value.get("stage_index") for value in selected] != list(
        range(1, _CAMPAIGN_STAGE_COUNT + 1)
    ):
        raise EngineProtocolError(
            "live catalog does not contain one ordered five-stage campaign"
        )
    return selected


def _zero_death(observation: Mapping[str, Any]) -> bool:
    player = observation.get("player")
    death = player.get("death") if isinstance(player, Mapping) else None
    return (
        not isinstance(death, bool)
        and isinstance(death, (int, float))
        and math.isfinite(float(death))
        and float(death) == 0.0
    )


def run_engine_mpc_campaign(
    client: EngineClient,
    *,
    difficulty: str,
    seed: int,
    player: str,
    controller: EngineMPC,
    config: EngineMPCCampaignConfig = EngineMPCCampaignConfig(),
) -> dict[str, Any]:
    """Run one reset and keep the same memory-free MPC through Stage 1-5."""

    if difficulty not in {"Normal", "Lunatic"}:
        raise ValueError("difficulty must be Normal or Lunatic")
    if controller.config.decision_interval != config.decision_interval:
        raise ValueError("controller and runner decision intervals differ")
    if controller.config.observation_delay != config.observation_delay:
        raise ValueError("controller and runner observation delays differ")
    if controller.config.region_dynamics_memory is not None:
        raise ValueError("campaign controller must not contain region dynamics memory")

    implementation_sha256 = source_tree_sha256()
    ping = client.ping()
    commands = ping.get("commands")
    if not isinstance(commands, list) or "reset_campaign" not in commands:
        raise EngineProtocolError("engine bridge does not advertise reset_campaign")
    runtime_source_verification = verify_runtime_source_fingerprints(ping)
    catalog_entries = _catalog_campaign_stages(client.catalog(), difficulty=difficulty)
    reset_response = client.reset_campaign(
        difficulty,
        seed=int(seed),
        player=player,
        options={},
    )
    reset_metadata = _validate_reset_metadata(
        reset_response,
        difficulty=difficulty,
        seed=int(seed),
        player=player,
    )
    raw = _observation(reset_response)
    campaign = _validated_campaign_snapshot(raw, difficulty=difficulty)
    if (
        campaign.get("stage_index") != 1
        or campaign.get("stages_completed") != 0
        or campaign.get("stage_transition_count") != 0
        or campaign.get("campaign_complete") is not False
    ):
        raise EngineProtocolError("campaign did not start at an uncompleted Stage 1")

    initial_episode_frame = _episode_frame(raw)
    controller.reset()
    delayed_observations: deque[Mapping[str, Any]] = deque(
        [raw] * (config.observation_delay + 1),
        maxlen=config.observation_delay + 1,
    )
    observation_feed = _MaturedControllerObservationFeed(
        controller,
        excluded_fields=("campaign",),
    )
    terminal_observations: deque[Mapping[str, Any]] = deque(
        [raw],
        maxlen=_TERMINAL_OBSERVATION_WINDOW_FRAMES,
    )
    terminal_controller_inputs: deque[dict[str, Any]] = deque(
        maxlen=_TERMINAL_CONTROLLER_WINDOW_DECISIONS,
    )
    display = client.set_rendering(config.render, every=config.render_every)
    if display.get("render") is not config.render or display.get("every") != config.render_every:
        raise EngineProtocolError("engine did not apply the requested display state")

    outcome_trace = _OutcomeTrace()
    outcome_trace.push(raw)
    render_performance = RenderPerformanceTrace()
    render_performance.push(raw)
    decisions: list[dict[str, Any]] = []
    stage_boundaries: list[dict[str, Any]] = []
    logical_frames = 0
    shot_frames = 0
    predicted_collision_plan_frames = 0
    terminal_before: Mapping[str, Any] | None = None
    terminal_action = None
    previous_transition_count = 0

    while raw.get("terminated") is not True and logical_frames < config.max_frames:
        campaign_at_start = _validated_campaign_snapshot(raw, difficulty=difficulty)
        delayed = delayed_observations[0]
        controller_input = observation_feed.update(delayed, raw)
        terminal_controller_inputs.append({
            "decision": len(decisions),
            "source_frame": _episode_frame(controller_input),
            "observation": controller_input,
        })
        decision = controller.select(controller_input)
        evaluation = _selected_evaluation(decision)
        action = _effective_action(decision)
        overlay_state = (
            controller.controller_overlay_state(decision, controller_input)
            if config.render else None
        )
        start_frame = _episode_frame(raw)
        requested = min(config.decision_interval, config.max_frames - logical_frames)
        advanced = 0
        boundary_after_action: dict[str, Any] | None = None

        for action_frame in range(requested):
            before = raw
            before_frame = _episode_frame(before)
            response = client.step(
                action,
                repeat=1,
                controller_overlay_state=(overlay_state if action_frame == 0 else None),
            )
            raw = _observation(response)
            terminal_observations.append(raw)
            after_frame = _episode_frame(raw)
            if (
                before_frame is None
                or after_frame is None
                or after_frame != before_frame + 1
            ):
                raise EngineProtocolError(
                    "engine campaign episode_frame did not advance by exactly one frame"
                )
            campaign_after = _validated_campaign_snapshot(raw, difficulty=difficulty)
            transition_count = int(campaign_after["stage_transition_count"])
            if transition_count < previous_transition_count or (
                transition_count > previous_transition_count + 1
            ):
                raise EngineProtocolError(
                    "campaign transition count did not advance by zero or one"
                )

            logical_frames += 1
            advanced += 1
            shot_frames += int(action.shoot)
            predicted_collision_plan_frames += int(evaluation.collided)
            outcome_trace.push(raw)
            render_performance.push(raw)

            if transition_count == previous_transition_count + 1:
                transition = campaign_after["transitions"][-1]
                boundary_after_action = {
                    "transition_count": transition_count,
                    "episode_frame": after_frame,
                    "from_stage_index": transition.get("from_stage_index"),
                    "from_stage_name": transition.get("from_stage_name"),
                    "to_stage_index": transition.get("to_stage_index"),
                    "to_stage_name": transition.get("to_stage_name"),
                    "active_content_seen": transition.get("active_content_seen"),
                    "controller_transient_state_cleared": transition_count < 5,
                }
                previous_transition_count = transition_count
                if transition_count < _CAMPAIGN_STAGE_COUNT:
                    controller.on_stage_boundary()
                    observation_feed.reset()
                    delayed_observations = deque(
                        [raw] * (config.observation_delay + 1),
                        maxlen=config.observation_delay + 1,
                    )
                    terminal_observations.clear()
                    terminal_observations.append(raw)
                    terminal_controller_inputs.clear()
                    stage_boundaries.append(boundary_after_action)
                elif raw.get("terminated") is not True:
                    raise EngineProtocolError(
                        "final campaign transition did not terminate the episode"
                    )
                terminal_before = before if raw.get("terminated") is True else None
                terminal_action = action if raw.get("terminated") is True else None
                break

            delayed_observations.append(raw)
            observation_feed.update(delayed_observations[0], raw)
            if raw.get("terminated") is True:
                terminal_before = before
                terminal_action = action
                break

        decisions.append({
            "decision": len(decisions),
            "control_source": "live_mpc",
            "stage_index": campaign_at_start.get("stage_index"),
            "stage_name": campaign_at_start.get("stage_name"),
            "campaign_transition_count_start": campaign_at_start.get(
                "stage_transition_count"
            ),
            "campaign_transition_count_end": _campaign_snapshot(raw).get(
                "stage_transition_count"
            ),
            "source_frame": decision.source_frame,
            "start_episode_frame": start_frame,
            "end_episode_frame": _episode_frame(raw),
            "requested_frames": requested,
            "advanced_frames": advanced,
            "action": action.to_dict(),
            "predicted_threat_count": len(decision.threats),
            "predicted_launch_motion_count": sum(
                threat.launch_motion_inferred for threat in decision.threats
            ),
            "predicted_accelerating_threat_count": sum(
                threat.acceleration_horizon > 0 for threat in decision.threats
            ),
            "predicted_collision": evaluation.collided,
            "predicted_collision_frames": evaluation.collision_frames,
            "predicted_earliest_collision_frame": evaluation.earliest_collision_frame,
            "predicted_minimum_margin": _finite_or_none(evaluation.minimum_margin),
            "predicted_minimum_nonregion_margin": _finite_or_none(
                evaluation.minimum_nonregion_margin
            ),
            "predicted_minimum_region_margin": _finite_or_none(
                evaluation.minimum_region_margin
            ),
            "predicted_immediate_corner_clearance": _finite_or_none(
                evaluation.immediate_corner_clearance
            ),
            "region_anchor": (
                None if decision.region_anchor is None else
                {"x": decision.region_anchor[0], "y": decision.region_anchor[1]}
            ),
            "region_crossing": decision.region_crossing,
            "region_path_margin": _finite_or_none(decision.region_path_margin),
            "region_evacuating": decision.region_evacuating,
            "region_target_rows_ahead": decision.region_target_rows_ahead,
            "region_navigation_mode": decision.region_navigation_mode,
            "region_current_component": decision.region_current_component,
            "region_target_component": decision.region_target_component,
            "region_portal": decision.region_portal,
            "region_deadline_slack": _finite_or_none(
                decision.region_deadline_slack
            ),
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
            "gap_selected_width": _finite_or_none(decision.gap_selected_width),
            "gap_selected_lifetime_frames": (
                decision.gap_selected_lifetime_frames
            ),
            "gap_navigation_mode": decision.gap_navigation_mode,
            "gap_plan_certified": decision.gap_plan_certified,
            "using_committed_plan": decision.using_committed_plan,
            "committed_plan_immediate_margin": _finite_or_none(
                decision.committed_plan_immediate_margin
            ),
            "committed_plan_current_horizon_margin": _finite_or_none(
                decision.committed_plan_current_horizon_margin
            ),
            "planned_actions": [
                value.to_dict() for value in decision.planned_actions
            ],
            "reporting_only_authority_player": _player_position(raw),
            "boundary_after_action": boundary_after_action,
        })

    engine_terminated = raw.get("terminated") is True
    engine_reason = raw.get("termination_reason") if engine_terminated else None
    termination_reason = engine_reason if engine_terminated else "max_frames"
    final_campaign = dict(_validated_campaign_snapshot(raw, difficulty=difficulty))
    outcome_evidence = outcome_trace.report(raw)
    final_episode_frame = _episode_frame(raw)
    engine_advanced_frames = None
    if initial_episode_frame is not None and final_episode_frame is not None:
        engine_advanced_frames = max(0, final_episode_frame - initial_episode_frame)

    completed = final_campaign.get("completed_stages")
    transitions = final_campaign.get("transitions")
    completed_values = completed if isinstance(completed, list) else []
    transition_values = transitions if isinstance(transitions, list) else []
    active_content_all_stages = (
        len(completed_values) == _CAMPAIGN_STAGE_COUNT
        and all(
            isinstance(value, Mapping) and value.get("active_content_seen") is True
            for value in completed_values
        )
        and len(transition_values) == _CAMPAIGN_STAGE_COUNT
        and all(
            isinstance(value, Mapping) and value.get("active_content_seen") is True
            for value in transition_values
        )
        and final_campaign.get("active_content_seen") is True
        and final_campaign.get("stage_active_content_seen") is True
    )
    all_live_mpc = bool(decisions) and all(
        value.get("control_source") == "live_mpc" for value in decisions
    )
    continuous_fire = (
        logical_frames > 0
        and shot_frames == logical_frames
        and all(
            value.get("action", {}).get("shoot") is True
            and value.get("action", {}).get("spell") is False
            for value in decisions
        )
    )
    external_memory = {
        "region_dynamics_memory_path": None,
        "region_dynamics_memory_sha256": None,
        "controller_region_dynamics_memory": None,
        "route_artifact": None,
        "route_library_artifact": None,
        "checkpoint": None,
        "action_prefix": None,
    }
    external_memory_free = all(value is None for value in external_memory.values())
    exact_boundaries = (
        len(stage_boundaries) == _CAMPAIGN_STAGE_COUNT - 1
        and all(
            value["transition_count"] == ordinal
            and value["from_stage_index"] == ordinal
            and value["to_stage_index"] == ordinal + 1
            and value["controller_transient_state_cleared"] is True
            for ordinal, value in enumerate(stage_boundaries, start=1)
        )
    )
    campaign_complete = (
        final_campaign.get("campaign_complete") is True
        and final_campaign.get("stage_count") == _CAMPAIGN_STAGE_COUNT
        and final_campaign.get("stages_completed") == _CAMPAIGN_STAGE_COUNT
        and final_campaign.get("stage_transition_count") == _CAMPAIGN_STAGE_COUNT
        and len(completed_values) == _CAMPAIGN_STAGE_COUNT
        and len(transition_values) == _CAMPAIGN_STAGE_COUNT
    )
    implementation_sha256_end = source_tree_sha256()
    implementation_source_unchanged = (
        implementation_sha256_end == implementation_sha256
    )
    success = (
        engine_terminated
        and engine_reason == "campaign_complete"
        and campaign_complete
        and active_content_all_stages
        and _zero_death(raw)
        and exact_boundaries
        and all_live_mpc
        and continuous_fire
        and external_memory_free
        and implementation_source_unchanged
    )

    runtime_identity = ping.get("runtime_identity")
    return {
        "schema_version": 1,
        "run_kind": "live_luastg_continuous_no_external_memory_campaign",
        "acceptance_claim": False,
        "implementation_sha256": implementation_sha256,
        "implementation_sha256_end": implementation_sha256_end,
        "implementation_source_unchanged": implementation_source_unchanged,
        "success": success,
        "passed": success,
        "episode_completed": success,
        "teacher_success": success,
        "pure_policy": False,
        "pure_policy_success": False,
        "pure_policy_validation_eligible": False,
        "success_criterion": (
            "one reset_campaign; exact ordered Stage 1-5 completion with active "
            "content in every stage; termination_reason=campaign_complete; finite "
            "final death=0; memory fields null; every decision live_mpc with "
            "shoot=true and spell=false; Python source unchanged during run"
        ),
        "episode_kind": "campaign",
        "difficulty": difficulty,
        "seed": int(seed),
        "player": player,
        "terminated": engine_terminated,
        "termination_reason": termination_reason,
        "engine_termination_reason": engine_reason,
        "frames": logical_frames,
        "engine_advanced_frames": engine_advanced_frames,
        "initial_episode_frame": initial_episode_frame,
        "final_episode_frame": final_episode_frame,
        "decision_count": len(decisions),
        "shoot_frames": shot_frames,
        "shoot_rate": shot_frames / logical_frames if logical_frames else 0.0,
        "continuous_fire": continuous_fire,
        "predicted_collision_plan_frames": predicted_collision_plan_frames,
        "reset_count": 1,
        "reset_command": "reset_campaign",
        "reset_metadata": reset_metadata,
        "native_replay": None,
        "replay_supported": False,
        "campaign_complete_evidence": campaign_complete,
        "all_stages_active_content_seen": active_content_all_stages,
        "all_control_sources_live_mpc": all_live_mpc,
        "stage_boundary_count": len(stage_boundaries),
        "stage_boundaries": stage_boundaries,
        "campaign": final_campaign,
        "catalog_stages": catalog_entries,
        "external_memory": external_memory,
        "external_memory_free": external_memory_free,
        "engine": {
            "protocol": ping.get("protocol"),
            "session_id": ping.get("session_id"),
            "process_nonce": ping.get("process_nonce"),
            "runtime_identity": (
                dict(runtime_identity) if isinstance(runtime_identity, Mapping) else {}
            ),
            "runtime_source_verification": runtime_source_verification,
        },
        "outcome_evidence": outcome_evidence,
        "render_performance": render_performance.report(),
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
        "terminal_observation_window": {
            "output_only_not_reused_by_controller": True,
            "authority_observations_are_controller_input": False,
            "controller_inputs_are_historical_live_inputs": True,
            "stage_index": final_campaign.get("stage_index"),
            "frame_capacity": _TERMINAL_OBSERVATION_WINDOW_FRAMES,
            "decision_capacity": _TERMINAL_CONTROLLER_WINDOW_DECISIONS,
            "observations": list(terminal_observations),
            "controller_inputs": list(terminal_controller_inputs),
        },
        "controller": {
            "kind": "visible_trajectory_mpc_teacher",
            "same_instance_across_stages": True,
            "stage_boundary_method": "on_stage_boundary",
            "config": asdict(controller.config),
            "uses_campaign_metadata_for_dodging": False,
            "uses_class_names_or_script_timers_for_dodging": False,
            "uses_delayed_visible_orientation": True,
        },
        "config": {
            **asdict(config),
            "reset_options": {},
            "region_dynamics_memory_path": None,
            "region_dynamics_memory_sha256": None,
            "route_artifact": None,
            "route_library_artifact": None,
            "checkpoint": None,
            "action_prefix": None,
            "authority_state_shield": False,
            "continuous_fire": True,
            "spell_forced_off": True,
            "campaign_metadata_is_controller_input": False,
        },
        "decisions": decisions,
    }


__all__ = [
    "EngineMPCCampaignConfig",
    "run_engine_mpc_campaign",
]
