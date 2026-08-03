"""Native DAgger collection on policy-visited LuaSTG trajectories."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
import math
import random
from typing import Any, Mapping

from .engine import EngineClient, EngineProtocolError
from .engine_runtime import verify_runtime_source_fingerprints
from .engine_mpc import CandidateEvaluation, EngineMPC, MPCDecision
from .engine_mpc_play import (
    _catalog_stage,
    _controller_observation,
    _nearest_collidables,
    _player_position,
)
from .engine_play import (
    VisualPolicyController,
    _OutcomeTrace,
    _commit_executed_action,
    _catalog_entry,
    _episode_frame,
    _observation,
    _stream_policy_control_inputs,
)
from .engine_vision import EngineStreamVision
from .native_dataset import NativeEpisodeBuffer, risk_from_clearance
from .protocol import Action
from .provenance import source_tree_sha256
from .render_performance import RenderPerformanceTrace
from .vision import VisionConfig


@dataclass(frozen=True, slots=True)
class EngineDAggerConfig:
    """Execution and intervention settings for one native DAgger episode."""

    max_frames: int = 7200
    decision_interval: int = 3
    observation_delay: int = 5
    teacher_probability: float = 0.25
    intervention_margin: float = 12.0
    intervention_regret: float = 8.0
    intervene_on_disagreement: bool = False
    # Retained for CLI/report compatibility; it no longer gates firing.
    shoot_minimum_margin: float = 12.0
    supervision_mode: str = "teacher"
    render: bool = False
    render_every: int = 1

    def __post_init__(self) -> None:
        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if self.decision_interval != 3:
            raise ValueError("native DAgger actions must be held for exactly three frames")
        if self.observation_delay < 0:
            raise ValueError("observation_delay cannot be negative")
        if not 0.0 <= self.teacher_probability <= 1.0:
            raise ValueError("teacher_probability must be in [0, 1]")
        finite_nonnegative = (
            self.intervention_margin,
            self.intervention_regret,
            self.shoot_minimum_margin,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in finite_nonnegative):
            raise ValueError("DAgger safety margins must be finite and nonnegative")
        if self.supervision_mode not in {"teacher", "corrective"}:
            raise ValueError(
                "supervision_mode must be 'teacher' or 'corrective'"
            )
        if not isinstance(self.intervene_on_disagreement, bool):
            raise ValueError("intervene_on_disagreement must be a Boolean")
        if (
            isinstance(self.render_every, bool)
            or not isinstance(self.render_every, int)
            or not 1 <= self.render_every <= 600
        ):
            raise ValueError("render_every must be an integer in [1, 600]")


def _evaluation_for_action(
    decision: MPCDecision,
    action: Action,
) -> CandidateEvaluation:
    matches = [
        evaluation
        for evaluation in decision.evaluations
        if evaluation.action.discrete == action.discrete
    ]
    # The MPC action set has one stationary action because focus speed is
    # irrelevant without movement, while the policy vocabulary has both.
    if not matches and action.move_x == 0 and action.move_y == 0:
        neutral = [
            evaluation
            for evaluation in decision.evaluations
            if evaluation.action.move_x == 0 and evaluation.action.move_y == 0
        ]
        if len(neutral) == 1:
            return replace(neutral[0], action=action)
    if len(matches) != 1:
        raise RuntimeError(
            "MPC decision must contain exactly one evaluation for every policy action"
        )
    return matches[0]


def _intervention_reason(
    student: CandidateEvaluation,
    teacher: CandidateEvaluation,
    *,
    forced: bool,
    disagreed: bool,
    intervene_on_disagreement: bool,
    margin: float,
    regret: float,
) -> str | None:
    """Return why the teacher must execute, keeping labels independent of the gate."""

    if forced:
        return "scheduled_teacher"
    if student.collided:
        return "predicted_collision"
    if student.minimum_margin < margin:
        return "minimum_margin"
    if teacher.minimum_margin - student.minimum_margin > regret:
        return "clearance_regret"
    if intervene_on_disagreement and disagreed:
        return "policy_disagreement"
    return None


def _movement_with_continuous_fire(movement: Action) -> Action:
    return Action(
        move_x=movement.move_x,
        move_y=movement.move_y,
        slow=movement.slow,
        shoot=True,
        spell=False,
    )


def run_engine_dagger_play(
    client: EngineClient,
    *,
    scenario: str,
    attack: int | None,
    seed: int,
    player: str,
    student: VisualPolicyController,
    teacher: EngineMPC,
    episode: NativeEpisodeBuffer,
    config: EngineDAggerConfig = EngineDAggerConfig(),
    vision_config: VisionConfig | None = None,
    stage: str | None = None,
) -> dict[str, Any]:
    """Run a policy trajectory, label every visited state, and report strict outcome.

    The student receives its checkpoint-declared streaming inputs. The exact-state
    MPC teacher is training-only: it supplies labels at every decision and executes
    only for the configured beta schedule or when the student's candidate is unsafe.
    """

    if student.inference_mode != "stream":
        raise ValueError("native DAgger requires a streaming policy checkpoint")
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
    if teacher.config.decision_interval != config.decision_interval:
        raise ValueError("teacher and runner decision intervals differ")
    if teacher.config.observation_delay != config.observation_delay:
        raise ValueError("teacher and runner observation delays differ")
    vision_config = vision_config or VisionConfig(
        history=1,
        observation_delay=config.observation_delay,
    )
    if vision_config.history != 1:
        raise ValueError("native DAgger stream observations must use history=1")
    if vision_config.observation_delay != config.observation_delay:
        raise ValueError("student and teacher observation delays differ")

    ping = client.ping()
    runtime_source_verification = verify_runtime_source_fingerprints(ping)
    catalog_response = client.catalog()
    if stage is None:
        assert attack is not None
        catalog_entry = _catalog_entry(catalog_response, scenario, attack)
        response = client.reset(
            scenario,
            attack,
            seed=int(seed),
            player=player,
            options={},
        )
    else:
        catalog_entry = _catalog_stage(catalog_response, stage)
        response = client.reset_stage(
            stage,
            seed=int(seed),
            player=player,
            options={},
        )

    raw = _observation(response)
    initial_episode_frame = _episode_frame(raw)
    teacher.reset()
    student.reset_for_seed(int(seed))
    stream_vision = EngineStreamVision(vision_config)
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
    rng = random.Random(int(seed))
    decisions: list[dict[str, Any]] = []
    logical_frames = 0
    shot_frames = 0
    interventions = 0
    scheduled_interventions = 0
    safety_interventions = 0
    policy_disagreement_interventions = 0
    agreements = 0
    terminal_before: Mapping[str, Any] | None = None
    terminal_action: Action | None = None
    previous_executed_action: Action | None = None

    while raw.get("terminated") is not True and logical_frames < config.max_frames:
        visible = stream_vision.observe()
        student_action = student.select(visible)
        controller_input = _controller_observation(delayed_observations[0], raw)
        teacher_decision = teacher.select(controller_input)
        teacher_evaluation = _evaluation_for_action(
            teacher_decision,
            teacher_decision.action,
        )
        teacher_action = _movement_with_continuous_fire(teacher_decision.action)
        student_evaluation = _evaluation_for_action(teacher_decision, student_action)
        forced = rng.random() < config.teacher_probability
        reason = _intervention_reason(
            student_evaluation,
            teacher_evaluation,
            forced=forced,
            disagreed=student_action.discrete != teacher_action.discrete,
            intervene_on_disagreement=config.intervene_on_disagreement,
            margin=config.intervention_margin,
            regret=config.intervention_regret,
        )
        intervened = reason is not None
        movement = teacher_action if intervened else student_action
        action = _movement_with_continuous_fire(movement)

        supervised_action = (
            action if config.supervision_mode == "corrective" else teacher_action
        )
        episode.record(
            visible,
            supervised_action,
            risk_from_clearance(teacher_evaluation.minimum_margin),
            previous_action=previous_executed_action,
        )
        agreements += int(student_action.discrete == teacher_action.discrete)
        interventions += int(intervened)
        scheduled_interventions += int(reason == "scheduled_teacher")
        safety_interventions += int(reason in {
            "predicted_collision",
            "minimum_margin",
            "clearance_regret",
        })
        policy_disagreement_interventions += int(reason == "policy_disagreement")

        start_frame = _episode_frame(raw)
        requested = min(config.decision_interval, config.max_frames - logical_frames)
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
            stream_vision.push(raw)
            delayed_observations.append(raw)
            trace.push(raw)
            render_performance.push(raw)
            logical_frames += 1
            advanced += 1
            shot_frames += int(action.shoot)
            if raw.get("terminated") is True:
                terminal_before = before
                terminal_action = action
                break
        if advanced:
            _commit_executed_action(student, action, frames=advanced)
            previous_executed_action = action
        decisions.append({
            "decision": len(decisions),
            "source_frame": int(visible.source_frame),
            "start_episode_frame": start_frame,
            "end_episode_frame": _episode_frame(raw),
            "advanced_frames": advanced,
            "student_action": student_action.to_dict(),
            "teacher_action": teacher_action.to_dict(),
            "executed_action": action.to_dict(),
            "supervised_action": supervised_action.to_dict(),
            "student_teacher_agreement": (
                student_action.discrete == teacher_action.discrete
            ),
            "teacher_intervened": intervened,
            "intervention_reason": reason,
            "student_predicted_collision": student_evaluation.collided,
            "student_predicted_minimum_margin": student_evaluation.minimum_margin,
            "teacher_predicted_minimum_margin": teacher_evaluation.minimum_margin,
        })

    engine_terminated = raw.get("terminated") is True
    engine_reason = raw.get("termination_reason") if engine_terminated else None
    termination_reason = engine_reason if engine_terminated else "max_frames"
    outcome_evidence = trace.report(raw)
    final_player = outcome_evidence.get("final_player")
    death_value = final_player.get("death") if isinstance(final_player, Mapping) else None
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
    engine_advanced = None
    if initial_episode_frame is not None and final_episode_frame is not None:
        engine_advanced = max(0, final_episode_frame - initial_episode_frame)
    runtime_identity = ping.get("runtime_identity")
    decision_count = len(decisions)
    return {
        "schema_version": 3,
        "run_kind": "live_luastg_native_dagger",
        "acceptance_claim": False,
        "training_only": True,
        "implementation_sha256": source_tree_sha256(),
        "success": success,
        "passed": success,
        "episode_completed": success,
        "teacher_assisted_success": success,
        "teacher_success": success,
        "pure_policy": False,
        "pure_policy_success": False,
        "pure_policy_validation_eligible": False,
        "success_criterion": (
            f"terminated with termination_reason={completion_reason} and "
            "outcome_evidence.final_player.death=0"
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
        "decision_count": decision_count,
        "shoot_frames": shot_frames,
        "shoot_rate": shot_frames / logical_frames if logical_frames else 0.0,
        "shoot_command_frames": shot_frames,
        "shoot_command_rate": shot_frames / logical_frames if logical_frames else 0.0,
        "continuous_fire": True,
        "student_teacher_agreements": agreements,
        "student_teacher_agreement_rate": (
            agreements / decision_count if decision_count else 0.0
        ),
        "teacher_interventions": interventions,
        "teacher_intervention_rate": (
            interventions / decision_count if decision_count else 0.0
        ),
        "scheduled_teacher_interventions": scheduled_interventions,
        "safety_teacher_interventions": safety_interventions,
        "policy_disagreement_interventions": policy_disagreement_interventions,
        "demonstration_supervision": {
            "mode": config.supervision_mode,
            "target": (
                "executed_action: student action when unassisted, teacher action "
                "when intervened"
                if config.supervision_mode == "corrective" else
                "teacher_action at every visited state"
            ),
            "teacher_correction_targets": (
                interventions if config.supervision_mode == "corrective" else 0
            ),
            "student_execution_targets": (
                decision_count - interventions
                if config.supervision_mode == "corrective" else 0
            ),
            "teacher_only_targets": (
                decision_count if config.supervision_mode == "teacher" else 0
            ),
        },
        "engine": {
            "protocol": ping.get("protocol"),
            "session_id": ping.get("session_id"),
            "process_nonce": ping.get("process_nonce"),
            "runtime_identity": (
                dict(runtime_identity) if isinstance(runtime_identity, Mapping) else {}
            ),
            "runtime_source_verification": runtime_source_verification,
            "catalog_entry": catalog_entry,
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
        "controller": {
            "kind": "streaming_visual_policy_with_training_only_mpc_teacher",
            "student": {
                "scenario_key": student.scenario_key,
                "device": student.device,
                "inference_mode": student.inference_mode,
                "proficiency": asdict(student.proficiency),
            },
            "teacher": {
                "config": asdict(teacher.config),
                "uses_raw_object_ids": True,
                "uses_raw_velocity_fields": True,
                "uses_class_names_or_script_timers_for_dodging": False,
            },
        },
        "config": {
            **asdict(config),
            "vision": asdict(vision_config),
            "reset_options": {},
            "continuous_fire": True,
            "shoot_minimum_margin_controls_fire": False,
            "spell_forced_off": True,
            "student_control_inputs": _stream_policy_control_inputs(student),
            "teacher_data_is_training_only": True,
            "failed_episode_labels_must_be_discarded": True,
        },
        "decisions": decisions,
    }


__all__ = [
    "EngineDAggerConfig",
    "run_engine_dagger_play",
]
