from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from stg_lab.engine import EngineProtocolError
from stg_lab.engine_dagger import (
    EngineDAggerConfig,
    _movement_with_continuous_fire,
    run_engine_dagger_play,
)
from stg_lab.engine_mpc import CandidateEvaluation, MPCConfig, MPCDecision, movement_actions
from stg_lab.engine_runtime import local_runtime_source_fingerprints
from stg_lab.native_dataset import NativeEpisodeBuffer, NativeEpisodeIdentity
from stg_lab.policy import resolve_proficiency
from stg_lab.protocol import Action
from stg_lab.vision import VisionConfig, VisionObservation


def _observation(
    frame: int,
    *,
    terminated: bool = False,
    reason: str | None = None,
    death: int = 0,
) -> dict[str, Any]:
    return {
        "episode_frame": frame,
        "terminated": terminated,
        "termination_reason": reason,
        "performance": {"native_fps": 120.0, "object_count": 1},
        "stage": {"card_index": 4},
        "world": {"pl": -192.0, "pr": 192.0, "pb": -224.0, "pt": 224.0},
        "player": {
            "x": 0.0,
            "y": -176.0,
            "a": 0.5,
            "b": 0.5,
            "hspeed": 4.0,
            "lspeed": 2.0,
            "death": death,
            "protect": 0,
            "status": "normal",
        },
        "enemy_bullets": [],
        "enemies": [{
            "id": 20,
            "x": 0.0,
            "y": 120.0,
            "a": 16.0,
            "b": 16.0,
            "hp": 100.0,
            "maxhp": 100.0,
            "collidable": False,
        }],
        "nontjt_enemies": [],
        "indestructibles": [],
        "lasers": [],
    }


class FakeClient:
    def __init__(self, *, death: int = 0) -> None:
        self.frame = 0
        self.death = death
        self.actions: list[Action] = []
        self.runtime_source_crc32 = local_runtime_source_fingerprints()[0]

    def ping(self):
        return {
            "protocol": 2,
            "session_id": "dagger-test",
            "process_nonce": "dagger-process",
            "runtime_identity": {
                "process_id": 42,
                "source_crc32": self.runtime_source_crc32,
            },
        }

    def catalog(self):
        return {"catalog": {"attacks": [
            {"scenario": "okuu:Lunatic", "attack": 3, "card_index": 4},
        ], "stages": []}}

    def reset(self, scenario, attack, *, seed, player, options):
        self.frame = 0
        return {"observation": _observation(0)}

    def set_rendering(self, enabled: bool, *, every: int = 1):
        return {"render": enabled, "every": every}

    def step(self, action: Action, *, repeat: int = 1):
        assert repeat == 1
        self.actions.append(action)
        self.frame += 1
        terminated = self.frame >= 6
        return {"observation": _observation(
            self.frame,
            terminated=terminated,
            reason="attack_complete" if terminated else None,
            death=self.death if terminated else 0,
        )}


class FixedStudent:
    inference_mode = "stream"
    scenario_key = "test"
    device = "cpu"
    proficiency = resolve_proficiency("expert")

    def __init__(self, action: Action) -> None:
        self.action = action
        self.commits: list[tuple[Action, int]] = []

    def reset_for_seed(self, seed: int) -> None:
        self.seed = seed

    def select(self, visible: VisionObservation) -> Action:
        assert visible.global_frames.shape[0] == 1
        return self.action

    def commit_executed_action(self, action: Action, *, frames: int) -> None:
        self.commits.append((action, frames))


class FixedTeacher:
    def __init__(
        self,
        *,
        student_collides: bool,
        student_minimum_margin: float = 30.0,
    ) -> None:
        self.config = MPCConfig(observation_delay=0, horizon_frames=36)
        self.student_collides = student_collides
        self.student_minimum_margin = student_minimum_margin

    def reset(self) -> None:
        return None

    def select(self, observation) -> MPCDecision:
        teacher_action = Action(move_x=1, slow=True)
        evaluations = []
        for action in movement_actions():
            is_student = action.discrete == Action(move_x=-1, slow=True).discrete
            evaluations.append(CandidateEvaluation(
                action=action,
                collided=self.student_collides and is_student,
                collision_frames=int(self.student_collides and is_student),
                earliest_collision_frame=(1 if self.student_collides and is_student else None),
                minimum_margin=(
                    -1.0
                    if self.student_collides and is_student else
                    self.student_minimum_margin
                    if is_student else
                    30.0
                ),
                boundary_penalty=0.0,
                boss_alignment=0.0,
            ))
        return MPCDecision(
            action=teacher_action,
            source_frame=int(observation["episode_frame"]),
            recomputed=True,
            threats=(),
            evaluations=tuple(evaluations),
            region_anchor=None,
            region_crossing=False,
            region_path_margin=None,
            region_evacuating=False,
            region_target_rows_ahead=0,
            region_navigation_mode="none",
            region_current_component=None,
            region_target_component=None,
            region_portal=None,
            region_deadline_slack=None,
            planned_actions=(teacher_action,),
            using_committed_plan=False,
            committed_plan_immediate_margin=None,
            committed_plan_current_horizon_margin=None,
            region_phase="unknown",
            region_phase_started_frame=None,
            region_learned_cycle_frames=None,
            region_frames_until_expansion=None,
            region_observed_radius=None,
        )


def _run(
    *,
    student_collides: bool,
    student_minimum_margin: float = 30.0,
    death: int = 0,
    student_action: Action = Action(move_x=-1, slow=True),
    supervision_mode: str = "teacher",
    intervene_on_disagreement: bool = False,
    teacher_probability: float = 0.0,
    intervention_margin: float = 0.0,
    intervention_regret: float = 100.0,
    minimum_safety_margin_gain: float | None = None,
    student_only_prefix_frames: int = 0,
    action_selection: str = "joint",
    record_teacher_evaluations: bool = False,
):
    client = FakeClient(death=death)
    episode = NativeEpisodeBuffer(NativeEpisodeIdentity(
        episode_kind="attack",
        scenario="okuu:Lunatic",
        attack=3,
        seed=42,
    ))
    student = FixedStudent(student_action)
    student.action_selection = action_selection
    report = run_engine_dagger_play(
        client,  # type: ignore[arg-type]
        scenario="okuu:Lunatic",
        attack=3,
        seed=42,
        player="reimu_player",
        student=student,  # type: ignore[arg-type]
        teacher=FixedTeacher(
            student_collides=student_collides,
            student_minimum_margin=student_minimum_margin,
        ),  # type: ignore[arg-type]
        episode=episode,
        config=EngineDAggerConfig(
            max_frames=12,
            observation_delay=0,
            teacher_probability=teacher_probability,
            intervention_margin=intervention_margin,
            intervention_regret=intervention_regret,
            minimum_safety_margin_gain=minimum_safety_margin_gain,
            student_only_prefix_frames=student_only_prefix_frames,
            intervene_on_disagreement=intervene_on_disagreement,
            supervision_mode=supervision_mode,
            record_teacher_evaluations=record_teacher_evaluations,
        ),
        vision_config=VisionConfig(
            global_width=12,
            global_height=14,
            local_width=10,
            local_height=10,
            history=1,
            observation_delay=0,
        ),
    )
    return client, episode, report, student


def test_dagger_reports_factorized_student_model_inputs() -> None:
    _client, _episode, report, _student = _run(
        student_collides=False,
        action_selection="factorized",
    )

    assert report["controller"]["student"]["action_selection"] == "factorized"
    assert (
        report["controller"]["student"]["action_selection_uses_safety_state"]
        is False
    )
    assert (
        "factorized_direction_speed_marginal_scores_from_model_logits"
        in report["config"]["student_control_inputs"]
    )


def test_dagger_records_teacher_labels_on_student_executed_states() -> None:
    client, episode, report, student_controller = _run(student_collides=False)

    assert report["success"] is True
    assert report["teacher_assisted_success"] is True
    assert report["teacher_success"] is True
    assert report["pure_policy"] is False
    assert report["pure_policy_success"] is False
    assert report["pure_policy_validation_eligible"] is False
    assert report["engine"]["runtime_source_verification"]["matched"] is True
    assert report["teacher_interventions"] == 0
    assert report["student_teacher_agreements"] == 0
    assert episode.decisions == report["decision_count"] == 2
    teacher = Action(move_x=1, slow=True).discrete
    student = Action(move_x=-1, slow=True).discrete
    assert episode.actions == [teacher, teacher]
    assert episode.previous_actions == [-1, student]
    assert all(action.discrete == student for action in client.actions)
    assert [
        (action.discrete, frames) for action, frames in student_controller.commits
    ] == [(student, 3), (student, 3)]
    assert report["continuous_fire"] is True
    assert report["shoot_frames"] == report["frames"]
    assert report["shoot_rate"] == 1.0
    assert all(action.shoot and not action.spell for action in client.actions)


def test_dagger_optionally_records_all_policy_action_evaluations() -> None:
    _client, episode, report, _student = _run(
        student_collides=True,
        record_teacher_evaluations=True,
    )

    teacher = Action(move_x=1, slow=True).discrete
    assert report["teacher_action_evaluations"]["recorded"] is True
    assert report["teacher_action_evaluations"]["action_count"] == 18
    assert episode.teacher_action_evaluation_mask == [True, True]
    assert episode.teacher_action_evaluations[0].shape == (18, 11)
    assert episode.teacher_action_regrets[0].shape == (18,)
    selected = episode.teacher_action_evaluations[0][:, -1]
    assert np.flatnonzero(selected).tolist() == [teacher]
    assert episode.teacher_action_regrets[0][teacher] == 0.0
    # Focus does not change stationary geometry, but both policy ids remain.
    np.testing.assert_array_equal(
        episode.teacher_action_evaluations[0][4, :-1],
        episode.teacher_action_evaluations[0][13, :-1],
    )


def test_dagger_continuous_fire_wrapper_preserves_only_movement_and_focus() -> None:
    action = _movement_with_continuous_fire(Action(
        move_x=-1,
        move_y=1,
        slow=False,
        shoot=False,
        spell=True,
    ))

    assert action == Action(
        move_x=-1,
        move_y=1,
        slow=False,
        shoot=True,
        spell=False,
    )


def test_dagger_low_margin_student_action_never_stops_continuous_fire() -> None:
    client, _episode, report, _student = _run(
        student_collides=False,
        student_minimum_margin=8.0,
        student_action=Action(
            move_x=-1,
            slow=True,
            shoot=False,
            spell=True,
        ),
        supervision_mode="corrective",
    )

    assert report["teacher_interventions"] == 0
    assert all(action.shoot and not action.spell for action in client.actions)
    assert all(
        decision["student_predicted_minimum_margin"] == 8.0
        and decision["executed_action"]["shoot"] is True
        and decision["supervised_action"]["shoot"] is True
        for decision in report["decisions"]
    )
    assert report["continuous_fire"] is True
    assert report["shoot_frames"] == report["frames"] == 6
    assert report["shoot_rate"] == 1.0
    assert report["config"]["shoot_minimum_margin_controls_fire"] is False


def test_dagger_intervenes_when_student_candidate_would_collide() -> None:
    client, episode, report, student_controller = _run(student_collides=True)

    assert report["success"] is True
    assert report["teacher_interventions"] == report["decision_count"]
    assert report["safety_teacher_interventions"] == report["decision_count"]
    assert all(
        decision["intervention_reason"] == "predicted_collision"
        for decision in report["decisions"]
    )
    assert all(action.move_x == 1 for action in client.actions)
    assert episode.decisions == 2
    assert episode.previous_actions == [-1, Action(move_x=1, slow=True).discrete]
    assert [
        (action.move_x, frames) for action, frames in student_controller.commits
    ] == [(1, 3), (1, 3)]


def test_dagger_student_only_prefix_collects_parent_states_before_rescue() -> None:
    client, _episode, report, _student = _run(
        student_collides=True,
        supervision_mode="corrective",
        student_only_prefix_frames=3,
    )

    assert [action.move_x for action in client.actions] == [-1, -1, -1, 1, 1, 1]
    assert report["teacher_interventions"] == 1
    assert report["student_only_prefix"] == {
        "configured_frames": 3,
        "decisions": 1,
        "suppressed_interventions": 1,
        "model_input": False,
        "purpose": (
            "collect parent-policy on-policy prefix states inside an episode "
            "that must still satisfy strict native success"
        ),
    }
    assert report["decisions"][0]["student_only_prefix_active"] is True
    assert (
        report["decisions"][0]["student_only_prefix_suppressed_intervention"]
        is True
    )
    assert report["decisions"][1]["student_only_prefix_active"] is False


@pytest.mark.parametrize("value", (-1, 1.5, True))
def test_dagger_rejects_invalid_student_only_prefix(value) -> None:
    with pytest.raises(ValueError, match="student_only_prefix_frames"):
        EngineDAggerConfig(student_only_prefix_frames=value)


@pytest.mark.parametrize(
    ("run_options", "candidate_reason", "minimum_margin_gain"),
    (
        (
            {"student_collides": True},
            "predicted_collision",
            31.0,
        ),
        (
            {
                "student_collides": False,
                "student_minimum_margin": 8.0,
                "intervention_margin": 12.0,
            },
            "minimum_margin",
            22.0,
        ),
        (
            {
                "student_collides": False,
                "student_minimum_margin": 20.0,
                "intervention_regret": 8.0,
            },
            "clearance_regret",
            10.0,
        ),
    ),
)
def test_dagger_minimum_margin_gain_gate_rejects_weak_safety_interventions(
    run_options,
    candidate_reason: str,
    minimum_margin_gain: float,
) -> None:
    client, _episode, report, _student = _run(
        **run_options,
        supervision_mode="corrective",
        minimum_safety_margin_gain=minimum_margin_gain + 0.5,
    )

    assert report["teacher_interventions"] == 0
    assert report["safety_teacher_interventions"] == 0
    assert all(action.move_x == -1 for action in client.actions)
    assert all(
        decision["candidate_intervention_reason"] == candidate_reason
        and decision["intervention_reason"] is None
        and decision["teacher_intervened"] is False
        and decision["teacher_predicted_minimum_margin_gain"]
        == minimum_margin_gain
        and decision["minimum_safety_margin_gain_gate_applied"] is True
        and decision["minimum_safety_margin_gain_gate_passed"] is False
        for decision in report["decisions"]
    )
    assert report["safety_intervention_margin_gain_gate"] == {
        "enabled": True,
        "minimum_gain": minimum_margin_gain + 0.5,
        "applies_to": [
            "clearance_regret",
            "minimum_margin",
            "predicted_collision",
        ],
        "candidate_decisions": 2,
        "accepted_decisions": 0,
        "rejected_decisions": 2,
        "unaffected_reasons": [
            "policy_disagreement",
            "scheduled_teacher",
        ],
    }
    assert report["config"]["minimum_safety_margin_gain"] == (
        minimum_margin_gain + 0.5
    )


def test_dagger_minimum_margin_gain_gate_accepts_threshold_equality() -> None:
    client, _episode, report, _student = _run(
        student_collides=True,
        minimum_safety_margin_gain=31.0,
    )

    assert report["teacher_interventions"] == report["decision_count"]
    assert report["safety_teacher_interventions"] == report["decision_count"]
    assert all(action.move_x == 1 for action in client.actions)
    assert all(
        decision["intervention_reason"] == "predicted_collision"
        and decision["minimum_safety_margin_gain_gate_passed"] is True
        for decision in report["decisions"]
    )
    gate = report["safety_intervention_margin_gain_gate"]
    assert gate["accepted_decisions"] == report["decision_count"]
    assert gate["rejected_decisions"] == 0


def test_dagger_margin_gain_gate_does_not_change_scheduled_interventions() -> None:
    client, _episode, report, _student = _run(
        student_collides=True,
        teacher_probability=1.0,
        minimum_safety_margin_gain=100.0,
    )

    assert report["scheduled_teacher_interventions"] == report["decision_count"]
    assert report["safety_teacher_interventions"] == 0
    assert all(action.move_x == 1 for action in client.actions)
    assert all(
        decision["intervention_reason"] == "scheduled_teacher"
        and decision["minimum_safety_margin_gain_gate_applied"] is False
        and decision["minimum_safety_margin_gain_gate_passed"] is None
        for decision in report["decisions"]
    )


def test_dagger_can_collect_only_current_policy_disagreements() -> None:
    client, episode, report, student_controller = _run(
        student_collides=False,
        supervision_mode="corrective",
        intervene_on_disagreement=True,
        minimum_safety_margin_gain=100.0,
    )

    teacher = Action(move_x=1, slow=True).discrete
    assert report["success"] is True
    assert report["teacher_interventions"] == report["decision_count"] == 2
    assert report["safety_teacher_interventions"] == 0
    assert report["policy_disagreement_interventions"] == report["decision_count"]
    assert all(
        decision["intervention_reason"] == "policy_disagreement"
        and decision["minimum_safety_margin_gain_gate_applied"] is False
        and decision["minimum_safety_margin_gain_gate_passed"] is None
        for decision in report["decisions"]
    )
    assert episode.actions == [teacher, teacher]
    assert all(action.discrete == teacher for action in client.actions)
    assert [
        (action.discrete, frames) for action, frames in student_controller.commits
    ] == [(teacher, 3), (teacher, 3)]


def test_corrective_dagger_supervises_the_actions_that_were_executed() -> None:
    _client, unassisted, report, _student = _run(
        student_collides=False,
        supervision_mode="corrective",
    )
    student_action = Action(move_x=-1, slow=True).discrete
    assert unassisted.actions == [student_action, student_action]
    assert report["demonstration_supervision"] == {
        "mode": "corrective",
        "target": (
            "executed_action: student action when unassisted, teacher action "
            "when intervened"
        ),
        "teacher_correction_targets": 0,
        "student_execution_targets": 2,
        "teacher_only_targets": 0,
    }
    assert all(
        decision["supervised_action"] == decision["executed_action"]
        for decision in report["decisions"]
    )

    _client, corrected, report, _student = _run(
        student_collides=True,
        supervision_mode="corrective",
    )
    teacher_action = Action(move_x=1, slow=True).discrete
    assert corrected.actions == [teacher_action, teacher_action]
    assert report["demonstration_supervision"]["teacher_correction_targets"] == 2
    assert report["demonstration_supervision"]["student_execution_targets"] == 0


def test_dagger_rejects_unknown_supervision_mode() -> None:
    with pytest.raises(ValueError, match="supervision_mode"):
        EngineDAggerConfig(supervision_mode="mixed")


@pytest.mark.parametrize("value", (-1.0, float("inf"), float("nan"), True))
def test_dagger_rejects_invalid_minimum_safety_margin_gain(value) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        EngineDAggerConfig(minimum_safety_margin_gain=value)


def test_dagger_rejects_stale_runtime_lua_before_reset() -> None:
    client = FakeClient()
    client.runtime_source_crc32 = {
        **client.runtime_source_crc32,
        "compat/testing/bridge.lua": "00000000",
    }
    episode = NativeEpisodeBuffer(NativeEpisodeIdentity(
        episode_kind="attack",
        scenario="okuu:Lunatic",
        attack=3,
        seed=42,
    ))

    with pytest.raises(EngineProtocolError, match="changed=.*bridge.lua"):
        run_engine_dagger_play(
            client,  # type: ignore[arg-type]
            scenario="okuu:Lunatic",
            attack=3,
            seed=42,
            player="reimu_player",
            student=FixedStudent(Action()),  # type: ignore[arg-type]
            teacher=FixedTeacher(student_collides=False),  # type: ignore[arg-type]
            episode=episode,
            config=EngineDAggerConfig(observation_delay=0),
            vision_config=VisionConfig(history=1, observation_delay=0),
        )


def test_dagger_completion_with_death_is_not_strict_success() -> None:
    _client, _episode, report, _student = _run(student_collides=True, death=1)

    assert report["termination_reason"] == "attack_complete"
    assert report["outcome_evidence"]["final_player"]["death"] == 1
    assert report["success"] is False
    assert report["passed"] is False


def test_dagger_maps_fast_stationary_policy_action_to_mpc_neutral_evaluation() -> None:
    client, episode, report, _student = _run(
        student_collides=False,
        student_action=Action(slow=False),
    )

    assert report["success"] is True
    assert episode.decisions == 2
    assert all(action.move_x == 0 and action.move_y == 0 for action in client.actions)
