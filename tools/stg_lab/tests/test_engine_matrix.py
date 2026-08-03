from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from stg_lab.engine_matrix import (
    EngineEpisodeTarget,
    EngineMatrixConfig,
    _episode_evidence,
    available_engine_profiles,
    controller_config_for_profile,
    run_engine_matrix,
    run_engine_policy_matrix,
    select_catalog_targets,
)
from stg_lab.engine_play import EnginePlayConfig, VisualPolicyController
from stg_lab.native_dataset import NativeDemonstrationBuilder
from stg_lab.protocol import Action
from stg_lab.vision import VisionConfig, VisionObservation


def catalog() -> dict:
    return {"catalog": {
        "attacks": [
            {"scenario": "boss:Normal", "attack": 1, "label": "N #1"},
            {"scenario": "boss:Normal", "attack": 2, "label": "N #2"},
            {"scenario": "boss:Lunatic", "attack": 1, "label": "L #1"},
        ],
        "stages": [
            {"stage": "Stage 1@Normal", "stage_index": 1},
            {"stage": "Stage 1@Lunatic", "stage_index": 1},
        ],
    }}


def policy_checkpoint_metadata(tmp_path: Path) -> dict:
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"test streaming policy checkpoint")
    return {
        "kind": "streaming_visual_policy",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_metadata": {
            "policy_config": {"inference_mode": "stream"},
        },
    }


def test_catalog_selection_preserves_order_and_rejects_missing_cases() -> None:
    selected = select_catalog_targets(
        catalog(),
        scenarios=("boss:Normal",),
        attacks=(2,),
        stages=("Stage 1@Lunatic",),
    )
    assert [value.target_id for value in selected] == [
        "attack:boss:Normal#2",
        "stage:Stage 1@Lunatic",
    ]

    all_targets = select_catalog_targets(
        catalog(), all_attacks=True, all_stages=True,
    )
    assert len(all_targets) == 5
    with pytest.raises(ValueError, match="lacks requested attacks"):
        select_catalog_targets(
            catalog(), scenarios=("boss:Lunatic",), attacks=(2,),
        )
    with pytest.raises(ValueError, match="absent"):
        select_catalog_targets(catalog(), stages=("Stage 9@Lunatic",))


def test_controller_profiles_bind_explicit_clearance_targets() -> None:
    config = EngineMatrixConfig(horizon_frames=36, observation_delay=0)
    current = controller_config_for_profile("current", config)
    general = controller_config_for_profile("general", config)
    legacy = controller_config_for_profile("legacy-clearance-12-1", config)
    novice = controller_config_for_profile("bullet-group-novice", config)
    intermediate = controller_config_for_profile("bullet-group-intermediate", config)
    expert = controller_config_for_profile("bullet-group-expert", config)
    assert available_engine_profiles() == (
        "current",
        "general",
        "legacy-clearance-12-1",
        "bullet-group-novice",
        "bullet-group-intermediate",
        "bullet-group-expert",
    )
    assert (current.safe_margin_target, current.region_safe_margin_target) == (20.0, 8.0)
    assert (
        current.minimum_direction_hold_frames,
        current.clearance_reward_cap,
        current.switch_margin_gain,
    ) == (12, 48.0, 8.0)
    assert (general.safe_margin_target, general.region_safe_margin_target) == (20.0, 8.0)
    assert (
        general.minimum_direction_hold_frames,
        general.clearance_reward_cap,
        general.switch_margin_gain,
    ) == (9, 36.0, 6.0)
    assert (legacy.safe_margin_target, legacy.region_safe_margin_target) == (12.0, 1.0)
    assert expert == current
    assert (
        novice.gap_minimum_group_size,
        intermediate.gap_minimum_group_size,
        expert.gap_minimum_group_size,
    ) == (5, 4, 3)
    assert (
        novice.gap_direction_tolerance_degrees,
        intermediate.gap_direction_tolerance_degrees,
        expert.gap_direction_tolerance_degrees,
    ) == (5.0, 8.0, 12.0)
    assert (
        novice.gap_safety_margin,
        intermediate.gap_safety_margin,
        expert.gap_safety_margin,
    ) == (18.0, 14.0, 10.0)
    assert (
        novice.gap_minimum_lifetime_frames,
        intermediate.gap_minimum_lifetime_frames,
        expert.gap_minimum_lifetime_frames,
    ) == (24, 18, 12)
    assert (
        novice.gap_entry_candidate_limit,
        intermediate.gap_entry_candidate_limit,
        expert.gap_entry_candidate_limit,
    ) == (2, 4, 8)
    assert (
        novice.gap_detour_beam_width,
        intermediate.gap_detour_beam_width,
        expert.gap_detour_beam_width,
    ) == (12, 24, 48)


def test_strict_success_requires_explicit_zero_death_evidence() -> None:
    evidence = _episode_evidence(
        {
            "success": True,
            "terminated": True,
            "engine_termination_reason": "attack_complete",
            "frames": 3,
            "decisions": [],
            "outcome_evidence": {"final_player": {}},
        },
        target=EngineEpisodeTarget("attack", "boss:Lunatic", 1),
        profile="current",
        trace_path=None,
    )

    assert evidence["strict_success"] is False
    assert evidence["died"] is False


def test_matrix_recomputes_strict_success_and_motion_metrics(
    tmp_path: Path,
) -> None:
    calls = []

    def fake_episode(_client, **kwargs):
        profile = (
            "current"
            if kwargs["controller"].config.safe_margin_target == 20.0
            else "legacy-clearance-12-1"
        )
        stage = kwargs["stage"]
        failed = stage is None and profile.startswith("legacy")
        reason = "player_hit" if failed else (
            "stage_complete" if stage is not None else "attack_complete"
        )
        calls.append((kwargs["scenario"], kwargs["attack"], stage, profile))
        return {
            "seed": kwargs["seed"],
            "scenario": kwargs["scenario"],
            "attack": kwargs["attack"],
            "episode_kind": "stage" if stage is not None else "attack",
            "stage": stage,
            # Deliberately lies for the failed attack; the matrix must ignore it.
            "success": True,
            "terminated": True,
            "termination_reason": reason,
            "engine_termination_reason": reason,
            "frames": 12,
            "engine_advanced_frames": 12,
            "decisions": [
                {"advanced_frames": 3, "action": {"move_x": 1, "move_y": 0, "slow": True}},
                {"advanced_frames": 3, "action": {"move_x": -1, "move_y": 0, "slow": True}},
                {"advanced_frames": 3, "action": {"move_x": 1, "move_y": 0, "slow": True}},
                {"advanced_frames": 3, "action": {"move_x": 1, "move_y": 1, "slow": True}},
            ],
            "engine": {
                "session_id": "matrix-session",
                "process_nonce": "matrix-process",
                "runtime_identity": {"process_id": 77},
            },
            "outcome_evidence": {
                "boss_hp_initial": 100.0,
                "boss_hp_last_observed": 0.0 if not failed else 70.0,
                "boss_hp_minimum_observed": 0.0 if not failed else 70.0,
                "player_path_distance": 30.0,
                "final_player": {"death": 1 if failed else 0},
            },
        }

    report = run_engine_matrix(
        object(),  # type: ignore[arg-type]
        targets=(
            EngineEpisodeTarget("attack", "boss:Lunatic", 3),
            EngineEpisodeTarget("stage", "Stage 5@Lunatic"),
        ),
        seeds=(101,),
        profiles=("current", "legacy-clearance-12-1"),
        config=EngineMatrixConfig(
            max_frames=30,
            horizon_frames=36,
            observation_delay=0,
        ),
        trace_directory=tmp_path / "traces",
        episode_runner=fake_episode,
    )

    assert len(calls) == 4
    assert report["overall"]["attempts"] == 4
    assert report["overall"]["strict_successes"] == 3
    assert report["overall"]["deaths"] == 1
    assert report["passed"] is False
    failed = next(value for value in report["episodes"] if not value["strict_success"])
    assert failed["runner_success"] is True
    assert failed["runner_success_matches_strict_metric"] is False
    assert failed["termination_reason"] == "player_hit"
    assert failed["smoothness"]["direction_changes"] == 3
    assert failed["smoothness"]["exact_reversals"] == 2
    assert failed["smoothness"]["aba_changes"] == 1
    assert failed["smoothness"]["direction_hold_frames"]["minimum"] == 3
    stage_episodes = [
        value for value in report["episodes"] if value["episode_kind"] == "stage"
    ]
    assert all(value["expected_completion_reason"] == "stage_complete" for value in stage_episodes)
    assert all(value["strict_success"] for value in stage_episodes)
    trace_paths = list((tmp_path / "traces").rglob("*.json"))
    assert len(trace_paths) == 4
    assert all(json.loads(path.read_text())["frames"] == 12 for path in trace_paths)


def test_matrix_rejects_engine_identity_changes() -> None:
    calls = 0

    def fake_episode(_client, **kwargs):
        nonlocal calls
        calls += 1
        return {
            "seed": kwargs["seed"],
            "scenario": kwargs["scenario"],
            "attack": kwargs["attack"],
            "episode_kind": "stage" if kwargs["stage"] is not None else "attack",
            "stage": kwargs["stage"],
            "success": True,
            "terminated": True,
            "engine_termination_reason": "attack_complete",
            "frames": 1,
            "decisions": [],
            "engine": {
                "session_id": "session",
                "process_nonce": "nonce",
                "runtime_identity": {"process_id": calls},
            },
            "outcome_evidence": {},
        }

    with pytest.raises(RuntimeError, match="one engine process"):
        run_engine_matrix(
            object(),  # type: ignore[arg-type]
            targets=(EngineEpisodeTarget("attack", "boss:Lunatic", 1),),
            seeds=(1, 2),
            config=EngineMatrixConfig(horizon_frames=36, observation_delay=0),
            episode_runner=fake_episode,
        )


def test_policy_matrix_accepts_portable_identity_without_os_pid(
    tmp_path: Path,
) -> None:
    def fake_episode(_client, **kwargs):
        return {
            "seed": kwargs["seed"],
            "scenario": kwargs["scenario"],
            "attack": kwargs["attack"],
            "episode_kind": "attack",
            "stage": None,
            "success": True,
            "pure_policy": True,
            "pure_policy_success": True,
            "pure_policy_validation_eligible": True,
            "visible_safety_interventions": 0,
            "terminated": True,
            "engine_termination_reason": "attack_complete",
            "frames": 3,
            "action_steps": [],
            "engine": {
                "session_id": "portable-session",
                "process_nonce": "portable-nonce",
                "runtime_identity": {},
            },
            "outcome_evidence": {"final_player": {"death": 0}},
        }

    report = run_engine_policy_matrix(
        object(),  # type: ignore[arg-type]
        targets=(EngineEpisodeTarget("attack", "boss:Lunatic", 1),),
        seeds=(1,),
        proficiencies=("expert",),
        controller_factory=lambda *_args: object.__new__(VisualPolicyController),
        controller_metadata=policy_checkpoint_metadata(tmp_path),
        config=EnginePlayConfig(
            max_frames=3,
            vision=VisionConfig(history=1, observation_delay=0),
        ),
        episode_runner=fake_episode,
    )

    assert report["passed"] is True
    assert report["pure_policy"] is True
    assert report["pure_policy_success"] is True
    assert report["engine_identity"]["process_id"] is None


def test_matrix_collects_only_strict_episode_demonstrations() -> None:
    calls = 0

    def fake_episode(_client, **kwargs):
        nonlocal calls
        calls += 1
        kwargs["decision_observer"](
            VisionObservation(
                global_frames=np.zeros((1, 6, 8, 8), dtype=np.float32),
                local_frames=np.zeros((1, 6, 8, 8), dtype=np.float32),
                source_frame=0,
            ),
            Action(move_x=1),
            0.25,
        )
        success = calls == 1
        reason = "attack_complete" if success else "player_hit"
        return {
            "seed": kwargs["seed"],
            "scenario": kwargs["scenario"],
            "attack": kwargs["attack"],
            "episode_kind": "attack",
            "stage": None,
            "success": success,
            "terminated": True,
            "termination_reason": reason,
            "engine_termination_reason": reason,
            "frames": 3,
            "decisions": [],
            "engine": {
                "session_id": "session",
                "process_nonce": "nonce",
                "runtime_identity": {"process_id": 7},
            },
            "outcome_evidence": {"final_player": {"death": 0 if success else 1}},
        }

    builder = NativeDemonstrationBuilder()
    report = run_engine_matrix(
        object(),  # type: ignore[arg-type]
        targets=(EngineEpisodeTarget("attack", "boss:Lunatic", 1),),
        seeds=(1, 2),
        config=EngineMatrixConfig(horizon_frames=36, observation_delay=0),
        episode_runner=fake_episode,
        demonstration_builder=builder,
    )

    assert report["overall"]["strict_successes"] == 1
    assert report["demonstration_collection"]["strict_successes_retained"] == 1
    demonstrations = builder.build()
    assert demonstrations.actions.shape == (1, 1)
    np.testing.assert_array_equal(demonstrations.episode_ids, (0,))


def test_policy_matrix_recomputes_results_and_reads_action_steps(
    tmp_path: Path,
) -> None:
    created = []

    def controller_factory(target, proficiency, seed):
        created.append((target.target_id, proficiency, seed))
        return object.__new__(VisualPolicyController)

    def fake_episode(_client, **kwargs):
        proficiency = created[-1][1]
        stage = kwargs["stage"]
        failed = proficiency == "novice"
        reason = "player_hit" if failed else (
            "stage_complete" if stage is not None else "attack_complete"
        )
        return {
            "seed": kwargs["seed"],
            "scenario": kwargs["scenario"],
            "attack": kwargs["attack"],
            "episode_kind": "stage" if stage is not None else "attack",
            "stage": stage,
            "success": not failed,
            "pure_policy": True,
            "pure_policy_success": not failed,
            "pure_policy_validation_eligible": True,
            "visible_safety_interventions": 0,
            "terminated": True,
            "engine_termination_reason": reason,
            "frames": 9,
            "action_steps": [
                {"advanced_frames": 3, "action": {"move_x": 1, "move_y": 0}},
                {"advanced_frames": 3, "action": {"move_x": -1, "move_y": 0}},
                {"advanced_frames": 3, "action": {"move_x": 1, "move_y": 0}},
            ],
            "controller": {"proficiency": {"name": proficiency}},
            "engine": {
                "session_id": "policy-matrix-session",
                "process_nonce": "policy-matrix-process",
                "runtime_identity": {"process_id": 88},
            },
            "outcome_evidence": {
                "boss_hp_initial": 100.0,
                "boss_hp_last_observed": 50.0 if failed else 0.0,
                "boss_hp_minimum_observed": 50.0 if failed else 0.0,
                "final_player": {"death": 1 if failed else 0},
            },
        }

    report = run_engine_policy_matrix(
        object(),  # type: ignore[arg-type]
        targets=(
            EngineEpisodeTarget("attack", "boss:Lunatic", 1),
            EngineEpisodeTarget("stage", "Stage 1@Normal"),
        ),
        seeds=(11,),
        proficiencies=("expert", "novice"),
        controller_factory=controller_factory,
        controller_metadata=policy_checkpoint_metadata(tmp_path),
        config=EnginePlayConfig(
            max_frames=30,
            vision=VisionConfig(history=1, observation_delay=0),
        ),
        trace_directory=tmp_path / "policy-traces",
        episode_runner=fake_episode,
    )

    assert len(created) == 4
    assert report["pure_policy"] is True
    assert report["pure_policy_success"] is False
    assert report["overall"]["attempts"] == 4
    assert report["overall"]["strict_successes"] == 2
    assert report["overall"]["deaths"] == 2
    assert report["passed"] is False
    expert = next(
        value for value in report["episodes"]
        if value["proficiency"] == "expert"
    )
    assert expert["smoothness"]["direction_changes"] == 2
    assert expert["smoothness"]["exact_reversals"] == 2
    assert expert["smoothness"]["aba_changes"] == 1
    assert expert["controller"]["proficiency"]["name"] == "expert"
    assert len(list((tmp_path / "policy-traces").rglob("*.json"))) == 4


def test_policy_matrix_rejects_unbound_checkpoint_metadata(tmp_path: Path) -> None:
    metadata = policy_checkpoint_metadata(tmp_path)
    metadata["checkpoint_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="SHA-256"):
        run_engine_policy_matrix(
            object(),  # type: ignore[arg-type]
            targets=(EngineEpisodeTarget("attack", "boss:Lunatic", 1),),
            seeds=(1,),
            proficiencies=("expert",),
            controller_factory=lambda *_args: object.__new__(VisualPolicyController),
            controller_metadata=metadata,
            config=EnginePlayConfig(
                max_frames=3,
                vision=VisionConfig(history=1, observation_delay=0),
            ),
            episode_runner=lambda *_args, **_kwargs: {},
        )
