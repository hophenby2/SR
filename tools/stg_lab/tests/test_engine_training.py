from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from stg_lab import engine_training
from stg_lab.engine_play import EnginePlayConfig
from stg_lab.engine_training import (
    CandidateController,
    CandidateStrategy,
    EngineTrainingConfig,
    generate_candidate_strategies,
    load_candidate_strategy,
    run_engine_training,
    write_strategy_artifact,
)
from stg_lab.protocol import Action
from stg_lab.vision import VisionConfig, VisionObservation


class NeutralController:
    def reset(self) -> None:
        self.calls = 0

    def select(self, _visible: VisionObservation) -> Action:
        self.calls += 1
        return Action(move_x=1, move_y=-1, slow=False)


def visible() -> VisionObservation:
    frames = np.zeros((1, 6, 4, 4), dtype=np.float32)
    return VisionObservation(frames, frames.copy(), source_frame=0)


def config() -> EnginePlayConfig:
    return EnginePlayConfig(
        max_frames=30,
        vision=VisionConfig(
            global_width=8,
            global_height=8,
            local_width=8,
            local_height=8,
            local_extent_x=24.0,
            local_extent_y=24.0,
            history=1,
            observation_delay=0,
        ),
        shoot_gate_radius=12.0,
        shoot_risk_threshold=0.25,
        shoot_motion_weight=0.5,
    )


def test_candidate_controller_applies_offset_axes_and_focus() -> None:
    base = NeutralController()
    delayed = CandidateController(base, CandidateStrategy(
        horizontal_sign=-1,
        vertical_sign=-1,
        slow_mode="focus",
        decision_offset=2,
    ))
    delayed.reset()
    first = delayed.select(visible())
    second = delayed.select(visible())
    third = delayed.select(visible())
    assert first.move_x == first.move_y == 0
    assert second.move_x == second.move_y == 0
    assert third == Action(move_x=-1, move_y=1, slow=True, shoot=True)
    assert base.calls == 1

    advanced_base = NeutralController()
    advanced = CandidateController(advanced_base, CandidateStrategy(decision_offset=-3))
    advanced.select(visible())
    assert advanced_base.calls == 4


def test_candidate_generation_is_seed_reproducible_and_baseline_first() -> None:
    first = generate_candidate_strategies(config(), count=12, search_seed=9)
    second = generate_candidate_strategies(config(), count=12, search_seed=9)
    other = generate_candidate_strategies(config(), count=12, search_seed=10)
    assert first == second
    assert first[0] == CandidateStrategy(
        shoot_gate_radius=12.0,
        shoot_risk_threshold=0.25,
        shoot_motion_weight=0.5,
    )
    assert first[1:] != other[1:]
    assert len({candidate.candidate_id for candidate in first}) == 12
    assert all(
        candidate.shoot_gate_radius == 12.0
        and candidate.shoot_risk_threshold == 0.25
        and candidate.shoot_motion_weight == 0.5
        for candidate in first
    )


def test_training_seed_sets_must_be_disjoint() -> None:
    with pytest.raises(ValueError, match="overlap"):
        EngineTrainingConfig(train_seeds=(1, 2), heldout_seeds=(2, 3))


def test_live_training_selects_only_by_strict_completion_and_writes_traces(
    monkeypatch,
    tmp_path: Path,
) -> None:
    client = object()
    calls = []

    def fake_play(selected_client, **kwargs):
        assert selected_client is client
        candidate = kwargs["controller_metadata"]["training_candidate"]
        phase = kwargs["controller_metadata"]["training_phase"]
        seed = kwargs["seed"]
        calls.append((id(selected_client), phase, seed, candidate["candidate_id"]))
        completed = candidate["horizontal_sign"] == -1
        # The baseline deliberately lies in runner_success. The training loop
        # must recompute its metric from termination evidence.
        return {
            "seed": seed,
            "success": True,
            "terminated": completed,
            "termination_reason": "attack_complete" if completed else "max_frames",
            "engine_termination_reason": "attack_complete" if completed else None,
            "frames": 20 if completed else 30,
            "decision_count": 7 if completed else 10,
            "shoot_rate": 0.5,
            "action_steps": [{"action": {"move_x": candidate["horizontal_sign"]}}],
            "engine": {
                "session_id": "training-session",
                "process_nonce": "one-process",
                "runtime_identity": {"process_id": 77},
            },
            "outcome_evidence": {
                "reporting_only_not_controller_input": True,
                "final_player": {"death": 0},
            },
        }

    monkeypatch.setattr(engine_training, "run_engine_play", fake_play)
    baseline = CandidateStrategy(
        shoot_gate_radius=12.0,
        shoot_risk_threshold=0.25,
        shoot_motion_weight=0.5,
    )
    mirrored = CandidateStrategy(
        horizontal_sign=-1,
        shoot_gate_radius=12.0,
        shoot_risk_threshold=0.25,
        shoot_motion_weight=0.5,
    )
    report = run_engine_training(
        client,  # type: ignore[arg-type]
        scenario="okuu:Lunatic",
        attack=3,
        player="reimu_player",
        controller_factory=NeutralController,
        controller_metadata={"kind": "test_controller"},
        play_config=config(),
        training_config=EngineTrainingConfig(
            train_seeds=(11, 12),
            heldout_seeds=(91,),
            candidate_count=2,
            search_seed=4,
        ),
        candidates=(baseline, mirrored),
        trace_directory=tmp_path / "traces",
    )

    assert len(calls) == 5
    assert {call[0] for call in calls} == {id(client)}
    assert [call[1] for call in calls] == ["train", "train", "train", "train", "heldout"]
    assert report["selected_candidate_index"] == 1
    assert report["selected_strategy"]["horizontal_sign"] == -1
    assert report["candidate_results"][0]["training"]["strict_successes"] == 0
    baseline_episode = report["candidate_results"][0]["episodes"][0]
    assert baseline_episode["runner_success"] is True
    assert baseline_episode["success"] is False
    assert baseline_episode["runner_success_matches_strict_metric"] is False
    assert report["heldout"]["strict_successes"] == 1
    assert report["passed"] is True
    assert report["train_heldout_disjoint"] is True
    assert report["same_engine_connection"] is True
    assert report["engine_identity"]["process_id"] == 77
    trace_paths = list((tmp_path / "traces").rglob("*.json"))
    assert len(trace_paths) == 5
    assert all(json.loads(path.read_text())["seed"] in {11, 12, 91} for path in trace_paths)

    artifact = tmp_path / "strategy.json"
    digest = write_strategy_artifact(artifact, report)
    assert len(digest) == 64
    assert load_candidate_strategy(artifact) == mirrored


@pytest.mark.parametrize("final_player", ({"death": 1}, {}))
def test_training_rejects_completion_without_explicit_zero_death(
    final_player,
) -> None:
    candidate = CandidateStrategy()
    evidence = engine_training._strict_episode_evidence(
        {
            "success": True,
            "terminated": True,
            "engine_termination_reason": "attack_complete",
            "outcome_evidence": {"final_player": final_player},
        },
        phase="heldout",
        candidate_index=0,
        candidate=candidate,
        trace_path=None,
    )

    assert evidence["success"] is False
    assert evidence["runner_success_matches_strict_metric"] is False
    assert evidence["final_death"] == final_player.get("death")
