from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from stg_lab.engine import EngineProtocolError
from stg_lab.engine_play import (
    EnginePlayConfig,
    VisualPolicyController,
    VisibleSafetySample,
    _stream_policy_control_inputs,
    load_route_controller,
    load_route_library_controller,
    run_engine_play,
    visible_safety_action,
)
from stg_lab.engine_runtime import local_runtime_source_fingerprints
from stg_lab.memory import EpisodicMemory
from stg_lab.policy import ProficiencyRuntime, resolve_proficiency
from stg_lab.protocol import Action
from stg_lab.vision import VisionConfig, VisionObservation


def observation(
    frame: int,
    *,
    terminated: bool = False,
    reason: str | None = None,
    nearby_threat: bool = False,
    death: int | None = None,
) -> dict[str, Any]:
    bullets = []
    if nearby_threat:
        bullets.append({
            "id": 7,
            "x": 0.0,
            "y": -176.0,
            "a": 6.0,
            "b": 6.0,
            "vx": 99.0,
            "vy": -99.0,
            "collidable": True,
            "class_name": "authority-only",
        })
    return {
        "episode_frame": frame,
        "terminated": terminated,
        "termination_reason": reason,
        "stage": {"timer": 9999, "card_index": 4},
        "resources": {"lifeleft": 99},
        "world": {
            "l": -192,
            "r": 192,
            "b": -224,
            "t": 224,
            "pl": -192,
            "pr": 192,
            "pb": -224,
            "pt": 224,
        },
        "player": {
            "x": 0.0,
            "y": -176.0,
            "a": 0.5,
            "b": 0.5,
            "hspeed": 4.0,
            "lspeed": 2.0,
            "death": int(reason == "player_hit") if death is None else death,
            "protect": 999,
        },
        "enemy_bullets": bullets,
        "enemies": [
            {"id": 20, "x": 0.0, "y": 120.0, "a": 16.0, "b": 16.0, "hp": 100.0},
            {"id": 21, "x": 40.0, "y": 180.0, "a": 16.0, "b": 16.0, "hp": 999999999.0},
        ],
        "nontjt_enemies": [],
        "indestructibles": [],
        "lasers": [],
    }


class FakeEngineClient:
    def __init__(
        self,
        *,
        terminate_at: int | None,
        reason: str | None,
        nearby_threat: bool = False,
        final_death: int | None = None,
    ) -> None:
        self.terminate_at = terminate_at
        self.reason = reason
        self.nearby_threat = nearby_threat
        self.final_death = final_death
        self.frame = 0
        self.actions: list[tuple[Action, int]] = []
        self.reset_call: dict[str, Any] | None = None
        self.render_calls: list[tuple[bool, int]] = []
        self.runtime_source_crc32 = local_runtime_source_fingerprints()[0]

    def ping(self):
        return {
            "protocol": 2,
            "session_id": "fake-live-session",
            "process_nonce": "fake-process",
            "runtime_identity": {
                "process_id": 42,
                "source_crc32": self.runtime_source_crc32,
            },
        }

    def catalog(self):
        return {"catalog": {
            "attacks": [
                {"scenario": "okuu:Lunatic", "attack": 3, "card_index": 4},
                {"scenario": "okuu:Lunatic", "attack": 4, "card_index": 5},
            ],
            "stages": [
                {"stage": "Stage 1@Normal", "name": "Stage 1"},
            ],
        }}

    def set_rendering(self, enabled: bool, *, every: int = 1):
        self.render_calls.append((enabled, every))
        return {"render": enabled, "every": every}

    def reset(self, scenario, attack, *, seed, player, options):
        self.frame = 0
        self.reset_call = {
            "scenario": scenario,
            "attack": attack,
            "seed": seed,
            "player": player,
            "options": options,
        }
        return {"observation": observation(0, nearby_threat=self.nearby_threat)}

    def reset_stage(self, stage, *, seed, player, options):
        self.frame = 0
        self.reset_call = {
            "stage": stage,
            "seed": seed,
            "player": player,
            "options": options,
        }
        return {"observation": observation(0, nearby_threat=self.nearby_threat)}

    def step(self, action: Action, *, repeat: int = 1):
        self.actions.append((action, repeat))
        self.frame += repeat
        terminated = self.terminate_at is not None and self.frame >= self.terminate_at
        return {"observation": observation(
            self.frame,
            terminated=terminated,
            reason=self.reason if terminated else None,
            nearby_threat=self.nearby_threat,
            death=self.final_death if terminated else None,
        )}


class AlternatingController:
    def reset(self) -> None:
        self.calls = 0

    def select(self, visible: VisionObservation) -> Action:
        assert isinstance(visible, VisionObservation)
        move_x = -1 if self.calls % 2 == 0 else 1
        self.calls += 1
        return Action(move_x=move_x, slow=True, shoot=False, spell=True)


class StubVisualPolicyController(VisualPolicyController):
    inference_mode = "stream"
    scenario_key = "attack:okuu:Lunatic#3"
    device = "cpu"
    proficiency = resolve_proficiency("expert")
    scenario_vocabulary = ("<unknown>", "attack:okuu:Lunatic#3")
    previous_action_size = 0

    def __init__(self) -> None:
        self.seed = 0
        self.decisions = 0

    def reset_for_seed(self, seed: int) -> None:
        self.seed = seed
        self.decisions = 0

    def select(self, visible: VisionObservation) -> Action:
        assert isinstance(visible, VisionObservation)
        self.decisions += 1
        return Action(move_x=1, slow=True)

    def commit_executed_action(self, action: Action, *, frames: int) -> None:
        assert isinstance(action, Action)
        assert frames > 0


class ContextOnlyPolicy:
    config = SimpleNamespace(
        memory_size=2,
        inference_mode="stream",
    )

    def to(self, _device: str):
        return self

    def eval(self):
        return self


def test_live_policy_rejects_identity_memory_without_vocabulary() -> None:
    with pytest.raises(ValueError, match="checkpoint-declared scenario vocabulary"):
        VisualPolicyController(
            ContextOnlyPolicy(),
            "stage5_boss3:lunatic",
            device="cpu",
        )


class ProficiencyController:
    def __init__(self, proficiency: str) -> None:
        self.proficiency = resolve_proficiency(proficiency)
        self.runtime = ProficiencyRuntime(self.proficiency)

    def reset_for_seed(self, seed: int) -> None:
        self.runtime.reset(seed)

    def select(self, visible: VisionObservation) -> Action:
        assert isinstance(visible, VisionObservation)
        return Action(move_y=1, slow=True)


class ExecutedCommitController:
    def reset(self) -> None:
        self.commits: list[tuple[Action, int]] = []

    def select(self, visible: VisionObservation) -> Action:
        assert isinstance(visible, VisionObservation)
        return Action(move_y=1, slow=True)

    def commit_executed_action(self, action: Action, *, frames: int) -> None:
        self.commits.append((action, frames))


def play_config(*, max_frames: int, observation_delay: int = 0) -> EnginePlayConfig:
    vision = VisionConfig(
        global_width=16,
        global_height=16,
        local_width=16,
        local_height=16,
        local_extent_x=24.0,
        local_extent_y=24.0,
        history=1,
        observation_delay=observation_delay,
    )
    return EnginePlayConfig(
        max_frames=max_frames,
        vision=vision,
        shoot_gate_radius=12.0,
        shoot_risk_threshold=0.25,
    )


def test_engine_play_rejects_stale_runtime_lua_before_reset() -> None:
    client = FakeEngineClient(terminate_at=3, reason="attack_complete")
    client.runtime_source_crc32 = {
        **client.runtime_source_crc32,
        "compat/testing/bridge.lua": "00000000",
    }

    with pytest.raises(EngineProtocolError, match="changed=.*bridge.lua"):
        run_engine_play(
            client,  # type: ignore[arg-type]
            scenario="okuu:Lunatic",
            attack=3,
            seed=42,
            player="reimu_player",
            controller=AlternatingController(),
            config=play_config(max_frames=3),
        )

    assert client.reset_call is None


def test_engine_play_requires_an_unshielded_model_for_pure_policy_success() -> None:
    pure = run_engine_play(
        FakeEngineClient(terminate_at=3, reason="attack_complete"),  # type: ignore[arg-type]
        scenario="okuu:Lunatic",
        attack=3,
        seed=42,
        player="reimu_player",
        controller=StubVisualPolicyController(),
        controller_metadata={"kind": "streaming_visual_policy"},
        config=play_config(max_frames=3),
    )

    shielded = run_engine_play(
        FakeEngineClient(terminate_at=3, reason="attack_complete"),  # type: ignore[arg-type]
        scenario="okuu:Lunatic",
        attack=3,
        seed=42,
        player="reimu_player",
        controller=StubVisualPolicyController(),
        controller_metadata={"kind": "streaming_visual_policy"},
        config=EnginePlayConfig(
            max_frames=3,
            vision=play_config(max_frames=3).vision,
            shoot_gate_radius=12.0,
            visible_safety_shield=True,
        ),
    )

    assert pure["success"] is True
    assert pure["pure_policy"] is True
    assert pure["unassisted_learned_policy"] is True
    assert pure["raw_model_action_execution"] is True
    assert pure["pure_policy_success"] is True
    assert pure["pure_policy_validation_eligible"] is True
    assert shielded["success"] is True
    assert shielded["visible_safety_interventions"] == 0
    assert shielded["pure_policy"] is False
    assert shielded["unassisted_learned_policy"] is False
    assert shielded["raw_model_action_execution"] is False
    assert shielded["pure_policy_success"] is False
    assert shielded["pure_policy_validation_eligible"] is False


def test_visible_safety_shield_uses_only_local_semantic_geometry() -> None:
    config = VisionConfig(
        global_width=9,
        global_height=9,
        local_width=9,
        local_height=9,
        local_extent_x=20.0,
        local_extent_y=20.0,
        history=1,
        observation_delay=0,
    )
    local = np.zeros((1, 6, 9, 9), dtype=np.float32)
    local[0, 0, 6, 4] = 1.0  # stationary occupied cell ten units above
    visible = VisionObservation(
        global_frames=np.zeros((1, 6, 9, 9), dtype=np.float32),
        local_frames=local,
        source_frame=12,
    )
    preferred = Action(move_y=1, slow=True)

    result = visible_safety_action(
        preferred,
        visible,
        config,
        horizon=12,
        minimum_margin=4.0,
    )

    assert result.intervened is True
    assert result.action.move_y != 1
    assert result.preferred_margin is not None and result.preferred_margin < 4.0
    assert result.selected_margin is not None and result.selected_margin >= 4.0
    assert result.threat_pixels == 1


def test_engine_play_visible_safety_uses_seeded_proficiency_limits(
    monkeypatch,
) -> None:
    from stg_lab import engine_play

    horizons = []

    def record_safety(preferred, _visible, _vision_config, *, horizon, minimum_margin):
        horizons.append((horizon, minimum_margin))
        return engine_play.VisibleSafetySample(
            preferred, False, 10.0, 10.0, 1,
        )

    monkeypatch.setattr(engine_play, "visible_safety_action", record_safety)

    def run(proficiency: str, seed: int, *, horizon_cap: int | None = None):
        return run_engine_play(
            FakeEngineClient(terminate_at=3, reason="attack_complete"),  # type: ignore[arg-type]
            scenario="okuu:Lunatic",
            attack=3,
            seed=seed,
            player="reimu_player",
            controller=ProficiencyController(proficiency),
            config=EnginePlayConfig(
                max_frames=3,
                vision=play_config(max_frames=3).vision,
                shoot_gate_radius=12.0,
                visible_safety_shield=True,
                visible_safety_horizon=horizon_cap,
            ),
        )

    expert = run("expert", 0)
    intermediate = run("intermediate", 0)
    novice_skipped = run("novice", 0)
    novice_applied = run("novice", 3)
    capped_expert = run("expert", 0, horizon_cap=5)

    assert horizons == [(12, 6.0), (6, 6.0), (3, 6.0), (5, 6.0)]
    assert expert["config"]["visible_safety_probability"] == 1.0
    assert expert["config"]["visible_safety_horizon"] == 12
    assert intermediate["config"]["visible_safety_probability"] == 0.65
    assert intermediate["visible_safety_checks"] == 1
    assert novice_skipped["config"]["visible_safety_probability"] == 0.25
    assert novice_skipped["visible_safety_checks"] == 0
    assert novice_skipped["visible_safety_probability_skips"] == 1
    assert novice_applied["visible_safety_checks"] == 1
    assert capped_expert["config"]["visible_safety_horizon"] == 5
    assert capped_expert["config"]["visible_safety_horizon_cap"] == 5
    assert all(
        report["config"]["authority_state_shield"] is False
        for report in (
            expert, intermediate, novice_skipped, novice_applied, capped_expert,
        )
    )


def test_engine_play_commits_only_the_visible_safety_executed_action(
    monkeypatch,
) -> None:
    from stg_lab import engine_play

    def override(preferred, _visible, _vision_config, **_options):
        assert preferred.move_y == 1
        return VisibleSafetySample(
            Action(move_x=-1, slow=True),
            True,
            -1.0,
            10.0,
            1,
        )

    monkeypatch.setattr(engine_play, "visible_safety_action", override)
    controller = ExecutedCommitController()
    client = FakeEngineClient(terminate_at=3, reason="attack_complete")

    report = run_engine_play(
        client,  # type: ignore[arg-type]
        scenario="okuu:Lunatic",
        attack=3,
        seed=42,
        player="reimu_player",
        controller=controller,
        config=EnginePlayConfig(
            max_frames=3,
            vision=play_config(max_frames=3).vision,
            shoot_gate_radius=12.0,
            visible_safety_shield=True,
        ),
    )

    assert report["visible_safety_interventions"] == 1
    assert report["action_steps"][0]["preferred_action"]["move_y"] == 1
    assert report["action_steps"][0]["action"]["move_x"] == -1
    assert len(controller.commits) == 1
    committed, frames = controller.commits[0]
    assert (committed.move_x, committed.move_y, frames) == (-1, 0, 3)


def test_visual_policy_controller_defers_motor_commit_until_execution(
    monkeypatch,
) -> None:
    from stg_lab import rollout

    calls = []

    def choose(*_args, **options):
        calls.append(options)
        return Action(move_y=1, slow=True), "next-hidden"

    class RecordingRuntime:
        def __init__(self) -> None:
            self.commits = []

        def commit(self, action: Action, *, decision_interval: int) -> None:
            self.commits.append((action, decision_interval))

    monkeypatch.setattr(rollout, "_policy_behavior_action", choose)
    controller = object.__new__(VisualPolicyController)
    controller.model = object()
    controller.device = "cpu"
    controller.memory = np.zeros(0, dtype=np.float32)
    controller.hidden = None
    controller.inference_mode = "stream"
    controller.runtime = RecordingRuntime()
    controller.decisions = 0
    visible = VisionObservation(
        global_frames=np.zeros((1, 6, 8, 8), dtype=np.float32),
        local_frames=np.zeros((1, 6, 8, 8), dtype=np.float32),
        source_frame=0,
    )

    preferred = controller.select(visible)

    assert preferred.move_y == 1
    assert calls[0]["commit_runtime"] is False
    assert controller.runtime.commits == []
    controller.commit_executed_action(Action(move_x=-1, slow=True), frames=3)
    assert controller.runtime.commits == [(Action(move_x=-1, slow=True), 3)]


def test_visual_policy_controller_records_previous_executed_motor_action() -> None:
    controller = object.__new__(VisualPolicyController)
    controller.runtime = type("Runtime", (), {
        "commit": lambda _self, _action, *, decision_interval: None,
    })()
    controller.previous_action_size = 18
    controller.previous_action_offset = 2
    controller.memory = np.zeros(20, dtype=np.float32)
    controller.memory[1] = 1.0

    action = Action(move_x=-1, move_y=1, slow=False)
    controller.commit_executed_action(action, frames=3)

    np.testing.assert_array_equal(controller.memory[:2], (0.0, 1.0))
    assert controller.memory[2:].sum() == 1.0
    assert controller.memory[2 + action.discrete] == 1.0


def test_stream_policy_control_inputs_report_learned_context() -> None:
    controller = object.__new__(VisualPolicyController)
    controller.scenario_vocabulary = ("<unknown>", "attack:okuu:Lunatic#3")
    controller.previous_action_size = 18

    assert _stream_policy_control_inputs(controller)[-2:] == [
        "registered_episode_identity_one_hot",
        "previous_executed_motor_action_one_hot",
    ]


def test_engine_play_holds_each_visible_decision_for_three_frames() -> None:
    client = FakeEngineClient(terminate_at=7, reason="attack_complete")
    observed = []
    report = run_engine_play(
        client,  # type: ignore[arg-type]
        scenario="okuu:Lunatic",
        attack=3,
        seed=42,
        player="reimu_player",
        controller=AlternatingController(),
        config=play_config(max_frames=30),
        decision_observer=lambda visible, action, risk: observed.append(
            (visible, action, risk)
        ),
    )

    assert client.reset_call == {
        "scenario": "okuu:Lunatic",
        "attack": 3,
        "seed": 42,
        "player": "reimu_player",
        "options": {},
    }
    assert all(repeat == 1 for _action, repeat in client.actions)
    assert [action.move_x for action, _repeat in client.actions] == [-1, -1, -1, 1, 1, 1, -1]
    assert all(action.shoot and not action.spell for action, _repeat in client.actions)
    assert [step["advanced_frames"] for step in report["action_steps"]] == [3, 3, 1]
    assert [step["source_frame"] for step in report["action_steps"]] == [0, 3, 6]
    assert report["success"] is True
    assert report["termination_reason"] == "attack_complete"
    assert report["frames"] == 7
    assert report["shoot_rate"] == 1.0
    assert report["config"]["reset_options"] == {}
    assert report["config"]["authority_state_shield"] is False
    assert client.render_calls == [(False, 1)]
    assert report["engine"]["runtime_identity"]["process_id"] == 42
    assert report["engine"]["catalog_entry"]["card_index"] == 4
    assert report["outcome_evidence"]["reporting_only_not_controller_input"] is True
    assert report["outcome_evidence"]["boss_hp_initial"] == 100.0
    assert report["unsafe_shot_frames"] is None
    assert report["unsafe_shot_frames_deprecated"] is True
    assert len(observed) == 3
    assert all(isinstance(value[0], VisionObservation) for value in observed)
    assert [value[1].move_x for value in observed] == [-1, 1, -1]
    assert all(value[2] == 0.0 for value in observed)


def test_engine_play_rejects_completion_with_a_recorded_death() -> None:
    client = FakeEngineClient(
        terminate_at=3,
        reason="attack_complete",
        final_death=1,
    )
    report = run_engine_play(
        client,  # type: ignore[arg-type]
        scenario="okuu:Lunatic",
        attack=3,
        seed=42,
        player="reimu_player",
        controller=AlternatingController(),
        config=play_config(max_frames=30),
    )

    assert report["termination_reason"] == "attack_complete"
    assert report["outcome_evidence"]["final_player"]["death"] == 1
    assert report["success"] is False
    assert report["passed"] is False
    assert "death=0" in report["success_criterion"]


def test_engine_play_accepts_only_zero_death_stage_completion() -> None:
    client = FakeEngineClient(terminate_at=3, reason="stage_complete", final_death=0)
    report = run_engine_play(
        client,  # type: ignore[arg-type]
        scenario="Stage 1@Normal",
        attack=None,
        stage="Stage 1@Normal",
        seed=43,
        player="reimu_player",
        controller=AlternatingController(),
        config=play_config(max_frames=30),
    )

    assert client.reset_call == {
        "stage": "Stage 1@Normal",
        "seed": 43,
        "player": "reimu_player",
        "options": {},
    }
    assert report["success"] is True
    assert report["episode_completed"] is True
    assert report["episode_kind"] == "stage"
    assert report["stage"] == "Stage 1@Normal"
    assert report["attack"] is None
    assert report["termination_reason"] == "stage_complete"
    assert report["engine"]["catalog_entry"]["name"] == "Stage 1"


def test_engine_play_rejects_boolean_false_as_zero_death() -> None:
    report = run_engine_play(
        FakeEngineClient(
            terminate_at=3,
            reason="attack_complete",
            final_death=False,
        ),  # type: ignore[arg-type]
        scenario="okuu:Lunatic",
        attack=3,
        seed=42,
        player="reimu_player",
        controller=AlternatingController(),
        config=play_config(max_frames=30),
    )

    assert report["termination_reason"] == "attack_complete"
    assert report["success"] is False
    assert report["passed"] is False
    assert report["episode_completed"] is False


def test_engine_play_visible_threat_is_diagnostic_and_never_stops_shooting() -> None:
    client = FakeEngineClient(
        terminate_at=None,
        reason=None,
        nearby_threat=True,
    )
    report = run_engine_play(
        client,  # type: ignore[arg-type]
        scenario="okuu:Lunatic",
        attack=4,
        seed=7,
        player="reimu_player",
        controller=AlternatingController(),
        config=play_config(max_frames=3),
    )

    assert all(action.shoot for action, _repeat in client.actions)
    diagnostic = report["action_steps"][0]["local_threat_diagnostic"]
    assert diagnostic["low_risk"] is False
    assert diagnostic["risk"] > 0.25
    assert diagnostic["reporting_only"] is True
    assert diagnostic["controls_fire"] is False
    assert report["terminated"] is False
    assert report["termination_reason"] == "max_frames"
    assert report["engine_termination_reason"] is None
    assert report["success"] is False
    assert report["continuous_fire"] is True
    assert report["shoot_frames"] == report["frames"] == 3
    assert report["shoot_rate"] == 1.0
    assert report["shoot_command_frames"] == report["frames"]
    assert report["shoot_command_rate"] == 1.0
    assert report["config"]["shoot_gate_controls_fire"] is False


def test_engine_play_player_hit_never_counts_as_success() -> None:
    client = FakeEngineClient(terminate_at=1, reason="player_hit")
    report = run_engine_play(
        client,  # type: ignore[arg-type]
        scenario="okuu:Lunatic",
        attack=3,
        seed=9,
        player="reimu_player",
        controller=AlternatingController(),
        config=play_config(max_frames=30),
    )
    assert report["terminated"] is True
    assert report["termination_reason"] == "player_hit"
    assert report["success"] is False


def test_existing_route_and_library_formats_load_from_sqlite(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite"
    route_cue = {
        "kind": "semantic_roi_mass",
        "channel": 0,
        "minimum_mass": 0.0,
        "roi": [-192.0, 192.0, -224.0, 224.0],
    }
    signature_cue = {
        "kind": "semantic_signature",
        "minimum_mass": 0.0,
        "channels": [0],
        "pooled_height": 1,
        "pooled_width": 1,
        "vector": [1.0],
    }
    with EpisodicMemory(database) as store:
        route_memory = store.remember(
            "stage5_boss3:lunatic",
            route_cue,
            death_point=None,
            trigger_lead=0,
            route=[Action(move_x=-1).to_dict()],
        )
        library_memory = store.remember(
            "stage5_boss4:lunatic",
            signature_cue,
            death_point=None,
            trigger_lead=0,
            route=[Action(move_x=1).to_dict()],
        )

    route_path = tmp_path / "route.json"
    route_path.write_text(json.dumps({
        "schema_version": 1,
        "route_id": "route-3",
        "scenario": "stage5_boss3:lunatic",
        "cue": route_cue,
        "trigger_lead": 0,
        "decision_interval": 3,
        "actions": [Action(move_x=-1).to_dict()],
        "source": {},
    }))
    library_path = tmp_path / "library.json"
    library_path.write_text(json.dumps({
        "schema_version": 1,
        "library_id": "library-4",
        "scenario": "stage5_boss4:lunatic",
        "memory_ids": [library_memory.id],
        "source": {},
    }))

    route, route_metadata = load_route_controller(route_path, database, route_memory.id)
    library, library_metadata = load_route_library_controller(library_path, database)
    assert route.config.shield is False
    assert route_metadata["memory_ids"] == [route_memory.id]
    assert library.config.shield is False
    assert library_metadata["memory_ids"] == [library_memory.id]
