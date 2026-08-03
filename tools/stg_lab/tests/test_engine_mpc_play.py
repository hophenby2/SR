from __future__ import annotations

from dataclasses import replace
import inspect
from typing import Any, Mapping

import pytest

from stg_lab.engine import EngineProtocolError
from stg_lab.engine_mpc import EngineMPC, MPCConfig, MPCDecision
from stg_lab.engine_mpc_play import (
    EngineMPCPlayConfig,
    _controller_observation,
    run_engine_mpc_play,
)
from stg_lab.engine_runtime import local_runtime_source_fingerprints
from stg_lab.protocol import Action
from stg_lab.vision import VisionConfig


def observation(
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
        "performance": {"native_fps": 59.5, "object_count": 320},
        "stage": {"card_index": 4},
        "safety_zone_overlay": {
            "schema_version": 2,
            "enabled": True,
            "data_source": "controller",
            "controller_revision": frame,
        },
        "world": {
            "pl": -192.0,
            "pr": 192.0,
            "pb": -224.0,
            "pt": 224.0,
        },
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


class FakeEngineClient:
    def __init__(
        self,
        *,
        terminate_at: int = 6,
        initial_frame: int = 0,
        frame_step: int = 1,
        completion_reason: str = "attack_complete",
        final_death: int = 0,
    ) -> None:
        self.terminate_at = terminate_at
        self.initial_frame = initial_frame
        self.frame_step = frame_step
        self.completion_reason = completion_reason
        self.final_death = final_death
        self.frame = initial_frame
        self.actions: list[Action] = []
        self.controller_overlay_states: list[Mapping[str, Any] | None] = []
        self.runtime_source_crc32 = local_runtime_source_fingerprints()[0]
        self.commands = [
            "ping", "catalog", "reset", "reset_stage", "step", "observe",
            "display", "save_replay", "close", "shutdown",
        ]
        self.replay_name: str | None = None
        self.replay_episode_kind: str | None = None
        self.replay_seed: int | None = None
        self.replay_player: str | None = None
        self.save_replay_calls: list[tuple[bool, str]] = []
        self.invalid_replay_save = False
        self.replay_frame_bytes_delta = 0

    def ping(self):
        return {
            "protocol": 2,
            "commands": self.commands,
            "session_id": "fake-mpc-prefix",
            "process_nonce": "fake-process",
            "runtime_identity": {
                "process_id": 42,
                "source_crc32": self.runtime_source_crc32,
            },
        }

    def catalog(self):
        return {"catalog": {"attacks": [
            {"scenario": "okuu:Lunatic", "attack": 3, "card_index": 4},
        ], "stages": [
            {"stage": "Stage 5@Lunatic", "stage_index": 5},
        ]}}

    def _reset_response(self, *, replay_name, episode_kind, stage_name, seed, player):
        self.frame = self.initial_frame
        response = {"observation": observation(self.frame)}
        if replay_name is not None:
            self.replay_name = replay_name
            self.replay_episode_kind = episode_kind
            self.replay_seed = seed
            self.replay_player = player
            response["reset"] = {"replay": {
                "schema_version": 1,
                "name": replay_name,
                "path": f"userdata/replay/test/analysis/{replay_name}.rep",
                "stage_name": stage_name,
                "random_seed": seed,
                "player": player,
                "episode_kind": episode_kind,
                "saved": False,
            }}
        return response

    def reset(
        self, scenario, attack, *, seed, player, options, replay_name=None,
    ):
        return self._reset_response(
            replay_name=replay_name,
            episode_kind="attack",
            stage_name="Spell Practice@Spell Practice",
            seed=seed,
            player=player,
        )

    def reset_stage(
        self, stage, *, seed, player, options, replay_name=None,
    ):
        assert stage == "Stage 5@Lunatic"
        return self._reset_response(
            replay_name=replay_name,
            episode_kind="stage",
            stage_name=stage,
            seed=seed,
            player=player,
        )

    def set_rendering(self, enabled: bool, *, every: int = 1):
        return {"render": enabled, "every": every}

    def step(
        self,
        action: Action,
        *,
        repeat: int = 1,
        controller_overlay_state: Mapping[str, Any] | None = None,
    ):
        assert repeat == 1
        self.actions.append(action)
        self.controller_overlay_states.append(controller_overlay_state)
        self.frame += self.frame_step
        terminated = self.frame >= self.terminate_at
        return {"observation": observation(
            self.frame,
            terminated=terminated,
            reason=self.completion_reason if terminated else None,
            death=self.final_death if terminated else 0,
        )}

    def save_replay(self, *, finish: bool, reason: str):
        self.save_replay_calls.append((finish, reason))
        if self.invalid_replay_save:
            return {"replay": {"saved": False}}
        assert self.replay_name is not None
        assert self.replay_episode_kind is not None
        assert self.replay_seed is not None
        assert self.replay_player is not None
        frame_count = len(self.actions) + 1
        return {"replay": {
            "schema_version": 1,
            "name": self.replay_name,
            "path": f"userdata/replay/test/analysis/{self.replay_name}.rep",
            "stage_name": (
                "Stage 5@Lunatic"
                if self.replay_episode_kind == "stage" else
                "Spell Practice@Spell Practice"
            ),
            "random_seed": self.replay_seed,
            "player": self.replay_player,
            "episode_kind": self.replay_episode_kind,
            "frame_count": frame_count,
            "frame_bytes_verified": frame_count + self.replay_frame_bytes_delta,
            "file_size": 512 + frame_count,
            "finish": finish,
            "group_finish": 1 if finish else 0,
            "reason": reason,
            "saved": True,
            "verified": True,
            "crc32": "89abcdef",
        }}


class PredictedCollisionMPC(EngineMPC):
    def select(self, observed) -> MPCDecision:
        decision = super().select(observed)
        evaluations = tuple(
            replace(
                evaluation,
                collided=True,
                collision_frames=1,
                earliest_collision_frame=1,
                minimum_margin=-1.0,
            )
            if evaluation.action.discrete == decision.action.discrete else
            evaluation
            for evaluation in decision.evaluations
        )
        return replace(decision, evaluations=evaluations)


def test_controller_observation_delays_hazards_but_not_own_player() -> None:
    delayed = observation(10)
    delayed["player"]["x"] = -40.0
    delayed["enemy_bullets"] = [{"id": 1, "x": 12.0, "y": 30.0}]
    current = observation(15)
    current["player"]["x"] = 24.0
    current["enemy_bullets"] = [{"id": 2, "x": -12.0, "y": 20.0}]

    visible = _controller_observation(delayed, current)

    assert visible["episode_frame"] == 10
    assert visible["enemy_bullets"] == delayed["enemy_bullets"]
    assert visible["player"]["x"] == 24.0
    assert visible["own_player_observation_delay"] == 0
    assert visible["own_player_observation_frame"] == 15
    assert "performance" not in visible
    assert "safety_zone_overlay" not in visible


def test_attack_complete_is_strict_live_policy_success() -> None:
    client = FakeEngineClient()
    report = run_engine_mpc_play(
        client,  # type: ignore[arg-type]
        scenario="okuu:Lunatic",
        attack=3,
        seed=42,
        player="reimu_player",
        controller=EngineMPC(MPCConfig(observation_delay=0, horizon_frames=36)),
        config=EngineMPCPlayConfig(
            max_frames=12,
            observation_delay=0,
            render=True,
        ),
    )

    assert report["success"] is True
    assert report["episode_completed"] is True
    assert report["teacher_success"] is True
    assert report["pure_policy"] is False
    assert report["pure_policy_success"] is False
    assert report["pure_policy_validation_eligible"] is False
    assert report["region_dynamics_training_eligible"] is True
    assert report["passed"] is True
    assert report["engine"]["runtime_source_verification"]["matched"] is True
    assert report["engine"]["runtime_source_verification"]["source_count"] == 18
    assert report["engine"]["safety_zone_overlay"] == {
        "schema_version": 2,
        "enabled": True,
        "data_source": "controller",
        "controller_revision": 6,
    }
    assert "recorded_prefix" not in report
    assert report["schema_version"] == 3
    assert all(item["control_source"] == "live_mpc" for item in report["decisions"])
    assert all(action.spell is False for action in client.actions)
    assert all(item["predicted_collision"] is False for item in report["decisions"])
    assert all(
        "predicted_minimum_nonregion_margin" in item
        and "predicted_minimum_region_margin" in item
        and "predicted_immediate_corner_clearance" in item
        and item["gap_bullet_group_count"] == 0
        and item["gap_corridor_count"] == 0
        and item["gap_selected_center"] is None
        and item["gap_selected_width"] is None
        and item["gap_selected_lifetime_frames"] is None
        and item["gap_navigation_mode"] == "inactive"
        for item in report["decisions"]
    )
    assert report["gap_prediction"] == {
        "enabled": True,
        "detected_decision_count": 0,
        "selected_decision_count": 0,
        "observe_decision_count": 0,
        "enter_decision_count": 0,
        "hold_decision_count": 0,
        "exit_decision_count": 0,
        "maximum_bullet_group_count": 0,
        "maximum_corridor_count": 0,
    }
    assert report["render_performance"]["dense_frames"]["median"] == 59.5
    assert report["continuous_fire"] is True
    assert report["shoot_frames"] == report["frames"]
    assert report["shoot_rate"] == 1.0
    assert report["config"]["controller_overlay_state_published"] is True
    assert all(action.shoot for action in client.actions)
    assert [state is not None for state in client.controller_overlay_states] == [
        True,
        False,
        False,
        True,
        False,
        False,
    ]
    first_overlay_state = client.controller_overlay_states[0]
    assert first_overlay_state is not None
    assert first_overlay_state["schema_version"] == 1
    assert first_overlay_state["region_navigation_active"] is False


@pytest.mark.parametrize("replay_name", [
    "CON",
    "prn",
    "Aux.rep",
    "nul.REP",
    "CON.analysis.rep",
    *(f"cOm{index}.rep" for index in range(1, 10)),
    *(f"LpT{index}" for index in range(1, 10)),
])
def test_replay_name_rejects_windows_reserved_basenames(
    replay_name: str,
) -> None:
    with pytest.raises(ValueError, match="Windows reserved basename"):
        EngineMPCPlayConfig(replay_name=replay_name)


@pytest.mark.parametrize("replay_name", [
    "console",
    "COM0",
    "COM10",
    "x.CON",
    "COM1-analysis",
])
def test_replay_name_accepts_nonreserved_near_matches(replay_name: str) -> None:
    assert EngineMPCPlayConfig(replay_name=replay_name).replay_name == replay_name


def test_replay_name_rejects_windows_trimmed_trailing_dot() -> None:
    with pytest.raises(ValueError, match="portable filename"):
        EngineMPCPlayConfig(replay_name="boss3-analysis.")


def test_native_replay_is_saved_for_successful_attack_with_group_finish_false() -> None:
    client = FakeEngineClient()
    config = EngineMPCPlayConfig(
        max_frames=12,
        observation_delay=0,
        replay_name="boss3-analysis.REP",
    )

    report = run_engine_mpc_play(
        client,  # type: ignore[arg-type]
        scenario="okuu:Lunatic",
        attack=3,
        seed=42,
        player="reimu_player",
        controller=EngineMPC(MPCConfig(observation_delay=0, horizon_frames=36)),
        config=config,
    )

    assert config.replay_name == "boss3-analysis"
    assert client.replay_name == "boss3-analysis"
    assert client.save_replay_calls == [(False, "attack_complete")]
    assert report["success"] is True
    assert report["native_replay"] == {
        "schema_version": 1,
        "name": "boss3-analysis",
        "path": "userdata/replay/test/analysis/boss3-analysis.rep",
        "stage_name": "Spell Practice@Spell Practice",
        "random_seed": 42,
        "player": "reimu_player",
        "episode_kind": "attack",
        "frame_count": 7,
        "frame_bytes_verified": 7,
        "file_size": 519,
        "finish": False,
        "group_finish": 0,
        "reason": "attack_complete",
        "saved": True,
        "verified": True,
        "crc32": "89abcdef",
    }
    assert report["config"]["replay_name"] == "boss3-analysis"


def test_native_replay_is_saved_when_attack_does_not_complete() -> None:
    client = FakeEngineClient(terminate_at=99)

    report = run_engine_mpc_play(
        client,  # type: ignore[arg-type]
        scenario="okuu:Lunatic",
        attack=3,
        seed=42,
        player="reimu_player",
        controller=EngineMPC(MPCConfig(observation_delay=0, horizon_frames=36)),
        config=EngineMPCPlayConfig(
            max_frames=6,
            observation_delay=0,
            replay_name="boss3-timeout",
        ),
    )

    assert report["success"] is False
    assert report["termination_reason"] == "max_frames"
    assert client.save_replay_calls == [(False, "max_frames")]
    assert report["native_replay"]["saved"] is True
    assert report["native_replay"]["finish"] is False


def test_native_replay_marks_only_strict_full_stage_success_as_finished() -> None:
    client = FakeEngineClient(completion_reason="stage_complete")

    report = run_engine_mpc_play(
        client,  # type: ignore[arg-type]
        scenario="Stage 5@Lunatic",
        attack=None,
        stage="Stage 5@Lunatic",
        seed=42,
        player="reimu_player",
        controller=EngineMPC(MPCConfig(observation_delay=0, horizon_frames=36)),
        config=EngineMPCPlayConfig(
            max_frames=12,
            observation_delay=0,
            replay_name="stage5-clear",
        ),
    )

    assert report["success"] is True
    assert client.save_replay_calls == [(True, "stage_complete")]
    assert report["native_replay"]["episode_kind"] == "stage"
    assert report["native_replay"]["finish"] is True

    failed_client = FakeEngineClient(
        completion_reason="stage_complete",
        final_death=1,
    )
    failed = run_engine_mpc_play(
        failed_client,  # type: ignore[arg-type]
        scenario="Stage 5@Lunatic",
        attack=None,
        stage="Stage 5@Lunatic",
        seed=43,
        player="reimu_player",
        controller=EngineMPC(MPCConfig(observation_delay=0, horizon_frames=36)),
        config=EngineMPCPlayConfig(
            max_frames=12,
            observation_delay=0,
            replay_name="stage5-death",
        ),
    )
    assert failed["success"] is False
    assert failed_client.save_replay_calls == [(False, "stage_complete")]
    assert failed["native_replay"]["finish"] is False


def test_native_replay_requires_advertised_bridge_command_before_reset() -> None:
    client = FakeEngineClient()
    client.commands.remove("save_replay")

    with pytest.raises(EngineProtocolError, match="does not advertise"):
        run_engine_mpc_play(
            client,  # type: ignore[arg-type]
            scenario="okuu:Lunatic",
            attack=3,
            seed=42,
            player="reimu_player",
            controller=EngineMPC(MPCConfig(observation_delay=0, horizon_frames=36)),
            config=EngineMPCPlayConfig(
                max_frames=12,
                observation_delay=0,
                replay_name="unsupported",
            ),
        )

    assert client.replay_name is None
    assert client.actions == []


def test_native_replay_rejects_invalid_save_response() -> None:
    client = FakeEngineClient()
    client.invalid_replay_save = True

    with pytest.raises(EngineProtocolError, match="invalid saved replay metadata"):
        run_engine_mpc_play(
            client,  # type: ignore[arg-type]
            scenario="okuu:Lunatic",
            attack=3,
            seed=42,
            player="reimu_player",
            controller=EngineMPC(MPCConfig(observation_delay=0, horizon_frames=36)),
            config=EngineMPCPlayConfig(
                max_frames=12,
                observation_delay=0,
                replay_name="invalid-save",
            ),
        )

    assert client.save_replay_calls == [(False, "attack_complete")]


def test_native_replay_rejects_unverified_frame_bytes() -> None:
    client = FakeEngineClient()
    client.replay_frame_bytes_delta = -1

    with pytest.raises(EngineProtocolError, match="invalid saved replay metadata"):
        run_engine_mpc_play(
            client,  # type: ignore[arg-type]
            scenario="okuu:Lunatic",
            attack=3,
            seed=42,
            player="reimu_player",
            controller=EngineMPC(MPCConfig(observation_delay=0, horizon_frames=36)),
            config=EngineMPCPlayConfig(
                max_frames=12,
                observation_delay=0,
                replay_name="invalid-frame-data",
            ),
        )

    assert client.save_replay_calls == [(False, "attack_complete")]


def test_predicted_collision_never_stops_continuous_fire() -> None:
    client = FakeEngineClient()
    report = run_engine_mpc_play(
        client,  # type: ignore[arg-type]
        scenario="okuu:Lunatic",
        attack=3,
        seed=42,
        player="reimu_player",
        controller=PredictedCollisionMPC(MPCConfig(
            observation_delay=0,
            horizon_frames=36,
        )),
        config=EngineMPCPlayConfig(
            max_frames=12,
            observation_delay=0,
            shoot_minimum_margin=1_000_000.0,
        ),
    )

    assert all(action.shoot and not action.spell for action in client.actions)
    assert all(item["predicted_collision"] for item in report["decisions"])
    assert report["continuous_fire"] is True
    assert report["shoot_frames"] == report["frames"] == 6
    assert report["shoot_rate"] == 1.0
    assert report["predicted_collision_plan_frames"] == 6
    assert report["unsafe_shot_frames"] is None
    assert report["unsafe_shot_frames_deprecated"] is True
    assert report["config"]["shoot_minimum_margin_controls_fire"] is False
    assert report["config"]["controller_overlay_state_published"] is False
    assert all(state is None for state in client.controller_overlay_states)


def test_engine_mpc_rejects_stale_runtime_lua_before_reset() -> None:
    client = FakeEngineClient()
    client.runtime_source_crc32 = {
        **client.runtime_source_crc32,
        "root.lua": "00000000",
    }

    with pytest.raises(EngineProtocolError, match="changed=.*root.lua"):
        run_engine_mpc_play(
            client,  # type: ignore[arg-type]
            scenario="okuu:Lunatic",
            attack=3,
            seed=42,
            player="reimu_player",
            controller=EngineMPC(MPCConfig(observation_delay=0, horizon_frames=36)),
            config=EngineMPCPlayConfig(max_frames=12, observation_delay=0),
        )


def test_disabled_gap_prediction_reports_only_inactive_diagnostics() -> None:
    report = run_engine_mpc_play(
        FakeEngineClient(),  # type: ignore[arg-type]
        scenario="okuu:Lunatic",
        attack=3,
        seed=42,
        player="reimu_player",
        controller=EngineMPC(MPCConfig(
            observation_delay=0,
            horizon_frames=36,
            gap_prediction_enabled=False,
        )),
        config=EngineMPCPlayConfig(max_frames=12, observation_delay=0),
    )

    assert report["gap_prediction"] == {
        "enabled": False,
        "detected_decision_count": 0,
        "selected_decision_count": 0,
        "observe_decision_count": 0,
        "enter_decision_count": 0,
        "hold_decision_count": 0,
        "exit_decision_count": 0,
        "maximum_bullet_group_count": 0,
        "maximum_corridor_count": 0,
    }
    assert all(
        item["gap_navigation_mode"] == "inactive"
        and item["gap_bullet_group_count"] == 0
        and item["gap_corridor_count"] == 0
        and item["gap_selected_center"] is None
        for item in report["decisions"]
    )


def test_action_artifact_replay_is_not_part_of_current_mpc_api() -> None:
    parameters = inspect.signature(run_engine_mpc_play).parameters
    assert "prefix_artifact" not in parameters
    assert "prefix_until_frame" not in parameters

    import stg_lab.engine_mpc_play as module

    assert not hasattr(module, "load_recorded_action_prefix")


def test_max_frames_is_not_success() -> None:
    report = run_engine_mpc_play(
        FakeEngineClient(terminate_at=99),  # type: ignore[arg-type]
        scenario="okuu:Lunatic",
        attack=3,
        seed=42,
        player="reimu_player",
        controller=EngineMPC(MPCConfig(observation_delay=0, horizon_frames=36)),
        config=EngineMPCPlayConfig(max_frames=6, observation_delay=0),
    )

    assert report["success"] is False
    assert report["passed"] is False
    assert report["terminated"] is False
    assert report["termination_reason"] == "max_frames"


def test_completion_with_a_recorded_death_is_not_success() -> None:
    report = run_engine_mpc_play(
        FakeEngineClient(final_death=1),  # type: ignore[arg-type]
        scenario="okuu:Lunatic",
        attack=3,
        seed=42,
        player="reimu_player",
        controller=EngineMPC(MPCConfig(observation_delay=0, horizon_frames=36)),
        config=EngineMPCPlayConfig(max_frames=12, observation_delay=0),
    )

    assert report["terminated"] is True
    assert report["termination_reason"] == "attack_complete"
    assert report["outcome_evidence"]["final_player"]["death"] == 1
    assert report["success"] is False
    assert report["passed"] is False
    assert "death=0" in report["success_criterion"]


def test_full_stage_requires_stage_complete() -> None:
    success = run_engine_mpc_play(
        FakeEngineClient(completion_reason="stage_complete"),  # type: ignore[arg-type]
        scenario="Stage 5@Lunatic",
        attack=None,
        stage="Stage 5@Lunatic",
        seed=42,
        player="reimu_player",
        controller=EngineMPC(MPCConfig(observation_delay=0, horizon_frames=36)),
        config=EngineMPCPlayConfig(max_frames=12, observation_delay=0),
    )
    assert success["success"] is True
    assert success["episode_kind"] == "stage"
    assert success["stage"] == "Stage 5@Lunatic"
    assert success["attack"] is None
    assert success["termination_reason"] == "stage_complete"

    wrong_reason = run_engine_mpc_play(
        FakeEngineClient(completion_reason="attack_complete"),  # type: ignore[arg-type]
        scenario="Stage 5@Lunatic",
        attack=None,
        stage="Stage 5@Lunatic",
        seed=42,
        player="reimu_player",
        controller=EngineMPC(MPCConfig(observation_delay=0, horizon_frames=36)),
        config=EngineMPCPlayConfig(max_frames=12, observation_delay=0),
    )
    assert wrong_reason["success"] is False


def test_engine_episode_frame_must_advance_one_per_step() -> None:

    with pytest.raises(EngineProtocolError, match="advance by exactly one"):
        run_engine_mpc_play(
            FakeEngineClient(frame_step=2),  # type: ignore[arg-type]
            scenario="okuu:Lunatic",
            attack=3,
            seed=42,
            player="reimu_player",
            controller=EngineMPC(MPCConfig(observation_delay=0)),
            config=EngineMPCPlayConfig(observation_delay=0),
        )


def test_native_decision_observer_receives_latest_stream_frames() -> None:
    samples = []
    report = run_engine_mpc_play(
        FakeEngineClient(),  # type: ignore[arg-type]
        scenario="okuu:Lunatic",
        attack=3,
        seed=42,
        player="reimu_player",
        controller=EngineMPC(MPCConfig(observation_delay=0, horizon_frames=36)),
        config=EngineMPCPlayConfig(max_frames=12, observation_delay=0),
        decision_observer=lambda visible, action, risk: samples.append(
            (visible, action, risk)
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

    assert report["success"] is True
    assert len(samples) == report["decision_count"]
    assert all(value[0].global_frames.shape == (1, 6, 14, 12) for value in samples)
    assert all(value[0].local_frames.shape == (1, 6, 10, 10) for value in samples)
    assert all(0.0 <= value[2] <= 1.0 for value in samples)
