from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from stg_lab.engine import EngineProtocolError
from stg_lab.engine_mpc import EngineMPC, MPCConfig
from stg_lab.engine_mpc_play import (
    EngineMPCPlayConfig,
    _controller_observation,
    load_recorded_action_prefix,
    run_engine_mpc_play,
)
from stg_lab.protocol import Action


def observation(
    frame: int,
    *,
    terminated: bool = False,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "episode_frame": frame,
        "terminated": terminated,
        "termination_reason": reason,
        "stage": {"card_index": 4},
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
    ) -> None:
        self.terminate_at = terminate_at
        self.initial_frame = initial_frame
        self.frame_step = frame_step
        self.frame = initial_frame
        self.actions: list[Action] = []

    def ping(self):
        return {
            "protocol": 2,
            "session_id": "fake-mpc-prefix",
            "process_nonce": "fake-process",
            "runtime_identity": {"process_id": 42},
        }

    def catalog(self):
        return {"catalog": {"attacks": [
            {"scenario": "okuu:Lunatic", "attack": 3, "card_index": 4},
        ]}}

    def reset(self, scenario, attack, *, seed, player, options):
        self.frame = self.initial_frame
        return {"observation": observation(self.frame)}

    def set_rendering(self, enabled: bool, *, every: int = 1):
        return {"render": enabled, "every": every}

    def step(self, action: Action, *, repeat: int = 1):
        assert repeat == 1
        self.actions.append(action)
        self.frame += self.frame_step
        terminated = self.frame >= self.terminate_at
        return {"observation": observation(
            self.frame,
            terminated=terminated,
            reason="attack_complete" if terminated else None,
        )}


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


def write_prefix(
    path: Path,
    *,
    scenario: str = "okuu:Lunatic",
    attack: int = 3,
    seed: int = 42,
    player: str = "reimu_player",
    spell: bool = False,
    start_episode_frame: int = 0,
    requested_frames: int = 3,
    advanced_frames: int = 3,
) -> None:
    path.write_text(json.dumps({
        "schema_version": 1,
        "run_kind": "live_luastg_delayed_visible_mpc_teacher",
        "scenario": scenario,
        "attack": attack,
        "seed": seed,
        "player": player,
        "initial_episode_frame": start_episode_frame,
        "decision_count": 1,
        "config": {
            "reset_options": {},
            "authority_state_shield": False,
            "spell_forced_off": True,
        },
        "decisions": [{
            "decision": 0,
            "start_episode_frame": start_episode_frame,
            "end_episode_frame": start_episode_frame + advanced_frames,
            "requested_frames": requested_frames,
            "advanced_frames": advanced_frames,
            "action": {
                "move_x": 1,
                "move_y": 0,
                "slow": True,
                "shoot": False,
                "spell": spell,
            },
        }],
    }), encoding="utf-8")


def update_prefix(path: Path, update) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    update(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_no_prefix_attack_complete_is_policy_validation_eligible() -> None:
    report = run_engine_mpc_play(
        FakeEngineClient(),  # type: ignore[arg-type]
        scenario="okuu:Lunatic",
        attack=3,
        seed=42,
        player="reimu_player",
        controller=EngineMPC(MPCConfig(observation_delay=0, horizon_frames=36)),
        config=EngineMPCPlayConfig(max_frames=12, observation_delay=0),
    )

    assert report["success"] is True
    assert report["episode_completed"] is True
    assert report["policy_validation_eligible"] is True
    assert report["passed"] is True


def test_recorded_prefix_replays_then_switches_to_live_mpc(tmp_path: Path) -> None:
    artifact = tmp_path / "prefix.json"
    write_prefix(artifact)
    client = FakeEngineClient()
    controller = EngineMPC(MPCConfig(observation_delay=0, horizon_frames=36))

    report = run_engine_mpc_play(
        client,  # type: ignore[arg-type]
        scenario="okuu:Lunatic",
        attack=3,
        seed=42,
        player="reimu_player",
        controller=controller,
        config=EngineMPCPlayConfig(max_frames=12, observation_delay=0),
        prefix_artifact=artifact,
        prefix_until_frame=3,
    )

    assert report["success"] is True
    assert report["episode_completed"] is True
    assert report["policy_validation_eligible"] is False
    assert report["passed"] is False
    assert [item["control_source"] for item in report["decisions"]] == [
        "recorded_prefix",
        "live_mpc",
    ]
    assert all(action.move_x == 1 and not action.shoot for action in client.actions[:3])
    assert all(action.spell is False for action in client.actions)
    assert report["recorded_prefix"]["used_decisions"] == 1
    assert report["recorded_prefix"]["used_frames"] == 3
    assert report["recorded_prefix"]["effective_live_switch_episode_frame"] == 3
    assert len(report["recorded_prefix"]["artifact_sha256"]) == 64
    assert report["decisions"][0]["predicted_collision"] is None
    assert report["decisions"][1]["predicted_collision"] is False


def test_prefix_cutoff_inside_decision_switches_at_decision_end(tmp_path: Path) -> None:
    artifact = tmp_path / "prefix-2024.json"
    write_prefix(artifact, start_episode_frame=2023)
    client = FakeEngineClient(initial_frame=2023, terminate_at=2029)
    controller = EngineMPC(MPCConfig(observation_delay=0, horizon_frames=36))

    report = run_engine_mpc_play(
        client,  # type: ignore[arg-type]
        scenario="okuu:Lunatic",
        attack=3,
        seed=42,
        player="reimu_player",
        controller=controller,
        config=EngineMPCPlayConfig(max_frames=6, observation_delay=0),
        prefix_artifact=artifact,
        prefix_until_frame=2024,
    )

    assert report["success"] is True
    assert report["policy_validation_eligible"] is False
    assert report["passed"] is False
    assert [item["start_episode_frame"] for item in report["decisions"]] == [2023, 2026]
    assert [item["control_source"] for item in report["decisions"]] == [
        "recorded_prefix",
        "live_mpc",
    ]
    assert report["recorded_prefix"]["until_episode_frame"] == 2024
    assert report["recorded_prefix"]["selected_decisions"] == 1
    assert report["recorded_prefix"]["effective_live_switch_episode_frame"] == 2026


def test_prefix_loader_rejects_identity_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "wrong-seed.json"
    write_prefix(artifact, seed=41)

    with pytest.raises(ValueError, match="identity does not match"):
        load_recorded_action_prefix(
            artifact,
            scenario="okuu:Lunatic",
            attack=3,
            seed=42,
            player="reimu_player",
        )


def test_prefix_loader_rejects_spell_actions(tmp_path: Path) -> None:
    artifact = tmp_path / "spell.json"
    write_prefix(artifact, spell=True)

    with pytest.raises(ValueError, match="uses a spell"):
        load_recorded_action_prefix(
            artifact,
            scenario="okuu:Lunatic",
            attack=3,
            seed=42,
            player="reimu_player",
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(schema_version=2), "schema_version must be 1"),
        (lambda value: value.update(decision_count=2), "decision_count does not match"),
        (
            lambda value: value.update(initial_episode_frame=1),
            "first decision does not start at initial_episode_frame",
        ),
    ],
)
def test_prefix_loader_rejects_invalid_schema_and_frame_metadata(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    artifact = tmp_path / "invalid-metadata.json"
    write_prefix(artifact)
    update_prefix(artifact, mutate)

    with pytest.raises(ValueError, match=message):
        load_recorded_action_prefix(
            artifact,
            scenario="okuu:Lunatic",
            attack=3,
            seed=42,
            player="reimu_player",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("move_x", True),
        ("move_y", 1.0),
        ("slow", 1),
        ("shoot", 0),
        ("spell", 0),
    ],
)
def test_prefix_loader_rejects_nonexact_action_types(
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    artifact = tmp_path / f"invalid-{field}.json"
    write_prefix(artifact)
    update_prefix(
        artifact,
        lambda payload: payload["decisions"][0]["action"].update({field: value}),
    )

    with pytest.raises(ValueError, match=field):
        load_recorded_action_prefix(
            artifact,
            scenario="okuu:Lunatic",
            attack=3,
            seed=42,
            player="reimu_player",
        )


def test_prefix_loader_rejects_extra_action_fields(tmp_path: Path) -> None:
    artifact = tmp_path / "extra-action-field.json"
    write_prefix(artifact)
    update_prefix(
        artifact,
        lambda payload: payload["decisions"][0]["action"].update(extra=False),
    )

    with pytest.raises(ValueError, match="action fields must be exactly"):
        load_recorded_action_prefix(
            artifact,
            scenario="okuu:Lunatic",
            attack=3,
            seed=42,
            player="reimu_player",
        )


def test_selected_prefix_decisions_must_be_complete_three_frame_holds(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "short-decision.json"
    write_prefix(artifact, requested_frames=2, advanced_frames=2)

    with pytest.raises(ValueError, match="exactly three frames"):
        run_engine_mpc_play(
            FakeEngineClient(),  # type: ignore[arg-type]
            scenario="okuu:Lunatic",
            attack=3,
            seed=42,
            player="reimu_player",
            controller=EngineMPC(MPCConfig(observation_delay=0)),
            config=EngineMPCPlayConfig(observation_delay=0),
            prefix_artifact=artifact,
            prefix_until_frame=2,
        )


def test_prefix_artifact_must_cover_cutoff(tmp_path: Path) -> None:
    artifact = tmp_path / "short-coverage.json"
    write_prefix(artifact)

    with pytest.raises(ValueError, match="does not cover"):
        run_engine_mpc_play(
            FakeEngineClient(),  # type: ignore[arg-type]
            scenario="okuu:Lunatic",
            attack=3,
            seed=42,
            player="reimu_player",
            controller=EngineMPC(MPCConfig(observation_delay=0)),
            config=EngineMPCPlayConfig(observation_delay=0),
            prefix_artifact=artifact,
            prefix_until_frame=4,
        )


def test_engine_reset_frame_must_match_recorded_prefix(tmp_path: Path) -> None:
    artifact = tmp_path / "reset-frame.json"
    write_prefix(artifact, start_episode_frame=1)

    with pytest.raises(EngineProtocolError, match="reset initial_episode_frame"):
        run_engine_mpc_play(
            FakeEngineClient(initial_frame=0),  # type: ignore[arg-type]
            scenario="okuu:Lunatic",
            attack=3,
            seed=42,
            player="reimu_player",
            controller=EngineMPC(MPCConfig(observation_delay=0)),
            config=EngineMPCPlayConfig(observation_delay=0),
            prefix_artifact=artifact,
            prefix_until_frame=2,
        )


def test_engine_episode_frame_must_advance_one_per_step(tmp_path: Path) -> None:
    artifact = tmp_path / "frame-skip.json"
    write_prefix(artifact)

    with pytest.raises(EngineProtocolError, match="advance by exactly one"):
        run_engine_mpc_play(
            FakeEngineClient(frame_step=2),  # type: ignore[arg-type]
            scenario="okuu:Lunatic",
            attack=3,
            seed=42,
            player="reimu_player",
            controller=EngineMPC(MPCConfig(observation_delay=0)),
            config=EngineMPCPlayConfig(observation_delay=0),
            prefix_artifact=artifact,
            prefix_until_frame=3,
        )


def test_prefix_options_must_be_supplied_together(tmp_path: Path) -> None:
    artifact = tmp_path / "prefix.json"
    write_prefix(artifact)

    with pytest.raises(ValueError, match="must be provided together"):
        run_engine_mpc_play(
            FakeEngineClient(),  # type: ignore[arg-type]
            scenario="okuu:Lunatic",
            attack=3,
            seed=42,
            player="reimu_player",
            controller=EngineMPC(MPCConfig(observation_delay=0)),
            config=EngineMPCPlayConfig(observation_delay=0),
            prefix_artifact=artifact,
        )
