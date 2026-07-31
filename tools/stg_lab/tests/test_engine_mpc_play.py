from __future__ import annotations

import inspect
from typing import Any

import pytest

from stg_lab.engine import EngineProtocolError
from stg_lab.engine_mpc import EngineMPC, MPCConfig
from stg_lab.engine_mpc_play import (
    EngineMPCPlayConfig,
    _controller_observation,
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
        "performance": {"native_fps": 59.5, "object_count": 320},
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
    assert "performance" not in visible


def test_attack_complete_is_strict_live_policy_success() -> None:
    client = FakeEngineClient()
    report = run_engine_mpc_play(
        client,  # type: ignore[arg-type]
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
    assert "recorded_prefix" not in report
    assert report["unsafe_shot_frames_excludes_recorded_actions"] is False
    assert all(item["control_source"] == "live_mpc" for item in report["decisions"])
    assert all(action.spell is False for action in client.actions)
    assert all(item["predicted_collision"] is False for item in report["decisions"])
    assert report["render_performance"]["dense_frames"]["median"] == 59.5


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
