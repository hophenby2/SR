from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stg_lab.engine_play import (
    EnginePlayConfig,
    load_route_controller,
    load_route_library_controller,
    run_engine_play,
)
from stg_lab.memory import EpisodicMemory
from stg_lab.protocol import Action
from stg_lab.vision import VisionConfig, VisionObservation


def observation(
    frame: int,
    *,
    terminated: bool = False,
    reason: str | None = None,
    nearby_threat: bool = False,
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
            "death": int(reason == "player_hit"),
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
    ) -> None:
        self.terminate_at = terminate_at
        self.reason = reason
        self.nearby_threat = nearby_threat
        self.frame = 0
        self.actions: list[tuple[Action, int]] = []
        self.reset_call: dict[str, Any] | None = None
        self.render_calls: list[tuple[bool, int]] = []

    def ping(self):
        return {
            "protocol": 2,
            "session_id": "fake-live-session",
            "process_nonce": "fake-process",
            "runtime_identity": {"process_id": 42},
        }

    def catalog(self):
        return {"catalog": {"attacks": [
            {"scenario": "okuu:Lunatic", "attack": 3, "card_index": 4},
            {"scenario": "okuu:Lunatic", "attack": 4, "card_index": 5},
        ]}}

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

    def step(self, action: Action, *, repeat: int = 1):
        self.actions.append((action, repeat))
        self.frame += repeat
        terminated = self.terminate_at is not None and self.frame >= self.terminate_at
        return {"observation": observation(
            self.frame,
            terminated=terminated,
            reason=self.reason if terminated else None,
            nearby_threat=self.nearby_threat,
        )}


class AlternatingController:
    def reset(self) -> None:
        self.calls = 0

    def select(self, visible: VisionObservation) -> Action:
        assert isinstance(visible, VisionObservation)
        move_x = -1 if self.calls % 2 == 0 else 1
        self.calls += 1
        return Action(move_x=move_x, slow=True, shoot=False, spell=True)


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


def test_engine_play_holds_each_visible_decision_for_three_frames() -> None:
    client = FakeEngineClient(terminate_at=7, reason="attack_complete")
    report = run_engine_play(
        client,  # type: ignore[arg-type]
        scenario="okuu:Lunatic",
        attack=3,
        seed=42,
        player="reimu_player",
        controller=AlternatingController(),
        config=play_config(max_frames=30),
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
    assert report["unsafe_shot_frames"] == 0


def test_engine_play_visible_threat_gate_suppresses_shooting_and_timeout_fails() -> None:
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

    assert all(not action.shoot for action, _repeat in client.actions)
    assert report["action_steps"][0]["shoot_gate"]["safe"] is False
    assert report["action_steps"][0]["shoot_gate"]["risk"] > 0.25
    assert report["terminated"] is False
    assert report["termination_reason"] == "max_frames"
    assert report["engine_termination_reason"] is None
    assert report["success"] is False
    assert report["shoot_rate"] == 0.0


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
