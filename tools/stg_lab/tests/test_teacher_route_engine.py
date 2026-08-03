from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from stg_lab.engine import EngineProtocolError
from stg_lab.engine_runtime import local_runtime_source_fingerprints
from stg_lab.native_dataset import NativeDemonstrationBuilder
from stg_lab.protocol import Action
from stg_lab.teacher_route_engine import (
    NativeRouteConfig,
    capture_route_conditioned_field,
    load_route_program,
    replay_teacher_route_strict,
)
from stg_lab.training import Demonstrations


def write_route(path: Path, *, frame_count: int = 3) -> None:
    path.write_text(json.dumps({
        "schema_version": 2,
        "kind": "offline_space_time_teacher_route",
        "timeline_semantics": "initial_state_plus_post_step_frames",
        "config": {
            "bounds": [-184.0, 184.0, -208.0, 192.0],
            "boundary_padding": 2.0,
            "fast_speed": 4.0,
            "focus_speed": 2.0,
            "start_x": 0.0,
            "start_y": -176.0,
        },
        "decisions": [{
            "frame_count": frame_count,
            "action": Action(move_x=1, slow=True).to_dict(),
            "minimum_clearance": 8.0,
        }],
    }), encoding="utf-8")


def observation(
    frame: int,
    x: float,
    *,
    hp: float,
    terminated: bool = False,
    reason: str | None = None,
    death: int | None = 0,
) -> dict[str, Any]:
    player = {"x": x, "y": -176.0, "a": 0.5, "b": 0.5}
    if death is not None:
        player["death"] = death
    return {
        "episode_frame": frame,
        "terminated": terminated,
        "termination_reason": reason,
        "player": player,
        "enemy_bullets": [{
            "id": frame,
            "x": 50.0,
            "y": 20.0,
            "a": 4.0,
            "b": 4.0,
            "rot": 30.0,
            "collidable": True,
        }],
        "enemies": [{
            "id": 20,
            "x": 0.0,
            "y": 100.0,
            "a": 16.0,
            "b": 16.0,
            "hp": hp,
            "maxhp": 100.0,
            "collidable": True,
        }],
        "nontjt_enemies": [],
        "indestructibles": [],
    }


class FakeClient:
    def __init__(
        self,
        *,
        terminate_at: int,
        reason: str,
        final_death: int | None = 0,
    ) -> None:
        self.terminate_at = terminate_at
        self.reason = reason
        self.final_death = final_death
        self.frame = 1
        self.x = 0.0
        self.reset_options: dict[str, Any] | None = None
        self.actions: list[Action] = []
        self.runtime_source_crc32 = local_runtime_source_fingerprints()[0]

    def ping(self) -> dict[str, Any]:
        return {
            "protocol": 2,
            "session_id": "teacher-test",
            "process_nonce": "fake-process",
            "runtime_identity": {
                "process_id": 42,
                "source_crc32": self.runtime_source_crc32,
            },
        }

    def reset(self, scenario, attack, *, seed, player, options):
        self.frame = 1
        self.x = 0.0
        self.reset_options = dict(options)
        return {"observation": observation(1, self.x, hp=100.0)}

    def step(self, action: Action, *, repeat: int = 1):
        assert repeat == 1
        self.actions.append(action)
        self.frame += 1
        self.x += 2.0 * action.move_x if action.slow else 4.0 * action.move_x
        terminated = self.frame >= self.terminate_at
        return {"observation": observation(
            self.frame,
            self.x,
            hp=max(0.0, 100.0 - 10.0 * (self.frame - 1)),
            terminated=terminated,
            reason=self.reason if terminated else None,
            death=self.final_death if terminated else 0,
        )}


def test_ghost_capture_extends_a_short_route_until_attack_complete(tmp_path) -> None:
    route_path = tmp_path / "route.json"
    write_route(route_path, frame_count=3)
    route = load_route_program(route_path)
    client = FakeClient(terminate_at=6, reason="attack_complete")
    report = capture_route_conditioned_field(
        client,  # type: ignore[arg-type]
        route=route,
        scenario="okuu:Lunatic",
        attack=4,
        seed=7,
        player="reimu_player",
        output_npz=tmp_path / "field.npz",
        output_json=tmp_path / "field.json",
        iteration=2,
    )

    assert client.reset_options == {
        "player_ghost": True,
        "player_collidable": False,
        "lifeleft": 99,
    }
    assert report["strict_success"] is False
    assert report["field_complete"] is True
    assert report["engine"]["runtime_source_verification"]["matched"] is True
    assert report["engine"]["runtime_source_verification"]["source_count"] == 18
    assert report["route_frames_used"] == 3
    assert report["fallback_frames"] == 2
    assert report["boss_hp_at_route_exhaustion"] == 70.0
    assert report["trace"]["maximum_route_divergence"] == 0.0
    with np.load(tmp_path / "field.npz") as payload:
        assert payload["frames"].tolist() == [1, 2, 3, 4, 5, 6]
        assert payload["offsets"].tolist() == [0, 2, 4, 6, 8, 10, 12]
        assert payload["threats"].shape == (12, 6)
        assert payload["player_positions"].shape == (6, 2)


def test_strict_replay_uses_default_collision_and_rejects_player_hit(tmp_path) -> None:
    route_path = tmp_path / "route.json"
    write_route(route_path)
    client = FakeClient(terminate_at=3, reason="player_hit")
    report = replay_teacher_route_strict(
        client,  # type: ignore[arg-type]
        route=load_route_program(route_path),
        scenario="okuu:Lunatic",
        attack=4,
        seed=7,
        player="reimu_player",
        output_json=tmp_path / "strict.json",
    )
    assert client.reset_options == {}
    assert report["strict_success"] is False
    assert report["termination_reason"] == "player_hit"
    assert report["route_frames_used"] == 2
    assert report["terminal_evidence"]["nearest_collidables_before"]


def test_teacher_route_rejects_stale_runtime_sources_before_reset(tmp_path) -> None:
    route_path = tmp_path / "route.json"
    write_route(route_path)
    client = FakeClient(terminate_at=4, reason="attack_complete")
    client.runtime_source_crc32["root.lua"] = "00000000"

    with pytest.raises(EngineProtocolError, match="root.lua"):
        replay_teacher_route_strict(
            client,  # type: ignore[arg-type]
            route=load_route_program(route_path),
            scenario="okuu:Lunatic",
            attack=4,
            seed=7,
            player="reimu_player",
            output_json=tmp_path / "strict.json",
        )

    assert client.reset_options is None


def test_strict_replay_only_accepts_native_attack_complete(tmp_path) -> None:
    route_path = tmp_path / "route.json"
    write_route(route_path)
    client = FakeClient(terminate_at=4, reason="attack_complete")
    report = replay_teacher_route_strict(
        client,  # type: ignore[arg-type]
        route=load_route_program(route_path),
        scenario="okuu:Lunatic",
        attack=4,
        seed=7,
        player="reimu_player",
        output_json=tmp_path / "strict.json",
        config=NativeRouteConfig(max_frames=10),
    )
    assert report["strict_success"] is True
    assert report["termination_reason"] == "attack_complete"
    assert report["final_death"] == 0
    assert report["terminal_evidence"]["final_player_death"] == 0
    assert report["trace"]["frames_advanced"] == 3


def test_strict_replay_distills_only_visible_decisions_after_zero_death(
    tmp_path,
) -> None:
    route_path = tmp_path / "route.json"
    write_route(route_path)
    builder = NativeDemonstrationBuilder()
    report = replay_teacher_route_strict(
        FakeClient(terminate_at=4, reason="attack_complete"),  # type: ignore[arg-type]
        route=load_route_program(route_path),
        scenario="okuu:Lunatic",
        attack=4,
        seed=7,
        player="reimu_player",
        output_json=tmp_path / "strict.json",
        config=NativeRouteConfig(max_frames=10),
        demonstration_builder=builder,
    )

    assert report["strict_success"] is True
    assert report["demonstration_collection"] == {
        "decisions_recorded": 1,
        "strict_success_retained": True,
        "model_input_excludes_route_and_absolute_frame": True,
    }
    manifest = builder.save(
        tmp_path / "boss4-visible.npz",
        manifest_path=tmp_path / "boss4-visible.manifest.json",
    )
    demonstrations = Demonstrations.load(tmp_path / "boss4-visible.npz")
    assert demonstrations.actions.tolist() == [[Action(move_x=1, slow=True).discrete]]
    assert demonstrations.risks.tolist() == [[0.5]]
    assert demonstrations.memory is None
    assert demonstrations.proficiency is None
    assert manifest["excluded_model_inputs"] == [
        "scenario_identity",
        "attack_identity",
        "absolute_frame",
        "script_phase",
        "recorded_route",
        "waypoints",
    ]


def test_failed_strict_replay_discards_visible_decisions(tmp_path) -> None:
    route_path = tmp_path / "route.json"
    write_route(route_path)
    builder = NativeDemonstrationBuilder()
    report = replay_teacher_route_strict(
        FakeClient(terminate_at=3, reason="player_hit"),  # type: ignore[arg-type]
        route=load_route_program(route_path),
        scenario="okuu:Lunatic",
        attack=4,
        seed=7,
        player="reimu_player",
        output_json=tmp_path / "strict.json",
        demonstration_builder=builder,
    )

    assert report["strict_success"] is False
    assert report["demonstration_collection"]["strict_success_retained"] is False
    assert builder.accepted_count == 0
    with pytest.raises(ValueError, match="no strictly successful"):
        builder.build()


@pytest.mark.parametrize("final_death", (1, None))
def test_strict_replay_rejects_completion_without_explicit_zero_death(
    tmp_path,
    final_death,
) -> None:
    route_path = tmp_path / "route.json"
    write_route(route_path)
    report = replay_teacher_route_strict(
        FakeClient(
            terminate_at=4,
            reason="attack_complete",
            final_death=final_death,
        ),  # type: ignore[arg-type]
        route=load_route_program(route_path),
        scenario="okuu:Lunatic",
        attack=4,
        seed=7,
        player="reimu_player",
        output_json=tmp_path / "strict.json",
        config=NativeRouteConfig(max_frames=10),
    )

    assert report["termination_reason"] == "attack_complete"
    assert report["final_death"] == final_death
    assert report["strict_success"] is False
