from __future__ import annotations

from typing import Any

import pytest

from stg_lab.engine_replay_analysis import (
    EngineReplayAnalysisConfig,
    _ReplayTelemetry,
    run_engine_replay_analysis,
)
from stg_lab.engine_runtime import local_runtime_source_fingerprints
from stg_lab.protocol import Action


def observation(
    frame: int,
    *,
    terminated: bool = False,
    reason: str | None = None,
) -> dict[str, Any]:
    player_x = -40.0 + (frame - 1) * 40.0
    wall = [
        {
            "id": 100 + index,
            "x": 0.0,
            "y": float(y),
            "a": 8.0,
            "b": 8.0,
            "collidable": True,
        }
        for index, y in enumerate(range(-220, 221, 16))
    ]
    return {
        "episode_frame": frame,
        "terminated": terminated,
        "termination_reason": reason,
        "world": {
            "l": -192.0,
            "r": 192.0,
            "b": -224.0,
            "t": 224.0,
            "pl": -192.0,
            "pr": 192.0,
            "pb": -224.0,
            "pt": 224.0,
        },
        "player": {
            "x": player_x,
            "y": -176.0,
            "a": 0.5,
            "b": 0.5,
            "slow": 1 if frame < 3 else 0,
            "death": 1 if reason == "player_hit" else 0,
            "protect": 0,
        },
        "resources": {"lifeleft": 2 if reason == "player_hit" else 3},
        "enemy_bullets": [{
            "id": 7,
            "x": player_x + 12.0,
            "y": -176.0,
            "a": 2.0,
            "b": 2.0,
            "vx": 3.0,
            "vy": 4.0,
            "speed": 5.0,
            "collidable": True,
        }],
        "enemies": [{
            "id": 20,
            "x": 0.0,
            "y": 120.0,
            "a": 16.0,
            "b": 16.0,
            "hp": 100.0 - 20.0 * frame,
            "maxhp": 100.0,
            "collidable": False,
        }, {
            "id": 21,
            "x": 120.0,
            "y": 180.0,
            "a": 16.0,
            "b": 16.0,
            "hp": 999999999.0,
            "maxhp": 999999999.0,
            "collidable": True,
        }],
        "nontjt_enemies": [],
        "indestructibles": wall,
        "lasers": [],
    }


class FakeClient:
    def __init__(self, *, terminal_frame: int = 3, reason: str = "replay_exhausted") -> None:
        self.terminal_frame = terminal_frame
        self.reason = reason
        self.frame = 0
        self.actions: list[tuple[Action, int]] = []
        self.render_calls: list[tuple[bool, int]] = []
        self.requested_path: str | None = None
        self.runtime_source_crc32 = local_runtime_source_fingerprints()[0]

    def ping(self) -> dict[str, Any]:
        return {
            "protocol": 2,
            "commands": ["ping", "reset_replay", "step", "display"],
            "session_id": "fake-replay-analysis",
            "process_nonce": "fake-process",
            "runtime_identity": {"source_crc32": self.runtime_source_crc32},
        }

    def reset_replay(self, path: str) -> dict[str, Any]:
        self.requested_path = path
        self.frame = 1
        return {
            "reset": {
                "episode_kind": "replay",
                "scenario": "okuu:Lunatic",
                "card_index": 4,
                "replay": {
                    "schema_version": 1,
                    "path": path,
                    "file_version": 1,
                    "game_name": "SR-master",
                    "game_version": 1,
                    "group_finish": 0,
                    "user_name": "HT",
                    "stage_name": "Spell Practice@Spell Practice",
                    "stage_player": "Reimu",
                    "random_seed": 24962,
                    "frame_count": 3,
                    "frame_data_position": 100,
                    "frame_bytes_verified": 3,
                    "file_size": 103,
                    "crc32": "1234abcd",
                    "scenario": "okuu:Lunatic",
                    "card_index": 4,
                },
            },
            "observation": observation(1),
        }

    def set_rendering(self, enabled: bool, *, every: int = 1) -> dict[str, Any]:
        self.render_calls.append((enabled, every))
        return {"render": enabled, "every": every}

    def step(self, action: Action, *, repeat: int = 1) -> dict[str, Any]:
        self.actions.append((action, repeat))
        self.frame += 1
        terminal = self.frame == self.terminal_frame
        return {
            "observation": observation(
                self.frame,
                terminated=terminal,
                reason=self.reason if terminal else None,
            ),
        }


def test_replay_analysis_consumes_every_frame_and_reports_live_metrics() -> None:
    client = FakeClient()
    report = run_engine_replay_analysis(
        client,  # type: ignore[arg-type]
        replay_path="slot3.rep",
        config=EngineReplayAnalysisConfig(
            render=True,
            render_every=2,
            region_grid_cell_size=16.0,
        ),
    )

    assert report["analysis_complete"] is True
    assert report["input_stream_fully_consumed"] is True
    assert report["frames_analyzed"] == 3
    assert report["trajectory"]["path_distance"] == pytest.approx(80.0)
    assert report["trajectory"]["slow_frames"] == 2
    assert report["trajectory"]["moving_to_moving_direction_changes"] == 0
    assert report["trajectory"]["turns_at_least_90_degrees"] == 0
    assert report["trajectory"]["turns_over_90_degrees"] == 0
    assert report["trajectory"]["slow_mode_changes"] == 1
    assert report["trajectory"]["slow_mode_changes_with_direction_change"] == 0
    assert report["bullets"]["collidable_count_per_frame"]["mean"] == 1.0
    assert report["bullets"]["speed"]["object_frame_weighted_mean"] == 5.0
    assert report["observed_outcome"]["boss_hp_initial"] == 80.0
    assert report["observed_outcome"]["boss_hp_minimum_observed"] == 40.0
    assert report["conservative_clearance"]["safety_levels"]["2"]["frames"] == 2
    assert report["conservative_clearance"]["safety_levels"]["3"]["frames"] == 1
    assert report["region_topology"]["region_crossing_frames"] == 1
    assert report["region_topology"]["component_switch_count"] == 1
    assert len(report["timeline"]) == 3
    assert client.render_calls == [(True, 2)]
    assert all(action == Action(shoot=False) and repeat == 1 for action, repeat in client.actions)


@pytest.mark.parametrize("reason", ("attack_complete", "player_hit"))
def test_replay_analysis_accepts_authoritative_early_terminal_reason(reason: str) -> None:
    client = FakeClient(terminal_frame=2, reason=reason)
    report = run_engine_replay_analysis(
        client,  # type: ignore[arg-type]
        replay_path="slot2.rep",
    )

    assert report["analysis_complete"] is True
    assert report["termination_reason"] == reason
    assert report["input_stream_fully_consumed"] is False
    assert report["frames_analyzed"] == 2
    assert report["unconsumed_input_frames"] == 1
    if reason == "player_hit":
        assert report["player_state"]["death_state_entries"] == 1


def test_replay_analysis_counts_attack_completion_on_last_input_as_fully_consumed() -> None:
    client = FakeClient(terminal_frame=3, reason="attack_complete")
    report = run_engine_replay_analysis(
        client,  # type: ignore[arg-type]
        replay_path="ai.rep",
    )

    assert report["analysis_complete"] is True
    assert report["input_stream_fully_consumed"] is True
    assert report["unconsumed_input_frames"] == 0


def test_replay_analysis_stops_at_configured_limit_without_claiming_completion() -> None:
    client = FakeClient(terminal_frame=3)
    report = run_engine_replay_analysis(
        client,  # type: ignore[arg-type]
        replay_path="slot3.rep",
        config=EngineReplayAnalysisConfig(max_frames=1),
    )

    assert report["analysis_complete"] is False
    assert report["terminated"] is False
    assert report["frames_analyzed"] == 1
    assert report["unconsumed_input_frames"] == 2


def test_replay_telemetry_learns_region_cycle_only_from_observed_radii() -> None:
    telemetry = _ReplayTelemetry(EngineReplayAnalysisConfig(timeline_every=1000))
    for frame in range(1, 261):
        phase_frame = (frame - 1) % 120 + 1
        if phase_frame <= 10:
            radius = 7.0
        elif phase_frame <= 40:
            radius = min(28.0, 7.0 + (phase_frame - 10) * 0.7)
        elif phase_frame <= 55:
            radius = 28.0
        elif phase_frame <= 85:
            radius = max(7.0, 28.0 - (phase_frame - 55) * 0.7)
        else:
            radius = 7.0
        sample = observation(1)
        sample["episode_frame"] = frame
        sample["player"]["x"] = -100.0
        sample["enemy_bullets"] = []
        sample["enemies"] = []
        for record in sample["indestructibles"]:
            record["a"] = radius
            record["b"] = radius
        telemetry.push(sample)

    phase = telemetry.report()["region_phase"]
    assert phase["learned_cycle_frames"] == pytest.approx(120.0)
    assert phase["expansion_starts"] == [10, 130, 250]
    assert {item["to"] for item in phase["transitions"]} == {
        "expanding",
        "maximum_hold",
        "contracting",
        "minimum_hold",
    }


def test_replay_trajectory_reports_humanlike_turn_and_slow_mode_metrics() -> None:
    telemetry = _ReplayTelemetry(EngineReplayAnalysisConfig(timeline_every=1000))
    samples = (
        ((0.0, 0.0), False),
        ((1.0, 0.0), False),
        ((2.0, 1.0), True),
        ((2.0, 2.0), True),
        ((1.0, 2.0), False),
        ((0.0, 1.0), False),
        ((1.0, 1.0), True),
        ((0.0, 1.0), True),
        ((0.0, 1.0), False),
        ((0.0, 2.0), True),
        ((0.0, 3.0), False),
    )
    for frame, ((x, y), slow) in enumerate(samples, start=1):
        sample = observation(1)
        sample["episode_frame"] = frame
        sample["player"]["x"] = x
        sample["player"]["y"] = y
        sample["player"]["slow"] = int(slow)
        sample["enemy_bullets"] = []
        sample["enemies"] = []
        sample["indestructibles"] = []
        telemetry.push(sample)

    trajectory = telemetry.report()["trajectory"]
    assert trajectory["direction_changes"] == 9
    assert trajectory["moving_to_moving_direction_changes"] == 6
    assert trajectory["turns_at_least_90_degrees"] == 3
    assert trajectory["turns_over_90_degrees"] == 2
    assert trajectory["exact_reversals"] == 1
    assert trajectory["movement_starts"] == 2
    assert trajectory["movement_stops"] == 1
    assert trajectory["slow_mode_changes"] == 6
    assert trajectory["slow_mode_changes_with_direction_change"] == 5


@pytest.mark.parametrize(
    "kwargs",
    (
        {"max_frames": 0},
        {"render_every": 0},
        {"timeline_every": 0},
        {"region_grid_cell_size": 0.0},
    ),
)
def test_replay_analysis_config_rejects_invalid_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        EngineReplayAnalysisConfig(**kwargs)
