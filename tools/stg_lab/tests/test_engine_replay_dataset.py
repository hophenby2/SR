from __future__ import annotations

from pathlib import Path
from typing import Any
import zlib

import numpy as np
import pytest

from stg_lab.cli import build_parser
from stg_lab.engine import EngineProtocolError
from stg_lab.engine_replay_dataset import (
    ReplayDemonstrationConfig,
    aggregate_replay_actions,
    collect_replay_demonstrations,
    project_replay_actions,
    replay_byte_action,
    save_replay_demonstrations,
)
from stg_lab.engine_runtime import local_runtime_source_fingerprints
from stg_lab.engine_vision import EngineStreamVision
from stg_lab.protocol import Action


def _observation(
    frame: int,
    *,
    terminated: bool = False,
    reason: str | None = None,
) -> dict[str, Any]:
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
            "x": float(frame),
            "y": -208.0,
            "a": 0.5,
            "b": 0.5,
            "slow": 1,
            "death": 100 if reason == "player_hit" else 0,
            "protect": 0,
        },
        "resources": {"lifeleft": 3},
        "enemy_bullets": [{
            "id": 7,
            "x": float(frame * 24),
            "y": -64.0,
            "a": 2.0,
            "b": 2.0,
            "vx": 0.0,
            "vy": -2.0,
            "collidable": True,
        }],
        "enemies": [{
            "id": 20,
            "x": 0.0,
            "y": 120.0,
            "a": 16.0,
            "b": 16.0,
            "hp": max(0.0, 7000.0 - frame * 1000.0),
            "collidable": False,
        }],
        "nontjt_enemies": [],
        "indestructibles": [],
        "lasers": [],
        "counts": {
            "enemy_bullets": 1,
            "enemies": 1,
            "nontjt_enemies": 0,
            "indestructibles": 0,
            "lasers": 0,
        },
    }


class _FakeClient:
    def __init__(self, replay: Path, *, reason: str = "attack_complete") -> None:
        self.replay = replay
        self.reason = reason
        self.frame = 0
        self.actions: list[tuple[Action, int]] = []
        self.source_crc32 = local_runtime_source_fingerprints()[0]

    def ping(self) -> dict[str, Any]:
        return {
            "protocol": 2,
            "commands": [
                "ping", "catalog", "reset_replay", "step", "display",
            ],
            "session_id": "human-replay-test",
            "process_nonce": "human-replay-process",
            "runtime_identity": {"source_crc32": self.source_crc32},
        }

    def catalog(self) -> dict[str, Any]:
        return {
            "catalog": {
                "attacks": [{
                    "scenario": "okuu:Lunatic",
                    "attack": 3,
                    "card_index": 4,
                }],
            },
        }

    def reset_replay(self, path: str) -> dict[str, Any]:
        raw = self.replay.read_bytes()
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
                    "random_seed": 10292,
                    "frame_count": 7,
                    "frame_data_position": len(raw) - 7,
                    "frame_bytes_verified": 7,
                    "file_size": len(raw),
                    "crc32": f"{zlib.crc32(raw) & 0xFFFFFFFF:08x}",
                    "scenario": "okuu:Lunatic",
                    "card_index": 4,
                    "spell_practice_index": 50,
                },
            },
            "observation": _observation(1),
        }

    def set_rendering(self, enabled: bool, *, every: int = 1) -> dict[str, Any]:
        return {"render": enabled, "every": every}

    def step(self, action: Action, *, repeat: int = 1) -> dict[str, Any]:
        self.actions.append((action, repeat))
        self.frame += 1
        terminal = self.frame == 7
        return {
            "observation": _observation(
                self.frame,
                terminated=terminal,
                reason=self.reason if terminal else None,
            ),
        }


def _write_replay(
    path: Path,
    *,
    spell: bool = False,
    special: bool = False,
    shoot_gap: bool = False,
) -> None:
    header = b"test-replay-data"
    slow_shoot = 8 | 4
    left = 32 | slow_shoot
    right = 16 | slow_shoot
    spell_bit = 2 if spell else 0
    special_bit = 1 if special else 0
    assisted_right = right | spell_bit | special_bit
    if shoot_gap:
        assisted_right &= ~4
    path.write_bytes(header + bytes([
        slow_shoot,
        left,
        left,
        left,
        assisted_right,
        right,
        right,
    ]))


def _write_mixed_replay(path: Path) -> None:
    header = b"test-replay-data"
    slow_shoot = 8 | 4
    left = 32 | slow_shoot
    right = 16 | slow_shoot
    path.write_bytes(header + bytes([
        slow_shoot,
        left,
        right,
        right,
        left,
        left,
        left,
    ]))


def _write_phase_replay(
    path: Path,
    *,
    tail_spell: bool = False,
    tail_shoot_gap: bool = False,
) -> None:
    header = b"test-replay-data"
    slow_shoot = 8 | 4
    left = 32 | slow_shoot
    right = 16 | slow_shoot
    tail = left | (2 if tail_spell else 0)
    if tail_shoot_gap:
        tail &= ~4
    path.write_bytes(header + bytes([
        slow_shoot,
        left,
        right,
        right,
        right,
        left,
        tail,
    ]))


def test_replay_byte_action_and_modal_primitive() -> None:
    assert replay_byte_action(128 | 32 | 8 | 4) == Action(
        move_x=-1,
        move_y=1,
        slow=True,
        shoot=True,
    )
    assert replay_byte_action(32 | 16 | 64).move_x == 0
    assert aggregate_replay_actions([
        Action(move_x=-1),
        Action(move_x=1),
        Action(move_x=1),
    ]) == Action(move_x=1)
    actions = [Action(move_x=-1), Action(move_x=1), Action(move_x=1)]
    assert project_replay_actions(actions, "first") == Action(move_x=-1)
    assert project_replay_actions(actions, "midpoint") == Action(move_x=1)
    with pytest.raises(ValueError, match="full window"):
        project_replay_actions(actions, "exact-hold")
    assert project_replay_actions(
        [Action(move_x=-1)] * 3,
        "exact-hold",
    ) == Action(move_x=-1)


@pytest.mark.parametrize("offset", (False, -1, 3))
def test_rejects_invalid_decision_phase_offset(offset: object) -> None:
    with pytest.raises(ValueError, match="decision_phase_offset"):
        ReplayDemonstrationConfig(decision_phase_offset=offset)  # type: ignore[arg-type]


def test_collects_only_complete_causal_three_frame_primitives(tmp_path: Path) -> None:
    replay = tmp_path / "slot3.rep"
    _write_replay(replay)
    client = _FakeClient(replay)

    report, builder = collect_replay_demonstrations(
        client,  # type: ignore[arg-type]
        replay_path="../slot3.rep",
        replay_file=replay,
        config=ReplayDemonstrationConfig(),
    )

    assert report["strict_success"] is True
    assert report["decision_count"] == 2
    assert report["causal_contract"]["future_replay_bytes_are_model_inputs"] is False
    assert report["causal_contract"]["future_visual_frames_are_model_inputs"] is False
    assert report["causal_contract"]["future_replay_bytes_are_supervision"] is True
    assert report["raw_input"]["spell_frames"] == 0
    assert report["raw_input"]["special_frames"] == 0
    assert report["raw_input"]["continuous_shoot"] is True
    assert report["aggregated_motor_labels"]["exact_reversals"] == 1
    assert report["aggregated_motor_labels"]["projection"] == "exact-hold"
    assert report["aggregated_motor_labels"]["changed_input_windows"] == 0
    assert report["aggregated_motor_labels"][
        "trajectory_execution_contract_satisfied"
    ] is True
    assert report["action_supervision"]["supervised_decisions"] == 2
    demonstrations = builder.build()
    assert demonstrations.actions[:, 0].tolist() == [12, 14]
    assert demonstrations.previous_actions[:, 0].tolist() == [13, 12]
    assert demonstrations.supervision_mask[:, 0].tolist() == [True, True]

    expected = EngineStreamVision(ReplayDemonstrationConfig().vision).reset(
        _observation(1),
    )
    future = EngineStreamVision(ReplayDemonstrationConfig().vision).reset(
        _observation(4),
    )
    np.testing.assert_array_equal(
        demonstrations.global_frames[0],
        expected.global_frames,
    )
    assert not np.array_equal(
        demonstrations.global_frames[0],
        future.global_frames,
    )
    assert len(client.actions) == 6
    assert all(action == Action(shoot=False) for action, _repeat in client.actions)

    output = tmp_path / "human.npz"
    manifest_path = tmp_path / "human.manifest.json"
    manifest = save_replay_demonstrations(
        builder,
        report,
        output,
        manifest_path,
    )
    assert output.is_file()
    assert manifest_path.is_file()
    assert manifest["run_kind"] == "strict_native_human_replay_demonstrations"
    assert manifest["demonstrator_proficiency"] == "expert"
    assert manifest["temporal_contract_version"] == 3
    assert manifest["decision_phase_offset"] == 0
    assert manifest["decision_phase"] == report["decision_phase"]
    assert manifest["action_supervision"]["supervised_decisions"] == 2
    assert manifest["artifact_compatibility"] == {
        "legacy_npz_requires_migration": False,
        "legacy_projection_modes": ["first", "midpoint", "modal"],
        "legacy_cli_option": "--action-projection",
        "supervision_mask_is_additive": True,
    }


def test_exact_hold_masks_mixed_window_but_preserves_replay_trajectory(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "mixed.rep"
    _write_mixed_replay(replay)

    report, builder = collect_replay_demonstrations(
        _FakeClient(replay),  # type: ignore[arg-type]
        replay_path="mixed.rep",
        replay_file=replay,
    )

    demonstrations = builder.build()
    assert demonstrations.actions[:, 0].tolist() == [14, 12]
    assert demonstrations.previous_actions[:, 0].tolist() == [13, 14]
    assert demonstrations.supervision_mask[:, 0].tolist() == [False, True]
    assert demonstrations.episode_ids.tolist() == [0, 0]
    assert report["decision_count"] == 2
    assert report["aggregated_motor_labels"]["changed_input_windows"] == 1
    assert report["aggregated_motor_labels"]["exact_hold_windows"] == 1
    assert report["aggregated_motor_labels"][
        "supervised_execution_mismatch_windows"
    ] == 0
    assert report["aggregated_motor_labels"][
        "trajectory_execution_contract_satisfied"
    ] is True
    assert report["action_supervision"] == {
        "mask": "supervision_mask",
        "supervised_decisions": 1,
        "unsupervised_context_decisions": 1,
        "trajectory_decisions": 2,
        "native_trajectory_advancement": (
            "all recorded framewise replay inputs in each three-frame window"
        ),
        "label_execution_contract": (
            "the labelled movement/speed action was executed on every native "
            "frame in its three-frame window"
        ),
        "mixed_window_context_action": (
            "terminal_native_action_with_supervision_disabled"
        ),
        "previous_action_contract": (
            "last native replay movement/speed input before the decision boundary"
        ),
    }


def test_shifted_phase_finds_exact_hold_without_future_input_leakage(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "phase.rep"
    _write_phase_replay(replay)
    client = _FakeClient(replay)
    config = ReplayDemonstrationConfig(decision_phase_offset=1)

    report, builder = collect_replay_demonstrations(
        client,  # type: ignore[arg-type]
        replay_path="phase.rep",
        replay_file=replay,
        config=config,
    )

    demonstrations = builder.build()
    assert report["strict_success"] is True
    assert report["frames_consumed"] == 7
    assert report["decision_count"] == 1
    assert demonstrations.actions[:, 0].tolist() == [14]
    assert demonstrations.previous_actions[:, 0].tolist() == [12]
    assert demonstrations.supervision_mask[:, 0].tolist() == [True]
    assert report["decision_phase"] == {
        "offset_frames": 1,
        "interval_frames": 3,
        "valid_offset_range": [0, 2],
        "prefix_frames_advanced": 1,
        "tail_outcome_frames_advanced": 2,
        "incomplete_terminal_window_frames": 0,
        "first_control_episode_frame": 2,
        "last_control_episode_frame": 2,
        "sampling_only": True,
        "model_input_fields_added": [],
        "offset_is_model_input": False,
    }
    assert report["aggregated_motor_labels"]["decision_phase_offset"] == 1
    assert report["aggregated_motor_labels"]["action_frame_range"] == {
        "first": 3,
        "last": 5,
    }
    assert report["causal_contract"][
        "decision_phase_offset_is_model_input"
    ] is False
    assert report["causal_contract"][
        "decision_phase_offset_changes_sampling_boundaries_only"
    ] is True
    assert report["causal_contract"][
        "phase_prefix_frames_are_past_at_first_model_input"
    ] is True
    phase = report["decision_phase"]
    assert report["frames_consumed"] == (
        1
        + phase["prefix_frames_advanced"]
        + report["decision_count"] * phase["interval_frames"]
        + phase["incomplete_terminal_window_frames"]
        + phase["tail_outcome_frames_advanced"]
    )
    assert report["source_frame_range"]["first"] <= phase[
        "first_control_episode_frame"
    ]

    expected_stream = EngineStreamVision(config.vision)
    expected_stream.reset(_observation(1))
    expected_stream.push(_observation(2))
    expected_visible = expected_stream.observe()
    np.testing.assert_array_equal(
        demonstrations.global_frames[0],
        expected_visible.global_frames,
    )
    np.testing.assert_array_equal(
        demonstrations.local_frames[0],
        expected_visible.local_frames,
    )
    assert len(client.actions) == 6


@pytest.mark.parametrize(
    ("offset", "masks", "first_control", "last_control", "action_range", "tail"),
    (
        (0, [False, False], 1, 4, {"first": 2, "last": 7}, 0),
        (2, [False], 3, 3, {"first": 4, "last": 6}, 1),
    ),
)
def test_other_sampling_phases_keep_mixed_windows_unsupervised(
    tmp_path: Path,
    offset: int,
    masks: list[bool],
    first_control: int,
    last_control: int,
    action_range: dict[str, int],
    tail: int,
) -> None:
    replay = tmp_path / f"phase-{offset}.rep"
    _write_phase_replay(replay)

    report, builder = collect_replay_demonstrations(
        _FakeClient(replay),  # type: ignore[arg-type]
        replay_path=f"phase-{offset}.rep",
        replay_file=replay,
        config=ReplayDemonstrationConfig(decision_phase_offset=offset),
    )

    demonstrations = builder.build()
    assert demonstrations.supervision_mask[:, 0].tolist() == masks
    phase = report["decision_phase"]
    assert phase["prefix_frames_advanced"] == offset
    assert phase["tail_outcome_frames_advanced"] == tail
    assert phase["first_control_episode_frame"] == first_control
    assert phase["last_control_episode_frame"] == last_control
    assert report["aggregated_motor_labels"]["action_frame_range"] == action_range
    assert report["frames_consumed"] == (
        1
        + phase["prefix_frames_advanced"]
        + report["decision_count"] * phase["interval_frames"]
        + phase["incomplete_terminal_window_frames"]
        + phase["tail_outcome_frames_advanced"]
    )


@pytest.mark.parametrize(
    ("tail_spell", "tail_shoot_gap", "raw_field", "expected_value"),
    (
        (True, False, "spell_frames", 1),
        (False, True, "continuous_shoot", False),
    ),
)
def test_shifted_phase_tail_still_participates_in_strict_acceptance(
    tmp_path: Path,
    tail_spell: bool,
    tail_shoot_gap: bool,
    raw_field: str,
    expected_value: object,
) -> None:
    replay = tmp_path / "phase-assisted.rep"
    _write_phase_replay(
        replay,
        tail_spell=tail_spell,
        tail_shoot_gap=tail_shoot_gap,
    )

    report, builder = collect_replay_demonstrations(
        _FakeClient(replay),  # type: ignore[arg-type]
        replay_path="phase-assisted.rep",
        replay_file=replay,
        config=ReplayDemonstrationConfig(decision_phase_offset=1),
    )

    assert report["frames_consumed"] == 7
    assert report["decision_phase"]["tail_outcome_frames_advanced"] == 2
    assert report["raw_input"][raw_field] == expected_value
    assert report["strict_success"] is False
    assert builder.accepted_count == 0


def test_legacy_first_projection_is_explicitly_audited(tmp_path: Path) -> None:
    replay = tmp_path / "legacy.rep"
    _write_mixed_replay(replay)

    report, builder = collect_replay_demonstrations(
        _FakeClient(replay),  # type: ignore[arg-type]
        replay_path="legacy.rep",
        replay_file=replay,
        config=ReplayDemonstrationConfig(action_projection="first"),
    )

    demonstrations = builder.build()
    assert demonstrations.actions[:, 0].tolist() == [12, 12]
    assert demonstrations.previous_actions[:, 0].tolist() == [13, 14]
    assert demonstrations.supervision_mask[:, 0].tolist() == [True, True]
    assert report["aggregated_motor_labels"][
        "supervised_execution_mismatch_windows"
    ] == 1
    assert report["aggregated_motor_labels"][
        "trajectory_execution_contract_satisfied"
    ] is False
    assert report["action_supervision"]["unsupervised_context_decisions"] == 0


def test_cli_defaults_to_exact_hold_and_keeps_projection_alias() -> None:
    parser = build_parser()
    base = [
        "engine-replay-collect-demos",
        "--replay", "slot3.rep",
        "--save-demos", "human.npz",
    ]
    assert parser.parse_args(base).action_projection == "exact-hold"
    assert parser.parse_args(base).decision_phase_offset == 0
    assert parser.parse_args(
        [*base, "--decision-phase-offset", "1"],
    ).decision_phase_offset == 1
    assert parser.parse_args(
        [*base, "--decision-phase-offset", "2"],
    ).decision_phase_offset == 2
    with pytest.raises(SystemExit):
        parser.parse_args([*base, "--decision-phase-offset", "3"])
    assert parser.parse_args(
        [*base, "--action-projection", "modal"],
    ).action_projection == "modal"
    assert parser.parse_args(
        [*base, "--action-aggregation", "first"],
    ).action_projection == "first"


def test_rejects_hit_or_spell_assisted_replay(tmp_path: Path) -> None:
    hit_replay = tmp_path / "hit.rep"
    _write_replay(hit_replay)
    hit_report, hit_builder = collect_replay_demonstrations(
        _FakeClient(hit_replay, reason="player_hit"),  # type: ignore[arg-type]
        replay_path="hit.rep",
        replay_file=hit_replay,
    )
    assert hit_report["strict_success"] is False
    assert hit_builder.accepted_count == 0

    spell_replay = tmp_path / "spell.rep"
    _write_replay(spell_replay, spell=True)
    spell_report, spell_builder = collect_replay_demonstrations(
        _FakeClient(spell_replay),  # type: ignore[arg-type]
        replay_path="spell.rep",
        replay_file=spell_replay,
    )
    assert spell_report["termination_reason"] == "attack_complete"
    assert spell_report["raw_input"]["spell_frames"] == 1
    assert spell_report["strict_success"] is False
    assert spell_builder.accepted_count == 0

    special_replay = tmp_path / "special.rep"
    _write_replay(special_replay, special=True)
    special_report, special_builder = collect_replay_demonstrations(
        _FakeClient(special_replay),  # type: ignore[arg-type]
        replay_path="special.rep",
        replay_file=special_replay,
    )
    assert special_report["raw_input"]["special_frames"] == 1
    assert special_report["strict_success"] is False
    assert special_builder.accepted_count == 0

    shoot_gap_replay = tmp_path / "shoot-gap.rep"
    _write_replay(shoot_gap_replay, shoot_gap=True)
    shoot_report, shoot_builder = collect_replay_demonstrations(
        _FakeClient(shoot_gap_replay),  # type: ignore[arg-type]
        replay_path="shoot-gap.rep",
        replay_file=shoot_gap_replay,
    )
    assert shoot_report["raw_input"]["continuous_shoot"] is False
    assert shoot_report["strict_success"] is False
    assert shoot_builder.accepted_count == 0


def test_rejects_native_replay_frame_drift(tmp_path: Path) -> None:
    replay = tmp_path / "drift.rep"
    _write_replay(replay)

    class DriftClient(_FakeClient):
        def step(self, action: Action, *, repeat: int = 1) -> dict[str, Any]:
            response = super().step(action, repeat=repeat)
            response["observation"]["episode_frame"] += 1
            return response

    with pytest.raises(EngineProtocolError, match="exactly one logical input frame"):
        collect_replay_demonstrations(
            DriftClient(replay),  # type: ignore[arg-type]
            replay_path="drift.rep",
            replay_file=replay,
        )


def test_rejects_replay_byte_exhaustion_without_native_termination(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "unterminated.rep"
    _write_replay(replay)

    class UnterminatedClient(_FakeClient):
        def step(self, action: Action, *, repeat: int = 1) -> dict[str, Any]:
            response = super().step(action, repeat=repeat)
            response["observation"]["terminated"] = False
            response["observation"]["termination_reason"] = None
            return response

    with pytest.raises(EngineProtocolError, match="without termination"):
        collect_replay_demonstrations(
            UnterminatedClient(replay),  # type: ignore[arg-type]
            replay_path="unterminated.rep",
            replay_file=replay,
        )
