from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from stg_lab.engine_mpc import (
    EngineMPC,
    MPCConfig,
    MPCDecision,
    RegionDynamicsMemory,
    load_region_dynamics_memory,
)


_BOUNDS = (-200.0, 200.0, -240.0, 256.0)
_ROW_X = (-96.0, -48.0, 0.0, 48.0, 96.0)
_ROW_Y = (-200.0, -120.0, -63.0, 0.0)


def _learned_region_dynamics() -> RegionDynamicsMemory:
    return RegionDynamicsMemory(
        minimum_radius=7.0,
        maximum_radius=28.0,
        growth_rate=0.7,
        contraction_rate=0.7,
        expanding_frames=30.0,
        maximum_hold_frames=30.0,
        contracting_frames=30.0,
        minimum_hold_frames=90.0,
        cycle_frames=180.0,
    )


def _radius(relative_frame: int) -> float:
    phase = relative_frame % 180
    if phase < 30:
        return 7.0 + 0.7 * phase
    if phase < 60:
        return 28.0
    if phase < 90:
        return 28.0 - 0.7 * (phase - 60)
    return 7.0


def _region_observation(frame: int, radius: float) -> dict[str, Any]:
    left, right, bottom, top = _BOUNDS
    walls = [
        {
            "id": row * 100 + column,
            "x": x,
            "y": y,
            "a": radius,
            "b": radius,
            "dx": 0.0,
            "dy": 0.0,
            "collidable": True,
        }
        for row, y in enumerate(_ROW_Y)
        for column, x in enumerate(_ROW_X)
    ]
    return {
        "episode_frame": frame,
        "world": {"pl": left, "pr": right, "pb": bottom, "pt": top},
        "player": {
            "x": -24.0,
            "y": -160.0,
            "a": 0.5,
            "b": 0.5,
            "dx": 0.0,
            "dy": 0.0,
            "hspeed": 4.0,
            "lspeed": 2.0,
        },
        "enemy_bullets": [],
        "enemies": [],
        "nontjt_enemies": [],
        "indestructibles": walls,
    }


def _config() -> MPCConfig:
    return MPCConfig(
        observation_delay=0,
        horizon_frames=36,
        beam_width=16,
        region_beam_width=64,
        region_dynamics_memory=_learned_region_dynamics(),
    )


def _assert_episode_state_is_clear(controller: EngineMPC) -> None:
    phase = controller._region_phase
    topology = controller._region_topology

    assert phase.phase == "unknown"
    assert phase.phase_started_frame is None
    assert phase.last_frame is None
    assert phase.history == []
    assert phase.expansion_starts == []
    assert phase.phase_starts == {}
    assert phase.phase_durations == {}
    assert topology.next_row_identity == 1
    assert topology.row_tracks == {}
    assert topology.target_component is None
    assert topology.portal is None
    assert topology.revision == 0
    assert controller._committed_plan == ()
    assert controller._committed_plan_is_region is False
    assert controller._committed_plan_evacuating is False
    assert controller._committed_plan_key is None
    assert controller.estimator.last_frame is None


def test_reset_clears_episode_phase_topology_and_committed_actions_twice() -> None:
    config = _config()
    reused = EngineMPC(config)
    first_observation = _region_observation(100, 7.0)
    next_observation = _region_observation(103, 9.1)

    reused.select(first_observation)
    for _episode in range(2):
        reused.select(next_observation)

        assert reused._region_phase.history
        assert reused._region_phase.phase_starts
        assert reused._region_topology.next_row_identity > 1
        assert reused._region_topology.row_tracks
        assert reused._region_topology.target_component is not None
        assert reused._committed_plan

        reused.reset()
        _assert_episode_state_is_clear(reused)

        restarted = reused.select(first_observation)
        fresh = EngineMPC(config).select(first_observation)
        assert restarted == fresh


def _relative_decision_contract(decision: MPCDecision) -> dict[str, Any]:
    phase_age = (
        None
        if decision.region_phase_started_frame is None
        else decision.source_frame - decision.region_phase_started_frame
    )
    return {
        "action": decision.action.discrete,
        "planned_actions": tuple(action.discrete for action in decision.planned_actions),
        "region_anchor": decision.region_anchor,
        "region_crossing": decision.region_crossing,
        "region_path_margin": decision.region_path_margin,
        "region_evacuating": decision.region_evacuating,
        "region_target_rows_ahead": decision.region_target_rows_ahead,
        "region_navigation_mode": decision.region_navigation_mode,
        "region_current_component": decision.region_current_component,
        "region_target_component": decision.region_target_component,
        "region_portal": decision.region_portal,
        "region_deadline_slack": decision.region_deadline_slack,
        "using_committed_plan": decision.using_committed_plan,
        "committed_plan_immediate_margin": decision.committed_plan_immediate_margin,
        "committed_plan_current_horizon_margin": (
            decision.committed_plan_current_horizon_margin
        ),
        "region_phase": decision.region_phase,
        "region_phase_age": phase_age,
        "region_learned_cycle_frames": decision.region_learned_cycle_frames,
        "region_frames_until_expansion": decision.region_frames_until_expansion,
        "region_observed_radius": decision.region_observed_radius,
    }


def _run_region_timeline(frame_shift: int) -> list[dict[str, Any]]:
    controller = EngineMPC(_config())
    decisions: list[dict[str, Any]] = []

    for relative_frame in range(0, 178, 3):
        current = _region_observation(
            500 + frame_shift + relative_frame,
            _radius(relative_frame),
        )
        decisions.append(_relative_decision_contract(controller.select(current)))
    return decisions


def test_controller_contract_is_invariant_to_episode_frame_translation() -> None:
    original = _run_region_timeline(0)
    shifted = _run_region_timeline(50_000)

    assert len(original) == len(shifted) == 60
    assert original == shifted
    assert {decision["region_phase"] for decision in original} >= {
        "expanding",
        "maximum_hold",
        "contracting",
        "minimum_hold",
    }
    assert all(decision["region_current_component"] for decision in original)
    assert all(decision["region_target_component"] for decision in original)


@pytest.mark.parametrize(
    "route_artifact",
    [
        {
            "schema_version": 1,
            "route_id": "sr-stage5-boss3-visible-v2",
            "scenario": "stage5_boss3:lunatic",
            "cue": {
                "kind": "semantic_roi_mass",
                "channel": 0,
                "minimum_mass": 0.5,
                "roi": [-192.0, 192.0, 80.0, 224.0],
            },
            "trigger_lead": 0,
            "decision_interval": 3,
            "actions": [{"move_x": 1, "shoot": True}],
            "source": {"kind": "standalone_simulation"},
            "generator": "experiments/build_route_memory_v2.py",
            "generator_sha256": "0" * 64,
        },
        {
            "schema_version": 1,
            "library_id": "sr-stage5-boss4-visible-v2",
            "scenario": "stage5_boss4:lunatic",
            "memory_ids": [1, 2, 3],
            "source": {"kind": "native_route_capture"},
        },
    ],
    ids=("simulated-single-route", "native-route-library"),
)
def test_region_memory_loader_rejects_complete_route_artifact_schemas(
    tmp_path: Path,
    route_artifact: dict[str, Any],
) -> None:
    path = tmp_path / "route.json"
    path.write_text(json.dumps(route_artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported top-level fields"):
        load_region_dynamics_memory(path)
