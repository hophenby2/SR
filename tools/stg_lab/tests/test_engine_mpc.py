from __future__ import annotations

from dataclasses import replace
import json
import math
from typing import Any

import numpy as np
import pytest

from stg_lab.engine_mpc import (
    CandidateEvaluation,
    EngineMPC,
    MPCConfig,
    PredictedThreat,
    RegionDynamicsMemory,
    VisibleTrackEstimator,
    _GapCorridor,
    _RegionAnchor,
    movement_actions,
)


def observation(
    frame: int,
    *,
    player_x: float = 0.0,
    player_y: float = 0.0,
    bullets: list[dict[str, Any]] | None = None,
    enemies: list[dict[str, Any]] | None = None,
    indestructibles: list[dict[str, Any]] | None = None,
    lasers: list[dict[str, Any]] | None = None,
    bounds: tuple[float, float, float, float] = (-100.0, 100.0, -100.0, 100.0),
) -> dict[str, Any]:
    left, right, bottom, top = bounds
    return {
        "episode_frame": frame,
        "world": {
            "pl": left,
            "pr": right,
            "pb": bottom,
            "pt": top,
        },
        "player": {
            "x": player_x,
            "y": player_y,
            "a": 0.5,
            "b": 0.5,
            "hspeed": 4.0,
            "lspeed": 2.0,
        },
        "enemy_bullets": list(bullets or []),
        "enemies": list(enemies or []),
        "nontjt_enemies": [],
        "indestructibles": list(indestructibles or []),
        "lasers": list(lasers or []),
    }


def bullet(object_id: int, x: float, y: float, **values: Any) -> dict[str, Any]:
    return {
        "id": object_id,
        "x": x,
        "y": y,
        "a": 2.0,
        "b": 2.0,
        "collidable": True,
        **values,
    }


def wall_object(
    object_id: int,
    x: float,
    y: float,
    *,
    radius: float,
    dx: float = -1.5,
    dy: float = -2.6,
) -> dict[str, Any]:
    return {
        "id": object_id,
        "x": x,
        "y": y,
        "a": radius,
        "b": radius,
        "dx": dx,
        "dy": dy,
        "collidable": True,
    }


def evaluation(decision, *, move_x: int, move_y: int, slow: bool):
    return next(
        item for item in decision.evaluations
        if item.action.move_x == move_x
        and item.action.move_y == move_y
        and item.action.slow is slow
    )


def test_action_set_has_17_unique_three_frame_movement_choices() -> None:
    actions = movement_actions()
    assert len(actions) == 17
    assert len({(action.move_x, action.move_y, action.slow) for action in actions}) == 17
    assert all(action.spell is False for action in actions)


def test_linear_bullet_prediction_avoids_neutral_collision() -> None:
    teacher = EngineMPC(MPCConfig(observation_delay=0, horizon_frames=36))
    decision = teacher.select(observation(
        0,
        bullets=[bullet(1, 0.0, 24.0, dx=0.0, dy=-2.0)],
    ))

    neutral = evaluation(decision, move_x=0, move_y=0, slow=True)
    selected = next(item for item in decision.evaluations if item.action == decision.action)
    assert neutral.collided is True
    assert selected.collided is False
    assert (decision.action.move_x, decision.action.move_y) != (0, 0)
    assert decision.action.spell is False
    assert decision.threats[0].at(12)[:2] == (0.0, 0.0)


def test_controller_overlay_state_uses_live_decision_and_phase_envelope() -> None:
    teacher = EngineMPC(MPCConfig(observation_delay=5, horizon_frames=36))
    observed = observation(
        12,
        bullets=[bullet(7, 20.0, 40.0, dx=1.5, dy=-2.0)],
    )
    decision = replace(
        teacher.select(observed),
        region_anchor=(24.0, -36.0),
    )
    radius_after_calls: list[int] = []

    def radius_after(future_frame: int) -> float | None:
        radius_after_calls.append(future_frame)
        return None if future_frame == 8 else 7.0 + future_frame * 0.1

    teacher._region_phase.radius_after = radius_after  # type: ignore[method-assign]
    state = teacher.controller_overlay_state(decision, observed)

    assert state["schema_version"] == 1
    assert state["revision"] == state["source_frame"] == decision.source_frame
    assert state["horizon_frames"] == 36
    assert state["future_start"] == 1
    assert state["danger_margin"] == teacher.config.danger_margin_target
    assert state["safe_margin"] == teacher.config.safe_margin_target
    assert state["region_safe_margin"] == teacher.config.region_safe_margin_target
    assert state["region_navigation_active"] is True
    assert state["player_radius"] == 0.5
    assert state["bounds"] == {
        "left": -92.0,
        "right": 92.0,
        "bottom": -84.0,
        "top": 68.0,
    }
    assert radius_after_calls == list(range(6, 42))
    assert len(state["region_phase_radii"]) == 36
    assert state["region_phase_radii"][2] is None
    assert state["threats"] == [{
        "key": decision.threats[0].key,
        "source": decision.threats[0].source,
        "x": decision.threats[0].x,
        "y": decision.threats[0].y,
        "vx": decision.threats[0].vx,
        "vy": decision.threats[0].vy,
        "radius": decision.threats[0].radius,
        "radius_rate": decision.threats[0].radius_rate,
        "radius_rate_horizon": decision.threats[0].radius_rate_horizon,
        "motion_horizon": decision.threats[0].motion_horizon,
    }]
    json.dumps(state, allow_nan=False)


def test_straight_laser_is_covered_without_duplicate_object_threat() -> None:
    laser = {
        "id": 8,
        "kind": "straight_laser",
        "x": -30.0,
        "y": 0.0,
        "rot": 0.0,
        "l1": 10.0,
        "l2": 40.0,
        "l3": 10.0,
        "w": 8.0,
        "collidable": True,
    }
    current = observation(0, bullets=[laser], lasers=[laser])
    estimator = VisibleTrackEstimator(MPCConfig(observation_delay=0))
    threats = estimator.update(current)

    assert threats
    assert {threat.source for threat in threats} == {"lasers"}
    assert all(threat.key.startswith("lasers:8:straight:") for threat in threats)
    assert min(threat.x - threat.radius for threat in threats) <= -30.0
    assert max(threat.x + threat.radius for threat in threats) >= 30.0

    teacher = EngineMPC(MPCConfig(observation_delay=0, horizon_frames=36))
    decision = teacher.select(current)
    neutral = evaluation(decision, move_x=0, move_y=0, slow=True)
    assert neutral.collided is True


def test_screen_length_laser_uses_bounded_conservative_circle_cover() -> None:
    laser = {
        "id": 9,
        "kind": "straight_laser",
        "x": -320.0,
        "y": 0.0,
        "rot": 0.0,
        "l1": 0.0,
        "l2": 640.0,
        "l3": 0.0,
        "w": 8.0,
        "collidable": True,
    }
    threats = VisibleTrackEstimator(MPCConfig(observation_delay=0)).update(
        observation(0, lasers=[laser]),
    )

    assert len(threats) == 20
    assert min(threat.x - threat.radius for threat in threats) <= -320.0
    assert max(threat.x + threat.radius for threat in threats) >= 320.0


def test_bent_laser_segments_and_visible_motion_are_tracked() -> None:
    def bent(offset: float) -> dict[str, Any]:
        return {
            "id": 12,
            "kind": "bent_laser",
            "x": offset,
            "y": 0.0,
            "w": 6.0,
            "collidable": True,
            "points": [
                {"slot": 1, "x": -20.0 + offset, "y": -10.0},
                {"slot": 2, "x": offset, "y": 0.0},
                {"slot": 3, "x": 20.0 + offset, "y": 10.0},
            ],
        }

    estimator = VisibleTrackEstimator(MPCConfig(observation_delay=0))
    first = estimator.update(observation(0, lasers=[bent(0.0)]))
    shifted = estimator.update(observation(3, lasers=[bent(6.0)]))

    assert first and shifted
    assert {threat.source for threat in shifted} == {"lasers"}
    assert all(threat.vx == pytest.approx(2.0) for threat in shifted)
    assert all(threat.vy == pytest.approx(0.0) for threat in shifted)


def test_visible_warning_hazards_are_kept_through_observation_delay() -> None:
    warning_bullet = bullet(15, 0.0, 8.0, collidable=False, dx=0.0, dy=0.0)
    warning_laser = {
        "id": 16,
        "kind": "straight_laser",
        "x": -20.0,
        "y": 0.0,
        "rot": 0.0,
        "l1": 0.0,
        "l2": 40.0,
        "l3": 0.0,
        "w": 4.0,
        "collidable": False,
    }
    threats = VisibleTrackEstimator(MPCConfig(observation_delay=5)).update(
        observation(0, bullets=[warning_bullet, warning_laser], lasers=[warning_laser]),
    )

    assert any(threat.key == "enemy_bullets:15" for threat in threats)
    assert any(threat.key.startswith("lasers:16:straight:") for threat in threats)
    assert all(threat.source != "indestructibles" for threat in threats)


def test_rotating_laser_uses_sample_motion_instead_of_object_center_dx() -> None:
    def rotating(angle: float) -> dict[str, Any]:
        return {
            "id": 18,
            "kind": "straight_laser",
            "x": 0.0,
            "y": 0.0,
            "dx": 0.0,
            "dy": 0.0,
            "rot": angle,
            "l1": 0.0,
            "l2": 40.0,
            "l3": 0.0,
            "w": 8.0,
            "collidable": True,
        }

    estimator = VisibleTrackEstimator(MPCConfig(observation_delay=0))
    estimator.update(observation(0, lasers=[rotating(0.0)]))
    threats = estimator.update(observation(3, lasers=[rotating(9.0)]))

    assert threats
    assert any(abs(threat.vy) > 0.1 for threat in threats)
    assert len({round(threat.vy, 3) for threat in threats}) > 1


def test_visible_displacement_and_growing_radius_are_extrapolated_by_id() -> None:
    estimator = VisibleTrackEstimator(MPCConfig(observation_delay=0))
    estimator.update(observation(0, bullets=[bullet(9, -20.0, 0.0, a=2.0, b=2.0)]))
    threats = estimator.update(observation(
        3,
        bullets=[bullet(9, -8.0, 0.0, a=8.0, b=8.0)],
    ))
    threat = threats[0]
    assert threat.key == "enemy_bullets:9"
    assert threat.vx == 4.0
    assert threat.vy == 0.0
    assert threat.radius_rate == 2.0
    # Radius velocity is deliberately extrapolated only over a short window;
    # scripted bullets frequently stop expanding before the motion horizon.
    assert threat.at(10) == (32.0, 0.0, 20.0)

    teacher = EngineMPC(MPCConfig(observation_delay=0))
    teacher.select(observation(0, bullets=[bullet(9, 18.0, 0.0, a=2.0, b=2.0)]))
    decision = teacher.select(observation(
        3,
        bullets=[bullet(9, 18.0, 0.0, a=8.0, b=8.0)],
    ))
    neutral = evaluation(decision, move_x=0, move_y=0, slow=True)
    assert neutral.collided is True
    assert neutral.earliest_collision_frame is not None


def test_nuke_radius_uses_the_learned_maximum_envelope_at_float_oscillation() -> None:
    config = MPCConfig(
        observation_delay=0,
        region_dynamics_memory=learned_region_dynamics(),
    )
    estimator = VisibleTrackEstimator(config)
    estimator.update(observation(
        0,
        indestructibles=[
            wall_object(9, 0.0, 0.0, radius=28.0, dx=0.0, dy=-3.0),
        ],
    ))
    dipped = estimator.update(observation(
        3,
        indestructibles=[
            wall_object(9, 0.0, -9.0, radius=27.3, dx=0.0, dy=-3.0),
        ],
    ))[0]

    assert dipped.radius_rate == 0.0
    assert dipped.radius_rate_horizon == 0
    assert dipped.at(3)[2] == 28.0

    rising = VisibleTrackEstimator(config)
    rising.update(observation(
        0,
        indestructibles=[
            wall_object(9, 0.0, 0.0, radius=27.3, dx=0.0, dy=-3.0),
        ],
    ))
    maximum = rising.update(observation(
        3,
        indestructibles=[
            wall_object(9, 0.0, -9.0, radius=28.0, dx=0.0, dy=-3.0),
        ],
    ))[0]
    assert maximum.radius_rate == 0.0
    assert maximum.at(60)[2] == 28.0


def test_nuke_minimum_float_oscillation_does_not_reset_growth_phase() -> None:
    estimator = VisibleTrackEstimator(MPCConfig(
        observation_delay=0,
        region_dynamics_memory=learned_region_dynamics(),
    ))
    estimator.update(observation(
        0,
        indestructibles=[
            wall_object(9, 0.0, 0.0, radius=6.3, dx=0.0, dy=-3.0),
        ],
    ))
    plateau = estimator.update(observation(
        3,
        indestructibles=[
            wall_object(9, 0.0, -9.0, radius=7.7, dx=0.0, dy=-3.0),
        ],
    ))[0]
    assert plateau.radius == 7.7
    assert plateau.radius_rate == 0.0

    growth = estimator.update(observation(
        6,
        indestructibles=[
            wall_object(9, 0.0, -18.0, radius=8.4, dx=0.0, dy=-3.0),
        ],
    ))[0]
    assert growth.radius_rate == 0.7
    assert growth.at(3)[2] == 10.5


def test_generic_indestructible_radius_uses_only_observed_motion() -> None:
    estimator = VisibleTrackEstimator(MPCConfig(observation_delay=0))
    estimator.update(observation(
        0,
        indestructibles=[
            wall_object(9, 0.0, 0.0, radius=27.3, dx=0.0, dy=-3.0),
        ],
    ))
    threat = estimator.update(observation(
        3,
        indestructibles=[
            wall_object(9, 0.0, -9.0, radius=28.0, dx=0.0, dy=-3.0),
        ],
    ))[0]

    assert threat.radius_rate == pytest.approx(0.7 / 3.0)
    assert threat.radius_rate_horizon == 6


def test_dx_and_observation_delay_compensate_to_estimated_current_position() -> None:
    estimator = VisibleTrackEstimator(MPCConfig(observation_delay=5))
    threat = estimator.update(observation(
        10,
        bullets=[bullet(4, -20.0, 5.0, dx=4.0, dy=-1.0)],
    ))[0]
    assert threat.x == 0.0
    assert threat.y == 0.0
    assert threat.observation_delay == 5

    displacement = VisibleTrackEstimator(MPCConfig(observation_delay=5))
    displacement.update(observation(0, bullets=[bullet(5, -20.0, 0.0)]))
    predicted = displacement.update(observation(3, bullets=[bullet(5, -8.0, 0.0)]))[0]
    assert predicted.vx == 4.0
    assert predicted.x == 12.0


def test_reappearing_reused_id_starts_a_fresh_visible_track() -> None:
    estimator = VisibleTrackEstimator(MPCConfig(observation_delay=0))
    estimator.update(observation(
        0,
        bullets=[bullet(5, -100.0, 0.0, dx=4.0, dy=0.0)],
    ))
    estimator.update(observation(3))
    threat = estimator.update(observation(
        6,
        bullets=[bullet(5, 50.0, 0.0, dx=1.0, dy=-2.0)],
    ))[0]

    assert threat.vx == 1.0
    assert threat.vy == -2.0
    assert threat.radius_rate == 0.0


def test_immediately_reused_id_rejects_inconsistent_cross_frame_motion() -> None:
    estimator = VisibleTrackEstimator(MPCConfig(observation_delay=0))
    estimator.update(observation(
        0,
        bullets=[bullet(5, -100.0, 0.0, dx=4.0, dy=0.0)],
    ))
    threat = estimator.update(observation(
        3,
        player_y=-176.0,
        bullets=[bullet(5, 50.0, 0.0, dx=-0.2, dy=-4.0)],
    ))[0]

    assert threat.vx == -0.2
    assert threat.vy == -4.0
    assert threat.radius_rate == 0.0


def test_immediately_reused_indestructible_id_rejects_inconsistent_motion() -> None:
    estimator = VisibleTrackEstimator(MPCConfig(observation_delay=0))
    estimator.update(observation(
        0,
        indestructibles=[
            wall_object(9, -100.0, 0.0, radius=7.0, dx=4.0, dy=0.0),
        ],
    ))
    threat = estimator.update(observation(
        3,
        indestructibles=[
            wall_object(9, 50.0, 0.0, radius=7.0, dx=-0.2, dy=-4.0),
        ],
    ))[0]

    assert threat.key == "indestructibles:9"
    assert threat.vx == -0.2
    assert threat.vy == -4.0
    assert threat.radius_rate == 0.0


def test_boundary_is_preferred_over_safe_boss_alignment() -> None:
    teacher = EngineMPC(MPCConfig(observation_delay=0))
    boss = {
        "id": 30,
        "x": 18.0,
        "y": 15.0,
        "maxhp": 1000.0,
        "collidable": False,
    }
    decision = teacher.select(observation(
        0,
        player_x=18.0,
        enemies=[boss],
        bounds=(-20.0, 20.0, -20.0, 20.0),
    ))
    assert decision.action.move_x == -1
    assert decision.action.spell is False


def test_teacher_holds_action_until_next_three_frame_decision() -> None:
    teacher = EngineMPC(MPCConfig(observation_delay=0))
    first = teacher.select({"observation": observation(0)})
    second = teacher.select(observation(1))
    third = teacher.select(observation(2))
    fourth = teacher.select(observation(3))
    assert first.recomputed is True
    assert second.recomputed is False
    assert third.recomputed is False
    assert fourth.recomputed is True
    assert second.action == first.action == third.action


def test_safe_direction_hysteresis_rejects_a_three_frame_target_flip() -> None:
    config = MPCConfig(
        observation_delay=0,
        preferred_y_fraction=0.5,
    )

    def with_boss(frame: int, boss_x: float) -> dict[str, Any]:
        return observation(
            frame,
            enemies=[{
                "id": 30,
                "x": boss_x,
                "y": 40.0,
                "maxhp": 1000.0,
                "collidable": False,
            }],
        )

    teacher = EngineMPC(config)
    first = teacher.select(with_boss(0, 80.0))
    held = teacher.select(with_boss(3, -80.0))
    no_aba = teacher.select(with_boss(6, 80.0))
    unconstrained = EngineMPC(config).select(with_boss(3, -80.0))

    assert (first.action.move_x, first.action.move_y) == (
        held.action.move_x,
        held.action.move_y,
    )
    assert (unconstrained.action.move_x, unconstrained.action.move_y) != (
        first.action.move_x,
        first.action.move_y,
    )
    assert held.recomputed is True
    assert (no_aba.action.move_x, no_aba.action.move_y) == (
        first.action.move_x,
        first.action.move_y,
    )
    assert held.action.spell is False


def test_region_navigation_only_breaks_hysteresis_for_urgent_progress() -> None:
    def anchor(x: float, mode: str, slack: float = 30.0) -> _RegionAnchor:
        return _RegionAnchor(
            x=x,
            y=0.0,
            crossing=mode == "evacuate",
            path_margin=20.0,
            evacuating=mode in {"preposition", "evacuate"},
            target_rows_ahead=1,
            navigation_mode=mode,
            current_component="band:0",
            target_component="exterior:right",
            portal="test",
            deadline_slack=slack,
        )

    preposition = EngineMPC(MPCConfig(
        observation_delay=0,
        preferred_y_fraction=0.5,
    ))
    preposition_anchors = iter((
        anchor(80.0, "preposition"),
        anchor(-80.0, "preposition"),
    ))
    preposition._region_anchor = lambda *_args: next(preposition_anchors)
    first = preposition.select(observation(0))
    held = preposition.select(observation(3))
    assert (held.action.move_x, held.action.move_y) == (
        first.action.move_x,
        first.action.move_y,
    )

    nonurgent = EngineMPC(MPCConfig(
        observation_delay=0,
        preferred_y_fraction=0.5,
    ))
    nonurgent_anchors = iter((
        anchor(80.0, "preposition"),
        anchor(-80.0, "evacuate"),
    ))
    nonurgent._region_anchor = lambda *_args: next(nonurgent_anchors)
    before = nonurgent.select(observation(0))
    held = nonurgent.select(observation(3))
    assert (held.action.move_x, held.action.move_y) == (
        before.action.move_x,
        before.action.move_y,
    )

    urgent = EngineMPC(MPCConfig(
        observation_delay=0,
        preferred_y_fraction=0.5,
    ))
    urgent_anchors = iter((
        anchor(80.0, "preposition"),
        anchor(-80.0, "evacuate", 0.0),
    ))
    urgent._region_anchor = lambda *_args: next(urgent_anchors)
    before = urgent.select(observation(0))
    escaped = urgent.select(observation(3))
    assert (escaped.action.move_x, escaped.action.move_y) != (
        before.action.move_x,
        before.action.move_y,
    )


def test_committed_plan_cannot_bypass_direction_hold() -> None:
    teacher = EngineMPC(MPCConfig())
    right = next(
        action for action in teacher.actions
        if (action.move_x, action.move_y, action.slow) == (1, 0, True)
    )
    left = next(
        action for action in teacher.actions
        if (action.move_x, action.move_y, action.slow) == (-1, 0, True)
    )
    teacher._last_action = right
    teacher._direction_started_frame = 100

    assert not teacher._committed_action_respects_direction_hold(
        left,
        right,
        103,
    )
    assert teacher._committed_action_respects_direction_hold(
        left,
        left,
        103,
    )
    assert not teacher._committed_action_respects_direction_hold(
        left,
        right,
        109,
    )
    assert teacher._committed_action_respects_direction_hold(
        left,
        right,
        112,
    )


def test_committed_old_direction_cannot_override_a_safety_release() -> None:
    teacher = EngineMPC(MPCConfig(
        observation_delay=0,
        preferred_y_fraction=0.5,
    ))

    def anchor(*_args: Any) -> _RegionAnchor:
        return _RegionAnchor(
            x=80.0,
            y=0.0,
            crossing=False,
            path_margin=20.0,
            evacuating=True,
            target_rows_ahead=1,
            navigation_mode="preposition",
            current_component="band:0",
            target_component="exterior:right",
            portal="test",
            deadline_slack=30.0,
        )

    teacher._region_anchor = anchor
    first = teacher.select(observation(0))
    committed = first.action
    replacement = next(
        action for action in teacher.actions
        if (action.move_x, action.move_y) == (
            -committed.move_x,
            -committed.move_y,
        )
        and action.slow == committed.slow
    )
    evaluations = tuple(
        replace(
            value,
            collided=False,
            collision_frames=0,
            earliest_collision_frame=None,
            minimum_margin=(
                8.0
                if value.action.discrete == committed.discrete else
                14.0
                if value.action.discrete == replacement.discrete else
                value.minimum_margin
            ),
        )
        for value in first.evaluations
    )
    proposed = replace(
        first,
        action=replacement,
        source_frame=3,
        evaluations=evaluations,
        planned_actions=(replacement,),
    )
    teacher._committed_plan = (committed,)
    teacher._committed_plan_is_region = True
    teacher._committed_plan_evacuating = proposed.region_evacuating
    teacher._committed_plan_key = (
        proposed.region_phase,
        proposed.region_phase_started_frame,
        proposed.region_navigation_mode,
        proposed.region_current_component,
        proposed.region_target_component,
        proposed.region_portal,
    )
    teacher._compute = lambda *_args: proposed

    selected = teacher.select(observation(3))

    assert selected.action.discrete == replacement.discrete
    assert selected.using_committed_plan is False


def test_beam_candidate_order_keeps_clearance_shortfall_ahead_of_smoothing() -> None:
    order = EngineMPC._beam_candidate_order(
        collided=np.asarray([False, False]),
        earliest_collision=np.asarray([61, 61]),
        collision_frames=np.asarray([0, 0]),
        danger_margin_shortfall=np.asarray([0.0, 0.0]),
        margin_shortfall=np.asarray([8.0, 0.0]),
        region_margin_shortfall=np.asarray([0.0, 0.0]),
        preference=np.asarray([0.0, 100.0]),
        minimum_margin=np.asarray([0.0, 8.0]),
    )

    assert order.tolist() == [1, 0]


def test_region_order_does_not_let_a_narrow_portal_hide_bullet_clearance() -> None:
    order = EngineMPC._beam_candidate_order(
        collided=np.asarray([False, False]),
        earliest_collision=np.asarray([61, 61]),
        collision_frames=np.asarray([0, 0]),
        danger_margin_shortfall=np.asarray([0.0, 18.0]),
        # Candidate zero stays outside the ordinary-danger reserve but passes
        # closer to a forced-region wall. Candidate one does the opposite.
        margin_shortfall=np.asarray([0.0, 18.0]),
        region_margin_shortfall=np.asarray([6.0, 0.0]),
        preference=np.asarray([100.0, 0.0]),
        minimum_margin=np.asarray([2.0, 8.0]),
    )

    assert order.tolist() == [0, 1]


def test_region_reserve_precedes_extra_clearance_outside_danger_zone() -> None:
    order = EngineMPC._beam_candidate_order(
        collided=np.asarray([False, False]),
        earliest_collision=np.asarray([61, 61]),
        collision_frames=np.asarray([0, 0]),
        danger_margin_shortfall=np.asarray([0.0, 0.0]),
        margin_shortfall=np.asarray([0.0, 2.0]),
        region_margin_shortfall=np.asarray([6.0, 0.0]),
        preference=np.asarray([0.0, 100.0]),
        minimum_margin=np.asarray([2.0, 8.0]),
    )

    assert order.tolist() == [1, 0]


def test_far_collision_timing_yields_to_escape_reserve() -> None:
    order = EngineMPC._beam_candidate_order(
        collided=np.asarray([True, True]),
        earliest_collision=np.asarray([60, 48]),
        collision_frames=np.asarray([1, 1]),
        danger_margin_shortfall=np.asarray([0.0, 0.0]),
        margin_shortfall=np.asarray([10.0, 0.0]),
        region_margin_shortfall=np.asarray([0.0, 0.0]),
        preference=np.asarray([0.0, 0.0]),
        minimum_margin=np.asarray([10.0, 8.0]),
        collision_priority_frames=36,
    )

    assert order.tolist() == [1, 0]


def test_near_collision_still_prioritizes_later_impact() -> None:
    order = EngineMPC._beam_candidate_order(
        collided=np.asarray([True, True]),
        earliest_collision=np.asarray([12, 24]),
        collision_frames=np.asarray([1, 4]),
        danger_margin_shortfall=np.asarray([0.0, 10.0]),
        margin_shortfall=np.asarray([0.0, 10.0]),
        region_margin_shortfall=np.asarray([0.0, 0.0]),
        preference=np.asarray([0.0, 100.0]),
        minimum_margin=np.asarray([10.0, 0.0]),
        collision_priority_frames=36,
    )

    assert order.tolist() == [1, 0]


def test_region_evaluation_reports_independent_threat_reserves() -> None:
    teacher = EngineMPC(MPCConfig(observation_delay=0, horizon_frames=36))
    decision = teacher.select(observation(
        0,
        player_x=0.0,
        player_y=-20.0,
        bullets=[bullet(1, 35.0, -20.0, dx=0.0, dy=0.0)],
        indestructibles=[
            wall_object(2, -35.0, -20.0, radius=7.0, dx=0.0, dy=0.0),
        ],
    ))
    selected = next(
        item for item in decision.evaluations
        if item.action.discrete == decision.action.discrete
    )

    assert math.isfinite(selected.minimum_nonregion_margin)
    assert math.isfinite(selected.minimum_region_margin)
    assert selected.minimum_margin == min(
        selected.minimum_nonregion_margin,
        selected.minimum_region_margin,
    )


def test_imminent_collision_interrupts_direction_hold_immediately() -> None:
    teacher = EngineMPC(MPCConfig(
        observation_delay=0,
        preferred_y_fraction=0.5,
    ))
    boss = {
        "id": 30,
        "x": 80.0,
        "y": 40.0,
        "maxhp": 1000.0,
        "collidable": False,
    }
    first = teacher.select(observation(0, enemies=[boss]))
    assert (first.action.move_x, first.action.move_y) == (1, -1)

    path_coordinate = 3.0 * math.sqrt(2.0)
    danger = bullet(
        31,
        path_coordinate,
        -path_coordinate,
        dx=0.0,
        dy=0.0,
    )
    boss["x"] = -80.0
    escaped = teacher.select(observation(3, enemies=[boss], bullets=[danger]))
    incumbent = next(
        item for item in escaped.evaluations
        if item.action.discrete == first.action.discrete
    )
    selected = next(
        item for item in escaped.evaluations
        if item.action.discrete == escaped.action.discrete
    )

    assert incumbent.collided is True
    assert incumbent.earliest_collision_frame == 2
    assert selected.collided is False
    assert escaped.action.discrete != first.action.discrete
    assert escaped.action.spell is False


def test_material_clearance_gain_can_interrupt_direction_hold() -> None:
    teacher = EngineMPC(MPCConfig(
        observation_delay=0,
        preferred_y_fraction=0.5,
        switch_margin_gain=6.0,
    ))
    boss = {
        "id": 30,
        "x": 80.0,
        "y": 40.0,
        "maxhp": 1000.0,
        "collidable": False,
    }
    first = teacher.select(observation(0, enemies=[boss]))
    boss["x"] = -80.0
    early = teacher.select(observation(
        3,
        enemies=[boss],
        bullets=[bullet(31, 10.0, 0.0, dx=0.0, dy=0.0)],
    ))
    assert early.action.discrete == first.action.discrete

    escaped = teacher.select(observation(
        9,
        enemies=[boss],
        bullets=[bullet(31, 10.0, 0.0, dx=0.0, dy=0.0)],
    ))
    incumbent = next(
        item for item in escaped.evaluations
        if item.action.discrete == first.action.discrete
    )
    selected = next(
        item for item in escaped.evaluations
        if item.action.discrete == escaped.action.discrete
    )

    assert incumbent.collided is False
    assert incumbent.minimum_margin > teacher.config.emergency_margin
    assert (
        selected.minimum_margin - incumbent.minimum_margin
        >= teacher.config.switch_margin_gain
    )
    assert escaped.action.discrete != first.action.discrete


def test_corner_escape_can_interrupt_direction_hold() -> None:
    teacher = EngineMPC(MPCConfig())
    inward = next(
        action for action in teacher.actions
        if (action.move_x, action.move_y, action.slow) == (1, 0, False)
    )
    trapped = next(
        action for action in teacher.actions
        if (action.move_x, action.move_y, action.slow) == (-1, 0, False)
    )
    teacher._last_action = trapped
    teacher._direction_started_frame = 100

    def candidate(action, corner_clearance: float) -> CandidateEvaluation:
        return CandidateEvaluation(
            action=action,
            collided=False,
            collision_frames=0,
            earliest_collision_frame=None,
            minimum_margin=20.0,
            boundary_penalty=0.0,
            boss_alignment=0.0,
            minimum_nonregion_margin=20.0,
            immediate_corner_clearance=corner_clearance,
        )

    selected = teacher._apply_direction_hold(
        0,
        (candidate(inward, 12.0), candidate(trapped, 0.0)),
        None,
        103,
    )

    assert selected == 0


def test_transition_penalty_distinguishes_switch_reverse_and_aba() -> None:
    teacher = EngineMPC(MPCConfig())
    right = next(
        action for action in teacher.actions
        if (action.move_x, action.move_y, action.slow) == (1, 0, True)
    )
    left = next(
        action for action in teacher.actions
        if (action.move_x, action.move_y, action.slow) == (-1, 0, True)
    )
    up = next(
        action for action in teacher.actions
        if (action.move_x, action.move_y, action.slow) == (0, 1, True)
    )

    ordinary_switch = teacher._transition_penalty(up, right, None)
    reverse = teacher._transition_penalty(left, right, None)
    aba_reverse = teacher._transition_penalty(left, right, left)

    assert ordinary_switch == teacher.config.direction_switch_penalty
    assert reverse == (
        teacher.config.direction_switch_penalty
        + teacher.config.direction_reverse_penalty
    )
    assert aba_reverse == reverse + teacher.config.direction_aba_penalty


def test_clearance_reserve_prefers_more_distance_after_basic_safety() -> None:
    preferred_y_fraction = 84.0 / 152.0
    legacy = EngineMPC(MPCConfig(
        observation_delay=0,
        preferred_y_fraction=preferred_y_fraction,
        danger_margin_target=12.0,
        safe_margin_target=12.0,
        clearance_reward_weight=0.0,
        corner_reserve_weight=0.0,
        minimum_direction_hold_frames=0,
        direction_switch_penalty=0.0,
        direction_reverse_penalty=0.0,
        direction_aba_penalty=0.0,
        speed_switch_penalty=0.0,
    ))
    safer = EngineMPC(MPCConfig(
        observation_delay=0,
        preferred_y_fraction=preferred_y_fraction,
    ))
    scene = observation(
        0,
        bullets=[bullet(31, -50.0, -30.0, dx=0.0, dy=0.0)],
        enemies=[{
            "id": 30,
            "x": -80.0,
            "y": 40.0,
            "maxhp": 1000.0,
            "collidable": False,
        }],
    )

    legacy_decision = legacy.select(scene)
    safer_decision = safer.select(scene)
    legacy_evaluation = next(
        item for item in legacy_decision.evaluations
        if item.action.discrete == legacy_decision.action.discrete
    )
    safer_evaluation = next(
        item for item in safer_decision.evaluations
        if item.action.discrete == safer_decision.action.discrete
    )

    assert legacy_evaluation.collided is False
    assert safer_evaluation.collided is False
    assert safer_evaluation.minimum_margin >= legacy_evaluation.minimum_margin + 4.0
    assert safer_evaluation.minimum_margin >= safer.config.safe_margin_target


def test_region_planner_keeps_a_non_downward_wall_transition() -> None:
    lower_x = (-203.0, -155.0, -107.0, -59.0, -11.0)
    upper_x = (-186.0, -138.0, -90.0, -42.0, 6.0)
    walls = [
        *(wall_object(index, x, -199.0, radius=7.0)
          for index, x in enumerate(lower_x)),
        *(wall_object(100 + index, x, -140.0, radius=7.0)
          for index, x in enumerate(upper_x)),
    ]
    teacher = EngineMPC(MPCConfig(observation_delay=0, horizon_frames=60))
    decision = teacher.select(observation(
        10,
        player_x=-17.0,
        player_y=-160.0,
        indestructibles=walls,
        bounds=(-192.0, 192.0, -224.0, 224.0),
    ))

    assert decision.region_crossing is True
    assert decision.region_anchor is not None
    assert decision.action.move_y >= 0
    selected = next(
        item for item in decision.evaluations
        if item.action.discrete == decision.action.discrete
    )
    assert selected.collided is False


def test_region_beam_commits_its_next_segment_when_it_remains_locally_safe() -> None:
    lower_x = (-203.0, -155.0, -107.0, -59.0, -11.0)
    upper_x = (-186.0, -138.0, -90.0, -42.0, 6.0)
    walls = [
        *(wall_object(index, x, -199.0, radius=7.0, dx=0.0, dy=0.0)
          for index, x in enumerate(lower_x)),
        *(wall_object(100 + index, x, -140.0, radius=7.0, dx=0.0, dy=0.0)
          for index, x in enumerate(upper_x)),
    ]
    teacher = EngineMPC(MPCConfig(observation_delay=0, horizon_frames=60))
    first = teacher.select(observation(
        10,
        player_x=-17.0,
        player_y=-160.0,
        indestructibles=walls,
        bounds=(-192.0, 192.0, -224.0, 224.0),
    ))
    second = teacher.select(observation(
        13,
        player_x=-17.0,
        player_y=-160.0,
        indestructibles=walls,
        bounds=(-192.0, 192.0, -224.0, 224.0),
    ))

    assert len(first.planned_actions) == 20
    assert first.planned_actions[0].discrete == first.action.discrete
    assert second.using_committed_plan is True
    assert second.action.discrete == first.planned_actions[1].discrete
    assert second.committed_plan_immediate_margin is not None
    assert second.committed_plan_immediate_margin > 0.0


def test_48_unit_gap_reaches_its_portal_closure_boundary_at_radius_17_5() -> None:
    def walls(radius: float) -> list[dict[str, Any]]:
        return [
            wall_object(
                row * 100 + column,
                x,
                y,
                radius=radius,
                dx=0.0,
                dy=0.0,
            )
            for row, y in enumerate((-60.0, 0.0))
            for column, x in enumerate((-48.0, 0.0, 48.0))
        ]

    def decide(radius: float):
        teacher = EngineMPC(MPCConfig(
            observation_delay=0,
            horizon_frames=60,
            preferred_y_fraction=74.0 / 192.0,
            region_dynamics_memory=learned_region_dynamics(),
        ))
        return teacher.select(observation(
            10,
            player_x=-24.0,
            player_y=-30.0,
            indestructibles=walls(radius),
            bounds=(-73.0, 73.0, -120.0, 120.0),
        ))

    below = decide(17.4)
    at_boundary = decide(17.5)
    above = decide(17.6)

    assert below.region_portal is not None
    assert ":gap:" in below.region_portal
    # A one-frame snapshot cannot establish whether an intermediate radius is
    # growing or shrinking, so its deadline remains conservatively unknown.
    assert below.region_deadline_slack == -math.inf
    # At 17.5 the required six-unit clearance has zero time remaining.
    assert at_boundary.region_portal is not None
    assert ":gap:" in at_boundary.region_portal
    assert at_boundary.region_deadline_slack is not None
    assert at_boundary.region_deadline_slack < 0.0
    assert above.region_portal is None
    assert above.region_path_margin == -math.inf


def test_side_portal_follows_translated_and_rotated_row_endpoints() -> None:
    radius = 20.0
    config = MPCConfig(
        observation_delay=0,
        horizon_frames=60,
        preferred_y_fraction=39.0 / 202.0,
    )
    side_offset = radius + 0.5 + 6.0 + config.region_safe_margin_target
    target_y_offset = radius + 0.5 + 6.0

    for angle_degrees, translation_x, expected_side, endpoint_index in (
        (4.0, -35.0, "right", 2),
        (-4.0, 35.0, "left", 0),
    ):
        angle = math.radians(angle_degrees)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        row = [
            (column, x * cosine + translation_x, x * sine - 20.0)
            for column, x in enumerate((-48.0, 0.0, 48.0))
        ]
        walls = [
            wall_object(
                object_id,
                x,
                y,
                radius=radius,
                dx=0.0,
                dy=0.0,
            )
            for object_id, x, y in row
        ]
        teacher = EngineMPC(config)
        decision = teacher.select(observation(
            10,
            player_x=0.0,
            player_y=-55.0,
            indestructibles=walls,
            bounds=(-140.0, 140.0, -110.0, 140.0),
        ))

        endpoint = row[endpoint_index]
        direction = 1.0 if expected_side == "right" else -1.0
        assert decision.region_portal is not None
        assert decision.region_portal.endswith(f":side:{expected_side}")
        assert decision.region_crossing is True
        assert decision.region_anchor is not None
        assert math.isclose(
            decision.region_anchor[0],
            endpoint[1] + direction * side_offset,
            abs_tol=1e-9,
        )
        assert math.isclose(
            decision.region_anchor[1],
            endpoint[2] + target_y_offset,
            abs_tol=1e-9,
        )


def test_region_row_identity_survives_endpoint_loss_and_motion() -> None:
    def moving_rows(frame: int, *, drop_left: bool) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        for row, base_y in enumerate((-60.0, 0.0)):
            columns = range(1 if drop_left else 0, 5)
            for column in columns:
                values.append(wall_object(
                    row * 100 + column,
                    -96.0 + 48.0 * column + frame,
                    base_y - frame,
                    radius=7.0,
                    dx=1.0,
                    dy=-1.0,
                ))
        return values

    teacher = EngineMPC(MPCConfig(observation_delay=0, horizon_frames=60))
    first = teacher.select(observation(
        0,
        player_x=0.0,
        player_y=-30.0,
        indestructibles=moving_rows(0, drop_left=False),
    ))
    second = teacher.select(observation(
        3,
        player_x=3.0,
        player_y=-33.0,
        indestructibles=moving_rows(3, drop_left=True),
    ))

    assert first.region_current_component == second.region_current_component
    assert first.region_current_component is not None
    assert "indestructibles:" not in first.region_current_component
    assert first.region_current_component.startswith("band:row:")


def learned_region_dynamics() -> RegionDynamicsMemory:
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


def learned_lateral_region_dynamics() -> RegionDynamicsMemory:
    return replace(
        learned_region_dynamics(),
        lateral_flow_cycle_frames=360.0,
        safe_side_rule="opposite_incoming_lateral_flow",
    )


def test_phase_flow_forecast_is_relative_and_mirrors_the_observed_flow() -> None:
    def decide(
        *,
        mirror: int,
        frame: int,
        dynamics: RegionDynamicsMemory | None = None,
    ):
        walls = [
            wall_object(
                row * 100 + column,
                mirror * x,
                y,
                radius=7.0,
                dx=mirror * 1.5,
                dy=-2.6,
            )
            for row, y in enumerate((60.0, 120.0, 180.0, 222.0))
            for column, x in enumerate(
                (-192.0, -144.0, -96.0, -48.0, 0.0, 48.0)
            )
        ]
        teacher = EngineMPC(MPCConfig(
            observation_delay=0,
            horizon_frames=60,
            region_dynamics_memory=(
                learned_lateral_region_dynamics()
                if dynamics is None else dynamics
            ),
        ))
        return teacher.select(observation(
            frame,
            player_x=mirror * 80.0,
            player_y=-160.0,
            indestructibles=walls,
            bounds=(-192.0, 192.0, -224.0, 224.0),
        ))

    left = decide(mirror=1, frame=10)
    shifted = decide(mirror=1, frame=441)
    mirrored = decide(mirror=-1, frame=10)
    legacy = decide(
        mirror=1,
        frame=10,
        dynamics=learned_region_dynamics(),
    )

    assert left.region_phase == shifted.region_phase == "minimum_hold"
    assert left.region_frames_until_expansion == shifted.region_frames_until_expansion
    assert left.region_target_component == shifted.region_target_component == (
        "exterior:left"
    )
    assert left.region_portal == shifted.region_portal == "phase-flow:left"
    assert left.region_anchor == shifted.region_anchor
    assert left.action.move_x == shifted.action.move_x == -1

    assert mirrored.region_target_component == "exterior:right"
    assert mirrored.region_portal == "phase-flow:right"
    assert mirrored.region_anchor is not None and left.region_anchor is not None
    assert mirrored.region_anchor[0] == pytest.approx(-left.region_anchor[0])
    assert mirrored.action.move_x == 1
    assert legacy.region_portal is not None
    assert not legacy.region_portal.startswith("phase-flow:")


def test_region_dynamics_prior_makes_first_cycle_portal_deadline_finite() -> None:
    row_x = (-96.0, -48.0, 0.0, 48.0, 96.0)
    walls = [
        wall_object(column, x, -120.0, radius=7.0, dx=0.0, dy=-2.0)
        for column, x in enumerate(row_x)
    ]
    teacher = EngineMPC(MPCConfig(
        observation_delay=0,
        horizon_frames=60,
        region_dynamics_memory=learned_region_dynamics(),
    ))
    decision = teacher.select(observation(
        500,
        player_x=0.0,
        player_y=-160.0,
        indestructibles=walls,
        bounds=(-192.0, 192.0, -224.0, 224.0),
    ))

    assert decision.region_phase == "minimum_hold"
    assert decision.region_learned_cycle_frames == 180.0
    assert decision.region_frames_until_expansion == 90.0
    assert decision.region_deadline_slack is not None
    assert math.isfinite(decision.region_deadline_slack)


def test_target_rows_ahead_comes_from_the_next_stable_band_geometry() -> None:
    row_x = (-96.0, -48.0, 0.0, 48.0, 96.0)

    def decide(row_y: tuple[float, ...]):
        walls = [
            wall_object(
                row * 100 + column,
                x,
                y,
                radius=7.0,
                dx=0.0,
                dy=0.0,
            )
            for row, y in enumerate(row_y)
            for column, x in enumerate(row_x)
        ]
        teacher = EngineMPC(MPCConfig(
            observation_delay=0,
            horizon_frames=60,
            region_safe_margin_target=1.0,
            region_dynamics_memory=learned_region_dynamics(),
        ))
        return teacher.select(observation(
            10,
            player_x=-24.0,
            player_y=-160.0,
            indestructibles=walls,
            bounds=(-200.0, 200.0, -240.0, 256.0),
        ))

    immediate_stable_band = decide((-200.0, -120.0, -60.0, 0.0))
    narrow_then_stable_band = decide((-200.0, -120.0, -63.0, 0.0))

    assert immediate_stable_band.region_target_rows_ahead == 1
    assert narrow_then_stable_band.region_target_rows_ahead == 2
    assert (
        immediate_stable_band.region_current_component
        == narrow_then_stable_band.region_current_component
    )
    assert (
        immediate_stable_band.region_target_component
        != narrow_then_stable_band.region_target_component
    )
    assert narrow_then_stable_band.region_target_component == "exterior:left"
    assert narrow_then_stable_band.region_portal == "row:2:side:left"


def test_default_region_reserve_skips_a_band_that_becomes_too_narrow() -> None:
    row_x = (-96.0, -48.0, 0.0, 48.0, 96.0)
    walls = [
        wall_object(
            row * 100 + column,
            x,
            y,
            radius=7.0,
            dx=0.0,
            dy=0.0,
        )
        for row, y in enumerate((-200.0, -120.0, -60.0, 0.0))
        for column, x in enumerate(row_x)
    ]

    def decide(region_margin: float):
        teacher = EngineMPC(MPCConfig(
            observation_delay=0,
            horizon_frames=60,
            region_safe_margin_target=region_margin,
            region_dynamics_memory=learned_region_dynamics(),
        ))
        decision = teacher.select(observation(
            10,
            player_x=-24.0,
            player_y=-160.0,
            indestructibles=walls,
            bounds=(-200.0, 200.0, -240.0, 256.0),
        ))
        selected = next(
            item for item in decision.evaluations
            if item.action.discrete == decision.action.discrete
        )
        return decision, selected

    legacy, legacy_evaluation = decide(1.0)
    safer, safer_evaluation = decide(8.0)

    assert legacy.region_target_rows_ahead == 1
    assert safer.region_target_rows_ahead == 3
    assert safer.region_crossing is True
    assert safer.region_path_margin is not None
    assert safer.region_path_margin > 0.0
    assert safer_evaluation.collided is False
    assert safer_evaluation.minimum_margin > 8.0
    assert legacy_evaluation.minimum_margin > 8.0


def test_persistent_region_intent_survives_a_blocked_straight_path() -> None:
    row_x = (-96.0, -48.0, 0.0, 48.0, 96.0)
    row_y = (-200.0, -120.0, -63.0, 0.0)
    walls = [
        wall_object(
            row * 100 + column,
            x,
            y,
            radius=7.0,
            dx=0.0,
            dy=0.0,
        )
        for row, y in enumerate(row_y)
        for column, x in enumerate(row_x)
    ]
    teacher = EngineMPC(MPCConfig(
        observation_delay=0,
        horizon_frames=60,
        region_dynamics_memory=learned_region_dynamics(),
    ))
    first_observation = observation(
        10,
        player_x=80.0,
        player_y=-160.0,
        indestructibles=walls,
        bounds=(-140.0, 140.0, -240.0, 256.0),
    )
    first_threats = teacher.estimator.update(first_observation)
    teacher._update_region_phase(first_observation, 10)
    first_player = teacher._player(first_observation, 0)
    first = teacher._region_anchor(
        first_player,
        teacher._bounds(first_observation, first_player[2]),
        first_threats,
        10,
    )

    blocked_observation = observation(
        13,
        player_x=80.0,
        player_y=-160.0,
        bullets=[bullet(900, 108.0, -160.0, a=18.0, b=18.0)],
        indestructibles=walls,
        bounds=(-140.0, 140.0, -240.0, 256.0),
    )
    blocked_threats = teacher.estimator.update(blocked_observation)
    teacher._update_region_phase(blocked_observation, 13)
    blocked_player = teacher._player(blocked_observation, 0)
    blocked = teacher._region_anchor(
        blocked_player,
        teacher._bounds(blocked_observation, blocked_player[2]),
        blocked_threats,
        13,
    )

    assert first is not None and blocked is not None
    assert first.target_component == "exterior:right"
    assert blocked.target_component == first.target_component
    assert blocked.path_margin < 0.0
    assert blocked.navigation_mode == "preposition"


def test_exterior_side_changes_when_row_motion_will_close_the_near_boundary() -> None:
    row_x = (-96.0, -48.0, 0.0, 48.0, 96.0)
    row_y = (-200.0, -120.0, -63.0, 0.0)

    def decide(vx: float):
        walls = [
            wall_object(
                row * 100 + column,
                x,
                y,
                radius=7.0,
                dx=vx,
                dy=0.0,
            )
            for row, y in enumerate(row_y)
            for column, x in enumerate(row_x)
        ]
        teacher = EngineMPC(MPCConfig(
            observation_delay=0,
            horizon_frames=60,
            region_dynamics_memory=learned_region_dynamics(),
        ))
        return teacher.select(observation(
            10,
            player_x=80.0,
            player_y=-160.0,
            indestructibles=walls,
            bounds=(-140.0, 140.0, -240.0, 256.0),
        ))

    stationary = decide(0.0)
    moving_toward_right_boundary = decide(2.0)

    assert stationary.region_target_component == "exterior:right"
    assert stationary.region_portal == "row:2:side:right"
    assert moving_toward_right_boundary.region_target_component == "exterior:left"
    assert moving_toward_right_boundary.region_portal == "row:2:side:left"


@pytest.mark.parametrize("changed_key", ("phase", "mode", "portal"))
def test_region_commitment_is_invalidated_when_topology_key_changes(
    changed_key: str,
) -> None:
    lower_x = (-203.0, -155.0, -107.0, -59.0, -11.0)
    upper_x = (-186.0, -138.0, -90.0, -42.0, 6.0)
    walls = [
        *(wall_object(index, x, -199.0, radius=7.0, dx=0.0, dy=0.0)
          for index, x in enumerate(lower_x)),
        *(wall_object(100 + index, x, -140.0, radius=7.0, dx=0.0, dy=0.0)
          for index, x in enumerate(upper_x)),
    ]
    teacher = EngineMPC(MPCConfig(observation_delay=0, horizon_frames=60))
    first = teacher.select(observation(
        10,
        player_x=-17.0,
        player_y=-160.0,
        indestructibles=walls,
        bounds=(-192.0, 192.0, -224.0, 224.0),
    ))
    original_compute = teacher._compute

    def changed_compute(current_observation, threats, source_frame):
        decision = original_compute(current_observation, threats, source_frame)
        if changed_key == "phase":
            return replace(
                decision,
                region_phase="expanding",
                region_phase_started_frame=source_frame,
            )
        if changed_key == "mode":
            return replace(decision, region_navigation_mode="preposition")
        assert decision.region_portal is not None
        return replace(decision, region_portal=f"{decision.region_portal}:changed")

    teacher._compute = changed_compute
    second = teacher.select(observation(
        13,
        player_x=-17.0,
        player_y=-160.0,
        indestructibles=walls,
        bounds=(-192.0, 192.0, -224.0, 224.0),
    ))

    assert first.region_anchor is not None
    assert first.planned_actions
    assert second.using_committed_plan is False
    assert second.planned_actions[0].discrete == second.action.discrete


def test_bottom_boundary_forms_the_first_safe_band() -> None:
    upper_x = (-186.0, -138.0, -90.0, -42.0, 6.0)
    walls = [
        wall_object(100 + index, x, -135.0, radius=7.0)
        for index, x in enumerate(upper_x)
    ]
    teacher = EngineMPC(MPCConfig(observation_delay=0, horizon_frames=60))
    decision = teacher.select(observation(
        10,
        player_x=-17.0,
        player_y=-160.0,
        indestructibles=walls,
        bounds=(-192.0, 192.0, -224.0, 224.0),
    ))

    assert decision.region_anchor is not None
    assert decision.region_crossing is True
    assert -160.0 < decision.region_anchor[1] < -120.0


def test_distant_first_wall_does_not_pull_player_up_from_bottom_anchor() -> None:
    upper_x = (-186.0, -138.0, -90.0, -42.0, 6.0)
    walls = [
        wall_object(100 + index, x, 200.0, radius=7.0)
        for index, x in enumerate(upper_x)
    ]
    teacher = EngineMPC(MPCConfig(observation_delay=0, horizon_frames=60))
    decision = teacher.select(observation(
        10,
        player_x=-17.0,
        player_y=-176.0,
        indestructibles=walls,
        bounds=(-192.0, 192.0, -224.0, 224.0),
    ))

    assert decision.region_crossing is False
    assert decision.region_anchor is not None
    assert decision.region_anchor[1] == -176.0


def test_growing_wall_holds_its_band_when_outer_crossing_path_is_blocked() -> None:
    lower_x = (-203.0, -155.0, -107.0, -59.0, -11.0)
    upper_x = (-186.0, -138.0, -90.0, -42.0, 6.0)

    def walls(radius: float, rewind: float = 0.0) -> list[dict[str, Any]]:
        return [
            *(wall_object(
                index,
                x + 1.5 * rewind,
                -199.0 + 2.6 * rewind,
                radius=radius,
            ) for index, x in enumerate(lower_x)),
            *(wall_object(
                100 + index,
                x + 1.5 * rewind,
                -140.0 + 2.6 * rewind,
                radius=radius,
            ) for index, x in enumerate(upper_x)),
        ]

    teacher = EngineMPC(MPCConfig(observation_delay=0, horizon_frames=60))
    teacher.select(observation(
        7,
        player_x=-17.0,
        player_y=-160.0,
        indestructibles=walls(18.9, rewind=3.0),
        bounds=(-192.0, 192.0, -224.0, 224.0),
    ))
    decision = teacher.select(observation(
        10,
        player_x=-17.0,
        player_y=-160.0,
        indestructibles=walls(21.0),
        bounds=(-192.0, 192.0, -224.0, 224.0),
    ))

    assert decision.region_crossing is False
    assert decision.region_path_margin is not None
    assert decision.region_path_margin < 0.0
    assert decision.region_anchor is not None
    assert decision.region_anchor[0] > 20.0
    assert decision.action.move_y < 0


def test_region_phase_learns_two_cycles_then_anticipates_the_next_expansion() -> None:
    row_x = (-186.0, -138.0, -90.0, -42.0, 6.0)
    row_y = (-200.0, -140.0, -80.0, -20.0)

    def walls(radius: float) -> list[dict[str, Any]]:
        return [
            wall_object(
                row * 100 + column,
                x,
                y,
                radius=radius,
                dx=0.0,
                dy=0.0,
            )
            for row, y in enumerate(row_y)
            for column, x in enumerate(row_x)
        ]

    teacher = EngineMPC(MPCConfig(observation_delay=0, horizon_frames=60))
    teacher.select(observation(
        97,
        player_x=-17.0,
        player_y=-160.0,
        indestructibles=walls(7.0),
        bounds=(-192.0, 192.0, -224.0, 224.0),
    ))
    teacher.select(observation(
        100,
        player_x=-17.0,
        player_y=-160.0,
        indestructibles=walls(7.0),
        bounds=(-192.0, 192.0, -224.0, 224.0),
    ))
    teacher.select(observation(
        103,
        player_x=-17.0,
        player_y=-160.0,
        indestructibles=walls(9.1),
        bounds=(-192.0, 192.0, -224.0, 224.0),
    ))
    growth = teacher.select(observation(
        106,
        player_x=-17.0,
        player_y=-160.0,
        indestructibles=walls(11.2),
        bounds=(-192.0, 192.0, -224.0, 224.0),
    ))
    teacher.select(observation(
        130,
        player_x=-17.0,
        player_y=-160.0,
        indestructibles=walls(28.0),
        bounds=(-192.0, 192.0, -224.0, 224.0),
    ))
    teacher.select(observation(
        133,
        player_x=-17.0,
        player_y=-160.0,
        indestructibles=walls(28.0),
        bounds=(-192.0, 192.0, -224.0, 224.0),
    ))
    teacher.select(observation(
        160,
        player_x=-17.0,
        player_y=-160.0,
        indestructibles=walls(20.0),
        bounds=(-192.0, 192.0, -224.0, 224.0),
    ))
    teacher.select(observation(
        190,
        player_x=-17.0,
        player_y=-160.0,
        indestructibles=walls(7.0),
        bounds=(-192.0, 192.0, -224.0, 224.0),
    ))
    teacher.select(observation(
        280,
        player_x=-17.0,
        player_y=-160.0,
        indestructibles=walls(7.0),
        bounds=(-192.0, 192.0, -224.0, 224.0),
    ))
    teacher.select(observation(
        283,
        player_x=-17.0,
        player_y=-160.0,
        indestructibles=walls(9.1),
        bounds=(-192.0, 192.0, -224.0, 224.0),
    ))
    teacher.select(observation(
        286,
        player_x=-17.0,
        player_y=-160.0,
        indestructibles=walls(11.2),
        bounds=(-192.0, 192.0, -224.0, 224.0),
    ))
    teacher.select(observation(
        310,
        player_x=-17.0,
        player_y=-160.0,
        indestructibles=walls(28.0),
        bounds=(-192.0, 192.0, -224.0, 224.0),
    ))
    teacher.select(observation(
        313,
        player_x=-17.0,
        player_y=-160.0,
        indestructibles=walls(28.0),
        bounds=(-192.0, 192.0, -224.0, 224.0),
    ))
    teacher.select(observation(
        340,
        player_x=-17.0,
        player_y=-160.0,
        indestructibles=walls(20.0),
        bounds=(-192.0, 192.0, -224.0, 224.0),
    ))
    teacher.select(observation(
        370,
        player_x=-17.0,
        player_y=-160.0,
        indestructibles=walls(7.0),
        bounds=(-192.0, 192.0, -224.0, 224.0),
    ))
    anticipated = teacher.select(observation(
        412,
        player_x=-17.0,
        player_y=-160.0,
        indestructibles=walls(7.0),
        bounds=(-192.0, 192.0, -224.0, 224.0),
    ))

    assert growth.region_evacuating is True
    assert anticipated.region_evacuating is True
    assert anticipated.region_learned_cycle_frames == 182.0
    assert anticipated.region_frames_until_expansion is not None
    assert abs(anticipated.region_frames_until_expansion - 48.0) <= 6.0
    assert anticipated.region_target_rows_ahead >= 1
    assert anticipated.region_navigation_mode in {"preposition", "evacuate"}
    assert anticipated.region_target_component == "exterior:right"
    assert any(action.move_x > 0 for action in anticipated.planned_actions[:5])


def _gap_test_config(**values: Any) -> MPCConfig:
    return replace(MPCConfig(
        observation_delay=0,
        horizon_frames=36,
        beam_width=32,
    ), **values)


def _gap_wavefront(
    points: tuple[tuple[float, float], ...],
    *,
    velocity: tuple[float, float] = (0.0, -2.0),
) -> list[dict[str, Any]]:
    return [
        bullet(
            index,
            x,
            y,
            dx=velocity[0],
            dy=velocity[1],
        )
        for index, (x, y) in enumerate(points)
    ]


def _gap_geometry(
    bullets: list[dict[str, Any]],
    *,
    player_x: float = -70.0,
    player_y: float = 0.0,
    config: MPCConfig | None = None,
):
    teacher = EngineMPC(config or _gap_test_config())
    current = observation(
        0,
        player_x=player_x,
        player_y=player_y,
        bullets=bullets,
    )
    threats = teacher.estimator.update(current)
    player = teacher._player(current, teacher.config.observation_delay)
    bounds = teacher._bounds(current, player[2])
    groups, corridors = teacher._gap_corridors(player, bounds, threats)
    return teacher, current, player, bounds, threats, groups, corridors


@pytest.mark.parametrize(
    ("points", "velocity", "player", "expected_center"),
    (
        (
            ((-80.0, 60.0), (-20.0, 60.0), (40.0, 60.0), (90.0, 60.0)),
            (0.0, -2.0),
            (-70.0, 0.0),
            (-50.0, 0.0),
        ),
        (
            ((-60.0, -70.0), (-60.0, -20.0), (-60.0, 30.0), (-60.0, 60.0)),
            (2.0, 0.0),
            (0.0, -60.0),
            (0.0, -45.0),
        ),
    ),
)
def test_parallel_bullet_wavefront_forms_a_persistent_rotatable_gap(
    points: tuple[tuple[float, float], ...],
    velocity: tuple[float, float],
    player: tuple[float, float],
    expected_center: tuple[float, float],
) -> None:
    teacher, _, current_player, bounds, threats, groups, _ = _gap_geometry(
        _gap_wavefront(points, velocity=velocity),
        player_x=player[0],
        player_y=player[1],
    )
    _, corridors, selected, _ = teacher._gap_navigation(
        current_player,
        bounds,
        threats,
        None,
    )

    assert len(groups) == 1
    assert groups[0].coverage_fraction >= 0.45
    assert selected is not None
    stable = selected
    assert stable.center == pytest.approx(expected_center)
    assert stable.usable_width >= 25.0
    assert stable.lifetime_frames == 36
    assert stable.arrival_frames == pytest.approx(30.0)
    assert stable.path_margin >= 8.0


@pytest.mark.parametrize(
    ("bullet_count", "expected_profiles"),
    (
        (3, {"bullet-group-expert"}),
        (4, {"bullet-group-intermediate", "bullet-group-expert"}),
        (
            5,
            {
                "bullet-group-novice",
                "bullet-group-intermediate",
                "bullet-group-expert",
            },
        ),
    ),
)
def test_bullet_group_profiles_recognize_increasingly_subtle_wavefronts(
    bullet_count: int,
    expected_profiles: set[str],
) -> None:
    from stg_lab.engine_matrix import apply_controller_profile

    points = tuple(
        (-80.0 + 40.0 * index, 60.0)
        for index in range(bullet_count)
    )
    recognized = set()
    for profile in (
        "bullet-group-novice",
        "bullet-group-intermediate",
        "bullet-group-expert",
    ):
        config = apply_controller_profile(profile, _gap_test_config())
        *_, groups, _corridors = _gap_geometry(
            _gap_wavefront(points),
            config=config,
        )
        if groups:
            recognized.add(profile)

    assert recognized == expected_profiles


@pytest.mark.parametrize(
    ("direction_offsets", "expected_profiles"),
    (
        (
            (-10.0, -5.0, 0.0, 5.0, 10.0),
            {"bullet-group-expert"},
        ),
        (
            (-6.0, -3.0, 0.0, 3.0, 6.0),
            {"bullet-group-intermediate", "bullet-group-expert"},
        ),
        (
            (0.0, 0.0, 0.0, 0.0, 0.0),
            {
                "bullet-group-novice",
                "bullet-group-intermediate",
                "bullet-group-expert",
            },
        ),
    ),
)
def test_bullet_group_profiles_handle_direction_noise_by_skill(
    direction_offsets: tuple[float, ...],
    expected_profiles: set[str],
) -> None:
    from stg_lab.engine_matrix import apply_controller_profile

    bullets = []
    for index, degrees in enumerate(direction_offsets):
        angle = math.radians(degrees)
        bullets.append(bullet(
            index,
            -80.0 + 40.0 * index,
            60.0,
            dx=2.0 * math.sin(angle),
            dy=-2.0 * math.cos(angle),
        ))
    recognized = set()
    for profile in (
        "bullet-group-novice",
        "bullet-group-intermediate",
        "bullet-group-expert",
    ):
        config = apply_controller_profile(profile, _gap_test_config())
        *_, groups, _corridors = _gap_geometry(bullets, config=config)
        if groups:
            recognized.add(profile)

    assert recognized == expected_profiles


@pytest.mark.parametrize(
    ("spacing", "expected_profiles"),
    (
        (32.0, {"bullet-group-expert"}),
        (40.0, {"bullet-group-intermediate", "bullet-group-expert"}),
        (
            50.0,
            {
                "bullet-group-novice",
                "bullet-group-intermediate",
                "bullet-group-expert",
            },
        ),
    ),
)
def test_bullet_group_profiles_require_level_appropriate_gap_width(
    spacing: float,
    expected_profiles: set[str],
) -> None:
    from stg_lab.engine_matrix import apply_controller_profile

    wavefront = _gap_wavefront(tuple(
        ((index - 2) * spacing, 60.0)
        for index in range(5)
    ))
    for value in wavefront:
        value["a"] = 1.0
        value["b"] = 1.0
    accepted = set()
    for profile in (
        "bullet-group-novice",
        "bullet-group-intermediate",
        "bullet-group-expert",
    ):
        config = apply_controller_profile(profile, _gap_test_config())
        *_, groups, corridors = _gap_geometry(wavefront, config=config)
        assert len(groups) == 1
        if corridors:
            accepted.add(profile)

    assert accepted == expected_profiles


def test_detected_parallel_gap_steers_toward_reachable_corridor() -> None:
    wavefront = _gap_wavefront((
        (-72.5, 40.0),
        (-37.5, 40.0),
        (-2.5, 40.0),
        (32.5, 40.0),
    ))
    current = observation(
        0,
        player_x=30.0,
        player_y=0.0,
        bullets=wavefront,
    )

    enabled = EngineMPC(_gap_test_config()).select(current)
    disabled = EngineMPC(_gap_test_config(
        gap_prediction_enabled=False,
    )).select(current)
    selected = next(
        value for value in enabled.evaluations
        if value.action.discrete == enabled.action.discrete
    )

    assert enabled.gap_selected_center == pytest.approx((15.0, 0.0))
    assert enabled.gap_selected_width == pytest.approx(10.0)
    assert enabled.gap_navigation_mode == "enter"
    assert enabled.action.move_x == -1
    assert disabled.action.move_x == 0
    assert selected.collided is False


def test_nonparallel_bullets_do_not_form_one_gap_group() -> None:
    bullets = [
        *(
            bullet(index, x, 60.0, dx=0.0, dy=-2.0)
            for index, x in enumerate((-70.0, -20.0))
        ),
        *(
            bullet(10 + index, x, 60.0, dx=1.0, dy=-1.4)
            for index, x in enumerate((20.0, 70.0))
        ),
    ]

    _, _, _, _, _, groups, corridors = _gap_geometry(bullets)

    assert groups == ()
    assert corridors == ()


def test_current_gap_is_rejected_when_parallel_edges_close_in_the_future() -> None:
    bullets = [
        bullet(index, x, 60.0, dx=vx, dy=-2.0)
        for index, (x, vx) in enumerate(zip(
            (-45.0, -15.0, 15.0, 45.0),
            (0.15, 0.05, -0.05, -0.15),
        ))
    ]

    _, _, _, _, _, groups, corridors = _gap_geometry(bullets)

    assert len(groups) == 1
    assert groups[0].coverage_fraction >= 0.45
    assert corridors == ()


def test_gap_width_includes_bullet_player_and_configured_safety_radii() -> None:
    def corridor_widths(spacing: float) -> tuple[float, ...]:
        points = tuple(
            ((index - 1.5) * spacing, 60.0)
            for index in range(4)
        )
        *_, corridors = _gap_geometry(_gap_wavefront(points))
        return tuple(value.usable_width for value in corridors)

    assert corridor_widths(28.9) == ()
    assert corridor_widths(29.0) == pytest.approx((4.0, 4.0, 4.0))


def test_large_third_bullet_can_block_every_otherwise_valid_gap_entry() -> None:
    wavefront = _gap_wavefront((
        (-80.0, 60.0),
        (-20.0, 60.0),
        (40.0, 60.0),
        (90.0, 60.0),
    ))
    baseline = _gap_geometry(wavefront)
    blocked = _gap_geometry([
        *wavefront,
        bullet(99, -40.0, 0.0, a=40.0, b=40.0, dx=0.0, dy=0.0),
    ])

    baseline_teacher, _, baseline_player, baseline_bounds, baseline_threats, _, _ = baseline
    _, baseline_corridors, baseline_selected, _ = baseline_teacher._gap_navigation(
        baseline_player,
        baseline_bounds,
        baseline_threats,
        None,
    )
    teacher, _, player, bounds, threats, _, blocked_geometry = blocked
    _, blocked_corridors, selected, mode = teacher._gap_navigation(
        player,
        bounds,
        threats,
        None,
    )

    assert baseline_selected is not None
    assert baseline_selected.path_margin >= 8.0
    assert len(blocked_corridors) == len(blocked_geometry)
    assert all(not value.entry_plan for value in blocked_corridors)
    assert selected is None
    assert mode == "observe"


def test_open_safe_gap_is_observed_without_unnecessary_lateral_chase() -> None:
    wavefront = _gap_wavefront((
        (-80.0, 60.0),
        (-20.0, 60.0),
        (40.0, 60.0),
        (90.0, 60.0),
    ))
    decision = EngineMPC(_gap_test_config()).select(observation(
        0,
        player_x=10.0,
        player_y=0.0,
        bullets=wavefront,
    ))

    assert decision.gap_bullet_group_count == 1
    assert decision.gap_corridor_count == 3
    assert decision.gap_selected_center is None
    assert decision.gap_navigation_mode == "observe"
    assert decision.action.move_x == 0


def test_player_already_inside_gap_holds_as_wavefront_reaches_guard_window() -> None:
    teacher = EngineMPC(_gap_test_config())
    modes = []
    centers = []
    for frame, y in zip((0, 3, 6, 9), (30.0, 24.0, 18.0, 12.0)):
        wavefront = _gap_wavefront((
            (-110.0, y),
            (-80.0, y),
            (-20.0, y),
            (40.0, y),
        ))
        decision = teacher.select(observation(
            frame,
            player_x=-40.0,
            player_y=0.0,
            bullets=wavefront,
        ))
        modes.append(decision.gap_navigation_mode)
        centers.append(decision.gap_selected_center)

    assert modes == ["hold", "hold", "hold", "hold"]
    assert all(value == pytest.approx((-50.0, 0.0)) for value in centers)


def test_gap_entry_certificate_uses_executable_three_frame_action_blocks() -> None:
    teacher = EngineMPC(_gap_test_config(horizon_frames=36))
    player = (0.0, 0.0, 0.5, 10.0, 2.0)
    bounds = (-100.0, 100.0, -100.0, 100.0)
    path_x, path_y, _travel, settled, plan = teacher._gap_entry_path(
        player,
        bounds,
        (55.556, 0.0),
        (1.0, 0.0),
        10.0,
        11.9,
        18,
    )

    previous = np.asarray(player[:2], dtype=np.float64)
    deltas = []
    for x, y in zip(path_x, path_y):
        current = np.asarray((x, y), dtype=np.float64)
        deltas.append(current - previous)
        previous = current
    for block_start in range(0, len(deltas), teacher.config.decision_interval):
        block = deltas[block_start:block_start + teacher.config.decision_interval]
        assert all(value == pytest.approx(block[0]) for value in block)
    assert len(plan) == len(path_x) // teacher.config.decision_interval
    assert settled is False
    margin, _, certified_plan = teacher._gap_path_margin(
        player,
        bounds,
        (55.556, 0.0),
        (1.0, 0.0),
        10.0,
        11.9,
        (),
    )
    assert margin == -math.inf
    assert certified_plan == ()


def test_gap_entry_certificate_finds_safe_discrete_detour() -> None:
    teacher = EngineMPC(_gap_test_config())
    current = observation(
        0,
        bullets=[bullet(99, 24.0, 0.0, dx=0.0, dy=0.0)],
    )
    threats = teacher.estimator.update(current)
    player = teacher._player(current, teacher.config.observation_delay)
    bounds = teacher._bounds(current, player[2])
    forecast = teacher._gap_threat_forecast(threats)
    direct_x, direct_y, _, direct_settled, _ = teacher._gap_entry_path(
        player,
        bounds,
        (55.556, 0.0),
        (1.0, 0.0),
        10.0,
        21.0,
        27,
    )
    direct_margin = float(np.min(
        np.hypot(
            direct_x[:, None] - forecast[0][:27],
            direct_y[:, None] - forecast[1][:27],
        ) - player[2] - forecast[2][:27]
    ))

    margin, travel, plan = teacher._gap_path_margin(
        player,
        bounds,
        (55.556, 0.0),
        (1.0, 0.0),
        10.0,
        21.0,
        threats,
        forecast,
    )

    assert direct_settled is True
    assert direct_margin == pytest.approx(-2.5)
    assert margin >= teacher.config.gap_path_minimum_margin
    assert travel <= 15.0
    assert plan
    assert plan[0].move_x == 1
    assert any(action.move_y != 0 for action in plan)


def test_enter_uses_certified_route_and_reaches_hold() -> None:
    teacher = EngineMPC(_gap_test_config())
    bullet_x = (-130.0, -120.0, -110.0, 7.5, 42.5, 77.5)
    boss = [{
        "id": 99,
        "x": -100.0,
        "y": 80.0,
        "hp": 1000.0,
        "maxhp": 1000.0,
        "a": 1.0,
        "b": 1.0,
        "collidable": False,
    }]

    def current(frame: int, player_x: float) -> dict[str, Any]:
        return observation(
            frame,
            player_x=player_x,
            player_y=0.0,
            bullets=_gap_wavefront(tuple(
                (x, 24.0 - 2.0 * frame) for x in bullet_x
            )),
            enemies=boss,
        )

    first = teacher.select(current(0, 0.0))
    second = teacher.select(current(3, 12.0))
    inside = teacher.select(current(6, 24.0))

    assert first.gap_selected_center == pytest.approx((25.0, 0.0))
    assert first.gap_navigation_mode == "enter"
    assert (first.action.move_x, first.action.move_y, first.action.slow) == (
        1,
        0,
        False,
    )
    assert first.planned_actions[0].discrete == first.action.discrete
    assert second.gap_navigation_mode == "enter"
    assert second.action.move_x == 1
    assert inside.gap_navigation_mode == "hold"


def test_gap_entry_certification_is_bounded_after_geometry_grouping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _gap_test_config(
        observation_delay=5,
        horizon_frames=60,
        gap_entry_candidate_limit=3,
    )
    teacher = EngineMPC(config)
    bullets = [
        bullet(
            row * 13 + column,
            -220.0 + 40.0 * column,
            -160.0 + 25.0 * row,
            dx=0.0,
            dy=-10.0,
        )
        for row in range(23)
        for column in range(13)
    ]
    current = observation(
        0,
        player_x=0.0,
        player_y=-176.0,
        bullets=bullets,
        bounds=(-300.0, 300.0, -300.0, 500.0),
    )
    threats = teacher.estimator.update(current)
    player = teacher._player(current, teacher.config.observation_delay)
    bounds = teacher._bounds(current, player[2])
    groups, corridors = teacher._gap_corridors(player, bounds, threats)
    calls = 0

    def reject(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return -math.inf, math.inf, ()

    monkeypatch.setattr(teacher, "_gap_path_margin", reject)
    _, _, selected, mode = teacher._gap_navigation(
        player,
        bounds,
        threats,
        None,
    )

    assert len(groups) == 23
    assert len(corridors) == 276
    assert calls == config.gap_entry_candidate_limit
    assert selected is None
    assert mode == "observe"


def test_vectorized_gap_forecast_matches_per_threat_prediction() -> None:
    teacher = EngineMPC(_gap_test_config())
    threats = (
        PredictedThreat(
            key="enemy_bullets:1",
            source="enemy_bullets",
            object_id=1,
            x=-10.0,
            y=20.0,
            vx=1.5,
            vy=-2.0,
            radius=2.0,
            radius_rate=0.25,
            source_frame=0,
            observation_delay=0,
            radius_rate_horizon=5,
            motion_horizon=9,
        ),
        PredictedThreat(
            key="lasers:2",
            source="lasers",
            object_id=2,
            x=30.0,
            y=-40.0,
            vx=-0.5,
            vy=0.75,
            radius=8.0,
            radius_rate=-0.4,
            source_frame=0,
            observation_delay=0,
            radius_rate_horizon=12,
            motion_horizon=36,
        ),
    )

    forecast = teacher._gap_threat_forecast(threats)

    for future_frame in range(1, teacher.config.horizon_frames + 1):
        for threat_index, threat in enumerate(threats):
            expected = teacher._threat_at(threat, future_frame)
            actual = tuple(
                values[future_frame - 1, threat_index] for values in forecast
            )
            assert actual == pytest.approx(expected)


def test_region_anchor_selects_a_compatible_reachable_gap() -> None:
    config = _gap_test_config(horizon_frames=60)
    wavefront = _gap_wavefront((
        (-80.0, 120.0),
        (-20.0, 120.0),
        (40.0, 120.0),
        (90.0, 120.0),
    ))
    teacher, _, player, bounds, threats, _, _ = _gap_geometry(
        wavefront,
        player_x=-70.0,
        config=config,
    )
    region = _RegionAnchor(
        x=65.0,
        y=0.0,
        crossing=False,
        path_margin=20.0,
        evacuating=True,
        target_rows_ahead=1,
        navigation_mode="preposition",
        current_component="band:0",
        target_component="exterior:right",
        portal="test",
        deadline_slack=30.0,
    )

    _, _, selected, mode = teacher._gap_navigation(
        player,
        bounds,
        threats,
        region,
    )

    assert selected is not None
    assert selected.center == pytest.approx((65.0, 0.0))
    assert mode == "enter"


def test_disabling_gap_prediction_preserves_legacy_action_and_empty_telemetry() -> None:
    wavefront = _gap_wavefront((
        (-80.0, 60.0),
        (-20.0, 60.0),
        (40.0, 60.0),
        (90.0, 60.0),
    ))
    current = observation(
        0,
        player_x=-70.0,
        player_y=0.0,
        bullets=wavefront,
    )
    enabled = EngineMPC(_gap_test_config()).select(current)
    disabled = EngineMPC(_gap_test_config(
        gap_prediction_enabled=False,
    )).select(current)

    assert enabled.gap_selected_center is not None
    assert disabled.gap_bullet_group_count == 0
    assert disabled.gap_corridor_count == 0
    assert disabled.gap_selected_center is None
    assert disabled.gap_selected_width is None
    assert disabled.gap_selected_lifetime_frames is None
    assert disabled.gap_navigation_mode == "inactive"
    assert (
        disabled.action.move_x,
        disabled.action.move_y,
        disabled.action.slow,
    ) == (1, -1, False)


def test_frame_rewind_clears_the_committed_gap_from_the_previous_run() -> None:
    teacher = EngineMPC(_gap_test_config())
    wavefront = _gap_wavefront((
        (-80.0, 60.0),
        (-20.0, 60.0),
        (40.0, 60.0),
        (90.0, 60.0),
    ))

    committed = teacher.select(observation(
        9,
        player_x=-70.0,
        player_y=0.0,
        bullets=wavefront,
    ))
    restarted = teacher.select(observation(
        0,
        player_x=0.0,
        player_y=0.0,
        bullets=[],
    ))

    assert committed.gap_selected_center is not None
    assert committed.gap_navigation_mode in {"enter", "hold"}
    assert restarted.gap_selected_center is None
    assert restarted.gap_navigation_mode == "inactive"


def test_gap_alignment_cannot_override_a_predicted_collision() -> None:
    forced_gap = _GapCorridor(
        key="test-gap",
        group_key="test-group",
        center_x=80.0,
        center_y=0.0,
        usable_width=30.0,
        lifetime_frames=36,
        arrival_frames=6.0,
        path_margin=30.0,
        normal_x=1.0,
        normal_y=0.0,
        member_count=4,
    )
    teacher = EngineMPC(_gap_test_config(
        gap_anchor_weight=1000.0,
    ))
    teacher._gap_navigation = lambda *_args: (
        (),
        (forced_gap,),
        forced_gap,
        "enter",
    )
    decision = teacher.select(observation(
        0,
        bullets=[bullet(1, 4.0, 0.0, dx=0.0, dy=0.0)],
    ))
    selected = next(
        value for value in decision.evaluations
        if value.action.discrete == decision.action.discrete
    )
    toward_gap = evaluation(
        decision,
        move_x=1,
        move_y=0,
        slow=False,
    )

    assert decision.gap_selected_center == (80.0, 0.0)
    assert toward_gap.collided is True
    assert toward_gap.minimum_margin < 0.0
    assert selected.collided is False
    assert selected.minimum_margin > 0.0


def test_gap_anchor_steers_the_first_action_when_safety_is_equal() -> None:
    gap = _GapCorridor(
        key="test-gap",
        group_key="test-group",
        center_x=-80.0,
        center_y=0.0,
        usable_width=10.0,
        lifetime_frames=36,
        arrival_frames=24.0,
        path_margin=20.0,
        normal_x=1.0,
        normal_y=0.0,
        member_count=4,
    )
    teacher = EngineMPC(_gap_test_config())
    teacher._gap_navigation = lambda *_args: (
        (),
        (gap,),
        gap,
        "enter",
    )

    decision = teacher.select(observation(0))

    assert decision.gap_selected_center == (-80.0, 0.0)
    assert decision.action.move_x == -1


def test_gap_anchor_does_not_expand_the_configured_beam_budget() -> None:
    teacher = EngineMPC(_gap_test_config(beam_width=8))
    gap = _GapCorridor(
        key="test-gap",
        group_key="test-group",
        center_x=40.0,
        center_y=0.0,
        usable_width=20.0,
        lifetime_frames=36,
        arrival_frames=12.0,
        path_margin=20.0,
        normal_x=1.0,
        normal_y=0.0,
        member_count=4,
    )
    count = 40
    kept = teacher._diverse_keep(
        np.arange(count, dtype=np.int64),
        np.arange(count, dtype=np.float64),
        np.zeros(count, dtype=np.float64),
        np.arange(count, dtype=np.int64) % len(teacher.actions),
        None,
        gap,
    )

    assert len(kept) == teacher.config.beam_width
    assert kept == pytest.approx(np.arange(teacher.config.beam_width))
