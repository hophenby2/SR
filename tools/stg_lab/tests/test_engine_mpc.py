from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

import pytest

from stg_lab.engine_mpc import (
    EngineMPC,
    MPCConfig,
    RegionDynamicsMemory,
    VisibleTrackEstimator,
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
    estimator = VisibleTrackEstimator(MPCConfig(observation_delay=0))
    estimator.update(observation(
        0,
        indestructibles=[wall_object(9, 0.0, 0.0, radius=28.0)],
    ))
    dipped = estimator.update(observation(
        3,
        indestructibles=[wall_object(9, 0.0, -9.0, radius=27.3)],
    ))[0]

    assert dipped.radius_rate == 0.0
    assert dipped.radius_rate_horizon == 0
    assert dipped.at(3)[2] == 28.0

    rising = VisibleTrackEstimator(MPCConfig(observation_delay=0))
    rising.update(observation(
        0,
        indestructibles=[wall_object(9, 0.0, 0.0, radius=27.3)],
    ))
    maximum = rising.update(observation(
        3,
        indestructibles=[wall_object(9, 0.0, -9.0, radius=28.0)],
    ))[0]
    assert maximum.radius_rate == 0.0
    assert maximum.at(60)[2] == 28.0


def test_nuke_minimum_float_oscillation_does_not_reset_growth_phase() -> None:
    estimator = VisibleTrackEstimator(MPCConfig(observation_delay=0))
    estimator.update(observation(
        0,
        indestructibles=[wall_object(9, 0.0, 0.0, radius=6.3)],
    ))
    plateau = estimator.update(observation(
        3,
        indestructibles=[wall_object(9, 0.0, -9.0, radius=7.7)],
    ))[0]
    assert plateau.radius == 7.7
    assert plateau.radius_rate == 0.0

    growth = estimator.update(observation(
        6,
        indestructibles=[wall_object(9, 0.0, -18.0, radius=8.4)],
    ))[0]
    assert growth.radius_rate == 0.7
    assert growth.at(3)[2] == 10.5


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
        player_y=120.0,
        bullets=[bullet(5, 50.0, 0.0, dx=-0.2, dy=-4.0)],
    ))[0]

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
            preferred_y=-30.0,
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
    side_offset = radius + 0.5 + 6.0 + 1.0
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
        teacher = EngineMPC(MPCConfig(
            observation_delay=0,
            horizon_frames=60,
            preferred_y=-55.0,
        ))
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
        teacher = EngineMPC(MPCConfig(observation_delay=0, horizon_frames=60))
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
    assert anticipated.region_learned_cycle_frames == 180.0
    assert anticipated.region_frames_until_expansion is not None
    assert abs(anticipated.region_frames_until_expansion - 48.0) <= 6.0
    assert anticipated.region_target_rows_ahead >= 1
    assert anticipated.region_navigation_mode in {"preposition", "evacuate"}
    assert anticipated.region_target_component == "exterior:right"
    assert any(action.move_x > 0 for action in anticipated.planned_actions[:5])
