import json
import math

import numpy as np
import pytest

from stg_lab.protocol import Action
from stg_lab.teacher_route import (
    ProtectedThreatTimeline,
    TeacherRouteConfig,
    solve_teacher_route,
    teacher_actions,
    validate_teacher_route,
)


def timeline(rows_by_frame: list[list[list[float]]]) -> ProtectedThreatTimeline:
    offsets = [0]
    rows: list[list[float]] = []
    for frame_rows in rows_by_frame:
        rows.extend(frame_rows)
        offsets.append(len(rows))
    return ProtectedThreatTimeline(
        frames=np.arange(1, len(rows_by_frame) + 1, dtype=np.int32),
        offsets=np.asarray(offsets, dtype=np.int64),
        threats=np.asarray(rows, dtype=np.float64).reshape((-1, 6)),
    )


def small_config(**changes: object) -> TeacherRouteConfig:
    values: dict[str, object] = {
        "bounds": (-20.0, 20.0, -20.0, 20.0),
        "boundary_padding": 0.0,
        "start_x": 0.0,
        "start_y": 0.0,
        "beam_width": 256,
        "position_bin_size": 0.5,
        "desired_clearance": 3.0,
        "clearance_cap": 12.0,
        "spatial_bin_size": 8.0,
        "boundary_margin": 0.0,
        "anchor_weight": 0.0,
        "anchor_x": 0.0,
        "anchor_y": 0.0,
    }
    values.update(changes)
    return TeacherRouteConfig(**values)


def test_capture_validation_rejects_nonconsecutive_frames() -> None:
    with pytest.raises(ValueError, match="consecutive"):
        ProtectedThreatTimeline(
            frames=np.array([1, 3], dtype=np.int32),
            offsets=np.array([0, 0, 0], dtype=np.int64),
            threats=np.empty((0, 6), dtype=np.float64),
        )


def test_warning_geometry_is_not_a_lethal_threat() -> None:
    capture = timeline([[[0.0, 0.0, 100.0, 100.0, 0.0, 1.0]]])
    validation = validate_teacher_route(
        capture,
        [],
        small_config(start_x=0.0, start_y=0.0),
    )
    assert validation.collision_free
    assert validation.minimum_clearance == 12.0


def test_replay_checks_every_frame_inside_a_three_frame_action() -> None:
    # Fast right crosses x=4, 8, 12.  The lethal ellipse exists only at the
    # middle sample, so endpoint-only checking would miss it.
    capture = timeline([
        [],
        [],
        [[8.0, 0.0, 1.0, 1.0, 37.0, 0.0]],
        [],
    ])
    validation = validate_teacher_route(
        capture,
        [Action(move_x=1)],
        small_config(player_radius=0.5),
    )
    assert not validation.collision_free
    assert validation.minimum_clearance == pytest.approx(-1.5)
    assert validation.minimum_clearance_frame == 3


def test_rotated_ellipse_margin_is_conservative_for_player_radius() -> None:
    angle = math.radians(45.0)
    # The player is 0.4 units outside the ellipse along its minor axis, less
    # than the 0.5 player radius.  Rotate the ellipse to ensure the calculation
    # is performed in local ellipse coordinates.
    displacement = (-2.4 * math.sin(angle), 2.4 * math.cos(angle))
    center = (-displacement[0], -displacement[1])
    capture = timeline([[[center[0], center[1], 6.0, 2.0, 45.0, 0.0]]])
    validation = validate_teacher_route(
        capture,
        [],
        small_config(player_radius=0.5),
    )
    assert not validation.collision_free
    assert validation.minimum_clearance == pytest.approx(-0.1)


def test_diagonal_motion_is_normalized_and_action_is_held_for_three_frames() -> None:
    capture = timeline([[], [], [], []])
    config = small_config()
    validation = validate_teacher_route(
        capture,
        [Action(move_x=1, move_y=1)],
        config,
    )
    expected = 3.0 * config.fast_speed / math.sqrt(2.0)
    assert validation.positions[-1] == pytest.approx((expected, expected))
    assert validation.path_distance == pytest.approx(3.0 * config.fast_speed)


def test_conservative_boundary_padding_is_applied_during_replay() -> None:
    capture = timeline([[], [], [], []])
    config = small_config(boundary_padding=2.0, start_x=17.0)
    validation = validate_teacher_route(capture, [Action(move_x=1)], config)
    assert validation.positions == ((17.0, 0.0),) + ((18.0, 0.0),) * 3


def test_solver_routes_around_a_later_blocker_and_emits_distillation_metadata(tmp_path) -> None:
    capture = timeline([
        [],
        [],
        [],
        [],
        [[0.0, 0.0, 3.0, 3.0, 0.0, 0.0]],
        [[0.0, 0.0, 3.0, 3.0, 0.0, 0.0]],
        [[0.0, 0.0, 3.0, 3.0, 0.0, 0.0]],
    ])
    config = small_config(start_x=0.0, start_y=0.0, beam_width=512)
    route = solve_teacher_route(capture, config)
    assert len(route.decisions) == 2
    assert all(item.frame_count == 3 for item in route.decisions)
    assert route.validation.collision_free
    assert route.validation.direction_reversals == 0
    assert math.hypot(*route.validation.positions[-1]) > 3.5

    output = tmp_path / "route.json"
    route.write_json(output)
    raw = json.loads(output.read_text(encoding="utf-8"))
    assert raw["purpose"] == "training_data_distillation_only"
    assert raw["not_for_deployment"] is True
    assert raw["not_acceptance_evidence"] is True
    assert raw["uses_complete_future_threat_timeline"] is True
    assert raw["timeline_semantics"] == "initial_state_plus_post_step_frames"
    assert len(raw["decisions"]) == 2


def test_initial_capture_frame_is_checked_before_any_action() -> None:
    capture = timeline([[[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]]])
    validation = validate_teacher_route(capture, [], small_config(player_radius=0.5))
    assert not validation.collision_free
    assert validation.positions == ((0.0, 0.0),)
    assert validation.minimum_clearance_frame == 1


def test_teacher_action_set_has_unique_stationary_control() -> None:
    actions = teacher_actions()
    assert len(actions) == 17
    assert sum(item.move_x == item.move_y == 0 for item in actions) == 1


def test_reference_route_penalty_prefers_the_conditioning_path() -> None:
    capture = timeline([[], [], [], []])
    reference = [Action(move_x=1)] * 3
    reference_positions = [(0.0, 0.0), (4.0, 0.0), (8.0, 0.0), (12.0, 0.0)]
    route = solve_teacher_route(
        capture,
        small_config(
            reference_action_penalty=100.0,
            reference_position_weight=10.0,
        ),
        reference_actions=reference,
        reference_positions=reference_positions,
    )
    assert route.decisions[0].action == Action(move_x=1)


def test_hard_clearance_overrides_an_unsafe_reference_route() -> None:
    capture = timeline([
        [],
        [[4.0, 0.0, 2.0, 2.0, 0.0, 0.0]],
        [[8.0, 0.0, 2.0, 2.0, 0.0, 0.0]],
        [[12.0, 0.0, 2.0, 2.0, 0.0, 0.0]],
    ])
    reference = [Action(move_x=1)] * 3
    route = solve_teacher_route(
        capture,
        small_config(
            reference_action_penalty=1000.0,
            reference_position_weight=100.0,
        ),
        reference_actions=reference,
        reference_positions=[(0.0, 0.0), (4.0, 0.0), (8.0, 0.0), (12.0, 0.0)],
    )
    assert route.validation.collision_free
    assert route.decisions[0].action != Action(move_x=1)
