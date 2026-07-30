import numpy as np

from stg_lab.planning import (
    PlannerConfig,
    RiskConfig,
    RiskField,
    SpatioTemporalPlanner,
    connected_components,
)
from stg_lab.scenarios import make_environment
from stg_lab.sim import SimulationConfig


def test_planner_uses_luastg_player_movement_bounds_and_real_offsets() -> None:
    env = make_environment(
        "stage5_boss4",
        duration_frames=30,
        config=SimulationConfig(reaction_frames=0),
    )
    planner = SpatioTemporalPlanner(PlannerConfig(risk=RiskConfig(
        horizon_frames=10,
        sample_every=4,
        cell_size=16.0,
    )))
    field = planner.build_field(env)
    assert tuple(field.frames) == (0, 4, 8, 10)
    assert field.bounds == (-184.0, 184.0, -208.0, 192.0)
    assert field.xs[0] == -184.0 and field.xs[-1] == 184.0
    assert field.ys[0] == -208.0 and field.ys[-1] == 192.0


def test_cached_timeline_planning_is_repeatable_without_source_mutation() -> None:
    env = make_environment(
        "stage5_boss4",
        seed=55,
        duration_frames=40,
        config=SimulationConfig(reaction_frames=0),
    )
    planner = SpatioTemporalPlanner(PlannerConfig(risk=RiskConfig(
        horizon_frames=18,
        sample_every=3,
        cell_size=20.0,
    )))
    first = planner.plan(env)
    first_cache_size = len(planner._layer_cache)
    second = planner.plan(env)
    assert env.frame == 0
    assert first.actions == second.actions
    assert first.waypoints == second.waypoints
    assert first.peak_level == second.peak_level
    assert len(planner._layer_cache) == first_cache_size


def test_planner_crosses_lower_danger_to_escape_a_disconnecting_safe_region() -> None:
    # The left safe component becomes lethal.  The only survivable route moves
    # through the level-1 bridge before reaching the disconnected right side.
    levels = np.full((5, 3, 5), 4, dtype=np.uint8)
    levels[:, 1, :2] = 0
    levels[:, 1, 2] = 1
    levels[:, 1, 3:] = 0
    levels[3:, 1, :2] = 4
    risk = np.choose(levels, [0.05, 0.5, 2.0, 5.0, 12.0]).astype(np.float32)
    field = RiskField(
        risk=risk,
        levels=levels,
        frames=np.arange(5, dtype=np.int32),
        xs=np.arange(5, dtype=np.float32),
        ys=np.arange(3, dtype=np.float32),
        bounds=(0.0, 4.0, 0.0, 2.0),
        player_radius=0.1,
        player_speed=1.0,
        focus_speed=1.0,
        sample_every=1,
    )
    initial_labels, count = connected_components(levels[0] == 0)
    assert count == 2
    assert initial_labels[1, 0] != initial_labels[1, 4]

    planner = SpatioTemporalPlanner(PlannerConfig(allow_diagonal=False))
    planner.build_field = lambda _source, *, observation=None: field  # type: ignore[method-assign]
    result = planner.plan({"frame": 0, "player": {"x": 0.0, "y": 1.0}})

    assert result.peak_level == 1
    assert [step.position for step in result.steps] == [
        (0.0, 1.0),
        (1.0, 1.0),
        (2.0, 1.0),
        (3.0, 1.0),
        (3.0, 1.0),
    ]
    assert [step.safety_level for step in result.steps] == [0, 0, 1, 0, 0]
    assert [step.region_id for step in result.steps] == [0, 0, None, 0, 0]
