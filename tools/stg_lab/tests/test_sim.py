from dataclasses import dataclass

import numpy as np
import pytest

from stg_lab.protocol import Action
from stg_lab.sim import CircleThreat, EllipseThreat, Outcome, SimulationConfig, STGEnvironment
from stg_lab.metrics import state_hash
from stg_lab.scenarios import make_environment


@dataclass
class EmptyScenario:
    name: str = "empty"
    duration_frames: int = 12

    def reset(self, env: STGEnvironment) -> None:
        return None

    def update(self, env: STGEnvironment) -> None:
        return None


@dataclass
class SweptScenario:
    name: str = "swept"
    duration_frames: int = 12

    def reset(self, env: STGEnvironment) -> None:
        env.add_threat(CircleThreat(-20.0, -176.0, 1.0, vx=40.0))

    def update(self, env: STGEnvironment) -> None:
        return None


@dataclass
class EphemeralScenario:
    name: str = "ephemeral"
    duration_frames: int = 8

    def reset(self, env: STGEnvironment) -> None:
        return None

    def update(self, env: STGEnvironment) -> None:
        if env.frame == 1:
            env.add_threat(CircleThreat(-20.0, 0.0, 1.0, vx=40.0, lifetime=2))


def test_reaction_delay_and_diagonal_speed() -> None:
    env = STGEnvironment(EmptyScenario(), config=SimulationConfig(reaction_frames=2))
    start_x = env.player.x
    env.step(Action(move_x=1))
    env.step(Action(move_x=1))
    assert env.player.x == start_x
    env.step(Action(move_x=1))
    assert env.player.x == start_x + env.player.speed

    env = STGEnvironment(EmptyScenario(), config=SimulationConfig(reaction_frames=0))
    env.step(Action(move_x=1, move_y=1))
    assert np.isclose(np.hypot(env.player.x, env.player.y + 176.0), env.player.speed)


def test_luastg_player_bounds_and_action_hold_are_separate() -> None:
    env = STGEnvironment(
        EmptyScenario(duration_frames=300),
        config=SimulationConfig(reaction_frames=0, action_hold_frames=3),
    )
    assert env.player.radius == 0.5
    start_x = env.player.x
    env.step(Action(move_x=1))
    env.step(Action(move_x=-1))
    env.step(Action(move_x=-1))
    assert env.player.x == start_x + 3 * env.player.speed
    env.step(Action(move_x=-1))
    assert env.player.x == start_x + 2 * env.player.speed

    env.set_player_position(999.0, -999.0)
    assert (env.player.x, env.player.y) == (184.0, -208.0)
    for _ in range(200):
        env.step(Action(move_x=-1, move_y=1))
        if env.done:
            break
    assert env.player.x >= -184.0
    assert env.player.y <= 192.0


def test_swept_collision_catches_fast_threat() -> None:
    env = STGEnvironment(SweptScenario(), config=SimulationConfig(reaction_frames=0))
    result = env.step(Action())
    assert result.done
    assert result.outcome is Outcome.HIT
    assert any(event.kind == "player_hit" for event in result.events)


def test_forecast_is_deterministic_and_does_not_mutate_source() -> None:
    env = STGEnvironment(SweptScenario(), config=SimulationConfig(reaction_frames=0), seed=9)
    forecast_a = env.forecast(horizon=4)
    forecast_b = env.forecast(horizon=4)
    assert env.frame == 0
    assert forecast_a == forecast_b
    assert [frame.frame for frame in forecast_a] == [1, 2, 3, 4]


def test_forecast_preserves_ephemeral_threats_and_terminal_offset() -> None:
    env = STGEnvironment(EphemeralScenario(), config=SimulationConfig(reaction_frames=0))
    forecast = env.forecast(horizon=4, step=4)
    assert [frame.offset for frame in forecast] == [4]
    assert any(threat.id == 1 for threat in forecast[0].swept_threats)

    terminal = STGEnvironment(EmptyScenario(duration_frames=3))
    forecast = terminal.forecast(horizon=10, step=4)
    assert [frame.offset for frame in forecast] == [3]


def test_empty_scenario_clears() -> None:
    env = STGEnvironment(EmptyScenario(duration_frames=3), config=SimulationConfig(reaction_frames=0))
    result = env.step(Action())
    result = env.step(Action())
    result = env.step(Action())
    assert result.outcome is Outcome.CLEAR


def test_ellipse_collision_uses_euclidean_circle_clearance() -> None:
    ellipse = EllipseThreat(0.0, 0.0, 4.0, 2.2)
    parameter = np.pi / 4.0
    boundary = np.asarray((4.0 * np.cos(parameter), 2.2 * np.sin(parameter)))
    normal = np.asarray((np.cos(parameter) / 4.0, np.sin(parameter) / 2.2))
    normal /= np.linalg.norm(normal)
    point = boundary + 1.99 * normal
    assert ellipse.collides_swept(tuple(point), tuple(point), 2.0)


def test_non_finite_geometry_is_rejected() -> None:
    with pytest.raises(ValueError):
        SimulationConfig(player_speed=float("nan"))
    with pytest.raises(ValueError):
        CircleThreat(float("inf"), 0.0, 1.0)


def test_scenario_seed_is_repeatable_but_not_ignored() -> None:
    actions = [Action(move_x=(frame % 3) - 1) for frame in range(80)]
    hashes: list[list[str]] = []
    for seed in (11, 11, 12):
        env = make_environment("stage5_boss4", seed=seed, duration_frames=100)
        trajectory = []
        for action in actions:
            result = env.step(action)
            trajectory.append(state_hash(result.observation))
            if result.done:
                break
        hashes.append(trajectory)
    assert hashes[0] == hashes[1]
    assert hashes[0] != hashes[2]
