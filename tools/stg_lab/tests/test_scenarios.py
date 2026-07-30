from stg_lab.protocol import Action
from stg_lab.scenarios import make_environment
from stg_lab.sim import SimulationConfig


def advance(env, frame: int) -> None:
    while env.frame < frame and not env.done:
        env._advance(Action(), build_semantic=False, detect_collision=False)


def test_boss3_uses_all_original_sources_without_hidden_telegraph() -> None:
    env = make_environment(
        "stage5_boss3",
        seed=4,
        duration_frames=260,
        config=SimulationConfig(reaction_frames=0),
    )
    assert len(env.scenario.source_xs) == 10
    assert env.scenario.high_radius == 28.0
    advance(env, 60)
    nukes = tuple(env.iter_threats(tag="expanding_nuke"))
    assert len(nukes) == 10
    assert all(not threat.warning for threat in nukes)
    assert {threat.metadata["source_index"] for threat in nukes} == set(range(10))


def test_boss4_star_timing_and_fan_density_match_lua_shape() -> None:
    env = make_environment(
        "stage5_boss4",
        seed=7,
        duration_frames=240,
        config=SimulationConfig(reaction_frames=0),
    )
    advance(env, 119)
    assert not tuple(env.iter_threats(tag="orbiting_star"))
    advance(env, 120)
    assert len(tuple(env.iter_threats(tag="orbiting_star"))) == 1
    assert not tuple(env.iter_threats(tag="rotating_fan_bullet"))
    advance(env, 151)
    assert len(tuple(env.iter_threats(tag="rotating_fan_bullet"))) == 18


def test_sr_scenarios_reject_non_60hz_configuration() -> None:
    try:
        make_environment("stage5_boss3", config=SimulationConfig(fps=30))
    except ValueError as error:
        assert "60 fps" in str(error)
    else:
        raise AssertionError("SR scenario unexpectedly accepted a non-60 Hz clock")
